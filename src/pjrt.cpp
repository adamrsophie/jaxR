// Thin glue between R and the PJRT C API (XLA runtime plugins).
//
// The PJRT plugin (a shared library exporting GetPjrtApi) is loaded at run time.
// All host data crosses the boundary in row-major (C) order; R reorders arrays.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include "pjrt_c_api.h"

static const PJRT_Api* g_api = nullptr;
static PJRT_Client* g_client = nullptr;
static std::vector<PJRT_Device*> g_devices;
static std::string g_platform;

// ---- errors -------------------------------------------------------------------------

static std::string take_error(PJRT_Error* err) {
  PJRT_Error_Message_Args m;
  std::memset(&m, 0, sizeof m);
  m.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  m.error = err;
  g_api->PJRT_Error_Message(&m);
  std::string msg(m.message, m.message_size);
  PJRT_Error_Destroy_Args d;
  std::memset(&d, 0, sizeof d);
  d.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  d.error = err;
  g_api->PJRT_Error_Destroy(&d);
  return msg;
}

// Raise an R error from a PJRT error. Only C style data is alive at this point
// except the std::string, which is copied to a stack buffer first.
#define PJRT_CHECK(expr, what)                                              \
  do {                                                                      \
    PJRT_Error* err__ = (expr);                                             \
    if (err__) {                                                            \
      char buf__[4096];                                                     \
      {                                                                     \
        std::string m__ = take_error(err__);                                \
        std::snprintf(buf__, sizeof buf__, "%s: %s", what, m__.c_str());    \
      }                                                                     \
      Rf_error("%s", buf__);                                                \
    }                                                                       \
  } while (0)

static void require_client() {
  if (!g_api || !g_client) Rf_error("jaxR: no XLA backend loaded (see jax_backend_init())");
}

// Optional hook run by the R main thread while it waits for device events, so host
// callbacks queued by FFI handlers on XLA worker threads can be serviced (see
// src/callbacks.cpp). Set with jaxr_set_pump().
typedef void (*jaxr_pump_fn)();
static jaxr_pump_fn g_pump = nullptr;
extern "C" void jaxr_set_pump(jaxr_pump_fn f) { g_pump = f; }

static void await_and_destroy(PJRT_Event* ev, const char* what) {
  if (!ev) return;
  if (g_pump) {
    for (;;) {
      PJRT_Event_IsReady_Args r;
      std::memset(&r, 0, sizeof r);
      r.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
      r.event = ev;
      PJRT_Error* e = g_api->PJRT_Event_IsReady(&r);
      if (e) { take_error(e); break; }
      if (r.is_ready) break;
      g_pump();
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
  }
  PJRT_Event_Await_Args a;
  std::memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  a.event = ev;
  PJRT_Error* err = g_api->PJRT_Event_Await(&a);
  PJRT_Event_Destroy_Args d;
  std::memset(&d, 0, sizeof d);
  d.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  d.event = ev;
  PJRT_Error* err2 = g_api->PJRT_Event_Destroy(&d);
  if (err2) take_error(err2);
  PJRT_CHECK(err, what);
}

// ---- plugin / client ------------------------------------------------------------------

// Whether the function table of the plugin has the member `name`: a plugin of the
// same major version may have an older, shorter table (struct_size says how long).
#define PJRT_HAS(api, name) \
  ((api)->struct_size >= offsetof(PJRT_Api, name) + sizeof((api)->name) && (api)->name)

// Releases a client made by a load that failed later, so that a new load may try again.
static void abandon_client(const PJRT_Api* api, PJRT_Client* client) {
  if (!client || !PJRT_HAS(api, PJRT_Client_Destroy)) return;
  PJRT_Client_Destroy_Args a;
  std::memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  a.client = client;
  PJRT_Error* err = api->PJRT_Client_Destroy(&a);
  if (err) take_error(err);
}

extern "C" SEXP jaxr_pjrt_load(SEXP path) {
  if (g_client) Rf_error("jaxR: an XLA backend is already loaded");
  const char* p = CHAR(STRING_ELT(path, 0));
  typedef const PJRT_Api* (*GetApiFn)();
#ifdef _WIN32
  HMODULE h = LoadLibraryA(p);
  if (!h) Rf_error("jaxR: cannot load PJRT plugin '%s'", p);
  GetApiFn get = (GetApiFn)GetProcAddress(h, "GetPjrtApi");
#else
  void* h = dlopen(p, RTLD_NOW | RTLD_LOCAL);
  if (!h) Rf_error("jaxR: cannot load PJRT plugin '%s': %s", p, dlerror());
  GetApiFn get = (GetApiFn)dlsym(h, "GetPjrtApi");
#endif
  if (!get) Rf_error("jaxR: '%s' does not export GetPjrtApi", p);
  const PJRT_Api* api = get();
  if (!api) Rf_error("jaxR: GetPjrtApi returned NULL");
  if (api->pjrt_api_version.major_version != PJRT_API_MAJOR)
    Rf_error("jaxR: PJRT API major version mismatch (plugin %d, jaxR %d)",
             api->pjrt_api_version.major_version, PJRT_API_MAJOR);
  // Every function jaxR calls unconditionally must be in the table.
#define REQUIRE(name) \
  if (!PJRT_HAS(api, name)) Rf_error("jaxR: the PJRT plugin does not provide " #name)
  REQUIRE(PJRT_Error_Destroy); REQUIRE(PJRT_Error_Message); REQUIRE(PJRT_Plugin_Initialize);
  REQUIRE(PJRT_Event_Destroy); REQUIRE(PJRT_Event_IsReady); REQUIRE(PJRT_Event_Await);
  REQUIRE(PJRT_Client_Create); REQUIRE(PJRT_Client_PlatformName);
  REQUIRE(PJRT_Client_AddressableDevices); REQUIRE(PJRT_Client_Compile);
  REQUIRE(PJRT_Client_BufferFromHostBuffer); REQUIRE(PJRT_Executable_Destroy);
  REQUIRE(PJRT_Executable_NumOutputs); REQUIRE(PJRT_Executable_Serialize);
  REQUIRE(PJRT_Executable_DeserializeAndLoad); REQUIRE(PJRT_LoadedExecutable_Destroy);
  REQUIRE(PJRT_LoadedExecutable_GetExecutable); REQUIRE(PJRT_LoadedExecutable_Execute);
  REQUIRE(PJRT_Buffer_Destroy); REQUIRE(PJRT_Buffer_ToHostBuffer); REQUIRE(PJRT_Buffer_ReadyEvent);
#undef REQUIRE
  g_api = api;

  PJRT_Plugin_Initialize_Args ia;
  std::memset(&ia, 0, sizeof ia);
  ia.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  PJRT_CHECK(g_api->PJRT_Plugin_Initialize(&ia), "PJRT_Plugin_Initialize");

  PJRT_Client_Create_Args ca;
  std::memset(&ca, 0, sizeof ca);
  ca.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  PJRT_CHECK(g_api->PJRT_Client_Create(&ca), "PJRT_Client_Create");
  PJRT_Client* client = ca.client;

  // The client is published only once every query succeeded; on an error it is
  // released, so that the session can load a backend again.
  PJRT_Client_AddressableDevices_Args da;
  std::memset(&da, 0, sizeof da);
  da.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  da.client = client;
  PJRT_Error* err = g_api->PJRT_Client_AddressableDevices(&da);
  PJRT_Client_PlatformName_Args pa;
  std::memset(&pa, 0, sizeof pa);
  pa.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  pa.client = client;
  if (!err) err = g_api->PJRT_Client_PlatformName(&pa);
  if (err) {
    std::string msg = take_error(err);
    abandon_client(g_api, client);
    Rf_error("jaxR: PJRT client query failed: %s", msg.c_str());
  }
  g_devices.assign(da.addressable_devices, da.addressable_devices + da.num_addressable_devices);
  g_platform.assign(pa.platform_name, pa.platform_name_size);
  g_client = client;

  SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
  SEXP nms = PROTECT(Rf_allocVector(STRSXP, 4));
  SET_VECTOR_ELT(out, 0, Rf_mkString(g_platform.c_str()));
  SET_STRING_ELT(nms, 0, Rf_mkChar("platform"));
  SET_VECTOR_ELT(out, 1, Rf_ScalarInteger((int)g_devices.size()));
  SET_STRING_ELT(nms, 1, Rf_mkChar("num_devices"));
  SET_VECTOR_ELT(out, 2, Rf_ScalarInteger(g_api->pjrt_api_version.major_version));
  SET_STRING_ELT(nms, 2, Rf_mkChar("api_major"));
  SET_VECTOR_ELT(out, 3, Rf_ScalarInteger(g_api->pjrt_api_version.minor_version));
  SET_STRING_ELT(nms, 3, Rf_mkChar("api_minor"));
  Rf_setAttrib(out, R_NamesSymbol, nms);
  UNPROTECT(2);
  return out;
}

extern "C" SEXP jaxr_pjrt_loaded() { return Rf_ScalarLogical(g_client != nullptr); }

// Accessors for other translation units (FFI handler registration).
extern "C" const PJRT_Api* jaxr_api() { return g_api; }
extern "C" PJRT_Client* jaxr_client() { return g_client; }

// ---- executables ----------------------------------------------------------------------

static void exe_finalizer(SEXP ptr) {
  PJRT_LoadedExecutable* e = (PJRT_LoadedExecutable*)R_ExternalPtrAddr(ptr);
  if (!e || !g_api) return;
  PJRT_LoadedExecutable_Destroy_Args d;
  std::memset(&d, 0, sizeof d);
  d.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  d.executable = e;
  PJRT_Error* err = g_api->PJRT_LoadedExecutable_Destroy(&d);
  if (err) take_error(err);
  R_ClearExternalPtr(ptr);
}

// Wraps a loaded executable for R and records its number of outputs.
static SEXP wrap_loaded(PJRT_LoadedExecutable* exe) {
  // Owned by R from here on, so that an error below releases the executable.
  SEXP ptr = PROTECT(R_MakeExternalPtr(exe, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ptr, exe_finalizer, TRUE);

  PJRT_LoadedExecutable_GetExecutable_Args ga;
  std::memset(&ga, 0, sizeof ga);
  ga.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  ga.loaded_executable = exe;
  PJRT_CHECK(g_api->PJRT_LoadedExecutable_GetExecutable(&ga), "GetExecutable");
  PJRT_Executable_NumOutputs_Args na;
  std::memset(&na, 0, sizeof na);
  na.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
  na.executable = ga.executable;
  PJRT_Error* err = g_api->PJRT_Executable_NumOutputs(&na);
  PJRT_Executable_Destroy_Args xd;
  std::memset(&xd, 0, sizeof xd);
  xd.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  xd.executable = ga.executable;
  PJRT_Error* err2 = g_api->PJRT_Executable_Destroy(&xd);
  if (err2) take_error(err2);
  PJRT_CHECK(err, "NumOutputs");

  Rf_setAttrib(ptr, Rf_install("num_outputs"), Rf_ScalarInteger((int)na.num_outputs));
  UNPROTECT(1);
  return ptr;
}

extern "C" SEXP jaxr_compile(SEXP code, SEXP options) {
  require_client();
  SEXP s = STRING_ELT(code, 0);
  PJRT_Program prog;
  std::memset(&prog, 0, sizeof prog);
  prog.struct_size = PJRT_Program_STRUCT_SIZE;
  prog.code = (char*)CHAR(s);
  prog.code_size = (size_t)LENGTH(s);
  prog.format = "mlir";
  prog.format_size = 4;

  PJRT_Client_Compile_Args a;
  std::memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  a.client = g_client;
  a.program = &prog;
  a.compile_options = (const char*)RAW(options);
  a.compile_options_size = (size_t)XLENGTH(options);
  PJRT_CHECK(g_api->PJRT_Client_Compile(&a), "XLA compilation failed");
  SEXP ptr = PROTECT(wrap_loaded(a.executable));
  // Programs with host callbacks (src/callbacks.cpp) are awaited by jaxr_execute so that
  // their side effects happen before it returns.
  SEXP cb_sym = Rf_install("host_callbacks");
  Rf_setAttrib(ptr, cb_sym, Rf_ScalarLogical(std::strstr(CHAR(s), "\"jaxr_r_callback\"") != nullptr));
  UNPROTECT(1);
  return ptr;
}

// Port of PjRtLoadedExecutable serialization (jax.experimental.serialize_executable):
// the platform specific bytes of a compiled executable.
extern "C" SEXP jaxr_executable_serialize(SEXP ptr) {
  require_client();
  PJRT_LoadedExecutable* exe = (PJRT_LoadedExecutable*)R_ExternalPtrAddr(ptr);
  if (!exe) Rf_error("jaxR: the executable has been released");
  PJRT_LoadedExecutable_GetExecutable_Args ga;
  std::memset(&ga, 0, sizeof ga);
  ga.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  ga.loaded_executable = exe;
  PJRT_CHECK(g_api->PJRT_LoadedExecutable_GetExecutable(&ga), "GetExecutable");
  PJRT_Executable_Serialize_Args sa;
  std::memset(&sa, 0, sizeof sa);
  sa.struct_size = PJRT_Executable_Serialize_Args_STRUCT_SIZE;
  sa.executable = ga.executable;
  PJRT_Error* err = g_api->PJRT_Executable_Serialize(&sa);
  SEXP out = R_NilValue;
  if (!err) {
    out = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t)sa.serialized_bytes_size));
    if (sa.serialized_bytes_size) std::memcpy(RAW(out), sa.serialized_bytes, sa.serialized_bytes_size);
    if (sa.serialized_executable_deleter) sa.serialized_executable_deleter(sa.serialized_executable);
  }
  PJRT_Executable_Destroy_Args xd;
  std::memset(&xd, 0, sizeof xd);
  xd.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  xd.executable = ga.executable;
  PJRT_Error* err2 = g_api->PJRT_Executable_Destroy(&xd);
  if (err2) take_error(err2);
  if (err) PJRT_CHECK(err, "Executable serialization failed");
  UNPROTECT(1);
  return out;
}

// The unloaded executable of a loaded one; the caller destroys it.
static PJRT_Executable* get_executable(SEXP ptr) {
  require_client();
  PJRT_LoadedExecutable* exe = (PJRT_LoadedExecutable*)R_ExternalPtrAddr(ptr);
  if (!exe) Rf_error("jaxR: the executable has been released");
  PJRT_LoadedExecutable_GetExecutable_Args ga;
  std::memset(&ga, 0, sizeof ga);
  ga.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  ga.loaded_executable = exe;
  PJRT_CHECK(g_api->PJRT_LoadedExecutable_GetExecutable(&ga), "GetExecutable");
  return ga.executable;
}

static void destroy_executable(PJRT_Executable* e) {
  PJRT_Executable_Destroy_Args xd;
  std::memset(&xd, 0, sizeof xd);
  xd.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  xd.executable = e;
  PJRT_Error* err = g_api->PJRT_Executable_Destroy(&xd);
  if (err) take_error(err);
}

// Port of Executable.cost_analysis: the cost properties XLA reports, as a named list
// (NULL when the plugin does not provide them).
extern "C" SEXP jaxr_executable_cost_analysis(SEXP ptr) {
  PJRT_Executable* e = get_executable(ptr);
  if (!PJRT_HAS(g_api, PJRT_Executable_GetCostAnalysis)) {
    destroy_executable(e);
    return R_NilValue;
  }
  PJRT_Executable_GetCostAnalysis_Args ca;
  std::memset(&ca, 0, sizeof ca);
  ca.struct_size = PJRT_Executable_GetCostAnalysis_Args_STRUCT_SIZE;
  ca.executable = e;
  PJRT_Error* err = g_api->PJRT_Executable_GetCostAnalysis(&ca);
  if (err) {
    take_error(err);
    destroy_executable(e);
    return R_NilValue;
  }
  // The properties belong to the executable: they are copied before it is destroyed.
  SEXP out = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)ca.num_properties));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t)ca.num_properties));
  for (size_t i = 0; i < ca.num_properties; i++) {
    const PJRT_NamedValue& v = ca.properties[i];
    SET_STRING_ELT(names, (R_xlen_t)i, Rf_mkCharLen(v.name, (int)v.name_size));
    SEXP val = R_NilValue;
    switch (v.type) {
      case PJRT_NamedValue_kString:
        val = Rf_ScalarString(Rf_mkCharLen(v.string_value, (int)v.value_size));
        break;
      case PJRT_NamedValue_kInt64:
        val = Rf_ScalarReal((double)v.int64_value);
        break;
      case PJRT_NamedValue_kInt64List:
        val = Rf_allocVector(REALSXP, (R_xlen_t)v.value_size);
        for (size_t j = 0; j < v.value_size; j++) REAL(val)[j] = (double)v.int64_array_value[j];
        break;
      case PJRT_NamedValue_kFloat:
        val = Rf_ScalarReal((double)v.float_value);
        break;
      case PJRT_NamedValue_kBool:
        val = Rf_ScalarLogical(v.bool_value ? 1 : 0);
        break;
    }
    SET_VECTOR_ELT(out, (R_xlen_t)i, val);
  }
  Rf_setAttrib(out, R_NamesSymbol, names);
  destroy_executable(e);
  UNPROTECT(2);
  return out;
}

// Port of Executable.memory_analysis (xla::CompiledMemoryStats), as a named numeric
// vector (NULL when the plugin does not provide it).
extern "C" SEXP jaxr_executable_memory_analysis(SEXP ptr) {
  PJRT_Executable* e = get_executable(ptr);
  if (!PJRT_HAS(g_api, PJRT_Executable_GetCompiledMemoryStats)) {
    destroy_executable(e);
    return R_NilValue;
  }
  PJRT_Executable_GetCompiledMemoryStats_Args m;
  std::memset(&m, 0, sizeof m);
  m.struct_size = PJRT_Executable_GetCompiledMemoryStats_Args_STRUCT_SIZE;
  m.executable = e;
  PJRT_Error* err = g_api->PJRT_Executable_GetCompiledMemoryStats(&m);
  destroy_executable(e);
  if (err) {
    take_error(err);
    return R_NilValue;
  }
  const char* nm[] = {"generated_code_size_in_bytes", "argument_size_in_bytes",
                      "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes",
                      "host_generated_code_size_in_bytes", "host_argument_size_in_bytes",
                      "host_output_size_in_bytes", "host_alias_size_in_bytes",
                      "host_temp_size_in_bytes", "peak_memory_in_bytes", "total_allocation_bytes",
                      "indefinite_allocations", "peak_unpadded_heap_bytes"};
  int64_t vals[] = {m.generated_code_size_in_bytes, m.argument_size_in_bytes,
                    m.output_size_in_bytes, m.alias_size_in_bytes, m.temp_size_in_bytes,
                    m.host_generated_code_size_in_bytes, m.host_argument_size_in_bytes,
                    m.host_output_size_in_bytes, m.host_alias_size_in_bytes,
                    m.host_temp_size_in_bytes, m.peak_memory_in_bytes, m.total_allocation_bytes,
                    m.indefinite_allocations, m.peak_unpadded_heap_bytes};
  const int n = sizeof(vals) / sizeof(vals[0]);
  SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, n));
  for (int i = 0; i < n; i++) {
    REAL(out)[i] = (double)vals[i];
    SET_STRING_ELT(names, i, Rf_mkChar(nm[i]));
  }
  Rf_setAttrib(out, R_NamesSymbol, names);
  UNPROTECT(2);
  return out;
}

extern "C" SEXP jaxr_executable_deserialize(SEXP bytes) {
  require_client();
  PJRT_Executable_DeserializeAndLoad_Args da;
  std::memset(&da, 0, sizeof da);
  da.struct_size = PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE;
  da.client = g_client;
  da.serialized_executable = (const char*)RAW(bytes);
  da.serialized_executable_size = (size_t)XLENGTH(bytes);
  PJRT_CHECK(g_api->PJRT_Executable_DeserializeAndLoad(&da), "Executable deserialization failed");
  SEXP ptr = PROTECT(wrap_loaded(da.loaded_executable));
  Rf_setAttrib(ptr, Rf_install("host_callbacks"), Rf_ScalarLogical(FALSE));
  UNPROTECT(1);
  return ptr;
}

// ---- buffers --------------------------------------------------------------------------

enum { T_PRED = PJRT_Buffer_Type_PRED, T_S8 = PJRT_Buffer_Type_S8, T_S16 = PJRT_Buffer_Type_S16,
       T_S32 = PJRT_Buffer_Type_S32, T_S64 = PJRT_Buffer_Type_S64, T_U8 = PJRT_Buffer_Type_U8,
       T_U16 = PJRT_Buffer_Type_U16, T_U32 = PJRT_Buffer_Type_U32, T_U64 = PJRT_Buffer_Type_U64,
       T_F16 = PJRT_Buffer_Type_F16, T_F32 = PJRT_Buffer_Type_F32, T_F64 = PJRT_Buffer_Type_F64,
       T_BF16 = PJRT_Buffer_Type_BF16, T_C64 = PJRT_Buffer_Type_C64, T_C128 = PJRT_Buffer_Type_C128 };

static size_t elem_size(int t) {
  switch (t) {
    case T_PRED: case T_S8: case T_U8: return 1;
    case T_S16: case T_U16: case T_F16: case T_BF16: return 2;
    case T_S32: case T_U32: case T_F32: return 4;
    case T_S64: case T_U64: case T_F64: case T_C64: return 8;
    case T_C128: return 16;
  }
  Rf_error("jaxR: unsupported buffer type %d", t);
  return 0;
}

// Rounds a double to nearest even in a binary format with `ebits` exponent bits and
// `mbits` mantissa bits (float16: 5, 10; bfloat16: 8, 7), directly from its bits, so
// that no intermediate rounding (through float) changes which side of a midpoint it is.
extern "C" uint16_t jaxr_f64_to_small(double v, int ebits, int mbits) {
  uint64_t x;
  std::memcpy(&x, &v, 8);
  uint64_t sign = (x >> 63) << (ebits + mbits);
  uint64_t mant = x & ((1ull << 52) - 1ull);
  int64_t exp = (int64_t)((x >> 52) & 0x7ffu);
  uint64_t emax = (1ull << ebits) - 1ull;
  if (exp == 0x7ff) {
    uint64_t q = mant ? (1ull << (mbits - 1)) : 0ull;
    return (uint16_t)(sign | (emax << mbits) | q);
  }
  if (exp == 0 && mant == 0) return (uint16_t)sign;
  int64_t bias = (1ll << (ebits - 1)) - 1;
  int64_t e = exp - 1023 + bias;
  int shift_n = 52 - mbits;
  uint64_t m = exp ? (mant | (1ull << 52)) : mant;
  if (!exp) e += 1;  // a double subnormal (far below either format)
  if (e >= (int64_t)emax) return (uint16_t)(sign | (emax << mbits));
  uint64_t out;
  uint64_t rem, mid;
  int shift;
  if (e <= 0) {
    shift = shift_n + 1 - (int)e;
    if (shift > 62) return (uint16_t)sign;
    out = m >> shift;
  } else {
    shift = shift_n;
    out = ((uint64_t)e << mbits) | ((m >> shift) & ((1ull << mbits) - 1ull));
  }
  rem = m & ((1ull << shift) - 1ull);
  mid = 1ull << (shift - 1);
  if (rem > mid || (rem == mid && (out & 1ull))) out++;
  return (uint16_t)(sign | out);
}

static float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t x;
  if (exp == 0) {
    if (mant == 0) x = sign;
    else {
      int e = -1;
      do { e++; mant <<= 1; } while (!(mant & 0x400u));
      x = sign | ((uint32_t)(127 - 15 - e) << 23) | ((mant & 0x3ffu) << 13);
    }
  } else if (exp == 0x1f) {
    x = sign | 0x7f800000u | (mant << 13);
  } else {
    x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

static float bf16_to_f32(uint16_t b) {
  uint32_t x = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

static double get_num(SEXP data, R_xlen_t i) {
  switch (TYPEOF(data)) {
    case REALSXP: return REAL(data)[i];
    case INTSXP: return (double)INTEGER(data)[i];
    case LGLSXP: return (double)LOGICAL(data)[i];
    default: Rf_error("jaxR: unsupported host data type");
  }
  return 0;
}

static void buffer_finalizer(SEXP ptr) {
  PJRT_Buffer* b = (PJRT_Buffer*)R_ExternalPtrAddr(ptr);
  if (!b || !g_api) return;
  PJRT_Buffer_Destroy_Args d;
  std::memset(&d, 0, sizeof d);
  d.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  d.buffer = b;
  PJRT_Error* err = g_api->PJRT_Buffer_Destroy(&d);
  if (err) take_error(err);
  R_ClearExternalPtr(ptr);
}

static SEXP wrap_buffer(PJRT_Buffer* b) {
  SEXP ptr = PROTECT(R_MakeExternalPtr(b, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ptr, buffer_finalizer, TRUE);
  UNPROTECT(1);
  return ptr;
}

// data: flat host vector in row-major order; type: PJRT_Buffer_Type; dims: shape.
extern "C" SEXP jaxr_buffer_from_host(SEXP data, SEXP type, SEXP dims, SEXP device) {
  require_client();
  int t = Rf_asInteger(type);
  R_xlen_t n = XLENGTH(data);
  size_t es = elem_size(t);
  // R_alloc memory is released by R on return and on error (Rf_error does not run C++
  // destructors, so no std::vector may be live across an error path).
  uint8_t* p = (uint8_t*)R_alloc((size_t)n * es + 1, 1);
  if (t == T_C128 || t == T_C64) {
    if (TYPEOF(data) != CPLXSXP) Rf_error("jaxR: complex buffer needs complex data");
    Rcomplex* c = COMPLEX(data);
    for (R_xlen_t i = 0; i < n; i++) {
      if (t == T_C128) {
        double v[2] = {c[i].r, c[i].i};
        std::memcpy(p + i * 16, v, 16);
      } else {
        float v[2] = {(float)c[i].r, (float)c[i].i};
        std::memcpy(p + i * 8, v, 8);
      }
    }
  } else {
    for (R_xlen_t i = 0; i < n; i++) {
      double v = get_num(data, i);
      switch (t) {
        case T_PRED: p[i] = (uint8_t)(v != 0); break;
        case T_S8: { int8_t w = (int8_t)(int64_t)v; std::memcpy(p + i, &w, 1); break; }
        case T_S16: { int16_t w = (int16_t)(int64_t)v; std::memcpy(p + 2 * i, &w, 2); break; }
        case T_S32: { int32_t w = (int32_t)(int64_t)v; std::memcpy(p + 4 * i, &w, 4); break; }
        case T_S64: { int64_t w = (int64_t)v; std::memcpy(p + 8 * i, &w, 8); break; }
        case T_U8: { uint8_t w = (uint8_t)(uint64_t)v; p[i] = w; break; }
        case T_U16: { uint16_t w = (uint16_t)(uint64_t)v; std::memcpy(p + 2 * i, &w, 2); break; }
        case T_U32: { uint32_t w = (uint32_t)(uint64_t)v; std::memcpy(p + 4 * i, &w, 4); break; }
        case T_U64: { uint64_t w = (uint64_t)v; std::memcpy(p + 8 * i, &w, 8); break; }
        case T_F16: { uint16_t w = jaxr_f64_to_small(v, 5, 10); std::memcpy(p + 2 * i, &w, 2); break; }
        case T_BF16: { uint16_t w = jaxr_f64_to_small(v, 8, 7); std::memcpy(p + 2 * i, &w, 2); break; }
        case T_F32: { float w = (float)v; std::memcpy(p + 4 * i, &w, 4); break; }
        case T_F64: std::memcpy(p + 8 * i, &v, 8); break;
        default: Rf_error("jaxR: unsupported buffer type %d", t);
      }
    }
  }
  int nd = LENGTH(dims);
  int64_t* d = (int64_t*)R_alloc((size_t)nd + 1, sizeof(int64_t));
  for (int i = 0; i < nd; i++) d[i] = (int64_t)INTEGER(dims)[i];
  int dev = Rf_asInteger(device);
  if (dev < 0 || dev >= (int)g_devices.size()) Rf_error("jaxR: invalid device %d", dev);

  PJRT_Client_BufferFromHostBuffer_Args a;
  std::memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  a.client = g_client;
  a.data = p;
  a.type = (PJRT_Buffer_Type)t;
  a.dims = d;
  a.num_dims = (size_t)nd;
  a.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
  a.device = g_devices[dev];
  PJRT_CHECK(g_api->PJRT_Client_BufferFromHostBuffer(&a), "BufferFromHostBuffer");
  // Owned by R before waiting, so that a failed transfer releases the buffer.
  SEXP res = PROTECT(wrap_buffer(a.buffer));
  if (a.done_with_host_buffer) await_and_destroy(a.done_with_host_buffer, "host transfer");
  UNPROTECT(1);
  return res;
}

// The bytes of a device buffer (row-major order), for exact constants.
extern "C" SEXP jaxr_buffer_raw(SEXP ptr, SEXP nbytes) {
  require_client();
  PJRT_Buffer* b = (PJRT_Buffer*)R_ExternalPtrAddr(ptr);
  if (!b) Rf_error("jaxR: buffer has been deleted");
  R_xlen_t n = (R_xlen_t)Rf_asReal(nbytes);
  SEXP out = PROTECT(Rf_allocVector(RAWSXP, n));
  PJRT_Buffer_ToHostBuffer_Args a;
  std::memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  a.src = b;
  a.dst = RAW(out);
  a.dst_size = (size_t)n;
  PJRT_CHECK(g_api->PJRT_Buffer_ToHostBuffer(&a), "ToHostBuffer");
  await_and_destroy(a.event, "device to host transfer");
  UNPROTECT(1);
  return out;
}

// Waits until a device buffer is ready (PJRT_Buffer_ReadyEvent), without a copy; R
// callbacks are serviced meanwhile.
extern "C" SEXP jaxr_buffer_ready(SEXP ptr) {
  require_client();
  PJRT_Buffer* b = (PJRT_Buffer*)R_ExternalPtrAddr(ptr);
  if (!b) Rf_error("jaxR: buffer has been deleted");
  PJRT_Buffer_ReadyEvent_Args a;
  std::memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
  a.buffer = b;
  PJRT_CHECK(g_api->PJRT_Buffer_ReadyEvent(&a), "Buffer_ReadyEvent");
  await_and_destroy(a.event, "XLA execution failed");
  return R_NilValue;
}

// Copy a device buffer to a flat R vector (row-major order).
extern "C" SEXP jaxr_buffer_to_host(SEXP ptr, SEXP type, SEXP size) {
  require_client();
  PJRT_Buffer* b = (PJRT_Buffer*)R_ExternalPtrAddr(ptr);
  if (!b) Rf_error("jaxR: buffer has been deleted");
  int t = Rf_asInteger(type);
  R_xlen_t n = (R_xlen_t)Rf_asReal(size);
  size_t es = elem_size(t);
  uint8_t* host = (uint8_t*)R_alloc((size_t)n * es + 1, 1);
  PJRT_Buffer_ToHostBuffer_Args a;
  std::memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  a.src = b;
  a.dst = host;
  a.dst_size = (size_t)n * es;
  PJRT_CHECK(g_api->PJRT_Buffer_ToHostBuffer(&a), "ToHostBuffer");
  await_and_destroy(a.event, "device to host transfer");
  const uint8_t* p = host;
  SEXP out;
  if (t == T_PRED) {
    out = PROTECT(Rf_allocVector(LGLSXP, n));
    for (R_xlen_t i = 0; i < n; i++) LOGICAL(out)[i] = p[i] != 0;
  } else if (t == T_C128 || t == T_C64) {
    out = PROTECT(Rf_allocVector(CPLXSXP, n));
    for (R_xlen_t i = 0; i < n; i++) {
      if (t == T_C128) {
        double v[2];
        std::memcpy(v, p + 16 * i, 16);
        COMPLEX(out)[i].r = v[0];
        COMPLEX(out)[i].i = v[1];
      } else {
        float v[2];
        std::memcpy(v, p + 8 * i, 8);
        COMPLEX(out)[i].r = v[0];
        COMPLEX(out)[i].i = v[1];
      }
    }
  } else {
    out = PROTECT(Rf_allocVector(REALSXP, n));
    double* o = REAL(out);
    for (R_xlen_t i = 0; i < n; i++) {
      switch (t) {
        case T_S8: { int8_t w; std::memcpy(&w, p + i, 1); o[i] = w; break; }
        case T_S16: { int16_t w; std::memcpy(&w, p + 2 * i, 2); o[i] = w; break; }
        case T_S32: { int32_t w; std::memcpy(&w, p + 4 * i, 4); o[i] = w; break; }
        case T_S64: { int64_t w; std::memcpy(&w, p + 8 * i, 8); o[i] = (double)w; break; }
        case T_U8: o[i] = p[i]; break;
        case T_U16: { uint16_t w; std::memcpy(&w, p + 2 * i, 2); o[i] = w; break; }
        case T_U32: { uint32_t w; std::memcpy(&w, p + 4 * i, 4); o[i] = w; break; }
        case T_U64: { uint64_t w; std::memcpy(&w, p + 8 * i, 8); o[i] = (double)w; break; }
        case T_F16: { uint16_t w; std::memcpy(&w, p + 2 * i, 2); o[i] = f16_to_f32(w); break; }
        case T_BF16: { uint16_t w; std::memcpy(&w, p + 2 * i, 2); o[i] = bf16_to_f32(w); break; }
        case T_F32: { float w; std::memcpy(&w, p + 4 * i, 4); o[i] = w; break; }
        case T_F64: std::memcpy(o + i, p + 8 * i, 8); break;
        default: Rf_error("jaxR: unsupported buffer type %d", t);
      }
    }
  }
  UNPROTECT(1);
  return out;
}

// Run an executable on device 0. args: list of buffer external pointers.
extern "C" SEXP jaxr_execute(SEXP exe, SEXP args) {
  require_client();
  PJRT_LoadedExecutable* e = (PJRT_LoadedExecutable*)R_ExternalPtrAddr(exe);
  if (!e) Rf_error("jaxR: executable has been deleted");
  int nout = Rf_asInteger(Rf_getAttrib(exe, Rf_install("num_outputs")));
  bool wait = Rf_asLogical(Rf_getAttrib(exe, Rf_install("host_callbacks"))) == TRUE;
  PJRT_Event* done = nullptr;
  R_xlen_t nargs = XLENGTH(args);
  PJRT_Buffer** in = (PJRT_Buffer**)R_alloc((size_t)nargs + 1, sizeof(PJRT_Buffer*));
  for (R_xlen_t i = 0; i < nargs; i++) {
    in[i] = (PJRT_Buffer*)R_ExternalPtrAddr(VECTOR_ELT(args, i));
    if (!in[i]) Rf_error("jaxR: argument %d buffer has been deleted", (int)i + 1);
  }
  PJRT_Buffer** out = (PJRT_Buffer**)R_alloc((size_t)nout + 1, sizeof(PJRT_Buffer*));
  for (int i = 0; i <= nout; i++) out[i] = nullptr;
  PJRT_Buffer* const* arg_lists[1] = {in};
  PJRT_Buffer** out_lists[1] = {out};
  PJRT_ExecuteOptions eo;
  std::memset(&eo, 0, sizeof eo);
  eo.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  PJRT_LoadedExecutable_Execute_Args a;
  std::memset(&a, 0, sizeof a);
  a.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  a.executable = e;
  a.options = &eo;
  a.argument_lists = arg_lists;
  a.num_devices = 1;
  a.num_args = (size_t)nargs;
  a.output_lists = out_lists;
  if (wait) a.device_complete_events = &done;
  PJRT_CHECK(g_api->PJRT_LoadedExecutable_Execute(&a), "XLA execution failed");
  SEXP res = PROTECT(Rf_allocVector(VECSXP, nout));
  for (int i = 0; i < nout; i++) SET_VECTOR_ELT(res, i, wrap_buffer(out[i]));
  // Wait for programs with host callbacks, servicing the callbacks, so that their side
  // effects happen before execution returns.
  if (wait) await_and_destroy(done, "XLA execution failed");
  UNPROTECT(1);
  return res;
}

extern "C" SEXP jaxr_buffer_delete(SEXP ptr) {
  buffer_finalizer(ptr);
  return R_NilValue;
}

// Weak references (for jax.live_arrays): the key is a jax array environment.
extern "C" SEXP jaxr_weakref_new(SEXP key) {
  return R_MakeWeakRef(key, R_NilValue, R_NilValue, FALSE);
}

extern "C" SEXP jaxr_weakref_get(SEXP w) { return R_WeakRefKey(w); }

// A weak reference whose value lives while its key does (an ephemeron: a path from
// the value back to the key does not keep the key alive).
extern "C" SEXP jaxr_weakref_pair(SEXP key, SEXP value) {
  return R_MakeWeakRef(key, value, R_NilValue, FALSE);
}

extern "C" SEXP jaxr_weakref_value(SEXP w) { return R_WeakRefValue(w); }

// Identity of an R object (its address), for hashing closures by identity.
extern "C" SEXP jaxr_address(SEXP x) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%p", (void*)x);
  return Rf_mkString(buf);
}

// The protected value of an external pointer: what it keeps alive.
extern "C" SEXP jaxr_extptr_prot(SEXP ptr) { return R_ExternalPtrProtected(ptr); }
extern "C" SEXP jaxr_extptr_set_prot(SEXP ptr, SEXP value) {
  R_SetExternalPtrProtected(ptr, value);
  return R_NilValue;
}

static const R_CallMethodDef call_methods[] = {
  {"jaxr_weakref_new", (DL_FUNC)&jaxr_weakref_new, 1},
  {"jaxr_weakref_get", (DL_FUNC)&jaxr_weakref_get, 1},
  {"jaxr_weakref_pair", (DL_FUNC)&jaxr_weakref_pair, 2},
  {"jaxr_weakref_value", (DL_FUNC)&jaxr_weakref_value, 1},
  {"jaxr_address", (DL_FUNC)&jaxr_address, 1},
  {"jaxr_extptr_prot", (DL_FUNC)&jaxr_extptr_prot, 1},
  {"jaxr_extptr_set_prot", (DL_FUNC)&jaxr_extptr_set_prot, 2},
  {"jaxr_pjrt_load", (DL_FUNC)&jaxr_pjrt_load, 1},
  {"jaxr_pjrt_loaded", (DL_FUNC)&jaxr_pjrt_loaded, 0},
  {"jaxr_compile", (DL_FUNC)&jaxr_compile, 2},
  {"jaxr_executable_serialize", (DL_FUNC)&jaxr_executable_serialize, 1},
  {"jaxr_executable_deserialize", (DL_FUNC)&jaxr_executable_deserialize, 1},
  {"jaxr_executable_cost_analysis", (DL_FUNC)&jaxr_executable_cost_analysis, 1},
  {"jaxr_executable_memory_analysis", (DL_FUNC)&jaxr_executable_memory_analysis, 1},
  {"jaxr_buffer_from_host", (DL_FUNC)&jaxr_buffer_from_host, 4},
  {"jaxr_buffer_to_host", (DL_FUNC)&jaxr_buffer_to_host, 3},
  {"jaxr_buffer_raw", (DL_FUNC)&jaxr_buffer_raw, 2},
  {"jaxr_buffer_ready", (DL_FUNC)&jaxr_buffer_ready, 1},
  {"jaxr_execute", (DL_FUNC)&jaxr_execute, 2},
  {"jaxr_buffer_delete", (DL_FUNC)&jaxr_buffer_delete, 1},
  {nullptr, nullptr, 0}
};

extern "C" void R_init_jaxR(DllInfo* dll) {
  R_registerRoutines(dll, nullptr, call_methods, nullptr, nullptr);
  R_useDynamicSymbols(dll, TRUE);
}
