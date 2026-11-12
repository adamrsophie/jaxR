// Host callbacks into R from compiled XLA programs (the R counterpart of jaxlib's
// xla_ffi_python_cpu_callback, used by jax/_src/callback.py:emit_python_callback).
//
// The lowerings of pure_callback, io_callback, debug_callback and debug_print emit a
// stablehlo.custom_call to the typed FFI target "jaxr_r_callback" with the attribute
// `index` (an id into the callback registry of R/callbacks.R). The handler below is
// registered through the PJRT FFI extension, as src/lapack_ffi.cpp registers its
// LAPACK targets.
//
// R may only run on the R main thread. When XLA runs the handler on that thread
// (the program executes inline inside PJRT_LoadedExecutable_Execute or while the main
// thread waits on an event), the handler calls R directly. Otherwise it queues the
// request and blocks until the main thread services the queue: src/pjrt.cpp calls the
// pump installed here (jaxr_set_pump) while it waits for device events. R errors never
// unwind through XLA frames: R is entered through R_tryEvalSilent and an R level
// tryCatch, and failures are returned to XLA as FFI errors.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include "xla/ffi/api/ffi.h"

#include "pjrt_c_api.h"
#include "pjrt_c_api_ffi_extension.h"

#define R_NO_REMAP
#define STRICT_R_HEADERS
#include <Rinternals.h>

extern "C" const PJRT_Api* jaxr_api();
extern "C" void jaxr_set_pump(void (*)());

namespace {

namespace ffi = xla::ffi;

struct Request {
  uint64_t index;
  ffi::RemainingArgs* args;
  ffi::RemainingRets* rets;
  std::string error;
  bool taken = false;
  bool done = false;
};

std::mutex g_mu;
std::condition_variable g_cv;
std::deque<Request*> g_queue;
std::thread::id g_main;
long g_calls_main = 0;
long g_calls_worker = 0;
bool g_servicing = false;  // the main thread is running a callback (guarded by g_mu)

// A worker gives up when the main thread has not picked its request up for this long
// while not running any callback (the R session is not waiting on XLA, so it would
// never service the queue).
// ponytail: fixed timeout; make it configurable if R legitimately stays away from
// XLA waits for longer while a program with callbacks is running.
constexpr auto kPickupTimeout = std::chrono::seconds(60);

void set_servicing(bool v) {
  std::lock_guard<std::mutex> lk(g_mu);
  g_servicing = v;
}

// ---- element conversion ----------------------------------------------------------------

uint16_t f32_to_f16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t mant = x & 0x007fffffu;
  int32_t exp = (int32_t)((x >> 23) & 0xffu);
  if (exp == 0xff) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
  int32_t e = exp - 127 + 15;
  if (e >= 0x1f) return (uint16_t)(sign | 0x7c00u);
  if (e <= 0) {
    if (e < -10) return (uint16_t)sign;
    mant |= 0x00800000u;
    uint32_t shift = (uint32_t)(14 - e);
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1u);
    uint32_t mid = 1u << (shift - 1u);
    if (rem > mid || (rem == mid && (half & 1u))) half++;
    return (uint16_t)(sign | half);
  }
  uint32_t half = sign | ((uint32_t)e << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;
  return (uint16_t)half;
}

float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {
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

uint16_t f32_to_bf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if ((x & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((x >> 16) | 0x40u);
  x += 0x7fffu + ((x >> 16) & 1u);
  return (uint16_t)(x >> 16);
}

float bf16_to_f32(uint16_t b) {
  uint32_t x = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

template <typename T>
T load(const void* p, size_t i) {
  T v;
  std::memcpy(&v, static_cast<const uint8_t*>(p) + i * sizeof(T), sizeof(T));
  return v;
}

template <typename T>
void store(void* p, size_t i, T v) {
  std::memcpy(static_cast<uint8_t*>(p) + i * sizeof(T), &v, sizeof(T));
}

// Device buffer -> flat R vector in row-major order (logical, double or complex).
// Returns R_NilValue for an unsupported element type.
SEXP buffer_to_r(const ffi::AnyBuffer& b) {
  size_t n = b.element_count();
  const void* p = b.untyped_data();
  ffi::DataType t = b.element_type();
  SEXP out;
  switch (t) {
    case ffi::PRED:
      out = Rf_allocVector(LGLSXP, (R_xlen_t)n);
      for (size_t i = 0; i < n; i++) LOGICAL(out)[i] = load<uint8_t>(p, i) != 0;
      return out;
    case ffi::C64:
    case ffi::C128:
      out = Rf_allocVector(CPLXSXP, (R_xlen_t)n);
      for (size_t i = 0; i < n; i++) {
        Rcomplex& c = COMPLEX(out)[i];
        if (t == ffi::C64) {
          c.r = load<float>(p, 2 * i);
          c.i = load<float>(p, 2 * i + 1);
        } else {
          c.r = load<double>(p, 2 * i);
          c.i = load<double>(p, 2 * i + 1);
        }
      }
      return out;
    default:
      break;
  }
  out = Rf_allocVector(REALSXP, (R_xlen_t)n);
  double* o = REAL(out);
  for (size_t i = 0; i < n; i++) {
    switch (t) {
      case ffi::S8: o[i] = load<int8_t>(p, i); break;
      case ffi::S16: o[i] = load<int16_t>(p, i); break;
      case ffi::S32: o[i] = load<int32_t>(p, i); break;
      case ffi::S64: o[i] = (double)load<int64_t>(p, i); break;
      case ffi::U8: o[i] = load<uint8_t>(p, i); break;
      case ffi::U16: o[i] = load<uint16_t>(p, i); break;
      case ffi::U32: o[i] = load<uint32_t>(p, i); break;
      case ffi::U64: o[i] = (double)load<uint64_t>(p, i); break;
      case ffi::F16: o[i] = f16_to_f32(load<uint16_t>(p, i)); break;
      case ffi::BF16: o[i] = bf16_to_f32(load<uint16_t>(p, i)); break;
      case ffi::F32: o[i] = load<float>(p, i); break;
      case ffi::F64: o[i] = load<double>(p, i); break;
      default: return R_NilValue;
    }
  }
  return out;
}

double num_at(SEXP v, R_xlen_t i) {
  switch (TYPEOF(v)) {
    case REALSXP: return REAL(v)[i];
    case INTSXP: return INTEGER(v)[i];
    case LGLSXP: return LOGICAL(v)[i];
    default: return 0;
  }
}

// Flat R vector (row-major, already validated in R) -> result buffer.
std::string r_to_buffer(SEXP v, ffi::AnyBuffer b) {
  size_t n = b.element_count();
  if ((size_t)XLENGTH(v) != n) return "internal: callback result has the wrong length";
  void* p = b.untyped_data();
  ffi::DataType t = b.element_type();
  if (t == ffi::C64 || t == ffi::C128) {
    if (TYPEOF(v) != CPLXSXP) return "internal: complex callback result expected";
    for (size_t i = 0; i < n; i++) {
      Rcomplex c = COMPLEX(v)[i];
      if (t == ffi::C64) {
        store<float>(p, 2 * i, (float)c.r);
        store<float>(p, 2 * i + 1, (float)c.i);
      } else {
        store<double>(p, 2 * i, c.r);
        store<double>(p, 2 * i + 1, c.i);
      }
    }
    return "";
  }
  if (TYPEOF(v) != REALSXP && TYPEOF(v) != INTSXP && TYPEOF(v) != LGLSXP)
    return "internal: numeric callback result expected";
  for (size_t i = 0; i < n; i++) {
    double x = num_at(v, (R_xlen_t)i);
    switch (t) {
      case ffi::PRED: store<uint8_t>(p, i, x != 0); break;
      case ffi::S8: store<int8_t>(p, i, (int8_t)(int64_t)x); break;
      case ffi::S16: store<int16_t>(p, i, (int16_t)(int64_t)x); break;
      case ffi::S32: store<int32_t>(p, i, (int32_t)(int64_t)x); break;
      case ffi::S64: store<int64_t>(p, i, (int64_t)x); break;
      case ffi::U8: store<uint8_t>(p, i, (uint8_t)(uint64_t)x); break;
      case ffi::U16: store<uint16_t>(p, i, (uint16_t)(uint64_t)x); break;
      case ffi::U32: store<uint32_t>(p, i, (uint32_t)(uint64_t)x); break;
      case ffi::U64: store<uint64_t>(p, i, (uint64_t)x); break;
      case ffi::F16: store<uint16_t>(p, i, f32_to_f16((float)x)); break;
      case ffi::BF16: store<uint16_t>(p, i, f32_to_bf16((float)x)); break;
      case ffi::F32: store<float>(p, i, (float)x); break;
      case ffi::F64: store<double>(p, i, x); break;
      default: return "unsupported result element type in a host callback";
    }
  }
  return "";
}

// ---- servicing (R main thread only) ----------------------------------------------------

// Call jaxR:::callback_dispatch(index, inputs) and copy its results into the result
// buffers. Token operands and results (ordered effects) carry no data and are skipped.
std::string service_in_r(Request* r) {
  ffi::RemainingArgs& args = *r->args;
  ffi::RemainingRets& rets = *r->rets;
  int nprot = 0;
  SEXP ins = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)args.size()));
  nprot++;
  R_xlen_t k = 0;
  for (size_t i = 0; i < args.size(); i++) {
    auto b = args.get<ffi::AnyBuffer>(i);
    if (!b.has_value()) {
      UNPROTECT(nprot);
      return "host callback: cannot decode operand";
    }
    if (b->element_type() == ffi::TOKEN) continue;
    SEXP v = PROTECT(buffer_to_r(*b));
    if (v == R_NilValue) {
      UNPROTECT(nprot + 1);
      return "unsupported operand element type in a host callback";
    }
    auto dims = b->dimensions();
    SEXP shape = PROTECT(Rf_allocVector(INTSXP, (R_xlen_t)dims.size()));
    for (size_t d = 0; d < dims.size(); d++) INTEGER(shape)[d] = (int)dims[d];
    Rf_setAttrib(v, Rf_install("shape"), shape);
    SET_VECTOR_ELT(ins, k++, v);
    UNPROTECT(2);
  }
  ins = PROTECT(Rf_xlengthgets(ins, k));
  nprot++;
  SEXP ns = PROTECT(R_FindNamespace(Rf_mkString("jaxR")));
  nprot++;
  SEXP idx = PROTECT(Rf_ScalarReal((double)r->index));
  nprot++;
  SEXP fn = Rf_install("callback_dispatch");
  SEXP call = PROTECT(Rf_lang3(fn, idx, ins));
  nprot++;
  int failed = 0;
  SEXP res = R_tryEvalSilent(call, ns, &failed);
  if (failed || res == nullptr) {
    UNPROTECT(nprot);
    return std::string("host callback failed: ") + R_curErrorBuf();
  }
  PROTECT(res);
  nprot++;
  // callback_dispatch returns a character scalar (error message) or a list of flat
  // result vectors, one per non-token result.
  if (TYPEOF(res) == STRSXP) {
    std::string msg = Rf_length(res) ? CHAR(STRING_ELT(res, 0)) : "host callback failed";
    UNPROTECT(nprot);
    return msg;
  }
  std::string err;
  R_xlen_t j = 0;
  for (size_t i = 0; i < rets.size() && err.empty(); i++) {
    auto b = rets.get<ffi::AnyBuffer>(i);
    if (!b.has_value()) {
      err = "host callback: cannot decode result";
      break;
    }
    if ((*b)->element_type() == ffi::TOKEN) continue;
    if (j >= XLENGTH(res)) {
      err = "internal: callback returned too few results";
      break;
    }
    err = r_to_buffer(VECTOR_ELT(res, j++), **b);
  }
  UNPROTECT(nprot);
  return err;
}

void pump() {
  for (;;) {
    Request* r;
    {
      std::lock_guard<std::mutex> lk(g_mu);
      if (g_queue.empty()) return;
      r = g_queue.front();
      g_queue.pop_front();
      r->taken = true;
    }
    g_cv.notify_all();
    set_servicing(true);
    std::string err = service_in_r(r);
    {
      std::lock_guard<std::mutex> lk(g_mu);
      g_servicing = false;
      r->error = err;
      r->done = true;
    }
    g_cv.notify_all();
  }
}

ffi::Error callback_handler(ffi::RemainingArgs args, ffi::RemainingRets rets, uint64_t index) {
  Request req;
  req.index = index;
  req.args = &args;
  req.rets = &rets;
  if (std::this_thread::get_id() == g_main) {
    g_calls_main++;
    set_servicing(true);
    req.error = service_in_r(&req);
    set_servicing(false);
  } else {
    std::unique_lock<std::mutex> lk(g_mu);
    g_calls_worker++;
    g_queue.push_back(&req);
    while (!req.taken && !g_cv.wait_for(lk, kPickupTimeout, [&] { return req.taken; })) {
      if (g_servicing) continue;  // the main thread is busy with another callback
      for (auto it = g_queue.begin(); it != g_queue.end(); ++it) {
        if (*it == &req) {
          g_queue.erase(it);
          break;
        }
      }
      return ffi::Error(ffi::ErrorCode::kDeadlineExceeded,
                        "host callback was not serviced by the R main thread");
    }
    g_cv.wait(lk, [&] { return req.done; });
  }
  if (!req.error.empty()) return ffi::Error(ffi::ErrorCode::kUnknown, req.error);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(kRCallback, callback_handler,
                       ffi::Ffi::Bind().RemainingArgs().RemainingRets().Attr<uint64_t>("index"));

std::string take_error(const PJRT_Api* api, PJRT_Error* err) {
  PJRT_Error_Message_Args m;
  std::memset(&m, 0, sizeof m);
  m.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  m.error = err;
  api->PJRT_Error_Message(&m);
  std::string msg(m.message, m.message_size);
  PJRT_Error_Destroy_Args d;
  std::memset(&d, 0, sizeof d);
  d.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  d.error = err;
  api->PJRT_Error_Destroy(&d);
  return msg;
}

std::string register_handler() {
  const PJRT_Api* api = jaxr_api();
  if (!api) return "no PJRT plugin is loaded";
  PJRT_FFI_Extension* ext = nullptr;
  for (PJRT_Extension_Base* e = api->extension_start; e; e = e->next) {
    if (e->type == PJRT_Extension_Type_FFI) ext = reinterpret_cast<PJRT_FFI_Extension*>(e);
  }
  if (!ext) return "the PJRT plugin has no FFI extension";
  if (ext->base.struct_size < PJRT_FFI_Extension_STRUCT_SIZE || !ext->register_handler)
    return "the FFI extension of the PJRT plugin cannot register handlers";
  static const char kTarget[] = "jaxr_r_callback";
  static const char kPlatform[] = "Host";
  PJRT_FFI_Register_Handler_Args a;
  std::memset(&a, 0, sizeof a);
  a.struct_size = PJRT_FFI_Register_Handler_Args_STRUCT_SIZE;
  a.target_name = kTarget;
  a.target_name_size = sizeof(kTarget) - 1;
  a.handler = reinterpret_cast<void*>(kRCallback);
  a.platform_name = kPlatform;
  a.platform_name_size = sizeof(kPlatform) - 1;
  a.traits = static_cast<PJRT_FFI_Handler_TraitsBits>(0);
  if (PJRT_Error* err = ext->register_handler(&a)) return take_error(api, err);
  return std::string();
}

}  // namespace

// Called from R (R/callbacks.R) right after the plugin loads, on the R main thread.
// Returns "" on success, else the failure message.
extern "C" SEXP jaxr_callbacks_register() {
  static bool done = false;
  std::string msg;
  if (!done) {
    g_main = std::this_thread::get_id();
    msg = register_handler();
    done = msg.empty();
    if (done) jaxr_set_pump(&pump);
  }
  return Rf_mkString(msg.c_str());
}

// Service queued callbacks now (effects_barrier).
extern "C" SEXP jaxr_callbacks_pump() {
  if (std::this_thread::get_id() == g_main) pump();
  return R_NilValue;
}

// Number of handler invocations on the main thread and on XLA worker threads.
extern "C" SEXP jaxr_callbacks_stats() {
  SEXP out = Rf_allocVector(REALSXP, 2);
  std::lock_guard<std::mutex> lk(g_mu);
  REAL(out)[0] = (double)g_calls_main;
  REAL(out)[1] = (double)g_calls_worker;
  return out;
}
