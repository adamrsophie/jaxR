// XLA FFI handlers for the LAPACK custom call targets of jaxlib's CPU backend
// (port of jaxlib/cpu/lapack_kernels.cc and jaxlib/cpu/tridiagonal_solve_kernels.cc).
//
// JAX's CPU lowerings of the linear algebra primitives call typed FFI targets named
// lapack_<s|d|c|z><routine>_ffi. The prebuilt PJRT CPU plugin does not ship them, so
// they are implemented here on R's LAPACK and registered through the PJRT FFI
// extension. R's LAPACK has double and double complex routines only: single precision
// buffers are widened to double, factored, and narrowed again.
//
// Handlers run on XLA worker threads. They must never call the R API, and they must
// never reach LAPACK's xerbla (R's version raises an R error), so every call passes
// valid arguments (lda >= 1) and empty problems return before LAPACK is called.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"

#include "pjrt_c_api.h"
#include "pjrt_c_api_ffi_extension.h"

#define USE_FC_LEN_T
#define R_NO_REMAP
#define STRICT_R_HEADERS
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#ifndef FCONE
#define FCONE
#endif

extern "C" const PJRT_Api* jaxr_api();

namespace {

namespace ffi = xla::ffi;

using cplx = std::complex<double>;
using Buf = ffi::AnyBuffer;
using Out = ffi::Result<ffi::AnyBuffer>;
using IntOut = ffi::ResultBuffer<ffi::S32>;

inline Rcomplex* rc(cplx* p) { return reinterpret_cast<Rcomplex*>(p); }

inline bool is_real_type(ffi::DataType t) { return t == ffi::F32 || t == ffi::F64; }
inline bool is_cplx_type(ffi::DataType t) { return t == ffi::C64 || t == ffi::C128; }

ffi::Error unsupported_dtype() {
  return ffi::Error::InvalidArgument(
      "jaxR LAPACK kernels support float32, float64, complex64 and complex128 only");
}

// ---- element conversion between XLA buffers and LAPACK work arrays -------------------

// Read n elements starting at element `off` of a real (f32/f64) buffer.
inline void load(const Buf& b, int64_t off, int64_t n, double* dst) {
  if (n <= 0) return;
  if (b.element_type() == ffi::F64) {
    std::memcpy(dst, static_cast<const double*>(b.untyped_data()) + off, n * sizeof(double));
  } else {
    const float* s = static_cast<const float*>(b.untyped_data()) + off;
    for (int64_t i = 0; i < n; ++i) dst[i] = s[i];
  }
}

// Read n elements starting at element `off` of a complex (c64/c128) buffer.
inline void load(const Buf& b, int64_t off, int64_t n, cplx* dst) {
  if (n <= 0) return;
  if (b.element_type() == ffi::C128) {
    std::memcpy(static_cast<void*>(dst), static_cast<const cplx*>(b.untyped_data()) + off,
                n * sizeof(cplx));
  } else {
    const std::complex<float>* s = static_cast<const std::complex<float>*>(b.untyped_data()) + off;
    for (int64_t i = 0; i < n; ++i) dst[i] = cplx(s[i].real(), s[i].imag());
  }
}

inline void store(const Buf& b, int64_t off, int64_t n, const double* src) {
  if (n <= 0) return;
  if (b.element_type() == ffi::F64) {
    std::memcpy(static_cast<double*>(b.untyped_data()) + off, src, n * sizeof(double));
  } else {
    float* d = static_cast<float*>(b.untyped_data()) + off;
    for (int64_t i = 0; i < n; ++i) d[i] = static_cast<float>(src[i]);
  }
}

inline void store(const Buf& b, int64_t off, int64_t n, const cplx* src) {
  if (n <= 0) return;
  if (b.element_type() == ffi::C128) {
    std::memcpy(static_cast<void*>(static_cast<cplx*>(b.untyped_data()) + off), src,
                n * sizeof(cplx));
  } else {
    std::complex<float>* d = static_cast<std::complex<float>*>(b.untyped_data()) + off;
    for (int64_t i = 0; i < n; ++i)
      d[i] = std::complex<float>(static_cast<float>(src[i].real()), static_cast<float>(src[i].imag()));
  }
}

// Port of jaxlib/ffi_helpers.h:SplitBatch2D.
struct Dims2 {
  int64_t batch, rows, cols;
};

inline bool split2d(ffi::Span<const int64_t> dims, Dims2* out) {
  if (dims.size() < 2) return false;
  int64_t batch = 1;
  for (size_t i = 0; i + 2 < dims.size(); ++i) batch *= dims[i];
  out->batch = batch;
  out->rows = dims[dims.size() - 2];
  out->cols = dims[dims.size() - 1];
  return true;
}

ffi::Error rank_error() {
  return ffi::Error::InvalidArgument("Argument must have at least 2 dimensions");
}

inline int lead(int64_t n) { return static_cast<int>(std::max<int64_t>(1, n)); }

// C++ exceptions (for example std::bad_alloc from a work array) must not unwind into
// XLA: they become FFI errors.
template <typename F>
ffi::Error guarded(F&& f) {
  try {
    return f();
  } catch (const std::exception& e) {
    return ffi::Error::Internal(std::string("jaxR LAPACK kernel: ") + e.what());
  } catch (...) {
    return ffi::Error::Internal("jaxR LAPACK kernel: unknown C++ exception");
  }
}

// Run a templated kernel on the work type matching the dtype of `buf`.
#define JAXR_BY_DTYPE(buf, fn, ...)                                              \
  guarded([&]() -> ffi::Error {                                                  \
    return is_cplx_type((buf).element_type())   ? fn<cplx>(__VA_ARGS__)          \
           : is_real_type((buf).element_type()) ? fn<double>(__VA_ARGS__)        \
                                                : unsupported_dtype();           \
  })

// Character attributes reach LAPACK unchanged; an invalid one would make LAPACK call
// xerbla, so they are validated first.
inline bool one_of(uint8_t c, const char* set) { return c != 0 && std::strchr(set, c) != nullptr; }

// Buffer checks at the boundary between HLO text and native code.
inline ffi::DataType real_of(ffi::DataType t) {
  return t == ffi::C64 ? ffi::F32 : t == ffi::C128 ? ffi::F64 : t;
}
inline ffi::DataType cplx_of(ffi::DataType t) {
  return t == ffi::F32 ? ffi::C64 : t == ffi::F64 ? ffi::C128 : t;
}
inline bool has(const Buf& b, ffi::DataType t, int64_t count) {
  return b.element_type() == t && static_cast<int64_t>(b.element_count()) == count;
}
inline bool has(const ffi::Buffer<ffi::S32>& b, int64_t count) {
  return static_cast<int64_t>(b.element_count()) == count;
}
inline bool fits_int(const Dims2& d) { return d.rows <= INT32_MAX && d.cols <= INT32_MAX; }

#define JAXR_REQUIRE(cond)                                                             \
  do {                                                                                 \
    if (!(cond))                                                                       \
      return ffi::Error::InvalidArgument(                                              \
          "jaxR LAPACK kernel: unexpected buffer type or shape (" #cond ")");          \
  } while (0)

// LAPACK workspaces, sized by a workspace query on first use and reused for every
// matrix of the batch.
template <typename W>
struct Scratch {
  std::vector<W> work;
  std::vector<double> rwork;
  std::vector<int> iwork;
};

inline int lwork_of(double q) { return std::max(1, static_cast<int>(q)); }
inline int lwork_of(cplx q) { return std::max(1, static_cast<int>(q.real())); }

// ---- LAPACK entry points overloaded on the work type ---------------------------------

extern "C" void F77_NAME(zgtsv)(const int* n, const int* nrhs, Rcomplex* dl, Rcomplex* d,
                                Rcomplex* du, Rcomplex* b, const int* ldb, int* info);

inline void xtrsm(char side, char uplo, char trans, char diag, int m, int n, double* a, int lda,
                  double* b, int ldb) {
  double alpha = 1.0;
  F77_CALL(dtrsm)(&side, &uplo, &trans, &diag, &m, &n, &alpha, a, &lda, b, &ldb
                  FCONE FCONE FCONE FCONE);
}
inline void xtrsm(char side, char uplo, char trans, char diag, int m, int n, cplx* a, int lda,
                  cplx* b, int ldb) {
  cplx alpha(1.0, 0.0);
  F77_CALL(ztrsm)(&side, &uplo, &trans, &diag, &m, &n, rc(&alpha), rc(a), &lda, rc(b), &ldb
                  FCONE FCONE FCONE FCONE);
}

inline void xpotrf(char uplo, int n, double* a, int lda, int* info) {
  F77_CALL(dpotrf)(&uplo, &n, a, &lda, info FCONE);
}
inline void xpotrf(char uplo, int n, cplx* a, int lda, int* info) {
  F77_CALL(zpotrf)(&uplo, &n, rc(a), &lda, info FCONE);
}

inline void xgetrf(int m, int n, double* a, int lda, int* ipiv, int* info) {
  F77_CALL(dgetrf)(&m, &n, a, &lda, ipiv, info);
}
inline void xgetrf(int m, int n, cplx* a, int lda, int* ipiv, int* info) {
  F77_CALL(zgetrf)(&m, &n, rc(a), &lda, ipiv, info);
}

inline void xgeqrf(int m, int n, double* a, int lda, double* tau, Scratch<double>& s, int* info) {
  if (s.work.empty()) {
    double q = 0;
    int lw = -1;
    F77_CALL(dgeqrf)(&m, &n, a, &lda, tau, &q, &lw, info);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(dgeqrf)(&m, &n, a, &lda, tau, s.work.data(), &lw, info);
}
inline void xgeqrf(int m, int n, cplx* a, int lda, cplx* tau, Scratch<cplx>& s, int* info) {
  if (s.work.empty()) {
    cplx q = 0;
    int lw = -1;
    F77_CALL(zgeqrf)(&m, &n, rc(a), &lda, rc(tau), rc(&q), &lw, info);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(zgeqrf)(&m, &n, rc(a), &lda, rc(tau), rc(s.work.data()), &lw, info);
}

inline void xgeqp3(int m, int n, double* a, int lda, int* jpvt, double* tau, Scratch<double>& s,
                   int* info) {
  if (s.work.empty()) {
    double q = 0;
    int lw = -1;
    F77_CALL(dgeqp3)(&m, &n, a, &lda, jpvt, tau, &q, &lw, info);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(dgeqp3)(&m, &n, a, &lda, jpvt, tau, s.work.data(), &lw, info);
}
inline void xgeqp3(int m, int n, cplx* a, int lda, int* jpvt, cplx* tau, Scratch<cplx>& s,
                   int* info) {
  if (s.work.empty()) {
    s.rwork.resize(static_cast<size_t>(std::max(1, 2 * n)));
    cplx q = 0;
    int lw = -1;
    F77_CALL(zgeqp3)(&m, &n, rc(a), &lda, jpvt, rc(tau), rc(&q), &lw, s.rwork.data(), info);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(zgeqp3)(&m, &n, rc(a), &lda, jpvt, rc(tau), rc(s.work.data()), &lw, s.rwork.data(),
                   info);
}

inline void xorgqr(int m, int n, int k, double* a, int lda, double* tau, Scratch<double>& s,
                   int* info) {
  if (s.work.empty()) {
    double q = 0;
    int lw = -1;
    F77_CALL(dorgqr)(&m, &n, &k, a, &lda, tau, &q, &lw, info);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(dorgqr)(&m, &n, &k, a, &lda, tau, s.work.data(), &lw, info);
}
inline void xorgqr(int m, int n, int k, cplx* a, int lda, cplx* tau, Scratch<cplx>& s, int* info) {
  if (s.work.empty()) {
    cplx q = 0;
    int lw = -1;
    F77_CALL(zungqr)(&m, &n, &k, rc(a), &lda, rc(tau), rc(&q), &lw, info);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(zungqr)(&m, &n, &k, rc(a), &lda, rc(tau), rc(s.work.data()), &lw, info);
}

// `transpose` selects Q^T for real and Q^H for complex matrices.
inline void xormqr(char side, bool transpose, int m, int n, int k, double* a, int lda, double* tau,
                   double* c, int ldc, Scratch<double>& s, int* info) {
  char trans = transpose ? 'T' : 'N';
  if (s.work.empty()) {
    double q = 0;
    int lw = -1;
    F77_CALL(dormqr)(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, &q, &lw, info FCONE FCONE);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(dormqr)(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, s.work.data(), &lw, info
                   FCONE FCONE);
}
inline void xormqr(char side, bool transpose, int m, int n, int k, cplx* a, int lda, cplx* tau,
                   cplx* c, int ldc, Scratch<cplx>& s, int* info) {
  char trans = transpose ? 'C' : 'N';
  if (s.work.empty()) {
    cplx q = 0;
    int lw = -1;
    F77_CALL(zunmqr)(&side, &trans, &m, &n, &k, rc(a), &lda, rc(tau), rc(c), &ldc, rc(&q), &lw,
                     info FCONE FCONE);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(zunmqr)(&side, &trans, &m, &n, &k, rc(a), &lda, rc(tau), rc(c), &ldc,
                   rc(s.work.data()), &lw, info FCONE FCONE);
}

// Port of jaxlib/cpu/lapack_kernels.cc:svd::GetRealWorkspaceSize.
inline int64_t gesdd_rwork_size(int64_t m, int64_t n, char jobz) {
  const int64_t mn = std::min(m, n), mx = std::max(m, n);
  if (jobz == 'N') return 7 * mn;
  return std::max(5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn);
}

inline void xgesdd(char jobz, int m, int n, double* a, int lda, double* sv, double* u, int ldu,
                   double* vt, int ldvt, Scratch<double>& s, int* info) {
  if (s.work.empty()) {
    s.iwork.resize(static_cast<size_t>(std::max(1, 8 * std::min(m, n))));
    double q = 0;
    int lw = -1;
    F77_CALL(dgesdd)(&jobz, &m, &n, a, &lda, sv, u, &ldu, vt, &ldvt, &q, &lw, s.iwork.data(),
                     info FCONE);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(dgesdd)(&jobz, &m, &n, a, &lda, sv, u, &ldu, vt, &ldvt, s.work.data(), &lw,
                   s.iwork.data(), info FCONE);
}
inline void xgesdd(char jobz, int m, int n, cplx* a, int lda, double* sv, cplx* u, int ldu,
                   cplx* vt, int ldvt, Scratch<cplx>& s, int* info) {
  if (s.work.empty()) {
    s.iwork.resize(static_cast<size_t>(std::max(1, 8 * std::min(m, n))));
    s.rwork.resize(static_cast<size_t>(std::max<int64_t>(1, gesdd_rwork_size(m, n, jobz))));
    cplx q = 0;
    int lw = -1;
    F77_CALL(zgesdd)(&jobz, &m, &n, rc(a), &lda, sv, rc(u), &ldu, rc(vt), &ldvt, rc(&q), &lw,
                     s.rwork.data(), s.iwork.data(), info FCONE);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(zgesdd)(&jobz, &m, &n, rc(a), &lda, sv, rc(u), &ldu, rc(vt), &ldvt, rc(s.work.data()),
                   &lw, s.rwork.data(), s.iwork.data(), info FCONE);
}

inline void xgesvd(char job, int m, int n, double* a, int lda, double* sv, double* u, int ldu,
                   double* vt, int ldvt, Scratch<double>& s, int* info) {
  if (s.work.empty()) {
    double q = 0;
    int lw = -1;
    F77_CALL(dgesvd)(&job, &job, &m, &n, a, &lda, sv, u, &ldu, vt, &ldvt, &q, &lw, info
                     FCONE FCONE);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(dgesvd)(&job, &job, &m, &n, a, &lda, sv, u, &ldu, vt, &ldvt, s.work.data(), &lw, info
                   FCONE FCONE);
}
inline void xgesvd(char job, int m, int n, cplx* a, int lda, double* sv, cplx* u, int ldu,
                   cplx* vt, int ldvt, Scratch<cplx>& s, int* info) {
  if (s.work.empty()) {
    s.rwork.resize(static_cast<size_t>(std::max(1, 5 * std::min(m, n))));
    cplx q = 0;
    int lw = -1;
    F77_CALL(zgesvd)(&job, &job, &m, &n, rc(a), &lda, sv, rc(u), &ldu, rc(vt), &ldvt, rc(&q), &lw,
                     s.rwork.data(), info FCONE FCONE);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(zgesvd)(&job, &job, &m, &n, rc(a), &lda, sv, rc(u), &ldu, rc(vt), &ldvt,
                   rc(s.work.data()), &lw, s.rwork.data(), info FCONE FCONE);
}

inline void xheevd(char jobz, char uplo, int n, double* a, int lda, double* w, Scratch<double>& s,
                   int* info) {
  if (s.work.empty()) {
    double q = 0;
    int iq = 0, lw = -1, liw = -1;
    F77_CALL(dsyevd)(&jobz, &uplo, &n, a, &lda, w, &q, &lw, &iq, &liw, info FCONE FCONE);
    s.work.resize(lwork_of(q));
    s.iwork.resize(static_cast<size_t>(std::max(1, iq)));
  }
  int lw = static_cast<int>(s.work.size()), liw = static_cast<int>(s.iwork.size());
  F77_CALL(dsyevd)(&jobz, &uplo, &n, a, &lda, w, s.work.data(), &lw, s.iwork.data(), &liw, info
                   FCONE FCONE);
}
inline void xheevd(char jobz, char uplo, int n, cplx* a, int lda, double* w, Scratch<cplx>& s,
                   int* info) {
  if (s.work.empty()) {
    cplx q = 0;
    double rq = 0;
    int iq = 0, lw = -1, lrw = -1, liw = -1;
    F77_CALL(zheevd)(&jobz, &uplo, &n, rc(a), &lda, w, rc(&q), &lw, &rq, &lrw, &iq, &liw, info
                     FCONE FCONE);
    s.work.resize(lwork_of(q));
    s.rwork.resize(static_cast<size_t>(lwork_of(rq)));
    s.iwork.resize(static_cast<size_t>(std::max(1, iq)));
  }
  int lw = static_cast<int>(s.work.size()), lrw = static_cast<int>(s.rwork.size()),
      liw = static_cast<int>(s.iwork.size());
  F77_CALL(zheevd)(&jobz, &uplo, &n, rc(a), &lda, w, rc(s.work.data()), &lw, s.rwork.data(), &lrw,
                   s.iwork.data(), &liw, info FCONE FCONE);
}

inline void xgehrd(int n, int ilo, int ihi, double* a, int lda, double* tau, Scratch<double>& s,
                   int* info) {
  if (s.work.empty()) {
    double q = 0;
    int lw = -1;
    F77_CALL(dgehrd)(&n, &ilo, &ihi, a, &lda, tau, &q, &lw, info);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(dgehrd)(&n, &ilo, &ihi, a, &lda, tau, s.work.data(), &lw, info);
}
inline void xgehrd(int n, int ilo, int ihi, cplx* a, int lda, cplx* tau, Scratch<cplx>& s,
                   int* info) {
  if (s.work.empty()) {
    cplx q = 0;
    int lw = -1;
    F77_CALL(zgehrd)(&n, &ilo, &ihi, rc(a), &lda, rc(tau), rc(&q), &lw, info);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(zgehrd)(&n, &ilo, &ihi, rc(a), &lda, rc(tau), rc(s.work.data()), &lw, info);
}

inline void xhetrd(char uplo, int n, double* a, int lda, double* d, double* e, double* tau,
                   Scratch<double>& s, int* info) {
  if (s.work.empty()) {
    double q = 0;
    int lw = -1;
    F77_CALL(dsytrd)(&uplo, &n, a, &lda, d, e, tau, &q, &lw, info FCONE);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(dsytrd)(&uplo, &n, a, &lda, d, e, tau, s.work.data(), &lw, info FCONE);
}
inline void xhetrd(char uplo, int n, cplx* a, int lda, double* d, double* e, cplx* tau,
                   Scratch<cplx>& s, int* info) {
  if (s.work.empty()) {
    cplx q = 0;
    int lw = -1;
    F77_CALL(zhetrd)(&uplo, &n, rc(a), &lda, d, e, rc(tau), rc(&q), &lw, info FCONE);
    s.work.resize(lwork_of(q));
  }
  int lw = static_cast<int>(s.work.size());
  F77_CALL(zhetrd)(&uplo, &n, rc(a), &lda, d, e, rc(tau), rc(s.work.data()), &lw, info FCONE);
}

inline void xgtsv(int n, int nrhs, double* dl, double* d, double* du, double* b, int ldb,
                  int* info) {
  F77_CALL(dgtsv)(&n, &nrhs, dl, d, du, b, &ldb, info);
}
inline void xgtsv(int n, int nrhs, cplx* dl, cplx* d, cplx* du, cplx* b, int ldb, int* info) {
  F77_CALL(zgtsv)(&n, &nrhs, rc(dl), rc(d), rc(du), rc(b), &ldb, info);
}

// ---- kernels -------------------------------------------------------------------------
//
// Every kernel loads one matrix of the batch into a double (or double complex) work
// array, calls LAPACK and stores the results. Inputs are fully read before any output
// is written because aliased operands share memory with their outputs.

// Port of jaxlib/cpu/lapack_kernels.cc:TriMatrixEquationSolver::Kernel.
template <typename W>
ffi::Error trsm_impl(Buf x, Buf y, Out y_out, uint8_t side, uint8_t uplo, uint8_t trans_x,
                     uint8_t diag) {
  Dims2 d;
  if (!split2d(y.dimensions(), &d)) return rank_error();
  JAXR_REQUIRE(fits_int(d));
  JAXR_REQUIRE(one_of(side, "LR") && one_of(uplo, "LU") && one_of(trans_x, "NTC") && one_of(diag, "NU"));
  const int64_t ld = side == 'L' ? d.rows : d.cols;
  const int64_t x_step = ld * ld, y_step = d.rows * d.cols;
  JAXR_REQUIRE(has(x, y.element_type(), d.batch * x_step));
  JAXR_REQUIRE(has(*y_out, y.element_type(), d.batch * y_step));
  std::vector<W> a(static_cast<size_t>(x_step)), b(static_cast<size_t>(y_step));
  for (int64_t i = 0; i < d.batch; ++i) {
    load(x, i * x_step, x_step, a.data());
    load(y, i * y_step, y_step, b.data());
    if (y_step > 0)
      xtrsm(static_cast<char>(side), static_cast<char>(uplo), static_cast<char>(trans_x),
            static_cast<char>(diag), static_cast<int>(d.rows), static_cast<int>(d.cols), a.data(),
            lead(ld), b.data(), lead(d.rows));
    store(*y_out, i * y_step, y_step, b.data());
  }
  return ffi::Error::Success();
}

ffi::Error trsm(Buf x, Buf y, Out y_out, uint8_t side, uint8_t uplo, uint8_t trans_x,
                uint8_t diag) {
  return JAXR_BY_DTYPE(y, trsm_impl, x, y, y_out, side, uplo, trans_x, diag);
}

// Port of jaxlib/cpu/lapack_kernels.cc:LuDecomposition::Kernel.
template <typename W>
ffi::Error getrf_impl(Buf x, Out x_out, IntOut ipiv, IntOut info) {
  Dims2 d;
  if (!split2d(x.dimensions(), &d)) return rank_error();
  JAXR_REQUIRE(fits_int(d));
  const int64_t step = d.rows * d.cols;
  const int64_t ipiv_step = std::min(d.rows, d.cols);
  JAXR_REQUIRE(has(*x_out, x.element_type(), d.batch * step));
  JAXR_REQUIRE(has(*ipiv, d.batch * ipiv_step) && has(*info, d.batch));
  int32_t* ipiv_data = ipiv->typed_data();
  int32_t* info_data = info->typed_data();
  std::vector<W> a(static_cast<size_t>(step));
  for (int64_t i = 0; i < d.batch; ++i) {
    int info_v = 0;
    load(x, i * step, step, a.data());
    if (step > 0)
      xgetrf(static_cast<int>(d.rows), static_cast<int>(d.cols), a.data(), lead(d.rows),
             ipiv_data + i * ipiv_step, &info_v);
    store(*x_out, i * step, step, a.data());
    info_data[i] = info_v;
  }
  return ffi::Error::Success();
}

ffi::Error getrf(Buf x, Out x_out, IntOut ipiv, IntOut info) {
  return JAXR_BY_DTYPE(x, getrf_impl, x, x_out, ipiv, info);
}

// Port of jaxlib/cpu/lapack_kernels.cc:QrFactorization::Kernel.
template <typename W>
ffi::Error geqrf_impl(Buf x, Out x_out, Out tau) {
  Dims2 d;
  if (!split2d(x.dimensions(), &d)) return rank_error();
  JAXR_REQUIRE(fits_int(d));
  const int64_t step = d.rows * d.cols, tau_step = std::min(d.rows, d.cols);
  JAXR_REQUIRE(has(*x_out, x.element_type(), d.batch * step));
  JAXR_REQUIRE(has(*tau, x.element_type(), d.batch * tau_step));
  std::vector<W> a(static_cast<size_t>(step)), t(static_cast<size_t>(tau_step));
  Scratch<W> s;
  for (int64_t i = 0; i < d.batch; ++i) {
    int info_v = 0;
    load(x, i * step, step, a.data());
    if (step > 0)
      xgeqrf(static_cast<int>(d.rows), static_cast<int>(d.cols), a.data(), lead(d.rows), t.data(),
             s, &info_v);
    store(*x_out, i * step, step, a.data());
    store(*tau, i * tau_step, tau_step, t.data());
  }
  return ffi::Error::Success();
}

ffi::Error geqrf(Buf x, Out x_out, Out tau) { return JAXR_BY_DTYPE(x, geqrf_impl, x, x_out, tau); }

// Port of jaxlib/cpu/lapack_kernels.cc:PivotingQrFactorization::Kernel.
template <typename W>
ffi::Error geqp3_impl(Buf x, ffi::Buffer<ffi::S32> jpvt, Out x_out, IntOut jpvt_out, Out tau) {
  Dims2 d;
  if (!split2d(x.dimensions(), &d)) return rank_error();
  JAXR_REQUIRE(fits_int(d));
  const int64_t step = d.rows * d.cols, tau_step = std::min(d.rows, d.cols);
  JAXR_REQUIRE(has(*x_out, x.element_type(), d.batch * step));
  JAXR_REQUIRE(has(*tau, x.element_type(), d.batch * tau_step));
  JAXR_REQUIRE(has(jpvt, d.batch * d.cols) && has(*jpvt_out, d.batch * d.cols));
  const int32_t* jpvt_data = jpvt.typed_data();
  int32_t* jpvt_out_data = jpvt_out->typed_data();
  std::vector<W> a(static_cast<size_t>(step)), t(static_cast<size_t>(tau_step));
  std::vector<int> p(static_cast<size_t>(d.cols));
  Scratch<W> s;
  for (int64_t i = 0; i < d.batch; ++i) {
    int info_v = 0;
    load(x, i * step, step, a.data());
    std::copy_n(jpvt_data + i * d.cols, d.cols, p.data());
    if (step > 0)
      xgeqp3(static_cast<int>(d.rows), static_cast<int>(d.cols), a.data(), lead(d.rows), p.data(),
             t.data(), s, &info_v);
    store(*x_out, i * step, step, a.data());
    std::copy_n(p.data(), d.cols, jpvt_out_data + i * d.cols);
    store(*tau, i * tau_step, tau_step, t.data());
  }
  return ffi::Error::Success();
}

ffi::Error geqp3(Buf x, ffi::Buffer<ffi::S32> jpvt, Out x_out, IntOut jpvt_out, Out tau) {
  return JAXR_BY_DTYPE(x, geqp3_impl, x, jpvt, x_out, jpvt_out, tau);
}

// Port of jaxlib/cpu/lapack_kernels.cc:OrthogonalQr::Kernel.
template <typename W>
ffi::Error orgqr_impl(Buf x, Buf tau, Out x_out) {
  Dims2 d;
  if (!split2d(x.dimensions(), &d)) return rank_error();
  JAXR_REQUIRE(fits_int(d) && tau.dimensions().size() >= 1);
  const int64_t step = d.rows * d.cols, k = tau.dimensions().back();
  // LAPACK requires m >= n >= k.
  JAXR_REQUIRE(d.rows >= d.cols && d.cols >= k);
  JAXR_REQUIRE(has(tau, x.element_type(), d.batch * k));
  JAXR_REQUIRE(has(*x_out, x.element_type(), d.batch * step));
  std::vector<W> a(static_cast<size_t>(step)), t(static_cast<size_t>(k));
  Scratch<W> s;
  for (int64_t i = 0; i < d.batch; ++i) {
    int info_v = 0;
    load(x, i * step, step, a.data());
    load(tau, i * k, k, t.data());
    if (step > 0)
      xorgqr(static_cast<int>(d.rows), static_cast<int>(d.cols), static_cast<int>(k), a.data(),
             lead(d.rows), t.data(), s, &info_v);
    store(*x_out, i * step, step, a.data());
  }
  return ffi::Error::Success();
}

ffi::Error orgqr(Buf x, Buf tau, Out x_out) { return JAXR_BY_DTYPE(x, orgqr_impl, x, tau, x_out); }

// Port of jaxlib/cpu/lapack_kernels.cc:OrthogonalQrMultiply::Kernel.
template <typename W>
ffi::Error ormqr_impl(Buf a, Buf tau, Buf c, bool left, bool transpose, Out c_out) {
  Dims2 d, da;
  if (!split2d(c.dimensions(), &d) || !split2d(a.dimensions(), &da)) return rank_error();
  JAXR_REQUIRE(fits_int(d) && fits_int(da) && tau.dimensions().size() >= 1);
  const int64_t c_step = d.rows * d.cols, a_step = da.rows * da.cols, k = tau.dimensions().back();
  const int64_t nq = left ? d.rows : d.cols;
  // LAPACK requires the order of Q to match the reflectors and k <= order.
  JAXR_REQUIRE(da.rows == nq && k <= nq && k <= da.cols && da.batch == d.batch);
  JAXR_REQUIRE(a.element_type() == c.element_type());
  JAXR_REQUIRE(has(tau, c.element_type(), d.batch * k));
  JAXR_REQUIRE(has(*c_out, c.element_type(), d.batch * c_step));
  std::vector<W> av(static_cast<size_t>(a_step)), t(static_cast<size_t>(k)),
      cv(static_cast<size_t>(c_step));
  Scratch<W> s;
  for (int64_t i = 0; i < d.batch; ++i) {
    int info_v = 0;
    load(a, i * a_step, a_step, av.data());
    load(tau, i * k, k, t.data());
    load(c, i * c_step, c_step, cv.data());
    if (c_step > 0 && k > 0)
      xormqr(left ? 'L' : 'R', transpose, static_cast<int>(d.rows), static_cast<int>(d.cols),
             static_cast<int>(k), av.data(), lead(nq), t.data(), cv.data(), lead(d.rows), s,
             &info_v);
    store(*c_out, i * c_step, c_step, cv.data());
  }
  return ffi::Error::Success();
}

ffi::Error ormqr(Buf a, Buf tau, Buf c, bool left, bool transpose, Out c_out) {
  return JAXR_BY_DTYPE(c, ormqr_impl, a, tau, c, left, transpose, c_out);
}

// Port of jaxlib/cpu/lapack_kernels.cc:CholeskyFactorization::Kernel.
template <typename W>
ffi::Error potrf_impl(Buf x, uint8_t uplo, Out x_out, IntOut info) {
  Dims2 d;
  if (!split2d(x.dimensions(), &d)) return rank_error();
  JAXR_REQUIRE(fits_int(d) && d.rows == d.cols && one_of(uplo, "LU"));
  const int64_t step = d.rows * d.cols;
  JAXR_REQUIRE(has(*x_out, x.element_type(), d.batch * step) && has(*info, d.batch));
  int32_t* info_data = info->typed_data();
  std::vector<W> a(static_cast<size_t>(step));
  for (int64_t i = 0; i < d.batch; ++i) {
    int info_v = 0;
    load(x, i * step, step, a.data());
    if (step > 0)
      xpotrf(static_cast<char>(uplo), static_cast<int>(d.rows), a.data(), lead(d.rows), &info_v);
    store(*x_out, i * step, step, a.data());
    info_data[i] = info_v;
  }
  return ffi::Error::Success();
}

ffi::Error potrf(Buf x, uint8_t uplo, Out x_out, IntOut info) {
  return JAXR_BY_DTYPE(x, potrf_impl, x, uplo, x_out, info);
}

// Port of jaxlib/cpu/lapack_kernels.cc:internal::SvdKernel (gesdd, divide and conquer)
// and internal::SvdQRKernel (gesvd, QR iteration).
template <typename W>
ffi::Error svd_impl(Buf x, Out x_out, Out sv, Out u, Out vt, IntOut info, uint8_t mode,
                    bool divide_and_conquer) {
  if (mode == 'O')
    return ffi::Error(ffi::ErrorCode::kUnimplemented,
                      "SVD: Current implementation does not support this computation mode");
  Dims2 d, du, dvt;
  if (!split2d(x.dimensions(), &d) || !split2d(u->dimensions(), &du) ||
      !split2d(vt->dimensions(), &dvt))
    return rank_error();
  JAXR_REQUIRE(fits_int(d) && fits_int(du) && fits_int(dvt));
  JAXR_REQUIRE(mode == 'A' || mode == 'S' || mode == 'N');
  const int64_t step = d.rows * d.cols, k = std::min(d.rows, d.cols);
  const int64_t u_step = du.rows * du.cols, vt_step = dvt.rows * dvt.cols;
  const ffi::DataType dt = x.element_type();
  JAXR_REQUIRE(has(*x_out, dt, d.batch * step) && has(*sv, real_of(dt), d.batch * k));
  JAXR_REQUIRE(has(*u, dt, d.batch * u_step) && has(*vt, dt, d.batch * vt_step));
  JAXR_REQUIRE(has(*info, d.batch));
  if (mode == 'A') JAXR_REQUIRE(du.rows == d.rows && du.cols == d.rows && dvt.rows == d.cols && dvt.cols == d.cols);
  if (mode == 'S') JAXR_REQUIRE(du.rows == d.rows && du.cols == k && dvt.rows == k && dvt.cols == d.cols);
  int32_t* info_data = info->typed_data();
  std::vector<W> a(static_cast<size_t>(step)), uv(static_cast<size_t>(std::max<int64_t>(1, u_step))),
      vtv(static_cast<size_t>(std::max<int64_t>(1, vt_step)));
  std::vector<double> svv(static_cast<size_t>(std::max<int64_t>(1, k)));
  Scratch<W> s;
  for (int64_t i = 0; i < d.batch; ++i) {
    int info_v = 0;
    load(x, i * step, step, a.data());
    if (step > 0) {
      if (divide_and_conquer)
        xgesdd(static_cast<char>(mode), static_cast<int>(d.rows), static_cast<int>(d.cols),
               a.data(), lead(d.rows), svv.data(), uv.data(), lead(d.rows), vtv.data(),
               lead(dvt.rows), s, &info_v);
      else
        xgesvd(static_cast<char>(mode), static_cast<int>(d.rows), static_cast<int>(d.cols),
               a.data(), lead(d.rows), svv.data(), uv.data(), lead(d.rows), vtv.data(),
               lead(dvt.rows), s, &info_v);
    }
    store(*x_out, i * step, step, a.data());
    store(*sv, i * k, k, svv.data());
    if (mode != 'N') {
      store(*u, i * u_step, u_step, uv.data());
      store(*vt, i * vt_step, vt_step, vtv.data());
    }
    info_data[i] = info_v;
  }
  return ffi::Error::Success();
}

ffi::Error gesdd(Buf x, Out x_out, Out sv, Out u, Out vt, IntOut info, uint8_t mode) {
  return JAXR_BY_DTYPE(x, svd_impl, x, x_out, sv, u, vt, info, mode, true);
}

ffi::Error gesvd(Buf x, Out x_out, Out sv, Out u, Out vt, IntOut info, uint8_t mode) {
  return JAXR_BY_DTYPE(x, svd_impl, x, x_out, sv, u, vt, info, mode, false);
}

// Port of jaxlib/cpu/lapack_kernels.cc:EigenvalueDecompositionSymmetric::Kernel and
// EigenvalueDecompositionHermitian::Kernel.
template <typename W>
ffi::Error heevd_impl(Buf x, uint8_t uplo, Out x_out, Out w, IntOut info, uint8_t mode) {
  Dims2 d;
  if (!split2d(x.dimensions(), &d)) return rank_error();
  JAXR_REQUIRE(fits_int(d) && d.rows == d.cols && one_of(uplo, "LU") && one_of(mode, "NV"));
  const int64_t n = d.cols, step = n * n;
  const ffi::DataType dt = x.element_type();
  JAXR_REQUIRE(has(*x_out, dt, d.batch * step) && has(*w, real_of(dt), d.batch * n));
  JAXR_REQUIRE(has(*info, d.batch));
  int32_t* info_data = info->typed_data();
  std::vector<W> a(static_cast<size_t>(step));
  std::vector<double> wv(static_cast<size_t>(n));
  Scratch<W> s;
  for (int64_t i = 0; i < d.batch; ++i) {
    int info_v = 0;
    load(x, i * step, step, a.data());
    if (n > 0)
      xheevd(static_cast<char>(mode), static_cast<char>(uplo), static_cast<int>(n), a.data(),
             lead(n), wv.data(), s, &info_v);
    store(*x_out, i * step, step, a.data());
    store(*w, i * n, n, wv.data());
    info_data[i] = info_v;
  }
  return ffi::Error::Success();
}

ffi::Error heevd(Buf x, uint8_t uplo, Out x_out, Out w, IntOut info, uint8_t mode) {
  return JAXR_BY_DTYPE(x, heevd_impl, x, uplo, x_out, w, info, mode);
}

// Port of jaxlib/cpu/lapack_kernels.h:UnpackEigenvectors. LAPACK returns the
// eigenvectors of a complex conjugate pair as the real and imaginary parts in two
// adjacent real columns.
void unpack_eigenvectors(int64_t n, const double* wi, const double* packed, cplx* unpacked) {
  for (int64_t j = 0; j < n;) {
    if (wi[j] == 0. || std::isnan(wi[j])) {
      for (int64_t i = 0; i < n; ++i) unpacked[j * n + i] = cplx(packed[j * n + i], 0.);
      ++j;
    } else {
      for (int64_t i = 0; i < n; ++i) {
        const double re = packed[j * n + i];
        const double im = j + 1 < n ? packed[(j + 1) * n + i] : 0.;
        unpacked[j * n + i] = cplx(re, im);
        if (j + 1 < n) unpacked[(j + 1) * n + i] = cplx(re, -im);
      }
      j += 2;
    }
  }
}

// Port of jaxlib/cpu/lapack_kernels.cc:EigenvalueDecomposition::Kernel (real input).
ffi::Error geev_real_impl(Buf x, uint8_t compute_left, uint8_t compute_right, Out wr, Out wi, Out vl,
                     Out vr, IntOut info) {
  Dims2 d;
  if (!split2d(x.dimensions(), &d)) return rank_error();
  const ffi::DataType dt = x.element_type();
  JAXR_REQUIRE(is_real_type(dt) && fits_int(d) && d.rows == d.cols);
  JAXR_REQUIRE(one_of(compute_left, "NV") && one_of(compute_right, "NV"));
  const int64_t n = d.cols, step = n * n;
  JAXR_REQUIRE(has(*wr, dt, d.batch * n) && has(*wi, dt, d.batch * n));
  JAXR_REQUIRE(has(*vl, cplx_of(dt), d.batch * step) && has(*vr, cplx_of(dt), d.batch * step));
  JAXR_REQUIRE(has(*info, d.batch));
  int32_t* info_data = info->typed_data();
  char jobvl = static_cast<char>(compute_left), jobvr = static_cast<char>(compute_right);
  std::vector<double> a(static_cast<size_t>(step)), wrv(static_cast<size_t>(n)),
      wiv(static_cast<size_t>(n)), vlw(static_cast<size_t>(std::max<int64_t>(1, step))),
      vrw(static_cast<size_t>(std::max<int64_t>(1, step))), work;
  std::vector<cplx> unpacked(static_cast<size_t>(step));
  const int nn = static_cast<int>(n), ld = lead(n);
  for (int64_t i = 0; i < d.batch; ++i) {
    load(x, i * step, step, a.data());
    if (!std::all_of(a.begin(), a.end(), [](double v) { return std::isfinite(v); })) {
      info_data[i] = -4;
      continue;
    }
    int info_v = 0;
    if (n > 0) {
      if (work.empty()) {
        double q = 0;
        int lw = -1;
        F77_CALL(dgeev)(&jobvl, &jobvr, &nn, a.data(), &ld, wrv.data(), wiv.data(), vlw.data(),
                        &ld, vrw.data(), &ld, &q, &lw, &info_v FCONE FCONE);
        work.resize(lwork_of(q));
      }
      int lw = static_cast<int>(work.size());
      F77_CALL(dgeev)(&jobvl, &jobvr, &nn, a.data(), &ld, wrv.data(), wiv.data(), vlw.data(), &ld,
                      vrw.data(), &ld, work.data(), &lw, &info_v FCONE FCONE);
    }
    info_data[i] = info_v;
    store(*wr, i * n, n, wrv.data());
    store(*wi, i * n, n, wiv.data());
    if (info_v == 0) {
      unpack_eigenvectors(n, wiv.data(), vlw.data(), unpacked.data());
      store(*vl, i * step, step, unpacked.data());
      unpack_eigenvectors(n, wiv.data(), vrw.data(), unpacked.data());
      store(*vr, i * step, step, unpacked.data());
    }
  }
  return ffi::Error::Success();
}

// Port of jaxlib/cpu/lapack_kernels.cc:EigenvalueDecompositionComplex::Kernel.
ffi::Error geev_cplx_impl(Buf x, uint8_t compute_left, uint8_t compute_right, Out w, Out vl, Out vr,
                     IntOut info) {
  Dims2 d;
  if (!split2d(x.dimensions(), &d)) return rank_error();
  const ffi::DataType dt = x.element_type();
  JAXR_REQUIRE(is_cplx_type(dt) && fits_int(d) && d.rows == d.cols);
  JAXR_REQUIRE(one_of(compute_left, "NV") && one_of(compute_right, "NV"));
  const int64_t n = d.cols, step = n * n;
  JAXR_REQUIRE(has(*w, dt, d.batch * n) && has(*vl, dt, d.batch * step));
  JAXR_REQUIRE(has(*vr, dt, d.batch * step) && has(*info, d.batch));
  int32_t* info_data = info->typed_data();
  char jobvl = static_cast<char>(compute_left), jobvr = static_cast<char>(compute_right);
  std::vector<cplx> a(static_cast<size_t>(step)), wv(static_cast<size_t>(n)),
      vlw(static_cast<size_t>(std::max<int64_t>(1, step))),
      vrw(static_cast<size_t>(std::max<int64_t>(1, step))), work;
  std::vector<double> rwork(static_cast<size_t>(std::max<int64_t>(1, 2 * n)));
  const int nn = static_cast<int>(n), ld = lead(n);
  for (int64_t i = 0; i < d.batch; ++i) {
    load(x, i * step, step, a.data());
    if (!std::all_of(a.begin(), a.end(), [](const cplx& z) {
          return std::isfinite(z.real()) && std::isfinite(z.imag());
        })) {
      info_data[i] = -4;
      continue;
    }
    int info_v = 0;
    if (n > 0) {
      if (work.empty()) {
        cplx q = 0;
        int lw = -1;
        F77_CALL(zgeev)(&jobvl, &jobvr, &nn, rc(a.data()), &ld, rc(wv.data()), rc(vlw.data()), &ld,
                        rc(vrw.data()), &ld, rc(&q), &lw, rwork.data(), &info_v FCONE FCONE);
        work.resize(lwork_of(q));
      }
      int lw = static_cast<int>(work.size());
      F77_CALL(zgeev)(&jobvl, &jobvr, &nn, rc(a.data()), &ld, rc(wv.data()), rc(vlw.data()), &ld,
                      rc(vrw.data()), &ld, rc(work.data()), &lw, rwork.data(), &info_v
                      FCONE FCONE);
    }
    info_data[i] = info_v;
    store(*w, i * n, n, wv.data());
    store(*vl, i * step, step, vlw.data());
    store(*vr, i * step, step, vrw.data());
  }
  return ffi::Error::Success();
}

ffi::Error sort_unimplemented() {
  return ffi::Error(ffi::ErrorCode::kUnimplemented,
                    "Ordering eigenvalues on the diagonal is not implemented");
}

// Port of jaxlib/cpu/lapack_kernels.cc:SchurDecomposition::Kernel (real input).
ffi::Error gees_real_impl(Buf x, uint8_t mode, uint8_t sort, Out x_out, Out vs, Out wr, Out wi,
                     IntOut sdim, IntOut info) {
  Dims2 d;
  if (!split2d(x.dimensions(), &d)) return rank_error();
  if (sort != 'N') return sort_unimplemented();
  const ffi::DataType dt = x.element_type();
  JAXR_REQUIRE(is_real_type(dt) && fits_int(d) && d.rows == d.cols && one_of(mode, "NV"));
  const int64_t n = d.cols, step = n * n;
  JAXR_REQUIRE(has(*x_out, dt, d.batch * step) && has(*vs, dt, d.batch * step));
  JAXR_REQUIRE(has(*wr, dt, d.batch * n) && has(*wi, dt, d.batch * n));
  JAXR_REQUIRE(has(*sdim, d.batch) && has(*info, d.batch));
  char jobvs = static_cast<char>(mode), sort_v = 'N';
  std::vector<double> a(static_cast<size_t>(step)),
      vsv(static_cast<size_t>(std::max<int64_t>(1, step))), wrv(static_cast<size_t>(n)),
      wiv(static_cast<size_t>(n)), work;
  const int nn = static_cast<int>(n), ld = lead(n);
  int bwork = 0;
  for (int64_t i = 0; i < d.batch; ++i) {
    int info_v = 0, sdim_v = 0;
    load(x, i * step, step, a.data());
    if (n > 0) {
      if (work.empty()) {
        double q = 0;
        int lw = -1;
        F77_CALL(dgees)(&jobvs, &sort_v, nullptr, &nn, a.data(), &ld, &sdim_v, wrv.data(),
                        wiv.data(), vsv.data(), &ld, &q, &lw, &bwork, &info_v FCONE FCONE);
        work.resize(lwork_of(q));
      }
      int lw = static_cast<int>(work.size());
      F77_CALL(dgees)(&jobvs, &sort_v, nullptr, &nn, a.data(), &ld, &sdim_v, wrv.data(),
                      wiv.data(), vsv.data(), &ld, work.data(), &lw, &bwork, &info_v FCONE FCONE);
    }
    store(*x_out, i * step, step, a.data());
    store(*vs, i * step, step, vsv.data());
    store(*wr, i * n, n, wrv.data());
    store(*wi, i * n, n, wiv.data());
    sdim->typed_data()[i] = sdim_v;
    info->typed_data()[i] = info_v;
  }
  return ffi::Error::Success();
}

// Port of jaxlib/cpu/lapack_kernels.cc:SchurDecompositionComplex::Kernel.
ffi::Error gees_cplx_impl(Buf x, uint8_t mode, uint8_t sort, Out x_out, Out vs, Out w, IntOut sdim,
                     IntOut info) {
  Dims2 d;
  if (!split2d(x.dimensions(), &d)) return rank_error();
  if (sort != 'N') return sort_unimplemented();
  const ffi::DataType dt = x.element_type();
  JAXR_REQUIRE(is_cplx_type(dt) && fits_int(d) && d.rows == d.cols && one_of(mode, "NV"));
  const int64_t n = d.cols, step = n * n;
  JAXR_REQUIRE(has(*x_out, dt, d.batch * step) && has(*vs, dt, d.batch * step));
  JAXR_REQUIRE(has(*w, dt, d.batch * n) && has(*sdim, d.batch) && has(*info, d.batch));
  char jobvs = static_cast<char>(mode), sort_v = 'N';
  std::vector<cplx> a(static_cast<size_t>(step)),
      vsv(static_cast<size_t>(std::max<int64_t>(1, step))), wv(static_cast<size_t>(n)), work;
  std::vector<double> rwork(static_cast<size_t>(std::max<int64_t>(1, n)));
  const int nn = static_cast<int>(n), ld = lead(n);
  int bwork = 0;
  for (int64_t i = 0; i < d.batch; ++i) {
    int info_v = 0, sdim_v = 0;
    load(x, i * step, step, a.data());
    if (n > 0) {
      if (work.empty()) {
        cplx q = 0;
        int lw = -1;
        F77_CALL(zgees)(&jobvs, &sort_v, nullptr, &nn, rc(a.data()), &ld, &sdim_v, rc(wv.data()),
                        rc(vsv.data()), &ld, rc(&q), &lw, rwork.data(), &bwork, &info_v
                        FCONE FCONE);
        work.resize(lwork_of(q));
      }
      int lw = static_cast<int>(work.size());
      F77_CALL(zgees)(&jobvs, &sort_v, nullptr, &nn, rc(a.data()), &ld, &sdim_v, rc(wv.data()),
                      rc(vsv.data()), &ld, rc(work.data()), &lw, rwork.data(), &bwork, &info_v
                      FCONE FCONE);
    }
    store(*x_out, i * step, step, a.data());
    store(*vs, i * step, step, vsv.data());
    store(*w, i * n, n, wv.data());
    sdim->typed_data()[i] = sdim_v;
    info->typed_data()[i] = info_v;
  }
  return ffi::Error::Success();
}

// Port of jaxlib/cpu/lapack_kernels.cc:HessenbergDecomposition::Kernel.
template <typename W>
ffi::Error gehrd_impl(Buf x, int32_t low, int32_t high, Out x_out, Out tau, IntOut info) {
  Dims2 d;
  if (!split2d(x.dimensions(), &d)) return rank_error();
  JAXR_REQUIRE(fits_int(d) && d.rows == d.cols && d.cols >= 1);
  const int64_t n = d.cols, step = n * n;
  // LAPACK requires 1 <= ilo <= ihi <= n.
  JAXR_REQUIRE(1 <= low && low <= high && high <= n);
  JAXR_REQUIRE(has(*x_out, x.element_type(), d.batch * step));
  JAXR_REQUIRE(has(*tau, x.element_type(), d.batch * (n - 1)) && has(*info, d.batch));
  int32_t* info_data = info->typed_data();
  std::vector<W> a(static_cast<size_t>(step)), t(static_cast<size_t>(n));
  Scratch<W> s;
  for (int64_t i = 0; i < d.batch; ++i) {
    int info_v = 0;
    load(x, i * step, step, a.data());
    xgehrd(static_cast<int>(n), low, high, a.data(), lead(n), t.data(), s, &info_v);
    store(*x_out, i * step, step, a.data());
    store(*tau, i * (n - 1), n - 1, t.data());
    info_data[i] = info_v;
  }
  return ffi::Error::Success();
}

ffi::Error gehrd(Buf x, int32_t low, int32_t high, Out x_out, Out tau, IntOut info) {
  return JAXR_BY_DTYPE(x, gehrd_impl, x, low, high, x_out, tau, info);
}

// Port of jaxlib/cpu/lapack_kernels.cc:TridiagonalReduction::Kernel.
template <typename W>
ffi::Error hetrd_impl(Buf x, uint8_t uplo, Out x_out, Out diag, Out off_diag, Out tau,
                      IntOut info) {
  Dims2 d;
  if (!split2d(x.dimensions(), &d)) return rank_error();
  JAXR_REQUIRE(fits_int(d) && d.rows == d.cols && d.cols >= 1 && one_of(uplo, "LU"));
  const int64_t n = d.cols, step = n * n;
  const ffi::DataType dt = x.element_type();
  JAXR_REQUIRE(has(*x_out, dt, d.batch * step) && has(*tau, dt, d.batch * (n - 1)));
  JAXR_REQUIRE(has(*diag, real_of(dt), d.batch * n));
  JAXR_REQUIRE(has(*off_diag, real_of(dt), d.batch * (n - 1)) && has(*info, d.batch));
  int32_t* info_data = info->typed_data();
  std::vector<W> a(static_cast<size_t>(step)), t(static_cast<size_t>(n));
  std::vector<double> dv(static_cast<size_t>(n)), ev(static_cast<size_t>(n));
  Scratch<W> s;
  for (int64_t i = 0; i < d.batch; ++i) {
    int info_v = 0;
    load(x, i * step, step, a.data());
    xhetrd(static_cast<char>(uplo), static_cast<int>(n), a.data(), lead(n), dv.data(), ev.data(),
           t.data(), s, &info_v);
    store(*x_out, i * step, step, a.data());
    store(*diag, i * n, n, dv.data());
    store(*off_diag, i * (n - 1), n - 1, ev.data());
    store(*tau, i * (n - 1), n - 1, t.data());
    info_data[i] = info_v;
  }
  return ffi::Error::Success();
}

ffi::Error hetrd(Buf x, uint8_t uplo, Out x_out, Out diag, Out off_diag, Out tau, IntOut info) {
  return JAXR_BY_DTYPE(x, hetrd_impl, x, uplo, x_out, diag, off_diag, tau, info);
}

// Port of jaxlib/cpu/lapack_kernels.cc:TridiagonalSolver::Kernel. The first element of
// dl and the last element of du are not part of the matrix.
template <typename W>
ffi::Error gtsv_impl(Buf dl, Buf d, Buf du, Buf b, Out dl_out, Out d_out, Out du_out, Out b_out,
                     IntOut info) {
  Dims2 db;
  if (!split2d(b.dimensions(), &db)) return rank_error();
  JAXR_REQUIRE(fits_int(db));
  const int64_t n = db.rows, b_step = db.rows * db.cols;
  const ffi::DataType dt = b.element_type();
  JAXR_REQUIRE(has(dl, dt, db.batch * n) && has(d, dt, db.batch * n) && has(du, dt, db.batch * n));
  JAXR_REQUIRE(has(*dl_out, dt, db.batch * n) && has(*d_out, dt, db.batch * n));
  JAXR_REQUIRE(has(*du_out, dt, db.batch * n) && has(*b_out, dt, db.batch * b_step));
  JAXR_REQUIRE(has(*info, db.batch));
  int32_t* info_data = info->typed_data();
  std::vector<W> dlv(static_cast<size_t>(n)), dv(static_cast<size_t>(n)),
      duv(static_cast<size_t>(n)), bv(static_cast<size_t>(b_step));
  for (int64_t i = 0; i < db.batch; ++i) {
    int info_v = 0;
    load(dl, i * n, n, dlv.data());
    load(d, i * n, n, dv.data());
    load(du, i * n, n, duv.data());
    load(b, i * b_step, b_step, bv.data());
    if (n > 0)
      xgtsv(static_cast<int>(n), static_cast<int>(db.cols), dlv.data() + 1, dv.data(), duv.data(),
            bv.data(), lead(n), &info_v);
    store(*dl_out, i * n, n, dlv.data());
    store(*d_out, i * n, n, dv.data());
    store(*du_out, i * n, n, duv.data());
    store(*b_out, i * b_step, b_step, bv.data());
    info_data[i] = info_v;
  }
  return ffi::Error::Success();
}

ffi::Error gtsv(Buf dl, Buf d, Buf du, Buf b, Out dl_out, Out d_out, Out du_out, Out b_out,
                IntOut info) {
  return JAXR_BY_DTYPE(b, gtsv_impl, dl, d, du, b, dl_out, d_out, du_out, b_out, info);
}

// ---- tridiagonal solve with perturbation of singular pivots --------------------------

template <typename S>
struct RealOf {
  using type = S;
};
template <typename T>
struct RealOf<std::complex<T>> {
  using type = T;
};

// Port of jaxlib/tridiagonal_solve_perturbed.h:MaybePerturbPivot (logic of LAPACK's
// xLAGTS). Adjusts the pivot so that neither rhs_row / pivot nor 1 / pivot overflow.
template <typename S>
void maybe_perturb_pivot(typename RealOf<S>::type perturb, S& pivot, S* rhs_row, int k_rhs) {
  using R = typename RealOf<S>::type;
  const R one(1.0);
  const R tiny = std::numeric_limits<R>::min();
  const R small = one / std::numeric_limits<R>::max();
  const R safemin = small < tiny ? tiny : small * (one + std::numeric_limits<R>::epsilon());
  const R bignum = one / safemin;
  R abs_pivot = std::abs(pivot);
  if (abs_pivot >= one) return;
  perturb = std::max(std::abs(perturb), safemin);
  perturb = std::copysign(perturb, std::real(pivot));
  bool stop = false;
  R max_factor = 0;
  for (int j = 0; j < k_rhs; ++j) max_factor = std::max(max_factor, R(std::abs(rhs_row[j])));
  while (abs_pivot < one && !stop) {
    if (abs_pivot < safemin) {
      if (abs_pivot == R(0.0) || max_factor * safemin > abs_pivot) {
        pivot += perturb;
        perturb *= R(2.0);
      } else {
        pivot *= bignum;
        for (int j = 0; j < k_rhs; ++j) rhs_row[j] *= bignum;
        stop = true;
      }
    } else if (max_factor > abs_pivot * bignum) {
      pivot += perturb;
      perturb *= R(2.0);
    } else {
      stop = true;
    }
    abs_pivot = std::abs(pivot);
  }
}

// Port of jaxlib/tridiagonal_solve_perturbed.h:
// SolveWithGaussianEliminationWithPivotingAndPerturbSingular (LAPACK's xLAGTF and
// xLAGTS fused). rhs and x are column major n by k_rhs; u is an n by 3 workspace.
template <typename S>
void solve_tridiagonal_perturbed(int n, int k_rhs, const S* subdiag, const S* diag,
                                 const S* superdiag, const S* rhs, S* x, S* u, S* rhs_row) {
  using R = typename RealOf<S>::type;
  const S zero = S(0.0);
  if (n == 0) return;
  auto U = [&](int i, int j) -> S& { return u[i + static_cast<int64_t>(j) * n]; };
  auto X = [&](int i, int j) -> S& { return x[i + static_cast<int64_t>(j) * n]; };
  auto B = [&](int i, int j) -> const S& { return rhs[i + static_cast<int64_t>(j) * n]; };
  if (n == 1) {
    S p = diag[0];
    R perturb = std::numeric_limits<R>::epsilon() * std::abs(p);
    for (int j = 0; j < k_rhs; ++j) rhs_row[j] = B(0, j);
    maybe_perturb_pivot(perturb, p, rhs_row, k_rhs);
    for (int j = 0; j < k_rhs; ++j) X(0, j) = rhs_row[j] / p;
    return;
  }
  std::fill_n(u, static_cast<int64_t>(n) * 3, zero);
  U(0, 0) = diag[0];
  U(0, 1) = superdiag[0];
  R max_abs_u = std::max(std::abs(U(0, 0)), std::abs(U(0, 1)));
  R scale1 = std::abs(U(0, 0)) + std::abs(U(0, 1));
  for (int j = 0; j < k_rhs; ++j) X(0, j) = B(0, j);
  for (int k = 0; k < n - 1; ++k) {
    U(k + 1, 0) = diag[k + 1];
    R scale2 = std::abs(subdiag[k + 1]) + std::abs(U(k + 1, 0));
    if (k < n - 2) scale2 += std::abs(superdiag[k + 1]);
    if (subdiag[k + 1] == zero) {
      scale1 = scale2;
      U(k + 1, 1) = superdiag[k + 1];
      U(k, 2) = zero;
      for (int j = 0; j < k_rhs; ++j) X(k + 1, j) = B(k + 1, j);
    } else {
      const R piv1 = U(k, 0) == zero ? R(0.0) : std::abs(U(k, 0)) / scale1;
      const R piv2 = std::abs(subdiag[k + 1]) / scale2;
      if (piv2 <= piv1) {
        scale1 = scale2;
        const S factor = subdiag[k + 1] / U(k, 0);
        U(k + 1, 0) -= factor * U(k, 1);
        U(k + 1, 1) = superdiag[k + 1];
        U(k, 2) = zero;
        for (int j = 0; j < k_rhs; ++j) X(k + 1, j) = B(k + 1, j) - factor * X(k, j);
      } else {
        const S factor = U(k, 0) / subdiag[k + 1];
        const S utmp = U(k, 1);
        U(k, 0) = subdiag[k + 1];
        U(k + 1, 0) = utmp - factor * diag[k + 1];
        U(k, 1) = diag[k + 1];
        if (k < n - 2) {
          U(k, 2) = superdiag[k + 1];
          U(k + 1, 1) = -factor * superdiag[k + 1];
        }
        for (int j = 0; j < k_rhs; ++j) {
          X(k + 1, j) = X(k, j) - factor * B(k + 1, j);
          X(k, j) = B(k + 1, j);
        }
        scale1 = std::abs(U(k + 1, 0)) + std::abs(U(k + 1, 1));
      }
    }
    if (k < n - 2) {
      for (int i = 0; i < 3; ++i) max_abs_u = std::max(max_abs_u, R(std::abs(U(k, i))));
    }
  }
  max_abs_u = std::max(max_abs_u, R(std::abs(U(n - 1, 0))));
  const R perturb = std::numeric_limits<R>::epsilon() * max_abs_u;
  S p = U(n - 1, 0);
  for (int j = 0; j < k_rhs; ++j) rhs_row[j] = X(n - 1, j);
  maybe_perturb_pivot(perturb, p, rhs_row, k_rhs);
  for (int j = 0; j < k_rhs; ++j) X(n - 1, j) = rhs_row[j] * (S(1.0) / p);
  p = U(n - 2, 0);
  for (int j = 0; j < k_rhs; ++j) rhs_row[j] = X(n - 2, j) - U(n - 2, 1) * X(n - 1, j);
  maybe_perturb_pivot(std::copysign(perturb, std::real(p)), p, rhs_row, k_rhs);
  for (int j = 0; j < k_rhs; ++j) X(n - 2, j) = rhs_row[j] * (S(1.0) / p);
  for (int k = n - 3; k >= 0; --k) {
    p = U(k, 0);
    for (int j = 0; j < k_rhs; ++j)
      rhs_row[j] = X(k, j) - U(k, 1) * X(k + 1, j) - U(k, 2) * X(k + 2, j);
    maybe_perturb_pivot(std::copysign(perturb, std::real(p)), p, rhs_row, k_rhs);
    for (int j = 0; j < k_rhs; ++j) X(k, j) = rhs_row[j] * (S(1.0) / p);
  }
}

template <typename S>
void tridiagonal_solve_perturbed_typed(const Dims2& db, Buf dl, Buf d, Buf du, Buf b, Out x_out) {
  const int64_t n = db.rows, b_step = db.rows * db.cols;
  std::vector<S> workspace(static_cast<size_t>(n * 3)), rhs_row(static_cast<size_t>(db.cols));
  const S* dl_ptr = static_cast<const S*>(dl.untyped_data());
  const S* d_ptr = static_cast<const S*>(d.untyped_data());
  const S* du_ptr = static_cast<const S*>(du.untyped_data());
  const S* b_ptr = static_cast<const S*>(b.untyped_data());
  S* x_ptr = static_cast<S*>(x_out->untyped_data());
  for (int64_t i = 0; i < db.batch; ++i) {
    solve_tridiagonal_perturbed<S>(static_cast<int>(n), static_cast<int>(db.cols), dl_ptr + i * n,
                                   d_ptr + i * n, du_ptr + i * n, b_ptr + i * b_step,
                                   x_ptr + i * b_step, workspace.data(), rhs_row.data());
  }
}

// Port of jaxlib/cpu/tridiagonal_solve_kernels.cc:TridiagonalSolvePerturbedFfiImpl. This
// kernel is not LAPACK and runs in the precision of its operands.
ffi::Error tridiagonal_solve_perturbed_impl(Buf dl, Buf d, Buf du, Buf b, Out x_out) {
  const ffi::DataType dt = d.element_type();
  if (dl.element_type() != dt || du.element_type() != dt || b.element_type() != dt ||
      x_out->element_type() != dt)
    return ffi::Error::InvalidArgument("All types must match");
  Dims2 db;
  if (!split2d(b.dimensions(), &db)) return rank_error();
  JAXR_REQUIRE(fits_int(db));
  const int64_t n = db.rows;
  JAXR_REQUIRE(has(dl, dt, db.batch * n) && has(d, dt, db.batch * n) && has(du, dt, db.batch * n));
  JAXR_REQUIRE(has(*x_out, dt, db.batch * db.rows * db.cols));
  JAXR_REQUIRE(b.untyped_data() != x_out->untyped_data());
  switch (dt) {
    case ffi::F32: tridiagonal_solve_perturbed_typed<float>(db, dl, d, du, b, x_out); break;
    case ffi::F64: tridiagonal_solve_perturbed_typed<double>(db, dl, d, du, b, x_out); break;
    case ffi::C64:
      tridiagonal_solve_perturbed_typed<std::complex<float>>(db, dl, d, du, b, x_out);
      break;
    case ffi::C128:
      tridiagonal_solve_perturbed_typed<std::complex<double>>(db, dl, d, du, b, x_out);
      break;
    default: return ffi::Error::InvalidArgument("Unsupported data type");
  }
  return ffi::Error::Success();
}

ffi::Error geev_real(Buf x, uint8_t compute_left, uint8_t compute_right, Out wr, Out wi, Out vl, Out vr, IntOut info) {
  return guarded([&] { return geev_real_impl(x, compute_left, compute_right, wr, wi, vl, vr, info); });
}

ffi::Error geev_cplx(Buf x, uint8_t compute_left, uint8_t compute_right, Out w, Out vl, Out vr, IntOut info) {
  return guarded([&] { return geev_cplx_impl(x, compute_left, compute_right, w, vl, vr, info); });
}

ffi::Error gees_real(Buf x, uint8_t mode, uint8_t sort, Out x_out, Out vs, Out wr, Out wi, IntOut sdim, IntOut info) {
  return guarded([&] { return gees_real_impl(x, mode, sort, x_out, vs, wr, wi, sdim, info); });
}

ffi::Error gees_cplx(Buf x, uint8_t mode, uint8_t sort, Out x_out, Out vs, Out w, IntOut sdim, IntOut info) {
  return guarded([&] { return gees_cplx_impl(x, mode, sort, x_out, vs, w, sdim, info); });
}

ffi::Error tridiagonal_solve_perturbed(Buf dl, Buf d, Buf du, Buf b, Out x_out) {
  return guarded([&] { return tridiagonal_solve_perturbed_impl(dl, d, du, b, x_out); });
}

// ---- handler bindings (argument order of jaxlib/cpu/lapack_kernels.cc) ---------------

using I32 = ffi::Buffer<ffi::S32>;

XLA_FFI_DEFINE_HANDLER_EXPLICIT(kTrsm, trsm,
                       ffi::Ffi::Bind().Arg<Buf>().Arg<Buf>().Ret<Buf>()
                           .Attr<uint8_t>("side").Attr<uint8_t>("uplo")
                           .Attr<uint8_t>("trans_x").Attr<uint8_t>("diag"));
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kGetrf, getrf, ffi::Ffi::Bind().Arg<Buf>().Ret<Buf>().Ret<I32>().Ret<I32>());
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kGeqrf, geqrf, ffi::Ffi::Bind().Arg<Buf>().Ret<Buf>().Ret<Buf>());
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kGeqp3, geqp3,
                       ffi::Ffi::Bind().Arg<Buf>().Arg<I32>().Ret<Buf>().Ret<I32>().Ret<Buf>());
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kOrgqr, orgqr, ffi::Ffi::Bind().Arg<Buf>().Arg<Buf>().Ret<Buf>());
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kOrmqr, ormqr,
                       ffi::Ffi::Bind().Arg<Buf>().Arg<Buf>().Arg<Buf>()
                           .Attr<bool>("left").Attr<bool>("transpose").Ret<Buf>());
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kPotrf, potrf,
                       ffi::Ffi::Bind().Arg<Buf>().Attr<uint8_t>("uplo").Ret<Buf>().Ret<I32>());
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kGesdd, gesdd,
                       ffi::Ffi::Bind().Arg<Buf>().Ret<Buf>().Ret<Buf>().Ret<Buf>().Ret<Buf>()
                           .Ret<I32>().Attr<uint8_t>("mode"));
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kGesvd, gesvd,
                       ffi::Ffi::Bind().Arg<Buf>().Ret<Buf>().Ret<Buf>().Ret<Buf>().Ret<Buf>()
                           .Ret<I32>().Attr<uint8_t>("mode"));
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kHeevd, heevd,
                       ffi::Ffi::Bind().Arg<Buf>().Attr<uint8_t>("uplo").Ret<Buf>().Ret<Buf>()
                           .Ret<I32>().Attr<uint8_t>("mode"));
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kGeevReal, geev_real,
                       ffi::Ffi::Bind().Arg<Buf>()
                           .Attr<uint8_t>("compute_left").Attr<uint8_t>("compute_right")
                           .Ret<Buf>().Ret<Buf>().Ret<Buf>().Ret<Buf>().Ret<I32>());
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kGeevCplx, geev_cplx,
                       ffi::Ffi::Bind().Arg<Buf>()
                           .Attr<uint8_t>("compute_left").Attr<uint8_t>("compute_right")
                           .Ret<Buf>().Ret<Buf>().Ret<Buf>().Ret<I32>());
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kGeesReal, gees_real,
                       ffi::Ffi::Bind().Arg<Buf>().Attr<uint8_t>("mode").Attr<uint8_t>("sort")
                           .Ret<Buf>().Ret<Buf>().Ret<Buf>().Ret<Buf>().Ret<I32>().Ret<I32>());
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kGeesCplx, gees_cplx,
                       ffi::Ffi::Bind().Arg<Buf>().Attr<uint8_t>("mode").Attr<uint8_t>("sort")
                           .Ret<Buf>().Ret<Buf>().Ret<Buf>().Ret<I32>().Ret<I32>());
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kHetrd, hetrd,
                       ffi::Ffi::Bind().Arg<Buf>().Attr<uint8_t>("uplo").Ret<Buf>().Ret<Buf>()
                           .Ret<Buf>().Ret<Buf>().Ret<I32>());
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kGehrd, gehrd,
                       ffi::Ffi::Bind().Arg<Buf>().Attr<int32_t>("low").Attr<int32_t>("high")
                           .Ret<Buf>().Ret<Buf>().Ret<I32>());
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kGtsv, gtsv,
                       ffi::Ffi::Bind().Arg<Buf>().Arg<Buf>().Arg<Buf>().Arg<Buf>()
                           .Ret<Buf>().Ret<Buf>().Ret<Buf>().Ret<Buf>().Ret<I32>());
XLA_FFI_DEFINE_HANDLER_EXPLICIT(kTridiagonalSolvePerturbed, tridiagonal_solve_perturbed,
                       ffi::Ffi::Bind().Arg<Buf>().Arg<Buf>().Arg<Buf>().Arg<Buf>().Ret<Buf>());

struct Target {
  const char* name;
  XLA_FFI_Handler* handler;
};

// Target names of jaxlib/cpu/lapack_kernels.cc and tridiagonal_solve_kernels.cc.
const Target kTargets[] = {
    {"lapack_strsm_ffi", kTrsm},       {"lapack_dtrsm_ffi", kTrsm},
    {"lapack_ctrsm_ffi", kTrsm},       {"lapack_ztrsm_ffi", kTrsm},
    {"lapack_sgetrf_ffi", kGetrf},     {"lapack_dgetrf_ffi", kGetrf},
    {"lapack_cgetrf_ffi", kGetrf},     {"lapack_zgetrf_ffi", kGetrf},
    {"lapack_sgeqrf_ffi", kGeqrf},     {"lapack_dgeqrf_ffi", kGeqrf},
    {"lapack_cgeqrf_ffi", kGeqrf},     {"lapack_zgeqrf_ffi", kGeqrf},
    {"lapack_sgeqp3_ffi", kGeqp3},     {"lapack_dgeqp3_ffi", kGeqp3},
    {"lapack_cgeqp3_ffi", kGeqp3},     {"lapack_zgeqp3_ffi", kGeqp3},
    {"lapack_sorgqr_ffi", kOrgqr},     {"lapack_dorgqr_ffi", kOrgqr},
    {"lapack_cungqr_ffi", kOrgqr},     {"lapack_zungqr_ffi", kOrgqr},
    {"lapack_sormqr_ffi", kOrmqr},     {"lapack_dormqr_ffi", kOrmqr},
    {"lapack_cunmqr_ffi", kOrmqr},     {"lapack_zunmqr_ffi", kOrmqr},
    {"lapack_spotrf_ffi", kPotrf},     {"lapack_dpotrf_ffi", kPotrf},
    {"lapack_cpotrf_ffi", kPotrf},     {"lapack_zpotrf_ffi", kPotrf},
    {"lapack_sgesdd_ffi", kGesdd},     {"lapack_dgesdd_ffi", kGesdd},
    {"lapack_cgesdd_ffi", kGesdd},     {"lapack_zgesdd_ffi", kGesdd},
    {"lapack_sgesvd_ffi", kGesvd},     {"lapack_dgesvd_ffi", kGesvd},
    {"lapack_cgesvd_ffi", kGesvd},     {"lapack_zgesvd_ffi", kGesvd},
    {"lapack_ssyevd_ffi", kHeevd},     {"lapack_dsyevd_ffi", kHeevd},
    {"lapack_cheevd_ffi", kHeevd},     {"lapack_zheevd_ffi", kHeevd},
    {"lapack_sgeev_ffi", kGeevReal},   {"lapack_dgeev_ffi", kGeevReal},
    {"lapack_cgeev_ffi", kGeevCplx},   {"lapack_zgeev_ffi", kGeevCplx},
    {"lapack_ssytrd_ffi", kHetrd},     {"lapack_dsytrd_ffi", kHetrd},
    {"lapack_chetrd_ffi", kHetrd},     {"lapack_zhetrd_ffi", kHetrd},
    {"lapack_sgees_ffi", kGeesReal},   {"lapack_dgees_ffi", kGeesReal},
    {"lapack_cgees_ffi", kGeesCplx},   {"lapack_zgees_ffi", kGeesCplx},
    {"lapack_sgehrd_ffi", kGehrd},     {"lapack_dgehrd_ffi", kGehrd},
    {"lapack_cgehrd_ffi", kGehrd},     {"lapack_zgehrd_ffi", kGehrd},
    {"lapack_sgtsv_ffi", kGtsv},       {"lapack_dgtsv_ffi", kGtsv},
    {"lapack_cgtsv_ffi", kGtsv},       {"lapack_zgtsv_ffi", kGtsv},
    {"tridiagonal_solve_perturbed_ffi", kTridiagonalSolvePerturbed},
};

std::string take_pjrt_error(const PJRT_Api* api, PJRT_Error* err) {
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

// Register every target with the plugin. Returns "" on success, else the reason.
std::string register_all() {
  const PJRT_Api* api = jaxr_api();
  if (!api) return "no PJRT plugin is loaded";
  PJRT_FFI_Extension* ext = nullptr;
  for (PJRT_Extension_Base* e = api->extension_start; e; e = e->next) {
    if (e->type == PJRT_Extension_Type_FFI) ext = reinterpret_cast<PJRT_FFI_Extension*>(e);
  }
  if (!ext) return "the PJRT plugin has no FFI extension";
  if (ext->base.struct_size < PJRT_FFI_Extension_STRUCT_SIZE || !ext->register_handler)
    return "the FFI extension of the PJRT plugin cannot register handlers";
  static const char kPlatform[] = "Host";
  for (const Target& t : kTargets) {
    PJRT_FFI_Register_Handler_Args a;
    std::memset(&a, 0, sizeof a);
    a.struct_size = PJRT_FFI_Register_Handler_Args_STRUCT_SIZE;
    a.target_name = t.name;
    a.target_name_size = std::strlen(t.name);
    a.handler = reinterpret_cast<void*>(t.handler);
    a.platform_name = kPlatform;
    a.platform_name_size = sizeof(kPlatform) - 1;
    a.traits = static_cast<PJRT_FFI_Handler_TraitsBits>(0);
    if (PJRT_Error* err = ext->register_handler(&a))
      return std::string(t.name) + ": " + take_pjrt_error(api, err);
  }
  return std::string();
}

}  // namespace

// Called from R (R/lax_linalg.R) right after the plugin loads. Returns a character
// scalar: "" when every LAPACK target is registered, else the failure message.
extern "C" SEXP jaxr_lapack_register() {
  static bool done = false;
  std::string msg;
  if (!done) {
    msg = register_all();
    done = msg.empty();
  }
  return Rf_mkString(msg.c_str());
}
