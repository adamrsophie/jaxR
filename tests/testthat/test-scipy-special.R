# jax.scipy.special. Expected values are from jax 0.12.0.1: float32 with x64 disabled,
# float64 with x64 enabled.

# Compare jax outputs with expected list(dtype, shape, row major values) triples. Finite
# values must agree elementwise to the relative tolerance; the others must be identical.
sp_expect <- function(got, ..., tol = NULL) {
  want <- list(...)
  if (inherits(got, "jax_value")) got <- list(got)
  testthat::expect_length(got, length(want))
  for (i in seq_along(want)) {
    w <- want[[i]]
    g <- got[[i]]
    testthat::expect_identical(dtype(g), w[[1L]])
    testthat::expect_identical(as.integer(dim(g)), as.integer(w[[2L]]))
    gv <- as.array(g)
    if (length(dim(gv)) > 1L) gv <- aperm(gv)
    gv <- as.vector(gv)
    wv <- w[[3L]]
    rtol <- tol %||% if (w[[1L]] %in% c("float64", "complex128")) 1e-12 else 1e-5
    fin <- is.finite(wv)
    parts <- function(v) if (is.complex(v)) c(Re(v), Im(v)) else as.numeric(v)
    testthat::expect_identical(parts(gv[!fin]), parts(wv[!fin]))
    err <- abs(gv[fin] - wv[fin])
    testthat::expect_true(all(err <= rtol * abs(wv[fin]) + rtol * 1e-3),
                          info = paste("max error", if (length(err)) max(err) else 0))
  }
}

# Stack f over the rows of its arguments (the reference for vmap).
sp_rows <- function(f, ...) {
  args <- list(...)
  n <- dim(args[[1L]])[1L]
  outs <- lapply(seq_len(n) - 1L, function(i) {
    do.call(f, lapply(args, lax_index_in_dim, index = i, axis = 0L, keepdims = FALSE))
  })
  if (inherits(outs[[1L]], "jax_value")) return(jnp_stack(outs))
  lapply(seq_along(outs[[1L]]), function(j) jnp_stack(lapply(outs, `[[`, j)))
}

test_that("gamma function family matches JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(scipy_special_gammaln(as_jax(array(c(0.5, 1, 2, 3.20000005, 100, 0, -1.5, -2, Inf,
    NaN)), "float32")), list("float32", c(10L), c(0.572364926, 0, 0, 0.885404885, 359.134216, Inf,
    0.860046983, Inf, Inf, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_gammaln(x))(as_jax(array(c(0.5, 1, 2,
    0x1.999999999999ap+1, 100, 0, -1.5, -2, Inf, NaN)), "float64")), list("float64", c(10L),
    c(0x1.250d048e7a1cp-1, 0, 0, 0x1.c553c810b3a58p-1, 0x1.67225b4879462p+8, Inf,
    0x1.b858151820f82p-1, Inf, Inf, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_gammasgn(as_jax(array(c(0.5, -0.5, -1.5, -2, 0, -0, NaN, Inf, -2.5)),
    "float32")), list("float32", c(9L), c(1, -1, 1, NaN, 1, -1, NaN, 1, -1)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_gammasgn(x))(as_jax(array(c(0.5, -0.5, -1.5, -2, 0, -0,
    NaN, Inf, -2.5)), "float64")), list("float64", c(9L), c(1, -1, 1, NaN, 1, -1, NaN, 1, -1)))
  enable_x64(FALSE)
  sp_expect(scipy_special_loggamma(as_jax(array(c(0.5, 3.20000005, -1.5, 0, NaN)), "float32")),
    list("float32", c(5L), c(0.572364926, 0.885404885, NaN, NaN, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_loggamma(x))(as_jax(array(c(0.5, 0x1.999999999999ap+1,
    -1.5, 0, NaN)), "float64")), list("float64", c(5L), c(0x1.250d048e7a1cp-1, 0x1.c553c810b3a58p-1,
    NaN, NaN, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_gamma(as_jax(array(c(0.5, 1, 3.20000005, -1.5, -2.5, 0, -0, -2, 30, NaN)),
    "float32")), list("float32", c(10L), c(1.77245378, 1, 2.42396569, 2.36327171, -0.945308685, Inf,
    -Inf, NaN, 8.84178847e+30, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_gamma(x))(as_jax(array(c(0.5, 1, 0x1.999999999999ap+1,
    -1.5, -2.5, 0, -0, -2, 30, NaN)), "float64")), list("float64", c(10L), c(0x1.c5bf891b4ef6dp+0,
    1, 0x1.364480377a495p+1, 0x1.2e7fb0bcdf4efp+1, -0x1.e3ff812e3217ep-1, Inf, -Inf, NaN,
    0x1.be6518687a6d2p+102, NaN)))
  sp_expect(scipy_special_polygamma(as_jax(array(c(1L, 2L, 0L, 3L)), "int64"), as_jax(array(c(0.5,
    1.5, 3.20000005, 10)), "float32")), list("float32", c(4L), c(4.93480206, -0.828796923,
    0.998839378, 0.00231990335)))
  sp_expect(jit(function(n, x) scipy_special_polygamma(n, x))(as_jax(array(c(1L, 2L, 0L, 3L)),
    "int64"), as_jax(array(c(0.5, 1.5, 0x1.999999999999ap+1, 10)), "float64")), list("float64",
    c(4L), c(0x1.3bd3cc9be45dfp+2, -0x1.a85808a40ab99p-1, 0x1.ff67cfa819028p-1,
    0x1.3012f876b806p-9)))
  enable_x64(FALSE)
  sp_expect(scipy_special_factorial(as_jax(array(c(0, 1, 5, 2.5, -1)), "float32")), list("float32",
    c(5L), c(1, 1, 120.000008, 3.32335091, 0)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_factorial(x))(as_jax(array(c(0, 1, 5, 2.5, -1)),
    "float64")), list("float64", c(5L), c(1, 1, 0x1.e000000000006p+6, 0x1.a96390899a078p+1, 0)))
  enable_x64(FALSE)
  sp_expect(scipy_special_multigammaln(as_jax(array(c(1.5, 2.5, 10)), "float32"), 3L),
    list("float32", c(3L), c(2.16867733, 1.88099539, 36.8128586)))
  enable_x64(TRUE)
  sp_expect(jit(function(a) scipy_special_multigammaln(a, 3L))(as_jax(array(c(1.5, 2.5, 10)),
    "float64")), list("float64", c(3L), c(0x1.159739b62fbb2p+1, 0x1.e188eb278c08p+0,
    0x1.2680bc06d35e7p+5)))
  enable_x64(FALSE)
  sp_expect(scipy_special_poch(as_jax(array(c(0.5, 2, 3.5, -1.5)), "float32"), as_jax(array(c(1.5,
    0, 2, 0.699999988)), "float32")), list("float32", c(4L), c(0.564189613, 1, 15.75, -2.4282248)))
  enable_x64(TRUE)
  sp_expect(jit(function(z, m) scipy_special_poch(z, m))(as_jax(array(c(0.5, 2, 3.5, -1.5)),
    "float64"), as_jax(array(c(1.5, 0, 2, 0x1.6666666666666p-1)), "float64")), list("float64",
    c(4L), c(0x1.20dd750429b6cp-1, 1, 0x1.f80000000000dp+3, -0x1.36d01001ad9e3p+1)))
  enable_x64(FALSE)
  sp_expect(scipy_special_gamma(as_jax(array(c(complex(real = 1, imaginary = 2), complex(real =
    -0.5, imaginary = -1.5), complex(real = 3, imaginary = -0.100000001), complex(real = -2,
    imaginary = 0), complex(real = 0.200000003, imaginary = 0), complex(real = -2.5, imaginary =
    3))), "complex64")), list("complex64", c(6L), c(complex(real = 0.151904032, imaginary =
    0.0198049173), complex(real = -0.139202669, imaginary = 0.0565531775), complex(real =
    1.98755896, imaginary = -0.1839827), complex(real = NaN, imaginary = NaN), complex(real =
    4.59084415, imaginary = 0), complex(real = 0.000479787588, imaginary = 0.000298855768))))
  enable_x64(TRUE)
  sp_expect(jit(function(z) scipy_special_gamma(z))(as_jax(array(c(complex(real = 1, imaginary = 2),
    complex(real = -0.5, imaginary = -1.5), complex(real = 3, imaginary = -0x1.999999999999ap-4),
    complex(real = -2, imaginary = 0), complex(real = 0x1.999999999999ap-3, imaginary = 0),
    complex(real = -2.5, imaginary = 3))), "complex128")), list("complex128", c(6L), c(complex(real
    = 0x1.3719721ccb5ebp-3, imaginary = 0x1.447bb0262adc4p-6), complex(real = -0x1.1d165296fae6bp-3,
    imaginary = 0x1.cf4863e5e3e5dp-5), complex(real = 0x1.fcd09f81e50f7p+0, imaginary =
    -0x1.78cbdfd531121p-3), complex(real = NaN, imaginary = NaN), complex(real =
    0x1.25d062250541p+2, imaginary = 0), complex(real = 0x1.f71838897c3eep-12, imaginary =
    0x1.395f7816e5886p-12))))
  enable_x64(FALSE)
  sp_expect(scipy_special_loggamma(as_jax(array(c(complex(real = 1, imaginary = 2), complex(real =
    -0.5, imaginary = -1.5), complex(real = 3, imaginary = -0.100000001), complex(real = -2,
    imaginary = 0), complex(real = 0.200000003, imaginary = 0), complex(real = -2.5, imaginary = 3),
    complex(real = -7.5, imaginary = 0.5))), "complex64")), list("complex64", c(7L), c(complex(real
    = -1.87607861, imaginary = 0.12964654), complex(real = -1.89544129, imaginary = 2.75569725),
    complex(real = 0.691173315, imaginary = -0.0923041254), complex(real = NaN, imaginary = NaN),
    complex(real = 1.52406406, imaginary = 0), complex(real = -7.4782362, imaginary = -5.72610378),
    complex(real = -9.30889702, imaginary = -24.0923729))))
  enable_x64(TRUE)
  sp_expect(jit(function(z) scipy_special_loggamma(z))(as_jax(array(c(complex(real = 1, imaginary =
    2), complex(real = -0.5, imaginary = -1.5), complex(real = 3, imaginary =
    -0x1.999999999999ap-4), complex(real = -2, imaginary = 0), complex(real = 0x1.999999999999ap-3,
    imaginary = 0), complex(real = -2.5, imaginary = 3), complex(real = -7.5, imaginary = 0.5))),
    "complex128")), list("complex128", c(7L), c(complex(real = -0x1.e046b30870b1cp+0, imaginary =
    0x1.09840204c77fp-3), complex(real = -0x1.e53baa54a8b72p+0, imaginary = 0x1.60bab54d1db86p+1),
    complex(real = 0x1.61e16d9d2d2bcp-1, imaginary = -0x1.7a13df4459d62p-4), complex(real = NaN,
    imaginary = NaN), complex(real = 0x1.86290bf25b627p+0, imaginary = 0), complex(real =
    -0x1.de9b6b5817f9ap+2, imaginary = -0x1.6e787e0d55d2cp+2), complex(real = -0x1.29e27e3ccf1f4p+3,
    imaginary = -0x1.817a5afe8a1cep+4))))
  enable_x64(FALSE)
  sp_expect(scipy_special_gammaln(as_jax(array(c(1L, 5L, 10L)), "int32")), list("float32", c(3L),
    c(0, 3.17805386, 12.8018274)))
})

test_that("beta function family matches JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(scipy_special_betaln(as_jax(array(c(0.5, 2, 30, 1000, 5)), "float32"),
    as_jax(array(c(1.5, 3, 2, 2000, 5)), "float32")), list("float32", c(5L), c(0.45158267,
    -2.48490667, -6.83518505, -1911.87451, -6.44571972)))
  enable_x64(TRUE)
  sp_expect(jit(function(a, b) scipy_special_betaln(a, b))(as_jax(array(c(0.5, 2, 30, 1000, 5)),
    "float64"), as_jax(array(c(1.5, 3, 2, 2000, 5)), "float64")), list("float64", c(5L),
    c(0x1.ce6bb25aa132p-2, -0x1.3e116bcd39e81p+1, -0x1.b573aa00d4da6p+2, -0x1.ddf7f9ade5ecp+10,
    -0x1.9c86ac6bdc2a7p+2)))
  enable_x64(FALSE)
  sp_expect(scipy_special_beta(as_jax(array(c(0.5, 2, -1.5, 30)), "float32"), as_jax(array(c(1.5, 3,
    0.699999988, 2)), "float32")), list("float32", c(4L), c(1.57079625, 0.0833333284, -0.534569681,
    0.00107526826)))
  enable_x64(TRUE)
  sp_expect(jit(function(a, b) scipy_special_beta(a, b))(as_jax(array(c(0.5, 2, -1.5, 30)),
    "float64"), as_jax(array(c(1.5, 3, 0x1.6666666666666p-1, 2)), "float64")), list("float64",
    c(4L), c(0x1.921fb54442d1cp+0, 0x1.555555555554ap-4, -0x1.11b31f6b06edbp-1,
    0x1.19e011d32fb66p-10)))
  enable_x64(FALSE)
  sp_expect(scipy_special_betainc(as_jax(array(c(2, 0.5)), "float32"), as_jax(array(c(3, 0.5)),
    "float32"), as_jax(array(c(0.200000003, 0.699999988)), "float32")), list("float32", c(2L),
    c(0.180800006, 0.630989909)))
  enable_x64(TRUE)
  sp_expect(jit(function(a, b, x) scipy_special_betainc(a, b, x))(as_jax(array(c(2, 0.5)),
    "float64"), as_jax(array(c(3, 0.5)), "float64"), as_jax(array(c(0x1.999999999999ap-3,
    0x1.6666666666666p-1)), "float64")), list("float64", c(2L), c(0x1.72474538ef356p-3,
    0x1.43111b092557dp-1)))
  enable_x64(FALSE)
  sp_expect(scipy_special_comb(as_jax(array(c(5, 10, 3, 4.5, -1)), "float32"), as_jax(array(c(2, 3,
    5, 2, 1)), "float32")), list("float32", c(5L), c(10, 120.000008, 0, 7.87499905, 0)))
  enable_x64(TRUE)
  sp_expect(jit(function(N, k) scipy_special_comb(N, k))(as_jax(array(c(5, 10, 3, 4.5, -1)),
    "float64"), as_jax(array(c(2, 3, 5, 2, 1)), "float64")), list("float64", c(5L),
    c(0x1.400000000000bp+3, 0x1.dffffffffffbbp+6, 0, 0x1.f800000000018p+2, 0)))
  enable_x64(FALSE)
  sp_expect(scipy_special_comb(as_jax(array(c(5, 10, 3, 0)), "float32"), as_jax(array(c(2, 3, 5,
    0)), "float32"), repetition = TRUE), list("float32", c(4L), c(15.000001, 220.000122, 21.0000076,
    1)))
  enable_x64(TRUE)
  sp_expect(jit(function(N, k) scipy_special_comb(N, k, repetition = TRUE))(as_jax(array(c(5, 10, 3,
    0)), "float64"), as_jax(array(c(2, 3, 5, 0)), "float64")), list("float64", c(4L),
    c(0x1.e00000000000bp+3, 0x1.b7fffffffffe1p+7, 0x1.500000000001p+4, 1)))
  enable_x64(FALSE)
  sp_expect(scipy_special_gammainc(as_jax(array(c(2, 0.5, 3)), "float32"), as_jax(array(c(0.5, 1.5,
    3.20000005)), "float32")), list("float32", c(3L), c(0.0902040079, 0.91673547, 0.620096385)))
  enable_x64(TRUE)
  sp_expect(jit(function(a, x) scipy_special_gammainc(a, x))(as_jax(array(c(2, 0.5, 3)), "float64"),
    as_jax(array(c(0.5, 1.5, 0x1.999999999999ap+1)), "float64")), list("float64", c(3L),
    c(0x1.7179c2ac4f788p-4, 0x1.d55e5a70068e5p-1, 0x1.3d7d41c0e1738p-1)))
  enable_x64(FALSE)
  sp_expect(scipy_special_gammaincc(as_jax(array(c(2, 0.5, 3)), "float32"), as_jax(array(c(0.5, 1.5,
    3.20000005)), "float32")), list("float32", c(3L), c(0.909796, 0.0832645521, 0.379903615)))
  enable_x64(TRUE)
  sp_expect(jit(function(a, x) scipy_special_gammaincc(a, x))(as_jax(array(c(2, 0.5, 3)),
    "float64"), as_jax(array(c(0.5, 1.5, 0x1.999999999999ap+1)), "float64")), list("float64", c(3L),
    c(0x1.d1d0c7aa7610fp-1, 0x1.550d2c7fcb8dap-4, 0x1.85057c7e3d191p-2)))
})

test_that("zeta and bernoulli match JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(scipy_special_zeta(as_jax(array(c(2, 3.5, 1, 0.5)), "float32"), as_jax(array(c(0.5, 1.5,
    2, 1)), "float32")), list("float32", c(4L), c(4.93480206, 0.307096153, Inf, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(s, q) scipy_special_zeta(s, q))(as_jax(array(c(2, 3.5, 1, 0.5)),
    "float64"), as_jax(array(c(0.5, 1.5, 2, 1)), "float64")), list("float64", c(4L),
    c(0x1.3bd3cc9be45dfp+2, 0x1.3a776abc481e8p-2, Inf, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_zeta(as_jax(array(c(2L, 3L)), "int32"), as_jax(array(c(0.5, 1.5)),
    "float32")), list("float32", c(2L), c(4.93480206, 0.414398372)))
  sp_expect(scipy_special_bernoulli(0L), list("float32", c(1L), c(1)))
  sp_expect(scipy_special_bernoulli(1L), list("float32", c(2L), c(1, -0.5)))
  sp_expect(scipy_special_bernoulli(2L), list("float32", c(3L), c(1, -0.5, 0.166666672)))
  sp_expect(scipy_special_bernoulli(3L), list("float32", c(4L), c(1, -0.5, 0.166666672, 0)))
  sp_expect(scipy_special_bernoulli(4L), list("float32", c(5L), c(1, -0.5, 0.166666672, 0,
    -0.0333332494)))
  sp_expect(scipy_special_bernoulli(9L), list("float32", c(10L), c(1, -0.5, 0.166666672, 0,
    -0.0333332457, 0, 0.0238095224, 0, -0.0333333313, 0)))
  sp_expect(scipy_special_bernoulli(12L), list("float32", c(13L), c(1, -0.5, 0.166666672, 0,
    -0.0333332457, 0, 0.0238095224, 0, -0.0333333313, 0, 0.0757575706, 0, -0.253113508)))
  enable_x64(TRUE)
  sp_expect({ d <- as_jax(1, "float64"); scipy_special_bernoulli(12L) }, list("float64", c(13L),
    c(1, -0.5, 0x1.5555555555555p-3, 0, -0x1.1110e3a25bd24p-5, 0, 0x1.86186181c49f1p-6, 0,
    -0x1.1111111110d68p-5, 0, 0x1.364d9364d9364p-4, 0, -0x1.033033033033p-2)))
})

test_that("error function family matches JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(scipy_special_erf(as_jax(array(c(0, 0.5, -1.5, 3, Inf, NaN)), "float32")),
    list("float32", c(6L), c(0, 0.520499885, -0.966105044, 0.999977946, 1, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_erf(x))(as_jax(array(c(0, 0.5, -1.5, 3, Inf, NaN)),
    "float64")), list("float64", c(6L), c(0, 0x1.0a7ef5c18edd2p-1, -0x1.eea5557137aep-1,
    0x1.fffd1ac4135fap-1, 1, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_erfc(as_jax(array(c(0, 0.5, -1.5, 3, 9, Inf, NaN)), "float32")),
    list("float32", c(7L), c(1, 0.479500175, 1.9661051, 2.20904967e-05, 0x1.198d420232f92p-121, 0,
    NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_erfc(x))(as_jax(array(c(0, 0.5, -1.5, 3, 9, Inf, NaN)),
    "float64")), list("float64", c(7L), c(1, 0x1.eb02147ce245cp-2, 0x1.f752aab89bd7p+0,
    0x1.729df6503422bp-16, 0x1.198d422be3f8ep-121, 0, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_erfinv(as_jax(array(c(0, 0.5, -0.899999976, 0.999000013, 1, -1, 1.5,
    NaN)), "float32")), list("float32", c(8L), c(0, 0.476936281, -1.16308701, 2.32675767, Inf, -Inf,
    NaN, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_erfinv(x))(as_jax(array(c(0, 0.5, -0x1.ccccccccccccdp-1,
    0.999, 1, -1, 1.5, NaN)), "float64")), list("float64", c(8L), c(0, 0x1.e861fbb24c008p-2,
    -0x1.29c014677065p+0, 0x1.29d311405cdbap+1, Inf, -Inf, NaN, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_erfcx(as_jax(array(c(-30, -9.39999962, -2, -0.5, 0, 0.300000012, 1, 1.5,
    2.25, 5, 30, 1e+10, 1.00000002e+30, Inf, -Inf, NaN)), "float32")), list("float32", c(16L),
    c(Inf, Inf, 108.94091, 1.95236063, 1, 0.734599352, 0.427583575, 0.321585417, 0.231087267,
    0.110704638, 0.0187958889, 5.64189598e-11, 0x1.6e2e13fc418bbp-101, 0, Inf, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_erfcx(x))(as_jax(array(c(-30, -0x1.2cccccccccccdp+3, -2,
    -0.5, 0, 0x1.3333333333333p-2, 1, 1.5, 2.25, 5, 30, 10000000000, 0x1.93e5939a08ceap+99, Inf,
    -Inf, NaN)), "float64")), list("float64", c(16L), c(Inf, 0x1.6432931eae92p+128,
    0x1.b3c37c70be791p+6, 0x1.f3cde5a30aa93p+0, 1, 0x1.781d676b434c4p-1, 0.427583576155807,
    0x1.494daffa2ad68p-2, 0x1.d94446d627932p-3, 0x1.c57239e943d1ap-4, 0x1.33f3abfd60d6fp-6,
    0x1.f044332d68161p-35, 0x1.6e2e12dc3773cp-101, 0, Inf, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_erfcinv(as_jax(array(c(0.100000001, 0.5, 1, 1.5, 1.89999998,
    1.00000001e-10, 0x1.4484bfeebc2ap-100, 0, 2, 2.5, -0.5, NaN)), "float32")), list("float32",
    c(12L), c(1.16308713, 0.476936281, 0, -0.476936281, -1.16308713, 4.57282495, 8.14861584, Inf,
    -Inf, NaN, NaN, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(y) scipy_special_erfcinv(y))(as_jax(array(c(0x1.999999999999ap-4, 0.5, 1,
    1.5, 0x1.e666666666666p+0, 1e-10, 0x1.4484bfeebc2ap-100, 0, 2, 2.5, -0.5, NaN)), "float64")),
    list("float64", c(12L), c(0x1.29c014677065p+0, 0x1.e861fbb24c00ap-2, 0, -0x1.e861fbb24c00ap-2,
    -0x1.29c014677064fp+0, 0x1.24a92a0d51497p+2, 0x1.04c176cf456d6p+3, Inf, -Inf, NaN, NaN, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_dawsn(as_jax(array(c(-30, -4, -2, 0, 0.300000012, 3.25, 4, 6.25, 30,
    1e+10, Inf, NaN)), "float32")), list("float32", c(12L), c(-0.0166759416, -0.129347995,
    -0.301340401, 0, 0.282631665, 0.162570879, 0.129347995, 0.0810660943, 0.0166759416,
    5.00000007e-11, 0, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_dawsn(x))(as_jax(array(c(-30, -4, -2, 0,
    0x1.3333333333333p-2, 3.25, 4, 6.25, 30, 10000000000, Inf, NaN)), "float64")), list("float64",
    c(12L), c(-0x1.1137f7bca4d95p-6, -0x1.08e79ad8e4938p-3, -0x1.3492932d91017p-2, 0,
    0x1.216a31f8527edp-2, 0x1.4cf1faca8d3b6p-3, 0x1.08e79ad8e4938p-3, 0x1.4c0bf5ece772p-4,
    0x1.1137f7bca4d95p-6, 0x1.b7cdfd9d7bdbbp-35, 0, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_wofz(as_jax(array(c(-2, 0.300000012, 4, 30)), "float32")),
    list("complex64", c(4L), c(complex(real = 0.0183156505, imaginary = -0.340026259), complex(real
    = 0.913931429, imaginary = 0.318915814), complex(real = 1.04308128e-07, imaginary =
    0.145953596), complex(real = 2.32830644e-10, imaginary = 0.018816784))))
  enable_x64(TRUE)
  sp_expect(jit(function(z) scipy_special_wofz(z))(as_jax(array(c(-2, 0x1.3333333333333p-2, 4, 30)),
    "float64")), list("complex128", c(4L), c(complex(real = 0.0183156388887636, imaginary =
    -0x1.5c2fd52853663p-2), complex(real = 0x1.d3eec9cf1197fp-1, imaginary = 0x1.4691d52ebdbaap-2),
    complex(real = 0x1.e355ba36p-24, imaginary = 0x1.2ae9b73ad7474p-3), complex(real = 0x1.12ap-50,
    imaginary = 0x1.344b50e81cd53p-6))))
  enable_x64(FALSE)
  sp_expect(scipy_special_wofz(as_jax(array(c(complex(real = 1, imaginary = 2), complex(real = -0.5,
    imaginary = -1.5), complex(real = 3, imaginary = -0.100000001), complex(real = 0, imaginary =
    0))), "complex64")), list("complex64", c(4L), c(complex(real = 0.218492627, imaginary =
    0.0929978266), complex(real = 0.742007077, imaginary = -14.8189449), complex(real =
    -0.00773692038, imaginary = 0.200883135), complex(real = 1.00000012, imaginary = 0))))
  enable_x64(TRUE)
  sp_expect(jit(function(z) scipy_special_wofz(z))(as_jax(array(c(complex(real = 1, imaginary = 2),
    complex(real = -0.5, imaginary = -1.5), complex(real = 3, imaginary = -0x1.999999999999ap-4),
    complex(real = 0, imaginary = 0))), "complex128")), list("complex128", c(4L), c(complex(real =
    0x1.bf790e682f718p-3, imaginary = 0x1.7ceb455f0dca2p-4), complex(real = 0x1.7be85d8fdc6e5p-1,
    imaginary = -0x1.da34c96cafff4p+3), complex(real = -0x1.fb0c0e74925fbp-8, imaginary =
    0x1.9b689a6d1aabp-3), complex(real = 0x1.0000000000089p+0, imaginary = 0))))
  sp_expect(scipy_special_erfcx(as_jax(array(c(complex(real = 1, imaginary = 2), complex(real =
    -0.5, imaginary = -1.5))), "complex128")), list("complex128", c(2L), c(complex(real =
    0x1.1f35edfc9549ep-3, imaginary = -0x1.c717d7126f3eep-3), complex(real = -0x1.6b7fa4a78898ep-3,
    imaginary = 0x1.3726238c75da9p-1))))
  enable_x64(FALSE)
  sp_expect(scipy_special_erfcx(as_jax(array(c(0.5, 2)), "float16")), list("float16", c(2L),
    c(0.615722656, 0.255371094)), tol = 2e-3)
  sp_expect(scipy_special_dawsn(as_jax(array(c(0.5, 2)), "float16")), list("float16", c(2L),
    c(0.424316406, 0.301269531)), tol = 2e-3)
})

test_that("normal distribution functions match JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(scipy_special_ndtr(as_jax(array(c(-40, -5, -0.5, 0, 0.300000012, 2, 10, NaN)),
    "float32")), list("float32", c(8L), c(0, 2.86651584e-07, 0.308537543, 0.5, 0.617911398,
    0.977249861, 1, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_ndtr(x))(as_jax(array(c(-40, -5, -0.5, 0,
    0x1.3333333333333p-2, 2, 10, NaN)), "float64")), list("float64", c(8L), c(0,
    0x1.33ca2f2133828p-22, 0x1.3bf143b9aa712p-2, 0.5, 0x1.3c5ee2cc40b78p-1, 0x1.f45a183e9b13dp-1, 1,
    NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_ndtri(as_jax(array(c(0.100000001, 0.5, 0.75, 0.99000001, 1.00000001e-10,
    0x1.4484bfeebc2ap-100, 0, 1, 1.5, -0.5, NaN)), "float32")), list("float32", c(11L),
    c(-1.2815516, 0, 0.674489737, 2.32634807, -6.361341, -11.4640245, -Inf, Inf, NaN, NaN, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(p) scipy_special_ndtri(p))(as_jax(array(c(0x1.999999999999ap-4, 0.5, 0.75,
    0x1.fae147ae147aep-1, 1e-10, 0x1.4484bfeebc2ap-100, 0, 1, 1.5, -0.5, NaN)), "float64")),
    list("float64", c(11L), c(-0x1.4813c36e26d33p+0, 0, 0x1.5956b87528a49p-1, 0x1.29c5c4630ff0ep+1,
    -0x1.97203597a2154p+2, -0x1.6ed94a4dacdc8p+3, -Inf, Inf, NaN, NaN, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_log_ndtr(as_jax(array(c(-40, -15, -5, -0.5, 0, 0.300000012, 2, 6, 10,
    NaN)), "float32")), list("float32", c(10L), c(-804.608459, -116.131386, -15.0649986,
    -1.17591178, -0.693147182, -0.481410205, -0.0230129175, -9.86589588e-10, -0x1.26c75e0079147p-77,
    NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_log_ndtr(x))(as_jax(array(c(-40, -15, -5, -0.5, 0,
    0x1.3333333333333p-2, 2, 6, 10, NaN)), "float64")), list("float64", c(10L),
    c(-0x1.924de16d8b53ap+9, -0x1.d08689bfbe165p+6, -0x1.e21477830f2bep+3, -0x1.2d088d9f06be5p+0,
    -0x1.62e42fefa39efp-1, -0x1.ecf6c90ff03bbp-2, -0x1.790b233d099bfp-6, -0x1.0f30f0023e915p-30,
    -0x1.26c75e84fb0e9p-77, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_log_ndtr(as_jax(array(c(-40, -25, -15)), "float32"), series_order = 20L),
    list("float32", c(3L), c(-804.608459, -316.639404, -116.131386)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_log_ndtr(x, series_order = 20L))(as_jax(array(c(-40, -25,
    -15)), "float64")), list("float64", c(3L), c(-0x1.924de16d8b4aep+9, -0x1.3ca3b03e433f5p+8,
    -0x1.d08689bfbe165p+6)))
  enable_x64(FALSE)
  sp_expect(scipy_special_log_ndtr(as_jax(array(c(-40, -25, -15)), "float32"), series_order = 0L),
    list("float32", c(3L), c(-804.607849, -316.637817, -116.126991)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_log_ndtr(x, series_order = 0L))(as_jax(array(c(-40, -25,
    -15)), "float64")), list("float64", c(3L), c(-0x1.924dccfad4ba2p+9, -0x1.3ca347cd40d79p+8,
    -0x1.d08689bfbe165p+6)))
})

test_that("logistic and entropy functions match JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(scipy_special_logit(as_jax(array(c(0.100000001, 0.5, 0.899999976, 0, 1, 1.5)),
    "float32")), list("float32", c(6L), c(-2.19722462, 0, 2.19722438, -Inf, Inf, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_logit(x))(as_jax(array(c(0x1.999999999999ap-4, 0.5,
    0x1.ccccccccccccdp-1, 0, 1, 1.5)), "float64")), list("float64", c(6L), c(-0x1.193ea7aad030ap+1,
    0, 0x1.193ea7aad030bp+1, -Inf, Inf, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_expit(as_jax(array(c(-30, -1, 0, 2, 40, NaN)), "float32")),
    list("float32", c(6L), c(9.35762359e-14, 0.268941432, 0.5, 0.880797029, 1, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_expit(x))(as_jax(array(c(-30, -1, 0, 2, 40, NaN)),
    "float64")), list("float64", c(6L), c(0x1.a56e0c2ac7cbfp-44, 0x1.136561454ba86p-2, 0.5,
    0x1.c2f7d5a8a79c9p-1, 1, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_xlogy(as_jax(array(c(0, 0, 2, 1, 0)), "float32"), as_jax(array(c(0, 3, 3,
    -1, NaN)), "float32")), list("float32", c(5L), c(0, 0, 2.19722462, NaN, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x, y) scipy_special_xlogy(x, y))(as_jax(array(c(0, 0, 2, 1, 0)),
    "float64"), as_jax(array(c(0, 3, 3, -1, NaN)), "float64")), list("float64", c(5L), c(0, 0,
    0x1.193ea7aad030bp+1, NaN, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_xlog1py(as_jax(array(c(0, 0, 2, 1, 0)), "float32"), as_jax(array(c(-1, 3,
    3, -2, NaN)), "float32")), list("float32", c(5L), c(0, 0, 2.77258873, NaN, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x, y) scipy_special_xlog1py(x, y))(as_jax(array(c(0, 0, 2, 1, 0)),
    "float64"), as_jax(array(c(-1, 3, 3, -2, NaN)), "float64")), list("float64", c(5L), c(0, 0,
    0x1.62e42fefa39efp+1, NaN, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_entr(as_jax(array(c(0, 0.5, 2, -1, NaN)), "float32")), list("float32",
    c(5L), c(-0, 0.346573591, -1.38629436, -Inf, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_entr(x))(as_jax(array(c(0, 0.5, 2, -1, NaN)), "float64")),
    list("float64", c(5L), c(-0, 0x1.62e42fefa39efp-2, -0x1.62e42fefa39efp+0, -Inf, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_boxcox(as_jax(array(c(0.5, 2, 3, 1)), "float32"), as_jax(array(c(0, 0.5,
    -1, 2)), "float32")), list("float32", c(4L), c(-0.693147182, 0.828427136, 0.666666687, 0)))
  enable_x64(TRUE)
  sp_expect(jit(function(x, l) scipy_special_boxcox(x, l))(as_jax(array(c(0.5, 2, 3, 1)),
    "float64"), as_jax(array(c(0, 0.5, -1, 2)), "float64")), list("float64", c(4L),
    c(-0x1.62e42fefa39efp-1, 0x1.a827999fcef32p-1, 0x1.5555555555556p-1, 0)))
  enable_x64(FALSE)
  sp_expect(scipy_special_boxcox1p(as_jax(array(c(0.5, 2, 3, 1)), "float32"), as_jax(array(c(0, 0.5,
    -1, 2)), "float32")), list("float32", c(4L), c(0.405465096, 1.46410167, 0.75, 1.5)))
  enable_x64(TRUE)
  sp_expect(jit(function(x, l) scipy_special_boxcox1p(x, l))(as_jax(array(c(0.5, 2, 3, 1)),
    "float64"), as_jax(array(c(0, 0.5, -1, 2)), "float64")), list("float64", c(4L),
    c(0x1.9f323ecbf984cp-2, 0x1.76cf5d0b09956p+0, 0.75, 1.5)))
  enable_x64(FALSE)
  sp_expect(scipy_special_kl_div(as_jax(array(c(0.5, 0, 1, 2, 0, 1, NaN, 1, 0.00100000005, 3)),
    "float32"), as_jax(array(c(0.25, 1, 0, 2.04999995, 0, -1, 1, 1, 1000, Inf)), "float32")),
    list("float32", c(10L), c(0.0965735912, 1, Inf, 0.000614773657, 0, Inf, NaN, 0, 999.985229,
    NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(p, q) scipy_special_kl_div(p, q))(as_jax(array(c(0.5, 0, 1, 2, 0, 1, NaN,
    1, 0.001, 3)), "float64"), as_jax(array(c(0.25, 1, 0, 0x1.0666666666666p+1, 0, -1, 1, 1, 1000,
    Inf)), "float64")), list("float64", c(10L), c(0x1.8b90bfbe8e7bcp-4, 1, Inf,
    0x1.4251adefa6689p-11, 0, Inf, NaN, 0, 0x1.f3fe1a867d571p+9, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_rel_entr(as_jax(array(c(0.5, 0, 1, 2, 0, 1, NaN, 1, 0.00100000005, 3)),
    "float32"), as_jax(array(c(0.25, 1, 0, 2.04999995, 0, -1, 1, 1, 1000, Inf)), "float32")),
    list("float32", c(10L), c(0.346573591, 0, Inf, -0.0493851788, 0, Inf, NaN, 0, -0.013815511,
    -Inf)))
  enable_x64(TRUE)
  sp_expect(jit(function(p, q) scipy_special_rel_entr(p, q))(as_jax(array(c(0.5, 0, 1, 2, 0, 1, NaN,
    1, 0.001, 3)), "float64"), as_jax(array(c(0.25, 1, 0, 0x1.0666666666666p+1, 0, -1, 1, 1, 1000,
    Inf)), "float64")), list("float64", c(10L), c(0x1.62e42fefa39efp-2, 0, Inf,
    -0x1.949052e1d202ep-5, 0, Inf, NaN, 0, -0x1.c4b4e70312687p-7, -Inf)))
  enable_x64(FALSE)
  sp_expect(scipy_special_xlogy(as_jax(array(c(0L, 2L)), "int32"), as_jax(array(c(3, 3)),
    "float32")), list("float32", c(2L), c(0, 2.19722462)))
})

test_that("logsumexp and softmax match JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(scipy_special_logsumexp(as_jax(matrix(c(1, 2, 3, -1, 0.5, 1000), 2, byrow = TRUE),
    "float32")), list("float32", integer(0), c(1000)))
  enable_x64(TRUE)
  sp_expect(jit(function(a) scipy_special_logsumexp(a))(as_jax(matrix(c(1, 2, 3, -1, 0.5, 1000), 2,
    byrow = TRUE), "float64")), list("float64", integer(0), c(1000)))
  enable_x64(FALSE)
  sp_expect(scipy_special_logsumexp(as_jax(matrix(c(1, 2, 3, -1, 0.5, 1000), 2, byrow = TRUE),
    "float32"), axis = 1L, keepdims = TRUE), list("float32", c(2L, 1L), c(3.40760589, 1000)))
  enable_x64(TRUE)
  sp_expect(jit(function(a) scipy_special_logsumexp(a, axis = 1L, keepdims =
    TRUE))(as_jax(matrix(c(1, 2, 3, -1, 0.5, 1000), 2, byrow = TRUE), "float64")), list("float64",
    c(2L, 1L), c(0x1.b42c6ea778b93p+1, 1000)))
  enable_x64(FALSE)
  sp_expect(scipy_special_logsumexp(as_jax(matrix(c(1, 2, 3, -1, 0.5, 1000), 2, byrow = TRUE),
    "float32"), axis = 0L, b = as_jax(matrix(c(1, -2, 0, 0.5, 1, 2), 2, byrow = TRUE), "float32")),
    list("float32", c(3L), c(1.06547642, NaN, 1000.69318)))
  enable_x64(TRUE)
  sp_expect(jit(function(a, b) scipy_special_logsumexp(a, axis = 0L, b = b))(as_jax(matrix(c(1, 2,
    3, -1, 0.5, 1000), 2, byrow = TRUE), "float64"), as_jax(matrix(c(1, -2, 0, 0.5, 1, 2), 2, byrow
    = TRUE), "float64")), list("float64", c(3L), c(0x1.10c3114d31fbep+0, NaN,
    0x1.f458b90bfbe8ep+9)))
  enable_x64(FALSE)
  sp_expect(scipy_special_logsumexp(as_jax(matrix(c(1, 2, 3, -1, 0.5, 1000), 2, byrow = TRUE),
    "float32"), b = as_jax(matrix(c(1, -2, 0, 0.5, 1, -2), 2, byrow = TRUE), "float32"), return_sign
    = TRUE), list("float32", integer(0), c(1000.69318)), list("float32", integer(0), c(-1)))
  enable_x64(TRUE)
  sp_expect(jit(function(a, b) scipy_special_logsumexp(a, b = b, return_sign =
    TRUE))(as_jax(matrix(c(1, 2, 3, -1, 0.5, 1000), 2, byrow = TRUE), "float64"), as_jax(matrix(c(1,
    -2, 0, 0.5, 1, -2), 2, byrow = TRUE), "float64")), list("float64", integer(0),
    c(0x1.f458b90bfbe8ep+9)), list("float64", integer(0), c(-1)))
  enable_x64(FALSE)
  sp_expect(scipy_special_logsumexp(as_jax(matrix(c(1, 2, 3, -1, 0.5, 1000), 2, byrow = TRUE),
    "float32"), axis = 1L, where = as_jax(matrix(c(TRUE, FALSE, TRUE, TRUE, TRUE, FALSE), 2, byrow =
    TRUE), "bool")), list("float32", c(2L), c(3.12692809, 0.701413333)))
  enable_x64(TRUE)
  sp_expect(jit(function(a, w) scipy_special_logsumexp(a, axis = 1L, where = w))(as_jax(matrix(c(1,
    2, 3, -1, 0.5, 1000), 2, byrow = TRUE), "float64"), as_jax(matrix(c(TRUE, FALSE, TRUE, TRUE,
    TRUE, FALSE), 2, byrow = TRUE), "bool")), list("float64", c(2L), c(0x1.903f2d54301d5p+1,
    0x1.671fa423d5084p-1)))
  enable_x64(FALSE)
  sp_expect(scipy_special_softmax(as_jax(matrix(c(1, 2, 3, -1, 0.5, 1000), 2, byrow = TRUE),
    "float32")), list("float32", c(2L, 3L), c(0, 0, 0, 0, 0, 1)))
  enable_x64(TRUE)
  sp_expect(jit(function(a) scipy_special_softmax(a))(as_jax(matrix(c(1, 2, 3, -1, 0.5, 1000), 2,
    byrow = TRUE), "float64")), list("float64", c(2L, 3L), c(0, 0, 0, 0, 0, 1)))
  enable_x64(FALSE)
  sp_expect(scipy_special_softmax(as_jax(matrix(c(1, 2, 3, -1, 0.5, 1000), 2, byrow = TRUE),
    "float32"), axis = 0L), list("float32", c(2L, 3L), c(0.880797029, 0.817574441, 0, 0.119202912,
    0.182425514, 1)))
  enable_x64(TRUE)
  sp_expect(jit(function(a) scipy_special_softmax(a, axis = 0L))(as_jax(matrix(c(1, 2, 3, -1, 0.5,
    1000), 2, byrow = TRUE), "float64")), list("float64", c(2L, 3L), c(0x1.c2f7d5a8a79c9p-1,
    0x1.a2991f2a97914p-1, 0, 0x1.e84152bac31aep-4, 0x1.759b8355a1bafp-3, 1)))
  enable_x64(FALSE)
  sp_expect(scipy_special_log_softmax(as_jax(matrix(c(1, 2, 3, -1, 0.5, 1000), 2, byrow = TRUE),
    "float32"), axis = -1L), list("float32", c(2L, 3L), c(-2.40760589, -1.40760589, -0.407605946,
    -1001, -999.5, 0)))
  enable_x64(TRUE)
  sp_expect(jit(function(a) scipy_special_log_softmax(a, axis = -1L))(as_jax(matrix(c(1, 2, 3, -1,
    0.5, 1000), 2, byrow = TRUE), "float64")), list("float64", c(2L, 3L), c(-0x1.342c6ea778b93p+1,
    -0x1.6858dd4ef1726p+0, -0x1.a163753bc5c9ap-2, -1001, -999.5, 0)))
  enable_x64(FALSE)
  sp_expect(scipy_special_log_softmax(as_jax(matrix(c(1, 2, 3, -1, 0.5, 1000), 2, byrow = TRUE),
    "float32")), list("float32", c(2L, 3L), c(-999, -998, -997, -1001, -999.5, 0)))
  enable_x64(TRUE)
  sp_expect(jit(function(a) scipy_special_log_softmax(a))(as_jax(matrix(c(1, 2, 3, -1, 0.5, 1000),
    2, byrow = TRUE), "float64")), list("float64", c(2L, 3L), c(-999, -998, -997, -1001, -999.5,
    0)))
  enable_x64(FALSE)
  sp_expect(scipy_special_softmax(as_jax(array(c(1L, 2L, 3L)), "int32")), list("float32", c(3L),
    c(0.0900305733, 0.244728476, 0.665240943)))
  sp_expect(scipy_special_logsumexp(as_jax(array(c(1L, 2L, 3L)), "int32")), list("float32",
    integer(0), c(3.40760589)))
})

test_that("Bessel functions match JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(scipy_special_i0e(as_jax(array(c(0, -0.5, 1.5, 9, 50, 9.99999975e-06)), "float32")),
    list("float32", c(6L), c(1.00000012, 0.645035326, 0.367433637, 0.134959534, 0.0565616265,
    0.999990106)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_i0e(x))(as_jax(array(c(0, -0.5, 1.5, 9, 50,
    0x1.4f8b588e368f1p-17)), "float64")), list("float64", c(6L), c(1, 0x1.4a42101eb0f47p-1,
    0x1.784084195afep-2, 0.134959524581723, 0x1.cf5a5415199f9p-5, 0x1.fffeb0754c5ecp-1)))
  enable_x64(FALSE)
  sp_expect(scipy_special_i1e(as_jax(array(c(0, -0.5, 1.5, 9, 50, 9.99999975e-06)), "float32")),
    list("float32", c(6L), c(0, -0.156420797, 0.21903941, 0.127224997, 0.0559931286,
    4.99994985e-06)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_i1e(x))(as_jax(array(c(0, -0.5, 1.5, 9, 50,
    0x1.4f8b588e368f1p-17)), "float64")), list("float64", c(6L), c(0, -0x1.40598cd0be9fp-3,
    0x1.c097b8ec11eb4p-3, 0x1.048e8849e9048p-3, 0x1.cab2177d77b38p-5, 0x1.4f8a7ca791d2cp-18)))
  enable_x64(FALSE)
  sp_expect(scipy_special_i0(as_jax(array(c(0, -0.5, 1.5, 9, 50, 9.99999975e-06)), "float32")),
    list("float32", c(6L), c(1.00000012, 1.06348348, 1.64672327, 1093.58838, 2.93255373e+20,
    1.00000012)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_i0(x))(as_jax(array(c(0, -0.5, 1.5, 9, 50,
    0x1.4f8b588e368f1p-17)), "float64")), list("float64", c(6L), c(1, 0x1.104072392c8p+0,
    0x1.a58fa6a5a3ad9p+0, 0x1.1165a799ae33bp+10, 0x1.fcb78c9d0a576p+67, 0x1.000000001b7cdp+0)))
  enable_x64(FALSE)
  sp_expect(scipy_special_i1(as_jax(array(c(0, -0.5, 1.5, 9, 50, 9.99999975e-06)), "float32")),
    list("float32", c(6L), c(0, -0.257894278, 0.981666505, 1030.91479, 2.90307873e+20,
    4.99999987e-06)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_i1(x))(as_jax(array(c(0, -0.5, 1.5, 9, 50,
    0x1.4f8b588e368f1p-17)), "float64")), list("float64", c(6L), c(0, -0x1.081571ddea057p-2,
    0x1.f69cfb6ca57ebp-1, 0x1.01ba8ad04fcf9p+10, 0x1.f79a9674e50dep+67, 0x1.4f8b588e4892dp-18)))
  enable_x64(FALSE)
  sp_expect(scipy_special_bessel_jn(as_jax(array(c(0.5, 2, 7.5)), "float32"), v = 3L),
    list("float32", c(4L, 3L), c(NaN, NaN, 0.266339689, NaN, NaN, 0.135248452, NaN, NaN,
    -0.23027344, NaN, NaN, -0.258060962)))
  enable_x64(TRUE)
  sp_expect(jit(function(z) scipy_special_bessel_jn(z, v = 3L))(as_jax(array(c(0.5, 2, 7.5)),
    "float64")), list("float64", c(4L, 3L), c(0x1.e07f1d54c3f34p-1, 0x1.ca873fb24cefap-3,
    0x1.10bb57e0e56e5p-2, 0x1.f02a71f4870d6p-3, 0x1.27487958371f1p-1, 0x1.14fd20aa5273cp-3,
    0x1.f56a93f863444p-6, 0x1.694d52d747c64p-2, -0x1.d79995fac5f0ep-3, 0x1.500880f70db5ap-9,
    0x1.081365fc429dp-3, -0x1.08411eba2acf7p-2)))
  enable_x64(FALSE)
  sp_expect(scipy_special_bessel_jn(as_jax(matrix(c(0.5, 2, 7.5, 1), 2, byrow = TRUE), "float32"), v
    = 1L, n_iter = 30L), list("float32", c(2L, 2L, 2L), c(0.938469887, 0.223890796, 0.266339689,
    0.765197635, 0.242268473, 0.576724827, 0.135248348, 0.440050572)))
  enable_x64(TRUE)
  sp_expect(jit(function(z) scipy_special_bessel_jn(z, v = 1L, n_iter = 30L))(as_jax(matrix(c(0.5,
    2, 7.5, 1), 2, byrow = TRUE), "float64")), list("float64", c(2L, 2L, 2L),
    c(0x1.e07f1d54c3f35p-1, 0x1.ca873fb24cef7p-3, 0x1.10bb57e0e56e8p-2, 0x1.87c7fdbd7b8efp-1,
    0x1.f02a71f4870d7p-3, 0x1.27487958371fp-1, 0x1.14fd20aa52739p-3, 0x1.c29c9ee970c6bp-2)))
  enable_x64(FALSE)
  sp_expect(scipy_special_i0(as_jax(array(c(0L, 1L, 2L)), "int32")), list("float32", c(3L),
    c(1.00000012, 1.26606596, 2.27958512)))
})

test_that("exponential, sine and cosine integrals match JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(scipy_special_expi(as_jax(array(c(-30, -2, -0.300000012, 0, 0.300000012, 1.89999998, 2,
    3, 6, 12, 20, 40, 70, Inf, NaN)), "float32")), list("float32", c(15L), c(-0x1.b37397f892cf6p-49,
    -0.0489005074, -0.905676663, -Inf, -0.302668512, 4.59371376, 4.9542346, 9.93383217, 85.9897614,
    14959.5342, 25615650, 6.03971884e+15, 3.64635285e+28, NaN, NaN)))
  enable_x64(TRUE)
  sp_expect(scipy_special_expi(as_jax(array(c(-30, -2, -0x1.3333333333333p-2, 0,
    0x1.3333333333333p-2, 0x1.e666666666666p+0, 2, 3, 6, 12, 20, 40, 70, Inf, NaN)), "float64")),
    list("float64", c(15L), c(-0x1.b37397e20f76ep-49, -0x1.9097cdc7f6565p-5, -0x1.cfb4d99f65a7cp-1,
    -Inf, -0x1.35eebdd6b302ap-2, 0x1.25ff67b129ce9p+2, 0x1.3d122cf9efdbp+2, 0x1.3de1f4d7d8d32p+3,
    0x1.57f5843502619p+6, 0x1.d37c42e699a86p+13, 0x1.86dd24a9ff9cdp+24, 0x1.575177da55f6ap+52,
    0x1.d747b13f2b734p+94, NaN, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_exp1(as_jax(array(c(0, 0.300000012, 1, 4, 30, 800, -1, Inf, NaN)),
    "float32")), list("float32", c(9L), c(Inf, 0.905676663, 0.219383955, 0.00377935241,
    0x1.b37397f892cf6p-49, 0, NaN, 0, NaN)))
  enable_x64(TRUE)
  sp_expect(scipy_special_exp1(as_jax(array(c(0, 0x1.3333333333333p-2, 1, 4, 30, 800, -1, Inf,
    NaN)), "float64")), list("float64", c(9L), c(Inf, 0x1.cfb4d99f65a7cp-1, 0x1.c14c5d3bf8f9cp-3,
    0x1.ef5e06002f2a7p-9, 0x1.b37397e20f76ep-49, 0, NaN, 0, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_expn(as_jax(array(c(0, 1, 2, 3, 5, 2, 0, 5000, 6000, 2.70000005, -1, 1, 3,
    NaN)), "float32"), as_jax(array(c(0.5, 0.5, 0, 0.5, 2.5, 30, 0, 3, 0.5, 1.5, 1, -1, 800, 1)),
    "float32")), list("float32", c(14L), c(1.21306133, 0.559773624, 1, 0.221604362, 0.0119073819,
    0x1.a635ba02cbe3dp-49, Inf, 9.95343089e-06, 0.00010109687, 0.0731007978, NaN, NaN, 0, NaN)))
  enable_x64(TRUE)
  sp_expect(scipy_special_expn(as_jax(array(c(0, 1, 2, 3, 5, 2, 0, 5000, 6000, 0x1.599999999999ap+1,
    -1, 1, 3, NaN)), "float64"), as_jax(array(c(0.5, 0.5, 0, 0.5, 2.5, 30, 0, 3, 0.5, 1.5, 1, -1,
    800, 1)), "float64")), list("float64", c(14L), c(0x1.368b2fc6f960ap+0, 0x1.1e9aa50574b82p-1, 1,
    0x1.c5d88249b3bd6p-3, 0x1.862e5777b4935p-7, 0x1.a635b8db2efccp-49, Inf, 0x1.4dfb527a0b3cap-17,
    0x1.a807ef1855eadp-14, 0x1.2b6bbaf7ea189p-4, NaN, NaN, 0, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_expn(as_jax(matrix(c(1, 3), 2, byrow = TRUE), "float32"),
    as_jax(array(c(0.5, 2.5, 10)), "float32")), list("float32", c(2L, 3L), c(0.559773624,
    0.0249149203, 4.15696923e-06, 0.221604362, 0.0162953697, 3.54876238e-06)))
  enable_x64(TRUE)
  sp_expect(scipy_special_expn(as_jax(matrix(c(1, 3), 2, byrow = TRUE), "float64"),
    as_jax(array(c(0.5, 2.5, 10)), "float64")), list("float64", c(2L, 3L), c(0x1.1e9aa50574b82p-1,
    0x1.9834bd5bdc855p-6, 0x1.16f82eb2ae2dfp-18, 0x1.c5d88249b3bd6p-3, 0x1.0afbbba323597p-6,
    0x1.dc4e8d87d93d9p-19)))
  enable_x64(FALSE)
  sp_expect(scipy_special_sici(as_jax(array(c(-30, -2, 0, 0.300000012, 4, 6, 8, 30, 2e+09, Inf,
    -Inf, NaN)), "float32")), list("float32", c(12L), c(-1.56675661, -1.60541296, 0, 0.298504055,
    1.75820315, 1.42468762, 1.57418692, 1.56675661, 1.57079637, 1.57079637, -1.57079637, NaN)),
    list("float32", c(12L), c(-0.0330324173, 0.422980845, -Inf, -0.649172962, -0.140981674,
    -0.0680572391, 0.122433886, -0.0330324173, 4.57355237e-10, 0, NaN, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_sici(x))(as_jax(array(c(-30, -2, 0, 0x1.3333333333333p-2,
    4, 6, 8, 30, 2000000000, Inf, -Inf, NaN)), "float64")), list("float64", c(12L),
    c(-0x1.9116f4e439771p+0, -0x1.9afc5847f10b8p+0, 0, 0x1.31ab0b4780007p-2, 0x1.c21999d582befp+0,
    0x1.6cb852c7c4a21p+0, 0x1.92fde85506872p+0, 0x1.9116f4e439771p+0, 0x1.921fb54364a81p+0,
    0x1.921fb54442d18p+0, -0x1.921fb54442d18p+0, NaN)), list("float64", c(12L),
    c(-0x1.0e99fffde5f4ap-5, 0x1.b121e2e9b12c6p-2, -Inf, -0x1.4c6065091e7c1p-1,
    -0x1.20bb032e1244p-3, -0x1.16c3314c7032cp-4, 0x1.f57d3b165b497p-4, -0x1.0e99fffde5f4ap-5,
    0x1.f6de0d721f46cp-32, 0, NaN, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_fresnel(as_jax(array(c(-30, -2, 0, 0.300000012, 1.60000002, 2, 6, 30,
    40000, Inf, -Inf, NaN)), "float32")), list("float32", c(12L), c(-0.489389688, -0.343415678, 0,
    0.0141169988, 0.638887644, 0.343415678, 0.446960747, 0.489389688, 0.5, 0.5, -0.5, NaN)),
    list("float32", c(12L), c(-0.499996245, -0.488253415, 0, 0.299400985, 0.365461797, 0.488253415,
    0.499531478, 0.499996245, 0.5, 0.5, -0.5, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) scipy_special_fresnel(x))(as_jax(array(c(-30, -2, 0,
    0x1.3333333333333p-2, 0x1.999999999999ap+0, 2, 6, 30, 40000, Inf, -Inf, NaN)), "float64")),
    list("float64", c(12L), c(-0x1.f522911aead01p-2, -0x1.5fa85c0e05e06p-2, 0, 0x1.ce95f660c2c2dp-7,
    0x1.471c4954fadfcp-1, 0x1.5fa85c0e05e06p-2, 0x1.c9b014f06e782p-2, 0x1.f522911aead01p-2,
    0x1.fffde9f6f1e3fp-2, 0.5, -0.5, NaN)), list("float64", c(12L), c(-0x1.ffff042a51674p-2,
    -0x1.f3f8b36d044cp-2, 0, 0x1.32962b6222d6ap-2, 0x1.763b966945a18p-2, 0x1.f3f8b36d044cp-2,
    0x1.ff852d570c8bfp-2, 0x1.ffff042a51674p-2, 0.5, 0.5, -0.5, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_fresnel(as_jax(array(c(0.5, 2)), "float16")), list("float16", c(2L),
    c(0.0647583008, 0.343505859)), list("float16", c(2L), c(0.492431641, 0.48828125)), tol = 2e-3)
})

test_that("spence and owens_t match JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(scipy_special_spence(as_jax(array(c(0, 0.300000012, 0.5, 1, 1.20000005, 1.5, 1.70000005,
    2, 30, -1, NaN)), "float32")), list("float32", c(11L), c(1.64493406, 0.889377594, 0.582240343,
    0, -0.19080019, -0.448414177, -0.605158329, -0.82246685, -7.28008461, NaN, NaN)))
  enable_x64(TRUE)
  sp_expect(scipy_special_spence(as_jax(array(c(0, 0x1.3333333333333p-2, 0.5, 1, 1.2, 1.5, 1.7, 2,
    30, -1, NaN)), "float64")), list("float64", c(11L), c(0x1.a51a6625307d3p+0,
    0x1.c75c810434968p-1, 0x1.2a1b6e2725678p-1, 0, -0x1.86c238fe9d4e4p-3, -0x1.cb2d180732026p-2,
    -0x1.35d75275e14efp-1, -0x1.a51a6625307dcp-1, -0x1.d1ece86894cfbp+2, NaN, NaN)))
  enable_x64(FALSE)
  sp_expect(scipy_special_owens_t(as_jax(array(c(-2, 0.300000012, 6, 0.5, 0, 0.100000001, 1, NaN, 1,
    0.5)), "float32"), as_jax(array(c(0.5, 2, 1, -3, 0.699999988, 5, 0, 1, Inf, 1)), "float32")),
    list("float32", c(10L), c(0.00862507802, 0.162604302, 4.93294794e-10, -0.151084036, 0.097200051,
    0.214695171, 0, NaN, 0.0793276355, 0.106671065)))
  enable_x64(TRUE)
  sp_expect(jit(function(h, a) scipy_special_owens_t(h, a))(as_jax(array(c(-2, 0x1.3333333333333p-2,
    6, 0.5, 0, 0x1.999999999999ap-4, 1, NaN, 1, 0.5)), "float64"), as_jax(array(c(0.5, 2, 1, -3,
    0x1.6666666666666p-1, 5, 0, 1, Inf, 1)), "float64")), list("float64", c(10L),
    c(0x1.1aa065efc720fp-7, 0x1.4d037c815c3cp-3, 0x1.0f30eefc15b2fp-31, -0x1.356b8cffabf05p-3,
    0x1.8e21a56263e83p-4, 0x1.b7b21abb7bed3p-3, 0, NaN, 0x1.44ed0bb7cb20cp-4,
    0x1.b4ecb76d956a4p-4)))
  sp_expect(scipy_special_owens_t(as_jax(array(c(0.300000012, 1)), "float32"), as_jax(array(c(0.5,
    2)), "float64")), list("float64", c(2L), c(0x1.1fef6f05061e4p-4, 0x1.4167db8e965cp-4)))
  sp_expect({ d <- as_jax(1, "float64"); scipy_special_owens_t(as_jax(array(c(0.300000012, 1)),
    "float32"), as_jax(array(c(0.5, 2)), "float32")) }, list("float64", c(2L),
    c(0x1.1fef6f0e56506p-4, 0x1.4167df42071bep-4)))
})

test_that("hypergeometric functions match JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(scipy_special_hyp1f1(as_jax(array(c(1.5, 2, 0, 1.5, 1.5, 3, -2, 0.5)), "float32"),
    as_jax(array(c(2.5, 3, 1.5, 1.5, 0, 2, 1.5, 1.5)), "float32"), as_jax(array(c(0.300000012, -4,
    2, 0.699999988, 1, 150, 1.20000005, -150)), "float32")), list("float32", c(8L), c(1.20088267,
    0.113552354, 1, 2.0137527, Inf, Inf, -0.21600008, -0)))
  enable_x64(TRUE)
  sp_expect(scipy_special_hyp1f1(as_jax(array(c(1.5, 2, 0, 1.5, 1.5, 3, -2, 0.5)), "float64"),
    as_jax(array(c(2.5, 3, 1.5, 1.5, 0, 2, 1.5, 1.5)), "float64"),
    as_jax(array(c(0x1.3333333333333p-2, -4, 2, 0x1.6666666666666p-1, 1, 150, 1.2, -150)),
    "float64")), list("float64", c(8L), c(0x1.336d0b7f0c5dbp+0, 0x1.d11ca9b3ace9ep-4, 1,
    0x1.01c2a61268987p+1, Inf, 0x1.9250c714a8ca9p+222, -0x1.ba5e353f7ced6p-3,
    -0x1.490c5e5eea278p-225)))
  enable_x64(FALSE)
  sp_expect(scipy_special_hyp1f1(as_jax(1.5, "float32"), as_jax(matrix(c(2.5, 3.5), 2, byrow =
    TRUE), "float32"), as_jax(array(c(0.300000012, -1)), "float32")), list("float32", c(2L, 2L),
    c(1.20088267, 0.568416953, 1.14000452, 0.669026613)))
  enable_x64(TRUE)
  sp_expect(scipy_special_hyp1f1(as_jax(1.5, "float64"), as_jax(matrix(c(2.5, 3.5), 2, byrow =
    TRUE), "float64"), as_jax(array(c(0x1.3333333333333p-2, -1)), "float64")), list("float64", c(2L,
    2L), c(0x1.336d0b7f0c5dbp+0, 0x1.23078ed4c5c2cp-1, 0x1.23d756a444b8cp+0, 0x1.568aa7cf4cd52p-1)))
  enable_x64(FALSE)
  sp_expect(scipy_special_hyp2f1(as_jax(array(c(1.5, 2, 0, 1, -2, 1.5, 1, 0.5, 2, 1.5, 1, 1)),
    "float32"), as_jax(array(c(2.5, 3, 1, 2, 3, 2.5, 2, 1.5, 3, 2.5, 2, 1)), "float32"),
    as_jax(array(c(4.5, 6, 2, 0, 4, 2.5, 1, 3.5, 4.5, 6, 4, 1.5)), "float32"),
    as_jax(array(c(0.300000012, 0.949999988, 0.5, 0.5, 0.699999988, 0.400000006, 0.300000012, 1,
    0.920000017, -0.800000012, 0.949999988, 1.5)), "float32")), list("float32", c(12L),
    c(1.32890928, 6.25137186, 1, Inf, 0.244000033, 2.15165734, 2.04081631, 1.47262144, 15.6832848,
    0.664880216, 2.44208217, Inf)))
  enable_x64(TRUE)
  sp_expect(scipy_special_hyp2f1(as_jax(array(c(1.5, 2, 0, 1, -2, 1.5, 1, 0.5, 2, 1.5, 1, 1)),
    "float64"), as_jax(array(c(2.5, 3, 1, 2, 3, 2.5, 2, 1.5, 3, 2.5, 2, 1)), "float64"),
    as_jax(array(c(4.5, 6, 2, 0, 4, 2.5, 1, 3.5, 4.5, 6, 4, 1.5)), "float64"),
    as_jax(array(c(0x1.3333333333333p-2, 0x1.e666666666666p-1, 0.5, 0.5, 0x1.6666666666666p-1,
    0x1.999999999999ap-2, 0x1.3333333333333p-2, 1, 0x1.d70a3d70a3d71p-1, -0x1.999999999999ap-1,
    0x1.e666666666666p-1, 1.5)), "float64")), list("float64", c(12L), c(0x1.5433653b33d67p+0,
    0x1.90167f33595a5p+2, 1, Inf, 0x1.f3b645a1cac0cp-3, 0x1.13698299dd679p+1, 0x1.05397829cbc15p+1,
    0x1.78fdb9effea54p+0, 0x1.f5dda3aa03a9cp+3, 0x1.546b30a9bb3e5p-1, 0x1.3896272d46d45p+1, Inf)))
})

test_that("Legendre functions and spherical harmonics match JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(suppressWarnings(scipy_special_lpmn_values(3L, 3L, as_jax(array(c(0.300000012, -0.5,
    0.899999976, 1)), "float32"), FALSE)), list("float32", c(4L, 4L, 4L), c(1, 1, 1, 1, 0.300000012,
    -0.5, 0.899999976, 1, -0.36500001, -0.125, 0.714999914, 1, -0.382500023, 0.4375, 0.472499788,
    0.99999994, 0, 0, 0, 0, -0.953939199, -0.866025388, -0.43588993, 0, -0.858545303, 1.29903805,
    -1.17690277, 0, 0.786999822, -0.324759513, -1.99419618, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.73000002,
    2.25, 0.570000052, 0, 4.09500027, -5.625, 2.5650003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -13.0212698, -9.74278545, -1.24228644, 0)))
  enable_x64(TRUE)
  sp_expect(jit(function(z) suppressWarnings(scipy_special_lpmn_values(3L, 3L, z,
    FALSE)))(as_jax(array(c(0x1.3333333333333p-2, -0.5, 0x1.ccccccccccccdp-1, 1)), "float64")),
    list("float64", c(4L, 4L, 4L), c(1, 1, 1, 1, 0x1.3333333333333p-2, -0.5, 0x1.ccccccccccccdp-1,
    1, -0x1.75c28f5c28f5cp-2, -0.125, 0x1.6e147ae147ae2p-1, 1, -0x1.87ae147ae147bp-2, 0.4375,
    0x1.e3d70a3d70a41p-2, 1, 0, 0, 0, 0, -0x1.e86ab810ea912p-1, -0x1.bb67ae8584caap-1,
    -0x1.be59eba3a1659p-2, 0, -0x1.b7933f426cb5cp-1, 0x1.4c8dc2e42398p+0, -0x1.2d497f0e7357cp+0, 0,
    0x1.92f1a4a78e518p-1, -0x1.4c8dc2e42398p-2, -0x1.fe83a58326ff6p+0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2.73, 0x1.1ffffffffffffp+1, 0x1.23d70a3d70a3bp-1, 0, 0x1.06147ae147ae1p+2,
    -0x1.67fffffffffffp+2, 0x1.4851eb851eb82p+1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0x1.a0ae3ea1a1b71p+3, -0x1.37c4e6b5e15e7p+3, -0x1.3e0677e49631dp+0, 0)))
  enable_x64(FALSE)
  sp_expect(suppressWarnings(scipy_special_lpmn_values(3L, 3L, as_jax(array(c(0.300000012, -0.5,
    0.899999976)), "float32"), TRUE)), list("float32", c(4L, 4L, 3L), c(0.282094777, 0.282094777,
    0.282094777, 0.146580741, -0.244301245, 0.439742208, -0.230235815, -0.0788478702, 0.451009899,
    -0.285479873, 0.326529235, 0.352651626, 0, 0, 0, -0.329580426, -0.299206704, -0.150597408,
    -0.221089274, 0.334523261, -0.303071439, 0.169561848, -0.0699705407, -0.429656476, 0, 0, 0, 0,
    0, 0, 0.351509511, 0.289705634, 0.0733921081, 0.279002041, -0.383244514, 0.174759522, 0, 0, 0,
    0, 0, 0, 0, 0, 0, -0.362185597, -0.270994812, -0.0345540978)))
  enable_x64(TRUE)
  sp_expect(jit(function(z) suppressWarnings(scipy_special_lpmn_values(3L, 3L, z,
    TRUE)))(as_jax(array(c(0x1.3333333333333p-2, -0.5, 0x1.ccccccccccccdp-1)), "float64")),
    list("float64", c(4L, 4L, 3L), c(0x1.20dd750429b6dp-2, 0x1.20dd750429b6dp-2,
    0x1.20dd750429b6dp-2, 0x1.2c3287b67ac5cp-3, -0x1.f454378577499p-3, 0x1.c24bcb91b828ap-2,
    -0x1.d785e3b3da18p-3, -0x1.42f601a8c679ap-4, 0x1.cdd58c9cde613p-2, -0x1.2454d768fe868p-2,
    0x1.4e5db1c867b9ap-2, 0x1.691d8290c1f18p-2, 0, 0, 0, -0x1.517d871da6cdp-2,
    -0x1.32633e6df28bcp-2, -0x1.346c69fc88aa5p-3, -0x1.c4ca727809bap-3, 0x1.568d44c02929bp-2,
    -0x1.36585bea42f63p-2, 0x1.5b433c583c0e3p-3, -0x1.1e9973d058e85p-4, -0x1.b7f7de21c52adp-2, 0, 0,
    0, 0, 0, 0, 0x1.67f21cd485307p-2, 0x1.28a8992aef2d9p-2, 0x1.2c9d3165888d9p-4,
    0x1.1db2b5a08232p-2, -0x1.887142a166e24p-2, 0x1.65e8513f3b037p-3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0x1.72e0c89e2d354p-2, -0x1.157faab4581e2p-2, -0x1.1b112f872ba25p-5)))
  enable_x64(FALSE)
  sp_expect(suppressWarnings(scipy_special_lpmn_values(0L, 0L, as_jax(array(c(0.300000012, -0.5)),
    "float32"), TRUE)), list("float32", c(1L, 1L, 2L), c(0.282094777, 0.282094777)))
  enable_x64(TRUE)
  sp_expect(jit(function(z) suppressWarnings(scipy_special_lpmn_values(0L, 0L, z,
    TRUE)))(as_jax(array(c(0x1.3333333333333p-2, -0.5)), "float64")), list("float64", c(1L, 1L, 2L),
    c(0x1.20dd750429b6dp-2, 0x1.20dd750429b6dp-2)))
  enable_x64(FALSE)
  sp_expect(suppressWarnings(scipy_special_lpmn_values(1L, 1L, as_jax(array(c(0.300000012, -0.5)),
    "float32"), FALSE)), list("float32", c(2L, 2L, 2L), c(1, 1, 0.300000012, -0.5, 0, 0,
    -0.953939199, -0.866025388)))
  enable_x64(TRUE)
  sp_expect(jit(function(z) suppressWarnings(scipy_special_lpmn_values(1L, 1L, z,
    FALSE)))(as_jax(array(c(0x1.3333333333333p-2, -0.5)), "float64")), list("float64", c(2L, 2L,
    2L), c(1, 1, 0x1.3333333333333p-2, -0.5, 0, 0, -0x1.e86ab810ea912p-1, -0x1.bb67ae8584caap-1)))
  enable_x64(FALSE)
  sp_expect(suppressWarnings(scipy_special_lpmn(4L, 4L, as_jax(array(c(0.300000012, -0.5,
    0.899999976)), "float32"))), list("float32", c(5L, 5L, 3L), c(1, 1, 1, 0.300000012, -0.5,
    0.899999976, -0.36500001, -0.125, 0.714999914, -0.382500023, 0.4375, 0.472499788, 0.0729374886,
    -0.2890625, 0.207937181, 0, 0, 0, -0.953939199, -0.866025388, -0.43588993, -0.858545303,
    1.29903805, -1.17690277, 0.786999822, -0.324759513, -1.99419618, 1.69562697, -1.35316467,
    -2.618608, 0, 0, 0, 0, 0, 0, 2.73000002, 2.25, 0.570000052, 4.09500027, -5.625, 2.5650003,
    -2.52524948, 4.21875, 6.65475082, 0, 0, 0, 0, 0, 0, 0, 0, 0, -13.0212698, -9.74278545,
    -1.24228644, -27.3446693, 34.0997505, -7.82640409, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    86.9504929, 59.0624924, 3.79050112)), list("float32", c(5L, 5L, 3L), c(0, 0, 0, 1, 1, 1,
    0.900000036, -1.5, 2.69999981, -0.825000048, 0.375, 4.57499933, -1.77750003, 1.56249988,
    6.00749779, 0, 0, 0, 0.31448549, -0.577350259, 2.06474137, -2.57878089, -1.73205078, 4.26713181,
    -4.55217743, 6.27868414, 3.56167579, 2.08818316, -5.77350235, -2.8631165, 0, 0, 0, 0, 0, 0,
    -1.80000007, 3, -5.39999962, 10.9499998, 3.75, -21.4499969, 30.3300018, -33.75, -45.0899811, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 12.8781796, -19.4855709, 17.6535416, -64.1047134, -1.43051147e-06,
    102.521301, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -114.660004, 157.5, -71.8200073)))
  enable_x64(TRUE)
  sp_expect(jit(function(z) suppressWarnings(scipy_special_lpmn(4L, 4L,
    z)))(as_jax(array(c(0x1.3333333333333p-2, -0.5, 0x1.ccccccccccccdp-1)), "float64")),
    list("float64", c(5L, 5L, 3L), c(1, 1, 1, 0x1.3333333333333p-2, -0.5, 0x1.ccccccccccccdp-1,
    -0x1.75c28f5c28f5cp-2, -0.125, 0x1.6e147ae147ae2p-1, -0x1.87ae147ae147bp-2, 0.4375,
    0x1.e3d70a3d70a41p-2, 0x1.2ac083126e97ap-4, -0.2890625, 0x1.a9db22d0e560cp-3, 0, 0, 0,
    -0x1.e86ab810ea912p-1, -0x1.bb67ae8584caap-1, -0x1.be59eba3a1659p-2, -0x1.b7933f426cb5cp-1,
    0x1.4c8dc2e42398p+0, -0x1.2d497f0e7357cp+0, 0x1.92f1a4a78e518p-1, -0x1.4c8dc2e42398p-2,
    -0x1.fe83a58326ff6p+0, 0x1.b2149b44cb59fp+0, -0x1.5a6900584fbe5p+0, -2.61860854033206, 0, 0, 0,
    0, 0, 0, 2.73, 0x1.1ffffffffffffp+1, 0x1.23d70a3d70a3bp-1, 0x1.06147ae147ae1p+2,
    -0x1.67fffffffffffp+2, 0x1.4851eb851eb82p+1, -0x1.433b645a1cac2p+1, 0x1.0dfffffffffffp+2,
    0x1.a9e76c8b43954p+2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0x1.a0ae3ea1a1b71p+3, -0x1.37c4e6b5e15e7p+3,
    -0x1.3e0677e49631dp+0, -0x1.b583c1c350337p+4, 0x1.10cc49df2532ap+5, -0x1.f4e3c9a19fc1ap+2, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x1.5bcd4fdf3b646p+6, 0x1.d87fffffffffep+5,
    0x1.e52f1a9fbe766p+1)), list("float64", c(5L, 5L, 3L), c(0, 0, 0, 1, 1, 1, 0x1.cccccccccccccp-1,
    -1.5, 0x1.599999999999ap+1, -0x1.a666666666666p-1, 0x1.7fffffffffffbp-2, 0x1.24ccccccccccdp+2,
    -0x1.c70a3d70a3d72p+0, 0x1.9000000000002p+0, 0x1.807ae147ae149p+2, 0, 0, 0,
    0x1.4208795cbc6aep-2, -0x1.279a74590331dp-1, 0x1.084973f1ba83p+1, -0x1.4a157c657453fp+1,
    -0x1.bb67ae8584caap+0, 0x1.1118b38ada543p+2, -0x1.2356ddcb7f425p+2, 0x1.91d5f6290057bp+2,
    0x1.c7e51b342e88fp+1, 0x1.0b4997ee403a1p+1, -0x1.7181116f43fe4p+2, -0x1.6e7a55aec52f3p+1, 0, 0,
    0, 0, 0, 0, -0x1.cccccccccccccp+0, 3, -0x1.599999999999ap+2, 0x1.5e66666666666p+3, 3.75,
    -0x1.5733333333334p+4, 0x1.e547ae147ae15p+4, -33.75, -0x1.68b851eb851eep+5, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0x1.9c1a0b4e45ea7p+3, -0x1.37c4e6b5e15e8p+4, 0x1.1a74e71d8c224p+4, -0x1.006b3a3c14bf8p+6,
    0x1p-47, 0x1.9a15d07e55e22p+6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -114.66,
    0x1.3afffffffffffp+7, -0x1.1f47ae147ae12p+6)))
  enable_x64(FALSE)
  sp_expect(suppressWarnings(scipy_special_lpmn(1L, 1L, as_jax(array(c(0.300000012, -0.5)),
    "float32"))), list("float32", c(2L, 2L, 2L), c(1, 1, 0.300000012, -0.5, 0, 0, -0.953939199,
    -0.866025388)), list("float32", c(2L, 2L, 2L), c(0, 0, 1, 1, 0, 0, 0.31448549, -0.577350259)))
  enable_x64(TRUE)
  sp_expect(jit(function(z) suppressWarnings(scipy_special_lpmn(1L, 1L,
    z)))(as_jax(array(c(0x1.3333333333333p-2, -0.5)), "float64")), list("float64", c(2L, 2L, 2L),
    c(1, 1, 0x1.3333333333333p-2, -0.5, 0, 0, -0x1.e86ab810ea912p-1, -0x1.bb67ae8584caap-1)),
    list("float64", c(2L, 2L, 2L), c(0, 0, 1, 1, 0, 0, 0x1.4208795cbc6aep-2,
    -0x1.279a74590331dp-1)))
  enable_x64(FALSE)
  sp_expect(suppressWarnings(scipy_special_lpmn(2L, 2L, as_jax(array(c(0.300000012, -0.5)),
    "float32"))), list("float32", c(3L, 3L, 2L), c(1, 1, 0.300000012, -0.5, -0.36500001, -0.125, 0,
    0, -0.953939199, -0.866025388, -0.858545303, 1.29903805, 0, 0, 0, 0, 2.73000002, 2.25)),
    list("float32", c(3L, 3L, 2L), c(0, 0, 1, 1, 0.900000036, -1.5, 0, 0, 0.31448549, -0.577350259,
    -2.57878089, -1.73205078, 0, 0, 0, 0, -1.80000007, 3)))
  enable_x64(TRUE)
  sp_expect(jit(function(z) suppressWarnings(scipy_special_lpmn(2L, 2L,
    z)))(as_jax(array(c(0x1.3333333333333p-2, -0.5)), "float64")), list("float64", c(3L, 3L, 2L),
    c(1, 1, 0x1.3333333333333p-2, -0.5, -0x1.75c28f5c28f5cp-2, -0.125, 0, 0, -0x1.e86ab810ea912p-1,
    -0x1.bb67ae8584caap-1, -0x1.b7933f426cb5cp-1, 0x1.4c8dc2e42398p+0, 0, 0, 0, 0, 2.73,
    0x1.1ffffffffffffp+1)), list("float64", c(3L, 3L, 2L), c(0, 0, 1, 1, 0x1.cccccccccccccp-1, -1.5,
    0, 0, 0x1.4208795cbc6aep-2, -0x1.279a74590331dp-1, -0x1.4a157c657453fp+1, -0x1.bb67ae8584caap+0,
    0, 0, 0, 0, -0x1.cccccccccccccp+0, 3)))
  enable_x64(FALSE)
  sp_expect(scipy_special_sph_harm_y(as_jax(array(c(1L, 2L, 3L, 0L, 3L)), "int32"),
    as_jax(array(c(0L, -1L, 2L, 0L, -3L)), "int32"), as_jax(array(c(0.300000012, 1.5, 2.20000005,
    0.899999976, 1.10000002)), "float32"), as_jax(array(c(0.699999988, 2.5, -1, 0.200000003, 4)),
    "float32")), list("complex64", c(5L), c(complex(real = 0.466779798, imaginary = 0), complex(real
    = -0.0436711535, imaginary = -0.0326233245), complex(real = 0.163604319, imaginary =
    0.357481956), complex(real = 0.282094777, imaginary = 0), complex(real = 0.249214143, imaginary
    = 0.158465296))))
  enable_x64(TRUE)
  sp_expect(scipy_special_sph_harm_y(as_jax(array(c(1L, 2L, 3L, 0L, 3L)), "int32"),
    as_jax(array(c(0L, -1L, 2L, 0L, -3L)), "int32"), as_jax(array(c(0x1.3333333333333p-2, 1.5,
    0x1.199999999999ap+1, 0x1.ccccccccccccdp-1, 0x1.199999999999ap+0)), "float64"),
    as_jax(array(c(0x1.6666666666666p-1, 2.5, -1, 0x1.999999999999ap-3, 4)), "float64")),
    list("complex128", c(5L), c(complex(real = 0x1.ddfb86ac5079cp-2, imaginary = 0), complex(real =
    -0x1.65c10c849762dp-5, imaginary = -0x1.0b4012d8cd502p-5), complex(real = 0x1.4f0fc9b2ddba6p-3,
    imaginary = 0x1.6e0fc206cd0b2p-2), complex(real = 0x1.20dd750429b6dp-2, imaginary = 0),
    complex(real = 0x1.fe63fc01aab9p-3, imaginary = 0x1.448972c7b4ef9p-3))))
  enable_x64(FALSE)
  sp_expect(scipy_special_sph_harm_y(as_jax(array(c(1L, 2L)), "int32"), as_jax(array(c(1L, -2L)),
    "int32"), as_jax(array(c(0.300000012, 1.5)), "float32"), as_jax(array(c(0.699999988, 2.5)),
    "float32"), n_max = 4L), list("complex64", c(2L), c(complex(real = -0.0780907571, imaginary =
    -0.0657749325), complex(real = 0.109023124, imaginary = 0.368554294))))
  enable_x64(TRUE)
  sp_expect(jit(function(n, m, t, p) scipy_special_sph_harm_y(n, m, t, p, n_max =
    4L))(as_jax(array(c(1L, 2L)), "int32"), as_jax(array(c(1L, -2L)), "int32"),
    as_jax(array(c(0x1.3333333333333p-2, 1.5)), "float64"), as_jax(array(c(0x1.6666666666666p-1,
    2.5)), "float64")), list("complex128", c(2L), c(complex(real = -0x1.3fdc1be384aa8p-4, imaginary
    = -0x1.0d6a0893c9dd7p-4), complex(real = 0x1.be8f05f28e5ep-4, imaginary =
    0x1.79664b1b38a4dp-2))))
})

test_that("gradients of the gamma and beta families match jax.grad", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(TRUE)
  sp_expect(jit(function(x) grad(function(x)
    jnp_sum(scipy_special_gamma(x)))(x))(as_jax(array(c(0.5, 0x1.999999999999ap+1, -1.5)),
    "float64")), list("float64", c(3L), c(-0x1.bd7834d3dc468p+1, 0x1.35e84695b92dap+1,
    0x1.a968770fb0ecp+0)))
  sp_expect(jit(function(x) grad(function(x)
    jnp_sum(scipy_special_loggamma(x)))(x))(as_jax(array(c(0.5, 0x1.999999999999ap+1)), "float64")),
    list("float64", c(2L), c(-0x1.f6a897d321505p+0, 0x1.ff67cfa819028p-1)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_betaln(a, b)), argnums =
    c(0L, 1L))(a, b))(as_jax(array(c(0.5, 2, 30, 5)), "float64"), as_jax(array(c(1.5, 3, 2, 5)),
    "float64")), list("float64", c(4L), c(-0x1.317217f7d1cfdp+1, -0x1.1555555555553p+0,
    -0x1.0ca9912c8c29ep-4, -0x1.7dc3dc3dc3dc8p-1)), list("float64", c(4L), c(-0x1.8b90bfbe8e7b8p-2,
    -0x1.2aaaaaaaaaaabp-1, -0x1.837cc539fba8ep+1, -0x1.7dc3dc3dc3dc8p-1)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_beta(a, b)), argnums =
    c(0L, 1L))(a, b))(as_jax(array(c(0.5, 2)), "float64"), as_jax(array(c(1.5, 3)), "float64")),
    list("float64", c(2L), c(-0x1.dfcafebb2b75p+1, -0x1.71c71c71c71b7p-4)), list("float64", c(2L),
    c(-0x1.36ad25dba28aap-1, -0x1.8e38e38e38e2ap-5)))
  sp_expect(jit(function(x) grad(function(x)
    jnp_sum(scipy_special_factorial(x)))(x))(as_jax(array(c(0.5, 3)), "float64")), list("float64",
    c(2L), c(0x1.08ea88ee561ffp-5, 0x1.e259642ac3774p+2)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_comb(a, b)), argnums =
    c(0L, 1L))(a, b))(as_jax(array(c(5, 10.5)), "float64"), as_jax(array(c(2, 3.5)), "float64")),
    list("float64", c(2L), c(0x1.2000000000008p+2, 0x1.36a31cf9595ap+6)), list("float64", c(2L),
    c(0x1.aaaaaaaaaaac4p+1, 0x1.fce292919f53cp+6)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(scipy_special_multigammaln(x,
    3L)))(x))(as_jax(array(c(1.5, 2.5)), "float64")), list("float64", c(2L),
    c(-0x1.408acbc4e0286p+1, 0x1.29951320ea5adp+0)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_poch(a, b)), argnums =
    c(0L, 1L))(a, b))(as_jax(array(c(0.5, 2, 3.5)), "float64"), as_jax(array(c(1.5,
    0x1.3333333333333p-2, 2)), "float64")), list("float64", c(3L), c(0x1.58a89613997bep+0,
    0x1.a789fc40119cap-3, 0x1.000000000000ap+3)), list("float64", c(3L), c(0x1.e882b02d892fcp-3,
    0x1.66700e6c2f2e4p-1, 0x1.95fed758f803p+4)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_zeta(a, b)), argnums =
    c(0L, 1L))(a, b))(as_jax(array(c(2, 3.5, 0.5)), "float64"), as_jax(array(c(0.5, 1.5, 1)),
    "float64")), list("float64", c(3L), c(0x1.bf823a80bf5a8p+0, -0.172822897058728, NaN)),
    list("float64", c(3L), c(-0x1.0d42c0452055cp+4, -0x1.48432d44dfb1ep-1, NaN)))
  enable_x64(FALSE)
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_zeta(a, b)), argnums =
    c(0L, 1L))(a, b))(as_jax(array(c(2, 3.5, 0.5)), "float32"), as_jax(array(c(0.5, 1.5, 1)),
    "float32")), list("float32", c(3L), c(1.74808109, -0.172822908, NaN)), list("float32", c(3L),
    c(-16.8287983, -0.64113754, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(s, q) vmap(grad(grad(scipy_special_zeta, argnums = 1L), argnums = 1L))(s,
    q))(as_jax(array(c(2, 3.5)), "float64"), as_jax(array(c(0.5, 1.5)), "float64")), list("float64",
    c(2L), c(0x1.85a2e8c290826p+6, 0x1.d1643f5546bdap+0)))
  sp_expect(jit(function(s, q) grad(function(s) jnp_sum(scipy_special_zeta(s,
    q)))(s))(as_jax(array(c(0.5, 2)), "float64"), as_jax(array(c(1, 1)), "float64")),
    list("float64", c(2L), c(NaN, -0x1.e00653256ab8ap-1)))
})

test_that("gradients of the logistic, entropy and softmax functions match jax.grad", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(TRUE)
  sp_expect(jit(function(x) grad(function(x)
    jnp_sum(scipy_special_logit(x)))(x))(as_jax(array(c(0x1.999999999999ap-4, 0.5,
    0x1.ccccccccccccdp-1)), "float64")), list("float64", c(3L), c(0x1.638e38e38e38dp+3, 4,
    0x1.638e38e38e38fp+3)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(jnp_sum(grad(function(y)
    jnp_sum(scipy_special_logit(y)))(x))))(x))(as_jax(array(c(0x1.999999999999ap-4,
    0x1.6666666666666p-1)), "float64")), list("float64", c(2L), c(-0x1.8b0fcd6e9e064p+6,
    0x1.223fdad91b486p+3)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(scipy_special_expit(x)))(x))(as_jax(array(c(-1,
    0, 2)), "float64")), list("float64", c(3L), c(0x1.92a946fa34394p-3, 0.25,
    0x1.ae0dc0f990c4cp-4)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_xlogy(a, b)), argnums =
    c(0L, 1L))(a, b))(as_jax(array(c(0, 2, 1)), "float64"), as_jax(array(c(3, 3, 0.5)), "float64")),
    list("float64", c(3L), c(0x1.193ea7aad030bp+0, 0x1.193ea7aad030bp+0, -0x1.62e42fefa39efp-1)),
    list("float64", c(3L), c(0, 0x1.5555555555555p-1, 2)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_xlog1py(a, b)), argnums =
    c(0L, 1L))(a, b))(as_jax(array(c(0, 2, 1)), "float64"), as_jax(array(c(3, 3, 0.5)), "float64")),
    list("float64", c(3L), c(0x1.62e42fefa39efp+0, 0x1.62e42fefa39efp+0, 0x1.9f323ecbf984cp-2)),
    list("float64", c(3L), c(0, 0.5, 0x1.5555555555555p-1)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(scipy_special_entr(x)))(x))(as_jax(array(c(0.5,
    2, 0.001)), "float64")), list("float64", c(3L), c(-0x1.3a37a020b8c22p-2, -0x1.b17217f7d1cf8p+0,
    0x1.7a18a998fffap+2)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_boxcox(a, b)), argnums =
    c(0L, 1L))(a, b))(as_jax(array(c(0.5, 2, 3)), "float64"), as_jax(array(c(0, 0.5, -1)),
    "float64")), list("float64", c(3L), c(2, 0x1.6a09e667f3bcdp-1, 0x1.c71c71c71c71cp-4)),
    list("float64", c(3L), c(0, 0x1.36f32e5f72754p-2, 0x1.33ac75c6ea69ep-2)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_boxcox1p(a, b)), argnums =
    c(0L, 1L))(a, b))(as_jax(array(c(0.5, 2, 3)), "float64"), as_jax(array(c(0, 0.5, -1)),
    "float64")), list("float64", c(3L), c(0x1.5555555555555p-1, 0x1.279a74590331dp-1, 0.0625)),
    list("float64", c(3L), c(0, 0x1.c147dc291f3b4p-1, 0x1.9d1bd0105c611p-2)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_kl_div(a, b)), argnums =
    c(0L, 1L))(a, b))(as_jax(array(c(0.5, 0, 2, 1, 3)), "float64"), as_jax(array(c(0.25, 1,
    0x1.0666666666666p+1, 1, Inf)), "float64")), list("float64", c(5L), c(0x1.62e42fefa39efp-1, 0,
    -0x1.949052e1d202ep-6, 0, -Inf)), list("float64", c(5L), c(-1, 1, 0x1.8f9c18f9c18e1p-6, -0, 1)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_rel_entr(a, b)), argnums =
    c(0L, 1L))(a, b))(as_jax(array(c(0.5, 0, 2, 1)), "float64"), as_jax(array(c(0.25, 1,
    0x1.0666666666666p+1, 1)), "float64")), list("float64", c(4L), c(0x1.b17217f7d1cf8p+0, 0,
    0x1.f35b7d68f16ffp-1, 1)), list("float64", c(4L), c(-2, 0, -0x1.f3831f3831f39p-1, -1)))
  sp_expect(jit(function(x) grad(function(x)
    jnp_sum(scipy_special_logsumexp(x)))(x))(as_jax(array(c(1, 2, 3, -1)), "float64")),
    list("float64", c(4L), c(0x1.6c537f76029dap-4, 0x1.ef2bac1f8869dp-3, 0x1.5080d8c5d3fd2p-1,
    0x1.8a7310e2652d8p-7)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_logsumexp(a, b = b)),
    argnums = c(0L, 1L))(a, b))(as_jax(array(c(1, 2, 3)), "float64"), as_jax(array(c(1, -0.5, 2)),
    "float64")), list("float64", c(3L), c(0x1.1c11f85f038fp-4, -0x1.821767b22365ap-4,
    0x1.066056f531fd6p+0)), list("float64", c(3L), c(0x1.1c11f85f038fp-4, 0x1.821767b22365ap-3,
    0x1.066056f531fd6p-1)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(lax_index_in_dim(scipy_special_softmax(x), 0L,
    keepdims = FALSE)))(x))(as_jax(array(c(1, 2, 3)), "float64")), list("float64", c(3L),
    c(0x1.4f90a945b4fdbp-4, -0x1.68fd49694f3a1p-6, -0x1.eaa2add6c25e6p-5)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(lax_index_in_dim(scipy_special_log_softmax(x),
    0L, keepdims = FALSE)))(x))(as_jax(array(c(1, 2, 3)), "float64")), list("float64", c(3L),
    c(0x1.d1e783412fa85p-1, -0x1.f534335ca4bcep-3, -0x1.549a766a06791p-1)))
})

test_that("custom derivatives of the normal and error functions match jax.grad", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(TRUE)
  sp_expect(jit(function(x) grad(function(x) jnp_sum(scipy_special_ndtr(x)))(x))(as_jax(array(c(-5,
    -0.5, 0x1.3333333333333p-2, 2)), "float64")), list("float64", c(4L), c(0x1.8f16964c8fd38p-20,
    0x1.6883d022086adp-2, 0x1.868a8709fbbbfp-2, 0x1.ba4b436e83ad3p-5)))
  sp_expect(jit(function(x) grad(function(x)
    jnp_sum(scipy_special_log_ndtr(x)))(x))(as_jax(array(c(-40, -5, -0.5, 0, 0x1.3333333333333p-2,
    6, 10)), "float64")), list("float64", c(7L), c(0x1.403322de073fcp+5, 0x1.4befae55d84e8p+2,
    0x1.241dac39ec665p+0, 0x1.9884533d43651p-1, 0x1.3c045f295a0f6p-1, 0x1.a1880fc3f1bbp-28,
    0x1.7416846b4d5d1p-74)))
  enable_x64(FALSE)
  sp_expect(jit(function(x) grad(function(x)
    jnp_sum(scipy_special_log_ndtr(x)))(x))(as_jax(array(c(-40, -5, -0.5, 0, 0.300000012, 6, 10)),
    "float32")), list("float32", c(7L), c(40.0253983, 5.18650436, 1.14107776, 0.797884583,
    0.617220879, 6.07588824e-09, 0x1.74169803c932fp-74)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) grad(function(x) jnp_sum(jnp_sum(grad(function(y)
    jnp_sum(scipy_special_log_ndtr(y)))(x))))(x))(as_jax(array(c(-5, 0x1.3333333333333p-2)),
    "float64")), list("float64", c(2L), c(-0x1.ef4269b53f72p-1, -0x1.21db820d0151ap-1)))
  sp_expect(jit(function(x) grad(function(x)
    jnp_sum(scipy_special_ndtri(x)))(x))(as_jax(array(c(0x1.999999999999ap-4, 0.5,
    0x1.fae147ae147aep-1, 1e-10)), "float64")), list("float64", c(4L), c(0x1.6cad033f2b127p+2,
    0x1.40d931ff62705p+1, 0x1.2c29da6e99e88p+5, 0x1.6e2522c8c9195p+30)))
  enable_x64(FALSE)
  sp_expect(jit(function(x) grad(function(x)
    jnp_sum(scipy_special_ndtri(x)))(x))(as_jax(array(c(0.100000001, 0.5, 0.99000001,
    1.00000001e-10)), "float32")), list("float32", c(4L), c(5.69806051, 2.50662827, 37.5204697,
    1.53572365e+09)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) grad(function(x) jnp_sum(jnp_sum(grad(function(y)
    jnp_sum(scipy_special_ndtri(y)))(x))))(x))(as_jax(array(c(0x1.999999999999ap-4,
    0x1.fae147ae147aep-1)), "float64")), list("float64", c(2L), c(-0x1.4cdfc91a54706p+5,
    0x1.995fc908ac5c7p+11)))
  sp_expect(jit(function(x) grad(function(x)
    jnp_sum(scipy_special_erfcinv(x)))(x))(as_jax(array(c(0x1.999999999999ap-4, 1,
    0x1.e666666666666p+0, 1e-10)), "float64")), list("float64", c(4L), c(-0x1.b6ca1b5a3c9d2p+1,
    -0x1.c5bf891b4ef6ap-1, -0x1.b6ca1b5a3c9cdp+1, -0x1.fdb77c0cc4246p+29)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(jnp_sum(grad(function(y)
    jnp_sum(scipy_special_erfcinv(y)))(x))))(x))(as_jax(array(c(0x1.999999999999ap-4,
    0x1.e666666666666p+0)), "float64")), list("float64", c(2L), c(0x1.b56031e2807f9p+4,
    -0x1.b56031e2807eep+4)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(scipy_special_erfcx(x)))(x))(as_jax(array(c(-2,
    0, 0x1.3333333333333p-2, 1.5, 5, 30)), "float64")), list("float64", c(6L),
    c(-0x1.b4e459e5c2a2cp+8, -0x1.20dd750429b6dp+0, -0x1.600fabfb2b0cbp-1, -0x1.4f1988444caf7p-3,
    -0x1.5d843497d4f3bp-6, -0x1.481e34f76a622p-11)))
  enable_x64(FALSE)
  sp_expect(jit(function(x) grad(function(x) jnp_sum(scipy_special_erfcx(x)))(x))(as_jax(array(c(-2,
    0, 0.300000012, 1.5, 5, 30)), "float32")), list("float32", c(6L), c(-436.892029, -1.12837911,
    -0.687619567, -0.163622916, -0.0213327892, -0.000625835499)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) vmap(grad(grad(scipy_special_erfcx)))(x))(as_jax(array(c(-2, 0,
    0x1.3333333333333p-2, 1, 1.5, 2.25, 5, 30)), "float64")), list("float64", c(8L),
    c(0x1.eb5cc973da71ep+10, 2, 0x1.0e7f1a39832eep+0, 0x1.3c27283c32cc4p-2, 0x1.37ea271bc54bcp-3,
    0x1.0615670e25a7cp-4, 0x1.08cf82b79a11cp-7, 0x1.5d6955b46bd2ap-15)))
  enable_x64(FALSE)
  sp_expect(jit(function(x) vmap(grad(grad(scipy_special_erfcx)))(x))(as_jax(array(c(-2, 0,
    0.300000012, 1, 1.5, 2.25, 5, 30)), "float32")), list("float32", c(8L), c(1965.44995, 2,
    1.05662692, 0.308743119, 0.152302086, 0.0639852881, 0.00808137655, 4.16530966e-05)))
  enable_x64(TRUE)
  sp_expect(jit(function(x)
    vmap(grad(grad(grad(scipy_special_ndtri))))(x))(as_jax(array(c(0x1.999999999999ap-4,
    0x1.6666666666666p-1)), "float64")), list("float64", c(2L), c(0x1.8c5905fa62201p+9,
    0x1.2701e889f1db7p+5)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(scipy_special_dawsn(x)))(x))(as_jax(array(c(-2,
    0x1.3333333333333p-2, 4, 30)), "float64")), list("float64", c(4L), c(-0x1.a494996c880b8p-3,
    0x1.a92cf1024da6cp-1, -0x1.1cf35b1c927p-5, -0x1.23c206d45ep-11)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(jnp_real(scipy_special_wofz(lax_complex(x, 0.5
    * x)))))(x))(as_jax(array(c(0x1.3333333333333p-2, -1.2)), "float64")), list("float64", c(2L),
    c(-0x1.8a350103fd80cp-1, 0x1.7803a340f6bcep+0)))
})

test_that("custom derivatives of the Bessel and integral functions match jax.grad", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(TRUE)
  sp_expect(jit(function(x) grad(function(x) jnp_sum(scipy_special_i0(x)))(x))(as_jax(array(c(0,
    -0.5, 1.5, 9)), "float64")), list("float64", c(4L), c(0, -0x1.081571ddea057p-2,
    0x1.f69cfb6ca57ebp-1, 0x1.01ba8ad04fcf9p+10)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(scipy_special_i1(x)))(x))(as_jax(array(c(0,
    -0.5, 1.5, 9, 0x1.12e0be826d695p-30)), "float64")), list("float64", c(5L), c(0.5,
    0x1.186b72946efa9p-1, 0x1.fc0bfb02d906dp-1, 0x1.e985693df555bp+9, 0.5)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(jnp_sum(grad(function(y)
    jnp_sum(scipy_special_i0(y)))(x))))(x))(as_jax(array(c(0, 1.5)), "float64")), list("float64",
    c(2L), c(0.5, 0x1.fc0bfb02d906dp-1)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(jnp_sum(grad(function(y)
    jnp_sum(scipy_special_i1(y)))(x))))(x))(as_jax(array(c(0, 0x1.12e0be826d695p-30, 1.5)),
    "float64")), list("float64", c(3L), c(0, 0x1.9c511dc3a41ep-32, 0x1.834c8af05e5d4p-1)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(scipy_special_i0e(x)))(x))(as_jax(array(c(0.5,
    -1.5)), "float64")), list("float64", c(2L), c(-0x1.f45759d502996p-2, 0x1.2fe94f46a410cp-3)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(scipy_special_bessel_jn(x, v =
    2L)))(x))(as_jax(array(c(0.5, 2)), "float64")), list("float64", c(2L), c(0x1.537923b6e7fb6p-2,
    -0x1.ab522c5658679p-2)))
  sp_expect(jit(function(x) grad(function(x)
    jnp_sum(scipy_special_spence(x)))(x))(as_jax(array(c(0x1.3333333333333p-2, 1.2, 1.7, 30)),
    "float64")), list("float64", c(4L), c(-0x1.b84f5fb58135bp+0, NaN, NaN, -0x1.e063ca8e0eccep-4)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(scipy_special_expi(x)))(x))(as_jax(array(c(-2,
    0x1.3333333333333p-2, 3, 20)), "float64")), list("float64", c(4L), c(-0x1.152aaa3bf81ccp-4,
    0x1.1ff849fca1628p+2, 0x1.ac7dcff3f96b3p+2, 0x1.7226d3c53ecdp+24)))
  sp_expect(jit(function(x) grad(function(x)
    jnp_sum(scipy_special_exp1(x)))(x))(as_jax(array(c(0x1.3333333333333p-2, 4)), "float64")),
    list("float64", c(2L), c(-0x1.3c151ad783cafp+1, -0x1.2c155b8213cf5p-8)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_expn(a, b)), argnums =
    c(0L, 1L))(a, b))(as_jax(array(c(0, 1, 3, 2)), "float64"), as_jax(array(c(0.5, 0.5, 2.5, 30)),
    "float64")), list("float64", c(4L), c(0, 0, 0, 0)), list("float64", c(4L),
    c(-0x1.d1d0c7aa7610fp+1, -0x1.368b2fc6f960ap+0, -0x1.445d96bf95bcp-6, -0x1.b37397e20f76ep-49)))
  sp_expect(jit(function(x) { x <- x; list(grad(function(x)
    jnp_sum(scipy_special_sici(x)[[1L]]))(x), grad(function(x)
    jnp_sum(scipy_special_sici(x)[[2L]]))(x)) })(as_jax(array(c(-2, 0x1.3333333333333p-2, 6, 30)),
    "float64")), list("float64", c(4L), c(0x1.d18f6ead1b445p-2, 0x1.f85abf98c8babp-1,
    -0x1.7d7ecb8eb83abp-5, -0x1.0dcc6aa315dfbp-5)), list("float64", c(4L), c(0x1.aa22657537205p-3,
    0x1.979c3862732c6p+1, 0x1.47bcf5ba56ca3p-3, 0x1.50f7a9c117169p-8)))
  sp_expect(jit(function(x, t) jvp(scipy_special_sici, list(x),
    list(t))[[2L]])(as_jax(array(c(0x1.3333333333333p-2, 6)), "float64"), as_jax(array(c(1, -2)),
    "float64")), list("float64", c(2L), c(0x1.f85abf98c8babp-1, 0x1.7d7ecb8eb83abp-4)),
    list("float64", c(2L), c(0x1.979c3862732c6p+1, -0x1.47bcf5ba56ca3p-2)))
  sp_expect(jit(function(x) { x <- x; list(grad(function(x)
    jnp_sum(scipy_special_fresnel(x)[[1L]]))(x), grad(function(x)
    jnp_sum(scipy_special_fresnel(x)[[2L]]))(x)) })(as_jax(array(c(-2, 0x1.3333333333333p-2, 6)),
    "float64")), list("float64", c(3L), c(0, 0x1.2090d3391aae9p-3, 0)), list("float64", c(3L), c(1,
    0x1.fae46180515dp-1, 1)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(scipy_special_owens_t(a, b)), argnums =
    c(0L, 1L))(a, b))(as_jax(array(c(-2, 0x1.3333333333333p-2, 0.5)), "float64"),
    as_jax(array(c(0.5, 2, -3)), "float64")), list("float64", c(3L), c(0x1.2df3101d7f064p-6,
    -0x1.60a75d0e1ef29p-4, 0x1.38584c2d20dcp-3)), list("float64", c(3L), c(0x1.567895e1cbb4cp-7,
    0x1.a070f9f4d847fp-6, 0x1.2ad5d90abe94dp-8)))
})

test_that("custom derivatives of the hypergeometric and Legendre functions match jax.grad", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(TRUE)
  sp_expect(jit(function(a, b, x) grad(function(a, b, x) jnp_sum(scipy_special_hyp1f1(a, b, x)),
    argnums = c(0L, 1L, 2L))(a, b, x))(as_jax(array(c(1.5, 2)), "float64"), as_jax(array(c(2.5, 3)),
    "float64"), as_jax(array(c(0x1.3333333333333p-2, -4)), "float64")), list("float64", c(2L),
    c(0x1.245bf89bf591ap-3, -0x1.41fcf14d92be7p-3)), list("float64", c(2L), c(-0x1.6317fe489b5dp-4,
    0x1.73e6feeca8944p-4)), list("float64", c(2L), c(0x1.7d6104e114f44p-1, 0x1.861752d327f75p-5)))
  sp_expect(jit(function(x) jacfwd(function(x) scipy_special_hyp1f1(1.5, 2.5,
    x))(x))(as_jax(array(c(0x1.3333333333333p-2, -4)), "float64")), list("float64", c(2L, 2L),
    c(0x1.7d6104e114f44p-1, 0, 0, 0x1.aeb6aff55e1f4p-5)))
  sp_expect(jit(function(a, b, c, x) grad(function(a, b, c, x) jnp_sum(scipy_special_hyp2f1(a, b, c,
    x)), argnums = c(0L, 1L, 2L, 3L))(a, b, c, x))(as_jax(array(c(1.5, 2)), "float64"),
    as_jax(array(c(2.5, 3)), "float64"), as_jax(array(c(4.5, 6)), "float64"),
    as_jax(array(c(0x1.3333333333333p-2, 0x1.3333333333333p-1)), "float64")), list("float64", c(2L),
    c(0x1.07debee8721fdp-2, 0x1.e690c809d81f6p-1)), list("float64", c(2L), c(0x1.46a816c3dddf4p-3,
    0x1.57b93d280955cp-1)), list("float64", c(2L), c(-0x1.75a8a7a3bcfb3p-4, -0x1.7449da204152p-2)),
    list("float64", c(2L), c(0x1.71ec7c555168fp+0, 0x1.0809438ca9785p+2)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(suppressWarnings(scipy_special_lpmn_values(2L,
    2L, x, FALSE))))(x))(as_jax(array(c(0x1.3333333333333p-2, -0.5)), "float64")), list("float64",
    c(2L), c(-0x1.1507a06d0ff9ep+1, 0x1.8658ba6fcce5p-3)))
  sp_expect(jit(function(t) grad(function(t) jnp_sum(jnp_real(scipy_special_sph_harm_y(c(1L, 2L),
    c(0L, -1L), t, as_jax(c(0.7, 2.5), "float64"), n_max =
    2L))))(t))(as_jax(array(c(0x1.3333333333333p-2, 1.5)), "float64")), list("float64", c(2L),
    c(-0x1.27b6f2cfea404p-3, 0x1.39b787ef41ebcp-1)))
})

test_that("base Math generics dispatch to the special functions", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  v <- c(0.5, 1.5, 3.2, 7.25)
  x <- as_jax(v, "float64")
  expect_jax_equal(lgamma(x), base::lgamma(v), dtype = "float64")
  expect_jax_equal(gamma(x), base::gamma(v), dtype = "float64")
  expect_jax_equal(digamma(x), base::digamma(v), dtype = "float64")
  expect_jax_equal(trigamma(x), base::trigamma(v), dtype = "float64")
  expect_jax_equal(lgamma(x), scipy_special_gammaln(x))
  expect_jax_equal(trigamma(x), scipy_special_polygamma(1L, x))
  # the generics are differentiable: d lgamma = digamma, d digamma = trigamma
  expect_jax_equal(grad(function(x) jnp_sum(lgamma(x)))(x), base::digamma(v))
  expect_jax_equal(grad(function(x) jnp_sum(digamma(x)))(x), base::trigamma(v))
  enable_x64(FALSE)
  expect_identical(dtype(trigamma(as_jax(v, "float32"))), "float32")
})

test_that("special functions vmap like stacked loops", {
  skip_if_no_backend()
  a <- as_jax(matrix(c(2, 0.5, 3, 1.5, 4, 0.7), 2, byrow = TRUE), "float32")
  b <- as_jax(matrix(c(3, 0.5, 1.5, 2, 1, 2.5), 2, byrow = TRUE), "float32")
  x <- as_jax(matrix(c(0.5, 1.5, 3.2, 2.5, 0.3, 6), 2, byrow = TRUE), "float32")
  both <- function(f, ...) expect_jax_equal(jit(vmap(f))(...), sp_rows(f, ...))
  # thin wrappers and custom JVP functions
  both(scipy_special_gammainc, a, x)
  both(scipy_special_betaln, a, b)
  both(scipy_special_xlogy, a, x)
  both(scipy_special_kl_div, a, x)
  both(scipy_special_zeta, a + 1, x)
  both(scipy_special_gamma, x)
  both(scipy_special_log_ndtr, x)
  both(scipy_special_ndtri, x / 7)
  both(scipy_special_erfcx, x)
  both(scipy_special_i1, x)
  both(function(x) scipy_special_logsumexp(x, axis = 0L), x)
  both(scipy_special_softmax, x)
  # piecewise and vectorized kernels (while loops per element)
  both(scipy_special_spence, x)
  both(scipy_special_expn, a, x)
  both(scipy_special_hyp1f1, a, b, x)
  # tuple outputs
  got <- jit(vmap(scipy_special_sici))(x)
  want <- sp_rows(scipy_special_sici, x)
  expect_jax_equal(got[[1L]], want[[1L]])
  expect_jax_equal(got[[2L]], want[[2L]])
  # mapped and unmapped operands, batch dimension not leading
  expect_jax_equal(vmap(scipy_special_gammaincc, in_axes = list(NULL, 1L))(as_jax(2, "float32"), x),
                   jnp_transpose(scipy_special_gammaincc(2, x)))
  # vmap of a gradient through a custom JVP, and the second order rule of erfcx (scalar only)
  x1 <- as_jax(c(-2, 0.3, 5), "float32")
  expect_jax_equal(vmap(grad(scipy_special_erfcx))(x1),
                   grad(function(x) jnp_sum(scipy_special_erfcx(x)))(x1))
})

test_that("gradients of special functions agree with finite differences", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  x <- as_jax(c(0.7, 2.3), "float64")
  a <- as_jax(c(1.5, 3.1), "float64")
  fd <- check_grad_fd
  fd(function(x) jnp_sum(scipy_special_gammaln(x)), x)
  fd(function(x) jnp_sum(scipy_special_gamma(x)), x)
  fd(function(x) jnp_sum(scipy_special_betaln(a, x)), x)
  fd(function(x) jnp_sum(scipy_special_gammainc(a, x)), x)
  fd(function(x) jnp_sum(scipy_special_gammaincc(x, a)), x)
  fd(function(x) jnp_sum(scipy_special_betainc(a, a + 1, x / 3)), x)
  fd(function(x) jnp_sum(scipy_special_zeta(x + 1, a)), x)
  fd(function(x) jnp_sum(scipy_special_zeta(a, x)), x)
  fd(function(x) jnp_sum(scipy_special_erfcx(x)), x)
  fd(function(x) jnp_sum(scipy_special_erfcinv(x / 3)), x)
  fd(function(x) jnp_sum(scipy_special_ndtri(x / 3)), x)
  fd(function(x) jnp_sum(scipy_special_log_ndtr(x)), x)
  fd(function(x) jnp_sum(scipy_special_dawsn(x)), x)
  fd(function(x) jnp_sum(scipy_special_i0(x) + scipy_special_i1(x)), x)
  fd(function(x) jnp_sum(scipy_special_expi(x) + scipy_special_exp1(x)), x)
  fd(function(x) jnp_sum(scipy_special_spence(x)), x)
  fd(function(x) { r <- scipy_special_sici(x); jnp_sum(r[[1L]] + r[[2L]]) }, x)
  fd(function(x) { r <- scipy_special_fresnel(x); jnp_sum(r[[1L]] - r[[2L]]) }, x)
  fd(function(x) jnp_sum(scipy_special_owens_t(x, a)), x)
  fd(function(x) jnp_sum(scipy_special_owens_t(a, x)), x)
  fd(function(x) jnp_sum(scipy_special_hyp1f1(a, a + x, x)), x)
  fd(function(x) jnp_sum(scipy_special_hyp2f1(a, x, a + 3, x / 4)), x)
  fd(function(x) jnp_sum(scipy_special_poch(x, a)), x)
  fd(function(x) jnp_sum(scipy_special_logsumexp(x * a)), x)
  fd(function(x) jnp_sum(scipy_special_kl_div(x, a) + scipy_special_rel_entr(a, x)), x)
  # second order through nested custom rules
  fd(function(x) jnp_sum(grad(function(y) jnp_sum(scipy_special_ndtri(y / 3)))(x)), x)
  fd(function(x) jnp_sum(grad(function(y) jnp_sum(scipy_special_log_ndtr(y)))(x)), x)
  fd(function(x) jnp_sum(vmap(grad(scipy_special_erfcx))(x)), x)
})

test_that("special functions promote dtypes like JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  x <- as_jax(c(0.5, 1.5), "float32")
  i <- as_jax(c(1L, 2L))
  expect_identical(dtype(scipy_special_gammaln(i)), "float32")
  expect_identical(dtype(scipy_special_betaln(i, x)), "float32")
  expect_identical(dtype(scipy_special_expit(TRUE)), "float32")
  expect_identical(dtype(scipy_special_xlogy(2L, x)), "float32")
  expect_identical(dtype(scipy_special_gammasgn(as_jax(c(0.5, -1.5), "float16"))), "float16")
  expect_identical(dtype(scipy_special_wofz(x)), "complex64")
  expect_identical(dtype(scipy_special_logsumexp(i)), "float32")
  expect_identical(dtype(scipy_special_logsumexp(x, b = i, return_sign = TRUE)[[2L]]), "float32")
  expect_identical(dim(scipy_special_logsumexp(jnp_ones(c(2L, 3L)), axis = 1L, keepdims = TRUE)),
                   c(2L, 1L))
  # weak scalars stay weak through thin wrappers
  expect_true(jaxR:::aval_of(scipy_special_gammaln(2.5))$weak)
  expect_identical(dtype(scipy_special_polygamma(1, x)), "float32")
  expect_identical(dtype(scipy_special_polygamma(as_jax(c(1L, 2L)), x)), "float32")
  enable_x64()
  # int32 promotes to float32 even with x64 enabled (jax 0.12.0.1).
  expect_identical(dtype(scipy_special_gammaln(i)), "float32")
  expect_identical(dtype(scipy_special_gammaln(as_jax(c(1, 2), "int64"))), "float64")
  expect_identical(dtype(scipy_special_gammaln(x)), "float32")
  expect_identical(dtype(scipy_special_betaln(x, as_jax(c(0.5, 1.5), "float64"))), "float64")
  expect_identical(dtype(scipy_special_wofz(as_jax(c(0.5, 1.5), "float64"))), "complex128")
  expect_identical(dtype(scipy_special_bernoulli(4L)), "float64")
})

test_that("special functions raise the errors of JAX", {
  skip_if_no_backend()
  z <- as_jax(c(1 + 1i, 2))
  x <- as_jax(c(0.5, 1.5), "float32")
  for (f in list(scipy_special_gammasgn, scipy_special_entr, scipy_special_dawsn,
                 scipy_special_erfcinv, scipy_special_expi, scipy_special_exp1)) {
    expect_error(f(z), "does not support complex-valued inputs")
  }
  expect_error(scipy_special_beta(z, z), "beta does not support complex-valued inputs")
  expect_error(scipy_special_betainc(x, x, z), "betainc does not support complex-valued inputs")
  expect_error(scipy_special_kl_div(z, z), "kl_div does not support complex-valued inputs")
  expect_error(scipy_special_rel_entr(z, z), "rel_entr does not support complex-valued inputs")
  expect_error(scipy_special_expn(x, z), "expn does not support complex-valued inputs")
  expect_error(scipy_special_polygamma(1L, z), "polygamma does not support complex-valued inputs")
  expect_error(scipy_special_poch(z, z), "poch does not support complex-valued inputs")
  expect_error(scipy_special_hyp1f1(z, z, z), "hyp1f1 does not support complex-valued inputs")
  expect_error(scipy_special_sici(z),
               "Argument `x` to sici must be real-valued. Got dtype complex64.")
  expect_error(scipy_special_fresnel(z), "Support for complex-valued inputs is not implemented yet")
  expect_error(scipy_special_bessel_jn(z, v = 1L), "complex input not supported")
  expect_error(scipy_special_gammaln(z), "lgamma does not accept dtype complex64")
  expect_error(scipy_special_i0(z), "Unsupported input type to jax.scipy.special.i0: complex64")
  expect_error(scipy_special_i1(z), "Unsupported input type to jax.scipy.special.i1: complex64")
  expect_error(scipy_special_polygamma(1.5, x),
               "Argument `n` to polygamma must be of integer type. Got dtype float32.")
  expect_error(scipy_special_polygamma(x, x), "Argument `n` to polygamma must be of integer type")
  expect_error(scipy_special_zeta(x), "Riemann zeta function not implemented")
  expect_error(scipy_special_factorial(x, exact = TRUE), "factorial with exact=True")
  expect_error(scipy_special_log_ndtr(x, series_order = 1.5),
               "series_order must be a Python integer")
  expect_error(scipy_special_log_ndtr(x, series_order = -1L), "series_order must be non-negative")
  expect_error(scipy_special_log_ndtr(x, series_order = 31L), "series_order must be <= 30")
  expect_error(scipy_special_log_ndtr(as_jax(1:2)), "x.dtype=int32 is not supported")
  expect_error(scipy_special_ndtr(as_jax(1:2)), "x.dtype=int32 is not supported, see docstring")
  expect_error(scipy_special_ndtri(as_jax(1:2)), "x.dtype=int32 is not supported, see docstring")
  expect_error(scipy_special_spence(as_jax(1:2)), "x.dtype=int32 is not supported, see docstring")
  expect_error(scipy_special_bernoulli(-1L), "n must be a non-negative integer")
  expect_error(scipy_special_sph_harm_y(1L, 0L, x, x, diff_n = 1L), "'diff_n' argument")
  lpmn_values <- function(...) suppressWarnings(scipy_special_lpmn_values(...))
  expect_error(suppressWarnings(scipy_special_lpmn(1L, 2L, x)),
               "Computations for m!=n are not yet supported")
  expect_error(lpmn_values(2L, 2L, jnp_ones(c(2L, 2L)), FALSE), "z must be a 1D array")
  expect_error(lpmn_values(2L, 2L, as_jax(1:2), FALSE), "z.dtype=int32 is not supported")
  expect_warning(scipy_special_lpmn_values(1L, 1L, x, FALSE), "lpmn_values is deprecated")
  # static arguments must be concrete
  expect_error(jit(function(d) scipy_special_multigammaln(x, d))(3L),
               class = "jax_concretization_error")
  expect_error(jit(function(n) scipy_special_bernoulli(n))(3L), class = "jax_concretization_error")
  expect_error(jit(function(n) scipy_special_sph_harm_y(n, c(0L, 1L), x, x))(c(1L, 2L)),
               "`n_max` argument of `jax.scipy.special.sph_harm_y` must be statically specified")
  # the second order rule of erfcx selects against a scalar threshold, as in JAX
  expect_error(grad(function(x) jnp_sum(grad(function(y) jnp_sum(scipy_special_erfcx(y)))(x)))(x),
               "select cases must have the same shapes")
})
