# Special function primitives of jax.lax (jax/_src/lax/special.py, erf.py, lgamma.py).
# Expected values are from jax 0.12.0.1: float32 with x64 disabled, float64 with x64 enabled.

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

test_that("polygamma, zeta and the scaled Bessel primitives match JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(lax_polygamma(as_jax(array(c(1, 2, 0, 3)), "float32"), as_jax(array(c(0.5, 1.5,
    3.20000005, 10)), "float32")), list("float32", c(4L), c(4.93480206, -0.828796923, 0.998839378,
    0.00231990335)))
  enable_x64(TRUE)
  sp_expect(jit(function(m, x) lax_polygamma(m, x))(as_jax(array(c(1, 2, 0, 3)), "float64"),
    as_jax(array(c(0.5, 1.5, 0x1.999999999999ap+1, 10)), "float64")), list("float64", c(4L),
    c(0x1.3bd3cc9be45dfp+2, -0x1.a85808a40ab99p-1, 0x1.ff67cfa819028p-1, 0x1.3012f876b806p-9)))
  enable_x64(FALSE)
  sp_expect(lax_zeta(as_jax(array(c(2, 3.5, 1, 0.5, 4)), "float32"), as_jax(array(c(0.5, 1.5, 2, 1,
    -1.5)), "float32")), list("float32", c(5L), c(4.93480206, 0.307096153, Inf, NaN, 32.4323807)))
  enable_x64(TRUE)
  sp_expect(jit(function(s, q) lax_zeta(s, q))(as_jax(array(c(2, 3.5, 1, 0.5, 4)), "float64"),
    as_jax(array(c(0.5, 1.5, 2, 1, -1.5)), "float64")), list("float64", c(5L),
    c(0x1.3bd3cc9be45dfp+2, 0x1.3a776abc481e8p-2, Inf, NaN, 0x1.03758350a844cp+5)))
  enable_x64(FALSE)
  sp_expect(lax_bessel_i0e(as_jax(array(c(0, -0.5, 1.5, 8, 9, 50, -Inf, NaN)), "float32")),
    list("float32", c(8L), c(1.00000012, 0.645035326, 0.367433637, 0.143431798, 0.134959534,
    0.0565616265, 0, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) lax_bessel_i0e(x))(as_jax(array(c(0, -0.5, 1.5, 8, 9, 50, -Inf, NaN)),
    "float64")), list("float64", c(8L), c(1, 0x1.4a42101eb0f47p-1, 0x1.784084195afep-2,
    0x1.25bf8fe241e6ap-3, 0.134959524581723, 0x1.cf5a5415199f9p-5, 0, NaN)))
  enable_x64(FALSE)
  sp_expect(lax_bessel_i1e(as_jax(array(c(0, -0.5, 1.5, 8, 9, 50, -Inf, NaN)), "float32")),
    list("float32", c(8L), c(0, -0.156420797, 0.21903941, 0.134142622, 0.127224997, 0.0559931286,
    -0, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) lax_bessel_i1e(x))(as_jax(array(c(0, -0.5, 1.5, 8, 9, 50, -Inf, NaN)),
    "float64")), list("float64", c(8L), c(0, -0x1.40598cd0be9fp-3, 0x1.c097b8ec11eb4p-3,
    0x1.12b94cad917c4p-3, 0x1.048e8849e9048p-3, 0x1.cab2177d77b38p-5, -0, NaN)))
  enable_x64(FALSE)
  sp_expect(lax_bessel_i0e(as_jax(array(c(0.5, 9)), "float16")), list("float16", c(2L),
    c(0.645019531, 0.135009766)), tol = 2e-3)
})

test_that("incomplete gamma primitives match JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(lax_igamma(as_jax(array(c(2, 0.5, 3, 10, 0, 1, 1, Inf, NaN, -1, 1)), "float32"),
    as_jax(array(c(0.5, 1.5, 3.20000005, 30, 1, 0, Inf, 1, 1, 1, -1)), "float32")), list("float32",
    c(11L), c(0.0902040079, 0.91673547, 0.620096385, 0.999992907, 1, 0, 1, 0, NaN, NaN, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(a, x) lax_igamma(a, x))(as_jax(array(c(2, 0.5, 3, 10, 0, 1, 1, Inf, NaN,
    -1, 1)), "float64"), as_jax(array(c(0.5, 1.5, 0x1.999999999999ap+1, 30, 1, 0, Inf, 1, 1, 1,
    -1)), "float64")), list("float64", c(11L), c(0x1.7179c2ac4f788p-4, 0x1.d55e5a70068e3p-1,
    0x1.3d7d41c0e1738p-1, 0x1.ffff1108a03b8p-1, 1, 0, 1, 0, NaN, NaN, NaN)))
  enable_x64(FALSE)
  sp_expect(lax_igammac(as_jax(array(c(2, 0.5, 3, 10, 0, 1, 1, Inf, NaN, -1, 1)), "float32"),
    as_jax(array(c(0.5, 1.5, 3.20000005, 30, 1, 0, Inf, 1, 1, 1, -1)), "float32")), list("float32",
    c(11L), c(0.909796, 0.0832645446, 0.379903615, 7.12175506e-06, 0, 1, 0, 1, NaN, NaN, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(a, x) lax_igammac(a, x))(as_jax(array(c(2, 0.5, 3, 10, 0, 1, 1, Inf, NaN,
    -1, 1)), "float64"), as_jax(array(c(0.5, 1.5, 0x1.999999999999ap+1, 30, 1, 0, Inf, 1, 1, 1,
    -1)), "float64")), list("float64", c(11L), c(0x1.d1d0c7aa7610fp-1, 0x1.550d2c7fcb8e6p-4,
    0x1.85057c7e3d191p-2, 0x1.ddeebf89001aep-18, 0, 1, 0, 1, NaN, NaN, NaN)))
  enable_x64(FALSE)
  sp_expect(lax_igamma_grad_a(as_jax(array(c(2, 0.5, 3, 10, 1, 0, NaN)), "float32"),
    as_jax(array(c(0.5, 1.5, 3.20000005, 30, 0, 1, 1)), "float32")), list("float32", c(7L),
    c(-0.151034236, -0.230194405, -0.227811873, -8.50020751e-06, 0, NaN, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(a, x) lax_igamma_grad_a(a, x))(as_jax(array(c(2, 0.5, 3, 10, 1, 0, NaN)),
    "float64"), as_jax(array(c(0.5, 1.5, 0x1.999999999999ap+1, 30, 0, 1, 1)), "float64")),
    list("float64", c(7L), c(-0x1.35517020cf43bp-3, -0x1.d770219582aa8p-3, -0x1.d28f120ba0e2p-3,
    -0x1.1d38300ccc0a3p-17, 0, NaN, NaN)))
  enable_x64(FALSE)
  sp_expect(lax_random_gamma_grad(as_jax(array(c(2, 0.5, 3, 10, 1, 0, NaN)), "float32"),
    as_jax(array(c(0.5, 1.5, 3.20000005, 30, 0, 1, 1)), "float32")), list("float32", c(7L),
    c(0.498026699, 2.23953056, 1.09156275, 1.67469382, 0, NaN, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(a, x) lax_random_gamma_grad(a, x))(as_jax(array(c(2, 0.5, 3, 10, 1, 0,
    NaN)), "float64"), as_jax(array(c(0.5, 1.5, 0x1.999999999999ap+1, 30, 0, 1, 1)), "float64")),
    list("float64", c(7L), c(0x1.fdfab7493e463p-2, 0x1.1ea8ed65b31b1p+1, 0x1.1770af459d4b7p+0,
    0x1.acb8c0ec233fdp+0, 0, NaN, NaN)))
  enable_x64(FALSE)
  sp_expect(lax_igamma(as_jax(array(c(0.5, 2)), "float16"), as_jax(array(c(0.5, 9)), "float16")),
    list("float16", c(2L), c(0.682617188, 0.998535156)), tol = 2e-3)
})

test_that("betainc matches JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(lax_betainc(as_jax(array(c(2, 0.5, 3, 10, 1, 0, 1, 2, NaN, 0x1.16c1ffff7a88bp-133)),
    "float32"), as_jax(array(c(3, 0.5, 1.5, 2, 1, 1, 0, 2, 1, 2)), "float32"),
    as_jax(array(c(0.200000003, 0.699999988, 0.99000001, 0.5, 0, 0.300000012, 0.300000012, 1.5, 0.5,
    0.5)), "float32")), list("float32", c(10L), c(0.180800006, 0.630989909, 0.995677292,
    0.005859375, 0, 1, 0, NaN, NaN, 1)))
  enable_x64(TRUE)
  sp_expect(jit(function(a, b, x) lax_betainc(a, b, x))(as_jax(array(c(2, 0.5, 3, 10, 1, 0, 1, 2,
    NaN, 0x1.16c262777579cp-133)), "float64"), as_jax(array(c(3, 0.5, 1.5, 2, 1, 1, 0, 2, 1, 2)),
    "float64"), as_jax(array(c(0x1.999999999999ap-3, 0x1.6666666666666p-1, 0x1.fae147ae147aep-1,
    0.5, 0, 0x1.3333333333333p-2, 0x1.3333333333333p-2, 1.5, 0.5, 0.5)), "float64")),
    list("float64", c(10L), c(0x1.72474538ef356p-3, 0x1.43111b092557dp-1, 0x1.fdc96aad1d042p-1,
    0x1.7fffffffffff6p-8, 0, 1, 0, NaN, NaN, 1)))
})

test_that("lgamma lowers as in JAX (float32 implementation, CHLO for float64)", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(lax_lgamma(as_jax(array(c(0.5, 1, 2, 3.20000005, 4, 7.5, 100, 1e+10,
    0x1.e17b83fceae89p+122, 0x1.e17b8404fedc3p+121, 0.300000012, 0.00100000005,
    0x1.16c1ffff7a88bp-133, 0, -0, -0.5, -1.5, -2.45700002, -2.74760008, -2.5, -3.1400001, -5.5,
    -7.19999981, -9.5, -12.3000002, -100.5, -1, -2, -100000000, -1.00000002e+30, Inf, -Inf, NaN)),
    "float32")), list("float32", c(33L), c(0.572364926, 0, 0, 0.885404885, 1.79175949, 7.53436422,
    359.134216, 2.20258517e+11, Inf, Inf, 1.09579802, 6.90717888, 92.1034088, Inf, Inf, 1.26551211,
    0.860046983, 3.74702868e-05, -0.000157999079, -0.0562437214, 0.0282017086, -4.51783228,
    -7.25480413, -12.7958946, -19.3919182, -364.90097, Inf, Inf, Inf, Inf, Inf, Inf, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) lax_lgamma(x))(as_jax(array(c(0.5, 1, 2, 0x1.999999999999ap+1, 4, 7.5,
    100, 10000000000, 0x1.e17b84357691bp+122, 0x1.e17b84357691bp+121, 0x1.3333333333333p-2, 0.001,
    0x1.16c262777579cp-133, 0, -0, -0.5, -1.5, -0x1.3a7ef9db22d0ep+1, -0x1.5fb15b573eab3p+1, -2.5,
    -0x1.91eb851eb851fp+1, -5.5, -0x1.ccccccccccccdp+2, -9.5, -0x1.899999999999ap+3, -100.5, -1, -2,
    -100000000, -0x1.93e5939a08ceap+99, Inf, -Inf, NaN)), "float64")), list("float64", c(33L),
    c(0x1.250d048e7a1cp-1, 0, 0, 0x1.c553c810b3a58p-1, 0x1.cab0bfa2a2005p+0, 0x1.e233060e41f8p+2,
    0x1.67225b4879462p+8, 0x1.9a43710f467c1p+37, 0x1.3cb575d392b59p+129, 0x1.3a19fbda96a4ap+128,
    0x1.188637a6c4192p+0, 0x1.ba0f3807161abp+2, 0x1.7069e2aa2aa5bp+6, Inf, Inf,
    0x1.43f89a3f0edd5p+0, 0x1.b858151820f82p-1, 0x1.3a8ab4338p-15, -0x1.4ba9ce1fdcp-13,
    -0x1.ccbf9f5ed0f6p-5, 0x1.ce120a0162c4p-6, -0x1.2124298f0b82ap+2, -0x1.d04ebc2b29ea8p+2,
    -0x1.9977f97d94f5ep+3, -0x1.36454b35882f7p+4, -0x1.6ce6a5dbefb8fp+8, Inf, Inf, Inf, Inf, Inf,
    Inf, NaN)))
  enable_x64(FALSE)
  sp_expect(lax_lgamma(as_jax(array(c(0.5, 9)), "float16")), list("float16", c(2L), c(0.572265625,
    10.6015625)), tol = 2e-3)
  sp_expect(lax_digamma(as_jax(array(c(0.5, 3.20000005, -1.5, 100, 0.00100000005)), "float32")),
    list("float32", c(5L), c(-1.96351004, 0.998839378, 0.70315671, 4.60016203, -1000.5755)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) lax_digamma(x))(as_jax(array(c(0.5, 0x1.999999999999ap+1, -1.5, 100,
    0.001)), "float64")), list("float64", c(5L), c(-0x1.f6a897d321505p+0, 0x1.ff67cfa819028p-1,
    0x1.680425af12b65p-1, 0x1.26690d4274475p+2, -0x1.f449ac574fcf8p+9)))
})

test_that("erfc with highest accuracy, exp_neg_sq and the accurate erf_inv match JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(jaxR:::bind(jaxR:::erfc_p, list(as_jax(array(c(-30, -6, -1.5, -0.300000012, 0,
    0.300000012, 0.899999976, 1, 1.5, 3, 9, 12, 30, Inf, -Inf, NaN)), "float32")), list(accuracy =
    "highest")), list("float32", c(16L), c(2, 2, 1.9661051, 1.32862675, 1, 0.671373248, 0.2030918,
    0.157299206, 0.0338948518, 2.20904967e-05, 0x1.198d420232f92p-121, 0, 0, 0, 2, NaN)))
  enable_x64(TRUE)
  sp_expect(jaxR:::bind(jaxR:::erfc_p, list(as_jax(array(c(-30, -6, -1.5, -0x1.3333333333333p-2, 0,
    0x1.3333333333333p-2, 0x1.ccccccccccccdp-1, 1, 1.5, 3, 9, 12, 30, Inf, -Inf, NaN)), "float64")),
    list(accuracy = "highest")), list("float64", c(16L), c(2, 2, 0x1.f752aab89bd7p+0,
    0x1.5420e22077a84p+0, 1, 0x1.57be3bbf10af8p-1, 0x1.9fee964dd7469p-3, 0x1.4226162fbddd4p-3,
    0x1.15aaa8ec85204p-5, 0x1.729df6503422bp-16, 0x1.198d422be3f8dp-121, 0x1.c90f21d2d475fp-213, 0,
    0, 2, NaN)))
  enable_x64(FALSE)
  sp_expect(jaxR:::sp_exp_neg_sq(as_jax(array(c(-30, -6, -1.5, 0, 0.300000012, 3, 10.5, 27, Inf,
    NaN)), "float32")), list("float32", c(10L), c(0, 0x1.0b6c3a01f0885p-52, 0.105399221, 1,
    0.913931191, 0.000123409802, 0, 0, 0, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) jaxR:::sp_exp_neg_sq(x))(as_jax(array(c(-30, -6, -1.5, 0,
    0x1.3333333333333p-2, 3, 10.5, 27, Inf, NaN)), "float64")), list("float64", c(10L), c(0,
    0x1.0b6c3afdde064p-52, 0x1.afb718e8457f7p-4, 1, 0x1.d3eec9cf11a26p-1, 0x1.02cf22526545ap-13,
    0x1.ec1f31193575dp-160, 0, 0, NaN)))
  enable_x64(FALSE)
  sp_expect(jaxR:::sp_erf_inv_impl(as_jax(array(c(0, 0.300000012, -0.899999976, 0.999000013,
    0.999999881, 1, -1, 1.5, NaN)), "float32")), list("float32", c(9L), c(0, 0.272462726,
    -1.16308713, 2.32675648, 3.74392128, Inf, -Inf, NaN, NaN)))
  enable_x64(TRUE)
  sp_expect(jaxR:::sp_erf_inv_impl(as_jax(array(c(0, 0x1.3333333333333p-2, -0x1.ccccccccccccdp-1,
    0.999, 0x1.fffffca501acbp-1, 1, -1, 1.5, NaN)), "float64")), list("float64", c(9L), c(0,
    0x1.1700774485ed3p-2, -0x1.29c014677065p+0, 0x1.29d311405cdb4p+1, 0x1.e21eb9013cbe8p+1, Inf,
    -Inf, NaN, NaN)))
})

test_that("the polynomial primitive and polyval match JAX", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(FALSE)
  sp_expect(jaxR:::sp_polynomial(as_jax(array(c(0.5, -1.5, 2)), "float32"), list(1, -2, 0.5, 3)),
    list("float32", c(3L), c(0.5, -5, 23)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) jaxR:::sp_polynomial(x, list(1, -2, 0.5, 3)))(as_jax(array(c(0.5, -1.5,
    2)), "float64")), list("float64", c(3L), c(0.5, -5, 23)))
  enable_x64(FALSE)
  sp_expect(jaxR:::sp_polynomial(as_jax(array(c(0.5, -1.5, 2, Inf)), "float32"), as.list(1:11 / 7)),
    list("float32", c(4L), c(0.569614947, 56.3698196, 2925.85693, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) jaxR:::sp_polynomial(x, as.list(1:11 / 7)))(as_jax(array(c(0.5, -1.5, 2,
    Inf)), "float64")), list("float64", c(4L), c(0x1.23a4924924924p-1, 0x1.c2f56db6db6dcp+5,
    0x1.6dbb6db6db6dcp+11, NaN)))
  enable_x64(FALSE)
  sp_expect(jaxR:::sp_polynomial(as_jax(array(c(0.5, -1.5, 2)), "float32"), list(as_jax(array(c(1,
    2, 3)), "float32"), as_jax(array(c(0.100000001, 0.200000003, 0.300000012)), "float32"), 2)),
    list("float32", c(3L), c(1.54999995, 6.19999981, 11.6000004)))
  enable_x64(TRUE)
  sp_expect(jit(function(x, c0, c1) jaxR:::sp_polynomial(x, list(c0, c1, 2)))(as_jax(array(c(0.5,
    -1.5, 2)), "float64"), as_jax(array(c(1, 2, 3)), "float64"),
    as_jax(array(c(0x1.999999999999ap-4, 0x1.999999999999ap-3, 0x1.3333333333333p-2)), "float64")),
    list("float64", c(3L), c(1.55, 0x1.8ccccccccccccp+2, 11.6)))
  enable_x64(FALSE)
  sp_expect(jaxR:::sp_polyval(c(3, 0.5, -2, 1), as_jax(array(c(0.5, -1.5, 2)), "float32")),
    list("float32", c(3L), c(0.5, -5, 23)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) jaxR:::sp_polyval(c(3, 0.5, -2, 1), x))(as_jax(array(c(0.5, -1.5, 2)),
    "float64")), list("float64", c(3L), c(0.5, -5, 23)))
  enable_x64(FALSE)
  sp_expect(jaxR:::sp_sinc(as_jax(array(c(0, 0.300000012, -0.5, 1, 2.5, -7.25, 1e+10, Inf, NaN)),
    "float32")), list("float32", c(9L), c(1, 0.858393669, 0.636619806, 0, 0.127323955,
    -0.0310453903, 0, 0, NaN)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) jaxR:::sp_sinc(x))(as_jax(array(c(0, 0x1.3333333333333p-2, -0.5, 1, 2.5,
    -7.25, 10000000000, Inf, NaN)), "float64")), list("float64", c(9L), c(1, 0x1.b77f60bebee61p-1,
    0x1.45f306dc9c882p-1, 0, 0x1.04c26be3b06cep-3, -0x1.fca5cde9d213fp-6, 0, 0, NaN)))
  sp_expect(grad(function(x) jnp_sum(jaxR:::sp_polynomial(x, list(1, -2, 0.5,
    3))))(as_jax(array(c(0.5, -1.5, 2)), "float64")), list("float64", c(3L), c(0.75, 16.75, 36)))
  sp_expect(grad(function(c0, c1, x) jnp_sum(jaxR:::sp_polynomial(x, list(c0, c1, 2))), argnums =
    c(0L, 1L))(as_jax(array(c(1, 2, 3)), "float64"), as_jax(array(c(0x1.999999999999ap-4,
    0x1.999999999999ap-3, 0x1.3333333333333p-2)), "float64"), as_jax(array(c(0.5, -1.5, 2)),
    "float64")), list("float64", c(3L), c(1, 1, 1)), list("float64", c(3L), c(0.5, -1.5, 2)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(jaxR:::sp_sinc(x)))(x))(as_jax(array(c(0,
    0x1.3333333333333p-2, 1, 2.5)), "float64")), list("float64", c(4L), c(0, -0x1.cdd6a18e0980ap-1,
    -1, -0x1.a1371305e7152p-5)))
  sp_expect(jit(function(x) grad(function(x)
    jnp_sum(jaxR:::sp_exp_neg_sq(x)))(x))(as_jax(array(c(-1.5, 0x1.3333333333333p-2, 30, Inf)),
    "float64")), list("float64", c(4L), c(0x1.43c952ae341f9p-2, -0x1.18c27915d7617p-1, -0, -0)))
})

test_that("gradients of the special primitives match jax.grad", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  enable_x64(TRUE)
  sp_expect(jit(function(x) grad(function(x) jnp_sum(lax_lgamma(x)))(x))(as_jax(array(c(0.5,
    0x1.999999999999ap+1, -1.5)), "float64")), list("float64", c(3L), c(-0x1.f6a897d321505p+0,
    0x1.ff67cfa819028p-1, 0x1.680425af12b65p-1)))
  enable_x64(FALSE)
  sp_expect(jit(function(x) grad(function(x) jnp_sum(lax_lgamma(x)))(x))(as_jax(array(c(0.5,
    3.20000005, -1.5)), "float32")), list("float32", c(3L), c(-1.96351004, 0.998839378,
    0.70315671)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) grad(function(x) jnp_sum(lax_digamma(x)))(x))(as_jax(array(c(0.5,
    0x1.999999999999ap+1, -1.5)), "float64")), list("float64", c(3L), c(0x1.3bd3cc9be45dfp+2,
    0x1.771ce6f817b35p-2, 0x1.2c22c9dc2b127p+3)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(lax_erf(x)))(x))(as_jax(array(c(0.5, -1.5, 3,
    30, Inf)), "float64")), list("float64", c(5L), c(0x1.c1efca49a5011p-1, 0x1.e723726b824a8p-4,
    0x1.2408e9ba3327fp-13, 0, 0)))
  enable_x64(FALSE)
  sp_expect(grad(function(x) jnp_sum(lax_erf(x)))(as_jax(array(c(0.5, -1.5, 3, 30, Inf)),
    "float32")), list("float32", c(5L), c(0.87878263, 0.118930295, 0.000139253054, 0, 0)))
  enable_x64(TRUE)
  sp_expect(jit(function(x) grad(function(x) jnp_sum(lax_erfc(x)))(x))(as_jax(array(c(0.5, -1.5, 3,
    30)), "float64")), list("float64", c(4L), c(-0x1.c1efca49a5011p-1, -0x1.e723726b824a8p-4,
    -0x1.2408e9ba3327fp-13, -0)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(lax_erf_inv(x)))(x))(as_jax(array(c(0.5,
    -0x1.ccccccccccccdp-1, 0x1.999999999999ap-4)), "float64")), list("float64", c(3L),
    c(0x1.1cd25bd382b6fp+0, 0x1.b6ca1b5a3c9d2p+1, 0x1.c9584a46900ddp-1)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(jnp_sum(grad(function(y)
    jnp_sum(lax_erf(y)))(x))))(x))(as_jax(array(c(0.5, -1.5, 3)), "float64")), list("float64",
    c(3L), c(-0x1.c1efca49a5011p-1, 0x1.6d5a95d0a1b7ep-2, -0x1.b60d5e974cbbfp-11)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(lax_polygamma(lax_full_like(x, 2),
    x)))(x))(as_jax(array(c(0.5, 0x1.999999999999ap+1)), "float64")), list("float64", c(2L),
    c(97.4090910340025, 0x1.8689d7a3518f8p-4)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(lax_igamma(a, b)), argnums = c(0L,
    1L))(a, b))(as_jax(array(c(2, 0.5, 3)), "float64"), as_jax(array(c(0.5, 1.5,
    0x1.999999999999ap+1)), "float64")), list("float64", c(3L), c(-0x1.35517020cf43bp-3,
    -0x1.d770219582ab1p-3, -0x1.d28f120ba0e31p-3)), list("float64", c(3L), c(0x1.368b2fc6f960ap-2,
    0x1.a503dcc210e22p-4, 0x1.ab6c3547adb45p-3)))
  sp_expect(jit(function(a, b) grad(function(a, b) jnp_sum(lax_igammac(a, b)), argnums = c(0L,
    1L))(a, b))(as_jax(array(c(2, 0.5, 3)), "float64"), as_jax(array(c(0.5, 1.5,
    0x1.999999999999ap+1)), "float64")), list("float64", c(3L), c(0x1.35517020cf43bp-3,
    0x1.d770219582ab1p-3, 0x1.d28f120ba0e31p-3)), list("float64", c(3L), c(-0x1.368b2fc6f960ap-2,
    -0x1.a503dcc210e22p-4, -0x1.ab6c3547adb45p-3)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(lax_betainc(as_jax(c(2, 0.5), "float64"),
    as_jax(c(3, 1.5), "float64"), x)))(x))(as_jax(array(c(0x1.999999999999ap-3,
    0x1.6666666666666p-1)), "float64")), list("float64", c(2L), c(0x1.89374bc6a7f06p+0,
    0x1.aac4914d31a57p-2)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(scipy_special_i0e(x)))(x))(as_jax(array(c(0.5,
    -1.5)), "float64")), list("float64", c(2L), c(-0x1.f45759d502996p-2, 0x1.2fe94f46a410cp-3)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(lax_bessel_i1e(x)))(x))(as_jax(array(c(0.5,
    -1.5, 9, 0, 0x1.79ca10c924223p-67)), "float64")), list("float64", c(5L), c(0x1.67fb9a0887f4cp-3,
    0x1.365faa50b258p-9, -0x1.a3888020152ap-8, 0.5, 0.5)))
  sp_expect(jit(function(x) grad(function(x) jnp_sum(jnp_sum(grad(function(y)
    jnp_sum(lax_bessel_i0e(y)))(x))))(x))(as_jax(array(c(0.5, -1.5)), "float64")), list("float64",
    c(2L), c(0x1.542a936ca349ep-1, 0x1.34c2cdefe6da2p-3)))
})

test_that("special primitives lower to the operations JAX emits", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  x <- as_jax(c(0.5, 1.5), "float32")
  expect_match(lower_text(function(x) lax_polygamma(x, x), x), "chlo.polygamma", fixed = TRUE)
  expect_match(lower_text(function(x) lax_zeta(x, x), x), "chlo.zeta", fixed = TRUE)
  expect_match(lower_text(lax_bessel_i1e, x), "chlo.bessel_i1e", fixed = TRUE)
  expect_match(lower_text(lax_igamma, x, x), "stablehlo.while", fixed = TRUE)
  expect_match(lower_text(lax_betainc, x, x, x), "stablehlo.while", fixed = TRUE)
  expect_no_match(lower_text(lax_bessel_i0e, x), "chlo.", fixed = TRUE)
  # float32 lgamma is the polynomial implementation of jax/_src/lax/lgamma.py
  expect_no_match(lower_text(lax_lgamma, x), "chlo.lgamma", fixed = TRUE)
  # erf.py:_erfcx_from_grad_large keeps the float32 split constant behind a barrier
  expect_match(lower_text(scipy_special_erfcx, x), "stablehlo.optimization_barrier", fixed = TRUE)
  enable_x64()
  expect_match(lower_text(lax_lgamma, as_jax(c(0.5, 1.5), "float64")), "chlo.lgamma", fixed = TRUE)
})

test_that("special primitives vmap like stacked loops", {
  skip_if_no_backend()
  a <- as_jax(matrix(c(2, 0.5, 3, 1.5, 4, 0.7), 2, byrow = TRUE), "float32")
  b <- as_jax(matrix(c(3, 0.5, 1.5, 2, 1, 2.5), 2, byrow = TRUE), "float32")
  x <- as_jax(matrix(c(0.5, 1.5, 3.2, 2.5, 0.3, 6), 2, byrow = TRUE), "float32")
  for (f in list(lax_igamma, lax_igammac, lax_igamma_grad_a, lax_random_gamma_grad,
                 function(a, x) lax_polygamma(lax_floor(a), x),
                 function(a, x) lax_zeta(a + 1, x))) {
    expect_jax_equal(jit(vmap(f))(a, x), sp_rows(f, a, x))
  }
  expect_jax_equal(jit(vmap(lax_betainc))(a, b, x / 7), sp_rows(lax_betainc, a, b, x / 7))
  for (f in list(lax_bessel_i0e, lax_bessel_i1e, lax_lgamma)) {
    expect_jax_equal(jit(vmap(f))(x), sp_rows(f, x))
  }
  # one unmapped operand and a batch dimension that is not leading
  expect_jax_equal(vmap(lax_igamma, in_axes = list(NULL, 1L))(as_jax(2, "float32"), x),
                   jnp_transpose(lax_igamma(lax_full_like(x, 2), x)))
  expect_jax_equal(vmap(lax_betainc, in_axes = list(1L, NULL, 0L), out_axes = 1L)(
    jnp_transpose(a), as_jax(1.5, "float32"), x / 7),
    jnp_transpose(lax_betainc(a, lax_full_like(a, 1.5), x / 7)))
  # gradients under vmap
  a1 <- as_jax(c(2, 0.5), "float32")
  x1 <- as_jax(c(0.5, 1.5), "float32")
  expect_jax_equal(vmap(grad(lax_igamma, argnums = 1L))(a1, x1),
                   grad(function(x) jnp_sum(lax_igamma(a1, x)))(x1))
  # the polynomial primitive broadcasts over mapped coefficients
  p <- function(x, c0) jaxR:::sp_polynomial(x, list(c0, 2, 0.5))
  expect_jax_equal(vmap(p)(x, a), sp_rows(p, x, a))
})

test_that("gradients of special primitives agree with finite differences", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  x <- as_jax(c(0.7, 2.3), "float64")
  a <- as_jax(c(1.5, 3.1), "float64")
  check_grad_fd(function(x) jnp_sum(lax_igamma(a, x)), x)
  check_grad_fd(function(a) jnp_sum(lax_igamma(a, x)), a)
  check_grad_fd(function(x) jnp_sum(lax_igammac(a, x)), x)
  check_grad_fd(function(a) jnp_sum(lax_igammac(a, x)), a)
  check_grad_fd(function(x) jnp_sum(lax_betainc(a, a + 1, x / 3)), x)
  check_grad_fd(function(x) jnp_sum(lax_polygamma(lax_full_like(x, 1), x)), x)
  check_grad_fd(function(x) jnp_sum(lax_bessel_i0e(x)), x)
  check_grad_fd(function(x) jnp_sum(lax_bessel_i1e(x)), x)
  check_grad_fd(function(x) jnp_sum(lax_lgamma(x)), x)
  check_grad_fd(function(x) jnp_sum(lax_erfc(x)), x)
  # second order: the JVP of erf goes through the custom rule of exp_neg_sq
  check_grad_fd(function(x) jnp_sum(grad(function(y) jnp_sum(lax_erf(y)))(x)), x)
  check_grad_fd(function(x) jnp_sum(grad(function(y) jnp_sum(lax_bessel_i1e(y)))(x)), x)
  # forward mode through the while loops of igamma
  jv <- jvp(function(x) lax_igamma(a, x), list(x), list(jnp_ones_like(x)))
  expect_jax_equal(jv[[2L]], grad(function(x) jnp_sum(lax_igamma(a, x)))(x))
})

test_that("special primitives check dtypes and report unsupported derivatives", {
  skip_if_no_backend()
  on.exit(enable_x64(FALSE))
  x <- as_jax(c(0.5, 1.5), "float32")
  expect_error(lax_igamma(as_jax(1:2), as_jax(1:2)), "igamma does not accept dtype int32")
  expect_error(lax_bessel_i0e(as_jax(1:2)), "bessel_i0e does not accept dtype int32")
  z <- as_jax(c(1 + 1i, 2))
  expect_error(lax_polygamma(z, z), "does not accept dtype complex64")
  expect_error(lax_zeta(x, as_jax(c(1, 2, 3), "float32")), "incompatible shapes")
  # R scalars adopt the dtype of the array argument
  expect_identical(dtype(lax_igamma(2, x)), "float32")
  expect_identical(dtype(lax_betainc(2, 3, x / 2)), "float32")
  expect_error(grad(function(a) jnp_sum(lax_betainc(a, x, x / 2)))(x),
               "Betainc gradient with respect to a and b not supported")
  expect_error(grad(function(m) jnp_sum(lax_polygamma(m, x)))(x),
               "polygamma gradient with respect to m is not supported")
  expect_error(grad(function(q) jnp_sum(lax_zeta(x + 1, q)))(x), "Differentiation rule for 'zeta'")
  expect_error(grad(function(a) jnp_sum(grad(function(b) jnp_sum(lax_igamma(b, x)))(a)))(x),
               "Differentiation rule for 'igamma_grad_a'")
  enable_x64()
  expect_error(lax_igamma(x, as_jax(c(0.5, 1.5), "float64")),
               "requires arguments to have the same dtypes")
  expect_identical(dtype(lax_random_gamma_grad(as_jax(c(0.5, 1.5), "float64"), 2)), "float64")
})
