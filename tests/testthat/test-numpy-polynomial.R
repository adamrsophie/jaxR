# jax.numpy polynomial functions (R/numpy_polynomial.R). Expected values from jax
# 0.12.0.1 (float64 as hex literals); each entry is list(value, dtype).

pol_exp <- list(
  polyval_f32 = list(array(c(0.5, 4.5, 1, 0.8359375, -14.3125, 20), dim = c(2, 3)), "float32"),
  polyval_f64 = list(array(c(0x1.0000000000000p-1, 0x1.2000000000000p+2, 1, 0x1.ac00000000000p-1, -0x1.ca00000000000p+3, 20), dim = c(2, 3)), "float64"),
  polyval_long_f32 = list(array(c(1.4285715, 290.85718, 7.8571424, 1.84127, -1.3713726, 6325.857), dim = c(2, 3)), "float32"),
  polyval_long_f64 = list(array(c(0x1.6db6db6db6db7p+0, 0x1.22db6db6db6dbp+8, 0x1.f6db6db6db6dbp+2, 0x1.d75d76db6db6ep+0, -0x1.5f12492492492p+0, 0x1.8b5db6db6db6ep+12), dim = c(2, 3)), "float64"),
  polyval_unroll_f32 = list(array(c(1.4285715, 290.85718, 7.8571424, 1.84127, -1.3713726, 6325.857), dim = c(2, 3)), "float32"),
  polyval_unroll_f64 = list(array(c(0x1.6db6db6db6db7p+0, 0x1.22db6db6db6dbp+8, 0x1.f6db6db6db6dbp+2, 0x1.d75d76db6db6ep+0, -0x1.5f12492492492p+0, 0x1.8b5db6db6db6ep+12), dim = c(2, 3)), "float64"),
  polyval_2d_f32 = list(c(6.75, 6), "float32"),
  polyval_2d_f64 = list(c(0x1.b000000000000p+2, 6), "float64"),
  polyval_1_f32 = list(array(c(3, 3, 3, 3, 3, 3), dim = c(2, 3)), "float32"),
  polyval_1_f64 = list(array(c(3, 3, 3, 3, 3, 3), dim = c(2, 3)), "float64"),
  polyadd_f32 = list(c(1.5, -3, 4, -0.5), "float32"),
  polyadd_f64 = list(c(0x1.8000000000000p+0, -3, 4, -0x1.0000000000000p-1), "float64"),
  polyadd_r_f32 = list(c(1.5, -3, 4, -0.5), "float32"),
  polyadd_r_f64 = list(c(0x1.8000000000000p+0, -3, 4, -0x1.0000000000000p-1), "float64"),
  polysub_f32 = list(c(-1.5, 3, 0, -1.5), "float32"),
  polysub_f64 = list(c(-0x1.8000000000000p+0, 3, 0, -0x1.8000000000000p+0), "float64"),
  polymul_f32 = list(c(3, -7.5, 7, -1, -0.5), "float32"),
  polymul_f64 = list(c(3, -0x1.e000000000000p+2, 7, -1, -0x1.0000000000000p-1), "float64"),
  polymul_trim_f32 = list(c(3, 6), "float32"),
  polymul_trim_f64 = list(c(3, 6), "float64"),
  polydiv_f321 = list(c(0.75, -1.125, 0.4375), "float32"),
  polydiv_f322 = list(c(0, 0, 0, 0.9375), "float32"),
  polydiv_f641 = list(c(0x1.8000000000000p-1, -0x1.2000000000000p+0, 0x1.c000000000000p-2), "float64"),
  polydiv_f642 = list(c(0, 0, 0, 0x1.e000000000000p-1), "float64"),
  polydiv_trim_f321 = list(c(1, -2), "float32"),
  polydiv_trim_f322 = list(numeric(0), "float32"),
  polydiv_trim_f641 = list(c(1, -2), "float64"),
  polydiv_trim_f642 = list(numeric(0), "float64"),
  polydiv_short_f321 = list(c(0), "float32"),
  polydiv_short_f322 = list(c(2, -1), "float32"),
  polydiv_short_f641 = list(c(0), "float64"),
  polydiv_short_f642 = list(c(2, -1), "float64"),
  polyder_f32 = list(c(4.5, -6, 2), "float32"),
  polyder_f64 = list(c(0x1.2000000000000p+2, -6, 2), "float64"),
  polyder2_f32 = list(c(9, -6), "float32"),
  polyder2_f64 = list(c(9, -6), "float64"),
  polyder0_f32 = list(c(1.5, -3, 2, 0.5), "float32"),
  polyder0_f64 = list(c(0x1.8000000000000p+0, -3, 2, 0x1.0000000000000p-1), "float64"),
  polyder_big_f32 = list(numeric(0), "float32"),
  polyder_big_f64 = list(numeric(0), "float64"),
  polyint_f32 = list(c(0.375, -1, 1, 0.5, 0), "float32"),
  polyint_f64 = list(c(0x1.8000000000000p-2, -1, 1, 0x1.0000000000000p-1, 0), "float64"),
  polyint2k_f32 = list(c(0.075, -0.25, 0.33333334, 0.25, 1, -2), "float32"),
  polyint2k_f64 = list(c(0x1.3333333333333p-4, -0x1.0000000000000p-2, 0x1.5555555555555p-2, 0x1.0000000000000p-2, 1, -2), "float64"),
  polyint3k_f32 = list(c(0.0125, -0.05, 0.083333336, 0.083333336, 2.5, 5, 5), "float32"),
  polyint3k_f64 = list(c(0x1.999999999999ap-7, -0x1.999999999999ap-5, 0x1.5555555555555p-4, 0x1.5555555555555p-4, 0x1.4000000000000p+1, 5, 5), "float64"),
  poly_f32 = list(c(1, 0.5, -2.5, 1), "float32"),
  poly_f64 = list(c(1, 0x1.0000000000000p-1, -0x1.4000000000000p+1, 1), "float64"),
  poly_empty_f32 = list(1, "float32"),
  poly_empty_f64 = list(1, "float64"),
  polyfit_f32 = list(c(-2.8285718, 1.7985723, 1.0416665), "float32"),
  polyfit_f64 = list(c(-0x1.6a0ea0ea0ea11p+1, 0x1.cc6f2d593bfb5p+0, 0x1.0aaaaaaaaaaa2p+0), "float64"),
  polyfit_w_f32 = list(c(-0.11239586, -2.7096634, 1.7897054, 1.0355884), "float32"),
  polyfit_w_f64 = list(c(-0x1.cc5e9f9d29923p-4, -0x1.5ad63b3c6eb5bp+1, 0x1.ca2a0de6e5853p+0, 0x1.091c53d632e89p+0), "float64"),
  polyfit_2d_f32 = list(array(c(-1.0299999, 1.4345237, 2.0600002, 0.8090477), dim = c(2, 2)), "float32"),
  polyfit_2d_f64 = list(array(c(-0x1.07ae147ae1477p+0, 0x1.6f3cf3cf3cf3bp+0, 0x1.07ae147ae147ep+1, 0x1.9e3b7d516eb08p-1), dim = c(2, 2)), "float64"),
  polyfit_full_f321 = list(c(-2.8285718, 1.7985723, 1.0416665), "float32"),
  polyfit_full_f322 = list(c(0.022738134), "float32"),
  polyfit_full_f323 = list(3, "int32"),
  polyfit_full_f324 = list(c(1.6388184, 0.5478889, 0.118709095), "float32"),
  polyfit_full_f325 = list(8.34465e-07, "float32"),
  polyfit_full_f641 = list(c(-0x1.6a0ea0ea0ea11p+1, 0x1.cc6f2d593bfb5p+0, 0x1.0aaaaaaaaaaa2p+0), "float64"),
  polyfit_full_f642 = list(c(0x1.748a7bdaf0e2fp-6), "float64"),
  polyfit_full_f643 = list(3, "int64"),
  polyfit_full_f644 = list(c(0x1.a3899b76c29a6p+0, 0x1.1884e29978092p-1, 0x1.e63b7fc3f0c1cp-4), "float64"),
  polyfit_full_f645 = list(0x1.c000000000000p-50, "float64"),
  polyfit_cov_f321 = list(c(-2.8285718, 1.7985723, 1.0416665), "float32"),
  polyfit_cov_f322 = list(array(c(0.08770426, -0.08770426, 0.012181147, -0.08770426, 0.09501295, -0.015835488, 0.012181144, -0.015835486, 0.0043310733), dim = c(3, 3)), "float32"),
  polyfit_cov_f641 = list(c(-0x1.6a0ea0ea0ea11p+1, 0x1.cc6f2d593bfb5p+0, 0x1.0aaaaaaaaaaa2p+0), "float64"),
  polyfit_cov_f642 = list(array(c(0x1.673c652568460p-4, -0x1.673c65256845fp-4, 0x1.8f26a9460214ep-7, -0x1.673c65256845ep-4, 0x1.852c183ddba15p-4, -0x1.0372bad3e7c14p-6, 0x1.8f26a9460214ap-7, -0x1.0372bad3e7c12p-6, 0x1.1bd739cb62324p-8), dim = c(3, 3)), "float64"),
  polyfit_covu_f321 = list(c(-1.0299999, 1.4345237), "float32"),
  polyfit_covu_f322 = list(array(c(1.2857141, -0.642857, -0.642857, 0.46428564), dim = c(2, 2)), "float32"),
  polyfit_covu_f641 = list(c(-0x1.07ae147ae1477p+0, 0x1.6f3cf3cf3cf3bp+0), "float64"),
  polyfit_covu_f642 = list(array(c(0x1.4924924924927p+0, -0x1.4924924924928p-1, -0x1.4924924924928p-1, 0x1.db6db6db6db72p-2), dim = c(2, 2)), "float64"),
  polyfit_cov2d_f321 = list(array(c(-1.0299999, 1.4345237, 2.0600002, 0.8090477), dim = c(2, 2)), "float32"),
  polyfit_cov2d_f322 = list(array(c(0.13919385, -0.069596924, -0.069596924, 0.050264444, 0.55677533, -0.27838764, -0.27838764, 0.20105775), dim = c(2, 2, 2)), "float32"),
  polyfit_cov2d_f641 = list(array(c(-0x1.07ae147ae1477p+0, 0x1.6f3cf3cf3cf3bp+0, 0x1.07ae147ae147ep+1, 0x1.9e3b7d516eb08p-1), dim = c(2, 2)), "float64"),
  polyfit_cov2d_f642 = list(array(c(0x1.1d11adff149b2p-3, -0x1.1d11adff149b3p-4, -0x1.1d11adff149b3p-4, 0x1.9bc434378f8acp-5, 0x1.1d11adff149b0p-1, -0x1.1d11adff149b1p-2, -0x1.1d11adff149b1p-2, 0x1.9bc434378f8aap-3), dim = c(2, 2, 2)), "float64"),
  roots_f32 = list(c(complex(real = 0.9999992251396179, imaginary = 0), complex(real = 2.0000040531158447, imaginary = 0), complex(real = 2.999995470046997, imaginary = 0)), "complex64"),
  roots_f64 = list(c(complex(real = 0x1.0000000000006p+0, imaginary = 0), complex(real = 0x1.fffffffffffe9p+0, imaginary = 0), complex(real = 0x1.7ffffffffffffp+1, imaginary = 0)), "complex128"),
  roots_c_f32 = list(c(complex(real = 0, imaginary = -1.9999998807907104), complex(real = -0, imaginary = 1.9999998807907104)), "complex64"),
  roots_c_f64 = list(c(complex(real = 0, imaginary = -0x1.0000000000001p+1), complex(real = -0, imaginary = 0x1.0000000000001p+1)), "complex128"),
  roots_lz_f32 = list(c(complex(real = 0.5, imaginary = 0), complex(real = 1, imaginary = 0)), "complex64"),
  roots_lz_f64 = list(c(complex(real = 0x1.0000000000000p-1, imaginary = 0), complex(real = 1, imaginary = 0)), "complex128"),
  roots_nostrip_f32 = list(c(complex(real = 0.5, imaginary = 0), complex(real = 1, imaginary = 0), complex(real = NaN, imaginary = NaN)), "complex64"),
  roots_nostrip_f64 = list(c(complex(real = 0x1.0000000000000p-1, imaginary = 0), complex(real = 1, imaginary = 0), complex(real = NaN, imaginary = NaN)), "complex128"),
  roots_jit_f32 = list(c(complex(real = 0.5, imaginary = 0), complex(real = 1, imaginary = 0), complex(real = NaN, imaginary = NaN), complex(real = NaN, imaginary = NaN)), "complex64"),
  roots_jit_f64 = list(c(complex(real = 0x1.0000000000000p-1, imaginary = 0), complex(real = 1, imaginary = 0), complex(real = NaN, imaginary = NaN), complex(real = NaN, imaginary = NaN)), "complex128"),
  roots_short_f32 = list(complex(0), "complex64"),
  roots_short_f64 = list(complex(0), "complex128"),
  poly_mat_f32 = list(c(complex(real = 1, imaginary = 0), complex(real = -5, imaginary = 0), complex(real = 5, imaginary = 0)), "complex64"),
  poly_mat_f64 = list(c(complex(real = 1, imaginary = 0), complex(real = -5, imaginary = 0), complex(real = 5, imaginary = 0)), "complex128"),
  polyval_int = list(c(11, 2), "int32"),
  polyadd_int = list(c(1, 3, 5), "int32"),
  polyder_int = list(c(2, 2), "float32"),
  polymul_int = list(c(1, 5, 6), "float32"),
  g_polyval_p = list(c(0x1.38a8ac0000000p+10, 0x1.4db2c00000000p+8, 0x1.6eb6000000000p+7, 0x1.90c0000000000p+4), "float64"),
  g_polyval_x = list(array(c(2, 8, 0x1.0000000000000p-1, 0x1.9000000000000p-1, 0x1.5200000000000p+4, 0x1.8800000000000p+4), dim = c(2, 3)), "float64"),
  g_polyval_long_x = list(array(c(0x1.753cb8ef74920p-3, -0x1.078d45e2ed56fp+8, -0x1.313608d95f960p-4, -0x1.218648c3cd950p-1, 0x1.4d73baae3718ap+1, 0x1.0f6cff1ee7a83p+12), dim = c(2, 3)), "float64"),
  g_polymul = list(c(27, -44, 30, -3), "float64"),
  g_polydiv = list(c(0x1.b000000000000p-2, -0x1.5000000000000p-1, 0x1.e000000000000p-1, 1), "float64"),
  g_polyint = list(c(0, 0x1.5555555555555p-4, 0x1.5555555555555p-2, 0x1.8000000000000p+0), "float64"),
  g_polyfit = list(c(0x1.3fffffffffffcp+3, 0x1.edb6db6db6dadp+0, -0x1.9b6db6db6db6dp+1, -0x1.5b6db6db6db6cp+2, -0x1.2db6db6db6dbbp+2, -0x1.124924924925ap+0, 0x1.6000000000008p+2), "float64"),
  g_poly = list(c(-0x1.4000000000000p+1, 0x1.0000000000000p-1, 3), "float64"),
  g_roots = list(c(-0x1.8fffffffffffcp+5, -0x1.7fffffffffffep+3, -2, -0x1.be31d9d4a9266p-52), "float64"),
  hess_polyval = list(0x1.333333333332cp-2, "float64"),
  vmap_polyval = list(array(c(0x1.1000000000000p+2, 0x1.e000000000000p+1, 11, 0), dim = c(2, 2)), "float64"),
  vmap_polymul = list(array(c(1, 0x1.8000000000000p+0, 1, 2, 0, 3, 4, 4), dim = c(2, 4)), "float64"),
  vmap_polyder = list(array(c(2, -2, 2, 0), dim = c(2, 2)), "float64"),
  end = NULL
)

pol_p <- c(1.5, -3, 2, 0.5)
pol_q <- c(2, -1)
pol_xs <- matrix(c(0, 1, -1.5, 2, 0.25, 3), 2, byrow = TRUE)
pol_xf <- seq(0, 1, length.out = 7)
pol_yf <- 1 + 2 * pol_xf - 3 * pol_xf^2 + c(0.1, -0.1, 0, 0.05, 0, -0.05, 0.02)
pol_Y2 <- cbind(pol_yf, rev(pol_yf) * 2)

pol_check <- function(name, got, tol) {
  if (!is.list(got)) got <- list(got)
  keys <- if (length(got) == 1L && !is.null(pol_exp[[name]])) name else paste0(name, seq_along(got))
  for (i in seq_along(got)) {
    e <- pol_exp[[keys[i]]]
    expect_identical(dtype(got[[i]]), e[[2L]], info = keys[i])
    g <- as.array(got[[i]])
    expect_equal(as.vector(g), as.vector(e[[1L]]), tolerance = tol, info = keys[i])
    if (length(dim(e[[1L]])) > 1L) expect_identical(as.integer(dim(g)), as.integer(dim(e[[1L]])), info = keys[i])
  }
}

# Roots sorted by real then imaginary part (eigenvalue order depends on the LAPACK build).
pol_sort_roots <- function(r) {
  v <- as.vector(as.array(r))
  as_jax(v[order(Re(v), Im(v))], dtype(r))
}

pol_cases <- list(
  polyval = function(dt) jnp_polyval(as_jax(pol_p, dt), as_jax(pol_xs, dt)),
  polyval_long = function(dt) jnp_polyval(as_jax((1:10) / 7, dt), as_jax(pol_xs, dt)),
  polyval_unroll = function(dt) jnp_polyval(as_jax((1:10) / 7, dt), as_jax(pol_xs, dt), unroll = 16L),
  polyval_2d = function(dt) jnp_polyval(as_jax(matrix(1:6, 3, byrow = TRUE), dt), as_jax(c(0.5, -2), dt)),
  polyval_1 = function(dt) jnp_polyval(as_jax(array(3, 1L), dt), as_jax(pol_xs, dt)),
  polyadd = function(dt) jnp_polyadd(as_jax(pol_p, dt), as_jax(pol_q, dt)),
  polyadd_r = function(dt) jnp_polyadd(as_jax(pol_q, dt), as_jax(pol_p, dt)),
  polysub = function(dt) jnp_polysub(as_jax(pol_q, dt), as_jax(pol_p, dt)),
  polymul = function(dt) jnp_polymul(as_jax(pol_p, dt), as_jax(pol_q, dt)),
  polymul_trim = function(dt) jnp_polymul(as_jax(c(0, 0, 1, 2), dt), as_jax(c(0, 3), dt),
                                          trim_leading_zeros = TRUE),
  polydiv = function(dt) jnp_polydiv(as_jax(pol_p, dt), as_jax(pol_q, dt)),
  polydiv_trim = function(dt) jnp_polydiv(as_jax(c(1, -3, 2), dt), as_jax(c(1, -1), dt),
                                          trim_leading_zeros = TRUE),
  polydiv_short = function(dt) jnp_polydiv(as_jax(pol_q, dt), as_jax(pol_p, dt)),
  polyder = function(dt) jnp_polyder(as_jax(pol_p, dt)),
  polyder2 = function(dt) jnp_polyder(as_jax(pol_p, dt), 2L),
  polyder0 = function(dt) jnp_polyder(as_jax(pol_p, dt), 0L),
  polyder_big = function(dt) jnp_polyder(as_jax(pol_p, dt), 5L),
  polyint = function(dt) jnp_polyint(as_jax(pol_p, dt)),
  polyint2k = function(dt) jnp_polyint(as_jax(pol_p, dt), 2L, k = as_jax(c(1, -2), dt)),
  polyint3k = function(dt) jnp_polyint(as_jax(pol_p, dt), 3L, k = 5),
  poly = function(dt) jnp_poly(as_jax(c(1, -2, 0.5), dt)),
  poly_empty = function(dt) jnp_poly(jnp_zeros(0L, dt)),
  polyfit = function(dt) jnp_polyfit(as_jax(pol_xf, dt), as_jax(pol_yf, dt), 2L),
  polyfit_w = function(dt) jnp_polyfit(as_jax(pol_xf, dt), as_jax(pol_yf, dt), 3L,
                                       w = as_jax(seq(1, 2, length.out = 7), dt)),
  polyfit_2d = function(dt) jnp_polyfit(as_jax(pol_xf, dt), as_jax(pol_Y2, dt), 1L),
  polyfit_full = function(dt) jnp_polyfit(as_jax(pol_xf, dt), as_jax(pol_yf, dt), 2L, full = TRUE),
  polyfit_cov = function(dt) jnp_polyfit(as_jax(pol_xf, dt), as_jax(pol_yf, dt), 2L, cov = TRUE),
  polyfit_covu = function(dt) jnp_polyfit(as_jax(pol_xf, dt), as_jax(pol_yf, dt), 1L, cov = "unscaled"),
  polyfit_cov2d = function(dt) jnp_polyfit(as_jax(pol_xf, dt), as_jax(pol_Y2, dt), 1L, cov = TRUE),
  roots = function(dt) pol_sort_roots(jnp_roots(as_jax(c(1, -6, 11, -6), dt))),
  roots_c = function(dt) pol_sort_roots(jnp_roots(as_jax(c(1, 0, 4), dt))),
  roots_lz = function(dt) pol_sort_roots(jnp_roots(as_jax(c(0, 0, 2, -3, 1), dt))),
  roots_nostrip = function(dt) pol_sort_roots(jnp_roots(as_jax(c(0, 2, -3, 1), dt), strip_zeros = FALSE)),
  roots_jit = function(dt) pol_sort_roots(jit(function(v) jnp_roots(v, strip_zeros = FALSE))(
    as_jax(c(0, 0, 2, -3, 1), dt))),
  roots_short = function(dt) jnp_roots(as_jax(3, dt)),
  poly_mat = function(dt) jnp_poly(as_jax(matrix(c(2, 1, 1, 3), 2), dt))
)

# The least squares, eigenvalue and singular value results go through R's LAPACK.
pol_lapack <- c("polyfit", "polyfit_w", "polyfit_2d", "polyfit_full", "polyfit_cov", "polyfit_covu",
                "polyfit_cov2d", "roots", "roots_c", "roots_lz", "roots_nostrip", "roots_jit", "poly_mat")

test_that("polynomial functions match JAX in float32 and float64", {
  skip_if_no_backend()
  for (nm in names(pol_cases)) {
    pol_check(paste0(nm, "_f32"), pol_cases[[nm]]("float32"), if (nm %in% pol_lapack) 1e-4 else 1e-6)
  }
  enable_x64()
  on.exit(enable_x64(FALSE))
  for (nm in names(pol_cases)) {
    pol_check(paste0(nm, "_f64"), pol_cases[[nm]]("float64"), if (nm %in% pol_lapack) 1e-10 else 1e-15)
  }
  # jit agrees with eager evaluation
  for (nm in setdiff(names(pol_cases), c(pol_lapack, "polymul_trim", "polydiv_trim", "poly_empty"))) {
    a <- pol_cases[[nm]]("float64")
    b <- jit(function() pol_cases[[nm]]("float64"))()
    expect_equal(lapply(if (is.list(a)) a else list(a), as.array),
                 lapply(if (is.list(b)) b else list(b), as.array), tolerance = 1e-15, info = nm)
  }
})

test_that("polynomial functions agree under jit with traced inputs", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  as_list <- function(v) lapply(if (is.list(v)) v else list(v), as.array)
  jcheck <- function(f, ..., tol = 1e-15) {
    expect_equal(as_list(jit(f)(...)), as_list(f(...)), tolerance = tol)
  }
  p <- as_jax(pol_p, "float64")
  q <- as_jax(pol_q, "float64")
  xs <- as_jax(pol_xs, "float64")
  xf <- as_jax(pol_xf, "float64")
  yf <- as_jax(pol_yf, "float64")
  Y2 <- as_jax(pol_Y2, "float64")
  w <- as_jax(seq(1, 2, length.out = 7), "float64")
  jcheck(jnp_polyval, p, xs)
  jcheck(function(c, x) jnp_polyval(c, x, unroll = 4L), as_jax((1:10) / 7, "float64"), xs)
  jcheck(jnp_polyadd, p, q)
  jcheck(jnp_polyadd, q, p)
  jcheck(jnp_polysub, q, p)
  jcheck(jnp_polymul, p, q)
  jcheck(jnp_polydiv, p, q)
  jcheck(jnp_polydiv, q, p)
  jcheck(function(c) jnp_polyder(c, 2L), p)
  jcheck(function(c, k) jnp_polyint(c, 2L, k = k), p, as_jax(c(1, -2), "float64"))
  jcheck(jnp_poly, as_jax(c(1, -2, 0.5), "float64"))
  jcheck(jnp_poly, as_jax(matrix(c(2, 1, 1, 3), 2), "float64"), tol = 1e-12)
  jcheck(function(x, y) jnp_polyfit(x, y, 2L), xf, yf, tol = 1e-12)
  jcheck(function(x, y, w) jnp_polyfit(x, y, 3L, w = w), xf, yf, w, tol = 1e-10)
  jcheck(function(x, y) jnp_polyfit(x, y, 2L, full = TRUE), xf, yf, tol = 1e-10)
  jcheck(function(x, y) jnp_polyfit(x, y, 2L, cov = TRUE), xf, yf, tol = 1e-10)
  jcheck(function(x, y) jnp_polyfit(x, y, 1L, cov = TRUE), xf, Y2, tol = 1e-10)
})

test_that("polynomial functions follow JAX dtype rules for integers", {
  skip_if_no_backend()
  pol_check("polyval_int", jnp_polyval(c(1L, 2L, 3L), c(2L, -1L)), 0)
  pol_check("polyadd_int", jnp_polyadd(c(1L, 2L), c(1L, 2L, 3L)), 0)
  pol_check("polyder_int", jnp_polyder(c(1L, 2L, 3L)), 0)
  pol_check("polymul_int", jnp_polymul(c(1L, 2L), c(1L, 3L)), 1e-6)
})

test_that("polynomial gradients match JAX and finite differences", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  p <- as_jax(pol_p, "float64")
  xs <- as_jax(pol_xs, "float64")
  fp <- function(c) jnp_sum(jnp_polyval(c, xs)^2)
  pol_check("g_polyval_p", grad(fp)(p), 1e-14)
  check_grad_fd(fp, p)
  fx <- function(x) jnp_sum(jnp_polyval(p, x))
  pol_check("g_polyval_x", grad(fx)(xs), 1e-14)
  fl <- function(x) jnp_sum(jnp_sin(jnp_polyval(jnp_arange(1, 11, dtype = "float64") / 7, x)))
  # Horner differentiated step by step (JAX differentiates the polynomial primitive)
  pol_check("g_polyval_long_x", grad(fl)(xs), 1e-10)
  check_grad_fd(fl, xs, eps = 1e-7, tolerance = 1e-4)
  fm <- function(c) jnp_sum(jnp_polymul(c, c(2, -1))^2)
  pol_check("g_polymul", grad(fm)(p), 1e-14)
  check_grad_fd(fm, p)
  fd <- function(c) {
    r <- jnp_polydiv(c, c(2, -1))
    jnp_sum(r[[1L]]^2) + jnp_sum(r[[2L]])
  }
  pol_check("g_polydiv", grad(fd)(p), 1e-14)
  check_grad_fd(fd, p)
  fi <- function(c) jnp_sum(jnp_polyint(c, 2L) * jnp_arange(6, dtype = "float64"))
  pol_check("g_polyint", grad(fi)(p), 1e-14)
  ff <- function(y) jnp_sum(jnp_polyfit(as_jax(pol_xf, "float64"), y, 2L) * c(1, -2, 3))
  pol_check("g_polyfit", grad(ff)(as_jax(pol_yf, "float64")), 1e-9)
  check_grad_fd(ff, as_jax(pol_yf, "float64"))
  fz <- function(z) jnp_sum(jnp_poly(z) * jnp_arange(1, 5, dtype = "float64"))
  pol_check("g_poly", grad(fz)(as_jax(c(1, -2, 0.5), "float64")), 1e-14)
  fr <- function(c) jnp_sum(jnp_real(jnp_roots(c))^2)
  g <- grad(fr)(as_jax(c(1, -6, 11, -6), "float64"))
  expect_equal(as.vector(as.array(g)), as.vector(pol_exp$g_roots[[1L]]), tolerance = 1e-8)
  # second order
  h <- jacfwd(grad(function(x) jnp_polyval(p, x)))(as_jax(0.7, "float64"))
  pol_check("hess_polyval", h, 1e-14)
})

test_that("polynomial functions batch under vmap", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  C <- as_jax(matrix(c(1, 2, 3, -1, 0, 4), 2, byrow = TRUE), "float64")
  pol_check("vmap_polyval", vmap(function(c) jnp_polyval(c, c(0.5, 2)))(C), 1e-15)
  A <- as_jax(matrix(c(1, 2, 3, 4), 2, byrow = TRUE), "float64")
  B <- as_jax(matrix(c(1, -1, 2, 0.5, 0, 1), 2, byrow = TRUE), "float64")
  pol_check("vmap_polymul", vmap(jnp_polymul)(A, B), 1e-15)
  pol_check("vmap_polyder", vmap(jnp_polyder)(C), 1e-15)
  loop <- jnp_stack(lapply(0:1, function(i) jnp_polyint(C[i], 2L, k = 1)))
  expect_jax_equal(vmap(function(c) jnp_polyint(c, 2L, k = 1))(C), loop)
  loop <- jnp_stack(lapply(0:1, function(i) jnp_polydiv(C[i], c(1, 2))[[1L]]))
  expect_jax_equal(vmap(function(c) jnp_polydiv(c, c(1, 2))[[1L]])(C), loop)
})

test_that("polynomial errors match JAX", {
  skip_if_no_backend()
  xf <- pol_xf
  yf <- pol_yf
  expect_error(jit(jnp_roots)(as_jax(c(1, 2))), class = "jax_concretization_error")
  expect_error(jit(function(a) jnp_polymul(a, a, trim_leading_zeros = TRUE))(jnp_ones(3L)),
               class = "jax_concretization_error")
  expect_error(jnp_roots(jnp_ones(c(2L, 2L))), "Input must be a rank-1 array.", fixed = TRUE)
  expect_error(jnp_polyder(pol_p, -1L), "Order of derivative must be positive", fixed = TRUE)
  expect_error(jnp_polyint(pol_p, -1L), "Order of integral must be positive (see polyder)", fixed = TRUE)
  expect_error(jnp_polyint(pol_p, 2L, k = jnp_ones(3L)),
               "k must be a scalar or a rank-1 array of length 1 or m.", fixed = TRUE)
  expect_error(jnp_polyfit(xf, yf, -1L), "expected deg >= 0", fixed = TRUE)
  expect_error(jnp_polyfit(matrix(1, 2, 2), c(1, 1), 1L), "expected 1D vector for x", fixed = TRUE,
               class = "jax_type_error")
  expect_error(jnp_polyfit(numeric(0), numeric(0), 1L), "expected non-empty vector for x", fixed = TRUE)
  expect_error(jnp_polyfit(xf, array(1, c(7L, 2L, 2L)), 1L), "expected 1D or 2D array for y", fixed = TRUE)
  expect_error(jnp_polyfit(xf, rep(1, 6), 1L), "expected x and y to have same length", fixed = TRUE)
  expect_error(jnp_polyfit(xf, yf, 1L, w = matrix(1, 7, 1)), "expected a 1-d array for weights",
               fixed = TRUE)
  expect_error(jnp_polyfit(xf, yf, 1L, w = rep(1, 6)), "expected w and y to have the same length",
               fixed = TRUE)
  expect_error(jnp_polyfit(xf[1:3], yf[1:3], 2L, cov = TRUE),
               "the number of data points must exceed order to scale the covariance matrix", fixed = TRUE)
  expect_error(jnp_poly(matrix(1, 2, 3)), "input must be 1d or non-empty square 2d array.", fixed = TRUE)
})
