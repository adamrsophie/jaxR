# jax.scipy.special (port of jax/_src/scipy/special.py and the modules it builds on:
# jax/_src/third_party/scipy/betaln.py and special.py, jax/_src/ops/special.py and the
# softmax functions of jax/_src/nn/functions.py).

sp_complex_error <- function(name) {
  jax_error(name, " does not support complex-valued inputs.", class = "jax_value_error")
}

# jnp.arange(start, stop, dtype = dt) for start < stop, always of rank 1.
sp_arange <- function(start, stop, dt) as_jax(array(seq(start, stop - 1L)), dt)

# Port of jax/_src/numpy/ufuncs.py:signbit for floating point input.
sp_signbit <- function(x) {
  info <- finfo(sp_dtype(x))
  xi <- lax_bitcast_convert_type(x, paste0("int", info$bits))
  lax_convert_element_type(lax_shift_right_arithmetic(xi, lax_full_like(xi, info$nexp + info$nmant)),
                           "bool")
}

#' Special functions
#'
#' Ports of `jax.scipy.special`. Arguments broadcast with NumPy rules and are promoted to
#' an inexact dtype with JAX type promotion; functions with custom derivative rules in JAX
#' carry the same rules here.
#' @param x,y,a,b,c,p,q,z,h Arrays or R values.
#' @param n,m Orders (arrays; integer valued for `scipy_special_polygamma`).
#' @param N,k Arrays for `scipy_special_comb`.
#' @param lmbda Box-Cox power.
#' @param d Dimension of the multivariate gamma function (a concrete integer).
#' @param exact Must be `FALSE`.
#' @param repetition Count combinations with repetition.
#' @param axis 0-based axis or axes; `NULL` means all axes.
#' @param keepdims Keep reduced dimensions as size one.
#' @param return_sign Also return the sign of the sum.
#' @param where Optional boolean mask of the elements to include.
#' @param series_order Number of terms (0 to 30) of the asymptotic series used by
#'   `scipy_special_log_ndtr` for very negative arguments.
#' @param v Highest order computed by `scipy_special_bessel_jn`.
#' @param n_iter Number of backward recurrence steps.
#' @param theta,phi Polar (colatitudinal) and azimuthal (longitudinal) angles.
#' @param diff_n Unsupported; must be `NULL`.
#' @param n_max Maximum degree (a concrete integer; defaults to the maximum of `n`).
#' @param is_normalized Whether the Legendre functions are normalized for spherical
#'   harmonics.
#' @return An array (a list of arrays for `scipy_special_fresnel`, `scipy_special_sici`,
#'   `scipy_special_lpmn` and `scipy_special_logsumexp` with `return_sign = TRUE`).
#' @name scipy_special
NULL

# ---- gamma and beta functions ---------------------------------------------------------------

#' @rdname scipy_special
#' @export
scipy_special_gammaln <- function(x) lax_lgamma(promote_args_inexact(x)[[1L]])

# Port of jax/_src/scipy/special.py:gammasgn.
sp_gammasgn <- jit(function(x) {
  x <- promote_args_inexact(x)[[1L]]
  dt <- sp_dtype(x)
  if (is_complex_dtype(dt)) sp_complex_error("gammasgn")
  floor_x <- lax_floor(x)
  x_negative <- x < 0
  sp_select(list((x_negative & (x == floor_x)) | jnp_isnan(x),
                 (x_negative & (floor_x %% 2 != 0)) | ((x == 0) & sp_signbit(x))),
            list(as_jax(NaN, dt), as_jax(-1, dt)), as_jax(1, dt))
})

#' @rdname scipy_special
#' @export
scipy_special_gammasgn <- function(x) sp_gammasgn(x)

# Lanczos coefficients for g = 7, N = 9 (GSL specfunc/gamma.c, lanczos_7_c).
sp_lanczos_g <- 7.0
sp_lanczos_coeffs <- c(0x1.ffffffffff95p-1, 0x1.52429b6c30b05p+9, -0x1.3ac8e8ed4171bp+10,
                       0x1.81a9661d3b4d8p+9, -0x1.613ae51a32f5dp+7, 0x1.903c27f8b9c81p+3,
                       -0x1.1bcb2992b2855p-3, 0x1.4f0514e4e324fp-17, 0x1.435508f3faeefp-23)

# Port of jax/_src/scipy/special.py:_complex_loggamma (Lanczos approximation with the
# reflection formula for Re(z) < 0.5).
sp_complex_loggamma <- function(z) {
  dt <- sp_dtype(z)
  nd <- length(sp_shape(z))
  needs_reflection <- jnp_real(z) < 0.5
  z_lanczos <- jnp_where(needs_reflection, 1.0 - z, z)
  zz <- z_lanczos - 1.0
  lead <- seq_len(nd) - 1L
  ks <- lax_expand_dims(as_jax(array(1:8), dt), lead)
  tail <- lax_expand_dims(as_jax(array(sp_lanczos_coeffs[-1L]), dt), lead)
  ag <- as_jax(sp_lanczos_coeffs[1L], dt) + jnp_sum(tail / (lax_expand_dims(zz, nd) + ks), axis = -1L)
  t <- zz + sp_lanczos_g + 0.5
  half_log2pi <- as_jax(0.5 * log(2.0 * pi), dt)
  log_gamma_lanczos <- half_log2pi + (zz + 0.5) * jnp_log(t) - t + jnp_log(ag)
  # Mask z to a safe value in the reflection branch so the unselected path never produces
  # NaN (which would contaminate gradients through the VJP of where: 0 * NaN = NaN).
  z_safe <- jnp_where(needs_reflection, z, jnp_full_like(z, 0.5))
  reflected <- jnp_log(as_jax(pi, dt)) - jnp_log(jnp_sin(pi * z_safe)) - log_gamma_lanczos
  jnp_where(needs_reflection, reflected, log_gamma_lanczos)
}

sp_complex_pole_or_nan <- function(z) {
  re <- jnp_real(z)
  is_pole <- (jnp_imag(z) == 0) & (re == jnp_floor(re)) & (re <= 0)
  is_pole | jnp_isnan(z)
}

# Port of jax/_src/scipy/special.py:_complex_gamma.
sp_complex_gamma <- function(z) {
  nan_val <- as_jax(complex(real = NaN, imaginary = NaN), sp_dtype(z))
  # Poles and NaN are masked before loggamma so the unselected path stays finite.
  safe <- sp_complex_pole_or_nan(z)
  z_safe <- jnp_where(safe, jnp_ones_like(z), z)
  jnp_where(safe, nan_val, jnp_exp(sp_complex_loggamma(z_safe)))
}

# Port of jax/_src/scipy/special.py:_complex_loggamma_scipy.
sp_complex_loggamma_scipy <- jit(function(z) {
  nan_val <- as_jax(complex(real = NaN, imaginary = NaN), sp_dtype(z))
  safe <- sp_complex_pole_or_nan(z)
  z_safe <- jnp_where(safe, jnp_ones_like(z), z)
  n <- jnp_maximum(0L, jnp_astype(jnp_ceil(0.5 - jnp_real(z_safe)), default_int()))
  cond_fun <- function(state) jnp_any(state[[1L]] < n)
  body_fun <- function(state) {
    k <- state[[1L]]
    sum_log <- state[[2L]]
    mask <- k < n
    zk <- jnp_where(mask, z_safe + k, jnp_ones_like(z_safe))
    list(k + 1L, sum_log + jnp_where(mask, jnp_log(zk), jnp_zeros_like(sum_log)))
  }
  # Shift z into the right half plane with loggamma(z) = loggamma(z + n) - sum(log(z + k)),
  # k = 0..n-1, to match SciPy's branch cuts along the negative real axis.
  sum_log <- lax_while_loop(cond_fun, body_fun, list(sp_int(0L), jnp_zeros_like(z_safe)))[[2L]]
  res <- sp_complex_loggamma(z_safe + n) - sum_log
  jnp_where(safe, nan_val, res)
})

# Port of jax/_src/scipy/special.py:loggamma.
#' @rdname scipy_special
#' @export
scipy_special_loggamma <- function(x) {
  x <- promote_args_inexact(x)[[1L]]
  if (is_complex_dtype(sp_dtype(x))) return(sp_complex_loggamma_scipy(x))
  jnp_where(x > 0, lax_lgamma(x), NaN)
}

# Port of jax/_src/scipy/special.py:gamma.
#' @rdname scipy_special
#' @export
scipy_special_gamma <- function(x) {
  x <- promote_args_inexact(x)[[1L]]
  if (is_complex_dtype(sp_dtype(x))) return(sp_complex_gamma(x))
  scipy_special_gammasgn(x) * lax_exp(lax_lgamma(x))
}

# Port of jax/_src/third_party/scipy/betaln.py:algdiv: log(gamma(a) / gamma(a + b)) for
# b >= 8, assuming a <= b.
sp_algdiv <- function(a, b) {
  c0 <- 0.833333333333333e-01
  c1 <- -0.277777777760991e-02
  c2 <- 0.793650666825390e-03
  c3 <- -0.595202931351870e-03
  c4 <- 0.837308034031215e-03
  c5 <- -0.165322962780713e-02
  h <- a / b
  c <- h / (1 + h)
  x <- h / (1 + h)
  d <- b + (a - 0.5)
  # sN = (1 - x^n) / (1 - x)
  x2 <- x * x
  s3 <- 1.0 + (x + x2)
  s5 <- 1.0 + (x + x2 * s3)
  s7 <- 1.0 + (x + x2 * s5)
  s9 <- 1.0 + (x + x2 * s7)
  s11 <- 1.0 + (x + x2 * s9)
  # w = del(b) - del(a + b), where ln(gamma(x)) = (x - 0.5) ln(x) - x + 0.5 ln(2 pi) + del(x)
  t <- (1.0 / b)^2
  w <- ((((c5 * s11 * t + c4 * s9) * t + c3 * s7) * t + c2 * s5) * t + c1 * s3) * t + c0
  w <- w * (c / b)
  u <- d * lax_log1p(a / b)
  v <- a * (lax_log(b) - 1.0)
  jnp_where(u <= v, (w - v) - u, (w - u) - v)
}

# Port of jax/_src/third_party/scipy/betaln.py:betaln.
#' @rdname scipy_special
#' @export
scipy_special_betaln <- function(a, b) {
  ab <- promote_args_inexact(a, b)
  a <- ab[[1L]]
  b <- ab[[2L]]
  # where (rather than minimum and maximum) keeps the gradient on one branch when a == b.
  new_a <- jnp_where(a <= b, a, b)
  new_b <- jnp_where(a <= b, b, a)
  small_b <- lax_lgamma(new_a) + (lax_lgamma(new_b) - lax_lgamma(new_a + new_b))
  large_b <- lax_lgamma(new_a) + sp_algdiv(new_a, new_b)
  jnp_where(new_b < 8, small_b, large_b)
}

# Port of jax/_src/scipy/special.py:factorial.
#' @rdname scipy_special
#' @export
scipy_special_factorial <- function(n, exact = FALSE) {
  if (exact) jax_error("factorial with exact=True", class = "jax_not_implemented_error")
  n <- promote_args_inexact(n)[[1L]]
  jnp_where(n < 0, 0L, lax_exp(lax_lgamma(n + 1L)))
}

# Port of jax/_src/scipy/special.py:comb.
#' @rdname scipy_special
#' @export
scipy_special_comb <- function(N, k, repetition = FALSE) {
  Nk <- promote_args_inexact(N, k)
  N <- Nk[[1L]]
  k <- Nk[[2L]]
  if (repetition) {
    cond <- (k == 0) & (N >= 0)
    result <- scipy_special_comb(N + k - 1L, k, repetition = FALSE)
    return(jnp_where(cond, 1.0, result))
  }
  cond <- (k <= N) & (N >= 0) & (k >= 0)
  safe_N <- jnp_where(cond, N, 0.0)
  safe_k <- jnp_where(cond, k, 0.0)
  result <- lax_exp(scipy_special_gammaln(safe_N + 1L) - scipy_special_gammaln(safe_k + 1L) -
                      scipy_special_gammaln(safe_N + 1L - safe_k))
  jnp_where(cond, result, 0.0)
}

# Port of jax/_src/scipy/special.py:beta.
#' @rdname scipy_special
#' @export
scipy_special_beta <- function(a, b) {
  ab <- promote_args_inexact(a, b)
  a <- ab[[1L]]
  b <- ab[[2L]]
  if (is_complex_dtype(sp_dtype(a))) sp_complex_error("beta")
  sign <- scipy_special_gammasgn(a) * scipy_special_gammasgn(b) * scipy_special_gammasgn(a + b)
  sign * lax_exp(scipy_special_betaln(a, b))
}

# Port of jax/_src/scipy/special.py:betainc.
#' @rdname scipy_special
#' @export
scipy_special_betainc <- function(a, b, x) {
  abx <- promote_args_inexact(a, b, x)
  if (is_complex_dtype(sp_dtype(abx[[3L]]))) sp_complex_error("betainc")
  lax_betainc(abx[[1L]], abx[[2L]], abx[[3L]])
}

#' @rdname scipy_special
#' @export
scipy_special_digamma <- function(x) lax_digamma(promote_args_inexact(x)[[1L]])

#' @rdname scipy_special
#' @export
scipy_special_gammainc <- function(a, x) {
  ax <- promote_args_inexact(a, x)
  lax_igamma(ax[[1L]], ax[[2L]])
}

#' @rdname scipy_special
#' @export
scipy_special_gammaincc <- function(a, x) {
  ax <- promote_args_inexact(a, x)
  lax_igammac(ax[[1L]], ax[[2L]])
}

# Port of jax/_src/scipy/special.py:multigammaln.
#' @rdname scipy_special
#' @export
scipy_special_multigammaln <- function(a, d) {
  d <- as.integer(as.array(concrete_or_error(d, "d argument of multigammaln")))
  ad <- promote_args_inexact(a, d)
  a <- ad[[1L]]
  d_ <- ad[[2L]]
  constant <- lax_mul(lax_mul(lax_mul(sp_const(a, 0.25), d_), lax_sub(d_, sp_const(a, 1))),
                      lax_log(sp_const(a, pi)))
  b <- lax_div(jnp_arange(d, dtype = sp_dtype(d_)), sp_const(a, 2))
  nd <- length(sp_shape(a))
  res <- jnp_sum(scipy_special_gammaln(jnp_expand_dims(a, nd) -
                                         jnp_expand_dims(b, seq_len(nd) - 1L)), axis = -1L)
  res + constant
}

# Port of jax/_src/scipy/special.py:poch (custom JVP in z and m).
sp_poch <- custom_jvp(function(z, m) {
  zm <- promote_args_inexact(z, m)
  z <- zm[[1L]]
  m <- zm[[2L]]
  if (is_complex_dtype(sp_dtype(z))) sp_complex_error("jnp.poch")
  jnp_where(m == 0, as_jax(1, sp_dtype(z)), scipy_special_gamma(z + m) / scipy_special_gamma(z))
})
sp_poch <- sp_defjvps(
  sp_poch,
  function(z_dot, primal_out, z, m) {
    (scipy_special_digamma(z + m) - scipy_special_digamma(z)) * sp_poch(z, m) * z_dot
  },
  function(m_dot, primal_out, z, m) scipy_special_digamma(z + m) * sp_poch(z, m) * m_dot)

#' @rdname scipy_special
#' @export
scipy_special_poch <- function(z, m) sp_poch(z, m)

# Port of jax/_src/scipy/special.py:polygamma.
#' @rdname scipy_special
#' @export
scipy_special_polygamma <- function(n, x) {
  # Whole valued R numbers count as integers, as Python ints do.
  if (!is_jax_value(n) && is.numeric(n) && all(n == round(n))) storage.mode(n) <- "integer"
  n_dtype <- sp_dtype(as_jax(n))
  if (!is_int_dtype(n_dtype))
    jax_error("Argument `n` to polygamma must be of integer type. Got dtype ", n_dtype, ".",
              class = "jax_value_error")
  nx <- promote_args_inexact(n, x)
  if (is_complex_dtype(sp_dtype(nx[[2L]]))) sp_complex_error("polygamma")
  lax_polygamma(nx[[1L]], nx[[2L]])
}

# Port of jax/_src/scipy/special.py:zeta.
sp_zeta <- custom_jvp(function(x, q) {
  xq <- promote_args_inexact(x, q)
  lax_zeta(xq[[1L]], xq[[2L]])
})

# Coefficients (2k)! / B_2k, with B the Bernoulli numbers (12, -720, 30240, -1209600,
# 47900160, -1307674368000 / 691, ...). The quotients of large integers are written as the
# exact doubles of the correctly rounded divisions.
sp_bernoulli_coefs <- c(12, -720, 30240, -1209600, 47900160, -0x1.c331393146053p+30, 74724249600,
  -0x1.5770bf3df715p+41, 0x1.a7b51e18d632bp+46, -0x1.055d6586f1431p+52, 0x1.427237ec0a9dep+57,
  -0x1.8dcd555480632p+62, 0x1.eac4c79e4706fp+67, -0x1.2ebb1540bcf69p+73, 0x1.757aa9601cf22p+78,
  -0x1.ccc2ed0971882p+83)

# Port of jax/_src/scipy/special.py:_zeta_series_expansion. There is no closed form for
# the derivative of the zeta function; this is formula (5) of Johansson, "Rigorous high
# precision computation of the Hurwitz zeta function and its derivatives" (2015), with the
# notation of the reference.
sp_zeta_series_expansion <- function(x, q) {
  sa <- promote_args_inexact(x, q)
  s <- sa[[1L]]
  a <- sa[[2L]]
  dt <- sp_dtype(a)
  typed <- function(v) as_jax(v, dt)
  s_ <- jnp_expand_dims(s, -1L)
  a_ <- jnp_expand_dims(a, -1L)
  # precision ~ N, M
  n_terms <- if (dt == "float32") 8L else 16L
  N <- typed(n_terms)
  k <- lax_expand_dims(typed(array(seq_len(n_terms) - 1L)), seq_along(sp_shape(a)) - 1L)
  # Operands are broadcast before pow: the JVP of pow_p in lax.R selects with a predicate of
  # the exponent's shape, which fails for size one broadcasting.
  pow <- function(x, y) do.call(jnp_power, jnp_broadcast_arrays(x, y))
  S <- jnp_sum(pow(a_ + k, -s_), -1L)
  I <- lax_div(pow(a + N, typed(1) - s), s - typed(1))
  T0 <- pow(a + N, -s)
  m <- lax_expand_dims(typed(array(seq_len(2L * n_terms) - 1L)), seq_along(sp_shape(s)) - 1L)
  s_over_a <- (s_ + m) / (a_ + N)
  T1 <- lax_slice_in_dim(jnp_cumprod(s_over_a, -1L), 0L, 2L * n_terms, stride = 2L, axis = -1L)
  T1 <- jnp_minimum(T1, typed(finfo(dt)$max))
  coefs <- lax_expand_dims(typed(array(sp_bernoulli_coefs[seq_len(n_terms)])),
                           seq_along(sp_shape(a)) - 1L)
  T1 <- T1 / coefs
  T <- T0 * (typed(0.5) + jnp_sum(T1, -1L))
  S + I + T
}

# Port of jax/_src/scipy/special.py:_zeta_jvp. The primal comes from the function itself:
# the series is an analytic continuation that is finite where zeta returns NaN (x < 1).
sp_zeta <- custom_jvp_defjvp(sp_zeta, function(primals, tangents) {
  primal_out <- sp_zeta(primals[[1L]], primals[[2L]])
  tangent_out <- jvp(sp_zeta_series_expansion, primals, tangents)[[2L]]
  # A multiplicative mask keeps the rule linear in the tangent, so reverse mode gives NaN too.
  tangent_out <- tangent_out * jnp_where(jnp_isnan(primal_out), NaN, 1.0)
  list(primal_out, tangent_out)
})

#' @rdname scipy_special
#' @export
scipy_special_zeta <- function(x, q = NULL) {
  if (is.null(q))
    jax_error("Riemann zeta function not implemented; pass q != None to compute the Hurwitz ",
              "Zeta function.", class = "jax_not_implemented_error")
  sp_zeta(x, q)
}

# ---- error function family --------------------------------------------------------------------

#' @rdname scipy_special
#' @export
scipy_special_erf <- function(x) lax_erf(promote_args_inexact(x)[[1L]])

#' @rdname scipy_special
#' @export
scipy_special_erfc <- function(x) lax_erfc(promote_args_inexact(x)[[1L]])

#' @rdname scipy_special
#' @export
scipy_special_erfinv <- function(x) lax_erf_inv(promote_args_inexact(x)[[1L]])

# ---- logistic, entropy and related functions --------------------------------------------------

# Port of jax/_src/scipy/special.py:logit.
sp_logit <- custom_jvp(function(x) {
  x <- promote_args_inexact(x)[[1L]]
  lax_log(lax_div(x, lax_sub(sp_const(x, 1), x)))
})
sp_logit <- sp_defjvps(sp_logit, function(g, ans, x) {
  lax_div(g, lax_mul(x, lax_sub(sp_const(x, 1), x)))
})

#' @rdname scipy_special
#' @export
scipy_special_logit <- function(x) sp_logit(x)

#' @rdname scipy_special
#' @export
scipy_special_expit <- function(x) lax_logistic(promote_args_inexact(x)[[1L]])

# Port of jax/_src/scipy/special.py:xlogy. xlogy(0, 0) is 0; NaN in y propagates even when
# x == 0, as in scipy.
sp_xlogy <- custom_jvp(function(x, y) {
  xy <- promote_args_inexact(x, y)
  x <- xy[[1L]]
  y <- xy[[2L]]
  jnp_where((x != 0) | jnp_isnan(y), lax_mul(x, lax_log(y)), jnp_zeros_like(x))
})
sp_xlogy <- custom_jvp_defjvp(sp_xlogy, function(primals, tangents) {
  x <- primals[[1L]]
  y <- primals[[2L]]
  result <- sp_xlogy(x, y)
  list(result, jnp_astype(tangents[[1L]] * lax_log(y) + tangents[[2L]] * x / y, sp_dtype(result)))
})

#' @rdname scipy_special
#' @export
scipy_special_xlogy <- function(x, y) sp_xlogy(x, y)

# Port of jax/_src/scipy/special.py:xlog1py. xlog1py(0, -1) is 0.
sp_xlog1py <- custom_jvp(function(x, y) {
  xy <- promote_args_inexact(x, y)
  x <- xy[[1L]]
  y <- xy[[2L]]
  jnp_where((x != 0) | jnp_isnan(y), lax_mul(x, lax_log1p(y)), jnp_zeros_like(x))
})
sp_xlog1py <- custom_jvp_defjvp(sp_xlog1py, function(primals, tangents) {
  x <- primals[[1L]]
  y <- primals[[2L]]
  result <- sp_xlog1py(x, y)
  list(result, jnp_astype(tangents[[1L]] * lax_log1p(y) + tangents[[2L]] * x / (1L + y),
                          sp_dtype(result)))
})

#' @rdname scipy_special
#' @export
scipy_special_xlog1py <- function(x, y) sp_xlog1py(x, y)

# Port of jax/_src/scipy/special.py:_xlogx: x log(x) with well defined derivatives.
sp_xlogx <- custom_jvp(function(x) sp_xlogy(x, x))
sp_xlogx <- custom_jvp_defjvp(sp_xlogx, function(primals, tangents) {
  x <- primals[[1L]]
  list(sp_xlogx(x), tangents[[1L]] * (lax_log(x) + 1L))
})

# Port of jax/_src/scipy/special.py:entr.
#' @rdname scipy_special
#' @export
scipy_special_entr <- function(x) {
  x <- promote_args_inexact(x)[[1L]]
  if (is_complex_dtype(sp_dtype(x))) sp_complex_error("entr")
  lax_select(lax_lt(x, sp_const(x, 0)), lax_full_like(x, -Inf), lax_neg(sp_xlogx(x)))
}

# Port of jax/_src/scipy/special.py:boxcox.
#' @rdname scipy_special
#' @export
scipy_special_boxcox <- function(x, lmbda) {
  xl <- promote_args_inexact(x, lmbda)
  x <- xl[[1L]]
  lmbda <- xl[[2L]]
  lmbda_is_zero <- lmbda == 0L
  safe_lmbda <- jnp_where(lmbda_is_zero, jnp_ones_like(lmbda), lmbda)
  log_x <- jnp_log(x)
  power_branch <- jnp_expm1(safe_lmbda * log_x) / safe_lmbda
  jnp_where(lmbda_is_zero, log_x, power_branch)
}

# Port of jax/_src/scipy/special.py:boxcox1p.
#' @rdname scipy_special
#' @export
scipy_special_boxcox1p <- function(x, lmbda) {
  xl <- promote_args_inexact(x, lmbda)
  x <- xl[[1L]]
  lmbda <- xl[[2L]]
  lmbda_is_zero <- lmbda == 0L
  safe_lmbda <- jnp_where(lmbda_is_zero, jnp_ones_like(lmbda), lmbda)
  log1p_x <- jnp_log1p(x)
  power_branch <- jnp_expm1(safe_lmbda * log1p_x) / safe_lmbda
  jnp_where(lmbda_is_zero, log1p_x, power_branch)
}

# Port of jax/_src/scipy/special.py:_log_p_over_q. Following scipy: log(p / q) through
# log1p((p - q) / q) when 0.5 < p / q < 2 (no cancellation near 1), log(p / q) when
# tiny < p / q < inf, and log(p) - log(q) otherwise (no overflow or underflow in p / q).
sp_log_p_over_q <- function(safe_p, safe_q) {
  ratio <- lax_div(safe_p, safe_q)
  near_one <- lax_bitwise_and(lax_gt(ratio, sp_const(safe_p, 0.5)),
                              lax_lt(ratio, sp_const(safe_p, 2.0)))
  tiny <- sp_const(safe_p, finfo(sp_dtype(safe_p))$tiny)
  normal_ratio <- lax_bitwise_and(lax_gt(ratio, tiny), lax_lt(ratio, sp_const(safe_p, Inf)))
  near_one_p <- jnp_where(near_one, safe_p, 1L)
  near_one_q <- jnp_where(near_one, safe_q, 1L)
  near_one_delta <- lax_sub(near_one_p, near_one_q)
  safe_rel_diff <- lax_div(near_one_delta, near_one_q)
  safe_ratio <- jnp_where(normal_ratio, ratio, 1L)
  log_p_over_q <- jnp_where(
    near_one, lax_log1p(safe_rel_diff),
    jnp_where(normal_ratio, lax_log(safe_ratio), lax_sub(lax_log(safe_p), lax_log(safe_q))))
  list(log_p_over_q = log_p_over_q, ratio = ratio, near_one = near_one,
       near_one_delta = near_one_delta, safe_rel_diff = safe_rel_diff)
}

sp_rel_entr_masks <- function(p, q) {
  zero <- sp_const(p, 0.0)
  both_gt_zero_mask <- lax_bitwise_and(lax_gt(p, zero), lax_gt(q, zero))
  list(zero = zero,
       nan_mask = lax_bitwise_or(jnp_isnan(p), jnp_isnan(q)),
       both_gt_zero_mask = both_gt_zero_mask,
       one_zero_mask = lax_bitwise_and(lax_eq(p, zero), lax_ge(q, zero)),
       safe_p = jnp_where(both_gt_zero_mask, p, 1L),
       safe_q = jnp_where(both_gt_zero_mask, q, 1L))
}

# Port of jax/_src/scipy/special.py:_rel_entr: shared implementation of rel_entr and
# kl_div, matching scipy.
sp_rel_entr <- custom_jvp(jit(function(p, q, kl) {
  mk <- sp_rel_entr_masks(p, q)
  zero <- mk$zero
  safe_p <- mk$safe_p
  safe_q <- mk$safe_q
  lp <- sp_log_p_over_q(safe_p, safe_q)
  u <- lp$safe_rel_diff
  log_val <- lax_mul(safe_p, lp$log_p_over_q)
  if (kl) {
    # For |u| = |(p - q) / q| < 0.05 evaluate the Taylor series
    # p log(1 + u) - (p - q) = (p - q) u (1/2 - u/6 + u^2/12 - u^3/20 + u^4/30 - ...)
    # directly, avoiding catastrophic cancellation and negative values near p = q.
    c1 <- sp_const(p, 1 / 2)
    c2 <- sp_const(p, 1 / 6)
    c3 <- sp_const(p, 1 / 12)
    c4 <- sp_const(p, 1 / 20)
    c5 <- sp_const(p, 1 / 30)
    poly <- c1 - u * (c2 - u * (c3 - u * (c4 - u * c5)))
    series_val <- lax_mul(lax_mul(lp$near_one_delta, u), poly)
    small_u <- lax_bitwise_and(lp$near_one, lax_lt(lax_abs(u), sp_const(p, 0.05)))
    pos_val <- jnp_where(small_u, series_val, lax_sub(log_val, lax_sub(safe_p, safe_q)))
  } else {
    pos_val <- log_val
  }
  jnp_where(mk$nan_mask, sp_const(p, NaN),
            jnp_where(mk$both_gt_zero_mask, pos_val,
                      jnp_where(mk$one_zero_mask, if (kl) q else zero, sp_const(p, Inf))))
}, static_argnums = 2L), nondiff_argnums = 2L)

# Port of jax/_src/scipy/special.py:_rel_entr_jvp.
sp_rel_entr <- custom_jvp_defjvp(sp_rel_entr, function(kl, primals, tangents) {
  p <- primals[[1L]]
  q <- primals[[2L]]
  ans <- sp_rel_entr(p, q, kl)
  mk <- sp_rel_entr_masks(p, q)
  zero <- mk$zero
  safe_p <- mk$safe_p
  safe_q <- mk$safe_q
  lp <- sp_log_p_over_q(safe_p, safe_q)
  if (kl) {
    not_posinf_q <- lax_lt(safe_q, sp_const(p, Inf))
    finite_p <- jnp_where(not_posinf_q, safe_p, 1L)
    finite_q <- jnp_where(not_posinf_q, safe_q, 1L)
    rel_diff <- jnp_where(not_posinf_q, lax_div(lax_sub(finite_p, finite_q), finite_q),
                          sp_const(p, -1.0))
    dp_pos <- lp$log_p_over_q
    dq_pos <- lax_neg(rel_diff)
  } else {
    dp_pos <- lax_add(lp$log_p_over_q, sp_const(p, 1.0))
    dq_pos <- lax_neg(lp$ratio)
  }
  nan <- sp_const(p, NaN)
  dp <- jnp_where(mk$nan_mask, nan, jnp_where(mk$both_gt_zero_mask, dp_pos, zero))
  dq <- jnp_where(mk$nan_mask, nan,
                  jnp_where(mk$both_gt_zero_mask, dq_pos,
                            jnp_where(mk$one_zero_mask, sp_const(p, if (kl) 1.0 else 0.0), zero)))
  ans_dot <- lax_add(lax_mul(tangents[[1L]], dp), lax_mul(tangents[[2L]], dq))
  list(ans, jnp_astype(ans_dot, sp_dtype(ans)))
})

# Port of jax/_src/scipy/special.py:kl_div.
#' @rdname scipy_special
#' @export
scipy_special_kl_div <- function(p, q) {
  pq <- promote_args_inexact(p, q)
  if (is_complex_dtype(sp_dtype(pq[[1L]]))) sp_complex_error("kl_div")
  sp_rel_entr(pq[[1L]], pq[[2L]], TRUE)
}

# Port of jax/_src/scipy/special.py:rel_entr.
#' @rdname scipy_special
#' @export
scipy_special_rel_entr <- function(p, q) {
  pq <- promote_args_inexact(p, q)
  if (is_complex_dtype(sp_dtype(pq[[1L]]))) sp_complex_error("rel_entr")
  sp_rel_entr(pq[[1L]], pq[[2L]], FALSE)
}

# ---- logsumexp and softmax ---------------------------------------------------------------------

# Port of jax/_src/ops/special.py:logsumexp. The reductions over `where` follow
# jax/_src/numpy/reductions.py:_reduction (masked elements are replaced by the identity).
#' @rdname scipy_special
#' @export
scipy_special_logsumexp <- function(a, axis = NULL, b = NULL, keepdims = FALSE,
                                    return_sign = FALSE, where = NULL) {
  if (!is.null(where)) a <- jnp_where(where, a, 0L)
  if (!is.null(b)) {
    ab <- promote_args_inexact(a, b)
    b_arr <- ab[[2L]]
    a_arr <- jnp_where(b_arr != 0L, ab[[1L]], -Inf)
  } else {
    a_arr <- promote_args_inexact(a)[[1L]]
  }
  dt <- sp_dtype(a_arr)
  dims <- reduction_axes(a_arr, axis)
  a_real <- jnp_real(a_arr)
  if (!is.null(where)) a_real <- jnp_where(where, a_real, -Inf)
  amax <- jnp_max(a_real, axis = dims, keepdims = keepdims)
  amax <- lax_stop_gradient(lax_select(jnp_isfinite(amax), amax, lax_full_like(amax, 0)))
  amax_with_dims <- if (keepdims) amax else lax_expand_dims(amax, dims)
  exp_a <- lax_exp(lax_sub(a_arr, jnp_astype(amax_with_dims, dt)))
  if (!is.null(b)) exp_a <- lax_mul(exp_a, b_arr)
  if (!is.null(where)) exp_a <- jnp_where(where, exp_a, 0L)
  sumexp <- jnp_sum(exp_a, axis = dims, keepdims = keepdims)
  sign <- lax_sign(sumexp)
  if (return_sign || !is_complex_dtype(dt)) sumexp <- jnp_abs(sumexp)
  out <- lax_add(lax_log(sumexp), jnp_astype(amax, sp_dtype(sumexp)))
  if (return_sign) return(list(out, sign))
  if (!is.null(b) && !is_complex_dtype(sp_dtype(out)))
    out <- jnp_where(sign < 0L, as_jax(NaN, sp_dtype(out)), out)
  out
}

# Port of jax/_src/nn/functions.py:_softmax_deprecated (jax_softmax_custom_jvp is False in
# 0.12.0.1) with where = None, as called by jax/_src/scipy/special.py:softmax.
#' @rdname scipy_special
#' @export
scipy_special_softmax <- function(x, axis = NULL) {
  x <- as_jax(x)
  x_max <- jnp_max(x, axis, keepdims = TRUE)
  unnormalized <- jnp_exp(x - lax_stop_gradient(x_max))
  unnormalized / jnp_sum(unnormalized, axis, keepdims = TRUE)
}

# Port of jax/_src/nn/functions.py:log_softmax with where = None (jitted with a static
# axis, as in JAX).
sp_log_softmax <- jit(function(x, axis) {
  x <- as_jax(x)
  x_max <- jnp_max(x, axis, keepdims = TRUE)
  shifted <- x - lax_stop_gradient(x_max)
  shifted_logsumexp <- jnp_log(jnp_sum(jnp_exp(shifted), axis, keepdims = TRUE))
  shifted - shifted_logsumexp
}, static_argnums = 1L)

#' @rdname scipy_special
#' @export
scipy_special_log_softmax <- function(x, axis = NULL) sp_log_softmax(x, axis)

# ---- normal distribution -----------------------------------------------------------------------

sp_check_float_dtype <- function(dt, fmt = "x.dtype=%s is not supported, see docstring for supported types.") {
  if (!dt %in% c("float32", "float64")) jax_error(sprintf(fmt, dt), class = "jax_type_error")
}

# Port of jax/_src/lax/special.py:ndtr.
sp_ndtr <- function(x) {
  dt <- sp_dtype(x)
  typed <- function(v) as_jax(v, dt)
  half_sqrt_2 <- typed(0.5 * sqrt(2))
  w <- x * half_sqrt_2
  z <- lax_abs(w)
  y <- lax_select(lax_lt(z, half_sqrt_2), typed(1) + lax_erf(w),
                  lax_select(lax_gt(w, typed(0)), typed(2) - lax_erfc(z), lax_erfc(z)))
  typed(0.5) * y
}

# Port of jax/_src/scipy/special.py:ndtr.
#' @rdname scipy_special
#' @export
scipy_special_ndtr <- function(x) {
  x <- jnp_asarray(x)
  sp_check_float_dtype(sp_dtype(x))
  sp_ndtr(x)
}

# Port of jax/_src/scipy/special.py:_double_factorial, np.prod(np.arange(n, 1, -2)): an
# int64 product that wraps around for n >= 35, as numpy's does. Limbs are 16 bits.
sp_double_factorial <- function(n) {
  limbs <- c(1, 0, 0, 0)
  for (f in if (n >= 2) seq(n, 2, by = -2) else numeric(0)) {
    carry <- 0
    for (i in 1:4) {
      v <- limbs[i] * f + carry
      limbs[i] <- v %% 65536
      carry <- v %/% 65536
    }
  }
  hi <- limbs[4L] * 65536 + limbs[3L]
  if (hi >= 2^31) hi <- hi - 2^32
  hi * 2^32 + (limbs[2L] * 65536 + limbs[1L])
}

# Port of jax/_src/scipy/special.py:_log_ndtr_asymptotic_series.
sp_log_ndtr_asymptotic_series <- function(x, series_order) {
  dt <- sp_dtype(x)
  if (series_order <= 0L) return(as_jax(1, dt))
  x_2 <- lax_square(x)
  even_sum <- jnp_zeros_like(x)
  odd_sum <- jnp_zeros_like(x)
  x_2n <- x_2
  for (n in seq_len(series_order)) {
    y <- as_jax(sp_double_factorial(2L * n - 1L), dt) / x_2n
    if (n %% 2L) odd_sum <- odd_sum + y else even_sum <- even_sum + y
    x_2n <- x_2n * x_2
  }
  as_jax(1, dt) + even_sum - odd_sum
}

# Port of jax/_src/scipy/special.py:_log_ndtr_lower: asymptotic expansion of log(cdf(x))
# for x << -1.
sp_log_ndtr_lower <- function(x, series_order) {
  dt <- sp_dtype(x)
  x_2 <- lax_square(x)
  log_scale <- -as_jax(0.5, dt) * x_2 - lax_log(-x) - as_jax(0.5 * log(2 * pi), dt)
  log_scale + lax_log(sp_log_ndtr_asymptotic_series(x, series_order))
}

# Port of jax/_src/scipy/special.py:_norm_logpdf.
sp_norm_logpdf <- function(x) {
  lax_sub(lax_mul(sp_const(x, -0.5), lax_square(x)), sp_const(x, log(sqrt(2 * pi))))
}

# Port of jax/_src/scipy/special.py:log_ndtr. The idea comes from Cephes
# (https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html) with these changes:
# for x >> 1, log(P[X < x]) = log(1 - P[X < -x]) is approximated by -P[X < -x]; one fixed
# series_order is used for all x; max and min keep the unselected branches finite so the
# gradient of the select is not NaN at x = 0.
sp_log_ndtr <- custom_jvp(function(x, series_order) {
  x_arr <- jnp_asarray(x)
  dt <- sp_dtype(x_arr)
  if (!dt %in% c("float32", "float64"))
    jax_error("x.dtype=", dt, " is not supported.", class = "jax_type_error")
  # Lower bounds: where the support of ndtr appears to be zero relative to scipy, made
  # more conservative. Upper bounds: where log(cdf(x)) is 0 and -cdf(-x) takes over.
  lower_segment <- as_jax(if (dt == "float64") -20 else -10, dt)
  upper_segment <- as_jax(if (dt == "float64") 8 else 5, dt)
  x_arr_gt_upper_segment <- lax_gt(x_arr, upper_segment)
  ndtr_arg <- jnp_where(x_arr_gt_upper_segment, -x_arr, lax_max(x_arr, lower_segment))
  # Both ndtr calls are combined to minimize the program size.
  ndtr_res <- sp_ndtr(ndtr_arg)
  jnp_where(x_arr_gt_upper_segment, -ndtr_res,
            jnp_where(lax_gt(x_arr, lower_segment), lax_log(ndtr_res),
                      sp_log_ndtr_lower(lax_min(x_arr, lower_segment), series_order)))
}, nondiff_argnums = 1L)
sp_log_ndtr <- custom_jvp_defjvp(sp_log_ndtr, function(series_order, primals, tangents) {
  x <- primals[[1L]]
  ans <- sp_log_ndtr(x, series_order)
  list(ans, lax_mul(tangents[[1L]], lax_exp(lax_sub(sp_norm_logpdf(x), ans))))
})

#' @rdname scipy_special
#' @export
scipy_special_log_ndtr <- function(x, series_order = 3L) {
  if (is_jax_value(series_order) || !is.numeric(series_order) || length(series_order) != 1L ||
      series_order != round(series_order))
    jax_error("series_order must be a Python integer.", class = "jax_type_error")
  if (series_order < 0) jax_error("series_order must be non-negative.", class = "jax_value_error")
  if (series_order > 30) jax_error("series_order must be <= 30.", class = "jax_value_error")
  sp_log_ndtr(x, as.integer(series_order))
}

# ---- Bessel functions --------------------------------------------------------------------------

#' @rdname scipy_special
#' @export
scipy_special_i0e <- function(x) lax_bessel_i0e(promote_args_inexact(x)[[1L]])

#' @rdname scipy_special
#' @export
scipy_special_i1e <- function(x) lax_bessel_i1e(promote_args_inexact(x)[[1L]])

# Port of jax/_src/numpy/lax_numpy.py:_i1_maclaurin: the kth derivative of i1 near zero
# from a two term Maclaurin series.
sp_i1_maclaurin <- custom_jvp(function(k, x) {
  if (k %% 2L == 0L) {
    m <- (k + 1L) %/% 2L
    lax_mul(sp_const(x, choose(k + 1L, m) / 2^(k + 1L)), x)
  } else {
    lax_full_like(x, choose(k, k %/% 2L) / 2^k)
  }
}, nondiff_argnums = 0L)
sp_i1_maclaurin <- custom_jvp_defjvp(sp_i1_maclaurin, function(k, primals, tangents) {
  x <- primals[[1L]]
  list(sp_i1_maclaurin(k, x), lax_mul(sp_i1_maclaurin(k + 1L, x), tangents[[1L]]))
})

# Port of jax/_src/numpy/lax_numpy.py:_i1 and i1_impl.
sp_i1_raw <- function(x) lax_mul(lax_exp(lax_abs(x)), lax_bessel_i1e(x))
# A closed form JVP from the Bessel recurrence (i1'(x) = i0(x) - i1(x) / x) is not viable
# because of the 0 / 0 singularity at x = 0, and autodiff through exp(abs(x)) * i1e(x)
# fails for higher derivatives near 0 (cusp of abs, cancellation). A recursive two term
# Maclaurin series is used for |x| <= sqrt(eps), where its truncation error is below
# machine precision.
sp_i1_impl <- custom_jvp(sp_i1_raw, function(primals, tangents) {
  out <- jvp(sp_i1_raw, primals, tangents)
  x <- primals[[1L]]
  cutoff <- sqrt(finfo(sp_dtype(x))$eps)
  use_series <- lax_le(lax_abs(x), sp_const(x, cutoff))
  list(out[[1L]], jnp_where(use_series, lax_mul(sp_i1_maclaurin(1L, x), tangents[[1L]]), out[[2L]]))
})

# Port of jax/_src/numpy/lax_numpy.py:i0_impl.
sp_i0_impl <- custom_jvp(function(x) {
  abs_x <- lax_abs(x)
  lax_mul(lax_exp(abs_x), lax_bessel_i0e(abs_x))
}, function(primals, tangents) {
  x <- primals[[1L]]
  list(sp_i0_impl(x), lax_mul(tangents[[1L]], sp_i1_impl(x)))
})

# Port of jax/_src/scipy/special.py:i0.
#' @rdname scipy_special
#' @export
scipy_special_i0 <- function(x) {
  x <- promote_args_inexact(x)[[1L]]
  if (!is_float_dtype(sp_dtype(x)))
    jax_error("Unsupported input type to jax.scipy.special.i0: ", sp_dtype(x), class = "jax_value_error")
  sp_i0_impl(x)
}

# Port of jax/_src/scipy/special.py:i1.
#' @rdname scipy_special
#' @export
scipy_special_i1 <- function(x) {
  x <- promote_args_inexact(x)[[1L]]
  if (!is_float_dtype(sp_dtype(x)))
    jax_error("Unsupported input type to jax.scipy.special.i1: ", sp_dtype(x), class = "jax_value_error")
  sp_i1_impl(x)
}

# Port of jax/_src/scipy/special.py:_bessel_jn_scan_body_fun and _bessel_jn.
sp_bessel_jn <- function(z, v, n_iter) {
  body <- function(carry, k) {
    f0 <- carry[[1L]]
    f1 <- carry[[2L]]
    bs <- carry[[3L]]
    z <- carry[[4L]]
    f <- 2.0 * (k + 1.0) * f1 / z - f0
    bs <- lax_cond(jnp_remainder(k, 2L) == 0L, function(u) u[[1L]] + 2.0 * u[[2L]],
                   function(u) u[[1L]], list(bs, f))
    list(list(f1, f, bs, z), f)
  }
  res <- lax_scan(body, list(sp_const(z, 0.0), sp_const(z, 1E-16), sp_const(z, 0.0), z),
                  lax_iota(sp_dtype(z), n_iter + 1L), reverse = TRUE)
  bs <- res[[1L]][[3L]]
  j_vals <- res[[2L]]
  f <- lax_index_in_dim(j_vals, 0L, 0L, keepdims = FALSE)  # the value of the last iteration
  lax_slice_in_dim(j_vals, 0L, v + 1L) / (bs - f)
}

sp_bessel_jn_nd <- jit(function(z, v, n_iter) {
  fun <- function(z) sp_bessel_jn(z, v, n_iter)
  for (i in seq_along(sp_shape(z))) fun <- vmap(fun)
  jnp_moveaxis(fun(z), -1L, 0L)
}, static_argnums = c(1L, 2L))

# Port of jax/_src/scipy/special.py:bessel_jn.
#' @rdname scipy_special
#' @export
scipy_special_bessel_jn <- function(z, v, n_iter = 50L) {
  z <- promote_dtypes_inexact(list(jnp_asarray(z)))[[1L]]
  if (is_complex_dtype(sp_dtype(z))) jax_error("complex input not supported.", class = "jax_value_error")
  v <- as.integer(as.array(concrete_or_error(v, "Argument v of bessel_jn.")))
  n_iter <- as.integer(as.array(concrete_or_error(n_iter, "Argument n_iter of bessel_jn.")))
  sp_bessel_jn_nd(z, v, n_iter)
}

# ---- Dawson's integral and the Faddeeva function ------------------------------------------------

# Rational approximation coefficients of dawsn (Cody, Paciorek and Thacher 1970).
# Region 1, [0, 3.25): y = ax P(ax^2) / Q(ax^2).
sp_dawsn_an <- c(0x1.8ffb30f7d51f1p-37, 0x1.d2e309db8f5fbp-31,
                 0x1.4e09113ca0bap-26, 0x1.ffb82d857b833p-21,
                 0x1.9d29007c3bd6ap-19, 0x1.71a3163a27174p-12,
                 -0x1.bdb92e4e9a921p-11, 0x1.5a357714ced03p-5,
                 -0x1.77ccca3261da7p-4, 0x1p+0)
sp_dawsn_ad <- c(0x1.a6ddf536ed65ap-36, 0x1.993234f10c1b4p-30,
                 0x1.bfc372910659cp-25, 0x1.559b5c414f013p-20,
                 0x1.860dcd4d604a3p-16, 0x1.555661ec43b62p-12,
                 0x1.c92fcf5507532p-9, 0x1.c9d9358f736f7p-6,
                 0x1.455fdbcb2a008p-3, 0x1.265bbc0f09197p-1,
                 0x1p+0)
# Region 2, [3.25, 6.25): y = 0.5 / ax (1 + (1 / ax^2) P(1 / ax^2) / Q(1 / ax^2)).
sp_dawsn_bn <- c(0x1.0495c52fe411ep-1, -0x1.f541cdebcb905p-3,
                 0x1.81a4b94e413c5p-4, -0x1.66562633da983p-6,
                 0x1.dffee25eba9bdp-9, -0x1.bbc454e5479acp-12,
                 0x1.2db061d28d773p-15, -0x1.2015dd001fa5bp-19,
                 0x1.86d8be5016991p-24, -0x1.4a3b14d9709fp-29,
                 0x1.3bfc2ac32b39ep-35)
sp_dawsn_bd <- c(1.00000000000000000000e0, -0x1.438083f2d47c7p-1,
                 0x1.e4c6875173c3ep-3, -0x1.b3a7e0ed1122bp-5,
                 0x1.15e2e53c1fb6p-7, -0x1.f105f8f05c7d8p-11,
                 0x1.4795fc069cc34p-14, -0x1.31eec145c9b53p-18,
                 0x1.961705729c1cdp-23, -0x1.51a2c0f7cf15cp-28,
                 0x1.3bfc202a6b56p-34)
# Region 3, [6.25, inf): same form as region 2.
sp_dawsn_cn <- c(-0x1.2e622ffa7ef2p-1, 0x1.422b1f29fbcb6p-1,
                 -0x1.6203e2f0a174ep-3, 0x1.0e11ab3d4d36bp-6,
                 -0x1.fe79cad3d09fbp-12)
sp_dawsn_cd <- c(1.00000000000000000000e0, -0x1.595ea2e7576e2p+1,
                 0x1.bb92c0388a954p+0, -0x1.932857b438c94p-2,
                 0x1.1a0885fe44f2dp-5, -0x1.fe79cad3d0a8dp-11)

# Port of jax/_src/scipy/special.py:_dawsn_impl. The approximations work on |x|; odd
# symmetry restores the sign (sign(0) is 0 and so is dawsn(0)).
sp_dawsn_impl <- function(x) {
  sign <- jnp_sign(x)
  ax <- lax_abs(x)
  ax2 <- lax_square(ax)
  t <- sp_const(x, 1) / ax2
  # Each region is evaluated on safe dummy inputs: at extreme ax the polynomials overflow
  # and inf / inf = NaN is not suppressed by select on all backends.
  in_r1 <- ax < sp_const(x, 3.25)
  safe_ax_r1 <- lax_select(in_r1, ax, lax_full_like(ax, 1))
  safe_ax2_r1 <- lax_square(safe_ax_r1)
  val_r1 <- safe_ax_r1 * sp_polyval(sp_dawsn_an, safe_ax2_r1) / sp_polyval(sp_dawsn_ad, safe_ax2_r1)

  in_r2 <- (ax >= sp_const(x, 3.25)) & (ax < sp_const(x, 6.25))
  safe_t_r2 <- lax_select(in_r2, t, lax_full_like(t, 1))
  safe_ax_r2 <- lax_select(in_r2, ax, lax_full_like(ax, 1))
  val_r2 <- (sp_const(x, 0.5) / safe_ax_r2) *
    (sp_const(x, 1) + safe_t_r2 * sp_polyval(sp_dawsn_bn, safe_t_r2) / sp_polyval(sp_dawsn_bd, safe_t_r2))

  in_r3 <- ax >= sp_const(x, 6.25)
  safe_t_r3 <- lax_select(in_r3, t, lax_full_like(t, 1))
  safe_ax_r3 <- lax_select(in_r3, ax, lax_full_like(ax, 1))
  val_r3 <- (sp_const(x, 0.5) / safe_ax_r3) *
    (sp_const(x, 1) + safe_t_r3 * sp_polyval(sp_dawsn_cn, safe_t_r3) / sp_polyval(sp_dawsn_cd, safe_t_r3))

  result <- lax_select(in_r1, val_r1, lax_select(ax < sp_const(x, 6.25), val_r2, val_r3))
  sign * result
}

# Port of jax/_src/scipy/special.py:_dawsn (float16 and bfloat16 are upcast to float32).
sp_dawsn <- custom_jvp(function(x) {
  dt <- sp_dtype(x)
  if (dt %in% c("float32", "float64")) return(sp_dawsn_impl(x))
  jnp_astype(sp_dawsn_impl(jnp_astype(x, "float32")), dt)
})
sp_dawsn <- sp_defjvps(sp_dawsn, function(g, ans, x) g * (sp_const(x, 1) - sp_const(x, 2) * x * ans))

# Port of jax/_src/scipy/special.py:dawsn.
#' @rdname scipy_special
#' @export
scipy_special_dawsn <- function(x) {
  x <- promote_args_inexact(x)[[1L]]
  if (is_complex_dtype(sp_dtype(x))) sp_complex_error("dawsn")
  sp_dawsn(x)
}

# Weideman (1994) N = 32 rational approximation of the Faddeeva function, valid for
# Im(z) >= 0: w(z) = 2 polyval(c, Z) / (L - iz)^2 + 1 / (sqrt(pi) (L - iz)) with
# Z = (L + iz) / (L - iz). Coefficients in descending degree.
sp_wofz_l <- 0x1.306fe0a31b715p+2
sp_wofz_c <- c(
  -0x1.6ee2f02ea7549p-40, 0x1.07432d31f78bap-38, 0x1.1a8ba992fe08bp-37, -0x1.7affb84c8761bp-36,
  -0x1.e7ad4c6a5493ep-35, 0x1.005e1892e6855p-33, 0x1.c8b580b28b07bp-32, -0x1.1f94108a4a309p-31,
  -0x1.b8e7ff861f125p-29, 0x1.beab9c5e68fafp-31, 0x1.98d6c2ae23379p-26, 0x1.89f1228092803p-26,
  -0x1.3e1bc33ba34eep-23, -0x1.c14304016d996p-22, 0x1.c8f77358ddd3bp-22, 0x1.2761bd4637758p-18,
  0x1.c9c06fbeb9022p-18, -0x1.67319d424457bp-16, -0x1.1236489e0483ap-13, -0x1.013f37ab5851cp-12,
  0x1.9ba970e0e2563p-12, 0x1.2831513d8b0cap-8, 0x1.37659861309f3p-6, 0x1.d5700b627a5f7p-5,
  0x1.1ff6a5dc3a45p-3, 0x1.2e89017d37519p-2, 0x1.178f24b258e91p-1, 0x1.cdc92d8080504p-1,
  0x1.5875952a2f948p+0, 0x1.d35f15b8926abp+0, 0x1.21bb971a2c848p+1, 0x1.493f998475b46p+1)

# Port of jax/_src/scipy/special.py:_wofz_upper.
sp_wofz_upper <- function(z) {
  re <- lax_real(z)
  L <- lax_complex(sp_const(re, sp_wofz_l), sp_const(re, 0))
  iz <- lax_complex(lax_neg(lax_imag(z)), re)
  denom <- L - iz
  Z <- (L + iz) / denom
  p <- sp_polyval(sp_wofz_c, Z)
  one_over_sqrtpi <- lax_complex(sp_const(re, 1 / sqrt(pi)), sp_const(re, 0))
  2L * p / (denom * denom) + one_over_sqrtpi / denom
}

# Port of jax/_src/scipy/special.py:_wofz.
sp_wofz <- custom_jvp(function(z) {
  sign <- lax_ge(lax_imag(z), sp_const(lax_real(z), 0))
  z_upper <- lax_select(sign, z, lax_neg(z))
  w_upper <- sp_wofz_upper(z_upper)
  correction <- 2L * lax_exp(lax_neg(z * z)) - w_upper
  lax_select(sign, w_upper, correction)
})
sp_wofz <- sp_defjvps(sp_wofz, function(g, ans, z) {
  re <- lax_real(z)
  g * (lax_neg(2L * z * ans) + lax_complex(sp_const(re, 0), sp_const(re, 2 / sqrt(pi))))
})

# Port of jax/_src/numpy/util.py:promote_args_complex for one argument.
sp_to_complex <- function(z) {
  z <- as_jax(z)
  a <- aval_of(z)
  inexact <- to_inexact_dtype(a$dtype)
  target <- if (inexact %in% c("float64", "complex128")) "complex128" else "complex64"
  lax_convert_element_type(z, target, weak_type = a$weak)
}

# Port of jax/_src/scipy/special.py:wofz.
#' @rdname scipy_special
#' @export
scipy_special_wofz <- function(z) sp_wofz(sp_to_complex(z))

# ---- exponential integrals (Cephes ei.c and expn.c) ---------------------------------------------

sp_euler_gamma <- 0x1.2788cfc6fb619p-1

# Port of jax/_src/scipy/special.py:_expint1 (0 < x <= 2).
sp_expint1 <- function(x) {
  A <- c(-0x1.566dbac131912p+2, 0x1.b50284772bde1p+7, -0x1.0509287cfdf6p+12,
         0x1.b0e788fe23e1cp+15, -0x1.439288850094ep+18, 0x1.84d3329d39885p+20)
  B <- c(1.0, -0x1.a40b38e2458b1p+5, 0x1.3ae76f9ab45f9p+10, -0x1.1275fbb82b302p+14,
         0x1.239d1b18fe99bp+17, -0x1.6432dd911d0fbp+19, 0x1.84d3329d39885p+20)
  f <- sp_polyval(A, x) / sp_polyval(B, x)
  x * f + sp_euler_gamma + jnp_log(x)
}

# Port of jax/_src/scipy/special.py:_eval_expint_k and _expint2 to _expint7.
sp_expint_k <- function(A, B) {
  force(A); force(B)
  function(x) {
    one <- sp_const(x, 1.0)
    w <- one / x
    f <- sp_polyval(A, w) / sp_polyval(B, w)
    f <- w * f + one
    jnp_exp(x) * w * f
  }
}
sp_expint_branches <- list(
  sp_expint1,
  # 2 <= x < 4
  sp_expint_k(c(0x1.44b312c56600cp-6, -0x1.458a9157b4e84p+0, -0x1.0b48d67f4712bp+1,
                0x1.60b5aeb6d1394p+1, -0x1.c38894e2430e9p-2, 0x1.7e353913d19bfp-5,
                -0x1.95060f1ed8415p-10, 0x1.281df6eb22cfcp-14),
              c(1.0, 0x1.79fbd1202ac41p+0, 0x1.2036c045d9b45p-1, 0x1.5bf578355e449p-3,
                0x1.7776a638453dp-6, 0x1.23a52299d590fp-8, 0x1.6a45353fafaa3p-13,
                0x1.4b9d9e7d80dcap-15)),
  # 4 <= x <= 8
  sp_expint_k(c(-0x1.5f8b0af9bb2f3p+0, -0x1.6ababbc532f19p-1, 0x1.94afc211defa1p+0,
                -0x1.0a64c5d6877b9p-2, 0x1.eaa5c2aef61fp-6, -0x1.1020c65f4722ap-10,
                0x1.6eabe4c82940ep-15, 0x1.1f77207a80cb5p-19),
              c(1.0, 0x1.b79058669c108p-1, 0x1.cb16a5db254cdp-2, 0x1.3ae5cfbee1be7p-4,
                0x1.9162ebefce877p-6, 0x1.cf0f531947054p-11, 0x1.e1656ff967816p-12,
                -0x1.3d6a2d0abf0abp-18, 0x1.65b76bde87f9dp-19)),
  # 8 <= x <= 16
  sp_expint_k(c(-0x1.0db0087489edcp+1, 0x1.bb947266e1a0fp+0, -0x1.f05b71f930b44p-3,
                0x1.7c8e1cfca4b28p-6, 0x1.f1a10c720ff6fp-13, -0x1.5df0a34d3247dp-14,
                0x1.c97bec7bcd44ap-17, -0x1.887f35747f1d8p-22, 0x1.f759a33c7c119p-27,
                0x1.538d3685d5ea6p-31),
              c(1.0, -0x1.d6a4a543a3e24p-3, 0x1.c4a3b9f44e68bp-4, -0x1.00a993060d9c5p-6,
                0x1.69e75f4002e4fp-9, -0x1.b62047a548d9cp-13, 0x1.1ea4c2589a8d9p-16,
                -0x1.ded0a119c2a52p-22, 0x1.df3162c367742p-27, 0x1.5225be05c9f6cp-31)),
  # 16 <= x <= 32
  sp_expint_k(c(-0x1.f76c3fab0f2dep-3, -0x1.2fcbf3946359dp-3, 0x1.28e3d92253f09p-4,
                -0x1.b9d0eae8f0cfdp-7, 0x1.60001c0ae19fdp-10, -0x1.4d21934d9c9e3p-14,
                0x1.62e5510432f5p-19, -0x1.6c2b02fd8be1p-25),
              c(1.0, -0x1.abb705ca3e639p-4, -0x1.12119ce5c9f9ep-2, 0x1.8d1b722487b45p-4,
                -0x1.066e156ab1f4fp-6, 0x1.8846b4369dd8bp-10, -0x1.62f0e47bc2ee7p-14,
                0x1.6e459c0c81e5bp-19, -0x1.6c2ac3c905a0ep-25)),
  # 32 <= x <= 64
  sp_expint_k(c(0x1.f0aa3faaa9a0bp-4, -0x1.2a24f8a0d965bp-1, 0x1.e10d57de68c95p-3,
                -0x1.f21447f0024ddp-6, 0x1.8bdbe5bd803a5p-10, -0x1.a74fe9bfef728p-16),
              c(1.0, -0x1.009398d02ac1ap+0, 0x1.2be65e0f94ecbp-2, -0x1.115e0da06387dp-5,
                0x1.991664e9dd0f9p-10, -0x1.a74fe9ba98fe2p-16)),
  # x > 64
  sp_expint_k(c(-0x1.8814eee7cd369p-1, 0x1.6092b383c3b2ep-1, -0x1.b4c18f5a8ce4ap-3,
                0x1.121cf6220b5d2p-5, -0x1.933f9a0f59f7p-9, 0x1.6e65c10b5f9f9p-13,
                -0x1.999cf7de777b4p-18, 0x1.05920f36cec09p-23, -0x1.2a89cfba332e3p-30),
              c(1.0, -0x1.e38895b57edf9p+0, 0x1.1112b4d77cf1bp+0, -0x1.19cbd38a23643p-2,
                0x1.4203f4ad3c0b3p-5, -0x1.bf91cd02bdaccp-9, 0x1.878171a913d92p-13,
                -0x1.a9d0c797e644dp-18, 0x1.0a3c3675b657cp-23, -0x1.2a89cfba32fcbp-30)))

# Port of jax/_src/scipy/special.py:_expi_pos (x >= 0).
sp_expi_pos <- function(x) {
  cc <- function(v) sp_const(x, v)
  conds <- c(list((cc(0) <= x) & (x <= cc(2))),
             lapply(1:5, function(i) (cc(2^i) < x) & (x <= cc(2^(i + 1)))))
  sp_piecewise(x, conds, sp_expint_branches)
}

# Port of jax/_src/scipy/special.py:expi.
sp_expi <- custom_jvp(jit(function(x) {
  x_arr <- promote_args_inexact(x)[[1L]]
  if (is_complex_dtype(sp_dtype(x_arr))) sp_complex_error("expi")
  sp_piecewise(x_arr, list(x_arr < 0L),
               list(function(x) -scipy_special_exp1(-x), sp_expi_pos))
}), function(primals, tangents) {
  x <- primals[[1L]]
  list(sp_expi(x), jnp_exp(x) / x * tangents[[1L]])
})

#' @rdname scipy_special
#' @export
scipy_special_expi <- function(x) sp_expi(x)

# Port of jax/_src/lax/control_flow/loops.py:fori_loop for traced bounds of equal dtype
# (a while loop; the bounds of the callers below are floating point counters).
sp_fori_while <- function(lower, upper, body_fun, init_val) {
  lax_while_loop(function(c) lax_lt(c[[1L]], c[[2L]]),
                 function(c) list(lax_add(c[[1L]], sp_const(c[[1L]], 1)), c[[2L]],
                                  body_fun(c[[1L]], c[[3L]])),
                 list(lower, upper, init_val))[[3L]]
}

sp_maxlog <- function(x) sp_const(x, log(finfo(sp_dtype(x))$max))

# Port of jax/_src/scipy/special.py:_expn1: 0 < x <= 1 and 1 <= n < 5000 (power series,
# DLMF 8.19.8).
sp_expn1 <- function(x, n) {
  cc <- sp_const
  MACHEP <- finfo(sp_dtype(x))$eps
  zero <- cc(x, 0.0)
  one <- cc(x, 1.0)
  in_domain <- (x > zero) & (x <= one) & (n >= one) & (n < cc(n, 5000))
  x_safe <- jnp_where(in_domain, x, one)
  n_safe <- jnp_where(in_domain, n, one)
  psi <- -sp_euler_gamma - jnp_log(x_safe)
  psi <- sp_fori_while(cc(n, 1), n_safe, function(i, psi) psi + one / i, psi)
  n1 <- jnp_where(n_safe == cc(n, 1), one + one, n_safe)
  init <- list(x = x, n = n, z = -x_safe, xk = zero, yk = one, pk = one - n_safe,
               ans = jnp_where(n_safe == cc(n, 1), zero, one / (one - n1)), t = cc(x, Inf))
  body <- function(d) {
    d$xk <- d$xk + one
    d$yk <- d$yk * (d$z / d$xk)
    d$pk <- d$pk + one
    d$ans <- d$ans + jnp_where(d$pk != zero, d$yk / d$pk, zero)
    d$t <- jnp_where(d$ans != zero, abs(d$yk / d$ans), one)
    d
  }
  # piecewise evaluates every branch, so the loop is bound to its domain.
  cond <- function(d) {
    (d$x > zero) & (d$x <= one) & (d$n >= one) & (d$n < cc(d$n, 5000)) & (d$t > MACHEP)
  }
  d <- lax_while_loop(cond, body, init)
  r <- n_safe - one
  sign <- jnp_where(r %% cc(r, 2) == zero, one, -one)
  log_term <- sign * jnp_exp(r * jnp_log(x_safe) - scipy_special_gammaln(n_safe)) * psi
  log_term - d$ans
}

# Port of jax/_src/scipy/special.py:_expn2: x > 1 and 1 <= n < 5000 (continued fraction,
# DLMF 8.19.17).
sp_expn2 <- function(x, n) {
  cc <- sp_const
  BIG <- cc(x, 0x1p+57)
  MACHEP <- finfo(sp_dtype(x))$eps
  MAXLOG <- sp_maxlog(x)
  zero <- cc(x, 0.0)
  one <- cc(x, 1.0)
  in_domain <- (x > one) & (x <= MAXLOG) & (n >= one) & (n < cc(n, 5000))
  x_safe <- jnp_where(in_domain, x, one + one)
  n_safe <- jnp_where(in_domain, n, one)
  init <- list(k = cc(n, 1), pkm2 = one, qkm2 = x_safe, pkm1 = one, qkm1 = x_safe + n_safe,
               ans = one / (x_safe + n_safe), t = cc(x, Inf), r = zero, x = x, n = n)
  body <- function(d) {
    x <- d$x
    n <- d$n
    d$k <- d$k + cc(d$k, 1)
    k <- d$k
    odd <- k %% cc(k, 2) == cc(k, 1)
    yk <- jnp_where(odd, one, x)
    xk <- jnp_where(odd, n + (k - cc(k, 1)) / cc(k, 2), k / cc(k, 2))
    pk <- d$pkm1 * yk + d$pkm2 * xk
    qk <- d$qkm1 * yk + d$qkm2 * xk
    nz <- qk != zero
    r <- jnp_where(nz, pk / qk, d$r)
    d$r <- r
    d$t <- jnp_where(nz, abs((d$ans - r) / r), one)
    d$ans <- jnp_where(nz, r, d$ans)
    d$pkm2 <- d$pkm1
    d$pkm1 <- pk
    d$qkm2 <- d$qkm1
    d$qkm1 <- qk
    is_big <- abs(pk) > BIG
    for (key in c("pkm1", "pkm2", "qkm1", "qkm2")) d[[key]] <- jnp_where(is_big, d[[key]] / BIG, d[[key]])
    d
  }
  cond <- function(d) {
    (d$x > one) & (d$x <= MAXLOG) & (d$n >= one) & (d$n < cc(d$n, 5000)) & (d$t > MACHEP)
  }
  d <- lax_while_loop(cond, body, init)
  d$ans * jnp_exp(-x_safe)
}

# Port of jax/_src/scipy/special.py:_expn3: n >= 5000 and 0 < x <= MAXLOG.
sp_expn3 <- function(x, n) {
  cc <- sp_const
  zero <- cc(x, 0.0)
  one <- cc(x, 1.0)
  MAXLOG <- sp_maxlog(x)
  in_domain <- (x > zero) & (x <= MAXLOG) & (n >= cc(n, 5000))
  x_safe <- jnp_where(in_domain, x, one)
  n_safe <- jnp_where(in_domain, n, cc(n, 5000))
  xk <- x_safe + n_safe
  lam <- x_safe / n_safe
  m <- (n_safe / xk) / xk
  ans <- cc(x, 6) * lam * lam - cc(x, 8) * lam + one
  ans <- m * ans + (one - cc(x, 2) * lam)
  ans <- m * ans + one
  ans <- m * ans + one
  ans * jnp_exp(-x_safe) / xk
}

# Port of jax/_src/scipy/special.py:_expn.
sp_expn <- custom_jvp(jit(sp_vectorize(function(n, x) {
  cc <- sp_const
  zero <- cc(x, 0)
  one <- cc(x, 1)
  MAXLOG <- sp_maxlog(x)
  # Non integer orders are truncated (as in JAX) rather than marked invalid.
  n <- jnp_floor(n)
  invalid <- (n < cc(n, 0)) | (x < zero) | jnp_isnan(n) | jnp_isnan(x) | jnp_isinf(n)
  valid <- !invalid
  conds <- list(
    invalid,
    valid & (x == zero) & (n < cc(n, 2)),
    valid & (x == zero) & (n >= cc(n, 2)),
    valid & (n == cc(n, 0)) & (x > zero),
    valid & (n > cc(n, 0)) & (x > MAXLOG),
    valid & (n >= cc(n, 5000)) & (x > zero) & (x <= MAXLOG),
    valid & (n > cc(n, 0)) & (n < cc(n, 5000)) & (x > one) & (x <= MAXLOG))
  n_minus_one <- jnp_where(valid & (n >= cc(n, 2)), n - one, one)
  x_safe <- jnp_where(valid & (x > zero), x, one)
  vals <- list(NaN, Inf,
               one / n_minus_one,  # safe denominator for n >= 2, guarded for n < 2
               jnp_exp(-x_safe) / x_safe, zero, sp_expn3, sp_expn2, sp_expn1)
  sp_piecewise(x, conds, vals, n = n)
})))

# Port of jax/_src/scipy/special.py:_expn_jvp.
sp_expn <- custom_jvp_defjvp(sp_expn, function(primals, tangents) {
  n <- primals[[1L]]
  x <- primals[[2L]]
  cc <- sp_const
  zero <- cc(x, 0)
  one <- cc(x, 1)
  MAXLOG <- sp_maxlog(x)
  n_trunc <- jnp_floor(n)
  primal_out <- sp_expn(n_trunc, x)
  invalid <- (n_trunc < cc(n_trunc, 0)) | (x < zero) | jnp_isnan(n_trunc) | jnp_isnan(x) |
    jnp_isinf(n_trunc)
  x_safe <- jnp_where((x > zero) & (x <= MAXLOG), x, one)
  deriv_n0 <- jnp_where(x == zero, cc(x, -Inf),
                        jnp_where(x > MAXLOG, zero,
                                  -(x_safe + one) * jnp_exp(-x_safe) / (x_safe * x_safe)))
  n_prev <- jnp_where(n_trunc >= cc(n_trunc, 1), n_trunc - cc(n_trunc, 1), cc(n_trunc, 0))
  deriv_n_pos <- -sp_expn(n_prev, x)
  deriv <- jnp_where(invalid, cc(x, NaN),
                     jnp_where(n_trunc == cc(n_trunc, 0), deriv_n0, deriv_n_pos))
  list(primal_out, lax_mul(tangents[[2L]], deriv))
})

# Port of jax/_src/scipy/special.py:expn.
#' @rdname scipy_special
#' @export
scipy_special_expn <- function(n, x) {
  nx <- promote_args_inexact(n, x)
  if (is_complex_dtype(sp_dtype(nx[[2L]]))) sp_complex_error("expn")
  sp_expn(nx[[1L]], nx[[2L]])
}

# Port of jax/_src/scipy/special.py:exp1.
#' @rdname scipy_special
#' @export
scipy_special_exp1 <- function(x) {
  x <- promote_args_inexact(x)[[1L]]
  if (is_complex_dtype(sp_dtype(x))) sp_complex_error("exp1")
  scipy_special_expn(1L, x)
}

# ---- sine and cosine integrals (Cephes sici.c) ---------------------------------------------------

# Port of jax/_src/numpy/ufuncs.py:sinc for real input: exact range reduction x = n + r,
# then sinc(r) = 1 + t Q(t) with t = r^2 and the Taylor coefficients of Q.
sp_sinc <- jit(function(x) {
  x <- promote_dtypes_inexact(list(x))[[1L]]
  orig_dtype <- sp_dtype(x)
  if (finfo(orig_dtype)$bits < 32L) x <- lax_convert_element_type(x, "float32")
  # x = n + r with n an integer and r in [-0.5, 0.5]; r is 0 when |x| is inf.
  n <- lax_stop_gradient(lax_round(x, 1L))
  is_inf <- lax_abs(x) == Inf
  r <- jnp_where(is_inf, 0L, x - n)
  t <- r * r
  # c_k = (-1)^k pi^(2k) / (2k + 1)!, low to high.
  q_coeffs <- if (sp_dtype(x) == "float64") {
    c(-0x1.a51a6625307d3p0, 0x1.9f9cb402bc46cp-1, -0x1.86a8e4720db67p-3, 0x1.ac6805cf350a6p-6,
      -0x1.33816aa4607abp-9, 0x1.374719fab3915p-13, -0x1.d42498d1ce099p-18,
      0x1.0fc992ff39e13p-22, -0x1.f5f9d970ca6dfp-28, 0x1.79788684225eap-33,
      -0x1.d7353939082fep-39)
  } else {
    c(-0x1.a51a66p0, 0x1.9f9cb4p-1, -0x1.86a8e4p-3, 0x1.ac6806p-6, -0x1.33816ap-9, 0x1.37471ap-13)
  }
  q <- sp_polynomial(t, as.list(q_coeffs))
  tq <- t * q
  sinc_sin <- 1L + tq
  num <- r + r * tq
  # Divide by (-1)^n x (by 1 when n == 0, avoiding 0 / 0 in autodiff).
  is_odd <- 2L * lax_round(0.5 * n, 1L) != n
  eq_n_zero <- n == 0L
  denom <- jnp_where(is_odd, -x, jnp_where(eq_n_zero, 1L, x))
  signed_scaled <- num / denom
  # At r == 0 the quotient may be -0.0; y - stop_gradient(y) is +0.0 with the derivatives
  # (sinc'(n) = (-1)^n / n) preserved.
  signed_scaled <- jnp_where(r == 0L, signed_scaled - lax_stop_gradient(signed_scaled), signed_scaled)
  result <- jnp_where(eq_n_zero, sinc_sin, signed_scaled)
  if (orig_dtype != sp_dtype(x)) result <- lax_convert_element_type(result, orig_dtype)
  result
})

# Port of jax/_src/scipy/special.py:_sici_series (0 <= x <= 4).
sp_sici_series <- function(x) {
  SN <- c(-0x1.7111e5de8a74ep-34, 0x1.8d5cfe68b2ba6p-25, -0x1.476916027de9dp-17,
          0x1.0019b2d3e26b7p-10, -0x1.52b702936e3f5p-5, 0x1p+0)
  SD <- c(0x1.1e1378a86806fp-39, 0x1.5fd682336d66bp-30, 0x1.da68b670ce50ap-22,
          0x1.a1ecefdb29b28p-14, 0x1.d195bcceb8cd2p-7, 0x1p+0)
  CN <- c(0x1.6448b3c15e40fp-36, -0x1.d0b6b8ff3e02ep-27, 0x1.e24725c6665bp-19,
          -0x1.f1085739b6027p-12, 0x1.d9c2594824532p-6, -0x1p+0)
  CD <- c(0x1.1eecff1ec7123p-38, 0x1.a5a2f7ba5686ap-29, 0x1.4abd82ff8d229p-20,
          0x1.4cdcb1d62892cp-12, 0x1.a1d0a2c50caf6p-5, 0x1p+2)
  t <- x * x
  si <- (x * sp_polyval(SN, t)) / sp_polyval(SD, t)
  ci <- sp_euler_gamma + jnp_log(x) + t * sp_polyval(CN, t) / sp_polyval(CD, t)
  list(jnp_where(x == 0L, 0.0, si), jnp_where(x == 0L, -Inf, ci))
}

# Port of jax/_src/scipy/special.py:_sici_asympt (4 < x <= 1e9).
sp_sici_asympt <- function(x) {
  s <- jnp_sin(x)
  c <- jnp_cos(x)
  z <- 1.0 / (x * x)
  FN4 <- c(0x1.0f1cbb40b7ae7p+2, 0x1.5d666f82283efp+2, 0x1.9eeee747100a9p+0,
           0x1.560790004019cp-3, 0x1.be5037f807e63p-8, 0x1.c8e9c3475f584p-14,
           0x1.26b045b354025p-21)
  FD4 <- c(1, 0x1.0547677e81fe5p+3, 0x1.d3bafe8916c6dp+2, 0x1.de302c966b9d7p+0,
           0x1.6e2a87221767fp-3, 0x1.cbdf83b78dc2dp-8, 0x1.cd847dc9514edp-14,
           0x1.26b046bb8da45p-21)
  GN4 <- c(0x1.64c325d2f5beap-4, 0x1.3906aec5679d1p-1, 0x1.96b66e7e7b929p-2,
           0x1.3298d27538956p-4, 0x1.612729a31eaecp-8, 0x1.53bcfa3b7bb66p-13,
           0x1.09b3de8512006p-19, 0x1.0ce46161a46dap-27)
  GD4 <- c(1, 0x1.a4dea09b8ead8p+0, 0x1.5524d75ac416ap-1, 0x1.95003bdca232dp-4,
           0x1.97e4c77cfc311p-8, 0x1.6b455c2837a3p-13, 0x1.10012a2ca2e74p-19,
           0x1.0ce46564d5a26p-27)
  FN8 <- c(0x1.d2d26f89b5d2fp-2, 0x1.6d6c16a1d4e48p-1, 0x1.484b7308de2ecp-3,
           0x1.7c51bb7a279a4p-7, 0x1.6e8957d091226p-12, 0x1.464b2ccabac1ap-18,
           0x1.12f5262091ae1p-25, 0x1.9e32f425e0304p-34, 0x1.b513da12e7a9dp-44)
  FD8 <- c(1.0, 0x1.d5bdca6088f72p-1, 0x1.6df2afebc0e46p-3, 0x1.9099bf9175badp-7,
           0x1.781eda8abe36fp-12, 0x1.4a77b32a6a5a8p-18, 0x1.148f14631660ap-25,
           0x1.9f0d7e12e9a4p-34, 0x1.b513da12e7a9dp-44)
  GN8 <- c(0x1.650c5d23286cap-1, 0x1.525741796d975p-2, 0x1.3b4aed7ff0c8p-5,
           0x1.c22628d5922dfp-10, 0x1.24b68b71a83bep-15, 0x1.74bab04aea33dp-22,
           0x1.d4677933ba2cep-30, 0x1.0f95d648214f3p-38, 0x1.c49458c0363a9p-49)
  GD8 <- c(1.0, 0x1.af7c34d1b37a3p+0, 0x1.f38f8ae181179p-2, 0x1.7f5082557a75ep-5,
           0x1.f2d1b7948715ep-10, 0x1.3519831592239p-15, 0x1.7f5f4b4039bedp-22,
           0x1.dab26c23c716fp-30, 0x1.10e9458ab177cp-38, 0x1.c49458c0363a9p-49)
  f4 <- sp_polyval(FN4, z) / (x * sp_polyval(FD4, z))
  g4 <- z * sp_polyval(GN4, z) / sp_polyval(GD4, z)
  f8 <- sp_polyval(FN8, z) / (x * sp_polyval(FD8, z))
  g8 <- z * sp_polyval(GN8, z) / sp_polyval(GD8, z)
  mask <- x < 8.0
  f <- jnp_where(mask, f4, f8)
  g <- jnp_where(mask, g4, g8)
  list((pi / 2) - f * c - g * s, f * s - g * c)
}

# Port of jax/_src/scipy/special.py:_sici_approx (x >= 1e9).
sp_sici_approx <- function(x) {
  si <- (pi / 2) - jnp_cos(x) / x
  ci <- jnp_sin(x) / x
  posinf <- x == Inf
  list(jnp_where(posinf, pi / 2, si), jnp_where(posinf, 0.0, ci))
}

# Port of jax/_src/scipy/special.py:sici.
sp_sici <- custom_jvp(jit(function(x) {
  x <- promote_args_inexact(x)[[1L]]
  if (is_complex_dtype(sp_dtype(x)))
    jax_error("Argument `x` to sici must be real-valued. Got dtype ", sp_dtype(x), ".",
              class = "jax_value_error")
  x_abs <- jnp_abs(x)
  series <- sp_sici_series(x_abs)
  asymp <- sp_sici_asympt(x_abs)
  approx <- sp_sici_approx(x_abs)
  conds <- list(x_abs <= 4L, (x_abs > 4L) & (x_abs <= 1e9))
  si <- sp_select(conds, list(series[[1L]], asymp[[1L]]), approx[[1L]])
  ci <- sp_select(conds, list(series[[2L]], asymp[[2L]]), approx[[2L]])
  list(jnp_sign(x) * si, jnp_where(x == -Inf, NaN, ci))
}), function(primals, tangents) {
  p <- primals[[1L]]
  t <- tangents[[1L]]
  list(sp_sici(p), list(sp_sinc(p / pi) * t, jnp_cos(p) / p * t))
})

#' @rdname scipy_special
#' @export
scipy_special_sici <- function(x) sp_sici(x)

# ---- Fresnel integrals (port of jax/_src/third_party/scipy/special.py) --------------------------

# Port of sincospisquaredhalf: accurate sin(pi x^2 / 2) and cos(pi x^2 / 2), following the
# sinpi and cospi functions of SciPy. With s = x %% 2, (x^2 / 2) %% 2 = (s (x - s / 2)) %% 2.
sp_sincospisquaredhalf <- jit(function(x) {
  x <- jnp_abs(x)
  fmod <- function(a, b) { ab <- promote_args_inexact(a, b); lax_rem(ab[[1L]], ab[[2L]]) }
  s <- fmod(x, 2.0)
  r <- fmod(s * (x - s / 2L), 2.0)
  sinpi <- jnp_where(r < 0.5, jnp_sin(pi * r),
                     jnp_where(r > 1.5, jnp_sin(pi * (r - 2.0)), -jnp_sin(pi * (r - 1.0))))
  cospi <- jnp_where(r == 0.5, 0.0,
                     jnp_where(r < 1.0, -jnp_sin(pi * (r - 0.5)), jnp_sin(pi * (r - 1.5))))
  list(sinpi, cospi)
})

# Cephes coefficients: fresnlf (single precision) and fresnl (double precision).
sp_fresnel_coeffs <- list(
  single = list(
    sn = c(+0x1.c4e59dc1b40f9p-30, -0x1.470253ed0a6b6p-23, +8.424748808502400e-6,
           -0x1.473a77dbbb21fp-12, +0x1.daca5c3983716p-8, -0x1.79fb2e0ba56c5p-4,
           +0x1.0c15237166da5p-1),
    cn = c(+0x1.e6cf441bccf2bp-27, -0x1.36a4564ce65e8p-20, +0x1.c3e9be1af38c4p-15,
           -0x1.a4943d6cf70eap-10, +0x1.cdc7ff97c9d4cp-6, -0x1.f952ba0663443p-3,
           +0x1.ffffff31d86dbp-1),
    fn = c(-0x1.bb1439f8a1cacp+40, +0x1.f920be3550674p+36, -0x1.efb0755905a62p+31,
           +0x1.18254528b225ap+26, -0x1.aa6275618646ep+19, +8.560515466275470e3,
           -0x1.9d26aa9606fc1p+6, +0x1.7fec665614f2fp+1),
    gn = c(-1.860843997624650e11, +0x1.7cfa6a96f74e6p+33, -0x1.686e35351fa21p+28,
           +0x1.8c470e484b113p+22, -0x1.3035e56f9062ap+16, +0x1.ae2585ec049ddp+9,
           -0x1.dde68e2ca6fcap+3, +0x1.fffded9efad13p-1)),
  double = list(
    sn = c(-0x1.75fa36d65b5c3p+11, +0x1.5a1d0172c05b6p+19, -0x1.e074624f6f5edp+25,
           +0x1.2fda860b77afep+31, -0x1.4a0ba347bf087p+35, +0x1.282cfe6e52457p+38),
    sd = c(+1.00000000000000000000e0, +0x1.1960532871072p+8, +0x1.64218fe9cdaa9p+15,
           +0x1.3bc2fb8d017b4p+22, +0x1.8fe51b5e5ea9ep+28, +0x1.4e664e6b222e9p+34,
           +0x1.1ad3b73de42b5p+39),
    cn = c(-0x1.ac80c2cfb62d3p-25, +0x1.3ee9262100f74p-17, -0x1.524420ed68786p-11,
           +0x1.3566a1059cc13p-6, -0x1.a4eac378a969p-3, +0x1p+0),
    cd = c(+0x1.197686a7fc6b3p-38, +0x1.f7449db0375b9p-31, +0x1.0c70802a45191p-23,
           +0x1.9a3eefa2a1d32p-17, +0x1.c718f3fdbd99dp-11, +0x1.51a076030fe74p-5,
           +0x1p+0),
    fn = c(+0x1.afa91d1b02391p-2, +0x1.25b30d2b90e5ep-3, +0x1.798e57b157dadp-7,
           +0x1.69c70c4950efcp-12, +0x1.37203c5357795p-18, +0x1.067b5484e87a1p-25,
           +0x1.c1f0a4998a7bbp-34, +0x1.835571459fdf7p-43, +0x1.35a2daf8fa9e4p-53,
           +0x1.636ef0e0b05f6p-65),
    fd = c(+1.00000000000000000000e0, +0x1.80cfeeb217a65p-1, +0x1.dec6ec1d46d7dp-4,
           +0x1.a615ea6ed56f3p-8, +0x1.4704a6017bd6p-13, +0x1.ef9b1923264f5p-20,
           +0x1.833b495525916p-27, +0x1.3cc8533d3a061p-35, +0x1.0926c6b9f079p-44,
           +0x1.a0e5d2352f48dp-55, +0x1.d9e9412b96141p-67),
    gn = c(+0x1.02463b420cacap-1, +0x1.93aaa67f87481p-3, +0x1.3371854ba3214p-6,
           +0x1.66a7948d1b33bp-11, +0x1.82577f9faabf3p-17, +0x1.a621c4dea6091p-24,
           +0x1.e9a94f200eb09p-32, +0x1.30bf5876689cbp-40, +0x1.8c7a03df8a964p-50,
           +0x1.edb24f17358a6p-61, +0x1.c409d624fbe2bp-73),
    gd = c(+1.00000000000000000000e0, +0x1.7996d23bc55c1p+0, +0x1.59dadefa1c34fp-2,
           +0x1.9f811e637aa09p-6, +0x1.ab2060fa0ae28p-11, +0x1.abf862c0a067bp-17,
           +0x1.c0071ab1c7428p-24, +0x1.fa8618e3cf1c6p-32, +0x1.366439c3def2bp-40,
           +0x1.900dc37c81936p-50, +0x1.ef5a184ff8364p-61, +0x1.c409d624fbe2bp-73)))

# Port of jax/_src/third_party/scipy/special.py:fresnel, a translation of SciPy's fresnl.h
# (double precision) and of Cephes fresnlf (single precision). float16 and bfloat16 are
# upcast to float32. As in Cephes, accuracy is only guaranteed on [-10, 10].
sp_fresnel <- custom_jvp(function(x) {
  xxa <- promote_args_inexact(x)[[1L]]
  original_dtype <- sp_dtype(xxa)
  if (is_complex_dtype(original_dtype))
    jax_error("Support for complex-valued inputs is not implemented yet.",
              class = "jax_not_implemented_error")
  if (original_dtype %in% c("float32", "float16", "bfloat16")) {
    xxa <- jnp_astype(xxa, "float32")
    co <- sp_fresnel_coeffs$single
  } else if (original_dtype == "float64") {
    co <- sp_fresnel_coeffs$double
  } else {
    jax_error("Support for ", original_dtype, " dtype is not implemented yet.",
              class = "jax_not_implemented_error")
  }
  single_precision <- sp_dtype(xxa) == "float32"
  x <- jnp_abs(xxa)
  x2 <- x * x
  s_inf <- c_inf <- 0.5

  # Small x
  t <- x2 * x2
  if (single_precision) {
    s_small <- x * x2 * sp_polyval(co$sn, t)
    c_small <- x * sp_polyval(co$cn, t)
  } else {
    s_small <- x * x2 * sp_polyval(co$sn, t) / sp_polyval(co$sd, t)
    c_small <- x * sp_polyval(co$cn, t) / sp_polyval(co$cd, t)
  }

  # Large x
  sc <- sp_sincospisquaredhalf(x)
  sinpi <- sc[[1L]]
  cospi <- sc[[2L]]
  if (single_precision) {
    c_large <- c_inf
    s_large <- s_inf
  } else {
    c_large <- 0.5 + 1L / (pi * x) * sinpi
    s_large <- 0.5 - 1L / (pi * x) * cospi
  }

  # Other x
  t <- pi * x2
  u <- 1.0 / (t * t)
  t <- 1.0 / t
  if (single_precision) {
    f <- 1.0 - u * sp_polyval(co$fn, u)
    g <- t * sp_polyval(co$gn, u)
  } else {
    f <- 1.0 - u * sp_polyval(co$fn, u) / sp_polyval(co$fd, u)
    g <- t * sp_polyval(co$gn, u) / sp_polyval(co$gd, u)
  }
  t <- pi * x
  c_other <- 0.5 + (f * sinpi - g * cospi) / t
  s_other <- 0.5 - (f * cospi + g * sinpi) / t

  isinf <- jnp_isinf(xxa)
  small <- x2 < 2.5625
  large <- x > 36974.0
  s <- jnp_where(isinf, s_inf, jnp_where(small, s_small, jnp_where(large, s_large, s_other)))
  c <- jnp_where(isinf, c_inf, jnp_where(small, c_small, jnp_where(large, c_large, c_other)))
  neg <- xxa < 0.0
  s <- jnp_where(neg, -s, s)
  c <- jnp_where(neg, -c, c)
  if (original_dtype != sp_dtype(xxa)) {
    s <- jnp_astype(s, original_dtype)
    c <- jnp_astype(c, original_dtype)
  }
  list(s, c)
}, function(primals, tangents) {
  x <- primals[[1L]]
  sc <- sp_sincospisquaredhalf(x)
  list(sp_fresnel(x), list(sc[[1L]] * tangents[[1L]], sc[[2L]] * tangents[[1L]]))
})

#' @rdname scipy_special
#' @export
scipy_special_fresnel <- function(x) sp_fresnel(x)

# ---- Owen's T function ---------------------------------------------------------------------------

# 13 point Gauss type quadrature on T(h, a) = (1 / 2 pi) int_0^a exp(-h^2 (1 + u^2) / 2) /
# (1 + u^2) du under u^2 = a^2 t, with 1 / (2 pi) and the Jacobian absorbed into the
# weights (Patefield and Tandy 2000, method T5).
sp_owens_t_quad_pts <- c(
  0x1.cbd3ca9e124fbp-9, 0x1.003ce7fc8b9a9p-5, 0x1.5d40bf651f4bfp-4, 0x1.4cb2f63066a9cp-3,
  0x1.08b75bffbd376p-2, 0x1.78e8cb3fa9863p-2, 0x1.f0a6b479fe41dp-2, 0x1.349ef17f64bbp-1,
  0x1.6df77e0e0ced2p-1, 0x1.a12794ee6ea8fp-1, 0x1.cb520a79d87e1p-1, 0x1.ea1b1c178de15p-1,
  0x1.fbcbae11a12ecp-1)
sp_owens_t_quad_wts <- c(
  0x1.3488c6b5e2b5fp-6, 0x1.3034010ccb167p-6, 0x1.279a065782553p-6, 0x1.1ad9bfe40b74ap-6,
  0x1.0a2100a507f3bp-6, 0x1.eb57c133a264dp-7, 0x1.bb87ca176de85p-7, 0x1.857df1b537732p-7,
  0x1.49fc727543262p-7, 0x1.09d93c0892d45p-7, 0x1.8bf646ff86b4dp-8, 0x1.fd5ff638e49dbp-9,
  0x1.b8381fb93992cp-10)

# Port of jax/_src/scipy/special.py:_owens_t_quadrature. The nodes and weights are float64
# arrays in JAX, so they take the default float dtype here as well.
sp_owens_t_quadrature <- function(h, a) {
  nd <- length(sp_shape(a))
  quad_pts <- lax_expand_dims(as_jax(array(sp_owens_t_quad_pts)), seq_len(nd) - 1L)
  r <- jnp_expand_dims(jnp_square(a), -1L) * quad_pts
  integrand <- jnp_exp(-0.5 * jnp_expand_dims(jnp_square(h), -1L) * (1. + r)) / (1. + r)
  a * jnp_matmul(integrand, as_jax(array(sp_owens_t_quad_wts)))
}

# Port of jax/_src/scipy/special.py:_owens_t_impl and _owens_t_jvp.
sp_owens_t <- custom_jvp(function(h, a) {
  h <- jnp_abs(h)
  abs_a <- jnp_abs(a)
  root_2 <- sp_const(h, sqrt(2))
  h_normed <- h / root_2
  modified_a <- jnp_where(abs_a <= 1., abs_a, jnp_reciprocal(abs_a))
  modified_h <- jnp_where(abs_a <= 1., h, abs_a * h)
  result <- sp_owens_t_quadrature(modified_h, modified_a)
  # Exact values for h = 0 and a = 1.
  result <- jnp_where(modified_h == 0., jnp_arctan(modified_a) / (2 * pi), result)
  result <- jnp_where(modified_a == 1.,
                      0.125 * lax_erfc(-modified_h / root_2) * lax_erfc(modified_h / root_2),
                      result)
  # Reciprocal correction for |a| > 1.
  normh <- lax_erfc(h_normed)
  normah <- lax_erfc(abs_a * h_normed)
  result <- jnp_where(
    abs_a > 1.,
    jnp_where(abs_a * h <= 0.67,
              (0.25 - 0.25 * lax_erf(h_normed) * lax_erf(abs_a * h_normed) - result),
              0.25 * (normh + normah - normh * normah) - result),
    result)
  result <- lax_sign(a) * result
  jnp_where(jnp_isnan(a) | jnp_isnan(h), jnp_full_like(result, NaN), result)
}, function(primals, tangents) {
  h <- primals[[1L]]
  a <- primals[[2L]]
  result <- sp_owens_t(h, a)
  root_2 <- sp_const(h, sqrt(2))
  # dT/dh = -exp(-h^2 / 2) erf(a h / sqrt(2)) / (2 sqrt(2 pi))
  dout_dh <- -lax_exp(-0.5 * lax_square(h)) * lax_erf(a * h / root_2) /
    (2. * sp_const(h, sqrt(2. * pi)))
  # dT/da = exp(-(a^2 + 1) h^2 / 2) / (2 pi (a^2 + 1))
  dout_da <- lax_exp(-0.5 * (lax_square(a) + 1.) * lax_square(h)) /
    (2. * pi * (lax_square(a) + 1.))
  list(result, jnp_astype(dout_dh * tangents[[1L]] + dout_da * tangents[[2L]], sp_dtype(result)))
})

# Port of jax/_src/scipy/special.py:owens_t.
#' @rdname scipy_special
#' @export
scipy_special_owens_t <- function(h, a) {
  ha <- promote_args_inexact(h, a)
  sp_owens_t(ha[[1L]], ha[[2L]])
}

# ---- Spence's function (Cephes spence.c) ----------------------------------------------------------

# Port of jax/_src/scipy/special.py:_spence_poly.
sp_spence_poly <- function(w) {
  A <- c(0x1.862d9a70a1bffp-15, 0x1.df74463beda34p-8, 0x1.121eb61515c68p-3,
         0x1.c266e64d1f41bp-1, 0x1.5b12621c05b6ap+1, 0x1.107238d7a3427p+2,
         0x1.a61b791772b23p+1, 0x1p+0)
  B <- c(0x1.6a472c69ef90ep-11, 0x1.a039ad7232021p-6, 0x1.21c4296f8343fp-2,
         0x1.6966dfa85ad29p+0, 0x1.d1aa28a4ec717p+1, 0x1.421936337526fp+2,
         0x1.c61b791772b23p+1, 0x1p+0)
  -w * sp_polyval(A, w) / sp_polyval(B, w)
}

# Port of jax/_src/scipy/special.py:_spence_calc.
sp_spence_calc <- function(x) {
  x2_bool <- x > 2.0
  x <- sp_piecewise(x, list(x2_bool), list(function(x) 1.0 / x, function(x) x))
  x1_5_bool <- x > 1.5
  x_5_bool <- x < 0.5
  x2_bool <- x2_bool | x1_5_bool
  w <- sp_piecewise(x, list(x1_5_bool, x_5_bool),
                    list(function(x) 1.0 / x - 1.0, function(x) -x, function(x) x - 1.0))
  y <- sp_spence_poly(w)
  y_flag_one <- pi^2 / 6.0 - jnp_log(x) * jnp_log(1.0 - x) - y
  y <- jnp_where(x_5_bool, y_flag_one, y)
  y_flag_two <- -0.5 * jnp_log(x)^2 - y
  jnp_where(x2_bool, y_flag_two, y)
}

# Port of jax/_src/scipy/special.py:_spence.
sp_spence <- jit(function(x) {
  sp_piecewise(x, list(x < 0.0, x == 1.0, x == 0.0), list(NaN, 0, pi^2 / 6, sp_spence_calc))
})

# Port of jax/_src/scipy/special.py:spence.
#' @rdname scipy_special
#' @export
scipy_special_spence <- function(x) {
  x <- jnp_asarray(x)
  sp_check_float_dtype(sp_dtype(x))
  sp_spence(x)
}

# ---- Bernoulli numbers ----------------------------------------------------------------------------

# Port of jax/_src/scipy/special.py:bernoulli (Chowla and Hartung algorithm).
#' @rdname scipy_special
#' @export
scipy_special_bernoulli <- function(n) {
  n <- as.integer(as.array(concrete_or_error(n, "Argument n of bernoulli")))
  if (n < 0L) jax_error("n must be a non-negative integer.", class = "jax_value_error")
  b3 <- as_jax(array(c(1, -1 / 2, 1 / 6)))
  if (n < 3L) return(lax_slice_in_dim(b3, 0L, n + 1L))
  dt <- sp_dtype(b3)
  n_even <- (n - 2L) %/% 2L
  if (n_even == 0L) return(lax_concatenate(list(b3, jnp_zeros(n - 2L, dt)), 0L))
  m <- as_jax(array(seq(4, n, by = 2)), dt)
  terms <- -(m - 1L) * m / 4L / pi^2
  q1 <- (1. / pi^2) * jnp_cumprod(terms)
  k <- sp_arange(2L, 50L, dt)  # 2^-50 < 1e-15
  q2 <- jnp_sum(jnp_power(lax_expand_dims(k, 1L), -lax_expand_dims(m, 0L)), axis = 0L)
  vals <- q1 * (1L + q2)
  # Odd indices above 1 are zero; the even ones hold `vals`.
  tail <- jnp_reshape(jnp_stack(list(jnp_zeros(n_even, dt), vals), axis = 1L), -1L)
  parts <- list(b3, tail)
  if (n %% 2L == 1L) parts <- c(parts, list(jnp_zeros(1L, dt)))
  lax_concatenate(parts, 0L)
}

# ---- erfcx, erfcinv and ndtri (built on the erf.py approximations) ----------------------------------

# Port of jax/_src/scipy/special.py:_erfcx_grad and _erfcx_grad_jvp. Letting AD differentiate
# the polynomial pieces would give slopes of polynomials, with jumps where the pieces meet.
# erfcx''(x) = 2 erfcx(x) + 2 x erfcx'(x) has mild cancellation for x <= 2.25 and is used
# there; for larger x AD differentiates erfcx_grad_large, whose error is smaller than the
# cancellation error of the formula. `ans` is the cached primal erfcx(x), not an independent
# input, so only the tangent of x matters. As in JAX, the select against the scalar threshold
# restricts this second order rule to scalar x (use vmap for arrays).
sp_erfcx_grad <- custom_jvp(function(x, ans) sp_erfcx_grad_impl(x, ans), function(primals, tangents) {
  x <- primals[[1L]]
  ans <- primals[[2L]]
  x_dot <- tangents[[1L]]
  grad_ans <- sp_erfcx_grad(x, ans)
  threshold <- sp_const(x, 2.25)
  is_large <- x > threshold
  zeros <- lax_full_like(x, 0.0)
  x_small <- lax_select(is_large, zeros, x)
  ans_small <- lax_select(is_large, zeros, ans)
  grad_small <- lax_select(is_large, zeros, grad_ans)
  small_dot <- x_dot * (2.0 * ans_small + 2.0 * x_small * grad_small)
  large_dot <- jvp(sp_erfcx_grad_large, list(lax_select(is_large, x, threshold)), list(x_dot))[[2L]]
  list(grad_ans, lax_select(is_large, large_dot, small_dot))
})

# Port of jax/_src/scipy/special.py:_erfcx and _erfcx_jvp.
sp_erfcx <- custom_jvp(function(x) sp_erfcx_impl(x), function(primals, tangents) {
  x <- primals[[1L]]
  ans <- sp_erfcx(x)
  list(ans, tangents[[1L]] * sp_erfcx_grad(x, ans))
})

# Port of jax/_src/scipy/special.py:erfcx.
#' @rdname scipy_special
#' @export
scipy_special_erfcx <- function(x) {
  x <- promote_args_inexact(x)[[1L]]
  dt <- sp_dtype(x)
  if (is_complex_dtype(dt)) return(sp_wofz(lax_complex(lax_neg(lax_imag(x)), lax_real(x))))
  if (finfo(dt)$bits < 32L) return(jnp_astype(sp_erfcx(jnp_astype(x, "float32")), dt))
  sp_erfcx(x)
}

# Port of jax/_src/scipy/special.py:_erfcinv_grad: erfcinv'(y) = -sqrt(pi) / 2 exp(z^2) with
# z = erfcinv(y). For |z| > 1 the identity -sqrt(pi) / 2 erfcx(|z|) / min(y, 2 - y) avoids
# amplifying the rounding error of z by 2 z^2.
sp_erfcinv_grad <- custom_jvp(function(y, z) {
  q <- jnp_where(y > 1.0, 2.0 - y, y)
  u <- lax_abs(z)
  is_tail <- u > 1.0
  z_central <- jnp_where(is_tail, 0.0, z)
  central <- -0.5 * sqrt(pi) * lax_exp(lax_square(z_central))
  is_finite_tail <- is_tail & (q > 0.0)
  u_tail <- jnp_where(is_finite_tail, u, 1.0)
  q_safe <- jnp_where(is_finite_tail, q, 1.0)
  tail_num <- -0.5 * sqrt(pi) * sp_erfcx_large(u_tail)
  tail <- jnp_where(q == 0.0, -Inf, tail_num / q_safe)
  jnp_where(is_tail, tail, central)
}, function(primals, tangents) {
  # z is the cached primal: d/dy (-sqrt(pi) / 2 exp(z^2)) = 2 z erfcinv'(y)^2.
  y <- primals[[1L]]
  z <- primals[[2L]]
  g <- sp_erfcinv_grad(y, z)
  list(g, tangents[[1L]] * ((2.0 * z * g) * g))
})

# Port of jax/_src/scipy/special.py:_erfcinv, _erfcinv_impl and _erfcinv_jvp.
sp_erfcinv <- custom_jvp(jit(function(y) {
  out_of_bounds <- (y < 0.0) | (y > 2.0) | jnp_isnan(y)
  is_edge <- (y == 0.0) | (y == 2.0)
  y_safe <- jnp_where(out_of_bounds | is_edge, 1.0, y)
  q_safe <- jnp_where(y_safe > 1.0, 2.0 - y_safe, y_safe)
  x_safe <- 1.0 - y_safe
  z <- sp_erf_inv_core(x_safe, q_safe, has_far_tail = TRUE)
  jnp_where(out_of_bounds, NaN, jnp_where(y == 0.0, Inf, jnp_where(y == 2.0, -Inf, z)))
}), function(primals, tangents) {
  y <- primals[[1L]]
  z <- sp_erfcinv(y)
  list(z, tangents[[1L]] * sp_erfcinv_grad(y, z))
})

# Port of jax/_src/scipy/special.py:erfcinv.
#' @rdname scipy_special
#' @export
scipy_special_erfcinv <- function(y) {
  y <- promote_args_inexact(y)[[1L]]
  dt <- sp_dtype(y)
  if (is_complex_dtype(dt)) sp_complex_error("erfcinv")
  if (finfo(dt)$bits < 32L) return(jnp_astype(sp_erfcinv(jnp_astype(y, "float32")), dt))
  sp_erfcinv(y)
}

# Port of jax/_src/scipy/special.py:_ndtri_grad: ndtri'(p) = sqrt(2 pi) exp(x^2 / 2) with
# x = ndtri(p). For |x| > sqrt(2) the identity sqrt(pi / 2) erfcx(|x| / sqrt(2)) /
# min(p, 1 - p) avoids amplifying the rounding error of x by x^2.
sp_ndtri_grad <- custom_jvp(function(p, x) {
  p_arr <- jnp_asarray(p, dtype = sp_dtype(x))
  a <- jnp_where(p_arr > 0.5, 1.0 - p_arr, p_arr)
  u <- lax_abs(x) * sqrt(0.5)
  is_tail <- u > 1.0
  x_central <- jnp_where(is_tail, 0.0, x)
  central <- sqrt(2.0 * pi) * lax_exp(0.5 * lax_square(x_central))
  is_finite_tail <- is_tail & (a > 0.0)
  u_tail <- jnp_where(is_finite_tail, u, 1.0)
  a_safe <- jnp_where(is_finite_tail, a, 0.5)
  tail_num <- sqrt(0.5 * pi) * sp_erfcx_large(u_tail)
  tail <- jnp_where(a == 0.0, Inf, tail_num / a_safe)
  jnp_where(is_tail, tail, central)
}, function(primals, tangents) {
  # x is the cached primal: d/dp (sqrt(2 pi) exp(x^2 / 2)) = x ndtri'(p)^2.
  p <- primals[[1L]]
  x <- primals[[2L]]
  g <- sp_ndtri_grad(p, x)
  list(g, tangents[[1L]] * ((x * g) * g))
})

# Port of jax/_src/scipy/special.py:_ndtri and _ndtri_jvp.
sp_ndtri <- custom_jvp(function(p) {
  p_arr <- jnp_asarray(p)
  out_of_bounds <- (p_arr < 0L) | (p_arr > 1L) | jnp_isnan(p_arr)
  is_edge <- (p_arr == 0L) | (p_arr == 1L)
  p_safe <- jnp_where(out_of_bounds | is_edge, 0.5, p_arr)
  a_safe <- jnp_where(p_safe > 0.5, 1L - p_safe, p_safe)
  x_safe <- 2L * (p_safe - 0.5)
  q_safe <- 2L * a_safe
  x <- sp_erf_inv_core(x_safe, q_safe, has_far_tail = TRUE, is_ndtri = TRUE)
  jnp_where(out_of_bounds, NaN, jnp_where(p_arr == 0L, -Inf, jnp_where(p_arr == 1L, Inf, x)))
}, function(primals, tangents) {
  p <- primals[[1L]]
  x <- sp_ndtri(p)
  list(x, tangents[[1L]] * sp_ndtri_grad(p, x))
})

# Port of jax/_src/scipy/special.py:ndtri.
#' @rdname scipy_special
#' @export
scipy_special_ndtri <- function(p) {
  sp_check_float_dtype(sp_dtype(as_jax(p)))
  sp_ndtri(p)
}

# ---- confluent and Gauss hypergeometric functions --------------------------------------------------
# The kernels are scalar functions lifted with sp_vectorize, as in JAX (jnp.vectorize): every
# element runs its own series.

sp_series_cond <- function(precision) {
  function(state) (state[[2L]] < 250L) & (lax_abs(state[[3L]]) / lax_abs(state[[1L]]) > precision)
}

# Port of jax/_src/scipy/special.py:_hyp1f1_serie: the power series of 1F1 (carry: serie, k,
# term).
sp_hyp1f1_serie <- function(a, b, x) {
  body <- function(state) {
    serie <- state[[1L]] + state[[3L]]
    k <- state[[2L]]
    term <- state[[3L]] * ((a + k) / (b + k) * x / (k + 1L))
    list(serie, k + 1L, term)
  }
  lax_while_loop(sp_series_cond(finfo(sp_dtype(x))$eps), body,
                 list(sp_const(x, 1), sp_int(1L), a / b * x))[[1L]]
}

# Port of jax/_src/scipy/special.py:_hyp1f1_asymptotic: the asymptotic expansion of 1F1.
sp_hyp1f1_asymptotic <- function(a, b, x) {
  body <- function(state) {
    serie <- state[[1L]] + state[[3L]]
    k <- state[[2L]]
    term <- state[[3L]] * ((b - a + k) * (1L - a + k) / (k + 1L) / x)
    list(serie, k + 1L, term)
  }
  serie <- lax_while_loop(sp_series_cond(finfo(sp_dtype(x))$eps), body,
                          list(sp_const(x, 1), sp_int(1L), (b - a) * (1L - a) / x))[[1L]]
  scipy_special_gamma(b) / scipy_special_gamma(a) * lax_exp(x) * jnp_power(x, a - b) * serie
}

# Ports of jax/_src/scipy/special.py:_hyp1f1_a_derivative and _hyp1f1_b_derivative: the
# series differentiated term by term in a or b.
sp_hyp1f1_param_derivative <- function(weight) {
  jit(sp_vectorize(function(a, b, x) {
    body <- function(state) {
      k <- state[[2L]]
      serie <- state[[1L]] + state[[3L]] * weight(a, b, k)
      term <- state[[3L]] * ((a + k) / (b + k) * x / (k + 1L))
      list(serie, k + 1L, term)
    }
    lax_while_loop(sp_series_cond(finfo(sp_dtype(x))$eps), body,
                   list(sp_const(x, 0), sp_int(1L), a / b * x))[[1L]]
  }))
}
sp_hyp1f1_a_derivative <- sp_hyp1f1_param_derivative(function(a, b, k) {
  scipy_special_digamma(a + k) - scipy_special_digamma(a)
})
sp_hyp1f1_b_derivative <- sp_hyp1f1_param_derivative(function(a, b, k) {
  scipy_special_digamma(b) - scipy_special_digamma(b + k)
})

# Port of jax/_src/scipy/special.py:hyp1f1 (https://doi.org/10.48550/arXiv.1407.7786).
sp_hyp1f1 <- custom_jvp(jit(sp_vectorize(function(a, b, x) {
  abx <- promote_args_inexact(a, b, x)
  a <- abx[[1L]]
  b <- abx[[2L]]
  x <- abx[[3L]]
  dt <- sp_dtype(x)
  if (is_complex_dtype(dt)) sp_complex_error("hyp1f1")
  result <- lax_cond(lax_abs(x) < 100L, sp_hyp1f1_serie, sp_hyp1f1_asymptotic, a, b, x)
  index <- (a == 0L) * 1L + ((a == b) & (a != 0L)) * 2L + ((b == 0L) & (a != 0L)) * 3L
  lax_select_n(lax_convert_element_type(index, "int32"), result, as_jax(1, dt), jnp_exp(x),
               as_jax(Inf, dt))
})))
sp_hyp1f1 <- sp_defjvps(
  sp_hyp1f1,
  function(a_dot, primal_out, a, b, x) sp_hyp1f1_a_derivative(a, b, x) * a_dot,
  function(b_dot, primal_out, a, b, x) sp_hyp1f1_b_derivative(a, b, x) * b_dot,
  function(x_dot, primal_out, a, b, x) a / b * sp_hyp1f1(a + 1L, b + 1L, x) * x_dot)

#' @rdname scipy_special
#' @export
scipy_special_hyp1f1 <- function(a, b, x) sp_hyp1f1(a, b, x)

# Port of jax/_src/scipy/special.py:_hyp2f1_terminal: the finite sum when a or b is a
# negative integer. Of a and b, the negative integer with the larger magnitude (that still
# is smaller in magnitude than a non positive c) bounds the loop.
sp_hyp2f1_terminal <- function(a, b, c, x) {
  dt <- sp_dtype(x)
  eps <- finfo(dt)$eps * 50
  ib <- jnp_round(b)
  mask <- jnp_logical_and(
    b < a,
    jnp_logical_and(jnp_abs(b - ib) < eps,
                    jnp_logical_not(jnp_logical_and(c %% 1L == 0L, jnp_logical_and(c <= 0L, c > b)))))
  orig_a <- a
  a <- jnp_where(mask, b, a)
  b <- jnp_where(mask, orig_a, b)
  a <- jnp_abs(a)
  body <- function(i, state) {
    term <- state[[2L]] * (-(a - i + 1L) / (c + i - 1L) * (b + i - 1L) / i * x)
    list(state[[1L]] + term, term)
  }
  sp_fori_while(as_jax(1, sp_dtype(a)), a + 1L, body, list(as_jax(1, dt), as_jax(1, dt)))[[1L]]
}

# Port of jax/_src/scipy/special.py:_hyp2f1_serie: the Gauss series (carry: serie, k, term).
sp_hyp2f1_serie <- function(a, b, c, x) {
  dt <- sp_dtype(x)
  rtol <- finfo(dt)$eps
  body <- function(state) {
    k <- state[[2L]]
    serie <- state[[1L]] + state[[3L]]
    term <- state[[3L]] * ((a + k - 1L) * (b + k - 1L) / (c + k - 1L) / k * x)
    list(serie, k + 1L, term)
  }
  cond <- function(state) (state[[2L]] < 250L) & (lax_abs(state[[3L]]) > rtol * lax_abs(state[[1L]]))
  lax_while_loop(cond, body, list(as_jax(0, dt), as_jax(1, dt), as_jax(1, dt)))[[1L]]
}

# Port of jax/_src/scipy/special.py:_hyp2f1_digamma_transform: the transformation for
# x > 0.9 when c - a - b is an integer (carry of the loop: a, ax, b, d1, e, p, q, r, s, t, y).
sp_hyp2f1_digamma_transform <- function(a, b, c, x) {
  dt <- sp_dtype(x)
  rtol <- finfo(dt)$eps
  psi <- scipy_special_digamma
  gam <- scipy_special_gamma
  d <- c - a - b
  s <- 1L - x
  rd <- jnp_round(d)
  e <- jnp_where(rd >= 0L, d, -d)
  d1 <- jnp_where(rd >= 0L, d, as_jax(0, sp_dtype(d)))
  d2 <- jnp_where(rd >= 0L, as_jax(0, sp_dtype(d)), d)
  ard <- jnp_astype(jnp_where(rd >= 0L, rd, -rd), "int32")
  ax <- jnp_log(s)
  y <- psi(1.0) + psi(1.0 + e) - psi(a + d1) - psi(b + d1) - ax
  y <- y / gam(e + 1.0)
  p <- (a + d1) * (b + d1) * s / gam(e + 2.0)
  cond <- function(state) {
    jnp_logical_and(state[[10L]] < 250L, jnp_abs(state[[7L]]) >= rtol * jnp_abs(state[[11L]]))
  }
  body <- function(state) {
    a <- state[[1L]]; ax <- state[[2L]]; b <- state[[3L]]; d1 <- state[[4L]]; e <- state[[5L]]
    p <- state[[6L]]; s <- state[[9L]]; t <- state[[10L]]; y <- state[[11L]]
    r <- psi(1.0 + t) + psi(1.0 + t + e) - psi(a + t + d1) - psi(b + t + d1) - ax
    q <- p * r
    y <- y + q
    p <- p * (s * (a + t + d1) / (t + 1.0))
    p <- p * ((b + t + d1) / (t + 1.0 + e))
    t <- t + 1.0
    list(a, ax, b, d1, e, p, q, r, s, t, y)
  }
  y <- lax_while_loop(cond, body,
                      list(a, ax, b, d1, e, p, y, as_jax(0, dt), s, as_jax(1, dt), y))[[11L]]
  compute_sum <- function(y) {
    for_body <- function(i, state) {
      a <- state[[1L]]; b <- state[[2L]]; d2 <- state[[3L]]; e <- state[[4L]]; p <- state[[5L]]
      s <- state[[6L]]; t <- state[[7L]]; y1 <- state[[8L]]
      r <- 1.0 - e + t
      p <- p * (s * (a + t + d2) * (b + t + d2) / r)
      t <- t + 1.0
      p <- p / t
      list(a, b, d2, e, p, s, t, y1 + p)
    }
    y1 <- sp_fori_while(as_jax(1L, "int32"), ard, for_body,
                        list(a, b, d2, e, as_jax(1, dt), s, as_jax(0, dt), as_jax(1, dt)))[[8L]]
    p <- gam(c)
    y1 <- y1 * (gam(e) * p / (gam(a + d1) * gam(b + d1)))
    y <- y * (p / (gam(a + d2) * gam(b + d2)))
    y <- jnp_where(lax_bitwise_and(ard, lax_full_like(ard, 1)) != 0L, -y, y)
    q <- jnp_power(s, rd)
    jnp_where(rd > 0L, y * q + y1, y + y1 * q)
  }
  jnp_where(rd == 0L, y * gam(c) / (gam(a) * gam(b)), compute_sum(y))
}

# Port of jax/_src/scipy/special.py:_hyp2f1_terminal_or_serie.
sp_hyp2f1_terminal_or_serie <- function(a, b, c, x) {
  eps <- finfo(sp_dtype(x))$eps * 50
  d <- c - a - b
  neg_int_a <- jnp_logical_and(a <= 0L, jnp_abs(a - jnp_round(a)) < eps)
  neg_int_b <- jnp_logical_and(b <= 0L, jnp_abs(b - jnp_round(b)) < eps)
  neg_int_a_or_b <- jnp_logical_or(neg_int_a, neg_int_b)
  index <- jnp_where(jnp_logical_and(x > 0.9, jnp_logical_not(neg_int_a_or_b)),
                     jnp_where(jnp_abs(d - jnp_round(d)) >= eps, 0L, 1L),
                     jnp_where(neg_int_a_or_b, 2L, 0L))
  lax_select_n(lax_convert_element_type(index, "int32"), sp_hyp2f1_serie(a, b, c, x),
               sp_hyp2f1_digamma_transform(a, b, c, x), sp_hyp2f1_terminal(a, b, c, x))
}

# Ports of jax/_src/scipy/special.py:_hyp2f1_a_derivative, _hyp2f1_b_derivative and
# _hyp2f1_c_derivative: the series differentiated term by term in a, b or c.
sp_hyp2f1_param_derivative <- function(weight) {
  jit(sp_vectorize(function(a, b, c, x) {
    body <- function(state) {
      k <- state[[2L]]
      serie <- state[[1L]] + state[[3L]] * weight(a, b, c, k)
      term <- state[[3L]] * ((a + k) * (b + k) / (c + k) / (k + 1L) * x)
      list(serie, k + 1L, term)
    }
    lax_while_loop(sp_series_cond(finfo(sp_dtype(x))$eps), body,
                   list(sp_const(x, 0), sp_int(1L), a * b / c * x))[[1L]]
  }))
}
sp_hyp2f1_a_derivative <- sp_hyp2f1_param_derivative(function(a, b, c, k) {
  scipy_special_digamma(a + k) - scipy_special_digamma(a)
})
sp_hyp2f1_b_derivative <- sp_hyp2f1_param_derivative(function(a, b, c, k) {
  scipy_special_digamma(b + k) - scipy_special_digamma(b)
})
sp_hyp2f1_c_derivative <- sp_hyp2f1_param_derivative(function(a, b, c, k) {
  scipy_special_digamma(c) - scipy_special_digamma(c + k)
})

# Port of jax/_src/scipy/special.py:hyp2f1 (https://doi.org/10.48550/arXiv.1407.7786).
sp_hyp2f1 <- custom_jvp(jit(sp_vectorize(function(a, b, c, x) {
  abcx <- promote_args_inexact(a, b, c, x)
  a <- abcx[[1L]]
  b <- abcx[[2L]]
  c <- abcx[[3L]]
  x <- abcx[[4L]]
  dt <- sp_dtype(x)
  eps <- finfo(dt)$eps * 50
  gam <- scipy_special_gamma
  and <- jnp_logical_and
  or <- jnp_logical_or
  d <- c - a - b
  s <- 1L - x
  ca <- c - a
  cb <- c - b
  id <- jnp_round(d)
  index <- jnp_where(
    or(x == 0L, and(or(a == 0L, b == 0L), c != 0L)), 0L,
    jnp_where(
      or(c == 0L, and(c < 0L, c %% 1L == 0L)), 1L,
      jnp_where(
        and(d <= -1L, jnp_logical_not(and(jnp_abs(d - id) >= eps, s < 0L))), 2L,
        jnp_where(
          and(d <= 0L, x == 1L), 1L,
          jnp_where(and(x < 1L, b == c), 3L,
                    jnp_where(and(x < 1L, a == c), 4L,
                              jnp_where(x > 1L, 1L, jnp_where(x == 1L, 5L, 6L))))))))
  lax_select_n(lax_convert_element_type(index, "int32"),
               as_jax(1, dt), as_jax(Inf, dt),
               jnp_power(s, d) * sp_hyp2f1_terminal_or_serie(ca, cb, c, x),
               jnp_power(s, -a), jnp_power(s, -b),
               gam(c) * gam(d) / (gam(ca) * gam(cb)),
               sp_hyp2f1_terminal_or_serie(a, b, c, x))
})))
sp_hyp2f1 <- sp_defjvps(
  sp_hyp2f1,
  function(a_dot, primal_out, a, b, c, x) sp_hyp2f1_a_derivative(a, b, c, x) * a_dot,
  function(b_dot, primal_out, a, b, c, x) sp_hyp2f1_b_derivative(a, b, c, x) * b_dot,
  function(c_dot, primal_out, a, b, c, x) sp_hyp2f1_c_derivative(a, b, c, x) * c_dot,
  function(x_dot, primal_out, a, b, c, x) a * b / c * sp_hyp2f1(a + 1L, b + 1L, c + 1L, x) * x_dot)

#' @rdname scipy_special
#' @export
scipy_special_hyp2f1 <- function(a, b, c, x) sp_hyp2f1(a, b, c, x)

# ---- associated Legendre functions and spherical harmonics -----------------------------------------

# Port of jax/_src/scipy/special.py:_gen_recurrence_mask: the coefficient masks d0 (l > m)
# and d1 (l > m + 1) of the recurrence over l, as (l_max + 1, l_max + 1) arrays indexed
# [m, l]. JAX multiplies them by a 3D mask with ones on the plane i + m - l == 0; the plane
# of iteration i is applied in sp_gen_associated_legendre.
sp_gen_recurrence_mask <- function(l_max, is_normalized, dt) {
  L <- l_max + 1L
  m_mat <- lax_broadcasted_iota(dt, c(L, L), 0L)
  l_mat <- lax_broadcasted_iota(dt, c(L, L), 1L)
  if (is_normalized) {
    c0 <- l_mat * l_mat
    c1 <- m_mat * m_mat
    c2 <- 2.0 * l_mat
    c3 <- (l_mat - 1.0) * (l_mat - 1.0)
    d0 <- jnp_sqrt((4.0 * c0 - 1.0) / (c0 - c1))
    d1 <- jnp_sqrt(((c2 + 1.0) * (c3 - c1)) / ((c2 - 3.0) * (c0 - c1)))
  } else {
    d0 <- (2.0 * l_mat - 1.0) / (l_mat - m_mat)
    d1 <- (l_mat + m_mat - 1.0) / (l_mat - m_mat)
  }
  gap <- outer(seq_len(L), seq_len(L), function(m, l) l - m)
  d_zeros <- jnp_zeros(c(L, L), dt)
  list(lax_select(as_jax(gap >= 1L, "bool"), d0, d_zeros),
       lax_select(as_jax(gap >= 2L, "bool"), d1, d_zeros))
}

# Port of jax/_src/scipy/special.py:_gen_associated_legendre: P_l^m(x) for 0 <= m, l <= l_max
# as an array of shape (l_max + 1, l_max + 1, length(x)) indexed [m, l, i], optionally
# normalized for spherical harmonics.
sp_gen_associated_legendre <- jit(function(l_max, x, is_normalized) {
  dt <- sp_dtype(x)
  n <- sp_shape(x)[1L]
  L <- l_max + 1L
  gap <- outer(seq_len(L), seq_len(L), function(m, l) l - m)
  initial_value <- if (is_normalized) 0.5 / jnp_sqrt(pi) else 1.0  # p(0, 0)
  diag <- jnp_broadcast_to(jnp_astype(initial_value, dt), c(1L, n))
  off <- jnp_zeros(c(1L, n), dt)
  if (l_max >= 1L) {
    a_idx <- sp_arange(1L, l_max + 1L, dt)
    b_idx <- jnp_arange(l_max, dtype = dt)
    if (is_normalized) {
      f_a <- jnp_cumprod(-1L * jnp_sqrt(1.0 + 0.5 / a_idx))
      f_b <- jnp_sqrt(2.0 * b_idx + 3.0)
    } else {
      f_a <- jnp_cumprod(1.0 - 2.0 * a_idx)
      f_b <- 2.0 * b_idx + 1.0
    }
    # Diagonal entries p(l, l) by recurrence.
    y <- jnp_cumprod(jnp_broadcast_to(jnp_sqrt(1.0 - x * x), c(l_max, n)), axis = 0L)
    p_diag <- initial_value * (lax_expand_dims(f_a, 1L) * y)
    # Entries p(l, l + 1) by recurrence.
    p_offdiag <- (lax_expand_dims(f_b, 1L) * lax_expand_dims(x, 0L)) * lax_slice_in_dim(
      lax_concatenate(list(diag, p_diag), 0L), 0L, l_max)
    diag <- lax_concatenate(list(diag, p_diag), 0L)
    off <- lax_concatenate(list(p_offdiag, off), 0L)
  }
  # p[m, m] = diag[m], p[m, m + 1] = off[m], zero elsewhere.
  full <- function(rows) lax_broadcast_in_dim(rows, c(L, L, n), c(0L, 2L))
  mask <- function(k) lax_broadcast_in_dim(as_jax(gap == k, "bool"), c(L, L, n), c(0L, 1L))
  p <- lax_select(mask(0L), full(diag), lax_select(mask(1L), full(off), jnp_zeros(c(L, L, n), dt)))
  if (l_max > 1L) {
    masks <- sp_gen_recurrence_mask(l_max, is_normalized, dt)
    roll <- function(a, shift) {
      lax_concatenate(list(lax_slice_in_dim(a, L - shift, L, axis = 1L),
                           lax_slice_in_dim(a, 0L, L - shift, axis = 1L)), 1L)
    }
    # ponytail: the loop over i = 2..l_max is unrolled (JAX runs it with fori_loop); the
    # operations per step are the same. Use lax_fori_loop if l_max gets large.
    for (i in 2:l_max) {
      plane <- as_jax(array(as.numeric(gap == i), c(L, L)), dt)
      coeff_0 <- lax_expand_dims(masks[[1L]] * plane, 2L)
      coeff_1 <- lax_expand_dims(masks[[2L]] * plane, 2L)
      h <- coeff_0 * (roll(p, 1L) * x) - coeff_1 * roll(p, 2L)
      p <- p + h
    }
  }
  p
}, static_argnums = c(0L, 2L))

sp_check_lpmn_args <- function(m, n, z) {
  z <- as_jax(z)
  sp_check_float_dtype(sp_dtype(z), "z.dtype=%s is not supported, see docstring for supported types.")
  if (length(sp_shape(z)) != 1L) jax_error("z must be a 1D array.", class = "jax_value_error")
  m <- as.integer(as.array(concrete_or_error(m, "Argument m of lpmn.")))
  n <- as.integer(as.array(concrete_or_error(n, "Argument n of lpmn.")))
  if (m != n)
    jax_error("Computations for m!=n are not yet supported.", class = "jax_not_implemented_error")
  list(l_max = n, z = z)
}

# Port of jax/_src/scipy/special.py:_gen_derivatives: derivatives of the (unnormalized)
# associated Legendre functions with respect to x, same layout as `p`.
sp_gen_derivatives <- jit(function(p, x) {
  dt <- sp_dtype(x)
  shp <- sp_shape(p)
  num_m <- shp[1L]
  num_l <- shp[2L]
  num_x <- shp[3L]
  zeros <- function(...) jnp_zeros(c(...), dt)
  set_block <- function(arr, update, start) lax_dynamic_update_slice(arr, update, as.list(start))
  # p_{l-1}^m, p_{l-1}^{m+2} and p_{l-1}^{m-2}.
  p_m_lm1 <- lax_concatenate(list(zeros(num_m, 1L, num_x), lax_slice_in_dim(p, 0L, num_l - 1L, axis = 1L)), 1L)
  shift_m <- function(a, k) {
    pad <- zeros(min(abs(k), num_m), num_l, num_x)
    if (abs(k) >= num_m) return(zeros(num_m, num_l, num_x))
    if (k > 0L) lax_concatenate(list(lax_slice_in_dim(a, k, num_m), pad), 0L)
    else lax_concatenate(list(pad, lax_slice_in_dim(a, 0L, num_m + k)), 0L)
  }
  p_mp2_lm1 <- shift_m(p_m_lm1, 2L)
  p_mm2_lm1 <- shift_m(p_m_lm1, -2L)
  # The derivative needs negative orders.
  row <- function(m, lo, hi) {
    lax_index_in_dim(lax_slice_in_dim(p, lo, hi, axis = 1L), m, 0L, keepdims = FALSE)
  }
  if (num_l > 2L) {
    l_vec <- sp_arange(1L, num_l - 1L, dt)
    coeff <- -1.0 / ((l_vec + 1L) * l_vec)
    update <- lax_expand_dims(coeff, 1L) * row(1L, 1L, num_l - 1L)
    p_mm2_lm1 <- set_block(p_mm2_lm1, lax_expand_dims(update, 0L), c(1L, 2L, 0L))
  }
  if (num_l > 3L) {
    l_vec <- sp_arange(2L, num_l - 1L, dt)
    coeff <- 1.0 / ((l_vec + 2L) * (l_vec + 1L) * l_vec * (l_vec - 1L))
    update <- lax_expand_dims(coeff, 1L) * row(2L, 2L, num_l - 1L)
    p_mm2_lm1 <- set_block(p_mm2_lm1, lax_expand_dims(update, 0L), c(0L, 3L, 0L))
  }
  m_mat <- lax_broadcasted_iota(dt, c(num_m, num_l), 0L)
  l_mat <- lax_broadcasted_iota(dt, c(num_m, num_l), 1L)
  upper <- outer(seq_len(num_m), seq_len(num_l), function(m, l) l >= m)
  not_row1 <- outer(seq_len(num_m), seq_len(num_l), function(m, l) m != 2L)
  masked <- function(a, keep = upper) {
    lax_expand_dims(lax_select(as_jax(keep, "bool"), a, zeros(num_m, num_l)), 2L)
  }
  a0 <- -0.5 / (m_mat - 1.0)
  b0 <- l_mat + m_mat
  c0 <- a0 * (b0 - 2.0) * (b0 - 1.0)
  # p_l^{m-1}
  p_mm1_l <- masked(a0, upper & not_row1) * p_m_lm1 + masked(c0, upper & not_row1) * p_mm2_lm1
  d0 <- -0.5 / (m_mat + 1.0)
  e0 <- d0 * b0 * (b0 + 1.0)
  # p_l^{m+1}
  p_mp1_l <- masked(d0) * p_mp2_lm1 + masked(e0) * p_m_lm1
  f0 <- b0 * (l_mat - m_mat + 1.0) / 2.0
  p_derivative <- masked(f0) * p_mm1_l - 0.5 * p_mp1_l
  # The singularity at m = 1 is treated separately.
  if (num_m > 1L) {
    l_vec <- jnp_arange(num_l, dtype = dt)
    g0 <- lax_expand_dims((l_vec + 1L) * l_vec, 1L) * row(0L, 0L, num_l)
    if (num_l > 2L) g0 <- g0 - row(2L, 0L, num_l)
    p_derivative_m0 <- lax_expand_dims(0.5 / jnp_sqrt(1L - x * x), 0L) * g0
    p_derivative <- set_block(p_derivative, lax_expand_dims(p_derivative_m0, 0L), c(1L, 0L, 0L))
    p_derivative <- set_block(p_derivative, zeros(1L, 1L, num_x), c(1L, 0L, 0L))
  }
  p_derivative
})

# Port of jax/_src/scipy/special.py:lpmn (deprecated in JAX; m must equal n).
#' @rdname scipy_special
#' @export
scipy_special_lpmn <- function(m, n, z) {
  warning("jax.scipy.special.lpmn is deprecated; no replacement is planned.")
  a <- sp_check_lpmn_args(m, n, z)
  p_vals <- sp_gen_associated_legendre(a$l_max, a$z, FALSE)
  list(p_vals, sp_gen_derivatives(p_vals, a$z))
}

# Port of jax/_src/scipy/special.py:lpmn_values (deprecated in JAX; m must equal n).
#' @rdname scipy_special
#' @export
scipy_special_lpmn_values <- function(m, n, z, is_normalized) {
  warning("jax.scipy.special.lpmn_values is deprecated; no replacement is planned.")
  a <- sp_check_lpmn_args(m, n, z)
  sp_gen_associated_legendre(a$l_max, a$z, isTRUE(is_normalized))
}

# Port of jax/_src/scipy/special.py:_sph_harm.
sp_sph_harm <- jit(function(n, m, theta, phi, n_max) {
  cos_colatitude <- jnp_cos(theta)
  legendre <- sp_gen_associated_legendre(n_max, cos_colatitude, TRUE)
  abs_m <- jnp_abs(m)
  legendre_val <- at_get(legendre, abs_m, n, jnp_arange(sp_shape(n)[1L]), mode = "clip")
  angle <- abs_m * phi
  vandermonde <- lax_complex(jnp_cos(angle), jnp_sin(angle))
  harmonics <- lax_complex(legendre_val * jnp_real(vandermonde), legendre_val * jnp_imag(vandermonde))
  # Negative order.
  jnp_where(m < 0L, jnp_power(-1.0, abs_m) * jnp_conjugate(harmonics), harmonics)
}, static_argnums = 4L)

# Port of jax/_src/scipy/special.py:sph_harm_y.
#' @rdname scipy_special
#' @export
scipy_special_sph_harm_y <- function(n, m, theta, phi, diff_n = NULL, n_max = NULL) {
  if (!is.null(diff_n))
    jax_error("The 'diff_n' argument to jax.scipy.special.sph_harm_y is not supported.",
              class = "jax_not_implemented_error")
  theta <- as_jax(theta)
  if (!length(sp_shape(theta))) theta <- lax_reshape(theta, 1L)
  what <- paste0("The `n_max` argument of `jax.scipy.special.sph_harm_y` must be statically ",
                 "specified to use `sph_harm_y` within JAX transformations.")
  if (is.null(n_max)) n_max <- max(as.array(concrete_or_error(n, what)))
  n_max <- as.integer(as.array(concrete_or_error(n_max, what)))
  sp_sph_harm(as_jax(n), as_jax(m), theta, as_jax(phi), n_max)
}
