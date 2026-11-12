# jax.scipy.stats (port of jax/_src/scipy/stats/*.py).
#
# Each distribution module of jax.scipy.stats becomes a family of functions named
# scipy_stats_<distribution>_<method>. Arguments are promoted with the jnp rules
# (promote_args_inexact), so R numbers act as weakly typed Python scalars. JAX's
# `_lax_const(x, v)` constants are written as R numbers in jnp operations, which have the
# same dtype after promotion.

st_args <- function(...) promote_args_inexact(...)

st_log1mexp <- function(x) nn_log1mexp(x)

# ---- norm ---------------------------------------------------------------------------------

#' Normal distribution
#'
#' Ports of `jax.scipy.stats.norm`: log density, density, distribution function and its
#' logarithm, survival function and its logarithm, percent point function and inverse
#' survival function of the normal distribution.
#' @param x Values.
#' @param q Probabilities.
#' @param loc Location (mean).
#' @param scale Scale (standard deviation).
#' @return An array.
#' @name scipy_stats_norm
#' @examples
#' \dontrun{
#' scipy_stats_norm_logpdf(c(-1, 0, 1))
#' scipy_stats_norm_ppf(0.975)
#' }
NULL

#' @rdname scipy_stats_norm
#' @export
scipy_stats_norm_logpdf <- function(x, loc = 0, scale = 1) {
  a <- st_args(x, loc, scale)
  x <- a[[1L]]; loc <- a[[2L]]; scale <- a[[3L]]
  scale_sqrd <- jnp_square(scale)
  log_normalizer <- jnp_log(jnp_multiply(2 * pi, scale_sqrd))
  quadratic <- jnp_true_divide(jnp_square(jnp_subtract(x, loc)), scale_sqrd)
  jnp_true_divide(jnp_add(log_normalizer, quadratic), -2)
}

#' @rdname scipy_stats_norm
#' @export
scipy_stats_norm_pdf <- function(x, loc = 0, scale = 1) jnp_exp(scipy_stats_norm_logpdf(x, loc, scale))

#' @rdname scipy_stats_norm
#' @export
scipy_stats_norm_cdf <- function(x, loc = 0, scale = 1) {
  a <- st_args(x, loc, scale)
  scipy_special_ndtr(jnp_true_divide(jnp_subtract(a[[1L]], a[[2L]]), a[[3L]]))
}

#' @rdname scipy_stats_norm
#' @export
scipy_stats_norm_logcdf <- function(x, loc = 0, scale = 1) {
  a <- st_args(x, loc, scale)
  scipy_special_log_ndtr(jnp_true_divide(jnp_subtract(a[[1L]], a[[2L]]), a[[3L]]))
}

#' @rdname scipy_stats_norm
#' @export
scipy_stats_norm_ppf <- function(q, loc = 0, scale = 1) {
  jnp_asarray(jnp_add(jnp_multiply(scipy_special_ndtri(q), scale), loc), dtype = "float")
}

#' @rdname scipy_stats_norm
#' @export
scipy_stats_norm_logsf <- function(x, loc = 0, scale = 1) {
  a <- st_args(x, loc, scale)
  scipy_stats_norm_logcdf(jnp_negative(a[[1L]]), jnp_negative(a[[2L]]), a[[3L]])
}

#' @rdname scipy_stats_norm
#' @export
scipy_stats_norm_sf <- function(x, loc = 0, scale = 1) {
  a <- st_args(x, loc, scale)
  scipy_stats_norm_cdf(jnp_negative(a[[1L]]), jnp_negative(a[[2L]]), a[[3L]])
}

#' @rdname scipy_stats_norm
#' @export
scipy_stats_norm_isf <- function(q, loc = 0, scale = 1) {
  a <- st_args(q, loc, scale)
  jnp_negative(scipy_stats_norm_ppf(a[[1L]], jnp_negative(a[[2L]]), a[[3L]]))
}

# ---- bernoulli ----------------------------------------------------------------------------

#' Bernoulli distribution
#'
#' Ports of `jax.scipy.stats.bernoulli`.
#' @param k Outcomes.
#' @param q Probabilities.
#' @param p Success probability.
#' @param loc Location.
#' @return An array.
#' @name scipy_stats_bernoulli
NULL

#' @rdname scipy_stats_bernoulli
#' @export
scipy_stats_bernoulli_logpmf <- function(k, p, loc = 0) {
  a <- st_args(k, p, loc)
  k <- a[[1L]]; p <- a[[2L]]; loc <- a[[3L]]
  x <- jnp_subtract(k, loc)
  log_probs <- jnp_add(scipy_special_xlogy(x, p), scipy_special_xlog1py(jnp_subtract(1, x), jnp_negative(p)))
  jnp_where(jnp_logical_or(jnp_less(x, 0), jnp_greater(x, 1)), -Inf, log_probs)
}

#' @rdname scipy_stats_bernoulli
#' @export
scipy_stats_bernoulli_pmf <- function(k, p, loc = 0) jnp_exp(scipy_stats_bernoulli_logpmf(k, p, loc))

#' @rdname scipy_stats_bernoulli
#' @export
scipy_stats_bernoulli_cdf <- function(k, p) {
  a <- st_args(k, p)
  k <- a[[1L]]; p <- a[[2L]]
  bad <- jnp_isnan(k) | jnp_isnan(p) | (p < 0) | (p > 1)
  jnp_select(list(bad, k < 0, (k >= 0) & (k < 1), k >= 1),
             list(NaN, jnp_zeros_like(k), jnp_subtract(1, p), jnp_ones_like(k)))
}

#' @rdname scipy_stats_bernoulli
#' @export
scipy_stats_bernoulli_ppf <- function(q, p) {
  a <- st_args(q, p)
  q <- a[[1L]]; p <- a[[2L]]
  bad <- jnp_isnan(q) | jnp_isnan(p) | (p < 0) | (p > 1) | (q < 0) | (q > 1)
  jnp_where(bad, NaN, jnp_where(q <= jnp_subtract(1, p), jnp_zeros_like(q), jnp_ones_like(q)))
}

# ---- beta ---------------------------------------------------------------------------------

#' Beta distribution
#'
#' Ports of `jax.scipy.stats.beta`.
#' @param x Values.
#' @param a,b Shape parameters.
#' @param loc Location.
#' @param scale Scale.
#' @return An array.
#' @name scipy_stats_beta
NULL

#' @rdname scipy_stats_beta
#' @export
scipy_stats_beta_logpdf <- function(x, a, b, loc = 0, scale = 1) {
  v <- st_args(x, a, b, loc, scale)
  x <- v[[1L]]; a <- v[[2L]]; b <- v[[3L]]; loc <- v[[4L]]; scale <- v[[5L]]
  shape_term <- jnp_negative(scipy_special_betaln(a, b))
  y <- jnp_true_divide(jnp_subtract(x, loc), scale)
  log_linear_term <- jnp_add(scipy_special_xlogy(jnp_subtract(a, 1), y),
                             scipy_special_xlog1py(jnp_subtract(b, 1), jnp_negative(y)))
  log_probs <- jnp_subtract(jnp_add(shape_term, log_linear_term), jnp_log(scale))
  result <- jnp_where(jnp_logical_or(x > jnp_add(loc, scale), x < loc), -Inf, log_probs)
  jnp_where((a <= 0) | (b <= 0) | (scale <= 0), NaN, result)
}

#' @rdname scipy_stats_beta
#' @export
scipy_stats_beta_pdf <- function(x, a, b, loc = 0, scale = 1) jnp_exp(scipy_stats_beta_logpdf(x, a, b, loc, scale))

#' @rdname scipy_stats_beta
#' @export
scipy_stats_beta_cdf <- function(x, a, b, loc = 0, scale = 1) {
  v <- st_args(x, a, b, loc, scale)
  y <- lax_clamp(0, jnp_true_divide(jnp_subtract(v[[1L]], v[[4L]]), v[[5L]]), 1)
  scipy_special_betainc(v[[2L]], v[[3L]], y)
}

#' @rdname scipy_stats_beta
#' @export
scipy_stats_beta_logcdf <- function(x, a, b, loc = 0, scale = 1) jnp_log(scipy_stats_beta_cdf(x, a, b, loc, scale))

#' @rdname scipy_stats_beta
#' @export
scipy_stats_beta_sf <- function(x, a, b, loc = 0, scale = 1) {
  v <- st_args(x, a, b, loc, scale)
  y <- lax_clamp(0, jnp_true_divide(jnp_subtract(v[[1L]], v[[4L]]), v[[5L]]), 1)
  scipy_special_betainc(v[[3L]], v[[2L]], jnp_subtract(1, y))
}

#' @rdname scipy_stats_beta
#' @export
scipy_stats_beta_logsf <- function(x, a, b, loc = 0, scale = 1) jnp_log(scipy_stats_beta_sf(x, a, b, loc, scale))

# ---- betabinom ----------------------------------------------------------------------------

#' Beta binomial distribution
#'
#' Ports of `jax.scipy.stats.betabinom`.
#' @param k Outcomes.
#' @param n Number of trials.
#' @param a,b Shape parameters.
#' @param loc Location.
#' @return An array.
#' @name scipy_stats_betabinom
NULL

#' @rdname scipy_stats_betabinom
#' @export
scipy_stats_betabinom_logpmf <- function(k, n, a, b, loc = 0) {
  v <- st_args(k, n, a, b, loc)
  k <- v[[1L]]; n <- v[[2L]]; a <- v[[3L]]; b <- v[[4L]]; loc <- v[[5L]]
  y <- jnp_subtract(jnp_floor(k), loc)
  combiln <- jnp_negative(jnp_add(jnp_log1p(n),
                                  scipy_special_betaln(jnp_add(jnp_subtract(n, y), 1), jnp_add(y, 1))))
  beta_lns <- jnp_subtract(scipy_special_betaln(jnp_add(y, a), jnp_add(jnp_subtract(n, y), b)),
                           scipy_special_betaln(a, b))
  log_probs <- jnp_add(combiln, beta_lns)
  log_probs <- jnp_where((y == 0) & (n == 0), 0, log_probs)
  y_cond <- (y < jnp_negative(loc)) | (y > n) | (jnp_add(y, a) <= 0)
  log_probs <- jnp_where(y_cond, -Inf, log_probs)
  jnp_where((n < 0) | (a <= 0) | (b <= 0), NaN, log_probs)
}

#' @rdname scipy_stats_betabinom
#' @export
scipy_stats_betabinom_pmf <- function(k, n, a, b, loc = 0) jnp_exp(scipy_stats_betabinom_logpmf(k, n, a, b, loc))

# ---- binom --------------------------------------------------------------------------------

#' Binomial distribution
#'
#' Ports of `jax.scipy.stats.binom`.
#' @param k Outcomes.
#' @param n Number of trials.
#' @param p Success probability.
#' @param loc Location.
#' @return An array.
#' @name scipy_stats_binom
NULL

#' @rdname scipy_stats_binom
#' @export
scipy_stats_binom_logpmf <- function(k, n, p, loc = 0) {
  v <- st_args(k, n, p, loc)
  k <- v[[1L]]; n <- v[[2L]]; p <- v[[3L]]; loc <- v[[4L]]
  y <- jnp_subtract(k, loc)
  comb_term <- jnp_subtract(scipy_special_gammaln(jnp_add(n, 1)),
                            jnp_add(scipy_special_gammaln(jnp_add(y, 1)),
                                    scipy_special_gammaln(jnp_add(jnp_subtract(n, y), 1))))
  log_linear_term <- jnp_add(scipy_special_xlogy(y, p),
                             scipy_special_xlog1py(jnp_subtract(n, y), jnp_negative(p)))
  log_probs <- jnp_add(comb_term, log_linear_term)
  log_probs <- jnp_where(((y == 0) & (n == 0)) | (log_linear_term == 0), 0, log_probs)
  jnp_where((k >= loc) & (k < jnp_add(jnp_add(loc, n), 1)), log_probs, -Inf)
}

#' @rdname scipy_stats_binom
#' @export
scipy_stats_binom_pmf <- function(k, n, p, loc = 0) jnp_exp(scipy_stats_binom_logpmf(k, n, p, loc))

# ---- cauchy -------------------------------------------------------------------------------

#' Cauchy distribution
#'
#' Ports of `jax.scipy.stats.cauchy`.
#' @param x Values.
#' @param q Probabilities.
#' @param loc Location.
#' @param scale Scale.
#' @return An array.
#' @name scipy_stats_cauchy
NULL

#' @rdname scipy_stats_cauchy
#' @export
scipy_stats_cauchy_logpdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  scaled_x <- jnp_true_divide(jnp_subtract(v[[1L]], v[[2L]]), v[[3L]])
  normalize_term <- jnp_log(jnp_multiply(pi, v[[3L]]))
  jnp_negative(jnp_add(normalize_term, jnp_log1p(jnp_multiply(scaled_x, scaled_x))))
}

#' @rdname scipy_stats_cauchy
#' @export
scipy_stats_cauchy_pdf <- function(x, loc = 0, scale = 1) jnp_exp(scipy_stats_cauchy_logpdf(x, loc, scale))

#' @rdname scipy_stats_cauchy
#' @export
scipy_stats_cauchy_cdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  scaled_x <- jnp_true_divide(jnp_subtract(v[[1L]], v[[2L]]), v[[3L]])
  # The constants take the dtype of the arguments before the division, as in JAX. Whether
  # a float32 tail reaches exactly zero depends on how the XLA build rounds atan(Inf).
  const <- function(v) lax_full_like(scaled_x, v, shape = integer(0))
  jnp_add(const(0.5), jnp_multiply(lax_div(const(1), const(pi)), jnp_arctan(scaled_x)))
}

#' @rdname scipy_stats_cauchy
#' @export
scipy_stats_cauchy_logcdf <- function(x, loc = 0, scale = 1) jnp_log(scipy_stats_cauchy_cdf(x, loc, scale))

#' @rdname scipy_stats_cauchy
#' @export
scipy_stats_cauchy_sf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  scipy_stats_cauchy_cdf(jnp_negative(v[[1L]]), jnp_negative(v[[2L]]), v[[3L]])
}

#' @rdname scipy_stats_cauchy
#' @export
scipy_stats_cauchy_logsf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  scipy_stats_cauchy_logcdf(jnp_negative(v[[1L]]), jnp_negative(v[[2L]]), v[[3L]])
}

#' @rdname scipy_stats_cauchy
#' @export
scipy_stats_cauchy_isf <- function(q, loc = 0, scale = 1) {
  v <- st_args(q, loc, scale)
  jnp_negative(scipy_stats_cauchy_ppf(v[[1L]], jnp_negative(v[[2L]]), v[[3L]]))
}

#' @rdname scipy_stats_cauchy
#' @export
scipy_stats_cauchy_ppf <- function(q, loc = 0, scale = 1) {
  v <- st_args(q, loc, scale)
  q <- v[[1L]]
  upper <- q > 0.5
  arg <- jnp_multiply(pi, jnp_where(upper, jnp_subtract(1, q), jnp_abs(q)))
  unscaled <- jnp_where(q == 0.5, jnp_zeros_like(q),
                        jnp_true_divide(jnp_where(upper, jnp_ones_like(q), jnp_full_like(q, -1)),
                                        jnp_tan(arg)))
  jnp_where((q < 0) | (q > 1), NaN, jnp_add(jnp_multiply(unscaled, v[[3L]]), v[[2L]]))
}

# ---- chi2 ---------------------------------------------------------------------------------

#' Chi-squared distribution
#'
#' Ports of `jax.scipy.stats.chi2`.
#' @param x Values.
#' @param df Degrees of freedom.
#' @param loc Location.
#' @param scale Scale.
#' @return An array.
#' @name scipy_stats_chi2
NULL

#' @rdname scipy_stats_chi2
#' @export
scipy_stats_chi2_logpdf <- function(x, df, loc = 0, scale = 1) {
  v <- st_args(x, df, loc, scale)
  x <- v[[1L]]; df <- v[[2L]]; loc <- v[[3L]]; scale <- v[[4L]]
  y <- jnp_true_divide(jnp_subtract(x, loc), scale)
  df_on_two <- jnp_true_divide(df, 2)
  kernel <- jnp_subtract(scipy_special_xlogy(jnp_subtract(df_on_two, 1), y), jnp_true_divide(y, 2))
  nrml_cnst <- jnp_negative(jnp_add(lax_lgamma(df_on_two), jnp_true_divide(jnp_multiply(log(2), df), 2)))
  log_probs <- jnp_add(jnp_subtract(nrml_cnst, jnp_log(scale)), kernel)
  jnp_where(x < loc, -Inf, log_probs)
}

#' @rdname scipy_stats_chi2
#' @export
scipy_stats_chi2_pdf <- function(x, df, loc = 0, scale = 1) jnp_exp(scipy_stats_chi2_logpdf(x, df, loc, scale))

#' @rdname scipy_stats_chi2
#' @export
scipy_stats_chi2_cdf <- function(x, df, loc = 0, scale = 1) {
  v <- st_args(x, df, loc, scale)
  y <- lax_clamp(0, jnp_true_divide(jnp_subtract(v[[1L]], v[[3L]]), jnp_multiply(v[[4L]], 2)), Inf)
  scipy_special_gammainc(jnp_true_divide(v[[2L]], 2), y)
}

#' @rdname scipy_stats_chi2
#' @export
scipy_stats_chi2_logcdf <- function(x, df, loc = 0, scale = 1) jnp_log(scipy_stats_chi2_cdf(x, df, loc, scale))

#' @rdname scipy_stats_chi2
#' @export
scipy_stats_chi2_sf <- function(x, df, loc = 0, scale = 1) {
  v <- st_args(x, df, loc, scale)
  y <- lax_clamp(0, jnp_true_divide(jnp_subtract(v[[1L]], v[[3L]]), jnp_multiply(v[[4L]], 2)), Inf)
  scipy_special_gammaincc(jnp_true_divide(v[[2L]], 2), y)
}

#' @rdname scipy_stats_chi2
#' @export
scipy_stats_chi2_logsf <- function(x, df, loc = 0, scale = 1) jnp_log(scipy_stats_chi2_sf(x, df, loc, scale))

# ---- dirichlet ----------------------------------------------------------------------------

#' Dirichlet distribution
#'
#' Ports of `jax.scipy.stats.dirichlet`. `x` holds the components along its first axis;
#' it may omit the last component, which is then `1 - sum(x)`.
#' @param x Points on the simplex.
#' @param alpha Concentration parameters (one dimensional).
#' @return An array.
#' @name scipy_stats_dirichlet
NULL

#' @rdname scipy_stats_dirichlet
#' @export
scipy_stats_dirichlet_logpdf <- function(x, alpha) {
  v <- promote_dtypes_inexact(list(x, alpha))
  x <- v[[1L]]; alpha <- v[[2L]]
  xs <- aval_of(x)$shape; ash <- aval_of(alpha)$shape
  if (length(ash) != 1L)
    jax_error("`alpha` must be one-dimensional; got alpha.shape=", py_tuple(ash),
              class = "jax_value_error")
  if (!xs[1L] %in% c(ash[1L], ash[1L] - 1L))
    jax_error("`x` must have either the same number of entries as `alpha` or one entry fewer; got ",
              "x.shape=", py_tuple(xs), ", alpha.shape=", py_tuple(ash), class = "jax_value_error")
  if (xs[1L] != ash[1L])
    x <- jnp_concatenate(list(x, jnp_subtract(1, jnp_sum(x, axis = 0L, keepdims = TRUE))), axis = 0L)
  normalize_term <- jnp_subtract(jnp_sum(scipy_special_gammaln(alpha)),
                                 scipy_special_gammaln(jnp_sum(alpha)))
  nd <- length(aval_of(x)$shape)
  if (nd > 1L) alpha <- lax_broadcast_in_dim(alpha, c(ash, rep(1L, nd - 1L)), 0L)
  log_probs <- jnp_subtract(jnp_sum(scipy_special_xlogy(jnp_subtract(alpha, 1), x), axis = 0L),
                            normalize_term)
  simplex <- jnp_all(x > 0, axis = 0L) & (jnp_abs(jnp_subtract(jnp_sum(x, axis = 0L), 1)) < 1e-6)
  jnp_where(simplex, log_probs, -Inf)
}

#' @rdname scipy_stats_dirichlet
#' @export
scipy_stats_dirichlet_pdf <- function(x, alpha) jnp_exp(scipy_stats_dirichlet_logpdf(x, alpha))

# ---- expon --------------------------------------------------------------------------------

#' Exponential distribution
#'
#' Ports of `jax.scipy.stats.expon`.
#' @param x Values.
#' @param q Probabilities.
#' @param loc Location.
#' @param scale Scale.
#' @return An array.
#' @name scipy_stats_expon
NULL

#' @rdname scipy_stats_expon
#' @export
scipy_stats_expon_logpdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  x <- v[[1L]]; loc <- v[[2L]]; scale <- v[[3L]]
  log_probs <- jnp_negative(jnp_add(jnp_true_divide(jnp_subtract(x, loc), scale), jnp_log(scale)))
  jnp_where(x < loc, -Inf, log_probs)
}

#' @rdname scipy_stats_expon
#' @export
scipy_stats_expon_pdf <- function(x, loc = 0, scale = 1) jnp_exp(scipy_stats_expon_logpdf(x, loc, scale))

#' @rdname scipy_stats_expon
#' @export
scipy_stats_expon_cdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  neg_scaled_x <- jnp_true_divide(jnp_subtract(v[[2L]], v[[1L]]), v[[3L]])
  jnp_where(v[[1L]] < v[[2L]], jnp_zeros_like(neg_scaled_x), jnp_negative(jnp_expm1(neg_scaled_x)))
}

#' @rdname scipy_stats_expon
#' @export
scipy_stats_expon_logcdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  scaled_x <- jnp_true_divide(jnp_subtract(v[[1L]], v[[2L]]), v[[3L]])
  below_loc <- v[[1L]] < v[[2L]]
  safe_x <- jnp_where(below_loc, 1, scaled_x)
  jnp_where(below_loc, -Inf, st_log1mexp(safe_x))
}

#' @rdname scipy_stats_expon
#' @export
scipy_stats_expon_logsf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  neg_scaled_x <- jnp_true_divide(jnp_subtract(v[[2L]], v[[1L]]), v[[3L]])
  jnp_where(v[[1L]] < v[[2L]], jnp_zeros_like(neg_scaled_x), neg_scaled_x)
}

#' @rdname scipy_stats_expon
#' @export
scipy_stats_expon_sf <- function(x, loc = 0, scale = 1) jnp_exp(scipy_stats_expon_logsf(x, loc, scale))

#' @rdname scipy_stats_expon
#' @export
scipy_stats_expon_ppf <- function(q, loc = 0, scale = 1) {
  v <- st_args(q, loc, scale)
  q <- v[[1L]]
  jnp_where(jnp_isnan(q) | (q < 0) | (q > 1), NaN,
            jnp_subtract(v[[2L]], jnp_multiply(v[[3L]], jnp_log1p(jnp_negative(q)))))
}

# ---- gamma --------------------------------------------------------------------------------

#' Gamma distribution
#'
#' Ports of `jax.scipy.stats.gamma`.
#' @param x Values.
#' @param a Shape parameter.
#' @param loc Location.
#' @param scale Scale.
#' @return An array.
#' @name scipy_stats_gamma
NULL

#' @rdname scipy_stats_gamma
#' @export
scipy_stats_gamma_logpdf <- function(x, a, loc = 0, scale = 1) {
  v <- st_args(x, a, loc, scale)
  x <- v[[1L]]; a <- v[[2L]]; loc <- v[[3L]]; scale <- v[[4L]]
  ok <- x >= loc
  y <- jnp_where(ok, jnp_true_divide(jnp_subtract(x, loc), scale), jnp_ones_like(x))
  log_linear_term <- jnp_subtract(scipy_special_xlogy(jnp_subtract(a, 1), y), y)
  shape_terms <- jnp_add(scipy_special_gammaln(a), jnp_log(scale))
  jnp_where(ok, jnp_subtract(log_linear_term, shape_terms), -Inf)
}

#' @rdname scipy_stats_gamma
#' @export
scipy_stats_gamma_pdf <- function(x, a, loc = 0, scale = 1) jnp_exp(scipy_stats_gamma_logpdf(x, a, loc, scale))

#' @rdname scipy_stats_gamma
#' @export
scipy_stats_gamma_cdf <- function(x, a, loc = 0, scale = 1) {
  v <- st_args(x, a, loc, scale)
  y <- lax_clamp(0, jnp_true_divide(jnp_subtract(v[[1L]], v[[3L]]), v[[4L]]), Inf)
  scipy_special_gammainc(v[[2L]], y)
}

#' @rdname scipy_stats_gamma
#' @export
scipy_stats_gamma_logcdf <- function(x, a, loc = 0, scale = 1) jnp_log(scipy_stats_gamma_cdf(x, a, loc, scale))

#' @rdname scipy_stats_gamma
#' @export
scipy_stats_gamma_sf <- function(x, a, loc = 0, scale = 1) {
  v <- st_args(x, a, loc, scale)
  y <- jnp_true_divide(jnp_subtract(v[[1L]], v[[3L]]), v[[4L]])
  jnp_where(y < 0, 1, scipy_special_gammaincc(v[[2L]], y))
}

#' @rdname scipy_stats_gamma
#' @export
scipy_stats_gamma_logsf <- function(x, a, loc = 0, scale = 1) jnp_log(scipy_stats_gamma_sf(x, a, loc, scale))

# ---- gennorm ------------------------------------------------------------------------------

#' Generalized normal distribution
#'
#' Ports of `jax.scipy.stats.gennorm`.
#' @param x Values.
#' @param beta Shape parameter.
#' @return An array.
#' @name scipy_stats_gennorm
NULL

#' @rdname scipy_stats_gennorm
#' @export
scipy_stats_gennorm_logpdf <- function(x, beta) {
  v <- st_args(x, beta)
  x <- v[[1L]]; beta <- v[[2L]]
  jnp_subtract(jnp_subtract(jnp_log(jnp_multiply(0.5, beta)), lax_lgamma(jnp_true_divide(1, beta))),
               jnp_power(jnp_abs(x), beta))
}

#' @rdname scipy_stats_gennorm
#' @export
scipy_stats_gennorm_cdf <- function(x, beta) {
  v <- st_args(x, beta)
  x <- v[[1L]]; beta <- v[[2L]]
  jnp_multiply(0.5, jnp_add(1, jnp_multiply(jnp_sign(x),
                                            lax_igamma(jnp_true_divide(1, beta), jnp_power(jnp_abs(x), beta)))))
}

#' @rdname scipy_stats_gennorm
#' @export
scipy_stats_gennorm_pdf <- function(x, beta) jnp_exp(scipy_stats_gennorm_logpdf(x, beta))

# ---- geom ---------------------------------------------------------------------------------

#' Geometric distribution
#'
#' Ports of `jax.scipy.stats.geom`.
#' @param k Outcomes.
#' @param p Success probability.
#' @param loc Location.
#' @return An array.
#' @name scipy_stats_geom
NULL

#' @rdname scipy_stats_geom
#' @export
scipy_stats_geom_logpmf <- function(k, p, loc = 0) {
  v <- st_args(k, p, loc)
  x <- jnp_subtract(v[[1L]], v[[3L]])
  log_probs <- jnp_add(scipy_special_xlog1py(jnp_subtract(x, 1), jnp_negative(v[[2L]])), jnp_log(v[[2L]]))
  jnp_where(x <= 0, -Inf, log_probs)
}

#' @rdname scipy_stats_geom
#' @export
scipy_stats_geom_pmf <- function(k, p, loc = 0) jnp_exp(scipy_stats_geom_logpmf(k, p, loc))

# ---- gumbel_l -----------------------------------------------------------------------------

#' Left skewed Gumbel distribution
#'
#' Ports of `jax.scipy.stats.gumbel_l`.
#' @param x Values.
#' @param p Probabilities.
#' @param loc Location.
#' @param scale Scale.
#' @return An array.
#' @name scipy_stats_gumbel_l
NULL

#' @rdname scipy_stats_gumbel_l
#' @export
scipy_stats_gumbel_l_logpdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  scale <- v[[3L]]
  z <- jnp_true_divide(jnp_subtract(v[[1L]], v[[2L]]), scale)
  log_pdf <- jnp_add(scipy_special_xlogy(-1, scale), jnp_subtract(z, jnp_exp(z)))
  jnp_where(scale > 0, log_pdf, NaN)
}

#' @rdname scipy_stats_gumbel_l
#' @export
scipy_stats_gumbel_l_pdf <- function(x, loc = 0, scale = 1) jnp_exp(scipy_stats_gumbel_l_logpdf(x, loc, scale))

#' @rdname scipy_stats_gumbel_l
#' @export
scipy_stats_gumbel_l_logcdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  scale <- v[[3L]]
  z <- jnp_true_divide(jnp_subtract(v[[1L]], v[[2L]]), scale)
  log_cdf <- jnp_log(jnp_negative(jnp_expm1(jnp_negative(jnp_exp(z)))))
  jnp_where(scale > 0, log_cdf, NaN)
}

#' @rdname scipy_stats_gumbel_l
#' @export
scipy_stats_gumbel_l_cdf <- function(x, loc = 0, scale = 1) jnp_exp(scipy_stats_gumbel_l_logcdf(x, loc, scale))

#' @rdname scipy_stats_gumbel_l
#' @export
scipy_stats_gumbel_l_ppf <- function(p, loc = 0, scale = 1) {
  v <- st_args(p, loc, scale)
  p <- v[[1L]]
  ok <- (p > 0) & (p < 1)
  t <- jnp_multiply(v[[3L]], jnp_log(scipy_special_xlog1py(-1, jnp_negative(p))))
  jnp_where(ok, jnp_add(v[[2L]], t), NaN)
}

#' @rdname scipy_stats_gumbel_l
#' @export
scipy_stats_gumbel_l_sf <- function(x, loc = 0, scale = 1) jnp_exp(scipy_stats_gumbel_l_logsf(x, loc, scale))

#' @rdname scipy_stats_gumbel_l
#' @export
scipy_stats_gumbel_l_logsf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  scale <- v[[3L]]
  z <- jnp_true_divide(jnp_subtract(v[[1L]], v[[2L]]), scale)
  jnp_where(scale > 0, jnp_negative(jnp_exp(z)), NaN)
}

# ---- gumbel_r -----------------------------------------------------------------------------

#' Right skewed Gumbel distribution
#'
#' Ports of `jax.scipy.stats.gumbel_r`.
#' @param x Values.
#' @param p Probabilities.
#' @param loc Location.
#' @param scale Scale.
#' @return An array.
#' @name scipy_stats_gumbel_r
NULL

#' @rdname scipy_stats_gumbel_r
#' @export
scipy_stats_gumbel_r_logpdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  scale <- v[[3L]]
  z <- jnp_true_divide(jnp_subtract(v[[1L]], v[[2L]]), scale)
  t2 <- jnp_negative(jnp_add(z, jnp_exp(jnp_negative(z))))
  jnp_where(scale > 0, jnp_add(scipy_special_xlogy(-1, scale), t2), NaN)
}

#' @rdname scipy_stats_gumbel_r
#' @export
scipy_stats_gumbel_r_pdf <- function(x, loc = 0, scale = 1) jnp_exp(scipy_stats_gumbel_r_logpdf(x, loc, scale))

#' @rdname scipy_stats_gumbel_r
#' @export
scipy_stats_gumbel_r_logcdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  scale <- v[[3L]]
  z <- jnp_true_divide(jnp_subtract(v[[1L]], v[[2L]]), scale)
  jnp_where(scale > 0, jnp_negative(jnp_exp(jnp_negative(z))), NaN)
}

#' @rdname scipy_stats_gumbel_r
#' @export
scipy_stats_gumbel_r_cdf <- function(x, loc = 0, scale = 1) jnp_exp(scipy_stats_gumbel_r_logcdf(x, loc, scale))

#' @rdname scipy_stats_gumbel_r
#' @export
scipy_stats_gumbel_r_ppf <- function(p, loc = 0, scale = 1) {
  v <- st_args(p, loc, scale)
  p <- v[[1L]]
  ok <- (p > 0) & (p < 1)
  t <- jnp_multiply(v[[3L]], jnp_log(scipy_special_xlogy(-1, p)))
  jnp_where(ok, jnp_subtract(v[[2L]], t), NaN)
}

#' @rdname scipy_stats_gumbel_r
#' @export
scipy_stats_gumbel_r_sf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  scale <- v[[3L]]
  neg_z <- jnp_true_divide(jnp_subtract(v[[2L]], v[[1L]]), scale)
  t1 <- jnp_exp(jnp_negative(jnp_exp(neg_z)))
  jnp_where(scale > 0, jnp_subtract(1, t1), NaN)
}

#' @rdname scipy_stats_gumbel_r
#' @export
scipy_stats_gumbel_r_logsf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  scale <- v[[3L]]
  neg_z <- jnp_true_divide(jnp_subtract(v[[2L]], v[[1L]]), scale)
  jnp_where(scale > 0, st_log1mexp(jnp_exp(neg_z)), NaN)
}

# ---- laplace ------------------------------------------------------------------------------

#' Laplace distribution
#'
#' Ports of `jax.scipy.stats.laplace`.
#' @param x Values.
#' @param loc Location.
#' @param scale Scale.
#' @return An array.
#' @name scipy_stats_laplace
NULL

#' @rdname scipy_stats_laplace
#' @export
scipy_stats_laplace_logpdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  linear_term <- jnp_true_divide(jnp_abs(jnp_subtract(v[[1L]], v[[2L]])), v[[3L]])
  jnp_negative(jnp_add(linear_term, jnp_log(jnp_multiply(2, v[[3L]]))))
}

#' @rdname scipy_stats_laplace
#' @export
scipy_stats_laplace_pdf <- function(x, loc = 0, scale = 1) jnp_exp(scipy_stats_laplace_logpdf(x, loc, scale))

#' @rdname scipy_stats_laplace
#' @export
scipy_stats_laplace_cdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  diff <- jnp_true_divide(jnp_subtract(v[[1L]], v[[2L]]), v[[3L]])
  half_exp <- jnp_multiply(0.5, jnp_exp(jnp_negative(jnp_abs(diff))))
  jnp_where(diff < 0, half_exp, jnp_subtract(1, half_exp))
}

# ---- logistic -----------------------------------------------------------------------------

#' Logistic distribution
#'
#' Ports of `jax.scipy.stats.logistic`.
#' @param x Values (probabilities for `ppf` and `isf`, as in JAX).
#' @param loc Location.
#' @param scale Scale.
#' @return An array.
#' @name scipy_stats_logistic
NULL

#' @rdname scipy_stats_logistic
#' @export
scipy_stats_logistic_logpdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  x <- jnp_true_divide(jnp_subtract(v[[1L]], v[[2L]]), v[[3L]])
  half_x <- jnp_true_divide(x, 2)
  jnp_subtract(jnp_multiply(-2, jnp_logaddexp(half_x, jnp_negative(half_x))), jnp_log(v[[3L]]))
}

#' @rdname scipy_stats_logistic
#' @export
scipy_stats_logistic_pdf <- function(x, loc = 0, scale = 1) jnp_exp(scipy_stats_logistic_logpdf(x, loc, scale))

#' @rdname scipy_stats_logistic
#' @export
scipy_stats_logistic_ppf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  jnp_add(jnp_multiply(scipy_special_logit(v[[1L]]), v[[3L]]), v[[2L]])
}

#' @rdname scipy_stats_logistic
#' @export
scipy_stats_logistic_sf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  scipy_special_expit(jnp_negative(jnp_true_divide(jnp_subtract(v[[1L]], v[[2L]]), v[[3L]])))
}

#' @rdname scipy_stats_logistic
#' @export
scipy_stats_logistic_isf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  jnp_add(jnp_multiply(jnp_negative(scipy_special_logit(v[[1L]])), v[[3L]]), v[[2L]])
}

#' @rdname scipy_stats_logistic
#' @export
scipy_stats_logistic_cdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  scipy_special_expit(jnp_true_divide(jnp_subtract(v[[1L]], v[[2L]]), v[[3L]]))
}

# ---- multinomial --------------------------------------------------------------------------

#' Multinomial distribution
#'
#' Ports of `jax.scipy.stats.multinomial`.
#' @param x Counts (integer typed), categories along the last axis.
#' @param n Number of trials (integer typed).
#' @param p Category probabilities.
#' @return An array.
#' @name scipy_stats_multinomial
NULL

#' @rdname scipy_stats_multinomial
#' @export
scipy_stats_multinomial_logpmf <- function(x, n, p) {
  p <- promote_dtypes_inexact(list(p))[[1L]]
  xn <- promote_dtypes_numeric(list(x, n))
  x <- xn[[1L]]; n <- xn[[2L]]
  if (!is_int_dtype(aval_of(x)$dtype))
    jax_error("x and n must be of integer type; got x.dtype=", aval_of(x)$dtype, ", n.dtype=",
              aval_of(n)$dtype, class = "jax_value_error")
  in_support <- jnp_equal(jnp_sum(x, axis = -1L, dtype = aval_of(x)$dtype), n)
  pdt <- aval_of(p)$dtype
  x <- jnp_astype(x, pdt)
  n <- jnp_astype(n, pdt)
  logprobs <- jnp_add(scipy_special_gammaln(jnp_add(n, 1)),
                      jnp_sum(jnp_subtract(scipy_special_xlogy(x, p), scipy_special_gammaln(jnp_add(x, 1))),
                              axis = -1L))
  jnp_where(in_support, logprobs, -Inf)
}

#' @rdname scipy_stats_multinomial
#' @export
scipy_stats_multinomial_pmf <- function(x, n, p) jnp_exp(scipy_stats_multinomial_logpmf(x, n, p))

# ---- multivariate_normal ------------------------------------------------------------------

#' Multivariate normal distribution
#'
#' Ports of `jax.scipy.stats.multivariate_normal`. `cov` may be a scalar (isotropic) or a
#' (batch of) covariance matrices.
#' @param x Values, the event dimension last.
#' @param mean Mean vector.
#' @param cov Covariance.
#' @param allow_singular Not supported (must be `NULL`), as in JAX.
#' @return An array.
#' @name scipy_stats_multivariate_normal
NULL

#' @rdname scipy_stats_multivariate_normal
#' @export
scipy_stats_multivariate_normal_logpdf <- function(x, mean, cov, allow_singular = NULL) {
  if (!is.null(allow_singular))
    jax_error("allow_singular argument of multivariate_normal.logpdf", class = "jax_not_implemented_error")
  v <- promote_dtypes_inexact(list(x, mean, cov))
  x <- v[[1L]]; mean <- v[[2L]]; cov <- v[[3L]]
  ms <- aval_of(mean)$shape
  cs <- aval_of(cov)$shape
  if (!length(ms)) {
    return(jnp_subtract(jnp_true_divide(jnp_multiply(-0.5, jnp_square(jnp_subtract(x, mean))), cov),
                        jnp_multiply(0.5, jnp_add(log(2 * pi), jnp_log(cov)))))
  }
  n <- ms[length(ms)]
  y <- jnp_subtract(x, mean)
  if (!length(cs)) {
    return(jnp_subtract(jnp_true_divide(jnp_multiply(-0.5, jnp_einsum("...i,...i->...", y, y)), cov),
                        jnp_multiply(n / 2, jnp_add(log(2 * pi), jnp_log(cov)))))
  }
  if (length(cs) < 2L || !identical(cs[length(cs) - c(1L, 0L)], c(n, n)))
    jax_error("multivariate_normal.logpdf got incompatible shapes", class = "jax_value_error")
  L <- lax_linalg_cholesky(cov)
  solve <- jnp_vectorize(function(a, b) lax_linalg_triangular_solve(a, b, lower = TRUE, transpose_a = TRUE),
                         signature = "(n,n),(n)->(n)")
  y <- solve(L, y)
  jnp_subtract(jnp_subtract(jnp_multiply(-0.5, jnp_einsum("...i,...i->...", y, y)), n / 2 * log(2 * pi)),
               jnp_sum(jnp_log(jnp_diagonal(L, axis1 = -1L, axis2 = -2L)), axis = -1L))
}

#' @rdname scipy_stats_multivariate_normal
#' @export
scipy_stats_multivariate_normal_pdf <- function(x, mean, cov) {
  jnp_exp(scipy_stats_multivariate_normal_logpdf(x, mean, cov))
}

# ---- nbinom -------------------------------------------------------------------------------

#' Negative binomial distribution
#'
#' Ports of `jax.scipy.stats.nbinom`.
#' @param k Outcomes.
#' @param n Number of successes.
#' @param p Success probability.
#' @param loc Location.
#' @return An array.
#' @name scipy_stats_nbinom
NULL

#' @rdname scipy_stats_nbinom
#' @export
scipy_stats_nbinom_logpmf <- function(k, n, p, loc = 0) {
  v <- st_args(k, n, p, loc)
  k <- v[[1L]]; n <- v[[2L]]; p <- v[[3L]]; loc <- v[[4L]]
  y <- jnp_subtract(k, loc)
  comb_term <- jnp_subtract(jnp_subtract(scipy_special_gammaln(jnp_add(y, n)), scipy_special_gammaln(n)),
                            scipy_special_gammaln(jnp_add(y, 1)))
  log_linear_term <- jnp_add(scipy_special_xlogy(n, p), scipy_special_xlogy(y, jnp_subtract(1, p)))
  jnp_where(k < loc, -Inf, jnp_add(comb_term, log_linear_term))
}

#' @rdname scipy_stats_nbinom
#' @export
scipy_stats_nbinom_pmf <- function(k, n, p, loc = 0) jnp_exp(scipy_stats_nbinom_logpmf(k, n, p, loc))

# ---- pareto -------------------------------------------------------------------------------

#' Pareto distribution
#'
#' Ports of `jax.scipy.stats.pareto`.
#' @param x Values.
#' @param q Probabilities.
#' @param b Shape parameter.
#' @param loc Location.
#' @param scale Scale.
#' @return An array.
#' @name scipy_stats_pareto
NULL

st_pareto_scaled <- function(v) jnp_true_divide(jnp_subtract(v[[1L]], v[[3L]]), v[[4L]])

#' @rdname scipy_stats_pareto
#' @export
scipy_stats_pareto_logpdf <- function(x, b, loc = 0, scale = 1) {
  v <- st_args(x, b, loc, scale)
  b <- v[[2L]]
  scaled_x <- st_pareto_scaled(v)
  normalize_term <- jnp_log(jnp_true_divide(v[[4L]], b))
  log_probs <- jnp_negative(jnp_add(normalize_term, jnp_multiply(jnp_add(b, 1), jnp_log(scaled_x))))
  jnp_where(scaled_x < 1, -Inf, log_probs)
}

#' @rdname scipy_stats_pareto
#' @export
scipy_stats_pareto_pdf <- function(x, b, loc = 0, scale = 1) jnp_exp(scipy_stats_pareto_logpdf(x, b, loc, scale))

#' @rdname scipy_stats_pareto
#' @export
scipy_stats_pareto_cdf <- function(x, b, loc = 0, scale = 1) {
  v <- st_args(x, b, loc, scale)
  scaled_x <- st_pareto_scaled(v)
  cdf <- jnp_subtract(1, jnp_power(scaled_x, jnp_negative(v[[2L]])))
  jnp_where(scaled_x < 1, jnp_zeros_like(cdf), cdf)
}

#' @rdname scipy_stats_pareto
#' @export
scipy_stats_pareto_logcdf <- function(x, b, loc = 0, scale = 1) {
  v <- st_args(x, b, loc, scale)
  scaled_x <- st_pareto_scaled(v)
  val <- jnp_log1p(jnp_negative(jnp_power(scaled_x, jnp_negative(v[[2L]]))))
  jnp_where(scaled_x < 1, -Inf, val)
}

#' @rdname scipy_stats_pareto
#' @export
scipy_stats_pareto_logsf <- function(x, b, loc = 0, scale = 1) {
  v <- st_args(x, b, loc, scale)
  scaled_x <- st_pareto_scaled(v)
  val <- jnp_negative(jnp_multiply(v[[2L]], jnp_log(scaled_x)))
  jnp_where(scaled_x < 1, jnp_zeros_like(val), val)
}

#' @rdname scipy_stats_pareto
#' @export
scipy_stats_pareto_sf <- function(x, b, loc = 0, scale = 1) jnp_exp(scipy_stats_pareto_logsf(x, b, loc, scale))

#' @rdname scipy_stats_pareto
#' @export
scipy_stats_pareto_ppf <- function(q, b, loc = 0, scale = 1) {
  v <- st_args(q, b, loc, scale)
  q <- v[[1L]]
  val <- jnp_add(v[[3L]], jnp_multiply(v[[4L]], jnp_power(jnp_subtract(1, q),
                                                          jnp_negative(jnp_true_divide(1, v[[2L]])))))
  jnp_where(jnp_isnan(q) | (q < 0) | (q > 1), NaN, val)
}

# ---- poisson ------------------------------------------------------------------------------

#' Poisson distribution
#'
#' Ports of `jax.scipy.stats.poisson`.
#' @param k Outcomes.
#' @param mu Rate.
#' @param loc Location.
#' @return An array.
#' @name scipy_stats_poisson
NULL

#' @rdname scipy_stats_poisson
#' @export
scipy_stats_poisson_logpmf <- function(k, mu, loc = 0) {
  v <- st_args(k, mu, loc)
  mu <- v[[2L]]
  x <- jnp_subtract(v[[1L]], v[[3L]])
  log_probs <- jnp_subtract(jnp_subtract(scipy_special_xlogy(x, mu), scipy_special_gammaln(jnp_add(x, 1))), mu)
  jnp_where((x < 0) | (jnp_round(x) != x), -Inf, log_probs)
}

#' @rdname scipy_stats_poisson
#' @export
scipy_stats_poisson_pmf <- function(k, mu, loc = 0) jnp_exp(scipy_stats_poisson_logpmf(k, mu, loc))

#' @rdname scipy_stats_poisson
#' @export
scipy_stats_poisson_cdf <- function(k, mu, loc = 0) {
  v <- st_args(k, mu, loc)
  x <- jnp_subtract(v[[1L]], v[[3L]])
  p <- scipy_special_gammaincc(jnp_floor(jnp_add(1, x)), v[[2L]])
  jnp_where(x < 0, jnp_zeros_like(p), p)
}

st_poisson_entropy_sum <- function(mu, max_k, upper) {
  dt <- aval_of(mu)$dtype
  k <- jnp_expand_dims(jnp_arange(max_k, dtype = dt), 1L)
  log_probs <- scipy_stats_poisson_logpmf(k, mu, 0)
  mask <- k < jnp_expand_dims(jnp_astype(jnp_ceil(upper), dt), 0L)
  jnp_sum(jnp_where(mask, jnp_multiply(jnp_negative(jnp_exp(log_probs)), log_probs), 0), axis = 0L)
}

#' @rdname scipy_stats_poisson
#' @export
scipy_stats_poisson_entropy <- function(mu, loc = 0) {
  v <- promote_dtypes_inexact(list(mu, loc))
  result_shape <- broadcast_shapes_impl(list(aval_of(v[[1L]])$shape, aval_of(v[[2L]])$shape))
  mu_flat <- jnp_ravel(v[[1L]])
  small_mu <- jnp_where((mu_flat == 0) | (mu_flat >= 10), 1, mu_flat)
  medium_mu <- jnp_where((mu_flat < 10) | (mu_flat >= 100), 10, mu_flat)
  large_mu <- jnp_where(mu_flat < 100, 100, mu_flat)
  large <- jnp_subtract(jnp_multiply(0.5, jnp_log(jnp_multiply(2 * pi * exp(1), large_mu))),
                        jnp_true_divide(1, jnp_multiply(12, large_mu)))
  result <- jnp_where(mu_flat == 0, jnp_zeros_like(mu_flat),
              jnp_where(mu_flat < 10, st_poisson_entropy_sum(small_mu, 35L, jnp_add(small_mu, 20)),
                jnp_where(mu_flat < 100,
                          st_poisson_entropy_sum(medium_mu, 250L,
                                                 jnp_add(jnp_add(medium_mu, jnp_multiply(10, jnp_sqrt(medium_mu))), 20)),
                          large)))
  jnp_broadcast_to(jnp_reshape(result, aval_of(v[[1L]])$shape), result_shape)
}

# ---- t ------------------------------------------------------------------------------------

#' Student t distribution
#'
#' Ports of `jax.scipy.stats.t`.
#' @param x Values.
#' @param df Degrees of freedom.
#' @param loc Location.
#' @param scale Scale.
#' @return An array.
#' @name scipy_stats_t
NULL

#' @rdname scipy_stats_t
#' @export
scipy_stats_t_logpdf <- function(x, df, loc = 0, scale = 1) {
  v <- st_args(x, df, loc, scale)
  df <- v[[2L]]; scale <- v[[4L]]
  scaled_x <- jnp_true_divide(jnp_subtract(v[[1L]], v[[3L]]), scale)
  df_over_two <- jnp_true_divide(df, 2)
  df_plus_one_over_two <- jnp_add(df_over_two, 0.5)
  normalize_term_const <- jnp_multiply(jnp_multiply(scale, scale), pi)
  normalize_term_tmp <- jnp_true_divide(jnp_log(jnp_multiply(normalize_term_const, df)), 2)
  normalize_term <- jnp_subtract(jnp_add(lax_lgamma(df_over_two), normalize_term_tmp),
                                 lax_lgamma(df_plus_one_over_two))
  quadratic <- jnp_true_divide(jnp_multiply(scaled_x, scaled_x), df)
  jnp_negative(jnp_add(normalize_term, jnp_multiply(df_plus_one_over_two, jnp_log1p(quadratic))))
}

#' @rdname scipy_stats_t
#' @export
scipy_stats_t_pdf <- function(x, df, loc = 0, scale = 1) jnp_exp(scipy_stats_t_logpdf(x, df, loc, scale))

# ---- truncnorm ----------------------------------------------------------------------------

#' Truncated normal distribution
#'
#' Ports of `jax.scipy.stats.truncnorm`. `a` and `b` are the truncation points in
#' standardized units.
#' @param x Values.
#' @param a,b Lower and upper truncation points.
#' @param loc Location.
#' @param scale Scale.
#' @return An array.
#' @name scipy_stats_truncnorm
NULL

st_log_diff <- function(x, y) {
  scipy_special_logsumexp(jnp_stack(list(x, y)),
                          b = jnp_stack(list(jnp_ones_like(x), jnp_negative(jnp_ones_like(y)))), axis = 0L)
}

st_log_gauss_mass <- function(a, b) {
  ab <- jnp_broadcast_arrays(jnp_array(a), jnp_array(b))
  a <- ab[[1L]]; b <- ab[[2L]]
  case_left <- b <= 0
  case_right <- a > 0
  case_central <- !(case_left | case_right)
  a_tail <- jnp_where(case_right, jnp_negative(b), a)
  b_tail <- jnp_where(case_right, jnp_negative(a), b)
  mass_tail <- st_log_diff(scipy_special_log_ndtr(b_tail), scipy_special_log_ndtr(a_tail))
  mass_central <- jnp_log1p(jnp_subtract(jnp_negative(scipy_special_ndtr(a)),
                                         scipy_special_ndtr(jnp_negative(b))))
  jnp_where(case_central, mass_central, mass_tail)
}

#' @rdname scipy_stats_truncnorm
#' @export
scipy_stats_truncnorm_logpdf <- function(x, a, b, loc = 0, scale = 1) {
  v <- st_args(x, a, b, loc, scale)
  x <- v[[1L]]; a <- v[[2L]]; b <- v[[3L]]; loc <- v[[4L]]; scale <- v[[5L]]
  val <- jnp_subtract(scipy_stats_norm_logpdf(x, loc, scale), st_log_gauss_mass(a, b))
  x_scaled <- jnp_true_divide(jnp_subtract(x, loc), scale)
  val <- jnp_where((x_scaled < a) | (x_scaled > b), -Inf, val)
  jnp_where(a < b, val, NaN)
}

#' @rdname scipy_stats_truncnorm
#' @export
scipy_stats_truncnorm_pdf <- function(x, a, b, loc = 0, scale = 1) {
  jnp_exp(scipy_stats_truncnorm_logpdf(x, a, b, loc, scale))
}

#' @rdname scipy_stats_truncnorm
#' @export
scipy_stats_truncnorm_logsf <- function(x, a, b, loc = 0, scale = 1) {
  v <- st_args(x, a, b, loc, scale)
  scipy_stats_truncnorm_logcdf(jnp_negative(v[[1L]]), jnp_negative(v[[3L]]), jnp_negative(v[[2L]]),
                               jnp_negative(v[[4L]]), v[[5L]])
}

#' @rdname scipy_stats_truncnorm
#' @export
scipy_stats_truncnorm_sf <- function(x, a, b, loc = 0, scale = 1) {
  jnp_exp(scipy_stats_truncnorm_logsf(x, a, b, loc, scale))
}

#' @rdname scipy_stats_truncnorm
#' @export
scipy_stats_truncnorm_logcdf <- function(x, a, b, loc = 0, scale = 1) {
  v <- st_args(x, a, b, loc, scale)
  xab <- jnp_broadcast_arrays(v[[1L]], v[[2L]], v[[3L]])
  a <- xab[[2L]]; b <- xab[[3L]]
  x <- jnp_true_divide(jnp_subtract(xab[[1L]], v[[4L]]), v[[5L]])
  lgm_ab <- st_log_gauss_mass(a, b)
  logcdf <- jnp_subtract(st_log_gauss_mass(a, x), lgm_ab)
  logsf <- jnp_subtract(st_log_gauss_mass(x, b), lgm_ab)
  logcdf <- jnp_select(list(x >= b, x <= a, logcdf > -0.1, x > a),
                       list(jnp_zeros_like(logcdf), jnp_full_like(logcdf, -Inf),
                            jnp_log1p(jnp_negative(jnp_exp(logsf))), logcdf), NaN)
  jnp_where(a < b, logcdf, NaN)
}

#' @rdname scipy_stats_truncnorm
#' @export
scipy_stats_truncnorm_cdf <- function(x, a, b, loc = 0, scale = 1) {
  jnp_exp(scipy_stats_truncnorm_logcdf(x, a, b, loc, scale))
}

# ---- uniform ------------------------------------------------------------------------------

#' Uniform distribution
#'
#' Ports of `jax.scipy.stats.uniform` on `[loc, loc + scale]`.
#' @param x Values.
#' @param q Probabilities.
#' @param loc Lower bound.
#' @param scale Width.
#' @return An array.
#' @name scipy_stats_uniform
NULL

#' @rdname scipy_stats_uniform
#' @export
scipy_stats_uniform_logpdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  z <- jnp_true_divide(jnp_subtract(v[[1L]], v[[2L]]), v[[3L]])
  log_probs <- jnp_negative(jnp_log(v[[3L]]))
  jnp_select(list((z < 0) | (z > 1), (z >= 0) & (z <= 1)),
             list(jnp_full_like(log_probs, -Inf), log_probs), NaN)
}

#' @rdname scipy_stats_uniform
#' @export
scipy_stats_uniform_pdf <- function(x, loc = 0, scale = 1) jnp_exp(scipy_stats_uniform_logpdf(x, loc, scale))

#' @rdname scipy_stats_uniform
#' @export
scipy_stats_uniform_cdf <- function(x, loc = 0, scale = 1) {
  v <- st_args(x, loc, scale)
  jnp_clip(jnp_true_divide(jnp_subtract(v[[1L]], v[[2L]]), v[[3L]]), 0, 1)
}

#' @rdname scipy_stats_uniform
#' @export
scipy_stats_uniform_ppf <- function(q, loc = 0, scale = 1) {
  v <- st_args(q, loc, scale)
  q <- v[[1L]]
  jnp_where(jnp_isnan(q) | (q < 0) | (q > 1), NaN, jnp_add(v[[2L]], jnp_multiply(v[[3L]], q)))
}

# ---- vonmises and wrapcauchy --------------------------------------------------------------

#' Circular distributions
#'
#' Ports of `jax.scipy.stats.vonmises` and `jax.scipy.stats.wrapcauchy`.
#' @param x Angles.
#' @param kappa Von Mises concentration.
#' @param c Wrapped Cauchy shape parameter in `(0, 1)`.
#' @return An array.
#' @name scipy_stats_circular
NULL

#' @rdname scipy_stats_circular
#' @export
scipy_stats_vonmises_logpdf <- function(x, kappa) {
  v <- st_args(x, kappa)
  x <- v[[1L]]; kappa <- v[[2L]]
  val <- jnp_subtract(jnp_subtract(jnp_multiply(kappa, jnp_subtract(jnp_cos(x), jnp_sign(kappa))),
                                   log(2 * pi)),
                      jnp_log(lax_bessel_i0e(kappa)))
  jnp_where(kappa >= 0, val, NaN)
}

#' @rdname scipy_stats_circular
#' @export
scipy_stats_vonmises_pdf <- function(x, kappa) jnp_exp(scipy_stats_vonmises_logpdf(x, kappa))

#' @rdname scipy_stats_circular
#' @export
scipy_stats_wrapcauchy_logpdf <- function(x, c) {
  v <- st_args(x, c)
  x <- v[[1L]]; c <- v[[2L]]
  val <- jnp_subtract(jnp_subtract(jnp_log(jnp_subtract(1, jnp_multiply(c, c))), log(2 * pi)),
                      jnp_log(jnp_subtract(jnp_add(1, jnp_multiply(c, c)),
                                           jnp_multiply(jnp_multiply(2, c), jnp_cos(x)))))
  jnp_where((c > 0) & (c < 1), jnp_where((x >= 0) & (x <= 2 * pi), val, -Inf), NaN)
}

#' @rdname scipy_stats_circular
#' @export
scipy_stats_wrapcauchy_pdf <- function(x, c) jnp_exp(scipy_stats_wrapcauchy_logpdf(x, c))

# ---- mode, rankdata, sem ------------------------------------------------------------------

st_check_nan_policy <- function(nan_policy, what) {
  if (!nan_policy %in% c("propagate", "omit", "raise"))
    jax_error("Illegal nan_policy value '", nan_policy,
              "'; expected one of {'propagate', 'omit', 'raise'}", class = "jax_value_error")
  if (nan_policy == "omit")
    jax_error("Logic for `nan_policy` of omit is not implemented", class = "jax_not_implemented_error")
  if (nan_policy == "raise")
    jax_error("In order to best JIT compile `", what, "`, we cannot know whether `x` contains nans. ",
              "Please check if nans exist in `x` outside of the `", what, "` function.",
              class = "jax_not_implemented_error")
}

#' Summary statistics
#'
#' Ports of `jax.scipy.stats.mode`, `jax.scipy.stats.rankdata` and
#' `jax.scipy.stats.sem`. Axes are 0-based.
#' @param a Array.
#' @param axis Axis (0-based) or `NULL` for the flattened array.
#' @param nan_policy Only `"propagate"` (plus `"omit"` for `sem`), as in JAX.
#' @param keepdims Keep the reduced axis with size one.
#' @param method Ranking method: `"average"`, `"min"`, `"max"`, `"dense"` or `"ordinal"`.
#' @param ddof Delta degrees of freedom.
#' @return `scipy_stats_mode()` returns a list with `mode` and `count`;
#'   `scipy_stats_rankdata()` and `scipy_stats_sem()` return arrays.
#' @name scipy_stats_summary
NULL

#' @rdname scipy_stats_summary
#' @export
scipy_stats_mode <- function(a, axis = 0L, nan_policy = "propagate", keepdims = FALSE) {
  x <- jnp_atleast_1d(a)
  st_check_nan_policy(nan_policy, "mode")
  input_shape <- aval_of(x)$shape
  r <- length(input_shape)
  if (!is.null(axis)) axis <- norm_axis(axis, r)
  output_shape <- if (isTRUE(keepdims)) {
    if (is.null(axis)) rep(1L, r) else replace(input_shape, axis + 1L, 1L)
  } else if (is.null(axis)) integer(0) else input_shape[-(axis + 1L)]
  if (is.null(axis)) {
    axis <- 0L
    x <- jnp_ravel(x)
  }
  helper <- function(x) {
    n <- prod_int(aval_of(x)$shape)
    if (n == 0L) return(list(jnp_array(NaN, dtype = default_float()), jnp_array(0, dtype = default_float())))
    vc <- jnp_unique(x, return_counts = TRUE, size = n)
    list(vc[[1L]][jnp_argmax(vc[[2L]])], jnp_max(vc[[2L]]))
  }
  x <- jnp_moveaxis(x, axis, 0L)
  s <- aval_of(x)$shape
  x <- jnp_reshape(x, c(s[1L], prod_int(drop_first(s, 1L))))
  out <- vmap(helper, in_axes = 1L)(x)
  list(mode = jnp_reshape(out[[1L]], output_shape), count = jnp_reshape(out[[2L]], output_shape))
}

#' @rdname scipy_stats_summary
#' @export
scipy_stats_rankdata <- function(a, method = "average", axis = NULL, nan_policy = "propagate") {
  st_check_nan_policy(nan_policy, "rankdata")
  if (!method %in% c("average", "min", "max", "dense", "ordinal"))
    jax_error("unknown method '", method, "'", class = "jax_value_error")
  if (!is.null(axis))
    return(jnp_apply_along_axis(function(v) scipy_stats_rankdata(v, method), axis, a))
  a <- jnp_ravel(a)
  out_dtype <- default_float()
  rank1 <- function(a) {
    n <- prod_int(aval_of(a)$shape)
    sk <- lax_sort_key_val(a, jnp_arange(n))
    arr <- sk[[1L]]; sorter <- sk[[2L]]
    inv <- at_set(jnp_empty_like(sorter), sorter, value = jnp_arange(n, dtype = aval_of(sorter)$dtype))
    if (method == "ordinal") return(jnp_astype(jnp_add(inv, 1L), out_dtype))
    obs <- jnp_concatenate(list(jnp_array(array(TRUE, dim = 1L)),
                                arr[sl(1L, NULL)] != arr[sl(NULL, -1L)]))
    dense <- jnp_cumsum(obs)[inv]
    if (method == "dense") return(jnp_astype(dense, out_dtype))
    count <- jnp_astype(jnp_nonzero(obs, size = n + 1L, fill_value = n)[[1L]], out_dtype)
    switch(method,
      max = count[dense],
      min = jnp_add(count[jnp_subtract(dense, 1L)], 1),
      average = jnp_multiply(0.5, jnp_add(jnp_add(count[dense], count[jnp_subtract(dense, 1L)]), 1)))
  }
  lax_cond(jnp_any(jnp_isnan(a)), function(a) jnp_full_like(a, NaN, dtype = out_dtype), rank1, a)
}

#' @rdname scipy_stats_summary
#' @export
scipy_stats_sem <- function(a, axis = 0L, ddof = 1L, nan_policy = "propagate", keepdims = FALSE) {
  b <- promote_args_inexact(a)[[1L]]
  dt <- aval_of(b)$dtype
  shape <- aval_of(b)$shape
  if (nan_policy == "propagate") {
    if (length(axis) > 1L)
      jax_error("tuple indices must be integers or slices, not tuple", class = "jax_type_error")
    size <- if (is.null(axis)) prod_int(shape) else shape[norm_axis(axis, length(shape)) + 1L]
    return(jnp_true_divide(jnp_std(b, axis, ddof = ddof, keepdims = keepdims),
                           jnp_astype(jnp_sqrt(size), dt)))
  }
  if (nan_policy == "omit") {
    count <- jnp_sum(!jnp_isnan(b), axis = axis, keepdims = keepdims)
    return(jnp_true_divide(jnp_nanstd(b, axis, ddof = ddof, keepdims = keepdims),
                           jnp_astype(jnp_sqrt(count), dt)))
  }
  jax_error(nan_policy, " is not supported", class = "jax_value_error")
}

# ---- gaussian_kde -------------------------------------------------------------------------

#' Gaussian kernel density estimation
#'
#' Port of `jax.scipy.stats.gaussian_kde`. The result is a pytree with the fields
#' `neff`, `dataset`, `weights`, `covariance` and `inv_cov`, and the JAX methods are
#' reached with `$`: `kde$evaluate(points)` (also `kde$pdf`), `kde$logpdf(points)`,
#' `kde$integrate_gaussian(mean, cov)`, `kde$integrate_box_1d(low, high)`,
#' `kde$integrate_kde(other)`, `kde$resample(key, shape)`, and the properties `kde$d`
#' and `kde$n`. `integrate_box` and `set_bandwidth` raise errors, as in JAX.
#' @param dataset Data with shape `(d, n)` (a vector is one dimensional data).
#' @param bw_method `"scott"` (default), `"silverman"`, a scalar factor or a function of
#'   the kde object.
#' @param weights Optional weights of the data points.
#' @return An object of class `jax_gaussian_kde`.
#' @export
#' @examples
#' \dontrun{
#' kde <- scipy_stats_gaussian_kde(c(-1, 0, 0.5, 2))
#' kde$evaluate(c(0, 1))
#' }
scipy_stats_gaussian_kde <- function(dataset, bw_method = NULL, weights = NULL) {
  dataset <- jnp_atleast_2d(dataset)
  if (is_complex_dtype(aval_of(dataset)$dtype))
    jax_error("gaussian_kde does not support complex data", class = "jax_not_implemented_error")
  ds <- aval_of(dataset)$shape
  if (!(prod_int(ds) > 1L))
    jax_error("`dataset` input should have multiple elements.", class = "jax_value_error")
  d <- ds[1L]; n <- ds[2L]
  if (!is.null(weights)) {
    v <- promote_dtypes_inexact(list(dataset, weights))
    dataset <- v[[1L]]
    weights <- jnp_atleast_1d(v[[2L]])
    weights <- jnp_true_divide(weights, jnp_sum(weights))
    ws <- aval_of(weights)$shape
    if (length(ws) != 1L) jax_error("`weights` input should be one-dimensional.", class = "jax_value_error")
    if (ws[1L] != n) jax_error("`weights` input should be of length n", class = "jax_value_error")
  } else {
    dataset <- promote_dtypes_inexact(list(dataset))[[1L]]
    weights <- jnp_full(n, 1 / n, dtype = aval_of(dataset)$dtype)
  }
  neff <- jnp_true_divide(1, jnp_sum(jnp_power(weights, 2L)))
  kde <- st_new_kde(neff, dataset, weights, NULL, NULL)
  bw_method <- bw_method %||% "scott"
  factor <- if (identical(bw_method, "scott")) {
    jnp_power(neff, -1 / (d + 4))
  } else if (identical(bw_method, "silverman")) {
    jnp_power(jnp_true_divide(jnp_multiply(neff, d + 2), 4), -1 / (d + 4))
  } else if (is.function(bw_method)) {
    bw_method(kde)
  } else if ((is.numeric(bw_method) && length(bw_method) == 1L) ||
             (is_jax_value(bw_method) && !length(aval_of(bw_method)$shape))) {
    bw_method
  } else {
    jax_error("`bw_method` should be 'scott', 'silverman', a scalar, or a callable.",
              class = "jax_value_error")
  }
  data_covariance <- jnp_atleast_2d(jnp_cov(dataset, rowvar = TRUE, bias = FALSE, aweights = weights))
  data_inv_cov <- jnp_linalg_inv(data_covariance)
  f2 <- jnp_power(factor, 2L)
  st_new_kde(neff, dataset, weights, jnp_multiply(data_covariance, f2), jnp_true_divide(data_inv_cov, f2))
}

st_new_kde <- function(neff, dataset, weights, covariance, inv_cov) {
  structure(list(neff = neff, dataset = dataset, weights = weights, covariance = covariance,
                 inv_cov = inv_cov), class = "jax_gaussian_kde")
}

register_pytree_node("jax_gaussian_kde",
  function(x) list(children = unname(unclass(x)), aux = NULL),
  function(aux, children) do.call(st_new_kde, children))

st_kde_field <- function(kde, name) .subset2(kde, name)

#' @export
`$.jax_gaussian_kde` <- function(x, name) {
  fields <- c("neff", "dataset", "weights", "covariance", "inv_cov")
  if (name %in% fields) return(.subset2(x, name))
  switch(name,
    d = aval_of(.subset2(x, "dataset"))$shape[1L],
    n = aval_of(.subset2(x, "dataset"))$shape[2L],
    evaluate = , pdf = function(points) st_kde_evaluate(x, points, FALSE),
    logpdf = function(points) st_kde_evaluate(x, points, TRUE),
    integrate_gaussian = function(mean, cov) st_kde_integrate_gaussian(x, mean, cov),
    integrate_box_1d = function(low, high) st_kde_integrate_box_1d(x, low, high),
    integrate_kde = function(other) st_kde_integrate_kde(x, other),
    resample = function(key, shape = integer(0)) st_kde_resample(x, key, shape),
    integrate_box = function(low_bounds, high_bounds, maxpts = NULL)
      jax_error("only 1D box integrations are supported; use `integrate_box_1d`",
                class = "jax_not_implemented_error"),
    set_bandwidth = function(bw_method = NULL)
      jax_error("dynamically changing the bandwidth method is not supported",
                class = "jax_not_implemented_error"),
    NULL)
}

#' @export
print.jax_gaussian_kde <- function(x, ...) {
  cat("gaussian_kde(d=", x$d, ", n=", x$n, ")\n", sep = "")
  invisible(x)
}

st_kde_reshape_points <- function(kde, points) {
  if (is_complex_dtype(aval_of(points)$dtype))
    jax_error("gaussian_kde does not support complex coordinates", class = "jax_not_implemented_error")
  points <- jnp_atleast_2d(points)
  ps <- aval_of(points)$shape
  if (ps[1L] != kde$d) {
    if (ps[1L] == 1L && ps[2L] == kde$d) return(jnp_reshape(points, c(kde$d, 1L)))
    jax_error("points have dimension ", ps[1L], ", dataset has dimension ", kde$d,
              class = "jax_value_error")
  }
  points
}

st_kde_evaluate <- function(kde, points, in_log) {
  points <- st_kde_reshape_points(kde, points)
  res <- st_gaussian_kernel_eval(in_log, jnp_transpose(kde$dataset), jnp_expand_dims(kde$weights, 1L),
                                 jnp_transpose(points), kde$inv_cov)
  res[, 0L]
}

st_gaussian_kernel_eval <- function(in_log, points, values, xi, precision) {
  v <- promote_dtypes_inexact(list(points, values, xi, precision))
  points <- v[[1L]]; values <- v[[2L]]; xi <- v[[3L]]; precision <- v[[4L]]
  d <- aval_of(points)$shape[2L]
  if (aval_of(xi)$shape[2L] != d)
    jax_error("points and xi must have same trailing dim", class = "jax_value_error")
  if (!identical(aval_of(precision)$shape, c(d, d)))
    jax_error("precision matrix must match data dims", class = "jax_value_error")
  whitening <- scipy_linalg_cholesky(precision, lower = TRUE)
  points <- jnp_dot(points, whitening)
  xi <- jnp_dot(xi, whitening)
  log_norm <- jnp_subtract(jnp_sum(jnp_log(jnp_diag(whitening))), 0.5 * d * log(2 * pi))
  kernel <- function(x_test, x_train, y_train) {
    arg <- jnp_subtract(log_norm, jnp_multiply(0.5, jnp_sum(jnp_square(jnp_subtract(x_train, x_test)))))
    if (in_log) jnp_add(jnp_log(y_train), arg) else jnp_multiply(y_train, jnp_exp(arg))
  }
  reduce <- if (in_log) scipy_special_logsumexp else jnp_sum
  reduced <- function(x) reduce(vmap(kernel, in_axes = list(NULL, 0L, 0L))(x, points, values), axis = 0L)
  vmap(reduced)(xi)
}

st_gaussian_kernel_convolve <- function(chol, norm, target, weights, mean) {
  diff <- jnp_subtract(target, jnp_expand_dims(mean, 1L))
  alpha <- scipy_linalg_cho_solve(chol, diff)
  arg <- jnp_multiply(0.5, jnp_sum(jnp_multiply(diff, alpha), axis = 0L))
  jnp_multiply(norm, jnp_sum(jnp_multiply(jnp_exp(jnp_negative(arg)), weights)))
}

st_kde_norm <- function(chol, d) {
  jnp_true_divide(1, jnp_multiply(sqrt(2 * pi)^d, jnp_prod(jnp_diag(chol[[1L]]))))
}

st_kde_integrate_gaussian <- function(kde, mean, cov) {
  mean <- jnp_atleast_1d(jnp_squeeze(mean))
  cov <- jnp_atleast_2d(cov)
  d <- kde$d
  if (!identical(aval_of(mean)$shape, d)) jax_error("mean does not have dimension ", d, class = "jax_value_error")
  if (!identical(aval_of(cov)$shape, c(d, d)))
    jax_error("covariance does not have dimension ", d, class = "jax_value_error")
  chol <- scipy_linalg_cho_factor(jnp_add(kde$covariance, cov))
  st_gaussian_kernel_convolve(chol, st_kde_norm(chol, d), kde$dataset, kde$weights, mean)
}

st_kde_integrate_box_1d <- function(kde, low, high) {
  if (kde$d != 1L) jax_error("integrate_box_1d() only handles 1D pdfs", class = "jax_value_error")
  if (length(aval_of(low)$shape) || length(aval_of(high)$shape))
    jax_error("the limits of integration in integrate_box_1d must be scalars", class = "jax_value_error")
  sigma <- jnp_squeeze(jnp_sqrt(kde$covariance))
  low <- jnp_squeeze(jnp_true_divide(jnp_subtract(low, kde$dataset), sigma))
  high <- jnp_squeeze(jnp_true_divide(jnp_subtract(high, kde$dataset), sigma))
  jnp_sum(jnp_multiply(kde$weights, jnp_subtract(scipy_special_ndtr(high), scipy_special_ndtr(low))))
}

st_kde_integrate_kde <- function(kde, other) {
  if (other$d != kde$d) jax_error("KDEs are not the same dimensionality", class = "jax_value_error")
  chol <- scipy_linalg_cho_factor(jnp_add(kde$covariance, other$covariance))
  norm <- st_kde_norm(chol, kde$d)
  if (kde$n < other$n) { sm <- kde; lg <- other } else { sm <- other; lg <- kde }
  f <- function(m) st_gaussian_kernel_convolve(chol, norm, lg$dataset, lg$weights, m)
  jnp_sum(jnp_multiply(vmap(f, in_axes = 1L)(sm$dataset), sm$weights))
}

st_kde_resample <- function(kde, key, shape = integer(0)) {
  keys <- random_split(key)
  ind <- random_choice(keys[0L], kde$n, shape = shape, p = kde$weights)
  cov <- kde$covariance
  eps <- random_multivariate_normal(keys[1L], jnp_zeros(kde$d, dtype = aval_of(cov)$dtype), cov,
                                    shape = shape, dtype = aval_of(kde$dataset)$dtype)
  jnp_add(kde$dataset[, ind], jnp_transpose(eps))
}
