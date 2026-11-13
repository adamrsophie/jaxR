# Special functions

Ports of `jax.scipy.special`. Arguments broadcast with NumPy rules and
are promoted to an inexact dtype with JAX type promotion; functions with
custom derivative rules in JAX carry the same rules here.

## Usage

``` r
scipy_special_gammaln(x)

scipy_special_gammasgn(x)

scipy_special_loggamma(x)

scipy_special_gamma(x)

scipy_special_betaln(a, b)

scipy_special_factorial(n, exact = FALSE)

scipy_special_comb(N, k, repetition = FALSE)

scipy_special_beta(a, b)

scipy_special_betainc(a, b, x)

scipy_special_digamma(x)

scipy_special_gammainc(a, x)

scipy_special_gammaincc(a, x)

scipy_special_multigammaln(a, d)

scipy_special_poch(z, m)

scipy_special_polygamma(n, x)

scipy_special_zeta(x, q = NULL)

scipy_special_erf(x)

scipy_special_erfc(x)

scipy_special_erfinv(x)

scipy_special_logit(x)

scipy_special_expit(x)

scipy_special_xlogy(x, y)

scipy_special_xlog1py(x, y)

scipy_special_entr(x)

scipy_special_boxcox(x, lmbda)

scipy_special_boxcox1p(x, lmbda)

scipy_special_kl_div(p, q)

scipy_special_rel_entr(p, q)

scipy_special_logsumexp(
  a,
  axis = NULL,
  b = NULL,
  keepdims = FALSE,
  return_sign = FALSE,
  where = NULL
)

scipy_special_softmax(x, axis = NULL)

scipy_special_log_softmax(x, axis = NULL)

scipy_special_ndtr(x)

scipy_special_log_ndtr(x, series_order = 3L)

scipy_special_i0e(x)

scipy_special_i1e(x)

scipy_special_i0(x)

scipy_special_i1(x)

scipy_special_bessel_jn(z, v, n_iter = 50L)

scipy_special_dawsn(x)

scipy_special_wofz(z)

scipy_special_expi(x)

scipy_special_expn(n, x)

scipy_special_exp1(x)

scipy_special_sici(x)

scipy_special_fresnel(x)

scipy_special_owens_t(h, a)

scipy_special_spence(x)

scipy_special_bernoulli(n)

scipy_special_erfcx(x)

scipy_special_erfcinv(y)

scipy_special_ndtri(p)

scipy_special_hyp1f1(a, b, x)

scipy_special_hyp2f1(a, b, c, x)

scipy_special_lpmn(m, n, z)

scipy_special_lpmn_values(m, n, z, is_normalized)

scipy_special_sph_harm_y(n, m, theta, phi, diff_n = NULL, n_max = NULL)
```

## Arguments

- x, y, a, b, c, p, q, z, h:

  Arrays or R values.

- n, m:

  Orders (arrays; integer valued for `scipy_special_polygamma`).

- exact:

  Must be `FALSE`.

- N, k:

  Arrays for `scipy_special_comb`.

- repetition:

  Count combinations with repetition.

- d:

  Dimension of the multivariate gamma function (a concrete integer).

- lmbda:

  Box-Cox power.

- axis:

  0-based axis or axes; `NULL` means all axes.

- keepdims:

  Keep reduced dimensions as size one.

- return_sign:

  Also return the sign of the sum.

- where:

  Optional boolean mask of the elements to include.

- series_order:

  Number of terms (0 to 30) of the asymptotic series used by
  `scipy_special_log_ndtr` for very negative arguments.

- v:

  Highest order computed by `scipy_special_bessel_jn`.

- n_iter:

  Number of backward recurrence steps.

- is_normalized:

  Whether the Legendre functions are normalized for spherical harmonics.

- theta, phi:

  Polar (colatitudinal) and azimuthal (longitudinal) angles.

- diff_n:

  Unsupported; must be `NULL`.

- n_max:

  Maximum degree (a concrete integer; defaults to the maximum of `n`).

## Value

An array (a list of arrays for `scipy_special_fresnel`,
`scipy_special_sici`, `scipy_special_lpmn` and `scipy_special_logsumexp`
with `return_sign = TRUE`).
