# Multivariate normal distribution

Ports of `jax.scipy.stats.multivariate_normal`. `cov` may be a scalar
(isotropic) or a (batch of) covariance matrices.

## Usage

``` r
scipy_stats_multivariate_normal_logpdf(x, mean, cov, allow_singular = NULL)

scipy_stats_multivariate_normal_pdf(x, mean, cov)
```

## Arguments

- x:

  Values, the event dimension last.

- mean:

  Mean vector.

- cov:

  Covariance.

- allow_singular:

  Not supported (must be `NULL`), as in JAX.

## Value

An array.
