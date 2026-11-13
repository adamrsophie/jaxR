# Gaussian kernel density estimation

Port of `jax.scipy.stats.gaussian_kde`. The result is a pytree with the
fields `neff`, `dataset`, `weights`, `covariance` and `inv_cov`, and the
JAX methods are reached with `$`: `kde$evaluate(points)` (also
`kde$pdf`), `kde$logpdf(points)`, `kde$integrate_gaussian(mean, cov)`,
`kde$integrate_box_1d(low, high)`, `kde$integrate_kde(other)`,
`kde$resample(key, shape)`, and the properties `kde$d` and `kde$n`.
`integrate_box` and `set_bandwidth` raise errors, as in JAX.

## Usage

``` r
scipy_stats_gaussian_kde(dataset, bw_method = NULL, weights = NULL)
```

## Arguments

- dataset:

  Data with shape `(d, n)` (a vector is one dimensional data).

- bw_method:

  `"scott"` (default), `"silverman"`, a scalar factor or a function of
  the kde object.

- weights:

  Optional weights of the data points.

## Value

An object of class `jax_gaussian_kde`.

## Examples

``` r
if (FALSE) { # \dontrun{
kde <- scipy_stats_gaussian_kde(c(-1, 0, 0.5, 2))
kde$evaluate(c(0, 1))
} # }
```
