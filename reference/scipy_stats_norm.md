# Normal distribution

Ports of `jax.scipy.stats.norm`: log density, density, distribution
function and its logarithm, survival function and its logarithm, percent
point function and inverse survival function of the normal distribution.

## Usage

``` r
scipy_stats_norm_logpdf(x, loc = 0L, scale = 1L)

scipy_stats_norm_pdf(x, loc = 0L, scale = 1L)

scipy_stats_norm_cdf(x, loc = 0L, scale = 1L)

scipy_stats_norm_logcdf(x, loc = 0L, scale = 1L)

scipy_stats_norm_ppf(q, loc = 0L, scale = 1L)

scipy_stats_norm_logsf(x, loc = 0L, scale = 1L)

scipy_stats_norm_sf(x, loc = 0L, scale = 1L)

scipy_stats_norm_isf(q, loc = 0L, scale = 1L)
```

## Arguments

- x:

  Values.

- loc:

  Location (mean).

- scale:

  Scale (standard deviation).

- q:

  Probabilities.

## Value

An array.

## Examples

``` r
if (FALSE) { # \dontrun{
scipy_stats_norm_logpdf(c(-1, 0, 1))
scipy_stats_norm_ppf(0.975)
} # }
```
