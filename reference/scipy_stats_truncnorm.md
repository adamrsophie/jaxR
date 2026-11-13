# Truncated normal distribution

Ports of `jax.scipy.stats.truncnorm`. `a` and `b` are the truncation
points in standardized units.

## Usage

``` r
scipy_stats_truncnorm_logpdf(x, a, b, loc = 0L, scale = 1L)

scipy_stats_truncnorm_pdf(x, a, b, loc = 0L, scale = 1L)

scipy_stats_truncnorm_logsf(x, a, b, loc = 0L, scale = 1L)

scipy_stats_truncnorm_sf(x, a, b, loc = 0L, scale = 1L)

scipy_stats_truncnorm_logcdf(x, a, b, loc = 0L, scale = 1L)

scipy_stats_truncnorm_cdf(x, a, b, loc = 0L, scale = 1L)
```

## Arguments

- x:

  Values.

- a, b:

  Lower and upper truncation points.

- loc:

  Location.

- scale:

  Scale.

## Value

An array.
