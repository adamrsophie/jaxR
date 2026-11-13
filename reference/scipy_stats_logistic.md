# Logistic distribution

Ports of `jax.scipy.stats.logistic`.

## Usage

``` r
scipy_stats_logistic_logpdf(x, loc = 0L, scale = 1L)

scipy_stats_logistic_pdf(x, loc = 0L, scale = 1L)

scipy_stats_logistic_ppf(x, loc = 0L, scale = 1L)

scipy_stats_logistic_sf(x, loc = 0L, scale = 1L)

scipy_stats_logistic_isf(x, loc = 0L, scale = 1L)

scipy_stats_logistic_cdf(x, loc = 0L, scale = 1L)
```

## Arguments

- x:

  Values (probabilities for `ppf` and `isf`, as in JAX).

- loc:

  Location.

- scale:

  Scale.

## Value

An array.
