# Circular distributions

Ports of `jax.scipy.stats.vonmises` and `jax.scipy.stats.wrapcauchy`.

## Usage

``` r
scipy_stats_vonmises_logpdf(x, kappa)

scipy_stats_vonmises_pdf(x, kappa)

scipy_stats_wrapcauchy_logpdf(x, c)

scipy_stats_wrapcauchy_pdf(x, c)
```

## Arguments

- x:

  Angles.

- kappa:

  Von Mises concentration.

- c:

  Wrapped Cauchy shape parameter in `(0, 1)`.

## Value

An array.
