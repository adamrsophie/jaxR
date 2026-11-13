# Dirichlet distribution

Ports of `jax.scipy.stats.dirichlet`. `x` holds the components along its
first axis; it may omit the last component, which is then `1 - sum(x)`.

## Usage

``` r
scipy_stats_dirichlet_logpdf(x, alpha)

scipy_stats_dirichlet_pdf(x, alpha)
```

## Arguments

- x:

  Points on the simplex.

- alpha:

  Concentration parameters (one dimensional).

## Value

An array.
