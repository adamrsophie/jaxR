# Polynomials

Ports of `jnp.poly`, `jnp.polyadd`, `jnp.polysub`, `jnp.polymul`,
`jnp.polydiv`, `jnp.polyder`, `jnp.polyint`, `jnp.polyval`,
`jnp.polyfit` and `jnp.roots`. Coefficient vectors run from the highest
degree to the constant term.

## Usage

``` r
jnp_roots(p, strip_zeros = TRUE)

jnp_polyfit(x, y, deg, rcond = NULL, full = FALSE, w = NULL, cov = FALSE)

jnp_poly(seq_of_zeros)

jnp_polyval(p, x, unroll = NULL)

jnp_polyadd(a1, a2)

jnp_polysub(a1, a2)

jnp_polyint(p, m = 1L, k = NULL)

jnp_polyder(p, m = 1L)

jnp_polymul(a1, a2, trim_leading_zeros = FALSE)

jnp_polydiv(u, v, trim_leading_zeros = FALSE)
```

## Arguments

- strip_zeros:

  `roots`: strip leading zeros (needs concrete values); with `FALSE` the
  result has one NaN entry per leading zero.

- x:

  Points at which to evaluate (`polyval`) or sample coordinates
  (`polyfit`).

- y:

  Sample values (1-D, or 2-D with one column per data set).

- deg:

  Degree of the fitted polynomial.

- rcond:

  Relative condition number for the least squares solve.

- full:

  Also return the residuals, rank, singular values and `rcond`.

- w:

  Optional weights of the samples.

- cov:

  Also return the covariance matrix of the coefficients (`TRUE`, or
  `"unscaled"`).

- seq_of_zeros:

  1-D array of roots, or a square matrix (its eigenvalues are used).

- unroll:

  Number of Horner steps unrolled before a loop is used.

- a1, a2, u, v, p:

  Coefficient arrays.

- m:

  Order of the derivative or integral.

- k:

  Integration constants (a scalar or one per integration).

- trim_leading_zeros:

  Whether to remove leading zeros (needs concrete values).

## Value

An array; `polydiv` returns `list(quotient, remainder)`, `polyfit` with
`full` or `cov` returns a list.
