# Covariance and correlation matrices

Ports of `jnp.cov` and `jnp.corrcoef`. Rows are variables and columns
are observations unless `rowvar = FALSE`.

## Usage

``` r
jnp_cov(
  m,
  y = NULL,
  rowvar = TRUE,
  bias = FALSE,
  ddof = NULL,
  fweights = NULL,
  aweights = NULL,
  dtype = NULL
)

jnp_corrcoef(x, y = NULL, rowvar = TRUE, dtype = NULL)
```

## Arguments

- m, x:

  Array of one or two dimensions.

- y:

  Optional additional variables, stacked below `m`.

- rowvar:

  Whether rows (rather than columns) are variables.

- bias:

  Normalize by `N` rather than `N - 1` (ignored when `ddof` is given).

- ddof:

  Delta degrees of freedom; `NULL` uses `bias`.

- fweights:

  Optional integer frequency weights of the observations.

- aweights:

  Optional observation weights.

- dtype:

  Optional inexact dtype of the computation.

## Value

An array.
