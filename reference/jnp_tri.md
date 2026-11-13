# Triangular matrices

Ports of `jnp.tri`, `jnp.tril` and `jnp.triu`.

## Usage

``` r
jnp_tri(N, M = NULL, k = 0L, dtype = NULL)

jnp_tril(m, k = 0L)

jnp_triu(m, k = 0L)
```

## Arguments

- N, M:

  Number of rows and columns (`M` defaults to `N`).

- k:

  Diagonal offset.

- dtype:

  Dtype (default float).

- m:

  Array of rank at least two.

## Value

An array.
