# Index grids

Ports of `jnp.indices` (a dense `(ndim, ...)` grid of 0-based indices,
or a list of broadcastable index arrays with `sparse = TRUE`) and
`jnp.ix_` (open mesh: the k-th 1-D input is reshaped to lie along axis
k).

## Usage

``` r
jnp_indices(dimensions, dtype = NULL, sparse = FALSE)

jnp_ix_(...)
```

## Arguments

- dimensions:

  Integer vector giving the grid shape.

- dtype:

  Optional integer dtype.

- sparse:

  Return a list of sparse index arrays.

- ...:

  1-D arrays.

## Value

An array or a list of arrays.
