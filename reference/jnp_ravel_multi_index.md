# Convert between flat and multi-dimensional indices

Ports of `jnp.ravel_multi_index` and `jnp.unravel_index`. Indices are
0-based.

## Usage

``` r
jnp_ravel_multi_index(
  multi_index,
  dims,
  mode = "raise",
  order = "C",
  dtype = NULL
)

jnp_unravel_index(indices, shape)
```

## Arguments

- multi_index:

  List of integer index arrays, one per dimension.

- dims, shape:

  Integer vector giving the array shape.

- mode:

  `"raise"` (requires concrete indices), `"wrap"`, `"clip"` or
  `"ignore"`.

- order:

  `"C"` or `"F"`.

- dtype:

  Optional integer dtype of the result.

- indices:

  Integer array of flat indices.

## Value

`jnp_ravel_multi_index` returns an integer array; `jnp_unravel_index`
returns a list of integer arrays, one per dimension (out of bound
indices are clipped).
