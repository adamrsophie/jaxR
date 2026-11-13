# Coordinate grids

Port of `jnp.meshgrid`.

## Usage

``` r
jnp_meshgrid(..., copy = TRUE, sparse = FALSE, indexing = "xy")
```

## Arguments

- ...:

  One dimensional arrays.

- copy:

  Must be `TRUE`.

- sparse:

  Return broadcastable (sparse) grids.

- indexing:

  `"xy"` (Cartesian) or `"ij"` (matrix).

## Value

A list of arrays.
