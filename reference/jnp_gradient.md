# Numerical gradient

Port of `jnp.gradient`: second order central differences in the interior
and first order differences at the edges, with scalar or 1-D coordinate
spacings.

## Usage

``` r
jnp_gradient(f, ..., axis = NULL, edge_order = NULL)
```

## Arguments

- f:

  Array.

- ...:

  Spacings: none (unit spacing), one for all axes, or one per axis; each
  a scalar or a 1-D array of coordinates.

- axis:

  0-based axis or axes; `NULL` for all.

- edge_order:

  Not supported (must be `NULL`).

## Value

An array for one axis, otherwise a list of arrays.
