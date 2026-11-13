# Shape as an array

Port of `jax.lax.shape_as_value` for static shapes: an integer array
holding the shape (`int32`, or `int64` when a dimension does not fit in
32 bits).

## Usage

``` r
lax_shape_as_value(shape)
```

## Arguments

- shape:

  Integer vector.

## Value

An integer array of length `length(shape)`.

## Examples

``` r
if (FALSE) { # \dontrun{
lax_shape_as_value(c(2L, 3L))
} # }
```
