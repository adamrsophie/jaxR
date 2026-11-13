# Broadcast shapes (lax version)

Port of `jax.lax.broadcast_shapes`: the shape that results from NumPy
broadcasting of the given shapes.

## Usage

``` r
lax_broadcast_shapes(...)
```

## Arguments

- ...:

  Integer shape vectors (one or more).

## Value

An integer vector.

## Examples

``` r
lax_broadcast_shapes(c(3L, 1L), 4L)
#> [1] 3 4
```
