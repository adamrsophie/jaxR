# Stack and unstack arrays

Ports of `jax.lax.stack` (join arrays of equal shape and dtype along a
new axis) and `jax.lax.unstack` (the list of slices along an axis).

## Usage

``` r
lax_stack(operands, axis = 0L)

lax_unstack(x, axis = 0L)
```

## Arguments

- operands:

  List of arrays.

- axis:

  0-based axis (of the result for `lax_stack`).

- x:

  Array.

## Value

`lax_stack()` returns an array, `lax_unstack()` a list of arrays.

## Examples

``` r
if (FALSE) { # \dontrun{
lax_stack(list(c(1, 2), c(3, 4)), axis = 1L)
lax_unstack(matrix(1:4, 2))
} # }
```
