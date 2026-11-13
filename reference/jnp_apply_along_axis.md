# Apply a function along axes

Ports of `jnp.apply_along_axis` (`func1d` is applied to 1-D slices along
`axis` by nested `vmap`) and `jnp.apply_over_axes` (`func(a, axis)` is
applied repeatedly, restoring reduced axes).

## Usage

``` r
jnp_apply_along_axis(func1d, axis, arr, ...)

jnp_apply_over_axes(func, a, axes)
```

## Arguments

- func1d:

  Function of a 1-D array.

- axis:

  0-based axis.

- arr, a:

  Array.

- ...:

  Further arguments to `func1d`.

- func:

  Function of an array and an axis.

- axes:

  0-based axes.

## Value

An array.
