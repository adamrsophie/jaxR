# Interpolation on a regular grid

Port of `jax.scipy.interpolate.RegularGridInterpolator`: linear or
nearest neighbor interpolation of values given on a rectilinear grid in
any number of dimensions. The constructor returns a callable object:
`interp(xi, method = NULL)` evaluates the interpolant at the points `xi`
(an array whose last axis has one coordinate per grid dimension, or a
list of coordinate arrays that broadcast together). Its fields
`interp$grid`, `interp$values`, `interp$method`, `interp$bounds_error`
and `interp$fill_value` are read with `$`. The object is a pytree, so it
can be passed through
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) and
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md).

## Usage

``` r
scipy_interpolate_RegularGridInterpolator(
  points,
  values,
  method = "linear",
  bounds_error = FALSE,
  fill_value = NaN
)

# S3 method for class 'scipy_interpolate_RegularGridInterpolator'
x$name
```

## Arguments

- points:

  List of 1-D arrays of strictly ascending grid coordinates, one per
  dimension (a single vector is taken as a one dimensional grid).

- values:

  Array of values on the grid; trailing dimensions beyond the grid are
  carried along.

- method:

  `"linear"` or `"nearest"`.

- bounds_error:

  Must be `FALSE` (bounds cannot be checked under jit).

- fill_value:

  Value for points outside the grid, or `NULL` to extrapolate.

- x:

  Interpolator object.

- name:

  Field name.

## Value

A callable interpolator; for `$`, the field.

## Examples

``` r
if (FALSE) { # \dontrun{
interp <- scipy_interpolate_RegularGridInterpolator(
  list(c(0, 1, 2), c(0, 1)), matrix(c(0, 1, 2, 3, 4, 5), 3, 2, byrow = TRUE))
interp(matrix(c(0.5, 0.5, 1.5, 0.25), 2, 2, byrow = TRUE))
} # }
```
