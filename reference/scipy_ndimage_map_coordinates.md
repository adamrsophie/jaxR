# Map an array to new coordinates by interpolation

Port of `jax.scipy.ndimage.map_coordinates`: the value of `input` at
each point of `coordinates` (0-based, fractional array coordinates) by
nearest (`order = 0`) or linear (`order = 1`) interpolation. Higher
orders are not implemented upstream. Integer inputs give rounded integer
outputs.

## Usage

``` r
scipy_ndimage_map_coordinates(
  input,
  coordinates,
  order,
  mode = "constant",
  cval = 0
)
```

## Arguments

- input:

  Array.

- coordinates:

  A list with one coordinate array per dimension of `input` (the arrays
  broadcast together), or an array whose first axis indexes the
  dimensions.

- order:

  Interpolation order, `0` or `1`.

- mode:

  Handling of points outside the array: `"constant"` (use `cval`),
  `"nearest"`, `"wrap"`, `"mirror"` or `"reflect"`.

- cval:

  Fill value for `mode = "constant"`.

## Value

An array with the broadcast shape of the coordinates and the dtype of
`input`.

## Examples

``` r
if (FALSE) { # \dontrun{
x <- matrix(as.numeric(0:11), 3, 4, byrow = TRUE)
scipy_ndimage_map_coordinates(x, list(c(0.5, 2), c(0.5, 1)), order = 1L)
} # }
```
