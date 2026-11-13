# Remove a linear or constant trend

Port of `jax.scipy.signal.detrend`.

## Usage

``` r
scipy_signal_detrend(
  data,
  axis = -1L,
  type = "linear",
  bp = 0L,
  overwrite_data = NULL
)
```

## Arguments

- data:

  Array.

- axis:

  0-based axis along which to detrend.

- type:

  `"linear"` (least squares line, piecewise between the breakpoints) or
  `"constant"` (the mean).

- bp:

  Static breakpoints (0-based positions along `axis`) for `"linear"`.

- overwrite_data:

  Not supported (must be `NULL`).

## Value

The detrended array (inexact dtype).

## Examples

``` r
if (FALSE) { # \dontrun{
scipy_signal_detrend(c(1, 3, 2, 5, 4))
} # }
```
