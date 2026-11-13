# Convolution and correlation of N-dimensional arrays

Ports of `jax.scipy.signal.convolve`, `convolve2d`, `correlate`,
`correlate2d` and `fftconvolve`. The direct methods use
[`lax_conv_general_dilated()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_general_dilated.md);
`fftconvolve` (and `method = "fft"`) multiplies Fourier transforms.
Inputs are promoted to an inexact dtype.

## Usage

``` r
scipy_signal_fftconvolve(in1, in2, mode = "full", axes = NULL)

scipy_signal_convolve(
  in1,
  in2,
  mode = "full",
  method = "auto",
  precision = NULL
)

scipy_signal_convolve2d(
  in1,
  in2,
  mode = "full",
  boundary = "fill",
  fillvalue = 0,
  precision = NULL
)

scipy_signal_correlate(
  in1,
  in2,
  mode = "full",
  method = "auto",
  precision = NULL
)

scipy_signal_correlate2d(
  in1,
  in2,
  mode = "full",
  boundary = "fill",
  fillvalue = 0,
  precision = NULL
)
```

## Arguments

- in1:

  Left-hand input array.

- in2:

  Right-hand input array, with the same number of dimensions as `in1`.

- mode:

  Output size: `"full"` (the full convolution), `"same"` (the size of
  `in1`, centered) or `"valid"` (only positions without zero padding).

- axes:

  Optional 0-based axes over which to convolve; the other axes are
  mapped over (they must have equal sizes). By default all axes are
  convolved.

- method:

  `"auto"` or `"direct"` (lax convolution) or `"fft"`.

- precision:

  Optional precision of the convolution (see
  [`lax_conv_general_dilated()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_general_dilated.md)).

- boundary:

  Only `"fill"` is supported.

- fillvalue:

  Only `0` is supported.

## Value

An array.

## Examples

``` r
if (FALSE) { # \dontrun{
scipy_signal_convolve(c(1, 2, 3), c(0, 1, 0.5))
scipy_signal_correlate2d(matrix(1:6, 2), matrix(c(1, -1), 1), mode = "same")
} # }
```
