# Spectral analysis: STFT, inverse STFT, cross and power spectral densities

Ports of `jax.scipy.signal.stft`, `istft`, `csd` and `welch`. Signals
are split in overlapping segments (`nperseg` long, `noverlap` overlap)
that are detrended, windowed and Fourier transformed. Each function
returns an unnamed list, like the tuples of JAX: `stft` gives
`list(f, t, Zxx)`, `istft` `list(t, x)`, `csd` `list(f, Pxy)` and
`welch` `list(f, Pxx)`.

## Usage

``` r
scipy_signal_stft(
  x,
  fs = 1,
  window = "hann",
  nperseg = 256L,
  noverlap = NULL,
  nfft = NULL,
  detrend = FALSE,
  return_onesided = TRUE,
  boundary = "zeros",
  padded = TRUE,
  axis = -1L
)

scipy_signal_csd(
  x,
  y,
  fs = 1,
  window = "hann",
  nperseg = NULL,
  noverlap = NULL,
  nfft = NULL,
  detrend = "constant",
  return_onesided = TRUE,
  scaling = "density",
  axis = -1L,
  average = "mean"
)

scipy_signal_welch(
  x,
  fs = 1,
  window = "hann",
  nperseg = NULL,
  noverlap = NULL,
  nfft = NULL,
  detrend = "constant",
  return_onesided = TRUE,
  scaling = "density",
  axis = -1L,
  average = "mean"
)

scipy_signal_istft(
  Zxx,
  fs = 1,
  window = "hann",
  nperseg = NULL,
  noverlap = NULL,
  nfft = NULL,
  input_onesided = TRUE,
  boundary = TRUE,
  time_axis = -1L,
  freq_axis = -2L
)
```

## Arguments

- x:

  Signal array (time along `axis`).

- fs:

  Sampling frequency.

- window:

  A 1-D array of window values, or a window of
  `scipy.signal.get_window`: a name (`"hann"`, `"hamming"`,
  `"blackman"`, ...) or a list of a name and its parameters
  (`list("kaiser", 8)`, `list("tukey", 0.25)`, `list("chebwin", 60)`,
  `list("dpss", 2)`), where Python takes a tuple. Named windows are
  periodic, as in SciPy.

- nperseg:

  Segment length (static integer).

- noverlap:

  Overlap of the segments (default `nperseg %/% 2`).

- nfft:

  FFT length (default `nperseg`).

- detrend:

  Detrending of each segment: `"constant"`, `"linear"`, a function, or
  `FALSE` for none.

- return_onesided:

  One sided spectrum for real data.

- boundary:

  Signal extension at both ends for `stft`: `"zeros"`, `"even"`,
  `"odd"`, `"constant"` or `NULL`; for `istft`, whether the input was
  extended.

- padded:

  Zero pad the signal to a whole number of segments.

- axis:

  0-based axis of the time samples.

- y:

  Second signal for `csd` (or `NULL`, giving the power spectral density
  of `x`).

- scaling:

  `"density"` or `"spectrum"`.

- average:

  Averaging over segments: `"mean"` or `"median"`.

- Zxx:

  STFT array.

- input_onesided:

  Whether `Zxx` is a one sided spectrum.

- time_axis, freq_axis:

  0-based axes of `Zxx`.

## Value

An unnamed list of arrays (see Description).

## Examples

``` r
if (FALSE) { # \dontrun{
x <- sin(seq(0, 20, length.out = 64))
res <- scipy_signal_stft(x, nperseg = 16L)
scipy_signal_istft(res[[3L]], nperseg = 16L)
scipy_signal_welch(x, nperseg = 16L)
} # }
```
