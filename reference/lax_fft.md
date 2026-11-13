# Fast Fourier transform

Port of `jax.lax.fft`: an FFT over the trailing `length(fft_lengths)`
dimensions of `x`. `RFFT` converts `x` to an inexact real dtype and
returns the `fft_lengths[n] %/% 2 + 1` non-negative frequency terms of
the last dimension; the other types convert `x` to a complex dtype.
`IRFFT` returns a real array whose trailing dimensions are
`fft_lengths`.

## Usage

``` r
lax_fft(x, fft_type, fft_lengths)
```

## Arguments

- x:

  Array.

- fft_type:

  One of `"FFT"`, `"IFFT"`, `"RFFT"`, `"IRFFT"` (or lower case), or a
  value of `lax_FftType`.

- fft_lengths:

  Integer vector of transform lengths (at most 3 for XLA).

## Value

An array.
