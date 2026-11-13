# FFT types

Port of `jax.lax.FftType`: `FFT` (forward complex to complex), `IFFT`
(inverse complex to complex), `RFFT` (forward real to complex) and
`IRFFT` (inverse complex to real). R has no enums, so the values are
integer codes;
[`lax_fft()`](https://adamrsophie.github.io/jaxR/reference/lax_fft.md)
also accepts the names as strings.

## Usage

``` r
lax_FftType
```

## Format

A named integer vector.
