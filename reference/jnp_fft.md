# Discrete Fourier transforms

Ports of `jax.numpy.fft`. The one dimensional transforms (`fft`, `ifft`,
`rfft`, `irfft`, `hfft`, `ihfft`) act along `axis`; the `2` variants
along two `axes` (default the last two) and the `n` variants along
`axes` (default all, or the last `length(s)`). Axes are 0-based;
negative values count from the end. Inputs are cropped or zero padded to
the transform lengths `n` or `s`.

## Usage

``` r
jnp_fft_fftn(a, s = NULL, axes = NULL, norm = NULL)

jnp_fft_ifftn(a, s = NULL, axes = NULL, norm = NULL)

jnp_fft_rfftn(a, s = NULL, axes = NULL, norm = NULL)

jnp_fft_irfftn(a, s = NULL, axes = NULL, norm = NULL)

jnp_fft_fft(a, n = NULL, axis = -1L, norm = NULL)

jnp_fft_ifft(a, n = NULL, axis = -1L, norm = NULL)

jnp_fft_rfft(a, n = NULL, axis = -1L, norm = NULL)

jnp_fft_irfft(a, n = NULL, axis = -1L, norm = NULL)

jnp_fft_hfft(a, n = NULL, axis = -1L, norm = NULL)

jnp_fft_ihfft(a, n = NULL, axis = -1L, norm = NULL)

jnp_fft_fft2(a, s = NULL, axes = c(-2L, -1L), norm = NULL)

jnp_fft_ifft2(a, s = NULL, axes = c(-2L, -1L), norm = NULL)

jnp_fft_rfft2(a, s = NULL, axes = c(-2L, -1L), norm = NULL)

jnp_fft_irfft2(a, s = NULL, axes = c(-2L, -1L), norm = NULL)
```

## Arguments

- a:

  Array.

- s:

  Transform lengths along `axes`.

- axes:

  0-based axes.

- norm:

  `NULL` or `"backward"` (no scaling of the forward transform),
  `"ortho"` (both directions scaled by `1 / sqrt(n)`) or `"forward"`
  (forward transform scaled by `1 / n`).

- n:

  Length of the transformed axis (`NULL` uses the input length; for the
  inverse real transforms the default is `2 * (m - 1)`).

- axis:

  0-based axis.

## Value

An array.
