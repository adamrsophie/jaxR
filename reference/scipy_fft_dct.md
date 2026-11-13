# Discrete cosine transforms

Ports of `jax.scipy.fft.dct`, `dctn`, `idct` and `idctn` (type 2 only).

## Usage

``` r
scipy_fft_dct(x, type = 2L, n = NULL, axis = -1L, norm = NULL)

scipy_fft_dctn(x, type = 2L, s = NULL, axes = NULL, norm = NULL)

scipy_fft_idct(x, type = 2L, n = NULL, axis = -1L, norm = NULL)

scipy_fft_idctn(x, type = 2L, s = NULL, axes = NULL, norm = NULL)
```

## Arguments

- x:

  Array.

- type:

  DCT type (only 2 is implemented).

- n:

  Length of the transform (the input is zero padded or cropped).

- axis:

  0-based axis.

- norm:

  `NULL`, `"backward"` or `"ortho"`.

- s:

  Transform lengths along `axes`.

- axes:

  0-based axes (default all, or the last `length(s)`).

## Value

An array.
