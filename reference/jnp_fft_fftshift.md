# Shift the zero frequency term to the center

Ports of `jax.numpy.fft.fftshift` and `jax.numpy.fft.ifftshift` (its
inverse).

## Usage

``` r
jnp_fft_fftshift(x, axes = NULL)

jnp_fft_ifftshift(x, axes = NULL)
```

## Arguments

- x:

  Array.

- axes:

  0-based axis or axes to shift (`NULL` shifts all).

## Value

An array.
