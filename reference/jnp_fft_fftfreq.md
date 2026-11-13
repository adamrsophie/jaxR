# Sample frequencies of the discrete Fourier transform

Ports of `jax.numpy.fft.fftfreq` and `jax.numpy.fft.rfftfreq`.

## Usage

``` r
jnp_fft_fftfreq(n, d = 1, dtype = NULL, device = NULL)

jnp_fft_rfftfreq(n, d = 1, dtype = NULL, device = NULL)
```

## Arguments

- n:

  Window length (a single integer).

- d:

  Sample spacing.

- dtype:

  Optional dtype (default the default float type).

- device:

  Ignored (single device).

## Value

A one dimensional array.
