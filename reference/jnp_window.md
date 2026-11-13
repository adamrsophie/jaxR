# Window functions

Ports of `jnp.bartlett`, `jnp.blackman`, `jnp.hamming`, `jnp.hanning`
and `jnp.kaiser` (default float dtype).

## Usage

``` r
jnp_blackman(M)

jnp_bartlett(M)

jnp_hamming(M)

jnp_hanning(M)

jnp_kaiser(M, beta)
```

## Arguments

- M:

  Number of points (static).

- beta:

  Shape parameter of the Kaiser window.

## Value

A one dimensional array.
