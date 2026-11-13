# Compare arrays elementwise within a tolerance

Ports of `jnp.isclose` and `jnp.allclose`:
`abs(a - b) <= atol + rtol * abs(b)`, with infinities equal only to
themselves and NaN equal to NaN when `equal_nan`.

## Usage

``` r
jnp_isclose(a, b, rtol = 1e-05, atol = 1e-08, equal_nan = FALSE)

jnp_allclose(a, b, rtol = 1e-05, atol = 1e-08, equal_nan = FALSE)
```

## Arguments

- a, b:

  Arrays.

- rtol:

  Relative tolerance.

- atol:

  Absolute tolerance.

- equal_nan:

  Whether NaN values compare equal.

## Value

A boolean array (`jnp_isclose`) or boolean scalar (`jnp_allclose`).
