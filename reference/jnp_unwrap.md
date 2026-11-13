# Unwrap a phase signal

Port of `jnp.unwrap`.

## Usage

``` r
jnp_unwrap(p, discont = NULL, axis = -1L, period = 2 * pi)
```

## Arguments

- p:

  Array.

- discont:

  Maximum discontinuity (default `period / 2`).

- axis:

  Axis (0-based).

- period:

  Period of the signal.

## Value

An array.
