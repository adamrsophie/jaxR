# Convolution and correlation of one dimensional arrays

Ports of `jnp.convolve` and `jnp.correlate`.

## Usage

``` r
jnp_convolve(
  a,
  v,
  mode = "full",
  precision = NULL,
  preferred_element_type = NULL
)

jnp_correlate(
  a,
  v,
  mode = "valid",
  precision = NULL,
  preferred_element_type = NULL
)
```

## Arguments

- a, v:

  One dimensional arrays.

- mode:

  `"full"`, `"same"` or `"valid"`.

- precision:

  Optional precision.

- preferred_element_type:

  Optional result dtype (inputs are then promoted without conversion to
  an inexact dtype).

## Value

A one dimensional array.
