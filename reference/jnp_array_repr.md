# NumPy style string representations

Ports of `jnp.array_repr` and `jnp.array_str` (NumPy's functions):
`"array([1., 2.], dtype=float32)"` and `"[1. 2.]"`.

## Usage

``` r
jnp_array_repr(
  arr,
  max_line_width = NULL,
  precision = NULL,
  suppress_small = NULL
)

jnp_array_str(
  arr,
  max_line_width = NULL,
  precision = NULL,
  suppress_small = NULL
)
```

## Arguments

- arr:

  An array.

- max_line_width:

  Maximum line width.

- precision:

  Number of digits.

- suppress_small:

  Print very small numbers as zero.

## Value

A string.
