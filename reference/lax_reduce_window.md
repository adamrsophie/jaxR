# Reduction over padded windows

Port of `jax.lax.reduce_window` (XLA ReduceWindowWithGeneralPadding).
With `lax_add`, `lax_max` or `lax_min` and their identity as initial
value the specialized reductions `reduce_window_sum`,
`reduce_window_max` and `reduce_window_min` are used.

## Usage

``` r
lax_reduce_window(
  operand,
  init_value,
  computation,
  window_dimensions,
  window_strides = NULL,
  padding = "VALID",
  base_dilation = NULL,
  window_dilation = NULL
)
```

## Arguments

- operand:

  Array or pytree (list) of arrays.

- init_value:

  Scalar or pytree of scalars with the structure of `operand`.

- computation:

  Function of two arguments with the structure of `operand`.

- window_dimensions:

  Integer vector of window sizes.

- window_strides:

  Optional integer vector of strides (default 1).

- padding:

  `"VALID"` (default), `"SAME"`, `"SAME_LOWER"` or a list of
  `c(low, high)` pairs.

- base_dilation, window_dilation:

  Optional integer vectors of dilations.

## Value

An array or a pytree of arrays with the structure of `operand`.
