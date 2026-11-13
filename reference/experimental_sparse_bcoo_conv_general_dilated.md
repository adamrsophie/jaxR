# Convolution with BCOO operands

Port of `jax.experimental.sparse.bcoo_conv_general_dilated`: one
dimensional convolutions of operands with leading dimensions `(1, 1)`,
unit strides, dilations and group counts.

## Usage

``` r
experimental_sparse_bcoo_conv_general_dilated(
  lhs,
  rhs,
  window_strides,
  padding,
  lhs_dilation = NULL,
  rhs_dilation = NULL,
  dimension_numbers = NULL,
  feature_group_count = 1L,
  batch_group_count = 1L,
  precision = NULL,
  preferred_element_type = NULL
)
```

## Arguments

- lhs, rhs:

  BCOO or dense arrays.

- window_strides, padding, lhs_dilation, rhs_dilation,
  dimension_numbers, feature_group_count, batch_group_count, precision,
  preferred_element_type:

  As for
  [`lax_conv_general_dilated()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_general_dilated.md).

## Value

A BCOO array.
