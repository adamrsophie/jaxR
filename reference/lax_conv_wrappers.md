# Convolution wrappers

Ports of `jax.lax.conv`, `jax.lax.conv_with_general_padding` and
`jax.lax.conv_transpose`, convenience wrappers around
[`lax_conv_general_dilated()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_general_dilated.md).

## Usage

``` r
lax_conv(
  lhs,
  rhs,
  window_strides,
  padding,
  precision = NULL,
  preferred_element_type = NULL
)

lax_conv_with_general_padding(
  lhs,
  rhs,
  window_strides,
  padding,
  lhs_dilation,
  rhs_dilation,
  precision = NULL,
  preferred_element_type = NULL
)

lax_conv_transpose(
  lhs,
  rhs,
  strides,
  padding,
  rhs_dilation = NULL,
  dimension_numbers = NULL,
  transpose_kernel = FALSE,
  precision = NULL,
  preferred_element_type = NULL,
  use_consistent_padding = FALSE
)
```

## Arguments

- lhs:

  Input array.

- rhs:

  Kernel array.

- window_strides:

  Integer vector of `n` strides.

- padding:

  `"SAME"`, `"SAME_LOWER"`, `"VALID"` or a list of `n` `c(low, high)`
  pairs.

- precision:

  Optional precision (`"default"`, `"high"`, `"highest"`, or a pair).

- preferred_element_type:

  Optional accumulation and result dtype.

- lhs_dilation, rhs_dilation:

  Optional integer vectors of `n` dilation factors.

- strides:

  Integer vector of `n` (fractional) strides of the transposed
  convolution.

- dimension_numbers:

  `NULL` (`NCHW`, `OIHW`, `NCHW` layouts), three spec strings such as
  `c("NHWC", "HWIO", "NHWC")`, or a `ConvDimensionNumbers`.

- transpose_kernel:

  Flip the spatial axes and swap the feature axes of the kernel
  (gradient of a forward convolution, as in Keras `Conv2DTranspose`).

- use_consistent_padding:

  Interpret explicit padding as the padding of the corresponding forward
  convolution.

## Value

The convolution.
