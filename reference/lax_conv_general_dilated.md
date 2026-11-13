# General n dimensional convolution

Port of `jax.lax.conv_general_dilated` (XLA Conv). Arrays have rank
`n + 2`.

## Usage

``` r
lax_conv_general_dilated(
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

- lhs:

  Input array.

- rhs:

  Kernel array.

- window_strides:

  Integer vector of `n` strides.

- padding:

  `"SAME"`, `"SAME_LOWER"`, `"VALID"` or a list of `n` `c(low, high)`
  pairs.

- lhs_dilation, rhs_dilation:

  Optional integer vectors of `n` dilation factors.

- dimension_numbers:

  `NULL` (`NCHW`, `OIHW`, `NCHW` layouts), three spec strings such as
  `c("NHWC", "HWIO", "NHWC")`, or a `ConvDimensionNumbers`.

- feature_group_count, batch_group_count:

  Group counts.

- precision:

  Optional precision (`"default"`, `"high"`, `"highest"`, or a pair).

- preferred_element_type:

  Optional accumulation and result dtype.

## Value

The convolution.
