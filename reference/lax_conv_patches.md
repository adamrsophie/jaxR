# Image patches and unshared convolutions

Ports of `jax.lax.conv_general_dilated_patches` (extract the patches of
the receptive field of
[`lax_conv_general_dilated()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_general_dilated.md)
into the output feature dimension, in the order
`"c" + spatial dims of rhs_spec`) and
`jax.lax.conv_general_dilated_local` (a locally connected layer: a
convolution with a separate kernel at each output position; the spatial
dimensions of `rhs` are the output positions and its `I` dimension holds
`(in feature, filter spatial...)`).

## Usage

``` r
lax_conv_general_dilated_patches(
  lhs,
  filter_shape,
  window_strides,
  padding,
  lhs_dilation = NULL,
  rhs_dilation = NULL,
  dimension_numbers = NULL,
  precision = NULL,
  preferred_element_type = NULL
)

lax_conv_general_dilated_local(
  lhs,
  rhs,
  window_strides,
  padding,
  filter_shape,
  lhs_dilation = NULL,
  rhs_dilation = NULL,
  dimension_numbers = NULL,
  precision = NULL
)
```

## Arguments

- lhs:

  Input array of rank `n + 2`.

- filter_shape:

  Integer vector of `n` receptive field sizes (in the order of the rhs
  spatial dimensions).

- window_strides:

  Integer vector of `n` strides.

- padding:

  `"SAME"`, `"VALID"` or a list of `n` `c(low, high)` pairs.

- lhs_dilation, rhs_dilation:

  Optional dilation factors.

- dimension_numbers:

  `NULL`, three layout strings or a
  [`lax_ConvDimensionNumbers()`](https://adamrsophie.github.io/jaxR/reference/lax_ConvDimensionNumbers.md)
  object.

- precision:

  Optional precision (see
  [lax_Precision](https://adamrsophie.github.io/jaxR/reference/lax_enums.md)).

- preferred_element_type:

  Optional result dtype.

- rhs:

  Kernel of the unshared convolution.

## Value

An array.

## Examples

``` r
if (FALSE) { # \dontrun{
x <- array(as.numeric(1:16), c(1, 1, 4, 4))
dim(lax_conv_general_dilated_patches(x, c(2L, 2L), c(1L, 1L), "VALID"))
} # }
```
