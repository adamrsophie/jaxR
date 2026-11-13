# Convolution output shapes

Ports of `jax.lax.conv_shape_tuple` (shapes in canonical order),
`jax.lax.conv_general_shape_tuple` and
`jax.lax.conv_transpose_shape_tuple`.

## Usage

``` r
lax_conv_shape_tuple(
  lhs_shape,
  rhs_shape,
  strides,
  pads,
  batch_group_count = 1L
)

lax_conv_general_shape_tuple(
  lhs_shape,
  rhs_shape,
  window_strides,
  padding,
  dimension_numbers
)

lax_conv_transpose_shape_tuple(
  lhs_shape,
  rhs_shape,
  window_strides,
  padding,
  dimension_numbers
)
```

## Arguments

- lhs_shape, rhs_shape:

  Integer vectors.

- strides, window_strides:

  Integer vectors.

- pads, padding:

  A padding string or a list of `c(low, high)` pairs.

- batch_group_count:

  Batch group count.

- dimension_numbers:

  Three spec strings or a `ConvDimensionNumbers`.

## Value

An integer vector.
