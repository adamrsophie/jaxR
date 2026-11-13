# Convert convolution dimension numbers to canonical form

Port of `jax.lax.conv_dimension_numbers`.

## Usage

``` r
lax_conv_dimension_numbers(lhs_shape, rhs_shape, dimension_numbers)
```

## Arguments

- lhs_shape, rhs_shape:

  Integer vectors.

- dimension_numbers:

  `NULL`, three spec strings, or a `ConvDimensionNumbers`.

## Value

A `ConvDimensionNumbers` object.
