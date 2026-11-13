# Shape of a windowed reduction

Port of `jax.lax.reduce_window_shape_tuple`.

## Usage

``` r
lax_reduce_window_shape_tuple(
  operand_shape,
  window_dimensions,
  window_strides,
  padding,
  base_dilation = NULL,
  window_dilation = NULL
)
```

## Arguments

- operand_shape:

  Integer vector.

- window_dimensions, window_strides:

  Integer vectors.

- padding:

  List of `c(low, high)` pairs.

- base_dilation, window_dilation:

  Optional integer vectors.

## Value

An integer vector.
