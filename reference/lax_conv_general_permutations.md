# Convolution dimension permutations

Port of `jax.lax.conv_general_permutations`: the 0-based permutations
(relative to the canonical `NC...`, `OI...`, `NC...` layouts) described
by three spec strings.

## Usage

``` r
lax_conv_general_permutations(dimension_numbers)
```

## Arguments

- dimension_numbers:

  Character vector (or list) of three spec strings such as
  `c("NHWC", "HWIO", "NHWC")`.

## Value

A list of three integer vectors.
