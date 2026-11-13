# Dimension numbers of dot products and convolutions

Constructors for the dimension number arguments of
[`lax_dot_general()`](https://adamrsophie.github.io/jaxR/reference/lax_dot_general.md)
(`jax.lax.DotDimensionNumbers`),
[`lax_conv_general_dilated()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_general_dilated.md)
(`jax.lax.ConvGeneralDilatedDimensionNumbers`) and
[`lax_ragged_dot_general()`](https://adamrsophie.github.io/jaxR/reference/lax_ragged_dot.md)
(`jax.lax.RaggedDotDimensionNumbers`). Dimensions are 0-based.

## Usage

``` r
lax_DotDimensionNumbers(
  lhs_contracting,
  rhs_contracting,
  lhs_batch = integer(0),
  rhs_batch = integer(0)
)

lax_ConvGeneralDilatedDimensionNumbers(lhs_spec, rhs_spec, out_spec)

lax_RaggedDotDimensionNumbers(
  dot_dimension_numbers,
  lhs_ragged_dimensions,
  rhs_group_dimensions
)
```

## Arguments

- lhs_contracting, rhs_contracting:

  Contracting dimensions.

- lhs_batch, rhs_batch:

  Batch dimensions.

- lhs_spec, rhs_spec, out_spec:

  Layout strings (such as `"NCHW"`, `"OIHW"`) or 0-based integer
  vectors.

- dot_dimension_numbers:

  Dot dimension numbers as for
  [`lax_dot_general()`](https://adamrsophie.github.io/jaxR/reference/lax_dot_general.md).

- lhs_ragged_dimensions:

  The lhs ragged dimension (one value).

- rhs_group_dimensions:

  The rhs group dimensions (zero or one value).

## Value

`lax_DotDimensionNumbers()` returns
`list(list(lhs_contracting, rhs_contracting), list(lhs_batch, rhs_batch))`.

`lax_ConvGeneralDilatedDimensionNumbers()` returns the three layout
strings (or, for integer vectors, a
[`lax_ConvDimensionNumbers()`](https://adamrsophie.github.io/jaxR/reference/lax_ConvDimensionNumbers.md)
object).

`lax_RaggedDotDimensionNumbers()` returns an object of class
`jax_RaggedDotDimensionNumbers`.

## Examples

``` r
lax_DotDimensionNumbers(1L, 0L)
#> [[1]]
#> [[1]][[1]]
#> [1] 1
#> 
#> [[1]][[2]]
#> [1] 0
#> 
#> 
#> [[2]]
#> [[2]][[1]]
#> integer(0)
#> 
#> [[2]][[2]]
#> integer(0)
#> 
#> 
lax_ConvGeneralDilatedDimensionNumbers("NHWC", "HWIO", "NHWC")
#> [1] "NHWC" "HWIO" "NHWC"
lax_RaggedDotDimensionNumbers(list(list(1L, 1L), list(integer(0), integer(0))), 0L, 0L)
#> $lhs_contracting
#> [1] 1
#> 
#> $rhs_contracting
#> [1] 1
#> 
#> $lhs_batch
#> integer(0)
#> 
#> $rhs_batch
#> integer(0)
#> 
#> $lhs_ragged_dimensions
#> [1] 0
#> 
#> $rhs_group_dimensions
#> [1] 0
#> 
#> attr(,"class")
#> [1] "jax_RaggedDotDimensionNumbers"
```
