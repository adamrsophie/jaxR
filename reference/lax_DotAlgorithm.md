# Dot product algorithm

Port of `jax.lax.DotAlgorithm`: the storage types the operands are
rounded to, the accumulation type and the decomposition of the products.

## Usage

``` r
lax_DotAlgorithm(
  lhs_precision_type,
  rhs_precision_type,
  accumulation_type,
  lhs_component_count = 1L,
  rhs_component_count = 1L,
  num_primitive_operations = 1L,
  allow_imprecise_accumulation = FALSE
)
```

## Arguments

- lhs_precision_type, rhs_precision_type, accumulation_type:

  Dtypes.

- lhs_component_count, rhs_component_count, num_primitive_operations:

  Counts of the decomposition (1 for algorithms without one).

- allow_imprecise_accumulation:

  Whether some steps may accumulate in lower precision.

## Value

An object of class `jax_DotAlgorithm`.

## Examples

``` r
lax_DotAlgorithm("float32", "float32", "float32")
#> $lhs_precision_type
#> [1] "float32"
#> 
#> $rhs_precision_type
#> [1] "float32"
#> 
#> $accumulation_type
#> [1] "float32"
#> 
#> $lhs_component_count
#> [1] 1
#> 
#> $rhs_component_count
#> [1] 1
#> 
#> $num_primitive_operations
#> [1] 1
#> 
#> $allow_imprecise_accumulation
#> [1] FALSE
#> 
#> attr(,"class")
#> [1] "jax_DotAlgorithm"
```
