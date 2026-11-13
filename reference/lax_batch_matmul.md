# Batched matrix multiplication

Port of `jax.lax.batch_matmul`: operands of equal rank (at least 2),
multiplied over their last two dimensions with the leading dimensions as
batch dimensions.

## Usage

``` r
lax_batch_matmul(lhs, rhs, precision = NULL)
```

## Arguments

- lhs, rhs:

  Arrays.

- precision:

  Optional precision (see
  [lax_Precision](https://adamrsophie.github.io/jaxR/reference/lax_enums.md)).

## Value

An array.

## Examples

``` r
if (FALSE) { # \dontrun{
lax_batch_matmul(array(1, c(2, 3, 4)), array(1, c(2, 4, 5)))
} # }
```
