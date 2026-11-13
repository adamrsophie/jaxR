# Scaled dot product

Port of `jax.lax.scaled_dot`: `(lhs * lhs_scale) %*% (rhs * rhs_scale)`
computed in `bfloat16` with `preferred_element_type` accumulation
(default `bfloat16`). Each scale has the shape of its operand except
along the contracting dimensions, where the operand size must be a
multiple (at least 2) of the scale size. Missing scales are ones (JAX
uses `float8_e8m0fnu` ones, a dtype jaxR does not have; the values are
equal). Not differentiable, as in JAX.

## Usage

``` r
lax_scaled_dot(
  lhs,
  rhs,
  lhs_scale = NULL,
  rhs_scale = NULL,
  dimension_numbers = NULL,
  preferred_element_type = NULL
)
```

## Arguments

- lhs, rhs:

  Arrays.

- lhs_scale, rhs_scale:

  Optional scale arrays.

- dimension_numbers:

  Dot dimension numbers; by default the last dimension of `lhs` is
  contracted with the second to last of `rhs` and the leading dimensions
  are batch dimensions.

- preferred_element_type:

  Result dtype (`"bfloat16"` or `"float32"`).

## Value

An array.

## Examples

``` r
if (FALSE) { # \dontrun{
lax_scaled_dot(matrix(1, 2, 4), matrix(1, 4, 3), preferred_element_type = "float32")
} # }
```
