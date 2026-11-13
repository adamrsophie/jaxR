# Ragged dot products

Ports of `jax.lax.ragged_dot` and `jax.lax.ragged_dot_general`. In
`lax_ragged_dot` the rows of `lhs` (shape `(m, k)`) are split into
consecutive groups of sizes `group_sizes` (length `g`) and group `i` is
multiplied by `rhs[i, , ]` (shape `(g, k, n)`), giving an `(m, n)`
result. `lax_ragged_dot_general` takes the general dimension numbers of
[`lax_RaggedDotDimensionNumbers()`](https://adamrsophie.github.io/jaxR/reference/lax_dimension_numbers.md):
the lhs ragged dimension is a non contracting, contracting (the result
gets a leading group dimension) or batch dimension.

## Usage

``` r
lax_ragged_dot_general(
  lhs,
  rhs,
  group_sizes,
  ragged_dot_dimension_numbers,
  precision = NULL,
  preferred_element_type = NULL,
  group_offset = NULL
)

lax_ragged_dot(
  lhs,
  rhs,
  group_sizes,
  precision = NULL,
  preferred_element_type = NULL,
  group_offset = NULL
)
```

## Arguments

- lhs, rhs:

  Arrays.

- group_sizes:

  Integer array of group sizes.

- ragged_dot_dimension_numbers:

  A
  [`lax_RaggedDotDimensionNumbers()`](https://adamrsophie.github.io/jaxR/reference/lax_dimension_numbers.md)
  object.

- precision:

  Optional precision.

- preferred_element_type:

  Optional result dtype.

- group_offset:

  Not supported (as in JAX); must be `NULL`.

## Value

An array.

## Examples

``` r
if (FALSE) { # \dontrun{
lhs <- matrix(1, 5, 2)
rhs <- array(1:12, c(3, 2, 2))
lax_ragged_dot(lhs, rhs, c(2L, 1L, 2L))
} # }
```
