# Approximate top k

Ports of `jax.lax.approx_max_k` and `jax.lax.approx_min_k`: the `k`
largest (smallest) values along `reduction_dimension` and their 0-based
indices. On CPU the result is exact (XLA's fallback sorts); the order of
tied values is implementation defined.

## Usage

``` r
lax_approx_max_k(
  operand,
  k,
  reduction_dimension = -1L,
  recall_target = 0.95,
  reduction_input_size_override = -1L,
  aggregate_to_topk = TRUE
)

lax_approx_min_k(
  operand,
  k,
  reduction_dimension = -1L,
  recall_target = 0.95,
  reduction_input_size_override = -1L,
  aggregate_to_topk = TRUE
)
```

## Arguments

- operand:

  Floating point array.

- k:

  Number of values.

- reduction_dimension:

  0-based dimension searched (negative counts from the end).

- recall_target:

  Recall target of the approximation.

- reduction_input_size_override:

  When positive, the size used to evaluate the recall instead of the
  size of the reduction dimension.

- aggregate_to_topk:

  When `FALSE` the results are not reduced to `k` values; the size of
  the reduction dimension is then implementation defined (at least `k`).

## Value

A list of the values and their `int32` indices.

## Examples

``` r
if (FALSE) { # \dontrun{
lax_approx_max_k(c(3, 1, 4, 1, 5, 9, 2, 6), 3L)
} # }
```
