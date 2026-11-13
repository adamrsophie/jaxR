# Scaled matrix multiplication

`nn_scaled_matmul` ports `jax.nn.scaled_matmul`: a block scaled batched
matmul of `lhs` `(B, M, K)` and `rhs` `(B, N, K)` with scales
`(B, M, K_a)` and `(B, N, K_b)`, equivalent to repeating each scale over
its block of `K / K_a` (or `K / K_b`) entries and contracting over `K`.
It uses JAX's CPU lowering (inputs are cast to
`preferred_element_type`). `nn_scaled_dot_general` ports
`jax.nn.scaled_dot_general` without quantization configs (it is then
[`lax_dot_general()`](https://adamrsophie.github.io/jaxR/reference/lax_dot_general.md));
the block scaled configurations need float8 and float4 dtypes, which
jaxR does not have.

## Usage

``` r
nn_scaled_matmul(
  lhs,
  rhs,
  lhs_scales,
  rhs_scales,
  preferred_element_type = "float32"
)

nn_scaled_dot_general(
  lhs,
  rhs,
  dimension_numbers,
  preferred_element_type = "float32",
  configs = NULL,
  implementation = NULL
)
```

## Arguments

- lhs, rhs:

  Operands.

- lhs_scales, rhs_scales:

  Block scales.

- preferred_element_type:

  Output dtype (default `"float32"`).

- dimension_numbers:

  `list(list(lhs_contracting, rhs_contracting), list(lhs_batch, rhs_batch))`
  with 0-based dimensions, as for
  [`lax_dot_general()`](https://adamrsophie.github.io/jaxR/reference/lax_dot_general.md).

- configs:

  Must be `NULL`.

- implementation:

  Deprecated and ignored (a warning is given).

## Value

An array of shape `(B, M, N)` (`nn_scaled_matmul`).

## Examples

``` r
if (FALSE) { # \dontrun{
a <- jnp_reshape(jnp_array(c(1, 2, 3)), c(1, 1, 3))
b <- jnp_reshape(jnp_array(c(4, 5, 6)), c(1, 1, 3))
s <- jnp_reshape(jnp_array(0.5), c(1, 1, 1))
nn_scaled_matmul(a, b, s, s)
} # }
```
