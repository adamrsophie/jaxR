# Activation functions

Ports of the `jax.nn` activation functions. All are elementwise except
`nn_glu`, which splits `x` in two halves along `axis` (0-based).
`nn_relu`, `nn_relu6` and `nn_log1mexp` have custom JVP rules as in JAX:
the derivative of `nn_relu` at 0 is 0, and the derivative of `nn_relu6`
is 0 at 0 and at 6. `nn_swish` is an alias of `nn_silu`, `nn_hard_swish`
of `nn_hard_silu` and `nn_tanh` of
[`jnp_tanh()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md).

## Usage

``` r
nn_identity(x)

nn_relu(x)

nn_squareplus(x, b = 4L)

nn_softplus(x)

nn_sparse_plus(x)

nn_soft_sign(x)

nn_sigmoid(x)

nn_sparse_sigmoid(x)

nn_silu(x)

nn_swish(x)

nn_mish(x)

nn_log_sigmoid(x)

nn_elu(x, alpha = 1)

nn_leaky_relu(x, negative_slope = 0.01)

nn_hard_tanh(x)

nn_celu(x, alpha = 1)

nn_selu(x)

nn_gelu(x, approximate = TRUE)

nn_glu(x, axis = -1L)

nn_relu6(x)

nn_hard_sigmoid(x)

nn_hard_silu(x)

nn_hard_swish(x)

nn_log1mexp(x)

nn_tanh(x)
```

## Arguments

- x:

  Input array.

- b:

  Smoothness parameter of `nn_squareplus`.

- alpha:

  Scalar or array of alpha values (`nn_elu`, `nn_celu`).

- negative_slope:

  Slope for negative inputs of `nn_leaky_relu`.

- approximate:

  Use the tanh approximation of `nn_gelu` (default) or the exact
  formulation with `erfc`.

- axis:

  0-based axis along which `nn_glu` splits its input.

## Value

An array.

## Details

Arguments are not promoted to floating point unless JAX does so:
`nn_relu` of an `int32` array is `int32`, while `nn_relu6` and `nn_gelu`
return floating point arrays.

## Examples

``` r
if (FALSE) { # \dontrun{
x <- jnp_array(c(-2, -1, -0.5, 0, 0.5, 1, 2))
nn_relu(x)
nn_gelu(x, approximate = FALSE)
grad(function(v) nn_relu(v))(0)
} # }
```
