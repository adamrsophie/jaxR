# Softmax, log-sum-exp and standardization

Ports of `jax.nn.softmax`, `jax.nn.log_softmax`, `jax.nn.logsumexp`,
`jax.nn.logmeanexp` and `jax.nn.standardize`. Axes are 0-based;
`axis = NULL` reduces over all axes. `where` is an optional boolean mask
of the elements to include: masked out elements are 0 in `nn_softmax`
and `-Inf` in `nn_log_softmax`.

## Usage

``` r
nn_logsumexp(
  a,
  axis = NULL,
  b = NULL,
  keepdims = FALSE,
  return_sign = FALSE,
  where = NULL
)

nn_logmeanexp(x, axis = NULL, where = NULL, keepdims = FALSE)

nn_log_softmax(x, axis = -1L, where = NULL)

nn_softmax(x, axis = -1L, where = NULL)

nn_standardize(
  x,
  axis = -1L,
  mean = NULL,
  variance = NULL,
  epsilon = 1e-05,
  where = NULL,
  algorithm = "fast"
)
```

## Arguments

- axis:

  Axis or axes (0-based) to reduce over, or `NULL` for all axes.

- b:

  Optional scaling factors for the exponentials of `nn_logsumexp`,
  broadcastable to the shape of `a`.

- keepdims:

  Keep reduced axes as size one dimensions.

- return_sign:

  Return `list(result, sign)` where `result` holds the logarithm of the
  absolute values of the sums.

- where:

  Optional boolean mask of the elements to include.

- x, a:

  Input array.

- mean, variance:

  Optional mean and variance used by `nn_standardize` (computed from `x`
  when `NULL`).

- epsilon:

  Correction added to the variance.

- algorithm:

  Variance algorithm of `nn_standardize`: `"fast"`
  (`mean(x^2) - mean(x)^2`, clipped at 0) or `"stable"`
  (`mean((x - mean(x))^2)`).

## Value

An array (a list for `nn_logsumexp` with `return_sign = TRUE`).

## Details

As in JAX 0.12.0.1 (`jax_softmax_custom_jvp = False`), `nn_softmax`
stops the gradient through the subtracted maximum; when the option
`jax_softmax_custom_jvp` is `TRUE` it uses the custom JVP rule
`y * (x_dot - sum(y * x_dot))` instead.

## Examples

``` r
if (FALSE) { # \dontrun{
x <- jnp_array(c(1, 2, 3))
nn_softmax(x)
nn_softmax(x, where = c(TRUE, FALSE, TRUE))
nn_logsumexp(x)
} # }
```
