# Operators and math functions for jax arrays

Arithmetic (`+ - * / ^ %% %/%`), comparison, logical (`& | !`) operators
and `%*%` follow `jax.numpy` semantics (broadcasting, type promotion).
Math group functions (`exp`, `log`, `sqrt`, `abs`, ...) and summaries
(`sum`, `max`, ...) map to the `jnp_` functions. These work on tracers
too, so ordinary R code is transformable by
[`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md),
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) and
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md).

## Usage

``` r
# S3 method for class 'jax_value'
Ops(e1, e2)

# S3 method for class 'jax_value'
matrixOps(x, y)

# S3 method for class 'jax_value'
chooseOpsMethod(x, y, mx, my, cl, reverse)

# S3 method for class 'jax_value'
Math(x, ...)

# S3 method for class 'jax_value'
Summary(..., na.rm = FALSE)

# S3 method for class 'jax_value'
mean(x, ...)

# S3 method for class 'jax_value'
t(x)
```

## Arguments

- e1, e2, x:

  Jax values or R data.

- y:

  Second operand.

- mx, my, cl, reverse:

  See
  [`base::chooseOpsMethod()`](https://rdrr.io/r/base/chooseOpsMethod.html).

- ...:

  Further arguments.

- na.rm:

  Ignored.
