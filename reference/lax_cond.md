# Conditional execution

Port of `jax.lax.cond`: applies `true_fun(...)` or `false_fun(...)` to
the operands depending on the scalar `pred`. Both branches are traced;
the predicate may be traced.

## Usage

``` r
lax_cond(pred, true_fun, false_fun, ...)
```

## Arguments

- pred:

  Scalar predicate.

- true_fun, false_fun:

  Functions of the operands with identical output types.

- ...:

  Operands (pytrees).
