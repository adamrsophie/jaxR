# Hessian

Port of `jax.hessian` (`jacfwd(jacrev(fun))`). For a symcompR expression
the call is forwarded to `symcompR::hessian()`.

## Usage

``` r
hessian(fun, argnums = 0L, has_aux = FALSE, holomorphic = FALSE, ...)
```

## Arguments

- fun:

  Function.

- argnums:

  0-based argument position(s); negative values count from the end. A
  list of positions (the analog of a Python tuple) always gives a list
  of Jacobians.

- has_aux:

  Whether `fun` returns `list(output, aux)`.

- holomorphic:

  Whether `fun` is promised to be holomorphic.

- ...:

  Passed to symcompR when `fun` is a symbolic expression. Not used
  otherwise.
