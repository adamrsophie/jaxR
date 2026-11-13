# Turn a symcompR expression into a jaxR function

The analogue of SymPy's `lambdify(args, expr, modules = "jax")`: the
returned function evaluates a symcompR expression with jaxR operations,
so it can be differentiated with
[`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md),
vectorized with
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) and
compiled with
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md).

## Usage

``` r
jax_lambdify(args, expr)
```

## Arguments

- args:

  A symcompR symbol or a list of symbols (the function's arguments, in
  order).

- expr:

  A symcompR expression, or a list of expressions (the function then
  returns a list).

## Value

An R function of jax arrays.
