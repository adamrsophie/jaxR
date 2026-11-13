# Custom rematerialization rules

Ports of `custom_vjp.defremat` and `custom_vjp.defremat_with_logs`.
`fwd` is the forward rule of
[`custom_vjp_defvjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md);
`rem(res, ...)` receives its residuals and the arguments of the function
and returns `list(out, res2)`; `bwd` is the backward rule and receives
`res2`. jaxR runs `rem` right after `fwd` on the forward pass, which is
what JAX does outside
[`checkpoint()`](https://adamrsophie.github.io/jaxR/reference/checkpoint.md);
under
[`checkpoint()`](https://adamrsophie.github.io/jaxR/reference/checkpoint.md)
JAX defers `rem` to the backward pass, which saves memory but gives the
same values.

## Usage

``` r
custom_vjp_defremat(f, fwd, rem, bwd)

custom_vjp_defremat_with_logs(f, fwd, rem, bwd)
```

## Arguments

- f:

  A function created by
  [`custom_vjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md).

- fwd, bwd:

  Optional rules (can be set later with
  [`custom_vjp_defvjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md)).

- rem:

  The rematerialization rule.

## Value

A function with the custom VJP rule.
