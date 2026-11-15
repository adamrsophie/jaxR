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
same values. A function that already has a VJP rule (from
[`custom_vjp_defvjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md))
keeps it: JAX uses the rematerialization rules for it under
[`checkpoint()`](https://adamrsophie.github.io/jaxR/reference/checkpoint.md)
only, where jaxR keeps the VJP rule too. A later `custom_vjp_defremat()`
replaces rules from an earlier one.

## Usage

``` r
custom_vjp_defremat(f, fwd, rem, bwd, generic_batching = FALSE)

custom_vjp_defremat_with_logs(f, fwd, rem, bwd, generic_batching = FALSE)
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

- generic_batching:

  How [`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md)
  handles a primal output that is shared across the batch (unbatched),
  as in JAX. By default, when such an output has a nonzero cotangent and
  `bwd` gives a batched cotangent for an unbatched argument, that is an
  error, since summing over the batch would count the shared cotangent
  once per example. With `TRUE`, `bwd` runs with the whole shared
  cotangent given to the first example (and zeros to the others), which
  is right for a correct rule; likewise an output of `fwd` that is
  batched where the function's output is not takes the value of the
  first example.

## Value

A function with the custom VJP rule.
