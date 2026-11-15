# Define a function with a custom VJP rule

Port of `jax.custom_vjp`. `fwd(...)` returns
`list(primal_out, residuals)`; `bwd(residuals, ct)` (preceded by any
`nondiff_argnums` arguments) returns a list with one cotangent per
differentiable argument (`NULL` for zero), each with the structure and
the leaf shapes of its argument. Forward mode differentiation
([`jvp()`](https://adamrsophie.github.io/jaxR/reference/jvp.md)) of such
a function raises an error, as in JAX.

## Usage

``` r
custom_vjp(fun, fwd = NULL, bwd = NULL, nondiff_argnums = integer(0))

custom_vjp_defvjp(
  f,
  fwd,
  bwd,
  symbolic_zeros = FALSE,
  optimize_remat = FALSE,
  generic_batching = FALSE
)
```

## Arguments

- fun:

  The function.

- fwd, bwd:

  Optional rules (can be set later with `custom_vjp_defvjp()`).

- nondiff_argnums:

  0-based positions of non-differentiable arguments.

- f:

  A function created by `custom_vjp()`.

- symbolic_zeros:

  Symbolic zeros mode (port of `symbolic_zeros=True` of
  `custom_vjp.defvjp`). The leaves of the arguments of `fwd` are then
  [`CustomVJPPrimal()`](https://adamrsophie.github.io/jaxR/reference/CustomVJPPrimal.md)
  records with elements `value` and `perturbed` (whether that argument
  is being differentiated), and the leaves of `ct` in `bwd` are
  [`SymbolicZero()`](https://adamrsophie.github.io/jaxR/reference/SymbolicZero.md)
  objects where a cotangent is known to be zero. `bwd` may return
  `SymbolicZero` objects too.

- optimize_remat:

  Run `fwd` through the `remat_opt` primitive (port of
  `optimize_remat=True` of `custom_vjp.defvjp`): values and derivatives
  are unchanged. jaxR keeps the forward computation whole (its dead code
  elimination has no rule that drops the primal part), so this saves no
  memory.

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

A function with a custom VJP rule.
