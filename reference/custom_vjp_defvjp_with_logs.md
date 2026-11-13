# Custom VJP rules that log or take gradient accumulators

Ports of `custom_vjp.defvjp_with_logs` and
`custom_vjp.defvjp_with_accums`.

## Usage

``` r
custom_vjp_defvjp_with_logs(
  f,
  fwd,
  bwd,
  symbolic_zeros = FALSE,
  optimize_remat = FALSE
)

custom_vjp_defvjp_with_accums(
  f,
  fwd,
  bwd,
  symbolic_zeros = FALSE,
  optimize_remat = FALSE
)
```

## Arguments

- f:

  A function created by
  [`custom_vjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md).

- fwd, bwd:

  Optional rules (can be set later with
  [`custom_vjp_defvjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md)).

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

## Value

A function with the custom VJP rule.

## Details

With `custom_vjp_defvjp_with_logs()`, `bwd` returns a pair
`list(in_cts, logs)`: the usual cotangents and a named list of values to
log out of the backward pass (or `NULL`). Apply the VJP function with
its `with_logs` method to receive them: `vjp_fun$with_logs(ct)` returns
`list(arg_cts, logs)`. A plain call drops the logs. Log names must be
unique within a backward pass.

With `custom_vjp_defvjp_with_accums()`, `bwd(res, g, ...)` receives one
gradient accumulator per argument of the function (with the structure of
the arguments, see
[ad_ValAccum](https://adamrsophie.github.io/jaxR/reference/ad_ValAccum.md))
after its two usual arguments, pushes the cotangent contributions into
them and returns `NULL` or a named list of logs.

## Examples

``` r
if (FALSE) { # \dontrun{
f <- custom_vjp(function(x, y) x * y)
f <- custom_vjp_defvjp_with_logs(f, function(x, y) list(x * y, list(x, y)),
  function(res, g) list(list(g * res[[2]], g * res[[1]]), list(g = g)))
vjp(f, 3, 4)[[2]]$with_logs(1)
} # }
```
