# Define a function with a custom vmap rule

Ports of `jax.custom_batching.custom_vmap` and `sequential_vmap`. A
`custom_vmap()` function behaves like `fun`, except under
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md), where
`vmap_rule` is used.

## Usage

``` r
custom_vmap(fun, vmap_rule = NULL)

custom_vmap_def_vmap(f, vmap_rule)

sequential_vmap(f)
```

## Arguments

- fun, f:

  A function. For `custom_vmap_def_vmap()`, a function created by
  `custom_vmap()`.

- vmap_rule:

  The vmap rule (can be set later with `custom_vmap_def_vmap()`).

## Value

A function with a custom vmap rule.

## Details

The rule is called as `vmap_rule(axis_size, in_batched, ...)`:
`axis_size` is the size of the mapped axis, `in_batched` is a list of
logicals with the structure of the arguments saying which of them are
batched, and `...` are the arguments, with the mapped axis first in
every batched one. It returns `list(out, out_batched)`, where
`out_batched` has the structure of `out` and says which outputs carry
the mapped axis (first).

As in JAX, a `custom_vmap()` function supports forward mode
differentiation but not reverse mode. To customize both, wrap the
`custom_vmap()` function in
[`custom_vjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md),
with
[`custom_vjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md)
on the outside.

`sequential_vmap(f)` is the special case whose rule calls `f` once per
element in a loop
([`lax_map()`](https://adamrsophie.github.io/jaxR/reference/lax_map.md)),
for functions that cannot handle a batch dimension.

## Examples

``` r
if (FALSE) { # \dontrun{
f <- custom_vmap(function(x, y) x + y, function(axis_size, in_batched, xs, ys) {
  list(xs * ys, TRUE)
})
f(2, 3)                                           # 5
vmap(f)(jnp_arange(3L), jnp_arange(1L, 4L))       # 0 2 6: the rule multiplies

g <- sequential_vmap(function(x) x + 1)
vmap(g)(jnp_arange(3L))
} # }
```
