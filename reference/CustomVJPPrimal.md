# Primal values of custom_vjp forward rules

Ports of `jax.custom_derivatives.CustomVJPPrimal` (a primal argument
with a flag telling whether it is perturbed, as received by a forward
rule in symbolic zeros mode) and
`jax.custom_derivatives.custom_vjp_primal_tree_values` (strip such
records from a pytree, keeping their values).

## Usage

``` r
CustomVJPPrimal(value, perturbed)

custom_vjp_primal_tree_values(tree)
```

## Arguments

- value:

  The primal value.

- perturbed:

  Whether the value carries a nonzero tangent.

- tree:

  A pytree whose leaves are `CustomVJPPrimal` objects.

## Value

`CustomVJPPrimal()` returns an object of class `jax_CustomVJPPrimal`;
`custom_vjp_primal_tree_values()` the pytree of values.

## Examples

``` r
p <- list(CustomVJPPrimal(1, TRUE), CustomVJPPrimal(2, FALSE))
custom_vjp_primal_tree_values(p)
#> [[1]]
#> [1] 1
#> 
#> [[2]]
#> [1] 2
#> 
```
