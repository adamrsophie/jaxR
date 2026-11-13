# Gradient utilities

`optimizers_l2_norm` ports `l2_norm`, the l2 norm of all leaves of a
pytree of arrays (useful for weight decay). `optimizers_clip_grads`
ports `clip_grads`: it rescales a pytree of gradients whose l2 norm
exceeds `max_norm` to norm `max_norm`.

## Usage

``` r
optimizers_l2_norm(tree)

optimizers_clip_grads(grad_tree, max_norm)
```

## Arguments

- tree, grad_tree:

  Pytree of arrays.

- max_norm:

  Maximum norm.

## Value

A scalar array (`l2_norm`) or a pytree with the structure of
`grad_tree`.

## Examples

``` r
if (FALSE) { # \dontrun{
optimizers_clip_grads(list(jnp_array(c(3, 4))), 1)
} # }
```
