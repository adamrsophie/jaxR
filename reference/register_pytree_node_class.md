# Register a class through its flatten methods

Ports of `jax.tree_util.register_pytree_node_class` and
`register_pytree_with_keys_class`. R classes have no methods of their
own, so the functions are found by name: `tree_flatten.<class>(x)`
(returning `list(children, aux)`),
`tree_unflatten.<class>(aux, children)` and, for the keyed variant,
`tree_flatten_with_keys.<class>(x)`.

## Usage

``` r
register_pytree_node_class(cls, envir = parent.frame())

register_pytree_with_keys_class(cls, envir = parent.frame())
```

## Arguments

- cls:

  Class name.

- envir:

  Environment where the methods are looked up.

## Value

`cls`, invisibly.
