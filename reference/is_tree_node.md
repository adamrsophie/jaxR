# Test whether a class is a pytree node type

Port of `jax.tree_util.is_tree_node`. R has no type objects: pass a
class name (`"list"` and `"NULL"` are the built-in containers) or an
object (its class is used).

## Usage

``` r
is_tree_node(typ)
```

## Arguments

- typ:

  A class name or an object.

## Value

A logical.
