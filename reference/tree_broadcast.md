# Broadcast a pytree prefix over a full pytree

Port of `jax.tree.broadcast` (`jax.tree_util.tree_broadcast`): each leaf
of `prefix_tree` is repeated over the leaves of the corresponding
subtree of `full_tree`.

## Usage

``` r
tree_broadcast(prefix_tree, full_tree, is_leaf = NULL)
```

## Arguments

- prefix_tree:

  A pytree that is a prefix of `full_tree`.

- full_tree:

  A pytree.

- is_leaf:

  Optional leaf predicate for `prefix_tree`.

## Value

A pytree with the structure of `full_tree`.
