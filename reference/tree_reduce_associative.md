# Reduce the leaves of a pytree with an associative operation

Port of `jax.tree.reduce_associative`: a balanced (parallel) reduction.

## Usage

``` r
tree_reduce_associative(operation, tree, identity, is_leaf = NULL)
```

## Arguments

- operation:

  Associative function of two values.

- tree:

  A pytree.

- identity:

  Value returned for a tree without leaves (required then).

- is_leaf:

  Optional leaf predicate.

## Value

The reduced value.
