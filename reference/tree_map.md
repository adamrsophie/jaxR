# Map and reduce over the leaves of pytrees

Ports of `jax.tree.map`, `jax.tree.all` and `jax.tree.reduce`.
`tree_map()` applies `f` to the leaves of one or more pytrees of the
same structure. `tree_all()` is `TRUE` when every leaf is true.
`tree_reduce()` folds the leaves from the left with `f`.

## Usage

``` r
tree_map(f, tree, ..., is_leaf = NULL)

tree_all(tree, is_leaf = NULL)

tree_reduce(f, tree, initializer, is_leaf = NULL)
```

## Arguments

- f:

  Function applied to leaves (`tree_map()`), or a function of two
  arguments that combines the accumulated value with the next leaf
  (`tree_reduce()`).

- tree:

  Pytree.

- ...:

  Further pytrees with the same structure as `tree`.

- is_leaf:

  Optional leaf predicate.

- initializer:

  Optional start value of `tree_reduce()`. Without it a tree with no
  leaves is an error.

## Value

`tree_map()`: a pytree with the structure of `tree`. `tree_all()`: a
logical. `tree_reduce()`: the accumulated value.
