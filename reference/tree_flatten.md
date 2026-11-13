# Flatten a pytree

Containers are plain lists, `NULL` and classes registered with
[`register_pytree_node()`](https://adamrsophie.github.io/jaxR/reference/register_pytree_node.md);
everything else is a leaf. A list whose elements all have distinct names
plays the role of a Python dict: as in JAX its leaves are visited in
sorted key order, and two such lists are matched by name, not by
position (in
[`tree_map()`](https://adamrsophie.github.io/jaxR/reference/tree_map.md),
cotangents, `in_axes` and so on).
[`tree_unflatten()`](https://adamrsophie.github.io/jaxR/reference/tree_unflatten.md)
restores the original order of the elements.

## Usage

``` r
tree_flatten(tree, is_leaf = NULL)

tree_leaves(tree, is_leaf = NULL)

tree_structure(tree, is_leaf = NULL)
```

## Arguments

- tree:

  A pytree (nested lists of arrays).

- is_leaf:

  Optional predicate marking subtrees to treat as leaves.

## Value

`tree_flatten()`: a list with `leaves` and `treedef`. `tree_leaves()`:
the list of leaves. `tree_structure()`: the treedef.
