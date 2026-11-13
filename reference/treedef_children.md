# Treedef helpers

Ports of `jax.tree_util.treedef_children`, `treedef_is_leaf` and
`treedef_tuple`. `treedef_is_leaf()` is `TRUE` for a treedef with one
node: a leaf, `NULL`, or a container without children (as in JAX).
`treedef_num_leaves()` is the port of `PyTreeDef.num_leaves`.

## Usage

``` r
treedef_num_leaves(treedef)

treedef_children(treedef)

treedef_is_leaf(treedef)

treedef_tuple(treedefs)
```

## Arguments

- treedef:

  A treedef (from
  [`tree_structure()`](https://adamrsophie.github.io/jaxR/reference/tree_flatten.md)).

- treedefs:

  A list of treedefs.

## Value

`treedef_children()`: a list of treedefs; `treedef_is_leaf()`: a
logical; `treedef_tuple()`: the treedef of an unnamed list of the given
structures; `treedef_num_leaves()`: the number of leaves.
