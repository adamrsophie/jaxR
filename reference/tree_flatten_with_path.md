# Flatten a pytree with key paths

Ports of `jax.tree.flatten_with_path`, `jax.tree.leaves_with_path` and
`jax.tree.map_with_path` (and their `jax.tree_util` aliases).

## Usage

``` r
tree_flatten_with_path(tree, is_leaf = NULL, is_leaf_takes_path = FALSE)

tree_leaves_with_path(tree, is_leaf = NULL, is_leaf_takes_path = FALSE)

tree_map_with_path(f, tree, ..., is_leaf = NULL, is_leaf_takes_path = FALSE)
```

## Arguments

- tree:

  A pytree.

- is_leaf:

  Optional predicate marking subtrees to treat as leaves.

- is_leaf_takes_path:

  Whether `is_leaf` takes `(path, x)` instead of `x`.

- f:

  Function of `(path, leaf, ...)` (one leaf per tree).

- ...:

  Further pytrees with the structure of `tree` (`tree_map_with_path()`).

## Value

`tree_flatten_with_path()`: a list with `leaves` (a list of
`list(path, leaf)` pairs, paths being lists of key entries) and
`treedef`. `tree_leaves_with_path()`: the pairs. `tree_map_with_path()`:
the mapped tree.

## Examples

``` r
tree_leaves_with_path(list(a = 1, b = list(2, 3)))
#> [[1]]
#> [[1]][[1]]
#> [[1]][[1]][[1]]
#> DictKey(key='a')
#> 
#> 
#> [[1]][[2]]
#> [1] 1
#> 
#> 
#> [[2]]
#> [[2]][[1]]
#> [[2]][[1]][[1]]
#> DictKey(key='b')
#> 
#> [[2]][[1]][[2]]
#> SequenceKey(idx=0)
#> 
#> 
#> [[2]][[2]]
#> [1] 2
#> 
#> 
#> [[3]]
#> [[3]][[1]]
#> [[3]][[1]][[1]]
#> DictKey(key='b')
#> 
#> [[3]][[1]][[2]]
#> SequenceKey(idx=1)
#> 
#> 
#> [[3]][[2]]
#> [1] 3
#> 
#> 
tree_map_with_path(function(p, x) paste(keystr(p), x), list(a = 1, b = list(2, 3)))
#> $a
#> [1] "['a'] 1"
#> 
#> $b
#> $b[[1]]
#> [1] "['b'][0] 2"
#> 
#> $b[[2]]
#> [1] "['b'][1] 3"
#> 
#> 
```
