# Flatten one level of a pytree

Ports of `jax.tree_util.flatten_one_level` and
`flatten_one_level_with_keys`.

## Usage

``` r
flatten_one_level(tree)

flatten_one_level_with_keys(tree)
```

## Arguments

- tree:

  A container (list, `NULL` or a registered node).

## Value

A list with `children` (for the `_with_keys` variant, `list(key, child)`
pairs) and `aux` (the names of a list, or the aux data of a registered
node).
