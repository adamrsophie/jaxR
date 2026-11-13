# Register a pytree node class with key paths

Port of `jax.tree_util.register_pytree_with_keys`: like
[`register_pytree_node()`](https://adamrsophie.github.io/jaxR/reference/register_pytree_node.md),
with a flatten function that also names each child, so that
[`tree_flatten_with_path()`](https://adamrsophie.github.io/jaxR/reference/tree_flatten_with_path.md)
reports these keys.

## Usage

``` r
register_pytree_with_keys(
  nodetype,
  flatten_with_keys,
  unflatten_func,
  flatten_func = NULL
)
```

## Arguments

- nodetype:

  Class name.

- flatten_with_keys:

  Function of a node returning
  `list(children = <list of list(key, child) pairs>, aux = <any>)`.

- unflatten_func:

  Function of `(aux, children)` returning a node.

- flatten_func:

  Optional faster flatten function returning `list(children, aux)` with
  the same children and aux.

## Value

`nodetype`, invisibly.
