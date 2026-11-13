# Register a class as a pytree node

Register a class as a pytree node

## Usage

``` r
register_pytree_node(class, flatten, unflatten)
```

## Arguments

- class:

  Class name (matched with
  [`inherits()`](https://rdrr.io/r/base/class.html)).

- flatten:

  Function of a node returning `list(children = <list>, aux = <any>)`.

- unflatten:

  Function of `(aux, children)` returning a node.

## Value

Invisibly `NULL`.
