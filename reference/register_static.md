# Register a class as a static pytree node

Port of `jax.tree_util.register_static`: instances have no leaves; the
object itself is kept in the treedef (so
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) treats it
as static).

## Usage

``` r
register_static(cls)
```

## Arguments

- cls:

  Class name.

## Value

`cls`, invisibly.
