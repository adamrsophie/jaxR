# Trace a function to a jaxpr

Port of `jax.make_jaxpr`.

## Usage

``` r
make_jaxpr(fun, static_argnums = integer(0), return_shape = FALSE)
```

## Arguments

- fun:

  Function of arrays (or pytrees of arrays).

- static_argnums:

  0-based positions of arguments treated as static; negative values
  count from the end.

- return_shape:

  Whether to return also the output structure, as a pytree like the
  output of `fun` with a
  [`ShapeDtypeStruct()`](https://adamrsophie.github.io/jaxR/reference/ShapeDtypeStruct.md)
  at each leaf.

## Value

A function with the same arguments as `fun` that returns the closed
jaxpr, or `list(jaxpr, shapes)` with `return_shape`.
