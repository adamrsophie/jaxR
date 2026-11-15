# Constrain the memory layout of values inside a computation

Port of `jax.experimental.layout.with_layout_constraint`: the leaves of
`x` get the layouts `layouts` (one layout, or a tree of layouts matching
`x`) inside a compiled function, which lets XLA store an intermediate
value in another dimension order. Values are unchanged; outside
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) the
arrays keep the row major layout of the CPU device.

## Usage

``` r
experimental_layout_with_layout_constraint(x, layouts)
```

## Arguments

- x:

  A pytree of arrays.

- layouts:

  A layout from
  [`experimental_layout_Layout()`](https://adamrsophie.github.io/jaxR/reference/experimental_layout_Layout.md),
  or a pytree of them.

## Value

`x`, with the constraints.

## Examples

``` r
if (FALSE) { # \dontrun{
f <- jit(function(x) experimental_layout_with_layout_constraint(x * 2,
  experimental_layout_Layout(c(1L, 0L))))
f(jnp_ones(c(2L, 3L)))
} # }
```
