# Piecewise defined function

Port of `jnp.piecewise`. Each function is evaluated on every element
(through
[`jnp_vectorize()`](https://adamrsophie.github.io/jaxR/reference/jnp_vectorize.md)
and
[`lax_switch()`](https://adamrsophie.github.io/jaxR/reference/lax_switch.md));
where several conditions hold the last wins.

## Usage

``` r
jnp_piecewise(x, condlist, funclist, ...)
```

## Arguments

- x:

  Array.

- condlist:

  List of boolean arrays (or a boolean array whose first axis indexes
  the conditions).

- funclist:

  List of functions or constants: one per condition, plus optionally a
  default as the last entry.

- ...:

  Extra arguments passed to the functions.

## Value

An array with the dtype of `x`.
