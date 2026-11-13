# Reverse the order of elements

Ports of `jnp.flip` (along the given axes, all by default), `jnp.fliplr`
(axis 1) and `jnp.flipud` (axis 0).

## Usage

``` r
jnp_flip(m, axis = NULL)

jnp_fliplr(m)

jnp_flipud(m)
```

## Arguments

- m:

  Array.

- axis:

  0-based axis or axes; `NULL` flips all axes.

## Value

An array.
