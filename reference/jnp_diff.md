# Differences between consecutive elements

Ports of `jnp.diff` (the `n`th discrete difference along `axis`; boolean
input gives `!=`) and `jnp.ediff1d` (differences of the flattened array
with optional values before and after).

## Usage

``` r
jnp_diff(a, n = 1L, axis = -1L, prepend = NULL, append = NULL)

jnp_ediff1d(ary, to_end = NULL, to_begin = NULL)
```

## Arguments

- a, ary:

  Array.

- n:

  Order of the difference.

- axis:

  0-based axis.

- prepend, append:

  Optional values joined before or after `a` along `axis`.

- to_end, to_begin:

  Optional values joined after or before the differences.

## Value

An array.
