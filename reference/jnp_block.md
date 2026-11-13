# Assemble an array from nested lists of blocks

Port of `jnp.block`: the innermost lists are concatenated along the last
axis, the next level along the second to last axis, and so on.

## Usage

``` r
jnp_block(arrays)
```

## Arguments

- arrays:

  A (nested) list of arrays.

## Value

An array.
