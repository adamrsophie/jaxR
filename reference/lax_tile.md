# Tile an array

Port of `jax.lax.tile`: repeats `operand` `reps[i]` times along
dimension `i`.

## Usage

``` r
lax_tile(operand, reps)
```

## Arguments

- operand:

  Array.

- reps:

  Integer vector with one entry per dimension of `operand`.

## Value

An array.

## Examples

``` r
if (FALSE) { # \dontrun{
lax_tile(matrix(1:4, 2, byrow = TRUE), c(2L, 3L))
} # }
```
