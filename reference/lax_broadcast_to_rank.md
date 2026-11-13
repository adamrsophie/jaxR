# Add leading dimensions of size one

Port of `jax.lax.broadcast_to_rank`: adds leading dimensions of size one
to give `x` rank `rank` (unchanged when it already has at least that
rank).

## Usage

``` r
lax_broadcast_to_rank(x, rank)
```

## Arguments

- x:

  Array.

- rank:

  Target rank.

## Value

An array.

## Examples

``` r
if (FALSE) { # \dontrun{
dim(lax_broadcast_to_rank(c(1, 2, 3), 3L))
} # }
```
