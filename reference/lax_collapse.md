# Collapse dimensions into one

Port of `jax.lax.collapse`: dimensions
`[start_dimension, stop_dimension)` are raveled (row major) into a
single dimension.

## Usage

``` r
lax_collapse(operand, start_dimension, stop_dimension = NULL)
```

## Arguments

- operand:

  Array.

- start_dimension:

  First dimension collapsed (0-based, inclusive).

- stop_dimension:

  End of the collapsed range (exclusive); `NULL` collapses all
  dimensions after `start_dimension`.

## Value

An array.

## Examples

``` r
if (FALSE) { # \dontrun{
dim(lax_collapse(array(0, c(2, 3, 4)), 0L, 2L))
} # }
```
