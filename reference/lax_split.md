# Split an array

Port of `jax.lax.split`: chunks of sizes `sizes` along `axis`.

## Usage

``` r
lax_split(operand, sizes, axis = 0L)
```

## Arguments

- operand:

  Array.

- sizes:

  Integer vector of chunk sizes; they must add up to the size of `axis`.

- axis:

  0-based axis.

## Value

A list of arrays.

## Examples

``` r
if (FALSE) { # \dontrun{
lax_split(1:5, c(2L, 3L))
} # }
```
