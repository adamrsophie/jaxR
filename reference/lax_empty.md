# Array of unspecified values

Port of `jax.lax.empty`. On CPU the values are zeros.

## Usage

``` r
lax_empty(shape, dtype)
```

## Arguments

- shape:

  Integer vector.

- dtype:

  Dtype.

## Value

An array.

## Examples

``` r
if (FALSE) { # \dontrun{
lax_empty(3L, "float32")
} # }
```
