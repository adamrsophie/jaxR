# Dtype of a value

Port of `jax.lax.dtype` (`jax._src.dtypes.dtype`): jax arrays keep their
dtype; R scalars, R vectors and dtype names are canonicalized (64-bit
types map to 32-bit ones unless x64 is enabled).

## Usage

``` r
lax_dtype(x)
```

## Arguments

- x:

  A jax array, R data or a dtype name.

## Value

A dtype name.

## Examples

``` r
lax_dtype(1.5)
#> [1] "float32"
lax_dtype("float64")
#> [1] "float32"
```
