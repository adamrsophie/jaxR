# Index and iterate key arrays

`keys[i]` indexes the leading (key array) dimensions of a typed key
array with the rules of `[` for jax arrays; `as.list(keys)` splits a key
array along its first dimension (port of `PRNGKeyArray.__iter__`).

## Usage

``` r
# S3 method for class 'jax_prng_key'
x[...]

# S3 method for class 'jax_prng_key'
as.list(x, ...)
```

## Arguments

- x:

  A key array from
  [`random_key()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  or
  [`random_split()`](https://adamrsophie.github.io/jaxR/reference/random_key.md).

- ...:

  Index arguments.

## Value

A key array, or a list of key arrays.
