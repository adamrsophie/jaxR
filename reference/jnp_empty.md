# Uninitialized arrays

Ports of `jnp.empty` and `jnp.empty_like`. On the CPU backend the
contents are zeros (JAX lowers `empty` to a zero broadcast there).

## Usage

``` r
jnp_empty(shape, dtype = NULL)

jnp_empty_like(prototype, dtype = NULL, shape = NULL)
```

## Arguments

- shape:

  Integer vector.

- dtype:

  Optional dtype (default float).

- prototype:

  Array whose shape and dtype are used.

## Value

An array.
