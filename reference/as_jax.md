# Convert R data to a jax array

Port of `jnp.asarray` for R inputs. R doubles become `float32` arrays
unless x64 is enabled, R integers `int32`, logicals `bool`, complex
values `complex64` (`complex128` with x64). A length one vector without
`dim` is a rank 0 array; R numeric scalars are weakly typed, as Python
scalars are in JAX.

## Usage

``` r
as_jax(x, dtype = NULL)
```

## Arguments

- x:

  R data, a jax array, or a tracer.

- dtype:

  Optional dtype.

## Value

A jax array (or tracer).
