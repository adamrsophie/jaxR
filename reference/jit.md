# Just-in-time compilation

Port of `jax.jit`. The function is traced once per combination of
argument shapes, dtypes, weak types, pytree structure and static
arguments, lowered to StableHLO and compiled by XLA. Under other
transformations (grad, vmap, ...) the cached jaxpr is evaluated under
the active transformation.

## Usage

``` r
jit(
  fun,
  static_argnums = NULL,
  static_argnames = NULL,
  donate_argnums = integer(0),
  inline = FALSE
)
```

## Arguments

- fun:

  Function of arrays (or pytrees of arrays).

- static_argnums:

  0-based positions of arguments treated as compile time constants
  (negative positions count from the end). Static values are compared
  with [`identical()`](https://rdrr.io/r/base/identical.html), so two
  closures are the same static argument only when they are the same
  function with the same environment.

- static_argnames:

  Names of arguments treated as static. As in JAX, when only one of
  `static_argnums` and `static_argnames` is given the other is inferred
  from the parameters of `fun`, so a static parameter is static however
  it is passed.

- donate_argnums:

  Accepted for compatibility; ignored.

- inline:

  Ignored.

## Value

The compiled function.
