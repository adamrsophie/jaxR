# Scan a function over leading array axes while carrying state

Port of `jax.lax.scan`: `f(carry, x)` returns `list(new_carry, y)`; the
result is `list(final_carry, stacked_ys)`.

## Usage

``` r
lax_scan(f, init, xs = NULL, length = NULL, reverse = FALSE, unroll = 1L)
```

## Arguments

- f:

  Function `(carry, x)`.

- init:

  Initial carry (pytree).

- xs:

  Pytree of arrays scanned along their leading axis, or `NULL`.

- length:

  Number of iterations (required when `xs` is `NULL`).

- reverse:

  Scan from the end.

- unroll:

  Number of iterations per loop step, or a logical. Validated as in JAX;
  the loop is always left rolled.
