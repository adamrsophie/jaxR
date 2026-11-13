# Vectorizing map

Port of `jax.vmap`. Axes are 0-based; `NULL` means not mapped.

## Usage

``` r
vmap(fun, in_axes = 0L, out_axes = 0L, axis_name = NULL, axis_size = NULL)
```

## Arguments

- fun:

  Function to vectorize.

- in_axes:

  Integer, `NULL`, or a list (one entry per argument, each possibly a
  nested list or a registered pytree node) giving the mapped axis of
  each input.

- out_axes:

  Like `in_axes` for the outputs.

- axis_name:

  Optional name for collectives (`lax_psum` and friends).

- axis_size:

  Size of the mapped axis; needed when no input is mapped. When given it
  must agree with the sizes of the mapped input axes.

## Value

The vectorized function.
