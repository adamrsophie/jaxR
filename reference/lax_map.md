# Map a function over leading array axes

Port of `jax.lax.map` (sequential, via scan).

## Usage

``` r
lax_map(f, xs, batch_size = NULL)
```

## Arguments

- f:

  Function of one pytree slice.

- xs:

  Pytree of arrays.

- batch_size:

  Optional batch size: slices are processed in batches with vmap.
