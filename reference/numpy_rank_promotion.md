# Run code with a configuration option set

Ports of the context managers `jax.numpy_rank_promotion`,
`jax.default_matmul_precision`, `jax.threefry_partitionable` and
`jax.default_prng_impl`: `code` is evaluated with the option set to
`new_val`.

## Usage

``` r
numpy_rank_promotion(new_val, code)

default_matmul_precision(new_val, code)

threefry_partitionable(new_val, code)

default_prng_impl(new_val, code)
```

## Arguments

- new_val:

  The value of the option while `code` runs.

- code:

  Expression to evaluate.

## Value

The value of `code`.
