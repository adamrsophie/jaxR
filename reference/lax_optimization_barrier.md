# Optimization barrier

Port of `jax.lax.optimization_barrier`: prevents XLA from moving
computations across.

## Usage

``` r
lax_optimization_barrier(operand)
```

## Arguments

- operand:

  A pytree of arrays.
