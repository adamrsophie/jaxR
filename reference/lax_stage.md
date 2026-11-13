# Lift a value into the current trace

Port of `jax.lax.stage`: logically the identity. Outside any
transformation it returns a jax array; inside one it returns a tracer of
the current trace.

## Usage

``` r
lax_stage(x)
```

## Arguments

- x:

  An array or R value.
