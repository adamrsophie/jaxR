# Wait for pending side effects

Port of `jax.effects_barrier`: services host callbacks that are still
queued.

## Usage

``` r
effects_barrier()
```

## Value

`NULL`, invisibly.
