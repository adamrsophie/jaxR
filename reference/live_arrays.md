# Live arrays

Port of `jax.live_arrays`: the jax arrays that are still reachable from
R (arrays freed by R's garbage collector are not listed; call
[`gc()`](https://rdrr.io/r/base/gc.html) first to collect garbage).

## Usage

``` r
live_arrays(platform = NULL)
```

## Arguments

- platform:

  Ignored (one CPU device).

## Value

A list of jax arrays.
