# Clip values to an interval

Port of `jnp.clip`: `minimum(max, maximum(min, arr))`; either bound may
be `NULL`. Complex values are rejected.

## Usage

``` r
jnp_clip(arr = NULL, min = NULL, max = NULL)
```

## Arguments

- arr:

  Array.

- min, max:

  Lower and upper bounds (arrays or scalars).

## Value

An array.
