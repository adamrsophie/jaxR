# Take elements along several axes

Port of `jax.lax.index_take`: `src[idxs[[1]][k], idxs[[2]][k], ...]`
along `axes` for every `k`. Indices wrap around.

## Usage

``` r
lax_index_take(src, idxs, axes)
```

## Arguments

- src:

  Array.

- idxs:

  List of 1-D integer arrays of equal length, one per entry of `axes`.

- axes:

  0-based axes of `src` (ascending).

## Value

An array whose leading dimension indexes the taken elements.
