# Bin indices of values

Port of `jnp.digitize`: the index of the bin of each value of `x`, for
increasing or decreasing `bins`.

## Usage

``` r
jnp_digitize(x, bins, right = FALSE, method = NULL)
```

## Arguments

- x:

  Array.

- bins:

  1-D monotonic array of bin edges.

- right:

  Whether bins include their right edge.

- method:

  Search method passed to
  [`jnp_searchsorted()`](https://adamrsophie.github.io/jaxR/reference/jnp_searchsorted.md).

## Value

An int32 array.
