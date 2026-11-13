# Histograms

Ports of `jnp.histogram_bin_edges`, `jnp.histogram`, `jnp.histogram2d`
and `jnp.histogramdd`. `bins` is a number of equal width bins or an
array of edges; the last bin includes its right edge. String `bins` are
not supported.

## Usage

``` r
jnp_histogram_bin_edges(a, bins = 10L, range = NULL, weights = NULL)

jnp_histogram(a, bins = 10L, range = NULL, weights = NULL, density = NULL)

jnp_histogram2d(x, y, bins = 10L, range = NULL, weights = NULL, density = NULL)

jnp_histogramdd(
  sample,
  bins = 10L,
  range = NULL,
  weights = NULL,
  density = NULL
)
```

## Arguments

- a, x, y:

  Arrays of values.

- bins:

  Number of bins, or bin edges; for 2-D and D-D histograms one per
  dimension may be given in a list.

- range:

  Optional `c(lower, upper)`; for 2-D and D-D histograms a list with one
  pair (or `NULL`) per dimension.

- weights:

  Optional weights.

- density:

  Normalize to a probability density.

- sample:

  Array of shape `(N, D)`.

## Value

`jnp_histogram_bin_edges`: the edges. `jnp_histogram`: a list of counts
and edges. `jnp_histogram2d`: a list of counts, x edges and y edges.
`jnp_histogramdd`: a list of counts and a list of edges.
