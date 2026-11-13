# Set operations on 1-D arrays

Ports of `jnp.union1d`, `jnp.intersect1d`, `jnp.setdiff1d`,
`jnp.setxor1d` and `jnp.isin`. Inputs are flattened. Without `size` the
inputs must be concrete; with a static `size` the result has exactly
`size` entries, padded with `fill_value`.

## Usage

``` r
jnp_union1d(ar1, ar2, size = NULL, fill_value = NULL)

jnp_setdiff1d(ar1, ar2, assume_unique = FALSE, size = NULL, fill_value = NULL)

jnp_setxor1d(ar1, ar2, assume_unique = FALSE, size = NULL, fill_value = NULL)

jnp_intersect1d(
  ar1,
  ar2,
  assume_unique = FALSE,
  return_indices = FALSE,
  size = NULL,
  fill_value = NULL
)

jnp_isin(
  element,
  test_elements,
  assume_unique = FALSE,
  invert = FALSE,
  method = "auto"
)
```

## Arguments

- ar1, ar2:

  Arrays.

- size:

  Optional static output size.

- fill_value:

  Padding value when `size` exceeds the number of results.

- assume_unique:

  Whether the inputs are known to have no repeated entries.

- return_indices:

  `intersect1d`: also return the 0-based indices of the common values in
  `ar1` and `ar2`.

- element, test_elements:

  `isin`: test each entry of `element` for membership in
  `test_elements`.

- invert:

  `isin`: return the negated result.

- method:

  `isin`: `"auto"`, `"compare_all"`, `"binary_search"` or `"sort"`.

## Value

An array (`intersect1d` with `return_indices = TRUE`: a list of three
arrays; `isin`: a boolean array of the shape of `element`).
