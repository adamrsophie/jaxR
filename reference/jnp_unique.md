# Unique elements of an array

Port of `jnp.unique` and of the array API functions `jnp.unique_all`,
`jnp.unique_counts`, `jnp.unique_inverse` and `jnp.unique_values`.
Without `size` the input must be concrete (outside of `jit`); with a
static `size` the result has exactly `size` entries along the unique
axis, padded with `fill_value` (by default the smallest unique value) or
truncated.

## Usage

``` r
jnp_unique(
  ar,
  return_index = FALSE,
  return_inverse = FALSE,
  return_counts = FALSE,
  axis = NULL,
  equal_nan = TRUE,
  size = NULL,
  fill_value = NULL,
  sorted = TRUE
)

jnp_unique_all(x, size = NULL, fill_value = NULL)

jnp_unique_counts(x, size = NULL, fill_value = NULL)

jnp_unique_inverse(x, size = NULL, fill_value = NULL)

jnp_unique_values(x, size = NULL, fill_value = NULL)
```

## Arguments

- ar, x:

  Array.

- return_index:

  Also return the 0-based indices of the first occurrences.

- return_inverse:

  Also return the indices that rebuild `ar` from the result.

- return_counts:

  Also return the number of occurrences of each unique value.

- axis:

  0-based axis along which to find unique slices; `NULL` flattens.

- equal_nan:

  Treat all NaN values as equal.

- size:

  Optional static number of unique values to return.

- fill_value:

  Padding value when `size` exceeds the number of unique values.

- sorted:

  Ignored (results are always sorted).

## Value

An array, or a list of arrays when any `return_*` flag is set.
`unique_all`, `unique_counts` and `unique_inverse` return named lists
with the fields `values`, `indices`, `inverse_indices` and `counts`.
