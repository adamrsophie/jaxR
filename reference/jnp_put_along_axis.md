# Put values along an axis

Port of `jnp.put_along_axis`. Arrays are immutable, so `inplace = FALSE`
must be passed; the updated array is returned.

## Usage

``` r
jnp_put_along_axis(arr, indices, values, axis, inplace = TRUE, mode = NULL)
```

## Arguments

- arr:

  Array.

- indices:

  Integer array with the same number of dimensions as `arr`.

- values:

  Values, broadcastable to the shape of `indices`.

- axis:

  0-based axis; `NULL` flattens `arr`.

- inplace:

  Must be `FALSE`.

- mode:

  Out of bounds mode (see
  [`at_set()`](https://adamrsophie.github.io/jaxR/reference/at.md)).

## Value

The updated array.
