# Select slices with a boolean condition

Ports of `jnp.compress` (slices along `axis` where `condition` is true)
and `jnp.extract` (elements of the flattened array). The result size
depends on the data, so inside transformations a static `size` is
required; the result is then padded with `fill_value` or truncated.

## Usage

``` r
jnp_compress(
  condition,
  a,
  axis = NULL,
  size = NULL,
  fill_value = 0,
  out = NULL
)

jnp_extract(condition, arr, size = NULL, fill_value = 0)
```

## Arguments

- condition:

  Boolean (or numeric) array; 1-D for `jnp_compress`.

- a, arr:

  Array.

- axis:

  0-based axis; `NULL` flattens `a`.

- size:

  Optional static result length.

- fill_value:

  Padding value when `size` is given.

- out:

  Not supported (must be `NULL`).

## Value

An array.
