# Sort keys and permute values alike

Port of `jax.lax.sort_key_val`.

## Usage

``` r
lax_sort_key_val(keys, values, dimension = -1L, is_stable = TRUE)
```

## Arguments

- keys:

  Array of sort keys.

- values:

  Array with the shape of `keys`.

- dimension:

  0-based dimension.

- is_stable:

  Use a stable sort.

## Value

`list(sorted_keys, permuted_values)`.
