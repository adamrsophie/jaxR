# Replace elements by mask or flat index

Ports of `jnp.place` and `jnp.put`. JAX arrays are immutable, so
`inplace` must be `FALSE` and the updated array is returned.

## Usage

``` r
jnp_place(arr, mask, vals, inplace = TRUE)

jnp_put(a, ind, v, mode = NULL, inplace = TRUE)
```

## Arguments

- arr, a:

  Array to update.

- mask:

  Boolean array with as many elements as `arr`.

- vals, v:

  Values (repeated as needed).

- inplace:

  Must be `FALSE`.

- ind:

  Flat 0-based indices into `a`.

- mode:

  `NULL` (out of bound indices are dropped), `"clip"` or `"wrap"`.

## Value

An array.
