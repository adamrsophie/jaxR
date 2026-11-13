# Delete or insert entries along an axis

Ports of `jnp.delete` and `jnp.insert`. `obj` is a 0-based index, a
vector or array of indices, a boolean mask (delete only) or a slice
built with [`sl()`](https://adamrsophie.github.io/jaxR/reference/sl.md).
Deleting with an index array needs concrete indices unless
`assume_unique_indices = TRUE`.

## Usage

``` r
jnp_delete(arr, obj, axis = NULL, assume_unique_indices = FALSE)

jnp_insert(arr, obj, values, axis = NULL)
```

## Arguments

- arr:

  Array.

- obj:

  Indices.

- axis:

  0-based axis; `NULL` flattens `arr`.

- assume_unique_indices:

  Whether the integer indices are known to be unique (enables use under
  `jit`).

- values:

  Values to insert (broadcast to the inserted slots).

## Value

An array.
