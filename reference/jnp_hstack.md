# Stack arrays

Ports of `jnp.hstack` (along axis 1, or 0 for 1-D inputs), `jnp.dstack`
(along axis 2 after
[`jnp_atleast_3d()`](https://adamrsophie.github.io/jaxR/reference/jnp_atleast_1d.md)),
`jnp.column_stack` (1-D inputs become columns) and `jnp.concat` (the
array API name of `jnp.concatenate`).

## Usage

``` r
jnp_hstack(tup, dtype = NULL)

jnp_dstack(tup, dtype = NULL)

jnp_column_stack(tup)

jnp_concat(arrays, axis = 0L)
```

## Arguments

- tup, arrays:

  A list of arrays, or an array whose leading axis is iterated over.

- dtype:

  Optional result dtype.

- axis:

  0-based axis; `NULL` flattens the inputs.

## Value

An array.
