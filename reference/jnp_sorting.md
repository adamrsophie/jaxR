# Sorting functions of jax.numpy

Ports of `jnp.sort`, `jnp.argsort`, `jnp.partition`, `jnp.argpartition`,
`jnp.sort_complex`, `jnp.lexsort` and `jnp.top_k`. Axes and returned
indices are 0-based.

## Usage

``` r
jnp_sort(
  a,
  axis = -1L,
  kind = NULL,
  order = NULL,
  stable = TRUE,
  descending = FALSE
)

jnp_argsort(
  a,
  axis = -1L,
  kind = NULL,
  order = NULL,
  stable = TRUE,
  descending = FALSE,
  dtype = NULL
)

jnp_partition(a, kth, axis = -1L)

jnp_argpartition(a, kth, axis = -1L)

jnp_sort_complex(a)

jnp_lexsort(keys, axis = -1L, batch_size = 16L)

jnp_top_k(a, k, axis = -1L, mode = "largest", sorted = TRUE)
```

## Arguments

- a:

  Array.

- axis:

  0-based axis (`NULL` flattens for `sort` and `argsort`).

- kind, order:

  Unsupported (must be `NULL`).

- stable:

  Use a stable sort.

- descending:

  Sort in descending order.

- dtype:

  Optional index dtype of `argsort`.

- kth:

  0-based index of the element in sorted position.

- keys:

  Array whose leading dimension indexes the keys, or a list of arrays.
  The last key is the primary one.

- batch_size:

  Number of keys sorted at once (all keys are always sorted in one pass
  here).

- k:

  Number of entries for `top_k`.

- mode:

  `"largest"` or `"smallest"`.

- sorted:

  Ignored (results are always sorted).

## Value

An array (a list of values and indices for `top_k`).
