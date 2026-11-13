# Batched coordinate (BCOO) sparse arrays

Port of `jax.experimental.sparse.BCOO`. Construct from
`list(data, indices)` and a shape, or with
`experimental_sparse_BCOO$fromdense(mat, nse, index_dtype, n_dense, n_batch)`.
A BCOO array is a pytree (leaves `data` and `indices`) with the
properties `$data`, `$indices`, `$shape`, `$dtype`, `$nse`, `$n_batch`,
`$n_sparse`, `$n_dense`, `$ndim`, `$size`, `$indices_sorted`,
`$unique_indices` and `$T`, and the methods `$todense()`,
`$transpose(axes)`, `$sum_duplicates(nse, remove_zeros)`,
`$sort_indices()`, `$update_layout(n_batch, n_dense, on_inefficient)`,
`$reshape(...)`, `$astype(dtype)`, `$sum(...)` and
`$block_until_ready()`. The arithmetic operators, `%*%`, comparisons and
`[` work through
[`experimental_sparse_sparsify()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_sparsify.md),
as the BCOO dunder methods do in JAX.

## Usage

``` r
experimental_sparse_BCOO(
  args,
  shape,
  indices_sorted = FALSE,
  unique_indices = FALSE
)
```

## Arguments

- args:

  `list(data, indices)`.

- shape:

  Shape of the array.

- indices_sorted, unique_indices:

  Flags describing the indices.

## Value

A BCOO array (class `jax_bcoo`).

## Examples

``` r
if (FALSE) { # \dontrun{
m <- experimental_sparse_BCOO(list(c(1, 2), matrix(c(0L, 1L, 1L, 0L), 2)), shape = c(2, 2))
m$todense()
m %*% c(1, 1)
} # }
```
