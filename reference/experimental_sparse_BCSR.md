# Batched compressed sparse row (BCSR) arrays

Port of `jax.experimental.sparse.BCSR`: `list(data, indices, indptr)`
plus a shape with exactly two sparse dimensions after the batch
dimensions. Class methods: `experimental_sparse_BCSR$fromdense()`,
`$from_bcoo()`. Properties and methods: `$data`, `$indices`, `$indptr`,
`$shape`, `$dtype`, `$nse`, `$n_batch`, `$n_sparse`, `$n_dense`,
`$todense()`, `$to_bcoo()`, `$sum_duplicates()`. `%*%` and `[` go
through
[`experimental_sparse_sparsify()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_sparsify.md).

## Usage

``` r
experimental_sparse_BCSR(
  args,
  shape,
  indices_sorted = FALSE,
  unique_indices = FALSE
)
```

## Arguments

- args:

  `list(data, indices, indptr)`.

- shape:

  Shape.

- indices_sorted, unique_indices:

  Flags.

## Value

A BCSR array (class `jax_bcsr`).
