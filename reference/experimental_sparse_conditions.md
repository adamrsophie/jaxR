# Sparse efficiency conditions

Ports of the warning and error classes of `jax.experimental.sparse`.
They are the condition classes jaxR signals:
`experimental_sparse_SparseEfficiencyError` is the class of errors
raised by
[`experimental_sparse_bcoo_update_layout()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_update_layout.md)
and the warnings carry `experimental_sparse_SparseEfficiencyWarning`.
`CuSparseEfficiencyWarning` is never signaled (there is no cuSPARSE
path) but kept for compatibility.

## Usage

``` r
experimental_sparse_SparseEfficiencyError

experimental_sparse_SparseEfficiencyWarning

experimental_sparse_CuSparseEfficiencyWarning
```

## Format

Character scalars naming condition classes.
