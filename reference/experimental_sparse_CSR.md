# Compressed sparse row and column matrices

Ports of `jax.experimental.sparse.CSR` and `CSC`:
`list(data, indices, indptr)` and a two dimensional shape. Class methods
`$fromdense()`; properties and methods `$data`, `$indices`, `$indptr`,
`$shape`, `$nse`, `$dtype`, `$todense()`, `$transpose()` (or `$T`, which
turns a CSR into a CSC and back) and `%*%` with dense operands.

## Usage

``` r
experimental_sparse_CSR(args, shape)

experimental_sparse_CSC(args, shape)
```

## Arguments

- args:

  `list(data, indices, indptr)`.

- shape:

  Shape.

## Value

A CSR (class `jax_csr`) or CSC (class `jax_csc`) matrix.
