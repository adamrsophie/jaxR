# Sparse direct solver

Port of `jax.experimental.sparse.linalg.spsolve`: solves `A x = b` for a
square matrix `A` given by its CSR buffers. As upstream on CPU, the
solve runs on the host: a sparse LU factorization with the Matrix
package (a dense solve for complex data or without Matrix), so `tol` and
`reorder` (GPU solver options) are only validated. It works under
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) and is
differentiable with respect to `data` and `b`; batching with
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) is not
supported.

## Usage

``` r
experimental_sparse_linalg_spsolve(
  data,
  indices,
  indptr,
  b,
  tol = 1e-06,
  reorder = 1L
)
```

## Arguments

- data:

  The stored values of the CSR matrix.

- indices:

  The 0-based column indices.

- indptr:

  The row pointer array, of length `n + 1`.

- b:

  The right hand side, of length `n`.

- tol:

  Singularity tolerance of the GPU solver.

- reorder:

  Fill-in reducing reordering scheme of the GPU solver (0 to 3).

## Value

The solution, with the dtype and size of `b`.

## Examples

``` r
if (FALSE) { # \dontrun{
m <- experimental_sparse_CSR$fromdense(matrix(c(4, 1, 0, 1, 3, 0, 0, 0, 2), 3))
experimental_sparse_linalg_spsolve(m$data, m$indices, m$indptr, jnp_array(c(1, 2, 3)))
} # }
```
