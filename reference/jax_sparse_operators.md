# Operators for sparse arrays

Arithmetic, comparison and `%*%` operators for BCOO arrays (and `%*%`
for BCSR, COO, CSR and CSC), ports of the sparse array dunder methods of
JAX (which wrap the jnp functions with
[`experimental_sparse_sparsify()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_sparsify.md)).

## Usage

``` r
# S3 method for class 'jax_sparse'
Ops(e1, e2)

# S3 method for class 'jax_sparse'
matrixOps(x, y)

# S3 method for class 'jax_sparse'
chooseOpsMethod(x, y, mx, my, cl, reverse)

# S3 method for class 'jax_sparse'
Math(x, ...)

# S3 method for class 'jax_sparse'
x[...]
```

## Arguments

- e1, e2, x, y:

  Sparse or dense operands.

- mx, my, cl, reverse:

  See
  [`base::chooseOpsMethod()`](https://rdrr.io/r/base/chooseOpsMethod.html).

- ...:

  Indices for `[`.
