# Abstract sparse classes

`experimental_sparse_JAXSparse` is the port of the abstract base class
`jax.experimental.sparse.JAXSparse`: every jaxR sparse array inherits
from class `jax_sparse` (test with `inherits(x, "jax_sparse")`).
`experimental_sparse_SparseTracer` names the tracer class of
`sparsify(use_tracer = TRUE)` in JAX; jaxR's sparsify uses the jaxpr
interpreter in both modes, so neither object can be instantiated.

## Usage

``` r
experimental_sparse_JAXSparse(...)

experimental_sparse_SparseTracer(...)
```

## Format

Class objects.

## Arguments

- ...:

  Ignored: neither class can be instantiated.
