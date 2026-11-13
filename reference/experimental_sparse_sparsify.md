# Sparsify a function

Port of `jax.experimental.sparse.sparsify`: the returned function
accepts BCOO and BCSR arrays wherever `f` accepts dense arrays, as long
as every primitive applied to a sparse value has a sparse rule
(elementwise functions that keep zeros, `+`, `-`, `*`, `/` by a dense
value, `%*%`, sums, reshapes, transposes, slicing, indexing, broadcast,
concatenation, `lax_while_loop`, `lax_cond` and `lax_scan` with sparse
carries, and
[`experimental_sparse_todense()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_todense.md)).

## Usage

``` r
experimental_sparse_sparsify(f, use_tracer = FALSE)
```

## Arguments

- f:

  A function.

- use_tracer:

  Accepted for compatibility; both modes use the jaxpr interpreter.

## Value

A function.

## Examples

``` r
if (FALSE) { # \dontrun{
m <- experimental_sparse_BCOO$fromdense(matrix(c(0, 1, 0, 2), 2))
experimental_sparse_sparsify(function(m, v) jnp_sin(m) %*% v)(m, c(1, 1))
} # }
```
