# Differentiation with sparse arguments

Ports of `jax.experimental.sparse.grad`, `value_and_grad`, `jacfwd`,
`jacrev` and `jacobian`: like
[`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md) and
friends, but sparse arguments are differentiated with respect to their
stored values, and their gradients are sparse arrays with the same
indices.

## Usage

``` r
experimental_sparse_grad(fun, argnums = 0L, has_aux = FALSE)

experimental_sparse_value_and_grad(fun, argnums = 0L, has_aux = FALSE)

experimental_sparse_jacfwd(fun, argnums = 0L, has_aux = FALSE)

experimental_sparse_jacrev(fun, argnums = 0L, has_aux = FALSE)

experimental_sparse_jacobian(fun, argnums = 0L, has_aux = FALSE)
```

## Arguments

- fun:

  Function to differentiate.

- argnums:

  0-based argument numbers.

- has_aux:

  Whether `fun` returns `list(value, aux)`.

## Value

A function.
