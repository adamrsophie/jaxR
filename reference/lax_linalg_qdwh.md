# QR-based dynamically weighted Halley iteration for the polar decomposition

Port of `jax.lax.linalg.qdwh`: computes the polar decomposition
`x = u h`.

## Usage

``` r
lax_linalg_qdwh(
  x,
  is_hermitian = FALSE,
  max_iterations = NULL,
  eps = NULL,
  dynamic_shape = NULL
)
```

## Arguments

- x:

  A full-rank matrix with shape `M x N`, `M >= N`. The matrix may be
  padded up to that size from a smaller true shape (`dynamic_shape`).

- is_hermitian:

  Unused; exists for compatibility with JAX.

- max_iterations:

  Maximum number of iterations (default 10).

- eps:

  The final result satisfies `|x_k - x_k-1| < |x_k| * (4 eps)^(1/3)`.

- dynamic_shape:

  Optional unpadded shape as `list(m, n)`.

## Value

`list(u, h, num_iters, is_converged)`.
