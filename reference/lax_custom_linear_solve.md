# Matrix free linear solve with implicit differentiation

Port of `jax.lax.custom_linear_solve`. Gradients are defined by implicit
differentiation at the solution instead of differentiating through
`solve`.

## Usage

``` r
lax_custom_linear_solve(
  matvec,
  b,
  solve,
  transpose_solve = NULL,
  symmetric = FALSE,
  has_aux = FALSE
)
```

## Arguments

- matvec:

  Linear function computing `A x`.

- b:

  Right hand side (pytree).

- solve:

  Function `(matvec, b)` returning `x` with `matvec(x) == b`.

- transpose_solve:

  Function `(vecmat, b)` solving the transposed system; needed for
  reverse mode unless `symmetric`.

- symmetric:

  Whether the linear map is symmetric.

- has_aux:

  Whether `solve` returns `list(x, aux)`.
