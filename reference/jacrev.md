# Jacobian by reverse mode

Port of `jax.jacrev` (and its alias `jax.jacobian`). For pytree inputs
and outputs the result has the output structure with each leaf replaced
by the input structure. Each block has the dtype and weak type of the
input it differentiates.

## Usage

``` r
jacrev(
  fun,
  argnums = 0L,
  has_aux = FALSE,
  holomorphic = FALSE,
  allow_int = FALSE
)

jacobian(
  fun,
  argnums = 0L,
  has_aux = FALSE,
  holomorphic = FALSE,
  allow_int = FALSE,
  ...
)
```

## Arguments

- fun:

  Function.

- argnums:

  0-based argument position(s); negative values count from the end. A
  list of positions (the analog of a Python tuple) always gives a list
  of Jacobians.

- has_aux:

  Whether `fun` returns `list(output, aux)`.

- holomorphic:

  Whether `fun` is promised to be holomorphic.

- allow_int:

  Whether integer and boolean inputs are allowed; their Jacobian blocks
  are zero.

- ...:

  Passed to `symcompR::jacobian()` when `fun` is a symbolic expression.
  Not used otherwise.

## Value

A function computing the Jacobian (or `list(jacobian, aux)`).
