# Gradient of a scalar function

Port of `jax.grad`.

## Usage

``` r
grad(
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

  Function returning a scalar.

- argnums:

  0-based position (or positions) of the differentiated positional
  arguments; negative values count from the end. With a vector of
  several positions, or a list of positions (the analog of a Python
  tuple), the gradient is a list.

- has_aux:

  Whether `fun` returns `list(value, aux)`.

- holomorphic:

  Whether `fun` is promised to be holomorphic. Inputs and outputs must
  then be complex.

- allow_int:

  Whether integer and boolean inputs are allowed. Their gradient is a
  float0 array.

- ...:

  Passed to `symcompR::grad()` when `fun` is a symbolic expression (the
  symbolic gradient of symcompR, so both packages can be attached
  together). Not used otherwise.

## Value

A function computing the gradient (or `list(grad, aux)` with `has_aux`).
