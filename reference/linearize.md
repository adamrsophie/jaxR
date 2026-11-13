# Linearize a function

Port of `jax.linearize`.

## Usage

``` r
linearize(fun, ..., has_aux = FALSE, in_nzs = NULL)
```

## Arguments

- fun:

  Function of pytrees.

- ...:

  Primal arguments.

- has_aux:

  Whether `fun` returns `list(output, aux)`.

- in_nzs:

  Optional tree of logicals, a prefix of the list of primal arguments:
  which inputs may have a nonzero tangent (see
  [`vjp()`](https://adamrsophie.github.io/jaxR/reference/vjp.md)). The
  linearized function still takes a tangent for every input and ignores
  those marked `FALSE`.

## Value

`list(out, jvp_fun)` (plus `aux` when `has_aux`) where `jvp_fun(...)`
maps tangents (one per primal argument, with the shapes and dtypes of
the primals; a float0 array for an integer primal) to output tangents.
