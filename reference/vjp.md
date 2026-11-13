# Vector-Jacobian product

Port of `jax.vjp`.

## Usage

``` r
vjp(fun, ..., has_aux = FALSE, in_nzs = NULL)
```

## Arguments

- fun:

  Function of pytrees of arrays.

- ...:

  Primal arguments.

- has_aux:

  Whether `fun` returns `list(output, aux)`.

- in_nzs:

  Optional tree of logicals, a prefix of the list of primal arguments
  (for example `list(TRUE, FALSE)`): which inputs may have a nonzero
  tangent. The tangent of an input marked `FALSE` is a symbolic zero
  during linearization, and its cotangent is zero. The default marks
  every input.

## Value

`list(out, vjp_fun)` (plus `aux` when `has_aux`), where `vjp_fun(ct)`
returns a list of cotangents, one per primal argument. `ct` must have
the structure of `out`, and each of its leaves the shape and dtype of
the matching output (as in JAX there is no broadcasting or casting; the
cotangent of an integer output is ignored).
