# Vector-Jacobian product

Port of `jax.vjp`.

## Usage

``` r
vjp(fun, ..., has_aux = FALSE, saveable_args = TRUE, in_nzs = NULL)
```

## Arguments

- fun:

  Function of pytrees of arrays.

- ...:

  Primal arguments.

- has_aux:

  Whether `fun` returns `list(output, aux)`.

- saveable_args:

  A tree of logicals, a prefix of the list of primal arguments like
  `in_nzs`, by default `TRUE`: which arguments may be kept for the
  backward pass. An argument array that the backward pass needs as it is
  (not a value computed from it) and that is marked `FALSE` is not kept:
  `vjp_fun$args_res` shows a `NotSaveable()` sentinel in its place, and
  the value must be put back (`vjp_fun$args_res[[i]] <- x`) before
  `vjp_fun` is called. Arguments the backward pass does not need show
  `NotNeeded()`. Only jaxR arrays given as arguments count; R numbers
  are converted to new arrays, like Python scalars in JAX.

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
cotangent of an integer output is ignored). `vjp_fun$out_nzs` is a tree
of logicals with the structure of `out` (plain lists in place of
registered containers) telling which outputs have a tangent that is not
known to be zero, and `vjp_fun$args_res` holds the saved arguments (see
`saveable_args`).
