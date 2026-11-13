# Separate forward and backward functions of a VJP

Port of `jax.fwd_and_bwd`. `fwd(...)` returns
`list(primals_out, residuals)` (plus `aux` with `has_aux`);
`bwd(residuals, ct)` returns the cotangents of the arguments in
`argnums` (one value for a single argnum, a list otherwise). The
residuals are a pytree, so `bwd` can be jitted once and reused.

## Usage

``` r
fwd_and_bwd(fun, argnums = 0L, has_aux = FALSE, jitted = TRUE)
```

## Arguments

- fun:

  Function.

- argnums:

  0-based positions of the differentiated arguments; negative values
  count from the end.

- has_aux:

  Whether `fun` returns `list(output, aux)`.

- jitted:

  Return jitted functions.

## Value

`list(fwd, bwd)`.
