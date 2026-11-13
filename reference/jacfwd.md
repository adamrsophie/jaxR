# Jacobian by forward mode

Port of `jax.jacfwd`. Each block has the dtype and weak type of the
output it differentiates.

## Usage

``` r
jacfwd(fun, argnums = 0L, has_aux = FALSE, holomorphic = FALSE)
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
