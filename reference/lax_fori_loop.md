# Loop from lower to upper

Port of `jax.lax.fori_loop`: `body_fun(i, val)` for `i` in
`lower..upper-1`. With concrete bounds the loop is a scan (reverse mode
differentiable); otherwise a while loop.

## Usage

``` r
lax_fori_loop(lower, upper, body_fun, init_val, unroll = NULL)
```

## Arguments

- lower, upper:

  Loop bounds (integers).

- body_fun:

  Function `(i, val)`.

- init_val:

  Initial value.

- unroll:

  Ignored.
