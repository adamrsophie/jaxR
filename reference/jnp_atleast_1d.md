# Ensure a minimum number of dimensions

Ports of `jnp.atleast_1d`, `jnp.atleast_2d` and `jnp.atleast_3d`.
Leading unit dimensions are added (`atleast_3d` gives `(1, n, 1)` for
1-D and `(m, n, 1)` for 2-D input).

## Usage

``` r
jnp_atleast_1d(...)

jnp_atleast_2d(...)

jnp_atleast_3d(...)
```

## Arguments

- ...:

  Arrays.

## Value

An array for one input, otherwise a list of arrays.
