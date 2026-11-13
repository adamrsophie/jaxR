# Grids and concatenation helpers

Ports of `jnp.mgrid`, `jnp.ogrid`, `jnp.r_` and `jnp.c_`. NumPy's slice
syntax is written with
[`sl()`](https://adamrsophie.github.io/jaxR/reference/sl.md):
`jnp.mgrid[0:3, 0:2]` is `jnp_mgrid(sl(0, 3), sl(0, 2))` and a complex
step gives a number of points, as in `jnp_mgrid(sl(0, 1, 5i))`. A single
slice gives a one dimensional result; pass a list of slices for the
tuple form. `jnp_r_` and `jnp_c_` concatenate arrays, scalars and
slices; an optional leading string is a directive (`"r"`, `"c"` or
`"axis,ndmin,trans1d"`).

## Usage

``` r
jnp_mgrid(...)

jnp_ogrid(...)

jnp_r_(...)

jnp_c_(...)
```

## Arguments

- ...:

  Slices (or, for `jnp_r_` and `jnp_c_`, items and a leading directive).

## Value

An array (`jnp_ogrid` with several slices returns a list).
