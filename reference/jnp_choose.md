# Construct an array from an index array and choices

Port of `jnp.choose`: `out[i] = choices[a[i]][i]` after broadcasting.

## Usage

``` r
jnp_choose(a, choices, out = NULL, mode = "raise")
```

## Arguments

- a:

  Integer array of 0-based choice indices.

- choices:

  A list of arrays, or an array whose leading axis indexes the choices.

- out:

  Not supported (must be `NULL`).

- mode:

  `"raise"` (needs concrete `a`), `"wrap"` or `"clip"`.

## Value

An array.
