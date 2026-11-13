# Split an array into sub-arrays

Ports of `jnp.array_split` (sections may differ in size by one),
`jnp.hsplit` (axis 1, or 0 for 1-D input) and `jnp.dsplit` (axis 2). A
single number gives the number of sections; a vector of length other
than one, a list or a 1-D array gives the 0-based split points (out of
range points give empty sections).

## Usage

``` r
jnp_array_split(ary, indices_or_sections, axis = 0L)

jnp_hsplit(ary, indices_or_sections)

jnp_dsplit(ary, indices_or_sections)
```

## Arguments

- ary:

  Array.

- indices_or_sections:

  Number of sections or split points.

- axis:

  0-based axis.

## Value

A list of arrays.
