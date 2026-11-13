# Permute, transpose and rotate axes

Ports of `jnp.permute_dims`, `jnp.matrix_transpose`, `jnp.rot90` and
`jnp.rollaxis`. Axes are 0-based.

## Usage

``` r
jnp_permute_dims(a, axes)

jnp_matrix_transpose(x)

jnp_rot90(m, k = 1L, axes = c(0L, 1L))

jnp_rollaxis(a, axis, start = 0L)
```

## Arguments

- a, x, m:

  Arrays.

- axes:

  For `jnp_permute_dims`, a permutation of the axes; for `jnp_rot90`,
  the two axes of the rotation plane.

- k:

  Number of 90 degree rotations.

- axis:

  Axis to roll.

- start:

  Position before which the axis is placed.

## Value

An array.
