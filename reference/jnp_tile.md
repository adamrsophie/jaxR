# Resize, tile, repeat, roll and split arrays

Ports of `jnp.resize`, `jnp.tile`, `jnp.repeat`, `jnp.roll`,
`jnp.split`, `jnp.vsplit`, `jnp.unstack` and `jnp.vstack`. Axes are
0-based.

## Usage

``` r
jnp_resize(a, new_shape)

jnp_tile(A, reps)

jnp_repeat(a, repeats, axis = NULL, total_repeat_length = NULL)

jnp_roll(a, shift, axis = NULL)

jnp_split(ary, indices_or_sections, axis = 0L)

jnp_vsplit(ary, indices_or_sections)

jnp_unstack(x, axis = 0L)

jnp_vstack(tup, dtype = NULL)
```

## Arguments

- a, A, ary, x:

  Arrays.

- new_shape:

  Integer vector.

- reps:

  Integer repetitions per axis.

- repeats:

  Integer repetitions (scalar or one per element along `axis`).

- axis:

  Axis (0-based); `NULL` flattens first where JAX does.

- total_repeat_length:

  Static output length along `axis`, needed when `repeats` is traced.

- shift:

  Integer shift(s); a traced value selects the dynamic implementation.

- indices_or_sections:

  A number of equal sections, or the split indices. Pass a list (or an
  array) to split at a single index, since a single R number means a
  number of sections.

- tup:

  List of arrays (or an array, split along its first axis).

- dtype:

  Optional result dtype.

## Value

An array (`jnp_split`, `jnp_vsplit` and `jnp_unstack` return lists).
