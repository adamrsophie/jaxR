# Shape helpers

Shape helpers

## Usage

``` r
jnp_shape(a)

jnp_ndim(a)

jnp_size(a, axis = NULL)

jnp_reshape(a, shape, order = "C")

jnp_ravel(a, order = "C")

jnp_transpose(a, axes = NULL)

jnp_moveaxis(a, source, destination)

jnp_swapaxes(a, axis1, axis2)

jnp_expand_dims(a, axis)

jnp_squeeze(a, axis = NULL)

jnp_broadcast_to(a, shape)

jnp_broadcast_arrays(...)

jnp_concatenate(arrays, axis = 0L, dtype = NULL)

jnp_stack(arrays, axis = 0L, out = NULL, dtype = NULL)

jnp_astype(a, dtype)
```

## Arguments

- a:

  Array.

- axis:

  0-based axis (or axes).

- shape:

  New shape (one entry may be -1).

- order:

  Only `"C"` is supported.

- axes:

  0-based permutation.

- source, destination:

  0-based axes.

- axis1, axis2:

  Axes to swap.

- ...:

  Arrays.

- arrays:

  List of arrays.

- dtype:

  Dtype.

- out:

  Unsupported; must be `NULL`.
