# Data type objects

Port of `jnp.dtype` (NumPy's `np.dtype`): the dtype named by a string,
of a scalar type such as
[jnp_float32](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md),
or of an array. Unlike
[`dtype()`](https://adamrsophie.github.io/jaxR/reference/dtype.md),
64-bit names are not canonicalized when x64 is disabled, and `"float"`,
`"int"`, `"uint"` and `"complex"` are the 64-bit types, as in NumPy.

## Usage

``` r
jnp_dtype(obj, align = FALSE, copy = FALSE)
```

## Arguments

- obj:

  A dtype name, scalar type or array.

- align, copy:

  Ignored (NumPy arguments).

## Value

A dtype string.
