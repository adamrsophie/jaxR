# Tensor contractions

Ports of `jnp.tensordot`, `jnp.matvec`, `jnp.vecmat` and `jnp.vecdot`
(`vecmat` and `vecdot` conjugate their first argument). Axes are
0-based.

## Usage

``` r
jnp_tensordot(a, b, axes = 2L, precision = NULL, preferred_element_type = NULL)

jnp_matvec(x1, x2)

jnp_vecmat(x1, x2)

jnp_vecdot(x1, x2, axis = -1L, precision = NULL, preferred_element_type = NULL)
```

## Arguments

- a, b, x1, x2:

  Arrays.

- axes:

  For `jnp_tensordot`: a number of trailing axes of `a` to contract with
  leading axes of `b`, or a list of two integer vectors (axes of `a`,
  axes of `b`).

- precision:

  Optional precision.

- preferred_element_type:

  Optional accumulation and result dtype.

- axis:

  Axis of `x1` and `x2` to contract.

## Value

An array.
