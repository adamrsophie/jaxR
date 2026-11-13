# Dtype queries and promotion

Ports of `jnp.issubdtype`, `jnp.isdtype`, `jnp.promote_types`,
`jnp.result_type` and `jnp.can_cast`. Dtypes may be given as names,
scalar types such as
[jnp_float32](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
or arrays. `promote_types` uses JAX's promotion lattice and does not
canonicalize 64-bit types; `result_type` promotes values (R scalars are
weakly typed) and canonicalizes like JAX. `can_cast` follows NumPy's
casting rules.

## Usage

``` r
jnp_issubdtype(arg1, arg2)

jnp_isdtype(dtype, kind)

jnp_promote_types(a, b)

jnp_result_type(...)

jnp_can_cast(from_, to, casting = "safe")
```

## Arguments

- arg1:

  A dtype, or an abstract category such as
  [jnp_floating](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md).

- arg2:

  A dtype or an abstract category.

- dtype:

  A dtype.

- kind:

  A dtype or one of `"bool"`, `"signed integer"`, `"unsigned integer"`,
  `"integral"`, `"real floating"`, `"complex floating"`, `"numeric"`, or
  a vector (list) of these.

- a, b:

  Dtypes.

- ...:

  Arrays, R values, dtypes or `NULL` (the default float type).

- from\_, to:

  Dtypes (or arrays).

- casting:

  One of `"no"`, `"equiv"`, `"safe"`, `"same_kind"`, `"unsafe"`.

## Value

A logical scalar, or a dtype string.
