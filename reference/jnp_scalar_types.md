# Scalar types

Ports of the `jax.numpy` scalar types. Each is a constructor:
`jnp_float32(x)` is `jnp_asarray(x, dtype = "float32")` (64-bit types
give 32-bit arrays unless x64 is enabled). They also work as dtype
arguments of
[`jnp_dtype()`](https://adamrsophie.github.io/jaxR/reference/jnp_dtype.md),
[`jnp_issubdtype()`](https://adamrsophie.github.io/jaxR/reference/jnp_dtype_functions.md),
[`jnp_promote_types()`](https://adamrsophie.github.io/jaxR/reference/jnp_dtype_functions.md),
[`jnp_can_cast()`](https://adamrsophie.github.io/jaxR/reference/jnp_dtype_functions.md)
and the info functions. `jnp_int_`, `jnp_uint`, `jnp_float_` and
`jnp_complex_` are the 64-bit types; `jnp_single`, `jnp_double`,
`jnp_csingle` and `jnp_cdouble` are aliases.

## Usage

``` r
jnp_bool_(x)

jnp_bool(x)

jnp_int8(x)

jnp_int16(x)

jnp_int32(x)

jnp_int64(x)

jnp_uint8(x)

jnp_uint16(x)

jnp_uint32(x)

jnp_uint64(x)

jnp_bfloat16(x)

jnp_float16(x)

jnp_float32(x)

jnp_float64(x)

jnp_complex64(x)

jnp_complex128(x)

jnp_int1(x)

jnp_uint1(x)

jnp_int2(x)

jnp_uint2(x)

jnp_int4(x)

jnp_uint4(x)

jnp_float4_e2m1fn(x)

jnp_float6_e2m3fn(x)

jnp_float6_e3m2fn(x)

jnp_float8_e3m4(x)

jnp_float8_e4m3(x)

jnp_float8_e8m0fnu(x)

jnp_float8_e4m3b11fnuz(x)

jnp_float8_e4m3fn(x)

jnp_float8_e4m3fnuz(x)

jnp_float8_e5m2(x)

jnp_float8_e5m2fnuz(x)

jnp_single(x)

jnp_double(x)

jnp_csingle(x)

jnp_cdouble(x)

jnp_int_(x)

jnp_uint(x)

jnp_float_(x)

jnp_complex_(x)

jnp_generic

jnp_number

jnp_integer

jnp_signedinteger

jnp_unsignedinteger

jnp_inexact

jnp_floating

jnp_complexfloating

jnp_character

jnp_flexible
```

## Arguments

- x:

  Value(s) to convert.

## Value

An array of the type.

## Details

The abstract types (`jnp_generic`, `jnp_number`, `jnp_integer`,
`jnp_signedinteger`, `jnp_unsignedinteger`, `jnp_inexact`,
`jnp_floating`, `jnp_complexfloating`, `jnp_character`, `jnp_flexible`)
are category names for
[`jnp_issubdtype()`](https://adamrsophie.github.io/jaxR/reference/jnp_dtype_functions.md).
