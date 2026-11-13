# The jax.dtypes module

Ports of `jax.dtypes`: `canonicalize_dtype` (64-bit types map to 32-bit
ones unless x64 is enabled), `issubdtype` (also for the extended
categories `dtypes_prng_key` and `dtypes_extended`, which typed PRNG key
arrays belong to), `result_type`, `itemsize_bits`, `scalar_type_of` (the
R type of scalars of the dtype: `"logical"`, `"integer"`, `"double"` or
`"complex"`, standing for Python's bool, int, float and complex),
`finfo` and `iinfo`, the dtype names `float0` and `bfloat16`, and
`TypePromotionError` (a condition constructor; jaxR signals it with
class `jax_type_promotion_error`).

## Usage

``` r
dtypes_canonicalize_dtype(dtype, allow_extended_dtype = FALSE)

dtypes_issubdtype(arg1, arg2)

dtypes_result_type(..., return_weak_type_flag = FALSE)

dtypes_itemsize_bits(dtype)

dtypes_scalar_type_of(x)

dtypes_finfo(dtype = "float")

dtypes_iinfo(dtype = "int")

dtypes_float0

dtypes_bfloat16

dtypes_prng_key

dtypes_extended

dtypes_TypePromotionError(message)
```

## Arguments

- dtype:

  A dtype name or a value.

- allow_extended_dtype:

  Accept extended dtypes (key types).

- arg1, arg2:

  Dtypes, values or categories.

- ...:

  Values or dtypes.

- return_weak_type_flag:

  Also return whether the result is weakly typed.

- x:

  A value or dtype.

- message:

  Error message.

## Value

See the descriptions above.
