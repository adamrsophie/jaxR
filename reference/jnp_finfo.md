# Machine limits of numeric types

Ports of `jnp.finfo` (ml_dtypes.finfo) and `jnp.iinfo`
(ml_dtypes.iinfo). The dtype is not canonicalized, so
`jnp_finfo("float64")` describes float64 even when x64 is disabled. For
complex types `finfo` describes the real component. 64-bit integer
bounds are rounded to the nearest double.

## Usage

``` r
jnp_finfo(dtype)

jnp_iinfo(int_type)
```

## Arguments

- dtype:

  A float or complex dtype, scalar type or array.

- int_type:

  An integer dtype, scalar type or array.

## Value

A list of limits: for `finfo`, `dtype`, `bits`, `eps`, `epsneg`, `max`,
`min`, `tiny`, `smallest_normal`, `smallest_subnormal`, `resolution`,
`precision`, `machep`, `negep`, `minexp`, `maxexp`, `nmant`, `nexp`,
`iexp`; for `iinfo`, `dtype`, `kind`, `bits`, `min`, `max`.
