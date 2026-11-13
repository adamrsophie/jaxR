# jax.numpy elementwise functions

Ports of the `jax.numpy` ufuncs. Arguments broadcast with NumPy rules
and are promoted with JAX type promotion.

## Usage

``` r
jnp_add(x1, x2)

jnp_subtract(x1, x2)

jnp_multiply(x1, x2)

jnp_true_divide(x1, x2)

jnp_divide(x1, x2)

jnp_floor_divide(x1, x2)

jnp_remainder(x1, x2)

jnp_mod(x1, x2)

jnp_power(x1, x2)

jnp_negative(x)

jnp_positive(x)

jnp_abs(x)

jnp_absolute(x)

jnp_sign(x)

jnp_exp(x)

jnp_exp2(x)

jnp_log(x)

jnp_log2(x)

jnp_log10(x)

jnp_log1p(x)

jnp_expm1(x)

jnp_sqrt(x)

jnp_cbrt(x)

jnp_square(x)

jnp_reciprocal(x)

jnp_sin(x)

jnp_cos(x)

jnp_tan(x)

jnp_arcsin(x)

jnp_arccos(x)

jnp_arctan(x)

jnp_sinh(x)

jnp_cosh(x)

jnp_tanh(x)

jnp_arcsinh(x)

jnp_arccosh(x)

jnp_arctanh(x)

jnp_arctan2(x1, x2)

jnp_floor(x)

jnp_ceil(x)

jnp_trunc(x)

jnp_round(x, decimals = 0L, out = NULL)

jnp_rint(x)

jnp_maximum(x1, x2)

jnp_minimum(x1, x2)

jnp_equal(x1, x2)

jnp_not_equal(x1, x2)

jnp_less(x1, x2)

jnp_less_equal(x1, x2)

jnp_greater(x1, x2)

jnp_greater_equal(x1, x2)

jnp_logical_and(x1, x2)

jnp_logical_or(x1, x2)

jnp_logical_xor(x1, x2)

jnp_logical_not(x)

jnp_bitwise_and(x1, x2)

jnp_bitwise_or(x1, x2)

jnp_bitwise_xor(x1, x2)

jnp_bitwise_not(x)

jnp_invert(x)

jnp_isfinite(x)

jnp_isnan(x)

jnp_isinf(x)

jnp_real(x)

jnp_imag(x)

jnp_conjugate(x)

jnp_conj(x)

jnp_fabs(x)

jnp_bitwise_invert(x)

jnp_left_shift(x1, x2)

jnp_bitwise_left_shift(x1, x2)

jnp_right_shift(x1, x2)

jnp_bitwise_right_shift(x1, x2)

jnp_bitwise_count(x)

jnp_float_power(x1, x2)

jnp_pow(x1, x2)

jnp_nextafter(x1, x2)

jnp_spacing(x)

jnp_acos(x)

jnp_acosh(x)

jnp_asin(x)

jnp_asinh(x)

jnp_atan(x)

jnp_atanh(x)

jnp_atan2(x1, x2)

jnp_copysign(x1, x2)

jnp_signbit(x)

jnp_divmod(x1, x2)

jnp_fmod(x1, x2)

jnp_deg2rad(x)

jnp_rad2deg(x)

jnp_degrees(x)

jnp_radians(x)

jnp_modf(x, out = NULL)

jnp_isposinf(x, out = NULL)

jnp_isneginf(x, out = NULL)

jnp_heaviside(x1, x2)

jnp_logaddexp(x1, x2)

jnp_logaddexp2(x1, x2)

jnp_ldexp(x1, x2)

jnp_frexp(x)

jnp_hypot(x1, x2)

jnp_sinc(x)

jnp_ufuncs
```

## Arguments

- x, x1, x2:

  Arrays or R values.

- decimals:

  Number of decimals.

- out:

  Not supported (must be `NULL`).

## Value

An array.

## Details

`jnp_ufuncs` is a list of the
[`jnp_ufunc()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufunc.md)
objects of `jax.numpy` (`jnp_ufuncs$add` is `jnp.add`): `add`,
`bitwise_and`, `bitwise_or`, `bitwise_xor`, `logaddexp`, `logaddexp2`,
`logical_and`, `logical_or`, `logical_xor`, `maximum`, `minimum`,
`multiply`, `negative` and `subtract`. They have the methods `reduce`,
`accumulate`, `reduceat`, `outer` and `at`, for example
`jnp_ufuncs$add$reduce(x)`. `jnp_divmod()`, `jnp_frexp()` and
`jnp_modf()` return lists of two arrays.
