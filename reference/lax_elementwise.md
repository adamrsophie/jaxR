# lax elementwise operations

Ports of the elementwise functions of `jax.lax`. Arguments must have
equal dtypes (R scalars are weakly typed and adopt the dtype of the
other argument) and shapes that are equal, rank 0, or differ only in
size one dimensions.

## Usage

``` r
lax_neg(x)

lax_sign(x)

lax_floor(x)

lax_ceil(x)

lax_round(x, rounding_method = 0L)

lax_is_finite(x)

lax_exp(x, accuracy = NULL)

lax_exp2(x, accuracy = NULL)

lax_log(x, accuracy = NULL)

lax_log1p(x, accuracy = NULL)

lax_expm1(x, accuracy = NULL)

lax_tanh(x, accuracy = NULL)

lax_logistic(x, accuracy = NULL)

lax_sin(x, accuracy = NULL)

lax_cos(x, accuracy = NULL)

lax_tan(x, accuracy = NULL)

lax_asin(x)

lax_acos(x)

lax_atan(x)

lax_sinh(x)

lax_cosh(x)

lax_asinh(x)

lax_acosh(x)

lax_atanh(x)

lax_sqrt(x, accuracy = NULL)

lax_rsqrt(x, accuracy = NULL)

lax_cbrt(x, accuracy = NULL)

lax_abs(x)

lax_lgamma(x)

lax_digamma(x)

lax_erf(x)

lax_erfc(x, accuracy = NULL)

lax_erf_inv(x)

lax_bitwise_not(x)

lax_population_count(x)

lax_clz(x)

lax_square(x)

lax_real(x)

lax_imag(x)

lax_conj(x)

lax_integer_pow(x, y)

lax_reduce_precision(x, exponent_bits, mantissa_bits)

lax_add(x, y)

lax_sub(x, y)

lax_mul(x, y)

lax_div(x, y)

lax_rem(x, y)

lax_pow(x, y)

lax_max(x, y)

lax_min(x, y)

lax_atan2(x, y)

lax_nextafter(x, y)

lax_bitwise_and(x, y)

lax_bitwise_or(x, y)

lax_bitwise_xor(x, y)

lax_shift_left(x, y)

lax_shift_right_arithmetic(x, y)

lax_shift_right_logical(x, y)

lax_complex(x, y)

lax_eq(x, y)

lax_ne(x, y)

lax_lt(x, y)

lax_le(x, y)

lax_gt(x, y)

lax_ge(x, y)
```

## Arguments

- x, y:

  Arrays.

- rounding_method:

  0 (away from zero) or 1 (to nearest even).

- accuracy:

  Optional accuracy request of the functions that take one in JAX: a
  [lax_AccuracyMode](https://adamrsophie.github.io/jaxR/reference/lax_enums.md)
  value, `"highest"` or `"default"`, or a
  [`lax_Tolerance()`](https://adamrsophie.github.io/jaxR/reference/lax_Tolerance.md).
  It is a hint to the backend; on the CPU backend only `lax_erfc()` with
  the highest accuracy changes the computation.

- exponent_bits, mantissa_bits:

  Target precision.

## Value

An array.
