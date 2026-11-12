# jax.numpy ufuncs that R/numpy_core.R does not define and the jnp.ufunc class (ports
# of jax/_src/numpy/ufuncs.py, jax/_src/numpy/ufunc_api.py and the reductions of
# jax/_src/numpy/reductions.py that back the ufunc reduce methods).
#
# Internal helpers are prefixed `ufn_`. Local ports that should be switched to shared
# versions once they exist: `ufn_polynomial` (lax.polynomial), `ufn_lax_hypot`
# (lax.hypot), `ufn_vectorize` (jnp.vectorize) and `ufn_reduction` (reductions._reduction,
# needed because jnp_sum and friends do not yet support `initial` and `where`).

# ---- helpers ------------------------------------------------------------------------------

ufn_dtype <- function(x) aval_of(x)$dtype

ufn_shape_str <- function(s) paste0("(", paste(s, collapse = ", "), if (length(s) == 1L) ",", ")")

# numpy.finfo fields used below (minexp, maxexp, smallest_subnormal).
ufn_finfo <- function(dt) {
  fi <- finfo(dt)
  emax <- 2^(fi$nexp - 1) - 1
  fi$maxexp <- emax + 1
  fi$minexp <- 1 - emax
  fi$smallest_subnormal <- 2^(1 - emax - fi$nmant)
  fi
}

# Scalar constant of the dtype of x (lax._const): weak only when x is a weak scalar.
ufn_c <- function(x, v) {
  a <- aval_of(x)
  lax_full_like(x, v, dtype = if (a$weak && !length(a$shape)) NULL else a$dtype, shape = integer(0))
}

# Strong scalar constant of dtype dt (a NumPy scalar or ufuncs._constant_like).
ufn_scalar <- function(v, dt) lax_full(integer(0), v, dt)

# Unsigned integer constants for the bit manipulation in ldexp and frexp. Values above
# 2^24 are built on device with shifts so that 64-bit masks never pass through a double.
ufn_uconst <- function(v, ut) ufn_scalar(v, ut)
ufn_ushift1 <- function(k, ut) lax_shift_left(ufn_uconst(1, ut), ufn_uconst(k, ut))

# ---- unary and binary elementwise functions (ufuncs.py) -------------------------------------

#' @rdname jnp_ufuncs
#' @export
jnp_fabs <- function(x) {
  # Port of ufuncs.py:fabs.
  x <- as_jax(x)
  if (is_complex_dtype(ufn_dtype(x)))
    jax_error("ufunc 'fabs' does not support complex dtypes", class = "jax_type_error")
  lax_abs(promote_args_inexact(x)[[1L]])
}

#' @rdname jnp_ufuncs
#' @export
jnp_bitwise_invert <- function(x) lax_bitwise_not(promote_args(x)[[1L]])

#' @rdname jnp_ufuncs
#' @export
jnp_left_shift <- function(x1, x2) {
  a <- promote_args_numeric(x1, x2)
  lax_shift_left(a[[1L]], a[[2L]])
}

#' @rdname jnp_ufuncs
#' @export
jnp_bitwise_left_shift <- function(x1, x2) jnp_left_shift(x1, x2)

#' @rdname jnp_ufuncs
#' @export
jnp_right_shift <- function(x1, x2) {
  # Port of ufuncs.py:right_shift: logical shift for unsigned, arithmetic otherwise.
  a <- promote_args_numeric(x1, x2)
  f <- if (is_unsigned_dtype(ufn_dtype(a[[1L]]))) lax_shift_right_logical else lax_shift_right_arithmetic
  f(a[[1L]], a[[2L]])
}

#' @rdname jnp_ufuncs
#' @export
jnp_bitwise_right_shift <- function(x1, x2) jnp_right_shift(x1, x2)

#' @rdname jnp_ufuncs
#' @export
jnp_bitwise_count <- function(x) {
  # Port of ufuncs.py:bitwise_count.
  x <- promote_args_numeric(x)[[1L]]
  lax_convert_element_type(lax_population_count(jnp_abs(x)), "uint8")
}

#' @rdname jnp_ufuncs
#' @export
jnp_float_power <- function(x1, x2) {
  a <- promote_args_inexact(x1, x2)
  lax_pow(a[[1L]], a[[2L]])
}

#' @rdname jnp_ufuncs
#' @export
jnp_pow <- function(x1, x2) jnp_power(x1, x2)

#' @rdname jnp_ufuncs
#' @export
jnp_nextafter <- function(x1, x2) {
  a <- promote_args_inexact(x1, x2)
  lax_nextafter(a[[1L]], a[[2L]])
}

#' @rdname jnp_ufuncs
#' @export
jnp_spacing <- function(x) {
  # Port of ufuncs.py:spacing.
  arr <- promote_args_inexact(x)[[1L]]
  dt <- ufn_dtype(arr)
  if (is_complex_dtype(dt))
    jax_error("jnp.spacing is not defined for complex inputs.", class = "jax_value_error")
  if (dt == "float16") return(lax_sub(lax_nextafter(arr, ufn_c(arr, Inf)), arr))
  result <- lax_sub(lax_nextafter(arr, jnp_copysign(Inf, arr)), arr)
  jnp_where(jnp_equal(result, 0),
            jnp_copysign(ufn_scalar(ufn_finfo(dt)$smallest_subnormal, dt), arr), result)
}

#' @rdname jnp_ufuncs
#' @export
jnp_acos <- function(x) jnp_arccos(promote_args(x)[[1L]])
#' @rdname jnp_ufuncs
#' @export
jnp_acosh <- function(x) jnp_arccosh(promote_args(x)[[1L]])
#' @rdname jnp_ufuncs
#' @export
jnp_asin <- function(x) jnp_arcsin(promote_args(x)[[1L]])
#' @rdname jnp_ufuncs
#' @export
jnp_asinh <- function(x) jnp_arcsinh(promote_args(x)[[1L]])
#' @rdname jnp_ufuncs
#' @export
jnp_atan <- function(x) jnp_arctan(promote_args(x)[[1L]])
#' @rdname jnp_ufuncs
#' @export
jnp_atanh <- function(x) jnp_arctanh(promote_args(x)[[1L]])
#' @rdname jnp_ufuncs
#' @export
jnp_atan2 <- function(x1, x2) {
  a <- promote_args(x1, x2)
  jnp_arctan2(a[[1L]], a[[2L]])
}

#' @rdname jnp_ufuncs
#' @export
jnp_copysign <- function(x1, x2) {
  # Port of ufuncs.py:copysign.
  a <- promote_args_inexact(x1, x2)
  if (is_complex_dtype(ufn_dtype(a[[1L]])))
    jax_error("copysign does not support complex-valued inputs", class = "jax_type_error")
  ax <- lax_abs(a[[1L]])
  jnp_where(jnp_signbit(a[[2L]]), lax_neg(ax), ax)
}

#' @rdname jnp_ufuncs
#' @export
jnp_signbit <- function(x) {
  # Port of ufuncs.py:signbit: the sign bit of the bit pattern.
  x <- promote_args(x)[[1L]]
  dt <- ufn_dtype(x)
  if (is_int_dtype(dt)) return(lax_lt(x, ufn_scalar(0, dt)))
  if (dt == "bool") return(lax_full_like(x, FALSE, dtype = "bool"))
  if (!is_float_dtype(dt))
    jax_error("jax.numpy.signbit is not well defined for ", dt, class = "jax_value_error")
  info <- finfo(dt)
  xi <- lax_bitcast_convert_type(x, paste0("int", info$bits))
  lax_convert_element_type(lax_shift_right_arithmetic(xi, info$nexp + info$nmant), "bool")
}

# Port of ufuncs.py:_float_divmod.
ufn_float_divmod <- function(x1, x2) {
  mod <- lax_rem(x1, x2)
  x1_corrected <- jnp_where(jnp_equal(x2, 0), x1, lax_sub(x1, mod))
  div <- lax_div(x1_corrected, x2)
  ind <- lax_bitwise_and(jnp_not_equal(mod, 0), jnp_not_equal(lax_sign(x2), lax_sign(mod)))
  mod <- lax_select(ind, lax_add(mod, x2), mod)
  div <- lax_select(ind, jnp_subtract(div, 1), div)
  list(lax_round(div), mod)
}

#' @rdname jnp_ufuncs
#' @export
jnp_divmod <- function(x1, x2) {
  # Port of ufuncs.py:divmod. Returns list(floor_divide(x1, x2), remainder(x1, x2)).
  a <- promote_args_numeric(x1, x2)
  if (is_int_dtype(ufn_dtype(a[[1L]])))
    return(list(jnp_floor_divide(a[[1L]], a[[2L]]), jnp_remainder(a[[1L]], a[[2L]])))
  ufn_float_divmod(a[[1L]], a[[2L]])
}

#' @rdname jnp_ufuncs
#' @export
jnp_fmod <- function(x1, x2) {
  # Port of ufuncs.py:fmod (C style remainder with the sign of x1).
  x1 <- as_jax(x1)
  x2 <- as_jax(x2)
  if (is_int_dtype(result_type(x1, x2)))
    x2 <- jnp_where(jnp_equal(x2, 0), lax_full_like(x2, 1), x2)
  a <- promote_args_numeric(x1, x2)
  lax_rem(a[[1L]], a[[2L]])
}

#' @rdname jnp_ufuncs
#' @export
jnp_deg2rad <- function(x) {
  x <- promote_args_inexact(x)[[1L]]
  lax_mul(x, ufn_c(x, pi / 180))
}
#' @rdname jnp_ufuncs
#' @export
jnp_rad2deg <- function(x) {
  x <- promote_args_inexact(x)[[1L]]
  lax_mul(x, ufn_c(x, 180 / pi))
}
#' @rdname jnp_ufuncs
#' @export
jnp_degrees <- function(x) jnp_rad2deg(x)
#' @rdname jnp_ufuncs
#' @export
jnp_radians <- function(x) jnp_deg2rad(x)

#' @rdname jnp_ufuncs
#' @param out Not supported (must be `NULL`).
#' @export
jnp_modf <- function(x, out = NULL) {
  # Port of ufuncs.py:modf. Returns list(fractional part, integral part).
  x <- promote_dtypes_inexact(list(x))[[1L]]
  if (!is.null(out))
    jax_error("The 'out' argument to jnp.modf is not supported.", class = "jax_not_implemented_error")
  whole <- jnp_where(lax_ge(x, ufn_c(x, 0)), jnp_floor(x), jnp_ceil(x))
  list(lax_sub(x, whole), whole)
}

# Port of ufuncs.py:_isposneginf (every float dtype of jaxR supports infinities).
ufn_isposneginf <- function(infinity, x, out) {
  if (!is.null(out))
    jax_error("The 'out' argument to isneginf/isposinf is not supported.",
              class = "jax_not_implemented_error")
  dt <- ufn_dtype(x)
  if (is_float_dtype(dt)) return(lax_eq(x, ufn_scalar(infinity, dt)))
  if (is_complex_dtype(dt))
    jax_error("isposinf/isneginf are not well defined for complex types", class = "jax_value_error")
  lax_full_like(x, FALSE, dtype = "bool")
}

#' @rdname jnp_ufuncs
#' @export
jnp_isposinf <- function(x, out = NULL) ufn_isposneginf(Inf, as_jax(x), out)
#' @rdname jnp_ufuncs
#' @export
jnp_isneginf <- function(x, out = NULL) ufn_isposneginf(-Inf, as_jax(x), out)

#' @rdname jnp_ufuncs
#' @export
jnp_heaviside <- function(x1, x2) {
  # Port of ufuncs.py:heaviside.
  a <- promote_dtypes_inexact(list(x1, x2))
  x1 <- a[[1L]]
  zero <- ufn_c(x1, 0)
  jnp_where(lax_lt(x1, zero), zero,
            jnp_where(lax_gt(x1, zero), ufn_c(x1, 1), jnp_where(lax_ne(x1, x1), x1, a[[2L]])))
}

# ---- logaddexp, logaddexp2 ----------------------------------------------------------------

#' @rdname jnp_ufuncs
#' @export
jnp_logaddexp <- function(x1, x2) {
  # Port of ufuncs.py:logaddexp (custom_jvp lax_other_logaddexp of R/lax_cumulative.R).
  a <- promote_args_inexact(x1, x2)
  lax_other_logaddexp(a[[1L]], a[[2L]])
}

# Port of lax/other.py:logaddexp2 with its custom JVP _logaddexp2_jvp.
ufn_lax_other_logaddexp2 <- custom_jvp(function(x1, x2) {
  x1 <- as_jax(x1)
  x2 <- as_jax(x2)
  dt <- ufn_dtype(x1)
  amax <- lax_max(x1, x2)
  invln2 <- ufn_c(amax, 1 / log(2))
  if (is_float_dtype(dt)) {
    delta <- lax_sub(x1, x2)
    return(lax_select(lax_ne(delta, delta), lax_add(x1, x2),
                      lax_add(amax, lax_mul(invln2, lax_log1p(lax_exp2(lax_neg(lax_abs(delta))))))))
  }
  if (is_complex_dtype(dt)) {
    delta <- lax_sub(lax_add(x1, x2), lax_mul(amax, ufn_c(amax, 2)))
    out <- lax_add(amax, lax_mul(invln2, lax_log1p(lax_exp2(delta))))
    return(lax_complex(lax_real(out), wrap_between(lax_imag(out), pi / log(2))))
  }
  jax_error("logaddexp2 requires floating-point or complex inputs; got ", dt,
            class = "jax_value_error")
}, jvp = function(primals, tangents) {
  x1 <- primals[[1L]]
  x2 <- primals[[2L]]
  primal_out <- ufn_lax_other_logaddexp2(x1, x2)
  ri_out <- replace_inf(primal_out)
  tangent_out <- lax_add(lax_mul(tangents[[1L]], lax_exp2(lax_sub(replace_inf(x1), ri_out))),
                         lax_mul(tangents[[2L]], lax_exp2(lax_sub(replace_inf(x2), ri_out))))
  list(primal_out, tangent_out)
})

#' @rdname jnp_ufuncs
#' @export
jnp_logaddexp2 <- function(x1, x2) {
  a <- promote_args_inexact(x1, x2)
  ufn_lax_other_logaddexp2(a[[1L]], a[[2L]])
}

# ---- ldexp and frexp (HiPrims Ldexp and Frexp of ufuncs.py) ---------------------------------

# Port of ufuncs.py:_flush_subnormals.
ufn_flush_subnormals <- function(x) {
  dt <- ufn_dtype(x)
  info <- finfo(dt)
  ut <- paste0("uint", info$bits)
  u <- lax_bitcast_convert_type(x, ut)
  exp_mask <- lax_shift_left(ufn_uconst(2^info$nexp - 1, ut), ufn_uconst(info$nmant, ut))
  sign_mask <- ufn_ushift1(info$bits - 1, ut)
  is_zero_or_sub <- lax_eq(lax_bitwise_and(u, exp_mask), ufn_uconst(0, ut))
  lax_bitcast_convert_type(lax_select(is_zero_or_sub, lax_bitwise_and(u, sign_mask), u), dt)
}

# Port of ufuncs.py:_ldexp_impl (exact bit manipulation of the exponent field).
ufn_ldexp_impl <- function(x, n) {
  x <- ufn_flush_subnormals(x)
  dt <- ufn_dtype(x)
  info <- ufn_finfo(dt)
  ut <- paste0("uint", info$bits)
  u <- lax_bitcast_convert_type(x, ut)
  exp_mask <- ufn_uconst(2^info$nexp - 1, ut)
  nmant <- ufn_uconst(info$nmant, ut)
  mant_mask <- lax_sub(ufn_ushift1(info$nmant, ut), ufn_uconst(1, ut))
  sign <- lax_bitwise_and(u, ufn_ushift1(info$bits - 1, ut))
  ex <- lax_bitwise_and(lax_shift_right_logical(u, nmant), exp_mask)
  mant <- lax_bitwise_and(u, mant_mask)
  is_zero_or_sub <- lax_eq(ex, ufn_uconst(0, ut))
  is_inf_or_nan <- lax_eq(ex, exp_mask)
  max_shift <- 2^(info$nexp + 1)
  ndt <- ufn_dtype(n)
  is_signed <- is_signed_dtype(ndt)
  if (dtype_bits(ndt) < 32L) n <- lax_convert_element_type(n, if (is_signed) "int32" else "uint32")
  low <- if (is_signed) -max_shift else 0
  n_i32 <- lax_convert_element_type(lax_clamp(low, n, max_shift), "int32")
  new_exp <- lax_add(lax_convert_element_type(ex, "int32"), n_i32)
  inf_bits <- lax_bitwise_or(sign, lax_shift_left(exp_mask, nmant))
  norm_exp_bits <- lax_shift_left(lax_convert_element_type(new_exp, ut), nmant)
  norm_bits <- lax_bitwise_or(lax_bitwise_or(sign, norm_exp_bits), mant)
  res_bits <- lax_select(lax_le(new_exp, 0L), sign,
                         lax_select(lax_ge(new_exp, as.integer(2^info$nexp - 1)), inf_bits, norm_bits))
  out_bits <- lax_select(is_inf_or_nan, u, lax_select(is_zero_or_sub, sign, res_bits))
  lax_bitcast_convert_type(out_bits, dt)
}

# Port of ufuncs.py:_frexp_impl.
ufn_frexp_impl <- function(x) {
  x <- ufn_flush_subnormals(x)
  dt <- ufn_dtype(x)
  info <- ufn_finfo(dt)
  ut <- paste0("uint", info$bits)
  u <- lax_bitcast_convert_type(x, ut)
  exp_mask <- ufn_uconst(2^info$nexp - 1, ut)
  nmant <- ufn_uconst(info$nmant, ut)
  mant_mask <- lax_sub(ufn_ushift1(info$nmant, ut), ufn_uconst(1, ut))
  bias <- info$maxexp - 1
  sign <- lax_bitwise_and(u, ufn_ushift1(info$bits - 1, ut))
  ex <- lax_bitwise_and(lax_shift_right_logical(u, nmant), exp_mask)
  mant <- lax_bitwise_and(u, mant_mask)
  is_zero_or_sub <- lax_eq(ex, ufn_uconst(0, ut))
  is_inf_or_nan <- lax_eq(ex, exp_mask)
  m_norm_bits <- lax_bitwise_or(lax_bitwise_or(sign, lax_shift_left(ufn_uconst(bias - 1, ut), nmant)),
                                mant)
  m_bits <- lax_select(is_inf_or_nan, u, lax_select(is_zero_or_sub, sign, m_norm_bits))
  e <- lax_select(lax_bitwise_or(is_inf_or_nan, is_zero_or_sub), lax_full_like(u, 0, dtype = "int32"),
                  lax_sub(lax_convert_element_type(ex, "int32"), as.integer(bias - 1)))
  list(lax_bitcast_convert_type(m_bits, dt), e)
}

# Port of the Ldexp HiPrim (abstract eval, expand as lowering, jvp, transpose, batch).
ufn_ldexp_p <- new_primitive("ldexp")
ufn_ldexp_p$abstract_eval <- function(avals, params) {
  new_aval(avals[[1L]]$shape, avals[[1L]]$dtype, FALSE)
}
ufn_ldexp_p$lowering <- lower_fun(function(args, params) ufn_ldexp_impl(args[[1L]], args[[2L]]))
ufn_ldexp_p$jvp <- function(primals, tangents, params) {
  out <- bind(ufn_ldexp_p, primals)
  t <- tangents[[1L]]
  list(out, if (is_zero(t)) zero_of(tangent_aval(aval_of(out))) else jnp_ldexp(t, primals[[2L]]))
}
ufn_ldexp_p$transpose <- function(ct, args, params) {
  list(if (is_zero(ct)) NULL else jnp_ldexp(ct, args[[2L]]), NULL)
}
ufn_ldexp_p$batch <- function(vals, dims, params, axis) {
  x1 <- vals[[1L]]
  x2 <- vals[[2L]]
  d1 <- dims[1L]
  if (!identical(d1, dims[2L])) {
    x1 <- bdim_at_front(x1, d1, axis$size)
    x2 <- bdim_at_front(x2, dims[2L], axis$size)
    d1 <- 0L
  }
  list(bind(ufn_ldexp_p, list(x1, x2)), d1)
}

# Port of the Frexp HiPrim.
ufn_frexp_p <- new_primitive("frexp", multiple_results = TRUE)
ufn_frexp_p$abstract_eval <- function(avals, params) {
  list(new_aval(avals[[1L]]$shape, avals[[1L]]$dtype, FALSE), new_aval(avals[[1L]]$shape, "int32", FALSE))
}
ufn_frexp_p$lowering <- lower_fun(function(args, params) ufn_frexp_impl(args[[1L]]),
                                  multiple_results = TRUE)
ufn_frexp_p$jvp <- function(primals, tangents, params) {
  outs <- bind(ufn_frexp_p, primals)
  t <- tangents[[1L]]
  m_dot <- if (is_zero(t)) zero_of(tangent_aval(aval_of(outs[[1L]])))
           else jnp_ldexp(t, lax_neg(outs[[2L]]))
  list(outs, list(m_dot, zero_of(tangent_aval(aval_of(outs[[2L]])))))
}
ufn_frexp_p$batch <- function(vals, dims, params, axis) {
  list(bind(ufn_frexp_p, vals), c(dims[1L], dims[1L]))
}

#' @rdname jnp_ufuncs
#' @export
jnp_ldexp <- function(x1, x2) {
  # Port of ufuncs.py:ldexp: x1 * 2^x2 for integer x2.
  x1 <- as_jax(x1)
  x2 <- as_jax(x2)
  d1 <- ufn_dtype(x1)
  d2 <- ufn_dtype(x2)
  if (is_complex_dtype(d1) || is_inexact_dtype(d2))
    jax_error("ldexp not supported for input types (", d1, ", ", d2, ")", class = "jax_value_error")
  x1 <- promote_args_inexact(x1)[[1L]]
  if (d2 == "bool") x2 <- lax_convert_element_type(x2, "int32")
  a <- do.call(jnp_broadcast_arrays, promote_shapes(list(x1, x2)))
  x1 <- lax_convert_element_type(a[[1L]], ufn_dtype(a[[1L]]), weak_type = FALSE)
  x2 <- lax_convert_element_type(a[[2L]], ufn_dtype(a[[2L]]), weak_type = FALSE)
  bind(ufn_ldexp_p, list(x1, x2))
}

#' @rdname jnp_ufuncs
#' @export
jnp_frexp <- function(x) {
  # Port of ufuncs.py:frexp. Returns list(mantissa, int32 exponent).
  x <- promote_dtypes_inexact(list(x))[[1L]]
  if (is_complex_dtype(ufn_dtype(x)))
    jax_error("frexp does not support complex-valued inputs", class = "jax_type_error")
  x <- lax_convert_element_type(x, ufn_dtype(x), weak_type = FALSE)
  bind(ufn_frexp_p, list(x))
}

# ---- hypot ----------------------------------------------------------------------------------

# Port of lax.py:hypot: returns list(r, x1 / r, x2 / r) with power of two rescaling.
ufn_lax_hypot <- function(x1, x2) {
  a1 <- lax_abs(x1)
  a2 <- lax_abs(x2)
  m <- lax_max(a1, a2)
  fi <- ufn_finfo(ufn_dtype(x1))
  k <- (fi$nmant - fi$minexp + 1) %/% 2
  e_hi <- (fi$maxexp - 1) %/% 2
  hi <- lax_ge(m, ufn_c(x1, 2^e_hi))
  lo <- lax_le(m, ufn_c(x1, 2^(e_hi - k)))
  s_down <- lax_full_like(x1, 2^-k)
  s_up <- lax_full_like(x1, 2^k)
  one <- lax_full_like(x1, 1)
  y1 <- lax_select(hi, lax_mul(x1, s_down), lax_select(lo, lax_mul(x1, s_up), x1))
  y2 <- lax_select(hi, lax_mul(x2, s_down), lax_select(lo, lax_mul(x2, s_up), x2))
  inv_scale <- lax_select(hi, s_up, lax_select(lo, s_down, one))
  r_scaled <- lax_sqrt(lax_add(lax_square(y1), lax_square(y2)))
  r <- lax_mul(r_scaled, inv_scale)
  inf <- lax_full_like(x1, Inf)
  r <- lax_select(lax_bitwise_or(lax_eq(a1, inf), lax_eq(a2, inf)), inf, r)
  list(r, lax_div(y1, r_scaled), lax_div(y2, r_scaled))
}

# Port of ufuncs.py:_hypot (custom_jvp) and _hypot_jvp.
ufn_hypot <- custom_jvp(ufn_lax_hypot, jvp = function(primals, tangents) {
  dx1 <- tangents[[1L]]
  dx2 <- tangents[[2L]]
  o <- ufn_hypot(primals[[1L]], primals[[2L]])
  r <- o[[1L]]
  u1 <- o[[2L]]
  u2 <- o[[3L]]
  dr <- lax_add(lax_mul(u1, dx1), lax_mul(u2, dx2))
  list(o, list(dr, lax_div(lax_sub(dx1, lax_mul(u1, dr)), r),
               lax_div(lax_sub(dx2, lax_mul(u2, dr)), r)))
})

#' @rdname jnp_ufuncs
#' @export
jnp_hypot <- function(x1, x2) {
  a <- promote_args_inexact(x1, x2)
  if (is_complex_dtype(ufn_dtype(a[[1L]])))
    jax_error("jnp.hypot is not well defined for complex-valued inputs. Please convert to ",
              "real values first, such as by using abs(x)", class = "jax_value_error")
  b <- jnp_broadcast_arrays(a[[1L]], a[[2L]])
  ufn_hypot(b[[1L]], b[[2L]])[[1L]]
}

# ---- sinc -------------------------------------------------------------------------------------

# Horner evaluation of sum(coeffs[k] x^(k-1)), the lowering of lax.polynomial
# (lax.py:_polynomial_lower). Derivatives come from differentiating the recurrence.
ufn_polynomial <- function(x, coeffs) {
  n <- length(coeffs)
  acc <- lax_full_like(x, coeffs[n], shape = integer(0))
  for (c in rev(coeffs[-n])) acc <- lax_add(lax_mul(acc, x), ufn_c(x, c))
  acc
}

ufn_sinc_q64 <- c(-0x1.a51a6625307d3p0, 0x1.9f9cb402bc46cp-1, -0x1.86a8e4720db67p-3,
                  0x1.ac6805cf350a6p-6, -0x1.33816aa4607abp-9, 0x1.374719fab3915p-13,
                  -0x1.d42498d1ce099p-18, 0x1.0fc992ff39e13p-22, -0x1.f5f9d970ca6dfp-28,
                  0x1.79788684225eap-33, -0x1.d7353939082fep-39)
ufn_sinc_q32 <- c(-0x1.a51a66p0, 0x1.9f9cb4p-1, -0x1.86a8e4p-3, 0x1.ac6806p-6, -0x1.33816ap-9,
                  0x1.37471ap-13)

# Port of ufuncs.py:_sinc_complex.
ufn_sinc_complex <- function(x) {
  re <- lax_real(x)
  eq_inf <- lax_bitwise_and(lax_eq(lax_abs(re), ufn_c(re, Inf)), lax_is_finite(lax_imag(x)))
  pi_x <- lax_mul(ufn_c(x, pi), x)
  safe_pi_x <- jnp_where(eq_inf, 1, pi_x)
  abs_u <- lax_abs(safe_pi_x)
  use_series <- lax_lt(abs_u, ufn_c(abs_u, 1))
  safe_u <- lax_full_like(safe_pi_x, 1, shape = integer(0))
  u_series <- jnp_where(use_series, safe_pi_x, 0)
  u_direct <- jnp_where(use_series, safe_u, safe_pi_x)
  eps <- finfo(ufn_dtype(x))$eps
  coeffs <- numeric(0)
  for (k in 0:99) {
    mag <- 1 / prod(as.numeric(seq_len(2 * k + 1)))
    coeffs <- c(coeffs, (-1)^k * mag)
    if (mag <= eps) break
  }
  series <- ufn_polynomial(lax_square(u_series), coeffs)
  result <- jnp_where(use_series, series, lax_div(lax_sin(u_direct), u_direct))
  jnp_where(eq_inf, 0, result)
}

#' @rdname jnp_ufuncs
#' @export
jnp_sinc <- function(x) {
  # Port of ufuncs.py:sinc: sin(pi x) / (pi x) by argument reduction and a minimax
  # polynomial. The derivative at x = 0 comes from the stop_gradient construction
  # below (JAX 0.12 no longer uses a custom_jvp here).
  x <- promote_dtypes_inexact(list(x))[[1L]]
  if (is_complex_dtype(ufn_dtype(x))) return(ufn_sinc_complex(x))
  orig_dtype <- ufn_dtype(x)
  if (finfo(orig_dtype)$bits < 32L) x <- lax_convert_element_type(x, "float32")
  n <- lax_stop_gradient(lax_round(x, 1L))
  is_inf <- lax_eq(lax_abs(x), ufn_c(x, Inf))
  r <- jnp_where(is_inf, 0, lax_sub(x, n))
  t <- lax_mul(r, r)
  q <- ufn_polynomial(t, if (ufn_dtype(x) == "float64") ufn_sinc_q64 else ufn_sinc_q32)
  tq <- lax_mul(t, q)
  sinc_sin <- lax_add(ufn_c(tq, 1), tq)
  num <- lax_add(r, lax_mul(r, tq))
  is_odd <- lax_ne(lax_mul(ufn_c(n, 2), lax_round(lax_mul(ufn_c(n, 0.5), n), 1L)), n)
  eq_n_zero <- lax_eq(n, ufn_c(n, 0))
  denom <- jnp_where(is_odd, lax_neg(x), jnp_where(eq_n_zero, 1, x))
  signed_scaled <- lax_div(num, denom)
  signed_scaled <- jnp_where(lax_eq(r, ufn_c(r, 0)), lax_sub(signed_scaled, lax_stop_gradient(signed_scaled)),
                             signed_scaled)
  result <- jnp_where(eq_n_zero, sinc_sin, signed_scaled)
  if (orig_dtype != ufn_dtype(x)) result <- lax_convert_element_type(result, orig_dtype)
  result
}

# ---- reductions behind the ufunc reduce methods (reductions.py) ---------------------------------

ufn_check_where <- function(name, where) {
  if (is.null(where)) return(NULL)
  where <- as_jax(where)
  if (ufn_dtype(where) != "bool")
    jax_error("jnp.", name, ": where must be None or a boolean array; got where.dtype=",
              ufn_dtype(where), ".", class = "jax_value_error")
  where
}

# Port of reductions._reduction_init_val as a scalar of dtype dt (init_val may also be a
# function of dt).
ufn_init_scalar <- function(dt, v) {
  if (is.function(v)) return(v(dt))
  if (dt == "bool") return(ufn_scalar(v > 0, "bool"))
  if (is_int_dtype(dt) && is.infinite(v)) {
    lo <- ufn_scalar(if (is_signed_dtype(dt)) -2^(dtype_bits(dt) - 1L) else 0, dt)
    return(if (v < 0) lo else lax_bitwise_not(lo))
  }
  ufn_scalar(v, dt)
}

# Port of reductions._reduction. `op` combines `initial` with the result and `reducer`
# is the monoid reduction of `op` (the lax.reduce pattern match of JAX).
ufn_reduction <- function(a, name, op, reducer, init_val, has_identity = TRUE, preproc = NULL,
                          bool_op = op, bool_reducer = reducer, upcast_f16 = FALSE,
                          axis = NULL, dtype = NULL, keepdims = FALSE, initial = NULL,
                          where = NULL, promote_integers = FALSE) {
  a <- as_jax(a)
  where <- ufn_check_where(name, where)
  if (is.null(initial) && !has_identity && !is.null(where))
    jax_error("reduction operation ", name, " does not have an identity, so to use a ",
              "where mask one has to specify 'initial'", class = "jax_value_error")
  if (!is.null(preproc)) a <- preproc(a)
  dims <- reduction_axes(a, axis)
  if (is.null(initial) && !has_identity && any(aval_of(a)$shape[dims + 1L] < 1L))
    jax_error("zero-size array to reduction operation ", name, " which has no identity",
              class = "jax_value_error")
  result_dtype <- if (!is.null(dtype)) dtype(dtype)
                  else if (promote_integers) sum_dtype(ufn_dtype(a)) else ufn_dtype(a)
  comp <- if (upcast_f16 && result_dtype %in% c("float16", "bfloat16")) "float32" else result_dtype
  a <- lax_convert_element_type(a, comp)
  if (comp == "bool") {
    op <- bool_op
    reducer <- bool_reducer
  }
  if (!is.null(where)) a <- jnp_where(where, a, ufn_init_scalar(comp, init_val))
  result <- reducer(a, dims)
  if (!is.null(initial)) {
    initial_arr <- lax_convert_element_type(as_jax(initial), comp)
    if (length(aval_of(initial_arr)$shape))
      jax_error("initial value must be a scalar. Got array of shape ",
                ufn_shape_str(aval_of(initial_arr)$shape), class = "jax_value_error")
    result <- op(initial_arr, result)
  }
  if (keepdims) result <- lax_expand_dims(result, dims)
  lax_convert_element_type(result, result_dtype)
}

ufn_cast_to_numeric <- function(x) promote_dtypes_numeric(list(x))[[1L]]
ufn_cast_to_bool <- function(x) {
  if (is_complex_dtype(ufn_dtype(x))) x <- lax_real(x)
  lax_convert_element_type(x, "bool")
}
ufn_require_integer <- function(x) {
  dt <- ufn_dtype(x)
  if (!(dt == "bool" || is_int_dtype(dt)))
    jax_error("integer argument required; got dtype=", dt, class = "jax_value_error")
  x
}

# Ports of reductions.sum, prod, _reduce_max, _reduce_min, _reduce_bitwise_* and
# _reduce_logical_*, with the signature of the ufunc reduce slot.
ufn_reduce_sum <- function(a, axis = NULL, dtype = NULL, keepdims = FALSE, initial = NULL,
                           where = NULL) {
  ufn_reduction(a, "sum", lax_add, lax_reduce_sum, 0, preproc = ufn_cast_to_numeric,
                bool_op = lax_bitwise_or, bool_reducer = lax_reduce_or, upcast_f16 = is.null(dtype),
                axis = axis, dtype = dtype, keepdims = keepdims, initial = initial, where = where,
                promote_integers = TRUE)
}
ufn_reduce_prod <- function(a, axis = NULL, dtype = NULL, keepdims = FALSE, initial = NULL,
                            where = NULL) {
  ufn_reduction(a, "prod", lax_mul, lax_reduce_prod, 1, preproc = ufn_cast_to_numeric,
                bool_op = lax_bitwise_and, bool_reducer = lax_reduce_and, upcast_f16 = is.null(dtype),
                axis = axis, dtype = dtype, keepdims = keepdims, initial = initial, where = where,
                promote_integers = TRUE)
}
ufn_reduce_max <- function(a, axis = NULL, dtype = NULL, keepdims = FALSE, initial = NULL,
                           where = NULL) {
  ufn_reduction(a, "max", lax_max, lax_reduce_max, -Inf, has_identity = FALSE,
                bool_op = lax_bitwise_or, bool_reducer = lax_reduce_or, axis = axis, dtype = dtype,
                keepdims = keepdims, initial = initial, where = where)
}
ufn_reduce_min <- function(a, axis = NULL, dtype = NULL, keepdims = FALSE, initial = NULL,
                           where = NULL) {
  ufn_reduction(a, "min", lax_min, lax_reduce_min, Inf, has_identity = FALSE,
                bool_op = lax_bitwise_and, bool_reducer = lax_reduce_and, axis = axis, dtype = dtype,
                keepdims = keepdims, initial = initial, where = where)
}
ufn_all_ones <- function(dt) lax_bitwise_not(ufn_scalar(0, dt))
ufn_bitwise_reducer <- function(name, op, reducer, init_val, preproc) {
  force(name); force(op); force(reducer); force(init_val); force(preproc)
  function(a, axis = NULL, dtype = NULL, keepdims = FALSE, initial = NULL, where = NULL) {
    ufn_reduction(a, name, op, reducer, init_val, preproc = preproc, axis = axis, dtype = dtype,
                  keepdims = keepdims, initial = initial, where = where)
  }
}
ufn_reduce_bitwise_and <- ufn_bitwise_reducer("reduce_bitwise_and", lax_bitwise_and, lax_reduce_and,
                                              ufn_all_ones, ufn_require_integer)
ufn_reduce_bitwise_or <- ufn_bitwise_reducer("reduce_bitwise_or", lax_bitwise_or, lax_reduce_or, 0,
                                             ufn_require_integer)
ufn_reduce_bitwise_xor <- ufn_bitwise_reducer("reduce_bitwise_xor", lax_bitwise_xor, lax_reduce_xor, 0,
                                              ufn_require_integer)
ufn_reduce_logical_and <- ufn_bitwise_reducer("reduce_logical_and", lax_bitwise_and, lax_reduce_and,
                                              TRUE, ufn_cast_to_bool)
ufn_reduce_logical_or <- ufn_bitwise_reducer("reduce_logical_or", lax_bitwise_or, lax_reduce_or,
                                             FALSE, ufn_cast_to_bool)
ufn_reduce_logical_xor <- ufn_bitwise_reducer("reduce_logical_xor", lax_bitwise_xor, lax_reduce_xor,
                                              FALSE, ufn_cast_to_bool)

# Port of reductions._logsumexp (the reduce method of jnp.logaddexp).
ufn_logsumexp <- function(a, axis = NULL, dtype = NULL, keepdims = FALSE, initial = NULL,
                          where = NULL) {
  if (!is.null(dtype)) dtype(dtype)
  where <- ufn_check_where("logsumexp", where)
  a_arr <- promote_dtypes_inexact(list(a))[[1L]]
  dims <- reduction_axes(a_arr, axis)
  amax <- ufn_reduce_max(jnp_real(a_arr), axis = dims, keepdims = keepdims, where = where,
                         initial = -Inf)
  amax <- lax_stop_gradient(lax_select(lax_is_finite(amax), amax, lax_full_like(amax, 0)))
  amax_with_dims <- if (keepdims) amax else lax_expand_dims(amax, dims)
  exp_a <- lax_exp(lax_sub(a_arr, lax_convert_element_type(amax_with_dims, ufn_dtype(a_arr))))
  sumexp <- ufn_reduce_sum(exp_a, axis = dims, keepdims = keepdims, where = where)
  result <- lax_add(lax_log(sumexp), lax_convert_element_type(amax, ufn_dtype(sumexp)))
  if (is.null(initial)) result else lax_other_logaddexp(initial, result)
}

# Port of reductions._logsumexp2 (the reduce method of jnp.logaddexp2).
ufn_logsumexp2 <- function(a, axis = NULL, dtype = NULL, keepdims = FALSE, initial = NULL,
                           where = NULL) {
  if (!is.null(dtype)) dtype(dtype)
  a <- as_jax(a)
  where <- ufn_check_where("logsumexp2", where)
  ln2 <- log(2)
  if (!is.null(initial)) initial <- if (is_jax_value(initial)) jnp_multiply(initial, ln2) else initial * ln2
  jnp_true_divide(ufn_logsumexp(jnp_multiply(a, ln2), axis = axis, dtype = dtype,
                                keepdims = keepdims, where = where, initial = initial), ln2)
}

# ---- jnp.ufunc (ufunc_api.py) ---------------------------------------------------------------------

#' NumPy style universal functions
#'
#' Port of `jax.numpy.ufunc`. `jnp_ufunc()` wraps an elementwise function `func` of
#' `nin` arrays returning `nout` arrays (a list when `nout > 1`) into a callable
#' ufunc object; calling it broadcasts the inputs and maps `func` over every element
#' (with [vmap()]) unless a vectorized `call` is supplied. `jnp_frompyfunc()` is
#' `jnp.frompyfunc`. The ufunc methods of JAX are the functions `jnp_ufunc_reduce()`
#' (`ufunc.reduce`), `jnp_ufunc_accumulate()`, `jnp_ufunc_reduceat()`,
#' `jnp_ufunc_outer()` and `jnp_ufunc_at()`; they are also reachable with `$`, as in
#' `jnp_ufuncs$add$reduce(x)`, and `u$nin`, `u$nout`, `u$nargs`, `u$identity` and
#' `u$name` give the properties.
#'
#' Axes are 0-based. `indices` of `jnp_ufunc_at()` is one index (see [at_get()]) or a
#' plain list of indices, one per dimension (a Python tuple). As in JAX, `at` cannot
#' operate in place: pass `inplace = FALSE` and use the result.
#' @param func Elementwise function of `nin` arguments.
#' @param nin,nout Number of inputs and outputs.
#' @param name Name of the ufunc (by default the name of `func` or `"<lambda>"`); for
#'   `$`, the property or method name.
#' @param nargs Number of arguments (default `nin`).
#' @param identity Identity element of a binary ufunc, or `NULL`.
#' @param call,reduce,accumulate,at,reduceat Optional specialized implementations.
#' @param ufunc A ufunc object.
#' @param a,A,B Arrays.
#' @param axis Axis (0-based); `NULL` reduces over all axes in `jnp_ufunc_reduce()`.
#' @param dtype Optional result dtype.
#' @param out Not supported (must be `NULL`).
#' @param keepdims Keep reduced dimensions with size one.
#' @param initial Initial value of the reduction.
#' @param where Boolean mask of the elements to reduce.
#' @param indices Indices (see Details).
#' @param b Second operand of `at` for binary ufuncs.
#' @param inplace Must be `FALSE` (JAX arrays are immutable).
#' @param x A ufunc object.
#' @param ... Ignored.
#' @return `jnp_ufunc()` and `jnp_frompyfunc()` return a ufunc object; the methods return
#'   arrays.
#' @examples
#' \dontrun{
#' jnp_ufuncs$add$reduce(jnp_arange(5))
#' u <- jnp_frompyfunc(function(x, y) x * y + 1, nin = 2, nout = 1, identity = 0)
#' jnp_ufunc_outer(u, c(1, 2), c(3, 4, 5))
#' }
#' @export
jnp_ufunc <- function(func, nin, nout, name = NULL, nargs = NULL, identity = NULL, call = NULL,
                      reduce = NULL, accumulate = NULL, at = NULL, reduceat = NULL) {
  if (is.null(name)) {
    s <- substitute(func)
    name <- if (is.name(s)) as.character(s) else "<lambda>"
  }
  index <- function(v, what) {
    if (!is.numeric(v) || length(v) != 1L || v != round(v))
      jax_error("'", what, "' must be an integer, got ", format(v), class = "jax_type_error")
    as.integer(v)
  }
  props <- list(func = func, name = name, nin = index(nin, "nin"), nout = index(nout, "nout"),
                nargs = index(nargs %||% nin, "nargs"), identity = identity, call = call,
                reduce = reduce, accumulate = accumulate, at = at, reduceat = reduceat)
  f <- function(..., out = NULL, where = NULL) {
    if (!is.null(out)) jax_error("out argument of ", ufn_repr(name), class = "jax_not_implemented_error")
    if (!is.null(where))
      jax_error("where argument of ", ufn_repr(name), class = "jax_not_implemented_error")
    (call %||% ufn_vectorize(func))(...)
  }
  structure(f, class = c("jnp_ufunc", "function"), ufunc = props)
}

#' @rdname jnp_ufunc
#' @export
jnp_frompyfunc <- function(func, nin, nout, identity = NULL) {
  s <- substitute(func)
  jnp_ufunc(func, nin, nout, name = if (is.name(s)) as.character(s) else "<lambda>",
            identity = identity)
}

ufn_repr <- function(name) paste0("<jnp.ufunc '", name, "'>")

ufn_props <- function(ufunc) {
  if (!inherits(ufunc, "jnp_ufunc"))
    jax_error("expected a jnp_ufunc object, got an object of class '", class(ufunc)[1L], "'",
              class = "jax_type_error")
  attr(ufunc, "ufunc")
}

ufn_check_binary <- function(p, what) {
  if (p$nin != 2L) jax_error(what, " only supported for binary ufuncs", class = "jax_value_error")
  if (p$nout != 1L)
    jax_error(what, " only supported for functions returning a single value", class = "jax_value_error")
}

# Port of vectorize.py:vectorize without a signature: broadcast the arguments and map
# func over every dimension. (JAX maps over unit dimensions without broadcasting them;
# the values are the same.)
ufn_vectorize <- function(func) {
  force(func)
  function(...) {
    args <- jnp_broadcast_arrays(...)
    f <- func
    nd <- if (length(args)) length(aval_of(args[[1L]])$shape) else 0L
    for (i in seq_len(nd)) f <- vmap(f)
    do.call(f, args)
  }
}

ufn_one <- function(x) lax_full_like(as_jax(x), 1, shape = integer(0))
ufn_func_dtype <- function(p, ...) eval_shape(p$func, ...)$dtype
ufn_take0 <- function(x, i) lax_dynamic_index_in_dim(x, i, 0L, keepdims = FALSE)

#' @rdname jnp_ufunc
#' @export
jnp_ufunc_reduce <- function(ufunc, a, axis = 0L, dtype = NULL, out = NULL, keepdims = FALSE,
                             initial = NULL, where = NULL) {
  # Port of ufunc_api.py:ufunc.reduce.
  p <- ufn_props(ufunc)
  ufn_check_binary(p, "reduce")
  if (!is.null(out))
    jax_error("out argument of ", p$name, ".reduce()", class = "jax_not_implemented_error")
  if (!is.null(where)) {
    where <- as_jax(where)
    if (is.null(p$identity) && is.null(initial))
      jax_error("reduction operation '", p$name, "' does not have an identity, so to use a ",
                "where mask one has to specify 'initial'.", class = "jax_value_error")
    if (ufn_dtype(where) != "bool")
      jax_error("where argument must have dtype=bool; got dtype=", ufn_dtype(where),
                class = "jax_value_error")
  }
  if (is.null(p$reduce))
    return(ufn_reduce_via_scan(ufunc, p, a, axis, dtype, keepdims, initial, where))
  p$reduce(a, axis = axis, dtype = dtype, keepdims = keepdims, initial = initial, where = where)
}

# Port of ufunc_api.py:ufunc._reduce_via_scan.
ufn_reduce_via_scan <- function(ufunc, p, arr, axis, dtype, keepdims, initial, where) {
  arr <- as_jax(arr)
  if (is.null(initial)) initial <- p$identity
  if (is.null(dtype)) dtype <- ufn_func_dtype(p, ufn_one(arr), ufn_one(arr))
  shape <- aval_of(arr)$shape
  if (!is.null(where)) where <- jnp_broadcast_to(where, shape)
  if (length(axis) > 1L) jax_error("tuple of axes", class = "jax_not_implemented_error")
  if (is.null(axis)) {
    final_shape <- if (keepdims) rep(1L, length(shape)) else integer(0)
    arr <- jnp_ravel(arr)
    if (!is.null(where)) where <- jnp_ravel(where)
    axis <- 0L
  } else {
    axis <- norm_axis(axis, length(shape))
    final_shape <- if (keepdims) replace(shape, axis + 1L, 1L) else shape[-(axis + 1L)]
  }
  if (axis != 0L) {
    arr <- jnp_moveaxis(arr, axis, 0L)
    if (!is.null(where)) where <- jnp_moveaxis(where, axis, 0L)
  }
  n <- aval_of(arr)$shape[1L]
  if (n == 0L) {
    if (is.null(initial))
      jax_error("zero-size array to reduction operation ", p$name, " which has no identity",
                class = "jax_value_error")
    return(lax_full(final_shape, initial, dtype))
  }
  body_fun <- function(i, val) {
    new <- ufunc(val, lax_convert_element_type(ufn_take0(arr, i), dtype))
    if (is.null(where)) new else jnp_where(ufn_take0(where, i), new, val)
  }
  if (is.null(initial)) {
    start_index <- 1L
    start_value <- ufn_take0(arr, 0L)
  } else {
    start_index <- 0L
    start_value <- initial
  }
  start_value <- jnp_broadcast_to(lax_convert_element_type(as_jax(start_value), dtype),
                                  drop_first(aval_of(arr)$shape, 1L))
  result <- lax_fori_loop(start_index, n, body_fun, start_value)
  if (keepdims) result <- lax_reshape(result, final_shape)
  result
}

#' @rdname jnp_ufunc
#' @export
jnp_ufunc_accumulate <- function(ufunc, a, axis = 0L, dtype = NULL, out = NULL) {
  # Port of ufunc_api.py:ufunc.accumulate.
  p <- ufn_props(ufunc)
  ufn_check_binary(p, "accumulate")
  if (!is.null(out))
    jax_error("out argument of ", p$name, ".accumulate()", class = "jax_not_implemented_error")
  if (is.null(p$accumulate)) return(ufn_accumulate_via_scan(ufunc, p, a, axis, dtype))
  p$accumulate(a, axis = axis, dtype = dtype)
}

# Port of ufunc_api.py:ufunc._accumulate_via_scan.
ufn_accumulate_via_scan <- function(ufunc, p, arr, axis, dtype) {
  arr <- as_jax(arr)
  if (is.null(dtype)) dtype <- ufn_func_dtype(p, ufn_one(arr), ufn_one(arr))
  if (is.null(axis) || length(axis) > 1L)
    jax_error("accumulate does not allow multiple axes", class = "jax_value_error")
  shape <- aval_of(arr)$shape
  axis <- norm_axis(axis, length(shape))
  if (prod_int(shape) == 0L) return(lax_full(shape, 0, dtype))
  arr <- jnp_moveaxis(arr, axis, 0L)
  first <- lax_convert_element_type(ufn_take0(arr, 0L), dtype)
  scan_fun <- function(carry, x) {
    i <- carry[[1L]]
    y <- jnp_where(lax_eq(i, 0L), first,
                   ufunc(lax_convert_element_type(carry[[2L]], dtype),
                         lax_convert_element_type(ufn_take0(arr, i), dtype)))
    list(list(lax_add(i, 1L), y), y)
  }
  res <- lax_scan(scan_fun, list(0L, first), NULL, length = shape[axis + 1L])
  jnp_moveaxis(res[[2L]], 0L, axis)
}

#' @rdname jnp_ufunc
#' @export
jnp_ufunc_reduceat <- function(ufunc, a, indices, axis = 0L, dtype = NULL, out = NULL) {
  # Port of ufunc_api.py:ufunc.reduceat.
  p <- ufn_props(ufunc)
  ufn_check_binary(p, "reduceat")
  if (!is.null(out))
    jax_error("out argument of ", p$name, ".reduceat()", class = "jax_not_implemented_error")
  if (is.null(p$reduceat)) return(ufn_reduceat_via_scan(ufunc, a, indices, axis, dtype))
  p$reduceat(a, indices, axis = axis, dtype = dtype)
}

# Port of ufunc_api.py:ufunc._reduceat_via_scan.
ufn_reduceat_via_scan <- function(ufunc, a, indices, axis, dtype) {
  a <- as_jax(a)
  if (is.numeric(indices) && !is_jax_value(indices)) storage.mode(indices) <- "integer"
  indices <- as_jax(indices)
  shape <- aval_of(a)$shape
  if (!length(shape))
    jax_error("reduceat: a must have 1 or more dimension, got a.shape=()", class = "jax_value_error")
  if (length(aval_of(indices)$shape) != 1L)
    jax_error("reduceat: indices must be one-dimensional, got indices.shape=",
              ufn_shape_str(aval_of(indices)$shape), class = "jax_value_error")
  if (!is.null(dtype)) a <- lax_convert_element_type(a, dtype)
  if (is.null(axis) || length(axis) != 1L)
    jax_error("reduceat requires a single integer axis.", class = "jax_value_error")
  axis <- norm_axis(axis, length(shape))
  out <- jnp_take(a, indices, axis = axis)
  nd <- length(aval_of(out)$shape)
  ind <- lax_expand_dims(jnp_concatenate(list(indices, lax_reshape(as_jax(shape[axis + 1L]), 1L))),
                         setdiff(seq_len(nd) - 1L, axis))
  m <- aval_of(ind)$shape[axis + 1L]
  ind_start <- lax_slice_in_dim(ind, 0L, m - 1L, axis = axis)
  ind_end <- lax_slice_in_dim(ind, 1L, m, axis = axis)
  loop_body <- function(i, out) {
    jnp_where(lax_bitwise_and(jnp_greater(i, ind_start), jnp_less(i, ind_end)),
              ufunc(out, jnp_take(a, lax_expand_dims(i, 0L), axis = axis)), out)
  }
  lax_fori_loop(0L, shape[axis + 1L], loop_body, out)
}

#' @rdname jnp_ufunc
#' @export
jnp_ufunc_outer <- function(ufunc, A, B) {
  # Port of ufunc_api.py:ufunc.outer.
  p <- ufn_props(ufunc)
  ufn_check_binary(p, "outer")
  A <- as_jax(A)
  B <- as_jax(B)
  ravel <- function(x) lax_reshape(x, prod_int(aval_of(x)$shape))
  result <- vmap(vmap(ufunc, in_axes = list(NULL, 0L)), in_axes = list(0L, NULL))(ravel(A), ravel(B))
  lax_reshape(result, c(aval_of(A)$shape, aval_of(B)$shape))
}

ufn_at_warning <- paste0(
  "Because JAX arrays are immutable, jnp.ufunc.at() cannot operate inplace like\n",
  "np.ufunc.at(). Instead, you can pass inplace=False and capture the result; e.g.\n",
  ">>> arr = jnp.add.at(arr, ind, val, inplace=False)\n")

#' @rdname jnp_ufunc
#' @export
jnp_ufunc_at <- function(ufunc, a, indices, b = NULL, inplace = TRUE) {
  # Port of ufunc_api.py:ufunc.at.
  p <- ufn_props(ufunc)
  if (inplace) jax_error(ufn_at_warning, class = "jax_not_implemented_error")
  if (is.null(p$at)) {
    if (is.null(b)) return(ufn_at_via_scan(ufunc, p, a, indices))
    return(ufn_at_via_scan(ufunc, p, a, indices, b))
  }
  if (is.null(b)) p$at(a, indices) else p$at(a, indices, b)
}

# A plain list is a tuple of indices; anything else is a single index.
ufn_index_tuple <- function(indices) {
  if (is.list(indices) && is.null(oldClass(indices))) indices else list(indices)
}
ufn_is_slice <- function(i) inherits(i, c("jax_slice", "jax_newaxis", "jax_ellipsis"))
ufn_at_get <- function(a, idx) do.call(at_get, c(list(a), idx))
ufn_at_update <- function(f, a, idx, value) do.call(f, c(list(a), idx, list(value = value)))

# Port of ufunc_api.py:ufunc._at_via_scan.
ufn_at_via_scan <- function(ufunc, p, a, indices, ...) {
  args <- list(...)
  dtype <- do.call(ufn_func_dtype, c(list(p, ufn_one(a)), lapply(args, ufn_one)))
  a <- lax_convert_element_type(as_jax(a), dtype)
  args <- lapply(args, function(v) lax_convert_element_type(as_jax(v), dtype))
  idx <- ufn_index_tuple(indices)
  if (!length(idx)) return(a)
  idx <- lapply(idx, function(i) {
    if (is.numeric(i) && !is_jax_value(i) && !ufn_is_slice(i)) storage.mode(i) <- "integer"
    i
  })
  arrays <- !vapply(idx, ufn_is_slice, NA)
  shapes <- lapply(idx[arrays], function(i) aval_of(as_jax(i))$shape)
  shape <- if (length(shapes)) broadcast_shapes_impl(shapes) else integer(0)
  if (!length(shape)) {
    return(ufn_at_update(at_set, a, idx, do.call(ufunc, c(list(ufn_at_get(a, idx)), args))))
  }
  n <- prod_int(shape)
  if (length(args)) {
    s0 <- aval_of(args[[1L]])$shape
    rest <- drop_first(s0, length(shape))
    args <- list(lax_reshape(jnp_broadcast_to(args[[1L]], c(shape, rest)), c(n, rest)))
  }
  idx[arrays] <- lapply(idx[arrays], function(i) jnp_ravel(jnp_broadcast_to(i, shape)))
  scan_fun <- function(carry, x) {
    i <- carry[[1L]]
    ai <- carry[[2L]]
    ix <- idx
    ix[arrays] <- lapply(idx[arrays], ufn_take0, i = i)
    vals <- lapply(args, ufn_take0, i = i)
    ai <- ufn_at_update(at_set, ai, ix, do.call(ufunc, c(list(ufn_at_get(ai, ix)), vals)))
    list(list(lax_add(i, 1L), ai), x)
  }
  # JAX scans len(indices[0]) steps, which is n whenever the first index is an array.
  lax_scan(scan_fun, list(0L, a), NULL, length = n)[[1L]][[2L]]
}

#' @rdname jnp_ufunc
#' @export
`$.jnp_ufunc` <- function(x, name) {
  p <- attr(x, "ufunc")
  switch(name,
    nin = , nout = , nargs = , identity = , name = p[[name]],
    reduce = function(...) jnp_ufunc_reduce(x, ...),
    accumulate = function(...) jnp_ufunc_accumulate(x, ...),
    reduceat = function(...) jnp_ufunc_reduceat(x, ...),
    outer = function(...) jnp_ufunc_outer(x, ...),
    at = function(...) jnp_ufunc_at(x, ...),
    jax_error("'jnp.ufunc' object has no attribute '", name, "'", class = "jax_value_error"))
}

#' @rdname jnp_ufunc
#' @export
print.jnp_ufunc <- function(x, ...) {
  cat(ufn_repr(attr(x, "ufunc")$name), "\n", sep = "")
  invisible(x)
}

# ---- the standard ufunc objects -------------------------------------------------------------

# Ports of ufuncs.py:_add_at, _multiply_at and _subtract_at.
ufn_bool_at <- function(f) {
  force(f)
  function(a, indices, b) {
    a <- as_jax(a)
    idx <- ufn_index_tuple(indices)
    if (ufn_dtype(a) != "bool") return(ufn_at_update(f, a, idx, b))
    a <- lax_convert_element_type(a, "int32")
    b <- lax_convert_element_type(lax_convert_element_type(as_jax(b), "bool"), "int32")
    lax_convert_element_type(ufn_at_update(f, a, idx, b), "bool")
  }
}
ufn_subtract_at <- function(a, indices, b) ufn_at_update(at_subtract, a, ufn_index_tuple(indices), b)

ufn_binary <- function(name, func, identity, reduce = NULL, accumulate = NULL, at = NULL) {
  jnp_ufunc(func, 2L, 1L, name = name, identity = identity, call = func, reduce = reduce,
            accumulate = accumulate, at = at)
}

#' @rdname jnp_ufuncs
#' @format NULL
#' @details `jnp_ufuncs` is a list of the [jnp_ufunc()] objects of `jax.numpy`
#'   (`jnp_ufuncs$add` is `jnp.add`): `add`, `bitwise_and`, `bitwise_or`, `bitwise_xor`,
#'   `logaddexp`, `logaddexp2`, `logical_and`, `logical_or`, `logical_xor`, `maximum`,
#'   `minimum`, `multiply`, `negative` and `subtract`. They have the methods `reduce`,
#'   `accumulate`, `reduceat`, `outer` and `at`, for example `jnp_ufuncs$add$reduce(x)`.
#'   `jnp_divmod()`, `jnp_frexp()` and `jnp_modf()` return lists of two arrays.
#' @export
jnp_ufuncs <- list(
  add = ufn_binary("add", function(x1, x2) jnp_add(x1, x2), 0,
                   reduce = function(...) ufn_reduce_sum(...),
                   accumulate = function(a, axis = 0L, dtype = NULL) jnp_cumsum(a, axis = axis, dtype = dtype),
                   at = ufn_bool_at(function(...) at_add(...))),
  bitwise_and = ufn_binary("bitwise_and", function(x1, x2) jnp_bitwise_and(x1, x2), -1,
                           reduce = function(...) ufn_reduce_bitwise_and(...)),
  bitwise_or = ufn_binary("bitwise_or", function(x1, x2) jnp_bitwise_or(x1, x2), 0,
                          reduce = function(...) ufn_reduce_bitwise_or(...)),
  bitwise_xor = ufn_binary("bitwise_xor", function(x1, x2) jnp_bitwise_xor(x1, x2), 0,
                           reduce = function(...) ufn_reduce_bitwise_xor(...)),
  logaddexp = ufn_binary("logaddexp", function(x1, x2) jnp_logaddexp(x1, x2), -Inf,
                         reduce = function(...) ufn_logsumexp(...)),
  logaddexp2 = ufn_binary("logaddexp2", function(x1, x2) jnp_logaddexp2(x1, x2), -Inf,
                          reduce = function(...) ufn_logsumexp2(...)),
  logical_and = ufn_binary("logical_and", function(x1, x2) jnp_logical_and(x1, x2), TRUE,
                           reduce = function(...) ufn_reduce_logical_and(...)),
  logical_or = ufn_binary("logical_or", function(x1, x2) jnp_logical_or(x1, x2), FALSE,
                          reduce = function(...) ufn_reduce_logical_or(...)),
  logical_xor = ufn_binary("logical_xor", function(x1, x2) jnp_logical_xor(x1, x2), FALSE,
                           reduce = function(...) ufn_reduce_logical_xor(...)),
  maximum = ufn_binary("maximum", function(x1, x2) jnp_maximum(x1, x2), NULL,
                       reduce = function(...) ufn_reduce_max(...)),
  minimum = ufn_binary("minimum", function(x1, x2) jnp_minimum(x1, x2), NULL,
                       reduce = function(...) ufn_reduce_min(...)),
  multiply = ufn_binary("multiply", function(x1, x2) jnp_multiply(x1, x2), 1,
                        reduce = function(...) ufn_reduce_prod(...),
                        accumulate = function(a, axis = 0L, dtype = NULL) jnp_cumprod(a, axis = axis, dtype = dtype),
                        at = ufn_bool_at(function(...) at_multiply(...))),
  negative = jnp_ufunc(function(x) jnp_negative(x), 1L, 1L, name = "negative",
                       call = function(x) jnp_negative(x)),
  subtract = ufn_binary("subtract", function(x1, x2) jnp_subtract(x1, x2), NULL,
                        at = ufn_subtract_at)
)
