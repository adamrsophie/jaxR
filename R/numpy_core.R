# jax.numpy core: promotion, broadcasting, ufuncs and basic array functions
# (port of jax/_src/numpy/util.py, ufuncs.py, lax_numpy.py, reductions.py subsets).

# ---- promotion helpers (jax._src.numpy.util) ---------------------------------------------

promote_dtypes <- function(args) {
  args <- lapply(args, as_jax)
  avs <- lapply(args, aval_of)
  if (length(args) < 2L) return(args)
  res <- promote_types(vapply(avs, `[[`, "", "dtype"), vapply(avs, `[[`, NA, "weak"))
  lapply(args, function(a) {
    av <- aval_of(a)
    if (av$dtype == res$dtype && av$weak == res$weak) a
    else lax_convert_element_type(a, res$dtype, weak_type = res$weak)
  })
}

# Port of dtypes.to_inexact_dtype: bool and integers up to 32 bits become float32,
# 64-bit integers float64 (then canonicalized).
to_inexact_dtype <- function(dt) {
  if (is_inexact_dtype(dt)) return(dt)
  canonicalize_dtype(if (dtype_bits(dt) > 32L) "float64" else "float32")
}

promote_dtypes_inexact <- function(args) {
  args <- promote_dtypes(args)
  lapply(args, function(a) {
    av <- aval_of(a)
    if (is_inexact_dtype(av$dtype)) a
    else lax_convert_element_type(a, to_inexact_dtype(av$dtype), weak_type = av$weak)
  })
}

promote_dtypes_numeric <- function(args) {
  args <- promote_dtypes(args)
  lapply(args, function(a) {
    av <- aval_of(a)
    if (av$dtype == "bool") lax_convert_element_type(a, "int32", weak_type = av$weak) else a
  })
}

# Rank promotion: prepend unit dimensions so lax size one broadcasting applies.
promote_shapes <- function(args) {
  if (length(args) < 2L) return(args)
  shapes <- lapply(args, function(a) aval_of(a)$shape)
  nd <- max(vapply(shapes, length, 1L))
  broadcast_shapes_impl(shapes)
  lapply(args, function(a) {
    s <- aval_of(a)$shape
    if (length(s) == 0L || length(s) == nd) return(a)
    if (identical(.jx$config$jax_numpy_rank_promotion, "raise"))
      jax_error("Operands could not be broadcast together with shapes ",
                paste(vapply(shapes, function(z) paste0("(", paste(z, collapse = ","), ")"), ""),
                      collapse = " "), " and with the config option jax_numpy_rank_promotion='raise'.")
    lax_reshape(a, c(rep(1L, nd - length(s)), s))
  })
}

# The ufuncs of jax.numpy are jit functions, and the jet rule of jit gives a constant
# operand explicit zero terms before the promotion inside: a weakly typed conversion of
# the constant then gives weakly typed zero terms. jet_weak_scope marks that region for
# process_jet; elsewhere (jnp_stack, lax functions) the zero terms of constants are
# strongly typed, as in JAX.
# The arguments are evaluated before the scope opens, so that code in them is outside.
jet_weak_scope <- function(f, args) {
  force(args)
  old <- .jx$jet_weak
  .jx$jet_weak <- TRUE
  on.exit(.jx$jet_weak <- old)
  promote_shapes(f(args))
}
promote_args <- function(...) jet_weak_scope(promote_dtypes, list(...))
promote_args_inexact <- function(...) jet_weak_scope(promote_dtypes_inexact, list(...))
promote_args_numeric <- function(...) jet_weak_scope(promote_dtypes_numeric, list(...))

# Port of ufuncs._to_bool: a value is true when it differs from zero (so a complex
# value with a zero real part and a nonzero imaginary part is true).
# Port of ufuncs._to_bool (ufn_c is lax._const).
to_bool <- function(x) if (aval_of(x)$dtype == "bool") x else lax_ne(x, ufn_c(x, 0))

# Port of ufuncs._complex_comparison: complex values order by real part, then by
# imaginary part.
complex_comparison <- function(op, x1, x2) {
  a <- promote_args(x1, x2)
  x <- a[[1L]]
  y <- a[[2L]]
  if (!is_complex_dtype(aval_of(x)$dtype)) return(op(x, y))
  lax_select(lax_eq(lax_real(x), lax_real(y)), op(lax_imag(x), lax_imag(y)),
             op(lax_real(x), lax_real(y)))
}

jnp_unary <- function(f, inexact = TRUE) {
  function(x) {
    x <- if (inexact) promote_dtypes_inexact(list(x))[[1L]] else as_jax(x)
    f(x)
  }
}

# ---- ufuncs --------------------------------------------------------------------------------

#' jax.numpy elementwise functions
#'
#' Ports of the `jax.numpy` ufuncs. Arguments broadcast with NumPy rules and are
#' promoted with JAX type promotion.
#' @param x,x1,x2 Arrays or R values.
#' @name jnp_ufuncs
#' @return An array.
NULL

#' @rdname jnp_ufuncs
#' @export
jnp_add <- function(x1, x2) {
  a <- promote_args(x1, x2)
  if (aval_of(a[[1L]])$dtype == "bool") return(lax_bitwise_or(a[[1L]], a[[2L]]))
  lax_add(a[[1L]], a[[2L]])
}
#' @rdname jnp_ufuncs
#' @export
jnp_subtract <- function(x1, x2) { a <- promote_args(x1, x2); lax_sub(a[[1L]], a[[2L]]) }
#' @rdname jnp_ufuncs
#' @export
jnp_multiply <- function(x1, x2) {
  a <- promote_args(x1, x2)
  if (aval_of(a[[1L]])$dtype == "bool") return(lax_bitwise_and(a[[1L]], a[[2L]]))
  lax_mul(a[[1L]], a[[2L]])
}
#' @rdname jnp_ufuncs
#' @export
jnp_true_divide <- function(x1, x2) { a <- promote_args_inexact(x1, x2); lax_div(a[[1L]], a[[2L]]) }
#' @rdname jnp_ufuncs
#' @export
jnp_divide <- jnp_true_divide
#' @rdname jnp_ufuncs
#' @export
jnp_floor_divide <- function(x1, x2) {
  a <- promote_args_numeric(x1, x2)
  x <- a[[1L]]; y <- a[[2L]]
  dt <- aval_of(x)$dtype
  if (is_unsigned_dtype(dt)) return(lax_div(x, y))
  if (is_signed_dtype(dt)) {
    zero <- lax_full_like(x, 0)
    q <- lax_div(x, y)
    sx <- lax_sign(x); sy <- lax_sign(y)
    adj <- lax_bitwise_and(lax_ne(sx, sy), lax_ne(lax_rem(x, y), zero))
    return(lax_select(adj, lax_sub(q, lax_full_like(q, 1)), q))
  }
  float_divmod(x, y)[[1L]]
}
# Port of jax/_src/numpy/ufuncs.py:_float_divmod (CPython float_divmod).
float_divmod <- function(x1, x2) {
  mod <- lax_rem(x1, x2)
  x1_corrected <- jnp_where(lax_eq(x2, lax_full_like(x2, 0)), x1, lax_sub(x1, mod))
  div <- lax_div(x1_corrected, x2)
  ind <- lax_bitwise_and(lax_ne(mod, lax_full_like(mod, 0)), lax_ne(lax_sign(x2), lax_sign(mod)))
  mod <- lax_select(ind, lax_add(mod, x2), mod)
  div <- lax_select(ind, lax_sub(div, lax_full_like(div, 1)), div)
  list(lax_round(div), mod)
}
#' @rdname jnp_ufuncs
#' @export
jnp_remainder <- function(x1, x2) {
  a <- promote_args_numeric(x1, x2)
  x <- a[[1L]]; y <- a[[2L]]
  zero <- lax_full_like(y, 0)
  dt <- aval_of(x)$dtype
  safe_y <- if (is_int_dtype(dt)) lax_select(lax_eq(y, zero), lax_full_like(y, 1), y) else y
  trunc_mod <- lax_rem(x, safe_y)
  trunc_mod_not_zero <- lax_ne(trunc_mod, lax_full_like(trunc_mod, 0))
  do_plus <- lax_bitwise_and(lax_ne(lax_lt(trunc_mod, lax_full_like(trunc_mod, 0)), lax_lt(y, zero)),
                             trunc_mod_not_zero)
  lax_select(do_plus, lax_add(trunc_mod, y), trunc_mod)
}
#' @rdname jnp_ufuncs
#' @export
jnp_mod <- jnp_remainder
#' @rdname jnp_ufuncs
#' @export
jnp_power <- function(x1, x2) {
  # Case 1 of jnp.power: a concrete integer scalar exponent uses integer_pow. R writes
  # integer exponents as plain numbers (x^2), so a whole valued R scalar counts as one;
  # so does a concrete rank 0 integer array.
  k <- if (!is_jax_value(x2) && is.numeric(x2) && length(x2) == 1L && is.null(dim(x2)) &&
           is.finite(x2) && x2 == round(x2)) x2
       else if (is_jax_array(x2) && !length(aval_of(x2)$shape) && is_int_dtype(aval_of(x2)$dtype))
         as.vector(array_host(x2))
       else NULL
  if (!is.null(k)) {
    a <- promote_dtypes_numeric(list(as_jax(x1)))[[1L]]
    # R doubles hold integers exactly up to 2^53 only.
    if (is_jax_array(x2) && abs(k) >= 2^53) return(pow_int_wide(a, x2, k < 0))
    return(lax_integer_pow(a, k))
  }
  x1 <- as_jax(x1)
  x2 <- as_jax(x2)
  a <- promote_args_numeric(x1, x2)
  if (is_int_dtype(aval_of(a[[1L]])$dtype)) return(pow_int_int(a[[1L]], a[[2L]]))
  # Case 3: an inexact base with an integer power keeps the dtype of the base.
  d1 <- aval_of(x1)$dtype
  if (is_inexact_dtype(d1) && is_int_dtype(aval_of(x2)$dtype))
    return(bind(pow_p, promote_shapes(list(x1, x2))))
  lax_pow(a[[1L]], a[[2L]])
}
# integer_pow for a concrete 64 bit exponent that an R double cannot hold exactly: the
# repeated squaring of its lowering, done on the device over all bits of the exponent.
pow_int_wide <- function(x, n, negative) {
  if (negative && !is_inexact_dtype(aval_of(x)$dtype))
    jax_error("Integers cannot be raised to negative powers, got integer_pow(",
              aval_long(aval_of(x)), ", ", format(as.vector(array_host(n)), scientific = FALSE), ")",
              class = "jax_type_error")
  # The magnitude of a negative exponent is formed unsigned, so that -2^63 has one.
  if (negative) pow_wide_neg(x, lax_neg(lax_convert_element_type(n, sub("^u?int", "uint", aval_of(n)$dtype))))
  else pow_wide_pos(x, n)
}

# x^m by repeated squaring over every bit of the integer array m (m >= 0).
pow_wide_loop <- function(x, m) {
  one <- lax_full_like(m, 1)
  zero <- lax_full_like(m, 0)
  acc <- lax_full_like(x, 1)
  for (i in seq_len(dtype_bits(aval_of(m)$dtype))) {
    acc <- lax_select(lax_ne(lax_bitwise_and(m, one), zero), lax_mul(acc, x), acc)
    x <- lax_mul(x, x)
    m <- lax_shift_right_logical(m, one)
  }
  acc
}

# The derivative is the rule of integer_pow, y * x^(y - 1) (the squares of the loop
# would give zero times infinity), with the exponent kept exact on the device. The
# functions are made on first use, since custom_jvp is defined in a later file.
pow_wide_fns <- local({
  fns <- NULL
  function() {
    if (!is.null(fns)) return(fns)
    pos <- custom_jvp(function(x, m) pow_wide_loop(x, m))
    pos <- custom_jvp_defjvp(pos, function(primals, tangents) {
      x <- primals[[1L]]
      m <- primals[[2L]]
      d <- lax_mul(lax_convert_element_type(m, aval_of(x)$dtype, weak_type = aval_of(x)$weak),
                   pos(x, lax_sub(m, lax_full_like(m, 1))))
      list(pos(x, m), lax_mul(tangents[[1L]], d))
    })
    neg <- custom_jvp(function(x, m) lax_div(lax_full_like(x, 1), pow_wide_loop(x, m)))
    neg <- custom_jvp_defjvp(neg, function(primals, tangents) {
      x <- primals[[1L]]
      m <- primals[[2L]]
      d <- lax_neg(lax_mul(lax_convert_element_type(m, aval_of(x)$dtype, weak_type = aval_of(x)$weak),
                           neg(x, lax_add(m, lax_full_like(m, 1)))))
      list(neg(x, m), lax_mul(tangents[[1L]], d))
    })
    fns <<- list(pos = pos, neg = neg)
    fns
  }
})
pow_wide_pos <- function(x, m) pow_wide_fns()$pos(x, m)
pow_wide_neg <- function(x, m) pow_wide_fns()$neg(x, m)
# Port of ufuncs._pow_int_int: binary exponentiation over the low 6 bits of the exponent
# (a larger exponent overflows every base above 1).
pow_int_int <- function(x1, x2) {
  sh <- broadcast_shapes_impl(list(aval_of(x1)$shape, aval_of(x2)$shape))
  x1 <- jnp_broadcast_to(x1, sh)
  x2 <- jnp_broadcast_to(x2, sh)
  # Strongly typed constants (ufuncs._constant_like), so the result is strongly typed.
  zero <- lax_full_like(x2, 0, dtype = aval_of(x2)$dtype)
  one <- lax_full_like(x2, 1, dtype = aval_of(x2)$dtype)
  # pow(0, x2) is zero for x2 != 0.
  acc <- lax_select(lax_bitwise_and(lax_eq(x1, zero), lax_ne(x2, zero)), zero, one)
  for (i in seq_len(6L)) {
    acc <- lax_select(lax_eq(lax_bitwise_and(x2, one), zero), acc, lax_mul(acc, x1))
    x1 <- lax_mul(x1, x1)
    x2 <- lax_shift_right_logical(x2, one)
  }
  acc
}
#' @rdname jnp_ufuncs
#' @export
jnp_negative <- function(x) lax_neg(as_jax(x))
#' @rdname jnp_ufuncs
#' @export
jnp_positive <- function(x) as_jax(x)
#' @rdname jnp_ufuncs
#' @export
jnp_abs <- function(x) {
  x <- as_jax(x)
  dt <- aval_of(x)$dtype
  if (is_unsigned_dtype(dt) || dt == "bool") x else lax_abs(x)
}
#' @rdname jnp_ufuncs
#' @export
jnp_absolute <- jnp_abs
#' @rdname jnp_ufuncs
#' @export
jnp_sign <- function(x) lax_sign(as_jax(x))
#' @rdname jnp_ufuncs
#' @export
jnp_exp <- jnp_unary(lax_exp)
#' @rdname jnp_ufuncs
#' @export
jnp_exp2 <- jnp_unary(lax_exp2)
#' @rdname jnp_ufuncs
#' @export
jnp_log <- jnp_unary(lax_log)
#' @rdname jnp_ufuncs
#' @export
jnp_log2 <- jnp_unary(function(x) lax_log2(x))
#' @rdname jnp_ufuncs
#' @export
jnp_log10 <- function(x) npa_log10(x)
#' @rdname jnp_ufuncs
#' @export
jnp_log1p <- jnp_unary(lax_log1p)
#' @rdname jnp_ufuncs
#' @export
jnp_expm1 <- jnp_unary(lax_expm1)
#' @rdname jnp_ufuncs
#' @export
jnp_sqrt <- jnp_unary(lax_sqrt)
#' @rdname jnp_ufuncs
#' @export
jnp_cbrt <- jnp_unary(lax_cbrt)
#' @rdname jnp_ufuncs
#' @export
jnp_square <- function(x) lax_square(promote_dtypes_numeric(list(x))[[1L]])
#' @rdname jnp_ufuncs
#' @export
jnp_reciprocal <- jnp_unary(function(x) lax_integer_pow(x, -1L))
#' @rdname jnp_ufuncs
#' @export
jnp_sin <- jnp_unary(lax_sin)
#' @rdname jnp_ufuncs
#' @export
jnp_cos <- jnp_unary(lax_cos)
#' @rdname jnp_ufuncs
#' @export
jnp_tan <- jnp_unary(lax_tan)
#' @rdname jnp_ufuncs
#' @export
jnp_arcsin <- jnp_unary(lax_asin)
#' @rdname jnp_ufuncs
#' @export
jnp_arccos <- jnp_unary(lax_acos)
#' @rdname jnp_ufuncs
#' @export
jnp_arctan <- jnp_unary(lax_atan)
#' @rdname jnp_ufuncs
#' @export
jnp_sinh <- jnp_unary(lax_sinh)
#' @rdname jnp_ufuncs
#' @export
jnp_cosh <- jnp_unary(lax_cosh)
#' @rdname jnp_ufuncs
#' @export
jnp_tanh <- jnp_unary(lax_tanh)
#' @rdname jnp_ufuncs
#' @export
jnp_arcsinh <- jnp_unary(lax_asinh)
#' @rdname jnp_ufuncs
#' @export
jnp_arccosh <- jnp_unary(lax_acosh)
#' @rdname jnp_ufuncs
#' @export
jnp_arctanh <- jnp_unary(lax_atanh)
#' @rdname jnp_ufuncs
#' @export
jnp_arctan2 <- function(x1, x2) { a <- promote_args_inexact(x1, x2); lax_atan2(a[[1L]], a[[2L]]) }
#' @rdname jnp_ufuncs
#' @export
jnp_floor <- function(x) {
  x <- as_jax(x)
  if (!is_inexact_dtype(aval_of(x)$dtype)) return(x)
  lax_floor(x)
}
#' @rdname jnp_ufuncs
#' @export
jnp_ceil <- function(x) {
  x <- as_jax(x)
  if (!is_inexact_dtype(aval_of(x)$dtype)) return(x)
  lax_ceil(x)
}
#' @rdname jnp_ufuncs
#' @export
jnp_trunc <- function(x) {
  x <- as_jax(x)
  if (!is_inexact_dtype(aval_of(x)$dtype)) return(x)
  lax_select(lax_lt(x, lax_full_like(x, 0)), lax_ceil(x), lax_floor(x))
}
#' @rdname jnp_ufuncs
#' @param decimals Number of decimals.
#' @export
jnp_round <- function(x, decimals = 0L, out = NULL) npa_round(x, decimals, out)
#' @rdname jnp_ufuncs
#' @export
# Port of jax/_src/numpy/ufuncs.py:rint.
jnp_rint <- function(x) {
  x <- as_jax(x)
  dt <- aval_of(x)$dtype
  if (dt == "bool" || is_int_dtype(dt)) return(lax_convert_element_type(x, default_float()))
  if (is_complex_dtype(dt)) return(lax_complex(jnp_rint(lax_real(x)), jnp_rint(lax_imag(x))))
  lax_round(x, 1L)
}
#' @rdname jnp_ufuncs
#' @export
jnp_maximum <- function(x1, x2) { a <- promote_args(x1, x2); lax_max(a[[1L]], a[[2L]]) }
#' @rdname jnp_ufuncs
#' @export
jnp_minimum <- function(x1, x2) { a <- promote_args(x1, x2); lax_min(a[[1L]], a[[2L]]) }
#' @rdname jnp_ufuncs
#' @export
jnp_equal <- function(x1, x2) { a <- promote_args(x1, x2); lax_eq(a[[1L]], a[[2L]]) }
#' @rdname jnp_ufuncs
#' @export
jnp_not_equal <- function(x1, x2) { a <- promote_args(x1, x2); lax_ne(a[[1L]], a[[2L]]) }
#' @rdname jnp_ufuncs
#' @export
jnp_less <- function(x1, x2) complex_comparison(lax_lt, x1, x2)
#' @rdname jnp_ufuncs
#' @export
jnp_less_equal <- function(x1, x2) complex_comparison(lax_le, x1, x2)
#' @rdname jnp_ufuncs
#' @export
jnp_greater <- function(x1, x2) complex_comparison(lax_gt, x1, x2)
#' @rdname jnp_ufuncs
#' @export
jnp_greater_equal <- function(x1, x2) complex_comparison(lax_ge, x1, x2)
#' @rdname jnp_ufuncs
#' @export
jnp_logical_and <- function(x1, x2) {
  a <- lapply(promote_args(x1, x2), to_bool)
  lax_bitwise_and(a[[1L]], a[[2L]])
}
#' @rdname jnp_ufuncs
#' @export
jnp_logical_or <- function(x1, x2) {
  a <- lapply(promote_args(x1, x2), to_bool)
  lax_bitwise_or(a[[1L]], a[[2L]])
}
#' @rdname jnp_ufuncs
#' @export
jnp_logical_xor <- function(x1, x2) {
  a <- lapply(promote_args(x1, x2), to_bool)
  lax_bitwise_xor(a[[1L]], a[[2L]])
}
#' @rdname jnp_ufuncs
#' @export
jnp_logical_not <- function(x) lax_bitwise_not(to_bool(as_jax(x)))
#' @rdname jnp_ufuncs
#' @export
jnp_bitwise_and <- function(x1, x2) { a <- promote_args(x1, x2); lax_bitwise_and(a[[1L]], a[[2L]]) }
#' @rdname jnp_ufuncs
#' @export
jnp_bitwise_or <- function(x1, x2) { a <- promote_args(x1, x2); lax_bitwise_or(a[[1L]], a[[2L]]) }
#' @rdname jnp_ufuncs
#' @export
jnp_bitwise_xor <- function(x1, x2) { a <- promote_args(x1, x2); lax_bitwise_xor(a[[1L]], a[[2L]]) }
#' @rdname jnp_ufuncs
#' @export
jnp_bitwise_not <- function(x) lax_bitwise_not(as_jax(x))
#' @rdname jnp_ufuncs
#' @export
jnp_invert <- jnp_bitwise_not
#' @rdname jnp_ufuncs
#' @export
jnp_isfinite <- function(x) {
  x <- as_jax(x)
  dt <- aval_of(x)$dtype
  if (is_float_dtype(dt)) return(lax_is_finite(x))
  if (is_complex_dtype(dt)) return(lax_bitwise_and(lax_is_finite(lax_real(x)), lax_is_finite(lax_imag(x))))
  lax_full_like(x, TRUE, dtype = "bool")
}
#' @rdname jnp_ufuncs
#' @export
jnp_isnan <- function(x) {
  x <- as_jax(x)
  lax_ne(x, x)
}
#' @rdname jnp_ufuncs
#' @export
jnp_isinf <- function(x) {
  x <- as_jax(x)
  dt <- aval_of(x)$dtype
  inf <- function(v) lax_eq(lax_abs(v), ufn_scalar(Inf, aval_of(v)$dtype))
  if (is_complex_dtype(dt)) return(lax_bitwise_or(inf(lax_real(x)), inf(lax_imag(x))))
  if (!is_float_dtype(dt)) return(lax_full_like(x, FALSE, dtype = "bool"))
  inf(x)
}
#' @rdname jnp_ufuncs
#' @export
jnp_real <- function(x) { x <- as_jax(x); if (is_complex_dtype(aval_of(x)$dtype)) lax_real(x) else x }
#' @rdname jnp_ufuncs
#' @export
jnp_imag <- function(x) {
  x <- as_jax(x)
  if (is_complex_dtype(aval_of(x)$dtype)) lax_imag(x) else lax_full_like(x, 0)
}
#' @rdname jnp_ufuncs
#' @export
jnp_conjugate <- function(x) { x <- as_jax(x); if (is_complex_dtype(aval_of(x)$dtype)) lax_conj(x) else x }
#' @rdname jnp_ufuncs
#' @export
jnp_conj <- jnp_conjugate

# ---- array creation ----------------------------------------------------------------------------

#' Convert to a jax array
#' @param a R data, array, or nested list of scalars.
#' @param dtype Optional dtype.
#' @param copy Ignored.
#' @export
jnp_asarray <- function(a, dtype = NULL, copy = NULL) {
  if (is.list(a) && is.null(oldClass(a))) {
    # An empty list is an empty float array, as np.array([]) is.
    if (!length(a)) return(jnp_zeros(0L, dtype %||% default_float()))
    parts <- lapply(a, jnp_asarray, dtype = dtype)
    out <- jnp_stack(parts)
    return(lax_convert_element_type(out, aval_of(out)$dtype))
  }
  as_jax(a, dtype)
}

#' @rdname jnp_asarray
#' @param ndmin Minimum rank of the result: unit dimensions are prepended as needed.
#' @export
jnp_array <- function(a, dtype = NULL, copy = TRUE, ndmin = 0L) {
  out <- jnp_asarray(a, dtype)
  s <- aval_of(out)$shape
  if (ndmin > length(s)) out <- lax_reshape(out, c(rep(1L, ndmin - length(s)), s))
  out
}

#' Arrays of zeros, ones, or a fill value
#' @param shape Integer vector.
#' @param dtype Dtype (default float).
#' @param fill_value Scalar.
#' @param a Array whose shape and dtype are used.
#' @name jnp_creation
NULL
#' @rdname jnp_creation
#' @export
jnp_zeros <- function(shape, dtype = NULL) lax_full(shape, 0, dtype %||% default_float())
#' @rdname jnp_creation
#' @export
jnp_ones <- function(shape, dtype = NULL) lax_full(shape, 1, dtype %||% default_float())
#' @rdname jnp_creation
#' @export
jnp_full <- function(shape, fill_value, dtype = NULL) {
  scalar <- if (is_jax_value(fill_value)) !length(aval_of(fill_value)$shape)
            else length(fill_value) == 1L && is.null(dim(fill_value))
  if (scalar) return(lax_full(shape, fill_value, dtype))
  jnp_broadcast_to(as_jax(fill_value, dtype), canon_shape(shape))
}
#' @rdname jnp_creation
#' @export
jnp_zeros_like <- function(a, dtype = NULL, shape = NULL) lax_full_like(a, 0, dtype, shape)
#' @rdname jnp_creation
#' @export
jnp_ones_like <- function(a, dtype = NULL, shape = NULL) lax_full_like(a, 1, dtype, shape)
#' @rdname jnp_creation
#' @export
jnp_full_like <- function(a, fill_value, dtype = NULL, shape = NULL) {
  scalar <- if (is_jax_value(fill_value)) !length(aval_of(fill_value)$shape)
            else length(fill_value) == 1L && is.null(dim(fill_value))
  if (scalar) return(lax_full_like(a, fill_value, dtype, shape))
  # An array fill value broadcasts to the target shape (and stays differentiable).
  # The fill array keeps its own dtype (an R integer vector is an int32 array) and is
  # then cast like an array, so that integers wrap instead of being range checked.
  av <- aval_of(a)
  dt <- if (is.null(dtype)) av$dtype else dtype(dtype)
  fv <- jnp_asarray(fill_value)
  if (aval_of(fv)$dtype != dt) fv <- lax_convert_element_type(fv, dt)
  jnp_broadcast_to(fv, if (is.null(shape)) av$shape else canon_shape(shape))
}

#' Evenly spaced values
#'
#' Port of `jnp.arange` (half-open interval, 0-based like NumPy). The dtype is the
#' promoted type of the arguments; a whole valued R number counts as an integer here,
#' as R has no integer literal without the `L` suffix (`jnp_arange(5)` is an integer
#' array, `jnp_arange(0.5, 3)` a float one). Pass `dtype` to choose.
#' @param start,stop,step Range specification; `jnp_arange(n)` gives `0..n-1`.
#' @param dtype Optional dtype.
#' @export
jnp_arange <- function(start, stop = NULL, step = NULL, dtype = NULL) {
  if (!is.null(dtype)) dtype <- dtype(dtype)
  args <- list(start, stop, step)
  names(args) <- c(if (is.null(stop) && is.null(step)) "stop" else "start", "stop", "step")
  present <- !vapply(args, is.null, NA)
  typs <- list()
  vals <- list(NULL, NULL, NULL)
  for (i in which(present)) {
    v <- args[[i]]
    nm <- names(args)[i]
    if (is_jax_value(v)) v <- concrete_or_error(v, paste0("It arose in the jnp.arange argument '", nm, "'"))
    scalar <- if (is_jax_value(v)) !length(aval_of(v)$shape) else length(v) == 1L && is.null(dim(v))
    if (!scalar)
      jax_error("jax.numpy.arange: arguments must be scalars; got ", nm, "=",
                paste(format(if (is_jax_value(v)) as.vector(array_host(v)) else v), collapse = " "),
                class = "jax_value_error")
    if (is.complex(v) || (is_jax_value(v) && is_complex_dtype(aval_of(v)$dtype)))
      jax_error("Passing complex start/stop/step to jnp.arange is no longer supported ",
                "starting in JAX v0.10.0.", class = "jax_value_error")
    typs[[length(typs) + 1L]] <-
      if (!is_jax_value(v) && is.double(v) && is.finite(v) && v == round(v))
        new_aval(integer(0), default_int(), TRUE)
      else aval_of(v)
    vals[[i]] <- if (is_jax_value(v)) as.numeric(as.vector(array_host(v))) else as.numeric(v)
    # A whole valued R number is an integer in the arithmetic with typed bounds too.
    if (!is_jax_value(v) && is.double(v) && is.finite(v) && v == round(v)) {
      if (abs(v) <= .Machine$integer.max) v <- as.integer(v)
      else if (default_int() == "int64" && abs(v) < 2^63)
        v <- ensure_compile_time_eval(lax_convert_element_type(as_jax(v, "int64"), "int64", weak_type = TRUE))
    }
    args[[i]] <- v
  }
  dt <- dtype %||% promote_types(vapply(typs, `[[`, "", "dtype"), vapply(typs, `[[`, NA, "weak"))$dtype
  conv <- function(v, d) lax_convert_element_type(as_jax(v), d)
  if (!present[2L]) {
    st <- 0
    start_arg <- 0L
    stop_arg <- args[[1L]]
    sp <- vals[[1L]]
  } else {
    st <- vals[[1L]]
    start_arg <- args[[1L]]
    stop_arg <- args[[2L]]
    sp <- vals[[2L]]
  }
  # The length comes from (stop - start) / step in the arithmetic of the arguments: R
  # numbers use doubles (like Python floats), arrays the precision of their dtype.
  typed <- any(vapply(args[present], is_jax_value, NA))
  host1 <- function(x) as.numeric(as.vector(array_host(x)))
  # The length is computed on concrete values also under jit (the bounds are static).
  span <- if (typed) ensure_compile_time_eval(jnp_subtract(stop_arg, start_arg)) else sp - st
  if (present[3L]) {
    # Start and step are converted before the multiplication, as in JAX.
    wd <- if (is_float_dtype(dt) && dtype_bits(dt) < 32L) "float32" else dt
    size <- max(0, ceiling(if (typed) host1(ensure_compile_time_eval(jnp_true_divide(span, args[[3L]])))
                           else span / vals[[3L]]))
    return(lax_convert_element_type(
      lax_add(conv(start_arg, wd), lax_mul(conv(args[[3L]], wd), lax_iota(wd, size))), dt))
  }
  if (st == 0) return(lax_iota(dt, max(0, ceiling(sp))))
  lax_add(conv(start_arg, dt), lax_iota(dt, max(0, ceiling(if (typed) host1(span) else span))))
}

#' Evenly spaced numbers over an interval
#'
#' Port of `jnp.linspace`.
#' @param start,stop Interval ends (scalars or arrays that broadcast).
#' @param num Number of samples.
#' @param endpoint Include `stop`.
#' @param retstep Also return the step: the result is then `list(samples, step)`.
#' @param dtype Optional dtype (default: the inexact promoted type of the ends).
#' @param axis 0-based axis of the samples in the result.
#' @export
jnp_linspace <- function(start, stop, num = 50L, endpoint = TRUE, retstep = FALSE, dtype = NULL,
                         axis = 0L) {
  npa_linspace(start, stop, num, endpoint, retstep, dtype, axis)
}

#' Identity matrix
#' @param N Rows.
#' @param M Columns (default N).
#' @param k Diagonal offset.
#' @param dtype Dtype.
#' @export
jnp_eye <- function(N, M = NULL, k = 0L, dtype = NULL) {
  M <- M %||% N
  dt <- dtype(dtype %||% default_float())
  r <- lax_broadcasted_iota("int32", c(N, M), 0L)
  c <- lax_broadcasted_iota("int32", c(N, M), 1L)
  lax_convert_element_type(lax_eq(lax_add(r, lax_full_like(r, as.integer(k))), c), dt)
}

#' @rdname jnp_eye
#' @param n Size.
#' @export
jnp_identity <- function(n, dtype = NULL) jnp_eye(n, dtype = dtype)

# ---- shape manipulation ------------------------------------------------------------------------

#' Shape helpers
#' @param a Array.
#' @param shape New shape (one entry may be -1).
#' @param axes 0-based permutation.
#' @param axis 0-based axis (or axes).
#' @param out Unsupported; must be `NULL`.
#' @param order Only `"C"` is supported.
#' @name jnp_shape
NULL

#' @rdname jnp_shape
#' @export
jnp_shape <- function(a) aval_of(a)$shape
#' @rdname jnp_shape
#' @export
jnp_ndim <- function(a) length(aval_of(a)$shape)
#' @rdname jnp_shape
#' @export
jnp_size <- function(a, axis = NULL) {
  shape <- aval_of(a)$shape
  if (is.null(axis)) return(prod_int(shape))
  prod_int(shape[norm_axis(axis, length(shape)) + 1L])
}

#' @rdname jnp_shape
#' @export
jnp_reshape <- function(a, shape, order = "C") {
  x <- as_jax(a)
  n <- prod_int(aval_of(x)$shape)
  if (is_jax_value(shape)) shape <- as.vector(concrete_or_error(shape, "the shape argument of jnp.reshape"))
  # list() is the empty tuple: the shape of a scalar.
  shape <- if (is.list(shape) && !length(shape)) integer(0) else unlist(shape)
  if (!is.numeric(shape) || anyNA(shape) || any(shape != round(shape)))
    jax_error("Shapes must be 1D sequences of concrete values of integer type, got ",
              paste(format(shape), collapse = ", "), ".", class = "jax_type_error")
  shape <- as.integer(shape)
  neg <- which(shape == -1L)
  if (length(neg) > 1L) jax_error("can only specify one unknown dimension")
  if (length(neg)) {
    known <- prod_int(shape[-neg])
    shape[neg] <- if (known == 0L) 0L else n %/% known
  }
  if (identical(order, "C")) return(lax_reshape(x, shape))
  if (identical(order, "F")) {
    # Column major: reshape the reversed layout, then reverse the axes back.
    r <- length(aval_of(x)$shape)
    out <- lax_reshape(x, rev(shape), rev(seq_len(r) - 1L))
    return(lax_transpose(out, rev(seq_along(shape) - 1L)))
  }
  if (identical(order, "A"))
    jax_error("np.reshape order=A is not implemented.", class = "jax_not_implemented_error")
  jax_error("Unexpected value for 'order' argument: ", order, ".", class = "jax_value_error")
}
#' @rdname jnp_shape
#' @export
jnp_ravel <- function(a, order = "C") {
  if (identical(order, "K"))
    jax_error("Ravel not implemented for order='K'.", class = "jax_not_implemented_error")
  jnp_reshape(a, -1L, order)
}
#' @rdname jnp_shape
#' @export
jnp_transpose <- function(a, axes = NULL) {
  x <- as_jax(a)
  r <- length(aval_of(x)$shape)
  axes <- if (is.null(axes)) rev(seq_len(r) - 1L) else norm_axis(axes, r)
  lax_transpose(x, axes)
}
#' @rdname jnp_shape
#' @param source,destination 0-based axes.
#' @export
jnp_moveaxis <- function(a, source, destination) {
  x <- as_jax(a)
  r <- length(aval_of(x)$shape)
  src <- norm_axis(source, r)
  dst <- norm_axis(destination, r)
  perm <- setdiff(seq_len(r) - 1L, src)
  ord <- order(dst)
  for (i in ord) perm <- append(perm, src[i], after = dst[i])
  lax_transpose(x, perm)
}
#' @rdname jnp_shape
#' @param axis1,axis2 Axes to swap.
#' @export
jnp_swapaxes <- function(a, axis1, axis2) {
  x <- as_jax(a)
  r <- length(aval_of(x)$shape)
  perm <- seq_len(r) - 1L
  i <- norm_axis(axis1, r); j <- norm_axis(axis2, r)
  perm[c(i, j) + 1L] <- perm[c(j, i) + 1L]
  lax_transpose(x, perm)
}
#' @rdname jnp_shape
#' @export
jnp_expand_dims <- function(a, axis) lax_expand_dims(as_jax(a), axis)
#' @rdname jnp_shape
#' @export
jnp_squeeze <- function(a, axis = NULL) {
  x <- as_jax(a)
  s <- aval_of(x)$shape
  if (is.null(axis)) axis <- which(s == 1L) - 1L
  lax_squeeze(x, axis)
}
#' @rdname jnp_shape
#' @export
jnp_broadcast_to <- function(a, shape) {
  x <- as_jax(a)
  s <- aval_of(x)$shape
  shape <- as.integer(shape)
  if (identical(s, shape)) return(x)
  bs <- broadcast_shapes_impl(list(s, shape))
  if (!identical(bs, shape))
    jax_error("Incompatible shapes for broadcasting: (", paste(s, collapse = ", "), ") and requested shape (",
              paste(shape, collapse = ", "), ")")
  nlead <- length(shape) - length(s)
  diff <- which(s != shape[nlead + seq_along(s)])
  x2 <- if (length(diff)) lax_squeeze(x, diff - 1L) else x
  keep <- setdiff(seq_along(s), diff)
  lax_broadcast_in_dim(x2, shape, nlead + keep - 1L)
}
#' @rdname jnp_shape
#' @param ... Arrays.
#' @export
jnp_broadcast_arrays <- function(...) {
  args <- lapply(list(...), as_jax)
  shape <- broadcast_shapes_impl(lapply(args, function(a) aval_of(a)$shape))
  lapply(args, jnp_broadcast_to, shape = shape)
}
#' @rdname jnp_shape
#' @param arrays List of arrays.
#' @export
jnp_concatenate <- function(arrays, axis = 0L, dtype = NULL) npa_concatenate(arrays, axis, dtype)
#' @rdname jnp_shape
#' @export
jnp_stack <- function(arrays, axis = 0L, out = NULL, dtype = NULL) {
  if (!length(arrays)) jax_error("Need at least one array to stack.", class = "jax_value_error")
  if (!is.null(out)) jax_error("The 'out' argument to jnp.stack is not supported.")
  # An array is the sequence of its slices along the first axis.
  if (!is.list(arrays)) arrays <- lax_unstack(as_jax(arrays), 0L)
  xs <- if (!is.null(dtype)) lapply(arrays, as_jax, dtype = dtype) else promote_dtypes(arrays)
  lax_stack(xs, axis)
}
#' @rdname jnp_shape
#' @param dtype Dtype.
#' @export
jnp_astype <- function(a, dtype) lax_convert_element_type(as_jax(a), dtype)

# ---- where / select ------------------------------------------------------------------------------

#' Select elements
#'
#' Three-argument form of `jnp.where`.
#' @param condition Boolean array.
#' @param x,y Values where condition is true / false.
#' @param size,fill_value One argument form only; see [jnp_nonzero()].
#' @export
jnp_where <- function(condition, x = NULL, y = NULL, size = NULL, fill_value = NULL) {
  if (is.null(x) && is.null(y)) return(jnp_nonzero(condition, size = size, fill_value = fill_value))
  if (!is.null(size) || !is.null(fill_value))
    jax_error("size and fill_value arguments cannot be used in three-term where function.",
              class = "jax_value_error")
  if (is.null(x) || is.null(y))
    jax_error("Either both or neither of the x and y arguments should be provided to jax.numpy.where.",
              class = "jax_value_error")
  c <- jnp_asarray(condition, dtype = "bool")
  v <- promote_dtypes(list(x, y))
  shape <- broadcast_shapes_impl(list(aval_of(c)$shape, aval_of(v[[1L]])$shape, aval_of(v[[2L]])$shape))
  c <- jnp_broadcast_to(c, shape)
  lax_select(c, jnp_broadcast_to(v[[1L]], shape), jnp_broadcast_to(v[[2L]], shape))
}

# ---- reductions ---------------------------------------------------------------------------------

# Sorted positional axes of a reduction (duplicates are an error, as in red_dims).
reduction_axes <- function(x, axis) sort(red_dims(x, axis))

keepdims_reshape <- function(out, x, axes) {
  s <- aval_of(x)$shape
  s[axes + 1L] <- 1L
  lax_reshape(out, s)
}

sum_dtype <- function(dt) {
  if (dt == "bool") return(default_int())
  if (is_unsigned_dtype(dt)) return(if (dtype_bits(dt) < dtype_bits(default_uint())) default_uint() else dt)
  if (is_signed_dtype(dt)) return(if (dtype_bits(dt) < dtype_bits(default_int())) default_int() else dt)
  dt
}

#' Reductions
#'
#' Ports of `jnp.sum`, `jnp.prod`, `jnp.max`, `jnp.min`, `jnp.mean`, `jnp.var`,
#' `jnp.std`, `jnp.all`, `jnp.any`, `jnp.argmax` and `jnp.argmin`. Axes are 0-based;
#' `NULL` reduces over all of them.
#' @param a Array.
#' @param axis Axis or axes.
#' @param dtype Optional result dtype (the accumulation dtype of `sum` and `prod`).
#' @param out Not supported (must be `NULL`).
#' @param keepdims Keep reduced dimensions as size one.
#' @param initial Optional scalar starting value of the reduction. `max` and `min`
#'   need it to reduce an empty array or to use `where`.
#' @param where Optional boolean mask of the elements to include.
#' @param promote_integers Whether integers narrower than the default integer are
#'   promoted to it (the NumPy rule).
#' @return An array.
#' @name jnp_reductions
NULL

#' @rdname jnp_reductions
#' @export
jnp_sum <- function(a, axis = NULL, dtype = NULL, out = NULL, keepdims = FALSE, initial = NULL,
                    where = NULL, promote_integers = TRUE) {
  red_sum(a, axis, dtype, out, keepdims, initial, where, promote_integers)
}
#' @rdname jnp_reductions
#' @export
jnp_prod <- function(a, axis = NULL, dtype = NULL, out = NULL, keepdims = FALSE, initial = NULL,
                     where = NULL, promote_integers = TRUE) {
  red_prod(a, axis, dtype, out, keepdims, initial, where, promote_integers)
}
#' @rdname jnp_reductions
#' @export
jnp_max <- function(a, axis = NULL, out = NULL, keepdims = FALSE, initial = NULL, where = NULL) {
  red_max(a, axis, out, keepdims, initial, where)
}
#' @rdname jnp_reductions
#' @export
jnp_min <- function(a, axis = NULL, out = NULL, keepdims = FALSE, initial = NULL, where = NULL) {
  red_min(a, axis, out, keepdims, initial, where)
}
#' @rdname jnp_reductions
#' @export
jnp_amax <- jnp_max
#' @rdname jnp_reductions
#' @export
jnp_amin <- jnp_min
#' @rdname jnp_reductions
#' @export
jnp_all <- function(a, axis = NULL, out = NULL, keepdims = FALSE, where = NULL) {
  red_all(a, axis, out, keepdims, where)
}
#' @rdname jnp_reductions
#' @export
jnp_any <- function(a, axis = NULL, out = NULL, keepdims = FALSE, where = NULL) {
  red_any(a, axis, out, keepdims, where)
}
#' @rdname jnp_reductions
#' @export
jnp_mean <- function(a, axis = NULL, dtype = NULL, out = NULL, keepdims = FALSE, where = NULL) {
  red_mean(a, axis, dtype, out, keepdims, where)
}
#' @rdname jnp_reductions
#' @param ddof Delta degrees of freedom: the divisor is `N - ddof`.
#' @param mean Optional precomputed mean (with the reduced dimensions kept).
#' @param correction Array API name of `ddof` (give one of the two).
#' @export
jnp_var <- function(a, axis = NULL, dtype = NULL, out = NULL, ddof = 0L, keepdims = FALSE,
                    where = NULL, mean = NULL, correction = NULL) {
  red_var(a, axis, dtype, out, ddof, keepdims, where, mean, correction)
}
#' @rdname jnp_reductions
#' @export
jnp_std <- function(a, axis = NULL, dtype = NULL, out = NULL, ddof = 0L, keepdims = FALSE,
                    where = NULL, mean = NULL, correction = NULL) {
  red_std(a, axis, dtype, out, ddof, keepdims, where, mean, correction)
}
#' @rdname jnp_reductions
#' @export
jnp_argmax <- function(a, axis = NULL, out = NULL, keepdims = FALSE) {
  red_check_out("argmax", out)
  red_arg_extreme(a, axis, isTRUE(keepdims), "argmax")
}
#' @rdname jnp_reductions
#' @export
jnp_argmin <- function(a, axis = NULL, out = NULL, keepdims = FALSE) {
  red_check_out("argmin", out)
  red_arg_extreme(a, axis, isTRUE(keepdims), "argmin")
}

# ---- linear algebra products --------------------------------------------------------------------

#' Matrix product with NumPy semantics
#' @param a,b Arrays.
#' @param precision Optional precision.
#' @param preferred_element_type Optional result dtype.
#' @export
jnp_matmul <- function(a, b, precision = NULL, preferred_element_type = NULL) {
  a <- as_jax(a); b <- as_jax(b)
  pref <- dot_preferred(a, b, preferred_element_type)
  as_ <- aval_of(a)$shape; bs <- aval_of(b)$shape
  for (i in 1:2) if (!length(list(as_, bs)[[i]]))
    jax_error("matmul input operand ", i - 1L, " must have ndim at least 1, but it has ndim 0",
              class = "jax_value_error")
  a_vec <- length(as_) == 1L
  b_vec <- length(bs) == 1L
  if (a_vec) a <- lax_expand_dims(a, 0L)
  if (b_vec) b <- lax_expand_dims(b, 1L)
  as_ <- aval_of(a)$shape; bs <- aval_of(b)$shape
  batch <- broadcast_shapes_impl(list(utils::head(as_, -2L), utils::head(bs, -2L)))
  nb <- length(batch)
  a <- jnp_broadcast_to(a, c(batch, utils::tail(as_, 2L)))
  b <- jnp_broadcast_to(b, c(batch, utils::tail(bs, 2L)))
  bd <- seq_len(nb) - 1L
  out <- dot_general_bind(a, b, list(list(nb + 1L, nb), list(bd, bd)), precision, pref$dtype)
  if (a_vec) out <- lax_squeeze(out, nb)
  if (b_vec) out <- lax_squeeze(out, length(aval_of(out)$shape) - 1L)
  lax_convert_element_type(out, pref$dtype, weak_type = pref$weak)
}

#' Dot product with NumPy semantics
#' @param a,b Arrays.
#' @param precision Optional precision.
#' @param preferred_element_type Optional result dtype.
#' @export
jnp_dot <- function(a, b, precision = NULL, preferred_element_type = NULL) {
  a <- as_jax(a); b <- as_jax(b)
  pref <- dot_preferred(a, b, preferred_element_type)
  ar <- length(aval_of(a)$shape); br <- length(aval_of(b)$shape)
  contract <- if (ar == 0L || br == 0L) list(integer(0), integer(0))
              else if (br == 1L) list(ar - 1L, 0L)
              else list(ar - 1L, br - 2L)
  out <- dot_general_bind(a, b, list(contract, list(integer(0), integer(0))), precision, pref$dtype)
  lax_convert_element_type(out, pref$dtype, weak_type = pref$weak)
}

# The result type of jnp.dot and jnp.matmul: the promoted type of the inputs, which is
# also the preferred element type of the product, or the type the caller asks for.
dot_preferred <- function(a, b, preferred_element_type) {
  if (!is.null(preferred_element_type)) return(list(dtype = dtype(preferred_element_type), weak = FALSE))
  aa <- aval_of(a); ba <- aval_of(b)
  promote_types(c(aa$dtype, ba$dtype), c(aa$weak, ba$weak))
}

#' Inner products
#' @param a,b Arrays.
#' @param precision,preferred_element_type See [lax_dot_general()].
#' @param out Unsupported; must be `NULL`.
#' @export
jnp_vdot <- function(a, b, precision = NULL, preferred_element_type = NULL) {
  x <- jnp_ravel(a); y <- jnp_ravel(b)
  if (is_complex_dtype(aval_of(x)$dtype)) x <- lax_conj(x)
  jnp_dot(x, y, precision = precision, preferred_element_type = preferred_element_type)
}

#' @rdname jnp_vdot
#' @export
jnp_inner <- function(a, b, precision = NULL, preferred_element_type = NULL) {
  ab <- promote_dtypes(list(a, b))
  ar <- length(aval_of(ab[[1L]])$shape); br <- length(aval_of(ab[[2L]])$shape)
  if (ar == 0L || br == 0L) {
    out <- jnp_multiply(ab[[1L]], ab[[2L]])
    return(if (is.null(preferred_element_type)) out else as_jax(out, preferred_element_type))
  }
  lax_dot_general(ab[[1L]], ab[[2L]], list(list(ar - 1L, br - 1L), list(integer(0), integer(0))),
                  precision, preferred_element_type)
}

#' @rdname jnp_vdot
#' @export
jnp_outer <- function(a, b, out = NULL) {
  if (!is.null(out)) jax_error("The 'out' argument to jnp.outer is not supported.")
  x <- jnp_ravel(a); y <- jnp_ravel(b)
  jnp_multiply(lax_expand_dims(x, 1L), y)
}
