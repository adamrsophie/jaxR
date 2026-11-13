# lax elementwise primitives (port of jax/_src/lax/lax.py, elementwise part).

# ---- shared rule builders -------------------------------------------------------------

kind_ok <- function(dt, kinds) {
  k <- dtype_kind(dt)
  switch(kinds,
    any = TRUE, float = k == "f", inexact = k %in% c("f", "c"), int = k %in% c("i", "u"),
    intbool = k %in% c("i", "u", "b"), number = k %in% c("i", "u", "f", "c"),
    real = k %in% c("i", "u", "f"), complex = k == "c", bool = k == "b", float_int = k %in% c("i", "u", "f"),
    ordered = k %in% c("i", "u", "f", "b"))
}

kinds_label <- c(any = "any", float = "floating", inexact = "inexact", int = "integer",
                 intbool = "integer or boolean", number = "numeric", real = "real",
                 complex = "complex", bool = "boolean", float_int = "real", ordered = "bool, floating, integer")

broadcasting_shape_rule <- function(name, avals) {
  shapes <- lapply(avals, `[[`, "shape")
  shapes <- shapes[vapply(shapes, length, 1L) > 0L]
  if (!length(shapes)) return(integer(0))
  ranks <- unique(vapply(shapes, length, 1L))
  if (length(ranks) != 1L)
    jax_error(name, ": arrays must have the same number of dimensions, got ",
              paste(sort(ranks), collapse = ", "), class = "jax_type_error")
  out <- integer(ranks)
  for (i in seq_len(ranks)) {
    ds <- vapply(shapes, `[`, 1L, i)
    non1 <- unique(ds[ds != 1L])
    if (length(non1) > 1L)
      jax_error(name, " got incompatible shapes for broadcasting: ",
                paste(vapply(shapes, function(s) paste0("(", paste(s, collapse = ", "), ")"), ""),
                      collapse = ", "), class = "jax_type_error")
    out[i] <- if (length(non1)) non1 else 1L
  }
  out
}

# Port of lax.check_same_dtypes.
check_same_dtypes <- function(name, avals) {
  dts <- vapply(avals, `[[`, "", "dtype")
  if (length(unique(dts)) > 1L)
    jax_error("lax.", name, " requires arguments to have the same dtypes, got ",
              paste(dts, collapse = ", "), ".", class = "jax_type_error")
  invisible()
}

naryop_abstract_eval <- function(name, kinds, out_dtype = NULL) {
  function(avals, params) {
    check_same_dtypes(name, avals)
    dts <- avals[[1L]]$dtype
    if (!kind_ok(dts, kinds))
      jax_error(name, " does not accept dtype ", dts, ". Accepted dtypes are subtypes of ",
                kinds_label[[kinds]], ".", class = "jax_type_error")
    shape <- broadcasting_shape_rule(name, avals)
    od <- if (is.function(out_dtype)) out_dtype(dts) else out_dtype %||% dts
    new_aval(shape, od, all(vapply(avals, `[[`, NA, "weak")))
  }
}

# Broadcast lowered operands to the output shape (lax size 1 / rank 0 broadcasting).
bcast_operands <- function(ctx, ins, avals_in, aval_out) {
  vapply(seq_along(ins), function(i) {
    a <- avals_in[[i]]
    if (identical(a$shape, aval_out$shape)) return(ins[[i]])
    target <- new_aval(aval_out$shape, a$dtype)
    dims <- if (length(a$shape)) seq_along(a$shape) - 1L else integer(0)
    hlo_broadcast(ctx, ins[[i]], a, target, dims)
  }, "")
}

lower_nary <- function(hlo, attrs = NULL) {
  function(ctx, ins, avals_in, avals_out, params) {
    ao <- avals_out[[1L]]
    ops <- bcast_operands(ctx, ins, avals_in, ao)
    types <- vapply(avals_in, function(a) tensor_type(new_aval(ao$shape, a$dtype)), "")
    emit_op(ctx, hlo, ops, types, tensor_type(ao), attrs = attrs)
  }
}

# Elementwise batching for lax broadcasting semantics.
# Port of jax/_src/interpreters/batching.py:broadcast_batcher.
lax_broadcast_batcher <- function(prim) {
  function(vals, dims, params, axis) {
    k <- which(!is.na(dims))[1L]
    shape <- aval_of(vals[[k]])$shape
    dim <- dims[k]
    agree <- all(vapply(seq_along(vals), function(i) {
      s <- aval_of(vals[[i]])$shape
      !length(s) || (identical(s, shape) && identical(dims[i], dim))
    }, NA))
    if (agree) {
      out <- bind(prim, vals, params)
      return(if (prim$multiple_results) list(out, rep(dim, length(out))) else list(out, dim))
    }
    vals <- Map(function(v, d) {
      if (!length(aval_of(v)$shape)) return(v)
      bdim_at_front(v, d, 1L)
    }, vals, dims)
    nd <- max(vapply(vals, function(v) length(aval_of(v)$shape), 1L))
    vals <- Map(function(v, d) {
      r <- length(aval_of(v)$shape)
      if (is.na(d) || r == nd) return(v)
      lax_expand_dims(v, seq(r, nd - 1L))
    }, vals, dims)
    out <- bind(prim, vals, params)
    if (prim$multiple_results) list(out, rep(0L, length(out))) else list(out, 0L)
  }
}

# Reduce a cotangent to the shape of an operand broadcast by lax size 1 broadcasting.
unbroadcast <- function(aval, ct) {
  if (is_zero(ct)) return(ct)
  cs <- aval_of(ct)$shape
  if (identical(cs, aval$shape)) return(ct)
  if (length(aval$shape) == 0L) return(lax_reduce_sum(ct, seq_along(cs) - 1L))
  dims <- which(aval$shape == 1L & cs != 1L) - 1L
  if (!length(dims)) return(ct)
  lax_reshape(lax_reduce_sum(ct, dims), aval$shape)
}

defjvp_zero <- function(prim) {
  prim$jvp <- function(primals, tangents, params) {
    ans <- bind(prim, primals, params)
    tz <- if (prim$multiple_results) lapply(ans, function(a) zero_of(tangent_aval(aval_of(a))))
          else zero_of(tangent_aval(aval_of(ans)))
    list(ans, tz)
  }
  invisible(prim)
}

def_unop <- function(name, hlo, kinds, out_dtype = NULL) {
  p <- new_primitive(name)
  p$abstract_eval <- naryop_abstract_eval(name, kinds, out_dtype)
  if (is.function(hlo)) p$lowering <- hlo else p$lowering <- lower_nary(hlo)
  p$batch <- lax_broadcast_batcher(p)
  p
}

def_binop <- function(name, hlo, kinds, out_dtype = NULL) def_unop(name, hlo, kinds, out_dtype)

# ---- operand helpers for R-level wrappers -------------------------------------------

# Convert arguments to jax values; weakly typed scalars take the dtype of the first
# strongly typed argument (lax functions require equal dtypes).
lax_args <- function(...) {
  args <- lapply(list(...), as_jax)
  avs <- lapply(args, aval_of)
  strong <- which(!vapply(avs, `[[`, NA, "weak"))
  if (length(strong)) {
    dt <- avs[[strong[1L]]]$dtype
    for (i in seq_along(args)) {
      if (avs[[i]]$weak && avs[[i]]$dtype != dt)
        args[[i]] <- lax_convert_element_type(args[[i]], dt, weak_type = TRUE)
    }
  }
  args
}

full_like_val <- function(x, v) lax_full_like(x, v)

# ---- unary primitives -------------------------------------------------------------------

neg_p <- def_unop("neg", "stablehlo.negate", "number")
# Port of lax._sign_lower_hlo: stablehlo.sign does not accept unsigned integers.
sign_p <- def_unop("sign", function(ctx, ins, ai, ao, p) {
  a <- ai[[1L]]
  t <- tensor_type(a)
  if (!is_unsigned_dtype(a$dtype)) return(emit_op(ctx, "stablehlo.sign", ins, t, t))
  z <- full_constant(ctx, 0, a)
  pred <- emit_op(ctx, "stablehlo.compare", c(ins, z), c(t, t), tensor_type(new_aval(a$shape, "bool")),
                  attrs = list(comparison_direction = "#stablehlo<comparison_direction EQ>",
                               compare_type = "#stablehlo<comparison_type UNSIGNED>"))
  emit_op(ctx, "stablehlo.select", c(pred, z, full_constant(ctx, 1, a)),
          c(tensor_type(new_aval(a$shape, "bool")), t, t), t)
}, "number")
floor_p <- def_unop("floor", "stablehlo.floor", "float")
ceil_p <- def_unop("ceil", "stablehlo.ceil", "float")
round_p <- def_unop("round", function(ctx, ins, ai, ao, p) {
  op <- if (identical(p$rounding_method, 1L)) "stablehlo.round_nearest_even" else "stablehlo.round_nearest_afz"
  emit_op(ctx, op, ins, tensor_type(ai[[1L]]), tensor_type(ao[[1L]]))
}, "float")
is_finite_p <- def_unop("is_finite", "stablehlo.is_finite", "float", out_dtype = "bool")
exp_p <- def_unop("exp", "stablehlo.exponential", "inexact")
exp2_p <- def_unop("exp2", NULL, "inexact")
log_p <- def_unop("log", "stablehlo.log", "inexact")
log1p_p <- def_unop("log1p", "stablehlo.log_plus_one", "inexact")
expm1_p <- def_unop("expm1", "stablehlo.exponential_minus_one", "inexact")
tanh_p <- def_unop("tanh", "stablehlo.tanh", "inexact")
logistic_p <- def_unop("logistic", "stablehlo.logistic", "inexact")
sin_p <- def_unop("sin", "stablehlo.sine", "inexact")
cos_p <- def_unop("cos", "stablehlo.cosine", "inexact")
tan_p <- def_unop("tan", "stablehlo.tan", "inexact")
asin_p <- def_unop("asin", "chlo.asin", "inexact")
acos_p <- def_unop("acos", "chlo.acos", "inexact")
atan_p <- def_unop("atan", "chlo.atan", "inexact")
sinh_p <- def_unop("sinh", "chlo.sinh", "inexact")
cosh_p <- def_unop("cosh", "chlo.cosh", "inexact")
asinh_p <- def_unop("asinh", "chlo.asinh", "inexact")
acosh_p <- def_unop("acosh", "chlo.acosh", "inexact")
atanh_p <- def_unop("atanh", "chlo.atanh", "inexact")
sqrt_p <- def_unop("sqrt", "stablehlo.sqrt", "inexact")
rsqrt_p <- def_unop("rsqrt", "stablehlo.rsqrt", "inexact")
cbrt_p <- def_unop("cbrt", "stablehlo.cbrt", "float")
abs_p <- def_unop("abs", "stablehlo.abs", "number",
                  out_dtype = function(dt) if (is_complex_dtype(dt)) complex_component(dt) else dt)
lgamma_p <- def_unop("lgamma", "chlo.lgamma", "float")
# The expansion of chlo.digamma in the plugin gives NaN at the pole at zero, where JAX
# gives -Inf (which the gradient of lgamma relies on). Negative integers stay NaN.
digamma_p <- def_unop("digamma", function(ctx, ins, ai, ao, p) {
  a <- ai[[1L]]
  t <- tensor_type(a)
  bt <- tensor_type(new_aval(a$shape, "bool"))
  y <- emit_op(ctx, "chlo.digamma", ins, t, t)
  is0 <- emit_op(ctx, "stablehlo.compare", c(ins[[1L]], full_constant(ctx, 0, a)), c(t, t), bt,
                 attrs = list(comparison_direction = "#stablehlo<comparison_direction EQ>",
                              compare_type = "#stablehlo<comparison_type FLOAT>"))
  emit_op(ctx, "stablehlo.select", c(is0, full_constant(ctx, -Inf, a), y), c(bt, t, t), t)
}, "float")
erf_p <- def_unop("erf", "chlo.erf", "float")
erfc_p <- def_unop("erfc", "chlo.erfc", "float")
erf_inv_p <- def_unop("erf_inv", "chlo.erf_inv", "float")
not_p <- def_unop("not", "stablehlo.not", "intbool")
population_count_p <- def_unop("population_count", "stablehlo.popcnt", "int")
clz_p <- def_unop("clz", "stablehlo.count_leading_zeros", "int")
square_p <- def_unop("square", function(ctx, ins, ai, ao, p) {
  t <- tensor_type(ai[[1L]])
  if (is_complex_dtype(ai[[1L]]$dtype)) return(emit_op(ctx, "chlo.square", ins, t, t))
  emit_op(ctx, "stablehlo.multiply", c(ins, ins), c(t, t), t)
}, "number")
real_p <- def_unop("real", "stablehlo.real", "complex", out_dtype = complex_component)
imag_p <- def_unop("imag", "stablehlo.imag", "complex", out_dtype = complex_component)
conj_p <- def_unop("conj", function(ctx, ins, ai, ao, p) {
  a <- ai[[1L]]
  ta <- tensor_type(a)
  if (!is_complex_dtype(a$dtype)) {
    return(emit_op(ctx, "stablehlo.complex", c(ins, full_constant(ctx, 0, a)), c(ta, ta),
                   tensor_type(ao[[1L]])))
  }
  ra <- new_aval(a$shape, complex_component(a$dtype))
  re <- emit_op(ctx, "stablehlo.real", ins, ta, tensor_type(ra))
  im <- emit_op(ctx, "stablehlo.imag", ins, ta, tensor_type(ra))
  nim <- emit_op(ctx, "stablehlo.negate", im, tensor_type(ra), tensor_type(ra))
  emit_op(ctx, "stablehlo.complex", c(re, nim), rep(tensor_type(ra), 2L), tensor_type(ao[[1L]]))
}, "number", out_dtype = complex_of)

exp2_p$lowering <- lower_fun(function(args, params) {
  x <- args[[1L]]
  lax_exp(lax_mul(lax_full_like(x, log(2)), x))
})

integer_pow_p <- def_unop("integer_pow", NULL, "number")
local({
  base_rule <- integer_pow_p$abstract_eval
  integer_pow_p$abstract_eval <- function(avals, params) {
    a <- avals[[1L]]
    if (is_int_dtype(a$dtype) && params$y < 0L)
      jax_error("Integers cannot be raised to negative powers, got integer_pow(", aval_str(a), ", ",
                params$y, ")", class = "jax_type_error")
    base_rule(avals, params)
  }
})
integer_pow_p$lowering <- lower_fun(function(args, params) {
  x <- args[[1L]]
  y <- params$y
  if (y == 0L) return(lax_full_like(x, 1))
  is_recip <- y < 0L
  y <- abs(y)
  acc <- NULL
  while (y > 0L) {
    # floor(y / 2) is exact for whole doubles (%% warns above 2^53).
    half <- floor(y / 2)
    if (y > 2 * half) acc <- if (is.null(acc)) x else lax_mul(acc, x)
    y <- half
    if (y > 0L) x <- lax_mul(x, x)
  }
  if (is_recip) lax_div(lax_full_like(acc, 1), acc) else acc
})

reduce_precision_p <- def_unop("reduce_precision", function(ctx, ins, ai, ao, p) {
  t <- tensor_type(ai[[1L]])
  emit_op(ctx, "stablehlo.reduce_precision", ins, t, t,
          attrs = list(exponent_bits = attr_i32(p$exponent_bits),
                       mantissa_bits = attr_i32(p$mantissa_bits)))
}, "float")

# ---- binary primitives -------------------------------------------------------------------

add_p <- def_binop("add", "stablehlo.add", "number")
sub_p <- def_binop("sub", "stablehlo.subtract", "number")
mul_p <- def_binop("mul", "stablehlo.multiply", "number")
div_p <- def_binop("div", "stablehlo.divide", "number")
rem_p <- def_binop("rem", "stablehlo.remainder", "real")
pow_p <- def_binop("pow", "stablehlo.power", "inexact")
# Port of lax._pow_dtype_rule: an inexact base takes an exponent of the same dtype or of
# any integer dtype (the integer exponent has its own derivative rule below).
pow_p$abstract_eval <- function(avals, params) {
  dx <- avals[[1L]]$dtype
  dy <- avals[[2L]]$dtype
  if (!is_inexact_dtype(dx))
    jax_error("pow does not accept dtype ", dx, " at position 0. Accepted dtypes at position 0 ",
              "are subtypes of complexfloating, floating.", class = "jax_type_error")
  if (!is_inexact_dtype(dy) && !is_int_dtype(dy))
    jax_error("pow does not accept dtype ", dy, " at position 1. Accepted dtypes at position 1 ",
              "are subtypes of complexfloating, floating, integer.", class = "jax_type_error")
  if (!is_int_dtype(dy) && dx != dy)
    jax_error("the first argument to pow must have an inexact dtype (float or complex), and the ",
              "second argument must have an inexact or integer dtype, and two inexact dtypes must ",
              "match, but got ", dx, " and ", dy, " respectively.", class = "jax_type_error")
  new_aval(broadcasting_shape_rule("pow", avals), dx, all(vapply(avals, `[[`, NA, "weak")))
}
# Port of lax._pow_lower: an integer exponent is converted to the dtype of the base.
pow_p$lowering <- function(ctx, ins, ai, ao, p) {
  if (ai[[1L]]$dtype != ai[[2L]]$dtype) {
    ya <- new_aval(ai[[2L]]$shape, ao[[1L]]$dtype)
    ins <- c(ins[[1L]], emit_op(ctx, "stablehlo.convert", ins[[2L]], tensor_type(ai[[2L]]),
                                tensor_type(ya)))
    ai[[2L]] <- ya
  }
  lower_nary("stablehlo.power")(ctx, ins, ai, ao, p)
}
# Port of mlir._minmax_hlo: complex values compare lexicographically as (real, imag)
# pairs (stablehlo.maximum and minimum have no complex ordering).
minmax_lowering <- function(hlo, dir) {
  function(ctx, ins, ai, ao, p) {
    o <- ao[[1L]]
    if (!is_complex_dtype(o$dtype)) return(lower_nary(hlo)(ctx, ins, ai, ao, p))
    ops <- bcast_operands(ctx, ins, ai, o)
    ct <- tensor_type(o)
    rt <- tensor_type(new_aval(o$shape, complex_component(o$dtype)))
    bt <- tensor_type(new_aval(o$shape, "bool"))
    part <- function(op, x) emit_op(ctx, op, x, ct, rt)
    cmp <- function(a, b, d) {
      emit_op(ctx, "stablehlo.compare", c(a, b), c(rt, rt), bt,
              attrs = list(comparison_direction = paste0("#stablehlo<comparison_direction ", d, ">"),
                           compare_type = "#stablehlo<comparison_type FLOAT>"))
    }
    rx <- part("stablehlo.real", ops[1L])
    ry <- part("stablehlo.real", ops[2L])
    real_eq <- cmp(rx, ry, "EQ")
    real_cmp <- cmp(rx, ry, dir)
    imag_cmp <- cmp(part("stablehlo.imag", ops[1L]), part("stablehlo.imag", ops[2L]), dir)
    which <- emit_op(ctx, "stablehlo.select", c(real_eq, imag_cmp, real_cmp), c(bt, bt, bt), bt)
    emit_op(ctx, "stablehlo.select", c(which, ops[1L], ops[2L]), c(bt, ct, ct), ct)
  }
}
max_p <- def_binop("max", minmax_lowering("stablehlo.maximum", "GT"), "any")
min_p <- def_binop("min", minmax_lowering("stablehlo.minimum", "LT"), "any")
atan2_p <- def_binop("atan2", "stablehlo.atan2", "inexact")
nextafter_p <- def_binop("nextafter", "chlo.next_after", "float")
and_p <- def_binop("and", "stablehlo.and", "intbool")
or_p <- def_binop("or", "stablehlo.or", "intbool")
xor_p <- def_binop("xor", "stablehlo.xor", "intbool")
shift_left_p <- def_binop("shift_left", "stablehlo.shift_left", "int")
shift_right_arithmetic_p <- def_binop("shift_right_arithmetic", "stablehlo.shift_right_arithmetic", "int")
shift_right_logical_p <- def_binop("shift_right_logical", "stablehlo.shift_right_logical", "int")
complex_p <- def_binop("complex", "stablehlo.complex", "float", out_dtype = complex_of)

compare_type <- function(dt) {
  k <- dtype_kind(dt)
  if (k %in% c("f", "c")) "FLOAT" else if (k == "i") "SIGNED" else "UNSIGNED"
}

def_compare <- function(name, dir, kinds = "any") {
  def_binop(name, function(ctx, ins, ai, ao, p) {
    a <- ai[[1L]]
    lower_nary("stablehlo.compare", list(
      comparison_direction = paste0("#stablehlo<comparison_direction ", dir, ">"),
      compare_type = paste0("#stablehlo<comparison_type ", compare_type(a$dtype), ">")))(
        ctx, ins, ai, ao, p)
  }, kinds, out_dtype = "bool")
}
eq_p <- def_compare("eq", "EQ")
ne_p <- def_compare("ne", "NE")
lt_p <- def_compare("lt", "LT", "ordered")
le_p <- def_compare("le", "LE", "ordered")
gt_p <- def_compare("gt", "GT", "ordered")
ge_p <- def_compare("ge", "GE", "ordered")

# add_any: tangent accumulation (ad_util.add_jaxvals_p).
add_any_p <- new_primitive("add_any")
add_any_p$abstract_eval <- function(avals, params) avals[[1L]]
add_any_p$lowering <- function(ctx, ins, ai, ao, p) {
  if (ai[[1L]]$dtype == "float0") return(ins[[1L]])
  lower_nary("stablehlo.add")(ctx, ins, ai, ao, p)
}
add_any_p$batch <- function(vals, dims, params, axis) {
  if (identical(dims[1L], dims[2L])) return(list(bind(add_any_p, vals), dims[1L]))
  vals <- Map(function(v, d) bdim_at_front(v, d, axis$size), vals, dims)
  list(bind(add_any_p, vals), 0L)
}

# ---- JVP and transpose rules -----------------------------------------------------------

defjvp_zero(floor_p); defjvp_zero(ceil_p); defjvp_zero(round_p)
defjvp_zero(is_finite_p)
# Port of lax._sign_jvp_rule. On the real line sign is piecewise constant. For complex
# x, sign(x) = exp(i theta) with theta = angle(x), so d sign(x) = i sign(x) dtheta with
# dtheta = Im(conj(sign(x)) dx) / |x| (zero at x == 0, where sign(x) == 0).
defjvp2(sign_p, function(g, ans, primals, params) {
  x <- primals[[1L]]
  if (!is_complex_dtype(aval_of(x)$dtype)) return(zero_of(tangent_aval(aval_of(ans))))
  dtheta <- lax_div(lax_imag(lax_mul(lax_conj(ans), g)), replace_zero(lax_abs(x)))
  lax_mul(ans, lax_complex(lax_full_like(dtheta, 0), dtheta))
})
# The tangent of the first operand passes through; the second has no tangent.
defjvp2(nextafter_p,
        function(g, ans, primals, params) broadcast_to_shape(g, aval_of(ans)$shape), NULL)
deflinear(reduce_precision_p,
          function(ct, args, params) list(bind(reduce_precision_p, list(ct), params)))
for (p in list(eq_p, ne_p, lt_p, le_p, gt_p, ge_p)) defjvp_zero(p)

deflinear(neg_p, function(ct, args, params) list(lax_neg(ct)))
deflinear(add_any_p, function(ct, args, params) list(ct, ct))
deflinear(real_p, function(ct, args, params) list(lax_complex(ct, lax_full_like(ct, 0))))
# Port of ad.deflinear2(imag_p, lambda t, _: [complex(zeros, neg(t))]).
deflinear(imag_p, function(ct, args, params) list(lax_complex(lax_full_like(ct, 0), lax_neg(ct))))
deflinear(conj_p, function(ct, args, params) {
  a <- args[[1L]]$aval
  if (is_complex_dtype(a$dtype)) list(lax_conj(ct)) else list(lax_real(ct))
})
deflinear(complex_p, function(ct, args, params) list(lax_real(ct), lax_neg(lax_imag(ct))))

arg_aval <- function(a) if (is_undefined(a)) a$aval else aval_of(a)

add_p$jvp <- function(primals, tangents, params) {
  ans <- bind(add_p, primals)
  ao <- aval_of(ans)
  ts <- Map(function(t, p) if (is_zero(t)) t else lax_broadcast_to_rank(t, ao), tangents, primals)
  list(ans, add_tangents(ts[[1L]], ts[[2L]]))
}
add_p$transpose <- function(ct, args, params) {
  list(unbroadcast(arg_aval(args[[1L]]), ct), unbroadcast(arg_aval(args[[2L]]), ct))
}
sub_p$jvp <- function(primals, tangents, params) {
  ans <- bind(sub_p, primals)
  ao <- aval_of(ans)
  t1 <- if (is_zero(tangents[[1L]])) tangents[[1L]] else lax_broadcast_to_rank(tangents[[1L]], ao)
  t2 <- if (is_zero(tangents[[2L]])) tangents[[2L]] else lax_neg(lax_broadcast_to_rank(tangents[[2L]], ao))
  list(ans, add_tangents(t1, t2))
}
sub_p$transpose <- function(ct, args, params) {
  list(unbroadcast(arg_aval(args[[1L]]), ct),
       if (is_zero(ct)) ct else unbroadcast(arg_aval(args[[2L]]), lax_neg(ct)))
}
mul_p$jvp <- function(primals, tangents, params) {
  x <- primals[[1L]]
  y <- primals[[2L]]
  ans <- bind(mul_p, primals)
  out <- add_tangents(
    if (is_zero(tangents[[1L]])) tangents[[1L]] else lax_mul(tangents[[1L]], y),
    if (is_zero(tangents[[2L]])) tangents[[2L]] else lax_mul(x, tangents[[2L]]))
  list(ans, out)
}
mul_p$transpose <- function(ct, args, params) {
  x <- args[[1L]]
  y <- args[[2L]]
  if (is_zero(ct)) return(list(ct, ct))
  if (is_undefined(x) && is_undefined(y))
    jax_error("Transpose rule for mul: both arguments are linear")
  if (is_undefined(x)) return(list(unbroadcast(x$aval, lax_mul(ct, y)), NULL))
  list(NULL, unbroadcast(y$aval, lax_mul(x, ct)))
}
div_p$jvp <- function(primals, tangents, params) {
  x <- primals[[1L]]
  y <- primals[[2L]]
  ans <- bind(div_p, primals)
  out <- add_tangents(
    if (is_zero(tangents[[1L]])) tangents[[1L]] else lax_div(tangents[[1L]], y),
    if (is_zero(tangents[[2L]])) tangents[[2L]] else
      lax_mul(lax_mul(lax_neg(tangents[[2L]]), x), lax_integer_pow(y, -2L)))
  list(ans, out)
}
div_p$transpose <- function(ct, args, params) {
  if (is_zero(ct)) return(list(ct, NULL))
  list(unbroadcast(args[[1L]]$aval, lax_div(ct, args[[2L]])), NULL)
}
rem_p$jvp <- function(primals, tangents, params) {
  x <- primals[[1L]]
  y <- primals[[2L]]
  ans <- bind(rem_p, primals)
  ao <- aval_of(ans)
  q <- lax_div(x, y)
  out <- add_tangents(
    if (is_zero(tangents[[1L]])) tangents[[1L]] else lax_broadcast_to_rank(tangents[[1L]], ao),
    if (is_zero(tangents[[2L]])) tangents[[2L]] else
      lax_mul(lax_neg(tangents[[2L]]), lax_mul(lax_sign(q), lax_floor(lax_abs(q)))))
  list(ans, out)
}

replace_zero <- function(x) lax_select(lax_eq(x, lax_full_like(x, 0)), lax_full_like(x, 1), x)

defjvp2(pow_p,
  function(g, ans, primals, params) {
    x <- primals[[1L]]
    y <- primals[[2L]]
    if (is_int_dtype(aval_of(y)$dtype)) {
      # Port of lax._pow_jvp_lhs for an integer exponent: the derivative is zero where
      # y == 0, so it is defined at x == 0.
      y <- lax_convert_element_type(y, aval_of(x)$dtype)
      sh <- broadcast_shapes_impl(list(aval_of(x)$shape, aval_of(y)$shape))
      x <- broadcast_to_shape(x, sh)
      y <- broadcast_to_shape(y, sh)
      zero <- lax_full_like(y, 0)
      jac <- lax_select(lax_eq(y, zero), zero,
                        lax_mul(replace_zero(y), lax_pow(x, lax_sub(y, lax_full_like(y, 1)))))
      return(lax_mul(g, jac))
    }
    jac <- lax_mul(y, lax_pow(x, lax_sub(y, lax_full_like(y, 1))))
    lax_mul(g, jac)
  },
  function(g, ans, primals, params) {
    x <- primals[[1L]]
    lax_mul(g, lax_mul(lax_log(replace_zero(x)), ans))
  })

balanced_eq <- function(x, z, y) {
  lax_div(lax_select(lax_eq(x, z), lax_full_like(z, 1), lax_full_like(z, 0)),
          lax_select(lax_eq(y, z), lax_full_like(z, 2), lax_full_like(z, 1)))
}
defjvp2(max_p,
  function(g, ans, primals, params) lax_mul(g, balanced_eq(primals[[1L]], ans, primals[[2L]])),
  function(g, ans, primals, params) lax_mul(g, balanced_eq(primals[[2L]], ans, primals[[1L]])))
defjvp2(min_p,
  function(g, ans, primals, params) lax_mul(g, balanced_eq(primals[[1L]], ans, primals[[2L]])),
  function(g, ans, primals, params) lax_mul(g, balanced_eq(primals[[2L]], ans, primals[[1L]])))
defjvp(atan2_p,
  function(g, primals, params) {
    x <- primals[[1L]]; y <- primals[[2L]]
    lax_mul(g, lax_div(y, lax_add(lax_square(x), lax_square(y))))
  },
  function(g, primals, params) {
    x <- primals[[1L]]; y <- primals[[2L]]
    lax_mul(g, lax_div(lax_neg(x), lax_add(lax_square(x), lax_square(y))))
  })

one <- function(x) lax_full_like(x, 1)
defjvp2(exp_p, function(g, ans, p, params) lax_mul(g, ans))
defjvp2(exp2_p, function(g, ans, p, params) lax_mul(lax_log(sp_const(p[[1L]], 2)), lax_mul(g, ans)))
defjvp(log_p, function(g, p, params) lax_div(g, p[[1L]]))
defjvp(log1p_p, function(g, p, params) lax_div(g, lax_add(p[[1L]], one(p[[1L]]))))
defjvp2(expm1_p, function(g, ans, p, params) lax_mul(g, bind(exp_p, p, params)))
defjvp2(tanh_p, function(g, ans, p, params) {
  x <- p[[1L]]
  if (identical(params$accuracy, "highest"))
    return(lax_mul(g, lax_mul(sp_const(x, 4),
                              lax_mul(bind(logistic_p, list(lax_mul(sp_const(x, 2), x)), params),
                                      bind(logistic_p, list(lax_mul(sp_const(x, -2), x)), params)))))
  lax_mul(g, lax_one_minus_square(ans))
})
defjvp2(logistic_p, function(g, ans, p, params) {
  if (identical(params$accuracy, "highest"))
    return(lax_mul(g, lax_mul(ans, lax_logistic(lax_neg(p[[1L]])))))
  lax_mul(g, lax_mul(ans, lax_sub(one(ans), ans)))
})
defjvp(sin_p, function(g, p, params) lax_mul(g, bind(cos_p, p, params)))
defjvp(cos_p, function(g, p, params) lax_neg(lax_mul(g, bind(sin_p, p, params))))
defjvp2(tan_p, function(g, ans, p, params) lax_mul(g, lax_add(one(ans), lax_square(ans))))
defjvp(asin_p, function(g, p, params) lax_mul(g, lax_rsqrt(lax_one_minus_square(p[[1L]]))))
defjvp(acos_p, function(g, p, params) lax_mul(g, lax_neg(lax_rsqrt(lax_one_minus_square(p[[1L]])))))
defjvp(atan_p, function(g, p, params) lax_div(g, lax_add(one(p[[1L]]), lax_square(p[[1L]]))))
defjvp(sinh_p, function(g, p, params) lax_mul(g, lax_cosh(p[[1L]])))
defjvp(cosh_p, function(g, p, params) lax_mul(g, lax_sinh(p[[1L]])))
# Port of lax._asinh_jvp: g * rsqrt(x^2 + 1) with the argument scaled by a power of two so
# that the square cannot overflow; the complex case keeps the signed zeros on the branch
# cuts and stays accurate near the branch points.
asinh_jvp <- function(g, x) {
  dt <- aval_of(x)$dtype
  maxexp <- jnp_finfo(dt)$maxexp
  inv_t <- 2^-(maxexp %/% 2L - 2L)
  max_scale <- 2^(maxexp %/% 2L + 2L)
  if (is_complex_dtype(dt)) {
    rx <- lax_real(x)
    ix <- lax_imag(x)
    max_abs <- lax_max(lax_abs(rx), lax_abs(ix))
    scale <- lax_stop_gradient(lax_clamp(sp_const(rx, 1), lax_mul(max_abs, sp_const(rx, inv_t)),
                                         sp_const(rx, max_scale)))
    inv_scale <- lax_reciprocal(scale)
    rx_s <- lax_mul(rx, inv_scale)
    ix_s <- lax_mul(ix, inv_scale)
    re_w <- lax_add(lax_square(rx_s), lax_one_minus_square(ix_s))
    im_w <- lax_mul(sp_const(rx_s, 2), lax_mul(rx_s, ix_s))
    r <- lax_rsqrt(lax_complex(re_w, im_w))
    deriv <- lax_select(lax_eq(max_abs, sp_const(rx, Inf)), lax_full_like(x, 0),
                        lax_complex(lax_mul(lax_real(r), inv_scale), lax_mul(lax_imag(r), inv_scale)))
  } else {
    scale <- lax_stop_gradient(lax_clamp(sp_const(x, 1), lax_mul(lax_abs(x), sp_const(x, inv_t)),
                                         sp_const(x, max_scale)))
    inv_scale <- lax_reciprocal(scale)
    x_scaled <- lax_mul(x, inv_scale)
    deriv <- lax_mul(inv_scale, lax_rsqrt(lax_add(lax_square(x_scaled), lax_square(inv_scale))))
  }
  lax_mul(g, deriv)
}
defjvp(asinh_p, function(g, p, params) asinh_jvp(g, p[[1L]]))
defjvp(acosh_p, function(g, p, params) {
  x <- p[[1L]]
  lax_mul(g, lax_mul(lax_rsqrt(lax_sub(x, one(x))), lax_rsqrt(lax_add(x, one(x)))))
})
defjvp(atanh_p, function(g, p, params) lax_div(g, lax_one_minus_square(p[[1L]])))
defjvp2(sqrt_p, function(g, ans, p, params) lax_mul(g, lax_div(lax_full_like(ans, 0.5), ans)))
defjvp2(rsqrt_p, function(g, ans, p, params) lax_mul(g, lax_mul(lax_full_like(ans, -0.5), lax_div(ans, p[[1L]]))))
defjvp2(cbrt_p, function(g, ans, p, params) lax_mul(g, lax_mul(lax_full_like(ans, 1 / 3), lax_integer_pow(ans, -2L))))
defjvp2(abs_p, function(g, ans, p, params) {
  x <- p[[1L]]
  if (is_complex_dtype(aval_of(x)$dtype)) {
    den <- lax_convert_element_type(replace_zero(ans), aval_of(x)$dtype)
    return(lax_real(lax_mul(g, lax_div(lax_conj(x), den))))
  }
  lax_select(lax_ge(x, lax_full_like(x, 0)), g, lax_neg(g))
})
defjvp(lgamma_p, function(g, p, params) lax_mul(g, lax_digamma(p[[1L]])))
defjvp(digamma_p, function(g, p, params) lax_mul(g, lax_polygamma(lax_full_like(p[[1L]], 1), p[[1L]])))
defjvp2(erf_inv_p, function(g, ans, p, params) {
  lax_mul(lax_full_like(ans, sqrt(pi) / 2), lax_mul(g, lax_exp(lax_square(ans))))
})
defjvp(square_p, function(g, p, params) lax_mul(g, lax_mul(lax_full_like(p[[1L]], 2), p[[1L]])))
defjvp(integer_pow_p, function(g, p, params) {
  y <- params$y
  x <- p[[1L]]
  if (y == 0L) return(lax_full_like(g, 0))
  if (y == 1L) return(g)
  if (y == 2L) return(lax_mul(g, lax_mul(sp_const(x, y), x)))
  # From 2^53 on, y - 1 is not exact in an R double (its parity would be lost):
  # x^(y - 1) is x^y / x there (0 at x = 0 for y > 1).
  if (abs(y) >= 2^53) {
    zero <- lax_full_like(x, 0)
    xm1 <- lax_select(lax_eq(x, zero), zero,
                      lax_div(lax_integer_pow(x, y), lax_select(lax_eq(x, zero), lax_full_like(x, 1), x)))
    return(lax_mul(g, lax_mul(sp_const(x, y), xm1)))
  }
  lax_mul(g, lax_mul(sp_const(x, y), lax_integer_pow(x, y - 1L)))
})

# ---- R-level lax functions -------------------------------------------------------------

unary <- function(prim) function(x) bind(prim, list(as_jax(x)))

# Port of lax.accuracy_attr for the unary functions that take `accuracy` in JAX.
accuracy_params <- function(accuracy) {
  if (is.null(accuracy)) return(list())
  if (inherits(accuracy, "jax_Tolerance")) return(list(accuracy = accuracy))
  mode <- if (is.character(accuracy) && length(accuracy) == 1L) tolower(accuracy)
          else if (identical(accuracy, lax_AccuracyMode$HIGHEST)) "highest"
          else if (identical(accuracy, lax_AccuracyMode$DEFAULT)) "default"
          else ""
  if (!mode %in% c("highest", "default"))
    jax_error("Accuracy ", paste(format(accuracy), collapse = " "), " not supported",
              class = "jax_not_implemented_error")
  list(accuracy = mode)
}
unary_acc <- function(prim) {
  function(x, accuracy = NULL) bind(prim, list(as_jax(x)), accuracy_params(accuracy))
}
binary <- function(prim) function(x, y) bind(prim, lax_args(x, y))

#' lax elementwise operations
#'
#' Ports of the elementwise functions of `jax.lax`. Arguments must have equal dtypes
#' (R scalars are weakly typed and adopt the dtype of the other argument) and shapes
#' that are equal, rank 0, or differ only in size one dimensions.
#' @param x,y Arrays.
#' @param accuracy Optional accuracy request of the functions that take one in JAX: a
#'   [lax_AccuracyMode] value, `"highest"` or `"default"`, or a [lax_Tolerance()]. It is
#'   a hint to the backend; on the CPU backend only `lax_erfc()` with the highest
#'   accuracy changes the computation.
#' @name lax_elementwise
#' @return An array.
NULL

#' @rdname lax_elementwise
#' @export
lax_neg <- unary(neg_p)
#' @rdname lax_elementwise
#' @export
lax_sign <- unary(sign_p)
#' @rdname lax_elementwise
#' @export
lax_floor <- unary(floor_p)
#' @rdname lax_elementwise
#' @export
lax_ceil <- unary(ceil_p)
#' @rdname lax_elementwise
#' @param rounding_method 0 (away from zero) or 1 (to nearest even).
#' @export
lax_round <- function(x, rounding_method = 0L) bind(round_p, list(as_jax(x)), list(rounding_method = as.integer(rounding_method)))
#' @rdname lax_elementwise
#' @export
lax_is_finite <- unary(is_finite_p)
#' @rdname lax_elementwise
#' @export
lax_exp <- unary_acc(exp_p)
#' @rdname lax_elementwise
#' @export
lax_exp2 <- unary_acc(exp2_p)
#' @rdname lax_elementwise
#' @export
lax_log <- unary_acc(log_p)
#' @rdname lax_elementwise
#' @export
lax_log1p <- unary_acc(log1p_p)
#' @rdname lax_elementwise
#' @export
lax_expm1 <- unary_acc(expm1_p)
#' @rdname lax_elementwise
#' @export
lax_tanh <- unary_acc(tanh_p)
#' @rdname lax_elementwise
#' @export
lax_logistic <- unary_acc(logistic_p)
#' @rdname lax_elementwise
#' @export
lax_sin <- unary_acc(sin_p)
#' @rdname lax_elementwise
#' @export
lax_cos <- unary_acc(cos_p)
#' @rdname lax_elementwise
#' @export
lax_tan <- unary_acc(tan_p)
#' @rdname lax_elementwise
#' @export
lax_asin <- unary(asin_p)
#' @rdname lax_elementwise
#' @export
lax_acos <- unary(acos_p)
#' @rdname lax_elementwise
#' @export
lax_atan <- unary(atan_p)
#' @rdname lax_elementwise
#' @export
lax_sinh <- unary(sinh_p)
#' @rdname lax_elementwise
#' @export
lax_cosh <- unary(cosh_p)
#' @rdname lax_elementwise
#' @export
lax_asinh <- unary(asinh_p)
#' @rdname lax_elementwise
#' @export
lax_acosh <- unary(acosh_p)
#' @rdname lax_elementwise
#' @export
lax_atanh <- unary(atanh_p)
#' @rdname lax_elementwise
#' @export
lax_sqrt <- unary_acc(sqrt_p)
#' @rdname lax_elementwise
#' @export
lax_rsqrt <- unary_acc(rsqrt_p)
#' @rdname lax_elementwise
#' @export
lax_cbrt <- unary_acc(cbrt_p)
#' @rdname lax_elementwise
#' @export
lax_abs <- unary(abs_p)
#' @rdname lax_elementwise
#' @export
lax_lgamma <- unary(lgamma_p)
#' @rdname lax_elementwise
#' @export
lax_digamma <- unary(digamma_p)
#' @rdname lax_elementwise
#' @export
lax_erf <- unary(erf_p)
#' @rdname lax_elementwise
#' @export
lax_erfc <- unary_acc(erfc_p)
#' @rdname lax_elementwise
#' @export
lax_erf_inv <- unary(erf_inv_p)
#' @rdname lax_elementwise
#' @export
lax_bitwise_not <- unary(not_p)
#' @rdname lax_elementwise
#' @export
lax_population_count <- unary(population_count_p)
#' @rdname lax_elementwise
#' @export
lax_clz <- unary(clz_p)
#' @rdname lax_elementwise
#' @export
lax_square <- unary(square_p)
#' @rdname lax_elementwise
#' @export
lax_real <- unary(real_p)
#' @rdname lax_elementwise
#' @export
lax_imag <- unary(imag_p)
#' @rdname lax_elementwise
#' @export
lax_conj <- unary(conj_p)
#' @rdname lax_elementwise
#' @export
lax_integer_pow <- function(x, y) {
  # An exponent beyond the range of R integers (a large unsigned value) stays a whole
  # double; the rules of the primitive only do arithmetic on it.
  bind(integer_pow_p, list(as_jax(x)),
       list(y = if (abs(y) <= .Machine$integer.max) as.integer(y) else trunc(y)))
}
#' @rdname lax_elementwise
#' @param exponent_bits,mantissa_bits Target precision.
#' @export
lax_reduce_precision <- function(x, exponent_bits, mantissa_bits) {
  bind(reduce_precision_p, list(as_jax(x)),
       list(exponent_bits = as.integer(exponent_bits), mantissa_bits = as.integer(mantissa_bits)))
}
#' @rdname lax_elementwise
#' @export
lax_add <- binary(add_p)
#' @rdname lax_elementwise
#' @export
lax_sub <- binary(sub_p)
#' @rdname lax_elementwise
#' @export
lax_mul <- binary(mul_p)
#' @rdname lax_elementwise
#' @export
lax_div <- binary(div_p)
#' @rdname lax_elementwise
#' @export
lax_rem <- binary(rem_p)
#' @rdname lax_elementwise
#' @export
lax_pow <- function(x, y) {
  x <- as_jax(x)
  y <- as_jax(y)
  # An integer exponent keeps its dtype (lax._pow_dtype_rule).
  if (is_inexact_dtype(aval_of(x)$dtype) && is_int_dtype(aval_of(y)$dtype))
    return(bind(pow_p, list(x, y)))
  bind(pow_p, lax_args(x, y))
}
#' @rdname lax_elementwise
#' @export
lax_max <- binary(max_p)
#' @rdname lax_elementwise
#' @export
lax_min <- binary(min_p)
#' @rdname lax_elementwise
#' @export
lax_atan2 <- binary(atan2_p)
#' @rdname lax_elementwise
#' @export
lax_nextafter <- binary(nextafter_p)
#' @rdname lax_elementwise
#' @export
lax_bitwise_and <- binary(and_p)
#' @rdname lax_elementwise
#' @export
lax_bitwise_or <- binary(or_p)
#' @rdname lax_elementwise
#' @export
lax_bitwise_xor <- binary(xor_p)
#' @rdname lax_elementwise
#' @export
lax_shift_left <- binary(shift_left_p)
#' @rdname lax_elementwise
#' @export
lax_shift_right_arithmetic <- binary(shift_right_arithmetic_p)
#' @rdname lax_elementwise
#' @export
lax_shift_right_logical <- binary(shift_right_logical_p)
#' @rdname lax_elementwise
#' @export
lax_complex <- binary(complex_p)
#' @rdname lax_elementwise
#' @export
lax_eq <- binary(eq_p)
#' @rdname lax_elementwise
#' @export
lax_ne <- binary(ne_p)
#' @rdname lax_elementwise
#' @export
lax_lt <- binary(lt_p)
#' @rdname lax_elementwise
#' @export
lax_le <- binary(le_p)
#' @rdname lax_elementwise
#' @export
lax_gt <- binary(gt_p)
#' @rdname lax_elementwise
#' @export
lax_ge <- binary(ge_p)
