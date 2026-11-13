# Remaining public jax.lax functions and the extras of jax.custom_derivatives and
# jax.ad_checkpoint (ports of jax/_src/lax/lax.py, ann.py, other.py, scaled_dot.py,
# control_flow/solves.py, control_flow/conditionals.py, parallel.py,
# jax/_src/custom_derivatives.py, jax/_src/ad_util.py and jax/_src/ad_checkpoint.py).
# Internal helpers of this file start with `lxe_` so they cannot collide with the
# helpers of other files.

# ---- enum like objects ------------------------------------------------------------------

#' Precision and algorithm settings of lax functions
#'
#' R counterparts of the enum like classes of `jax.lax`:
#' * `lax_Precision`: the names `"DEFAULT"`, `"HIGH"` and `"HIGHEST"` (port of
#'   `jax.lax.Precision`), accepted by the `precision` argument of [lax_dot_general()],
#'   [lax_dot()] and [lax_conv_general_dilated()].
#' * `lax_RoundingMethod`: `AWAY_FROM_ZERO = 0L`, `TO_NEAREST_EVEN = 1L`, the values of
#'   the `rounding_method` argument of [lax_round()].
#' * `lax_AccuracyMode`: `HIGHEST = 1L`, `DEFAULT = 2L` (port of `jax.lax.AccuracyMode`).
#' * `lax_DotAlgorithmPreset`: the named presets of `jax.lax.DotAlgorithmPreset`.
#' @format Named lists.
#' @name lax_enums
#' @examples
#' lax_Precision$HIGHEST
#' lax_RoundingMethod$TO_NEAREST_EVEN
NULL

#' @rdname lax_enums
#' @export
lax_Precision <- list(DEFAULT = "DEFAULT", HIGH = "HIGH", HIGHEST = "HIGHEST")

#' @rdname lax_enums
#' @export
lax_RoundingMethod <- list(AWAY_FROM_ZERO = 0L, TO_NEAREST_EVEN = 1L)

#' @rdname lax_enums
#' @export
lax_AccuracyMode <- list(HIGHEST = 1L, DEFAULT = 2L)

lxe_preset_names <- c("DEFAULT", "ANY_F8_ANY_F8_F32", "ANY_F8_ANY_F8_F32_FAST_ACCUM",
                      "ANY_F8_ANY_F8_ANY", "ANY_F8_ANY_F8_ANY_FAST_ACCUM", "F16_F16_F16",
                      "F16_F16_F32", "BF16_BF16_BF16", "BF16_BF16_F32", "BF16_BF16_F32_X3",
                      "BF16_BF16_F32_X6", "BF16_BF16_F32_X9", "TF32_TF32_F32",
                      "TF32_TF32_F32_X3", "F32_F32_F32", "F64_F64_F64")

#' @rdname lax_enums
#' @export
lax_DotAlgorithmPreset <- stats::setNames(
  lapply(lxe_preset_names, function(n) structure(n, class = "jax_DotAlgorithmPreset")),
  lxe_preset_names)

#' @export
print.jax_DotAlgorithmPreset <- function(x, ...) {
  cat("DotAlgorithmPreset.", unclass(x), "\n", sep = "")
  invisible(x)
}

#' Tolerance of a unary function
#'
#' Port of `jax.lax.Tolerance`: at most two of the tolerances may be set (`atol` and
#' `rtol`, or `atol` and `ulps`).
#' @param atol,rtol Nonnegative absolute and relative tolerances.
#' @param ulps Nonnegative tolerance in units in the last place.
#' @return An object of class `jax_Tolerance`.
#' @examples
#' lax_Tolerance(atol = 1e-6)
#' @export
lax_Tolerance <- function(atol = 0, rtol = 0, ulps = 0L) {
  if (atol < 0 || rtol < 0 || ulps < 0)
    jax_error("Tolerances must be non-negative.", class = "jax_value_error")
  if (atol == 0 && rtol == 0 && ulps == 0)
    jax_error("At least one of atol, rtol, or ulps must be set.", class = "jax_value_error")
  structure(list(atol = atol, rtol = rtol, ulps = as.integer(ulps)), class = "jax_Tolerance")
}

#' Dot product algorithm
#'
#' Port of `jax.lax.DotAlgorithm`: the storage types the operands are rounded to, the
#' accumulation type and the decomposition of the products.
#' @param lhs_precision_type,rhs_precision_type,accumulation_type Dtypes.
#' @param lhs_component_count,rhs_component_count,num_primitive_operations Counts of the
#'   decomposition (1 for algorithms without one).
#' @param allow_imprecise_accumulation Whether some steps may accumulate in lower
#'   precision.
#' @return An object of class `jax_DotAlgorithm`.
#' @examples
#' lax_DotAlgorithm("float32", "float32", "float32")
#' @export
lax_DotAlgorithm <- function(lhs_precision_type, rhs_precision_type, accumulation_type,
                             lhs_component_count = 1L, rhs_component_count = 1L,
                             num_primitive_operations = 1L,
                             allow_imprecise_accumulation = FALSE) {
  structure(list(lhs_precision_type = dtype(lhs_precision_type),
                 rhs_precision_type = dtype(rhs_precision_type),
                 accumulation_type = dtype(accumulation_type),
                 lhs_component_count = as.integer(lhs_component_count),
                 rhs_component_count = as.integer(rhs_component_count),
                 num_primitive_operations = as.integer(num_primitive_operations),
                 allow_imprecise_accumulation = isTRUE(allow_imprecise_accumulation)),
            class = "jax_DotAlgorithm")
}

#' Canonical precision specification
#'
#' Port of `jax.lax.PrecisionLike` (a union type in JAX) as the function that
#' canonicalizes such a value (`lax.canonicalize_precision`): `NULL`, a precision name
#' or a pair of names, a [lax_DotAlgorithm()] or a preset of [lax_DotAlgorithmPreset]
#' (or its name).
#' @param precision A precision like value.
#' @return `NULL`, a pair of names from [lax_Precision], a `jax_DotAlgorithm` or a
#'   `jax_DotAlgorithmPreset`.
#' @examples
#' lax_PrecisionLike("highest")
#' lax_PrecisionLike(c("default", "high"))
#' @export
lax_PrecisionLike <- function(precision = NULL) {
  strings <- c(highest = "HIGHEST", float32 = "HIGHEST", high = "HIGH", bfloat16_3x = "HIGH",
               tensorfloat32 = "HIGH", default = "DEFAULT", bfloat16 = "DEFAULT",
               fastest = "DEFAULT")
  bad <- function() {
    jax_error("Precision argument must be one of:\n- None,\n- a string in [",
              paste0("'", names(strings), "'", collapse = ", "), "],\n",
              "- a lax.Precision value,\n- a tuple of two lax.Precision values or strings,\n",
              "- a lax.DotAlgorithmPreset or the name of one of these presets, or\n",
              "- a lax.DotAlgorithm value;\nbut got ",
              paste(format(unclass(precision)), collapse = ", "), ".", class = "jax_value_error")
  }
  if (is.null(precision)) {
    p <- .jx$config$jax_default_matmul_precision
    return(if (is.null(p)) NULL else lax_PrecisionLike(p))
  }
  if (inherits(precision, c("jax_DotAlgorithm", "jax_DotAlgorithmPreset"))) return(precision)
  if (is.list(precision)) precision <- unlist(precision)
  if (!is.character(precision) || !length(precision) %in% 1:2 || anyNA(precision)) bad()
  one <- function(s) {
    if (s %in% names(strings)) return(strings[[s]])
    if (s %in% unlist(lax_Precision)) return(s)
    NULL
  }
  if (length(precision) == 1L) {
    p <- one(precision)
    if (!is.null(p)) return(c(p, p))
    if (precision %in% lxe_preset_names) return(lax_DotAlgorithmPreset[[precision]])
    bad()
  }
  ps <- lapply(precision, one)
  if (any(vapply(ps, is.null, NA))) bad()
  unlist(ps)
}

#' Dimension numbers of dot products and convolutions
#'
#' Constructors for the dimension number arguments of [lax_dot_general()]
#' (`jax.lax.DotDimensionNumbers`), [lax_conv_general_dilated()]
#' (`jax.lax.ConvGeneralDilatedDimensionNumbers`) and [lax_ragged_dot_general()]
#' (`jax.lax.RaggedDotDimensionNumbers`). Dimensions are 0-based.
#' @param lhs_contracting,rhs_contracting Contracting dimensions.
#' @param lhs_batch,rhs_batch Batch dimensions.
#' @return `lax_DotDimensionNumbers()` returns
#'   `list(list(lhs_contracting, rhs_contracting), list(lhs_batch, rhs_batch))`.
#' @name lax_dimension_numbers
#' @examples
#' lax_DotDimensionNumbers(1L, 0L)
#' lax_ConvGeneralDilatedDimensionNumbers("NHWC", "HWIO", "NHWC")
#' lax_RaggedDotDimensionNumbers(list(list(1L, 1L), list(integer(0), integer(0))), 0L, 0L)
NULL

#' @rdname lax_dimension_numbers
#' @export
lax_DotDimensionNumbers <- function(lhs_contracting, rhs_contracting, lhs_batch = integer(0),
                                    rhs_batch = integer(0)) {
  list(list(as.integer(lhs_contracting), as.integer(rhs_contracting)),
       list(as.integer(lhs_batch), as.integer(rhs_batch)))
}

#' @rdname lax_dimension_numbers
#' @param lhs_spec,rhs_spec,out_spec Layout strings (such as `"NCHW"`, `"OIHW"`) or
#'   0-based integer vectors.
#' @return `lax_ConvGeneralDilatedDimensionNumbers()` returns the three layout strings
#'   (or, for integer vectors, a [lax_ConvDimensionNumbers()] object).
#' @export
lax_ConvGeneralDilatedDimensionNumbers <- function(lhs_spec, rhs_spec, out_spec) {
  if (is.character(lhs_spec) && is.character(rhs_spec) && is.character(out_spec))
    return(c(lhs_spec, rhs_spec, out_spec))
  lax_ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)
}

#' @rdname lax_dimension_numbers
#' @param dot_dimension_numbers Dot dimension numbers as for [lax_dot_general()].
#' @param lhs_ragged_dimensions The lhs ragged dimension (one value).
#' @param rhs_group_dimensions The rhs group dimensions (zero or one value).
#' @return `lax_RaggedDotDimensionNumbers()` returns an object of class
#'   `jax_RaggedDotDimensionNumbers`.
#' @export
lax_RaggedDotDimensionNumbers <- function(dot_dimension_numbers, lhs_ragged_dimensions,
                                          rhs_group_dimensions) {
  dn <- dot_dimension_numbers
  structure(list(lhs_contracting = as.integer(dn[[1L]][[1L]]),
                 rhs_contracting = as.integer(dn[[1L]][[2L]]),
                 lhs_batch = as.integer(dn[[2L]][[1L]] %||% integer(0)),
                 rhs_batch = as.integer(dn[[2L]][[2L]] %||% integer(0)),
                 lhs_ragged_dimensions = as.integer(lhs_ragged_dimensions),
                 rhs_group_dimensions = as.integer(rhs_group_dimensions)),
            class = "jax_RaggedDotDimensionNumbers")
}

# ---- small functions ------------------------------------------------------------------------

#' Broadcast shapes (lax version)
#'
#' Port of `jax.lax.broadcast_shapes`: the shape that results from NumPy broadcasting
#' of the given shapes.
#' @param ... Integer shape vectors (one or more).
#' @return An integer vector.
#' @examples
#' lax_broadcast_shapes(c(3L, 1L), 4L)
#' @export
lax_broadcast_shapes <- function(...) {
  shapes <- lapply(list(...), as.integer)
  if (!length(shapes)) jax_error("broadcast_shapes requires at least one shape", class = "jax_value_error")
  for (s in shapes) {
    if (any(s < 0L))
      jax_error("Only non-negative indices are allowed when broadcasting static shapes, ",
                "but got shape ", fmt_tuple(s), ".", class = "jax_type_error")
  }
  if (length(shapes) == 1L) return(shapes[[1L]])
  nd <- max(vapply(shapes, length, 1L))
  ps <- lapply(shapes, function(s) c(rep(1L, nd - length(s)), s))
  out <- integer(nd)
  for (i in seq_len(nd)) {
    d <- vapply(ps, `[`, 1L, i)
    non1 <- unique(d[d != 1L])
    if (length(non1) > 1L)
      jax_error("Incompatible shapes for broadcasting: shapes=[",
                paste(vapply(shapes, fmt_tuple, ""), collapse = ", "), "]",
                class = "jax_value_error")
    out[i] <- if (length(non1)) non1 else 1L
  }
  out
}

#' Add leading dimensions of size one
#'
#' Port of `jax.lax.broadcast_to_rank`: adds leading dimensions of size one to give `x`
#' rank `rank` (unchanged when it already has at least that rank).
#' @param x Array.
#' @param rank Target rank.
#' @return An array.
#' @examples
#' \dontrun{
#' dim(lax_broadcast_to_rank(c(1, 2, 3), 3L))
#' }
#' @export
lax_broadcast_to_rank <- function(x, rank) {
  # Internal callers of R/lax.R pass an aval (broadcast to its shape); this branch goes
  # once they use the renamed internal helper (see the proposed diff for R/lax_shape.R).
  if (inherits(rank, "jax_aval")) return(broadcast_to_shape(x, rank$shape))
  x <- as_jax(x)
  nd <- length(aval_of(x)$shape)
  rank <- as.integer(rank)
  if (nd >= rank) return(x)
  lax_broadcast(x, rep(1L, rank - nd))
}

#' Collapse dimensions into one
#'
#' Port of `jax.lax.collapse`: dimensions `[start_dimension, stop_dimension)` are
#' raveled (row major) into a single dimension.
#' @param operand Array.
#' @param start_dimension First dimension collapsed (0-based, inclusive).
#' @param stop_dimension End of the collapsed range (exclusive); `NULL` collapses all
#'   dimensions after `start_dimension`.
#' @return An array.
#' @examples
#' \dontrun{
#' dim(lax_collapse(array(0, c(2, 3, 4)), 0L, 2L))
#' }
#' @export
lax_collapse <- function(operand, start_dimension, stop_dimension = NULL) {
  x <- as_jax(operand)
  s <- aval_of(x)$shape
  n <- length(s)
  # Port of Python's slice(start, stop).indices(n) for step 1.
  idx <- function(v, default) {
    if (is.null(v)) return(default)
    v <- as.integer(v)
    if (v < 0L) v <- max(v + n, 0L)
    min(v, n)
  }
  lo <- idx(start_dimension, 0L)
  hi <- idx(stop_dimension, n)
  if (hi < lo)
    jax_error("Invalid dimension range passed to collapse: ", fmt_tuple(s), "[", start_dimension,
              ":", if (is.null(stop_dimension)) "None" else stop_dimension, "]",
              class = "jax_value_error")
  size <- prod_int(s[seq_len(hi - lo) + lo])
  lax_reshape(x, c(s[seq_len(lo)], size, drop_first(s, hi)))
}

#' Batched matrix multiplication
#'
#' Port of `jax.lax.batch_matmul`: operands of equal rank (at least 2), multiplied over
#' their last two dimensions with the leading dimensions as batch dimensions.
#' @param lhs,rhs Arrays.
#' @param precision Optional precision (see [lax_Precision]).
#' @return An array.
#' @examples
#' \dontrun{
#' lax_batch_matmul(array(1, c(2, 3, 4)), array(1, c(2, 4, 5)))
#' }
#' @export
lax_batch_matmul <- function(lhs, rhs, precision = NULL) {
  l <- as_jax(lhs)
  r <- as_jax(rhs)
  ln <- length(aval_of(l)$shape)
  rn <- length(aval_of(r)$shape)
  if (min(ln, rn) < 2L)
    jax_error("Arguments to batch_matmul must be at least 2D, got ", ln, ", ", rn,
              class = "jax_value_error")
  if (ln != rn)
    jax_error("Arguments to batch_matmul must have same ndim, got ", ln, ", ", rn,
              class = "jax_value_error")
  batch <- seq_len(ln - 2L) - 1L
  lax_dot_general(l, r, list(list(ln - 1L, rn - 2L), list(batch, batch)), precision = precision)
}

#' Elementwise reciprocal
#'
#' Port of `jax.lax.reciprocal` (`integer_pow(x, -1)`).
#' @param x Array.
#' @return An array.
#' @examples
#' \dontrun{
#' lax_reciprocal(c(1, 2, 4))
#' }
#' @export
lax_reciprocal <- function(x) lax_integer_pow(x, -1L)

#' Dtype of a value
#'
#' Port of `jax.lax.dtype` (`jax._src.dtypes.dtype`): jax arrays keep their dtype; R
#' scalars, R vectors and dtype names are canonicalized (64-bit types map to 32-bit
#' ones unless x64 is enabled).
#' @param x A jax array, R data or a dtype name.
#' @return A dtype name.
#' @examples
#' lax_dtype(1.5)
#' lax_dtype("float64")
#' @export
lax_dtype <- function(x) {
  if (is.null(x)) jax_error("Invalid argument to dtype: None.", class = "jax_value_error")
  if (inherits(x, "jax_value")) return(aval_of(x)$dtype)
  if (inherits(x, "jax_aval")) return(x$dtype)
  if (is.character(x) && length(x) == 1L) {
    out <- tryCatch(dtype(x), jax_error = function(e) NULL)
    if (is.null(out) || out == "float0")
      jax_error("Value '", x, "' with dtype ", x, " is not a valid JAX array type. Only ",
                "arrays of numeric types are supported by JAX.", class = "jax_type_error")
    return(out)
  }
  if (is.logical(x)) return("bool")
  if (is.integer(x)) return(if (length(x) == 1L && is.null(dim(x))) default_int() else "int32")
  if (is.double(x)) return(default_float())
  if (is.complex(x)) return(default_complex())
  jax_error("Cannot determine dtype of ", class(x)[1L], class = "jax_type_error")
}

#' Shape as an array
#'
#' Port of `jax.lax.shape_as_value` for static shapes: an integer array holding the
#' shape (`int32`, or `int64` when a dimension does not fit in 32 bits).
#' @param shape Integer vector.
#' @return An integer array of length `length(shape)`.
#' @examples
#' \dontrun{
#' lax_shape_as_value(c(2L, 3L))
#' }
#' @export
lax_shape_as_value <- function(shape) {
  shape <- as.numeric(shape)
  dt <- if (any(shape > .Machine$integer.max)) "int64" else "int32"
  if (!length(shape)) return(lax_full(0L, 0, dt))
  # Built directly: the int64 of a large dimension is kept without 64 bit mode.
  new_array(new_aval(length(shape), dt), host = array(shape, dim = length(shape)))
}

# ---- elementwise primitives ------------------------------------------------------------------

# Port of lax._log2_lower: log(x) times 1/ln(2) (real and imaginary parts for complex).
log2_p <- def_unop("log2", function(ctx, ins, ai, ao, p) {
  a <- ai[[1L]]
  t <- tensor_type(a)
  lx <- emit_op(ctx, "stablehlo.log", ins, t, t)
  if (is_complex_dtype(a$dtype)) {
    ra <- new_aval(a$shape, complex_component(a$dtype))
    rt <- tensor_type(ra)
    k <- full_constant(ctx, 1.4426950408889634, ra)
    re <- emit_op(ctx, "stablehlo.multiply", c(emit_op(ctx, "stablehlo.real", lx, t, rt), k),
                  c(rt, rt), rt)
    im <- emit_op(ctx, "stablehlo.multiply", c(emit_op(ctx, "stablehlo.imag", lx, t, rt), k),
                  c(rt, rt), rt)
    return(emit_op(ctx, "stablehlo.complex", c(re, im), c(rt, rt), t))
  }
  emit_op(ctx, "stablehlo.multiply", c(lx, full_constant(ctx, 1.4426950408889634, a)),
          c(t, t), t)
}, "inexact")
defjvp(log2_p, function(g, p, params) {
  x <- p[[1L]]
  lax_div(g, lax_mul(x, lax_log(lax_full_like(x, 2))))
})

# Port of lax.mulhi_p (naryop over integers lowered to chlo.mulhi, zero derivative).
mulhi_p <- def_binop("mulhi", "chlo.mulhi", "int")
defjvp_zero(mulhi_p)

# Port of lax.one_minus_square_p: (1 + x) (1 - x) with the JVP -2 x g.
one_minus_square_p <- def_unop("one_minus_square", lower_fun(function(args, params) {
  x <- args[[1L]]
  lax_mul(lax_add(one(x), x), lax_sub(one(x), x))
}), "number")
defjvp(one_minus_square_p, function(g, p, params) {
  lax_mul(g, lax_mul(lax_full_like(p[[1L]], -2), p[[1L]]))
})

#' More lax elementwise operations
#'
#' Ports of `jax.lax.log2` (base 2 logarithm), `jax.lax.mulhi` (the high half of the
#' double width product of integers) and `jax.lax.one_minus_square` (`1 - x^2`,
#' computed as `(1 + x) (1 - x)` with the derivative `-2 x`).
#' @param x,y Arrays (integers for `lax_mulhi`).
#' @param accuracy Optional accuracy request of `lax_log2()` (see [lax_elementwise]).
#' @return An array.
#' @name lax_elementwise_extra
#' @examples
#' \dontrun{
#' lax_log2(c(1, 2, 8))
#' lax_one_minus_square(0.5)
#' }
NULL

#' @rdname lax_elementwise_extra
#' @export
lax_log2 <- function(x, accuracy = NULL) bind(log2_p, list(as_jax(x)), accuracy_params(accuracy))

#' @rdname lax_elementwise_extra
#' @export
lax_mulhi <- function(x, y) bind(mulhi_p, lax_args(x, y))

#' @rdname lax_elementwise_extra
#' @export
lax_one_minus_square <- function(x) bind(one_minus_square_p, list(as_jax(x)))

# ---- polynomial -----------------------------------------------------------------------------

polynomial_p <- new_primitive("polynomial")
polynomial_p$abstract_eval <- naryop_abstract_eval("polynomial", "number")
polynomial_p$batch <- lax_broadcast_batcher(polynomial_p)

# Port of lax._polynomial_jvp.
polynomial_p$jvp <- function(primals, tangents, params) {
  x <- primals[[1L]]
  coeffs <- primals[-1L]
  y <- bind(polynomial_p, primals, params)
  dot_y <- zero_of(tangent_aval(aval_of(y)))
  dx <- tangents[[1L]]
  if (!is_zero(dx)) {
    deriv <- lapply(seq_along(coeffs)[-1L], function(k) {
      lax_mul(lax_full_like(coeffs[[k]], k - 1L), coeffs[[k]])
    })
    dot_y <- add_tangents(dot_y, lax_mul(dx, lax_polynomial(x, deriv, unroll = params$unroll)))
  }
  dc <- tangents[-1L]
  if (any(!vapply(dc, is_zero, NA))) {
    concrete <- Map(function(c, d) if (is_zero(d)) lax_full_like(c, 0) else d, coeffs, dc)
    dot_y <- add_tangents(dot_y, lax_polynomial(x, concrete, unroll = params$unroll))
  }
  list(y, dot_y)
}

# Port of lax._polynomial_transpose (linear in the coefficients only).
polynomial_p$transpose <- function(ct, args, params) {
  x <- args[[1L]]
  coeffs <- args[-1L]
  if (is_undefined(x)) jax_error("internal: polynomial is not linear in x")
  out <- list(NULL)
  if (is_zero(ct)) {
    for (k in seq_along(coeffs))
      out[k + 1L] <- list(if (is_undefined(coeffs[[k]])) zero_of(coeffs[[k]]$aval) else NULL)
    return(out)
  }
  xk <- ct
  for (k in seq_along(coeffs)) {
    c <- coeffs[[k]]
    out[k + 1L] <- list(if (is_undefined(c)) unbroadcast(c$aval, xk) else NULL)
    if (k < length(coeffs)) xk <- lax_mul(xk, x)
  }
  out
}

# Port of lax._polynomial_lower: Horner's rule, unrolled up to `unroll` terms (default
# 8) and a scan over the stacked coefficients beyond.
polynomial_p$lowering <- lower_fun(function(args, params) {
  x <- args[[1L]]
  coeffs <- args[-1L]
  n <- length(coeffs)
  thr <- params$unroll %||% 8L
  if (n <= thr) {
    acc <- coeffs[[n]]
    for (c in rev(coeffs[-n])) acc <- lax_add(lax_mul(acc, x), c)
    return(acc)
  }
  avals <- lapply(args, aval_of)
  out_shape <- broadcasting_shape_rule("polynomial", avals)
  coeff_shape <- broadcasting_shape_rule("polynomial", avals[-1L])
  b <- lapply(rev(coeffs), function(c) broadcast_to_shape(c, coeff_shape))
  weak <- all(vapply(avals, `[[`, NA, "weak"))
  init <- lax_convert_element_type(lax_full_like(x, 0, shape = out_shape), avals[[1L]]$dtype,
                                   weak_type = weak)
  lax_scan(function(acc, c) list(lax_add(lax_mul(acc, x), c), NULL), init, lax_stack(b, 0L),
           unroll = thr)[[1L]]
})

#' Evaluate a polynomial
#'
#' Port of `jax.lax.polynomial`: Horner's rule for
#' `coeffs[[1]] + x * coeffs[[2]] + x^2 * coeffs[[3]] + ...` (coefficients from the
#' lowest degree to the highest). Derivatives with respect to `x` are lower degree
#' polynomials.
#' @param x Array.
#' @param coeffs List (or vector) of coefficients; arrays must have the rank of `x` or
#'   be scalars.
#' @param unroll Number of terms unrolled before a scan is used (default 8).
#' @return An array.
#' @examples
#' \dontrun{
#' lax_polynomial(3, c(1, 5, 2))  # 34
#' }
#' @export
lax_polynomial <- function(x, coeffs, unroll = NULL) {
  x <- as_jax(x)
  dt <- aval_of(x)$dtype
  if (!is.list(coeffs)) coeffs <- as.list(coeffs)
  # Scalar coefficients are constants of the dtype of x (lax._const).
  cs <- lapply(coeffs, function(c) {
    if (!is_tracer(c) && !length(aval_of(c)$shape)) return(sp_const(x, c))
    lax_convert_element_type(as_jax(c), dt)
  })
  if (!length(cs)) return(lax_full_like(x, 0))
  if (length(cs) == 1L) {
    shape <- broadcasting_shape_rule("polynomial", list(aval_of(x), aval_of(cs[[1L]])))
    return(broadcast_to_shape(cs[[1L]], shape))
  }
  bind(polynomial_p, c(list(x), cs),
       list(unroll = if (is.null(unroll)) NULL else as.integer(unroll)))
}

# ---- tile -----------------------------------------------------------------------------------

tile_p <- new_primitive("tile")
tile_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  reps <- params$reps
  if (length(a$shape) != length(reps))
    jax_error("reps length must be equal to the ndim of x, got len(reps)=", length(reps),
              " and x.ndim=", length(a$shape), ".", class = "jax_type_error")
  new_aval(a$shape * reps, a$dtype, a$weak)
}
# Port of lax._tile_lower: reshape to (1, d0, 1, d1, ...), broadcast to
# (r0, d0, r1, d1, ...) and reshape to the output.
tile_p$lowering <- function(ctx, ins, ai, ao, p) {
  a <- ai[[1L]]
  if (!length(a$shape)) return(ins[[1L]])
  ea <- new_aval(as.vector(rbind(1L, a$shape)), a$dtype)
  ba <- new_aval(as.vector(rbind(p$reps, a$shape)), a$dtype)
  x <- emit_op(ctx, "stablehlo.reshape", ins, tensor_type(a), tensor_type(ea))
  x <- hlo_broadcast(ctx, x, ea, ba, seq_along(ea$shape) - 1L)
  emit_op(ctx, "stablehlo.reshape", x, tensor_type(ba), tensor_type(ao[[1L]]))
}
# Port of lax._tile_transpose_rule.
deflinear(tile_p, function(ct, args, params) {
  a <- args[[1L]]$aval
  r <- length(a$shape)
  if (!r) return(list(ct))
  ctr <- lax_reshape(ct, as.vector(rbind(params$reps, a$shape)))
  list(lax_reduce_sum(ctr, 2L * (seq_len(r) - 1L)))
})
# Port of lax._tile_batch_rule.
tile_p$batch <- function(vals, dims, params, axis) {
  d <- dims[1L]
  list(bind(tile_p, vals, list(reps = append(params$reps, 1L, after = d))), d)
}

#' Tile an array
#'
#' Port of `jax.lax.tile`: repeats `operand` `reps[i]` times along dimension `i`.
#' @param operand Array.
#' @param reps Integer vector with one entry per dimension of `operand`.
#' @return An array.
#' @examples
#' \dontrun{
#' lax_tile(matrix(1:4, 2, byrow = TRUE), c(2L, 3L))
#' }
#' @export
lax_tile <- function(operand, reps) {
  bind(tile_p, list(as_jax(operand)), list(reps = canon_shape(reps)))
}

# ---- split ----------------------------------------------------------------------------------

split_p <- new_primitive("split", multiple_results = TRUE)
# Port of lax._split_shape_rule (dtype and weak type of the operand).
split_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  sizes <- params$sizes
  ax <- params$axis
  if (any(sizes < 0L))
    jax_error("Sizes passed to split must be nonnegative, got [", paste(sizes, collapse = ", "),
              "]", class = "jax_value_error")
  if (a$shape[ax + 1L] != sum(sizes))
    jax_error("Sum of sizes ", sum(sizes), " must be equal to dimension ", ax,
              " of the operand shape [", paste(a$shape, collapse = ", "), "]",
              class = "jax_value_error")
  lapply(sizes, function(s) {
    sh <- a$shape
    sh[ax + 1L] <- s
    new_aval(sh, a$dtype, a$weak)
  })
}
# Port of lax._split_lower: one slice per output.
split_p$lowering <- function(ctx, ins, ai, ao, p) {
  a <- ai[[1L]]
  r <- length(a$shape)
  start <- rep(0L, r)
  lim <- a$shape
  outs <- character(length(ao))
  for (i in seq_along(ao)) {
    lim[p$axis + 1L] <- start[p$axis + 1L] + ao[[i]]$shape[p$axis + 1L]
    outs[i] <- emit_op(ctx, "stablehlo.slice", ins, tensor_type(a), tensor_type(ao[[i]]),
                       attrs = list(start_indices = attr_i64_array(start),
                                    limit_indices = attr_i64_array(lim),
                                    strides = attr_i64_array(rep(1L, r))))
    start[p$axis + 1L] <- lim[p$axis + 1L]
  }
  outs
}
# Port of lax._split_transpose_rule.
deflinear(split_p, function(cts, args, params) {
  a <- args[[1L]]$aval
  if (all(vapply(cts, is_zero, NA))) return(list(zero_of(a)))
  cts <- lapply(cts, function(c) if (is_zero(c)) zeros_aval(strip_weak(c$aval)) else c)
  list(lax_concatenate(cts, params$axis))
})
# Port of lax._split_batch_rule.
split_p$batch <- function(vals, dims, params, axis) {
  d <- dims[1L]
  ax <- if (params$axis >= d) params$axis + 1L else params$axis
  out <- bind(split_p, vals, list(sizes = params$sizes, axis = ax))
  list(out, rep(d, length(out)))
}

#' Split an array
#'
#' Port of `jax.lax.split`: chunks of sizes `sizes` along `axis`.
#' @param operand Array.
#' @param sizes Integer vector of chunk sizes; they must add up to the size of `axis`.
#' @param axis 0-based axis.
#' @return A list of arrays.
#' @examples
#' \dontrun{
#' lax_split(1:5, c(2L, 3L))
#' }
#' @export
lax_split <- function(operand, sizes, axis = 0L) {
  x <- as_jax(operand)
  if (!is.numeric(sizes) || anyNA(sizes) || any(sizes != round(sizes)))
    jax_error("Sizes passed to split must be integers, got ", paste(format(sizes), collapse = ", "),
              class = "jax_value_error")
  bind(split_p, list(x), list(sizes = as.integer(sizes),
                              axis = norm_axis(axis, length(aval_of(x)$shape))))
}

# ---- stack / unstack ------------------------------------------------------------------------

lxe_shapes_str <- function(avals) paste(vapply(avals, function(a) fmt_tuple(a$shape), ""), collapse = ", ")

stack_p <- new_primitive("stack")
# Ports of lax._stack_shape_rule and lax._stack_dtype_rule (check_same_dtypes).
stack_p$abstract_eval <- function(avals, params) {
  if (!length(avals)) jax_error("stack expects at least one operand, got 0.", class = "jax_value_error")
  if (length(unique(vapply(avals, function(a) length(a$shape), 1L))) != 1L)
    jax_error("Cannot stack arrays with different numbers of dimensions: got ",
              lxe_shapes_str(avals), ".", class = "jax_value_error")
  if (length(unique(lapply(avals, `[[`, "shape"))) != 1L)
    jax_error("All input arrays must have the same shape. Got ", lxe_shapes_str(avals), ".",
              class = "jax_value_error")
  dts <- vapply(avals, `[[`, "", "dtype")
  if (length(unique(dts)) > 1L)
    jax_error("lax.stack requires arguments to have the same dtypes, got ",
              paste(dts, collapse = ", "), ".", class = "jax_type_error")
  new_aval(append(avals[[1L]]$shape, length(avals), after = params$axis), dts[1L],
           all(vapply(avals, `[[`, NA, "weak")))
}
# Port of lax._stack_lower: broadcast each operand to a size one axis, then a tree of
# concatenations of at most 16 operands (lax._concatenate_tree).
stack_p$lowering <- function(ctx, ins, ai, ao, p) {
  a <- ai[[1L]]
  ax <- p$axis
  ea <- new_aval(append(a$shape, 1L, after = ax), a$dtype)
  bd <- setdiff(seq_len(length(a$shape) + 1L) - 1L, ax)
  xs <- vapply(seq_along(ins), function(i) hlo_broadcast(ctx, ins[[i]], ai[[i]], ea, bd), "")
  sizes <- rep(1L, length(xs))
  while (length(xs) > 1L) {
    grp <- (seq_along(xs) - 1L) %/% 16L
    nx <- character(0)
    ns <- integer(0)
    for (g in unique(grp)) {
      sel <- which(grp == g)
      sh <- ea$shape
      sh[ax + 1L] <- sum(sizes[sel])
      ns <- c(ns, sh[ax + 1L])
      nx <- c(nx, emit_op(ctx, "stablehlo.concatenate", xs[sel],
                          vapply(sizes[sel], function(s) { z <- ea$shape; z[ax + 1L] <- s; tensor_type(new_aval(z, a$dtype)) }, ""),
                          tensor_type(new_aval(sh, a$dtype)),
                          attrs = list(dimension = attr_i64(ax))))
    }
    xs <- nx
    sizes <- ns
  }
  xs[[1L]]
}
# Port of lax._stack_transpose_rule.
deflinear(stack_p, function(ct, args, params) {
  if (is_zero(ct)) return(lapply(args, function(a) if (is_undefined(a)) zero_of(a$aval) else NULL))
  out <- bind(unstack_p, list(ct), list(axis = params$axis))
  Map(function(o, a) if (is_undefined(a)) o else NULL, out, args)
})
# Port of lax._stack_batch_rule.
stack_p$batch <- function(vals, dims, params, axis) {
  ax <- params$axis
  bd <- dims[1L]
  if (!is.na(bd) && all(!is.na(dims)) && all(dims == bd)) {
    if (ax <= bd) return(list(bind(stack_p, vals, list(axis = ax)), bd + 1L))
    return(list(bind(stack_p, vals, list(axis = ax + 1L)), bd))
  }
  ops <- Map(function(v, d) bdim_at_front(v, d, axis$size), vals, dims)
  list(bind(stack_p, ops, list(axis = ax + 1L)), 0L)
}

unstack_p <- new_primitive("unstack", multiple_results = TRUE)
# Port of lax._unstack_shape_rule.
unstack_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  if (!length(a$shape))
    jax_error("unstack requires arrays with rank > 0, however a scalar array of shape () was passed.",
              class = "jax_value_error")
  ax <- params$axis
  rep(list(new_aval(a$shape[-(ax + 1L)], a$dtype, a$weak)), a$shape[ax + 1L])
}
# Port of lax._unstack_lower: slice and reshape per output.
unstack_p$lowering <- function(ctx, ins, ai, ao, p) {
  a <- ai[[1L]]
  r <- length(a$shape)
  ax <- p$axis
  sa <- a
  sa$shape[ax + 1L] <- 1L
  lim <- a$shape
  outs <- character(length(ao))
  for (i in seq_along(ao)) {
    start <- rep(0L, r)
    start[ax + 1L] <- i - 1L
    lim[ax + 1L] <- i
    s <- emit_op(ctx, "stablehlo.slice", ins, tensor_type(a), tensor_type(sa),
                 attrs = list(start_indices = attr_i64_array(start),
                              limit_indices = attr_i64_array(lim),
                              strides = attr_i64_array(rep(1L, r))))
    outs[i] <- emit_op(ctx, "stablehlo.reshape", s, tensor_type(sa), tensor_type(ao[[i]]))
  }
  outs
}
# Port of lax._unstack_transpose_rule.
deflinear(unstack_p, function(cts, args, params) {
  a <- args[[1L]]$aval
  if (all(vapply(cts, is_zero, NA))) return(list(zero_of(a)))
  cts <- lapply(cts, function(c) if (is_zero(c)) zeros_aval(strip_weak(c$aval)) else c)
  list(bind(stack_p, cts, list(axis = params$axis)))
})
# Port of lax._unstack_batch_rule.
unstack_p$batch <- function(vals, dims, params, axis) {
  bd <- dims[1L]
  ax <- params$axis
  if (ax < bd) {
    out_axis <- ax
    out_bd <- bd - 1L
  } else {
    out_axis <- ax + 1L
    out_bd <- bd
  }
  out <- bind(unstack_p, vals, list(axis = out_axis))
  list(out, rep(out_bd, length(out)))
}

#' Stack and unstack arrays
#'
#' Ports of `jax.lax.stack` (join arrays of equal shape and dtype along a new axis) and
#' `jax.lax.unstack` (the list of slices along an axis).
#' @param operands List of arrays.
#' @param axis 0-based axis (of the result for `lax_stack`).
#' @param x Array.
#' @return `lax_stack()` returns an array, `lax_unstack()` a list of arrays.
#' @name lax_stack
#' @examples
#' \dontrun{
#' lax_stack(list(c(1, 2), c(3, 4)), axis = 1L)
#' lax_unstack(matrix(1:4, 2))
#' }
NULL

#' @rdname lax_stack
#' @export
lax_stack <- function(operands, axis = 0L) {
  if (!length(operands)) jax_error("stack expects at least one operand, got 0.", class = "jax_value_error")
  xs <- do.call(lax_args, unname(as.list(operands)))
  ax <- norm_axis(axis, length(aval_of(xs[[1L]])$shape) + 1L)
  bind(stack_p, xs, list(axis = ax))
}

#' @rdname lax_stack
#' @export
lax_unstack <- function(x, axis = 0L) {
  x <- as_jax(x)
  r <- length(aval_of(x)$shape)
  ax <- if (r) norm_axis(axis, r) else as.integer(axis)
  bind(unstack_p, list(x), list(axis = ax))
}

# ---- empty ----------------------------------------------------------------------------------

empty_p <- new_primitive("empty")
empty_p$abstract_eval <- function(avals, params) new_aval(params$shape, params$dtype)
# Port of lax._empty_lower (CPU): a broadcast zero.
empty_p$lowering <- function(ctx, ins, ai, ao, p) full_constant(ctx, 0, ao[[1L]])

#' Array of unspecified values
#'
#' Port of `jax.lax.empty`. On CPU the values are zeros.
#' @param shape Integer vector.
#' @param dtype Dtype.
#' @return An array.
#' @examples
#' \dontrun{
#' lax_empty(3L, "float32")
#' }
#' @export
lax_empty <- function(shape, dtype) {
  bind(empty_p, list(), list(shape = canon_shape(shape), dtype = dtype(dtype)))
}

# ---- approx_max_k / approx_min_k ------------------------------------------------------------

# Port of XLA's ApproxTopKReductionOutputSize (approx_topk_shape.cc), the output size of
# the reduction dimension when aggregate_to_topk is FALSE.
lxe_approx_top_k_output_size <- function(input_size, rank, top_k, recall_target,
                                         input_size_override) {
  log2_floor <- function(x) if (x <= 0) -1 else floor(log2(x))
  log2_ceil <- function(x) if (x <= 1) 0 else ceiling(log2(x))
  ceil_ratio <- function(a, b) (a + b - 1) %/% b
  tiling <- if (rank == 1L) 1024 else 128
  if (input_size <= tiling) return(as.integer(input_size))
  if (input_size_override >= 0 && input_size > input_size_override)
    jax_error("reduction_input_size_override: ", input_size_override, " should be greater ",
              "equals to operands[reduction_dim]: ", input_size, class = "jax_value_error")
  logical_size <- if (input_size_override >= 0) input_size_override else input_size
  if (top_k == 1L) return(as.integer(tiling))
  # The recall target is a C++ float.
  rt <- readBin(writeBin(as.numeric(recall_target), raw(), size = 4L), "double", size = 4L)
  if (rt == 1) return(as.integer(input_size))
  if (rt <= 0 || rt > 1) jax_error("recall_target should range in (0,1]", class = "jax_value_error")
  m <- min(max(floor((1 - top_k) / log(rt)), tiling), input_size)
  lr <- log2_floor(logical_size %/% m)
  if (lr == 0) return(as.integer(input_size))
  if (input_size_override >= 0) lr <- min(lr, log2_ceil(input_size %/% tiling))
  as.integer(ceil_ratio(ceil_ratio(input_size, tiling), 2^lr) * tiling)
}

approx_top_k_p <- new_primitive("approx_top_k", multiple_results = TRUE)
# Port of ann._approx_top_k_abstract_eval.
approx_top_k_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  k <- params$k
  if (k <= 0L) jax_error("k must be positive, got ", k, class = "jax_value_error")
  if (!length(a$shape))
    jax_error("approx_top_k operand must have >= 1 dimension, got ", fmt_tuple(a$shape),
              class = "jax_type_error")
  dims <- a$shape
  rd <- norm_axis(params$reduction_dimension, length(dims), "reduction_dimension")
  if (dims[rd + 1L] < k)
    jax_error("k must be smaller than the size of reduction_dim ", dims[rd + 1L], ", got ", k,
              class = "jax_value_error")
  if (!is_float_dtype(a$dtype)) jax_error("operand must be a floating type", class = "jax_value_error")
  dims[rd + 1L] <- if (params$aggregate_to_topk) k else
    lxe_approx_top_k_output_size(dims[rd + 1L], length(dims), k, params$recall_target,
                                 params$reduction_input_size_override)
  list(new_aval(dims, a$dtype, a$weak), new_aval(dims, "int32"))
}

# CPU lowering. JAX emits the custom call ApproxTopK with is_fallback = true and a
# comparator function; jaxR modules hold a single function, so this emits what XLA's
# ApproxTopKFallback expands that call into: a variadic reduce selecting by the
# comparator for one output, otherwise a sort of (operand, iota) by the value only
# comparator followed by a slice.
approx_top_k_p$lowering <- function(ctx, ins, ai, ao, params) {
  a <- ai[[1L]]
  r <- length(a$shape)
  rd <- norm_axis(params$reduction_dimension, r)
  ia <- new_aval(a$shape, "int32")
  iota <- emit_op(ctx, "stablehlo.iota", character(0), character(0), tensor_type(ia),
                  attrs = list(iota_dimension = attr_i64(rd)))
  st <- tensor_type(new_aval(integer(0), a$dtype))
  it <- tensor_type(new_aval(integer(0), "int32"))
  bt <- tensor_type(new_aval(integer(0), "bool"))
  dir <- paste0("#stablehlo<comparison_direction ", if (params$is_max_k) "GT" else "LT", ">")
  cmp <- function(sub, x, y) {
    emit_op(sub, "stablehlo.compare", c(x, y), c(st, st), bt, attrs = list(comparison_direction = dir))
  }
  k_out <- ao[[1L]]$shape[rd + 1L]
  if (k_out == 1L) {
    init_v <- ir_constant(ctx, if (params$is_max_k) -Inf else Inf, new_aval(integer(0), a$dtype))
    init_i <- ir_constant(ctx, -1, new_aval(integer(0), "int32"))
    ra <- new_aval(a$shape[-(rd + 1L)], a$dtype)
    ri <- new_aval(a$shape[-(rd + 1L)], "int32")
    region <- build_region(ctx, c(st, it, st, it), function(sub, args) {
      pred <- cmp(sub, args[1L], args[3L])
      v <- emit_op(sub, "stablehlo.select", c(pred, args[1L], args[3L]), c(bt, st, st), st)
      i <- emit_op(sub, "stablehlo.select", c(pred, args[2L], args[4L]), c(bt, it, it), it)
      list(values = c(v, i), types = c(st, it))
    })
    red <- emit_op(ctx, "stablehlo.reduce", c(ins[[1L]], iota, init_v, init_i),
                   c(tensor_type(a), tensor_type(ia), st, it), c(tensor_type(ra), tensor_type(ri)),
                   attrs = list(dimensions = attr_i64_array(rd)), regions = region)
    return(c(emit_op(ctx, "stablehlo.reshape", red[1L], tensor_type(ra), tensor_type(ao[[1L]])),
             emit_op(ctx, "stablehlo.reshape", red[2L], tensor_type(ri), tensor_type(ao[[2L]]))))
  }
  region <- build_region(ctx, c(st, st, it, it), function(sub, args) {
    list(values = cmp(sub, args[1L], args[2L]), types = bt)
  })
  srt <- emit_op(ctx, "stablehlo.sort", c(ins[[1L]], iota), c(tensor_type(a), tensor_type(ia)),
                 c(tensor_type(a), tensor_type(ia)),
                 attrs = list(dimension = attr_i64(rd), is_stable = attr_bool(FALSE)),
                 regions = region)
  lim <- a$shape
  lim[rd + 1L] <- k_out
  sl <- function(v, ti, to) {
    emit_op(ctx, "stablehlo.slice", v, ti, to,
            attrs = list(start_indices = attr_i64_array(rep(0L, r)),
                         limit_indices = attr_i64_array(lim), strides = attr_i64_array(rep(1L, r))))
  }
  c(sl(srt[1L], tensor_type(a), tensor_type(ao[[1L]])),
    sl(srt[2L], tensor_type(ia), tensor_type(ao[[2L]])))
}

# Port of ann._approx_top_k_batch_rule.
approx_top_k_p$batch <- function(vals, dims, params, axis) {
  d <- dims[1L]
  r <- length(aval_of(vals[[1L]])$shape)
  dim_map <- setdiff(seq_len(r) - 1L, d)
  rd <- params$reduction_dimension
  new_rd <- dim_map[if (rd < 0L) length(dim_map) + rd + 1L else rd + 1L]
  out <- bind(approx_top_k_p, vals, update_params(params, list(reduction_dimension = new_rd)))
  list(out, c(d, d))
}

# Port of ann._approx_top_k_jvp (a gather of the tangent at the returned indices).
approx_top_k_p$jvp <- function(primals, tangents, params) {
  out <- bind(approx_top_k_p, primals, params)
  zero_idx <- zero_of(tangent_aval(aval_of(out[[2L]])))
  t <- tangents[[1L]]
  if (is_zero(t)) return(list(out, list(zero_of(tangent_aval(aval_of(out[[1L]]))), zero_idx)))
  rd <- norm_axis(params$reduction_dimension, length(aval_of(out[[2L]])$shape))
  list(out, list(jnp_take_along_axis(t, out[[2L]], axis = rd), zero_idx))
}

#' Approximate top k
#'
#' Ports of `jax.lax.approx_max_k` and `jax.lax.approx_min_k`: the `k` largest
#' (smallest) values along `reduction_dimension` and their 0-based indices. On CPU the
#' result is exact (XLA's fallback sorts); the order of tied values is implementation
#' defined.
#' @param operand Floating point array.
#' @param k Number of values.
#' @param reduction_dimension 0-based dimension searched (negative counts from the end).
#' @param recall_target Recall target of the approximation.
#' @param reduction_input_size_override When positive, the size used to evaluate the
#'   recall instead of the size of the reduction dimension.
#' @param aggregate_to_topk When `FALSE` the results are not reduced to `k` values; the
#'   size of the reduction dimension is then implementation defined (at least `k`).
#' @return A list of the values and their `int32` indices.
#' @name lax_approx_max_k
#' @examples
#' \dontrun{
#' lax_approx_max_k(c(3, 1, 4, 1, 5, 9, 2, 6), 3L)
#' }
NULL

lxe_approx_top_k <- function(operand, k, reduction_dimension, recall_target, is_max_k,
                             reduction_input_size_override, aggregate_to_topk) {
  bind(approx_top_k_p, list(as_jax(operand)),
       list(k = npa_index(k, "k"), reduction_dimension = as.integer(reduction_dimension),
            recall_target = as.numeric(recall_target), is_max_k = is_max_k,
            reduction_input_size_override = as.integer(reduction_input_size_override),
            aggregate_to_topk = isTRUE(aggregate_to_topk)))
}

#' @rdname lax_approx_max_k
#' @export
lax_approx_max_k <- function(operand, k, reduction_dimension = -1L, recall_target = 0.95,
                             reduction_input_size_override = -1L, aggregate_to_topk = TRUE) {
  lxe_approx_top_k(operand, k, reduction_dimension, recall_target, TRUE,
                   reduction_input_size_override, aggregate_to_topk)
}

#' @rdname lax_approx_max_k
#' @export
lax_approx_min_k <- function(operand, k, reduction_dimension = -1L, recall_target = 0.95,
                             reduction_input_size_override = -1L, aggregate_to_topk = TRUE) {
  lxe_approx_top_k(operand, k, reduction_dimension, recall_target, FALSE,
                   reduction_input_size_override, aggregate_to_topk)
}

# ---- dot product helpers -------------------------------------------------------------------

# Port of lax._dot_general_dtype_rule (XLA's HigherPrecisionType) and _maybe_upcast.
lxe_dot_dtype <- function(ldt, rdt, preferred_element_type, name = "lax.dot_general") {
  props <- function(dt) {
    c_dt <- complex_component(dt)
    k <- dtype_kind(c_dt)
    fl <- k == "f"
    maxexp <- c(float16 = 16, bfloat16 = 128, float32 = 128, float64 = 1024)
    nmant <- c(float16 = 10, bfloat16 = 7, float32 = 23, float64 = 52)
    c(is_complex_dtype(dt), if (fl) maxexp[[c_dt]] else -1, if (fl) nmant[[c_dt]] else -1,
      if (c_dt == "bool") 1 else dtype_bits(c_dt), k != "u")
  }
  cmp <- function(a, b) {
    for (i in seq_along(a)) if (a[i] != b[i]) return(sign(a[i] - b[i]))
    0
  }
  o <- cmp(props(ldt), props(rdt))
  res <- if (o > 0) ldt else if (o < 0) rdt else {
    if (ldt != rdt) jax_error(name, " argument type error: ", ldt, ", ", rdt, class = "jax_type_error")
    ldt
  }
  pet <- preferred_element_type
  if (is.null(pet) || pet == res) return(res)
  width <- function(dt) if (dt == "bool") 1L else dtype_bits(complex_component(dt))
  if (!is_float_dtype(res) && width(pet) < width(res))
    jax_error("`preferred_element_type` must not be narrower than the original type, got ",
              "preferred_element_type of ", pet, " for result type of ", res, ".",
              class = "jax_type_error")
  pet
}

# Single precision name for lax_dot_general (which takes one name) from a precision like
# value; on CPU precision does not change results.
lxe_dot_precision <- function(precision) {
  p <- lax_PrecisionLike(precision)
  if (is.null(p) || !is.character(p)) return(NULL)
  p[1L]
}

# Port of numpy moveaxis (util.moveaxis) for several axes.
lxe_moveaxis <- function(x, source, destination) {
  r <- length(aval_of(x)$shape)
  source <- norm_axis(source, r)
  destination <- norm_axis(destination, r)
  order <- setdiff(seq_len(r) - 1L, source)
  o <- order(destination)
  for (j in o) order <- append(order, source[j], after = destination[j])
  lax_transpose(x, order)
}

# ---- ragged_dot_general ---------------------------------------------------------------------

lxe_ragged_mode_names <- c("RAGGED_NONCONTRACTING", "RAGGED_CONTRACTING", "RAGGED_BATCH")

# Port of lax._ragged_dot_mode_and_dim: 1 non contracting, 2 contracting, 3 batch.
lxe_ragged_mode <- function(lhs_rank, rdn) {
  rd <- rdn$lhs_ragged_dimensions[1L]
  nonc <- setdiff(seq_len(lhs_rank) - 1L, c(rdn$lhs_contracting, rdn$lhs_batch))
  if (rd %in% nonc) return(1L)
  if (rd %in% rdn$lhs_contracting) return(2L)
  if (rd %in% rdn$lhs_batch) return(3L)
  jax_error("lhs_ragged_dim ", rd, " not found in lhs_noncontracting [", paste(nonc, collapse = ", "),
            "], lhs_contracting [", paste(rdn$lhs_contracting, collapse = ", "), "], or lhs_batch [",
            paste(rdn$lhs_batch, collapse = ", "), "].", class = "jax_type_error")
}

# Port of lax._ragged_dot_prefix_dims.
lxe_ragged_prefix <- function(mode, rank, ragged_dim, batch, contract) {
  nonc <- setdiff(seq_len(rank) - 1L, c(contract, batch))
  switch(mode,
         c(batch, nonc[seq_len(match(ragged_dim, nonc) - 1L)]),
         c(batch, contract[seq_len(match(ragged_dim, contract) - 1L)]),
         batch[seq_len(match(ragged_dim, batch) - 1L)])
}

lxe_rdn_dot <- function(rdn) {
  list(list(rdn$lhs_contracting, rdn$rhs_contracting), list(rdn$lhs_batch, rdn$rhs_batch))
}

ragged_dot_general_p <- new_primitive("ragged_dot_general")

# Ports of lax._ragged_dot_general_shape_rule and _ragged_dot_general_dtype_rule.
ragged_dot_general_p$abstract_eval <- function(avals, params) {
  l <- avals[[1L]]
  r <- avals[[2L]]
  g <- avals[[3L]]
  rdn <- params$ragged_dot_dimension_numbers
  in_range <- function(d, rank, what, arg) {
    if (d < 0L || d >= rank)
      jax_error("ragged_dot_general requires ", what, " numbers to be nonnegative and less ",
                "than the number of axes of the ", arg, " value, got ", d, " for ", arg,
                " of rank ", rank, ".", class = "jax_type_error")
  }
  if (length(rdn$lhs_ragged_dimensions) != 1L)
    jax_error("ragged_dot_general expects exactly one lhs ragged dimension.", class = "jax_type_error")
  rd <- rdn$lhs_ragged_dimensions[1L]
  in_range(rd, length(l$shape), "lhs ragged dimension", "lhs")
  mode <- lxe_ragged_mode(length(l$shape), rdn)
  if (!length(g$shape)) jax_error("expected rank of group_sizes to be >=1.", class = "jax_type_error")
  if (length(g$shape) != 1L) {
    pre <- lxe_ragged_prefix(mode, length(l$shape), rd, rdn$lhs_batch, rdn$lhs_contracting)
    expected <- c(l$shape[pre + 1L], g$shape[length(g$shape)])
    if (!identical(as.integer(g$shape), as.integer(expected)))
      jax_error("expected group_sizes to have shape ", fmt_tuple(expected), ", got ",
                fmt_tuple(g$shape), ".", class = "jax_type_error")
  }
  num_groups <- g$shape[length(g$shape)]
  rg <- rdn$rhs_group_dimensions
  if (mode %in% 2:3) {
    if (length(rg))
      jax_error("ragged_dot_general requires zero group dimensions in the rhs when lhs ragged ",
                "dimension is contracting or batch.", class = "jax_type_error")
  } else {
    if (length(rg) != 1L)
      jax_error("ragged_dot_general requires exactly one rhs group dimension when lhs ragged ",
                "dimension is noncontracting.", class = "jax_type_error")
    in_range(rg, length(r$shape), "rhs group dimension", "rhs")
    if (rg %in% c(rdn$rhs_batch, rdn$rhs_contracting))
      jax_error("ragged_dot_general requires rhs group dimension numbers to be distinct from ",
                "contracting and batch dimensions.", class = "jax_type_error")
    if (r$shape[rg + 1L] != num_groups)
      jax_error("expected rhs group dimension size to be ", num_groups, ", got ",
                r$shape[rg + 1L], ".", class = "jax_type_error")
  }
  lc <- rdn$lhs_contracting; rc <- rdn$rhs_contracting
  lb <- rdn$lhs_batch; rb <- rdn$rhs_batch
  if (length(lb) != length(rb))
    jax_error("dot_general requires equal numbers of lhs_batch and rhs_batch dimensions, got ",
              "lhs_batch ", fmt_tuple(lb), " and rhs_batch ", fmt_tuple(rb), ".", class = "jax_type_error")
  if (!identical(l$shape[lb + 1L], r$shape[rb + 1L]))
    jax_error("dot_general requires lhs batch dimensions and rhs batch dimensions to have the ",
              "same shape, got ", fmt_tuple(l$shape[lb + 1L]), " and ", fmt_tuple(r$shape[rb + 1L]),
              ".", class = "jax_type_error")
  if (!identical(l$shape[lc + 1L], r$shape[rc + 1L]))
    jax_error("dot_general requires contracting dimensions to have the same shape, got ",
              fmt_tuple(l$shape[lc + 1L]), " and ", fmt_tuple(r$shape[rc + 1L]), ".",
              class = "jax_type_error")
  shape <- c(l$shape[lb + 1L], l$shape[setdiff(seq_along(l$shape) - 1L, c(lc, lb)) + 1L],
             r$shape[setdiff(seq_along(r$shape) - 1L, c(rc, rb, rg)) + 1L])
  if (mode == 2L) shape <- c(num_groups, shape)
  if (!is_int_dtype(g$dtype))
    jax_error("ragged_dot_general requires that group_sizes.dtype is subtype of np.integer.",
              class = "jax_type_error")
  dt <- lxe_dot_dtype(l$dtype, r$dtype, params$preferred_element_type, "lax.ragged_dot_general")
  new_aval(shape, dt)
}

# Port of lax._ragged_dot_general_impl (the CPU lowering): the ragged dimension is
# expanded into a dense group dimension by masking, then one dot_general.
lxe_ragged_impl <- function(lhs, rhs, group_sizes, rdn, precision, pet) {
  ragged_to_dense <- function(x, gs, dim) {
    shape <- c(aval_of(gs)$shape, aval_of(x)$shape)
    x <- lax_broadcast_in_dim(x, shape, seq_along(aval_of(x)$shape))
    iota <- lax_broadcasted_iota(aval_of(gs)$dtype, shape, dim + 1L)
    ends <- lax_cumsum(gs)
    starts <- lax_sub(ends, gs)
    ends <- lax_broadcast_in_dim(ends, shape, 0L)
    starts <- lax_broadcast_in_dim(starts, shape, 0L)
    mask <- lax_bitwise_and(lax_le(starts, iota), lax_lt(iota, ends))
    lax_select(mask, x, lax_full_like(x, 0))
  }
  batched <- function(dim, axes) {
    if (!length(axes)) return(function(x, gs) ragged_to_dense(x, gs, dim))
    xa <- axes[1L]
    decr <- function(d) ifelse(d >= xa, d - 1L, d)
    vmap(batched(decr(dim), decr(axes[-1L])), in_axes = list(xa, 0L))
  }
  expand <- function(x, dim, gs, axes) {
    out <- batched(dim, axes)(x, gs)
    nd <- length(aval_of(x)$shape)
    unsorted <- c(axes + 1L, 0L, setdiff(seq_len(nd) - 1L, axes) + 1L)
    lax_transpose(out, order(unsorted) - 1L)
  }
  ls <- aval_of(lhs)$shape
  mode <- lxe_ragged_mode(length(ls), rdn)
  rd <- rdn$lhs_ragged_dimensions[1L]
  lc <- rdn$lhs_contracting; rc <- rdn$rhs_contracting
  lb <- rdn$lhs_batch; rb <- rdn$rhs_batch
  l_prefix <- lxe_ragged_prefix(mode, length(ls), rd, lb, lc)
  if (length(aval_of(group_sizes)$shape) == 1L)
    group_sizes <- lax_broadcast(group_sizes, ls[l_prefix + 1L])
  dg <- function(a, b, dn) lax_dot_general(a, b, dn, precision = precision, preferred_element_type = pet)
  switch(mode,
    dg(expand(lhs, rd, group_sizes, l_prefix), rhs,
       list(list(c(lc + 1L, 0L), c(rc, rdn$rhs_group_dimensions[1L])), list(lb + 1L, rb))),
    {
      rr <- rc[match(rd, lc)]
      r_prefix <- lxe_ragged_prefix(mode, length(aval_of(rhs)$shape), rr, rb, rc)
      dg(expand(lhs, rd, group_sizes, l_prefix), expand(rhs, rr, group_sizes, r_prefix),
         list(list(lc + 1L, rc + 1L), list(c(0L, lb + 1L), c(0L, rb + 1L))))
    },
    dg(lhs, rhs, lxe_rdn_dot(rdn)))
}

ragged_dot_general_p$lowering <- lower_fun(function(args, params) {
  lxe_ragged_impl(args[[1L]], args[[2L]], args[[3L]], params$ragged_dot_dimension_numbers,
                  params$precision, params$preferred_element_type)
})

# Port of lax._ragged_dot_general_jvp_rule.
ragged_dot_general_p$jvp <- function(primals, tangents, params) {
  out <- bind(ragged_dot_general_p, primals, params)
  dx <- tangents[[1L]]
  dy <- tangents[[2L]]
  t1 <- if (is_zero(dx)) dx else bind(ragged_dot_general_p, list(dx, primals[[2L]], primals[[3L]]), params)
  t2 <- if (is_zero(dy)) dy else bind(ragged_dot_general_p, list(primals[[1L]], dy, primals[[3L]]), params)
  list(out, add_tangents(t1, t2))
}

# Port of lax._ragged_dot_general_transpose_rule (implemented for the non contracting
# mode only, as in JAX).
ragged_dot_general_p$transpose <- function(ct, args, params) {
  x <- args[[1L]]
  y <- args[[2L]]
  gs <- args[[3L]]
  rdn <- params$ragged_dot_dimension_numbers
  xa <- arg_aval(x)
  ya <- arg_aval(y)
  xc <- rdn$lhs_contracting; yc <- rdn$rhs_contracting
  xb <- rdn$lhs_batch; yb <- rdn$rhs_batch
  x_kept <- setdiff(seq_along(xa$shape) - 1L, c(xc, xb))
  yg <- rdn$rhs_group_dimensions
  y_kept <- setdiff(seq_along(ya$shape) - 1L, c(yc, yb, yg))
  mode <- lxe_ragged_mode(length(xa$shape), rdn)
  rd <- rdn$lhs_ragged_dimensions[1L]
  if (is_zero(ct)) return(list(NULL, NULL, NULL))
  unimpl <- function(fn) {
    jax_error("Unimplemented ", fn, " for ragged dot general in mode ", lxe_ragged_mode_names[mode],
              ".", class = "jax_not_implemented_error")
  }
  grad <- function(lhs, rhs, dims, unsorted, aval) {
    out <- bind(ragged_dot_general_p, list(lhs, rhs, gs), update_params(params, list(ragged_dot_dimension_numbers = dims)))
    out <- lax_transpose(out, order(unsorted) - 1L)
    if (aval_of(out)$dtype != aval$dtype) out <- lax_convert_element_type(out, aval$dtype, aval$weak)
    out
  }
  x_bar <- NULL
  y_bar <- NULL
  if (is_undefined(x)) {
    if (mode != 1L) unimpl("grad_x_dims")
    rr <- ranges_like(xb, x_kept, y_kept)
    dims <- lax_RaggedDotDimensionNumbers(list(list(rr[[3L]], y_kept), list(rr[[1L]], yb)),
                                          length(xb) + match(rd, x_kept) - 1L, yg)
    x_bar <- grad(ct, y, dims, c(xb, x_kept, xc[order(yc)]), xa)
  }
  if (is_undefined(y)) {
    if (mode != 1L) unimpl("grad_y_dims")
    rr <- ranges_like(xb, x_kept, y_kept)
    dims <- lax_RaggedDotDimensionNumbers(list(list(x_kept, rr[[2L]]), list(xb, rr[[1L]])), rd,
                                          integer(0))
    y_bar <- grad(x, ct, dims, c(yg, yb, yc[order(xc)], y_kept), ya)
  }
  list(x_bar, y_bar, NULL)
}

# Port of lax._ragged_dot_general_batch_rule (through _dot_batch_rule and
# _dot_general_batch_dim_nums with both operands batched at 0). JAX requires every
# operand to be batched at dimension 0; here batch dimensions are moved to the front
# and unbatched operands broadcast first.
ragged_dot_general_p$batch <- function(vals, dims, params, axis) {
  n <- axis$size
  lhs <- bdim_at_front(vals[[1L]], dims[1L], n)
  rhs <- bdim_at_front(vals[[2L]], dims[2L], n)
  gs <- vals[[3L]]
  if (!is.na(dims[3L])) gs <- moveaxis(gs, dims[3L], 0L)
  else if (length(aval_of(gs)$shape) > 1L) gs <- lax_broadcast(gs, n)
  rdn <- params$ragged_dot_dimension_numbers
  mode <- lxe_ragged_mode(length(aval_of(vals[[1L]])$shape) - !is.na(dims[1L]), rdn)
  new <- rdn
  new$lhs_contracting <- rdn$lhs_contracting + 1L
  new$rhs_contracting <- rdn$rhs_contracting + 1L
  new$lhs_batch <- c(0L, rdn$lhs_batch + 1L)
  new$rhs_batch <- c(0L, rdn$rhs_batch + 1L)
  new$lhs_ragged_dimensions <- rdn$lhs_ragged_dimensions + 1L
  new$rhs_group_dimensions <- rdn$rhs_group_dimensions + 1L
  out <- bind(ragged_dot_general_p, list(lhs, rhs, gs),
              update_params(params, list(ragged_dot_dimension_numbers = new)))
  list(out, if (mode == 2L) 1L else 0L)
}

#' Ragged dot products
#'
#' Ports of `jax.lax.ragged_dot` and `jax.lax.ragged_dot_general`. In `lax_ragged_dot`
#' the rows of `lhs` (shape `(m, k)`) are split into consecutive groups of sizes
#' `group_sizes` (length `g`) and group `i` is multiplied by `rhs[i, , ]` (shape
#' `(g, k, n)`), giving an `(m, n)` result. `lax_ragged_dot_general` takes the general
#' dimension numbers of [lax_RaggedDotDimensionNumbers()]: the lhs ragged dimension is a
#' non contracting, contracting (the result gets a leading group dimension) or batch
#' dimension.
#' @param lhs,rhs Arrays.
#' @param group_sizes Integer array of group sizes.
#' @param precision Optional precision.
#' @param preferred_element_type Optional result dtype.
#' @param group_offset Not supported (as in JAX); must be `NULL`.
#' @param ragged_dot_dimension_numbers A [lax_RaggedDotDimensionNumbers()] object.
#' @return An array.
#' @name lax_ragged_dot
#' @examples
#' \dontrun{
#' lhs <- matrix(1, 5, 2)
#' rhs <- array(1:12, c(3, 2, 2))
#' lax_ragged_dot(lhs, rhs, c(2L, 1L, 2L))
#' }
NULL

#' @rdname lax_ragged_dot
#' @export
lax_ragged_dot_general <- function(lhs, rhs, group_sizes, ragged_dot_dimension_numbers,
                                   precision = NULL, preferred_element_type = NULL,
                                   group_offset = NULL) {
  if (!is.null(group_offset))
    jax_error("Unimplemented group_offset support.", class = "jax_not_implemented_error")
  if (!inherits(ragged_dot_dimension_numbers, "jax_RaggedDotDimensionNumbers"))
    jax_error("ragged_dot_dimension_numbers must be a lax_RaggedDotDimensionNumbers object",
              class = "jax_type_error")
  args <- lax_args(lhs, rhs)
  bind(ragged_dot_general_p, c(args, list(as_jax(group_sizes))),
       list(ragged_dot_dimension_numbers = ragged_dot_dimension_numbers,
            precision = lxe_dot_precision(precision),
            preferred_element_type = if (is.null(preferred_element_type)) NULL
                                     else dtype(preferred_element_type)))
}

#' @rdname lax_ragged_dot
#' @export
lax_ragged_dot <- function(lhs, rhs, group_sizes, precision = NULL, preferred_element_type = NULL,
                           group_offset = NULL) {
  lax_ragged_dot_general(lhs, rhs, group_sizes,
                         lax_RaggedDotDimensionNumbers(list(list(1L, 1L), list(integer(0), integer(0))), 0L, 0L),
                         precision = precision, preferred_element_type = preferred_element_type,
                         group_offset = group_offset)
}

# ---- conv_general_dilated_patches / conv_general_dilated_local ---------------------------------

#' Image patches and unshared convolutions
#'
#' Ports of `jax.lax.conv_general_dilated_patches` (extract the patches of the
#' receptive field of [lax_conv_general_dilated()] into the output feature dimension,
#' in the order `"c" + spatial dims of rhs_spec`) and
#' `jax.lax.conv_general_dilated_local` (a locally connected layer: a convolution with
#' a separate kernel at each output position; the spatial dimensions of `rhs` are the
#' output positions and its `I` dimension holds `(in feature, filter spatial...)`).
#' @param lhs Input array of rank `n + 2`.
#' @param filter_shape Integer vector of `n` receptive field sizes (in the order of the
#'   rhs spatial dimensions).
#' @param window_strides Integer vector of `n` strides.
#' @param padding `"SAME"`, `"VALID"` or a list of `n` `c(low, high)` pairs.
#' @param lhs_dilation,rhs_dilation Optional dilation factors.
#' @param dimension_numbers `NULL`, three layout strings or a
#'   [lax_ConvDimensionNumbers()] object.
#' @param precision Optional precision (see [lax_Precision]).
#' @param preferred_element_type Optional result dtype.
#' @param rhs Kernel of the unshared convolution.
#' @return An array.
#' @name lax_conv_patches
#' @examples
#' \dontrun{
#' x <- array(as.numeric(1:16), c(1, 1, 4, 4))
#' dim(lax_conv_general_dilated_patches(x, c(2L, 2L), c(1L, 1L), "VALID"))
#' }
NULL

#' @rdname lax_conv_patches
#' @export
lax_conv_general_dilated_patches <- function(lhs, filter_shape, window_strides, padding,
                                             lhs_dilation = NULL, rhs_dilation = NULL,
                                             dimension_numbers = NULL, precision = NULL,
                                             preferred_element_type = NULL) {
  x <- as_jax(lhs)
  ls <- aval_of(x)$shape
  filter_shape <- as.integer(filter_shape)
  dn <- lax_conv_dimension_numbers(ls, c(1L, 1L, filter_shape), dimension_numbers)
  spatial <- prod_int(filter_shape)
  n_channels <- ls[dn$lhs_spec[2L] + 1L]
  dt <- aval_of(x)$dtype
  eye <- lax_convert_element_type(lax_eq(lax_broadcasted_iota("int32", c(spatial, spatial), 0L),
                                         lax_broadcasted_iota("int32", c(spatial, spatial), 1L)), dt)
  rhs <- lax_broadcast_in_dim(eye, c(n_channels, spatial, spatial), c(1L, 2L))
  rhs <- lax_reshape(rhs, c(n_channels * spatial, 1L, filter_shape))
  rhs <- lxe_moveaxis(rhs, c(0L, 1L), dn$rhs_spec[1:2])
  lax_conv_general_dilated(x, rhs, window_strides, padding, lhs_dilation = lhs_dilation,
                           rhs_dilation = rhs_dilation, dimension_numbers = dn,
                           feature_group_count = n_channels,
                           precision = if (is.null(precision)) NULL else c(lax_PrecisionLike(precision)[1L], "DEFAULT"),
                           preferred_element_type = preferred_element_type)
}

#' @rdname lax_conv_patches
#' @export
lax_conv_general_dilated_local <- function(lhs, rhs, window_strides, padding, filter_shape,
                                           lhs_dilation = NULL, rhs_dilation = NULL,
                                           dimension_numbers = NULL, precision = NULL) {
  x <- as_jax(lhs)
  cp <- lax_PrecisionLike(precision)
  if (!is.null(cp) && !is.character(cp))
    jax_error("Unsupported precision for conv_general_dilated_local: ", format(unclass(precision)),
              class = "jax_value_error")
  patches <- lax_conv_general_dilated_patches(x, filter_shape, window_strides, padding,
                                              lhs_dilation = lhs_dilation, rhs_dilation = rhs_dilation,
                                              dimension_numbers = dimension_numbers,
                                              precision = if (is.null(cp)) NULL else cp[1L])
  dn <- lax_conv_dimension_numbers(aval_of(x)$shape, c(1L, 1L, as.integer(filter_shape)),
                                   dimension_numbers)
  lhs_b <- drop_first(dn$out_spec, 2L)
  rhs_b <- drop_first(dn$rhs_spec, 2L)
  rhs_b <- rhs_b[order(lhs_b)]
  lhs_b <- sort(lhs_b)
  out <- lax_dot_general(patches, rhs, list(list(dn$out_spec[2L], dn$rhs_spec[2L]), list(lhs_b, rhs_b)),
                         precision = lxe_dot_precision(precision))
  lxe_moveaxis(out, c(-2L, -1L), dn$out_spec[1:2])
}

# ---- composite --------------------------------------------------------------------------------

composite_p <- new_primitive("composite", multiple_results = TRUE)
composite_p$abstract_eval <- function(avals, params) jaxpr_out_avals(params$call_jaxpr$jaxpr)
# Port of lax._composite_impl.
composite_p$impl <- function(args, params) eval_closed(params$call_jaxpr, args)
# Port of lax._composite_lowering. JAX emits stablehlo.composite with the decomposition
# as a private function; jaxR modules hold a single function, so the decomposition is
# inlined (what XLA does with a composite it does not recognize).
composite_p$lowering <- function(ctx, ins, ai, ao, params) {
  cj <- params$call_jaxpr
  jaxpr_subcomp(ctx, cj$jaxpr, lower_closed_consts(ctx, cj), as.list(ins))
}
# Ports of lax.composite_jvp and lax.composite_transpose.
composite_p$jvp <- function(primals, tangents, params) {
  jax_error("JVP rule for composite not implemented. You can use `jax.custom_jvp` to add ",
            "support. See https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html",
            class = "jax_value_error")
}
composite_p$transpose <- function(cts, args, params) {
  jax_error("Transpose rule for composite not implemented. You can use`jax.custom_jvp` or ",
            "`jax.custom_vjp` to add support. See ",
            "https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html",
            class = "jax_value_error")
}

#' Composite operation
#'
#' Port of `jax.lax.composite`: wraps `decomposition` as a named composite operation
#' that compilers may replace by a specialized implementation. Unnamed arguments of the
#' returned function are inputs; named arguments are attributes passed to
#' `decomposition` (they must not be traced values). On CPU the decomposition is
#' compiled. Composites are not differentiable (use [custom_jvp()]) and have no
#' batching rule, as in JAX.
#' @param decomposition Function implementing the operation.
#' @param name Name of the operation.
#' @param version Version of its semantics.
#' @return A function.
#' @examples
#' \dontrun{
#' tangent <- lax_composite(function(x) lax_sin(x) / lax_cos(x), name = "my.tangent")
#' tangent(c(0, pi / 4))
#' }
#' @export
lax_composite <- function(decomposition, name, version = 0L) {
  force(decomposition)
  function(...) {
    dots <- list(...)
    nms <- names(dots) %||% rep("", length(dots))
    inputs <- dots[!nzchar(nms)]
    attrs <- dots[nzchar(nms)]
    if (any(vapply(tree_leaves(attrs), is_tracer, NA)))
      jax_error("Found a JAX Tracer as an attribute in the decomposition for the composite op '",
                name, "'. This means that the decomposition function closes over a value that ",
                "is involved in a JAX transformation. Any values that aren't explicitly known ",
                "at compile time must be explicitly passed as arguments to the composite.",
                "\n\nNote: If you are passing jax arrays as attributes, use numpy arrays instead.",
                class = c("jax_unexpected_tracer_error", "jax_escaped_tracer_error", "jax_type_error"))
    fl <- tree_flatten(unname(inputs))
    t <- trace_flat_fun(function(...) do.call(decomposition, c(list(...), attrs)), fl$treedef,
                        lapply(fl$leaves, aval_of))
    cj <- closed_jaxpr(convert_constvars(t$cj$jaxpr))
    out <- bind(composite_p, c(t$cj$consts, lapply(fl$leaves, as_jax)),
                list(name = name, attributes = attrs, version = as.integer(version), call_jaxpr = cj))
    tree_unflatten(t$out_tree, out)
  }
}

# ---- scaled_dot ---------------------------------------------------------------------------

# Port of scaled_dot._validate_operand_scale.
lxe_validate_scale <- function(side, op, scale, contracting) {
  for (i in seq_along(op$shape)) {
    size <- op$shape[i]
    if ((i - 1L) %in% contracting) {
      if (size %% scale$shape[i] != 0L)
        jax_error(side, " contracting dim ", i - 1L, " of size ", size, " must be divisible by ",
                  "its scale's dim size ", scale$shape[i], ".", class = "jax_type_error")
      s <- size %/% scale$shape[i]
      if (s < 2L)
        jax_error("The ratio of ", side, " contracting dim ", i - 1L, " to its scale's dim size (",
                  s, ") must be at least 2.", class = "jax_type_error")
    } else if (scale$shape[i] != size) {
      jax_error(side, " dim ", i - 1L, " of size ", size, " does not match scale dim size ",
                scale$shape[i], ".", class = "jax_type_error")
    }
  }
}

scaled_dot_p <- new_primitive("scaled_dot")
# Ports of scaled_dot._scaled_dot_validate_inputs and _scaled_dot_abstract_eval.
scaled_dot_p$abstract_eval <- function(avals, params) {
  l <- avals[[1L]]; r <- avals[[2L]]; ls <- avals[[3L]]; rs <- avals[[4L]]
  dn <- params$dimension_numbers
  lc <- dn[[1L]][[1L]]; rc <- dn[[1L]][[2L]]
  lb <- dn[[2L]][[1L]]; rb <- dn[[2L]][[2L]]
  nd <- vapply(list(l, r, ls, rs), function(a) length(a$shape), 1L)
  if (max(nd) != min(nd))
    jax_error("All input tensors must have the same rank. Got lhs rank: ", nd[1L], " rhs rank: ",
              nd[2L], " lhs_scale rank: ", nd[3L], " rhs_scale rank: ", nd[4L], ".",
              class = "jax_type_error")
  if (length(lb) != length(rb))
    jax_error("LHS and RHS must have the same number of batch dimensions, got ", length(lb),
              " and ", length(rb), ".", class = "jax_type_error")
  if (length(lc) != length(rc))
    jax_error("LHS and RHS must have the same number of contracting dimensions, got ",
              length(lc), " and ", length(rc), ".", class = "jax_type_error")
  for (j in seq_along(lb)) {
    s <- c(l$shape[lb[j] + 1L], r$shape[rb[j] + 1L], ls$shape[lb[j] + 1L], rs$shape[rb[j] + 1L])
    if (max(s) != min(s))
      jax_error("All input tensors must have the same batch dimension size for batch dims (",
                lb[j], ", ", rb[j], ").", class = "jax_type_error")
  }
  for (j in seq_along(lc)) {
    if (l$shape[lc[j] + 1L] != r$shape[rc[j] + 1L])
      jax_error("LHS contracting dim ", lc[j], " of size ", l$shape[lc[j] + 1L],
                " does not match RHS contracting dim ", rc[j], " of size ", r$shape[rc[j] + 1L],
                ".", class = "jax_type_error")
  }
  lxe_validate_scale("LHS", l, ls, lc)
  lxe_validate_scale("RHS", r, rs, rc)
  shape <- c(l$shape[lb + 1L], l$shape[setdiff(seq_along(l$shape) - 1L, c(lc, lb)) + 1L],
             r$shape[setdiff(seq_along(r$shape) - 1L, c(rc, rb)) + 1L])
  new_aval(shape, params$preferred_element_type %||% "bfloat16")
}

# Port of scaled_dot._scale_broadcast.
lxe_scale_broadcast <- function(scale, operand_shape, contracting) {
  for (i in contracting) {
    ss <- aval_of(scale)$shape
    if (ss[i + 1L] != operand_shape[i + 1L]) {
      mult <- operand_shape[i + 1L] %/% ss[i + 1L]
      scale <- lax_expand_dims(scale, i + 1L)
      scale <- lax_broadcast_in_dim(scale, append(ss, mult, after = i + 1L),
                                    seq_len(length(ss) + 1L) - 1L)
      ns <- ss
      ns[i + 1L] <- ss[i + 1L] * mult
      scale <- lax_reshape(scale, ns)
    }
  }
  scale
}

# Port of scaled_dot._scaled_dot_impl (the decomposition of the xla.scaled_dot
# composite, inlined).
scaled_dot_p$lowering <- lower_fun(function(args, params) {
  dn <- params$dimension_numbers
  scaled <- function(x, s, contracting) {
    s <- lxe_scale_broadcast(s, aval_of(x)$shape, contracting)
    lax_mul(lax_convert_element_type(x, "bfloat16"), lax_convert_element_type(s, "bfloat16"))
  }
  lhs <- scaled(args[[1L]], args[[3L]], dn[[1L]][[1L]])
  rhs <- scaled(args[[2L]], args[[4L]], dn[[1L]][[2L]])
  lax_dot_general(lhs, rhs, dn, preferred_element_type = params$preferred_element_type)
})

# Port of scaled_dot._scaled_dot_batching_rule.
scaled_dot_p$batch <- function(vals, dims, params, axis) {
  vals <- Map(function(v, d) bdim_at_front(v, d, axis$size), vals, dims)
  dn <- params$dimension_numbers
  ndn <- list(list(dn[[1L]][[1L]] + 1L, dn[[1L]][[2L]] + 1L),
              list(c(0L, dn[[2L]][[1L]] + 1L), c(0L, dn[[2L]][[2L]] + 1L)))
  list(bind(scaled_dot_p, vals, update_params(params, list(dimension_numbers = ndn))), 0L)
}

#' Scaled dot product
#'
#' Port of `jax.lax.scaled_dot`: `(lhs * lhs_scale) %*% (rhs * rhs_scale)` computed in
#' `bfloat16` with `preferred_element_type` accumulation (default `bfloat16`). Each scale
#' has the shape of its operand except along the contracting dimensions, where the
#' operand size must be a multiple (at least 2) of the scale size. Missing scales are
#' ones (JAX uses `float8_e8m0fnu` ones, a dtype jaxR does not have; the values are
#' equal). Not differentiable, as in JAX.
#' @param lhs,rhs Arrays.
#' @param lhs_scale,rhs_scale Optional scale arrays.
#' @param dimension_numbers Dot dimension numbers; by default the last dimension of
#'   `lhs` is contracted with the second to last of `rhs` and the leading dimensions are
#'   batch dimensions.
#' @param preferred_element_type Result dtype (`"bfloat16"` or `"float32"`).
#' @return An array.
#' @examples
#' \dontrun{
#' lax_scaled_dot(matrix(1, 2, 4), matrix(1, 4, 3), preferred_element_type = "float32")
#' }
#' @export
lax_scaled_dot <- function(lhs, rhs, lhs_scale = NULL, rhs_scale = NULL, dimension_numbers = NULL,
                           preferred_element_type = NULL) {
  lhs <- as_jax(lhs)
  rhs <- as_jax(rhs)
  n <- length(aval_of(lhs)$shape)
  if (is.null(dimension_numbers)) {
    if (n <= 1L)
      jax_error("scaled_dot does not support 0-rank and 1-rank operands.", class = "jax_type_error")
    b <- seq_len(n - 2L) - 1L
    dimension_numbers <- list(list(n - 1L, n - 2L), list(b, b))
  }
  dn <- list(list(as.integer(dimension_numbers[[1L]][[1L]]), as.integer(dimension_numbers[[1L]][[2L]])),
             list(as.integer(dimension_numbers[[2L]][[1L]] %||% integer(0)),
                  as.integer(dimension_numbers[[2L]][[2L]] %||% integer(0))))
  dummy <- function(x, contracting) {
    s <- aval_of(x)$shape
    s[contracting + 1L] <- 1L
    lax_full(s, 1, "bfloat16")
  }
  lhs_scale <- if (is.null(lhs_scale)) dummy(lhs, dn[[1L]][[1L]]) else as_jax(lhs_scale)
  rhs_scale <- if (is.null(rhs_scale)) dummy(rhs, dn[[1L]][[2L]]) else as_jax(rhs_scale)
  pet <- dtype(preferred_element_type %||% "bfloat16")
  bind(scaled_dot_p, list(lhs, rhs, lhs_scale, rhs_scale),
       list(dimension_numbers = dn, preferred_element_type = pet))
}

# ---- platform_dependent / pcast ---------------------------------------------------------------

#' Platform dependent code
#'
#' Port of `jax.lax.platform_dependent`. jaxR runs on the CPU, so the `cpu` branch (or
#' `default` when there is none) is applied to the arguments.
#' @param ... Arguments passed to the selected branch (unnamed) and the branches, named by
#'   platform (`cpu`, `cuda`, `rocm`, `tpu`, ...).
#' @param default Optional branch for platforms without their own branch.
#' @return The result of the selected branch.
#' @examples
#' \dontrun{
#' lax_platform_dependent(2, cpu = function(x) x + 1, default = function(x) x - 1)
#' }
#' @export
lax_platform_dependent <- function(..., default = NULL) {
  dots <- list(...)
  nms <- names(dots) %||% rep("", length(dots))
  args <- dots[!nzchar(nms)]
  branches <- dots[nzchar(nms)]
  for (p in names(branches)) {
    if (!is.function(branches[[p]]))
      jax_error("lax.platform_dependent: the '", p, "' branch must be a callable.", class = "jax_type_error")
    if (p == "gpu")
      jax_error("Use 'cuda', 'rocm', or 'oneapi' for lax.platform_dependent.", class = "jax_value_error")
  }
  if (!is.null(default) && !is.function(default))
    jax_error("lax.platform_dependent: the 'default' branch must be a callable.", class = "jax_type_error")
  f <- branches[["cpu"]] %||% default
  if (is.null(f)) {
    plats <- paste0("(", paste0("('", names(branches), "',)", collapse = ", "),
                    if (length(branches) == 1L) "," else "", ")")
    jax_error("MLIR translation rule for primitive 'platform_index(platforms=", plats,
              ")' not found for platform cpu", class = "jax_not_implemented_error")
  }
  do.call(f, unname(args))
}

#' Cast between replication types
#'
#' Port of `jax.lax.pcast` for `vmap` axes. jaxR has no `shard_map`, so values are never
#' varying across a named axis in the type system: the cast to `"varying"` is the
#' identity (after checking that the axis names are bound), and the casts to
#' `"reduced"` and `"unreduced"` raise as they do in JAX for invariant values.
#' @param x A pytree of arrays.
#' @param axis_name Axis name or character vector of names (`NULL` or empty returns
#'   `x`).
#' @param to One of `"varying"`, `"reduced"` or `"unreduced"`.
#' @return `x`.
#' @examples
#' \dontrun{
#' vmap(function(v) lax_pcast(v, "i", to = "varying"), axis_name = "i")(c(1, 2))
#' }
#' @export
lax_pcast <- function(x, axis_name, to) {
  axes <- if (is.list(axis_name)) unlist(axis_name) else axis_name
  if (!length(axes)) return(x)
  if (!(length(to) == 1L && to %in% c("unreduced", "reduced", "varying")))
    jax_error("Got unexpected `to` value. Allowed `to` values are: {'reduced', 'unreduced', ",
              "'varying'}", class = "jax_value_error")
  if (to == "unreduced")
    jax_error("Unsupported pcast from=invarying, to='unreduced'", class = "jax_value_error")
  for (a in axes) axis_size_of(a)
  if (to == "reduced") jax_error("pcast to 'reduced' requires shard_map", class = "jax_not_implemented_error")
  x
}

# ---- tokens and dce_sink ------------------------------------------------------------------------

# Tokens order side effects in XLA. jaxR has no token type in its avals or lowering, so
# a token is a host sentinel: a pytree node with no leaves, which passes through jit and
# the other transformations unchanged.
lxe_token <- function() structure(list(), class = "jax_token")
register_pytree_node("jax_token", function(x) list(children = list(), aux = NULL),
                     function(aux, children) lxe_token())

#' @export
print.jax_token <- function(x, ...) {
  cat("Token()\n")
  invisible(x)
}

#' Tokens for ordering side effects
#'
#' Ports of `jax.lax.create_token` and `jax.lax.after_all`. jaxR represents a token as a
#' host value with no array leaves (there is no token type in its compiled programs), so
#' tokens can be created, merged and returned from transformed functions but order
#' nothing.
#' @param x Ignored (kept for compatibility, as in JAX).
#' @param ... Tokens.
#' @return A token.
#' @name lax_tokens
#' @examples
#' t <- lax_create_token()
#' lax_after_all(t, lax_create_token())
NULL

#' @rdname lax_tokens
#' @export
lax_create_token <- function(x = NULL) lxe_token()

#' @rdname lax_tokens
#' @export
lax_after_all <- function(...) {
  for (t in list(...)) {
    if (!inherits(t, "jax_token"))
      jax_error("Arguments to after_all must be tokens", class = "jax_type_error")
  }
  lxe_token()
}

#' Mark values as used
#'
#' Port of `jax.lax.dce_sink`: marks the computation of `val` as used so that dead code
#' elimination keeps it. jaxR programs have no side effects that dead code elimination
#' could drop, so this only checks that `val` is a pytree of arrays.
#' @param val A pytree of arrays.
#' @param prevent_mlir_dce Accepted for compatibility.
#' @return `NULL`, invisibly.
#' @examples
#' \dontrun{
#' lax_dce_sink(c(1, 2))
#' }
#' @export
lax_dce_sink <- function(val, prevent_mlir_dce = FALSE) {
  for (l in tree_leaves(val)) as_jax(l)
  invisible(NULL)
}

# ---- custom_root ----------------------------------------------------------------------------

# Port of control_flow.common._check_tree.
lxe_check_tree <- function(func_name, expected_name, actual, expected, has_aux = FALSE) {
  if (has_aux) {
    if (!identical(actual$type, "list") || length(actual$children) != 2L)
      jax_error(func_name, "() produced a pytree with structure ", format(actual), ", but a ",
                "pytree tuple with auxiliary output was expected because has_aux was set to True.",
                class = "jax_value_error")
    actual <- actual$children[[1L]]
  }
  if (!treedef_equal(actual, expected))
    jax_error(func_name, "() output pytree structure must match ", expected_name, ", got ",
              format(actual), " and ", format(expected), ".", class = "jax_type_error")
}

lxe_hoist <- function(t) list(jaxpr = closed_jaxpr(convert_constvars(t$cj$jaxpr)), consts = t$cj$consts)

# Operands: f consts, solve consts, linearize-and-solve consts, then the initial guess.
custom_root_p <- new_primitive("custom_root", multiple_results = TRUE)
custom_root_p$abstract_eval <- function(avals, params) jaxpr_out_avals(params$jaxprs$solve$jaxpr)

lxe_root_split <- function(args, cl) {
  nf <- cl[["f"]]
  ns <- cl[["solve"]]
  nl <- cl[["l_and_s"]]
  list(f = args[seq_len(nf)], solve = args[nf + seq_len(ns)], l_and_s = args[nf + ns + seq_len(nl)],
       guess = drop_first(args, nf + ns + nl))
}

custom_root_p$lowering <- lower_fun(function(args, params) {
  s <- lxe_root_split(args, params$const_lengths)
  eval_closed(params$jaxprs$solve, c(s$solve, s$guess))
}, multiple_results = TRUE)

# Port of solves._root_jvp (implicit function theorem): for f(p, x*(p)) = 0,
# dx* = -(d_x f)^-1 (d_p f dp), with d_p f dp the JVP of f at the solution with respect
# to the values f closes over and (d_x f)^-1 applied by tangent_solve. Tangents of the
# initial guess and of the solver's own values do not contribute.
custom_root_p$jvp <- function(primals, tangents, params) {
  sol <- bind(custom_root_p, primals, params)
  n <- params$n_sol
  s <- lxe_root_split(primals, params$const_lengths)
  f_dot <- lxe_root_split(tangents, params$const_lengths)$f
  if (all(vapply(f_dot, is_zero, NA)))
    return(list(sol, lapply(sol, function(o) zero_of(tangent_aval(aval_of(o))))))
  j <- params$jaxprs
  solution <- sol[seq_len(n)]
  f_at_solution <- function(...) eval_closed(j$f, c(list(...), solution))
  rhs <- jvp(f_at_solution, s$f, lapply(f_dot, instantiate_zero))[[2L]]
  x_dot <- eval_closed(j$l_and_s, c(s$l_and_s, solution, rhs))
  list(sol, c(lapply(x_dot, lax_neg),
              lapply(drop_first(sol, n), function(o) zero_of(tangent_aval(aval_of(o))))))
}

# Batching: the solution is always batched (the initial guess is broadcast when it is
# not), as the custom_jvp batching of JAX's _custom_root does.
custom_root_p$batch <- function(vals, dims, params, axis) {
  size <- axis$size
  cl <- params$const_lengths
  vals <- Map(function(v, d) if (is.na(d)) v else moveaxis(v, d, 0L), vals, dims)
  bat <- !is.na(dims)
  nc <- cl[["f"]] + cl[["solve"]] + cl[["l_and_s"]]
  for (i in setdiff(seq_along(vals), seq_len(nc))) {
    if (!bat[i]) vals[[i]] <- lax_broadcast(vals[[i]], size)
    bat[i] <- TRUE
  }
  b <- lxe_root_split(bat, cl)
  n <- params$n_sol
  j <- params$jaxprs
  bj <- function(cj, batched) batch_jaxpr(cj, size, axis$name, batched, TRUE)$jaxpr
  jaxprs <- list(f = bj(j$f, c(b$f, rep(TRUE, n))),
                 solve = bj(j$solve, c(b$solve, rep(TRUE, n))),
                 l_and_s = bj(j$l_and_s, c(b$l_and_s, rep(TRUE, 2L * n))))
  out <- bind(custom_root_p, vals, update_params(params, list(jaxprs = jaxprs)))
  list(out, rep(0L, length(out)))
}

#' Differentiable root finding
#'
#' Port of `jax.lax.custom_root`: solves `f(x) = 0` with `solve(f, initial_guess)` and
#' defines derivatives with respect to the values `f` closes over by implicit
#' differentiation at the solution (assuming `f(solution) == 0`), without
#' differentiating through `solve`.
#' @param f Function of a pytree returning a pytree of the same structure.
#' @param initial_guess Pytree of arrays.
#' @param solve Function `(f, initial_guess)` returning the root (or `list(root, aux)`
#'   with `has_aux`).
#' @param tangent_solve Function `(g, y)` solving the linear system `g(x) = y`, where `g`
#'   is `f` linearized at the root; for scalars `function(g, y) y / g(1)`.
#' @param has_aux Whether `solve` returns `list(root, aux)`.
#' @return The result of `solve(f, initial_guess)`.
#' @examples
#' \dontrun{
#' sqrt_newton <- function(a) {
#'   f <- function(x) x^2 - a
#'   solve <- function(f, x0) lax_fori_loop(0L, 20L, function(i, x) x - f(x) / (2 * x), x0)
#'   lax_custom_root(f, 1, solve, function(g, y) y / g(1))
#' }
#' grad(sqrt_newton)(2)  # 1 / (2 sqrt(2))
#' }
#' @export
lax_custom_root <- function(f, initial_guess, solve, tangent_solve, has_aux = FALSE) {
  gfl <- tree_flatten(initial_guess)
  g_leaves <- lapply(gfl$leaves, as_jax)
  g_avals <- lapply(g_leaves, aval_of)
  ft <- trace_flat_fun(f, tree_structure(list(initial_guess)), g_avals)
  lxe_check_tree("f", "initial_guess", ft$out_tree, gfl$treedef)
  st <- trace_flat_fun(function(x) solve(f, x), tree_structure(list(initial_guess)), g_avals)
  lxe_check_tree("solve", "initial_guess", st$out_tree, gfl$treedef, has_aux)
  t_avals <- lapply(g_avals, tangent_aval)
  lt <- trace_flat_fun(function(x, b) tangent_solve(linearize(f, x)[[2L]], b),
                       tree_structure(list(initial_guess, initial_guess)), c(g_avals, t_avals))
  lxe_check_tree("tangent_solve", "x", lt$out_tree, gfl$treedef)
  fh <- lxe_hoist(ft)
  sh <- lxe_hoist(st)
  lh <- lxe_hoist(lt)
  out <- bind(custom_root_p, c(fh$consts, sh$consts, lh$consts, g_leaves),
              list(const_lengths = c(f = length(fh$consts), solve = length(sh$consts),
                                     l_and_s = length(lh$consts)),
                   jaxprs = list(f = fh$jaxpr, solve = sh$jaxpr, l_and_s = lh$jaxpr),
                   n_sol = length(g_leaves)))
  tree_unflatten(st$out_tree, out)
}

# ---- linear_call ----------------------------------------------------------------------------

# Operands: callee consts, residuals, linear inputs.
linear_call_p <- new_primitive("linear_call", multiple_results = TRUE)
linear_call_p$abstract_eval <- function(avals, params) jaxpr_out_avals(params$callee$jaxpr)
# Ports of custom_derivatives._linear_call_impl and its lowering (mlir.lower_fun).
linear_call_p$impl <- function(args, params) eval_closed(params$callee, args)
linear_call_p$lowering <- lower_fun(function(args, params) eval_closed(params$callee, args),
                                    multiple_results = TRUE)

# Port of custom_derivatives._linear_call_jvp_rule.
linear_call_p$jvp <- function(primals, tangents, params) {
  k <- params$num_callee_consts + params$num_res
  if (!all(vapply(tangents[seq_len(k)], is_zero, NA)))
    jax_error("linear_call: residual arguments must not be differentiated",
              class = "jax_assertion_error")
  cr <- primals[seq_len(k)]
  out <- bind(linear_call_p, primals, params)
  tans <- bind(linear_call_p, c(cr, lapply(drop_first(tangents, k), instantiate_zero)), params)
  list(out, tans)
}

# Port of custom_derivatives._linear_call_transpose_rule: the transpose is a linear_call
# of fun_transpose whose own transpose is fun.
linear_call_p$transpose <- function(cts, args, params) {
  tt <- params$transpose_thunk()
  nc <- params$num_callee_consts
  nr <- params$num_res
  f_consts <- args[seq_len(nc)]
  res <- args[nc + seq_len(nr)]
  callee <- params$callee
  ct_avals <- drop_first(jaxpr_in_avals(tt$jaxpr$jaxpr), length(tt$consts) + nr)
  cts <- Map(function(c, a) if (is_zero(c)) zeros_aval(strip_weak(a)) else c, cts, ct_avals)
  out <- bind(linear_call_p, c(tt$consts, res, cts),
              list(callee = tt$jaxpr, transpose_thunk = function() list(jaxpr = callee, consts = f_consts),
                   num_callee_consts = length(tt$consts), num_res = nr))
  c(rep(list(NULL), nc + nr), out)
}

#' Linear function with a custom transpose
#'
#' Port of `jax.custom_derivatives.linear_call`: calls `fun(residual_args, linear_args)`,
#' a function linear in `linear_args`, whose transpose (used by reverse mode
#' differentiation and [linear_transpose()]) is `fun_transpose(residual_args, ct)`.
#' The transpose of the transpose is `fun` again. There is no batching rule, as in JAX.
#' @param fun Function `(residual_args, linear_args)` linear in its second argument.
#' @param fun_transpose Function `(residual_args, ct)` returning a pytree with the
#'   structure of `linear_args`.
#' @param residual_args Pytree of arguments in which `fun` need not be linear.
#' @param linear_args Pytree of the linear arguments.
#' @return `fun(residual_args, linear_args)`.
#' @examples
#' \dontrun{
#' f <- function(r, x) x / r
#' t <- function(r, ct) ct / r
#' div_add <- function(x, denom) x + linear_call(f, t, denom, x)
#' linear_transpose(function(x) div_add(x, 3), 1)(18)  # 24
#' }
#' @export
linear_call <- function(fun, fun_transpose, residual_args, linear_args) {
  rfl <- tree_flatten(residual_args)
  lfl <- tree_flatten(linear_args)
  res <- lapply(rfl$leaves, as_jax)
  lin <- lapply(lfl$leaves, as_jax)
  nr <- length(res)
  res_avals <- lapply(res, aval_of)
  store <- new.env(parent = emptyenv())
  cj <- trace_to_jaxpr(function(ls) {
    out <- fun(tree_unflatten(rfl$treedef, ls[seq_len(nr)]), tree_unflatten(lfl$treedef, drop_first(ls, nr)))
    ofl <- tree_flatten(out)
    store$out_tree <- ofl$treedef
    lapply(ofl$leaves, as_jax)
  }, c(res_avals, lapply(lin, aval_of)))
  out_tree <- store$out_tree
  out_avals <- jaxpr_out_avals(cj$jaxpr)
  cache <- new.env(parent = emptyenv())
  transpose_thunk <- function() {
    if (is.null(cache$t)) {
      tj <- trace_to_jaxpr(function(ls) {
        out <- fun_transpose(tree_unflatten(rfl$treedef, ls[seq_len(nr)]),
                             tree_unflatten(out_tree, drop_first(ls, nr)))
        ofl <- tree_flatten(out)
        if (!treedef_equal(ofl$treedef, lfl$treedef))
          jax_error("transpose output pytree structure must match that of linear inputs, got ",
                    "output structure ", format(ofl$treedef), " and input structure ",
                    format(lfl$treedef), ".", class = "jax_type_error")
        lapply(ofl$leaves, as_jax)
      }, c(res_avals, out_avals))
      cache$t <- list(jaxpr = closed_jaxpr(convert_constvars(tj$jaxpr)), consts = tj$consts)
    }
    cache$t
  }
  out <- bind(linear_call_p, c(cj$consts, res, lin),
              list(callee = closed_jaxpr(convert_constvars(cj$jaxpr)), transpose_thunk = transpose_thunk,
                   num_callee_consts = length(cj$consts), num_res = nr))
  tree_unflatten(out_tree, out)
}

# ---- symbolic zeros and custom_vjp primals ----------------------------------------------------

#' Symbolic zeros
#'
#' Ports of `jax.custom_derivatives.SymbolicZero` (a zero tangent known only by its
#' abstract value; `dim()` gives its shape and `$aval` its [ShapeDtypeStruct()]) and
#' `jax.custom_derivatives.zero_from_primal` (zero tangents for the leaves of a pytree,
#' as arrays or as symbolic zeros).
#' @param aval A [ShapeDtypeStruct()] (the tangent type).
#' @param val A pytree of arrays.
#' @param symbolic_zeros Return `SymbolicZero` objects instead of zero arrays.
#' @return `SymbolicZero()` returns an object of class `jax_SymbolicZero`;
#'   `zero_from_primal()` a pytree with the structure of `val`.
#' @name SymbolicZero
#' @examples
#' z <- SymbolicZero(ShapeDtypeStruct(c(2L, 3L), "float32"))
#' dim(z)
#' \dontrun{
#' zero_from_primal(list(a = c(1, 2), b = 3L))
#' }
NULL

#' @rdname SymbolicZero
#' @export
SymbolicZero <- function(aval) {
  if (!inherits(aval, "jax_aval")) aval <- aval_of(aval)
  structure(list(aval = aval), class = "jax_SymbolicZero")
}

#' @export
print.jax_SymbolicZero <- function(x, ...) {
  cat("SymbolicZero(", aval_str(x$aval), ")\n", sep = "")
  invisible(x)
}

#' @export
dim.jax_SymbolicZero <- function(x) x$aval$shape

#' @rdname SymbolicZero
#' @export
zero_from_primal <- function(val, symbolic_zeros = FALSE) {
  tree_map(function(x) {
    ta <- tangent_aval(aval_of(x))
    if (symbolic_zeros) SymbolicZero(ta) else zeros_aval(ta)
  }, val)
}

#' Primal values of custom_vjp forward rules
#'
#' Ports of `jax.custom_derivatives.CustomVJPPrimal` (a primal argument with a flag
#' telling whether it is perturbed, as received by a forward rule in symbolic zeros
#' mode) and `jax.custom_derivatives.custom_vjp_primal_tree_values` (strip such records
#' from a pytree, keeping their values).
#' @param value The primal value.
#' @param perturbed Whether the value carries a nonzero tangent.
#' @param tree A pytree whose leaves are `CustomVJPPrimal` objects.
#' @return `CustomVJPPrimal()` returns an object of class `jax_CustomVJPPrimal`;
#'   `custom_vjp_primal_tree_values()` the pytree of values.
#' @name CustomVJPPrimal
#' @examples
#' p <- list(CustomVJPPrimal(1, TRUE), CustomVJPPrimal(2, FALSE))
#' custom_vjp_primal_tree_values(p)
NULL

#' @rdname CustomVJPPrimal
#' @export
CustomVJPPrimal <- function(value, perturbed) {
  structure(list(value = value, perturbed = isTRUE(perturbed)), class = "jax_CustomVJPPrimal")
}

#' @rdname CustomVJPPrimal
#' @export
custom_vjp_primal_tree_values <- function(tree) {
  tree_map(function(leaf) {
    if (!inherits(leaf, "jax_CustomVJPPrimal"))
      jax_error("unexpected leaf type ", class(leaf)[1L], class = "jax_type_error")
    leaf$value
  }, tree)
}

# ---- checkpoint extras -------------------------------------------------------------------------

#' Rematerialization decisions of checkpoint policies
#'
#' Ports of `jax.ad_checkpoint.Saveable`, `Recompute` and `Offloadable`, the values a
#' policy of [checkpoint()] may return besides `TRUE` and `FALSE`: save the value,
#' recompute it, or offload it from memory space `src` to `dst` (jaxR has one memory
#' space, so offloading saves). `Saveable` and `Offloadable()` count as `TRUE` and
#' `Recompute` as `FALSE` (they are classed logical values).
#' @param src,dst Memory kinds (for example `"device"` and `"pinned_host"`).
#' @format `Saveable` and `Recompute` are classed logical scalars.
#' @name remat_policy_values
#' @examples
#' policy <- function(prim, params) if (prim == "sin") Saveable else Recompute
#' f <- checkpoint(function(x) sin(sin(x)), policy = policy)
NULL

#' @rdname remat_policy_values
#' @export
Saveable <- structure(TRUE, class = "jax_remat_case")

#' @rdname remat_policy_values
#' @export
Recompute <- structure(FALSE, class = "jax_remat_case")

#' @rdname remat_policy_values
#' @export
Offloadable <- function(src, dst) {
  structure(TRUE, src = src, dst = dst, class = c("jax_offloadable", "jax_remat_case"))
}

#' @export
print.jax_remat_case <- function(x, ...) {
  if (inherits(x, "jax_offloadable")) {
    cat("Offloadable(src='", attr(x, "src"), "', dst='", attr(x, "dst"), "')\n", sep = "")
  } else {
    cat(if (isTRUE(unclass(x))) "Saveable" else "Recompute", "\n")
  }
  invisible(x)
}

#' @export
`$.jax_offloadable` <- function(x, name) attr(x, name, exact = TRUE)

#' Name a value for checkpoint policies (forward pass)
#'
#' Port of `jax.ad_checkpoint.checkpoint_name_fwd`; without the experimental remat3
#' mode of JAX it is [checkpoint_name()].
#' @param x A pytree of arrays.
#' @param name Name used by [checkpoint_policies].
#' @return `x`.
#' @examples
#' \dontrun{
#' checkpoint_name_fwd(c(1, 2), "a")
#' }
#' @export
checkpoint_name_fwd <- function(x, name) checkpoint_name(x, name)

# Port of ad_checkpoint._saved_residuals over the jaxpr whose outputs are the residuals
# of the linearized function.
lxe_saved_residuals <- function(jaxpr, arg_names) {
  out <- list()
  add <- function(aval, src) out[[length(out) + 1L]] <<- list(aval = aval, src = src)
  outvars <- jaxpr$outvars
  # reduce_precision does not count as the producer; look through it.
  subst <- integer(0)
  for (e in jaxpr$eqns) {
    if (identical(e$prim$name, "reduce_precision") && is_var(e$invars[[1L]]))
      subst[as.character(e$outvars[[1L]]$id)] <- e$invars[[1L]]$id
  }
  look <- function(id) {
    while (!is.na(subst[as.character(id)])) id <- subst[[as.character(id)]]
    id
  }
  res_ids <- unique(vapply(Filter(is_var, outvars), function(v) look(v$id), 1L))
  for (a in Filter(is_lit, outvars)) add(a$aval, "from a literal")
  for (v in jaxpr$constvars) if (v$id %in% res_ids) add(v$aval, "from a constant")
  for (i in seq_along(jaxpr$invars)) {
    v <- jaxpr$invars[[i]]
    if (v$id %in% res_ids) {
      add(v$aval, if (nzchar(arg_names[i])) paste0("from the argument ", arg_names[i])
                  else "from the argument at flattened index {i}")
    }
  }
  for (e in jaxpr$eqns) {
    for (v in e$outvars) {
      if (!(v$id %in% res_ids)) next
      if (identical(e$prim$name, "name")) {
        add(v$aval, paste0("named '", e$params$name, "' from unknown"))
      } else {
        add(v$aval, paste0("output of ", e$prim$name, " from unknown"))
      }
    }
  }
  out
}

# Argument names of the flattened arguments (port of api_util.debug_info arg_names):
# formal names, `..k` for arguments matched by `...`, and R style paths into pytrees.
lxe_arg_names <- function(f, args) {
  fm <- names(formals(f))
  has_dots <- "..." %in% fm
  an <- names(args) %||% rep("", length(args))
  positional <- if (has_dots) fm[seq_len(match("...", fm) - 1L)] else fm
  unused <- setdiff(positional, an[nzchar(an)])
  paths <- function(x, prefix) {
    if (is.list(x) && is.null(oldClass(x))) {
      nms <- names(x)
      return(unlist(lapply(seq_along(x), function(i) {
        paths(x[[i]], if (!is.null(nms) && nzchar(nms[i])) paste0(prefix, "$", nms[i])
                      else paste0(prefix, "[[", i, "]]"))
      })))
    }
    rep(prefix, length(tree_leaves(x)))
  }
  pos <- 0L
  dots <- 0L
  unlist(lapply(seq_along(args), function(i) {
    nm <- an[i]
    if (!nzchar(nm)) {
      pos <<- pos + 1L
      nm <- if (pos <= length(unused)) unused[pos] else if (has_dots) {
        dots <<- dots + 1L
        paste0("..", dots)
      } else ""
    }
    if (!nzchar(nm)) return(rep("", length(tree_leaves(args[[i]]))))
    paths(args[[i]], nm)
  }))
}

#' Residuals saved for reverse mode differentiation
#'
#' Ports of `jax.ad_checkpoint.saved_residuals` and `print_saved_residuals`: the values
#' that reverse mode differentiation of `f` at the given arguments stores between the
#' forward and the backward pass, with where each comes from (an argument, a constant
#' or the output of a primitive). Source locations are not recorded by jaxR and print as
#' `unknown`.
#' @param f Function.
#' @param ... Arguments of `f`.
#' @return `saved_residuals()` returns a list of `list(aval, src)`;
#'   `print_saved_residuals()` prints one line per residual and returns it invisibly.
#' @examples
#' \dontrun{
#' print_saved_residuals(function(x) sin(x) * cos(x), 1)
#' }
#' @export
saved_residuals <- function(f, ...) {
  args <- list(...)
  fl <- tree_flatten(args)
  leaves <- lapply(fl$leaves, as_jax)
  store <- new.env(parent = emptyenv())
  cj <- trace_to_jaxpr(function(ls) {
    linearize_flat(flat_fun(f, fl$treedef, store), ls)$consts
  }, lapply(leaves, aval_of))
  lxe_saved_residuals(cj$jaxpr, lxe_arg_names(f, args))
}

#' @rdname saved_residuals
#' @export
print_saved_residuals <- function(f, ...) {
  res <- saved_residuals(f, ...)
  for (r in res) cat(aval_str(r$aval), " ", r$src, "\n", sep = "")
  invisible(res)
}
