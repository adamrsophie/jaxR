# lax shape and indexing primitives (port of jax/_src/lax/lax.py and slicing.py).

norm_axis <- function(axis, rank, what = "axis") {
  axis <- as.integer(axis)
  out <- ifelse(axis < 0L, axis + rank, axis)
  if (any(out < 0L | out >= max(rank, 1L)) && !(rank == 0L && all(out == 0L)))
    jax_error(what, " ", paste(axis, collapse = ","), " is out of bounds for array of rank ",
              rank, class = "jax_value_error")
  out
}

types_of <- function(avals) vapply(avals, tensor_type, "")

# Port of core.canonicalize_shape: sizes are concrete non-negative integers (a
# fractional size is an error, not something to truncate).
canon_shape <- function(shape) {
  if (is.null(shape)) return(integer(0))
  if (is_jax_value(shape)) shape <- as.vector(concrete_or_error(shape, "a shape argument"))
  if (is.list(shape)) {
    # list() is the empty tuple: the shape of a scalar.
    if (!length(shape)) return(integer(0))
    shape <- unlist(lapply(shape, function(s)
      if (is_jax_value(s)) as.vector(concrete_or_error(s, "a shape argument")) else s))
  }
  if (!is.numeric(shape) || anyNA(shape) || any(shape != round(shape)) || any(shape < 0))
    jax_error("Shapes must be 1D sequences of concrete values of integer type, got ",
              paste(format(shape), collapse = ", "), ".", class = "jax_type_error")
  as.integer(shape)
}

# ---- convert_element_type ---------------------------------------------------------------

convert_element_type_p <- new_primitive("convert_element_type")
convert_element_type_p$abstract_eval <- function(avals, params) {
  new_aval(avals[[1L]]$shape, params$new_dtype, params$weak_type)
}
convert_element_type_p$lowering <- function(ctx, ins, ai, ao, params) {
  a <- ai[[1L]]
  o <- ao[[1L]]
  x <- ins[[1L]]
  if (a$dtype == o$dtype) return(x)
  if (a$dtype == "float0" || o$dtype == "float0") return(full_constant(ctx, 0, o))
  if (is_complex_dtype(a$dtype) && !is_complex_dtype(o$dtype)) {
    ra <- new_aval(a$shape, complex_component(a$dtype))
    x <- emit_op(ctx, "stablehlo.real", x, tensor_type(a), tensor_type(ra))
    a <- ra
    if (a$dtype == o$dtype) return(x)
  }
  if (o$dtype == "bool") {
    z <- full_constant(ctx, 0, a)
    return(emit_op(ctx, "stablehlo.compare", c(x, z), rep(tensor_type(a), 2L), tensor_type(o),
                   attrs = list(comparison_direction = "#stablehlo<comparison_direction NE>",
                                compare_type = paste0("#stablehlo<comparison_type ",
                                                      compare_type(a$dtype), ">"))))
  }
  emit_op(ctx, "stablehlo.convert", x, tensor_type(a), tensor_type(o))
}
convert_element_type_p$jvp <- function(primals, tangents, params) {
  ans <- bind(convert_element_type_p, primals, params)
  t <- tangents[[1L]]
  if (is_zero(t) || !is_inexact_dtype(params$new_dtype)) return(list(ans, zero_of(tangent_aval(aval_of(ans)))))
  list(ans, bind(convert_element_type_p, list(t),
                 list(new_dtype = params$new_dtype, weak_type = isTRUE(params$weak_type))))
}
convert_element_type_p$transpose <- function(ct, args, params) {
  if (is_zero(ct)) return(list(ct))
  old <- args[[1L]]$aval$dtype
  if (is_complex_dtype(aval_of(ct)$dtype) && !is_complex_dtype(old)) ct <- lax_real(ct)
  # Port of _convert_element_type_transpose_rule: the cotangent gets the dtype and the
  # weak type of the operand.
  list(bind(convert_element_type_p, list(ct),
            list(new_dtype = old, weak_type = isTRUE(args[[1L]]$aval$weak))))
}
convert_element_type_p$batch <- function(vals, dims, params, axis) {
  list(bind(convert_element_type_p, vals, params), dims[1L])
}

#' Convert the dtype of an array
#' @param operand Array.
#' @param new_dtype Target dtype.
#' @param weak_type Whether the result is weakly typed.
#' @export
lax_convert_element_type <- function(operand, new_dtype, weak_type = FALSE) {
  x <- as_jax(operand)
  dt <- dtype(new_dtype)
  a <- aval_of(x)
  if (a$dtype == dt && a$weak == weak_type) return(x)
  bind(convert_element_type_p, list(x), list(new_dtype = dt, weak_type = isTRUE(weak_type)))
}

bitcast_convert_type_p <- new_primitive("bitcast_convert_type")
bitcast_convert_type_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  b1 <- dtype_bits(a$dtype)
  b2 <- dtype_bits(params$new_dtype)
  shape <- if (b1 == b2) a$shape else if (b1 > b2) c(a$shape, b1 %/% b2) else {
    if (tail(a$shape, 1L) != b2 %/% b1) jax_error("bitcast_convert_type: last dimension mismatch")
    head(a$shape, -1L)
  }
  new_aval(shape, params$new_dtype)
}
bitcast_convert_type_p$lowering <- function(ctx, ins, ai, ao, p) {
  emit_op(ctx, "stablehlo.bitcast_convert", ins, tensor_type(ai[[1L]]), tensor_type(ao[[1L]]))
}
bitcast_convert_type_p$batch <- function(vals, dims, params, axis) {
  list(bind(bitcast_convert_type_p, list(moveaxis(vals[[1L]], dims[1L], 0L)), params), 0L)
}
defjvp_zero(bitcast_convert_type_p)

#' Bitcast an array to another dtype
#' @param operand Array.
#' @param new_dtype Target dtype.
#' @export
lax_bitcast_convert_type <- function(operand, new_dtype) {
  bind(bitcast_convert_type_p, list(as_jax(operand)), list(new_dtype = dtype(new_dtype)))
}

# ---- broadcast_in_dim ----------------------------------------------------------------------

broadcast_in_dim_p <- new_primitive("broadcast_in_dim")
broadcast_in_dim_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  shape <- params$shape
  bd <- params$broadcast_dimensions
  if (length(bd) != length(a$shape))
    jax_error("broadcast_in_dim broadcast_dimensions must have length equal to operand ndim")
  for (i in seq_along(bd)) {
    if (!(a$shape[i] == 1L || a$shape[i] == shape[bd[i] + 1L]))
      jax_error("broadcast_in_dim operand dimension sizes must either be 1, or be equal to ",
                "their corresponding dimensions in the target broadcast shape; got operand of ",
                "shape (", paste(a$shape, collapse = ", "), "), target broadcast shape (",
                paste(shape, collapse = ", "), "), broadcast_dimensions (",
                paste(bd, collapse = ", "), ")", class = "jax_type_error")
  }
  new_aval(shape, a$dtype, a$weak)
}
broadcast_in_dim_p$lowering <- function(ctx, ins, ai, ao, p) {
  hlo_broadcast(ctx, ins[[1L]], ai[[1L]], ao[[1L]], p$broadcast_dimensions)
}
deflinear(broadcast_in_dim_p, function(ct, args, params) {
  a <- args[[1L]]$aval
  shape <- params$shape
  bd <- params$broadcast_dimensions
  unit <- bd[a$shape == 1L & shape[bd + 1L] != 1L]
  sum_dims <- sort(c(setdiff(seq_along(shape) - 1L, bd), unit))
  r <- if (length(sum_dims)) lax_reduce_sum(ct, sum_dims) else ct
  list(lax_reshape(r, a$shape))
})
broadcast_in_dim_p$batch <- function(vals, dims, params, axis) {
  x <- moveaxis(vals[[1L]], dims[1L], 0L)
  n <- aval_of(x)$shape[1L]
  list(bind(broadcast_in_dim_p, list(x),
            list(shape = c(n, params$shape), broadcast_dimensions = c(0L, params$broadcast_dimensions + 1L))),
       0L)
}

#' Broadcast an array into a target shape
#' @param operand Array.
#' @param shape Target shape.
#' @param broadcast_dimensions 0-based output dimension of each operand dimension.
#' @export
lax_broadcast_in_dim <- function(operand, shape, broadcast_dimensions) {
  x <- as_jax(operand)
  shape <- canon_shape(shape)
  bd <- as.integer(broadcast_dimensions)
  if (identical(aval_of(x)$shape, shape) && identical(bd, seq_along(shape) - 1L)) return(x)
  bind(broadcast_in_dim_p, list(x), list(shape = shape, broadcast_dimensions = bd))
}

#' Add leading dimensions
#' @param operand Array.
#' @param sizes Sizes of the new leading dimensions.
#' @export
lax_broadcast <- function(operand, sizes) {
  x <- as_jax(operand)
  r <- length(aval_of(x)$shape)
  sizes <- canon_shape(sizes)
  lax_broadcast_in_dim(x, c(sizes, aval_of(x)$shape), length(sizes) + seq_len(r) - 1L)
}

# Broadcast x (rank 0 or same rank with size 1 dims) to `shape`.
broadcast_to_shape <- function(x, shape) {
  a <- aval_of(x)
  if (identical(a$shape, as.integer(shape))) return(x)
  dims <- if (length(a$shape)) seq_along(a$shape) - 1L else integer(0)
  lax_broadcast_in_dim(x, shape, dims)
}
#' Broadcast an array to the shape of another
#'
#' Port of `jax.lax.broadcast_like`.
#' @param arr Array to broadcast.
#' @param like_arr Array whose shape is matched.
#' @export
lax_broadcast_like <- function(arr, like_arr) broadcast_to_shape(arr, aval_of(like_arr)$shape)

#' Insert size one dimensions
#' @param array Array.
#' @param dimensions 0-based positions of the new dimensions in the result.
#' @export
lax_expand_dims <- function(array, dimensions) {
  x <- as_jax(array)
  s <- aval_of(x)$shape
  nd <- length(s) + length(dimensions)
  dims <- sort(norm_axis(dimensions, nd))
  out_shape <- integer(nd)
  keep <- setdiff(seq_len(nd) - 1L, dims)
  out_shape[dims + 1L] <- 1L
  out_shape[keep + 1L] <- s
  lax_broadcast_in_dim(x, out_shape, keep)
}

# ---- reshape / squeeze / transpose / rev ---------------------------------------------------

reshape_p <- new_primitive("reshape")
reshape_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  if (prod(a$shape) != prod(params$new_sizes))
    jax_error("reshape total size must be unchanged, got new_sizes (",
              paste(params$new_sizes, collapse = ", "), ") for shape (",
              paste(a$shape, collapse = ", "), ").", class = "jax_type_error")
  new_aval(params$new_sizes, a$dtype, a$weak)
}
reshape_p$lowering <- function(ctx, ins, ai, ao, p) {
  emit_op(ctx, "stablehlo.reshape", ins, tensor_type(ai[[1L]]), tensor_type(ao[[1L]]))
}
deflinear(reshape_p, function(ct, args, params) list(lax_reshape(ct, args[[1L]]$aval$shape)))
reshape_p$batch <- function(vals, dims, params, axis) {
  x <- moveaxis(vals[[1L]], dims[1L], 0L)
  n <- aval_of(x)$shape[1L]
  list(bind(reshape_p, list(x), list(new_sizes = c(n, params$new_sizes))), 0L)
}

#' Reshape an array (C order)
#' @param operand Array.
#' @param new_sizes New shape.
#' @param dimensions Optional permutation applied before reshaping.
#' @export
lax_reshape <- function(operand, new_sizes, dimensions = NULL) {
  x <- as_jax(operand)
  if (!is.null(dimensions)) x <- lax_transpose(x, dimensions)
  new_sizes <- canon_shape(new_sizes)
  if (identical(aval_of(x)$shape, new_sizes)) return(x)
  bind(reshape_p, list(x), list(new_sizes = new_sizes))
}

py_tuple <- function(x) paste0("(", paste(x, collapse = ", "), if (length(x) == 1L) ",", ")")

squeeze_p <- new_primitive("squeeze")
squeeze_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  dims <- params$dimensions
  if (anyDuplicated(dims)) jax_error("dimensions are not unique: ", py_tuple(dims),
                                     class = "jax_value_error")
  if (any(dims < 0L | dims >= length(a$shape)))
    jax_error("dimensions outside range [0, ndim): ", py_tuple(dims), class = "jax_value_error")
  if (any(a$shape[dims + 1L] != 1L))
    jax_error("cannot select an axis to squeeze out which has size not equal to one, got ",
              "shape=", py_tuple(a$shape), " and dimensions=", py_tuple(dims),
              class = "jax_value_error")
  keep <- !(seq_along(a$shape) - 1L) %in% dims
  new_aval(a$shape[keep], a$dtype, a$weak)
}
squeeze_p$lowering <- function(ctx, ins, ai, ao, p) {
  emit_op(ctx, "stablehlo.reshape", ins, tensor_type(ai[[1L]]), tensor_type(ao[[1L]]))
}
deflinear(squeeze_p, function(ct, args, params) list(lax_expand_dims(ct, params$dimensions)))
squeeze_p$batch <- function(vals, dims, params, axis) {
  x <- moveaxis(vals[[1L]], dims[1L], 0L)
  list(bind(squeeze_p, list(x), list(dimensions = params$dimensions + 1L)), 0L)
}

#' Remove size one dimensions
#' @param array Array.
#' @param dimensions 0-based dimensions to remove.
#' @export
lax_squeeze <- function(array, dimensions) {
  x <- as_jax(array)
  dims <- sort(norm_axis(dimensions, length(aval_of(x)$shape)))
  if (!length(dims)) return(x)
  bind(squeeze_p, list(x), list(dimensions = dims))
}

# Identity primitives: copy_p (array copies, kept in jaxprs), stage_p (lifts a value into
# the current trace) and device_put_p (one device: transfers are identities).
identity_primitive <- function(name, multiple_results = FALSE) {
  p <- new_primitive(name, multiple_results)
  p$abstract_eval <- function(avals, params) if (multiple_results) avals else avals[[1L]]
  p$lowering <- function(ctx, ins, ai, ao, params) if (multiple_results) ins else ins[[1L]]
  p$batch <- function(vals, dims, params, axis) {
    out <- bind(p, vals, params)
    if (multiple_results) list(out, dims) else list(out, dims[1L])
  }
  p
}
copy_p <- identity_primitive("copy")
deflinear(copy_p, function(ct, args, params) list(bind(copy_p, list(ct))))
stage_p <- identity_primitive("stage")
stage_p$impl <- function(args, params) args[[1L]]
deflinear(stage_p, function(ct, args, params) list(ct))
device_put_p <- identity_primitive("device_put", multiple_results = TRUE)
device_put_p$impl <- function(args, params) args
deflinear(device_put_p, function(cts, args, params) cts)

#' Lift a value into the current trace
#'
#' Port of `jax.lax.stage`: logically the identity. Outside any transformation it returns
#' a jax array; inside one it returns a tracer of the current trace.
#' @param x An array or R value.
#' @export
lax_stage <- function(x) bind(stage_p, list(as_jax(x)))

transpose_p <- new_primitive("transpose")
transpose_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  perm <- params$permutation
  if (!setequal(perm, seq_along(a$shape) - 1L) || length(perm) != length(a$shape))
    jax_error("transpose permutation isn't a permutation of operand dimensions, got permutation ",
              py_tuple(perm), " for operand shape ", py_tuple(a$shape), ".", class = "jax_type_error")
  new_aval(a$shape[perm + 1L], a$dtype, a$weak)
}
transpose_p$lowering <- function(ctx, ins, ai, ao, p) {
  emit_op(ctx, "stablehlo.transpose", ins, tensor_type(ai[[1L]]), tensor_type(ao[[1L]]),
          attrs = list(permutation = attr_i64_array(p$permutation)))
}
deflinear(transpose_p, function(ct, args, params) list(lax_transpose(ct, order(params$permutation) - 1L)))
transpose_p$batch <- function(vals, dims, params, axis) {
  d <- dims[1L]
  perm <- c(d, ifelse(params$permutation < d, params$permutation, params$permutation + 1L))
  list(bind(transpose_p, vals, list(permutation = as.integer(perm))), 0L)
}

#' Permute dimensions
#' @param operand Array.
#' @param permutation 0-based permutation.
#' @export
lax_transpose <- function(operand, permutation) {
  x <- as_jax(operand)
  perm <- as.integer(permutation)
  if (identical(perm, seq_along(perm) - 1L)) return(x)
  bind(transpose_p, list(x), list(permutation = perm))
}

rev_p <- new_primitive("rev")
rev_p$abstract_eval <- function(avals, params) avals[[1L]]
rev_p$lowering <- function(ctx, ins, ai, ao, p) {
  emit_op(ctx, "stablehlo.reverse", ins, tensor_type(ai[[1L]]), tensor_type(ao[[1L]]),
          attrs = list(dimensions = attr_i64_array(p$dimensions)))
}
deflinear(rev_p, function(ct, args, params) list(lax_rev(ct, params$dimensions)))
rev_p$batch <- function(vals, dims, params, axis) {
  d <- dims[1L]
  list(bind(rev_p, vals, list(dimensions = ifelse(params$dimensions >= d, params$dimensions + 1L,
                                                  params$dimensions))), d)
}

#' Reverse dimensions
#' @param operand Array.
#' @param dimensions 0-based dimensions.
#' @export
lax_rev <- function(operand, dimensions) {
  x <- as_jax(operand)
  dims <- norm_axis(dimensions, length(aval_of(x)$shape))
  if (!length(dims)) return(x)
  bind(rev_p, list(x), list(dimensions = as.integer(dims)))
}

# ---- select_n / clamp ------------------------------------------------------------------------

select_n_p <- new_primitive("select_n")
select_n_p$abstract_eval <- function(avals, params) {
  w <- avals[[1L]]
  cases <- avals[-1L]
  if (!length(cases)) jax_error("select must have at least one case", class = "jax_type_error")
  for (c in cases) if (!aval_eq(c, cases[[1L]]))
    jax_error("select cases must have the same shapes and dtypes, got [",
              paste(vapply(cases, aval_str, ""), collapse = ", "), "].", class = "jax_type_error")
  if (length(w$shape) && !identical(w$shape, cases[[1L]]$shape))
    jax_error("select `which` must be scalar or have the same shape as cases, got `which` shape ",
              py_tuple(w$shape), " but case shape ", py_tuple(cases[[1L]]$shape), ".",
              class = "jax_type_error")
  if (!(w$dtype == "bool" || is_int_dtype(w$dtype)))
    jax_error("select `which` must be boolean or integer type", class = "jax_type_error")
  if (w$dtype == "bool" && length(cases) > 2L)
    jax_error("select with boolean `which` cannot have > 2 cases.", class = "jax_type_error")
  out <- cases[[1L]]
  out$weak <- all(vapply(cases, `[[`, NA, "weak"))
  out
}
select_n_p$lowering <- function(ctx, ins, ai, ao, p) {
  o <- ao[[1L]]
  w <- ins[[1L]]
  wa <- ai[[1L]]
  cases <- ins[-1L]
  if (length(cases) == 1L) return(cases[[1L]])
  ot <- tensor_type(o)
  wa_full <- new_aval(o$shape, wa$dtype)
  if (!identical(wa$shape, o$shape)) {
    w <- hlo_broadcast(ctx, w, wa, wa_full, integer(0))
    wa <- wa_full
  }
  rec <- function(lo, hi) {
    if (hi - lo == 1L) return(cases[[lo + 1L]])
    mid <- (lo + hi) %/% 2L
    if (wa$dtype == "bool") {
      pred <- w
    } else {
      m <- full_constant(ctx, mid, wa)
      pred <- emit_op(ctx, "stablehlo.compare", c(w, m), rep(tensor_type(wa), 2L),
                      tensor_type(new_aval(o$shape, "bool")),
                      attrs = list(comparison_direction = "#stablehlo<comparison_direction LT>",
                                   compare_type = if (is_unsigned_dtype(wa$dtype))
                                     "#stablehlo<comparison_type UNSIGNED>" else
                                     "#stablehlo<comparison_type SIGNED>"))
    }
    a <- rec(lo, mid)
    b <- rec(mid, hi)
    # Port of lax._select_lower: hlo.select(which, cases[1], cases[0]) for bool which.
    emit_op(ctx, "stablehlo.select", if (wa$dtype == "bool") c(pred, b, a) else c(pred, a, b),
            c(tensor_type(new_aval(o$shape, "bool")), ot, ot), ot)
  }
  rec(0L, length(cases))
}
deflinear(select_n_p, function(ct, args, params) {
  which <- args[[1L]]
  cases <- args[-1L]
  out <- list(NULL)
  for (i in seq_along(cases)) {
    if (!is_undefined(cases[[i]])) { out[[i + 1L]] <- NULL; next }
    if (is_zero(ct)) { out[i + 1L] <- list(ct); next }
    z <- lax_full_like(ct, 0)
    parts <- rep(list(z), length(cases))
    parts[[i]] <- ct
    out[[i + 1L]] <- do.call(lax_select_n, c(list(which), parts))
  }
  length(out) <- length(args)
  out
}, linear = NULL)
select_n_p$jvp <- function(primals, tangents, params) {
  ans <- bind(select_n_p, primals)
  ts <- tangents[-1L]
  if (all(vapply(ts, is_zero, NA))) return(list(ans, zero_of(tangent_aval(aval_of(ans)))))
  ts <- lapply(ts, instantiate_zero)
  list(ans, bind(select_n_p, c(primals[1L], ts)))
}
select_n_p$batch <- function(vals, dims, params, axis) {
  n <- axis$size
  which <- vals[[1L]]
  cases <- vals[-1L]
  cdims <- dims[-1L]
  if (all(!is.na(cdims)) && length(unique(cdims)) == 1L && is.na(dims[1L]) &&
      length(aval_of(which)$shape) == 0L)
    return(list(bind(select_n_p, vals), cdims[1L]))
  cases <- Map(function(c, d) bdim_at_front(c, d, n), cases, cdims)
  cs <- aval_of(cases[[1L]])$shape
  if (is.na(dims[1L])) {
    if (length(aval_of(which)$shape)) which <- bdim_at_front(which, NA, n)
  } else {
    which <- moveaxis(which, dims[1L], 0L)
    if (length(aval_of(which)$shape) == 1L && length(cs) > 1L)
      which <- lax_broadcast_in_dim(which, cs, 0L)
  }
  list(bind(select_n_p, c(list(which), cases)), 0L)
}

#' Select between cases
#' @param which Boolean or int32 array (or scalar) selecting the case.
#' @param ... Case arrays.
#' @export
lax_select_n <- function(which, ...) {
  cases <- lax_args(...)
  bind(select_n_p, c(list(as_jax(which)), cases))
}

#' Elementwise select
#' @param pred Boolean array.
#' @param on_true,on_false Arrays.
#' @export
lax_select <- function(pred, on_true, on_false) lax_select_n(pred, on_false, on_true)

clamp_p <- new_primitive("clamp")
clamp_p$abstract_eval <- function(avals, params) {
  x <- avals[[2L]]
  # Ports of lax._clamp_shape_rule and lax._clamp_dtype_rule.
  for (i in c(1L, 3L)) {
    nm <- if (i == 1L) "min" else "max"
    b <- avals[[i]]
    if (length(b$shape) && !identical(b$shape, x$shape))
      jax_error("clamp requires ", nm, ".shape == operand.shape or ", nm, ".shape == (), got ", nm,
                ".shape=", py_tuple(b$shape), ", operand.shape=", py_tuple(x$shape), ".",
                class = "jax_type_error")
  }
  check_same_dtypes("clamp", avals)
  # The result is weakly typed only when all three operands are.
  new_aval(x$shape, x$dtype, all(vapply(avals, `[[`, NA, "weak")))
}
clamp_p$lowering <- function(ctx, ins, ai, ao, p) {
  o <- ao[[1L]]
  ops <- bcast_operands(ctx, ins, ai, o)
  emit_op(ctx, "stablehlo.clamp", ops, rep(tensor_type(o), 3L), tensor_type(o))
}
defjvp(clamp_p,
  function(g, p, params) {
    mn <- p[[1L]]; x <- p[[2L]]; mx <- p[[3L]]
    lax_select(lax_bitwise_and(lax_gt(mn, x), lax_lt(mn, mx)), lax_broadcast_like(g, x), lax_full_like(x, 0))
  },
  function(g, p, params) {
    mn <- p[[1L]]; x <- p[[2L]]; mx <- p[[3L]]
    lax_select(lax_bitwise_and(lax_gt(x, mn), lax_lt(x, mx)), g, lax_full_like(x, 0))
  },
  function(g, p, params) {
    x <- p[[2L]]; mx <- p[[3L]]
    lax_select(lax_lt(mx, x), lax_broadcast_like(g, x), lax_full_like(x, 0))
  })
clamp_p$batch <- function(vals, dims, params, axis) {
  n <- axis$size
  x <- bdim_at_front(vals[[2L]], dims[2L], n)
  xs <- aval_of(x)$shape
  b <- function(v, d) {
    if (is.na(d) && length(aval_of(v)$shape) == 0L) return(v)
    broadcast_to_shape(bdim_at_front(v, d, n), xs)
  }
  list(bind(clamp_p, list(b(vals[[1L]], dims[1L]), x, b(vals[[3L]], dims[3L]))), 0L)
}

#' Clamp values
#' @param min,x,max Arrays (min and max may be scalars).
#' @export
lax_clamp <- function(min, x, max) bind(clamp_p, lax_args(min, x, max)[c(1L, 2L, 3L)])

# ---- iota / full ------------------------------------------------------------------------------

iota_p <- new_primitive("iota")
# Port of lax._iota_abstract_eval.
iota_p$abstract_eval <- function(avals, params) {
  if (!kind_ok(params$dtype, "number"))
    jax_error("iota does not accept dtype ", params$dtype, ". Accepted dtypes are subtypes of ",
              "floating, complexfloating, integer.", class = "jax_type_error")
  new_aval(params$shape, params$dtype)
}
iota_p$lowering <- function(ctx, ins, ai, ao, p) {
  emit_op(ctx, "stablehlo.iota", character(0), character(0), tensor_type(ao[[1L]]),
          attrs = list(iota_dimension = attr_i64(p$dimension)))
}

#' Iota (0, 1, 2, ...)
#' @param dtype Dtype.
#' @param size Length.
#' @export
lax_iota <- function(dtype, size) lax_broadcasted_iota(dtype, size, 0L)

#' Broadcasted iota
#' @param dtype Dtype.
#' @param shape Shape.
#' @param dimension 0-based dimension that counts.
#' @export
lax_broadcasted_iota <- function(dtype, shape, dimension) {
  shape <- canon_shape(shape)
  bind(iota_p, list(), list(dtype = dtype(dtype), shape = shape,
                            dimension = norm_axis(dimension, length(shape))))
}

#' Array filled with a value
#' @param shape Shape.
#' @param fill_value Scalar.
#' @param dtype Optional dtype.
#' @export
lax_full <- function(shape, fill_value, dtype = NULL) {
  v <- as_jax(fill_value)
  if (length(aval_of(v)$shape))
    jax_error("full must be called with scalar fill_value, got fill_value.shape ",
              py_tuple(aval_of(v)$shape), ".", class = "jax_type_error")
  if (is.null(dtype)) return(lax_broadcast(v, canon_shape(shape)))
  # An R fill value is cast on the host like a NumPy cast (-1 as uint8 is 255); a complex
  # value keeps its real part in a real dtype.
  if (is.complex(fill_value) && !is_complex_dtype(dtype(dtype))) fill_value <- Re(fill_value)
  lax_full_like(v, fill_value, dtype = dtype, shape = shape)
}

#' Array like x filled with a value
#' @param x Array.
#' @param fill_value Scalar.
#' @param dtype Optional dtype.
#' @param shape Optional shape.
#' @export
lax_full_like <- function(x, fill_value, dtype = NULL, shape = NULL) {
  a <- aval_of(x)
  dt <- if (is.null(dtype)) a$dtype else dtype(dtype)
  sh <- if (is.null(shape)) a$shape else canon_shape(shape)
  if (dt == "float0") return(zeros_aval(new_aval(sh, "float0")))
  weak <- is.null(dtype) && a$weak
  if (is_jax_value(fill_value)) {
    # An array (or traced) fill value is converted on the device.
    if (length(aval_of(fill_value)$shape))
      jax_error("full must be called with scalar fill_value, got fill_value.shape ",
                py_tuple(aval_of(fill_value)$shape), ".", class = "jax_type_error")
    v <- lax_convert_element_type(fill_value, dt, weak_type = weak)
    return(if (length(sh)) lax_broadcast(v, sh) else v)
  }
  if (length(r_shape(fill_value)))
    jax_error("full must be called with scalar fill_value, got fill_value.shape ",
              py_tuple(r_shape(fill_value)), ".", class = "jax_type_error")
  k <- dtype_kind(dt)
  host <- if (k == "c") as.complex(fill_value)
          else if (dt == "bool") !isFALSE(as.logical(fill_value))   # NaN is true
          else as.double(fill_value)
  if (k %in% c("i", "u") && !is.finite(host)) {
    # A non finite fill of an integer array is converted on the device, like the Python
    # float it stands for (infinities saturate).
    v <- lax_convert_element_type(new_array(new_aval(integer(0), default_float()), host = host), dt,
                                  weak_type = weak)
    return(if (length(sh)) lax_broadcast(v, sh) else v)
  }
  if (k %in% c("i", "u")) {
    host <- trunc(host)
    b <- dtype_bits(dt)
    # Integer fill values wrap like NumPy casts (-1 as uint8 is 255, 128 as int8 is -128).
    if (b <= 32L && is.finite(host)) {
      host <- host %% 2^b
      if (k == "i" && host >= 2^(b - 1L)) host <- host - 2^b
    }
  }
  sa <- new_aval(integer(0), dt, weak)
  v <- if (is_ext_dtype(dt)) {
    # The transfer code has no narrow types: convert a wide value on the device.
    wide <- new_array(new_aval(integer(0), if (k == "f") "float64" else "int32"), host = host)
    lax_convert_element_type(wide, dt, weak_type = weak)
  } else if (host_value_exact(host, dt)) new_array(sa, host = host)
  else new_array(sa, buf = buffer_from_r(host, dt, integer(0)))
  if (!length(sh)) return(v)
  lax_broadcast(v, sh)
}

host_value_exact <- function(h, dt) {
  dt %in% c("float64", "complex128", "bool", "int32", "int64", "uint32", "int16", "int8",
            "uint8", "uint16")
}

# ---- slicing ----------------------------------------------------------------------------------

slice_p <- new_primitive("slice")
slice_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  st <- params$start_indices
  li <- params$limit_indices
  sd <- params$strides %||% rep(1L, length(st))
  # Port of slicing._slice_shape_rule. JAX reports a length mismatch as a ValueError
  # (from safe_map, before the shape rule runs), so the class follows that.
  if (length(st) != length(a$shape))
    jax_error("slice start_indices must have length equal to the number of dimensions of the ",
              "operand, got indices ", py_tuple(st), " for operand shape ", py_tuple(a$shape), ".",
              class = "jax_value_error")
  if (length(li) != length(st))
    jax_error("slice limit_indices must have the same length as start_indices, got start_indices ",
              py_tuple(st), " and limit_indices ", py_tuple(li), ".", class = "jax_value_error")
  if (any(li > a$shape))
    jax_error("slice limit_indices must be less than or equal to operand shape, got limit_indices ",
              py_tuple(li), " for operand shape ", py_tuple(a$shape), ".", class = "jax_type_error")
  if (any(st < 0L))
    jax_error("slice start_indices must be greater than or equal to zero, got start_indices of ",
              py_tuple(st), ".", class = "jax_type_error")
  if (any(li < st))
    jax_error("slice limit_indices must be greater than or equal to start_indices, got ",
              "start_indices ", py_tuple(st), " and limit_indices ", py_tuple(li), ".",
              class = "jax_type_error")
  if (length(sd) != length(a$shape))
    jax_error("slice strides must have length equal to the number of dimensions of the operand, ",
              "got strides ", py_tuple(sd), " for operand shape ", py_tuple(a$shape), ".",
              class = "jax_type_error")
  if (any(sd <= 0L))
    jax_error("slice strides must be positive, got ", py_tuple(sd), class = "jax_type_error")
  new_aval(as.integer(ifelse(li > st, (li - st + sd - 1L) %/% sd, 0L)), a$dtype, a$weak)
}
slice_p$lowering <- function(ctx, ins, ai, ao, p) {
  sd <- p$strides %||% rep(1L, length(p$start_indices))
  emit_op(ctx, "stablehlo.slice", ins, tensor_type(ai[[1L]]), tensor_type(ao[[1L]]),
          attrs = list(start_indices = attr_i64_array(p$start_indices),
                       limit_indices = attr_i64_array(p$limit_indices),
                       strides = attr_i64_array(sd)))
}
deflinear(slice_p, function(ct, args, params) {
  a <- args[[1L]]$aval
  st <- params$start_indices
  sd <- params$strides %||% rep(1L, length(st))
  cs <- aval_of(ct)$shape
  real_limits <- st + ifelse(cs == 0L, 0L, (cs - 1L) * sd + 1L)
  cfg <- lapply(seq_along(st), function(i) c(st[i], a$shape[i] - real_limits[i], sd[i] - 1L))
  list(lax_pad(ct, lax_full_like(lax_full(integer(0), 0, aval_of(ct)$dtype), 0), cfg))
})
slice_p$batch <- function(vals, dims, params, axis) {
  d <- dims[1L]
  x <- vals[[1L]]
  n <- aval_of(x)$shape[d + 1L]
  ins <- function(v, val) append(v, val, after = d)
  sd <- params$strides %||% rep(1L, length(params$start_indices))
  list(bind(slice_p, list(x), list(start_indices = ins(params$start_indices, 0L),
                                    limit_indices = ins(params$limit_indices, n),
                                    strides = ins(sd, 1L))), d)
}

#' Static slice
#' @param operand Array.
#' @param start_indices,limit_indices 0-based start (inclusive) and limit (exclusive).
#' @param strides Optional strides.
#' @export
lax_slice <- function(operand, start_indices, limit_indices, strides = NULL) {
  bind(slice_p, list(as_jax(operand)),
       list(start_indices = as.integer(start_indices), limit_indices = as.integer(limit_indices),
            strides = if (is.null(strides)) NULL else as.integer(strides)))
}

#' Slice along one axis
#' @param operand Array.
#' @param start_index,limit_index Bounds.
#' @param stride Stride.
#' @param axis 0-based axis.
#' @export
lax_slice_in_dim <- function(operand, start_index, limit_index, stride = 1L, axis = 0L) {
  x <- as_jax(operand)
  s <- aval_of(x)$shape
  ax <- norm_axis(axis, length(s))
  st <- rep(0L, length(s))
  li <- s
  sd <- rep(1L, length(s))
  # NULL bounds are the ends of the axis; negative bounds count from its end.
  n <- s[ax + 1L]
  lo <- if (is.null(start_index)) 0L else as.integer(start_index)
  hi <- if (is.null(limit_index)) n else as.integer(limit_index)
  st[ax + 1L] <- if (lo < 0L) lo + n else lo
  li[ax + 1L] <- if (hi < 0L) hi + n else hi
  sd[ax + 1L] <- as.integer(stride)
  lax_slice(x, st, li, sd)
}

#' Index along one axis
#' @param operand Array.
#' @param index 0-based index.
#' @param axis 0-based axis.
#' @param keepdims Keep the indexed dimension.
#' @export
lax_index_in_dim <- function(operand, index, axis = 0L, keepdims = TRUE) {
  x <- as_jax(operand)
  s <- aval_of(x)$shape
  ax <- norm_axis(axis, length(s))
  index <- as.integer(index)
  if (index < 0L) index <- index + s[ax + 1L]
  out <- lax_slice_in_dim(x, index, index + 1L, 1L, ax)
  if (keepdims) out else lax_squeeze(out, ax)
}

lax_stack0 <- function(xs) {
  lax_concatenate(lapply(xs, function(x) lax_expand_dims(x, 0L)), 0L)
}

# The index operands of dynamic_slice and dynamic_update_slice share one integer dtype.
check_index_dtypes <- function(name, idx, got) {
  dts <- vapply(idx, `[[`, "", "dtype")
  if (length(unique(dts)) > 1L || !all(vapply(dts, is_int_dtype, NA)))
    jax_error("index arguments to ", name, " must be integers of the same type, ", got,
              paste(dts, collapse = ", "), class = "jax_type_error")
  invisible()
}

dynamic_slice_p <- new_primitive("dynamic_slice")
dynamic_slice_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  idx <- avals[-1L]
  if (length(idx) != length(a$shape))
    jax_error("Length of slice indices must match number of operand dimensions (", length(idx),
              " vs ", py_tuple(a$shape), ")", class = "jax_value_error")
  # Ports of slicing._dynamic_slice_shape_rule and slicing._dynamic_slice_dtype_rule.
  ss <- params$slice_sizes
  if (length(ss) != length(a$shape) || any(ss > a$shape))
    jax_error("slice slice_sizes must be less than or equal to operand shape, got slice_sizes ",
              py_tuple(ss), " for operand shape ", py_tuple(a$shape), ".", class = "jax_type_error")
  if (any(ss < 0L))
    jax_error("slice slice_sizes must be greater than or equal to zero, got slice_sizes of ",
              py_tuple(ss), ".", class = "jax_type_error")
  if (any(vapply(idx, function(i) length(i$shape) > 0L, NA)))
    jax_error("start_indices arguments to dynamic_slice must be scalars", class = "jax_type_error")
  check_index_dtypes("dynamic_slice", idx, "got: ")
  new_aval(params$slice_sizes, a$dtype, a$weak)
}
dynamic_slice_p$lowering <- function(ctx, ins, ai, ao, p) {
  emit_op(ctx, "stablehlo.dynamic_slice", ins, types_of(ai), tensor_type(ao[[1L]]),
          attrs = list(slice_sizes = attr_i64_array(p$slice_sizes)))
}
deflinear(dynamic_slice_p, function(ct, args, params) {
  a <- args[[1L]]$aval
  z <- lax_full(a$shape, 0, a$dtype)
  c(list(do.call(lax_dynamic_update_slice_raw, c(list(z, ct), args[-1L]))),
    rep(list(NULL), length(args) - 1L))
}, linear = 1L)
# Batching rules for dynamic_slice and dynamic_update_slice are installed by
# R/lax_slicing.R (ports of slicing._dynamic_slice_batching_rule and friends).


clamp_start_indices <- function(operand_shape, idx, sizes) {
  lapply(seq_along(idx), function(i) as_jax(idx[[i]], NULL))
}

#' Dynamic slice
#' @param operand Array.
#' @param start_indices List (or vector) of 0-based scalar start indices.
#' @param slice_sizes Sizes.
#' @param allow_negative_indices Whether negative start indices count from the end of
#'   the axis (one flag, or one per axis). When `FALSE` a negative R number is an error
#'   and a negative array index is clamped like any other out of bounds index.
#' @export
lax_dynamic_slice <- function(operand, start_indices, slice_sizes,
                              allow_negative_indices = TRUE) {
  x <- as_jax(operand)
  idx <- normalize_start_indices(start_indices, aval_of(x)$shape, allow_negative_indices)
  bind(dynamic_slice_p, c(list(x), idx), list(slice_sizes = canon_shape(slice_sizes)))
}

normalize_start_indices <- function(start_indices, shape, allow_negative = TRUE) {
  idx <- if (is.list(start_indices)) start_indices
         else if (is_jax_value(start_indices)) {
           if (length(aval_of(start_indices)$shape) != 1L)
             jax_error("Slice indices must be a 1D sequence, got ",
                       py_tuple(aval_of(start_indices)$shape), class = "jax_value_error")
           lapply(seq_len(aval_of(start_indices)$shape) - 1L, function(i)
             lax_index_in_dim(start_indices, i, 0L, keepdims = FALSE))
         } else as.list(start_indices)
  if (length(idx) != length(shape))
    jax_error("Length of slice indices must match number of operand dimensions (",
              length(idx), " vs ", py_tuple(shape), ")", class = "jax_value_error")
  if (!length(allow_negative) %in% c(1L, length(idx)))
    jax_error("allow_negative_indices must be one flag or one flag per axis, got ",
              length(allow_negative), " flags for ", length(idx), " axes", class = "jax_value_error")
  allow_negative <- rep_len(as.logical(allow_negative), length(idx))
  for (j in seq_along(idx)) {
    i <- idx[[j]]
    if (!allow_negative[j] && !is_jax_value(i) && is.numeric(i) && length(i) == 1L && i < 0)
      jax_error("Index ", i, " is out of bounds for dimension ", shape[j],
                " if allow_negative_indices=False", class = "jax_value_error")
  }
  idx <- lapply(idx, function(i) {
    i <- as_jax(i)
    if (!is_int_dtype(aval_of(i)$dtype))
      jax_error("index arguments to dynamic_slice must be integers of the same type, got: ",
                aval_of(i)$dtype, class = "jax_type_error")
    i
  })
  # R scalars adopt the dtype of the array indices; array indices of different integer
  # dtypes are an error, as in JAX.
  idx <- do.call(lax_args, unname(idx))
  # negative indices wrap around, as in jax.lax.dynamic_slice
  Map(function(i, n, neg) {
    if (!neg) return(i)
    if (is_jax_array(i) && length(aval_of(i)$shape) == 0L) {
      v <- as.numeric(array_host(i))
      if (v < 0) return(lax_convert_element_type(as_jax(v + n), aval_of(i)$dtype))
      return(i)
    }
    lax_select(lax_lt(i, lax_full_like(i, 0)), lax_add(i, lax_full_like(i, n)), i)
  }, idx, as.list(shape), allow_negative)
}

dynamic_update_slice_p <- new_primitive("dynamic_update_slice")
dynamic_update_slice_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  u <- avals[[2L]]
  idx <- avals[-(1:2)]
  # Ports of slicing._dynamic_update_slice_shape_rule and _dynamic_update_slice_dtype_rule.
  if (length(u$shape) != length(a$shape))
    jax_error("dynamic_update_slice update must have the same rank as operand, got update shape ",
              py_tuple(u$shape), " for operand shape ", py_tuple(a$shape), ".", class = "jax_type_error")
  if (length(idx) != length(a$shape))
    jax_error("dynamic_update_slice start_indices must have length equal to the rank of operand, ",
              "got ", length(idx), " indices for operand shape ", py_tuple(a$shape), ".",
              class = "jax_type_error")
  if (any(u$shape > a$shape))
    jax_error("dynamic_update_slice update shape must be smaller than operand shape, got update ",
              "shape ", py_tuple(u$shape), " for operand shape ", py_tuple(a$shape), ".",
              class = "jax_type_error")
  if (any(vapply(idx, function(i) length(i$shape) > 0L, NA)))
    jax_error("start_indices arguments to dynamic_update_slice must be scalars",
              class = "jax_type_error")
  check_same_dtypes("dynamic_update_slice", list(a, u))
  check_index_dtypes("dynamic_update_slice", idx, "got ")
  a
}
dynamic_update_slice_p$lowering <- function(ctx, ins, ai, ao, p) {
  emit_op(ctx, "stablehlo.dynamic_update_slice", ins, types_of(ai), tensor_type(ao[[1L]]))
}
deflinear(dynamic_update_slice_p, function(ct, args, params) {
  idx <- args[-(1:2)]
  ua <- arg_aval(args[[2L]])
  cx <- if (is_undefined(args[[1L]]))
    do.call(lax_dynamic_update_slice_raw, c(list(ct, lax_full(ua$shape, 0, ua$dtype)), idx)) else NULL
  cu <- if (is_undefined(args[[2L]]))
    bind(dynamic_slice_p, c(list(ct), idx), list(slice_sizes = ua$shape)) else NULL
  c(list(cx, cu), rep(list(NULL), length(idx)))
}, linear = 1:2)
lax_dynamic_update_slice_raw <- function(operand, update, ...) {
  bind(dynamic_update_slice_p, c(list(operand, update), list(...)))
}

#' Dynamic update slice
#' @param operand Array.
#' @param update Array written into operand.
#' @param start_indices 0-based start indices.
#' @param allow_negative_indices Whether negative start indices count from the end of
#'   the axis (one flag, or one per axis).
#' @export
lax_dynamic_update_slice <- function(operand, update, start_indices,
                                     allow_negative_indices = TRUE) {
  args <- lax_args(operand, update)
  idx <- normalize_start_indices(start_indices, aval_of(args[[1L]])$shape,
                                 allow_negative_indices)
  bind(dynamic_update_slice_p, c(args, idx))
}

# ---- concatenate / pad ------------------------------------------------------------------------

concatenate_p <- new_primitive("concatenate")
concatenate_p$abstract_eval <- function(avals, params) {
  d <- params$dimension
  a1 <- avals[[1L]]
  shapes_str <- function() paste(vapply(avals, function(z) py_tuple(z$shape), ""), collapse = ", ")
  if (length(unique(vapply(avals, function(z) length(z$shape), 1L))) != 1L)
    jax_error("Cannot concatenate arrays with different numbers of dimensions: got ", shapes_str(),
              ".", class = "jax_type_error")
  if (d < 0L || d >= length(a1$shape))
    jax_error("concatenate dimension out of bounds: dimension ", d, " for shapes ", shapes_str(),
              ".", class = "jax_type_error")
  check_same_dtypes("concatenate", avals)
  for (a in avals) {
    if (any(a$shape[-(d + 1L)] != a1$shape[-(d + 1L)]))
      jax_error("Cannot concatenate arrays with shapes that differ in dimensions other than the ",
                "one being concatenated: concatenating along dimension ", d, " for shapes ",
                paste(vapply(avals, function(z) paste0("(", paste(z$shape, collapse = ", "), ")"), ""),
                      collapse = ", "), ".", class = "jax_type_error")
  }
  s <- a1$shape
  s[d + 1L] <- sum(vapply(avals, function(a) a$shape[d + 1L], 1L))
  new_aval(s, a1$dtype, all(vapply(avals, `[[`, NA, "weak")))
}
concatenate_p$lowering <- function(ctx, ins, ai, ao, p) {
  if (length(ins) == 1L) return(ins[[1L]])
  emit_op(ctx, "stablehlo.concatenate", ins, types_of(ai), tensor_type(ao[[1L]]),
          attrs = list(dimension = attr_i64(p$dimension)))
}
deflinear(concatenate_p, function(ct, args, params) {
  d <- params$dimension
  off <- 0L
  lapply(args, function(a) {
    s <- arg_aval(a)$shape
    n <- s[d + 1L]
    r <- if (is_undefined(a)) {
      if (is_zero(ct)) zero_of(a$aval) else lax_slice_in_dim(ct, off, off + n, 1L, d)
    } else NULL
    off <<- off + n
    r
  })
})
concatenate_p$batch <- function(vals, dims, params, axis) {
  n <- axis$size
  vals <- Map(function(v, d) bdim_at_front(v, d, n), vals, dims)
  list(bind(concatenate_p, vals, list(dimension = params$dimension + 1L)), 0L)
}

#' Concatenate arrays
#' @param operands List of arrays.
#' @param dimension 0-based dimension.
#' @export
lax_concatenate <- function(operands, dimension) {
  xs <- do.call(lax_args, operands)
  if (!length(xs)) jax_error("concatenate requires at least one operand")
  d <- norm_axis(dimension, length(aval_of(xs[[1L]])$shape))
  if (length(xs) == 1L) return(xs[[1L]])
  bind(concatenate_p, xs, list(dimension = d))
}

pad_p <- new_primitive("pad")
pad_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  # Ports of lax._pad_dtype_rule and lax._pad_shape_rule.
  cfg <- params$padding_config
  cfg_str <- function() py_tuple(vapply(cfg, py_tuple, ""))
  if (length(avals[[2L]]$shape))
    jax_error("padding_value must be a scalar; got np.shape(padding_value)=",
              py_tuple(avals[[2L]]$shape), class = "jax_value_error")
  if (length(cfg) != length(a$shape))
    jax_error("length of padding_config must equal the number of axes of operand, got ",
              "padding_config ", cfg_str(), " for operand shape ", py_tuple(a$shape),
              class = "jax_value_error")
  if (any(vapply(cfg, function(c) c[3L] < 0L, NA)))
    jax_error("interior padding in padding_config must be nonnegative, got padding_config ",
              cfg_str(), class = "jax_value_error")
  if (avals[[2L]]$dtype != a$dtype)
    jax_error("pad operand and padding_value must be same dtype: got ", a$dtype, " and ",
              avals[[2L]]$dtype, ".", class = "jax_type_error")
  s <- vapply(seq_along(cfg), function(i) {
    lo <- cfg[[i]][1L]; hi <- cfg[[i]][2L]; it <- cfg[[i]][3L]
    n <- a$shape[i]
    as.integer(lo + hi + n + max(0L, n - 1L) * it)
  }, 1L)
  if (any(s < 0L))
    jax_error("Dimension size after padding is not at least 0, got result shape ", py_tuple(s),
              ", for padding_config ", cfg_str(), " and operand shape ", py_tuple(a$shape),
              class = "jax_value_error")
  new_aval(s, a$dtype, a$weak)
}
pad_p$lowering <- function(ctx, ins, ai, ao, p) {
  cfg <- p$padding_config
  emit_op(ctx, "stablehlo.pad", ins, types_of(ai), tensor_type(ao[[1L]]),
          attrs = list(edge_padding_low = attr_i64_array(vapply(cfg, `[`, 1L, 1L)),
                       edge_padding_high = attr_i64_array(vapply(cfg, `[`, 1L, 2L)),
                       interior_padding = attr_i64_array(vapply(cfg, `[`, 1L, 3L))))
}
deflinear(pad_p, function(ct, args, params) {
  cfg <- params$padding_config
  ca <- aval_of(ct)
  total <- function(x) lax_reduce_sum(x, seq_along(aval_of(x)$shape) - 1L)
  t_op <- function() {
    unpad <- lapply(cfg, function(c) c(-c[1L], -c[2L], 0L))
    up <- lax_pad(ct, lax_full(integer(0), 0, ca$dtype), unpad)
    us <- aval_of(up)$shape
    lax_slice(up, rep(0L, length(us)), us, vapply(cfg, `[`, 1L, 3L) + 1L)
  }
  if (is_zero(ct)) return(list(ct, ct))
  t_operand <- if (is_undefined(args[[1L]])) t_op() else NULL
  t_padv <- if (is_undefined(args[[2L]])) {
    lax_sub(total(ct), total(t_operand %||% t_op()))
  } else NULL
  list(t_operand, t_padv)
})
pad_p$batch <- function(vals, dims, params, axis) {
  if (is.na(dims[2L])) {
    d <- dims[1L]
    cfg <- append(params$padding_config, list(c(0L, 0L, 0L)), after = d)
    return(list(bind(pad_p, vals, list(padding_config = cfg)), d))
  }
  fallback_batch_rule(pad_p)(vals, dims, params, axis)
}
allow_fallback_batching("pad")

#' Pad an array
#' @param operand Array.
#' @param padding_value Scalar.
#' @param padding_config List of `c(low, high, interior)` per dimension.
#' @export
lax_pad <- function(operand, padding_value, padding_config) {
  args <- lax_args(operand, padding_value)
  cfg <- lapply(padding_config, function(c) as.integer(c))
  bind(pad_p, args, list(padding_config = cfg))
}
