# lax reductions, generic reduce, argmin/argmax and dot_general.

reduce_aval <- function(a, axes, dtype = NULL) {
  keep <- setdiff(seq_along(a$shape) - 1L, axes)
  new_aval(a$shape[keep + 1L], dtype %||% a$dtype, a$weak && is.null(dtype))
}

# `op` is a StableHLO operation name or a lowering function (see minmax_lowering).
reduce_region <- function(ctx, scalar_t, op, aval = NULL) {
  build_region(ctx, c(scalar_t, scalar_t), function(sub, args) {
    v <- if (is.function(op)) op(sub, args, list(aval, aval), list(aval), NULL)
         else emit_op(sub, op, args, c(scalar_t, scalar_t), scalar_t)
    list(values = v, types = scalar_t)
  })
}

def_reduce <- function(name, hlo, init, kinds = "any") {
  p <- new_primitive(name)
  p$abstract_eval <- function(avals, params) {
    a <- avals[[1L]]
    if (!kind_ok(a$dtype, kinds))
      jax_error(name, " does not accept dtype ", a$dtype, class = "jax_type_error")
    reduce_aval(a, params$axes)
  }
  p$lowering <- function(ctx, ins, ai, ao, params) {
    a <- ai[[1L]]
    sa <- new_aval(integer(0), a$dtype)
    st <- tensor_type(sa)
    minmax <- hlo %in% c("stablehlo.maximum", "stablehlo.minimum")
    # Complex extremes are lexicographic, as in lax.max and lax.min.
    op <- if (minmax && is_complex_dtype(a$dtype))
      minmax_lowering(hlo, if (hlo == "stablehlo.maximum") "GT" else "LT") else hlo
    # 64 bit integer extremes are not representable as R doubles (see lax_windowed.R).
    c0 <- if (is_int_dtype(a$dtype) && dtype_bits(a$dtype) == 64L && minmax)
      identity_constant(ctx, if (hlo == "stablehlo.maximum") "max" else "min", a$dtype)
    else ir_constant(ctx, init(a$dtype), sa)
    emit_op(ctx, "stablehlo.reduce", c(ins[[1L]], c0), c(tensor_type(a), st),
            tensor_type(ao[[1L]]), attrs = list(dimensions = attr_i64_array(params$axes)),
            regions = reduce_region(ctx, st, op, sa))
  }
  p$batch <- function(vals, dims, params, axis) {
    d <- dims[1L]
    axes <- ifelse(params$axes >= d, params$axes + 1L, params$axes)
    list(bind(p, vals, list(axes = as.integer(axes))), d - sum(params$axes < d))
  }
  p
}

# Port of dtypes.supports_inf: these formats of ml_dtypes have no infinity, so the
# identities of max and min are their extreme finite values.
supports_inf <- function(dt) {
  !dt %in% c("float8_e4m3b11fnuz", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2fnuz")
}
# The 4 and 6 bit formats have no infinity either: JAX builds the identity with NumPy,
# which stores an infinity as their extreme finite value.
finite_identity <- function(dt) {
  !supports_inf(dt) || dt %in% c("float4_e2m1fn", "float6_e2m3fn", "float6_e3m2fn")
}
identity_max <- function(dt) {
  k <- dtype_kind(dt)
  if (k == "b") FALSE
  else if (k %in% c("f", "c")) { if (finite_identity(dt)) finfo(dt)$min else -Inf }
  else iinfo(dt)$min
}
identity_min <- function(dt) {
  k <- dtype_kind(dt)
  if (k == "b") TRUE
  else if (k %in% c("f", "c")) { if (finite_identity(dt)) finfo(dt)$max else Inf }
  else iinfo(dt)$max
}

reduce_sum_p <- def_reduce("reduce_sum", "stablehlo.add", function(dt) 0, "number")
reduce_prod_p <- def_reduce("reduce_prod", "stablehlo.multiply", function(dt) 1, "number")
reduce_max_p <- def_reduce("reduce_max", "stablehlo.maximum", identity_max)
reduce_min_p <- def_reduce("reduce_min", "stablehlo.minimum", identity_min)
reduce_and_p <- def_reduce("reduce_and", "stablehlo.and",
                           function(dt) if (dt == "bool") TRUE else if (is_unsigned_dtype(dt)) iinfo(dt)$max else -1,
                           "intbool")
reduce_or_p <- def_reduce("reduce_or", "stablehlo.or", function(dt) FALSE, "intbool")
reduce_xor_p <- def_reduce("reduce_xor", "stablehlo.xor", function(dt) FALSE, "intbool")

deflinear(reduce_sum_p, function(ct, args, params) {
  a <- args[[1L]]$aval
  keep <- setdiff(seq_along(a$shape) - 1L, params$axes)
  list(lax_broadcast_in_dim(ct, a$shape, keep))
})

reduce_chooser_jvp <- function(prim) {
  function(primals, tangents, params) {
    x <- primals[[1L]]
    ans <- bind(prim, primals, params)
    g <- tangents[[1L]]
    if (is_zero(g)) return(list(ans, zero_of(tangent_aval(aval_of(ans)))))
    a <- aval_of(x)
    keep <- setdiff(seq_along(a$shape) - 1L, params$axes)
    loc <- lax_convert_element_type(lax_eq(x, lax_broadcast_in_dim(ans, a$shape, keep)),
                                    aval_of(g)$dtype)
    counts <- lax_reduce_sum(loc, params$axes)
    list(ans, lax_div(lax_reduce_sum(lax_mul(g, loc), params$axes), counts))
  }
}
reduce_max_p$jvp <- reduce_chooser_jvp(reduce_max_p)
reduce_min_p$jvp <- reduce_chooser_jvp(reduce_min_p)

# d prod = sum_i g_i prod_{j != i} x_j, computed with exclusive cumulative products so
# that zeros are handled exactly.
reduce_prod_p$jvp <- function(primals, tangents, params) {
  x <- primals[[1L]]
  ans <- bind(reduce_prod_p, primals, params)
  g <- tangents[[1L]]
  if (is_zero(g)) return(list(ans, zero_of(tangent_aval(aval_of(ans)))))
  a <- aval_of(x)
  axes <- params$axes
  keep <- setdiff(seq_along(a$shape) - 1L, axes)
  perm <- c(keep, axes)
  n_keep <- prod_int(a$shape[keep + 1L])
  n_red <- prod_int(a$shape[axes + 1L])
  # An empty product is the constant 1.
  if (n_red == 0L) return(list(ans, zero_of(tangent_aval(aval_of(ans)))))
  xt <- lax_reshape(lax_transpose(x, perm), c(n_keep, n_red))
  gt <- lax_reshape(lax_transpose(g, perm), c(n_keep, n_red))
  onecol <- lax_full(c(n_keep, 1L), 1, a$dtype)
  left <- lax_concatenate(list(onecol, lax_slice(lax_cumprod(xt, 1L), c(0L, 0L), c(n_keep, max(n_red - 1L, 0L)))), 1L)
  right <- lax_concatenate(list(lax_slice(lax_cumprod(xt, 1L, reverse = TRUE), c(0L, 1L), c(n_keep, n_red)), onecol), 1L)
  t <- lax_reduce_sum(lax_mul(gt, lax_mul(left, right)), 1L)
  list(ans, lax_reshape(t, a$shape[keep + 1L]))
}
for (p in list(reduce_and_p, reduce_or_p, reduce_xor_p)) defjvp_zero(p)

reducer <- function(prim) {
  function(operand, axes) {
    x <- as_jax(operand)
    axes <- norm_axis(axes, length(aval_of(x)$shape))
    if (anyDuplicated(axes))
      jax_error("duplicate value in 'axes' of reduction: ", py_tuple(axes),
                class = "jax_value_error")
    axes <- sort(axes)
    if (!length(axes)) return(x)
    bind(prim, list(x), list(axes = as.integer(axes)))
  }
}

#' lax reductions
#' @param operand Array.
#' @param axes 0-based axes to reduce.
#' @name lax_reductions
NULL
#' @rdname lax_reductions
#' @export
lax_reduce_sum <- reducer(reduce_sum_p)
#' @rdname lax_reductions
#' @export
lax_reduce_prod <- reducer(reduce_prod_p)
#' @rdname lax_reductions
#' @export
lax_reduce_max <- reducer(reduce_max_p)
#' @rdname lax_reductions
#' @export
lax_reduce_min <- reducer(reduce_min_p)
#' @rdname lax_reductions
#' @export
lax_reduce_and <- reducer(reduce_and_p)
#' @rdname lax_reductions
#' @export
lax_reduce_or <- reducer(reduce_or_p)
#' @rdname lax_reductions
#' @export
lax_reduce_xor <- reducer(reduce_xor_p)

# ---- generic reduce ---------------------------------------------------------------------------

reduce_p <- new_primitive("reduce", multiple_results = TRUE)
reduce_p$abstract_eval <- function(avals, params) {
  n <- length(avals) %/% 2L
  ops <- avals[seq_len(n)]
  inits <- avals[n + seq_len(n)]
  dts <- function(l) vapply(l, function(a) a$dtype, "")
  if (!identical(dts(ops), dts(inits)))
    jax_error("reduce operand dtypes should match corresponding initial value dtypes, got operands=[",
              paste(vapply(ops, aval_str, ""), collapse = ", "), "] and initial_values=[",
              paste(vapply(inits, aval_str, ""), collapse = ", "), "]", class = "jax_type_error")
  Map(function(a, i) {
    r <- reduce_aval(a, params$dimensions)
    new_aval(r$shape, r$dtype, a$weak && i$weak)
  }, ops, inits)
}
reduce_p$lowering <- function(ctx, ins, ai, ao, params) {
  n <- length(ins) %/% 2L
  jaxpr <- params$jaxpr
  sts <- vapply(ai[n + seq_len(n)], tensor_type, "")
  region <- build_region(ctx, c(sts, sts), function(sub, args) {
    outs <- jaxpr_subcomp(sub, jaxpr$jaxpr, list(), as.list(args))
    list(values = outs, types = sts)
  })
  emit_op(ctx, "stablehlo.reduce", ins, types_of(ai), vapply(ao, tensor_type, ""),
          attrs = list(dimensions = attr_i64_array(params$dimensions)), regions = region)
}
# Port of lax._reduce_jvp: the reduction is rebuilt as a balanced tree of reducer calls
# over the flattened reduced axes, which the JVP trace then differentiates.
reduce_p$jvp <- function(primals, tangents, params) {
  n <- length(primals) %/% 2L
  inits <- primals[n + seq_len(n)]
  if (!all(vapply(tangents[n + seq_len(n)], is_zero, NA)))
    jax_error("Gradient of general lax.reduce with non-zero tangents for initial values ",
              "to reduction not implemented", class = "jax_not_implemented_error")
  shape <- aval_of(primals[[1L]])$shape
  axes <- params$dimensions
  keep <- setdiff(seq_along(shape) - 1L, axes)
  out_shape <- shape[keep + 1L]
  n_red <- prod_int(shape[axes + 1L])
  cj <- params$jaxpr
  # Each call reduces two (k, m) blocks elementwise.
  reducer <- vmap(vmap(function(...) eval_closed(cj, list(...))))
  tree <- function(xs) {
    xs <- lapply(xs, function(x)
      lax_reshape(lax_transpose(x, c(axes, keep)), c(n_red, prod_int(out_shape))))
    k <- n_red
    while (k > 1L) {
      k1 <- (k + 1L) %/% 2L
      a <- lapply(xs, lax_slice_in_dim, 0L, k1)
      b <- lapply(xs, lax_slice_in_dim, k1, k)
      if (k - k1 != k1)
        b <- Map(function(x, i) lax_pad(x, i, list(c(0L, 1L, 0L), c(0L, 0L, 0L))), b, inits)
      xs <- do.call(reducer, c(a, b))
      k <- k1
    }
    if (n_red == 0L) return(lapply(inits, function(i) lax_broadcast(i, out_shape)))
    lapply(xs, lax_reshape, out_shape)
  }
  jvp_flat(tree, primals[seq_len(n)], tangents[seq_len(n)])
}
reduce_p$batch <- function(vals, dims, params, axis) {
  n <- length(vals) %/% 2L
  if (any(!is.na(dims[n + seq_len(n)]))) jax_error("reduce: batched init values not supported")
  ops <- Map(function(v, d) bdim_at_front(v, d, axis$size), vals[seq_len(n)], dims[seq_len(n)])
  out <- bind(reduce_p, c(ops, vals[n + seq_len(n)]),
              list(jaxpr = params$jaxpr, dimensions = params$dimensions + 1L))
  list(out, rep(0L, n))
}

# Port of lax._get_monoid_reducer: a known monoid with its identity as the initial value
# is one of the specialized (and differentiable) reductions.
monoid_reducer <- function(op, inits) {
  if (length(inits) != 1L || is_tracer(inits[[1L]])) return(NULL)
  a <- aval_of(inits[[1L]])
  if (length(a$shape)) return(NULL)
  val <- as.vector(array_host(inits[[1L]]))
  is <- function(f, identity) identical(op, f) && isTRUE(val == identity)
  intlike <- a$dtype == "bool" || is_int_dtype(a$dtype)
  all_ones <- if (a$dtype == "bool") TRUE else if (is_unsigned_dtype(a$dtype)) iinfo(a$dtype)$max else -1
  if (is(lax_add, 0)) lax_reduce_sum
  else if (is(lax_mul, 1)) lax_reduce_prod
  else if (intlike && is(lax_bitwise_or, 0)) lax_reduce_or
  else if (intlike && is(lax_bitwise_xor, 0)) lax_reduce_xor
  else if (intlike && is(lax_bitwise_and, all_ones)) lax_reduce_and
  else if (is(lax_max, identity_max(a$dtype))) lax_reduce_max
  else if (is(lax_min, identity_min(a$dtype))) lax_reduce_min
}

#' General reduction
#'
#' Port of `jax.lax.reduce`.
#' @param operands Array or list of arrays.
#' @param init_values Scalar or list of scalars.
#' @param computation Function of `(accumulators, values)`; with several operands it
#'   receives two lists and returns a list.
#' @param dimensions 0-based dimensions to reduce.
#' @export
lax_reduce <- function(operands, init_values, computation, dimensions) {
  multi <- is.list(operands) && is.null(oldClass(operands))
  ops <- if (multi) lapply(operands, as_jax) else list(as_jax(operands))
  raw <- if (multi) init_values else list(init_values)
  inits <- lapply(raw, as_jax)
  if (length(ops) != length(inits))
    jax_error("Must have same total number of operands as init_values: ", length(ops),
              " vs. ", length(inits), class = "jax_value_error")
  mono <- monoid_reducer(computation, inits)
  if (!is.null(mono)) {
    # The specialized reductions bypass the weak type rule of the primitive.
    out <- mono(ops[[1L]], dimensions)
    return(lax_convert_element_type(out, aval_of(out)$dtype,
                                    weak_type = aval_of(ops[[1L]])$weak && aval_of(inits[[1L]])$weak))
  }
  # An R number adopts the dtype of its operand; an array keeps its dtype, and the
  # primitive rejects one that differs.
  inits <- Map(function(i, o, r) {
    if (is_jax_value(r)) i
    else lax_convert_element_type(i, aval_of(o)$dtype, weak_type = aval_of(i)$weak)
  }, inits, ops, raw)
  n <- length(ops)
  savals <- lapply(inits, function(i) new_aval(integer(0), aval_of(i)$dtype))
  cj <- trace_hoisted(function(args) {
    a <- args[seq_len(n)]
    b <- args[n + seq_len(n)]
    r <- if (multi) computation(a, b) else list(computation(a[[1L]], b[[1L]]))
    lapply(r, as_jax)
  }, c(savals, savals))
  if (length(cj$consts)) jax_error("lax_reduce computation must not close over traced values")
  dims <- sort(norm_axis(dimensions, length(aval_of(ops[[1L]])$shape)))
  out <- bind(reduce_p, c(ops, inits), list(jaxpr = cj$jaxpr, dimensions = as.integer(dims)))
  if (multi) out else out[[1L]]
}

# ---- argmin / argmax --------------------------------------------------------------------------

def_argminmax <- function(name, cmp, identity) {
  p <- new_primitive(name)
  p$abstract_eval <- function(avals, params) {
    a <- avals[[1L]]
    if (a$shape[params$axes + 1L] < 1L)
      jax_error("argmin and argmax require non-empty reduced dimension. operand.shape=",
                py_tuple(a$shape), " axis=", params$axes, class = "jax_value_error")
    if (!is_int_dtype(params$index_dtype))
      jax_error("index_dtype must be an integer type, but got ", params$index_dtype,
                class = "jax_type_error")
    reduce_aval(a, params$axes, params$index_dtype)
  }
  p$lowering <- lower_fun(function(args, params) {
    x <- args[[1L]]
    a <- aval_of(x)
    ax <- params$axes
    idx <- lax_broadcasted_iota(params$index_dtype, a$shape, ax)
    red <- function(acc, op) {
      op_val <- op[[1L]]; op_idx <- op[[2L]]
      acc_val <- acc[[1L]]; acc_idx <- acc[[2L]]
      pick_op_val <- lax_bitwise_or(cmp(op_val, acc_val), lax_ne(op_val, op_val))
      pick_op_idx <- lax_bitwise_or(pick_op_val,
                                    lax_bitwise_and(lax_eq(op_val, acc_val), lax_lt(op_idx, acc_idx)))
      list(lax_select(pick_op_val, op_val, acc_val), lax_select(pick_op_idx, op_idx, acc_idx))
    }
    out <- lax_reduce(list(x, idx), list(identity(a$dtype), 0), function(acc, op) red(op, acc), ax)
    out[[2L]]
  })
  p$batch <- function(vals, dims, params, axis) {
    d <- dims[1L]
    axes <- ifelse(params$axes >= d, params$axes + 1L, params$axes)
    list(bind(p, vals, list(axes = as.integer(axes), index_dtype = params$index_dtype)),
         d - sum(params$axes < d))
  }
  defjvp_zero(p)
  p
}
argmax_p <- def_argminmax("argmax", lax_gt, identity_max)
argmin_p <- def_argminmax("argmin", lax_lt, identity_min)

#' Index of the maximum along an axis
#' @param operand Array.
#' @param axis 0-based axis.
#' @param index_dtype Integer dtype of the result.
#' @export
lax_argmax <- function(operand, axis, index_dtype = "int32") {
  x <- as_jax(operand)
  bind(argmax_p, list(x), list(axes = norm_axis(axis, length(aval_of(x)$shape)),
                               index_dtype = dtype(index_dtype)))
}

#' @rdname lax_argmax
#' @export
lax_argmin <- function(operand, axis, index_dtype = "int32") {
  x <- as_jax(operand)
  bind(argmin_p, list(x), list(axes = norm_axis(axis, length(aval_of(x)$shape)),
                               index_dtype = dtype(index_dtype)))
}

# ---- dot_general --------------------------------------------------------------------------------

dot_general_p <- new_primitive("dot_general")

remaining_dims <- function(n, ...) setdiff(seq_len(n) - 1L, c(...))

dot_general_p$abstract_eval <- function(avals, params) {
  l <- avals[[1L]]
  r <- avals[[2L]]
  dn <- params$dimension_numbers
  lc <- dn$lhs_contracting; rc <- dn$rhs_contracting
  lb <- dn$lhs_batch; rb <- dn$rhs_batch
  if (length(lc) != length(rc) || length(lb) != length(rb))
    jax_error("dot_general requires equal numbers of lhs and rhs contracting/batch dimensions")
  if (any(l$shape[lc + 1L] != r$shape[rc + 1L]))
    jax_error("dot_general requires contracting dimensions to have the same shape, got (",
              paste(l$shape[lc + 1L], collapse = ", "), ") and (",
              paste(r$shape[rc + 1L], collapse = ", "), ").", class = "jax_type_error")
  if (any(l$shape[lb + 1L] != r$shape[rb + 1L]))
    jax_error("dot_general requires lhs batch dimensions and rhs batch dimensions to have the same shape")
  shape <- c(l$shape[lb + 1L], l$shape[remaining_dims(length(l$shape), lc, lb) + 1L],
             r$shape[remaining_dims(length(r$shape), rc, rb) + 1L])
  new_aval(shape, dot_general_dtype(l$dtype, r$dtype, params$preferred_element_type),
           l$weak && r$weak)
}

# Port of lax._dot_general_dtype_rule and lax._maybe_upcast: operands of different dtypes
# give the dtype of higher precision (XLA's HigherPrecisionType), and
# preferred_element_type may widen the result but not narrow a non-float one.
dot_general_dtype <- function(ld, rd, pet = NULL) {
  component <- function(dt) if (is_complex_dtype(dt)) complex_component(dt) else dt
  width <- function(dt) if (dt == "bool") 1L else dtype_bits(component(dt))
  prop <- function(dt) {
    cd <- component(dt)
    fi <- if (is_float_dtype(cd)) finfo(cd) else list(nexp = -1, nmant = -1)
    c(is_complex_dtype(dt), fi$nexp, fi$nmant, width(dt), !is_unsigned_dtype(cd))
  }
  d <- prop(ld) - prop(rd)
  d <- d[d != 0]
  res <- if (!length(d)) {
    if (ld != rd)
      jax_error("lax.dot_general argument type error: ", ld, ", ", rd, class = "jax_type_error")
    ld
  } else if (d[1L] > 0) ld else rd
  if (is.null(pet) || pet == res) return(res)
  if (!is_float_dtype(res) && width(pet) < width(res))
    jax_error("`preferred_element_type` must not be narrower than the original type, got ",
              "preferred_element_type of ", pet, " for result type of ", res, ".",
              class = "jax_type_error")
  pet
}

dot_dn_attr <- function(dn) {
  f <- function(nm, v) if (length(v)) paste0(nm, " = ", attr_i64_list(v)) else NULL
  parts <- c(f("lhs_batching_dimensions", dn$lhs_batch), f("rhs_batching_dimensions", dn$rhs_batch),
             f("lhs_contracting_dimensions", dn$lhs_contracting),
             f("rhs_contracting_dimensions", dn$rhs_contracting))
  paste0("#stablehlo.dot<", paste(parts, collapse = ", "), ">")
}

dot_general_p$lowering <- function(ctx, ins, ai, ao, params) {
  o <- ao[[1L]]
  ops <- ins
  ait <- ai
  # Port of lax._handle_dot_precision on CPU: operands of different dtypes are converted
  # to the result dtype. With equal dtypes only a non-float operand of a float result is
  # converted; XLA widens the rest while it accumulates.
  mixed <- ai[[1L]]$dtype != ai[[2L]]$dtype
  for (i in 1:2) {
    if (ai[[i]]$dtype != o$dtype &&
        (mixed || (is_float_dtype(o$dtype) && !is_float_dtype(ai[[i]]$dtype)))) {
      na <- new_aval(ai[[i]]$shape, o$dtype)
      ops[[i]] <- convert_element_type_p$lowering(ctx, ops[[i]], ai[i], list(na), list())
      ait[[i]] <- na
    }
  }
  attrs <- list(dot_dimension_numbers = dot_dn_attr(params$dimension_numbers))
  prec <- params$precision
  if (!is.null(prec)) {
    pv <- toupper(prec)
    attrs$precision_config <- paste0("[#stablehlo<precision ", pv, ">, #stablehlo<precision ", pv, ">]")
  }
  emit_op(ctx, "stablehlo.dot_general", ops, types_of(ait), tensor_type(o), attrs = attrs)
}

ranges_like <- function(...) {
  xs <- list(...)
  start <- 0L
  lapply(xs, function(x) {
    r <- start + seq_along(x) - 1L
    start <<- start + length(x)
    r
  })
}

dot_transpose_lhs <- function(g, x_aval, y, dn, params, swap_ans = FALSE) {
  xc <- dn$lhs_contracting; yc <- dn$rhs_contracting
  xb <- dn$lhs_batch; yb <- dn$rhs_batch
  y_aval <- arg_aval(y)
  x_kept <- remaining_dims(length(x_aval$shape), xc, xb)
  y_kept <- remaining_dims(length(y_aval$shape), yc, yb)
  if (swap_ans) {
    rr <- ranges_like(xb, y_kept, x_kept)
    ans_batch <- rr[[1L]]; ans_y <- rr[[2L]]
  } else {
    rr <- ranges_like(xb, x_kept, y_kept)
    ans_batch <- rr[[1L]]; ans_y <- rr[[3L]]
  }
  dims <- list(lhs_contracting = ans_y, rhs_contracting = y_kept,
               lhs_batch = ans_batch, rhs_batch = yb)
  x_contract_sorted_by_y <- xc[order(yc)]
  out_axes <- order(c(xb, x_kept, x_contract_sorted_by_y)) - 1L
  r <- bind(dot_general_p, list(g, y), list(dimension_numbers = dims, precision = params$precision,
                                           preferred_element_type = params$preferred_element_type))
  r <- lax_transpose(r, out_axes)
  if (aval_of(r)$dtype != x_aval$dtype) r <- lax_convert_element_type(r, x_aval$dtype)
  r
}

dot_general_p$transpose <- function(ct, args, params) {
  x <- args[[1L]]
  y <- args[[2L]]
  dn <- params$dimension_numbers
  if (is_zero(ct)) return(list(NULL, NULL))
  if (is_undefined(x)) {
    return(list(dot_transpose_lhs(ct, x$aval, y, dn, params), NULL))
  }
  sdn <- list(lhs_contracting = dn$rhs_contracting, rhs_contracting = dn$lhs_contracting,
              lhs_batch = dn$rhs_batch, rhs_batch = dn$lhs_batch)
  list(NULL, dot_transpose_lhs(ct, y$aval, x, sdn, params, swap_ans = TRUE))
}

dot_general_p$jvp <- function(primals, tangents, params) {
  ans <- bind(dot_general_p, primals, params)
  t1 <- tangents[[1L]]
  t2 <- tangents[[2L]]
  out <- add_tangents(
    if (is_zero(t1)) t1 else bind(dot_general_p, list(t1, primals[[2L]]), params),
    if (is_zero(t2)) t2 else bind(dot_general_p, list(primals[[1L]], t2), params))
  list(ans, out)
}

shift_dims <- function(v, d) ifelse(v >= d, v + 1L, v)

dot_general_p$batch <- function(vals, dims, params, axis) {
  dn <- params$dimension_numbers
  l <- vals[[1L]]; r <- vals[[2L]]
  ld <- dims[1L]; rd <- dims[2L]
  if (!is.na(ld) && !is.na(rd)) {
    l <- moveaxis(l, ld, 0L)
    r <- moveaxis(r, rd, 0L)
    ndn <- list(lhs_contracting = dn$lhs_contracting + 1L, rhs_contracting = dn$rhs_contracting + 1L,
                lhs_batch = c(0L, dn$lhs_batch + 1L), rhs_batch = c(0L, dn$rhs_batch + 1L))
    return(list(bind(dot_general_p, list(l, r), update_params(params, list(dimension_numbers = ndn))), 0L))
  }
  if (!is.na(ld)) {
    l <- moveaxis(l, ld, 0L)
    ndn <- update_params(dn, list(lhs_contracting = dn$lhs_contracting + 1L, lhs_batch = dn$lhs_batch + 1L))
    return(list(bind(dot_general_p, list(l, r), update_params(params, list(dimension_numbers = ndn))),
                length(dn$lhs_batch)))
  }
  r <- moveaxis(r, rd, 0L)
  ndn <- update_params(dn, list(rhs_contracting = dn$rhs_contracting + 1L, rhs_batch = dn$rhs_batch + 1L))
  lr <- length(aval_of(l)$shape)
  n_lfree <- lr - length(dn$lhs_contracting) - length(dn$lhs_batch)
  list(bind(dot_general_p, list(l, r), update_params(params, list(dimension_numbers = ndn))),
       length(dn$lhs_batch) + n_lfree)
}

#' General dot product
#'
#' Port of `jax.lax.dot_general`.
#' @param lhs,rhs Arrays.
#' @param dimension_numbers `list(list(lhs_contracting, rhs_contracting),
#'   list(lhs_batch, rhs_batch))` with 0-based dimensions.
#' @param precision Optional precision (`"default"`, `"high"`, `"highest"`).
#' @param preferred_element_type Optional result dtype.
#' @export
lax_dot_general <- function(lhs, rhs, dimension_numbers, precision = NULL,
                            preferred_element_type = NULL) {
  args <- lax_args(lhs, rhs)
  dn <- list(lhs_contracting = as.integer(dimension_numbers[[1L]][[1L]]),
             rhs_contracting = as.integer(dimension_numbers[[1L]][[2L]]),
             lhs_batch = as.integer(dimension_numbers[[2L]][[1L]] %||% integer(0)),
             rhs_batch = as.integer(dimension_numbers[[2L]][[2L]] %||% integer(0)))
  bind(dot_general_p, args, list(dimension_numbers = dn, precision = precision,
                                 preferred_element_type = if (is.null(preferred_element_type)) NULL
                                                          else dtype(preferred_element_type)))
}

#' Matrix and vector products
#' @param lhs,rhs Arrays of rank 1 or 2.
#' @param precision Optional precision.
#' @param preferred_element_type Optional result dtype.
#' @export
lax_dot <- function(lhs, rhs, precision = NULL, preferred_element_type = NULL) {
  l <- as_jax(lhs)
  r <- as_jax(rhs)
  lr <- length(aval_of(l)$shape)
  rr <- length(aval_of(r)$shape)
  ls <- aval_of(l)$shape
  rs <- aval_of(r)$shape
  if (!(lr %in% 1:2 && rr %in% 1:2 && ls[lr] == rs[1L]))
    jax_error("jax.lax.dot: dimension_numbers must be specified when not performing simple ",
              "un-batched matrix-matrix, matrix-vector, or vector-vector products; got lhs_shape=",
              py_tuple(ls), " rhs_shape=", py_tuple(rs), class = "jax_value_error")
  lax_dot_general(l, r, list(list(lr - 1L, 0L), list(integer(0), integer(0))), precision,
                  preferred_element_type)
}
