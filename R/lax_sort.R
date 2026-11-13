# Sorting (port of the sort and top_k parts of jax/_src/lax/lax.py) and the
# jax.numpy sorting functions (port of jax/_src/numpy/sorting.py and of searchsorted from
# jax/_src/numpy/lax_numpy.py and jax/_src/numpy/hijax.py).

# ---- total order comparisons (lax.eq_to_p, le_to_p, lt_to_p) -------------------------

# Port of lax._compare_lower_hlo with total_order = True.
def_compare_to <- function(name, dir, ordered) {
  p <- new_primitive(name)
  base <- naryop_abstract_eval(name, "any", "bool")
  p$abstract_eval <- function(avals, params) {
    for (a in avals) {
      if (ordered && is_complex_dtype(a$dtype))
        jax_error(name, " does not accept dtype ", a$dtype, ". Accepted dtypes are subtypes ",
                  "of integer, floating, bool.", class = "jax_type_error")
    }
    base(avals, params)
  }
  p$lowering <- function(ctx, ins, ai, ao, params) {
    k <- dtype_kind(ai[[1L]]$dtype)
    ct <- if (k %in% c("f", "c")) "TOTALORDER" else if (k == "i") "SIGNED" else "UNSIGNED"
    lower_nary("stablehlo.compare", list(
      comparison_direction = paste0("#stablehlo<comparison_direction ", dir, ">"),
      compare_type = paste0("#stablehlo<comparison_type ", ct, ">")))(ctx, ins, ai, ao, params)
  }
  p$batch <- lax_broadcast_batcher(p)
  defjvp_zero(p)
  p
}
eq_to_p <- def_compare_to("eq_to", "EQ", FALSE)
le_to_p <- def_compare_to("le_to", "LE", TRUE)
lt_to_p <- def_compare_to("lt_to", "LT", TRUE)

# ---- comparators ------------------------------------------------------------------------

# Port of lax._canonicalize_float_for_sort: -0.0 becomes 0.0 and every NaN the same NaN.
canonicalize_float_for_sort <- function(x) {
  result <- lax_select(lax_eq(x, lax_full_like(x, 0)), lax_full_like(x, 0), x)
  lax_select(lax_ne(x, x), lax_full_like(result, NaN), result)
}

# Port of lax._operands_to_keys. `operands` is (x0, y0, x1, y1, ...).
operands_to_keys <- function(operands, num_keys = 1L) {
  x_keys <- list()
  y_keys <- list()
  for (i in seq_len(num_keys)) {
    x <- operands[[2L * i - 1L]]
    y <- operands[[2L * i]]
    dt <- aval_of(x)$dtype
    if (is_complex_dtype(dt)) {
      x_keys <- c(x_keys, list(canonicalize_float_for_sort(lax_real(x)),
                               canonicalize_float_for_sort(lax_imag(x))))
      y_keys <- c(y_keys, list(canonicalize_float_for_sort(lax_real(y)),
                               canonicalize_float_for_sort(lax_imag(y))))
    } else if (is_float_dtype(dt)) {
      x_keys <- c(x_keys, list(canonicalize_float_for_sort(x)))
      y_keys <- c(y_keys, list(canonicalize_float_for_sort(y)))
    } else {
      x_keys <- c(x_keys, list(x))
      y_keys <- c(y_keys, list(y))
    }
  }
  list(x_keys, y_keys)
}

# Ports of lax._sort_lt_comparator and lax._sort_le_comparator: lexicographic
# comparison on the first num_keys operand pairs, with a total order for floats.
sort_comparator <- function(operands, num_keys = 1L, last = lt_to_p) {
  k <- operands_to_keys(operands, num_keys)
  p <- NULL
  for (i in rev(seq_along(k[[1L]]))) {
    xk <- k[[1L]][[i]]
    yk <- k[[2L]][[i]]
    p <- if (is.null(p)) bind(last, list(xk, yk))
         else lax_bitwise_or(bind(lt_to_p, list(xk, yk)),
                             lax_bitwise_and(bind(eq_to_p, list(xk, yk)), p))
  }
  p
}
sort_lt_comparator <- function(operands, num_keys = 1L) sort_comparator(operands, num_keys, lt_to_p)
sort_le_comparator <- function(operands, num_keys = 1L) sort_comparator(operands, num_keys, le_to_p)

# ---- take along an axis ---------------------------------------------------------------

# out[..., i, ...] = x[..., idx[..., i, ...], ...] along `axis` (idx has the shape of x
# except along axis): the gather of lax._sort_jvp and lax._top_k_jvp, with every other
# dimension a batching dimension. A gather selects values, so non-finite entries
# elsewhere in x never reach the result.
take_along_axis_gather <- function(x, idx, axis) {
  r <- length(aval_of(x)$shape)
  bd <- setdiff(seq_len(r) - 1L, axis)
  dn <- lax_GatherDimensionNumbers(offset_dims = integer(0), collapsed_slice_dims = axis,
                                   start_index_map = axis, operand_batching_dims = bd,
                                   start_indices_batching_dims = bd)
  lax_gather(x, lax_expand_dims(idx, r), dn, rep(1L, r), mode = "promise_in_bounds")
}

# ---- sort ---------------------------------------------------------------------------------

sort_p <- new_primitive("sort", multiple_results = TRUE)

# Port of lax._sort_abstract_eval.
sort_p$abstract_eval <- function(avals, params) {
  s0 <- avals[[1L]]$shape
  if (any(vapply(avals, function(a) !identical(a$shape, s0), NA)))
    jax_error("Arguments to sort must have equal shapes, got: ",
              paste(vapply(avals, function(a) fmt_tuple(a$shape), ""), collapse = " "),
              class = "jax_type_error")
  avals
}

# Port of lax._sort_lower.
sort_p$lowering <- function(ctx, ins, ai, ao, params) {
  scalar_avals <- lapply(ai, function(a) new_aval(integer(0), a$dtype))
  pair_avals <- rep(scalar_avals, each = 2L)
  pair_types <- vapply(pair_avals, tensor_type, "")
  cmp <- lower_fun(function(args, p) sort_lt_comparator(args, p$num_keys))
  region <- build_region(ctx, pair_types, function(sub, args) {
    out <- cmp(sub, args, pair_avals, list(new_aval(integer(0), "bool")), params)
    list(values = out, types = scalar_type("bool"))
  })
  emit_op(ctx, "stablehlo.sort", ins, types_of(ai), vapply(ao, tensor_type, ""),
          attrs = list(dimension = attr_i64(params$dimension),
                       is_stable = attr_bool(params$is_stable)),
          regions = region)
}

# Port of lax._sort_jvp (the gather is take_along_axis_gather).
sort_p$jvp <- function(primals, tangents, params) {
  shape <- aval_of(primals[[1L]])$shape
  index_dtype <- if (any(shape > 4294967295)) "uint64" else "uint32"
  iota <- lax_broadcasted_iota(index_dtype, shape, params$dimension)
  out <- bind(sort_p, c(primals, list(iota)), params)
  n <- length(primals)
  idx <- out[[n + 1L]]
  tangents_out <- lapply(tangents, function(t) {
    if (is_zero(t)) t else take_along_axis_gather(t, idx, params$dimension)
  })
  list(out[seq_len(n)], tangents_out)
}

# Port of lax._sort_batch_rule.
sort_p$batch <- function(vals, dims, params, axis) {
  k <- which(!is.na(dims))[1L]
  proto <- vals[[k]]
  new_bdim <- dims[k]
  pshape <- aval_of(proto)$shape
  r <- length(pshape)
  new_args <- Map(function(v, d) {
    if (is.na(d)) lax_broadcast_in_dim(v, pshape, setdiff(seq_len(r) - 1L, new_bdim))
    else moveaxis(v, d, new_bdim)
  }, vals, dims)
  new_dimension <- params$dimension + as.integer(new_bdim <= params$dimension)
  out <- bind(sort_p, new_args, modifyList(params, list(dimension = new_dimension)))
  list(out, rep(new_bdim, length(new_args)))
}

#' Sort arrays
#'
#' Port of `jax.lax.sort`. Floating point values follow a total order in which -0.0
#' and 0.0 are equivalent and NaNs sort to the end; complex values sort
#' lexicographically on (real, imag).
#' @param operand Array or list of arrays of equal shapes.
#' @param dimension 0-based dimension to sort along (negative counts from the end).
#' @param is_stable Use a stable sort.
#' @param num_keys Number of leading operands used as lexicographic sort keys.
#' @return The sorted array, or a list of arrays permuted like the keys.
#' @export
lax_sort <- function(operand, dimension = -1L, is_stable = TRUE, num_keys = 1L) {
  if (is.list(operand) && is.null(oldClass(operand))) {
    if (!length(operand)) jax_error("Sort requires at least one operand", class = "jax_type_error")
    if (!(num_keys >= 1L && num_keys <= length(operand)))
      jax_error("num_keys=", num_keys, " must be between 1 and len(operand)=", length(operand),
                class = "jax_value_error")
    ops <- lapply(operand, as_jax)
    dimension <- norm_axis(dimension, length(aval_of(ops[[1L]])$shape))
    return(bind(sort_p, ops, list(dimension = dimension, is_stable = isTRUE(is_stable),
                                  num_keys = as.integer(num_keys))))
  }
  if (num_keys != 1L)
    jax_error("num_keys=", num_keys, " must equal 1 for a single operand.",
              class = "jax_value_error")
  x <- as_jax(operand)
  dimension <- norm_axis(dimension, length(aval_of(x)$shape))
  bind(sort_p, list(x), list(dimension = dimension, is_stable = isTRUE(is_stable),
                             num_keys = 1L))[[1L]]
}

#' Sort keys and permute values alike
#'
#' Port of `jax.lax.sort_key_val`.
#' @param keys Array of sort keys.
#' @param values Array with the shape of `keys`.
#' @param dimension 0-based dimension.
#' @param is_stable Use a stable sort.
#' @return `list(sorted_keys, permuted_values)`.
#' @export
lax_sort_key_val <- function(keys, values, dimension = -1L, is_stable = TRUE) {
  k <- as_jax(keys)
  dimension <- norm_axis(dimension, length(aval_of(k)$shape))
  bind(sort_p, list(k, as_jax(values)),
       list(dimension = dimension, is_stable = isTRUE(is_stable), num_keys = 1L))
}

# ---- top_k --------------------------------------------------------------------------------

top_k_p <- new_primitive("top_k", multiple_results = TRUE)

# Port of lax._top_k_abstract_eval.
top_k_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  k <- params$k
  axis <- params$axis
  if (is_complex_dtype(a$dtype))
    jax_error("top_k is not compatible with complex inputs.", class = "jax_value_error")
  if (k < 0L) jax_error("k argument to top_k must be nonnegative, got ", k, class = "jax_value_error")
  if (!length(a$shape))
    jax_error("top_k operand must have >= 1 dimension, got ()", class = "jax_type_error")
  if (!(axis >= 0L && axis < length(a$shape)))
    jax_error("axis argument out of range: axis=", axis, " for operand.shape=",
              fmt_tuple(a$shape), class = "jax_value_error")
  shape <- a$shape
  if (shape[axis + 1L] < k)
    jax_error("k argument to top_k must be no larger than size along axis; got k=", k,
              " with shape=[", paste(shape, collapse = ", "), "] and axis=", axis,
              class = "jax_value_error")
  shape[axis + 1L] <- k
  list(new_aval(shape, a$dtype, a$weak), new_aval(shape, "int32"))
}

# Port of lax._top_k_lower (static k): chlo.top_k along the last dimension.
top_k_p$lowering <- function(ctx, ins, ai, ao, params) {
  a <- ai[[1L]]
  nd <- length(a$shape)
  axis <- params$axis
  x <- ins[[1L]]
  perm <- NULL
  if (axis != nd - 1L) {
    perm <- seq_len(nd) - 1L
    perm[c(axis, nd - 1L) + 1L] <- perm[c(nd - 1L, axis) + 1L]
    ta <- new_aval(a$shape[perm + 1L], a$dtype)
    x <- emit_op(ctx, "stablehlo.transpose", x, tensor_type(a), tensor_type(ta),
                 attrs = list(permutation = attr_i64_array(perm)))
    a <- ta
  }
  outs_t <- if (is.null(perm)) ao else lapply(ao, function(o) new_aval(o$shape[perm + 1L], o$dtype))
  res <- emit_op(ctx, "chlo.top_k", x, tensor_type(a), vapply(outs_t, tensor_type, ""),
                 attrs = list(is_stable = attr_bool(params$is_stable), k = attr_i64(params$k)))
  if (is.null(perm)) return(res)
  vapply(seq_along(res), function(i) {
    emit_op(ctx, "stablehlo.transpose", res[[i]], tensor_type(outs_t[[i]]), tensor_type(ao[[i]]),
            attrs = list(permutation = attr_i64_array(perm)))
  }, "")
}

# Port of lax._top_k_jvp (the gather is take_along_axis_gather).
top_k_p$jvp <- function(primals, tangents, params) {
  out <- bind(top_k_p, primals, params)
  t <- tangents[[1L]]
  t_out <- if (is_zero(t)) zero_of(tangent_aval(aval_of(out[[1L]])))
           else take_along_axis_gather(t, out[[2L]], params$axis)
  list(out, list(t_out, zero_of(tangent_aval(aval_of(out[[2L]])))))
}

# Port of lax._top_k_batch_rule.
top_k_p$batch <- function(vals, dims, params, axis) {
  d <- dims[1L]
  ax <- params$axis
  if (d <= ax) ax <- ax + 1L
  list(bind(top_k_p, vals, modifyList(params, list(axis = ax))), c(d, d))
}

#' Top k values and their indices
#'
#' Port of `jax.lax.top_k`.
#' @param operand Array of a non complex dtype.
#' @param k Number of entries.
#' @param axis 0-based axis (default last).
#' @param is_stable Preserve the order of equal elements.
#' @return `list(values, indices)`; indices are 0-based `int32`.
#' @export
lax_top_k <- function(operand, k, axis = -1L, is_stable = TRUE) {
  x <- as_jax(operand)
  if (is.numeric(k) && length(k) == 1L && !is.na(k) && k < 0)
    jax_error("k argument to top_k must be nonnegative, got ", k, class = "jax_value_error")
  k <- canon_shape(k)
  if (length(k) != 1L) jax_error("k argument to top_k must be a scalar", class = "jax_type_error")
  if (k < 0L) jax_error("k argument to top_k must be nonnegative, got ", k, class = "jax_value_error")
  axis <- norm_axis(axis, length(aval_of(x)$shape))
  bind(top_k_p, list(x), list(k = k, axis = axis, is_stable = isTRUE(is_stable)))
}

# ---- jax.numpy sorting ---------------------------------------------------------------------

# Port of numpy.sorting._order_reversing_invert.
order_reversing_invert <- function(x) {
  k <- dtype_kind(aval_of(x)$dtype)
  if (k %in% c("b", "i", "u")) lax_bitwise_not(x) else lax_neg(x)
}

check_sort_kind_order <- function(name, kind, order) {
  if (!is.null(kind))
    jax_error("'kind' argument to ", name, " is not supported. Use stable=True or ",
              "stable=False to specify sort stability.", class = "jax_type_error")
  if (!is.null(order))
    jax_error("'order' argument to ", name, " is not supported.", class = "jax_type_error")
}

#' Sorting functions of jax.numpy
#'
#' Ports of `jnp.sort`, `jnp.argsort`, `jnp.partition`, `jnp.argpartition`,
#' `jnp.sort_complex`, `jnp.lexsort` and `jnp.top_k`. Axes and returned indices are
#' 0-based.
#' @param a Array.
#' @param axis 0-based axis (`NULL` flattens for `sort` and `argsort`).
#' @param kind,order Unsupported (must be `NULL`).
#' @param stable Use a stable sort.
#' @param descending Sort in descending order.
#' @param dtype Optional index dtype of `argsort`.
#' @return An array (a list of values and indices for `top_k`).
#' @name jnp_sorting
NULL

#' @rdname jnp_sorting
#' @export
jnp_sort <- function(a, axis = -1L, kind = NULL, order = NULL, stable = TRUE, descending = FALSE) {
  check_sort_kind_order("sort", kind, order)
  arr <- as_jax(a)
  if (is.null(axis)) {
    arr <- lax_reshape(arr, prod_int(aval_of(arr)$shape))
    axis <- 0L
  }
  dimension <- norm_axis(axis, length(aval_of(arr)$shape))
  result <- lax_sort(arr, dimension = dimension, is_stable = stable)
  if (descending) lax_rev(result, dimension) else result
}

#' @rdname jnp_sorting
#' @export
jnp_argsort <- function(a, axis = -1L, kind = NULL, order = NULL, stable = TRUE,
                        descending = FALSE, dtype = NULL) {
  check_sort_kind_order("argsort", kind, order)
  arr <- as_jax(a)
  if (is.null(axis)) {
    arr <- lax_reshape(arr, prod_int(aval_of(arr)$shape))
    axis <- 0L
  }
  shape <- aval_of(arr)$shape
  dimension <- norm_axis(axis, length(shape))
  idx_dtype <- if (!is.null(dtype)) dtype(dtype)
               else if (shape[dimension + 1L] > 2147483647) "int64" else default_int()
  iota <- lax_broadcasted_iota(idx_dtype, shape, dimension)
  if (descending && stable) {
    arr <- lax_rev(arr, dimension)
    iota <- lax_rev(iota, dimension)
  }
  indices <- lax_sort_key_val(arr, iota, dimension = dimension, is_stable = stable)[[2L]]
  if (descending) lax_rev(indices, dimension) else indices
}

#' @rdname jnp_sorting
#' @param kth 0-based index of the element in sorted position.
#' @export
jnp_partition <- function(a, kth, axis = -1L) {
  arr <- as_jax(a)
  if (is_complex_dtype(aval_of(arr)$dtype))
    jax_error("jnp.partition for complex dtype is not implemented.",
              class = "jax_not_implemented_error")
  r <- length(aval_of(arr)$shape)
  axis <- norm_axis(axis, r)
  kth <- norm_axis(kth, aval_of(arr)$shape[axis + 1L])
  arr <- jnp_swapaxes(arr, axis, -1L)
  n <- aval_of(arr)$shape[r]
  bottom <- order_reversing_invert(lax_top_k(order_reversing_invert(arr), kth + 1L)[[1L]])
  top <- lax_top_k(arr, n - kth - 1L)[[1L]]
  out <- lax_concatenate(list(bottom, top), r - 1L)
  jnp_swapaxes(out, -1L, axis)
}

#' @rdname jnp_sorting
#' @export
jnp_argpartition <- function(a, kth, axis = -1L) {
  arr <- as_jax(a)
  if (is_complex_dtype(aval_of(arr)$dtype))
    jax_error("jnp.argpartition for complex dtype is not implemented.",
              class = "jax_not_implemented_error")
  r <- length(aval_of(arr)$shape)
  axis <- norm_axis(axis, r)
  kth <- norm_axis(kth, aval_of(arr)$shape[axis + 1L])
  arr <- jnp_swapaxes(arr, axis, -1L)
  shape <- aval_of(arr)$shape
  n <- shape[r]
  bottom_ind <- lax_top_k(order_reversing_invert(arr), kth + 1L)[[2L]]
  # JAX sets proxy[bottom_ind] = 0 with a scatter under vmap; the same mask is built by
  # comparing every position with the selected indices.
  cmp_shape <- c(shape[-r], kth + 1L, n)
  hit <- lax_eq(lax_broadcast_in_dim(bottom_ind, cmp_shape, seq_len(r) - 1L),
                lax_broadcasted_iota("int32", cmp_shape, r))
  proxy <- lax_select(lax_reduce_or(hit, r - 1L), lax_full(shape, 0), lax_full(shape, 1))
  top_ind <- lax_top_k(proxy, n - kth - 1L)[[2L]]
  out <- lax_concatenate(list(bottom_ind, top_ind), r - 1L)
  jnp_swapaxes(out, -1L, axis)
}

#' @rdname jnp_sorting
#' @export
jnp_sort_complex <- function(a) {
  a <- lax_sort(as_jax(a))
  lax_convert_element_type(a, sig_complex_dtype(aval_of(a)$dtype))
}

lexsort_index_dtype <- function(size) if (size > 2147483647) "int64" else default_int()

#' @rdname jnp_sorting
#' @param keys Array whose leading dimension indexes the keys, or a list of arrays.
#'   The last key is the primary one.
#' @param batch_size Number of keys sorted at once (all keys are always sorted in one
#'   pass here).
#' @export
jnp_lexsort <- function(keys, axis = -1L, batch_size = 16L) {
  batch_size <- as.integer(batch_size)
  if (batch_size < 1L)
    jax_error("lexsort: batch_size must be positive, got ", batch_size, class = "jax_value_error")
  if (is.list(keys) && is.null(oldClass(keys))) {
    if (!length(keys))
      jax_error("need sequence of keys with len > 0 in lexsort", class = "jax_type_error")
    keys <- lapply(keys, as_jax)
    if (length(unique(lapply(keys, function(k) aval_of(k)$shape))) > 1L)
      jax_error("all keys need to be the same shape", class = "jax_value_error")
  } else {
    k <- as_jax(keys)
    s <- aval_of(k)$shape
    if (!length(s) || s[1L] == 0L)
      jax_error("need sequence of keys with len > 0 in lexsort", class = "jax_type_error")
    if (length(s) == 1L) return(lax_full(integer(0), 0L, default_int()))
    keys <- lapply(seq_len(s[1L]) - 1L, function(i) lax_index_in_dim(k, i, 0L, keepdims = FALSE))
  }
  shape <- aval_of(keys[[1L]])$shape
  if (!length(shape)) return(lax_full(integer(0), 0L, default_int()))
  axis <- norm_axis(axis, length(shape))
  # ponytail: JAX sorts more than batch_size keys in batches (fori_loop and gathers);
  # one sort over all keys gives the same permutation.
  iota <- lax_broadcasted_iota(lexsort_index_dtype(shape[axis + 1L]), shape, axis)
  out <- lax_sort(c(rev(keys), list(iota)), dimension = axis, num_keys = length(keys))
  out[[length(out)]]
}

#' @rdname jnp_sorting
#' @param k Number of entries for `top_k`.
#' @param mode `"largest"` or `"smallest"`.
#' @param sorted Ignored (results are always sorted).
#' @export
jnp_top_k <- function(a, k, axis = -1L, mode = "largest", sorted = TRUE) {
  arr <- as_jax(a)
  if (is_complex_dtype(aval_of(arr)$dtype))
    jax_error("top_k is not compatible with complex inputs.", class = "jax_value_error")
  if (!mode %in% c("largest", "smallest"))
    jax_error("mode must be 'largest' or 'smallest', got '", mode, "'", class = "jax_value_error")
  axis <- norm_axis(axis, length(aval_of(arr)$shape))
  if (mode == "largest") return(lax_top_k(arr, k, axis = axis))
  r <- lax_top_k(order_reversing_invert(arr), k, axis = axis)
  list(order_reversing_invert(r[[1L]]), r[[2L]])
}

# ---- searchsorted -------------------------------------------------------------------------

# Port of hijax._searchsorted_scan_impl: binary search with lax.scan.
searchsorted_scan_impl <- function(sorted_arr, query, side, dtype) {
  n <- aval_of(sorted_arr)$shape[1L]
  if (n == 0L) return(lax_full(aval_of(query)$shape, 0L, dtype))
  if (length(aval_of(query)$shape) > 0L)
    return(vmap(function(q) searchsorted_scan_impl(sorted_arr, q, side, dtype))(query))
  op <- if (side == "left") sort_le_comparator else sort_lt_comparator
  unsigned_dtype <- if (dtype_bits(dtype) == 64L) "uint64" else "uint32"
  body_fun <- function(state, x) {
    low <- state[[1L]]
    high <- state[[2L]]
    mid <- lax_add(low, lax_div(lax_sub(high, low), lax_full_like(low, 2)))
    a_mid <- lax_reshape(lax_dynamic_slice(sorted_arr, list(mid), 1L), integer(0))
    go_left <- op(list(query, a_mid))
    list(list(lax_select(go_left, low, mid), lax_select(go_left, mid, high)), NULL)
  }
  n_levels <- as.integer(ceiling(log2(n + 1)))
  init <- list(lax_full(integer(0), 0, unsigned_dtype), lax_full(integer(0), n, unsigned_dtype))
  carry <- lax_scan(body_fun, init, NULL, length = n_levels)[[1L]]
  lax_convert_element_type(carry[[2L]], dtype)
}

# Port of hijax._searchsorted_sort_impl. JAX writes the inverse permutation with a
# scatter (`.at[perm].set(idx)`); sorting the permutation gives the same ranks.
searchsorted_sort_impl <- function(sorted_arr, query, side, dtype) {
  n <- prod_int(aval_of(sorted_arr)$shape)
  qsize <- prod_int(aval_of(query)$shape)
  working_dtype <- if (n + qsize <= 2147483647) "int32" else "int64"
  rank <- function(x) {
    idx <- lax_iota(working_dtype, prod_int(aval_of(x)$shape))
    perm <- lax_sort_key_val(x, idx)[[2L]]
    lax_sort_key_val(perm, idx)[[2L]]
  }
  q <- lax_reshape(query, qsize)
  index <- if (side == "left") {
    lax_slice(rank(lax_concatenate(list(q, sorted_arr), 0L)), 0L, qsize)
  } else {
    lax_slice(rank(lax_concatenate(list(sorted_arr, q), 0L)), n, n + qsize)
  }
  lax_convert_element_type(lax_reshape(lax_sub(index, rank(q)), aval_of(query)$shape), dtype)
}

# Port of hijax._searchsorted_compare_all_impl.
searchsorted_compare_all_impl <- function(sorted_arr, query, side, dtype) {
  op <- if (side == "left") sort_lt_comparator else sort_le_comparator
  comparisons <- vmap(function(a, q) op(list(a, q)), in_axes = list(0L, NULL))(sorted_arr, query)
  jnp_sum(comparisons, axis = 0L, dtype = dtype)
}

#' Find insertion indices in a sorted array
#'
#' Port of `jnp.searchsorted` (0-based indices), with the methods `"scan"` (binary
#' search with [lax_scan()]), `"scan_unrolled"` (the same here), `"sort"` and
#' `"compare_all"`.
#' @param a Sorted one dimensional array.
#' @param v Query values.
#' @param side `"left"` or `"right"`.
#' @param sorter Optional 0-based indices that sort `a`.
#' @param method One of `"scan"`, `"scan_unrolled"`, `"sort"`, `"compare_all"`.
#' @return Integer array with the shape of `v`.
#' @export
jnp_searchsorted <- function(a, v, side = "left", sorter = NULL, method = "scan") {
  a <- as_jax(a)
  if (length(aval_of(a)$shape) != 1L)
    jax_error("a should be 1-dimensional", class = "jax_value_error")
  av <- promote_dtypes(list(a, v))
  a <- av[[1L]]
  v <- av[[2L]]
  if (!is.null(sorter)) a <- take_along_axis_gather(a, as_jax(sorter), 0L)
  n <- aval_of(a)$shape[1L]
  dtype <- if (n > 2147483647) "int64" else "int32"
  if (!side %in% c("left", "right"))
    jax_error("invalid argument side='", side, "', expected 'left' or 'right'",
              class = "jax_value_error")
  methods <- c("compare_all", "scan", "scan_unrolled", "sort")
  if (!method %in% methods)
    jax_error("invalid argument method='", method, "', expected one of ['",
              paste(methods, collapse = "', '"), "']", class = "jax_value_error")
  impl <- switch(method, compare_all = searchsorted_compare_all_impl,
                 sort = searchsorted_sort_impl, searchsorted_scan_impl)
  impl(a, v, side, dtype)
}
