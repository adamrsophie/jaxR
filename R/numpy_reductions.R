# jax.numpy reductions: ports of jax/_src/numpy/reductions.py (nan reductions, minmax,
# average, ptp, count_nonzero, quantiles and medians) and of the reductions defined in
# jax/_src/numpy/lax_numpy.py (nanargmax, nanargmin, cov, corrcoef).
#
# The `red_` functions are complete ports of the core reductions (`where`, `initial`,
# `promote_integers`, `correction`, `mean`); everything here is built on them. The older
# jnp_sum, jnp_mean, ... of R/numpy_core.R can become thin wrappers over them.

# ---- private helpers ------------------------------------------------------------------------

red_value_error <- function(...) jax_error(..., class = "jax_value_error")

red_tuple <- function(x) {
  if (length(x) == 1L) paste0("(", x, ",)") else paste0("(", paste(x, collapse = ", "), ")")
}

# Python range(a, b).
red_range <- function(a, b) if (b > a) seq.int(a, b - 1L) else integer(0)

red_ndim <- function(x) length(aval_of(x)$shape)

red_check_out <- function(name, out) {
  if (!is.null(out))
    jax_error("The 'out' argument to jnp.", name, " is not supported.",
              class = "jax_not_implemented_error")
}

# Port of reductions.py:check_where.
red_check_where <- function(name, where) {
  if (is.null(where)) return(NULL)
  where <- as_jax(where)
  dt <- aval_of(where)$dtype
  if (dt != "bool")
    red_value_error("jnp.", name, ": where must be None or a boolean array; got where.dtype=dtype('",
                    dt, "').")
  where
}

# Port of reductions.py:_ensure_optional_axes: axes must be static.
red_static_axis <- function(axis) {
  if (is.null(axis) || !is_jax_value(axis)) return(axis)
  as.integer(as.vector(concrete_or_error(axis, "the axis argument (it must be known statically)")))
}

# Port of reductions.py:_reduction_dims (positional axes; the order given is kept).
red_dims <- function(a, axis) {
  r <- red_ndim(a)
  if (is.null(axis)) return(seq_len(r) - 1L)
  axis <- red_static_axis(axis)
  dims <- norm_axis(axis, r)
  if (anyDuplicated(dims)) red_value_error("duplicate value in 'axis': ", red_tuple(axis))
  dims
}

# Port of reductions.py:_upcast_f16.
red_upcast_f16 <- function(dt) if (dt %in% c("float16", "bfloat16")) "float32" else dt

red_cast_to_numeric <- function(x) promote_dtypes_numeric(list(x))[[1L]]

# Port of reductions.py:_cast_to_bool.
red_cast_to_bool <- function(x) {
  if (is_complex_dtype(aval_of(x)$dtype)) x <- lax_real(x)
  lax_convert_element_type(x, "bool")
}

red_isnan <- function(x) lax_ne(x, x)

# Port of reductions.py:_reduction_init_val: a strong scalar of x's dtype; infinities
# saturate to the integer extremes.
red_init_val <- function(x, v) {
  dt <- aval_of(x)$dtype
  if (dt == "bool") return(lax_full(integer(0), v > 0, "bool"))
  if (is_int_dtype(dt) && is.infinite(v))
    return(gs_int_extreme_like(lax_full(integer(0), 0, dt), if (v < 0) "min" else "max"))
  lax_full(integer(0), v, dt)
}

red_reduce_fn <- function(op) {
  switch(op, add = lax_reduce_sum, mul = lax_reduce_prod, max = lax_reduce_max,
         min = lax_reduce_min, and = lax_reduce_and, or = lax_reduce_or, xor = lax_reduce_xor)
}

red_binop <- function(op) {
  switch(op, add = lax_add, mul = lax_mul, max = lax_max, min = lax_min,
         and = lax_bitwise_and, or = lax_bitwise_or, xor = lax_bitwise_xor)
}

# Port of reductions.py:_reduction. `op` names the monoid ("add", "mul", "max", "min",
# "and", "or", "xor"); lax.reduce with these monoids binds the reduce_* primitives. On
# booleans max and min are the logical or and and.
red_reduction <- function(a, name, op, init_val, has_identity = TRUE, preproc = NULL,
                          bool_op = NULL, upcast_f16_for_computation = FALSE, axis = NULL,
                          dtype = NULL, out = NULL, keepdims = FALSE, initial = NULL,
                          where = NULL, promote_integers = FALSE) {
  bool_op <- bool_op %||% op
  red_check_out(name, out)
  a <- as_jax(a)
  where <- red_check_where(name, where)
  if (is.null(initial) && !has_identity && !is.null(where))
    red_value_error("reduction operation ", name, " does not have an identity, so to use a ",
                    "where mask one has to specify 'initial'")
  if (!is.null(preproc)) a <- preproc(a)
  dims <- red_dims(a, axis)
  if (is.null(initial) && !has_identity && any(aval_of(a)$shape[dims + 1L] < 1L))
    red_value_error("zero-size array to reduction operation ", name, " which has no identity")
  result_dtype <- if (!is.null(dtype)) dtype(dtype)
                  else if (promote_integers) promote_integer_dtype(aval_of(a)$dtype)
                  else aval_of(a)$dtype
  computation_dtype <- if (upcast_f16_for_computation && is_inexact_dtype(result_dtype))
    red_upcast_f16(result_dtype) else result_dtype
  a <- lax_convert_element_type(a, computation_dtype)
  if (computation_dtype == "bool") op <- switch(bool_op, max = "or", min = "and", bool_op)
  if (!is.null(where)) a <- jnp_where(where, a, red_init_val(a, init_val))
  result <- red_reduce_fn(op)(a, dims)
  if (!is.null(initial)) {
    initial_arr <- lax_convert_element_type(initial, computation_dtype)
    if (red_ndim(initial_arr))
      red_value_error("initial value must be a scalar. Got array of shape ",
                      red_tuple(aval_of(initial_arr)$shape))
    result <- red_binop(op)(initial_arr, result)
  }
  if (keepdims) result <- lax_expand_dims(result, dims)
  lax_convert_element_type(result, result_dtype)
}

# Ports of reductions.py:_reduce_sum, _reduce_prod, _reduce_max, _reduce_min,
# _reduce_all and _reduce_any.
red_sum <- function(a, axis = NULL, dtype = NULL, out = NULL, keepdims = FALSE, initial = NULL,
                    where = NULL, promote_integers = TRUE) {
  red_reduction(a, "sum", "add", 0, preproc = red_cast_to_numeric, bool_op = "or",
                upcast_f16_for_computation = is.null(dtype), axis = axis, dtype = dtype,
                out = out, keepdims = keepdims, initial = initial, where = where,
                promote_integers = promote_integers)
}
red_prod <- function(a, axis = NULL, dtype = NULL, out = NULL, keepdims = FALSE, initial = NULL,
                     where = NULL, promote_integers = TRUE) {
  red_reduction(a, "prod", "mul", 1, preproc = red_cast_to_numeric, bool_op = "and",
                upcast_f16_for_computation = is.null(dtype), axis = axis, dtype = dtype,
                out = out, keepdims = keepdims, initial = initial, where = where,
                promote_integers = promote_integers)
}
red_max <- function(a, axis = NULL, out = NULL, keepdims = FALSE, initial = NULL, where = NULL,
                    name = "max") {
  red_reduction(a, name, "max", -Inf, has_identity = FALSE, axis = axis, out = out,
                keepdims = keepdims, initial = initial, where = where)
}
red_min <- function(a, axis = NULL, out = NULL, keepdims = FALSE, initial = NULL, where = NULL,
                    name = "min") {
  red_reduction(a, name, "min", Inf, has_identity = FALSE, axis = axis, out = out,
                keepdims = keepdims, initial = initial, where = where)
}
red_all <- function(a, axis = NULL, out = NULL, keepdims = FALSE, where = NULL) {
  red_reduction(a, "all", "and", TRUE, preproc = red_cast_to_bool, axis = axis, out = out,
                keepdims = keepdims, where = where)
}
red_any <- function(a, axis = NULL, out = NULL, keepdims = FALSE, where = NULL) {
  red_reduction(a, "any", "or", FALSE, preproc = red_cast_to_bool, axis = axis, out = out,
                keepdims = keepdims, where = where)
}

# Port of reductions.py:_count.
red_count <- function(a, axis, keepdims, where, dtype) {
  if (is.null(where)) {
    shape <- aval_of(a)$shape
    n <- if (is.null(axis)) prod(shape) else prod(shape[red_dims(a, axis) + 1L])
    return(lax_convert_element_type(n, dtype))
  }
  red_sum(jnp_broadcast_to(where, aval_of(a)$shape), axis, dtype = dtype, keepdims = keepdims)
}

# Port of reductions.py:_mean.
red_mean <- function(a, axis = NULL, dtype = NULL, out = NULL, keepdims = FALSE, where = NULL,
                     upcast_f16_for_computation = is.null(dtype)) {
  a <- as_jax(a)
  where <- red_check_where("mean", where)
  red_check_out("mean", out)
  result_dtype <- if (is.null(dtype)) to_inexact_dtype(aval_of(a)$dtype) else dtype(dtype)
  computation_dtype <- if (upcast_f16_for_computation && is_inexact_dtype(result_dtype))
    red_upcast_f16(result_dtype) else result_dtype
  normalizer <- red_count(a, axis, keepdims, where, computation_dtype)
  s <- red_sum(a, axis, dtype = computation_dtype, keepdims = keepdims, where = where)
  lax_convert_element_type(lax_div(s, normalizer), result_dtype)
}

# Port of reductions.py:_var_promote_types: list(computation dtype, result dtype).
red_var_promote_types <- function(a_dtype, dtype) {
  if (!is.null(dtype)) {
    if (!is_complex_dtype(dtype) && is_complex_dtype(a_dtype))
      red_value_error("jax.numpy.var does not yet support real dtype parameters when computing ",
                      "the variance of an array of complex values. The semantics of numpy.var ",
                      "seem unclear in this case. Please comment on ",
                      "https://github.com/jax-ml/jax/issues/2283 if this behavior is important ",
                      "to you.")
    computation_dtype <- dtype
  } else if (!is_inexact_dtype(a_dtype)) {
    dtype <- to_inexact_dtype(a_dtype)
    computation_dtype <- dtype
  } else {
    dtype <- complex_component(a_dtype)
    computation_dtype <- a_dtype
  }
  list(red_upcast_f16(computation_dtype), dtype)
}

red_correction <- function(ddof, correction) {
  if (is.null(correction)) return(ddof)
  if (!(is.numeric(ddof) && length(ddof) == 1L && ddof == 0))
    red_value_error("ddof and correction can't be provided simultaneously.")
  correction
}

# Ports of reductions.py:var and _var.
red_var <- function(a, axis = NULL, dtype = NULL, out = NULL, ddof = 0L, keepdims = FALSE,
                    where = NULL, mean = NULL, correction = NULL) {
  correction <- red_correction(ddof, correction)
  a <- as_jax(a)
  where <- red_check_where("var", where)
  if (!is.null(dtype)) dtype <- dtype(dtype)
  red_check_out("var", out)
  pt <- red_var_promote_types(aval_of(a)$dtype, dtype)
  computation_dtype <- pt[[1L]]
  dtype <- pt[[2L]]
  a <- lax_convert_element_type(a, computation_dtype)
  a_mean <- if (is.null(mean)) red_mean(a, axis, dtype = computation_dtype, keepdims = TRUE,
                                        where = where)
            else lax_convert_element_type(mean, computation_dtype)
  centered <- lax_sub(a, a_mean)
  if (is_complex_dtype(computation_dtype)) {
    centered <- lax_real(lax_mul(centered, lax_conj(centered)))
    computation_dtype <- aval_of(centered)$dtype
  } else {
    centered <- lax_square(centered)
  }
  normalizer <- red_count(a, axis, keepdims, where, computation_dtype)
  normalizer <- lax_sub(normalizer, lax_convert_element_type(correction, computation_dtype))
  result <- red_sum(centered, axis, dtype = computation_dtype, keepdims = keepdims, where = where)
  result <- lax_convert_element_type(lax_div(result, normalizer), dtype)
  jnp_where(lax_gt(normalizer, 0), result, NaN)
}

# Ports of reductions.py:std and _std.
red_std <- function(a, axis = NULL, dtype = NULL, out = NULL, ddof = 0L, keepdims = FALSE,
                    where = NULL, mean = NULL, correction = NULL) {
  correction <- red_correction(ddof, correction)
  a <- as_jax(a)
  red_check_where("std", where)
  if (!is.null(dtype)) {
    dtype <- dtype(dtype)
    if (!is_inexact_dtype(dtype))
      red_value_error("dtype argument to jnp.std must be inexact; got ", dtype)
  }
  red_check_out("std", out)
  lax_sqrt(red_var(a, axis, dtype, correction = correction, keepdims = keepdims, where = where,
                   mean = mean))
}

# Port of lax_numpy.py:_argmax / _argmin.
red_arg_extreme <- function(a, axis, keepdims, which) {
  a <- as_jax(a)
  if (is.null(axis)) {
    dims <- seq_len(red_ndim(a)) - 1L
    a <- jnp_ravel(a)
    axis <- 0L
  } else {
    axis <- norm_axis(red_static_axis(axis), red_ndim(a))
    dims <- axis
  }
  if (aval_of(a)$shape[axis + 1L] == 0L)
    red_value_error("attempt to get ", which, " of an empty sequence")
  f <- if (which == "argmax") lax_argmax else lax_argmin
  result <- f(a, axis, default_int())
  if (keepdims) lax_expand_dims(result, dims) else result
}

# Port of reductions.py:_nan_reduction.
red_nan_reduction <- function(a, name, reduction, init_val, nan_if_all_nan, axis = NULL,
                              keepdims = FALSE, where = NULL, ...) {
  a <- as_jax(a)
  where <- red_check_where(name, where)
  if (!is_inexact_dtype(aval_of(a)$dtype))
    return(reduction(a, axis = axis, keepdims = keepdims, where = where, ...))
  out <- reduction(jnp_where(red_isnan(a), red_init_val(a, init_val), a), axis = axis,
                   keepdims = keepdims, where = where, ...)
  if (!nan_if_all_nan) return(out)
  jnp_where(red_all(red_isnan(a), axis = axis, keepdims = keepdims),
            lax_full(integer(0), NaN, aval_of(a)$dtype), out)
}

# ---- nan reductions ---------------------------------------------------------------------------

#' Reductions that ignore NaN
#'
#' Ports of `jnp.nansum`, `jnp.nanprod`, `jnp.nanmax`, `jnp.nanmin`, `jnp.nanmean`,
#' `jnp.nanvar`, `jnp.nanstd`, `jnp.nanargmax` and `jnp.nanargmin`. NaN entries are
#' skipped; integer and boolean inputs reduce like the plain reductions. `nanmax` and
#' `nanmin` (without `initial`) give NaN where every entry is NaN, `nanargmax` and
#' `nanargmin` give -1 there.
#' @param a Array.
#' @param axis 0-based axis or axes to reduce; `NULL` reduces all. `nanargmax` and
#'   `nanargmin` take a single axis.
#' @param dtype Optional accumulation and result dtype.
#' @param out Not supported (must be `NULL`).
#' @param keepdims Keep reduced dimensions as size one.
#' @param initial Optional scalar starting value of the reduction.
#' @param where Optional boolean mask of the elements to include.
#' @param ddof Delta degrees of freedom.
#' @param mean Optional precomputed mean (broadcastable against `a` with kept dims).
#' @return An array.
#' @name jnp_nan_reductions
NULL

#' @rdname jnp_nan_reductions
#' @export
jnp_nanmin <- function(a, axis = NULL, out = NULL, keepdims = FALSE, initial = NULL,
                       where = NULL) {
  red_nan_reduction(a, "nanmin", red_min, Inf, is.null(initial), axis = axis, out = out,
                    keepdims = keepdims, initial = initial, where = where)
}

#' @rdname jnp_nan_reductions
#' @export
jnp_nanmax <- function(a, axis = NULL, out = NULL, keepdims = FALSE, initial = NULL,
                       where = NULL) {
  red_nan_reduction(a, "nanmax", red_max, -Inf, is.null(initial), axis = axis, out = out,
                    keepdims = keepdims, initial = initial, where = where)
}

#' @rdname jnp_nan_reductions
#' @export
jnp_nansum <- function(a, axis = NULL, dtype = NULL, out = NULL, keepdims = FALSE,
                       initial = NULL, where = NULL) {
  if (!is.null(dtype)) dtype <- dtype(dtype)
  red_nan_reduction(a, "nansum", red_sum, 0, FALSE, axis = axis, dtype = dtype, out = out,
                    keepdims = keepdims, initial = initial, where = where)
}

#' @rdname jnp_nan_reductions
#' @export
jnp_nanprod <- function(a, axis = NULL, dtype = NULL, out = NULL, keepdims = FALSE,
                        initial = NULL, where = NULL) {
  if (!is.null(dtype)) dtype <- dtype(dtype)
  red_nan_reduction(a, "nanprod", red_prod, 1, FALSE, axis = axis, dtype = dtype, out = out,
                    keepdims = keepdims, initial = initial, where = where)
}

#' @rdname jnp_nan_reductions
#' @export
jnp_nanmean <- function(a, axis = NULL, dtype = NULL, out = NULL, keepdims = FALSE,
                        where = NULL) {
  a <- as_jax(a)
  where <- red_check_where("nanmean", where)
  red_check_out("nanmean", out)
  if (!is_inexact_dtype(aval_of(a)$dtype))
    return(red_mean(a, axis, dtype, out, keepdims, where = where))
  dtype <- if (is.null(dtype)) to_inexact_dtype(aval_of(a)$dtype) else dtype(dtype)
  normalizer <- red_sum(lax_bitwise_not(red_isnan(a)), axis = axis, dtype = dtype,
                        keepdims = keepdims, where = where)
  lax_div(jnp_nansum(a, axis, dtype = dtype, keepdims = keepdims, where = where), normalizer)
}

#' @rdname jnp_nan_reductions
#' @export
jnp_nanvar <- function(a, axis = NULL, dtype = NULL, out = NULL, ddof = 0L, keepdims = FALSE,
                       where = NULL, mean = NULL) {
  a <- as_jax(a)
  where <- red_check_where("nanvar", where)
  if (!is.null(dtype)) dtype <- dtype(dtype)
  red_check_out("nanvar", out)
  # Port of reductions.py:_nanvar.
  pt <- red_var_promote_types(aval_of(a)$dtype, dtype)
  computation_dtype <- pt[[1L]]
  dtype <- pt[[2L]]
  a <- lax_convert_element_type(a, computation_dtype)
  a_mean <- if (is.null(mean)) jnp_nanmean(a, axis, dtype = computation_dtype, keepdims = TRUE,
                                           where = where)
            else lax_convert_element_type(mean, computation_dtype)
  centered <- jnp_where(red_isnan(a), 0, lax_sub(a, a_mean))
  centered <- if (is_complex_dtype(aval_of(centered)$dtype))
    lax_real(lax_mul(centered, lax_conj(centered))) else lax_square(centered)
  normalizer <- red_sum(lax_bitwise_not(red_isnan(a)), axis = axis, keepdims = keepdims,
                        where = where)
  normalizer <- jnp_subtract(normalizer, ddof)
  normalizer_mask <- lax_le(normalizer, lax_full_like(normalizer, 0))
  result <- red_sum(centered, axis, keepdims = keepdims, where = where)
  result <- jnp_where(normalizer_mask, NaN, result)
  divisor <- jnp_where(normalizer_mask, 1L, normalizer)
  result <- lax_div(result, lax_convert_element_type(divisor, aval_of(result)$dtype))
  lax_convert_element_type(result, dtype)
}

#' @rdname jnp_nan_reductions
#' @export
jnp_nanstd <- function(a, axis = NULL, dtype = NULL, out = NULL, ddof = 0L, keepdims = FALSE,
                       where = NULL, mean = NULL) {
  a <- as_jax(a)
  where <- red_check_where("nanstd", where)
  if (!is.null(dtype)) dtype <- dtype(dtype)
  red_check_out("nanstd", out)
  lax_sqrt(jnp_nanvar(a, axis = axis, dtype = dtype, ddof = ddof, keepdims = keepdims,
                      where = where, mean = mean))
}

red_nan_arg_extreme <- function(a, axis, out, keepdims, which) {
  red_check_out(paste0("nan", which), out)
  a <- as_jax(a)
  keepdims <- isTRUE(keepdims)
  if (!is_inexact_dtype(aval_of(a)$dtype)) return(red_arg_extreme(a, axis, keepdims, which))
  nan_mask <- jnp_isnan(a)
  a <- jnp_where(nan_mask, if (which == "argmax") -Inf else Inf, a)
  res <- red_arg_extreme(a, axis, keepdims, which)
  jnp_where(red_all(nan_mask, axis = axis, keepdims = keepdims), -1L, res)
}

# Ports of lax_numpy.py:nanargmax / _nanargmax and nanargmin / _nanargmin.
#' @rdname jnp_nan_reductions
#' @export
jnp_nanargmax <- function(a, axis = NULL, out = NULL, keepdims = NULL) {
  red_nan_arg_extreme(a, axis, out, keepdims, "argmax")
}

#' @rdname jnp_nan_reductions
#' @export
jnp_nanargmin <- function(a, axis = NULL, out = NULL, keepdims = NULL) {
  red_nan_arg_extreme(a, axis, out, keepdims, "argmin")
}

# ---- statistics ---------------------------------------------------------------------------------

#' Statistical reductions
#'
#' Ports of `jnp.minmax` (minimum and maximum in one call), `jnp.ptp` (range),
#' `jnp.count_nonzero` and `jnp.average` (weighted mean).
#' @param a Array.
#' @param axis 0-based axis or axes to reduce; `NULL` reduces all.
#' @param out Not supported (must be `NULL`).
#' @param keepdims Keep reduced dimensions as size one.
#' @param initial `minmax`: a scalar starting value for both reductions, or a list of
#'   two (for the minimum and the maximum).
#' @param where `minmax`: optional boolean mask (needs `initial`).
#' @param weights `average`: optional weights, of the shape of `a` or of the reduced axes.
#' @param returned `average`: also return the sum of the weights.
#' @return An array; `minmax` returns `list(min, max)` and `average` with
#'   `returned = TRUE` returns `list(average, sum_of_weights)`.
#' @name jnp_statistics
NULL

# Port of reductions.py:minmax and _reduce_minmax (the variadic reduce is two reductions).
#' @rdname jnp_statistics
#' @export
jnp_minmax <- function(a, axis = NULL, out = NULL, keepdims = FALSE, initial = NULL,
                       where = NULL) {
  if (is.list(initial)) {
    init_min <- initial[[1L]]
    init_max <- initial[[2L]]
  } else {
    init_min <- init_max <- initial
  }
  red_check_out("minmax", out)
  if ((is.null(init_min) || is.null(init_max)) && !is.null(where))
    red_value_error("reduction operation minmax does not have an identity, so to use a where ",
                    "mask one has to specify 'initial'")
  list(red_min(a, axis, keepdims = keepdims, initial = init_min, where = where, name = "minmax"),
       red_max(a, axis, keepdims = keepdims, initial = init_max, where = where, name = "minmax"))
}

# Port of reductions.py:ptp / _ptp.
#' @rdname jnp_statistics
#' @export
jnp_ptp <- function(a, axis = NULL, out = NULL, keepdims = FALSE) {
  red_check_out("ptp", out)
  a <- as_jax(a)
  lax_sub(red_max(a, axis, keepdims = keepdims), red_min(a, axis, keepdims = keepdims))
}

# Port of reductions.py:count_nonzero.
#' @rdname jnp_statistics
#' @export
jnp_count_nonzero <- function(a, axis = NULL, keepdims = FALSE) {
  a <- as_jax(a)
  red_sum(lax_ne(a, lax_full(integer(0), 0, aval_of(a)$dtype)), axis = axis,
          dtype = default_int(), keepdims = keepdims)
}

# Port of reductions.py:average / _average.
#' @rdname jnp_statistics
#' @export
jnp_average <- function(a, axis = NULL, weights = NULL, returned = FALSE, keepdims = FALSE) {
  a <- as_jax(a)
  axis <- red_static_axis(axis)
  axis_tuple <- if (is.null(axis)) seq_len(red_ndim(a)) - 1L else norm_axis(axis, red_ndim(a))
  if (is.null(weights)) {
    a <- promote_dtypes_inexact(list(a))[[1L]]
    avg <- red_mean(a, axis, keepdims = keepdims)
    weights_sum <- lax_full(integer(0), prod(aval_of(a)$shape[axis_tuple + 1L]),
                            aval_of(avg)$dtype)
  } else {
    aw <- promote_dtypes_inexact(list(a, weights))
    a <- aw[[1L]]
    weights <- aw[[2L]]
    a_shape <- aval_of(a)$shape
    w_shape <- aval_of(weights)$shape
    if (!identical(a_shape, w_shape)) {
      if (is.null(axis))
        red_value_error("Axis must be specified when shapes of a and weights differ.")
      if (!identical(w_shape, a_shape[axis_tuple + 1L]))
        red_value_error("Shape of weights must be consistent with shape of a along specified axis.")
      new_shape <- replace(rep(1L, length(a_shape)), axis_tuple + 1L, a_shape[axis_tuple + 1L])
      weights <- lax_reshape(weights, new_shape, dimensions = order(axis_tuple) - 1L)
    }
    weights_sum <- red_sum(weights, axis, keepdims = keepdims)
    avg <- jnp_true_divide(red_sum(jnp_multiply(a, weights), axis, keepdims = keepdims),
                           weights_sum)
  }
  if (!returned) return(avg)
  if (!identical(aval_of(avg)$shape, aval_of(weights_sum)$shape))
    weights_sum <- jnp_broadcast_to(weights_sum, aval_of(avg)$shape)
  list(avg, weights_sum)
}

# ---- quantiles ----------------------------------------------------------------------------------

#' Quantiles, percentiles and medians
#'
#' Ports of `jnp.quantile`, `jnp.nanquantile`, `jnp.percentile`, `jnp.nanpercentile`,
#' `jnp.median` and `jnp.nanmedian`. The `nan` variants ignore NaN entries; the others
#' return NaN for slices that contain one.
#' @param a Array.
#' @param q Quantile (in \[0, 1\]) or percentile (in \[0, 100\]); a scalar or 1-D array.
#'   A 1-D `q` adds a leading dimension to the result.
#' @param axis 0-based axis or axes to reduce; `NULL` reduces all.
#' @param out Not supported (must be `NULL`).
#' @param overwrite_input Not supported (must be `FALSE`).
#' @param method One of `"linear"`, `"lower"`, `"higher"`, `"midpoint"`, `"nearest"`,
#'   `"inverted_cdf"`.
#' @param keepdims Keep reduced dimensions as size one.
#' @param weights Optional weights (only with `method = "inverted_cdf"`).
#' @return An array.
#' @name jnp_quantile
NULL

red_quantile_args <- function(name, out, overwrite_input) {
  if (isTRUE(overwrite_input) || !is.null(out))
    red_value_error("jax.numpy.", name, " does not support overwrite_input=True or out != None")
}

#' @rdname jnp_quantile
#' @export
jnp_quantile <- function(a, q, axis = NULL, out = NULL, overwrite_input = FALSE,
                         method = "linear", keepdims = FALSE, weights = NULL) {
  red_quantile_args("quantile", out, overwrite_input)
  red_quantile(as_jax(a), as_jax(q), axis, method, keepdims, FALSE,
               if (is.null(weights)) NULL else as_jax(weights))
}

#' @rdname jnp_quantile
#' @export
jnp_nanquantile <- function(a, q, axis = NULL, out = NULL, overwrite_input = FALSE,
                            method = "linear", keepdims = FALSE, weights = NULL) {
  red_quantile_args("nanquantile", out, overwrite_input)
  red_quantile(as_jax(a), as_jax(q), axis, method, keepdims, TRUE,
               if (is.null(weights)) NULL else as_jax(weights))
}

#' @rdname jnp_quantile
#' @export
jnp_percentile <- function(a, q, axis = NULL, out = NULL, overwrite_input = FALSE,
                           method = "linear", keepdims = FALSE, weights = NULL) {
  q <- promote_dtypes_inexact(list(q))[[1L]]
  jnp_quantile(a, jnp_true_divide(q, 100), axis = axis, out = out,
               overwrite_input = overwrite_input, method = method, keepdims = keepdims,
               weights = weights)
}

#' @rdname jnp_quantile
#' @export
jnp_nanpercentile <- function(a, q, axis = NULL, out = NULL, overwrite_input = FALSE,
                              method = "linear", keepdims = FALSE, weights = NULL) {
  q <- promote_dtypes_inexact(list(q))[[1L]]
  jnp_nanquantile(a, jnp_true_divide(q, 100), axis = axis, out = out,
                  overwrite_input = overwrite_input, method = method, keepdims = keepdims,
                  weights = weights)
}

#' @rdname jnp_quantile
#' @export
jnp_median <- function(a, axis = NULL, out = NULL, overwrite_input = FALSE, keepdims = FALSE) {
  a <- as_jax(a)
  axis <- red_static_axis(axis)
  reduction_axes <- if (is.null(axis)) seq_len(red_ndim(a)) - 1L else norm_axis(axis, red_ndim(a))
  if (any(aval_of(a)$shape[reduction_axes + 1L] == 0L))
    red_value_error("jax.numpy.median reduction axis must be non-empty")
  jnp_quantile(a, 0.5, axis = axis, out = out, overwrite_input = overwrite_input,
               keepdims = keepdims, method = "midpoint")
}

#' @rdname jnp_quantile
#' @export
jnp_nanmedian <- function(a, axis = NULL, out = NULL, overwrite_input = FALSE, keepdims = FALSE) {
  jnp_nanquantile(a, 0.5, axis = axis, out = out, overwrite_input = overwrite_input,
                  keepdims = keepdims, method = "midpoint")
}

# Port of reductions.py:_quantile.
red_quantile <- function(a, q, axis, method, keepdims, squash_nans, weights = NULL) {
  methods <- c("linear", "lower", "higher", "midpoint", "nearest", "inverted_cdf")
  if (!(is.character(method) && length(method) == 1L && method %in% methods))
    red_value_error("method can only be 'linear', 'lower', 'higher', 'midpoint', 'nearest' or ",
                    "'inverted_cdf'")
  axis <- red_static_axis(axis)
  if (!is.null(weights)) {
    if (is_complex_dtype(aval_of(weights)$dtype)) red_value_error("Weights cannot be complex types.")
    if (method != "inverted_cdf")
      jax_error(method, " doesn't support weights. Only method 'inverted_cdf' supports weights.",
                class = "jax_not_implemented_error")
    aw <- promote_dtypes_inexact(list(a, weights))
    a <- aw[[1L]]
    weights <- aw[[2L]]
    if (!identical(aval_of(weights)$shape, aval_of(a)$shape)) {
      if (is.null(axis)) red_value_error("Weights shape must match 'a' shape when axis is None.")
      ax_tuple <- norm_axis(axis, red_ndim(a))
      want <- aval_of(a)$shape[ax_tuple + 1L]
      if (!identical(aval_of(weights)$shape, want))
        red_value_error("Weights shape ", red_tuple(aval_of(weights)$shape),
                        " must match reduction axes ", red_tuple(want))
      weights <- lax_broadcast_in_dim(weights, aval_of(a)$shape, ax_tuple)
    }
  } else {
    a <- promote_dtypes_inexact(list(a))[[1L]]
  }
  keepdim <- integer(0)
  if (is_complex_dtype(aval_of(a)$dtype))
    red_value_error("quantile does not support complex input, as the operation is poorly defined.")
  if (is.null(axis)) {
    if (keepdims) keepdim <- rep(1L, red_ndim(a))
    a <- jnp_ravel(a)
    if (!is.null(weights)) weights <- jnp_ravel(weights)
    axis <- 0L
  } else if (length(axis) != 1L) {
    # Several axes, or none (an empty tuple reduces over a new unit dimension).
    keepdim <- aval_of(a)$shape
    nd <- red_ndim(a)
    axis <- norm_axis(axis, nd)
    if (anyDuplicated(axis)) red_value_error("repeated axis")
    keepdim[axis + 1L] <- 1L
    keep <- setdiff(seq_len(nd) - 1L, axis)
    dimensions <- seq_len(nd) - 1L
    for (i in seq_along(keep)) {
      s <- keep[i]
      tmp <- dimensions[i]
      dimensions[i] <- dimensions[s + 1L]
      dimensions[s + 1L] <- tmp
    }
    shp <- aval_of(a)$shape
    new_shape <- c(shp[keep + 1L], prod_int(shp[axis + 1L]))
    a <- lax_reshape(a, new_shape, dimensions)
    if (!is.null(weights)) weights <- lax_reshape(weights, new_shape, dimensions)
    axis <- red_ndim(a) - 1L
  } else {
    axis <- norm_axis(axis, red_ndim(a))
  }
  q_shape <- aval_of(q)$shape
  q_ndim <- length(q_shape)
  if (q_ndim > 1L) red_value_error("q must be have rank <= 1, got shape ", red_tuple(q_shape))
  a_shape <- aval_of(a)$shape
  q_orig <- q
  int_t <- default_int()
  if (squash_nans && a_shape[axis + 1L] == 0L) {
    if (keepdims && length(keepdim)) {
      result_shape <- keepdim
    } else {
      result_shape <- a_shape
      if (keepdims) result_shape[axis + 1L] <- 1L else result_shape <- result_shape[-(axis + 1L)]
    }
    return(lax_full(result_shape, NaN, aval_of(a)$dtype))
  }
  if (squash_nans) {
    a <- jnp_where(red_isnan(a), NaN, a)
    if (!is.null(weights)) {
      kv <- lax_sort_key_val(a, weights, dimension = axis)
      a <- kv[[1L]]
      weights <- kv[[2L]]
    } else {
      a <- lax_sort(a, dimension = axis, is_stable = FALSE)
    }
    counts <- red_sum(lax_bitwise_not(red_isnan(a)), axis = axis, dtype = aval_of(q)$dtype,
                      keepdims = keepdims)
    shape_after_reduction <- aval_of(counts)$shape
    q <- lax_expand_dims(q, red_range(q_ndim, length(shape_after_reduction) + q_ndim))
    counts <- lax_expand_dims(counts, red_range(0L, q_ndim))
    q <- lax_mul(q, lax_sub(counts, lax_full_like(counts, 1)))
    low <- lax_floor(q)
    high <- lax_ceil(q)
    high_weight <- lax_sub(q, low)
    low_weight <- lax_sub(lax_full_like(high_weight, 1), high_weight)
    cm1 <- lax_sub(counts, lax_full_like(counts, 1))
    low <- lax_max(lax_full_like(low, 0), lax_min(low, cm1))
    high <- lax_max(lax_full_like(high, 0), lax_min(high, cm1))
    low <- lax_convert_element_type(low, int_t)
    high <- lax_convert_element_type(high, int_t)
    out_shape <- c(q_shape, shape_after_reduction)
    index <- lapply(seq_along(shape_after_reduction) - 1L, function(dim)
      lax_broadcasted_iota(int_t, out_shape, dim + q_ndim))
    if (keepdims) index[[axis + 1L]] <- low else index <- append(index, list(low), after = axis)
    low_value <- do.call(`[`, c(list(a), index))
    index[[axis + 1L]] <- high
    high_value <- do.call(`[`, c(list(a), index))
  } else {
    a <- jnp_where(red_any(red_isnan(a), axis = axis, keepdims = TRUE), NaN, a)
    if (!is.null(weights)) {
      kv <- lax_sort_key_val(a, weights, dimension = axis)
      a <- kv[[1L]]
      weights <- kv[[2L]]
    } else {
      a <- lax_sort(a, dimension = axis, is_stable = FALSE)
    }
    n <- lax_convert_element_type(a_shape[axis + 1L], aval_of(q)$dtype)
    nm1 <- lax_sub(n, lax_full_like(n, 1))
    q <- lax_mul(q, nm1)
    low <- lax_floor(q)
    high <- lax_ceil(q)
    high_weight <- lax_sub(q, low)
    low_weight <- lax_sub(lax_full_like(high_weight, 1), high_weight)
    low <- lax_clamp(lax_full_like(low, 0), low, nm1)
    high <- lax_clamp(lax_full_like(high, 0), high, nm1)
    low <- lax_convert_element_type(low, int_t)
    high <- lax_convert_element_type(high, int_t)
    slice_sizes <- a_shape
    slice_sizes[axis + 1L] <- 1L
    nd <- length(a_shape)
    dnums <- lax_GatherDimensionNumbers(
      offset_dims = red_range(q_ndim, if (keepdims) nd + q_ndim else nd + q_ndim - 1L),
      collapsed_slice_dims = if (keepdims) integer(0) else axis,
      start_index_map = axis)
    low_value <- lax_gather(a, lax_expand_dims(low, q_ndim), dnums, slice_sizes)
    high_value <- lax_gather(a, lax_expand_dims(high, q_ndim), dnums, slice_sizes)
    if (q_ndim == 1L) {
      low_weight <- lax_broadcast_in_dim(low_weight, aval_of(low_value)$shape, 0L)
      high_weight <- lax_broadcast_in_dim(high_weight, aval_of(high_value)$shape, 0L)
    }
  }
  q_dtype <- aval_of(q)$dtype
  result <- switch(method,
    linear = lax_add(lax_mul(lax_convert_element_type(low_value, q_dtype), low_weight),
                     lax_mul(lax_convert_element_type(high_value, q_dtype), high_weight)),
    lower = low_value,
    higher = high_value,
    nearest = lax_select(lax_le(high_weight, lax_full_like(high_weight, 0.5)), low_value,
                         high_value),
    midpoint = lax_mul(lax_add(low_value, high_value), lax_full_like(low_value, 0.5)),
    inverted_cdf = red_inverted_cdf(a, weights, q_orig, axis, keepdims, squash_nans, a_shape))
  if (keepdims && length(keepdim)) {
    if (q_ndim > 0L) keepdim <- c(q_shape[1L], keepdim)
    result <- lax_reshape(result, keepdim)
  }
  lax_convert_element_type(result, aval_of(a)$dtype)
}

# The method = "inverted_cdf" branch of reductions.py:_quantile.
red_inverted_cdf <- function(a, weights, q_orig, axis, keepdims, squash_nans, a_shape) {
  q_ndim <- red_ndim(q_orig)
  q_dtype <- aval_of(q_orig)$dtype
  int_t <- default_int()
  if (is.null(weights)) weights <- lax_full_like(a, 1)
  zeros <- lax_full_like(weights, 0)
  bad_weights <- lax_bitwise_or(lax_lt(weights, zeros), red_isnan(weights))
  nan_data <- red_isnan(a)
  clean_weights <- lax_select(lax_bitwise_or(bad_weights, nan_data), zeros, weights)
  cum_weights <- jnp_cumsum(clean_weights, axis = axis)
  total_weight <- lax_index_in_dim(cum_weights, a_shape[axis + 1L] - 1L, axis = axis,
                                   keepdims = keepdims)
  tw_f <- lax_expand_dims(total_weight, red_range(0L, q_ndim))
  q_f <- lax_reshape(q_orig, c(aval_of(q_orig)$shape, rep(1L, red_ndim(total_weight))))
  target_w <- lax_mul(q_f, tw_f)
  target_w_aligned <- if (keepdims) target_w else lax_expand_dims(target_w, axis + q_ndim)
  cw_f <- lax_expand_dims(cum_weights, red_range(0L, q_ndim))
  is_less <- lax_lt(cw_f, target_w_aligned)
  idx <- red_sum(lax_convert_element_type(is_less, int_t), axis = axis + q_ndim,
                 keepdims = keepdims)
  valid_counts <- if (squash_nans)
    red_sum(lax_bitwise_not(nan_data), axis = axis, dtype = q_dtype, keepdims = keepdims)
  else lax_full_like(total_weight, a_shape[axis + 1L], dtype = q_dtype)
  limit <- lax_sub(valid_counts, lax_full_like(valid_counts, 1))
  max_idx <- lax_convert_element_type(limit, int_t)
  max_idx_f <- lax_convert_element_type(lax_expand_dims(max_idx, red_range(0L, q_ndim)),
                                        aval_of(idx)$dtype)
  idx <- lax_max(lax_full_like(idx, 0), lax_min(idx, max_idx_f))
  idx_take <- if (keepdims) lax_squeeze(idx, q_ndim + axis) else idx
  if (q_ndim == 0L) {
    result <- jnp_take_along_axis(a, lax_expand_dims(idx_take, axis), axis = axis)
    result <- lax_squeeze(result, axis)
  } else {
    perm <- c(red_range(q_ndim, q_ndim + axis), red_range(0L, q_ndim),
              red_range(q_ndim + axis, red_ndim(idx_take)))
    result <- jnp_take_along_axis(a, lax_transpose(idx_take, perm), axis = axis)
    result <- lax_transpose(result, order(perm) - 1L)
  }
  if (keepdims) result <- lax_expand_dims(result, q_ndim + axis)
  all_nan_data <- lax_eq(valid_counts, lax_full_like(valid_counts, 0))
  any_bad_weight <- red_any(bad_weights, axis = axis, keepdims = keepdims)
  force_nan <- lax_bitwise_or(any_bad_weight,
                              if (squash_nans) all_nan_data
                              else red_any(nan_data, axis = axis, keepdims = keepdims))
  force_nan_f <- lax_expand_dims(force_nan, red_range(0L, q_ndim))
  jnp_where(force_nan_f, lax_full_like(result, NaN), result)
}

# ---- covariance -----------------------------------------------------------------------------------

red_atleast_2d <- function(x) {
  s <- aval_of(x)$shape
  if (length(s) >= 2L) x else lax_reshape(x, c(rep(1L, 2L - length(s)), s))
}

red_dtype_repr <- function(dtype) if (is.character(dtype)) paste0("'", dtype, "'") else format(dtype)

#' Covariance and correlation matrices
#'
#' Ports of `jnp.cov` and `jnp.corrcoef`. Rows are variables and columns are
#' observations unless `rowvar = FALSE`.
#' @param m,x Array of one or two dimensions.
#' @param y Optional additional variables, stacked below `m`.
#' @param rowvar Whether rows (rather than columns) are variables.
#' @param bias Normalize by `N` rather than `N - 1` (ignored when `ddof` is given).
#' @param ddof Delta degrees of freedom; `NULL` uses `bias`.
#' @param fweights Optional integer frequency weights of the observations.
#' @param aweights Optional observation weights.
#' @param dtype Optional inexact dtype of the computation.
#' @return An array.
#' @name jnp_cov
NULL

# Port of lax_numpy.py:cov.
#' @rdname jnp_cov
#' @export
jnp_cov <- function(m, y = NULL, rowvar = TRUE, bias = FALSE, ddof = NULL, fweights = NULL,
                    aweights = NULL, dtype = NULL) {
  if (!is.null(y)) {
    my <- promote_args_inexact(m, y)
    m <- my[[1L]]
    y <- my[[2L]]
    if (red_ndim(y) > 2L) red_value_error("y has more than 2 dimensions")
  } else {
    m <- promote_args_inexact(m)[[1L]]
  }
  if (red_ndim(m) > 2L) red_value_error("m has more than 2 dimensions")
  if (!is.null(dtype) && !is_inexact_dtype(dtype(dtype)))
    red_value_error("cov: dtype must be a subclass of float or complex; got dtype=",
                    red_dtype_repr(dtype))
  X <- red_atleast_2d(m)
  if (!rowvar && red_ndim(m) != 1L) X <- lax_transpose(X, c(1L, 0L))
  if (aval_of(X)$shape[1L] == 0L) return(lax_full(c(0L, 0L), 0, default_float()))
  if (!is.null(y)) {
    y_arr <- red_atleast_2d(y)
    if (!rowvar && aval_of(y_arr)$shape[1L] != 1L) y_arr <- lax_transpose(y_arr, c(1L, 0L))
    X <- jnp_concatenate(list(X, y_arr), axis = 0L)
  }
  xs <- aval_of(X)$shape
  if (xs[2L] == 0L)
    return(lax_full(if (xs[1L] == 1L) integer(0) else c(xs[1L], xs[1L]), NaN, aval_of(X)$dtype))
  if (is.null(ddof)) ddof <- if (bias == 0) 1L else 0L
  w <- NULL
  if (!is.null(fweights)) {
    fweights <- as_jax(fweights)
    if (red_ndim(fweights) > 1L)
      jax_error("cannot handle multidimensional fweights", class = "jax_runtime_error")
    if (aval_of(fweights)$shape[1L] != xs[2L])
      jax_error("incompatible numbers of samples and fweights", class = "jax_runtime_error")
    if (!is_int_dtype(aval_of(fweights)$dtype))
      jax_error("fweights must be integer.", class = "jax_type_error")
    w <- jnp_abs(fweights)
  }
  if (!is.null(aweights)) {
    aweights <- as_jax(aweights)
    if (red_ndim(aweights) > 1L)
      jax_error("cannot handle multidimensional aweights", class = "jax_runtime_error")
    if (aval_of(aweights)$shape[1L] != xs[2L])
      jax_error("incompatible numbers of samples and aweights", class = "jax_runtime_error")
    aweights <- jnp_abs(aweights)
    w <- if (is.null(w)) aweights else jnp_multiply(w, aweights)
  }
  if (!is.null(dtype)) {
    X <- jnp_astype(X, dtype)
    if (!is.null(w)) w <- jnp_astype(w, dtype)
  }
  aw <- jnp_average(X, axis = 1L, weights = w, returned = TRUE)
  avg <- aw[[1L]]
  w_sum <- lax_index_in_dim(aw[[2L]], 0L, keepdims = FALSE)
  f <- if (is.null(w)) xs[2L] - ddof
       else if (ddof == 0) w_sum
       else if (is.null(aweights)) jnp_subtract(w_sum, ddof)
       else jnp_subtract(w_sum, jnp_true_divide(jnp_multiply(ddof, red_sum(jnp_multiply(w, aweights))),
                                               w_sum))
  X <- jnp_subtract(X, lax_expand_dims(avg, 1L))
  X_T <- jnp_transpose(if (is.null(w)) X else jnp_multiply(X, lax_reshape(w, c(1L, xs[2L]))))
  jnp_squeeze(jnp_true_divide(jnp_dot(X, jnp_conjugate(X_T)), f))
}

red_clip_unit <- function(x) jnp_minimum(1, jnp_maximum(-1, x))

# Port of lax_numpy.py:corrcoef.
#' @rdname jnp_cov
#' @export
jnp_corrcoef <- function(x, y = NULL, rowvar = TRUE, dtype = NULL) {
  if (!is.null(dtype) && !is_inexact_dtype(dtype(dtype)))
    red_value_error("corrcoef: dtype must be a subclass of float or complex; got dtype=",
                    red_dtype_repr(dtype))
  c <- jnp_cov(x, y, rowvar, dtype = dtype)
  if (!red_ndim(c)) return(jnp_divide(c, c))
  i <- lax_iota(default_int(), aval_of(c)$shape[1L])
  d <- c[i, i]
  stddev <- lax_convert_element_type(jnp_sqrt(jnp_real(d)), aval_of(c)$dtype)
  c <- jnp_true_divide(jnp_true_divide(c, lax_expand_dims(stddev, 1L)),
                       lax_expand_dims(stddev, 0L))
  real_part <- red_clip_unit(jnp_real(c))
  if (is_complex_dtype(aval_of(c)$dtype))
    return(lax_complex(real_part, red_clip_unit(jnp_imag(c))))
  real_part
}
