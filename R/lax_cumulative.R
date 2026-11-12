# Associative scan and cumulative reductions (port of the associative_scan and
# cumulative reduction sections of jax/_src/lax/control_flow/loops.py) and the
# jax.numpy cumulative functions (port of jax/_src/numpy/reductions.py).
# The window reducers live in lax_windowed.R, which is collated later; they are only
# referenced inside function bodies.

# ---- associative_scan -------------------------------------------------------------------

# Port of loops._interleave.
interleave_along <- function(a, b, axis) {
  sa <- aval_of(a)$shape
  na <- sa[axis + 1L]
  nb <- aval_of(b)$shape[axis + 1L]
  a_pad <- rep(list(c(0L, 0L, 0L)), length(sa))
  b_pad <- a_pad
  a_pad[[axis + 1L]] <- c(0L, if (na == nb) 1L else 0L, 1L)
  b_pad[[axis + 1L]] <- c(1L, if (na == nb) 0L else 1L, 1L)
  op <- if (aval_of(a)$dtype == "bool") lax_bitwise_or else lax_add
  op(lax_pad(a, lax_full(integer(0), 0, aval_of(a)$dtype), a_pad),
     lax_pad(b, lax_full(integer(0), 0, aval_of(b)$dtype), b_pad))
}

#' Parallel prefix scan with an associative operation
#'
#' Port of `jax.lax.associative_scan`.
#' @param fn Associative function `fn(a, b)` applied elementwise over `axis`; its
#'   arguments and result have the structure of `elems`.
#' @param elems Array or pytree (list) of arrays with equal sizes along `axis`.
#' @param reverse Scan from the end.
#' @param axis 0-based axis.
#' @return Arrays with the structure of `elems`.
#' @export
lax_associative_scan <- function(fn, elems, reverse = FALSE, axis = 0L) {
  if (!is.function(fn))
    jax_error("lax.associative_scan: fn argument should be callable.", class = "jax_type_error")
  fl <- tree_flatten(elems)
  elems_flat <- lapply(fl$leaves, as_jax)
  axis <- norm_axis(axis, length(aval_of(elems_flat[[1L]])$shape))
  if (reverse) elems_flat <- lapply(elems_flat, lax_rev, dimensions = axis)
  combine <- function(a_flat, b_flat) {
    c <- fn(tree_unflatten(fl$treedef, a_flat), tree_unflatten(fl$treedef, b_flat))
    lapply(tree_leaves(c), as_jax)
  }
  num_elems <- aval_of(elems_flat[[1L]])$shape[axis + 1L]
  sizes <- vapply(elems_flat, function(e) aval_of(e)$shape[axis + 1L], 1L)
  if (any(sizes != num_elems))
    jax_error("Array inputs to associative_scan must have the same first dimension. (saw: [",
              paste(vapply(elems_flat, function(e) fmt_tuple(aval_of(e)$shape), ""),
                    collapse = ", "), "])", class = "jax_value_error")
  len <- function(e) aval_of(e)$shape[axis + 1L]
  sl <- function(e, start, limit, stride = 1L) lax_slice_in_dim(e, start, limit, stride, axis)
  scan <- function(elems) {
    n <- len(elems[[1L]])
    if (n < 2L) return(elems)
    reduced <- combine(lapply(elems, sl, 0L, n - 1L, 2L), lapply(elems, sl, 1L, n, 2L))
    odd <- scan(reduced)
    even <- if (n %% 2L == 0L) {
      combine(lapply(odd, function(e) sl(e, 0L, len(e) - 1L)), lapply(elems, sl, 2L, n, 2L))
    } else {
      combine(odd, lapply(elems, sl, 2L, n, 2L))
    }
    even <- Map(function(e, r) lax_concatenate(list(sl(e, 0L, 1L), r), axis), elems, even)
    Map(interleave_along, even, odd, MoreArgs = list(axis = axis))
  }
  scans <- scan(elems_flat)
  if (reverse) scans <- lapply(scans, lax_rev, dimensions = axis)
  tree_unflatten(fl$treedef, scans)
}

# ---- logaddexp (jax/_src/lax/other.py) ---------------------------------------------------

# Port of other._wrap_between.
wrap_between <- function(x, a) {
  dt <- aval_of(x)$dtype
  av <- lax_full(integer(0), a, dt)
  two_a <- lax_full(integer(0), 2 * a, dt)
  rem <- lax_rem(lax_add(x, av), two_a)
  rem <- lax_select(lax_lt(rem, lax_full_like(rem, 0)), lax_add(rem, two_a), rem)
  lax_sub(rem, av)
}

# Port of other._replace_inf.
replace_inf <- function(x) {
  re_x <- if (is_complex_dtype(aval_of(x)$dtype)) lax_real(x) else x
  lax_select(lax_eq(re_x, lax_full_like(re_x, Inf)), lax_full_like(x, 0), x)
}

# Port of other.logaddexp, a custom_jvp function (its rule is other._logaddexp_jvp).
lax_other_logaddexp <- custom_jvp(function(x1, x2) {
  x1 <- as_jax(x1)
  x2 <- as_jax(x2)
  dt <- aval_of(x1)$dtype
  amax <- lax_max(x1, x2)
  if (is_float_dtype(dt)) {
    delta <- lax_sub(x1, x2)
    return(lax_select(lax_ne(delta, delta), lax_add(x1, x2),
                      lax_add(amax, lax_log1p(lax_exp(lax_neg(lax_abs(delta)))))))
  }
  if (is_complex_dtype(dt)) {
    delta <- lax_sub(lax_add(x1, x2), lax_mul(amax, lax_full_like(amax, 2)))
    out <- lax_add(amax, lax_log1p(lax_exp(delta)))
    return(lax_complex(lax_real(out), wrap_between(lax_imag(out), pi)))
  }
  jax_error("logaddexp requires floating-point or complex inputs; got ", dt,
            class = "jax_value_error")
}, jvp = function(primals, tangents) {
  x1 <- primals[[1L]]
  x2 <- primals[[2L]]
  primal_out <- lax_other_logaddexp(x1, x2)
  ri_out <- replace_inf(primal_out)
  tangent_out <- lax_add(lax_mul(tangents[[1L]], lax_exp(lax_sub(replace_inf(x1), ri_out))),
                         lax_mul(tangents[[2L]], lax_exp(lax_sub(replace_inf(x2), ri_out))))
  list(primal_out, tangent_out)
})

# ---- cumulative reduction primitives ----------------------------------------------------

# Port of loops.cumred_reduce_window_impl.
cumred_reduce_window_impl <- function(window_reduce, x, axis, reverse) {
  s <- aval_of(x)$shape
  n <- s[axis + 1L]
  if (n == 0L) return(x)
  padding <- rep(list(c(0L, 0L)), length(s))
  padding[[axis + 1L]] <- if (reverse) c(0L, n - 1L) else c(n - 1L, 0L)
  strides <- rep(1L, length(s))
  window_dims <- rep(1L, length(s))
  window_dims[axis + 1L] <- n
  window_reduce(x, window_dims, strides, padding)
}

# Port of loops._cumulative_reduction_primitive (shape, dtype, batching and the
# reduce_window based lowering registered for all platforms but GPU cumsum).
def_cumred <- function(name, window_reduce) {
  p <- new_primitive(name)
  p$abstract_eval <- function(avals, params) {
    a <- avals[[1L]]
    axis <- params$axis
    if (axis < 0L) jax_error("XLA operations do not allow negative axes", class = "jax_value_error")
    if (axis >= length(a$shape))
      jax_error("axis ", axis, " is out of bounds for array of shape ", fmt_tuple(a$shape),
                class = "jax_value_error")
    if (!kind_ok(a$dtype, "number"))
      jax_error(name, " does not accept dtype ", a$dtype, ". Accepted dtypes are subtypes ",
                "of number.", class = "jax_type_error")
    new_aval(a$shape, a$dtype, a$weak)
  }
  p$batch <- function(vals, dims, params, axis) {
    d <- dims[1L]
    ax <- if (params$axis < d) params$axis else params$axis + 1L
    list(bind(p, vals, list(axis = ax, reverse = params$reverse)), d)
  }
  p$lowering <- lower_fun(function(args, params) {
    cumred_reduce_window_impl(window_reduce, args[[1L]], params$axis, params$reverse)
  })
  p
}

cumsum_p <- def_cumred("cumsum", function(...) reduce_window_sum(...))
cumlogsumexp_p <- def_cumred("cumlogsumexp", function(...) reduce_window_logaddexp(...))
cumprod_p <- def_cumred("cumprod", function(...) reduce_window_prod(...))
cummax_p <- def_cumred("cummax", function(...) reduce_window_max(...))
cummin_p <- def_cumred("cummin", function(...) reduce_window_min(...))

# Port of loops._cumsum_transpose_rule (ad.deflinear2).
deflinear(cumsum_p, function(ct, args, params) {
  if (is_zero(ct)) return(list(zero_of(args[[1L]]$aval)))
  list(lax_cumsum(ct, axis = params$axis, reverse = !params$reverse))
})

# Port of loops._cumulative_jvp_rule: differentiate the parallel prefix scan.
cumulative_jvp_rule <- function(combine_fn) {
  function(primals, tangents, params) {
    jvp(function(x) lax_associative_scan(combine_fn, x, reverse = params$reverse,
                                         axis = params$axis),
        list(primals[[1L]]), list(instantiate_zero(tangents[[1L]])))
  }
}
cumprod_p$jvp <- cumulative_jvp_rule(function(a, b) lax_mul(a, b))
cummin_p$jvp <- cumulative_jvp_rule(function(a, b) lax_min(a, b))
cummax_p$jvp <- cumulative_jvp_rule(function(a, b) lax_max(a, b))
cumlogsumexp_p$jvp <- cumulative_jvp_rule(function(a, b) lax_other_logaddexp(a, b))

cumulative_binder <- function(prim) {
  function(operand, axis = 0L, reverse = FALSE) {
    bind(prim, list(as_jax(operand)), list(axis = as.integer(axis), reverse = isTRUE(reverse)))
  }
}

#' Cumulative reductions
#'
#' Ports of `jax.lax.cumsum`, `cumprod`, `cummax`, `cummin` and `cumlogsumexp`. As in
#' JAX the axis must be nonnegative (0-based).
#' @param operand Array of a numeric dtype.
#' @param axis 0-based axis.
#' @param reverse Accumulate from the end.
#' @return An array of the shape and dtype of `operand`.
#' @name lax_cumulative
NULL
#' @rdname lax_cumulative
#' @export
lax_cumsum <- cumulative_binder(cumsum_p)
#' @rdname lax_cumulative
#' @export
lax_cumprod <- cumulative_binder(cumprod_p)
#' @rdname lax_cumulative
#' @export
lax_cummax <- cumulative_binder(cummax_p)
#' @rdname lax_cumulative
#' @export
lax_cummin <- cumulative_binder(cummin_p)
#' @rdname lax_cumulative
#' @export
lax_cumlogsumexp <- cumulative_binder(cumlogsumexp_p)

# ---- jax.numpy cumulative functions -----------------------------------------------------

# Port of numpy.reductions._promote_integer_dtype.
promote_integer_dtype <- function(dt) {
  if (dt == "bool") return(default_int())
  if (is_unsigned_dtype(dt))
    return(if (dtype_bits(dt) < dtype_bits(default_uint())) default_uint() else dt)
  if (is_signed_dtype(dt))
    return(if (dtype_bits(dt) < dtype_bits(default_int())) default_int() else dt)
  dt
}

# Port of numpy.reductions._cumulative_reduction.
cumulative_reduction <- function(name, reduction, a, axis, dtype, fill_nan = FALSE,
                                 fill_value = 0, promote_integers = FALSE) {
  a <- as_jax(a)
  if (is.null(axis) || length(aval_of(a)$shape) == 0L)
    a <- lax_reshape(a, prod_int(aval_of(a)$shape))
  if (is.null(axis)) axis <- 0L
  axis <- norm_axis(axis, length(aval_of(a)$shape))
  if (fill_nan) a <- lax_select(lax_ne(a, a), lax_full_like(a, fill_value), a)
  adt <- aval_of(a)$dtype
  if (is.null(dtype)) {
    result_type <- if (promote_integers || adt == "bool") promote_integer_dtype(adt) else adt
    computation_type <- result_type
  } else if (dtype(dtype) == "bool") {
    result_type <- "bool"
    if (adt != "bool") a <- lax_ne(a, lax_full_like(a, 0))
    computation_type <- promote_integer_dtype(result_type)
  } else {
    result_type <- dtype(dtype)
    computation_type <- result_type
  }
  a <- lax_convert_element_type(a, computation_type)
  lax_convert_element_type(reduction(a, axis), result_type)
}

#' Cumulative sums and products
#'
#' Ports of `jnp.cumsum`, `jnp.cumprod`, `jnp.nancumsum`, `jnp.nancumprod`,
#' `jnp.cumulative_sum` and `jnp.cumulative_prod`.
#' @param a,x Array.
#' @param axis 0-based axis; `NULL` flattens `a` (for `cumulative_*` it may be omitted
#'   only for one dimensional input).
#' @param dtype Optional dtype of the result.
#' @param out Unsupported (must be `NULL`).
#' @param include_initial Include the initial value (0 or 1) as first element.
#' @return An array.
#' @name jnp_cumulative
NULL

check_out_null <- function(name, out) {
  if (!is.null(out))
    jax_error("The 'out' argument to jnp.", name, " is not supported",
              class = "jax_not_implemented_error")
}

#' @rdname jnp_cumulative
#' @export
jnp_cumsum <- function(a, axis = NULL, dtype = NULL, out = NULL) {
  check_out_null("cumsum", out)
  cumulative_reduction("cumsum", lax_cumsum, a, axis, dtype)
}
#' @rdname jnp_cumulative
#' @export
jnp_cumprod <- function(a, axis = NULL, dtype = NULL, out = NULL) {
  check_out_null("cumprod", out)
  cumulative_reduction("cumprod", lax_cumprod, a, axis, dtype)
}
#' @rdname jnp_cumulative
#' @export
jnp_nancumsum <- function(a, axis = NULL, dtype = NULL, out = NULL) {
  check_out_null("nancumsum", out)
  cumulative_reduction("nancumsum", lax_cumsum, a, axis, dtype, fill_nan = TRUE, fill_value = 0)
}
#' @rdname jnp_cumulative
#' @export
jnp_nancumprod <- function(a, axis = NULL, dtype = NULL, out = NULL) {
  check_out_null("nancumprod", out)
  cumulative_reduction("nancumprod", lax_cumprod, a, axis, dtype, fill_nan = TRUE,
                       fill_value = 1)
}

cumulative_api_axis <- function(x, axis, what) {
  r <- length(aval_of(x)$shape)
  if (r == 0L)
    jax_error("The input must be non-scalar to take a cumulative ", what, ", however a ",
              "scalar value or scalar array was given.", class = "jax_value_error")
  if (is.null(axis)) {
    if (r > 1L)
      jax_error("The input array has rank ", r, ", however axis was not set to an explicit ",
                "value. The axis argument is only optional for one-dimensional arrays.",
                class = "jax_value_error")
    axis <- 0L
  }
  norm_axis(axis, r)
}

with_initial <- function(out, x, axis, value) {
  zs <- aval_of(x)$shape
  zs[axis + 1L] <- 1L
  lax_concatenate(list(lax_full(zs, value, aval_of(out)$dtype), out), axis)
}

#' @rdname jnp_cumulative
#' @export
jnp_cumulative_sum <- function(x, axis = NULL, dtype = NULL, include_initial = FALSE) {
  x <- as_jax(x)
  axis <- cumulative_api_axis(x, axis, "sum")
  out <- cumulative_reduction("_cumsum_with_promotion", lax_cumsum, x, axis, dtype,
                              promote_integers = TRUE)
  if (include_initial) out <- with_initial(out, x, axis, 0)
  out
}
#' @rdname jnp_cumulative
#' @export
jnp_cumulative_prod <- function(x, axis = NULL, dtype = NULL, include_initial = FALSE) {
  x <- as_jax(x)
  axis <- cumulative_api_axis(x, axis, "product")
  out <- cumulative_reduction("cumulative_prod", lax_cumprod, x, axis, dtype)
  if (include_initial) out <- with_initial(out, x, axis, 1)
  out
}

# Helpers for base::cummax / base::cummin on jax values (R/methods.R): cumulative
# maximum and minimum of the flattened array.
jnp_cumulative_max <- function(x) {
  x <- as_jax(x)
  lax_cummax(lax_reshape(x, prod_int(aval_of(x)$shape)), axis = 0L)
}
jnp_cumulative_min <- function(x) {
  x <- as_jax(x)
  lax_cummin(lax_reshape(x, prod_int(aval_of(x)$shape)), axis = 0L)
}
