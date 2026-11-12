# jax.nn: activations, normalizations and attention (port of jax/_src/nn/functions.py).
# Functions that upstream decorates with `api.jit` call a private jitted function with
# fixed positional arguments (static arguments by position), so values match the wheel.

# ---- private helpers ----------------------------------------------------------------

nnp_dtype <- function(x) aval_of(x)$dtype
nnp_shape <- function(x) aval_of(x)$shape

# Python style tuple and list text for error messages.
nnp_tuple <- function(x) {
  if (length(x) == 1L) paste0("(", x, ",)") else paste0("(", paste(x, collapse = ", "), ")")
}
nnp_list <- function(x) paste0("[", paste(x, collapse = ", "), "]")

# The masked reductions of jax.numpy, with the argument order used below.
nnp_max <- function(x, axis, keepdims = FALSE, where = NULL, initial = NULL)
  jnp_max(x, axis, keepdims = keepdims, initial = initial, where = where)
nnp_sum <- function(x, axis, keepdims = FALSE, where = NULL, dtype = NULL)
  jnp_sum(x, axis, dtype = dtype, keepdims = keepdims, where = where)
nnp_count <- function(x, axis, keepdims, where, dtype) red_count(x, axis, keepdims, where, dtype)
nnp_mean <- function(x, axis, keepdims = FALSE, where = NULL)
  jnp_mean(x, axis, keepdims = keepdims, where = where)

# Port of jax/_src/lax/other.py:_wrap_between.
nnp_wrap_between <- function(x, a) {
  dt <- nnp_dtype(x)
  av <- lax_full(integer(0), a, dt)
  two_a <- lax_full(integer(0), 2 * a, dt)
  rem <- lax_rem(lax_add(x, av), two_a)
  rem <- lax_select(lax_lt(rem, lax_full_like(rem, 0)), lax_add(rem, two_a), rem)
  lax_sub(rem, av)
}

# Port of jax/_src/lax/other.py:_replace_inf.
nnp_replace_inf <- function(x) {
  re_x <- if (is_complex_dtype(nnp_dtype(x))) lax_real(x) else x
  lax_select(lax_eq(re_x, lax_full_like(re_x, Inf)), lax_full_like(x, 0), x)
}

# Port of jax/_src/lax/other.py:logaddexp with its custom JVP _logaddexp_jvp. Same rule as
# lax_other_logaddexp in R/lax_cumulative.R (kept private here so jax.nn does not depend
# on that file).
nnp_lax_logaddexp <- custom_jvp(function(x1, x2) {
  x1 <- as_jax(x1)
  x2 <- as_jax(x2)
  dt <- nnp_dtype(x1)
  amax <- lax_max(x1, x2)
  if (is_float_dtype(dt)) {
    delta <- lax_sub(x1, x2)
    return(lax_select(lax_ne(delta, delta), lax_add(x1, x2),
                      lax_add(amax, lax_log1p(lax_exp(lax_neg(lax_abs(delta)))))))
  }
  if (is_complex_dtype(dt)) {
    delta <- lax_sub(lax_add(x1, x2), lax_mul(amax, lax_full_like(amax, 2)))
    out <- lax_add(amax, lax_log1p(lax_exp(delta)))
    return(lax_complex(lax_real(out), nnp_wrap_between(lax_imag(out), pi)))
  }
  jax_error("logaddexp requires floating-point or complex inputs; got ", dt,
            class = "jax_value_error")
}, jvp = function(primals, tangents) {
  primal_out <- nnp_lax_logaddexp(primals[[1L]], primals[[2L]])
  ri_out <- nnp_replace_inf(primal_out)
  tangent_out <- lax_add(
    lax_mul(tangents[[1L]], lax_exp(lax_sub(nnp_replace_inf(primals[[1L]]), ri_out))),
    lax_mul(tangents[[2L]], lax_exp(lax_sub(nnp_replace_inf(primals[[2L]]), ri_out))))
  list(primal_out, tangent_out)
})

# Port of jax/_src/numpy/ufuncs.py:logaddexp.
nnp_logaddexp <- function(x1, x2) {
  a <- promote_args_inexact(x1, x2)
  nnp_lax_logaddexp(a[[1L]], a[[2L]])
}

# Port of jax/_src/ops/special.py:logsumexp (jax.nn.logsumexp). Same rule as
# scipy_special_logsumexp in R/scipy_special.R.
nnp_logsumexp <- function(a, axis = NULL, b = NULL, keepdims = FALSE, return_sign = FALSE,
                          where = NULL) {
  if (!is.null(where)) a <- jnp_where(where, a, 0L)
  if (!is.null(b)) {
    ab <- promote_args_inexact(a, b)
    b_arr <- ab[[2L]]
    a_arr <- jnp_where(jnp_not_equal(b_arr, 0L), ab[[1L]], -Inf)
  } else {
    a_arr <- promote_args_inexact(a)[[1L]]
  }
  dt <- nnp_dtype(a_arr)
  dims <- reduction_axes(a_arr, axis)
  amax <- nnp_max(jnp_real(a_arr), dims, keepdims, where, initial = -Inf)
  amax <- lax_stop_gradient(lax_select(jnp_isfinite(amax), amax, lax_full_like(amax, 0)))
  amax_with_dims <- if (keepdims) amax else lax_expand_dims(amax, dims)
  exp_a <- lax_exp(lax_sub(a_arr, jnp_astype(amax_with_dims, dt)))
  if (!is.null(b)) exp_a <- lax_mul(exp_a, b_arr)
  sumexp <- nnp_sum(exp_a, dims, keepdims, where)
  sign <- lax_sign(sumexp)
  if (return_sign || !is_complex_dtype(dt)) sumexp <- jnp_abs(sumexp)
  out <- lax_add(lax_log(sumexp), jnp_astype(amax, nnp_dtype(sumexp)))
  if (return_sign) return(list(out, sign))
  if (!is.null(b) && !is_complex_dtype(nnp_dtype(out)))
    out <- jnp_where(jnp_less(sign, 0L), as_jax(NaN, nnp_dtype(out)), out)
  out
}

# ---- activations --------------------------------------------------------------------

#' Activation functions
#'
#' Ports of the `jax.nn` activation functions. All are elementwise except `nn_glu`,
#' which splits `x` in two halves along `axis` (0-based). `nn_relu`, `nn_relu6` and
#' `nn_log1mexp` have custom JVP rules as in JAX: the derivative of `nn_relu` at 0 is 0,
#' and the derivative of `nn_relu6` is 0 at 0 and at 6. `nn_swish` is an alias of
#' `nn_silu`, `nn_hard_swish` of `nn_hard_silu` and `nn_tanh` of [jnp_tanh()].
#'
#' Arguments are not promoted to floating point unless JAX does so: `nn_relu` of an
#' `int32` array is `int32`, while `nn_relu6` and `nn_gelu` return floating point
#' arrays.
#' @param x Input array.
#' @param b Smoothness parameter of `nn_squareplus`.
#' @param alpha Scalar or array of alpha values (`nn_elu`, `nn_celu`).
#' @param negative_slope Slope for negative inputs of `nn_leaky_relu`.
#' @param approximate Use the tanh approximation of `nn_gelu` (default) or the exact
#'   formulation with `erfc`.
#' @param axis 0-based axis along which `nn_glu` splits its input.
#' @return An array.
#' @examples
#' \dontrun{
#' x <- jnp_array(c(-2, -1, -0.5, 0, 0.5, 1, 2))
#' nn_relu(x)
#' nn_gelu(x, approximate = FALSE)
#' grad(function(v) nn_relu(v))(0)
#' }
#' @name nn_activations
NULL

nnp_identity <- jit(function(x) as_jax(x))

#' @rdname nn_activations
#' @export
nn_identity <- function(x) nnp_identity(x)

# Port of functions.py:relu (custom_jvp of a jitted function; rule from relu.defjvps).
nnp_relu <- custom_jvp(jit(function(x) jnp_maximum(x, 0L)), jvp = function(primals, tangents) {
  x <- primals[[1L]]
  g <- tangents[[1L]]
  list(nnp_relu(x), lax_select(jnp_greater(x, 0L), g, lax_full_like(g, 0L)))
})

#' @rdname nn_activations
#' @export
nn_relu <- function(x) nnp_relu(x)

nnp_squareplus <- jit(function(x, b) {
  x <- as_jax(x)
  y <- x + jnp_sqrt(jnp_square(x) + b)
  y / 2L
})

#' @rdname nn_activations
#' @export
nn_squareplus <- function(x, b = 4L) nnp_squareplus(x, b)

nnp_softplus <- jit(function(x) nnp_logaddexp(x, 0L))

#' @rdname nn_activations
#' @export
nn_softplus <- function(x) nnp_softplus(x)

nnp_sparse_plus <- jit(function(x) {
  x <- as_jax(x)
  jnp_where(x <= -1, 0, jnp_where(x >= 1, x, (x + 1)^2L / 4L))
})

#' @rdname nn_activations
#' @export
nn_sparse_plus <- function(x) nnp_sparse_plus(x)

nnp_soft_sign <- jit(function(x) {
  x <- as_jax(x)
  x / (jnp_abs(x) + 1L)
})

#' @rdname nn_activations
#' @export
nn_soft_sign <- function(x) nnp_soft_sign(x)

nnp_sigmoid <- jit(function(x) lax_logistic(x))

#' @rdname nn_activations
#' @export
nn_sigmoid <- function(x) nnp_sigmoid(x)

nnp_sparse_sigmoid <- jit(function(x) 0.5 * jnp_clip(as_jax(x) + 1, 0, 2))

#' @rdname nn_activations
#' @export
nn_sparse_sigmoid <- function(x) nnp_sparse_sigmoid(x)

nnp_silu <- jit(function(x) {
  x <- as_jax(x)
  x * nn_sigmoid(x)
})

#' @rdname nn_activations
#' @export
nn_silu <- function(x) nnp_silu(x)

#' @rdname nn_activations
#' @export
nn_swish <- nn_silu

nnp_mish <- jit(function(x) {
  x <- as_jax(x)
  x * jnp_tanh(nn_softplus(x))
})

#' @rdname nn_activations
#' @export
nn_mish <- function(x) nnp_mish(x)

nnp_log_sigmoid <- jit(function(x) -nn_softplus(-as_jax(x)))

#' @rdname nn_activations
#' @export
nn_log_sigmoid <- function(x) nnp_log_sigmoid(x)

nnp_elu <- jit(function(x, alpha) {
  x <- as_jax(x)
  jnp_where(x > 0L, x, alpha * jnp_expm1(jnp_where(x > 0L, 0, x)))
})

#' @rdname nn_activations
#' @export
nn_elu <- function(x, alpha = 1.0) nnp_elu(x, alpha)

nnp_leaky_relu <- jit(function(x, negative_slope) {
  x <- as_jax(x)
  jnp_where(x >= 0L, x, negative_slope * x)
})

#' @rdname nn_activations
#' @export
nn_leaky_relu <- function(x, negative_slope = 1e-2) nnp_leaky_relu(x, negative_slope)

nnp_hard_tanh <- jit(function(x) {
  x <- as_jax(x)
  jnp_where(x > 1L, 1L, jnp_where(x < -1L, -1L, x))
})

#' @rdname nn_activations
#' @export
nn_hard_tanh <- function(x) nnp_hard_tanh(x)

nnp_celu <- jit(function(x, alpha) {
  jnp_maximum(x, 0) + alpha * jnp_expm1(jnp_minimum(x, 0) / alpha)
})

#' @rdname nn_activations
#' @export
nn_celu <- function(x, alpha = 1.0) nnp_celu(x, alpha)

# alpha = 1.6732632423543772848170429916717 and scale = 1.0507009873554804934193349852946
# as Python parses them.
nnp_selu <- jit(function(x) {
  0x1.0cfabd6a91132p+0 * nn_elu(x, 0x1.ac5afad782cf1p+0)
})

#' @rdname nn_activations
#' @export
nn_selu <- function(x) nnp_selu(x)

# Not jitted upstream ("this jit was found to change numerics in a test").
#' @rdname nn_activations
#' @export
nn_gelu <- function(x, approximate = TRUE) {
  x <- promote_args_inexact(x)[[1L]]
  dt <- nnp_dtype(x)
  if (approximate) {
    sqrt_2_over_pi <- as_jax(sqrt(2 / pi), dt)
    cdf <- 0.5 * (1.0 + jnp_tanh(sqrt_2_over_pi * (x + 0.044715 * (x^3L))))
    return(x * cdf)
  }
  sqrt_half <- as_jax(sqrt(0.5), dt)
  jnp_asarray(0.5 * x * lax_erfc(-x * sqrt_half), dtype = dt)
}

nnp_glu <- jit(function(x, axis) {
  x <- as_jax(x)
  s <- nnp_shape(x)
  ax <- norm_axis(axis, length(s))
  size <- s[ax + 1L]
  if (size %% 2L != 0L) jax_error("axis size must be divisible by 2")
  half <- size %/% 2L
  x1 <- lax_slice_in_dim(x, 0L, half, axis = ax)
  x2 <- lax_slice_in_dim(x, half, size, axis = ax)
  x1 * nn_sigmoid(x2)
}, static_argnums = 1L)

#' @rdname nn_activations
#' @export
nn_glu <- function(x, axis = -1L) nnp_glu(x, as.integer(axis))

nnp_relu6 <- custom_jvp(jit(function(x) jnp_minimum(jnp_maximum(x, 0L), 6)),
                        jvp = function(primals, tangents) {
  x <- primals[[1L]]
  g <- tangents[[1L]]
  keep <- jnp_logical_and(jnp_greater(x, 0L), jnp_less(x, 6L))
  list(nnp_relu6(x), lax_select(keep, g, lax_full_like(g, 0L)))
})

#' @rdname nn_activations
#' @export
nn_relu6 <- function(x) nnp_relu6(x)

nnp_hard_sigmoid <- jit(function(x) nn_relu6(as_jax(x) + 3) / 6)

#' @rdname nn_activations
#' @export
nn_hard_sigmoid <- function(x) nnp_hard_sigmoid(x)

nnp_hard_silu <- jit(function(x) {
  x <- as_jax(x)
  x * nn_hard_sigmoid(x)
})

#' @rdname nn_activations
#' @export
nn_hard_silu <- function(x) nnp_hard_silu(x)

#' @rdname nn_activations
#' @export
nn_hard_swish <- nn_hard_silu

# Port of functions.py:log1mexp (custom_jvp of a jitted function; rule g / expm1(x)).
nnp_log1mexp <- custom_jvp(jit(function(x) {
  x <- as_jax(x)
  c <- jnp_log(2.0)
  jnp_where(x < c, jnp_log(-jnp_expm1(-x)), jnp_log1p(-jnp_exp(-x)))
}), jvp = function(primals, tangents) {
  x <- primals[[1L]]
  list(nnp_log1mexp(x), tangents[[1L]] / jnp_expm1(x))
})

#' @rdname nn_activations
#' @export
nn_log1mexp <- function(x) nnp_log1mexp(x)

#' @rdname nn_activations
#' @export
nn_tanh <- function(x) jnp_tanh(x)

# ---- softmax and normalization ----------------------------------------------------------

#' Softmax, log-sum-exp and standardization
#'
#' Ports of `jax.nn.softmax`, `jax.nn.log_softmax`, `jax.nn.logsumexp`,
#' `jax.nn.logmeanexp` and `jax.nn.standardize`. Axes are 0-based; `axis = NULL`
#' reduces over all axes. `where` is an optional boolean mask of the elements to include:
#' masked out elements are 0 in `nn_softmax` and `-Inf` in `nn_log_softmax`.
#'
#' As in JAX 0.12.0.1 (`jax_softmax_custom_jvp = False`), `nn_softmax` stops the gradient
#' through the subtracted maximum; when the option `jax_softmax_custom_jvp` is `TRUE` it
#' uses the custom JVP rule `y * (x_dot - sum(y * x_dot))` instead.
#' @param x,a Input array.
#' @param axis Axis or axes (0-based) to reduce over, or `NULL` for all axes.
#' @param where Optional boolean mask of the elements to include.
#' @param b Optional scaling factors for the exponentials of `nn_logsumexp`, broadcastable
#'   to the shape of `a`.
#' @param keepdims Keep reduced axes as size one dimensions.
#' @param return_sign Return `list(result, sign)` where `result` holds the logarithm of
#'   the absolute values of the sums.
#' @param mean,variance Optional mean and variance used by `nn_standardize` (computed
#'   from `x` when `NULL`).
#' @param epsilon Correction added to the variance.
#' @param algorithm Variance algorithm of `nn_standardize`: `"fast"` (`mean(x^2) -
#'   mean(x)^2`, clipped at 0) or `"stable"` (`mean((x - mean(x))^2)`).
#' @return An array (a list for `nn_logsumexp` with `return_sign = TRUE`).
#' @examples
#' \dontrun{
#' x <- jnp_array(c(1, 2, 3))
#' nn_softmax(x)
#' nn_softmax(x, where = c(TRUE, FALSE, TRUE))
#' nn_logsumexp(x)
#' }
#' @name nn_softmax
NULL

#' @rdname nn_softmax
#' @export
nn_logsumexp <- function(a, axis = NULL, b = NULL, keepdims = FALSE, return_sign = FALSE,
                         where = NULL) {
  nnp_logsumexp(a, axis, b, keepdims, return_sign, where)
}

nnp_logmeanexp <- jit(function(x, axis, where, keepdims) {
  lse <- nnp_logsumexp(x, axis = axis, where = where, keepdims = keepdims)
  count <- nnp_count(as_jax(x), axis, keepdims, where, nnp_dtype(lse))
  lse - jnp_log(count)
}, static_argnums = c(1L, 3L))

#' @rdname nn_softmax
#' @export
nn_logmeanexp <- function(x, axis = NULL, where = NULL, keepdims = FALSE) {
  nnp_logmeanexp(x, axis, where, keepdims)
}

nnp_log_softmax <- jit(function(x, axis, where) {
  x <- as_jax(x)
  x_max <- nnp_max(x, axis, keepdims = TRUE, where = where, initial = -Inf)
  x_safe <- if (is.null(where)) x else jnp_where(where, x, -Inf)
  shifted <- x_safe - lax_stop_gradient(x_max)
  shifted_logsumexp <- jnp_log(nnp_sum(jnp_exp(shifted), axis, keepdims = TRUE, where = where))
  result <- shifted - shifted_logsumexp
  if (!is.null(where)) return(jnp_where(where, result, -Inf))
  result
}, static_argnums = 1L)

#' @rdname nn_softmax
#' @export
nn_log_softmax <- function(x, axis = -1L, where = NULL) nnp_log_softmax(x, axis, where)

# Port of functions.py:_softmax (custom_jvp with nondiff axis).
nnp_softmax_fun <- function(x, axis, where, initial) {
  x_max <- nnp_max(x, axis, keepdims = TRUE, where = where, initial = initial)
  x_safe <- if (is.null(where)) as_jax(x) else jnp_where(where, x, initial)
  unnormalized <- jnp_exp(x_safe - x_max)
  result <- unnormalized / nnp_sum(unnormalized, axis, keepdims = TRUE, where = where)
  if (!is.null(where)) result <- jnp_where(where, result, 0L)
  result
}

nnp_softmax <- custom_jvp(nnp_softmax_fun, jvp = function(axis, primals, tangents) {
  x <- primals[[1L]]
  where <- primals[[2L]]
  x_dot <- tangents[[1L]]
  y <- nnp_softmax(x, axis, where, primals[[3L]])
  list(y, y * (x_dot - nnp_sum(y * x_dot, axis, keepdims = TRUE, where = where)))
}, nondiff_argnums = 1L)

# Port of functions.py:_softmax_deprecated.
nnp_softmax_deprecated <- function(x, axis, where, initial) {
  x_max <- nnp_max(x, axis, keepdims = TRUE, where = where, initial = initial)
  x_safe <- if (is.null(where)) as_jax(x) else jnp_where(where, x, initial)
  unnormalized <- jnp_exp(x_safe - lax_stop_gradient(x_max))
  result <- unnormalized / nnp_sum(unnormalized, axis, keepdims = TRUE, where = where)
  if (!is.null(where)) result <- jnp_where(where, result, 0L)
  result
}

# Not jitted upstream ("this jit was found to change numerics in a test").
#' @rdname nn_softmax
#' @export
nn_softmax <- function(x, axis = -1L, where = NULL) {
  if (isTRUE(.jx$config$jax_softmax_custom_jvp)) nnp_softmax(x, axis, where, -Inf)
  else nnp_softmax_deprecated(x, axis, where, -Inf)
}

nnp_standardize <- jit(function(x, axis, mean, variance, epsilon, where, algorithm) {
  x <- as_jax(x)
  if (is.null(mean)) mean <- nnp_mean(x, axis, keepdims = TRUE, where = where)
  if (is.null(variance)) {
    if (identical(algorithm, "stable")) {
      variance <- nnp_mean(jnp_square(jnp_subtract(x, mean)), axis, keepdims = TRUE,
                           where = where)
    } else if (identical(algorithm, "fast")) {
      variance <- nnp_mean(jnp_square(x), axis, keepdims = TRUE, where = where) -
        jnp_square(mean)
      variance <- jnp_clip(variance, 0L)
    } else {
      jax_error("Unknown algorithm '", algorithm, "'. Expected 'fast' or 'stable'.",
                class = "jax_value_error")
    }
  }
  jnp_subtract(x, mean) * lax_rsqrt(variance + epsilon)
}, static_argnums = c(1L, 6L))

#' @rdname nn_softmax
#' @export
nn_standardize <- function(x, axis = -1L, mean = NULL, variance = NULL, epsilon = 1e-5,
                           where = NULL, algorithm = "fast") {
  nnp_standardize(x, axis, mean, variance, epsilon, where, algorithm)
}

# ---- one_hot ------------------------------------------------------------------------------

nnp_one_hot <- jit(function(x, num_classes, dtype, axis) {
  nd <- length(nnp_shape(x))
  out_axis <- norm_axis(axis, nd + 1L)
  lhs <- lax_expand_dims(x, out_axis)
  rhs_shape <- append(rep(1L, nd), num_classes, after = out_axis)
  rhs <- lax_broadcasted_iota(nnp_dtype(x), rhs_shape, out_axis)
  jnp_astype(jnp_equal(lhs, rhs), dtype)
}, static_argnums = 1:3)

#' One-hot encoding
#'
#' Port of `jax.nn.one_hot`. Each index in `x` (0-based) becomes a vector of zeros of
#' length `num_classes` with a one at the index; indices outside `[0, num_classes)` give
#' all zeros.
#' @param x Integer array of indices (other dtypes give a deprecation warning, as in JAX).
#' @param num_classes Number of classes (a concrete integer).
#' @param dtype Output dtype (default float).
#' @param axis 0-based position of the new axis in the result, or the name of a mapped
#'   axis (`vmap(axis_name = )`), in which case `num_classes` must equal its size.
#' @param out_sharding Must be `NULL` (sharding is not supported).
#' @return An array with one more dimension than `x`.
#' @examples
#' \dontrun{
#' nn_one_hot(jnp_array(c(0L, 1L, 2L)), 3L)
#' }
#' @export
nn_one_hot <- function(x, num_classes, dtype = NULL, axis = -1L, out_sharding = NULL) {
  what <- "The error arose in jax.nn.one_hot argument `num_classes`."
  if (is_jax_value(num_classes)) num_classes <- concrete_or_error(num_classes, what)
  num_classes <- as.integer(num_classes)
  if (!is.null(out_sharding)) jax_error("nn_one_hot: out_sharding is not supported")
  x <- jnp_asarray(x)
  if (!is_int_dtype(nnp_dtype(x)))
    warning("jax.nn.one_hot input should be integer-typed; got dtype=", nnp_dtype(x),
            call. = FALSE)
  dtype <- if (is.null(dtype)) default_float() else dtype(dtype)
  if (is.character(axis)) {
    axis_size <- lax_axis_size(axis)
    if (num_classes != axis_size)
      jax_error("Expected num_classes to match the size of axis ", axis, ", but ",
                num_classes, " != ", axis_size, class = "jax_value_error")
    return(jnp_asarray(jnp_equal(x, lax_axis_index(axis)), dtype = dtype))
  }
  nnp_one_hot(x, num_classes, dtype, as.integer(axis))
}

# ---- dot_product_attention --------------------------------------------------------------

# Port of functions.py:_get_large_negative: -0.7 times the largest finite value of the
# dtype, capped at the float32 range and computed in the smaller of the two dtypes.
nnp_large_negative <- function(dt) {
  f32max <- finfo("float32")$max
  dmax <- finfo(dt)$max
  small <- if (dmax < f32max) dt else "float32"
  lax_convert_element_type(as_jax(-0.7 * min(dmax, f32max), small), dt)
}

# Ports of functions.py:_get_causal_mask, _get_window_mask, _get_padding_mask_logits and
# _get_padding_mask_encoded (jnp.tril and jnp.arange written with iota).
nnp_causal_mask <- function(t, s) {
  rows <- lax_broadcasted_iota("int32", c(t, s), 0L)
  cols <- lax_broadcasted_iota("int32", c(t, s), 1L)
  lax_reshape(lax_ge(rows, cols), c(1L, 1L, t, s))
}

nnp_window_mask <- function(t, s, local_window_size) {
  query_pos <- lax_broadcasted_iota("int32", c(t, s), 0L)
  key_pos <- lax_broadcasted_iota("int32", c(t, s), 1L)
  left_mask <- query_pos <= key_pos + as.integer(local_window_size[1L])
  right_mask <- query_pos >= key_pos - as.integer(local_window_size[2L])
  lax_reshape(jnp_logical_and(right_mask, left_mask), c(1L, 1L, t, s))
}

nnp_padding_mask_logits <- function(t, s, q_seqlen, kv_seqlen) {
  q_mask <- TRUE
  kv_mask <- TRUE
  if (!is.null(q_seqlen)) {
    q_indices <- lax_reshape(jnp_arange(0L, t), c(1L, t, 1L))
    q_mask <- q_indices < lax_expand_dims(q_seqlen, c(1L, 2L))
  }
  if (!is.null(kv_seqlen)) {
    kv_indices <- lax_reshape(jnp_arange(0L, s), c(1L, 1L, s))
    kv_mask <- kv_indices < lax_expand_dims(kv_seqlen, c(1L, 2L))
  }
  lax_expand_dims(jnp_logical_and(q_mask, kv_mask), 1L)
}

nnp_padding_mask_encoded <- function(t, q_seqlen) {
  q_indices <- lax_reshape(jnp_arange(0L, t), c(1L, t))
  lax_expand_dims(q_indices < lax_expand_dims(q_seqlen, 1L), c(2L, 3L))
}

# Port of functions.py:_apply_masks.
nnp_apply_masks <- function(logits, mask, is_causal, q_seqlen, kv_seqlen, local_window_size) {
  if (is.null(mask) && !is_causal && is.null(q_seqlen) && is.null(kv_seqlen) &&
      is.null(local_window_size)) return(logits)
  combined_mask <- jnp_ones_like(logits, dtype = "bool")
  if (!is.null(mask)) {
    if (nnp_dtype(mask) != "bool") jax_error("mask must be a boolean array")
    combined_mask <- jnp_logical_and(combined_mask, mask)
  }
  s <- nnp_shape(logits)
  t_len <- s[3L]
  s_len <- s[4L]
  if (is_causal) combined_mask <- jnp_logical_and(combined_mask, nnp_causal_mask(t_len, s_len))
  if (!is.null(local_window_size))
    combined_mask <- jnp_logical_and(combined_mask,
                                     nnp_window_mask(t_len, s_len, local_window_size))
  if (!is.null(q_seqlen) || !is.null(kv_seqlen))
    combined_mask <- jnp_logical_and(combined_mask,
                                     nnp_padding_mask_logits(t_len, s_len, q_seqlen, kv_seqlen))
  jnp_where(combined_mask, logits, nnp_large_negative(nnp_dtype(logits)))
}

# Port of functions.py:_dot_product_attention_core. The einsums are written as
# dot_general with the operand order and dimension numbers jnp.einsum chooses
# ("BTNH,BSNH->BNTS" and "BNTS,BSNH->BTNH"). The bfloat16/float16 dot algorithm presets
# are not applied (jaxR has no DotAlgorithmPreset); upstream falls back to the default
# precision when a preset is unsupported.
nnp_attention_core <- function(query, key, value, bias, mask, is_causal, scale, q_seqlen,
                               kv_seqlen, local_window_size, return_residual) {
  logits_dtype <- promote_types_dtype(nnp_dtype(query), "float32")
  logits <- lax_dot_general(query, key, list(list(3L, 3L), list(c(0L, 2L), c(0L, 2L))),
                            preferred_element_type = logits_dtype)
  logits <- logits * jnp_asarray(scale, dtype = nnp_dtype(logits))
  if (!is.null(bias)) logits <- jnp_astype(logits + bias, nnp_dtype(logits))
  padded_logits <- nnp_apply_masks(logits, mask, is_causal, q_seqlen, kv_seqlen,
                                   local_window_size)
  padded_logits <- jnp_astype(padded_logits, "float32")
  probs <- jnp_astype(nn_softmax(padded_logits, axis = -1L), nnp_dtype(key))
  encoded <- lax_dot_general(value, probs, list(list(1L, 3L), list(c(0L, 2L), c(0L, 1L))))
  encoded <- lax_transpose(encoded, c(0L, 3L, 1L, 2L))
  if (!is.null(q_seqlen)) {
    m <- nnp_padding_mask_encoded(nnp_shape(encoded)[2L], q_seqlen)
    encoded <- encoded * jnp_astype(m, nnp_dtype(encoded))
  }
  if (return_residual) {
    lse <- jnp_astype(nn_logsumexp(padded_logits, axis = -1L), nnp_dtype(key))
    lse <- lax_transpose(lse, c(0L, 2L, 1L))
    return(list(encoded, lax_stop_gradient(lse)))
  }
  encoded
}

# Port of functions.py:_dot_product_attention_xla.
nnp_attention_xla <- function(query, key, value, bias, mask, is_causal, scale, q_seqlen,
                              kv_seqlen, local_window_size, return_residual) {
  qs <- nnp_shape(query)
  ks <- nnp_shape(key)
  n_kv <- ks[3L]
  g <- qs[3L] %/% n_kv
  query <- jnp_reshape(query, c(qs[1L], qs[2L], n_kv, g, qs[4L]))
  to_grouped <- function(t) {
    if (is.null(t)) return(NULL)
    ts <- nnp_shape(t)
    if (ts[2L] == 1L) return(jnp_broadcast_to(lax_expand_dims(t, 2L), c(ts[1:2], g, ts[3:4])))
    if (ts[2L] != qs[3L]) jax_error("bias and mask must have 1 or N heads")
    jnp_reshape(t, c(ts[1L], n_kv, g, ts[3:4]))
  }
  core <- function(q, b, m) {
    nnp_attention_core(q, key, value, b, m, is_causal, scale, q_seqlen, kv_seqlen,
                       local_window_size, return_residual)
  }
  out <- vmap(core, in_axes = list(3L, 2L, 2L), out_axes = 3L)(query, to_grouped(bias),
                                                               to_grouped(mask))
  if (return_residual) {
    return(list(jnp_reshape(out[[1L]], qs), jnp_reshape(out[[2L]], qs[1:3])))
  }
  jnp_reshape(out, qs)
}

#' Scaled dot product attention
#'
#' Port of `jax.nn.dot_product_attention` (the `"xla"` implementation): for each head,
#' `softmax(Q K^T * scale + bias) V`. Shapes use `B` batch, `T` query length, `S`
#' key/value length, `N` query heads, `K` key/value heads (`N` must be a multiple of `K`:
#' grouped and multi-query attention) and `H` head dimension.
#' @param query Query array of shape `(B, T, N, H)` or `(T, N, H)`.
#' @param key Key array of shape `(B, S, K, H)` or `(S, K, H)`.
#' @param value Value array with the shape of `key`.
#' @param bias Optional bias added to the logits, broadcastable to `(B, N, T, S)`.
#' @param mask Optional boolean mask broadcastable to `(B, N, T, S)`; `TRUE` marks the
#'   elements that take part in attention.
#' @param scale Scale of the logits (default `1 / sqrt(H)`).
#' @param is_causal Apply a causal mask.
#' @param query_seq_lengths,key_value_seq_lengths Optional `int32` arrays of shape `(B)`
#'   with the valid lengths of the query and key/value sequences.
#' @param local_window_size Optional `c(left, right)` window (a single value is used for
#'   both sides): each query attends to keys at offsets `-left` to `right`.
#' @param implementation `NULL` or `"xla"` (`"cudnn"` needs a GPU and is not available).
#' @param return_residual Also return the log-sum-exp of the logits, of shape `(B, T, N)`
#'   or `(T, N)`.
#' @return An array with the shape of `query`, or `list(output, residual)`.
#' @examples
#' \dontrun{
#' q <- random_normal(random_key(0), c(2, 3, 4, 5))
#' k <- random_normal(random_key(1), c(2, 6, 2, 5))
#' nn_dot_product_attention(q, k, k, is_causal = TRUE)
#' }
#' @export
nn_dot_product_attention <- function(query, key, value, bias = NULL, mask = NULL,
                                     scale = NULL, is_causal = FALSE,
                                     query_seq_lengths = NULL, key_value_seq_lengths = NULL,
                                     local_window_size = NULL, implementation = NULL,
                                     return_residual = FALSE) {
  output_shape <- nnp_shape(jnp_asarray(query))
  residual_shape <- utils::head(output_shape, -1L)
  ensure_4d <- function(t) {
    t <- jnp_asarray(t)
    n <- 4L - length(nnp_shape(t))
    if (n > 0L) lax_expand_dims(t, seq_len(n) - 1L) else t
  }
  query_arr <- ensure_4d(query)
  key_arr <- ensure_4d(key)
  value_arr <- ensure_4d(value)
  if (!is.null(bias)) bias <- ensure_4d(bias)
  if (!is.null(mask)) mask <- ensure_4d(mask)
  if (!is.null(query_seq_lengths)) query_seq_lengths <- jnp_asarray(query_seq_lengths)
  if (!is.null(key_value_seq_lengths)) key_value_seq_lengths <- jnp_asarray(key_value_seq_lengths)
  if (length(local_window_size) == 1L) local_window_size <- rep(local_window_size, 2L)

  check <- function(t, shape, dtype, name) {
    if (is.null(t)) return(invisible())
    ts <- nnp_shape(t)
    if (length(ts) != length(shape))
      jax_error(name, " ndim should be ", length(shape), ", but got ", length(ts),
                class = "jax_value_error")
    if (!is.null(dtype) && nnp_dtype(t) != dtype)
      jax_error(name, " dtype should be ", dtype, ", but got ", nnp_dtype(t),
                class = "jax_value_error")
    if (any(shape != -1L & ts != shape))
      jax_error(name, " shape should be ", nnp_list(shape), ": but got ", nnp_tuple(ts),
                class = "jax_value_error")
  }
  ks <- nnp_shape(key_arr)
  b <- ks[1L]
  h <- ks[4L]
  check(value_arr, ks, nnp_dtype(key_arr), "value")
  check(query_arr, c(b, -1L, -1L, h), nnp_dtype(key_arr), "query")
  check(mask, rep(-1L, 4L), "bool", "mask")
  check(bias, rep(-1L, 4L), NULL, "bias")
  check(query_seq_lengths, b, "int32", "query_seq_lengths")
  check(key_value_seq_lengths, b, "int32", "key_value_seq_lengths")
  n_q <- nnp_shape(query_arr)[3L]
  if (n_q %% ks[3L] != 0L)
    jax_error("The number of query heads must be a multiple of key/value heads, but got ",
              n_q, " vs ", ks[3L], class = "jax_value_error")
  scale_val <- if (is.null(scale)) 1.0 / sqrt(h) else scale

  if (!is.null(implementation)) {
    if (identical(implementation, "cudnn"))
      jax_error("The cudnn implementation of dot_product_attention requires a GPU; jaxR runs ",
                "on CPU (use implementation = \"xla\" or NULL)", class = "jax_value_error")
    if (!identical(implementation, "xla"))
      jax_error("Unsupported implementation option: ", implementation, class = "jax_value_error")
  }
  out <- nnp_attention_xla(query_arr, key_arr, value_arr, bias, mask, is_causal, scale_val,
                           query_seq_lengths, key_value_seq_lengths, local_window_size,
                           return_residual)
  if (return_residual) {
    return(list(jnp_reshape(out[[1L]], output_shape), jnp_reshape(out[[2L]], residual_shape)))
  }
  jnp_reshape(out, output_shape)
}

# ---- scaled matmul ----------------------------------------------------------------------

# Port of jax/_src/cudnn/scaled_matmul_stablehlo.py:_scaled_matmul_cpu_lowering_impl (the
# CPU lowering of the scaled_matmul primitive, jitted as in _scaled_matmul).
nnp_scaled_matmul_cpu <- jit(function(a, b, a_scales, b_scales, preferred_element_type) {
  a <- jnp_astype(a, preferred_element_type)
  b <- jnp_astype(b, preferred_element_type)
  a_scales <- jnp_astype(a_scales, preferred_element_type)
  b_scales <- jnp_astype(b_scales, preferred_element_type)
  as_ <- nnp_shape(a)
  n <- nnp_shape(b)[2L]
  k_a <- nnp_shape(a_scales)[3L]
  k_b <- nnp_shape(b_scales)[3L]
  a_block_size <- as_[3L] %/% k_a
  b_block_size <- as_[3L] %/% k_b
  a_scaled <- jnp_reshape(jnp_reshape(a, c(as_[1:2], k_a, a_block_size)) *
                            jnp_reshape(a_scales, c(as_[1:2], k_a, 1L)), as_)
  b_scaled <- jnp_reshape(jnp_reshape(b, c(as_[1L], n, k_b, b_block_size)) *
                            jnp_reshape(b_scales, c(as_[1L], n, k_b, 1L)), c(as_[1L], n, as_[3L]))
  lax_dot_general(a_scaled, b_scaled, list(list(2L, 2L), list(0L, 0L)),
                  preferred_element_type = preferred_element_type)
}, static_argnums = 4L)

#' Scaled matrix multiplication
#'
#' `nn_scaled_matmul` ports `jax.nn.scaled_matmul`: a block scaled batched matmul of
#' `lhs` `(B, M, K)` and `rhs` `(B, N, K)` with scales `(B, M, K_a)` and `(B, N, K_b)`,
#' equivalent to repeating each scale over its block of `K / K_a` (or `K / K_b`) entries
#' and contracting over `K`. It uses JAX's CPU lowering (inputs are cast to
#' `preferred_element_type`). `nn_scaled_dot_general` ports `jax.nn.scaled_dot_general`
#' without quantization configs (it is then [lax_dot_general()]); the block scaled
#' configurations need float8 and float4 dtypes, which jaxR does not have.
#' @param lhs,rhs Operands.
#' @param lhs_scales,rhs_scales Block scales.
#' @param preferred_element_type Output dtype (default `"float32"`).
#' @param dimension_numbers `list(list(lhs_contracting, rhs_contracting), list(lhs_batch,
#'   rhs_batch))` with 0-based dimensions, as for [lax_dot_general()].
#' @param configs Must be `NULL`.
#' @param implementation Deprecated and ignored (a warning is given).
#' @return An array of shape `(B, M, N)` (`nn_scaled_matmul`).
#' @examples
#' \dontrun{
#' a <- jnp_reshape(jnp_array(c(1, 2, 3)), c(1, 1, 3))
#' b <- jnp_reshape(jnp_array(c(4, 5, 6)), c(1, 1, 3))
#' s <- jnp_reshape(jnp_array(0.5), c(1, 1, 1))
#' nn_scaled_matmul(a, b, s, s)
#' }
#' @name nn_scaled_matmul
NULL

#' @rdname nn_scaled_matmul
#' @export
nn_scaled_matmul <- function(lhs, rhs, lhs_scales, rhs_scales,
                             preferred_element_type = "float32") {
  ops <- lapply(list(lhs, rhs, lhs_scales, rhs_scales), jnp_asarray)
  shapes <- lapply(ops, nnp_shape)
  if (!all(vapply(shapes, length, 1L) == 3L))
    jax_error("scaled_matmul requires all inputs to be 3-dimensional arrays",
              class = "jax_value_error")
  sa <- shapes[[1L]]; sb <- shapes[[2L]]; sas <- shapes[[3L]]; sbs <- shapes[[4L]]
  if (sa[3L] != sb[3L] || sa[1L] != sb[1L])
    jax_error("scaled_matmul requires inputs a and b to have matching batch (B) and ",
              "contract (K) dimensions, but got shapes ", nnp_tuple(sa), " and ", nnp_tuple(sb),
              class = "jax_value_error")
  if (sas[3L] != sbs[3L] || sas[1L] != sbs[1L])
    jax_error("scaled_matmul requires scales to have matching batch (B) and contract (K) ",
              "dimensions, but got shapes ", nnp_tuple(sas), " and ", nnp_tuple(sbs),
              class = "jax_value_error")
  if (sas[2L] != sa[2L] || sbs[2L] != sb[2L])
    jax_error("scaled_matmul requires scales to match non-contract dimensions of inputs, but ",
              "got shapes a: ", nnp_tuple(sa), ", b: ", nnp_tuple(sb), ", a_scales: ",
              nnp_tuple(sas), ", b_scales: ", nnp_tuple(sbs), class = "jax_value_error")
  nnp_scaled_matmul_cpu(ops[[1L]], ops[[2L]], ops[[3L]], ops[[4L]],
                        dtype(preferred_element_type))
}

#' @rdname nn_scaled_matmul
#' @export
nn_scaled_dot_general <- function(lhs, rhs, dimension_numbers, preferred_element_type = "float32",
                                  configs = NULL, implementation = NULL) {
  if (!is.null(implementation))
    warning("Backend selector, now ignored. The system chooses the backend automatically.",
            call. = FALSE)
  if (!is.null(configs))
    jax_error("nn_scaled_dot_general with quantization configs needs the float8 and float4 ",
              "dtypes, which jaxR does not support", class = "jax_not_implemented_error")
  lax_dot_general(lhs, rhs, dimension_numbers, preferred_element_type = preferred_element_type)
}
