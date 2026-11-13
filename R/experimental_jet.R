# Taylor mode automatic differentiation: port of jax/experimental/jet.py.

# ---- trace ------------------------------------------------------------------------

# A jet tracer holds a primal and the list of its Taylor coefficients (`terms`). A value
# without a tracer of the trace has the zero series, written NULL here (jet.zero_series).
jet_pair <- function(x, trace) {
  if (is_own(x, trace)) list(x$primal, x$terms) else list(x, NULL)
}

jet_tracer <- function(trace, primal, terms) {
  if (is.null(terms)) return(primal)
  new_tracer(trace, aval_of(primal), primal = primal, terms = terms)
}

# Port of JetTrace.process_primitive.
process_jet <- function(trace, prim, args, params) {
  ov <- prim$override$jet
  if (!is.null(ov)) return(ov(trace, args, params))
  pairs <- lapply(args, jet_pair, trace = trace)
  primals <- lapply(pairs, `[[`, 1L)
  zero <- vapply(pairs, function(p) is.null(p[[2L]]), NA)
  if (all(zero)) {
    out <- with_trace(trace$parent, bind(prim, primals, params))
    # In JAX the dtype promotion of a jax.numpy ufunc runs inside the jet rule of that
    # function (a jit primitive there), after the zero terms of a constant operand are
    # made: a weakly typed conversion of a constant (an R number next to an array) so
    # gives weakly typed zero terms. The mark carries that to the next rule.
    if (isTRUE(.jx$jet_weak) && identical(prim$name, "convert_element_type") &&
        isTRUE(params$weak_type))
      attr(out, "jet_weak_zeros") <- trace$tag
    return(out)
  }
  res <- with_trace(trace$parent, {
    series <- lapply(seq_along(pairs), function(i) {
      if (!zero[i]) return(pairs[[i]][[2L]])
      a <- aval_of(primals[[i]])
      weak <- identical(attr(primals[[i]], "jet_weak_zeros"), trace$tag)
      rep(list(zeros_aval(new_aval(a$shape, a$dtype, weak))), trace$order)
    })
    if (is.null(prim$jet))
      jax_error("jet rule for '", prim$name, "' not implemented", class = "jax_key_error")
    prim$jet(primals, series, params)
  })
  if (is.null(res[[2L]])) return(res[[1L]])
  if (prim$multiple_results) {
    Map(function(p, ts) jet_tracer(trace, p, ts), res[[1L]], res[[2L]])
  } else {
    jet_tracer(trace, res[[1L]], res[[2L]])
  }
}

# Port of JetTrace.process_custom_jvp_call and process_custom_vjp_call: the custom rules
# are ignored and the function itself is traced.
custom_jvp_call_p$override$jet <- function(trace, args, params) params$fun(args)
custom_vjp_call_p$override$jet <- function(trace, args, params) params$fun(args)

# ---- jet --------------------------------------------------------------------------

jet_fact <- function(n) lax_exp(lax_lgamma(n + 1))

#' Taylor mode higher order automatic differentiation
#'
#' Port of `jax.experimental.jet.jet`. Where [jvp()] pushes one tangent through a
#' function, `experimental_jet_jet()` pushes a truncated Taylor polynomial: given the
#' value `h0` of an inner function and its derivatives `h1, ..., hK` along a direction,
#' it returns the value of `fun(h)` and its derivatives of order 1 to K, without nesting
#' first order differentiation K times.
#'
#' Rules exist for the primitives that JAX has rules for (arithmetic, the elementary
#' functions, linear structural operations, reductions, `dot_general`, convolutions,
#' gathers and scatter-add). As in JAX, control flow primitives have no rule, and custom
#' derivative rules are ignored: the function itself is differentiated.
#'
#' A rule for another primitive is a function `function(primals_in, series_in, params)`
#' assigned to the `jet` field of the primitive object (the counterpart of
#' `jet.jet_rules[prim] = rule`); it returns `list(primal_out, series_out)`.
#' @param fun Function of one or more arrays, returning an array or a pytree of arrays.
#' @param primals List of the arguments of `fun`: the point of expansion.
#' @param series List with one entry per argument; each entry is the list of the
#'   higher order terms of that argument. All entries have the same length, the order
#'   of the expansion.
#' @param factorial_scaled If `TRUE` (the default) the terms of the input and of the
#'   output are derivatives. If `FALSE` they are Taylor coefficients, the derivatives
#'   divided by the factorial of their order.
#' @return `list(primals_out, series_out)`: the value of `fun` and, with the same
#'   structure, the list of the higher order terms of each output.
#' @examples
#' \dontrun{
#' # f(h(z)) = sin(z^3) at z = 0.5: h and its first two derivatives go in.
#' h <- list(0.5^3, 3 * 0.5^2, 6 * 0.5)
#' out <- experimental_jet_jet(jnp_sin, list(h[[1]]), list(h[2:3]))
#' out[[1]]        # sin(h0)
#' out[[2]][[1]]   # cos(h0) * h1
#' out[[2]][[2]]   # -sin(h0) * h1^2 + cos(h0) * h2
#' }
#' @export
experimental_jet_jet <- function(fun, primals, series, factorial_scaled = TRUE) {
  order <- unique(lengths(series))
  if (length(order) != 1L)
    jax_error("jet terms have inconsistent lengths for different arguments",
              class = "jax_value_error")
  n <- min(length(primals), length(series))
  primals <- unname(primals)[seq_len(n)]
  series <- lapply(unname(series)[seq_len(n)], as.list)
  for (i in seq_len(n)) {
    if (!treedef_is_leaf(tree_structure(primals[[i]])))
      jax_error("primal value at position ", i - 1L, " is not an array", class = "jax_value_error")
    for (j in seq_len(order)) {
      if (!treedef_is_leaf(tree_structure(series[[i]][[j]])))
        jax_error("term ", j - 1L, " for argument ", i - 1L, " is not an array",
                  class = "jax_value_error")
    }
  }
  if (factorial_scaled)
    series <- lapply(series, function(terms) {
      lapply(seq_along(terms), function(k) jnp_true_divide(terms[[k]], jet_fact(k)))
    })
  trace <- new_trace("jet", current_trace())
  trace$order <- order
  on.exit(trace$alive <- FALSE, add = TRUE)
  ins <- Map(function(p, ts) {
    p <- as_jax(p)
    new_tracer(trace, aval_of(p), primal = p, terms = lapply(ts, as_jax))
  }, primals, series)
  fl <- with_trace(trace, tree_flatten(do.call(fun, ins)))
  pairs <- lapply(fl$leaves, function(v) jet_pair(if (is_tracer(v)) v else as_jax(v), trace))
  out_primals <- lapply(pairs, `[[`, 1L)
  out_terms <- lapply(pairs, function(p) {
    if (is.null(p[[2L]])) rep(list(jnp_zeros_like(p[[1L]])), order) else p[[2L]]
  })
  if (factorial_scaled)
    out_terms <- lapply(out_terms, function(terms) {
      lapply(seq_along(terms), function(k) jnp_multiply(terms[[k]], jet_fact(k)))
    })
  list(tree_unflatten(fl$treedef, out_primals), tree_unflatten(fl$treedef, out_terms))
}

# jet.jet2: Taylor coefficients in and out. Rules call it with (primals_in, series_in).
jet2 <- function(fun, primals, series) experimental_jet_jet(fun, primals, series, FALSE)

# ---- rule helpers -----------------------------------------------------------------

# Python's sum() starts from the integer 0, and the rules below are written with the
# operators of arrays, so the helpers are the jnp functions behind those operators.
jet_sum <- function(xs) Reduce(jnp_add, xs, 0L)
`%j*%` <- function(a, b) jnp_multiply(a, b)
`%j/%` <- function(a, b) jnp_true_divide(a, b)
`%j+%` <- function(a, b) jnp_add(a, b)
`%j-%` <- function(a, b) jnp_subtract(a, b)

# The k-th terms of all arguments, k = 1, ..., order (zip(*series_in)).
jet_zip <- function(series_in) {
  lapply(seq_along(series_in[[1L]]), function(k) lapply(series_in, `[[`, k))
}

# v'(t) = c(t) u'(t): v[k] = sum_{j=1}^{k} j c[k-j] u[j] / k. Lists are 0-based in the
# source; here element i + 1 holds coefficient i.
jet_conv <- function(k, cs, u) {
  jet_sum(lapply(seq_len(k), function(j) (j %j*% cs[[k - j + 1L]]) %j*% u[[j + 1L]])) %j/% k
}

jet_defzero <- function(prim) {
  prim$jet <- function(primals_in, series_in, params) list(bind(prim, primals_in, params), NULL)
}

jet_deflinear <- function(prim) {
  prim$jet <- function(primals_in, series_in, params) {
    primal_out <- bind(prim, primals_in, params)
    series_out <- lapply(jet_zip(series_in), function(terms_in) bind(prim, terms_in, params))
    if (prim$multiple_results)
      series_out <- lapply(seq_along(primal_out), function(i) lapply(series_out, `[[`, i))
    list(primal_out, series_out)
  }
}

# jet.def_deriv: the rule of a unary primitive from its first derivative.
jet_def_deriv <- function(prim, deriv) {
  prim$jet <- function(primals_in, series_in, params) {
    x <- primals_in[[1L]]
    series <- series_in[[1L]]
    primal_out <- bind(prim, list(x))
    d <- jet2(deriv, primals_in, series_in)
    cs <- c(list(d[[1L]]), d[[2L]])
    u <- c(list(x), series)
    list(primal_out, lapply(seq_along(series), function(k) jet_conv(k, cs, u)))
  }
}

# jet.def_comp: the rule of a primitive as the jet of a composition of other primitives.
jet_def_comp <- function(prim, comp) {
  prim$jet <- function(primals_in, series_in, params) jet2(comp, primals_in, series_in)
}

# A NumPy float64 scalar is strongly typed, unlike a Python float; np.log(2) is one.
jet_np64 <- function(v) jnp_asarray(v, dtype = "float64")

# ---- rules ------------------------------------------------------------------------

for (p in list(le_p, lt_p, gt_p, ge_p, eq_p, ne_p, not_p, and_p, or_p, xor_p, floor_p, ceil_p,
               round_p, sign_p, stop_gradient_p, is_finite_p, shift_left_p,
               shift_right_arithmetic_p, shift_right_logical_p, bitcast_convert_type_p,
               ufn_frexp_p))
  jet_defzero(p)

for (p in list(neg_p, real_p, complex_p, conj_p, imag_p, add_p, add_any_p, sub_p,
               convert_element_type_p, broadcast_in_dim_p, concatenate_p, split_p, stack_p,
               unstack_p, pad_p, reshape_p, squeeze_p, rev_p, transpose_p, slice_p,
               reduce_sum_p, reduce_window_sum_p, fft_p, copy_p, device_put_p, cumsum_p))
  jet_deflinear(p)
rm(p)

dynamic_slice_p$jet <- function(primals_in, series_in, params) {
  start <- primals_in[-1L]
  list(bind(dynamic_slice_p, primals_in, params),
       lapply(series_in[[1L]], function(t) bind(dynamic_slice_p, c(list(t), start), params)))
}

dynamic_update_slice_p$jet <- function(primals_in, series_in, params) {
  start <- primals_in[-(1:2)]
  list(bind(dynamic_update_slice_p, primals_in),
       lapply(jet_zip(series_in), function(terms_in) {
         bind(dynamic_update_slice_p, c(terms_in[1:2], start), params)
       }))
}

# The parallel prefix scan is used whatever the backend, because reduce_window is not
# differentiable to arbitrary order.
jet_cumulative <- function(combine) function(primals_in, series_in, params) {
  jet2(function(x) lax_associative_scan(combine, x, reverse = params$reverse, axis = params$axis),
       primals_in, series_in)
}
cumprod_p$jet <- jet_cumulative(function(a, b) lax_mul(a, b))
cummax_p$jet <- jet_cumulative(function(a, b) lax_max(a, b))
cummin_p$jet <- jet_cumulative(function(a, b) lax_min(a, b))

jet_def_deriv(erf_p, function(x) lax_mul(sp_const(x, 2 / sqrt(pi)), lax_exp(lax_neg(lax_square(x)))))

jet_def_comp(exp2_p, function(x) lax_exp(x %j*% jet_np64(log(2))))
jet_def_comp(expm1_p, function(x) lax_exp(x) %j-% 1L)
jet_def_comp(log2_p, function(x) lax_log(x) %j/% jet_np64(log(2)))
jet_def_comp(log1p_p, function(x) lax_log(1L %j+% x))
jet_def_comp(sqrt_p, function(x) jnp_power(x, 0.5))
jet_def_comp(square_p, function(x) x %j*% x)

one_minus_square_p$jet <- function(primals_in, series_in, params) {
  sq <- square_p$jet(primals_in, series_in, list())
  list(lax_one_minus_square(primals_in[[1L]]), lapply(sq[[2L]], lax_neg))
}

polynomial_p$jet <- function(primals_in, series_in, params) {
  jet2(function(x, ...) {
    coeffs <- list(...)
    acc <- coeffs[[length(coeffs)]]
    for (cf in rev(coeffs[-length(coeffs)])) acc <- lax_add(lax_mul(acc, x), cf)
    acc
  }, primals_in, series_in)
}

jet_def_comp(rsqrt_p, function(x) jnp_power(x, -0.5))
jet_def_deriv(asinh_p, function(x) asinh_jvp(lax_full_like(x, 1), x))
jet_def_comp(acosh_p, function(x) lax_log(x %j+% (lax_sqrt(x %j-% 1L) %j*% lax_sqrt(x %j+% 1L))))
jet_def_comp(atanh_p, function(x) 0.5 %j*% lax_log(lax_div(1L %j+% x, 1L %j-% x)))
jet_def_comp(erfc_p, function(x) 1L %j-% lax_erf(x))
jet_def_comp(rem_p, function(x, y) x %j-% (y %j*% lax_floor(x %j/% y)))
jet_def_comp(clamp_p, function(a, x, b) lax_min(lax_max(a, x), b))

erf_inv_p$jet <- function(primals_in, series_in, params) {
  x <- primals_in[[1L]]
  series <- series_in[[1L]]
  K <- length(series)
  u <- c(list(x), series)
  v <- vector("list", K + 1L)
  v[[1L]] <- lax_erf_inv(x)
  if (K == 0L) return(list(v[[1L]], list()))
  # The derivative is taken on the codomain: d/dx erf_inv(x) = sqrt(pi) / 2 exp(y^2). It
  # is propagated by hand through the square and the exponential.
  deriv_const <- jet_np64(sqrt(pi) / 2)
  cc <- vector("list", K)
  tmp_sq <- vector("list", K)
  tmp_exp <- vector("list", K)
  tmp_sq[[1L]] <- lax_square(v[[1L]])
  tmp_exp[[1L]] <- lax_exp(tmp_sq[[1L]])
  cc[[1L]] <- lax_mul(deriv_const, tmp_exp[[1L]])
  for (k in seq_len(K - 1L)) {
    v[[k + 1L]] <- jet_conv(k, cc, u)
    tmp_sq[[k + 1L]] <- jet_sum(lapply(0:k, function(j) v[[k - j + 1L]] %j*% v[[j + 1L]]))
    tmp_exp[[k + 1L]] <- jet_conv(k, tmp_exp, tmp_sq)
    cc[[k + 1L]] <- deriv_const %j*% tmp_exp[[k + 1L]]
  }
  v[[K + 1L]] <- jet_conv(K, cc, u)
  list(v[[1L]], v[-1L])
}

exp_p$jet <- function(primals_in, series_in, params) {
  x <- primals_in[[1L]]
  u <- c(list(x), series_in[[1L]])
  v <- list(lax_exp(x))
  for (k in seq_along(series_in[[1L]])) v[[k + 1L]] <- jet_conv(k, v, u)
  list(v[[1L]], v[-1L])
}

pow_p$jet <- function(primals_in, series_in, params) {
  xs <- jet2(function(x, y) lax_mul(y, lax_log(x)), primals_in, series_in)
  u <- c(list(xs[[1L]]), xs[[2L]])
  v <- list(jnp_power(primals_in[[1L]], primals_in[[2L]]))
  for (k in seq_along(xs[[2L]])) v[[k + 1L]] <- jet_conv(k, v, u)
  list(v[[1L]], v[-1L])
}

jet_pow_by_squaring <- function(x, n) {
  if (n < 0) return(jet_pow_by_squaring(1L %j/% x, -n))
  if (n == 0) return(1L)
  if (n %% 2 == 0) return(jet_pow_by_squaring(x %j*% x, n / 2))
  x %j*% jet_pow_by_squaring(x %j*% x, (n - 1) / 2)
}

integer_pow_p$jet <- function(primals_in, series_in, params) {
  y <- params$y
  if (y == 0) return(jet2(function(x) jnp_ones_like(x), primals_in, series_in))
  jet2(function(x) jet_pow_by_squaring(x, y), primals_in, series_in)
}

jet_logistic <- function(primals_in, series_in, params) {
  x <- primals_in[[1L]]
  u <- c(list(x), series_in[[1L]])
  v <- list(lax_logistic(x))
  # e holds the terms of the derivative, sigmoid (1 - sigmoid).
  e <- list(v[[1L]] %j*% (1L %j-% v[[1L]]))
  for (k in seq_along(series_in[[1L]])) {
    v[[k + 1L]] <- jet_conv(k, e, u)
    e[[k + 1L]] <- ((1L %j-% v[[1L]]) %j*% v[[k + 1L]]) %j-%
      jet_sum(lapply(seq_len(k), function(j) v[[j + 1L]] %j*% v[[k - j + 1L]]))
  }
  list(v[[1L]], v[-1L])
}
logistic_p$jet <- jet_logistic

tanh_p$jet <- function(primals_in, series_in, params) {
  out <- jet_logistic(list(2L %j*% primals_in[[1L]]),
                      list(lapply(series_in[[1L]], function(s) 2L %j*% s)), params)
  list((2L %j*% out[[1L]]) %j-% 1L, lapply(out[[2L]], function(s) 2L %j*% s))
}

log_p$jet <- function(primals_in, series_in, params) {
  x <- primals_in[[1L]]
  u <- c(list(x), series_in[[1L]])
  v <- list(lax_log(x))
  for (k in seq_along(series_in[[1L]])) {
    conv <- jet_sum(lapply(seq_len(k - 1L), function(j) (j %j*% v[[j + 1L]]) %j*% u[[k - j + 1L]]))
    v[[k + 1L]] <- (u[[k + 1L]] %j-% (conv %j/% k)) %j/% u[[1L]]
  }
  list(v[[1L]], v[-1L])
}

atan2_p$jet <- function(primals_in, series_in, params) {
  primal_out <- lax_atan2(primals_in[[1L]], primals_in[[2L]])
  q <- jet2(function(x, y) lax_div(x, y), primals_in, series_in)
  one <- sp_const(q[[1L]], 1)
  d <- jet2(function(x) lax_div(one, 1L %j+% lax_square(x)), q[1L], list(q[[2L]]))
  cs <- c(list(d[[1L]]), d[[2L]])
  u <- c(list(q[[1L]]), q[[2L]])
  list(primal_out, lapply(seq_along(q[[2L]]), function(k) jet_conv(k, cs, u)))
}

div_p$jet <- function(primals_in, series_in, params) {
  u <- c(primals_in[1L], series_in[[1L]])
  w <- c(primals_in[2L], series_in[[2L]])
  v <- list()
  for (k in seq_along(u) - 1L) {
    conv <- jet_sum(lapply(seq_len(k) - 1L, function(j) v[[j + 1L]] %j*% w[[k - j + 1L]]))
    v[[k + 1L]] <- (u[[k + 1L]] %j-% conv) %j/% w[[1L]]
  }
  list(v[[1L]], v[-1L])
}

# sin and cos (sign -1) and sinh and cosh (sign 1) are propagated together.
jet_sinusoidal <- function(sign, s_fun, c_fun, which) function(primals_in, series_in, params) {
  x <- primals_in[[1L]]
  u <- c(list(x), series_in[[1L]])
  sn <- list(s_fun(x))
  cs <- list(c_fun(x))
  for (k in seq_along(series_in[[1L]])) {
    sn[[k + 1L]] <- jet_sum(lapply(seq_len(k), function(j) {
      (j %j*% u[[j + 1L]]) %j*% cs[[k - j + 1L]]
    })) %j/% k
    cs[[k + 1L]] <- (jet_sum(lapply(seq_len(k), function(j) {
      (j %j*% u[[j + 1L]]) %j*% sn[[k - j + 1L]]
    })) %j/% k) %j*% sign
  }
  out <- if (which == 1L) sn else cs
  list(out[[1L]], out[-1L])
}
sin_p$jet <- jet_sinusoidal(-1L, function(x) lax_sin(x), function(x) lax_cos(x), 1L)
cos_p$jet <- jet_sinusoidal(-1L, function(x) lax_sin(x), function(x) lax_cos(x), 2L)
sinh_p$jet <- jet_sinusoidal(1L, function(x) lax_sinh(x), function(x) lax_cosh(x), 1L)
cosh_p$jet <- jet_sinusoidal(1L, function(x) lax_sinh(x), function(x) lax_cosh(x), 2L)

jet_bilinear <- function(prim) {
  prim$jet <- function(primals_in, series_in, params) {
    u <- c(primals_in[1L], series_in[[1L]])
    w <- c(primals_in[2L], series_in[[2L]])
    v <- lapply(seq_along(u) - 1L, function(k) {
      jet_sum(lapply(0:k, function(j) bind(prim, list(u[[j + 1L]], w[[k - j + 1L]]), params)))
    })
    list(v[[1L]], v[-1L])
  }
}
jet_bilinear(dot_general_p)
jet_bilinear(mul_p)
jet_bilinear(conv_general_dilated_p)

gather_p$jet <- function(primals_in, series_in, params) {
  list(bind(gather_p, primals_in, params),
       lapply(series_in[[1L]], function(g) bind(gather_p, list(g, primals_in[[2L]]), params)))
}

jet_reduce_chooser <- function(prim) {
  prim$jet <- function(primals_in, series_in, params) {
    operand <- primals_in[[1L]]
    gs <- series_in[[1L]]
    primal_out <- bind(prim, list(operand), params)
    if (!length(gs)) return(list(primal_out, list()))
    axes <- params$axes
    shape <- aval_of(operand)$shape
    shape[axes + 1L] <- 1L
    location_indicators <- lax_convert_element_type(
      lax_eq(operand, lax_reshape(primal_out, shape)), aval_of(gs[[1L]])$dtype)
    counts <- lax_reduce_sum(location_indicators, axes)
    list(primal_out, lapply(gs, function(g) {
      lax_div(lax_reduce_sum(lax_mul(g, location_indicators), axes), counts)
    }))
  }
}
jet_reduce_chooser(reduce_max_p)
jet_reduce_chooser(reduce_min_p)

abs_p$jet <- function(primals_in, series_in, params) {
  x <- primals_in[[1L]]
  zero <- lax_full_like(x, 0, shape = integer(0))
  negs <- lax_select(lax_lt(x, zero), lax_full_like(x, -1), lax_full_like(x, 1))
  list(bind(abs_p, list(x), params), lapply(series_in[[1L]], function(t) negs %j*% t))
}

select_n_p$jet <- function(primals_in, series_in, params) {
  b <- primals_in[[1L]]
  list(bind(select_n_p, primals_in),
       lapply(jet_zip(series_in), function(terms_in) bind(select_n_p, c(list(b), terms_in[-1L]))))
}

# max and min: the larger (smaller) operand is selected, and ties average the two.
jet_minmax <- function(cmp) function(primals_in, series_in, params) {
  xy <- jnp_broadcast_arrays(primals_in[[1L]], primals_in[[2L]])
  pick <- cmp(xy[[1L]], xy[[2L]])
  tie <- jnp_equal(xy[[1L]], xy[[2L]])
  list(lax_select(pick, xy[[1L]], xy[[2L]]),
       lapply(jet_zip(series_in), function(terms_in) {
         t <- jnp_broadcast_arrays(terms_in[[1L]], terms_in[[2L]])
         lax_select(tie, (t[[1L]] %j+% t[[2L]]) %j/% 2L, lax_select(pick, t[[1L]], t[[2L]]))
       }))
}
max_p$jet <- jet_minmax(function(x, y) jnp_greater(x, y))
min_p$jet <- jet_minmax(function(x, y) jnp_less(x, y))

scatter_add_p$jet <- function(primals_in, series_in, params) {
  idx <- primals_in[[2L]]
  list(bind(scatter_add_p, primals_in, params),
       lapply(jet_zip(series_in), function(terms_in) {
         bind(scatter_add_p, list(terms_in[[1L]], idx, terms_in[[3L]]), params)
       }))
}
