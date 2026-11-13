# lax special functions (port of jax/_src/lax/special.py, with the pieces of
# jax/_src/lax/lax.py, erf.py and lgamma.py that it builds on).

# ---- private helpers ---------------------------------------------------------------------

# Port of jax/_src/lax/lax.py:_const: a scalar with the dtype and weak type of `example`.
# Port of lax._const: a scalar constant with the dtype of `example`. It is weakly typed
# only when `example` stands for a Python scalar: a weakly typed scalar whose dtype is
# the default one of its kind.
sp_const <- function(example, value) {
  example <- as_jax(example)
  a <- aval_of(example)
  py_scalar <- a$weak && !length(a$shape) &&
    a$dtype == switch(dtype_kind(a$dtype), f = default_float(), i = default_int(),
                      c = default_complex(), a$dtype)
  lax_full_like(example, value, dtype = if (py_scalar) NULL else a$dtype, shape = integer(0))
}

sp_dtype <- function(x) aval_of(x)$dtype
sp_shape <- function(x) if (inherits(x, "jax_aval")) x$shape else aval_of(x)$shape
sp_isnan <- function(x) lax_ne(x, x)

# Python integers are weakly typed with the default integer dtype.
sp_int <- function(v) lax_convert_element_type(as_jax(as.integer(v)), default_int(), weak_type = TRUE)

# Port of jax/_src/lax/special.py:_any.
sp_any <- function(pred) lax_reduce_or(pred, seq_along(sp_shape(pred)) - 1L)

# Port of jax/_src/lax/special.py:_up_and_broadcast.
sp_up_and_broadcast <- function(doit) {
  function(...) {
    args <- lax_args(...)
    shape <- broadcast_shapes_impl(lapply(args, sp_shape))
    args <- lapply(args, broadcast_to_shape, shape = shape)
    a_dtype <- sp_dtype(args[[1L]])
    needs_upcast <- a_dtype %in% c("bfloat16", "float16")
    if (needs_upcast) args <- lapply(args, lax_convert_element_type, new_dtype = "float32")
    result <- do.call(doit, c(args, list(dtype = if (needs_upcast) "float32" else a_dtype)))
    if (needs_upcast) result <- lax_convert_element_type(result, a_dtype)
    result
  }
}

# ---- polynomial (port of jax/_src/lax/lax.py:polynomial_p) --------------------------------
# Internal until lax.R exports lax_polynomial; erf.py and jnp.polyval evaluate through it.

sp_polynomial_p <- new_primitive("polynomial")
sp_polynomial_p$abstract_eval <- naryop_abstract_eval("polynomial", "number")
sp_polynomial_p$batch <- lax_broadcast_batcher(sp_polynomial_p)

# Port of jax/_src/lax/lax.py:_polynomial_lower. Past the unroll threshold JAX runs the same
# Horner recurrence as a scan starting from zeros; it is emitted here as straight line code.
sp_polynomial_p$lowering <- lower_fun(function(args, params) {
  x <- args[[1L]]
  coeffs <- args[-1L]
  n <- length(coeffs)
  if (n <= (params$unroll %||% 8L)) {
    acc <- coeffs[[n]]
    for (c in rev(coeffs[-n])) acc <- lax_add(lax_mul(acc, x), c)
    return(acc)
  }
  acc <- lax_full_like(x, 0, shape = broadcasting_shape_rule("polynomial", lapply(args, aval_of)))
  for (c in rev(coeffs)) acc <- lax_add(lax_mul(acc, x), c)
  acc
})

# Port of jax/_src/lax/lax.py:_polynomial_jvp.
sp_polynomial_p$jvp <- function(primals, tangents, params) {
  x <- primals[[1L]]
  coeffs <- primals[-1L]
  y <- bind(sp_polynomial_p, primals, params)
  dot_y <- zero_of(tangent_aval(aval_of(y)))
  if (!is_zero(tangents[[1L]])) {
    deriv <- lapply(seq_along(coeffs)[-1L], function(i) {
      lax_mul(sp_const(coeffs[[i]], i - 1L), coeffs[[i]])
    })
    dot_y <- add_tangents(dot_y, lax_mul(tangents[[1L]], sp_polynomial(x, deriv, params$unroll)))
  }
  dot_c <- tangents[-1L]
  if (!all(vapply(dot_c, is_zero, NA))) {
    conc <- Map(function(c, d) if (is_zero(d)) lax_full_like(c, 0) else d, coeffs, dot_c)
    dot_y <- add_tangents(dot_y, sp_polynomial(x, conc, params$unroll))
  }
  list(y, dot_y)
}

# Port of jax/_src/lax/lax.py:_polynomial_transpose (linear in the coefficients).
sp_polynomial_p$transpose <- function(ct, args, params) {
  x <- args[[1L]]
  coeffs <- args[-1L]
  if (is_zero(ct)) return(vector("list", length(args)))
  out <- vector("list", length(args))
  xk_ct <- ct
  for (k in seq_along(coeffs)) {
    if (is_undefined(coeffs[[k]])) out[k + 1L] <- list(unbroadcast(coeffs[[k]]$aval, xk_ct))
    if (k < length(coeffs)) xk_ct <- lax_mul(xk_ct, x)
  }
  out
}

# Port of jax/_src/lax/lax.py:polynomial. `coeffs` run from the constant term upward.
sp_polynomial <- function(x, coeffs, unroll = NULL) {
  x <- as_jax(x)
  dt <- sp_dtype(x)
  cs <- lapply(coeffs, function(c) {
    if (!is_jax_value(c) && is.null(dim(c)) && length(c) == 1L) sp_const(x, c)
    else lax_convert_element_type(c, dt)
  })
  if (!length(cs)) return(lax_full_like(x, 0))
  if (length(cs) == 1L) {
    shape <- broadcasting_shape_rule("polynomial", list(aval_of(x), aval_of(cs[[1L]])))
    return(broadcast_to_shape(cs[[1L]], shape))
  }
  bind(sp_polynomial_p, c(list(x), cs),
       if (is.null(unroll)) list() else list(unroll = as.integer(unroll)))
}

# Port of jax/_src/numpy/polynomial.py:polyval for a coefficient vector held in R (highest
# degree first) with the dtype of x, which is how every caller in scipy/special.py uses it.
sp_polyval <- function(p, x) {
  x <- as_jax(x)
  dt <- sp_dtype(x)
  sp_polynomial(x, lapply(rev(p), function(c) as_jax(c, dt)))
}

# ---- jax.numpy helpers used by scipy/special.py --------------------------------------------

# Port of jax/_src/numpy/lax_numpy.py:select (the first true condition wins).
sp_select <- function(condlist, choicelist, default = 0) {
  conds <- lapply(condlist, jnp_asarray, dtype = "bool")
  choices <- promote_dtypes(c(list(default), choicelist))
  shape <- broadcast_shapes_impl(lapply(c(conds, choices), sp_shape))
  idx <- lax_full(shape, 0L, "int32")
  for (i in rev(seq_along(conds))) {
    idx <- lax_select(jnp_broadcast_to(conds[[i]], shape), lax_full(shape, i, "int32"), idx)
  }
  do.call(lax_select_n, c(list(idx), lapply(choices, jnp_broadcast_to, shape = shape)))
}

# vmap with its (lazily evaluated) in_axes argument forced, for use in loops.
sp_vmap_over <- function(f, in_axes) {
  force(in_axes)
  vmap(f, in_axes = in_axes)
}

# Port of jax/_src/numpy/vectorize.py:vectorize for scalar core dimensions: broadcasting is
# pushed into vmap, so `f` always sees scalars.
sp_vectorize <- function(f) {
  function(...) {
    args <- lapply(list(...), as_jax)
    shapes <- lapply(args, sp_shape)
    bshape <- broadcast_shapes_impl(shapes)
    nb <- length(bshape)
    squeezed <- lapply(args, function(a) {
      sq <- which(sp_shape(a) == 1L) - 1L
      if (length(sq)) lax_squeeze(a, sq) else a
    })
    rev_filled <- lapply(shapes, function(s) rev(c(rep(1L, nb - length(s)), s)))
    vf <- f
    expand <- integer(0)
    for (negdim in seq_len(nb)) {
      sizes <- vapply(rev_filled, `[`, 1L, negdim)
      if (all(sizes == 1L)) {
        expand <- c(expand, nb - negdim)
      } else {
        vf <- sp_vmap_over(vf, lapply(sizes, function(s) if (s == 1L) NULL else 0L))
      }
    }
    result <- do.call(vf, squeezed)
    if (!length(expand)) return(result)
    tree_map(function(r) lax_expand_dims(r, sort(expand)), result)
  }
}

# Port of jax/_src/numpy/lax_numpy.py:piecewise. Every entry of `funclist` is a function
# of x or a constant; the last true condition wins and a trailing extra entry is the default.
sp_piecewise <- function(x, condlist, funclist, ...) {
  x <- as_jax(x)
  nc <- length(condlist)
  nf <- length(funclist)
  if (nf == nc + 1L) funclist <- c(funclist[nf], funclist[-nf])
  else if (nf == nc) funclist <- c(list(0), funclist)
  else jax_error("with ", nc, " condition(s), either ", nc, " or ", nc + 1L,
                 " functions are expected; got ", nf, class = "jax_value_error")
  dt <- sp_dtype(x)
  extra <- list(...)
  branches <- lapply(funclist, function(f) {
    force(f)
    if (is.function(f)) function(x) lax_convert_element_type(do.call(f, c(list(x), extra)), dt)
    else function(x) as_jax(f, dt)
  })
  conds <- lapply(condlist, jnp_asarray, dtype = "bool")
  shape <- broadcast_shapes_impl(lapply(conds, sp_shape))
  idx <- lax_full(shape, 0L, "int32")
  for (i in seq_len(nc)) {
    idx <- lax_select(jnp_broadcast_to(conds[[i]], shape), lax_full(shape, i, "int32"), idx)
  }
  sp_vectorize(function(i, x) lax_switch(i, branches, x))(idx, x)
}

# Port of jax/_src/custom_derivatives.py:custom_jvp.defjvps: one rule g(tangent, primal_out,
# ...primals) per argument (NULL for none). The primal output comes from the function
# itself, so its custom rule also applies to higher order derivatives.
sp_defjvps <- function(f, ...) {
  gs <- list(...)
  self <- custom_jvp_defjvp(f, function(primals, tangents) {
    primal_out <- do.call(self, primals)
    out <- NULL
    for (i in seq_along(gs)) {
      if (is.null(gs[[i]])) next
      t <- do.call(gs[[i]], c(list(tangents[[i]], primal_out), primals))
      out <- if (is.null(out)) t else tree_map(function(a, b) add_tangents(as_jax(a), as_jax(b)), out, t)
    }
    list(primal_out, out)
  })
  self
}

# ---- continued fractions and series (port of tensorflow/compiler/xla/client/lib/math.cc
# as transcribed in jax/_src/lax/special.py) --------------------------------------------------

# Port of jax/_src/lax/special.py:lentz_thompson_barnett_algorithm.
# State: iteration, unconverged, c, d, h.
sp_lentz_thompson_barnett <- function(num_iterations, small, threshold, nth_partial_numerator,
                                      nth_partial_denominator, inputs) {
  cond_fn <- function(state) {
    lax_bitwise_and(lax_lt(state[[1L]], sp_int(num_iterations)), sp_any(state[[2L]]))
  }
  body_fn <- function(state) {
    iteration <- state[[1L]]
    unconverged <- state[[2L]]
    partial_numerator <- do.call(nth_partial_numerator, c(list(iteration), inputs))
    partial_denominator <- do.call(nth_partial_denominator, c(list(iteration), inputs))
    c <- lax_add(partial_denominator, lax_div(partial_numerator, state[[3L]]))
    small_constant <- lax_full_like(c, small)
    c <- lax_select(lax_lt(lax_abs(c), small_constant), small_constant, c)
    d <- lax_add(partial_denominator, lax_mul(partial_numerator, state[[4L]]))
    d <- lax_select(lax_lt(lax_abs(d), small_constant), small_constant, d)
    d <- lax_integer_pow(d, -1L)
    delta <- lax_mul(c, d)
    h <- lax_mul(state[[5L]], delta)
    tolerance_comparison <- lax_ge(lax_abs(lax_sub(delta, sp_const(delta, 1))),
                                   sp_const(delta, threshold))
    list(lax_add(iteration, sp_int(1L)),
         lax_bitwise_and(unconverged, tolerance_comparison),
         lax_select(unconverged, c, state[[3L]]),
         lax_select(unconverged, d, state[[4L]]),
         lax_select(unconverged, h, state[[5L]]))
  }
  partial_denominator <- do.call(nth_partial_denominator, c(list(sp_int(0L)), inputs))
  small_c <- sp_const(partial_denominator, small)
  h <- lax_select(lax_lt(lax_abs(partial_denominator), small_c),
                  lax_broadcast_in_dim(small_c, sp_shape(partial_denominator), integer(0)),
                  partial_denominator)
  state <- list(sp_int(1L), lax_full_like(h, TRUE, dtype = "bool"), h, lax_full_like(h, 0), h)
  lax_while_loop(cond_fn, body_fn, state)[[5L]]
}

# Port of jax/_src/lax/special.py:regularized_incomplete_beta_impl.
sp_regularized_incomplete_beta_impl <- function(a, b, x, dtype) {
  shape <- sp_shape(a)
  eps <- finfo(dtype)$eps
  # Partial numerator of the continued fraction (http://dlmf.nist.gov/8.17.E23); the
  # numerator of the first iteration is one.
  nth_partial_betainc_numerator <- function(iteration, a, b, x) {
    iteration_bcast <- lax_broadcast_in_dim(iteration, shape, integer(0))
    iteration_is_even <- lax_eq(jnp_remainder(iteration_bcast, lax_full_like(iteration_bcast, 2)),
                                lax_full_like(iteration_bcast, 0))
    iteration_is_one <- lax_eq(iteration_bcast, lax_full_like(iteration_bcast, 1))
    iteration_minus_one <- lax_sub(iteration_bcast, lax_full_like(iteration_bcast, 1))
    m <- jnp_floor_divide(iteration_minus_one, lax_full_like(iteration_minus_one, 2))
    m_is_zero <- lax_eq(m, lax_full_like(m, 0))
    m <- lax_convert_element_type(m, dtype)
    one <- lax_full_like(a, 1)
    two <- lax_full_like(a, 2)
    # When a is close to zero and m == 0, zero_numerator avoids inaccuracies when FTZ or
    # DAZ is enabled.
    zero_numerator <- -(a + b) * x / (a + one)
    even_numerator <- lax_select(m_is_zero, zero_numerator,
                                 -(a + m) * (a + b + m) * x / ((a + two * m) * (a + two * m + one)))
    odd_numerator <- m * (b - m) * x / ((a + two * m - one) * (a + two * m))
    one_numerator <- lax_full_like(x, 1)
    numerator <- lax_select(iteration_is_even, even_numerator, odd_numerator)
    lax_select(iteration_is_one, one_numerator, numerator)
  }
  nth_partial_betainc_denominator <- function(iteration, a, b, x) {
    iteration_bcast <- lax_broadcast_in_dim(iteration, shape, integer(0))
    lax_select(lax_eq(iteration_bcast, lax_full_like(iteration_bcast, 0)),
               lax_full_like(x, 0), lax_full_like(x, 1))
  }
  zero <- function(v) lax_full_like(v, 0)
  a_is_zero <- lax_bitwise_or(lax_eq(a, zero(a)), lax_eq(b, lax_full_like(b, Inf)))
  b_is_zero <- lax_bitwise_or(lax_eq(b, zero(b)), lax_eq(a, lax_full_like(a, Inf)))
  x_is_zero <- lax_eq(x, zero(x))
  x_is_one <- lax_eq(x, lax_full_like(x, 1))
  x_is_not_zero <- lax_bitwise_not(x_is_zero)
  x_is_not_one <- lax_bitwise_not(x_is_one)
  is_nan <- lax_bitwise_or(lax_bitwise_or(sp_isnan(a), sp_isnan(b)), sp_isnan(x))

  result_is_zero <- lax_bitwise_or(lax_bitwise_and(b_is_zero, x_is_not_one),
                                   lax_bitwise_and(a_is_zero, x_is_zero))
  result_is_one <- lax_bitwise_or(lax_bitwise_and(a_is_zero, x_is_not_zero),
                                  lax_bitwise_and(b_is_zero, x_is_one))
  result_is_nan <- lax_bitwise_or(lax_bitwise_or(lax_bitwise_or(
    lax_lt(a, zero(a)), lax_lt(b, zero(b))), lax_lt(x, zero(x))), lax_gt(x, lax_full_like(x, 1)))
  result_is_nan <- lax_bitwise_or(result_is_nan,
                                  lax_bitwise_or(lax_bitwise_and(a_is_zero, b_is_zero), is_nan))

  # The continued fraction converges rapidly when x < (a + 1) / (a + b + 2)
  # (http://dlmf.nist.gov/8.17.E23); otherwise use the symmetry relation
  # (http://dlmf.nist.gov/8.17.E4).
  converges_rapidly <- lax_lt(x, (a + lax_full_like(a, 1)) / (a + b + lax_full_like(b, 2)))
  a_orig <- a
  a <- lax_select(converges_rapidly, a, b)
  b <- lax_select(converges_rapidly, b, a_orig)
  x <- lax_select(converges_rapidly, x, lax_sub(lax_full_like(x, 1), x))

  continued_fraction <- sp_lentz_thompson_barnett(
    num_iterations = if (dtype == "float32") 200L else 600L,
    small = eps / 2, threshold = eps / 2,
    nth_partial_numerator = nth_partial_betainc_numerator,
    nth_partial_denominator = nth_partial_betainc_denominator,
    inputs = list(a, b, x))

  # For very small a, and to avoid division by zero, use
  # a * gamma(a) = gamma(a + 1) -> 1 as a -> 0+.
  very_small <- finfo(dtype)$tiny * 2
  lbeta_ab_small_a <- lax_lgamma(b) - lax_lgamma(a + b)
  lbeta_ab <- lax_lgamma(a) + lbeta_ab_small_a
  factor <- lax_select(lax_lt(a, lax_full_like(a, very_small)),
                       lax_exp(lax_log1p(-x) * b - lbeta_ab_small_a),
                       lax_exp(lax_log(x) * a + lax_log1p(-x) * b - lbeta_ab) / a)
  result <- continued_fraction * factor
  result <- lax_select(converges_rapidly, result, lax_sub(lax_full_like(result, 1), result))
  result <- lax_select(result_is_zero, lax_full_like(a, 0), result)
  result <- lax_select(result_is_one, lax_full_like(a, 1), result)
  lax_select(result_is_nan, lax_full_like(a, NaN), result)
}

# Port of jax/_src/lax/special.py:_igamma_series. `mode` is one of "value", "derivative",
# "sample_derivative" (IgammaMode).
sp_igamma_series <- function(ax, x, a, enabled, dtype, mode) {
  eps <- finfo(dtype)$eps
  cond_fn <- function(vals) sp_any(vals[[1L]])
  body_fn <- function(vals) {
    enabled <- vals[[1L]]
    r <- vals[[2L]]
    c <- vals[[3L]]
    ans <- vals[[4L]]
    x <- vals[[5L]]
    dc_da <- vals[[6L]]
    dans_da <- vals[[7L]]
    r <- r + 1
    dc_da <- dc_da * (x / r) - (c * x) / (r * r)
    dans_da <- dans_da + dc_da
    c <- c * (x / r)
    ans <- ans + c
    conditional <- if (mode == "value") lax_bitwise_and(enabled, c / ans > eps)
                   else lax_bitwise_and(enabled, abs(dc_da / dans_da) > eps)
    list(conditional,
         lax_select(enabled, r, vals[[2L]]),
         lax_select(enabled, c, vals[[3L]]),
         lax_select(enabled, ans, vals[[4L]]),
         lax_select(enabled, x, vals[[5L]]),
         lax_select(enabled, dc_da, vals[[6L]]),
         lax_select(enabled, dans_da, vals[[7L]]))
  }
  init_vals <- list(enabled, a, lax_full_like(a, 1), lax_full_like(a, 1), x,
                    lax_full_like(a, 0), lax_full_like(a, 0))
  vals <- lax_while_loop(cond_fn, body_fn, init_vals)
  ans <- vals[[4L]]
  dans_da <- vals[[7L]]
  if (mode == "value") return((ans * ax) / a)
  dlogax_da <- lax_log(x) - lax_digamma(a + 1)
  if (mode == "derivative") return(ax * (ans * dlogax_da + dans_da) / a)
  -(dans_da + ans * dlogax_da) * x / a
}

# Port of jax/_src/lax/special.py:_igammac_continued_fraction.
sp_igammac_continued_fraction <- function(ax, x, a, enabled, dtype, mode) {
  eps <- finfo(dtype)$eps
  cond_fn <- function(vals) {
    c <- vals[[6L]]
    lax_bitwise_and(c < 2000, sp_any(vals[[1L]]))
  }
  body_fn <- function(vals) {
    enabled <- vals[[1L]]; ans <- vals[[2L]]; t <- vals[[3L]]; y <- vals[[4L]]; z <- vals[[5L]]
    c <- vals[[6L]]; pkm1 <- vals[[7L]]; qkm1 <- vals[[8L]]; pkm2 <- vals[[9L]]
    qkm2 <- vals[[10L]]; dpkm2_da <- vals[[11L]]; dqkm2_da <- vals[[12L]]
    dpkm1_da <- vals[[13L]]; dqkm1_da <- vals[[14L]]; dans_da <- vals[[15L]]

    c <- c + 1
    y <- y + 1
    z <- z + 2
    yc <- y * c
    pk <- pkm1 * z - pkm2 * yc
    qk <- qkm1 * z - qkm2 * yc
    qk_is_nonzero <- lax_ne(qk, sp_const(qk, 0))
    r <- pk / qk

    t <- lax_select(qk_is_nonzero, lax_abs(lax_div(lax_sub(ans, r), r)), lax_full_like(r, 1))
    ans <- lax_select(qk_is_nonzero, r, ans)

    dpk_da <- dpkm1_da * z - pkm1 - dpkm2_da * yc + pkm2 * c
    dqk_da <- dqkm1_da * z - qkm1 - dqkm2_da * yc + qkm2 * c
    dans_da_new <- lax_select(qk_is_nonzero, lax_div(dpk_da - ans * dqk_da, qk), dans_da)
    grad_conditional <- lax_select(qk_is_nonzero, lax_abs(dans_da_new - dans_da),
                                   lax_full_like(dans_da, 1))

    pkm2 <- pkm1
    pkm1 <- pk
    qkm2 <- qkm1
    qkm1 <- qk

    dpkm2_da <- dpkm1_da
    dqkm2_da <- dqkm1_da
    dpkm1_da <- dpk_da
    dqkm1_da <- dqk_da

    rescale <- lax_gt(lax_abs(pk), lax_integer_pow(sp_const(pk, eps), -1L))
    scaled <- function(v) lax_select(rescale, lax_mul(v, sp_const(v, eps)), v)
    pkm2 <- scaled(pkm2)
    pkm1 <- scaled(pkm1)
    qkm2 <- scaled(qkm2)
    qkm1 <- scaled(qkm1)

    dpkm2_da <- scaled(dpkm2_da)
    dqkm2_da <- scaled(dqkm2_da)
    dpkm1_da <- scaled(dpkm1_da)
    dqkm1_da <- scaled(dqkm1_da)

    conditional <- if (mode == "value") lax_bitwise_and(enabled, t > eps)
                   else lax_bitwise_and(enabled, grad_conditional > eps)

    list(conditional,
         lax_select(enabled, ans, vals[[2L]]),
         lax_select(enabled, t, vals[[3L]]),
         lax_select(enabled, y, vals[[4L]]),
         lax_select(enabled, z, vals[[5L]]),
         c,
         lax_select(enabled, pkm1, vals[[7L]]),
         lax_select(enabled, qkm1, vals[[8L]]),
         lax_select(enabled, pkm2, vals[[9L]]),
         lax_select(enabled, qkm2, vals[[10L]]),
         lax_select(enabled, dpkm2_da, vals[[11L]]),
         lax_select(enabled, dqkm2_da, vals[[12L]]),
         lax_select(enabled, dpkm1_da, vals[[13L]]),
         lax_select(enabled, dqkm1_da, vals[[14L]]),
         lax_select(enabled, dans_da_new, vals[[15L]]))
  }

  y <- 1 - a
  z <- x + y + 1
  c <- sp_const(x, 0)
  pkm2 <- lax_full_like(x, 1)
  qkm2 <- x
  pkm1 <- x + 1
  qkm1 <- z * x
  ans <- pkm1 / qkm1
  t <- lax_full_like(x, 1)
  dpkm2_da <- lax_full_like(x, 0)
  dqkm2_da <- lax_full_like(x, 0)
  dpkm1_da <- lax_full_like(x, 0)
  dqkm1_da <- -x
  dans_da <- (dpkm1_da - ans * dqkm1_da) / qkm1
  init_vals <- list(enabled, ans, t, y, z, c, pkm1, qkm1, pkm2, qkm2,
                    dpkm2_da, dqkm2_da, dpkm1_da, dqkm1_da, dans_da)

  vals <- lax_while_loop(cond_fn, body_fn, init_vals)
  ans <- vals[[2L]]
  if (mode == "value") return(ans * ax)
  dans_da <- vals[[15L]]
  dlogax_da <- lax_log(x) - lax_digamma(a)
  if (mode == "derivative") return(lax_mul(ax, lax_add(lax_mul(ans, dlogax_da), dans_da)))
  lax_neg(lax_add(dans_da, lax_mul(ans, dlogax_da)) * x)
}

sp_or_all <- function(...) Reduce(lax_bitwise_or, list(...))
sp_neg_log_max <- function(dtype) lax_neg(lax_log(as_jax(finfo(dtype)$max, dtype)))

# Port of jax/_src/lax/special.py:igamma_impl.
sp_igamma_impl <- function(a, x, dtype) {
  is_nan <- lax_bitwise_or(sp_isnan(a), sp_isnan(x))
  x_is_infinity <- lax_eq(x, sp_const(x, Inf))
  a_is_infinity <- lax_eq(a, sp_const(a, Inf))
  a_is_zero <- lax_eq(a, sp_const(a, 0))
  x_is_zero <- lax_eq(x, sp_const(x, 0))
  domain_error <- sp_or_all(lax_lt(x, sp_const(x, 0)), lax_lt(a, sp_const(a, 0)),
                            lax_bitwise_and(a_is_zero, x_is_zero),
                            lax_bitwise_and(a_is_infinity, x_is_infinity), is_nan)
  use_igammac <- lax_bitwise_and(lax_ge(x, sp_const(x, 1)), lax_gt(x, a))
  ax <- a * lax_log(x) - x - lax_lgamma(a)
  underflow <- lax_lt(ax, sp_neg_log_max(dtype))
  ax <- lax_exp(ax)
  enabled <- lax_bitwise_not(sp_or_all(x_is_zero, domain_error, underflow, x_is_infinity,
                                       a_is_infinity))
  output <- lax_select(
    use_igammac,
    1 - sp_igammac_continued_fraction(ax, x, a, lax_bitwise_and(enabled, use_igammac),
                                      dtype, "value"),
    sp_igamma_series(ax, x, a, lax_bitwise_and(enabled, lax_bitwise_not(use_igammac)),
                     dtype, "value"))
  output <- lax_select(a_is_infinity, lax_full_like(a, 0), output)
  output <- lax_select(x_is_zero, lax_full_like(a, 0), output)
  output <- lax_select(x_is_infinity, lax_full_like(a, 1), output)
  lax_select(domain_error, lax_full_like(a, NaN), output)
}

# Port of jax/_src/lax/special.py:igammac_impl.
sp_igammac_impl <- function(a, x, dtype) {
  is_nan <- lax_bitwise_or(sp_isnan(a), sp_isnan(x))
  a_is_zero <- lax_eq(a, sp_const(a, 0))
  x_is_zero <- lax_eq(x, sp_const(x, 0))
  x_is_infinity <- lax_eq(x, sp_const(x, Inf))
  a_is_infinity <- lax_eq(a, sp_const(a, Inf))
  domain_error <- sp_or_all(lax_lt(x, sp_const(x, 0)), lax_lt(a, sp_const(a, 0)),
                            lax_bitwise_and(a_is_zero, x_is_zero),
                            lax_bitwise_and(a_is_infinity, x_is_infinity), is_nan)
  use_igamma <- lax_bitwise_or(lax_lt(x, sp_const(x, 1)), lax_lt(x, a))
  ax <- a * lax_log(x) - x - lax_lgamma(a)
  underflow <- lax_lt(ax, sp_neg_log_max(dtype))
  enabled <- lax_bitwise_not(sp_or_all(domain_error, underflow, x_is_infinity, a_is_zero,
                                       a_is_infinity))
  ax <- lax_exp(ax)
  igamma_call <- sp_igamma_series(ax, x, a, lax_bitwise_and(enabled, use_igamma), dtype, "value")
  igammac_cf_call <- sp_igammac_continued_fraction(
    ax, x, a, lax_bitwise_and(enabled, lax_bitwise_not(use_igamma)), dtype, "value")
  output <- lax_select(use_igamma, 1 - igamma_call, igammac_cf_call)
  output <- lax_select(a_is_infinity, lax_full_like(a, 1), output)
  output <- lax_select(lax_bitwise_or(x_is_infinity, a_is_zero), lax_full_like(output, 0), output)
  lax_select(domain_error, lax_full_like(a, NaN), output)
}

# Port of jax/_src/lax/special.py:igamma_grad_a_impl and random_gamma_grad_impl, which
# differ only in the IgammaMode.
sp_igamma_derivative_impl <- function(mode) {
  function(a, x, dtype) {
    is_nan <- lax_bitwise_or(sp_isnan(a), sp_isnan(x))
    x_is_zero <- lax_eq(x, lax_full_like(x, 0))
    domain_error <- lax_bitwise_or(lax_lt(x, lax_full_like(x, 0)), lax_le(a, lax_full_like(a, 0)))
    use_igammac <- lax_bitwise_and(lax_gt(x, lax_full_like(x, 1)), lax_gt(x, a))
    ax <- a * lax_log(x) - x - lax_lgamma(a)
    underflow <- lax_lt(ax, sp_neg_log_max(dtype))
    ax <- lax_exp(ax)
    enabled <- lax_bitwise_not(sp_or_all(x_is_zero, domain_error, underflow, is_nan))
    output <- lax_select(
      use_igammac,
      -sp_igammac_continued_fraction(ax, x, a, lax_bitwise_and(enabled, use_igammac), dtype, mode),
      sp_igamma_series(ax, x, a, lax_bitwise_and(enabled, lax_bitwise_not(use_igammac)),
                       dtype, mode))
    output <- lax_select(x_is_zero, lax_full_like(output, 0), output)
    lax_select(lax_bitwise_or(domain_error, is_nan), lax_full_like(a, NaN), output)
  }
}
sp_igamma_grad_a_impl <- sp_igamma_derivative_impl("derivative")
sp_random_gamma_grad_impl <- sp_igamma_derivative_impl("sample_derivative")

# ---- Bessel functions ----------------------------------------------------------------------

# Port of jax/_src/lax/special.py:evaluate_chebyshev_polynomial.
sp_evaluate_chebyshev_polynomial <- function(x, coefficients) {
  b0 <- lax_full_like(x, 0)
  b1 <- lax_full_like(x, 0)
  b2 <- lax_full_like(x, 0)
  for (c in coefficients) {
    b2 <- b1
    b1 <- b0
    b0 <- x * b1 - b2 + lax_full_like(x, c)
  }
  0.5 * (b0 - b2)
}

# Cephes coefficients of i0e (jax/_src/lax/special.py:_i0e_impl32 and _i0e_impl64).
sp_i0e_coeffs <- list(
  a32 = c(-0x1.beaf68c0b30abp-27, 0x1.03b769d4d6435p-24,
          -0x1.1ec638f227f8dp-22, 0x1.2bf24978cf4acp-20,
          -0x1.2866fcba56427p-18, 0x1.13f58be9a2859p-16,
          -0x1.e2b2659c41d5ap-15, 0x1.8b51b74107cabp-13,
          -0x1.2e2fd1f15eb52p-11, 0x1.adc758a12100ep-10,
          -0x1.1b65e201aa849p-8, 0x1.59961f3dde3ddp-7,
          -0x1.84e9ef121b6fp-6, 0x1.93e8acea8a32dp-5,
          -0x1.84b70342d06eap-4, 0x1.5f7ac77ac88cp-3,
          -0x1.37febc057cd8dp-2, 0x1.5a84e9035a22ap-1),
  b32 = c(0x1.d2c64a9225b87p-29, 0x1.8569280d6d56dp-26,
          0x1.b8007d9cd616ep-23, 0x1.8412bc101c586p-19,
          0x1.20fa378999e52p-14, 0x1.b998ca2e59049p-9,
          0x1.9be62aca809cbp-1),
  a64 = c(-0x1.45cb72134d0efp-58, 0x1.33362977da589p-55,
          -0x1.184eb721ebbb4p-52, 0x1.ee6d893f65ebap-50,
          -0x1.a5022c297fbebp-47, 0x1.59b464b262627p-44,
          -0x1.1164c62ee1afp-41, 0x1.9fe2fe19bd324p-39,
          -0x1.2fc957a946abcp-36, 0x1.a98becc743c1p-34,
          -0x1.1d4fe13ae9556p-31, 0x1.6d903a454cb34p-29,
          -0x1.beaf68c0b30abp-27, 0x1.03b769d4d6435p-24,
          -0x1.1ec638f227f8dp-22, 0x1.2bf24978cf4acp-20,
          -0x1.2866fcba56427p-18, 0x1.13f58be9a2859p-16,
          -0x1.e2b2659c41d5ap-15, 0x1.8b51b74107cabp-13,
          -0x1.2e2fd1f15eb52p-11, 0x1.adc758a12100ep-10,
          -0x1.1b65e201aa849p-8, 0x1.59961f3dde3ddp-7,
          -0x1.84e9ef121b6fp-6, 0x1.93e8acea8a32dp-5,
          -0x1.84b70342d06eap-4, 0x1.5f7ac77ac88cp-3,
          -0x1.37febc057cd8dp-2, 0x1.5a84e9035a22ap-1),
  b64 = c(-0x1.0adb754ca8b19p-57, -0x1.646da6611913p-58,
          0x1.9be1812d98421p-55, 0x1.3f3dd076041cdp-55,
          -0x1.4600babd21fe4p-52, -0x1.8aee7d908de38p-52,
          0x1.fee7da3eafb1fp-50, 0x1.12a919094e6d7p-48,
          -0x1.583fe7e65629ap-47, -0x1.75d99cf68bb32p-45,
          0x1.156ff0d5fc545p-46, 0x1.b1c8c6b83c073p-42,
          0x1.94347fa268cecp-41, -0x1.f904303178d66p-40,
          -0x1.d0fd7357e7bf2p-37, -0x1.1511d08397425p-35,
          0x1.a24feabe8004fp-37, 0x1.0f9ccc0f46f75p-31,
          0x1.d2c64a9225b87p-29, 0x1.8569280d6d56dp-26,
          0x1.b8007d9cd616ep-23, 0x1.8412bc101c586p-19,
          0x1.20fa378999e52p-14, 0x1.b998ca2e59049p-9,
          0x1.9be62aca809cbp-1))

# Port of jax/_src/lax/special.py:_i0e_impl32, _i0e_impl64 and bessel_i0e_impl (Cephes i0e).
sp_bessel_i0e_impl <- function(x) {
  x_dtype <- sp_dtype(x)
  if (!x_dtype %in% c("float32", "float64")) {
    # f16 is upcast: the Cephes coefficients do not have enough precision for it.
    return(lax_convert_element_type(sp_bessel_i0e_impl(lax_convert_element_type(x, "float32")),
                                    x_dtype))
  }
  wide <- x_dtype == "float64"
  x <- lax_abs(x)
  half <- lax_full_like(x, 0.5)
  two <- lax_full_like(x, 2)
  thirty_two <- lax_full_like(x, 32)
  result_le_8 <- sp_evaluate_chebyshev_polynomial(
    half * x - two, if (wide) sp_i0e_coeffs$a64 else sp_i0e_coeffs$a32)
  result_gt_8 <- lax_div(sp_evaluate_chebyshev_polynomial(
    thirty_two / x - two, if (wide) sp_i0e_coeffs$b64 else sp_i0e_coeffs$b32), lax_sqrt(x))
  lax_select(x <= 8, result_le_8, result_gt_8)
}

# ---- coefficient tables ------------------------------------------------------------------------

# Coefficient tables of jax/_src/lax/erf.py (Sollya fpminimax fits; lowest degree first),
# generated from the JAX 0.12.0.1 sources as exact hexadecimal literals.
sp_erf_tables <- list(
  ERF32_T_M1 = c(-0x1.4226160000000p-3, 0x1.b5daf40000000p-3, 0x1.cf71140000000p-5, 0x1.9ae15a0000000p-7,
               0x1.35b6620000000p-9, 0x1.5853f20000000p-12, 0x1.4b5ec60000000p-14),
  ERF64_T_M1 = c(-0x1.4226162fbddd5p-3, 0x1.b5db0451254fap-3, 0x1.cf6d2bc57822cp-5, 0x1.9b3c1054ff0dfp-7,
               0x1.31cd0ceac3b4dp-9, 0x1.84227cc6c435fp-12, 0x1.ac2510dffaf4dp-15,
               0x1.a0df95c08a7d3p-18, 0x1.6af2214c994d3p-21, 0x1.1d3815deb5114p-24,
               0x1.9f6d498ffeab3p-28, 0x1.dee5605ee0a79p-32, 0x1.f987104017746p-35),
  ERFC32_G_0_1 = c(-0x1.067f260000000p-1, 0x1.6ff85e0000000p-1, -0x1.5502060000000p-1, 0x1.fc9b520000000p-2,
                 -0x1.47907a0000000p-2, 0x1.7976dc0000000p-3, -0x1.8c2f180000000p-4,
                 0x1.87796c0000000p-5, -0x1.8d572a0000000p-6, 0x1.3d5cd40000000p-7),
  ERFC32_G_1_2 = c(0x1.5d8fcc0000000p-2, -0x1.3c8f3a0000000p-2, 0x1.308bc60000000p-3, 0x1.40234e0000000p-7,
                 -0x1.c6c0c80000000p-4, 0x1.28d0aa0000000p-3, -0x1.fb57040000000p-4,
                 0x1.07c7940000000p-4),
  ERFC32_G_2_8 = c(0x1.fcc3220000000p-2, -0x1.6c3cca0000000p-2, -0x1.8a59f40000000p-3, 0x1.67556e0000000p-1,
                 -0x1.5363380000000p-1, -0x1.be92bc0000000p-3, 0x1.a70c6e0000000p+0,
                 -0x1.0e28400000000p+1),
  ERFC32_G_8_INF = c(0x1.20dd740000000p-1, -0x1.b147860000000p-1, 0x1.0d8cd00000000p+1, -0x1.9bd5960000000p+2),
  ERFC64_G_0_1 = c(-0x1.067f263ec85e7p-1, 0x1.6ff861544dbffp-1, -0x1.55021bd369e1dp-1, 0x1.fc9a0570ff834p-2,
                 -0x1.4786f912f56aep-2, 0x1.79973b6973c28p-3, -0x1.8e2e1451bc1b5p-4,
                 0x1.85b049641853ap-5, -0x1.65a4048dd402ep-6, 0x1.363682a7210e9p-7,
                 -0x1.ffb88ced16238p-9, 0x1.934c7b5caf5ecp-10, -0x1.30ef3121b1896p-11,
                 0x1.bbfeb6475aa4cp-13, -0x1.382eba5243836p-14, 0x1.a581276bdfc2ap-16,
                 -0x1.1299bedad2447p-17, 0x1.9078339c927cep-19, -0x1.0f0a671941893p-20),
  ERFC64_G_12_INF = c(0x1.20dd750429b6dp-1, -0x1.b14c2f863e731p-1, 0x1.0ecf9db3a64c3p+1, -0x1.d9eb537ff0ebfp+2,
                    0x1.0a943fa7b85f5p+5, -0x1.6e827a9457943p+7, 0x1.28f33e00c3a43p+10,
                    -0x1.0b1a528ddb599p+13, 0x1.94b1159b4e5f9p+15),
  ERFC64_G_1_2 = c(0x1.5d8fcc4ff299fp-2, -0x1.3c8f3c0e9713fp-2, 0x1.308b5d651d692p-3, 0x1.40539e936fdf9p-7,
                 -0x1.c6a3b77e2cbdcp-4, 0x1.28273e28c6857p-3, -0x1.fb639fa6102fap-4,
                 0x1.23e180fbac86ep-4, -0x1.333520f71b863p-7, -0x1.68564659148d6p-5,
                 0x1.3f0ccd3895528p-4, -0x1.69af0c554c522p-4, 0x1.3a785279c5119p-4,
                 -0x1.94c906e0eee1bp-5, 0x1.0d917547d50b9p-6, 0x1.4af285199dcf0p-6,
                 -0x1.2cdbbf2120b51p-4, 0x1.61423e2bd69edp-4),
  ERFC64_G_2_4 = c(0x1.e564643a22ec1p-2, -0x1.7d2245a3fab2dp-2, -0x1.3a22e2716d9b4p-4, 0x1.11bd8bc5ed735p-1,
                 -0x1.4dfa3c7c26ea4p-1, 0x1.b7b2f445357eep-3, 0x1.67b14c9ce96fap-1,
                 -0x1.a31adbd26f844p+0, 0x1.ccbc138251430p+0, -0x1.83c9b3a976bcbp-2,
                 -0x1.71000220096a0p+1, 0x1.bdbeaa0846d38p+2, -0x1.24148b88fc5a6p+3,
                 0x1.33e7aee6b38dbp+2, 0x1.b80e9a3646cb3p+3, -0x1.34f75fb625e77p+5),
  ERFC64_G_4_12 = c(0x1.20dd7504362c6p-1, 0x1.c76beba6f246cp-28, -0x1.b14c47c4fdd1fp-1, 0x1.2769f70118d67p-15,
                  0x1.0ead78b59c354p+1, 0x1.5842cd55f8dc8p-6, -0x1.ed7ea8a970b71p+2,
                  0x1.a8269ce322d02p+1, 0x1.8f40d19cd55a0p+2, 0x1.4dcb55c4285fbp+7,
                  -0x1.d85fd73493477p+9, 0x1.3224fc40f7246p+11, -0x1.cabf0ba59224fp+11,
                  0x1.85ccc633f9b37p+11, -0x1.26fdae0850b99p+10),
  ERFINV32_P1 = c(0x1.508eb40000000p+0, 0x1.006e840000000p-2, -0x1.44d9560000000p-11, -0x1.f2946e0000000p-10,
                0x1.a737f80000000p-13, 0x1.026b3a0000000p-16, -0x1.64551e0000000p-18,
                0x1.5cd3320000000p-24, 0x1.9385580000000p-24),
  ERFINV32_P2 = c(-0x1.566be20000000p-3, 0x1.5cfcea0000000p-9, 0x1.c065740000000p-8, -0x1.c3ce240000000p-9,
                0x1.2db6820000000p-10, -0x1.c6429e0000000p-13, -0x1.b5b0460000000p-15,
                0x1.c345a00000000p-15, -0x1.9174540000000p-17),
  ERFINV32_P3 = c(-0x1.0c3f680000000p-3, 0x1.19ac700000000p-7, -0x1.0cd7bc0000000p-11, 0x1.7f59740000000p-18,
                0x1.9bff7c0000000p-18, -0x1.7464b00000000p-20, 0x1.3446760000000p-22,
                -0x1.7798f20000000p-24, 0x1.b7293c0000000p-27),
  ERFINV64_P1 = c(0x1.7083a336cd764p+0, 0x1.fce34214befaep-3, -0x1.9d94b8d1cc026p-9, -0x1.821fe73cf0223p-10,
                0x1.cf8b8bcbdb1f1p-13, 0x1.8d38bc4f859a4p-21, -0x1.1aedd3b03ed11p-18,
                0x1.c9571aa3cf69ap-22, 0x1.57ea41d40b6f7p-25, -0x1.b146771d77374p-27,
                0x1.ecf346d136385p-32, 0x1.d672a920915cfp-33, -0x1.0c6f0e94ac237p-35,
                -0x1.9087b763adfbfp-40, 0x1.b747c74f0b4fcp-41, -0x1.9feaa0741881cp-45,
                -0x1.b73d097d71e3dp-47, 0x1.1d26f17f2447ap-49, 0x1.b058ee070e9a4p-54,
                -0x1.2ffccb301a77fp-55),
  ERFINV64_P2 = c(-0x1.566be9baa624ep-3, 0x1.5cf6bc26d4b0dp-9, 0x1.c0820afc709fep-8, -0x1.c38c06b7f2ac1p-9,
                0x1.2b7b99b463d5ep-10, -0x1.cfecefc93c5cfp-13, -0x1.32cc7e7e51e7bp-15,
                0x1.d490539604ecbp-15, -0x1.9420cb7edb35ap-16, 0x1.5677bcc9afc2ap-19,
                0x1.a67db4feb28e9p-19, -0x1.0e35ac15520b9p-19, 0x1.9883ecf6aa351p-22,
                0x1.9d40c1d9402a5p-23, -0x1.57e6b5295de64p-23, 0x1.315f1ab1eab22p-25,
                0x1.ffd498b130146p-27, -0x1.6ad67e67ce1c9p-27, 0x1.56ca0f8fc7bebp-31,
                0x1.5f3c32a44f21dp-31),
  ERFINV64_P3 = c(-0x1.336445cb78457p-3, 0x1.5186909252081p-7, -0x1.22eb371abdb35p-13, -0x1.c2f0c4ba317f3p-13,
                0x1.3eb336279e7cbp-14, -0x1.49de79761feacp-16, 0x1.2dcd131fc6114p-18,
                -0x1.014c93b9c770ap-20, 0x1.a10fbfbd8714bp-23, -0x1.426c13d3d9282p-25,
                0x1.d1064f3f365e7p-28, -0x1.176ce2d17ac0dp-30, 0x1.9ca8e696af3dap-35,
                0x1.0b05814447eebp-34, -0x1.5e5575ec8c5f6p-35, 0x1.3d5558cee1095p-36,
                -0x1.f4afbb5188accp-38, 0x1.068d714fd68e8p-39, 0x1.64aead41961a4p-47,
                -0x1.db803224318bap-44),
  ERFINV64_P4 = c(-0x1.da072a61ba4eap-4, 0x1.b8e2742818d39p-8, -0x1.accd56c8ea75fp-12, 0x1.58dc927ba7941p-16,
                -0x1.fc7f589c31317p-23, -0x1.461c7a5711770p-23, 0x1.1a167f5d1e39ep-25,
                -0x1.5e5f5e288cc6cp-28, 0x1.7fef58ca557e1p-31, -0x1.89982c2ca9eeep-34,
                0x1.82f614b7d99ebp-37, -0x1.7166cfcfa4668p-40, 0x1.583f8c4c2d1d8p-43,
                -0x1.3b923c3a99e19p-46, 0x1.23838035a7f2ap-49, -0x1.04939cf5158e2p-52,
                0x1.7b33c11940dc3p-56, -0x1.473c3d074aaf5p-59, 0x1.16a8e1e7620a6p-61,
                -0x1.dd5eb07dcdc59p-65),
  ERFINV64_P5 = c(-0x1.2b971f40e5e38p-4, 0x1.3fdeef10d1f97p-9, -0x1.8070a7ed3f146p-14, 0x1.dd995aa777e37p-19,
                -0x1.29c10b551c6b3p-23, 0x1.6d4bc74ea8be4p-28, -0x1.af290403d711cp-33,
                0x1.d76e99f341a57p-38, -0x1.b3d937cccea3dp-43, 0x1.c1c18da7dae7ep-49,
                0x1.62b6d3c46a513p-53, -0x1.9cc37db7a34aap-56, 0x1.0275a925b8617p-59,
                -0x1.1afa28919d34bp-63, 0x1.5cd5a45f9f345p-67, -0x1.532d6710bbc58p-71,
                0x1.4964074ec7ae3p-77, -0x1.0f4635900acefp-81, 0x1.0dcd16a13181bp-82,
                -0x1.dfdfd0a3f5a5cp-87),
  NDTRI32_P1 = c(0x1.dbf6ce0000000p+0, 0x1.6aa6300000000p-2, -0x1.cb58ea0000000p-11, -0x1.608b440000000p-9,
               0x1.2b210e0000000p-12, 0x1.6d16940000000p-16, -0x1.f460ec0000000p-18,
               0x1.fd77900000000p-24, 0x1.0d3e140000000p-23),
  NDTRI32_P2 = c(-0x1.e354320000000p-3, 0x1.00abfa0000000p-8, 0x1.3d10880000000p-7, -0x1.3f797e0000000p-8,
               0x1.aaacbe0000000p-10, -0x1.4136240000000p-12, -0x1.354cac0000000p-14,
               0x1.3f1a2a0000000p-14, -0x1.1c21520000000p-16),
  NDTRI32_P3 = c(-0x1.7931a00000000p-3, 0x1.934bc80000000p-7, -0x1.7c39960000000p-11, 0x1.0fafbe0000000p-17,
               0x1.244c8c0000000p-17, -0x1.07e8100000000p-19, 0x1.ad9bd80000000p-22,
               -0x1.08fd020000000p-23, 0x1.3cee080000000p-26),
  NDTRI64_P1 = c(0x1.0494328c11f68p+1, 0x1.67d684b8bd0d5p-2, -0x1.247225e75ba10p-8, -0x1.110805d0604b2p-9,
               0x1.47c6a064c0769p-12, 0x1.18e0cb604677ep-20, -0x1.901f3e49ecb21p-18,
               0x1.43636db9d70a2p-21, 0x1.e65e87a338d15p-25, -0x1.325f34412e34ep-26,
               0x1.5c925408877cap-31, 0x1.4ca8526267166p-32, -0x1.7ba3536bbbec5p-35,
               -0x1.1b3d75563af1ap-39, 0x1.36b513a3c3d5bp-40, -0x1.25f97285d3bc1p-44,
               -0x1.3728996b01934p-46, 0x1.92e5e5fa2cb94p-49, 0x1.34c83cb67b8a6p-53,
               -0x1.acf9c4a320a36p-55),
  NDTRI64_P2 = c(-0x1.e441d8be4c3adp-3, 0x1.ed8234b66041ap-9, 0x1.3d249de327a56p-7, -0x1.3f4abbe9c170dp-8,
               0x1.a7885c3c296ffp-10, -0x1.480b7df92653ep-12, -0x1.b1e108357f884p-15,
               0x1.4b531a7fc52ffp-14, -0x1.1dc3003028c2fp-15, 0x1.e4528e6b26637p-19,
               0x1.2abf090c0722bp-18, -0x1.7e2254f12d186p-19, 0x1.20dd5ed0a343cp-21,
               0x1.24368bd08457fp-22, -0x1.e65a6ebcfc888p-23, 0x1.afde942fab73cp-25,
               0x1.69efbb197c7aep-26, -0x1.0093c0225e999p-26, 0x1.e4a0876288ad6p-31,
               0x1.f0d47baf6bcb8p-31),
  NDTRI64_P3 = c(-0x1.b2b7b1a2ff71cp-3, 0x1.dd5549ebcfec0p-7, -0x1.9b6bdc0636ce0p-13, -0x1.3edcf340d0579p-12,
               0x1.c2b5bdb78bfcep-14, -0x1.d28159778d37ap-16, 0x1.aacfa8c92a2f4p-18,
               -0x1.6be03bf46678dp-20, 0x1.26e8395f1f627p-22, -0x1.c7f968bb2147fp-25,
               0x1.48d07eb481d54p-27, -0x1.8b2187adf74bfp-30, 0x1.254242c3809eep-34,
               0x1.78b8f4cdb8771p-34, -0x1.f20e8edb60dc0p-35, 0x1.c421e187080b9p-36,
               -0x1.5cf04feb50101p-37, 0x1.65ad647afa4f8p-39, -0x1.1e0712564e325p-46,
               -0x1.21aaa0ee852d4p-43),
  NDTRI64_P4 = c(-0x1.4f30422c407cfp-3, 0x1.37c0967c8044fp-7, -0x1.2f3578ef61b8dp-11, 0x1.e7b53d46362c8p-16,
               -0x1.678fdeb8075acp-22, -0x1.cd30e1818fe2dp-23, 0x1.8eeeb8c790b52p-25,
               -0x1.ef8067d440c86p-28, 0x1.0f7ba63868244p-30, -0x1.165033b6c620dp-33,
               0x1.119f78d647a46p-36, -0x1.0534d25c884dap-39, 0x1.e6d71c7211099p-43,
               -0x1.be493db6c5e64p-46, 0x1.9c43a232d9c07p-49, -0x1.70813fb19231ap-52,
               0x1.0c2174a0c8ce1p-55, -0x1.ced2868e22e89p-59, 0x1.8a17cd7e96200p-61,
               -0x1.518570c182bf7p-64),
  NDTRI64_P5 = c(-0x1.a7af65743d660p-4, 0x1.c45d6cafd81f2p-9, -0x1.0fd715b65ab27p-13, 0x1.51b6a937534b5p-18,
               -0x1.a5167dcb6d500p-23, 0x1.024da42857645p-27, -0x1.30e0565828cb7p-32,
               0x1.4d5a4d0326a65p-37, -0x1.3430e913d10b6p-42, 0x1.3e08cf0897759p-48,
               0x1.f599e21836bc3p-53, -0x1.23ec07f0c81c5p-55, 0x1.6d98804f9ac97p-59,
               -0x1.8fcbc70762358p-63, 0x1.ecf92b25a9dabp-67, -0x1.e15c809d29bf6p-71,
               0x1.d53b0574b4157p-77, -0x1.6fe0b0aeaa851p-81, 0x1.7d21d290ddc3fp-82,
               -0x1.553dbd7f1f0afp-86))

# Coefficient tables of jax/_src/lax/lgamma.py, generated the same way.
sp_lgamma_tables <- list(
  LOG_Q_HI = c(0x1.5555560000000p-2, -0x1.fffff20000000p-3, 0x1.9999900000000p-3, -0x1.555c5c0000000p-3,
             0x1.249b480000000p-3, -0x1.fe01ac0000000p-4, 0x1.c375300000000p-4,
             -0x1.b33c1a0000000p-4, 0x1.b2f1360000000p-4, -0x1.0448480000000p-4),
  POS_HI_05_15 = c(0x1.2788d00000000p-1, -0x1.f646520000000p-3, 0x1.3e53d60000000p-3, -0x1.d7aaaa0000000p-4,
                 0x1.79f6720000000p-4, -0x1.3bdb700000000p-4, 0x1.0eabb80000000p-4,
                 -0x1.f7b9500000000p-5, 0x1.dddba80000000p-5, -0x1.f647f00000000p-6,
                 0x1.127e1e0000000p-6, -0x1.62e36c0000000p-4, 0x1.837a100000000p-4),
  POS_HI_15_4 = c(0x1.72c20a0000000p-2, -0x1.0d3a360000000p-4, 0x1.06a6440000000p-6, -0x1.21a4520000000p-8,
                0x1.55c08c0000000p-10, -0x1.a385cc0000000p-12, 0x1.05e3520000000p-13,
                -0x1.5f79c00000000p-15, 0x1.04bf3a0000000p-16, -0x1.ce70fe0000000p-19,
                0x1.ccff740000000p-23, -0x1.feb34e0000000p-21, 0x1.1177240000000p-21),
  STIRLING_HI = c(0x1.5555560000000p-4, -0x1.6c16c20000000p-9, 0x1.a01a020000000p-11, -0x1.3813820000000p-11,
                0x1.b951e20000000p-11),
  SINC_HI = c(-0x1.a51a660000000p+0, -0x1.1513060000000p-1, -0x1.5b4f0a0000000p-2, -0x1.ff7a740000000p-3,
            -0x1.b6a08e0000000p-3, -0x1.72eace0000000p-4, -0x1.5491500000000p-2),
  NEG_TABLE = list(
    c(-0x1.d3fe4c0000000p-2, 0x1.0000000000000p+0, -0x1.260dbc0000000p-3, 0x1.025f7a0000000p-2,
      -0x1.4273c20000000p-5, 0x1.6e3aae0000000p-5, -0x1.0d4afe0000000p-7, 0x1.1510b20000000p-7,
      -0x1.6b258a0000000p-10, 0x1.6d0a6e0000000p-10, -0x1.9fef700000000p-13,
      0x1.a044a40000000p-13, -0x1.a0145a0000000p-16, 0x1.a01faa0000000p-16,
      -0x1.71dda40000000p-19, 0x1.71ded00000000p-19),
    c(0x1.ff07940000000p-27, 0x0.0p+0, -0x1.3cb35e0000000p-28, 0x1.e427000000000p-27,
      -0x1.9958080000000p-30, 0x1.e80d480000000p-34, -0x1.6db2f40000000p-35,
      0x1.15037c0000000p-34, 0x1.06e6f80000000p-35, 0x1.7e50c00000000p-39, 0x1.e17f4e0000000p-42,
      -0x1.bbe00a0000000p-40, 0x1.a1fb900000000p-47, -0x1.cf752a0000000p-42,
      0x1.214d2c0000000p-47, 0x1.7c026a0000000p-44),
    c(0x1.051eb80000000p-2, -0x1.0000000000000p+0, 0x1.5c28f60000000p-2, 0x1.7ae1480000000p-2,
      0x1.ae147a0000000p-2, 0x1.ae147a0000000p-2, 0x1.ae147a0000000p-2, 0x1.ae147a0000000p-2,
      0x1.ae147a0000000p-2, 0x1.ae147a0000000p-2, 0x1.ae147a0000000p-2, 0x1.ae147a0000000p-2,
      0x1.ae147a0000000p-2, 0x1.ae147a0000000p-2, 0x1.ae147a0000000p-2, 0x1.ae147a0000000p-2),
    c(-0x1.62a56a0000000p+0, 0x0.0p+0, -0x1.1e09560000000p+1, -0x1.ee9dc80000000p-1,
      -0x1.0df5320000000p+1, -0x1.da61f80000000p+0, -0x1.0389220000000p+1, -0x1.f8800c0000000p+0,
      -0x1.00a9a20000000p+1, -0x1.fea9620000000p+0, -0x1.001a300000000p+1, -0x1.ffcb8c0000000p+0,
      -0x1.00037a0000000p+1, -0x1.fff90a0000000p+0, -0x1.0000680000000p+1, -0x1.ffff300000000p+0),
    c(0x1.563c7e0000000p+1, 0x0.0p+0, -0x1.ac45060000000p-4, 0x1.78eada0000000p+0,
      -0x1.978cce0000000p-4, 0x1.46261a0000000p-3, -0x1.bdaf660000000p-6, 0x1.e745ca0000000p-6,
      -0x1.5278760000000p-8, 0x1.576a920000000p-8, -0x1.a2edb80000000p-11, 0x1.a3c9000000000p-11,
      -0x1.bd50880000000p-14, 0x1.bd6d280000000p-14, -0x1.a1ea720000000p-17,
      0x1.a19bf60000000p-17),
    c(0x1.69eae60000000p+2, 0x0.0p+0, -0x1.4414a80000000p-1, 0x1.540b040000000p+0,
      -0x1.835dfa0000000p-1, -0x1.faeb780000000p-2, -0x1.630c020000000p-1, -0x1.45e1180000000p-1,
      -0x1.57f8e60000000p-1, -0x1.52a51a0000000p-1, -0x1.55be3a0000000p-1, -0x1.54ec8a0000000p-1,
      -0x1.5563700000000p-1, -0x1.55479a0000000p-1, -0x1.5557200000000p-1, -0x1.5553e00000000p-1),
    c(0x1.b4c28a0000000p+3, 0x0.0p+0, 0x1.798d2a0000000p-3, 0x1.54e8620000000p+1,
      -0x1.48b8be0000000p-4, 0x1.78cfb00000000p-3, -0x1.afd00a0000000p-6, 0x1.f5a8d00000000p-6,
      -0x1.50cf980000000p-8, 0x1.58ec280000000p-8, -0x1.a292100000000p-11, 0x1.a3f3660000000p-11,
      -0x1.bce8680000000p-14, 0x1.bd1dac0000000p-14, -0x1.6abd600000000p-17,
      0x1.778ed60000000p-17),
    c(0x1.d0e58e0000000p+4, 0x0.0p+0, -0x1.85bf900000000p-5, 0x1.9588840000000p+1,
      -0x1.e1b2f60000000p-2, -0x1.a0c9280000000p-3, -0x1.b407ca0000000p-2, -0x1.79aed60000000p-2,
      -0x1.9ebe9c0000000p-2, -0x1.9418bc0000000p-2, -0x1.9a514a0000000p-2, -0x1.98ae360000000p-2,
      -0x1.999c240000000p-2, -0x1.9964960000000p-2, -0x1.99856a0000000p-2, -0x1.997e940000000p-2),
    c(0x1.02c5500000000p+6, 0x0.0p+0, 0x1.1b65240000000p-1, 0x1.35078a0000000p+2,
      -0x1.efbe880000000p-5, 0x1.b3ca4c0000000p-3, -0x1.a58bb20000000p-6, 0x1.0517de0000000p-5,
      -0x1.520eee0000000p-8, 0x1.5d8d460000000p-8, -0x1.a5c0980000000p-11, 0x1.a7a1f00000000p-11,
      -0x1.c4bdf40000000p-14, 0x1.c478e40000000p-14, -0x1.28da160000000p-15,
      0x1.ef93480000000p-16),
    c(0x1.208c800000000p+7, 0x0.0p+0, 0x1.0187e40000000p-1, 0x1.8e69520000000p+2,
      -0x1.5aac820000000p-2, -0x1.e31a240000000p-5, -0x1.415d7c0000000p-2, -0x1.0603520000000p-2,
      -0x1.2c57e40000000p-2, -0x1.21856c0000000p-2, -0x1.27c90c0000000p-2, -0x1.261ef80000000p-2,
      -0x1.270d6a0000000p-2, -0x1.26d3ce0000000p-2, -0x1.26dcda0000000p-2, -0x1.26da880000000p-2),
    c(0x1.622bd20000000p+8, 0x0.0p+0, 0x1.bf8ca00000000p-1, 0x1.26a72c0000000p+3,
      -0x1.58f7f80000000p-5, 0x1.8ee3840000000p-3, -0x1.67bbc40000000p-6, 0x1.ca188a0000000p-6,
      -0x1.28bf1c0000000p-8, 0x1.348dc20000000p-8, -0x1.74735a0000000p-11, 0x1.76cd1a0000000p-11,
      -0x1.7143e20000000p-14, 0x1.75d81e0000000p-14, 0x1.1574960000000p-13,
      -0x1.88597a0000000p-14),
    c(0x1.e642c00000000p+9, 0x0.0p+0, 0x1.a875780000000p-1, 0x1.f414380000000p+3,
      -0x1.d7c6380000000p-3, -0x1.8587e20000000p-7, -0x1.ba279a0000000p-3, -0x1.58bcd80000000p-3,
      -0x1.9c7d980000000p-3, -0x1.8ab67a0000000p-3, -0x1.9653b40000000p-3, -0x1.939a120000000p-3,
      -0x1.955b080000000p-3, -0x1.950a180000000p-3, -0x1.9631f80000000p-3, -0x1.95fb360000000p-3),
    c(0x1.07fd3c0000000p+11, 0x0.0p+0, 0x1.7bd7340000000p+1, 0x1.8f55820000000p+4,
      -0x1.a01c440000000p-7, 0x1.e22b180000000p-2, -0x1.3bcda00000000p-5, 0x1.ceba840000000p-5,
      -0x1.0d023a0000000p-7, 0x1.1ce97a0000000p-7, -0x1.5184400000000p-10, 0x1.53b0a00000000p-10,
      -0x1.89f6f80000000p-13, 0x1.83fe260000000p-13, -0x1.6aaa200000000p-12,
      0x1.097aba0000000p-12),
    c(0x1.0ab50c0000000p+11, 0x0.0p+0, 0x1.f037a80000000p+1, 0x1.2b70300000000p+4,
      -0x1.28c8ea0000000p-2, 0x1.fbc9b40000000p-3, -0x1.5c5cca0000000p-2, -0x1.ea300e0000000p-3,
      -0x1.37a7840000000p-2, -0x1.24eb4e0000000p-2, -0x1.2e89e80000000p-2, -0x1.2ba62c0000000p-2,
      -0x1.2d05ec0000000p-2, -0x1.2c97e20000000p-2, -0x1.2be3760000000p-2, -0x1.2bfd7e0000000p-2)))

# ---- erf.py: polynomial approximations for erfc, erfcx, erfinv and erfcinv ------------------

# Port of jax/_src/lax/erf.py:_half_mantissa: x with the low half of its mantissa cleared
# (12 significant bits remain in float32, 26 in float64), so that its square is exact.
sp_half_mantissa <- function(x) {
  dt <- sp_dtype(x)
  if (dt == "float32") {
    x_u <- lax_bitcast_convert_type(x, "uint32")
    return(lax_bitcast_convert_type(lax_bitwise_and(x_u, sp_const(x_u, 0xFFFFF000)), "float32"))
  }
  if (dt == "float64") {
    x_u <- lax_bitcast_convert_type(x, "uint64")
    return(lax_bitcast_convert_type(lax_bitwise_and(x_u, sp_const(x_u, 0xFFFFFFFFF8000000)),
                                    "float64"))
  }
  jax_error("Unsupported dtype for _half_mantissa: ", dt, class = "jax_type_error")
}

# Port of jax/_src/lax/erf.py:_split_square: x^2 = s_hi + s_lo with s_hi = x_hi^2 exact, so
# exp(+-x^2) = exp(+-s_hi) exp(+-s_lo) does not magnify the rounding error of x^2.
sp_split_square <- function(x) {
  x_hi <- sp_half_mantissa(x)
  x_lo <- x - x_hi
  list(x_hi * x_hi, x_lo * (x + x_hi))
}

# Port of jax/_src/lax/erf.py:_expm1_lo: exp(u) - 1 for the small remainder u = +-s_lo.
sp_expm1_lo <- function(u) {
  if (sp_dtype(u) == "float32") return(u * sp_polynomial(u, list(1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0)))
  u * sp_polynomial(u, list(1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0))
}

sp_erf_max_ax <- function(x) if (sp_dtype(x) == "float32") 11.0 else 28.0

# Port of jax/_src/lax/erf.py:exp_neg_sq and _exp_neg_sq_jvp: exp(-x^2) through
# _split_square for float32 and float64.
sp_exp_neg_sq <- custom_jvp(function(x) {
  if (!sp_dtype(x) %in% c("float32", "float64")) return(lax_exp(lax_neg(lax_square(x))))
  ax_exp <- lax_min(abs(x), sp_const(x, sp_erf_max_ax(x)))
  ss <- sp_split_square(ax_exp)
  e <- lax_exp(-ss[[1L]])
  e + e * sp_expm1_lo(-ss[[2L]])
}, function(primals, tangents) {
  x <- as_jax(primals[[1L]])
  ans <- sp_exp_neg_sq(x)
  # For |x| >= max_ax, ans underflows to 0; at x = +-inf, -2 x ans would be inf * 0 = NaN
  # although the true derivative is 0. Replacing x with 0 there cannot change a finite result.
  safe_x <- lax_select(abs(x) < sp_const(x, sp_erf_max_ax(x)), x, lax_full_like(x, 0.0))
  list(ans, -sp_const(x, 2.0) * safe_x * ans * tangents[[1L]])
})

sp_erf_table <- function(x, name32, name64 = sub("32", "64", name32, fixed = TRUE)) {
  dt <- sp_dtype(x)
  if (dt == "float32") return(sp_erf_tables[[name32]])
  if (dt == "float64") return(sp_erf_tables[[name64]])
  jax_error("Unsupported dtype: ", dt, class = "jax_type_error")
}
sp_poly_full <- function(x, coeffs) sp_polynomial(x, as.list(coeffs), unroll = length(coeffs))

# Port of jax/_src/lax/erf.py:_erfc_small: erfc on |x| <= 1 from R(1 - w) - 1 =
# erf(x) / x - 1 expanded in w = 1 - x^2.
sp_erfc_small <- function(x) {
  d <- 1.0 - x
  w <- d * (1.0 + x)
  r_m1 <- sp_poly_full(w, sp_erf_table(x, "ERF32_T_M1"))
  lax_select(x < 0.25, 1.0 - x * (1.0 + r_m1), d - x * r_m1)
}

# Port of jax/_src/lax/erf.py:_erfcx_from_grad_large: erfcx(ax) = (1 / sqrt(pi) +
# 0.5 erfcx'(ax)) / ax, with 1 / sqrt(pi) split in two parts in float32.
sp_erfcx_from_grad_large <- function(ax, grad_large) {
  if (sp_dtype(ax) == "float32") {
    num <- as_jax(0x1.20dd76p-1, "float32") +
      lax_optimization_barrier(as_jax(-0x1.f7ac92p-26, "float32") + 0.5 * grad_large)
    scale <- lax_select(ax > 2.0^64, lax_full_like(ax, 2.0^-64), lax_full_like(ax, 1.0))
    return((num * scale) / (ax * scale))
  }
  num <- sp_const(grad_large, 0x1.20dd750429b6dp-1) + 0.5 * grad_large
  num / ax
}

# Port of jax/_src/lax/erf.py:erfcx_grad_large: erfcx'(ax) for ax >= 1.
sp_erfcx_grad_large <- function(ax) {
  u <- lax_integer_pow(ax, -1L)
  t <- lax_square(u)
  tb <- sp_erf_tables
  dt <- sp_dtype(ax)
  if (dt == "float32") {
    g_1_2 <- sp_poly_full(u - 0.75, tb$ERFC32_G_1_2)
    g_2_8 <- sp_poly_full(u - 0.3125, tb$ERFC32_G_2_8)
    g_8_inf <- sp_poly_full(t, tb$ERFC32_G_8_INF)
    g <- lax_select(ax < 2.0, g_1_2, lax_select(ax < 8.0, g_2_8, g_8_inf))
  } else if (dt == "float64") {
    g_1_2 <- sp_poly_full(u - 0.75, tb$ERFC64_G_1_2)
    g_2_4 <- sp_poly_full(u - 0.375, tb$ERFC64_G_2_4)
    g_4_12 <- sp_poly_full(u, tb$ERFC64_G_4_12)
    g_12_inf <- sp_poly_full(t, tb$ERFC64_G_12_INF)
    g <- lax_select(ax < 2.0, g_1_2,
                    lax_select(ax < 4.0, g_2_4, lax_select(ax < 12.0, g_4_12, g_12_inf)))
  } else {
    jax_error("Unsupported dtype: ", dt, class = "jax_type_error")
  }
  -t * g
}

# Port of jax/_src/lax/erf.py:erfcx_large: erfcx(ax) for ax >= 1.
sp_erfcx_large <- function(ax) {
  ax <- lax_max(ax, sp_const(ax, 1.0))
  sp_erfcx_from_grad_large(ax, sp_erfcx_grad_large(ax))
}

# Port of jax/_src/lax/erf.py:erfc_impl (the lowering of erfc with accuracy HIGHEST).
sp_erfc_impl <- function(x) {
  ax <- abs(x)
  small <- sp_erfc_small(lax_clamp(sp_const(x, -1.0), x, sp_const(x, 1.0)))
  ax_exp <- lax_min(ax, sp_const(ax, sp_erf_max_ax(x)))
  ss <- sp_split_square(ax_exp)
  y <- sp_erfcx_large(ax)
  expm1_neg_lo <- sp_expm1_lo(-ss[[2L]])
  large_pos <- lax_exp(-ss[[1L]]) * (y + y * expm1_neg_lo)
  large <- lax_select(x < 0.0, 2.0 - large_pos, large_pos)
  lax_select(ax < 1.0, small, large)
}

# Port of jax/_src/lax/erf.py:erfcx_impl.
sp_erfcx_impl <- function(x) {
  ax <- abs(x)
  f32 <- sp_dtype(x) == "float32"
  max_exp_ax <- if (f32) 9.382414 else 0x1.aa0f4d2e063cep+4
  min_x <- if (f32) -9.382415 else -0x1.aa0f4d2e063cfp+4
  ax_exp <- lax_min(ax, sp_const(ax, max_exp_ax))
  ss <- sp_split_square(ax_exp)
  e <- lax_exp(ss[[1L]])
  expm1_pos_lo <- sp_expm1_lo(ss[[2L]])
  y_small <- sp_erfc_small(lax_clamp(sp_const(x, -1.0), x, sp_const(x, 1.0)))
  small <- e * (y_small + y_small * expm1_pos_lo)
  large_pos <- sp_erfcx_large(ax)
  large_neg <- lax_select(x <= min_x, lax_full_like(x, Inf), (e + e * expm1_pos_lo) * 2.0 - large_pos)
  large <- lax_select(x < 0.0, large_neg, large_pos)
  lax_select(ax < 1.0, small, large)
}

# Port of jax/_src/lax/erf.py:erfcx_grad_impl: erfcx'(x) from the identity
# 2 x erfcx(x) - 2 / sqrt(pi) for x < 0 and from polynomials fit to erfcx' for x >= 0.
sp_erfcx_grad_impl <- function(x, ans) {
  is_neg <- x < 0.0
  is_large <- x >= 1.0
  zeros <- lax_full_like(x, 0.0)
  x_neg <- lax_select(is_neg, x, zeros)
  ans_neg <- lax_select(is_neg, ans, zeros)
  direct_neg <- 2.0 * x_neg * ans_neg - 2.0 / sqrt(pi)
  s <- lax_clamp(sp_const(x, 0.0), x, sp_const(x, 1.0)) - 0.5
  grad_0_1 <- sp_poly_full(s, sp_erf_table(x, "ERFC32_G_0_1"))
  grad_large <- sp_erfcx_grad_large(lax_max(x, sp_const(x, 1.0)))
  lax_select(is_neg, direct_neg, lax_select(is_large, grad_large, grad_0_1))
}

# Port of jax/_src/lax/erf.py:_piece_index and _select_piece.
sp_piece_index <- function(conds) {
  if (length(conds) == 1L) return(lax_bitwise_not(conds[[1L]]))
  idx <- lax_convert_element_type(lax_bitwise_not(conds[[1L]]), "int32")
  for (cond in conds[-1L]) idx <- idx + lax_convert_element_type(lax_bitwise_not(cond), "int32")
  idx
}
sp_select_piece <- function(w, idx, vals) {
  do.call(lax_select_n, c(list(idx), lapply(vals, function(v) lax_full_like(w, v))))
}

# Port of jax/_src/lax/erf.py:erf_inv_core: erfinv(x), or sqrt(2) erfinv(x) when `is_ndtri`,
# from x in (-1, 1) and q = 1 - |x| (so that 1 - x^2 = q (1 + |x|) has no cancellation
# near |x| = 1). erfinv passes (x, 1 - |x|) without the far tail pieces; erfcinv(y) passes
# (1 - y, min(y, 2 - y)); ndtri(p) passes (2 (p - 0.5), 2 min(p, 1 - p)).
sp_erf_inv_core <- function(x, q, has_far_tail = TRUE, is_ndtri = FALSE) {
  ax <- abs(x)
  w <- -lax_log(q * (1.0 + ax))
  dt <- sp_dtype(x)
  tb <- sp_erf_tables
  if (dt == "float32") {
    ps <- if (is_ndtri) tb[c("NDTRI32_P1", "NDTRI32_P2", "NDTRI32_P3")]
          else tb[c("ERFINV32_P1", "ERFINV32_P2", "ERFINV32_P3")]
    conds <- if (has_far_tail) list(w < 3.5, w < 18.0625) else list(w < 3.5)
    shifts <- if (has_far_tail) c(3.0, 7.0) else 3.0
    c1 <- 1.75
    unroll <- 9L
    k_h <- if (is_ndtri) 0x1.6a0000p0 else 1.0
  } else if (dt == "float64") {
    ps <- if (is_ndtri) tb[paste0("NDTRI64_P", 1:5)] else tb[paste0("ERFINV64_P", 1:5)]
    conds <- if (has_far_tail) list(w < 4.5, w < 16.0, w < 37.0, w < 144.0)
             else list(w < 4.5, w < 16.0)
    shifts <- if (has_far_tail) c(3.0, 5.0, 9.0, 19.5) else c(3.0, 5.0)
    c1 <- 2.25
    unroll <- 20L
    k_h <- if (is_ndtri) 0x1.6a09e68000000p0 else 1.0
  } else {
    jax_error("Unsupported dtype for erf_inv_core: ", dt, class = "jax_type_error")
  }
  ps <- ps[seq_len(length(conds) + 1L)]
  idx <- sp_piece_index(conds)
  w_safe <- lax_select(conds[[1L]], lax_full_like(w, 1.0), w)
  z <- lax_sqrt(w_safe)
  z_h <- sp_half_mantissa(z)
  z_l <- (w_safe - z_h * z_h) / (z + z_h)
  shift <- if (length(shifts) == 1L) lax_full_like(w, shifts) else sp_select_piece(w, idx, c(0.0, shifts))
  t <- lax_select(conds[[1L]], w - c1, z - shift)
  coeffs <- lapply(seq_len(min(lengths(ps))), function(j) {
    sp_select_piece(w, idx, vapply(ps, `[`, 0, j))
  })
  poly <- sp_polynomial(t, coeffs, unroll = unroll)
  tail_mag <- k_h * z_h + lax_optimization_barrier(k_h * z_l + poly)
  sign_x <- lax_select(x < 0.0, lax_full_like(x, -1.0), lax_full_like(x, 1.0))
  lax_select(conds[[1L]], x * poly, sign_x * tail_mag)
}

# Port of jax/_src/lax/special.py:_erf_inv_impl (the lowering of erf_inv when
# jax_accurate_erf_inv is set).
sp_erf_inv_impl <- function(x) {
  dt <- sp_dtype(x)
  if (!dt %in% c("float32", "float64"))
    return(lax_convert_element_type(sp_erf_inv_impl(lax_convert_element_type(x, "float32")), dt))
  ax <- abs(x)
  one <- sp_const(x, 1.0)
  out_of_bounds <- lax_bitwise_or(lax_gt(ax, one), sp_isnan(x))
  is_one <- lax_eq(ax, one)
  x_safe <- lax_select(lax_bitwise_or(out_of_bounds, is_one), lax_full_like(x, 0.0), x)
  q_safe <- lax_sub(one, abs(x_safe))
  res <- sp_erf_inv_core(x_safe, q_safe, has_far_tail = FALSE)
  lax_select(out_of_bounds, lax_full_like(x, NaN),
             lax_select(is_one, lax_mul(x, sp_const(x, Inf)), res))
}

# ---- lgamma.py: float32 lgamma ----------------------------------------------------------------

# Port of jax/_src/lax/lgamma.py:_split12: (x_hi, x_lo) with 12 mantissa bits in x_hi, so
# that products of high parts are exact in float32.
sp_split12 <- function(x) {
  xi <- lax_bitcast_convert_type(x, "int32")
  x_hi <- lax_bitcast_convert_type(lax_bitwise_and(xi, lax_full_like(xi, -4096)), "float32")
  list(x_hi, x - x_hi)
}

# Port of jax/_src/lax/lgamma.py:_horner (coefficients in ascending powers).
sp_horner <- function(x, coeffs) {
  n <- length(coeffs)
  p <- if (is_jax_value(coeffs[[n]])) coeffs[[n]] else lax_full_like(x, coeffs[[n]])
  for (c in rev(coeffs[-n])) p <- p * x + c
  p
}

# Port of jax/_src/lax/lgamma.py:_log_f32: ln(x 2^-sub_shift) as a compensated
# (log_hi, log_lo) pair.
sp_log_f32 <- function(x, sub_shift = 0L) {
  bits <- lax_bitcast_convert_type(x, "int32")
  exp_off <- lax_shift_right_arithmetic(bits - 1060439283L, lax_full_like(bits, 23))
  k <- lax_convert_element_type(exp_off - sub_shift, "float32")
  mant_bits <- bits - lax_shift_left(exp_off, lax_full_like(bits, 23))
  m <- lax_bitcast_convert_type(mant_bits, "float32")
  f <- m - 1.0
  f2 <- f * f
  ln1pf <- f - 0.5 * f2 + (f2 * f) * sp_horner(f, as.list(sp_lgamma_tables$LOG_Q_HI))
  h_d <- k * 0.693115234375
  l_d <- ln1pf + k * 3.19461832987e-05
  log_hi <- h_d + l_d
  log_lo <- (h_d - log_hi) + l_d
  list(log_hi, log_lo)
}

# Port of jax/_src/lax/lgamma.py:_lgamma_pos_small: lgamma on [0.5, 4] as (x - 1)(x - 2) P(d),
# returned as (lg_small, res_hi, res_lo); the unsummed pair keeps extra precision for the
# reflection formula.
sp_lgamma_pos_small <- function(x_small) {
  tb <- sp_lgamma_tables
  in_15_4 <- x_small > 1.5
  u <- x_small - 1.0
  v <- u - 1.0
  d <- u - lax_select(in_15_4, lax_full_like(u, 1.75), lax_full_like(u, 0.0))
  coeffs_hi <- Map(function(c1, c2) {
    lax_select(in_15_4, lax_full_like(x_small, c2), lax_full_like(x_small, c1))
  }, tb$POS_HI_05_15, tb$POS_HI_15_4)
  p <- sp_horner(d, coeffs_hi)
  us <- sp_split12(u)
  vs <- sp_split12(v)
  uv_hi <- us[[1L]] * vs[[1L]]
  uv_lo <- us[[2L]] * v + us[[1L]] * vs[[2L]]
  ps <- sp_split12(p)
  res_hi <- uv_hi * ps[[1L]]
  res_lo <- uv_lo * p + uv_hi * ps[[2L]]
  list(res_hi + res_lo, res_hi, res_lo)
}

# Port of jax/_src/lax/lgamma.py:_lgamma_pos_large: the Stirling series for x >= 4, returned
# as (p_hi, s_pos, psi_large) with p_hi + s_pos ~ lgamma(x) and psi_large ~ digamma(x).
sp_lgamma_pos_large <- function(x_gt4) {
  lg <- sp_log_f32(x_gt4)
  s <- lg[[1L]]
  hs <- sp_split12(s)
  h_x <- hs[[1L]]
  l_x <- hs[[2L]] + lg[[2L]]
  b <- x_gt4 - 0.5
  bs <- sp_split12(b)
  hx_m1 <- h_x - 1.0
  p_hi <- bs[[1L]] * hx_m1
  p_lo <- bs[[2L]] * hx_m1 + b * l_x
  inv_x <- lax_integer_pow(x_gt4, -1L)
  sp <- sp_horner(inv_x * inv_x, as.list(sp_lgamma_tables$STIRLING_HI))
  s_pos <- p_lo + 0x1.acfe38p-2 + inv_x * sp
  list(p_hi, s_pos, s - 0.5 * inv_x)
}

# Port of jax/_src/lax/lgamma.py:_lgamma_neg_reflection: Euler's reflection formula
# -(ln|delta| + lgamma(1 - x) - ln(pi delta / sin(pi delta))).
sp_lgamma_neg_reflection <- function(delta, log_delta_hi, lg_pos_hi, lg_pos_lo, dy_corr) {
  d2 <- delta * delta
  sinc_term <- d2 * sp_horner(d2, as.list(sp_lgamma_tables$SINC_HI))
  -((log_delta_hi + lg_pos_hi) + (lg_pos_lo + (sinc_term + dy_corr)))
}

# Port of jax/_src/lax/lgamma.py:_lgamma_neg_root: lgamma near its negative zeros through
# rational root window polynomials in w = (delta - r) / (delta + r). Returns
# (in_root_win, w P(w)). JAX gathers the table columns by index; here a one hot row times
# the table reads the same values (the other terms of each sum are exact zeros).
sp_lgamma_neg_root <- function(x_neg_safe, n) {
  n_root <- lax_select((x_neg_safe > -2.6) & (x_neg_safe < -2.0), lax_full_like(x_neg_safe, -2.0), n)
  delta_root <- x_neg_safe - n_root
  k <- lax_convert_element_type(-n_root, "int32")
  in_k_range <- (k >= 2L) & (k <= 9L)
  side <- lax_select(delta_root > 0.0, lax_full_like(k, 1), lax_full_like(k, 0))
  idx <- lax_select(in_k_range, 2L * (k - 2L) + side, lax_full_like(k, 1))
  nd <- length(sp_shape(idx))
  slots <- lax_expand_dims(lax_iota("int32", 16L), seq_len(nd) - 1L)
  one_hot <- lax_convert_element_type(lax_expand_dims(idx, nd) == slots, "float32")
  table <- as_jax(do.call(cbind, sp_lgamma_tables$NEG_TABLE), "float32")  # [slot, row]
  looked_up <- lax_dot_general(one_hot, table, list(list(nd, 0L), list(integer(0), integer(0))))
  cols <- lapply(seq_len(14L) - 1L, function(j) lax_index_in_dim(looked_up, j, nd, keepdims = FALSE))
  r_hi <- cols[[1L]]
  r_lo <- cols[[2L]]
  w_bound <- cols[[3L]]
  d0 <- delta_root - r_hi
  d <- delta_root + r_hi
  rcp <- lax_integer_pow(d, -1L)
  w <- (d0 - r_lo) * rcp
  w <- w + ((d0 - w * d) - r_lo * (1.0 + w)) * rcp
  in_root_win <- lax_abs(w) <= w_bound
  w_safe <- lax_select(in_root_win, w, lax_full_like(x_neg_safe, 0.0))
  list(in_root_win, w_safe * sp_horner(w_safe, cols[-(1:3)]))
}

# Port of jax/_src/lax/lgamma.py:lgamma_impl: float32 lgamma for all normal, subnormal and
# special inputs. x >= 0.5 is evaluated directly (factored minimax polynomials on [0.5, 4],
# the Stirling series above 4); x < 0.5 reflects to y = 1 - x through Euler's formula or
# through the root windows near the negative zeros on (-10, -2).
sp_lgamma_impl <- function(x) {
  max_finite_arg <- 0x1.895f1950b67a5p+121  # largest float32 whose lgamma is finite in float32
  # Subnormals are normalized with integer arithmetic before any float operation.
  bits <- lax_bitcast_convert_type(x, "int32")
  abs_bits <- lax_bitwise_and(bits, lax_full_like(bits, 2147483647))
  is_zero <- abs_bits == 0L
  is_sub <- (abs_bits > 0L) & (abs_bits < 8388608L)
  sub_shift <- lax_select(is_sub, lax_clz(abs_bits) - 8L, lax_full_like(bits, 0))
  x_norm <- lax_bitcast_convert_type(
    lax_select(is_sub,
               lax_bitwise_or(lax_bitwise_and(bits, lax_full_like(bits, -2147483648)),
                              lax_shift_left(abs_bits, sub_shift)),
               bits), "float32")

  is_pos <- x_norm >= 0.5
  # 1.0 stands in for 0.0 so inactive branches avoid 0 / 0 and log(0).
  x_safe <- lax_select(is_zero, lax_full_like(x, 1.0), x_norm)

  # For x <= -2^23 every float32 is an integer pole. Inputs with x >= 0.5 and large negative
  # integers are clamped to -0.5 so round() cannot overflow and delta stays away from zero.
  x_neg_safe <- lax_select(lax_bitwise_not(is_pos) & (x_safe > -8388608.0), x_safe,
                           lax_full_like(x, -0.5))
  # x = n + delta with n = round(x) and delta in [-0.5, 0.5] (exact); the reflected argument
  # y = 1 - x = (1 - n) - delta is formed as a Fast2Sum pair (y, y_lo).
  n <- lax_round(x_neg_safe, 1L)
  delta <- x_neg_safe - n
  abs_delta <- lax_abs(delta)
  y <- (1.0 - n) - delta
  y_lo <- ((1.0 - n) - y) - delta

  # Both halves evaluate lgamma at an argument >= 0.5, clamped to keep the Stirling series
  # free of inf - inf.
  arg_pos <- lax_select(is_pos, x_safe, y)
  x_bounded <- lax_min(arg_pos, lax_full_like(x, max_finite_arg))

  # ln|delta| for the reflection formula (sub_shift applies to subnormal x).
  lg <- sp_log_f32(abs_delta, sub_shift)
  log_hi <- lg[[1L]]
  log_lo <- lg[[2L]]

  small <- sp_lgamma_pos_small(x_bounded)
  large <- sp_lgamma_pos_large(x_bounded)
  p_hi <- large[[1L]]
  s_pos <- large[[2L]]
  psi <- large[[3L]]
  in_le_four <- x_bounded <= 4.0
  lg_pos <- lax_select(in_le_four, small[[1L]], p_hi + s_pos)
  lg_pos_hi <- lax_select(in_le_four, small[[2L]], p_hi)
  lg_pos_lo <- lax_select(in_le_four, small[[3L]], s_pos)

  # Reflection (with the first order correction psi(y) y_lo for the rounding of y) or the
  # root windows, where reflection cancels catastrophically.
  refl <- sp_lgamma_neg_reflection(delta, log_hi, lg_pos_hi, lg_pos_lo + log_lo, psi * y_lo)
  root <- sp_lgamma_neg_root(x_neg_safe, n)
  neg_res <- lax_select(root[[1L]], root[[2L]], refl)

  res <- lax_select(is_pos, lg_pos, neg_res)
  # -0.0 at x = 1 and 2 becomes +0.0.
  res <- lax_select(res == 0.0, lax_full_like(x, 0.0), res)
  # +inf at the poles (x = +-0, negative integers, -inf) and on overflow; NaN is preserved.
  is_neg_int <- lax_bitwise_not(is_pos) & (x_safe == lax_floor(x_safe))
  is_pole_or_inf <- is_zero | is_neg_int | (x > max_finite_arg)
  res <- lax_select(is_pole_or_inf, lax_full_like(x, Inf), res)
  lax_select(x != x, lax_full_like(x, NaN), res)
}

# ---- rules of the special primitives defined in lax.R ------------------------------------------
# lgamma_p, erf_p, erfc_p and erf_inv_p are created in lax.R with CHLO lowerings and
# exp(-x^2) JVPs; the rules of jax/_src/lax/special.py are installed here.

# Lowering through a function of jax values with the traced jaxpr cached per input types
# (mlir.lower_fun with mlir.cache_lowering): eager calls do not retrace the implementation.
sp_lower_fun_cached <- function(fun) {
  cache <- new.env(hash = TRUE, parent = emptyenv())
  function(ctx, ins, avals_in, avals_out, params) {
    key <- paste(c(vapply(avals_in, aval_key, ""), x64_enabled()), collapse = ";")
    cj <- cache[[key]]
    if (is.null(cj)) {
      cj <- with_trace(eval_trace(), trace_to_jaxpr(function(args) list(fun(args, params)),
                                                    avals_in, parent = eval_trace()))
      assign(key, cj, envir = cache)
    }
    consts <- lapply(seq_along(cj$consts), function(i) {
      ir_constant(ctx, array_host(cj$consts[[i]]), cj$jaxpr$constvars[[i]]$aval)
    })
    jaxpr_subcomp(ctx, cj$jaxpr, consts, as.list(ins))
  }
}

# Port of jax/_src/lax/special.py:_lgamma_lowering.
sp_lgamma_lower_f32 <- sp_lower_fun_cached(function(args, params) {
  v <- args[[1L]]
  dt <- sp_dtype(v)
  if (dt == "float32") return(sp_lgamma_impl(v))
  lax_convert_element_type(sp_lgamma_impl(lax_convert_element_type(v, "float32")), dt)
})
lgamma_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  if (avals_in[[1L]]$dtype == "float64")
    return(lower_nary("chlo.lgamma")(ctx, ins, avals_in, avals_out, params))
  sp_lgamma_lower_f32(ctx, ins, avals_in, avals_out, params)
}

defjvp(erf_p, function(g, primals, params) {
  x <- primals[[1L]]
  lax_mul(sp_const(x, 2 / sqrt(pi)), lax_mul(g, sp_exp_neg_sq(x)))
})
defjvp(erfc_p, function(g, primals, params) {
  x <- primals[[1L]]
  lax_mul(sp_const(x, -2 / sqrt(pi)), lax_mul(g, sp_exp_neg_sq(x)))
})

# Port of jax/_src/lax/special.py:_erfc_lowering (accuracy is normalized by
# accuracy_params()).
sp_erfc_lower_highest <- sp_lower_fun_cached(function(args, params) sp_erfc_impl(args[[1L]]))
erfc_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  if (identical(params$accuracy, "highest") && avals_in[[1L]]$dtype %in% c("float32", "float64"))
    return(sp_erfc_lower_highest(ctx, ins, avals_in, avals_out, params))
  lower_nary("chlo.erfc")(ctx, ins, avals_in, avals_out, params)
}

# Port of jax/_src/lax/special.py:_erf_inv_lowering. jax_accurate_erf_inv (default FALSE in
# JAX) is not a jaxR option yet; the accurate lowering is used once the option exists.
sp_erf_inv_lower_accurate <- sp_lower_fun_cached(function(args, params) sp_erf_inv_impl(args[[1L]]))
erf_inv_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  if (isTRUE(.jx$config$jax_accurate_erf_inv))
    return(sp_erf_inv_lower_accurate(ctx, ins, avals_in, avals_out, params))
  lower_nary("chlo.erf_inv")(ctx, ins, avals_in, avals_out, params)
}

# ---- primitives -----------------------------------------------------------------------------

regularized_incomplete_beta_p <- def_binop("regularized_incomplete_beta", NULL, "float")
regularized_incomplete_beta_p$lowering <- sp_lower_fun_cached(function(args, params) {
  do.call(sp_up_and_broadcast(sp_regularized_incomplete_beta_impl), args)
})
sp_betainc_grad_not_implemented <- function(g, primals, params) {
  jax_error("Betainc gradient with respect to a and b not supported.", class = "jax_value_error")
}
# Port of jax/_src/lax/special.py:betainc_gradx.
defjvp(regularized_incomplete_beta_p, sp_betainc_grad_not_implemented,
       sp_betainc_grad_not_implemented,
       function(g, primals, params) {
         a <- primals[[1L]]; b <- primals[[2L]]; x <- primals[[3L]]
         lbeta <- lax_lgamma(a) + lax_lgamma(b) - lax_lgamma(a + b)
         partial_x <- lax_exp((b - 1) * lax_log1p(-x) + (a - 1) * lax_log(x) - lbeta)
         partial_x * g
       })

# The expansion of chlo.polygamma in the plugin gives NaN for order zero at zero, where
# JAX gives -Inf (the pole of digamma).
polygamma_p <- def_binop("polygamma", function(ctx, ins, ai, ao, p) {
  a <- ao[[1L]]
  t <- tensor_type(a)
  bt <- tensor_type(new_aval(a$shape, "bool"))
  ops <- bcast_operands(ctx, ins, ai, a)
  y <- emit_op(ctx, "chlo.polygamma", ops, c(t, t), t)
  eq0 <- function(v) {
    emit_op(ctx, "stablehlo.compare", c(v, full_constant(ctx, 0, a)), c(t, t), bt,
            attrs = list(comparison_direction = "#stablehlo<comparison_direction EQ>",
                         compare_type = "#stablehlo<comparison_type FLOAT>"))
  }
  pole <- emit_op(ctx, "stablehlo.and", c(eq0(ops[[1L]]), eq0(ops[[2L]])), c(bt, bt), bt)
  emit_op(ctx, "stablehlo.select", c(pole, full_constant(ctx, -Inf, a), y), c(bt, t, t), t)
}, "float")
# Port of jax/_src/lax/special.py:polygamma_gradm and polygamma_gradx.
defjvp(polygamma_p,
       function(g, primals, params) {
         jax_error("polygamma gradient with respect to m is not supported", class = "jax_value_error")
       },
       function(g, primals, params) {
         m <- primals[[1L]]
         g * lax_polygamma(lax_add(m, sp_const(m, 1)), primals[[2L]])
       })

igamma_p <- def_binop("igamma", NULL, "float")
igamma_p$lowering <- sp_lower_fun_cached(function(args, params) {
  do.call(sp_up_and_broadcast(sp_igamma_impl), args)
})
igamma_grad_a_p <- def_binop("igamma_grad_a", NULL, "float")
igamma_grad_a_p$lowering <- sp_lower_fun_cached(function(args, params) {
  do.call(sp_up_and_broadcast(sp_igamma_grad_a_impl), args)
})
igammac_p <- def_binop("igammac", NULL, "float")
igammac_p$lowering <- sp_lower_fun_cached(function(args, params) {
  do.call(sp_up_and_broadcast(sp_igammac_impl), args)
})

# Port of jax/_src/lax/special.py:igamma_gradx and igamma_grada.
sp_igamma_gradx <- function(g, primals, params) {
  a <- primals[[1L]]; x <- primals[[2L]]
  g * lax_exp(-x + (a - lax_full_like(a, 1)) * lax_log(x) - lax_lgamma(a))
}
sp_igamma_grada <- function(g, primals, params) g * lax_igamma_grad_a(primals[[1L]], primals[[2L]])
defjvp(igamma_p, sp_igamma_grada, sp_igamma_gradx)
defjvp(igammac_p,
       function(g, primals, params) -sp_igamma_grada(g, primals, params),
       function(g, primals, params) -sp_igamma_gradx(g, primals, params))

zeta_p <- def_binop("zeta", "chlo.zeta", "float")

bessel_i0e_p <- def_unop("bessel_i0e", NULL, "float")
bessel_i0e_p$lowering <- sp_lower_fun_cached(function(args, params) sp_bessel_i0e_impl(args[[1L]]))
defjvp2(bessel_i0e_p, function(g, y, primals, params) {
  x <- primals[[1L]]
  g * (lax_bessel_i1e(x) - lax_sign(x) * y)
})

bessel_i1e_p <- def_unop("bessel_i1e", "chlo.bessel_i1e", "float")
# Port of jax/_src/lax/special.py:_bessel_i1e_jvp.
defjvp2(bessel_i1e_p, function(g, y, primals, params) {
  x <- primals[[1L]]
  eps <- finfo(sp_dtype(x))$eps
  x_is_not_tiny <- abs(x) > eps
  safe_x <- lax_select(x_is_not_tiny, x, lax_full_like(x, eps))
  dy_dx <- lax_bessel_i0e(safe_x) - y * (lax_sign(safe_x) + lax_integer_pow(safe_x, -1L))
  dy_dx <- lax_select(x_is_not_tiny, dy_dx, lax_full_like(x, 0.5))
  g * dy_dx
})

# ---- public functions -----------------------------------------------------------------------

#' lax special functions
#'
#' Ports of the special functions of `jax.lax`: the regularized incomplete beta integral
#' (`lax_betainc`), the regularized incomplete gamma functions and their derivatives
#' (`lax_igamma`, `lax_igammac`, `lax_igamma_grad_a`, `lax_random_gamma_grad`), the
#' polygamma and Hurwitz zeta functions and the exponentially scaled modified Bessel
#' functions of order 0 and 1. Arguments must be floating point arrays with equal dtypes
#' (R scalars adopt the dtype of the other arguments).
#' @param a,b,x,m,q Arrays.
#' @return An array.
#' @name lax_special
NULL

#' @rdname lax_special
#' @export
lax_betainc <- function(a, b, x) bind(regularized_incomplete_beta_p, lax_args(a, b, x))
#' @rdname lax_special
#' @export
lax_polygamma <- function(m, x) bind(polygamma_p, lax_args(m, x))
#' @rdname lax_special
#' @export
lax_igamma <- function(a, x) bind(igamma_p, lax_args(a, x))
#' @rdname lax_special
#' @export
lax_igammac <- function(a, x) bind(igammac_p, lax_args(a, x))
#' @rdname lax_special
#' @export
lax_igamma_grad_a <- function(a, x) bind(igamma_grad_a_p, lax_args(a, x))
#' @rdname lax_special
#' @export
lax_random_gamma_grad <- function(a, x) sp_up_and_broadcast(sp_random_gamma_grad_impl)(a, x)
#' @rdname lax_special
#' @export
lax_zeta <- function(x, q) bind(zeta_p, lax_args(x, q))
#' @rdname lax_special
#' @export
lax_bessel_i0e <- function(x) bind(bessel_i0e_p, list(as_jax(x)))
#' @rdname lax_special
#' @export
lax_bessel_i1e <- function(x) bind(bessel_i1e_p, list(as_jax(x)))
