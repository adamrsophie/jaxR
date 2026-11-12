# jax.scipy.integrate (port of jax/_src/scipy/integrate.py) and jax.experimental.ode
# (port of jax/experimental/ode.py: Dormand-Prince integration with adaptive steps and
# the adjoint method for reverse mode differentiation).

#' Integrate along an axis with the trapezoid rule
#'
#' Port of `jax.scipy.integrate.trapezoid` (the same function as `jnp.trapezoid`).
#' @param y Array of values.
#' @param x Optional sample points (default spacing `dx`).
#' @param dx Spacing when `x` is `NULL`.
#' @param axis Axis of integration (0-based).
#' @return An array.
#' @export
scipy_integrate_trapezoid <- function(y, x = NULL, dx = 1.0, axis = -1L) {
  # Port of jax/_src/scipy/integrate.py:trapezoid
  jnp_trapezoid(y, x = x, dx = dx, axis = axis)
}

# ---- jax.experimental.ode -------------------------------------------------------------

# Port of jax.closure_convert (R/api_misc.R:closure_convert): traces `fun` at the
# types of the example arguments and hoists its closed-over traced values of inexact
# dtype, which may carry derivatives. Returns list(converted, consts) where
# converted(<args>, <consts>) evaluates `fun`.
ode_closure_convert <- function(fun, ...) {
  fl <- tree_flatten(list(...))
  store <- new.env(parent = emptyenv())
  cj <- trace_to_jaxpr(flat_fun(fun, fl$treedef, store),
                       lapply(fl$leaves, function(l) aval_of(as_jax(l))))
  out_tree <- store$out_tree
  consts <- cj$consts
  hoist <- vapply(consts, function(c) is_tracer(c) && is_inexact_dtype(aval_of(c)$dtype), NA)
  nargs <- length(list(...))
  converted <- function(...) {
    all <- list(...)
    cs <- consts
    cs[hoist] <- drop_first(all, nargs)
    afl <- tree_flatten(all[seq_len(nargs)])
    tree_unflatten(out_tree, eval_jaxpr(cj$jaxpr, cs, lapply(afl$leaves, as_jax)))
  }
  list(converted, consts[hoist])
}

# Port of ode.py:ravel_first_arg.
ode_ravel_first_arg <- function(f, unravel) {
  force(f)
  force(unravel)
  function(y_flat, ...) ravel_pytree(f(unravel(y_flat), ...))$flat
}

# jnp.polyval(p, x) for coefficients along the leading axis of p (Horner's scheme).
ode_polyval <- function(p, x) {
  y <- jnp_zeros_like(lax_index_in_dim(p, 0L, keepdims = FALSE))
  for (i in seq_len(aval_of(p)$shape[1L]) - 1L) y <- y * x + lax_index_in_dim(p, i, keepdims = FALSE)
  y
}

# Port of ode.py:interp_fit_dopri.
ode_interp_fit_dopri <- function(y0, y1, k, dt) {
  # Fit a polynomial to the results of a Runge-Kutta step.
  dps_c_mid <- as_jax(c(
    6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2,
    -2691868925 / 45128329728 / 2, 187940372067 / 1594534317056 / 2,
    -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2), aval_of(y0)$dtype)
  y_mid <- y0 + jnp_astype(dt, aval_of(y0)$dtype) * jnp_dot(dps_c_mid, k)
  jnp_stack(ode_fit_4th_order_polynomial(y0, y1, y_mid, lax_index_in_dim(k, 0L, keepdims = FALSE),
                                         lax_index_in_dim(k, 6L, keepdims = FALSE), dt))
}

# Port of ode.py:fit_4th_order_polynomial.
ode_fit_4th_order_polynomial <- function(y0, y1, y_mid, dy0, dy1, dt) {
  dt <- jnp_astype(dt, aval_of(y0)$dtype)
  a <- -2. * dt * dy0 + 2. * dt * dy1 - 8. * y0 - 8. * y1 + 16. * y_mid
  b <- 5. * dt * dy0 - 3. * dt * dy1 + 18. * y0 + 14. * y1 - 32. * y_mid
  c <- -4. * dt * dy0 + dt * dy1 - 11. * y0 - 5. * y1 + 16. * y_mid
  d <- dt * dy0
  e <- y0
  list(a, b, c, d, e)
}

# Port of ode.py:initial_step_size.
ode_initial_step_size <- function(fun, t0, y0, order, rtol, atol, f0) {
  # Algorithm from:
  # E. Hairer, S. P. Norsett G. Wanner,
  # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
  yf <- promote_dtypes_inexact(list(y0, f0))
  y0 <- yf[[1L]]
  f0 <- yf[[2L]]
  dt <- aval_of(y0)$dtype
  scale <- atol + jnp_abs(y0) * rtol
  d0 <- jnp_linalg_norm(y0 / jnp_astype(scale, dt))
  d1 <- jnp_linalg_norm(f0 / jnp_astype(scale, dt))
  h0 <- jnp_where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)
  y1 <- y0 + jnp_astype(h0, dt) * f0
  f1 <- fun(y1, t0 + h0)
  d2 <- jnp_linalg_norm((f1 - f0) / jnp_astype(scale, dt)) / h0
  h1 <- jnp_where((d1 <= 1e-15) & (d2 <= 1e-15), jnp_maximum(1e-6, h0 * 1e-3),
                  jnp_power(0.01 / jnp_maximum(d1, d2), 1. / (order + 1.)))
  jnp_minimum(100. * h0, h1)
}

# Port of ode.py:runge_kutta_step (Dopri5 Butcher tableaux).
ode_runge_kutta_step <- function(func, y0, f0, t0, dt) {
  fdt <- aval_of(f0)$dtype
  alpha <- as_jax(c(1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1., 0), aval_of(dt)$dtype)
  beta <- as_jax(matrix(c(
    1 / 5, 0, 0, 0, 0, 0, 0,
    3 / 40, 9 / 40, 0, 0, 0, 0, 0,
    44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0,
    19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0,
    9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0,
    35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0), 6L, 7L, byrow = TRUE), fdt)
  c_sol <- as_jax(c(35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0), fdt)
  c_error <- as_jax(c(
    35 / 384 - 1951 / 21600, 0, 500 / 1113 - 22642 / 50085, 125 / 192 - 451 / 720,
    -2187 / 6784 - -12231 / 42400, 11 / 84 - 649 / 6300, -1. / 60.), fdt)
  dtf <- jnp_astype(dt, fdt)
  n <- aval_of(f0)$shape[1L]
  k <- at_set(jnp_zeros(c(7L, n), dtype = fdt), 0L, value = f0)
  k <- lax_fori_loop(1L, 7L, function(i, k) {
    ti <- t0 + dt * alpha[i - 1L]
    yi <- y0 + dtf * jnp_dot(beta[i - 1L], k)
    at_set(k, i, value = func(yi, ti))
  }, k)
  y1 <- dtf * jnp_dot(c_sol, k) + y0
  y1_error <- dtf * jnp_dot(c_error, k)
  f1 <- lax_index_in_dim(k, 6L, keepdims = FALSE)
  list(y1, f1, y1_error, k)
}

# Port of ode.py:abs2.
ode_abs2 <- function(x) {
  if (is_complex_dtype(aval_of(x)$dtype)) {
    lax_integer_pow(jnp_real(x), 2L) + lax_integer_pow(jnp_imag(x), 2L)
  } else {
    lax_integer_pow(x, 2L)
  }
}

# Port of ode.py:mean_error_ratio.
ode_mean_error_ratio <- function(error_estimate, rtol, atol, y0, y1) {
  err_tol <- atol + rtol * jnp_maximum(jnp_abs(y0), jnp_abs(y1))
  err_ratio <- error_estimate / jnp_astype(err_tol, aval_of(error_estimate)$dtype)
  jnp_sqrt(jnp_mean(ode_abs2(err_ratio)))
}

# Port of ode.py:optimal_step_size.
ode_optimal_step_size <- function(last_step, mean_error_ratio, safety = 0.9, ifactor = 10.0,
                                  dfactor = 0.2, order = 5.0) {
  dfactor <- jnp_where(mean_error_ratio < 1, 1.0, dfactor)
  factor <- jnp_minimum(ifactor, jnp_maximum(jnp_power(mean_error_ratio, -1.0 / order) * safety,
                                             dfactor))
  jnp_where(mean_error_ratio == 0, last_step * ifactor, last_step * factor)
}

ode_clip <- function(x, hi) jnp_minimum(hi, jnp_maximum(0., x))

# Port of ode.py:_odeint (the primal computation of the custom_vjp function).
ode_odeint_impl <- function(func, rtol, atol, mxstep, hmax, y0, ts, ...) {
  args <- list(...)
  func_ <- function(y, t) do.call(func, c(list(y, t), args))
  nt <- aval_of(ts)$shape[1L]
  t0 <- ts[0L]
  scan_fun <- function(carry, target_t) {
    cond_fun <- function(state) {
      (state[[4L]] < target_t) & (state[[1L]] < mxstep) & (state[[5L]] > 0)
    }
    body_fun <- function(state) {
      y <- state[[2L]]
      f <- state[[3L]]
      t <- state[[4L]]
      dt <- state[[5L]]
      rk <- ode_runge_kutta_step(func_, y, f, t, dt)
      next_t <- t + dt
      error_ratio <- ode_mean_error_ratio(rk[[3L]], rtol, atol, y, rk[[1L]])
      new_interp_coeff <- ode_interp_fit_dopri(y, rk[[1L]], rk[[4L]], dt)
      dt <- ode_clip(ode_optimal_step_size(dt, error_ratio), hmax)
      new <- list(state[[1L]] + 1L, rk[[1L]], rk[[2L]], next_t, dt, t, new_interp_coeff)
      old <- list(state[[1L]] + 1L, y, f, t, dt, state[[6L]], state[[7L]])
      accept <- error_ratio <= 1.
      Map(function(a, b) jnp_where(accept, a, b), new, old)
    }
    carry <- drop_first(lax_while_loop(cond_fun, body_fun, c(list(0L), carry)), 1L)
    t <- carry[[3L]]
    last_t <- carry[[5L]]
    interp_coeff <- carry[[6L]]
    relative_output_time <- (target_t - last_t) / (t - last_t)
    y_target <- ode_polyval(interp_coeff, jnp_astype(relative_output_time,
                                                     aval_of(interp_coeff)$dtype))
    list(carry, y_target)
  }
  f0 <- func_(y0, t0)
  dt <- ode_clip(ode_initial_step_size(func_, t0, y0, 4, rtol, atol, f0), hmax)
  interp_coeff <- jnp_stack(rep(list(y0), 5L))
  init_carry <- list(y0, f0, t0, dt, t0, interp_coeff)
  ys <- lax_scan(scan_fun, init_carry, lax_slice(ts, 1L, nt))[[2L]]
  jnp_concatenate(list(lax_expand_dims(y0, 0L), ys))
}

# Port of ode.py:_odeint_fwd.
ode_odeint_fwd <- function(func, rtol, atol, mxstep, hmax, y0, ts, ...) {
  ys <- ode_odeint(func, rtol, atol, mxstep, hmax, y0, ts, ...)
  list(ys, list(ys, ts, list(...)))
}

# Port of ode.py:_odeint_rev (adjoint method, Appendix C of arXiv:1806.07366).
ode_odeint_rev <- function(func, rtol, atol, mxstep, hmax, res, g) {
  ys <- res[[1L]]
  ts <- res[[2L]]
  args <- res[[3L]]
  nt <- aval_of(ts)$shape[1L]
  aug_dynamics <- function(augmented_state, t, ...) {
    # Original system augmented with vjp_y, vjp_t and vjp_args.
    y <- augmented_state[[1L]]
    y_bar <- augmented_state[[2L]]
    # `t` here is negative time, so we need to negate again to get back to normal time.
    # See the `odeint` invocation in `scan_fun` below.
    r <- do.call(vjp, c(list(func, y, -t), list(...)))
    c(list(-r[[1L]]), r[[2L]](y_bar))
  }
  scan_fun <- function(carry, i) {
    y_bar <- carry[[1L]]
    t0_bar <- carry[[2L]]
    args_bar <- carry[[3L]]
    # Compute effect of moving measurement time
    # `t_bar` should not be complex as it represents time
    t_bar <- jnp_real(jnp_dot(do.call(func, c(list(ys[i], ts[i]), args)), g[i]))
    t0_bar <- t0_bar - t_bar
    # Run augmented system backwards to previous observation
    out <- do.call(odeint, c(list(aug_dynamics, list(ys[i], y_bar, t0_bar, args_bar),
                                  jnp_stack(list(-ts[i], -ts[i - 1L]))), args,
                             list(rtol = rtol, atol = atol, mxstep = mxstep, hmax = hmax)))
    second <- function(x) tree_map(function(l) l[1L], x)
    y_bar <- second(out[[2L]]) + g[i - 1L]
    list(list(y_bar, second(out[[3L]]), second(out[[4L]])), t_bar)
  }
  init_carry <- list(g[nt - 1L], 0., tree_map(jnp_zeros_like, args))
  r <- lax_scan(scan_fun, init_carry, jnp_arange(nt - 1L, 0L, -1L))
  ts_bar <- jnp_concatenate(list(lax_expand_dims(r[[1L]][[2L]], 0L), lax_rev(r[[2L]], 0L)))
  c(list(r[[1L]][[1L]], ts_bar), r[[1L]][[3L]])
}

ode_odeint <- custom_vjp(ode_odeint_impl, ode_odeint_fwd, ode_odeint_rev,
                         nondiff_argnums = 0:4)

#' Integrate a system of ordinary differential equations
#'
#' Port of `jax.experimental.ode.odeint`: adaptive step size Dormand-Prince (Runge-Kutta
#' 4(5)) integration. Reverse mode derivatives use the adjoint method (an augmented
#' system integrated backwards in time); forward mode differentiation is not supported,
#' as in JAX. `experimental_ode_odeint` is an alias.
#' @param func Function `func(y, t, ...)` returning the time derivative of `y` (same
#'   structure as `y0`).
#' @param y0 Initial state (array or pytree of arrays).
#' @param t Strictly increasing float array of evaluation times; `t[0]` is the time of
#'   `y0`.
#' @param ... Extra arguments passed to `func` (arrays or pytrees of arrays).
#' @param rtol,atol Relative and absolute local error tolerances.
#' @param mxstep Maximum number of steps per output time.
#' @param hmax Maximum step size.
#' @return The solution at the times `t`: the structure of `y0` with a new leading axis
#'   of length `length(t)`.
#' @export
odeint <- function(func, y0, t, ..., rtol = 1.4e-8, atol = 1.4e-8, mxstep = Inf, hmax = Inf) {
  # Port of jax/experimental/ode.py:odeint
  args <- list(...)
  for (arg in tree_leaves(args)) {
    if (!(is_jax_value(arg) || is.numeric(arg) || is.logical(arg) || is.complex(arg)))
      jax_error("The contents of odeint *args must be arrays or scalars, but got ",
                paste(format(arg), collapse = " "), ".", class = "jax_type_error")
  }
  t <- as_jax(t)
  if (!is_float_dtype(aval_of(t)$dtype))
    jax_error("t must be an array of floats, but got ", aval_str(aval_of(t)), ".",
              class = "jax_type_error")
  cc <- do.call(ode_closure_convert, c(list(func, y0, t[0L]), args))
  do.call(ode_odeint_wrapper, c(list(cc[[1L]], rtol, atol, mxstep, hmax, y0, t), args, cc[[2L]]))
}

#' @rdname odeint
#' @export
experimental_ode_odeint <- odeint

# Port of ode.py:_odeint_wrapper. JAX jits it and sets the default matmul precision to
# highest, which does not change CPU results; here it runs eagerly.
ode_odeint_wrapper <- function(func, rtol, atol, mxstep, hmax, y0, ts, ...) {
  rp <- ravel_pytree(y0)
  func <- ode_ravel_first_arg(func, rp$unravel)
  out <- ode_odeint(func, rtol, atol, mxstep, hmax, rp$flat, ts, ...)
  vmap(rp$unravel)(out)
}
