# jax.scipy.optimize (ports of jax/_src/scipy/optimize/minimize.py, bfgs.py, _lbfgs.py
# and line_search.py). Loop states are named lists (the NamedTuples of JAX).

opt_dot <- function(a, b) jnp_dot(a, b, precision = "highest")
# NamedTuple._replace: update the named fields of a state list.
opt_update <- function(state, upd) {
  state[names(upd)] <- upd
  state
}
opt_replace <- function(state, ...) opt_update(state, list(...))

# Port of line_search.py:_binary_replace.
opt_binary_replace <- function(replace_bit, original, new, keys = names(new)) {
  stats::setNames(lapply(keys, function(k) jnp_where(replace_bit, new[[k]], original[[k]])), keys)
}

# Port of line_search.py:_cubicmin.
opt_cubicmin <- function(a, fa, fpa, b, fb, c, fc) {
  dt <- result_type(a, fa, fpa, b, fb, c, fc)
  C <- fpa
  db <- b - a
  dc <- c - a
  denom <- lax_integer_pow(db * dc, 2L) * (db - dc)
  d1 <- jnp_astype(jnp_stack(list(
    jnp_stack(list(lax_integer_pow(dc, 2L), -lax_integer_pow(db, 2L))),
    jnp_stack(list(-lax_integer_pow(dc, 3L), lax_integer_pow(db, 3L))))), dt)
  d2 <- jnp_astype(jnp_stack(list(fb - fa - C * db, fc - fa - C * dc)), dt)
  AB <- opt_dot(d1, d2) / denom
  A <- AB[0L]
  B <- AB[1L]
  radical <- B * B - 3.0 * A * C
  a + (-B + jnp_sqrt(radical)) / (3.0 * A)
}

# Port of line_search.py:_quadmin.
opt_quadmin <- function(a, fa, fpa, b, fb) {
  D <- fa
  C <- fpa
  db <- b - a
  B <- (fb - D - C * db) / lax_integer_pow(db, 2L)
  a - C / (2.0 * B)
}

# Port of line_search.py:_zoom.
opt_zoom <- function(restricted_func_and_grad, wolfe_one, wolfe_two, a_lo, phi_lo, dphi_lo,
                     a_hi, phi_hi, dphi_hi, g_0, pass_through) {
  # Evaluate the arguments now: a promise first forced inside the traced loop below
  # would leak a tracer.
  force(pass_through)
  state <- list(done = FALSE, failed = FALSE, j = 0L, a_lo = a_lo, phi_lo = phi_lo,
                dphi_lo = dphi_lo, a_hi = a_hi, phi_hi = phi_hi, dphi_hi = dphi_hi,
                a_rec = (a_lo + a_hi) / 2.0, phi_rec = (phi_lo + phi_hi) / 2.0, a_star = 1.0,
                phi_star = phi_lo, dphi_star = dphi_lo, g_star = g_0, nfev = 0L, ngev = 0L)
  delta1 <- 0.2
  delta2 <- 0.1
  body <- function(state) {
    # Body of zoom algorithm. We use boolean arithmetic to avoid using jax.cond
    # so that it works on GPU/TPU.
    dalpha <- state$a_hi - state$a_lo
    a <- jnp_minimum(state$a_hi, state$a_lo)
    b <- jnp_maximum(state$a_hi, state$a_lo)
    cchk <- delta1 * dalpha
    qchk <- delta2 * dalpha
    # This will cause the line search to stop, and since the Wolfe conditions are not
    # satisfied the minimization should stop too.
    threshold <- if (finfo(aval_of(dalpha)$dtype)$bits < 64L) 1e-5 else 1e-10
    state <- opt_replace(state, failed = state$failed | (dalpha <= threshold))
    # Cubmin is sometimes nan, though in this case the bounds check will fail.
    a_j_cubic <- opt_cubicmin(state$a_lo, state$phi_lo, state$dphi_lo, state$a_hi,
                              state$phi_hi, state$a_rec, state$phi_rec)
    use_cubic <- (state$j > 0L) & (a_j_cubic > a + cchk) & (a_j_cubic < b - cchk)
    a_j_quad <- opt_quadmin(state$a_lo, state$phi_lo, state$dphi_lo, state$a_hi, state$phi_hi)
    use_quad <- !use_cubic & (a_j_quad > a + qchk) & (a_j_quad < b - qchk)
    a_j_bisection <- (state$a_lo + state$a_hi) / 2.0
    use_bisection <- !use_cubic & !use_quad
    a_j <- jnp_where(use_cubic, a_j_cubic, state$a_rec)
    a_j <- jnp_where(use_quad, a_j_quad, a_j)
    a_j <- jnp_where(use_bisection, a_j_bisection, a_j)
    # TODO(jakevdp): should we use some sort of fixed-point approach here instead?
    pg <- restricted_func_and_grad(a_j)
    phi_j <- jnp_astype(pg[[1L]], aval_of(state$phi_lo)$dtype)
    dphi_j <- jnp_astype(pg[[2L]], aval_of(state$dphi_lo)$dtype)
    g_j <- jnp_astype(pg[[3L]], aval_of(state$g_star)$dtype)
    state <- opt_replace(state, nfev = state$nfev + 1L, ngev = state$ngev + 1L)
    hi_to_j <- wolfe_one(a_j, phi_j) | (phi_j >= state$phi_lo)
    star_to_j <- wolfe_two(dphi_j) & !hi_to_j
    hi_to_lo <- (dphi_j * (state$a_hi - state$a_lo) >= 0.0) & !hi_to_j & !star_to_j
    lo_to_j <- !hi_to_j & !star_to_j
    state <- opt_update(state, opt_binary_replace(hi_to_j, state, list(
      a_hi = a_j, phi_hi = phi_j, dphi_hi = dphi_j, a_rec = state$a_hi, phi_rec = state$phi_hi)))
    # for termination
    state <- opt_update(state, c(list(done = star_to_j | state$done), opt_binary_replace(
      star_to_j, state, list(a_star = a_j, phi_star = phi_j, dphi_star = dphi_j, g_star = g_j))))
    state <- opt_update(state, opt_binary_replace(hi_to_lo, state, list(
      a_hi = state$a_lo, phi_hi = state$phi_lo, dphi_hi = state$dphi_lo, a_rec = state$a_hi,
      phi_rec = state$phi_hi)))
    state <- opt_update(state, opt_binary_replace(lo_to_j & !hi_to_lo, state, list(
      a_rec = state$a_lo, phi_rec = state$phi_lo)))
    state <- opt_update(state, opt_binary_replace(lo_to_j, state, list(
      a_lo = a_j, phi_lo = phi_j, dphi_lo = dphi_j)))
    state <- opt_replace(state, j = state$j + 1L)
    # Choose higher cutoff for maxiter than Scipy as Jax takes longer to find the same
    # value (possibly floating point issues).
    opt_replace(state, failed = state$failed | (state$j >= 30L))
  }
  lax_while_loop(function(state) !state$done & !pass_through & !state$failed, body, state)
}

#' Inexact line search satisfying the strong Wolfe conditions
#'
#' Port of `jax._src.scipy.optimize.line_search.line_search` (algorithm 3.5 of Nocedal
#' and Wright, Numerical Optimization), the line search used by
#' [scipy_optimize_minimize()]. It is not part of the public `jax.scipy.optimize`
#' namespace.
#' @param f Function of `x` returning a scalar.
#' @param xk Starting point.
#' @param pk Search direction.
#' @param old_fval,gfk Value and gradient of `f` at `xk` (computed if `NULL`).
#' @param old_old_fval Value of `f` at the previous point, used for the initial step.
#' @param c1,c2 Constants of the Wolfe conditions.
#' @param maxiter Maximum number of iterations.
#' @return A named list with `failed`, `nit`, `nfev`, `ngev`, `k`, `a_k` (the step),
#'   `f_k`, `g_k` (value and gradient at the new point) and `status` (0: converged,
#'   1: the zoom failed, 3: `maxiter` reached).
#' @export
scipy_optimize_line_search <- function(f, xk, pk, old_fval = NULL, old_old_fval = NULL,
                                       gfk = NULL, c1 = 1e-4, c2 = 0.9, maxiter = 20L) {
  # Port of jax/_src/scipy/optimize/line_search.py:line_search
  xp <- promote_dtypes_inexact(list(xk, pk))
  xk <- xp[[1L]]
  pk <- xp[[2L]]
  pdt <- aval_of(pk)$dtype
  restricted_func_and_grad <- function(t) {
    t <- lax_convert_element_type(as_jax(t), pdt)
    vg <- value_and_grad(f)(xk + t * pk)
    list(vg[[1L]], jnp_real(opt_dot(vg[[2L]], pk)), vg[[2L]])
  }
  computed <- is.null(old_fval) || is.null(gfk)
  if (computed) {
    pg <- restricted_func_and_grad(0L)
    phi_0 <- pg[[1L]]
    dphi_0 <- pg[[2L]]
    gfk <- pg[[3L]]
  } else {
    phi_0 <- as_jax(old_fval)
    gfk <- as_jax(gfk)
    dphi_0 <- jnp_real(opt_dot(gfk, pk))
  }
  if (!is.null(old_old_fval)) {
    candidate_start_value <- (1.01 * 2) * (phi_0 - old_old_fval) / dphi_0
    start_value <- jnp_where(candidate_start_value > 1, 1.0, candidate_start_value)
  } else {
    start_value <- 1L
  }
  wolfe_one <- function(a_i, phi_i) phi_i > phi_0 + c1 * a_i * dphi_0
  wolfe_two <- function(dphi_i) jnp_abs(dphi_i) <= -c2 * dphi_0
  state <- list(done = FALSE, failed = FALSE, i = 1L, a_i1 = 0.0, phi_i1 = phi_0,
                dphi_i1 = dphi_0, nfev = if (computed) 1L else 0L,
                ngev = if (computed) 1L else 0L, a_star = 0.0, phi_star = phi_0,
                dphi_star = dphi_0, g_star = gfk)
  star_keys <- c("a_star", "phi_star", "dphi_star", "g_star")
  body <- function(state) {
    # no amax in this version, we just double as in scipy.
    # unlike original algorithm we do our next choice at the start of this loop
    a_i <- jnp_where(state$i == 1L, start_value, state$a_i1 * 2.0)
    pg <- restricted_func_and_grad(a_i)
    phi_i <- pg[[1L]]
    dphi_i <- pg[[2L]]
    g_i <- pg[[3L]]
    state <- opt_replace(state, nfev = state$nfev + 1L, ngev = state$ngev + 1L)
    star_to_zoom1 <- wolfe_one(a_i, phi_i) | ((phi_i >= state$phi_i1) & (state$i > 1L))
    star_to_i <- wolfe_two(dphi_i) & !star_to_zoom1
    star_to_zoom2 <- (dphi_i >= 0.0) & !star_to_zoom1 & !star_to_i
    zoom1 <- opt_zoom(restricted_func_and_grad, wolfe_one, wolfe_two, state$a_i1, state$phi_i1,
                      state$dphi_i1, a_i, phi_i, dphi_i, gfk, !star_to_zoom1)
    state <- opt_replace(state, nfev = state$nfev + zoom1$nfev, ngev = state$ngev + zoom1$ngev)
    zoom2 <- opt_zoom(restricted_func_and_grad, wolfe_one, wolfe_two, a_i, phi_i, dphi_i,
                      state$a_i1, state$phi_i1, state$dphi_i1, gfk, !star_to_zoom2)
    state <- opt_replace(state, nfev = state$nfev + zoom2$nfev, ngev = state$ngev + zoom2$ngev)
    state <- opt_update(state, c(list(
      done = star_to_zoom1 | state$done, failed = (star_to_zoom1 & zoom1$failed) | state$failed),
      opt_binary_replace(star_to_zoom1, state, zoom1, star_keys)))
    state <- opt_update(state, c(list(done = star_to_i | state$done), opt_binary_replace(
      star_to_i, state, list(a_star = a_i, phi_star = phi_i, dphi_star = dphi_i, g_star = g_i))))
    state <- opt_update(state, c(list(
      done = star_to_zoom2 | state$done, failed = (star_to_zoom2 & zoom2$failed) | state$failed),
      opt_binary_replace(star_to_zoom2, state, zoom2, star_keys)))
    opt_replace(state, i = state$i + 1L, a_i1 = a_i, phi_i1 = phi_i, dphi_i1 = dphi_i)
  }
  state <- lax_while_loop(function(state) !state$done & (state$i <= maxiter) & !state$failed,
                          body, state)
  status <- jnp_where(state$failed, 1L, jnp_where(state$i > maxiter, 3L, 0L))
  alpha_k <- as_jax(state$a_star)
  # Step sizes which are too small causes the optimizer to get stuck with a direction of
  # zero in <64 bit mode; avoid with a floor on minimum step size.
  if (finfo(aval_of(alpha_k)$dtype)$bits != 64L)
    alpha_k <- jnp_where(jnp_abs(alpha_k) < 1e-8, jnp_sign(alpha_k) * 1e-8, alpha_k)
  list(failed = state$failed | !state$done, nit = state$i - 1L, nfev = state$nfev,
       ngev = state$ngev, k = state$i, a_k = alpha_k, f_k = state$phi_star, g_k = state$g_star,
       status = status)
}

# Port of jax/_src/scipy/optimize/bfgs.py:minimize_bfgs.
opt_minimize_bfgs <- function(fun, x0, maxiter = NULL, norm = Inf, gtol = 1e-5,
                              line_search_maxiter = 10L) {
  x0 <- as_jax(x0)
  if (is.null(maxiter)) maxiter <- prod_int(aval_of(x0)$shape) * 200L
  d <- aval_of(x0)$shape[1L]
  initial_H <- jnp_eye(d, dtype = aval_of(x0)$dtype)
  vg <- value_and_grad(fun)(x0)
  f_0 <- vg[[1L]]
  g_0 <- vg[[2L]]
  state <- list(converged = jnp_linalg_norm(g_0, ord = norm) < gtol, failed = FALSE, k = 0L,
                nfev = 1L, ngev = 1L, nhev = 0L, x_k = x0, f_k = f_0, g_k = g_0, H_k = initial_H,
                old_old_fval = f_0 + jnp_linalg_norm(g_0) / 2, status = 0L,
                line_search_status = 0L)
  cond_fun <- function(state) !state$converged & !state$failed & (state$k < maxiter)
  body_fun <- function(state) {
    p_k <- -opt_dot(state$H_k, state$g_k)
    ls <- scipy_optimize_line_search(fun, state$x_k, p_k, old_fval = state$f_k,
                                     old_old_fval = state$old_old_fval, gfk = state$g_k,
                                     maxiter = line_search_maxiter)
    state <- opt_replace(state, nfev = state$nfev + ls$nfev, ngev = state$ngev + ls$ngev,
                         failed = ls$failed, line_search_status = ls$status)
    s_k <- ls$a_k * p_k
    x_kp1 <- state$x_k + s_k
    f_kp1 <- ls$f_k
    g_kp1 <- ls$g_k
    y_k <- g_kp1 - state$g_k
    rho_k <- jnp_reciprocal(opt_dot(y_k, s_k))
    sy_k <- lax_expand_dims(s_k, 1L) * lax_expand_dims(y_k, 0L)
    w <- jnp_eye(d, dtype = aval_of(rho_k)$dtype) - rho_k * sy_k
    # einsum("ij,jk,lk", w, H_k, w), contracted in the order opt_einsum picks.
    H_kp1 <- opt_dot(opt_dot(w, state$H_k), jnp_transpose(w)) +
      rho_k * lax_expand_dims(s_k, 1L) * lax_expand_dims(s_k, 0L)
    H_kp1 <- jnp_where(jnp_isfinite(rho_k), H_kp1, state$H_k)
    converged <- jnp_linalg_norm(g_kp1, ord = norm) < gtol
    opt_replace(state, converged = converged, k = state$k + 1L, x_k = x_kp1, f_k = f_kp1,
                g_k = g_kp1, H_k = H_kp1, old_old_fval = state$f_k)
  }
  state <- lax_while_loop(cond_fun, body_fun, state)
  status <- jnp_where(state$converged, 0L, jnp_where(state$k == maxiter, 1L, jnp_where(
    state$failed, 2L + state$line_search_status, -1L)))
  opt_replace(state, status = status)
}

# Port of jax/_src/scipy/optimize/_lbfgs.py:_minimize_lbfgs.
opt_minimize_lbfgs <- function(fun, x0, maxiter = NULL, norm = Inf, maxcor = 10L,
                               ftol = 2.220446049250313e-09, gtol = 1e-5, maxfun = NULL,
                               maxgrad = NULL, maxls = 20L) {
  x0 <- as_jax(x0)
  d <- aval_of(x0)$shape[1L]
  dt <- aval_of(x0)$dtype
  if (is.null(maxiter) && is.null(maxfun) && is.null(maxgrad)) maxiter <- d * 200L
  if (is.null(maxiter)) maxiter <- Inf
  if (is.null(maxfun)) maxfun <- Inf
  if (is.null(maxgrad)) maxgrad <- Inf
  vg <- value_and_grad(fun)(x0)
  state_initial <- list(
    converged = FALSE, failed = FALSE, k = 0L, nfev = 1L, ngev = 1L, x_k = x0, f_k = vg[[1L]],
    g_k = vg[[2L]], s_history = jnp_zeros(c(maxcor, d), dtype = dt),
    y_history = jnp_zeros(c(maxcor, d), dtype = dt), rho_history = jnp_zeros(maxcor, dtype = dt),
    gamma = 1.0, status = 0L, ls_status = 0L)
  cond_fun <- function(state) !state$converged & !state$failed
  body_fun <- function(state) {
    p_k <- opt_two_loop_recursion(state)
    ls <- scipy_optimize_line_search(f = fun, xk = state$x_k, pk = p_k, old_fval = state$f_k,
                                     gfk = state$g_k, maxiter = maxls)
    s_k <- jnp_astype(ls$a_k, aval_of(p_k)$dtype) * p_k
    x_kp1 <- state$x_k + s_k
    f_kp1 <- ls$f_k
    g_kp1 <- ls$g_k
    y_k <- g_kp1 - state$g_k
    rho_k_inv <- jnp_real(opt_dot(y_k, s_k))
    rho_k <- jnp_astype(jnp_reciprocal(rho_k_inv), aval_of(y_k)$dtype)
    gamma <- rho_k_inv / jnp_real(opt_dot(jnp_conjugate(y_k), y_k))
    status <- jnp_array(0L)
    status <- jnp_where(state$f_k - f_kp1 < ftol, 4L, status)
    status <- jnp_where(state$ngev >= maxgrad, 3L, status)
    status <- jnp_where(state$nfev >= maxfun, 2L, status)
    status <- jnp_where(state$k >= maxiter, 1L, status)
    status <- jnp_where(ls$failed, 5L, status)
    converged <- jnp_linalg_norm(g_kp1, ord = norm) < gtol
    list(converged = converged, failed = (status > 0L) & !converged, k = state$k + 1L,
         nfev = state$nfev + ls$nfev, ngev = state$ngev + ls$ngev,
         x_k = jnp_astype(x_kp1, aval_of(state$x_k)$dtype),
         f_k = jnp_astype(f_kp1, aval_of(state$f_k)$dtype),
         g_k = jnp_astype(g_kp1, aval_of(state$g_k)$dtype),
         s_history = opt_update_history(state$s_history, s_k),
         y_history = opt_update_history(state$y_history, y_k),
         rho_history = opt_update_history(state$rho_history, rho_k),
         gamma = jnp_astype(gamma, aval_of(state$g_k)$dtype),
         status = jnp_where(converged, 0L, status), ls_status = ls$status)
  }
  lax_while_loop(cond_fun, body_fun, state_initial)
}

# Port of _lbfgs.py:_two_loop_recursion.
opt_two_loop_recursion <- function(state) {
  dt <- aval_of(state$rho_history)$dtype
  his_size <- aval_of(state$rho_history)$shape[1L]
  curr_size <- jnp_where(state$k < his_size, state$k, his_size)
  q <- -jnp_conjugate(state$g_k)
  a_his <- jnp_zeros_like(state$rho_history)
  qa <- lax_fori_loop(0L, curr_size, function(j, carry) {
    i <- his_size - 1L - j
    a_i <- state$rho_history[i] *
      jnp_astype(jnp_real(opt_dot(jnp_conjugate(state$s_history[i]), carry[[1L]])), dt)
    list(carry[[1L]] - a_i * jnp_conjugate(state$y_history[i]), at_set(carry[[2L]], i, value = a_i))
  }, list(q, a_his))
  a_his <- qa[[2L]]
  q <- state$gamma * qa[[1L]]
  lax_fori_loop(0L, curr_size, function(j, q) {
    i <- his_size - curr_size + j
    b_i <- state$rho_history[i] * jnp_astype(jnp_real(opt_dot(state$y_history[i], q)), dt)
    q + (a_his[i] - b_i) * state$s_history[i]
  }, q)
}

# Port of _lbfgs.py:_update_history_vectors and _update_history_scalars:
# jnp.roll(history, -1, axis = 0).at[-1].set(new).
opt_update_history <- function(history, new) {
  n <- aval_of(history)$shape[1L]
  jnp_concatenate(list(lax_slice_in_dim(history, 1L, n), lax_expand_dims(new, 0L)), axis = 0L)
}

#' Results of an optimization
#'
#' Port of the `jax.scipy.optimize.OptimizeResults` named tuple: a named list (a pytree,
#' so it can be returned from [jit()]).
#' @param x Solution.
#' @param success Whether the optimization succeeded.
#' @param status Solver specific status code (0 means converged).
#' @param fun Function value at `x`.
#' @param jac Gradient at `x`.
#' @param hess_inv Approximate inverse Hessian at `x` (`NULL` for L-BFGS).
#' @param nfev,njev Numbers of function and gradient evaluations.
#' @param nit Number of iterations.
#' @return A named list with these fields.
#' @export
scipy_optimize_OptimizeResults <- function(x, success, status, fun, jac, hess_inv, nfev, njev,
                                           nit) {
  list(x = x, success = success, status = status, fun = fun, jac = jac, hess_inv = hess_inv,
       nfev = nfev, njev = njev, nit = nit)
}

#' Minimize a scalar function of one or more variables
#'
#' Port of `jax.scipy.optimize.minimize`. Supported methods are `"BFGS"` and
#' `"l-bfgs-experimental-do-not-rely-on-this"` (case insensitive). The whole
#' optimization is one [lax_while_loop()], so the function can be wrapped in [jit()]
#' and mapped with [vmap()].
#'
#' Options for BFGS: `maxiter` (default `200 * length(x0)`), `norm` (order of the
#' gradient norm, default `Inf`), `gtol` (default `1e-5`), `line_search_maxiter`
#' (default 10). Status codes: 0 converged, 1 `maxiter` reached, 2 + the line search
#' status if the line search failed, -1 otherwise.
#'
#' Options for L-BFGS: `maxiter`, `norm`, `maxcor` (history size, default 10), `ftol`,
#' `gtol`, `maxfun`, `maxgrad`, `maxls` (line search iterations, default 20).
#' @param fun Objective `fun(x, ...)` returning a scalar; `x` is a 1-D array.
#' @param x0 Initial guess (1-D).
#' @param args List of extra arguments passed to `fun` after `x`.
#' @param method Solver name.
#' @param tol Ignored (as in JAX); use the solver options.
#' @param options Named list of solver options.
#' @return A [scipy_optimize_OptimizeResults()] list.
#' @export
scipy_optimize_minimize <- function(fun, x0, args = list(), method, tol = NULL, options = NULL) {
  # Port of jax/_src/scipy/optimize/minimize.py:minimize
  if (is.null(options)) options <- list()
  if (!is.list(args) || !is.null(oldClass(args)))
    jax_error("args argument to jax.scipy.optimize.minimize must be a tuple, got ",
              paste(format(args), collapse = " "), class = "jax_type_error")
  fun_with_args <- function(x) do.call(fun, c(list(x), args))
  if (tolower(method) == "bfgs") {
    results <- do.call(opt_minimize_bfgs, c(list(fun_with_args, x0), options))
    success <- results$converged & !results$failed
    return(scipy_optimize_OptimizeResults(
      x = results$x_k, success = success, status = results$status, fun = results$f_k,
      jac = results$g_k, hess_inv = results$H_k, nfev = results$nfev, njev = results$ngev,
      nit = results$k))
  }
  if (tolower(method) == "l-bfgs-experimental-do-not-rely-on-this") {
    results <- do.call(opt_minimize_lbfgs, c(list(fun_with_args, x0), options))
    success <- results$converged & !results$failed
    return(scipy_optimize_OptimizeResults(
      x = results$x_k, success = success, status = results$status, fun = results$f_k,
      jac = results$g_k, hess_inv = NULL, nfev = results$nfev, njev = results$ngev,
      nit = results$k))
  }
  jax_error("Method ", method, " not recognized", class = "jax_value_error")
}
