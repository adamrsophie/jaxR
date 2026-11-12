# jax.example_libraries.optimizers (port of jax/example_libraries/optimizers.py).
#
# An optimizer is a triple of functions list(init_fn, update_fn, params_fn):
#   init_fn(params) -> opt_state
#   update_fn(step, grads, opt_state) -> opt_state
#   params_fn(opt_state) -> params
# The single array versions written by each optimizer below are lifted to pytrees of
# parameters by optimizers_optimizer(), which stores the state as an OptimizerState: the
# flattened per parameter states with the parameter treedef and the per parameter state
# treedefs.

#' Optimizer state and triple
#'
#' `optimizers_OptimizerState` ports the `OptimizerState` named tuple: a pytree node with
#' one child (`packed_state`, the list of flattened per parameter states) and the
#' parameter treedef and per parameter state treedefs as auxiliary data.
#' `optimizers_Optimizer` ports the `Optimizer` named tuple
#' `list(init_fn, update_fn, params_fn)`. `optimizers_optimizer` ports the decorator that
#' turns a constructor of single array optimizer functions into one that works on pytrees
#' of parameters (nested lists of arrays).
#' @param packed_state List of flattened per parameter states.
#' @param tree_def Treedef of the parameters.
#' @param subtree_defs List of treedefs of the per parameter states.
#' @param init_fn,update_fn,params_fn The optimizer functions: `init_fn(params)`,
#'   `update_fn(step, grads, opt_state)` and `params_fn(opt_state)` (the `get_params`
#'   function).
#' @param opt_maker A function returning `list(init, update, get_params)` that work on
#'   single arrays: `init(x0)` returns a state pytree, `update(i, g, state)` the new state
#'   and `get_params(state)` the array.
#' @return `optimizers_OptimizerState()` an object of class `OptimizerState`,
#'   `optimizers_Optimizer()` a named list and `optimizers_optimizer()` an optimizer
#'   constructor.
#' @examples
#' \dontrun{
#' my_sgd <- optimizers_optimizer(function(step_size) {
#'   list(function(x0) x0, function(i, g, x) x - step_size * g, function(x) x)
#' })
#' opt <- my_sgd(0.1)
#' state <- opt$init_fn(list(w = jnp_ones(2)))
#' }
#' @name optimizers_state
NULL

#' @rdname optimizers_state
#' @export
optimizers_OptimizerState <- function(packed_state, tree_def, subtree_defs) {
  structure(list(packed_state = packed_state, tree_def = tree_def, subtree_defs = subtree_defs),
            class = "OptimizerState")
}

register_pytree_node("OptimizerState",
  function(x) list(children = list(x$packed_state), aux = list(x$tree_def, x$subtree_defs)),
  function(aux, children) optimizers_OptimizerState(children[[1L]], aux[[1L]], aux[[2L]]))

#' @rdname optimizers_state
#' @export
optimizers_Optimizer <- function(init_fn, update_fn, params_fn) {
  list(init_fn = init_fn, update_fn = update_fn, params_fn = params_fn)
}

#' @rdname optimizers_state
#' @export
optimizers_optimizer <- function(opt_maker) {
  force(opt_maker)
  function(...) {
    fns <- opt_maker(...)
    init <- fns[[1L]]
    update <- fns[[2L]]
    get_params <- fns[[3L]]

    tree_init <- function(x0_tree) {
      fl <- tree_flatten(x0_tree)
      states <- lapply(lapply(fl$leaves, init), tree_flatten)
      optimizers_OptimizerState(lapply(states, `[[`, "leaves"), fl$treedef,
                                lapply(states, `[[`, "treedef"))
    }

    tree_update <- function(i, grad_tree, opt_state) {
      tree <- opt_state$tree_def
      subtrees <- opt_state$subtree_defs
      gfl <- tree_flatten(grad_tree)
      if (!treedef_equal(gfl$treedef, tree))
        jax_error("optimizer update function was passed a gradient tree that did not match ",
                  "the parameter tree structure with which it was initialized: parameter ",
                  "tree ", format(tree), " and grad tree ", format(gfl$treedef), ".",
                  class = "jax_type_error")
      states <- Map(tree_unflatten, subtrees, opt_state$packed_state)
      new_states <- lapply(Map(function(g, s) update(i, g, s), gfl$leaves, states), tree_flatten)
      for (k in seq_along(new_states)) {
        if (!treedef_equal(new_states[[k]]$treedef, subtrees[[k]]))
          jax_error("optimizer update function produced an output structure that did not ",
                    "match its input structure: input ", format(subtrees[[k]]), " and output ",
                    format(new_states[[k]]$treedef), ".", class = "jax_type_error")
      }
      optimizers_OptimizerState(lapply(new_states, `[[`, "leaves"), tree, subtrees)
    }

    tree_get_params <- function(opt_state) {
      states <- Map(tree_unflatten, opt_state$subtree_defs, opt_state$packed_state)
      tree_unflatten(opt_state$tree_def, lapply(unname(states), get_params))
    }

    optimizers_Optimizer(tree_init, tree_update, tree_get_params)
  }
}

# ---- optimizers ---------------------------------------------------------------------

# jnp.asarray(b, dtype) ** (i + 1): a concrete integer step uses integer_pow (as in
# jnp.power); a traced step uses lax.pow.
opt_beta_pow <- function(b, dt, i) jnp_power(as_jax(b, dt), i + 1L)

#' Optimizers
#'
#' Ports of the optimizers of `jax.example_libraries.optimizers`. Each constructor returns
#' an optimizer `list(init_fn, update_fn, params_fn)` (see [optimizers_Optimizer()]) that
#' works on pytrees of parameters: `init_fn(params)` creates the state,
#' `update_fn(step, grads, opt_state)` applies one update (`step` is the 0-based
#' iteration index) and `params_fn(opt_state)` extracts the parameters.
#' @param step_size Positive scalar, or a schedule (a function of the iteration index
#'   returning a positive scalar).
#' @param mass Momentum coefficient (`momentum`, `nesterov`).
#' @param momentum Momentum parameter (`adagrad`, `rmsprop_momentum`, `sm3`).
#' @param gamma Decay parameter (`rmsprop`, `rmsprop_momentum`).
#' @param eps Small constant for numerical stability.
#' @param b1,b2 Exponential decay rates of the first and second moment estimates (`adam`,
#'   `adamax`).
#' @return An optimizer `list(init_fn, update_fn, params_fn)`.
#' @examples
#' \dontrun{
#' opt <- optimizers_adam(1e-3)
#' state <- opt$init_fn(list(w = jnp_ones(3), b = 0))
#' grads <- list(w = jnp_ones(3), b = 1)
#' state <- opt$update_fn(0L, grads, state)
#' opt$params_fn(state)
#' }
#' @name optimizers
NULL

opt_sgd <- optimizers_optimizer(function(step_size) {
  step_size <- optimizers_make_schedule(step_size)
  optimizers_Optimizer(
    function(x0) x0,
    function(i, g, x) x - step_size(i) * g,
    function(x) x)
})

#' @rdname optimizers
#' @export
optimizers_sgd <- function(step_size) opt_sgd(step_size)

opt_momentum <- optimizers_optimizer(function(step_size, mass) {
  force(mass)
  step_size <- optimizers_make_schedule(step_size)
  list(
    function(x0) list(x0, jnp_zeros_like(x0)),
    function(i, g, state) {
      velocity <- mass * state[[2L]] + g
      list(state[[1L]] - step_size(i) * velocity, velocity)
    },
    function(state) state[[1L]])
})

#' @rdname optimizers
#' @export
optimizers_momentum <- function(step_size, mass) opt_momentum(step_size, mass)

opt_nesterov <- optimizers_optimizer(function(step_size, mass) {
  force(mass)
  step_size <- optimizers_make_schedule(step_size)
  list(
    function(x0) list(x0, jnp_zeros_like(x0)),
    function(i, g, state) {
      velocity <- mass * state[[2L]] + g
      list(state[[1L]] - step_size(i) * (mass * velocity + g), velocity)
    },
    function(state) state[[1L]])
})

#' @rdname optimizers
#' @export
optimizers_nesterov <- function(step_size, mass) opt_nesterov(step_size, mass)

opt_adagrad <- optimizers_optimizer(function(step_size, momentum = 0.9) {
  force(momentum)
  step_size <- optimizers_make_schedule(step_size)
  list(
    function(x0) list(x0, jnp_zeros_like(x0), jnp_zeros_like(x0)),
    function(i, g, state) {
      g_sq <- state[[2L]] + jnp_square(g)
      g_sq_inv_sqrt <- jnp_where(g_sq > 0L, 1 / jnp_sqrt(g_sq), 0.0)
      m <- (1 - momentum) * (g * g_sq_inv_sqrt) + momentum * state[[3L]]
      list(state[[1L]] - step_size(i) * m, g_sq, m)
    },
    function(state) state[[1L]])
})

#' @rdname optimizers
#' @export
optimizers_adagrad <- function(step_size, momentum = 0.9) opt_adagrad(step_size, momentum)

opt_rmsprop <- optimizers_optimizer(function(step_size, gamma = 0.9, eps = 1e-8) {
  force(gamma); force(eps)
  step_size <- optimizers_make_schedule(step_size)
  list(
    function(x0) list(x0, jnp_zeros_like(x0)),
    function(i, g, state) {
      avg_sq_grad <- state[[2L]] * gamma + jnp_square(g) * (1 - gamma)
      list(state[[1L]] - step_size(i) * g / jnp_sqrt(avg_sq_grad + eps), avg_sq_grad)
    },
    function(state) state[[1L]])
})

#' @rdname optimizers
#' @export
optimizers_rmsprop <- function(step_size, gamma = 0.9, eps = 1e-8) {
  opt_rmsprop(step_size, gamma, eps)
}

opt_rmsprop_momentum <- optimizers_optimizer(function(step_size, gamma = 0.9,
                                                             eps = 1e-8, momentum = 0.9) {
  force(gamma); force(eps); force(momentum)
  step_size <- optimizers_make_schedule(step_size)
  list(
    function(x0) list(x0, jnp_zeros_like(x0), jnp_zeros_like(x0)),
    function(i, g, state) {
      avg_sq_grad <- state[[2L]] * gamma + jnp_square(g) * (1 - gamma)
      mom <- momentum * state[[3L]] + step_size(i) * g / jnp_sqrt(avg_sq_grad + eps)
      list(state[[1L]] - mom, avg_sq_grad, mom)
    },
    function(state) state[[1L]])
})

#' @rdname optimizers
#' @export
optimizers_rmsprop_momentum <- function(step_size, gamma = 0.9, eps = 1e-8, momentum = 0.9) {
  opt_rmsprop_momentum(step_size, gamma, eps, momentum)
}

opt_adam <- optimizers_optimizer(function(step_size, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  force(b1); force(b2); force(eps)
  step_size <- optimizers_make_schedule(step_size)
  list(
    function(x0) list(x0, jnp_zeros_like(x0), jnp_zeros_like(x0)),
    function(i, g, state) {
      m <- (1 - b1) * g + b1 * state[[2L]]  # First moment estimate.
      v <- (1 - b2) * jnp_square(g) + b2 * state[[3L]]  # Second moment estimate.
      dt <- dtype(m)
      mhat <- m / (1L - opt_beta_pow(b1, dt, i))  # Bias correction.
      vhat <- v / (1L - opt_beta_pow(b2, dt, i))
      list(state[[1L]] - step_size(i) * mhat / (jnp_sqrt(vhat) + eps), m, v)
    },
    function(state) state[[1L]])
})

#' @rdname optimizers
#' @export
optimizers_adam <- function(step_size, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  opt_adam(step_size, b1, b2, eps)
}

opt_adamax <- optimizers_optimizer(function(step_size, b1 = 0.9, b2 = 0.999,
                                                   eps = 1e-8) {
  force(b1); force(b2); force(eps)
  step_size <- optimizers_make_schedule(step_size)
  list(
    function(x0) list(x0, jnp_zeros_like(x0), jnp_zeros_like(x0)),
    function(i, g, state) {
      m <- (1 - b1) * g + b1 * state[[2L]]  # First moment estimate.
      u <- jnp_maximum(b2 * state[[3L]], jnp_abs(g))  # Exponentially weighted infinity norm.
      x <- state[[1L]] - (step_size(i) / (1L - opt_beta_pow(b1, dtype(m), i))) * m / (u + eps)
      list(x, m, u)
    },
    function(state) state[[1L]])
})

#' @rdname optimizers
#' @export
optimizers_adamax <- function(step_size, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  opt_adamax(step_size, b1, b2, eps)
}

opt_sm3 <- optimizers_optimizer(function(step_size, momentum = 0.9) {
  force(momentum)
  step_size <- optimizers_make_schedule(step_size)
  # x[idx] with None everywhere except a full slice at `axis`.
  broadcast_into <- function(ndim, x, axis) {
    if (ndim == 0L) x else lax_expand_dims(x, setdiff(seq_len(ndim) - 1L, axis))
  }
  list(
    function(x0) {
      x0 <- as_jax(x0)
      # x_shape is a tuple of ints upstream: a list of R integers here.
      x_shape <- as.list(dim(x0))
      if (!length(dim(x0))) x0 <- jnp_reshape(x0, 1L)
      vs <- lapply(dim(x0), function(sz) jnp_zeros(sz, dtype = dtype(x0)))
      list(x0, jnp_zeros_like(x0), vs, x_shape)
    },
    function(i, g, state) {
      x <- state[[1L]]
      g <- as_jax(g)
      gnd <- length(dim(g))
      vs <- Map(function(v, k) broadcast_into(gnd, v, k), state[[3L]], seq_along(state[[3L]]) - 1L)
      accum <- Reduce(jnp_minimum, vs) + jnp_square(g)
      accum_inv_sqrt <- jnp_where(accum > 0L, 1 / jnp_sqrt(accum), 0L)
      m <- (1 - momentum) * (g * accum_inv_sqrt) + momentum * state[[2L]]
      x <- x - step_size(i) * m
      nd <- length(dim(x))
      vs <- lapply(seq_len(nd) - 1L, function(j) jnp_max(accum, setdiff(seq_len(nd) - 1L, j)))
      list(x, m, vs, state[[4L]])
    },
    function(state) jnp_reshape(state[[1L]], canon_shape(state[[4L]]))
  )
})

#' @rdname optimizers
#' @export
optimizers_sm3 <- function(step_size, momentum = 0.9) opt_sm3(step_size, momentum)

# ---- learning rate schedules ----------------------------------------------------------

#' Learning rate schedules
#'
#' Ports of the step size schedules of `jax.example_libraries.optimizers`. A schedule is
#' a function of the (0-based) iteration index returning the step size.
#' `optimizers_make_schedule` returns a schedule unchanged and turns a scalar into a
#' constant schedule.
#' @param step_size Initial (or constant) step size.
#' @param decay_steps,decay_rate Decay parameters: the step size is multiplied by
#'   `decay_rate` every `decay_steps` iterations (`exponential_decay`) or divided by
#'   `1 + decay_rate * i / decay_steps` (`inverse_time_decay`).
#' @param staircase Decay in discrete steps (`floor(i / decay_steps)`).
#' @param final_step_size,power Final value and power of `polynomial_decay`.
#' @param boundaries,values Boundaries (one shorter than `values`) and step sizes of
#'   `piecewise_constant`.
#' @param scalar_or_schedule A scalar or a schedule.
#' @return A schedule function.
#' @examples
#' \dontrun{
#' s <- optimizers_piecewise_constant(c(10, 20), c(1, 0.1, 0.01))
#' s(15L)
#' }
#' @name optimizers_schedules
NULL

#' @rdname optimizers_schedules
#' @export
optimizers_constant <- function(step_size) {
  force(step_size)
  function(i) step_size
}

#' @rdname optimizers_schedules
#' @export
optimizers_exponential_decay <- function(step_size, decay_steps, decay_rate) {
  force(step_size); force(decay_steps); force(decay_rate)
  function(i) step_size * decay_rate^(i / decay_steps)
}

#' @rdname optimizers_schedules
#' @export
optimizers_inverse_time_decay <- function(step_size, decay_steps, decay_rate, staircase = FALSE) {
  force(step_size); force(decay_steps); force(decay_rate)
  if (staircase) return(function(i) step_size / (1L + decay_rate * jnp_floor(i / decay_steps)))
  function(i) step_size / (1L + decay_rate * i / decay_steps)
}

#' @rdname optimizers_schedules
#' @export
optimizers_polynomial_decay <- function(step_size, decay_steps, final_step_size, power = 1.0) {
  force(step_size); force(decay_steps); force(final_step_size); force(power)
  function(step_num) {
    step_num <- jnp_minimum(step_num, decay_steps)
    base <- 1L - step_num / decay_steps
    # A float power (Python float upstream) uses lax.pow; an integer power integer_pow.
    step_mult <- if (is.double(power) && !is_jax_value(power)) {
      a <- promote_args_numeric(base, power)
      lax_pow(a[[1L]], a[[2L]])
    } else base^power
    step_mult * (step_size - final_step_size) + final_step_size
  }
}

#' @rdname optimizers_schedules
#' @export
optimizers_piecewise_constant <- function(boundaries, values) {
  boundaries <- jnp_array(boundaries)
  values <- jnp_array(values)
  if (!(length(dim(boundaries)) == 1L && length(dim(values)) == 1L))
    jax_error("boundaries and values must be sequences", class = "jax_value_error")
  if (dim(boundaries)[1L] != dim(values)[1L] - 1L)
    jax_error("boundaries length must be one shorter than values length",
              class = "jax_value_error")
  function(i) {
    idx <- jnp_sum(i > boundaries)
    lax_reshape(lax_dynamic_slice(values, list(idx), 1L), integer(0))
  }
}

#' @rdname optimizers_schedules
#' @export
optimizers_make_schedule <- function(scalar_or_schedule) {
  if (is.function(scalar_or_schedule)) return(scalar_or_schedule)
  if (length(opt_shape(scalar_or_schedule)) == 0L)
    return(optimizers_constant(scalar_or_schedule))
  jax_error(class(scalar_or_schedule)[1L], class = "jax_type_error")
}

opt_shape <- function(x) if (is_jax_value(x)) aval_of(x)$shape else r_shape(x)

# ---- utilities ------------------------------------------------------------------------

#' Gradient utilities
#'
#' `optimizers_l2_norm` ports `l2_norm`, the l2 norm of all leaves of a pytree of arrays
#' (useful for weight decay). `optimizers_clip_grads` ports `clip_grads`: it rescales a
#' pytree of gradients whose l2 norm exceeds `max_norm` to norm `max_norm`.
#' @param tree,grad_tree Pytree of arrays.
#' @param max_norm Maximum norm.
#' @return A scalar array (`l2_norm`) or a pytree with the structure of `grad_tree`.
#' @examples
#' \dontrun{
#' optimizers_clip_grads(list(jnp_array(c(3, 4))), 1)
#' }
#' @name optimizers_utils
NULL

#' @rdname optimizers_utils
#' @export
optimizers_l2_norm <- function(tree) {
  leaves <- tree_leaves(tree)
  jnp_sqrt(Reduce(`+`, lapply(leaves, function(x) jnp_vdot(x, x)), 0L))
}

#' @rdname optimizers_utils
#' @export
optimizers_clip_grads <- function(grad_tree, max_norm) {
  norm <- optimizers_l2_norm(grad_tree)
  tree_map(function(g) jnp_where(norm < max_norm, g, g * (max_norm / norm)), grad_tree)
}

# ---- serialization utilities ------------------------------------------------------------

#' Optimizer state serialization
#'
#' `optimizers_unpack_optimizer_state` converts an `OptimizerState` into a pytree with the
#' structure of the parameters whose leaves are `JoinPoint` objects holding each
#' parameter's state (a pytree), useful for serialization.
#' `optimizers_pack_optimizer_state` is its inverse. `optimizers_JoinPoint` marks the
#' boundary between the two nested pytrees (it is a leaf).
#' @param opt_state An `OptimizerState`.
#' @param marked_pytree A pytree with `JoinPoint` leaves.
#' @param subtree The pytree held by a `JoinPoint`.
#' @return A pytree with `JoinPoint` leaves, an `OptimizerState`, or a `JoinPoint`.
#' @examples
#' \dontrun{
#' opt <- optimizers_momentum(0.1, 0.9)
#' state <- opt$init_fn(list(a = jnp_ones(2)))
#' marked <- optimizers_unpack_optimizer_state(state)
#' state2 <- optimizers_pack_optimizer_state(marked)
#' }
#' @name optimizers_serialization
NULL

#' @rdname optimizers_serialization
#' @export
optimizers_JoinPoint <- function(subtree) structure(list(subtree = subtree), class = "JoinPoint")

#' @rdname optimizers_serialization
#' @export
optimizers_unpack_optimizer_state <- function(opt_state) {
  subtrees <- Map(tree_unflatten, opt_state$subtree_defs, opt_state$packed_state)
  tree_unflatten(opt_state$tree_def, lapply(unname(subtrees), optimizers_JoinPoint))
}

#' @rdname optimizers_serialization
#' @export
optimizers_pack_optimizer_state <- function(marked_pytree) {
  fl <- tree_flatten(marked_pytree)
  if (!all(vapply(fl$leaves, inherits, NA, what = "JoinPoint")))
    jax_error("pack_optimizer_state requires a pytree whose leaves are all JoinPoint objects")
  subtrees <- lapply(lapply(fl$leaves, `[[`, "subtree"), tree_flatten)
  optimizers_OptimizerState(lapply(subtrees, `[[`, "leaves"), fl$treedef,
                            lapply(subtrees, `[[`, "treedef"))
}
