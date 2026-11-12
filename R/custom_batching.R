# Port of jax.custom_batching (jax/_src/custom_batching.py): functions with a user
# defined vmap rule.

custom_vmap_p <- new_primitive("custom_vmap_call", multiple_results = TRUE)
custom_vmap_p$abstract_eval <- function(avals, params) jaxpr_out_avals(params$call_jaxpr$jaxpr)
custom_vmap_p$lowering <- call_jaxpr_lowering
custom_vmap_p$override$eval <- function(trace, args, params) params$fun(args)
custom_vmap_p$override$jaxpr <- stage_custom(custom_vmap_p)
custom_vmap_p$override$pe <- stage_custom(custom_vmap_p)

# Port of custom_vmap_batching. `params$rule(axis_size, in_batched, vals)` takes flat
# operands with their batch dimension at 0 and returns list(outs, batched).
custom_vmap_p$batch <- function(vals, dims, params, axis) {
  vals <- Map(function(v, d) if (is.na(d)) v else moveaxis(v, d, 0L), vals, as.list(dims))
  res <- params$rule(axis$size, !is.na(dims), vals)
  list(res$outs, ifelse(res$batched, 0L, NA_integer_))
}

# Port of vmap_unrestricted: run f (list -> list) under a batch trace and report which
# outputs are batched (their batch dimension is moved to 0).
vmap_unrestricted <- function(f, vals, dims, size) {
  trace <- new_batch_trace(current_trace(), size, NULL)
  on.exit(trace$alive <- FALSE, add = TRUE)
  ins <- Map(function(v, d) batch_tracer(trace, v, d), vals, as.list(dims))
  outs <- with_trace(trace, f(ins))
  batched <- vapply(outs, is_own, NA, trace = trace)
  list(outs = lapply(outs, function(o) if (is_own(o, trace)) moveaxis(o$val, o$bdim, 0L) else o),
       batched = batched)
}

# Port of custom_vmap_jvp: the JVP of a custom_vmap function is again a custom_vmap
# function, whose rule differentiates the user rule (jvp_of_rule_rule). Operands are the
# primals followed by the tangents of the inexact primals; outputs are the primal
# outputs followed by the tangents of the inexact outputs.
custom_vmap_p$jvp <- function(primals, tangents, params) {
  n <- length(primals)
  out_avals <- jaxpr_out_avals(params$call_jaxpr$jaxpr)
  m <- length(out_avals)
  diff_in <- vapply(primals, function(p) is_inexact_dtype(aval_of(p)$dtype), NA)
  diff_out <- vapply(out_avals, function(a) is_inexact_dtype(a$dtype), NA)
  nd <- sum(diff_in)
  ts <- Map(function(t, p) if (is_zero(t)) zeros_aval(tangent_aval(aval_of(p))) else t,
            tangents[diff_in], primals[diff_in])
  # jvp() ignores the tangent of a non-inexact primal; the primal itself stands in.
  jvp_flat <- function(g, v) {
    p <- v[seq_len(n)]
    t <- p
    t[diff_in] <- v[n + seq_len(nd)]
    r <- jvp(function(...) g(list(...)), p, t)
    c(r[[1L]], r[[2L]][diff_out])
  }
  fun <- params$fun
  rule <- params$rule
  fun_jvp <- function(v) jvp_flat(fun, v)
  rule_jvp <- function(axis_size, in_batched, v) {
    bp <- in_batched[seq_len(n)]
    bt <- rep(FALSE, n)
    bt[diff_in] <- in_batched[n + seq_len(nd)]
    mutual <- bp & bt
    store <- new.env(parent = emptyenv())
    to_jvp <- function(prims) {
      r <- rule(axis_size, mutual, prims)
      store$batched <- r$batched
      r$outs
    }
    dims <- c(ifelse(bp & !bt, 0L, NA_integer_), ifelse((bt & !bp)[diff_in], 0L, NA_integer_))
    res <- vmap_unrestricted(function(vv) jvp_flat(to_jvp, vv), v, dims, axis_size)
    list(outs = res$outs,
         batched = c(store$batched | res$batched[seq_len(m)],
                     store$batched[diff_out] | res$batched[m + seq_len(sum(diff_out))]))
  }
  ops <- c(primals, ts)
  cj <- trace_to_jaxpr(fun_jvp, lapply(ops, aval_of))
  outs <- bind(custom_vmap_p, ops, list(fun = fun_jvp, rule = rule_jvp, call_jaxpr = cj))
  out_t <- vector("list", m)
  k <- 0L
  for (j in seq_len(m)) {
    out_t[[j]] <- if (diff_out[j]) outs[[m + (k <- k + 1L)]] else zero_of(tangent_aval(out_avals[[j]]))
  }
  list(outs[seq_len(m)], out_t)
}

make_custom_vmap <- function(fun, vmap_rule) {
  f <- function(...) {
    args <- list(...)
    if (is.null(vmap_rule))
      jax_error("No batching rule defined for custom_vmap function using def_vmap.",
                class = "jax_attribute_error")
    fl <- flatten_args(args)
    store <- new.env(parent = emptyenv())
    fun_flat <- function(leaves) {
      out <- do.call(fun, tree_unflatten(fl$treedef, leaves))
      ofl <- tree_flatten(out)
      store$out_tree <- ofl$treedef
      lapply(ofl$leaves, as_jax)
    }
    cj <- trace_to_jaxpr(fun_flat, lapply(fl$leaves, aval_of))
    out_tree <- store$out_tree
    rule_flat <- function(axis_size, in_batched, vals) {
      res <- do.call(vmap_rule, c(list(axis_size, tree_unflatten(fl$treedef, as.list(in_batched))),
                                  tree_unflatten(fl$treedef, vals)))
      if (!is.list(res) || !is.null(oldClass(res)) || length(res) != 2L)
        jax_error("custom vmap rule must return list(out, out_batched)", class = "jax_type_error")
      o <- tree_flatten(res[[1L]])
      ob <- tree_flatten(res[[2L]])
      if (!treedef_equal(o$treedef, ob$treedef))
        jax_error("structure of output value and output batching specification returned by ",
                  "custom vmap rule do not match.\nOutput values: ", format(o$treedef),
                  "\nBatching spec: ", format(ob$treedef), class = "jax_value_error")
      if (!treedef_equal(o$treedef, out_tree))
        jax_error("structure of output returned by custom vmap rule does not match that of ",
                  "original custom-vmapped function.\nOriginal output: ", format(out_tree),
                  "\nRule output: ", format(o$treedef), class = "jax_value_error")
      list(outs = lapply(o$leaves, as_jax), batched = vapply(ob$leaves, isTRUE, NA))
    }
    outs <- bind(custom_vmap_p, fl$leaves, list(fun = fun_flat, rule = rule_flat, call_jaxpr = cj))
    tree_unflatten(out_tree, outs)
  }
  structure(f, class = c("jax_custom_vmap", "function"), fun = fun, vmap_rule = vmap_rule)
}

#' Define a function with a custom vmap rule
#'
#' Ports of `jax.custom_batching.custom_vmap` and `sequential_vmap`. A `custom_vmap()`
#' function behaves like `fun`, except under [vmap()], where `vmap_rule` is used.
#'
#' The rule is called as `vmap_rule(axis_size, in_batched, ...)`: `axis_size` is the size
#' of the mapped axis, `in_batched` is a list of logicals with the structure of the
#' arguments saying which of them are batched, and `...` are the arguments, with the
#' mapped axis first in every batched one. It returns `list(out, out_batched)`, where
#' `out_batched` has the structure of `out` and says which outputs carry the mapped
#' axis (first).
#'
#' As in JAX, a `custom_vmap()` function supports forward mode differentiation but not
#' reverse mode. To customize both, wrap the `custom_vmap()` function in
#' [custom_vjp()], with `custom_vjp()` on the outside.
#'
#' `sequential_vmap(f)` is the special case whose rule calls `f` once per element in a
#' loop ([lax_map()]), for functions that cannot handle a batch dimension.
#' @param fun,f A function. For `custom_vmap_def_vmap()`, a function created by
#'   `custom_vmap()`.
#' @param vmap_rule The vmap rule (can be set later with `custom_vmap_def_vmap()`).
#' @return A function with a custom vmap rule.
#' @examples
#' \dontrun{
#' f <- custom_vmap(function(x, y) x + y, function(axis_size, in_batched, xs, ys) {
#'   list(xs * ys, TRUE)
#' })
#' f(2, 3)                                           # 5
#' vmap(f)(jnp_arange(3L), jnp_arange(1L, 4L))       # 0 2 6: the rule multiplies
#'
#' g <- sequential_vmap(function(x) x + 1)
#' vmap(g)(jnp_arange(3L))
#' }
#' @export
custom_vmap <- function(fun, vmap_rule = NULL) {
  if (!is.function(fun)) jax_error("custom_vmap: fun must be a function", class = "jax_type_error")
  make_custom_vmap(fun, vmap_rule)
}

#' @rdname custom_vmap
#' @export
custom_vmap_def_vmap <- function(f, vmap_rule) {
  if (!inherits(f, "jax_custom_vmap"))
    jax_error("custom_vmap_def_vmap: f must be created by custom_vmap()", class = "jax_type_error")
  make_custom_vmap(attr(f, "fun"), vmap_rule)
}

#' @rdname custom_vmap
#' @export
sequential_vmap <- function(f) {
  g <- NULL
  rule <- function(axis_size, in_batched, ...) {
    fl <- tree_flatten(list(...))
    mask <- vapply(tree_flatten(in_batched)$leaves, isTRUE, NA)
    to_map <- function(mapped) {
      leaves <- fl$leaves
      leaves[mask] <- mapped
      do.call(g, tree_unflatten(fl$treedef, leaves))
    }
    out <- lax_map(to_map, fl$leaves[mask])
    list(out, tree_map(function(l) TRUE, out))
  }
  g <- custom_vmap(f, rule)
  g
}
