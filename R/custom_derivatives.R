# Custom derivative rules (port of jax/_src/custom_derivatives.py and the custom call
# handling of jax/_src/interpreters/ad.py, batching.py and partial_eval.py).

# ---- shared helpers -------------------------------------------------------------------

is_symbolic_zero <- function(x) inherits(x, "jax_SymbolicZero")

# Run f (list -> list) under a new batch trace with the given input dims (0 or NA).
# Returns the outputs (batch dimension moved to 0) and which of them are batched.
batched_run <- function(f, dims, size, name, vals) {
  trace <- new_batch_trace(current_trace(), size, name)
  on.exit(trace$alive <- FALSE, add = TRUE)
  ins <- Map(function(v, d) {
    if (is_symbolic_zero(v)) {
      # A symbolic zero stays symbolic, with the type of one element of the batch.
      if (!is.na(d)) v$aval$shape <- v$aval$shape[-(d + 1L)]
      return(v)
    }
    batch_tracer(trace, v, d)
  }, vals, as.list(dims))
  outs <- with_trace(trace, f(ins))
  own <- vapply(outs, is_own, NA, trace = trace)
  list(vals = Map(function(o, b) if (b) moveaxis(o$val, o$bdim, 0L) else o, outs, own),
       batched = own)
}

# Give each value the batch status in `want` (port of batching.matchaxis for a batch
# dimension at 0): unbatched values are broadcast where a batch dimension is wanted.
match_batched <- function(vals, have, want, size) {
  Map(function(v, h, w) {
    if (h == w) return(v)
    if (!w)
      jax_error("vmap of a function with a custom derivative rule: the rule gives a batched ",
                "value for an output that the function itself leaves unbatched")
    if (is_zero(v)) return(zero_of(new_aval(c(size, v$aval$shape), v$aval$dtype)))
    lax_broadcast(v, size)
  }, vals, have, want)
}

# Which outputs of a batched custom call carry the batch dimension is settled by the
# rule that runs while the call is bound (the function itself, or its derivative rule
# under differentiation), as in JAX. Later runs follow that choice.
batch_flags <- function() {
  st <- new.env(parent = emptyenv())
  st$open <- TRUE
  st$settle <- function(batched) {
    if (st$open) st$out <- batched
    st$out
  }
  st
}

flatten_args <- function(args) {
  fl <- tree_flatten(args)
  list(leaves = lapply(fl$leaves, as_jax), treedef = fl$treedef)
}

stage_custom <- function(prim) {
  function(trace, args, params) {
    if (!trace$forward_known || any(vapply(args, is_own, NA, trace = trace)))
      return(stage_eqn(trace, prim, args, params))
    with_trace(trace$parent, bind(prim, args, params))
  }
}

call_jaxpr_lowering <- function(ctx, ins, avals_in, avals_out, params) {
  cj <- params$call_jaxpr
  consts <- lapply(seq_along(cj$consts), function(i) {
    c <- cj$consts[[i]]
    if (is_tracer(c)) jax_error("custom derivative function closes over a traced value; pass it as an argument")
    ir_constant(ctx, array_host(c), cj$jaxpr$constvars[[i]]$aval)
  })
  jaxpr_subcomp(ctx, cj$jaxpr, consts, as.list(ins))
}

# The primal output of a derivative rule must have the structure and the leaf types of
# the output of the function (ports of the checks in _flatten_jvp and _flatten_fwd).
check_rule_primal_out <- function(what, tree, avals, store) {
  if (is.null(store$out_avals)) return(invisible())
  describe <- function(td, av) format(tree_unflatten(td, lapply(av, aval_long)))
  if (!treedef_equal(tree, store$out_tree))
    jax_error(what, " must produce a pair (list of length two) where the first element ",
              "represents the primal output (equal to the output of the function, and in ",
              "particular of the same container/pytree structure), but instead its first ",
              "element had container/pytree structure:\n    ", format(tree),
              "\nwhile the function had output container/pytree structure:\n    ",
              format(store$out_tree), ".", class = "jax_type_error")
  if (!all(mapply(aval_eq, avals, store$out_avals)))
    jax_error(what, " must produce a pair (list of length two) where the first element ",
              "represents the primal output (equal to the output of the function, and in ",
              "particular with leaves of the same shape/dtype), but instead its first ",
              "element had shapes/dtypes of:\n    ",
              paste(vapply(avals, aval_long, ""), collapse = ", "),
              "\nwhile the function had output shapes/dtypes of:\n    ",
              paste(vapply(store$out_avals, aval_long, ""), collapse = ", "),
              class = "jax_type_error")
}

# ---- custom_jvp ---------------------------------------------------------------------

custom_jvp_call_p <- new_primitive("custom_jvp_call", multiple_results = TRUE)
custom_jvp_call_p$abstract_eval <- function(avals, params) jaxpr_out_avals(params$call_jaxpr$jaxpr)
custom_jvp_call_p$lowering <- call_jaxpr_lowering
custom_jvp_call_p$override$eval <- function(trace, args, params) params$fun(args)
custom_jvp_call_p$override$jaxpr <- stage_custom(custom_jvp_call_p)
custom_jvp_call_p$override$pe <- stage_custom(custom_jvp_call_p)

# Port of JVPTrace.process_custom_jvp_call.
custom_jvp_call_p$override$jvp <- function(trace, args, params) {
  pts <- lapply(args, primal_tangent, trace = trace)
  primals <- lapply(pts, `[[`, 1L)
  tangents <- lapply(pts, `[[`, 2L)
  if (all(vapply(tangents, is_zero, NA)))
    return(with_trace(trace$parent, bind(custom_jvp_call_p, primals, params)))
  outs <- with_trace(trace$parent, {
    tangents <- if (isTRUE(params$symbolic_zeros))
      lapply(tangents, function(t) if (is_zero(t)) SymbolicZero(t$aval) else t)
    else lapply(tangents, instantiate_zero)
    params$jvp(primals, tangents)
  })
  n <- length(outs) %/% 2L
  Map(function(p, t) maybe_jvp_tracer(trace, p, t), outs[seq_len(n)], outs[n + seq_len(n)])
}

# Port of BatchTrace.process_custom_jvp_call.
custom_jvp_call_p$override$batch <- function(trace, args, params) {
  own <- vapply(args, is_own, NA, trace = trace)
  if (!any(own)) return(with_trace(trace$parent, bind(custom_jvp_call_p, args, params)))
  size <- trace$axis_size
  name <- trace$axis_name
  vals <- args
  dims <- rep(NA_integer_, length(args))
  for (i in which(own)) {
    vals[[i]] <- with_trace(trace$parent, moveaxis(args[[i]]$val, args[[i]]$bdim, 0L))
    dims[i] <- 0L
  }
  fun <- params$fun
  jvp_fun <- params$jvp
  st <- batch_flags()
  fun_b <- function(v) {
    r <- batched_run(fun, dims, size, name, v)
    match_batched(r$vals, r$batched, st$settle(r$batched), size)
  }
  # An output is batched when its primal or its tangent is (batch_custom_jvp_subtrace).
  jvp_b <- function(primals, tangents) {
    n <- length(primals)
    r <- batched_run(function(v) jvp_fun(v[seq_len(n)], v[n + seq_len(n)]), c(dims, dims),
                     size, name, c(primals, tangents))
    m <- length(r$vals) %/% 2L
    want <- st$settle(r$batched[seq_len(m)] | r$batched[m + seq_len(m)])
    match_batched(r$vals, r$batched, c(want, want), size)
  }
  cj <- with_trace(trace$parent, trace_to_jaxpr(fun_b, lapply(vals, aval_of)))
  outs <- with_trace(trace$parent, bind(custom_jvp_call_p, vals,
    update_params(params, list(fun = fun_b, jvp = jvp_b, call_jaxpr = cj))))
  st$open <- FALSE
  Map(function(o, b) batch_tracer(trace, o, if (b) 0L else NA_integer_), outs, st$out)
}

custom_jvp_call_p$transpose <- function(ct, args, params) {
  cj <- params$call_jaxpr
  backward_pass(cj$jaxpr, cj$consts, args, ct)
}

# Port of api_util.resolve_kwargs: named arguments are matched to the formals of `fun`
# and passed by position, so that the argument tree does not depend on how the caller
# named them. Left as is when the matched formals are not a prefix of the formals or
# `fun` takes `...`.
resolve_args <- function(fun, args) {
  nms <- names(args)
  if (is.null(nms) || !any(nzchar(nms))) return(args)
  fm <- names(formals(fun))
  if (is.null(fm) || "..." %in% fm) return(args)
  syms <- lapply(seq_along(args), function(i) as.name(paste0(".a", i)))
  names(syms) <- nms
  mc <- tryCatch(match.call(fun, as.call(c(list(quote(f)), syms))), error = function(e) NULL)
  if (is.null(mc)) return(args)
  got <- as.list(mc)[-1L]
  if (!identical(names(got), fm[seq_along(got)])) return(args)
  unname(args[as.integer(sub(".a", "", vapply(got, as.character, ""), fixed = TRUE))])
}

make_custom_jvp <- function(fun, jvp, nondiff_argnums, symbolic_zeros = FALSE) {
  f <- function(...) {
    args <- resolve_args(fun, list(...))
    if (is.null(jvp))
      jax_error("No JVP defined for custom_jvp function. Did you forget to use custom_jvp_defjvp?",
                class = "jax_attribute_error")
    # As in JAX, the rule receives args[i] for every i of nondiff_argnums (a negative i
    # counts from the end), but only the nonnegative positions are held out of
    # differentiation.
    nda <- as.integer(nondiff_argnums)
    nd <- ifelse(nda < 0L, nda + length(args), nda) + 1L
    diff_pos <- setdiff(seq_along(args), nda[nda >= 0L] + 1L)
    fl <- flatten_args(args[diff_pos])
    store <- new.env(parent = emptyenv())
    rebuild <- function(leaves) {
      a <- args
      a[diff_pos] <- tree_unflatten(fl$treedef, leaves)
      a
    }
    fun_flat <- function(leaves) {
      out <- do.call(fun, rebuild(leaves))
      ofl <- tree_flatten(out)
      outs <- lapply(ofl$leaves, as_jax)
      store$out_tree <- ofl$treedef
      store$out_avals <- lapply(outs, aval_of)
      outs
    }
    jvp_flat <- function(primals, tangents) {
      p <- tree_unflatten(fl$treedef, primals)
      t <- tree_unflatten(fl$treedef, tangents)
      res <- do.call(jvp, c(args[nd], list(p, t)))
      if (!is.list(res) || !is.null(oldClass(res)) || length(res) != 2L)
        jax_error("Custom JVP rule must produce a pair (list of length two) representing ",
                  "primal and tangent outputs", class = "jax_type_error")
      po <- tree_flatten(res[[1L]])
      to <- tryCatch(flatten_up_to(po$treedef, res[[2L]]), error = function(e)
        jax_error("Custom JVP rule must produce primal and tangent outputs with equal ",
                  "container (pytree) structures, but got ", format(po$treedef), " and ",
                  format(tree_structure(res[[2L]])), " respectively.", class = "jax_type_error"))
      po_l <- lapply(po$leaves, as_jax)
      pa <- lapply(po_l, aval_of)
      check_rule_primal_out("Custom JVP rule", po$treedef, pa, store)
      # A NULL tangent stands for zero; a SymbolicZero stays symbolic.
      to_l <- Map(function(t, a) {
        if (is.null(t) || is_zero(t)) zeros_aval(tangent_aval(a))
        else if (is_symbolic_zero(t)) t else as_jax(t)
      }, to, pa)
      ea <- lapply(pa, tangent_aval)
      ta <- lapply(to_l, function(t) if (is_symbolic_zero(t)) t$aval else aval_of(t))
      bad <- !mapply(aval_eq, ea, ta)
      if (any(bad)) {
        if (length(bad) == 1L)
          jax_error("Custom JVP rule must produce primal and tangent outputs with ",
                    "corresponding shapes and dtypes. Expected ", aval_long(ea[[1L]]),
                    " (tangent type of ", aval_long(pa[[1L]]), ") but got ",
                    aval_long(ta[[1L]]), ".", class = "jax_type_error")
        jax_error("Custom JVP rule must produce primal and tangent outputs with ",
                  "corresponding shapes and dtypes, but got:\n",
                  paste0("  primal ", vapply(pa[bad], aval_long, ""), " with tangent ",
                         vapply(ta[bad], aval_long, ""), ", expecting tangent ",
                         vapply(ea[bad], aval_long, ""), collapse = "\n"),
                  class = "jax_type_error")
      }
      c(po_l, Map(function(t, a) if (is_symbolic_zero(t)) zero_of(a) else t, to_l, ea))
    }
    cj <- trace_to_jaxpr(fun_flat, lapply(fl$leaves, aval_of))
    out_tree <- store$out_tree
    outs <- bind(custom_jvp_call_p, fl$leaves,
                 list(fun = fun_flat, jvp = jvp_flat, call_jaxpr = cj,
                      symbolic_zeros = isTRUE(symbolic_zeros)))
    tree_unflatten(out_tree, outs)
  }
  structure(f, class = c("jax_custom_jvp", "function"), fun = fun, jvp = jvp,
            nondiff_argnums = nondiff_argnums, symbolic_zeros = isTRUE(symbolic_zeros))
}

#' Define a function with a custom JVP rule
#'
#' Port of `jax.custom_jvp`. The rule receives `(primals, tangents)` (lists of the
#' differentiable arguments, preceded by any `nondiff_argnums` arguments) and returns
#' `list(primal_out, tangent_out)`. Each tangent output must have the shape and dtype
#' of its primal output (float0 for an integer output); `NULL` or a [SymbolicZero()]
#' stands for a zero tangent.
#' @param fun The function.
#' @param jvp Optional JVP rule (can be set later with [custom_jvp_defjvp()]).
#' @param nondiff_argnums 0-based positions of non-differentiable (static) arguments.
#' @return A function with a custom JVP rule.
#' @examples
#' \dontrun{
#' f <- custom_jvp(function(x) sin(x), jvp = function(primals, tangents) {
#'   list(sin(primals[[1]]), cos(primals[[1]]) * tangents[[1]])
#' })
#' }
#' @export
custom_jvp <- function(fun, jvp = NULL, nondiff_argnums = integer(0)) {
  make_custom_jvp(fun, jvp, nondiff_argnums)
}

#' @rdname custom_jvp
#' @param f A function created by [custom_jvp()].
#' @param symbolic_zeros Whether the rule receives a [SymbolicZero()] object in place
#'   of each tangent that is known to be zero (instead of an array of zeros), so that
#'   it can skip work.
#' @export
custom_jvp_defjvp <- function(f, jvp, symbolic_zeros = FALSE) {
  make_custom_jvp(attr(f, "fun"), jvp, attr(f, "nondiff_argnums"), symbolic_zeros)
}

#' @rdname custom_jvp
#' @param ... One JVP function per argument, `g(tangent, primal_out, ...primals)`, or
#'   `NULL` for arguments without a derivative.
#' @export
custom_jvp_defjvps <- function(f, ...) {
  gs <- list(...)
  if (length(attr(f, "nondiff_argnums")))
    jax_error("Can't use custom_jvp_defjvps with nondiff_argnums.", class = "jax_type_error")
  custom_jvp_defjvp(f, function(primals, tangents) {
    ans <- do.call(attr(f, "fun"), primals)
    out <- NULL
    for (i in seq_along(gs)) {
      if (is.null(gs[[i]])) next
      t <- do.call(gs[[i]], c(list(tangents[[i]], ans), primals))
      out <- if (is.null(out)) t else tree_map(jnp_add, out, t)
    }
    if (is.null(out)) out <- tree_map(jnp_zeros_like, ans)
    list(ans, out)
  })
}

# ---- custom_vjp ---------------------------------------------------------------------

custom_vjp_call_p <- new_primitive("custom_vjp_call", multiple_results = TRUE)
custom_vjp_call_p$abstract_eval <- function(avals, params) jaxpr_out_avals(params$call_jaxpr$jaxpr)
custom_vjp_call_p$lowering <- call_jaxpr_lowering
custom_vjp_call_p$override$eval <- function(trace, args, params) params$fun(args)
custom_vjp_call_p$override$jaxpr <- stage_custom(custom_vjp_call_p)
custom_vjp_call_p$override$pe <- stage_custom(custom_vjp_call_p)

custom_lin_p <- new_primitive("custom_lin", multiple_results = TRUE)
custom_lin_p$abstract_eval <- function(avals, params) params$out_avals
custom_lin_p$override$eval <- function(trace, args, params) {
  jax_error("can't apply forward-mode autodiff (jvp) to a custom_vjp function.",
            class = "jax_type_error")
}
custom_lin_p$override$jvp <- function(trace, args, params) {
  if (!any(vapply(args, is_own, NA, trace = trace)))
    return(with_trace(trace$parent, bind(custom_lin_p, args, params)))
  jax_error("can't apply forward-mode autodiff (jvp) to a custom_vjp function.",
            class = "jax_type_error")
}
custom_lin_p$batch <- function(vals, dims, params, axis) {
  jax_error("can't apply forward-mode autodiff (jvp) to a custom_vjp function.",
            class = "jax_type_error")
}
# Port of ad._custom_lin_transpose_fancy: cotangents for the non-zero tangent inputs.
custom_lin_p$transpose <- function(cts, args, params) {
  nres <- params$num_res
  res <- args[seq_len(nres)]
  cts <- Map(function(c, a) {
    if (!is_zero(c)) c else if (isTRUE(params$symbolic_zeros)) SymbolicZero(a) else zeros_aval(a)
  }, cts, params$out_avals)
  cts_in <- params$bwd(res, cts)
  nz <- cts_in[!params$in_zeros]
  c(rep(list(NULL), nres), nz)
}

# Port of JVPTrace.process_custom_vjp_call.
custom_vjp_call_p$override$jvp <- function(trace, args, params) {
  pts <- lapply(args, primal_tangent, trace = trace)
  primals <- lapply(pts, `[[`, 1L)
  tangents <- lapply(pts, `[[`, 2L)
  if (all(vapply(tangents, is_zero, NA)))
    return(with_trace(trace$parent, bind(custom_vjp_call_p, primals, params)))
  in_zeros <- vapply(tangents, is_zero, NA)
  r <- with_trace(trace$parent, params$fwd(primals, !in_zeros))
  primals_out <- r$out
  res <- r$res
  out_avals <- lapply(primals_out, function(p) tangent_aval(aval_of(p)))
  tangents_out <- with_trace(trace$parent, bind(custom_lin_p, c(res, tangents[!in_zeros]),
    list(num_res = length(res), bwd = params$bwd, out_avals = out_avals, in_zeros = in_zeros,
         symbolic_zeros = isTRUE(params$symbolic_zeros))))
  Map(function(p, t) maybe_jvp_tracer(trace, p, t), primals_out, tangents_out)
}

# Port of BatchTrace.process_custom_vjp_call and batch_custom_vjp_bwd.
custom_vjp_call_p$override$batch <- function(trace, args, params) {
  own <- vapply(args, is_own, NA, trace = trace)
  if (!any(own)) return(with_trace(trace$parent, bind(custom_vjp_call_p, args, params)))
  size <- trace$axis_size
  name <- trace$axis_name
  vals <- args
  dims <- rep(NA_integer_, length(args))
  for (i in which(own)) {
    vals[[i]] <- with_trace(trace$parent, moveaxis(args[[i]]$val, args[[i]]$bdim, 0L))
    dims[i] <- 0L
  }
  fun <- params$fun
  fwd <- params$fwd
  bwd <- params$bwd
  st <- batch_flags()
  fun_b <- function(v) {
    r <- batched_run(fun, dims, size, name, v)
    match_batched(r$vals, r$batched, st$settle(r$batched), size)
  }
  # Residuals keep the batch status the forward rule gives them.
  fwd_b <- function(primals, perturbed) {
    n <- 0L
    r <- batched_run(function(v) {
      o <- fwd(v, perturbed)
      n <<- length(o$out)
      c(o$out, o$res)
    }, dims, size, name, primals)
    io <- seq_len(n)
    st$res <- drop_first(r$batched, n)
    list(out = match_batched(r$vals[io], r$batched[io], st$settle(r$batched[io]), size),
         res = drop_first(r$vals, n))
  }
  # Cotangents of unbatched arguments are summed over the batch.
  bwd_b <- function(res, cts) {
    nres <- length(res)
    in_dims <- c(ifelse(st$res, 0L, NA_integer_), ifelse(st$out, 0L, NA_integer_))
    r <- batched_run(function(v) bwd(v[seq_len(nres)], v[nres + seq_len(length(cts))]),
                     in_dims, size, name, c(res, cts))
    Map(function(c, b, d) {
      if (is.na(d)) { if (b) lax_reduce_sum(c, 0L) else c }
      else if (b) c else lax_broadcast(c, size)
    }, r$vals, r$batched, as.list(dims))
  }
  cj <- with_trace(trace$parent, trace_to_jaxpr(fun_b, lapply(vals, aval_of)))
  outs <- with_trace(trace$parent, bind(custom_vjp_call_p, vals,
    update_params(params, list(fun = fun_b, fwd = fwd_b, bwd = bwd_b, call_jaxpr = cj))))
  st$open <- FALSE
  Map(function(o, b) batch_tracer(trace, o, if (b) 0L else NA_integer_), outs, st$out)
}

make_custom_vjp <- function(fun, fwd, bwd, nondiff_argnums, symbolic_zeros = FALSE) {
  symbolic_zeros <- isTRUE(symbolic_zeros)
  f <- function(...) {
    args <- resolve_args(fun, list(...))
    if (is.null(fwd) || is.null(bwd))
      jax_error("No VJP defined for custom_vjp function. Did you forget to use custom_vjp_defvjp?",
                class = "jax_attribute_error")
    # As in JAX, the rule receives args[i] for every i of nondiff_argnums (a negative i
    # counts from the end), but only the nonnegative positions are held out of
    # differentiation.
    nda <- as.integer(nondiff_argnums)
    nd <- ifelse(nda < 0L, nda + length(args), nda) + 1L
    diff_pos <- setdiff(seq_along(args), nda[nda >= 0L] + 1L)
    fl <- flatten_args(args[diff_pos])
    store <- new.env(parent = emptyenv())
    rebuild <- function(leaves) {
      a <- args
      a[diff_pos] <- tree_unflatten(fl$treedef, leaves)
      a
    }
    fun_flat <- function(leaves) {
      out <- do.call(fun, rebuild(leaves))
      ofl <- tree_flatten(out)
      outs <- lapply(ofl$leaves, as_jax)
      store$out_tree <- ofl$treedef
      store$out_avals <- lapply(outs, aval_of)
      outs
    }
    # In symbolic zeros mode the forward rule receives CustomVJPPrimal records telling
    # which arguments are perturbed.
    fwd_flat <- function(primals, perturbed) {
      ins <- if (symbolic_zeros) Map(CustomVJPPrimal, primals, as.list(perturbed)) else primals
      r <- do.call(fwd, rebuild(ins))
      if (!is.list(r) || !is.null(oldClass(r)) || length(r) != 2L)
        jax_error("Custom VJP fwd rule must produce a pair (list of length two) where the ",
                  "first element represents the primal output and the second element ",
                  "represents residuals (values stored from the forward pass for use on the ",
                  "backward pass)", class = "jax_type_error")
      ofl <- tree_flatten(r[[1L]])
      rfl <- tree_flatten(r[[2L]])
      outs <- lapply(ofl$leaves, as_jax)
      check_rule_primal_out("Custom VJP fwd rule", ofl$treedef, lapply(outs, aval_of), store)
      store$res_tree <- rfl$treedef
      list(out = outs, res = lapply(rfl$leaves, as_jax))
    }
    # Port of _flatten_bwd: one cotangent per leaf of the differentiable arguments.
    bwd_flat <- function(res, cts) {
      r <- do.call(bwd, c(args[nd], list(tree_unflatten(store$res_tree, res),
                                         tree_unflatten(store$out_tree, cts))))
      cts_in <- if (is.list(r) && is.null(oldClass(r)) && length(r) == length(diff_pos))
        tryCatch(flatten_up_to(fl$treedef, r, null_ok = TRUE), error = function(e) NULL)
      if (is.null(cts_in) && length(fl$leaves))
        jax_error("Custom VJP bwd rule must produce an output with the same container ",
                  "(pytree) structure as the arguments of the primal function, and in ",
                  "particular must produce a list of length equal to the number of arguments ",
                  "to the primal function, but got bwd output structure ",
                  format(tree_structure(r)), " for primal input structure ",
                  format(fl$treedef), ".", class = "jax_type_error")
      Map(function(c, p, i) {
        a <- aval_of(p)
        want <- tangent_aval(a)
        if (is.null(c) || want$dtype == "float0") return(zeros_aval(want))
        sz <- is_symbolic_zero(c)
        if (!sz) c <- as_jax(c)
        ca <- if (sz) c$aval else aval_of(c)
        if (!identical(ca$shape, a$shape) || (sz && ca$dtype != want$dtype)) {
          kp <- keystr(tree_leaves_with_path(unname(args[diff_pos]))[[i]][[1L]])
          if (sz)
            jax_error("Custom VJP bwd rule produced a SymbolicZero with a shape/dtype that ",
                      "does not match the corresponding input tangent shape/dtype: at output",
                      kp, " the SymbolicZero had shape/dtype ", aval_long(ca), " while the ",
                      "corresponding input had shape/dtype ", aval_long(a), ". Consider just ",
                      "returning a NULL here instead of a SymbolicZero object.",
                      class = "jax_value_error")
          jax_error("Custom VJP bwd rule must produce an output with the same type as the ",
                    "arguments of the primal function, but at output", kp, " the bwd rule ",
                    "produced an output of type ", aval_long(ca), " corresponding to an input ",
                    "of type ", aval_long(a), ", so the shapes do not match",
                    class = "jax_value_error")
        }
        if (sz) return(zeros_aval(want))
        # Port of _temporary_dtype_exception: a cotangent of another dtype is passed on as
        # it is.
        c
      }, cts_in %||% list(), fl$leaves, seq_along(fl$leaves))
    }
    cj <- trace_to_jaxpr(fun_flat, lapply(fl$leaves, aval_of))
    out_tree <- store$out_tree
    outs <- bind(custom_vjp_call_p, fl$leaves,
                 list(fun = fun_flat, fwd = fwd_flat, bwd = bwd_flat, call_jaxpr = cj,
                      symbolic_zeros = symbolic_zeros))
    tree_unflatten(out_tree, outs)
  }
  structure(f, class = c("jax_custom_vjp", "function"), fun = fun, fwd = fwd, bwd = bwd,
            nondiff_argnums = nondiff_argnums, symbolic_zeros = symbolic_zeros)
}

#' Define a function with a custom VJP rule
#'
#' Port of `jax.custom_vjp`. `fwd(...)` returns `list(primal_out, residuals)`;
#' `bwd(residuals, ct)` (preceded by any `nondiff_argnums` arguments) returns a list with
#' one cotangent per differentiable argument (`NULL` for zero), each with the structure
#' and the leaf shapes of its argument. Forward mode differentiation ([jvp()]) of such a
#' function raises an error, as in JAX.
#' @param fun The function.
#' @param fwd,bwd Optional rules (can be set later with [custom_vjp_defvjp()]).
#' @param nondiff_argnums 0-based positions of non-differentiable arguments.
#' @param symbolic_zeros Symbolic zeros mode (port of `symbolic_zeros=True` of
#'   `custom_vjp.defvjp`). The leaves of the arguments of `fwd` are then
#'   [CustomVJPPrimal()] records with elements `value` and `perturbed` (whether that
#'   argument is being differentiated), and the leaves of `ct` in `bwd` are
#'   [SymbolicZero()] objects where a cotangent is known to be zero. `bwd` may return
#'   `SymbolicZero` objects too.
#' @param optimize_remat Run `fwd` through the `remat_opt` primitive (port of
#'   `optimize_remat=True` of `custom_vjp.defvjp`): values and derivatives are unchanged.
#'   jaxR keeps the forward computation whole (its dead code elimination has no rule that
#'   drops the primal part), so this saves no memory.
#' @return A function with a custom VJP rule.
#' @export
custom_vjp <- function(fun, fwd = NULL, bwd = NULL, nondiff_argnums = integer(0)) {
  make_custom_vjp(fun, fwd, bwd, nondiff_argnums)
}

#' @rdname custom_vjp
#' @param f A function created by [custom_vjp()].
#' @export
custom_vjp_defvjp <- function(f, fwd, bwd, symbolic_zeros = FALSE, optimize_remat = FALSE) {
  if (isTRUE(symbolic_zeros) && isTRUE(optimize_remat))
    jax_error("remat optimization for custom_vjp does not support symbolic zeros",
              class = "jax_not_implemented_error")
  if (isTRUE(optimize_remat)) fwd <- remat_opt_wrap_fwd(fwd)
  make_custom_vjp(attr(f, "fun"), fwd, bwd, attr(f, "nondiff_argnums"), symbolic_zeros)
}

# ---- remat_opt ----------------------------------------------------------------------

remat_opt_p <- new_primitive("remat_opt", multiple_results = TRUE)
remat_opt_p$impl <- function(args, params) eval_closed(params$fwd_jaxpr, args)
remat_opt_p$abstract_eval <- function(avals, params) jaxpr_out_avals(params$fwd_jaxpr$jaxpr)
remat_opt_p$lowering <- lower_fun(function(args, params) eval_closed(params$fwd_jaxpr, args),
                                  multiple_results = TRUE)
remat_opt_p$jvp <- function(primals, tangents, params) {
  f <- function(...) eval_closed(params$fwd_jaxpr, list(...))
  r <- jvp(f, primals, lapply(tangents, instantiate_zero))
  list(r[[1L]], r[[2L]])
}
remat_opt_p$batch <- function(vals, dims, params, axis) {
  f <- function(...) eval_closed(params$fwd_jaxpr, list(...))
  ax <- lapply(dims, function(d) if (is.na(d)) NULL else d)
  out <- do.call(vmap(f, in_axes = ax), vals)
  list(out, rep(0L, length(out)))
}
remat_opt_p$transpose <- function(cts, args, params) {
  jax_error("remat optimization for custom_vjp does not support higher-order AD",
            class = "jax_not_implemented_error")
}

# Port of optimize_remat_of_custom_vjp_fwd: the forward rule becomes one remat_opt
# equation whose outputs are the residuals followed by the primal outputs.
remat_opt_wrap_fwd <- function(fwd) {
  force(fwd)
  function(...) {
    fl <- tree_flatten(list(...))
    leaves <- lapply(fl$leaves, as_jax)
    store <- new.env(parent = emptyenv())
    h <- trace_hoisted(function(ls) {
      out <- do.call(fwd, tree_unflatten(fl$treedef, ls))
      pf <- tree_flatten(out[[1L]])
      rf <- tree_flatten(out[[2L]])
      store$prim_tree <- pf$treedef
      store$res_tree <- rf$treedef
      store$nres <- length(rf$leaves)
      lapply(c(rf$leaves, pf$leaves), as_jax)
    }, lapply(leaves, aval_of))
    outs <- bind(remat_opt_p, c(h$consts, leaves),
                 list(fwd_jaxpr = h$jaxpr, num_consts = length(h$consts), num_res = store$nres))
    nres <- store$nres
    list(tree_unflatten(store$prim_tree, outs[seq_along(outs) > nres]),
         tree_unflatten(store$res_tree, outs[seq_len(nres)]))
  }
}

# ---- stop_gradient ------------------------------------------------------------------

stop_gradient_p <- new_primitive("stop_gradient")
stop_gradient_p$abstract_eval <- function(avals, params) avals[[1L]]
stop_gradient_p$lowering <- function(ctx, ins, ai, ao, params) ins[[1L]]
stop_gradient_p$batch <- function(vals, dims, params, axis) list(bind(stop_gradient_p, vals), dims[1L])
defjvp_zero(stop_gradient_p)

#' Stop gradients
#'
#' Port of `jax.lax.stop_gradient`: the identity, with a zero derivative.
#' @param x A pytree of arrays.
#' @export
lax_stop_gradient <- function(x) {
  tree_map(function(l) {
    l <- as_jax(l)
    if (is_inexact_dtype(aval_of(l)$dtype)) bind(stop_gradient_p, list(l)) else l
  }, x)
}

#' @rdname lax_stop_gradient
#' @export
stop_gradient <- lax_stop_gradient
