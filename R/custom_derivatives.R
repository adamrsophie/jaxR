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

# Port of api_util.resolve_kwargs: the call is matched to the formals of `fun` (as
# inspect.Signature.bind does) and the missing arguments get their defaults
# (apply_defaults), so that the argument tree does not depend on how the caller wrote
# the call. Formals before `...` are then passed by position, the `...` arguments
# follow as given, and formals after `...` (keyword only in Python) keep their names.
# The arguments are left as they are when the call cannot be matched or a formal
# without a default is missing.
resolve_args <- function(fun, args) {
  fm <- formals(fun)
  if (!length(fm)) return(args)
  fnames <- names(fm)
  syms <- lapply(seq_along(args), function(i) as.name(paste0(".a", i)))
  names(syms) <- names(args)
  mc <- tryCatch(match.call(fun, as.call(c(list(quote(f)), syms)), expand.dots = FALSE),
                 error = function(e) NULL)
  if (is.null(mc)) return(args)
  got <- as.list(mc)[-1L]
  idx <- function(sym) as.integer(sub(".a", "", as.character(sym), fixed = TRUE))
  env <- new.env(parent = environment(fun) %||% globalenv())
  for (nm in setdiff(fnames, "...")) {
    if (!is.null(got[[nm]])) {
      assign(nm, args[[idx(got[[nm]])]], envir = env)
    } else if (!identical(fm[[nm]], quote(expr = ))) {
      eval(call("delayedAssign", nm, fm[[nm]], env, env))
    } else {
      return(args)
    }
  }
  vals <- tryCatch(mget(setdiff(fnames, "..."), envir = env), error = function(e) NULL)
  if (is.null(vals)) return(args)
  dots_at <- match("...", fnames)
  before <- if (is.na(dots_at)) fnames else fnames[seq_len(dots_at - 1L)]
  out <- unname(vals[before])
  if (!is.na(dots_at)) {
    d <- got[["..."]]
    dv <- lapply(d, function(sym) args[[idx(sym)]])
    names(dv) <- names(d)
    # Formals after `...` are keyword-only arguments: as with ba.args in JAX, the ones
    # not given are left out (the function then uses its default, and missing() holds).
    after <- fnames[-seq_len(dots_at)]
    after <- after[!vapply(after, function(nm) is.null(got[[nm]]), NA)]
    out <- c(out, dv, vals[after])
    if (!length(dv) && !length(after)) names(out) <- NULL
  }
  out
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
    # The uses of keys while tracing are undone. A trace that stages the call itself
    # applies the signature (as for checkpoint); any other runs the function or its
    # rule, which consume the keys themselves.
    kr_revert(attr(cj, "kr"))
    tr <- current_trace()
    kr_sig <- if (inherits(tr, "jaxpr_trace") && !isTRUE(tr$forward_known)) attr(cj, "kr")
    kr_apply_in(kr_sig, fl$leaves, "custom_derivative")
    outs <- bind(custom_jvp_call_p, fl$leaves,
                 list(fun = fun_flat, jvp = jvp_flat, call_jaxpr = cj,
                      symbolic_zeros = isTRUE(symbolic_zeros)))
    kr_apply_out(kr_sig, fl$leaves, outs)
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
    # The rule logs into its own collector; the logs leave the batched trace as
    # batched outputs.
    sub <- NULL
    ltree <- NULL
    r <- batched_run(function(v) {
      sub <<- vjp_logs_begin()
      on.exit(vjp_logs_end(sub))
      o <- bwd(v[seq_len(nres)], v[nres + seq_len(length(cts))])
      lf <- tree_flatten(sub$items)
      ltree <<- lf$treedef
      c(o, lapply(lf$leaves, as_jax))
    }, in_dims, size, name, c(res, cts))
    nl <- treedef_num_leaves(ltree)
    io <- seq_len(length(r$vals) - nl)
    if (sub$unsupported && !is.null(.jx$vjp_logs)) .jx$vjp_logs$unsupported <- TRUE
    if (nl) vjp_logs_add(tree_unflatten(ltree, Map(function(v, b) if (b) v else lax_broadcast(v, size),
                                                   drop_first(r$vals, length(io)),
                                                   drop_first(r$batched, length(io)))))
    Map(function(c, b, d) {
      if (is.na(d)) { if (b) lax_reduce_sum(c, 0L) else c }
      else if (b) c else lax_broadcast(c, size)
    }, r$vals[io], r$batched[io], as.list(dims))
  }
  cj <- with_trace(trace$parent, trace_to_jaxpr(fun_b, lapply(vals, aval_of)))
  outs <- with_trace(trace$parent, bind(custom_vjp_call_p, vals,
    update_params(params, list(fun = fun_b, fwd = fwd_b, bwd = bwd_b, call_jaxpr = cj))))
  st$open <- FALSE
  Map(function(o, b) batch_tracer(trace, o, if (b) 0L else NA_integer_), outs, st$out)
}

make_custom_vjp <- function(fun, fwd, bwd, nondiff_argnums, symbolic_zeros = FALSE,
                            mode = "plain") {
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
      store$perturbed <- perturbed
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
    # Port of _flatten_bwd: one cotangent per leaf of the differentiable arguments. It is
    # made in its own environment, so that the backward pass does not keep the arguments.
    bwd_flat <- custom_vjp_bwd_flat(bwd, args[nd], store, fl$treedef, lapply(fl$leaves, aval_of),
                                    unname(args[diff_pos]), length(diff_pos), mode)
    cj <- trace_to_jaxpr(fun_flat, lapply(fl$leaves, aval_of))
    out_tree <- store$out_tree
    # The uses of keys while tracing are undone. A trace that stages the call itself
    # applies the signature (as for checkpoint); any other runs the function or its
    # rule, which consume the keys themselves.
    kr_revert(attr(cj, "kr"))
    tr <- current_trace()
    kr_sig <- if (inherits(tr, "jaxpr_trace") && !isTRUE(tr$forward_known)) attr(cj, "kr")
    kr_apply_in(kr_sig, fl$leaves, "custom_derivative")
    outs <- bind(custom_vjp_call_p, fl$leaves,
                 list(fun = fun_flat, fwd = fwd_flat, bwd = bwd_flat, call_jaxpr = cj,
                      symbolic_zeros = symbolic_zeros))
    kr_apply_out(kr_sig, fl$leaves, outs)
    tree_unflatten(out_tree, outs)
  }
  structure(f, class = c("jax_custom_vjp", "function"), fun = fun, fwd = fwd, bwd = bwd,
            nondiff_argnums = nondiff_argnums, symbolic_zeros = symbolic_zeros, mode = mode)
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

#' Custom VJP rules that log or take gradient accumulators
#'
#' Ports of `custom_vjp.defvjp_with_logs` and `custom_vjp.defvjp_with_accums`.
#'
#' With `custom_vjp_defvjp_with_logs()`, `bwd` returns a pair `list(in_cts, logs)`:
#' the usual cotangents and a named list of values to log out of the backward pass
#' (or `NULL`). Apply the VJP function with its `with_logs` method to receive them:
#' `vjp_fun$with_logs(ct)` returns `list(arg_cts, logs)`. A plain call drops the logs.
#' Log names must be unique within a backward pass.
#'
#' With `custom_vjp_defvjp_with_accums()`, `bwd(res, g, ...)` receives one gradient
#' accumulator per argument of the function (with the structure of the arguments, see
#' [ad_ValAccum]) after its two usual arguments, pushes the cotangent contributions
#' into them and returns `NULL` or a named list of logs.
#' @inheritParams custom_vjp_defvjp
#' @return A function with the custom VJP rule.
#' @examples
#' \dontrun{
#' f <- custom_vjp(function(x, y) x * y)
#' f <- custom_vjp_defvjp_with_logs(f, function(x, y) list(x * y, list(x, y)),
#'   function(res, g) list(list(g * res[[2]], g * res[[1]]), list(g = g)))
#' vjp(f, 3, 4)[[2]]$with_logs(1)
#' }
#' @export
custom_vjp_defvjp_with_logs <- function(f, fwd, bwd, symbolic_zeros = FALSE, optimize_remat = FALSE) {
  g <- custom_vjp_defvjp(f, fwd, bwd, symbolic_zeros, optimize_remat)
  make_custom_vjp(attr(g, "fun"), attr(g, "fwd"), bwd, attr(g, "nondiff_argnums"),
                  symbolic_zeros, mode = "logs")
}

#' @rdname custom_vjp_defvjp_with_logs
#' @export
custom_vjp_defvjp_with_accums <- function(f, fwd, bwd, symbolic_zeros = FALSE, optimize_remat = FALSE) {
  g <- custom_vjp_defvjp(f, fwd, bwd, symbolic_zeros, optimize_remat)
  make_custom_vjp(attr(g, "fun"), attr(g, "fwd"), bwd, attr(g, "nondiff_argnums"),
                  symbolic_zeros, mode = "accums")
}

#' Custom rematerialization rules
#'
#' Ports of `custom_vjp.defremat` and `custom_vjp.defremat_with_logs`. `fwd` is the
#' forward rule of [custom_vjp_defvjp()]; `rem(res, ...)` receives its residuals and the
#' arguments of the function and returns `list(out, res2)`; `bwd` is the backward rule
#' and receives `res2`. jaxR runs `rem` right after `fwd` on the forward pass, which is
#' what JAX does outside [checkpoint()]; under [checkpoint()] JAX defers `rem` to the
#' backward pass, which saves memory but gives the same values.
#' @inheritParams custom_vjp_defvjp
#' @param rem The rematerialization rule.
#' @return A function with the custom VJP rule.
#' @export
custom_vjp_defremat <- function(f, fwd, rem, bwd) {
  custom_vjp_defvjp(f, defremat_fwd(fwd, rem), bwd)
}

#' @rdname custom_vjp_defremat
#' @export
custom_vjp_defremat_with_logs <- function(f, fwd, rem, bwd) {
  custom_vjp_defvjp_with_logs(f, defremat_fwd(fwd, rem), bwd)
}

defremat_fwd <- function(fwd, rem) {
  force(fwd); force(rem)
  function(...) {
    r <- fwd(...)
    r2 <- rem(r[[2L]], ...)
    if (!is.list(r2) || !is.null(oldClass(r2)) || length(r2) != 2L)
      jax_error("custom_vjp rem rule must return a pair list(out, res)", class = "jax_type_error")
    list(r[[1L]], r2[[2L]])
  }
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

# The backward rule of a custom_vjp function as a function of flat residuals and
# cotangents (_flatten_bwd). `diff_args` is used only for the paths in error messages.
custom_vjp_bwd_flat <- function(bwd, nondiff, store, treedef, leaf_avals, diff_args, n_diff,
                                mode = "plain") {
  force(bwd); force(nondiff); force(store); force(treedef); force(leaf_avals); force(n_diff)
  force(mode)
  paths <- vapply(tree_leaves_with_path(diff_args), function(kp) keystr(kp[[1L]]), "")
  rm(diff_args)
  arg_path <- function(i) paths[[i]]
  function(res, cts) {
    py_res <- tree_unflatten(store$res_tree, res)
    py_cts <- tree_unflatten(store$out_tree, cts)
    if (mode == "accums") {
      # Port of call_with_accums: one accumulator per argument leaf, a NullAccum for an
      # argument that is not perturbed or has no tangent (float0).
      pert <- store$perturbed %||% rep(TRUE, length(leaf_avals))
      accs <- Map(function(a, p) {
        t <- tangent_aval(a)
        new_accum(if (isTRUE(p) && t$dtype != "float0") "ValAccum" else "NullAccum", t)
      }, leaf_avals, as.list(pert))
      logs <- do.call(bwd, c(nondiff, list(py_res, py_cts), tree_unflatten(treedef, accs)))
      vjp_logs_check(logs, paste0("Custom VJP bwd rule was registered with ",
                                  "custom_vjp_defvjp_with_accums, and so must return NULL or ",
                                  "a named list; its result"))
      vjp_logs_add(logs)
      r <- lapply(accs, accum_freeze)
      cts_in <- r
    } else {
      r <- do.call(bwd, c(nondiff, list(py_res, py_cts)))
      if (mode == "logs") {
        if (!is.list(r) || !is.null(oldClass(r)) || length(r) != 2L)
          jax_error("Custom VJP bwd rule was registered with custom_vjp_defvjp_with_logs and ",
                    "so must produce a pair list(in_cts, logs), but got ", format(r),
                    class = "jax_type_error")
        vjp_logs_check(r[[2L]], paste0("Custom VJP bwd rule was registered with ",
                                       "custom_vjp_defvjp_with_logs, and so the second element ",
                                       "of the pair it returns"))
        vjp_logs_add(r[[2L]])
        r <- r[[1L]]
      }
      cts_in <- NULL
    }
    if (is.null(cts_in)) cts_in <- if (is.list(r) && is.null(oldClass(r)) && length(r) == n_diff)
      tryCatch(flatten_up_to(treedef, r, null_ok = TRUE), error = function(e) NULL)
    if (is.null(cts_in) && length(leaf_avals))
      jax_error("Custom VJP bwd rule must produce an output with the same container ",
                "(pytree) structure as the arguments of the primal function, and in ",
                "particular must produce a list of length equal to the number of arguments ",
                "to the primal function, but got bwd output structure ",
                format(tree_structure(r)), " for primal input structure ",
                format(treedef), ".", class = "jax_type_error")
    Map(function(c, a, i) {
      want <- tangent_aval(a)
      if (is.null(c) || want$dtype == "float0") return(zeros_aval(want))
      sz <- is_symbolic_zero(c)
      if (!sz) c <- as_jax(c)
      ca <- if (sz) c$aval else aval_of(c)
      if (!identical(ca$shape, a$shape) || (sz && ca$dtype != want$dtype)) {
        kp <- arg_path(i)
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
    }, cts_in %||% list(), leaf_avals, seq_along(leaf_avals))
  }
}
