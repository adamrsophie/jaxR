# Automatic differentiation: JVP trace, linearize by partial evaluation, transposition.

# ---- JVP trace --------------------------------------------------------------------

new_jvp_trace <- function(parent, tag = NULL) new_trace("jvp", parent, tag)

maybe_jvp_tracer <- function(trace, primal, tangent) {
  if (is_zero(tangent)) return(primal)
  if (aval_of(tangent)$dtype == "float0") return(primal)
  new_tracer(trace, aval_of(primal), primal = primal, tangent = tangent)
}

primal_tangent <- function(x, trace) {
  if (is_own(x, trace)) return(list(x$primal, x$tangent))
  list(x, zero_of(tangent_aval(aval_of(x))))
}

process_jvp <- function(trace, prim, args, params) {
  ov <- prim$override$jvp
  if (!is.null(ov)) return(ov(trace, args, params))
  pts <- lapply(args, primal_tangent, trace = trace)
  primals <- lapply(pts, `[[`, 1L)
  tangents <- lapply(pts, `[[`, 2L)
  if (all(vapply(tangents, is_zero, NA)))
    return(with_trace(trace$parent, bind(prim, primals, params)))
  if (is.null(prim$jvp))
    jax_error("Differentiation rule for '", prim$name, "' not implemented")
  res <- with_trace(trace$parent, prim$jvp(primals, tangents, params))
  if (prim$multiple_results) {
    Map(function(p, t) maybe_jvp_tracer(trace, p, t), res[[1L]], res[[2L]])
  } else {
    maybe_jvp_tracer(trace, res[[1L]], res[[2L]])
  }
}

# ---- rule helpers (ports of ad.defjvp, deflinear, defbilinear) ---------------------

add_tangents <- function(a, b) {
  if (is_zero(a)) return(b)
  if (is_zero(b)) return(a)
  bind(add_any_p, list(a, b))
}

# Per-argument JVP functions g(t, primals, params) (NULL for non-differentiable args).
defjvp <- function(prim, ...) {
  gs <- list(...)
  prim$jvp <- function(primals, tangents, params) {
    ans <- bind(prim, primals, params)
    out <- NULL
    for (i in seq_along(tangents)) {
      t <- tangents[[i]]
      if (is_zero(t) || i > length(gs) || is.null(gs[[i]])) next
      contrib <- gs[[i]](t, primals, params)
      out <- if (is.null(out)) contrib else add_tangents(out, contrib)
    }
    if (is.null(out)) out <- zero_of(tangent_aval(aval_of(ans)))
    list(ans, out)
  }
  invisible(prim)
}

# Like defjvp but each g also receives the primal output: g(t, ans, primals, params).
defjvp2 <- function(prim, ...) {
  gs <- list(...)
  prim$jvp <- function(primals, tangents, params) {
    ans <- bind(prim, primals, params)
    out <- NULL
    for (i in seq_along(tangents)) {
      t <- tangents[[i]]
      if (is_zero(t) || i > length(gs) || is.null(gs[[i]])) next
      contrib <- gs[[i]](t, ans, primals, params)
      out <- if (is.null(out)) contrib else add_tangents(out, contrib)
    }
    if (is.null(out)) out <- zero_of(tangent_aval(aval_of(ans)))
    list(ans, out)
  }
  invisible(prim)
}

# Primitive jointly linear in the arguments `linear` (1-based positions); other
# arguments are treated as constants.
deflinear <- function(prim, transpose, linear = NULL) {
  prim$jvp <- function(primals, tangents, params) {
    ans <- bind(prim, primals, params)
    lin <- linear %||% seq_along(primals)
    if (all(vapply(tangents[lin], is_zero, NA))) {
      tz <- if (prim$multiple_results) lapply(ans, function(a) zero_of(tangent_aval(aval_of(a))))
            else zero_of(tangent_aval(aval_of(ans)))
      return(list(ans, tz))
    }
    targs <- primals
    for (i in lin) targs[[i]] <- instantiate_zero(tangents[[i]])
    list(ans, bind(prim, targs, params))
  }
  prim$transpose <- transpose
  invisible(prim)
}

# ---- transposition ----------------------------------------------------------------

undefined_primal <- function(aval) structure(list(aval = aval), class = "jax_undefined")
is_undefined <- function(x) inherits(x, "jax_undefined")

# Transpose a linear jaxpr. `args` has an undefined_primal for every linear input and
# values for the others. Returns cotangents for every input (Zero for non-linear ones).
backward_pass <- function(jaxpr, consts, args, cts_out) {
  n <- jaxpr$nvars
  primal_env <- vector("list", n)
  is_lin <- logical(n)
  # As in JAX, an input that is linear is the UndefinedPrimal the caller passes, with its
  # aval (a weak type can differ from the one of the variable).
  lin_in <- vector("list", n)
  for (i in seq_along(jaxpr$constvars)) primal_env[[jaxpr$constvars[[i]]$id]] <- consts[[i]]
  for (i in seq_along(jaxpr$invars)) {
    id <- jaxpr$invars[[i]]$id
    if (is_undefined(args[[i]])) {
      is_lin[id] <- TRUE
      lin_in[[id]] <- args[[i]]
    } else primal_env[[id]] <- args[[i]]
  }
  rd <- function(a) if (is_lit(a)) a$val else primal_env[[a$id]]
  lin_eqns <- list()
  for (e in jaxpr$eqns) {
    ins_lin <- vapply(e$invars, function(a) is_var(a) && is_lin[a$id], NA)
    if (any(ins_lin)) {
      for (v in e$outvars) is_lin[v$id] <- TRUE
      lin_eqns[[length(lin_eqns) + 1L]] <- e
    } else {
      out <- bind(e$prim, lapply(e$invars, rd), e$params)
      if (!e$prim$multiple_results) out <- list(out)
      for (j in seq_along(e$outvars)) primal_env[[e$outvars[[j]]$id]] <- out[[j]]
    }
  }
  ct_env <- vector("list", n)
  add_ct <- function(id, ct) {
    if (is.null(ct) || is_zero(ct)) return()
    cur <- ct_env[[id]]
    ct_env[[id]] <<- if (is.null(cur)) ct else add_tangents(cur, ct)
  }
  for (i in seq_along(jaxpr$outvars)) {
    a <- jaxpr$outvars[[i]]
    if (is_var(a) && is_lin[a$id]) add_ct(a$id, cts_out[[i]])
  }
  for (e in rev(lin_eqns)) {
    cts <- lapply(e$outvars, function(v) ct_env[[v$id]] %||% zero_of(v$aval))
    if (all(vapply(cts, is_zero, NA))) next
    invals <- lapply(e$invars, function(a) {
      if (is_var(a) && is_lin[a$id]) lin_in[[a$id]] %||% undefined_primal(a$aval) else rd(a)
    })
    if (is.null(e$prim$transpose))
      jax_error("Transpose rule (for reverse-mode differentiation) for '", e$prim$name,
                "' not implemented")
    cts_in <- e$prim$transpose(if (e$prim$multiple_results) cts else cts[[1L]], invals, e$params)
    for (j in seq_along(e$invars)) {
      a <- e$invars[[j]]
      if (is_var(a) && is_lin[a$id] && j <= length(cts_in)) add_ct(a$id, cts_in[[j]])
    }
  }
  lapply(seq_along(jaxpr$invars), function(i) {
    if (!is_undefined(args[[i]])) return(zero_of(tangent_aval(aval_of(args[[i]]))))
    ct_env[[jaxpr$invars[[i]]$id]] %||% zero_of(jaxpr$invars[[i]]$aval)
  })
}

# ---- linearize / vjp --------------------------------------------------------------

# fun: list of values -> list of values. Returns primals_out, the linear jaxpr with its
# residual consts, and which inputs carry tangents.
linearize_flat <- function(fun, primals, in_nzs = NULL, keep = NULL) {
  pe <- new_jaxpr_trace(current_trace(), forward_known = TRUE)
  # Arrays in `keep` (the arguments of vjp) stay residuals even when they are scalars,
  # so that vjp can tell which residuals are arguments.
  if (length(keep)) pe$keep_const <- keep
  jt <- new_jvp_trace(pe)
  on.exit({ jt$alive <- FALSE; pe$alive <- FALSE }, add = TRUE)
  diff_in <- vapply(primals, function(p) is_inexact_dtype(aval_of(p)$dtype), NA)
  if (!is.null(in_nzs)) diff_in <- diff_in & in_nzs
  tan_in <- list()
  ins <- vector("list", length(primals))
  for (i in seq_along(primals)) {
    p <- primals[[i]]
    if (diff_in[i]) {
      t <- new_tracer(pe, tangent_aval(aval_of(p)))
      tan_in[[length(tan_in) + 1L]] <- t
      ins[[i]] <- new_tracer(jt, aval_of(p), primal = p, tangent = t)
    } else {
      ins[[i]] <- p
    }
  }
  outs <- with_trace(jt, fun(ins))
  pts <- lapply(outs, primal_tangent, trace = jt)
  primals_out <- lapply(pts, `[[`, 1L)
  for (p in primals_out) if (is_own(p, pe))
    jax_error("internal: primal output of linearize depends on tangents")
  out_zero <- vapply(pts, function(pt) is_zero(pt[[2L]]), NA)
  tangents_out <- lapply(pts, function(pt) instantiate_zero(pt[[2L]]))
  lin <- to_jaxpr(pe, tan_in, tangents_out)
  list(primals_out = primals_out, jaxpr = lin$jaxpr, consts = lin$consts, diff_in = diff_in,
       in_avals = lapply(primals, aval_of), out_zero = out_zero)
}

# Port of api.tuptree_flags: expand a tree of logicals that is a prefix of `td` (its
# lists are matched by their number of children, and by name when both are dicts) to one
# flag per leaf.
tuptree_flags <- function(prefix, td, name) {
  if (is.logical(prefix) && length(prefix) == 1L && !is.na(prefix)) return(rep(prefix, td$n))
  # NULL is a container without children: it matches any such part of the arguments.
  if (is.null(prefix) && !identical(td$type, "leaf") && !length(td$children)) return(logical(0))
  # A registered container matches any container through its children (containers
  # are matched by their number of children only, as in JAX).
  h <- pytree_handler(prefix)
  if (!is.null(h) && !identical(td$type, "leaf")) prefix <- unname(h$flatten(prefix)$children)
  if (!is.list(prefix) || !is.null(oldClass(prefix)) || identical(td$type, "leaf") ||
      length(prefix) != length(td$children))
    jax_error("the ", name, " argument must be a tree of logicals that is a prefix of the ",
              "primal arguments, got ", format(tree_structure(prefix)), " for ", format(td),
              class = "jax_value_error")
  if (!is.null(td$ord) && !is.null(dict_order(names(prefix)))) prefix <- prefix[td$names]
  as.logical(unlist(Map(function(p, k) tuptree_flags(p, k, name), prefix, td$children)))
}

vjp_flat <- function(fun, primals, in_nzs = NULL, keep = NULL) {
  lin <- linearize_flat(fun, primals, in_nzs, keep)
  # The residuals are handed back to the caller and passed in on each call, so that the
  # function does not keep arguments the caller chose not to save.
  consts <- lin$consts
  lin$consts <- NULL
  n <- length(primals)
  rm(primals)
  f_vjp <- function(cts, consts) {
    args <- lapply(lin$jaxpr$invars, function(v) undefined_primal(v$aval))
    cts <- Map(function(ct, a) if (is.null(ct)) zero_of(a$aval) else ct, cts, lin$jaxpr$outvars)
    cts_lin <- backward_pass(lin$jaxpr, consts, args, cts)
    out <- vector("list", n)
    k <- 0L
    for (i in seq_len(n)) {
      if (lin$diff_in[i]) {
        k <- k + 1L
        out[[i]] <- instantiate_zero(cts_lin[[k]])
      } else {
        out[[i]] <- zeros_aval(tangent_aval(lin$in_avals[[i]]))
      }
    }
    out
  }
  # The function sees only what it needs, not this frame with the residuals.
  # The primal outputs are left out: one of them may be an argument that is not saved.
  environment(f_vjp) <- list2env(list(lin = lin[c("jaxpr", "diff_in", "in_avals")], n = n),
                                 parent = topenv())
  list(primals_out = lin$primals_out, f_vjp = f_vjp, consts = consts, out_zero = lin$out_zero)
}

# Port of api.mirror_tree_map: a tree of `vals` (one per leaf of `td`) in the shape of
# `td`, with plain lists in place of registered containers.
mirror_tree <- function(td, vals) {
  pos <- 0L
  build <- function(d) {
    switch(d$type,
      leaf = { pos <<- pos + 1L; vals[[pos]] },
      null = NULL,
      list = {
        out <- lapply(d$children, build)
        nms <- d$names
        if (!is.null(d$ord)) {
          back <- order(d$ord)
          out <- out[back]
          nms <- nms[back]
        }
        names(out) <- nms
        out
      },
      # The children of a registered container are positional, whatever their names.
      unname(lapply(d$children, build)))
  }
  build(td)
}

vjp_sentinel <- function(cls) structure(list(), class = c(cls, "jax_vjp_sentinel"))

#' @export
print.jax_vjp_sentinel <- function(x, ...) {
  cat(sub("^jax_", "", class(x)[1L]), "()\n", sep = "")
  invisible(x)
}

#' @export
`$.jax_VJP` <- function(x, name) {
  st <- attr(x, "state")
  switch(name, args_res = st$args_res, out_nzs = st$out_nzs,
         # Ports of VJP.with_logs (the function then returns list(cotangents, logs)) and
         # VJP.with_refs (see ad_GradValue()).
         with_logs = structure(x, want_logs = TRUE),
         with_refs = function(...) structure(x, refs = list(...)),
         jax_error("'VJP' object has no attribute '", name, "'", class = "jax_attribute_error"))
}

#' @export
`$<-.jax_VJP` <- function(x, name, value) {
  if (!identical(name, "args_res"))
    jax_error("cannot set attribute '", name, "' of a VJP function", class = "jax_attribute_error")
  assign("args_res", value, envir = attr(x, "state"))
  x
}

#' @export
`$.jax_linearized` <- function(x, name) {
  if (identical(name, "out_nzs")) return(attr(x, "out_nzs"))
  jax_error("linearized function has no attribute '", name, "'", class = "jax_attribute_error")
}

#' @export
print.jax_VJP <- function(x, ...) {
  cat("<VJP function>\n")
  invisible(x)
}

#' @export
print.jax_linearized <- function(x, ...) {
  cat("<linearized function>\n")
  invisible(x)
}

# Wrap a function of pytrees as a function of flat leaves; out tree recorded in `store`.
flat_fun <- function(fun, in_treedef, store) {
  function(leaves) {
    out <- do.call(fun, tree_unflatten(in_treedef, leaves))
    fl <- tree_flatten(out)
    store$out_tree <- fl$treedef
    lapply(fl$leaves, as_jax)
  }
}

# ---- argument checks (ports of the checks in jax/_src/api.py) ---------------------------

aval_long <- function(a) paste0(a$dtype, "[", paste(a$shape, collapse = ","), "]")
aval_repr <- function(a) paste0("ShapedArray(", aval_long(a), if (a$weak) ", weak_type=True", ")")

check_callable <- function(fun, name) {
  if (!is.function(fun))
    jax_error(name, ": expected a function, got an object of class '", class(fun)[1L], "'",
              class = "jax_type_error")
}

# Port of api_util.argnums_partial2 (with _ensure_index_tuple and _ensure_inbounds).
# `argnums` holds 0-based positions; negative values count from the end. A list (the
# analog of a Python tuple) always gives a list of results, even with one element.
argnums_partial <- function(fun, argnums, args) {
  nums <- unlist(argnums)
  if (!is.numeric(nums) || !length(nums) || anyNA(nums) || any(nums != trunc(nums)))
    jax_error("argnums must be an integer or a vector (or list) of integers",
              class = "jax_type_error")
  nums <- as.integer(nums)
  n <- length(args)
  bad <- nums < -n | nums >= n
  if (any(bad))
    jax_error("Positional argument indices, e.g. for `static_argnums`, must have value greater ",
              "than or equal to -length(args) and less than length(args), but got value ",
              nums[bad][1L], " for length(args) == ", n, ".", class = "jax_value_error")
  pos <- nums %% n + 1L
  f <- function(...) {
    a <- args
    a[pos] <- list(...)
    do.call(fun, a)
  }
  list(f = f, dyn = unname(args[pos]), single = !is.list(argnums) && length(nums) == 1L)
}

check_input_dtype_revderiv <- function(name, holomorphic, allow_int, x) {
  if (is_ref(x))
    jax_error(name, " cannot differentiate with respect to a Ref-typed argument, because the ",
              "gradient for a ref must itself be accumulated into a ref. (Ref arguments not ",
              "selected by argnums are fine: they are not differentiated.)",
              class = "jax_type_error")
  dt <- aval_of(x)$dtype
  if (holomorphic && !is_complex_dtype(dt))
    jax_error(name, " with holomorphic = TRUE requires inputs with complex dtype, but got ", dt,
              ".", class = "jax_type_error")
  if (is_inexact_dtype(dt)) return(invisible())
  if (dt == "float0")
    jax_error(name, " requires numerical-valued inputs (input dtype that is a sub-dtype of ",
              "bool or number), but got ", dt, ".", class = "jax_type_error")
  if (!allow_int)
    jax_error(name, " requires real- or complex-valued inputs (input dtype that is a ",
              "sub-dtype of inexact), but got ", dt, ". If you want to use Boolean- or ",
              "integer-valued inputs, use vjp or set allow_int to TRUE.",
              class = "jax_type_error")
}

check_output_dtype_revderiv <- function(name, holomorphic, x) {
  dt <- aval_of(x)$dtype
  if (holomorphic) {
    if (!is_complex_dtype(dt))
      jax_error(name, " with holomorphic = TRUE requires outputs with complex dtype, but got ",
                dt, ".", class = "jax_type_error")
  } else if (is_complex_dtype(dt)) {
    jax_error(name, " requires real-valued outputs (output dtype that is a sub-dtype of ",
              "floating), but got ", dt, ". For holomorphic differentiation, pass ",
              "holomorphic = TRUE. For differentiation of non-holomorphic functions involving ",
              "complex outputs, use vjp directly.", class = "jax_type_error")
  } else if (!is_float_dtype(dt)) {
    jax_error(name, " requires real-valued outputs (output dtype that is a sub-dtype of ",
              "floating), but got ", dt, ". For differentiation of functions with integer ",
              "outputs, use vjp directly.", class = "jax_type_error")
  }
}

check_input_dtype_jacfwd <- function(holomorphic, x) {
  dt <- aval_of(x)$dtype
  if (holomorphic) {
    if (!is_complex_dtype(dt))
      jax_error("jacfwd with holomorphic = TRUE requires inputs with complex dtype, but got ",
                dt, ".", class = "jax_type_error")
  } else if (!is_float_dtype(dt)) {
    jax_error("jacfwd requires real-valued inputs (input dtype that is a sub-dtype of ",
              "floating), but got ", dt, ". For holomorphic differentiation, pass ",
              "holomorphic = TRUE. For differentiation of non-holomorphic functions involving ",
              "complex inputs or integer inputs, use jvp directly.", class = "jax_type_error")
  }
}

check_output_dtype_jacfwd <- function(holomorphic, aval) {
  if (holomorphic && !is_complex_dtype(aval$dtype))
    jax_error("jacfwd with holomorphic = TRUE requires outputs with complex dtype, but got ",
              aval$dtype, ".", class = "jax_type_error")
}

check_scalar <- function(x) {
  msg <- "Gradient only defined for scalar-output functions. Output "
  if (!is_jax_value(x))
    jax_error(msg, "was a ", if (is.list(x)) "list" else class(x)[1L], ".", class = "jax_type_error")
  sh <- aval_of(x)$shape
  if (!length(sh)) return(invisible())
  extract <- if (prod_int(sh) == 1L)
    paste0("extract the scalar with jnp_reshape(output, integer(0))",
           if (length(sh) == 1L) " or output[1]", ", ") else ""
  jax_error(msg, "had shape: ", py_tuple(sh), ". To get the gradient, ", extract,
            "reduce the output to a scalar with jnp_sum(output), or use jacobian() to ",
            "differentiate a function with a non-scalar output.", class = "jax_type_error")
}

# Leaves of `tree` matched against `treedef` (named lists by name). `fail` receives the
# treedef of `tree` when the structures differ.
match_leaves <- function(treedef, tree, fail) {
  parts <- tryCatch(flatten_up_to(treedef, tree), error = function(e) NULL)
  if (is.null(parts) || any(vapply(parts, is_container, NA))) fail(tree_structure(tree))
  lapply(parts, as_jax)
}

# Port of _vjp_check_ct_avals: a cotangent has the shape and dtype of its output (the
# weak type is ignored); any dtype is accepted for the float0 cotangent of an integer
# output.
check_ct_avals <- function(cts, out_avals) {
  for (i in seq_along(cts)) {
    got <- aval_of(cts[[i]])
    want <- tangent_aval(out_avals[[i]])
    if (aval_eq(got, want) || (want$dtype == "float0" && identical(got$shape, want$shape))) next
    jax_error("unexpected JAX type (e.g. shape/dtype) for argument to VJP function: got ",
              aval_long(got), ", but expected ", aval_long(want), " because the corresponding ",
              "output of the differentiated function had JAX type ", aval_long(out_avals[[i]]),
              class = "jax_value_error")
  }
}

# Wrap a function returning list(output, aux): the aux value leaves the transformation
# as a plain value and is recorded in `store`.
aux_fun <- function(fun, store, name) {
  function(...) {
    r <- fun(...)
    if (!is.list(r) || !is.null(oldClass(r)) || length(r) != 2L)
      jax_error(name, ": a function used with has_aux = TRUE must return list(output, aux)",
                class = "jax_type_error")
    tr <- current_trace()
    store$aux <- tree_map(function(l) strip_own(l, tr), r[[2L]])
    r[[1L]]
  }
}

# The VJP function lives in its own environment, which holds the residuals but not the
# arguments of vjp(), so that arguments marked not saveable are not kept.
make_vjp_fun <- function(st, spec, opaque, n_in, arg_paths, out_tree, out_avals, in_tree, f_vjp,
                         in_avals) {
  # Forced, so that no promise keeps the frame of vjp() (and its arguments) alive.
  force(st); force(spec); force(opaque); force(n_in); force(arg_paths)
  force(out_tree); force(out_avals); force(in_tree); force(f_vjp); force(in_avals)
  vjp_fun <- function(ct) {
    self <- sys.function()
    refs <- attr(self, "refs")
    kinds <- if (!is.null(refs)) vjp_ref_kinds(refs, in_tree, in_avals, arg_paths)
    ar <- tree_leaves(st$args_res, is_leaf = function(x) inherits(x, "jax_vjp_sentinel"))
    if (length(ar) != n_in)
      jax_error("args_res of a VJP function must have the structure of the primal ",
                "arguments", class = "jax_value_error")
    miss <- unique(spec[!is.na(spec)][vapply(ar[spec[!is.na(spec)]], inherits, NA,
                                              "jax_NotSaveable")])
    if (length(miss))
      jax_error("the VJP function was applied before restoring its not-saveable residuals.\n\n",
                "Because `saveable_args` was passed to `vjp`, some argument values that\n",
                "would have been saved for the backward pass were instead replaced with\n",
                "`NotSaveable()` sentinels. Before the VJP function can be applied, these\n",
                "values must be restored, e.g. by assigning to the VJP function's `args_res`\n",
                "attribute. The values not yet restored correspond to:\n",
                paste0("  * the argument at position args", arg_paths[miss],
                       " of the differentiated function;", collapse = "\n"),
                class = "jax_value_error")
    consts <- opaque
    consts[!is.na(spec)] <- lapply(ar[spec[!is.na(spec)]], as_jax)
    cts <- match_leaves(out_tree, ct, function(got)
      jax_error("unexpected tree structure of argument to vjp function: got ", format(got),
                ", but expected to match ", format(out_tree), class = "jax_value_error"))
    check_ct_avals(cts, out_avals)
    # A float0 cotangent carries no information: use a symbolic zero.
    cts <- Map(function(c, a) if (is_inexact_dtype(a$dtype)) c else NULL, cts, out_avals)
    # The logs of the backward rules that run (custom_vjp_defvjp_with_logs and others).
    log <- vjp_logs_begin()
    on.exit(vjp_logs_end(log), add = TRUE)
    out <- f_vjp(cts, consts)
    if (!is.null(kinds))
      out <- Map(function(o, k) switch(k$kind, value = o, none = ad_DidntWant(), ref = {
        if (!is.null(o) && !is_zero(o)) k$ref[] <- k$ref[] + o
        ad_GradRef()
      }), out, kinds)
    res <- tree_unflatten(in_tree, out)
    if (!isTRUE(attr(self, "want_logs"))) return(res)
    # Logs from rules inside the body of a control flow primitive or checkpoint are
    # values of its finished trace: JAX stacks or sums them, jaxR does not.
    if (log$unsupported)
      jax_error("backward-pass logs from a custom rule inside the body of a control flow ",
                "primitive (scan, cond, while_loop, ...) or of checkpoint are not supported in jaxR",
                class = "jax_not_implemented_error")
    list(res, log$items)
  }
  structure(vjp_fun, class = c("jax_VJP", "function"), state = st)
}

#' Vector-Jacobian product
#'
#' Port of `jax.vjp`.
#' @param fun Function of pytrees of arrays.
#' @param ... Primal arguments.
#' @param has_aux Whether `fun` returns `list(output, aux)`.
#' @param saveable_args A tree of logicals, a prefix of the list of primal arguments
#'   like `in_nzs`, by default `TRUE`: which arguments may be kept for the backward
#'   pass. An argument array that the backward pass needs as it is (not a value computed
#'   from it) and that is marked `FALSE` is not kept: `vjp_fun$args_res` shows a
#'   `NotSaveable()` sentinel in its place, and the value must be put back
#'   (`vjp_fun$args_res[[i]] <- x`) before `vjp_fun` is called. Arguments the backward
#'   pass does not need show `NotNeeded()`. Only jaxR arrays given as arguments count;
#'   R numbers are converted to new arrays, like Python scalars in JAX.
#' @param in_nzs Optional tree of logicals, a prefix of the list of primal arguments
#'   (for example `list(TRUE, FALSE)`): which inputs may have a nonzero tangent. The
#'   tangent of an input marked `FALSE` is a symbolic zero during linearization, and its
#'   cotangent is zero. The default marks every input.
#' @return `list(out, vjp_fun)` (plus `aux` when `has_aux`), where `vjp_fun(ct)` returns a
#'   list of cotangents, one per primal argument. `ct` must have the structure of `out`,
#'   and each of its leaves the shape and dtype of the matching output (as in JAX there
#'   is no broadcasting or casting; the cotangent of an integer output is ignored).
#'   `vjp_fun$out_nzs` is a tree of logicals with the structure of `out` (plain lists in
#'   place of registered containers) telling which outputs have a tangent that is not
#'   known to be zero, and `vjp_fun$args_res` holds the saved arguments (see
#'   `saveable_args`).
#' @export
vjp <- function(fun, ..., has_aux = FALSE, saveable_args = TRUE, in_nzs = NULL) {
  check_callable(fun, "vjp")
  primals <- list(...)
  fl <- tree_flatten(primals)
  store <- new.env(parent = emptyenv())
  f <- if (has_aux) aux_fun(fun, store, "vjp") else fun
  saveable <- tuptree_flags(saveable_args, fl$treedef, "saveable_args")
  nz <- if (!is.null(in_nzs)) tuptree_flags(in_nzs, fl$treedef, "in_nzs")
  # As in JAX, only arrays passed in count as arguments saved verbatim; R values are
  # converted to new arrays, like Python scalars.
  given <- vapply(fl$leaves, function(l) is_jax_array(l) || is_tracer(l), NA)
  leaves <- lapply(fl$leaves, as_jax)
  res <- vjp_flat(flat_fun(f, fl$treedef, store), leaves, nz, keep = leaves[given])
  out_tree <- store$out_tree
  out_avals <- lapply(res$primals_out, aval_of)
  # Port of the RSpec analysis: residuals that are argument arrays come from args_res.
  # As with the dict of ids in JAX, an array passed twice maps to its last position, and
  # every position holding a used array is used.
  spec <- vapply(res$consts, function(cn) {
    k <- which(given & vapply(leaves, identical, NA, cn))
    if (length(k)) k[length(k)] else NA_integer_
  }, 1L)
  used_arrays <- leaves[unique(spec[!is.na(spec)])]
  used <- which(given & vapply(leaves, function(l) any(vapply(used_arrays, identical, NA, l)), NA))
  opaque <- res$consts
  opaque[!is.na(spec)] <- list(NULL)
  res$consts <- NULL
  st <- new.env(parent = emptyenv())
  st$args_res <- mirror_tree(fl$treedef, lapply(seq_along(leaves), function(i) {
    if (!(i %in% used)) vjp_sentinel("jax_NotNeeded")
    else if (saveable[i]) leaves[[i]] else vjp_sentinel("jax_NotSaveable")
  }))
  st$out_nzs <- mirror_tree(out_tree, as.list(!res$out_zero))
  arg_paths <- vapply(tree_leaves_with_path(unname(primals)), function(kp) keystr(kp[[1L]]), "")
  vjp_fun <- make_vjp_fun(st, spec, opaque, length(given), arg_paths, out_tree, out_avals,
                          fl$treedef, res$f_vjp, lapply(leaves, aval_of))
  out <- tree_unflatten(out_tree, res$primals_out)
  if (has_aux) list(out, vjp_fun, lower_aux(store$aux)) else list(out, vjp_fun)
}

# Aux outputs are computed under the JVP trace of linearize, which has finished when
# they are returned: replace such tracers by their primal values.
lower_aux <- function(aux) {
  tree_map(function(l) {
    while (is_tracer(l) && identical(l$trace$kind, "jvp") && !l$trace$alive) l <- l$primal
    l
  }, aux)
}

#' Linearize a function
#'
#' Port of `jax.linearize`.
#' @param fun Function of pytrees.
#' @param ... Primal arguments.
#' @param has_aux Whether `fun` returns `list(output, aux)`.
#' @param in_nzs Optional tree of logicals, a prefix of the list of primal arguments:
#'   which inputs may have a nonzero tangent (see [vjp()]). The linearized function
#'   still takes a tangent for every input and ignores those marked `FALSE`.
#' @return `list(out, jvp_fun)` (plus `aux` when `has_aux`) where `jvp_fun(...)` maps
#'   tangents (one per primal argument, with the shapes and dtypes of the primals; a
#'   float0 array for an integer primal) to output tangents. `jvp_fun$out_nzs` tells
#'   which outputs have a tangent that is not known to be zero (see [vjp()]).
#' @export
linearize <- function(fun, ..., has_aux = FALSE, in_nzs = NULL) {
  check_callable(fun, "linearize")
  primals <- list(...)
  fl <- tree_flatten(primals)
  store <- new.env(parent = emptyenv())
  f <- if (has_aux) aux_fun(fun, store, "linearize") else fun
  nz <- if (!is.null(in_nzs)) tuptree_flags(in_nzs, fl$treedef, "in_nzs")
  lin <- linearize_flat(flat_fun(f, fl$treedef, store), lapply(fl$leaves, as_jax), nz)
  out_tree <- store$out_tree
  f_jvp <- function(...) {
    ts <- match_leaves(fl$treedef, list(...), function(got)
      jax_error("expected ", format(fl$treedef), ", got ", format(got), class = "jax_type_error"))
    for (i in seq_along(ts)) {
      want <- tangent_aval(lin$in_avals[[i]])
      if (!aval_eq(aval_of(ts[[i]]), want))
        jax_error("linearized function called on tangent values inconsistent with the ",
                  "original primal values:\nGot tangent aval ", aval_long(aval_of(ts[[i]])),
                  " for primal aval ", aval_long(lin$in_avals[[i]]), " but expected ",
                  aval_long(want), ".", class = "jax_value_error")
    }
    tree_unflatten(out_tree, eval_jaxpr(lin$jaxpr, lin$consts, ts[lin$diff_in]))
  }
  f_jvp <- structure(f_jvp, class = c("jax_linearized", "function"),
                     out_nzs = mirror_tree(out_tree, as.list(!lin$out_zero)))
  out <- tree_unflatten(out_tree, lin$primals_out)
  if (has_aux) list(out, f_jvp, lower_aux(store$aux)) else list(out, f_jvp)
}

#' Jacobian-vector product
#'
#' Port of `jax.jvp`.
#' @param fun Function of pytrees.
#' @param primals List of primal arguments.
#' @param tangents List of tangents with the same structure as `primals`. Each tangent
#'   has the shape and dtype of its primal; the tangent of an integer or boolean primal
#'   is a float0 array (`jnp_zeros(shape, "float0")`).
#' @param has_aux Whether `fun` returns `list(output, aux)`.
#' @return `list(primals_out, tangents_out)` (plus aux).
#' @export
jvp <- function(fun, primals, tangents, has_aux = FALSE) {
  check_callable(fun, "jvp")
  plain <- function(x) is.list(x) && is.null(oldClass(x))
  if (!plain(primals) || !plain(tangents))
    jax_error("primal and tangent arguments to jvp must be lists; found ", class(primals)[1L],
              " and ", class(tangents)[1L], ".", class = "jax_type_error")
  fl <- tree_flatten(primals)
  ps <- lapply(fl$leaves, as_jax)
  ts <- match_leaves(fl$treedef, tangents, function(got)
    jax_error("primal and tangent arguments to jvp must have the same tree structure; ",
              "primals have tree structure ", format(fl$treedef), " whereas tangents have ",
              "tree structure ", format(got), ".", class = "jax_type_error"))
  for (i in seq_along(ps)) {
    pa <- aval_of(ps[[i]])
    ta <- aval_of(ts[[i]])
    if (ta$dtype != tangent_dtype(pa$dtype))
      jax_error("primal and tangent arguments to jvp do not match; dtypes must be equal, or ",
                "in case of int/bool primal dtype the tangent dtype must be float0. Got primal ",
                "dtype ", pa$dtype, " and so expected tangent dtype ", tangent_dtype(pa$dtype),
                ", but got tangent dtype ", ta$dtype, " instead.", class = "jax_type_error")
    if (!identical(pa$shape, ta$shape))
      jax_error("jvp called with different primal and tangent shapes; Got primal shape ",
                py_tuple(pa$shape), " and tangent shape as ", py_tuple(ta$shape),
                class = "jax_value_error")
  }
  store <- new.env(parent = emptyenv())
  f <- if (has_aux) aux_fun(fun, store, "jvp") else fun
  ff <- flat_fun(f, fl$treedef, store)
  trace <- new_jvp_trace(current_trace())
  on.exit(trace$alive <- FALSE, add = TRUE)
  ins <- Map(function(p, t) {
    a <- aval_of(p)
    if (!is_inexact_dtype(a$dtype)) return(p)
    new_tracer(trace, a, primal = p, tangent = t)
  }, ps, ts)
  outs <- with_trace(trace, ff(ins))
  pts <- lapply(outs, primal_tangent, trace = trace)
  po <- lapply(pts, `[[`, 1L)
  to <- lapply(pts, function(pt) instantiate_zero(pt[[2L]]))
  res <- list(tree_unflatten(store$out_tree, po), tree_unflatten(store$out_tree, to))
  if (has_aux) c(res, list(store$aux)) else res
}

# jvp over flat lists for rules that differentiate a helper function: tangents may be
# symbolic zeros, and so may the returned ones.
jvp_flat <- function(fun, primals, tangents) {
  trace <- new_jvp_trace(current_trace())
  on.exit(trace$alive <- FALSE, add = TRUE)
  outs <- with_trace(trace, fun(Map(function(p, t) maybe_jvp_tracer(trace, p, t),
                                    primals, tangents)))
  pts <- lapply(outs, primal_tangent, trace = trace)
  list(lapply(pts, `[[`, 1L), lapply(pts, `[[`, 2L))
}

# has_aux outputs leave a JVP transformation as values (their primal part), as in JAX.
strip_own <- function(x, trace) {
  while (is_own(x, trace)) x <- x$primal
  x
}

#' Value and gradient of a scalar function
#'
#' Port of `jax.value_and_grad`.
#' @param fun Function returning a scalar.
#' @param argnums 0-based position (or positions) of the differentiated positional
#'   arguments; negative values count from the end. With a vector of several positions,
#'   or a list of positions (the analog of a Python tuple), the gradient is a list.
#' @param has_aux Whether `fun` returns `list(value, aux)`.
#' @param holomorphic Whether `fun` is promised to be holomorphic. Inputs and outputs
#'   must then be complex.
#' @param allow_int Whether integer and boolean inputs are allowed. Their gradient is a
#'   float0 array.
#' @return A function returning `list(value, grad)` (or `list(list(value, aux), grad)`).
#' @export
value_and_grad <- function(fun, argnums = 0L, has_aux = FALSE, holomorphic = FALSE,
                           allow_int = FALSE) {
  check_callable(fun, "value_and_grad")
  force(argnums)
  function(...) {
    args <- list(...)
    mx <- suppressWarnings(max(unlist(argnums)))
    if (is.numeric(mx) && is.finite(mx) && mx >= length(args))
      jax_error("differentiating with respect to argnums=",
                if (length(unlist(argnums)) == 1L) mx else py_tuple(unlist(argnums)),
                " requires at least ", mx + 1L, " positional arguments to be passed by the ",
                "caller, but got only ", length(args), " positional arguments.",
                class = "jax_type_error")
    ap <- argnums_partial(fun, argnums, args)
    for (l in tree_leaves(ap$dyn)) check_input_dtype_revderiv("grad", holomorphic, allow_int, l)
    r <- do.call(vjp, c(list(ap$f), ap$dyn, list(has_aux = has_aux)))
    ans <- r[[1L]]
    check_scalar(ans)
    check_output_dtype_revderiv("grad", holomorphic, ans)
    g <- r[[2L]](lax_full_like(ans, 1))
    if (ap$single) g <- g[[1L]]
    list(if (has_aux) list(ans, r[[3L]]) else ans, g)
  }
}

#' Gradient of a scalar function
#'
#' Port of `jax.grad`.
#' @inheritParams value_and_grad
#' @param ... Passed to `symcompR::grad()` when `fun` is a symbolic expression (the
#'   symbolic gradient of symcompR, so both packages can be attached together). Not
#'   used otherwise.
#' @return A function computing the gradient (or `list(grad, aux)` with `has_aux`).
#' @export
grad <- function(fun, argnums = 0L, has_aux = FALSE, holomorphic = FALSE, allow_int = FALSE,
                 ...) {
  if (!is.function(fun)) {
    if (missing(argnums)) return(delegate_symcompR("grad", fun, ...))
    return(delegate_symcompR("grad", fun, argnums, ...))
  }
  check_no_dots("grad", ...)
  vg <- value_and_grad(fun, argnums, has_aux, holomorphic, allow_int)
  function(...) {
    r <- vg(...)
    if (has_aux) list(r[[2L]], r[[1L]][[2L]]) else r[[2L]]
  }
}

check_no_dots <- function(name, ...) {
  if (...length())
    jax_error(name, ": unused argument", if (...length() > 1L) "s", " (",
              paste(names(list(...)) %||% rep("", ...length()), collapse = ", "), ")",
              class = "jax_type_error")
}

#' Transpose a linear function
#'
#' Port of `jax.linear_transpose`.
#' @param fun A linear function.
#' @param ... Primal example arguments (only shapes and dtypes are used).
#' @return A function mapping output cotangents (with the structure, shapes and dtypes
#'   of the output of `fun`) to a list of input cotangents.
#' @export
linear_transpose <- function(fun, ...) {
  check_callable(fun, "linear_transpose")
  primals <- list(...)
  fl <- tree_flatten(primals)
  avals <- lapply(fl$leaves, aval_of)
  store <- new.env(parent = emptyenv())
  cj <- trace_to_jaxpr(flat_fun(fun, fl$treedef, store), avals)
  out_tree <- store$out_tree
  out_avals <- lapply(cj$jaxpr$outvars, function(v) if (is_lit(v)) aval_of(v$val) else v$aval)
  dts <- c(vapply(avals, `[[`, "", "dtype"), vapply(out_avals, `[[`, "", "dtype"))
  if (!(all(dts == "float0" | vapply(dts, is_inexact_dtype, NA)) ||
        all(dts == "float0" | vapply(dts, dtype_kind, "") %in% c("i", "u"))))
    jax_error("linear_transpose only supports [float, complex, or float0] -> [float, complex, ",
              "or float0], and integer -> integer functions, but got [",
              paste(vapply(avals, `[[`, "", "dtype"), collapse = ", "), "] -> [",
              paste(vapply(out_avals, `[[`, "", "dtype"), collapse = ", "), "].",
              class = "jax_type_error")
  function(ct) {
    cts <- match_leaves(out_tree, ct, function(got)
      jax_error("cotangent tree does not match function output, expected ", format(out_tree),
                " but got ", format(got), class = "jax_type_error"))
    if (!all(mapply(function(c, a) aval_eq(aval_of(c), a), cts, out_avals)))
      jax_error("cotangent type does not match function output, expected (",
                paste(vapply(out_avals, aval_repr, ""), collapse = ", "), ") but got (",
                paste(vapply(cts, function(c) aval_repr(aval_of(c)), ""), collapse = ", "), ")",
                class = "jax_type_error")
    cts <- Map(function(c, a) if (a$dtype == "float0") zero_of(a) else c, cts, out_avals)
    args <- lapply(avals, undefined_primal)
    res <- backward_pass(cj$jaxpr, cj$consts, args, cts)
    tree_unflatten(fl$treedef, lapply(res, instantiate_zero))
  }
}

# ---- Jacobians (port of jax/_src/api.py jacfwd, jacrev, hessian) ----------------------

# Port of _possible_downcast: `example` is an aval giving the dtype and weak type.
possible_downcast <- function(x, example) {
  a <- aval_of(x)
  # The float0 cotangent of an integer input becomes zeros of the input dtype.
  if (a$dtype == "float0")
    return(lax_convert_element_type(lax_full(a$shape, 0, example$dtype), example$dtype,
                                    example$weak))
  if (is_complex_dtype(a$dtype) && !is_complex_dtype(example$dtype)) x <- lax_real(x)
  lax_convert_element_type(x, example$dtype, example$weak)
}

# Port of _unravel_array_into_pytree: split `arr` along `axis` into one block per leaf
# of `pytree`; block i has the shape of leaf i in place of that axis. Blocks take the
# dtype and weak type of `example` (an aval), or of their leaf when it is NULL.
unravel_array_into_pytree <- function(pytree, axis, example, arr) {
  fl <- tree_flatten(pytree)
  sh <- aval_of(arr)$shape
  if (axis < 0L) axis <- axis + length(sh)
  off <- 0L
  parts <- vector("list", length(fl$leaves))
  for (i in seq_along(parts)) {
    a <- if (inherits(fl$leaves[[i]], "jax_aval")) fl$leaves[[i]] else aval_of(fl$leaves[[i]])
    n <- prod_int(a$shape)
    piece <- if (aval_of(arr)$dtype == "float0") {
      zeros_aval(new_aval(c(sh[seq_len(axis)], a$shape, sh[-seq_len(axis + 1L)]), "float0"))
    } else {
      lax_reshape(lax_slice_in_dim(arr, off, off + n, 1L, axis),
                  c(sh[seq_len(axis)], a$shape, sh[-seq_len(axis + 1L)]))
    }
    off <- off + n
    parts[[i]] <- possible_downcast(piece, example %||% a)
  }
  tree_unflatten(fl$treedef, parts)
}

# Port of _std_basis: a pytree like `pytree` (of arrays or avals) whose leaves have a
# leading axis of size N, the total number of elements, holding the standard basis.
std_basis <- function(pytree) {
  avals <- lapply(tree_leaves(pytree), function(l) if (inherits(l, "jax_aval")) l else aval_of(l))
  n <- sum(vapply(avals, function(a) prod_int(a$shape), 1L))
  dt <- if (length(avals)) promote_types(vapply(avals, `[[`, "", "dtype"),
                                         vapply(avals, `[[`, NA, "weak"))$dtype
        else default_float()
  unravel_array_into_pytree(pytree, 1L, NULL, jnp_eye(n, dtype = dt))
}

#' Jacobian by reverse mode
#'
#' Port of `jax.jacrev` (and its alias `jax.jacobian`). For pytree inputs and outputs
#' the result has the output structure with each leaf replaced by the input structure.
#' Each block has the dtype and weak type of the input it differentiates.
#' @param fun Function.
#' @param argnums 0-based argument position(s); negative values count from the end. A
#'   list of positions (the analog of a Python tuple) always gives a list of Jacobians.
#' @param has_aux Whether `fun` returns `list(output, aux)`.
#' @param holomorphic Whether `fun` is promised to be holomorphic.
#' @param allow_int Whether integer and boolean inputs are allowed; their Jacobian
#'   blocks are zero.
#' @return A function computing the Jacobian (or `list(jacobian, aux)`).
#' @export
jacrev <- function(fun, argnums = 0L, has_aux = FALSE, holomorphic = FALSE, allow_int = FALSE) {
  check_callable(fun, "jacrev")
  force(argnums)
  function(...) {
    ap <- argnums_partial(fun, argnums, list(...))
    for (l in tree_leaves(ap$dyn)) check_input_dtype_revderiv("jacrev", holomorphic, allow_int, l)
    r <- do.call(vjp, c(list(ap$f), ap$dyn, list(has_aux = has_aux)))
    y <- r[[1L]]
    for (l in tree_leaves(y)) check_output_dtype_revderiv("jacrev", holomorphic, l)
    jac <- vmap(r[[2L]])(std_basis(y))
    example <- ap$dyn
    if (ap$single) {
      jac <- jac[[1L]]
      example <- example[[1L]]
    }
    jac_tree <- tree_map(function(ex, arr) unravel_array_into_pytree(y, 0L, aval_of(ex), arr),
                         example, jac)
    jac_tree <- tree_transpose(tree_structure(example), tree_structure(y), jac_tree)
    if (has_aux) list(jac_tree, r[[3L]]) else jac_tree
  }
}

#' @rdname jacrev
#' @param ... Passed to `symcompR::jacobian()` when `fun` is a symbolic expression. Not
#'   used otherwise.
#' @export
jacobian <- function(fun, argnums = 0L, has_aux = FALSE, holomorphic = FALSE,
                     allow_int = FALSE, ...) {
  if (!is.function(fun)) return(delegate_symcompR("jacobian", fun, argnums, ...))
  check_no_dots("jacobian", ...)
  jacrev(fun, argnums, has_aux, holomorphic, allow_int)
}

#' Jacobian by forward mode
#'
#' Port of `jax.jacfwd`. Each block has the dtype and weak type of the output it
#' differentiates.
#' @inheritParams jacrev
#' @export
jacfwd <- function(fun, argnums = 0L, has_aux = FALSE, holomorphic = FALSE) {
  check_callable(fun, "jacfwd")
  force(argnums)
  function(...) {
    ap <- argnums_partial(fun, argnums, list(...))
    for (l in tree_leaves(ap$dyn)) check_input_dtype_jacfwd(holomorphic, l)
    store <- new.env(parent = emptyenv())
    # The types of the primal outputs are recorded on the way: only the tangents (and
    # the aux output) leave the vmap.
    pushfwd <- function(ts) {
      r <- jvp(ap$f, ap$dyn, ts, has_aux = has_aux)
      store$y <- tree_map(aval_of, r[[1L]])
      r[-1L]
    }
    r <- vmap(pushfwd, out_axes = if (has_aux) list(-1L, NULL) else list(-1L))(std_basis(ap$dyn))
    y <- store$y
    for (a in tree_leaves(y)) check_output_dtype_jacfwd(holomorphic, a)
    example <- if (ap$single) ap$dyn[[1L]] else ap$dyn
    jac_tree <- tree_map(function(ya, arr) unravel_array_into_pytree(example, -1L, ya, arr),
                         y, r[[1L]])
    if (has_aux) list(jac_tree, r[[2L]]) else jac_tree
  }
}

#' Hessian
#'
#' Port of `jax.hessian` (`jacfwd(jacrev(fun))`). For a symcompR expression the call is
#' forwarded to `symcompR::hessian()`.
#' @inheritParams jacrev
#' @param ... Passed to symcompR when `fun` is a symbolic expression. Not used otherwise.
#' @export
hessian <- function(fun, argnums = 0L, has_aux = FALSE, holomorphic = FALSE, ...) {
  if (!is.function(fun)) return(delegate_symcompR("hessian", fun, argnums, ...))
  check_no_dots("hessian", ...)
  jacfwd(jacrev(fun, argnums, has_aux, holomorphic), argnums, has_aux, holomorphic)
}

delegate_symcompR <- function(name, ...) {
  if (!requireNamespace("symcompR", quietly = TRUE))
    jax_error(name, "() expects a function; symbolic expressions need the symcompR package")
  do.call(getExportedValue("symcompR", name), list(...))
}
