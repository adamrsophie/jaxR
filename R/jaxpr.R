# Jaxprs: staging trace, partial evaluation trace, evaluation, printing.

new_var <- function(id, aval) structure(list(id = id, aval = aval), class = "jax_var")
new_lit <- function(val) structure(list(val = val, aval = aval_of(val)), class = "jax_lit")
is_var <- function(x) inherits(x, "jax_var")
is_lit <- function(x) inherits(x, "jax_lit")

# A staging trace. With forward_known = TRUE it is the partial evaluation trace:
# primitives whose inputs contain none of its tracers run on the parent trace.
new_jaxpr_trace <- function(parent, forward_known = FALSE) {
  t <- new_trace(if (forward_known) "pe" else "jaxpr", parent)
  class(t) <- c(if (forward_known) "pe_trace", "jaxpr_trace", "jax_trace")
  t$forward_known <- forward_known
  t$eqns <- list()
  t$neqns <- 0L
  t$consts <- list()
  t$constvars <- list()
  t$const_index <- new.env(parent = emptyenv())
  t
}

# Atom (var or literal) for a value used inside a staging trace.
to_atom <- function(trace, x) {
  if (is_own(x, trace)) return(new_var(x$id, x$aval))
  # A literal is written from R numbers, which hold 53 bits: a larger 64 bit integer
  # stays an operand, so no bits are lost.
  if (is_jax_array(x) && length(x$aval$shape) == 0L && !is_complex_dtype(x$aval$dtype) &&
      x$aval$dtype != "float0" &&
      (!x$aval$dtype %in% c("int64", "uint64") || abs(array_host(x)) < 2^53) &&
      (is.null(trace$keep_const) || !any(vapply(trace$keep_const, identical, NA, x))))
    return(new_lit(x))
  key <- if (is_tracer(x)) as.character(x$id) else NULL
  if (!is.null(key)) {
    v <- trace$const_index[[key]]
    if (!is.null(v)) return(v)
  }
  v <- new_var(next_id(), aval_of(x))
  n <- length(trace$consts) + 1L
  trace$consts[[n]] <- x
  trace$constvars[[n]] <- v
  if (!is.null(key)) assign(key, v, envir = trace$const_index)
  v
}

# Effects (io_callback, debug callbacks) of jaxprs held in the parameters of an
# equation (control flow bodies, checkpoint and custom derivative calls) belong to the
# equation, as the effects of a jaxpr do in JAX.
params_effectful <- function(params) {
  for (p in params) {
    if (inherits(p, "jax_jaxpr")) {
      if (isTRUE(p$effectful)) return(TRUE)
    } else if (inherits(p, "jax_closed_jaxpr")) {
      if (isTRUE(p$jaxpr$effectful)) return(TRUE)
    } else if (is.list(p) && is.null(oldClass(p)) && length(p)) {
      if (params_effectful(p)) return(TRUE)
    }
  }
  FALSE
}

eqn_effectful <- function(e) {
  if (!is.null(e$effectful)) e$effectful else isTRUE(e$prim$effectful) || params_effectful(e$params)
}

stage_eqn <- function(trace, prim, args, params) {
  avals <- lapply(args, aval_of)
  out_avals <- abstract_eval(prim, avals, params)
  if (!prim$multiple_results) out_avals <- list(out_avals)
  ins <- lapply(args, function(a) to_atom(trace, a))
  outs <- lapply(out_avals, function(a) new_tracer(trace, a))
  eqn <- list(prim = prim, invars = ins, params = params,
              outvars = lapply(outs, function(o) new_var(o$id, o$aval)),
              effectful = isTRUE(prim$effectful) || params_effectful(params))
  trace$neqns <- trace$neqns + 1L
  trace$eqns[[trace$neqns]] <- eqn
  if (prim$multiple_results) outs else outs[[1L]]
}

process_jaxpr <- function(trace, prim, args, params) {
  if (trace$forward_known) {
    if (!any(vapply(args, is_own, NA, trace = trace)))
      return(with_trace(trace$parent, bind(prim, args, params)))
    ov <- prim$override$pe
    if (!is.null(ov)) return(ov(trace, args, params))
    if (!is.null(prim$partial_eval)) return(prim$partial_eval(trace, args, params))
  } else {
    ov <- prim$override$jaxpr
    if (!is.null(ov)) return(ov(trace, args, params))
    if (identical(prim$name, "convert_element_type")) {
      folded <- fold_convert_constant(args[[1L]], params)
      if (!is.null(folded)) return(folded)
    }
  }
  stage_eqn(trace, prim, args, params)
}

# Port of lax._convert_elt_type_folding_rule (pe.const_fold_rules): the conversion of a
# scalar constant is done on the host instead of being staged, so `x * 2` stages
# `mul x 2.0` like JAX. Values without an exact image in the new dtype are not folded
# and keep the conversion of the backend.
fold_convert_constant <- function(x, params) {
  dt <- params$new_dtype
  if (!is_jax_array(x) || length(x$aval$shape)) return(NULL)
  sd <- x$aval$dtype
  if (is_ext_dtype(dt) || is_ext_dtype(sd) || dt == "float0" || sd == "float0") return(NULL)
  h <- as.vector(array_host(x))
  if (is.complex(h) && !is_complex_dtype(dt)) h <- Re(h)
  if (is.na(h) && !is_inexact_dtype(dt)) return(NULL)
  if (is_int_dtype(dt)) {
    ii <- iinfo(dt)
    if (!is.finite(h) || trunc(h) < ii$min || trunc(h) > ii$max) return(NULL)
    h <- trunc(h)   # in range, so the conversion truncates toward zero
  }
  # A double holds 64-bit integers exactly only below 2^53.
  if (is_int_dtype(sd) && abs(h) >= 2^53) return(NULL)
  v <- lax_full_like(x, h, dtype = dt)
  v$aval <- new_aval(integer(0), dt, isTRUE(params$weak_type))
  v
}

# Build a jaxpr from a staging trace. Returns list(jaxpr, consts).
to_jaxpr <- function(trace, in_tracers, outs) {
  invars <- lapply(in_tracers, function(t) new_var(t$id, t$aval))
  outvars <- lapply(outs, function(o) to_atom(trace, o))
  jaxpr <- list(constvars = trace$constvars, invars = invars, outvars = outvars,
                eqns = trace$eqns[seq_len(trace$neqns)])
  res <- dce_jaxpr(jaxpr, trace$consts)
  list(jaxpr = renumber_jaxpr(res$jaxpr), consts = res$consts)
}

atom_ids <- function(atoms) {
  ids <- integer(0)
  for (a in atoms) if (is_var(a)) ids <- c(ids, a$id)
  ids
}

# Dead code elimination. Equations with effects (the io_callback and debug callbacks of
# R/callbacks.R and R/debug.R, also inside nested jaxprs) are always kept.
dce_jaxpr <- function(jaxpr, consts) {
  live <- new.env(parent = emptyenv())
  for (id in atom_ids(jaxpr$outvars)) assign(as.character(id), TRUE, envir = live)
  keep <- logical(length(jaxpr$eqns))
  for (i in rev(seq_along(jaxpr$eqns))) {
    e <- jaxpr$eqns[[i]]
    if (eqn_effectful(e) ||
        any(vapply(e$outvars, function(v) exists(as.character(v$id), envir = live, inherits = FALSE), NA))) {
      keep[i] <- TRUE
      for (id in atom_ids(e$invars)) assign(as.character(id), TRUE, envir = live)
    }
  }
  jaxpr$eqns <- jaxpr$eqns[keep]
  used_c <- vapply(jaxpr$constvars, function(v) exists(as.character(v$id), envir = live, inherits = FALSE), NA)
  jaxpr$constvars <- jaxpr$constvars[used_c]
  list(jaxpr = jaxpr, consts = consts[used_c])
}

# Renumber variables to 1..n so evaluation can use a plain list as environment.
renumber_jaxpr <- function(jaxpr) {
  map <- new.env(parent = emptyenv())
  n <- 0L
  fresh <- function(v) {
    n <<- n + 1L
    assign(as.character(v$id), n, envir = map)
    new_var(n, v$aval)
  }
  rd <- function(a) {
    if (is_lit(a)) return(a)
    k <- map[[as.character(a$id)]]
    if (is.null(k)) jax_error("internal: unbound variable in jaxpr")
    new_var(k, a$aval)
  }
  jaxpr$constvars <- lapply(jaxpr$constvars, fresh)
  jaxpr$invars <- lapply(jaxpr$invars, fresh)
  jaxpr$eqns <- lapply(jaxpr$eqns, function(e) {
    e$invars <- lapply(e$invars, rd)
    e$outvars <- lapply(e$outvars, fresh)
    e
  })
  jaxpr$outvars <- lapply(jaxpr$outvars, rd)
  jaxpr$nvars <- n
  jaxpr$effectful <- any(vapply(jaxpr$eqns, eqn_effectful, NA))
  class(jaxpr) <- "jax_jaxpr"
  jaxpr
}

closed_jaxpr <- function(jaxpr, consts = list()) {
  structure(list(jaxpr = jaxpr, consts = consts), class = "jax_closed_jaxpr")
}

# Evaluate a jaxpr under the current trace (so evaluation is itself transformable).
eval_jaxpr <- function(jaxpr, consts, args) {
  env <- vector("list", jaxpr$nvars)
  for (i in seq_along(jaxpr$constvars)) env[[jaxpr$constvars[[i]]$id]] <- consts[[i]]
  if (length(args) != length(jaxpr$invars))
    jax_error("internal: eval_jaxpr got ", length(args), " args for ", length(jaxpr$invars), " invars")
  for (i in seq_along(jaxpr$invars)) env[[jaxpr$invars[[i]]$id]] <- args[[i]]
  rd <- function(a) if (is_lit(a)) a$val else env[[a$id]]
  for (e in jaxpr$eqns) {
    out <- bind(e$prim, lapply(e$invars, rd), e$params)
    if (e$prim$multiple_results) {
      for (j in seq_along(e$outvars)) env[[e$outvars[[j]]$id]] <- out[[j]]
    } else {
      env[[e$outvars[[1L]]$id]] <- out
    }
  }
  lapply(jaxpr$outvars, rd)
}

eval_closed <- function(cj, args) eval_jaxpr(cj$jaxpr, cj$consts, args)

# Turn constvars into leading invars.
convert_constvars <- function(jaxpr) {
  jaxpr$invars <- c(jaxpr$constvars, jaxpr$invars)
  jaxpr$constvars <- list()
  jaxpr
}

# Trace a function of a flat list of values to a closed jaxpr.
# `fun` takes a list of values and returns a list of values.
trace_to_jaxpr <- function(fun, in_avals, parent = current_trace()) {
  trace <- new_jaxpr_trace(parent)
  on.exit(trace$alive <- FALSE, add = TRUE)
  in_tracers <- lapply(in_avals, function(a) new_tracer(trace, a))
  outs <- with_trace(trace, fun(in_tracers))
  res <- to_jaxpr(trace, in_tracers, outs)
  closed_jaxpr(res$jaxpr, res$consts)
}

# Trace with constants hoisted to explicit leading inputs. Returns list(jaxpr (closed,
# no consts), consts (values to pass as leading operands)).
trace_hoisted <- function(fun, in_avals) {
  cj <- trace_to_jaxpr(fun, in_avals)
  list(jaxpr = closed_jaxpr(convert_constvars(cj$jaxpr)), consts = cj$consts)
}

jaxpr_out_avals <- function(jaxpr) lapply(jaxpr$outvars, `[[`, "aval")
jaxpr_in_avals <- function(jaxpr) lapply(jaxpr$invars, `[[`, "aval")

# ---- printing ---------------------------------------------------------------------

var_name <- function(i) {
  s <- ""
  repeat {
    s <- paste0(letters[(i - 1L) %% 26L + 1L], s)
    i <- (i - 1L) %/% 26L
    if (i == 0L) break
  }
  s
}

atom_str <- function(a) {
  if (is_lit(a)) {
    v <- as.vector(array_host(a$val))
    if (is.logical(v)) return(if (v) "True" else "False")
    s <- format(v, digits = 8)
    if (is_float_dtype(a$aval$dtype) && !grepl("[.eEnN]", s)) s <- paste0(s, ".0")
    return(paste0(s, if (a$aval$weak) "" else paste0(":", aval_str(a$aval))))
  }
  var_name(a$id)
}

param_str <- function(v) {
  if (inherits(v, "jax_closed_jaxpr") || inherits(v, "jax_jaxpr")) return("{...}")
  if (is.function(v)) return("<fn>")
  if (is.null(v)) return("None")
  if (is.list(v)) return(paste0("(", paste(vapply(v, param_str, ""), collapse = ","), ")"))
  if (inherits(v, "jax_aval")) return(aval_str(v))
  if (length(v) == 1L) return(format(v))
  paste0("(", paste(format(v), collapse = ","), ")")
}

format_jaxpr <- function(jaxpr, indent = "") {
  vd <- function(v) paste0(var_name(v$id), ":", aval_str(v$aval))
  head <- paste0("{ lambda ", paste(vapply(jaxpr$constvars, vd, ""), collapse = " "),
                 "; ", paste(vapply(jaxpr$invars, vd, ""), collapse = " "), ". let")
  lines <- vapply(jaxpr$eqns, function(e) {
    ps <- e$params
    ps <- ps[!vapply(ps, is.function, NA)]
    pstr <- if (length(ps)) paste0("[", paste(names(ps), vapply(ps, param_str, ""), sep = "=",
                                              collapse = " "), "]") else ""
    paste0(indent, "    ", paste(vapply(e$outvars, vd, ""), collapse = " "), " = ",
           e$prim$name, pstr, " ", paste(vapply(e$invars, atom_str, ""), collapse = " "))
  }, "")
  out <- paste0(indent, "  in (", paste(vapply(jaxpr$outvars, atom_str, ""), collapse = ", "), ") }")
  paste(c(head, lines, out), collapse = "\n")
}

#' @export
print.jax_jaxpr <- function(x, ...) {
  cat(format_jaxpr(x), "\n")
  invisible(x)
}

#' @export
print.jax_closed_jaxpr <- function(x, ...) {
  cat(format_jaxpr(x$jaxpr), "\n")
  invisible(x)
}

#' Trace a function to a jaxpr
#'
#' Port of `jax.make_jaxpr`.
#' @param fun Function of arrays (or pytrees of arrays).
#' @param static_argnums 0-based positions of arguments treated as static; negative
#'   values count from the end.
#' @param return_shape Whether to return also the output structure, as a pytree like the
#'   output of `fun` with a [ShapeDtypeStruct()] at each leaf.
#' @return A function with the same arguments as `fun` that returns the closed jaxpr, or
#'   `list(jaxpr, shapes)` with `return_shape`.
#' @export
make_jaxpr <- function(fun, static_argnums = integer(0), return_shape = FALSE) {
  if (!is.function(fun))
    jax_error("make_jaxpr: expected a function, got an object of class '", class(fun)[1L], "'",
              class = "jax_type_error")
  function(...) {
    args <- list(...)
    st <- as.integer(static_argnums)
    st <- ifelse(st < 0L, st + length(args), st) + 1L
    dyn <- setdiff(seq_along(args), st)
    fl <- tree_flatten(args[dyn])
    avals <- lapply(fl$leaves, aval_of)
    out_tree <- NULL
    cj <- trace_to_jaxpr(function(leaves) {
      a <- args
      a[dyn] <- tree_unflatten(fl$treedef, leaves)
      ofl <- tree_flatten(do.call(fun, a))
      out_tree <<- ofl$treedef
      ofl$leaves
    }, avals)
    if (!return_shape) return(cj)
    shapes <- lapply(jaxpr_out_avals(cj$jaxpr), function(a) {
      ShapeDtypeStruct(a$shape, a$dtype, weak_type = isTRUE(a$weak))
    })
    list(cj, tree_unflatten(out_tree, shapes))
  }
}

# ---- partial evaluation of jaxprs -------------------------------------------------

# Split a closed jaxpr into known and unknown parts.
#   unknowns: logical per input; instantiate: logical per output (force unknown).
# Returns list(known: closed jaxpr (known ins -> known outs ++ residuals),
#              unknown: closed jaxpr (residuals ++ unknown ins -> unknown outs),
#              out_unknowns, num_res).
partial_eval_jaxpr <- function(cj, unknowns, instantiate = FALSE) {
  jaxpr <- cj$jaxpr
  n_out <- length(jaxpr$outvars)
  instantiate <- rep_len(instantiate, n_out)
  K <- new_jaxpr_trace(current_trace())
  U <- new_jaxpr_trace(K, forward_known = TRUE)
  on.exit({ K$alive <- FALSE; U$alive <- FALSE }, add = TRUE)
  in_avals <- jaxpr_in_avals(jaxpr)
  k_in <- list()
  u_in <- list()
  ins <- vector("list", length(in_avals))
  for (i in seq_along(in_avals)) {
    if (unknowns[i]) {
      t <- new_tracer(U, in_avals[[i]])
      u_in[[length(u_in) + 1L]] <- t
    } else {
      t <- new_tracer(K, in_avals[[i]])
      k_in[[length(k_in) + 1L]] <- t
    }
    ins[[i]] <- t
  }
  outs <- with_trace(U, eval_jaxpr(jaxpr, cj$consts, ins))
  out_unknowns <- vapply(outs, is_own, NA, trace = U) | instantiate
  u_res <- to_jaxpr(U, u_in, outs[out_unknowns])
  residuals <- u_res$consts
  unknown <- closed_jaxpr(convert_constvars(u_res$jaxpr))
  k_res <- to_jaxpr(K, k_in, c(outs[!out_unknowns], residuals))
  known <- closed_jaxpr(k_res$jaxpr, k_res$consts)
  list(known = known, unknown = unknown, out_unknowns = out_unknowns,
       num_res = length(residuals))
}
