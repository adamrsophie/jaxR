# Mutable array references (port of the jax.Ref API of jax/_src/core.py, jax/_src/ref.py
# and jax/_src/indexing.py).
#
# A Ref is a mutable box holding an array. Reads and writes inside transformed functions
# act on traced values, like Python side state during JAX tracing; refs created inside a
# transformed function need no special handling. jit discharges the refs that a traced
# function reads or writes but did not create (arguments and closed over refs): their
# values become extra inputs of the compiled program, their final values extra outputs,
# and each call writes those outputs back into the refs.

is_ref <- function(x) inherits(x, "jax_ref")

ref_new_box <- function(value) {
  e <- new.env(parent = emptyenv())
  e$value <- value
  e$aval <- new_aval(aval_of(value)$shape, aval_of(value)$dtype)
  e$owner <- .jx$ref_ctx$id %||% 0L
  e$freed <- FALSE
  e$frozen <- FALSE
  class(e) <- "jax_ref"
  e
}

#' Mutable array references
#'
#' Ports of `jax.new_ref`, `jax.empty_ref`, `jax.free_ref`, `jax.freeze`, `jax.Ref` and
#' `jax.ds`. `new_ref(x)` makes a reference holding the array `x`; read it with `r[...]`
#' (NumPy indexing, `r[]` for the whole value) or `r$get()`, and write it with
#' `r[...] <- value`, `r$set(value, ...)` or `r$addupdate(value, ...)`. `freeze(r)`
#' returns the value and makes the reference read only; `free_ref(r)` releases it. `ds()`
#' builds a dynamic slice index `ds(start, size, stride)` for reference indexing.
#'
#' Refs may be read and written inside [jit()], [grad()] and [vmap()]: refs created
#' inside the transformed function behave as in JAX. [jit()] also handles refs passed as
#' arguments or closed over (their updates become outputs written back after each
#' call). [vmap()] maps a ref passed as an argument through its value: the function sees
#' a ref holding one element, and the ref holds the stacked result afterwards. A ref
#' that the body of [lax_scan()], [lax_while_loop()], [lax_fori_loop()] or a branch of
#' [lax_cond()] closes over is state of that operation. In other cases, writing an outer
#' ref inside a transformation leaves it holding a traced value that is unusable
#' outside, as JAX forbids.
#' @param init_val Initial value.
#' @param ty A [ShapeDtypeStruct()] for `empty_ref` (the value starts as zeros).
#' @param ref A reference.
#' @param memory_space,kind,pin Accepted for compatibility.
#' @param start,size,stride Start (static or traced), static size and stride of `ds()`.
#' @param ... Ignored: `Ref` names the class of references (port of `jax.Ref`) and
#'   cannot be called to build one.
#' @return `new_ref()` and `empty_ref()` return a reference (class `jax_ref`); `freeze()`
#'   returns an array; `free_ref()` returns `NULL` invisibly; `ds()` returns an index.
#' @name refs
NULL

#' @rdname refs
#' @export
new_ref <- function(init_val, memory_space = NULL, kind = NULL) {
  ref_new_box(as_jax(init_val))
}

#' @rdname refs
#' @export
empty_ref <- function(ty, memory_space = NULL, pin = FALSE) {
  a <- if (inherits(ty, "jax_aval")) ty else aval_of(ty)
  ref_new_box(lax_full(a$shape, 0, a$dtype))
}

#' @rdname refs
#' @export
free_ref <- function(ref) {
  ref_check(ref, "free_ref")
  assign("freed", TRUE, envir = ref)
  assign("value", NULL, envir = ref)
  invisible(NULL)
}

#' @rdname refs
#' @export
freeze <- function(ref) {
  ref_check(ref, "freeze")
  v <- ref_read(ref)
  assign("frozen", TRUE, envir = ref)
  v
}

#' @rdname refs
#' @format NULL
#' @export
Ref <- structure(function(...) {
  jax_error("Ref objects are created with new_ref() or empty_ref()", class = "jax_type_error")
}, class = c("jax_sparse_class", "function"), methods = list(), class_name = "Ref")

#' @rdname refs
#' @export
ds <- function(start, size, stride = 1L) {
  if (stride < 0L) jax_error("`stride` must be >= 0.", class = "jax_value_error")
  structure(list(start = start, size = as.integer(size), stride = as.integer(stride)), class = "jax_ds")
}

ref_check <- function(ref, what) {
  if (!is_ref(ref)) jax_error(what, " expects a Ref, got ", class(ref)[1L], class = "jax_type_error")
  if (isTRUE(ref$freed)) jax_error("Ref was freed with free_ref()", class = "jax_value_error")
}

# ---- jit discharge context --------------------------------------------------------------

# Make `ref` an input of the jit being traced (once), unless that jit created it.
ref_capture <- function(ref) {
  ctx <- .jx$ref_ctx
  if (is.null(ctx) || identical(ref$owner, ctx$id)) return(invisible())
  for (r in ctx$refs) if (identical(r, ref)) return(invisible())
  if (isTRUE(ctx$watch))
    stop(structure(class = c("jax_ref_touched", "condition"),
                   list(message = "a Ref was used by a control flow body that does not thread it",
                        call = NULL, ref = ref)))
  ref_check_alias(ref, ctx$originals)
  saved <- ref_value(ref)
  t <- new_tracer(ctx$trace, new_aval(aval_of(saved)$shape, aval_of(saved)$dtype, aval_of(saved)$weak))
  n <- length(ctx$refs) + 1L
  ctx$refs[[n]] <- ref
  ctx$saved[[n]] <- saved
  ctx$tracers[[n]] <- t
  assign("value", t, envir = ref)
  invisible()
}

# ---- refs closed over by control flow bodies ---------------------------------------------

# JAX keeps a Ref that the body of a scan, cond or while_loop closes over as state of the
# operation. Here the bodies run under a watch context (ref_body): the first use of an
# outer ref aborts the trace with the condition jax_ref_touched, and ref_threaded builds
# the operation again with that ref passed through the bodies as an explicit value.
# `run(refs)` builds the operation for the refs found so far.
ref_threaded <- function(run) {
  refs <- list()
  repeat {
    r <- tryCatch(run(refs), jax_ref_touched = function(e) e)
    if (!inherits(r, "jax_ref_touched")) return(r)
    # A use outside the bodies of this operation belongs to an enclosing operation.
    if (any(vapply(refs, identical, NA, r$ref))) stop(r)
    refs <- c(refs, list(r$ref))
  }
}

# Run the body `thunk()` with the threaded `refs` holding `vals`. Returns the result and
# the values the refs hold afterwards; the refs get their outer values back on exit.
ref_body <- function(refs, vals, thunk) {
  ctx <- new.env(parent = emptyenv())
  ctx$id <- next_id()
  ctx$watch <- TRUE
  ctx$refs <- refs
  old <- .jx$ref_ctx
  saved <- lapply(refs, ref_value)
  .jx$ref_ctx <- ctx
  on.exit({
    .jx$ref_ctx <- old
    for (i in seq_along(refs)) assign("value", saved[[i]], envir = refs[[i]])
  }, add = TRUE)
  for (i in seq_along(refs)) assign("value", vals[[i]], envir = refs[[i]])
  out <- thunk()
  list(out = out, finals = lapply(refs, ref_value))
}

# Port of api_util._check_no_aliased_closed_over_refs: a ref the function closes over
# must not also be one of its arguments.
ref_check_alias <- function(ref, arg_refs) {
  for (a in arg_refs) {
    if (identical(a, ref))
      jax_error("when tracing a function for jit, a mutable array reference of type Ref{",
                aval_long(ref$aval), "} was both closed over and passed as an argument",
                class = "jax_value_error")
  }
}

# A stand-in for an argument ref while a jit is traced, so that reads and writes through
# the argument are told apart from those through a closed over ref.
ref_proxy <- function(ref) {
  e <- list2env(as.list.environment(ref, all.names = TRUE), parent = emptyenv())
  class(e) <- class(ref)
  e
}

ref_value <- function(ref) get("value", envir = ref)

ref_read <- function(ref) {
  ref_check(ref, "reading a Ref")
  ref_capture(ref)
  ref_value(ref)
}

ref_write <- function(ref, value) {
  ref_check(ref, "writing a Ref")
  if (isTRUE(ref$frozen)) jax_error("Cannot write to a frozen Ref", class = "jax_value_error")
  ref_capture(ref)
  assign("value", value, envir = ref)
  invisible(ref)
}

# Convert a written value to the ref's dtype (weak values) or reject a mismatch.
ref_coerce <- function(ref, value, what = "swap") {
  v <- as_jax(value)
  dt <- ref$aval$dtype
  va <- aval_of(v)
  if (va$dtype != dt) {
    if (!va$weak) jax_error("Invalid dtype for `", what, "`. Ref dtype: ", dt, ". Value dtype: ", va$dtype, ".",
                            class = "jax_value_error")
    v <- lax_convert_element_type(v, dt)
  }
  v
}

ref_index_get <- function(cur, idx) {
  if (!length(idx)) return(cur)
  idx <- ref_static_ds(idx, aval_of(cur)$shape)
  if (inherits(idx, "jax_dynamic_ds")) {
    d <- idx
    return(lax_dynamic_slice_in_dim(cur, d$start, d$size, axis = d$axis))
  }
  do.call(`[`, c(list(cur), idx))
}

ref_index_update <- function(cur, idx, value, op) {
  if (!length(idx)) {
    new <- if (op == "set") value else lax_add(cur, value)
    return(lax_convert_element_type(jnp_broadcast_to(new, aval_of(cur)$shape), aval_of(cur)$dtype))
  }
  idx <- ref_static_ds(idx, aval_of(cur)$shape)
  if (inherits(idx, "jax_dynamic_ds")) {
    d <- idx
    if (op == "add") value <- lax_add(lax_dynamic_slice_in_dim(cur, d$start, d$size, axis = d$axis), value)
    sh <- aval_of(cur)$shape; sh[d$axis + 1L] <- d$size
    return(lax_dynamic_update_slice_in_dim(cur, lax_convert_element_type(jnp_broadcast_to(value, sh), aval_of(cur)$dtype),
                                           d$start, axis = d$axis))
  }
  f <- if (op == "set") at_set else at_add
  do.call(f, c(list(cur), idx, list(value = value)))
}

# ds() indices with static starts become slices; one dynamic ds becomes a dynamic slice.
ref_static_ds <- function(idx, shape) {
  is_ds <- vapply(idx, function(i) inherits(i, "jax_ds"), NA)
  if (!any(is_ds)) return(idx)
  dyn <- which(is_ds & vapply(idx, function(i) inherits(i, "jax_ds") && is_jax_value(i$start) &&
                                is_tracer(i$start), NA))
  if (length(dyn)) {
    if (length(idx) == 1L || (length(dyn) == 1L && all(vapply(idx[-dyn], function(i) inherits(i, "jax_slice") &&
                                                              is.null(i$start) && is.null(i$stop), NA)))) {
      d <- idx[[dyn]]
      if (d$stride != 1L) jax_error("ds with a traced start needs stride 1", class = "jax_not_implemented_error")
      return(structure(list(start = d$start, size = d$size, axis = dyn - 1L), class = "jax_dynamic_ds"))
    }
    jax_error("Ref indexing with a traced ds() start supports one ds index with full slices elsewhere",
              class = "jax_not_implemented_error")
  }
  lapply(idx, function(i) {
    if (!inherits(i, "jax_ds")) return(i)
    st <- as.integer(as.vector(concrete_or_error(as_jax(i$start), "ds start")))
    sl(st, st + i$size * i$stride, i$stride)
  })
}

#' @export
`[.jax_ref` <- function(x, ...) ref_index_get(ref_read(x), idx_collect(...))

#' @export
`[<-.jax_ref` <- function(x, ..., value) {
  idx <- idx_collect(...)
  cur <- ref_read(x)
  ref_write(x, ref_index_update(cur, idx, ref_coerce(x, value), "set"))
  x
}

#' @export
`$.jax_ref` <- function(x, name) {
  f <- function(n) get(n, envir = x)
  switch(name,
    value = ref_read(x),
    aval = f("aval"),
    shape = f("aval")$shape,
    dtype = f("aval")$dtype,
    ndim = length(f("aval")$shape),
    size = prod_int(f("aval")$shape),
    get = function(...) ref_index_get(ref_read(x), idx_collect(...)),
    set = function(value, ...) {
      ref_write(x, ref_index_update(ref_read(x), idx_collect(...), ref_coerce(x, value), "set"))
      invisible(NULL)
    },
    addupdate = function(value, ...) {
      ref_write(x, ref_index_update(ref_read(x), idx_collect(...), ref_coerce(x, value, "addupdate"), "add"))
      invisible(NULL)
    },
    swap = function(value, ...) {
      idx <- idx_collect(...)
      cur <- ref_read(x)
      old <- ref_index_get(cur, idx)
      ref_write(x, ref_index_update(cur, idx, ref_coerce(x, value), "set"))
      old
    },
    freeze = function() freeze(x),
    get(name, envir = x))
}

#' @export
`$<-.jax_ref` <- function(x, name, value) {
  assign(name, value, envir = x)
  x
}

#' @export
print.jax_ref <- function(x, ...) {
  if (isTRUE(get("freed", envir = x))) {
    cat("Ref(<freed>)\n")
  } else {
    cat("Ref(", aval_str(get("aval", envir = x)), ")\n", sep = "")
  }
  invisible(x)
}

#' @export
format.jax_ref <- function(x, ...) paste0("Ref(", aval_str(get("aval", envir = x)), ")")

# Arithmetic on a ref uses its value (as reading `ref[...]` would).
#' @export
Ops.jax_ref <- function(e1, e2) {
  v1 <- if (is_ref(e1)) ref_read(e1) else e1
  if (missing(e2)) return(get(.Generic)(v1))
  v2 <- if (is_ref(e2)) ref_read(e2) else e2
  get(.Generic)(v1, v2)
}

# Trace `fun` for jit with ref discharge. `call_refs` are the refs among the arguments,
# in flattened order. Returns the closed jaxpr, the captured refs (as argument positions
# or closed over objects) and the out tree.
ref_jit_trace <- function(fun_leaves, avals, call_refs) {
  trace <- new_jaxpr_trace(current_trace())
  ctx <- new.env(parent = emptyenv())
  ctx$id <- next_id()
  ctx$trace <- trace
  ctx$refs <- list(); ctx$saved <- list(); ctx$tracers <- list()
  ctx$originals <- call_refs
  ctx$proxies <- lapply(call_refs, ref_proxy)
  old <- .jx$ref_ctx
  .jx$ref_ctx <- ctx
  restore <- function() {
    for (i in seq_along(ctx$refs)) assign("value", ctx$saved[[i]], envir = ctx$refs[[i]])
    .jx$ref_ctx <- old
    trace$alive <- FALSE
  }
  on.exit(restore(), add = TRUE)
  in_tracers <- lapply(avals, function(a) new_tracer(trace, a))
  kr <- kr_begin()
  if (!is.null(kr)) on.exit(.jx$kr_log <- kr$old, add = TRUE)
  outs <- with_trace(trace, fun_leaves(in_tracers))
  krs <- if (!is.null(kr)) kr_end(kr, trace, in_tracers, outs)
  finals <- lapply(ctx$refs, function(r) as_jax(ref_value(r)))
  res <- to_jaxpr_extra(trace, c(in_tracers, ctx$tracers), c(outs, finals))
  where <- lapply(ctx$refs, function(r) {
    k <- which(vapply(ctx$proxies, identical, NA, r))
    if (length(k)) list(arg = k[1L]) else list(ref = r)
  })
  list(cj = closed_jaxpr(res$jaxpr, res$consts), where = where, n_out = length(outs), kr = krs)
}

# to_jaxpr with all inputs given as tracers (no dead code elimination of inputs).
to_jaxpr_extra <- function(trace, in_tracers, outs) to_jaxpr(trace, in_tracers, outs)
