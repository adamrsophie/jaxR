# Gradient accumulators, backward pass logs and the markers of the `with_refs` method of
# VJP functions (ports of jax.ad and of the accumulators of jax._src.interpreters.ad).

#' Markers for the `with_refs` method of VJP functions
#'
#' Ports of `jax.ad.GradValue`, `jax.ad.DontWant`, `jax.ad.GradRef` and
#' `jax.ad.DidntWant`. The arguments of `vjp_fun$with_refs(...)` (see [vjp()]) have the
#' structure of the primal arguments; each leaf is a [new_ref()] (the gradient is added
#' to it), `ad_GradValue()` (the gradient is returned, the default) or `ad_DontWant()`
#' (the gradient is not returned). The VJP function then returns `ad_GradRef()` for
#' the leaves accumulated into a ref and `ad_DidntWant()` for the leaves not wanted.
#'
#' Unlike JAX, jaxR computes every gradient and adds it into the ref after the backward
#' pass, and custom rules receive the accumulator kind they declare whatever the caller
#' asks for (see [ad_ValAccum]).
#' @return A marker object.
#' @examples
#' \dontrun{
#' r <- vjp(function(x, y) x * y, 3, 4)
#' ref <- new_ref(as_jax(0, "float32"))
#' r[[2]]$with_refs(ad_DontWant(), ref)(1)
#' ref[]  # 3
#' }
#' @export
ad_GradValue <- function() vjp_sentinel("jax_GradValue")

#' @rdname ad_GradValue
#' @export
ad_DontWant <- function() vjp_sentinel("jax_DontWant")

#' @rdname ad_GradValue
#' @export
ad_GradRef <- function() vjp_sentinel("jax_GradRef")

#' @rdname ad_GradValue
#' @export
ad_DidntWant <- function() vjp_sentinel("jax_DidntWant")

#' Gradient accumulators
#'
#' The kinds of gradient accumulator a backward rule can receive (ports of `ValAccum`,
#' `RefAccum` and `NullAccum` of JAX's autodiff). A rule defined with
#' [custom_vjp_defvjp_with_accums()] or `custom_gradient(with_accums = )` gets one
#' accumulator per argument leaf and pushes cotangent contributions into it:
#'
#' * a `ValAccum` collects values: `acc$accum(x)` adds `x`;
#' * a `RefAccum` holds a ref, `acc$ref`, to add to in place (`acc$ref[i] <-
#'   acc$ref[i] + g`); `acc$accum(x)` adds to all of it;
#' * a `NullAccum` (for an argument that is not differentiated) ignores what is
#'   pushed.
#'
#' `ad_ValAccum` and `ad_RefAccum` are the kinds that `custom_gradient(with_accums = )`
#' declares; `acc$aval` is the type of the cotangent.
#'
#' In JAX the kind also depends on the caller: a rule receives a `RefAccum` for an
#' argument whose gradient the caller accumulates into a ref (with `with_refs`), a
#' `NullAccum` for one it does not want, and a `TypeError` is raised when this
#' contradicts the declared kind. In jaxR a rule always receives the kind it declares
#' (a `ValAccum` for [custom_vjp_defvjp_with_accums()]), or a `NullAccum` for an
#' argument that is not differentiated; a `RefAccum` holds a fresh ref whose value
#' becomes an ordinary cotangent, so no update is done in place on the caller's ref.
#' @format Marker objects.
#' @export
ad_ValAccum <- structure(list(kind = "ValAccum"), class = "jax_accum_kind")

#' @rdname ad_ValAccum
#' @export
ad_RefAccum <- structure(list(kind = "RefAccum"), class = "jax_accum_kind")

#' @rdname ad_ValAccum
#' @export
ad_NullAccum <- structure(list(kind = "NullAccum"), class = "jax_accum_kind")

#' @export
print.jax_accum_kind <- function(x, ...) {
  cat("<class '", x$kind, "'>\n", sep = "")
  invisible(x)
}

# Port of ct_check: the cotangent has the shape of the accumulator (dtypes may differ).
accum_check <- function(acc, x) {
  a <- if (is_symbolic_zero(x)) x$aval else aval_of(x)
  if (!identical(a$shape, acc$aval$shape))
    jax_error("Expected cotangent type ", aval_str(acc$aval), " but got ", aval_str(a),
              class = "jax_value_error")
}

new_accum <- function(kind, aval) {
  e <- new.env(parent = emptyenv())
  e$aval <- aval
  e$kind <- kind
  if (kind == "ValAccum") {
    e$val <- NULL
    e$accum <- function(x) {
      if (is.null(x) || is_symbolic_zero(x)) return(invisible(NULL))
      x <- as_jax(x)
      accum_check(e, x)
      e$val <- if (is.null(e$val)) x else add_tangents(e$val, x)
      invisible(NULL)
    }
  } else if (kind == "RefAccum") {
    e$ref <- new_ref(lax_full(aval$shape, 0, aval$dtype))
    e$accum <- function(x) {
      if (is.null(x) || is_symbolic_zero(x)) return(invisible(NULL))
      x <- as_jax(x)
      accum_check(e, x)
      e$ref[] <- e$ref[] + x
      invisible(NULL)
    }
  } else {
    e$accum <- function(x) invisible(NULL)
  }
  class(e) <- c(paste0("jax_", kind), "jax_GradAccum")
  e
}

# The cotangent an accumulator holds (NULL for none).
accum_freeze <- function(acc) {
  switch(acc$kind, ValAccum = acc$val, RefAccum = acc$ref[], NULL)
}

#' @export
print.jax_GradAccum <- function(x, ...) {
  cat(x$kind, "(", aval_str(x$aval), ")\n", sep = "")
  invisible(x)
}

# ---- backward pass logs -------------------------------------------------------------

# Each call of a VJP function collects the logs of the backward rules it runs. Keys are
# unique within a backward pass, even when the logs are dropped. A rule run under
# another trace (the body of a control flow primitive or of checkpoint, traced once
# per branch or more) cannot log out of it: the collector records that instead.
vjp_logs_begin <- function() {
  log <- new.env(parent = emptyenv())
  log$items <- list()
  log$trace <- current_trace()
  log$unsupported <- FALSE
  log$old <- .jx$vjp_logs
  .jx$vjp_logs <- log
  log
}

vjp_logs_end <- function(log) .jx$vjp_logs <- log$old

vjp_logs_check <- function(logs, what) {
  if (is.null(logs)) return(invisible())
  ok <- is.list(logs) && is.null(oldClass(logs)) &&
    (!length(logs) || (!is.null(names(logs)) && all(nzchar(names(logs)))))
  if (!ok)
    jax_error(what, " must be NULL or a named list of backward-pass log entries (a dict), ",
              "but got ", class(logs)[1L], ".", class = "jax_type_error")
  invisible()
}

vjp_logs_add <- function(logs) {
  if (!length(logs)) return(invisible())
  log <- .jx$vjp_logs
  if (is.null(log)) return(invisible())
  if (!identical(current_trace(), log$trace)) {
    log$unsupported <- TRUE
    return(invisible())
  }
  dup <- intersect(names(logs), names(log$items))
  if (length(dup))
    jax_error("Duplicate backward-pass log key '", dup[1L], "' from a custom backward rule",
              class = "jax_value_error")
  log$items[names(logs)] <- logs
  invisible()
}

# ---- with_refs ----------------------------------------------------------------------

# Port of _vjp_accum for the leaves given to with_refs: checks each and returns the
# kind ("value", "ref" or "none").
vjp_ref_kinds <- function(refs, in_tree, in_avals, arg_paths) {
  fl <- tryCatch(flatten_up_to(in_tree, refs), error = function(e) NULL)
  if (is.null(fl))
    jax_error("unexpected tree structure in the arguments to a VJP function's `with_refs` ",
              "method.\n\nThe arguments to `with_refs` must match the pytree structure of ",
              "the primal arguments of the differentiated function, with one entry for ",
              "each array argument. Each entry must be a `Ref` (to accumulate this ",
              "argument's gradient into the ref in-place), `ad_GradValue()` (to have this ",
              "argument's gradient returned as a value, the default behavior), or ",
              "`ad_DontWant()` (to skip computing this argument's gradient).\n\nBut the ",
              "tree structures differ: got ", format(tree_structure(refs, is_leaf = is_ref_leaf)),
              " for ", format(in_tree), class = "jax_value_error")
  Map(function(x, a, p) {
    if (inherits(x, "jax_GradValue")) return(list(kind = "value"))
    if (inherits(x, "jax_DontWant")) return(list(kind = "none"))
    if (inherits(x, "jax_ref")) {
      want <- tangent_aval(a)
      got <- aval_of(x[])
      if (!identical(got$shape, want$shape) || got$dtype != want$dtype)
        jax_error("unexpected JAX type (e.g. shape/dtype) for gradient ref passed to the ",
                  "VJP function's `with_refs` method for the argument at position args", p,
                  " of the differentiated function: the given ref has type Ref{",
                  aval_str(got), "}, but accumulating this argument's gradient requires a ",
                  "ref of type Ref{", aval_str(want), "}", class = "jax_value_error")
      return(list(kind = "ref", ref = x))
    }
    jax_error("each argument of the `with_refs` method of a VJP function must be a Ref, ",
              "ad_GradValue() or ad_DontWant(), got ", class(x)[1L], class = "jax_value_error")
  }, fl, in_avals, arg_paths)
}

is_ref_leaf <- function(x) inherits(x, "jax_ref") || inherits(x, "jax_vjp_sentinel")
