# Core: primitives, traces, tracers and bind (stackless design of jax/_src/core.py).

.prims <- new.env(parent = emptyenv())

new_primitive <- function(name, multiple_results = FALSE) {
  p <- new.env(parent = emptyenv())
  p$name <- name
  p$multiple_results <- multiple_results
  p$impl <- NULL
  p$abstract_eval <- NULL
  p$jvp <- NULL
  p$transpose <- NULL
  p$batch <- NULL
  p$partial_eval <- NULL
  p$override <- list()
  # Port of Primitive.bind: positional arguments are operands, named ones are params.
  p$bind <- function(...) {
    a <- list(...)
    named <- nzchar(names(a) %||% character(length(a)))
    bind(p, unname(a[!named]), a[named])
  }
  class(p) <- "jax_primitive"
  assign(name, p, envir = .prims)
  p
}

#' Primitive objects
#'
#' Every operation of jaxR is a primitive, as in JAX. Primitive objects print as
#' `<primitive name>` and expose `$name`, `$multiple_results` and `$bind(...)`, the
#' port of `Primitive.bind`: unnamed arguments are operands and named arguments are
#' parameters, so `lax_add_p$bind(1, 2)` equals `lax_add(1, 2)` and
#' `lax_reshape_p$bind(x, new_sizes = c(2L, 3L))` reshapes `x`. The exported objects
#' carry the JAX names with the module prefix: `jax.lax.add_p` is `lax_add_p`,
#' `jax.lax.linalg.qr_p` is `lax_linalg_qr_p`, `jax.custom_derivatives.custom_jvp_call_p`
#' is `custom_jvp_call_p`.
#' @param x A primitive.
#' @param ... Ignored.
#' @name jax_primitive
#' @examples
#' lax_add_p
#' \dontrun{
#' lax_add_p$bind(1, 2)
#' }
NULL

#' @rdname jax_primitive
#' @export
print.jax_primitive <- function(x, ...) {
  cat("<primitive ", x$name, ">\n", sep = "")
  invisible(x)
}

# ---- traces -----------------------------------------------------------------------

new_trace <- function(kind, parent, tag = NULL) {
  t <- new.env(parent = emptyenv())
  t$kind <- kind
  t$parent <- parent
  t$tag <- tag %||% next_id()
  t$alive <- TRUE
  class(t) <- c(paste0(kind, "_trace"), "jax_trace")
  t
}

eval_trace <- function() new_trace("eval", NULL)

# Whether constants are staged into a jaxpr, where a conversion of a constant is folded
# (a trace of the stack builds one; the partial evaluation traces of grad and vjp hand
# constants to their parent).
staging <- function() {
  t <- current_trace()
  while (!is.null(t)) {
    if (inherits(t, "jaxpr_trace") && !isTRUE(t$forward_known)) return(TRUE)
    t <- t$parent
  }
  FALSE
}

current_trace <- function() {
  if (is.null(.jx$trace)) .jx$trace <- eval_trace()
  .jx$trace
}

with_trace <- function(trace, expr) {
  old <- .jx$trace
  .jx$trace <- trace
  on.exit(.jx$trace <- old, add = TRUE)
  expr
}

#' @export
print.jax_trace <- function(x, ...) {
  cat("<", x$kind, " trace ", x$tag, if (!x$alive) " (dead)", ">\n", sep = "")
  invisible(x)
}

process_primitive <- function(trace, prim, args, params) {
  switch(trace$kind,
    eval = process_eval(trace, prim, args, params),
    jvp = process_jvp(trace, prim, args, params),
    batch = process_batch(trace, prim, args, params),
    jet = process_jet(trace, prim, args, params),
    jaxpr = , pe = process_jaxpr(trace, prim, args, params),
    jax_error("internal: unknown trace kind ", trace$kind))
}

process_eval <- function(trace, prim, args, params) {
  for (a in args) if (is_tracer(a)) escaped_tracer_error(a)
  ov <- prim$override$eval
  if (!is.null(ov)) return(ov(trace, args, params))
  if (!is.null(prim$impl)) return(prim$impl(args, params))
  apply_primitive(prim, args, params)
}

escaped_tracer_error <- function(x) {
  jax_error("Encountered an unexpected tracer (", format(x), "). A function transformed ",
            "by jaxR leaked an intermediate value from its transformation, for example ",
            "by assigning it to a global or enclosing variable.",
            class = c("jax_unexpected_tracer_error", "jax_escaped_tracer_error", "jax_type_error"))
}

# ---- tracers ----------------------------------------------------------------------

new_tracer <- function(trace, aval, ...) {
  t <- new.env(parent = emptyenv())
  t$trace <- trace
  t$aval <- aval
  t$id <- next_id()
  extra <- list(...)
  for (n in names(extra)) assign(n, extra[[n]], envir = t)
  class(t) <- c("jax_tracer", "jax_value")
  t
}

is_own <- function(x, trace) is_tracer(x) && x$trace$tag == trace$tag

#' @export
format.jax_tracer <- function(x, ...) {
  paste0("Traced<", aval_str(x$aval), ">with<", x$trace$kind, "_trace>")
}

#' @export
print.jax_tracer <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
length.jax_tracer <- function(x) prod_int(x$aval$shape)

#' @export
dim.jax_tracer <- function(x) x$aval$shape

#' @export
length.jax_array <- function(x) prod_int(x$aval$shape)

#' @export
dim.jax_array <- function(x) x$aval$shape

tracer_concrete_value <- function(x) {
  if (!is_tracer(x)) return(as_jax(x))
  if (!x$trace$alive) return(NULL)
  if (x$trace$kind == "jvp") return(tracer_concrete_value(x$primal))
  NULL
}

# ---- bind -------------------------------------------------------------------------

bind <- function(prim, args, params = list()) {
  for (i in seq_along(args)) {
    a <- args[[i]]
    if (is_tracer(a)) {
      if (!a$trace$alive) escaped_tracer_error(a)
    } else if (!is_jax_array(a)) {
      args[[i]] <- as_jax(a)
    }
  }
  process_primitive(current_trace(), prim, args, params)
}

bind1 <- function(prim, ..., params = list()) bind(prim, list(...), params)

# Abstract evaluation helper.
abstract_eval <- function(prim, avals, params) {
  if (is.null(prim$abstract_eval))
    jax_error("primitive '", prim$name, "' has no abstract evaluation rule")
  prim$abstract_eval(avals, params)
}

# ---- symbolic zeros ---------------------------------------------------------------

tangent_dtype <- function(dt) if (is_inexact_dtype(dt)) dt else "float0"

# Port of ShapedArray.to_tangent_aval: the tangent type keeps the weak type of the primal.
tangent_aval <- function(a) new_aval(a$shape, tangent_dtype(a$dtype), isTRUE(a$weak))

zero_of <- function(aval) structure(list(aval = aval), class = "jax_zero")
is_zero <- function(x) inherits(x, "jax_zero")

zeros_aval <- function(aval) {
  n <- prod_int(aval$shape)
  data <- switch(dtype_kind(aval$dtype), b = , V = logical(n), c = complex(n), double(n))
  new_array(aval, host = with_shape(data, aval$shape))
}

instantiate_zero <- function(x) if (is_zero(x)) zeros_aval(x$aval) else x

#' @export
print.jax_zero <- function(x, ...) {
  cat("Zero(", aval_str(x$aval), ")\n", sep = "")
  invisible(x)
}
