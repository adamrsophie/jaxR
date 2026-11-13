# Port of jax.stages: the stages a jit function goes through (trace, lower, compile), as
# objects that can be inspected and, for the last one, called.

#' Ahead of time tracing, lowering and compilation
#'
#' Port of `jax.stages`. A function made by [jit()] has the methods of JAX's `Wrapped`
#' protocol, reached with `$`:
#'
#' * `f$trace(...)` traces `f` for the given arguments and returns a *Traced* stage.
#' * `f$lower(...)` is `f$trace(...)$lower()`: a *Lowered* stage.
#' * `f$eval_shape(...)` returns the shapes and dtypes of the result ([eval_shape()]).
#' * `f$clear_cache()` drops the compiled versions of `f`.
#'
#' The arguments can be arrays or [ShapeDtypeStruct()] values. Each stage is read with
#' `$` as well:
#'
#' | Stage | Fields and methods |
#' |---|---|
#' | Traced | `jaxpr`, `fun_name`, `lower()`; a Traced stage is a function and can be called |
#' | Lowered | `as_text(dialect = NULL)`, `compiler_ir(dialect = NULL)`, `compile()`, `cost_analysis()` |
#' | Compiled | `as_text()`, `cost_analysis()`, `memory_analysis()`, `runtime_executable()`, `call(...)`; a Compiled stage is a function and can be called |
#' | all | `args_info`, `out_info`, `in_tree`, `out_tree`, `in_avals` |
#'
#' `as_text()` of a Lowered stage is the StableHLO module (the only dialect). The
#' analyses return `NULL`, as they do in JAX when the backend provides none. A Compiled
#' stage takes the non static arguments of `f`, accepts only the shapes and dtypes it
#' was compiled for, and cannot be transformed again.
#' @param x A function made by [jit()], or a stage.
#' @param name Field or method name.
#' @param ... Ignored.
#' @return The field, or a function for a method.
#' @examples
#' \dontrun{
#' f <- jit(function(x) jnp_sum(x^2))
#' lowered <- f$lower(as_jax(c(1, 2, 3)))
#' cat(lowered$as_text())
#' compiled <- lowered$compile()
#' compiled(as_jax(c(1, 2, 3)))
#' }
#' @name jax_stages
#' @export
`$.jax_jitted` <- function(x, name) {
  switch(name,
    trace = function(...) stg_trace(x, list(...)),
    lower = function(...) stg_lower(stg_trace(x, list(...))),
    eval_shape = function(...) eval_shape(attr(x, "fun"), ...),
    clear_cache = function() invisible(utils::clrhash(environment(x)$cache)),
    stg_no_attribute("jit function", name))
}

stg_no_attribute <- function(what, name) {
  jax_error("'", what, "' object has no attribute '", name, "'", class = "jax_attribute_error")
}

stg_is_aval <- function(x) inherits(x, "jax_aval")

# ponytail: functions that write to refs passed as arguments are not staged (the trace
# here does not thread ref state); add when someone needs AOT compilation of those.
stg_trace <- function(jf, args) {
  st <- attr(jf, "static")
  nms <- names(args) %||% rep("", length(args))
  static <- (seq_along(args) - 1L) %in% st$nums | (nzchar(nms) & nms %in% st$names)
  r <- do.call(make_jaxpr(attr(jf, "fun"), static_argnums = which(static) - 1L, return_shape = TRUE),
               args)
  fl <- tree_flatten(args[!static], is_leaf = stg_is_aval)
  in_avals <- lapply(fl$leaves, aval_of)
  stage <- list(jaxpr = r[[1L]], fun_name = "jit_fun", out_info = r[[2L]],
                args_info = tree_unflatten(fl$treedef, in_avals), in_tree = fl$treedef,
                out_tree = tree_structure(r[[2L]], is_leaf = stg_is_aval), in_avals = in_avals)
  traced <- function(...) jf(...)
  stage$lower <- function() stg_lower(traced)
  # `lower` looks `traced` up when it is called, so it sees the classed function.
  traced <- structure(traced, stage = stage, class = c("jax_stages_traced", "jax_stage", "function"))
  traced
}

stg_dialect <- function(dialect) {
  if (!is.null(dialect) && !identical(dialect, "stablehlo"))
    jax_error("Unknown dialect ", dialect, " (jaxR lowers to stablehlo only)", class = "jax_value_error")
}

stg_lower <- function(traced) {
  st <- attr(traced, "stage")
  text <- lower_jaxpr_module(st$jaxpr$jaxpr, st$fun_name)
  as_text <- function(dialect = NULL, debug_info = FALSE) { stg_dialect(dialect); text }
  structure(c(st[c("args_info", "out_info", "in_tree", "out_tree", "in_avals")], list(
    as_text = as_text, compiler_ir = function(dialect = NULL) as_text(dialect),
    cost_analysis = function() NULL,
    compile = function(compiler_options = NULL) stg_compile(st, text))),
    class = c("jax_stages_lowered", "jax_stage"))
}

stg_compile <- function(st, text) {
  entry <- new.env(parent = emptyenv())
  entry$exe <- compile_mlir(text)
  stg_compiled(entry, st$jaxpr$consts, jaxpr_out_avals(st$jaxpr$jaxpr),
               st[c("args_info", "out_info", "in_tree", "out_tree", "in_avals")])
}

# The compiled stage: runs the executable in `entry` on the constants and the arguments.
stg_compiled <- function(entry, consts, out_avals, meta) {
  compiled <- function(...) {
    fl <- tree_flatten(list(...))
    if (!identical(treedef_key(fl$treedef), treedef_key(meta$in_tree)))
      jax_error("function compiled for a different input pytree than the one it was called with",
                class = "jax_type_error")
    if (any(vapply(fl$leaves, is_tracer, NA)))
      jax_error("Cannot apply JAX transformations to a function lowered and compiled for a ",
                "particular signature. Detected argument of Tracer type.", class = "jax_type_error")
    leaves <- lapply(fl$leaves, as_jax)
    for (i in seq_along(leaves)) {
      a <- aval_of(leaves[[i]])
      e <- meta$in_avals[[i]]
      if (!identical(a$shape, e$shape) || a$dtype != e$dtype)
        jax_error("Argument types differ from the types for which this computation was compiled. ",
                  "Argument ", i - 1L, " compiled with ", aval_str(e), " and called with ", aval_str(a),
                  class = "jax_type_error")
    }
    bufs <- c(lapply(consts, function(c) array_buf(as_jax(c))), lapply(leaves, array_buf))
    outs <- Map(function(b, a) new_array(a, buf = b), execute_exe(entry$exe, bufs), out_avals)
    check_debug_values(outs, "jit_fun")
    tree_unflatten(meta$out_tree, outs)
  }
  stage <- c(meta, list(
    call = compiled, as_text = function() NULL, cost_analysis = function() NULL,
    memory_analysis = function() NULL, runtime_executable = function() entry$exe,
    consts = consts, out_avals = out_avals))
  structure(compiled, stage = stage, class = c("jax_stages_compiled", "jax_stage", "function"))
}

#' @rdname jax_stages
#' @export
`$.jax_stage` <- function(x, name) {
  fields <- if (is.function(x)) attr(x, "stage") else unclass(x)
  if (!name %in% names(fields)) stg_no_attribute(class(x)[1L], name)
  fields[[name]]
}

#' @rdname jax_stages
#' @export
print.jax_stage <- function(x, ...) {
  what <- c(jax_stages_traced = "Traced", jax_stages_lowered = "Lowered", jax_stages_compiled = "Compiled")
  ins <- vapply((if (is.function(x)) attr(x, "stage") else unclass(x))$in_avals, aval_str, "")
  cat("<", what[[class(x)[1L]]], " stage of a jit function; arguments: ",
      if (length(ins)) paste(ins, collapse = ", ") else "none", ">\n", sep = "")
  invisible(x)
}

#' @rdname jax_stages
#' @export
print.jax_jitted <- function(x, ...) {
  cat("<jit function>\n")
  print(attr(x, "fun"), ...)
  invisible(x)
}
