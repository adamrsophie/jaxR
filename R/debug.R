# Debugging callbacks (port of jax/_src/debugging.py and jax/_src/debugger/core.py):
# jax.debug.callback, jax.debug.print, jax.debug.log and jax.debug.breakpoint. They
# lower through the host callback machinery of R/callbacks.R.

#' Debug effects
#'
#' Ports of `jax.debug.DebugEffect` and `jax.debug.OrderedDebugEffect`: the effects of
#' debug callbacks, as names. Unordered debug callbacks carry `"Debug"`, ordered ones
#' `"OrderedDebug"`.
#' @export
debug_DebugEffect <- "Debug"

#' @rdname debug_DebugEffect
#' @export
debug_OrderedDebugEffect <- "OrderedDebug"

# ---- primitives ---------------------------------------------------------------------------

# Port of debugging.py:debug_batching_rule: unrolls the callback over the mapped axis.
dbg_batching_rule <- function(prim) {
  function(vals, dims, params, axis) {
    for (i in seq_len(axis$size) - 1L) {
      a <- Map(function(v, d) if (is.na(d)) v else lax_index_in_dim(v, i, d, keepdims = FALSE),
               vals, as.list(dims))
      bind(prim, a, params)
    }
    list(list(), integer(0))
  }
}

dbg_define <- function(prim, spec_of) {
  prim$effectful <- TRUE
  prim$abstract_eval <- function(avals, params) list()
  prim$impl <- function(args, params) {
    host <- lapply(args, function(a) array_host(concrete_or_error(a, "a debug callback")))
    cb_call(spec_of(params), host)
    list()
  }
  # Debug callbacks are transparent to differentiation (debug_callback_jvp_rule,
  # debug_callback_transpose_rule).
  prim$jvp <- function(primals, tangents, params) list(bind(prim, primals, params), list())
  prim$transpose <- function(ct, args, params) rep(list(NULL), length(args))
  prim$batch <- dbg_batching_rule(prim)
  prim$lowering <- function(ctx, ins, avals_in, avals_out, params) {
    cb_check_backend()
    cb_emit(ctx, spec_of(params), ins, avals_in, list(), has_side_effect = TRUE,
            ordered = isTRUE(params$ordered))
  }
  invisible(prim)
}

debug_callback_p <- new_primitive("debug_callback", multiple_results = TRUE)
dbg_define(debug_callback_p, function(params) params$callback)

debug_print_p <- new_primitive("debug_print", multiple_results = TRUE)
dbg_define(debug_print_p, function(params) {
  list(kind = "print", fmt = params$fmt, in_tree = params$in_tree, static = params$static_args,
       has_placeholders = params$has_placeholders, dtypes = params$dtypes,
       digits = params$digits, log = params$log)
})

# ---- argument handling ----------------------------------------------------------------------

# Port of debugging.py:_split_callback_args: array leaves are passed through the
# callback, all other leaves are kept as static values.
dbg_split_args <- function(args) {
  fl <- tree_flatten(args)
  dyn <- vapply(fl$leaves, function(x) {
    is_jax_value(x) || ((is.numeric(x) || is.logical(x) || is.complex(x)) && !is.object(x) &&
                          !anyNA(x[!is.nan(x)]))
  }, NA)
  list(in_tree = fl$treedef, dyn = lapply(fl$leaves[dyn], as_jax),
       static = list(idx = which(!dyn), vals = fl$leaves[!dyn]))
}

# Port of debugging.py:merge_callback_args (flat leaves only).
cb_merge_static <- function(static, dyn) {
  n <- length(static$idx) + length(dyn)
  out <- vector("list", n)
  out[static$idx] <- static$vals
  out[setdiff(seq_len(n), static$idx)] <- dyn
  out
}

# ---- debug.callback --------------------------------------------------------------------------

#' Call an R function for debugging, also inside compiled code
#'
#' Port of `jax.debug.callback`. `callback` is called with the arguments, array leaves
#' replaced by their values as R arrays, and its result is ignored. Inside [jit()] the
#' call happens when the compiled program runs; under [vmap()] it is called once per
#' batch element; it is transparent to differentiation (it sees primal values only).
#' Called without `callback`, it returns a function taking `(callback, ...)`, the
#' two-call form of JAX.
#' @param callback An R function.
#' @param ... Arguments passed to `callback` (pytrees; non-array leaves are passed as
#'   static values).
#' @param ordered Whether ordered debug callbacks keep their program order.
#' @param partitioned Ignored (single device).
#' @return `NULL`, invisibly (or a function in the two-call form).
#' @export
debug_callback <- function(callback = NULL, ..., ordered = FALSE, partitioned = FALSE) {
  run <- function(callback, ...) {
    if (!is.function(callback))
      jax_error("first argument to jax.debug.callback must be callable, but got an object of ",
                "type ", class(callback)[1L], class = "jax_type_error")
    sp <- dbg_split_args(list(...))
    bind(debug_callback_p, sp$dyn, list(
      callback = list(kind = "debug", fun = callback, in_tree = sp$in_tree, static = sp$static),
      effect = if (ordered) "OrderedDebug" else "Debug", ordered = isTRUE(ordered),
      partitioned = isTRUE(partitioned)))
    invisible(NULL)
  }
  if (!is.null(callback)) return(run(callback, ...))
  if (...length())
    jax_error("debug_callback received unexpected arguments in the two-call form",
              class = "jax_type_error")
  run
}

# ---- debug.print ----------------------------------------------------------------------------

# Parse a Python format string into literal pieces and replacement fields
# (string.Formatter.parse, without nested fields in format specs).
dbg_parse_fmt <- function(fmt) {
  ch <- strsplit(fmt, "", fixed = TRUE)[[1L]]
  n <- length(ch)
  pieces <- list()
  lit <- character(0)
  push_lit <- function() {
    if (length(lit)) pieces[[length(pieces) + 1L]] <<- paste(lit, collapse = "")
    lit <<- character(0)
  }
  i <- 1L
  while (i <= n) {
    c <- ch[i]
    nxt <- if (i < n) ch[i + 1L] else ""
    if (c == "{" && nxt == "{") {
      lit <- c(lit, "{")
      i <- i + 2L
    } else if (c == "}" && nxt == "}") {
      lit <- c(lit, "}")
      i <- i + 2L
    } else if (c == "{") {
      j <- i + 1L
      depth <- 1L
      while (j <= n) {
        if (ch[j] == "{") depth <- depth + 1L
        if (ch[j] == "}") depth <- depth - 1L
        if (depth == 0L) break
        j <- j + 1L
      }
      if (j > n) jax_error("expected '}' before end of string", class = "jax_value_error")
      body <- if (j > i + 1L) paste(ch[(i + 1L):(j - 1L)], collapse = "") else ""
      m <- regmatches(body, regexec("^([^!:]*)(![rsa])?(:(.*))?$", body))[[1L]]
      if (!length(m)) jax_error("invalid format field {", body, "}", class = "jax_value_error")
      push_lit()
      pieces[[length(pieces) + 1L]] <- list(name = m[2L], conv = sub("!", "", m[3L]),
                                            spec = m[5L])
      i <- j + 1L
    } else if (c == "}") {
      jax_error("Single '}' encountered in format string", class = "jax_value_error")
    } else {
      lit <- c(lit, c)
      i <- i + 1L
    }
  }
  push_lit()
  pieces
}

# string.Formatter.vformat with check_unused_args of debugging._DebugPrintFormatChecker.
# `render(value, field)` formats one value.
dbg_vformat <- function(fmt, args, kwargs, render, check = TRUE) {
  pieces <- dbg_parse_fmt(fmt)
  auto <- NA
  next_idx <- 0L
  used_pos <- integer(0)
  used_kw <- character(0)
  out <- character(length(pieces))
  for (k in seq_along(pieces)) {
    p <- pieces[[k]]
    if (is.character(p)) {
      out[k] <- p
      next
    }
    name <- p$name
    if (grepl("[.[]", name))
      jax_error("debug_print supports plain field names only, got {", name, "}",
                class = "jax_value_error")
    if (name == "") {
      if (isFALSE(auto))
        jax_error("cannot switch from manual field specification to automatic field numbering",
                  class = "jax_value_error")
      auto <- TRUE
      name <- as.character(next_idx)
      next_idx <- next_idx + 1L
    } else if (grepl("^[0-9]+$", name)) {
      if (isTRUE(auto))
        jax_error("cannot switch from automatic field numbering to manual field specification",
                  class = "jax_value_error")
      auto <- FALSE
    }
    if (grepl("^[0-9]+$", name)) {
      idx <- as.integer(name)
      if (idx >= length(args))
        jax_error("Replacement index ", idx, " out of range for positional args tuple",
                  class = "jax_index_error")
      used_pos <- c(used_pos, idx)
      out[k] <- render(args[[idx + 1L]], p)
    } else {
      if (!name %in% names(kwargs)) jax_error("KeyError: '", name, "'", class = "jax_key_error")
      used_kw <- c(used_kw, name)
      out[k] <- render(kwargs[[name]], p)
    }
  }
  if (check) {
    unused <- setdiff(seq_along(args) - 1L, used_pos)
    if (length(unused))
      jax_error("Unused positional arguments to `jax.debug.print`: [",
                paste(vapply(args[unused + 1L], dbg_repr, ""), collapse = ", "), "]",
                class = "jax_value_error")
    unused_kw <- setdiff(names(kwargs), used_kw)
    if (length(unused_kw))
      jax_error("Unused keyword arguments to `jax.debug.print`: [",
                paste0("'", unused_kw, "'", collapse = ", "), "]. You may be passing an ",
                "f-string (i.e, `f\"{x}\"`) into `jax.debug.print` and instead should pass ",
                "in a regular string.", class = "jax_value_error")
  }
  paste(out, collapse = "")
}

dbg_repr <- function(x) {
  if (is_jax_value(x)) return(format(x))
  paste(deparse(x), collapse = " ")
}

# Port of the has_placeholders test of debug_print: the first parsed item is a field.
dbg_has_placeholders <- function(fmt) {
  m <- regexpr("\\{\\{|\\}\\}|\\{", fmt)
  m > 0L && regmatches(fmt, m) == "{"
}

dbg_split_named <- function(lst) {
  nms <- names(lst) %||% rep("", length(lst))
  list(args = unname(lst[!nzchar(nms)]), kwargs = lst[nzchar(nms)])
}

# A value of debug_print: host data with its dtype.
dbg_value <- function(data, dtype) structure(list(data = data, dtype = dtype), class = "jaxr_dbg_value")

# Format one value with R's formatting of arrays (digits as print() of jax arrays).
dbg_format_value <- function(x, spec = "", digits = getOption("digits")) {
  if (inherits(x, "jaxr_dbg_value")) {
    d <- x$data
    dt <- x$dtype
    if (is_int_dtype(dt) && all(abs(d) < 2^31)) storage.mode(d) <- "integer"
    dig <- switch(dt, float16 = , bfloat16 = 4L, float32 = , complex64 = min(7L, digits), digits)
    if (nzchar(spec)) return(dbg_format_spec(d, spec))
    shape <- r_shape(d)
    if (!length(dim(d))) return(paste(format(d, digits = dig), collapse = " "))
    out <- if (length(shape) >= 3L) utils::capture.output(print_c_order(d, shape, dig))
           else utils::capture.output(print(d, digits = dig))
    return(paste(out, collapse = "\n"))
  }
  if (is.list(x) && is.null(oldClass(x))) {
    nms <- names(x) %||% rep("", length(x))
    parts <- vapply(seq_along(x), function(i) {
      v <- dbg_format_value(x[[i]], "", digits)
      if (nzchar(nms[i])) paste0(nms[i], " = ", v) else v
    }, "")
    return(paste0("list(", paste(parts, collapse = ", "), ")"))
  }
  if (nzchar(spec) && (is.numeric(x) || is.character(x))) return(dbg_format_spec(x, spec))
  if (is.character(x) && length(x) == 1L) return(x)
  paste(utils::capture.output(print(x)), collapse = "\n")
}

# A Python format spec on numbers or strings, through sprintf().
dbg_format_spec <- function(d, spec) {
  if (!grepl("^[-+ #0]*[0-9]*(\\.[0-9]+)?[diouxXeEfFgGs]?$", spec))
    jax_error("unsupported format spec '", spec, "' in debug_print", class = "jax_value_error")
  if (!grepl("[a-zA-Z]$", spec)) spec <- paste0(spec, if (is.character(d)) "s" else "g")
  if (grepl("[dioxX]$", spec)) d <- round(d)
  s <- sprintf(paste0("%", sub("F$", "f", spec)), as.vector(d))
  if (length(s) == 1L) s else paste0("[", paste(s, collapse = " "), "]")
}

# Port of debugging.py:_format_print_callback.
dbg_format_print <- function(spec, dyn) {
  vals <- Map(function(d, dt) dbg_value(d, dt), dyn, spec$dtypes)
  tree <- tree_unflatten(spec$in_tree, cb_merge_static(spec$static, vals))
  sp <- dbg_split_named(tree)
  render <- function(v, field) dbg_format_value(v, field$spec, spec$digits)
  msg <- if (spec$has_placeholders) {
    dbg_vformat(spec$fmt, sp$args, sp$kwargs, render, check = FALSE)
  } else {
    paste(c(spec$fmt, vapply(sp$args, dbg_format_value, "", digits = spec$digits)), collapse = " ")
  }
  if (isTRUE(spec$log)) message(msg) else cat(msg, "\n", sep = "")
  invisible(NULL)
}

dbg_print_impl <- function(fmt, args, ordered, partitioned, skip_format_check, log) {
  if (!is.character(fmt) || length(fmt) != 1L)
    jax_error("debug_print: fmt must be a character string", class = "jax_type_error")
  if (!skip_format_check) {
    sp <- dbg_split_named(args)
    dbg_vformat(fmt, sp$args, sp$kwargs, function(v, field) "", check = TRUE)
  }
  has_placeholders <- nzchar(fmt) && dbg_has_placeholders(fmt)
  if (!has_placeholders && length(dbg_split_named(args)$kwargs))
    jax_error("Format without placeholders should not have kwargs.", class = "jax_value_error")
  sa <- dbg_split_args(args)
  bind(debug_print_p, sa$dyn, list(
    fmt = fmt, ordered = isTRUE(ordered), partitioned = isTRUE(partitioned),
    in_tree = sa$in_tree, static_args = sa$static, has_placeholders = has_placeholders,
    dtypes = vapply(sa$dyn, function(a) aval_of(a)$dtype, ""), digits = getOption("digits"),
    log = isTRUE(log)))
  invisible(NULL)
}

#' Print values, also inside compiled code
#'
#' Port of `jax.debug.print`. `fmt` is a Python format string (`"x = {x}"`,
#' `"{} and {}"`, `"{0}"`, with optional specs such as `"{x:.3f}"`); its fields are
#' filled from the unnamed (positional) and named arguments. Arrays are formatted with
#' R's formatting (as `print()` shows them); scalars as `format()` shows them. Inside
#' [jit()] the values are printed when the compiled program runs; under [vmap()] once
#' per batch element. Without placeholders the arguments are appended to `fmt`,
#' separated by spaces. `debug_log()` writes the message with [message()] instead.
#' Called without `fmt`, they return a function taking `(fmt, ...)` (two-call form).
#' @param fmt Format string.
#' @param ... Values to format (unnamed: positional fields; named: named fields).
#' @param ordered Whether ordered debug prints keep their program order.
#' @param partitioned Ignored (single device).
#' @param skip_format_check Skip the check that every argument is used by `fmt`.
#' @return `NULL`, invisibly (or a function in the two-call form).
#' @examples
#' \dontrun{
#' f <- jit(function(x) { debug_print("x = {x}, sum = {s:.2f}", x = x, s = jnp_sum(x)); x * 2 })
#' f(as_jax(c(1, 2, 3)))
#' }
#' @export
debug_print <- function(fmt = NULL, ..., ordered = FALSE, partitioned = FALSE,
                        skip_format_check = FALSE) {
  if (!is.null(fmt))
    return(dbg_print_impl(fmt, list(...), ordered, partitioned, skip_format_check, FALSE))
  if (...length())
    jax_error("debug_print received unexpected arguments in the two-call form",
              class = "jax_type_error")
  function(fmt, ...) dbg_print_impl(fmt, list(...), ordered, partitioned, skip_format_check, FALSE)
}

#' @rdname debug_print
#' @export
debug_log <- function(fmt = NULL, ..., ordered = FALSE, partitioned = FALSE,
                      skip_format_check = FALSE) {
  if (!is.null(fmt))
    return(dbg_print_impl(fmt, list(...), ordered, partitioned, skip_format_check, TRUE))
  if (...length())
    jax_error("debug_print received unexpected arguments in the two-call form",
              class = "jax_type_error")
  function(fmt, ...) dbg_print_impl(fmt, list(...), ordered, partitioned, skip_format_check, TRUE)
}

# ---- debug.breakpoint -------------------------------------------------------------------------

#' Enter the R browser at a point in a (possibly compiled) program
#'
#' Port of `jax.debug.breakpoint`. The variables of the calling frames are captured;
#' traced values among them are passed through a debug callback, so inside [jit()] the
#' breakpoint triggers when the compiled program runs, with their values (as R arrays).
#' The default debugger evaluates [browser()] in an environment holding the variables
#' of the innermost frame (with the other frames as `.frames`).
#' @param backend `NULL` (the R browser) or a function called with the list of frame
#'   environments (innermost first), a jaxR extension standing for JAX's debugger
#'   registry.
#' @param filter_frames Drop frames of jaxR itself.
#' @param num_frames Number of frames to capture (all if `NULL`).
#' @param ordered Whether ordered breakpoints keep their program order.
#' @param token Optional pytree of arrays: the breakpoint runs once its value is
#'   computed and it is returned unchanged (thread it into later computation).
#' @param ... Ignored.
#' @return `token` (invisibly `NULL` when no token is given).
#' @export
debug_breakpoint <- function(backend = NULL, filter_frames = TRUE, num_frames = NULL,
                             ordered = FALSE, token = NULL, ...) {
  if (!is.null(token) && isTRUE(ordered))
    jax_error("`ordered` and `token` are mutually exclusive arguments.", class = "jax_value_error")
  if (!is.null(backend) && !is.function(backend) && !identical(backend, "cli"))
    jax_error("debug_breakpoint: backend must be NULL, \"cli\" or a function", class = "jax_value_error")
  envs <- rev(sys.frames()[seq_len(sys.nframe() - 1L)])
  ns <- asNamespace("jaxR")
  if (filter_frames) envs <- Filter(function(e) !identical(topenv(e), ns), envs)
  if (!is.null(num_frames)) envs <- envs[seq_len(min(as.integer(num_frames), length(envs)))]
  frames <- lapply(envs, function(e) {
    nms <- setdiff(ls(e, all.names = TRUE), "...")
    vals <- lapply(nms, function(nm) tryCatch(get(nm, envir = e, inherits = FALSE),
                                              error = function(err) NULL))
    names(vals) <- nms
    vals[!vapply(vals, is.null, NA)]
  })
  fl <- tree_flatten(frames)
  dyn <- vapply(fl$leaves, is_tracer, NA)
  static <- list(idx = which(!dyn), vals = fl$leaves[!dyn])
  debugger <- function(...) {
    fr <- tree_unflatten(fl$treedef, cb_merge_static(static, list(...)))
    fr <- lapply(fr, function(v) {
      lapply(v, function(l) tree_map(function(z) if (is_jax_array(z)) array_host(z) else z, l))
    })
    envs <- lapply(fr, list2env)
    if (is.function(backend)) return(backend(envs))
    e <- if (length(envs)) envs[[1L]] else new.env()
    assign(".frames", envs, envir = e)
    # Open R's interactive debugger with `e` as its evaluation frame.
    if (interactive()) eval(quote(browser()), e)
    invisible(NULL)
  }
  if (is.null(token)) {
    do.call(debug_callback, c(list(debugger), fl$leaves[dyn], list(ordered = ordered)))
    return(invisible(NULL))
  }
  tl <- lax_stop_gradient(list(token, fl$leaves[dyn]))
  wrapper <- function(x, ...) {
    debugger(...)
    x
  }
  do.call(pure_callback, c(list(wrapper, tl[[1L]], tl[[1L]]), tl[[2L]]))
}
