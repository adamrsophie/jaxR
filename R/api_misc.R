# Top level jax API that makes sense on one CPU device (port of parts of
# jax/_src/api.py, jax/_src/xla_bridge.py, jax/_src/core.py,
# jax/_src/custom_derivatives.py, jax/_src/environment_info.py, jax/_src/config.py) and
# the jax.dtypes module (jax/_src/dtypes.py).

# ---- names and scopes ------------------------------------------------------------------------

#' Name a function or a scope of staged computations
#'
#' Ports of `jax.named_call` and `jax.named_scope`. JAX records these names in the
#' name stack of staged operations (they appear in profiler traces and HLO metadata);
#' jaxR keeps no name stack, so `named_call()` returns `fun` and `named_scope()`
#' evaluates `code`. Both are accepted so that JAX code ports unchanged.
#' @param fun A function.
#' @param name A name (`named_call()`: optional).
#' @param code Expression to evaluate.
#' @return `named_call()`: `fun`; `named_scope()`: the value of `code`.
#' @export
named_call <- function(fun, name = NULL) {
  if (!is.function(fun)) jax_error("named_call: fun must be a function", class = "jax_type_error")
  fun
}

#' @rdname named_call
#' @export
named_scope <- function(name, code) {
  if (!is.character(name) || length(name) != 1L)
    jax_error("named_scope name argument must be a string.", class = "jax_type_error")
  code
}

# ---- core helpers ------------------------------------------------------------------------------

#' Evaluate eagerly at trace time
#'
#' Port of `jax.ensure_compile_time_eval`: inside a staged function ([jit()],
#' [lax_scan()] bodies, ...), operations in `code` whose operands are all concrete are
#' evaluated now (constant folding) instead of being staged, so their results can
#' drive R control flow. Operations on traced values are staged as usual.
#' @param code Expression to evaluate.
#' @return The value of `code`.
#' @examples
#' \dontrun{
#' f <- jit(function(x) {
#'   z <- ensure_compile_time_eval(sin(sin(as_jax(3))) > 0)
#'   if (as.logical(z)) sin(x) else cos(x)
#' })
#' }
#' @export
ensure_compile_time_eval <- function(code) {
  # Port of config.eager_constant_folding: staging traces evaluate equations without
  # staged operands on their parent trace while `code` runs.
  touched <- list()
  t <- .jx$trace
  while (!is.null(t)) {
    if (identical(t$kind, "jaxpr") && !isTRUE(t$forward_known)) {
      t$forward_known <- TRUE
      touched[[length(touched) + 1L]] <- t
    }
    t <- t$parent
  }
  on.exit(for (tt in touched) tt$forward_known <- FALSE, add = TRUE)
  code
}

#' The abstract type of a value
#'
#' Port of `jax.typeof`: the `ShapedArray` (shape, dtype and weak type) of an array,
#' tracer or R value. R numbers are weakly typed (`~float32[]`).
#' @param x A value.
#' @return A `jax_aval`.
#' @export
jax_typeof <- function(x) {
  if (is_jax_value(x) || inherits(x, "jax_aval")) return(aval_of(x))
  if ((is.numeric(x) || is.logical(x) || is.complex(x)) && !is.object(x)) return(aval_of(as_jax(x)))
  jax_error("Argument '", paste(deparse(x), collapse = " "), "' of type ", class(x)[1L],
            " is not a valid JAX type", class = "jax_type_error")
}

#' Print information about the jaxR installation
#'
#' Port of `jax.print_environment_info`.
#' @param return_string Return the text instead of printing it.
#' @return The text, invisibly unless `return_string` is `TRUE`.
#' @export
print_environment_info <- function(return_string = FALSE) {
  ensure_backend()
  b <- .jx$backend
  info <- paste0(
    "jaxR:   ", utils::packageVersion("jaxR"), " (mirrors jax 0.12.0.1)\n",
    "R:      ", R.version.string, "\n",
    "PJRT:   ", b$plugin, " (C API ", b$api_major, ".", b$api_minor, ")\n",
    "device info: ", default_backend(), "-", b$num_devices, ", ", b$num_devices, " local devices\n",
    "process_count: 1\n",
    "x64:    ", x64_enabled(), "\n",
    "platform: ", paste(Sys.info()[c("sysname", "release", "machine")], collapse = " "), "\n")
  if (return_string) return(info)
  cat(info)
  invisible(info)
}

# ---- devices and processes (jax/_src/xla_bridge.py) -----------------------------------------

#' Backend, devices and processes
#'
#' Ports of `jax.default_backend`, `jax.device_count`, `jax.local_device_count`,
#' `jax.local_devices`, `jax.process_index`, `jax.process_count` and
#' `jax.process_indices` (and the deprecated aliases `jax.host_id`, `jax.host_count`,
#' `jax.host_ids`, which warn as in JAX). jaxR runs one process on the devices of one
#' PJRT client.
#' @param backend Ignored (one backend).
#' @param process_index Process index (only 0 exists).
#' @param host_id Deprecated alias of `process_index`.
#' @return The platform name, a count, an index, or device descriptions (as
#'   [devices()]).
#' @export
default_backend <- function() {
  ensure_backend()
  tolower(.jx$backend$platform)
}

#' @rdname default_backend
#' @export
device_count <- function(backend = NULL) {
  ensure_backend()
  .jx$backend$num_devices
}

#' @rdname default_backend
#' @export
local_device_count <- function(backend = NULL) device_count(backend)

#' @rdname default_backend
#' @export
local_devices <- function(process_index = NULL, backend = NULL, host_id = NULL) {
  p <- process_index %||% host_id %||% 0L
  if (p != 0L) jax_error("Process index ", p, " out of range (one process)", class = "jax_value_error")
  devices()
}

#' @rdname default_backend
#' @export
process_index <- function(backend = NULL) 0L

#' @rdname default_backend
#' @export
process_count <- function(backend = NULL) 1L

#' @rdname default_backend
#' @export
process_indices <- function(backend = NULL) 0L

#' @rdname default_backend
#' @export
host_id <- function(backend = NULL) {
  warning("jax.process_index has been renamed to jax.process_index. This alias will ",
          "eventually be removed; please update your code.", call. = FALSE)
  process_index(backend)
}

#' @rdname default_backend
#' @export
host_count <- function(backend = NULL) {
  warning("jax.process_count has been renamed to jax.process_count. This alias will ",
          "eventually be removed; please update your code.", call. = FALSE)
  process_count(backend)
}

#' @rdname default_backend
#' @export
host_ids <- function(backend = NULL) {
  warning("jax.process_indexs has been renamed to jax.process_indices. This alias will ",
          "eventually be removed; please update your code.", call. = FALSE)
  process_indices(backend)
}

#' Run code with a default device
#'
#' Port of the `jax.default_device` context manager. jaxR places arrays on its single
#' device, so this only checks that `device` is one of [devices()] (or `NULL`).
#' @param device A device description from [devices()], or `NULL`.
#' @param code Expression to evaluate.
#' @return The value of `code`.
#' @export
default_device <- function(device, code) {
  if (!is.null(device) && !(is.character(device) && length(device) == 1L && device %in% devices()))
    jax_error("default_device: unknown device ", format(device), "; devices are ",
              paste(devices(), collapse = ", "), class = "jax_value_error")
  code
}

#' Wait for pending side effects
#'
#' Port of `jax.effects_barrier`: services host callbacks that are still queued.
#' @return `NULL`, invisibly.
#' @export
effects_barrier <- function() {
  if (!is.null(.jx$backend)) .Call("jaxr_callbacks_pump", PACKAGE = "jaxR")
  invisible(NULL)
}

#' Copy arrays to the host
#'
#' Port of `jax.copy_to_host_async`. jaxR copies synchronously: the host copies of the
#' array leaves are fetched (and cached) now.
#' @param x A pytree.
#' @return `x`, invisibly.
#' @export
copy_to_host_async <- function(x) {
  for (l in tree_leaves(x)) if (is_jax_array(l)) array_host(l)
  invisible(x)
}

# ---- configuration context managers (jax/_src/config.py) -------------------------------------

#' Run code with a configuration option set
#'
#' Ports of the context managers `jax.numpy_rank_promotion`,
#' `jax.default_matmul_precision`, `jax.threefry_partitionable` and
#' `jax.default_prng_impl`: `code` is evaluated with the option set to `new_val`.
#' @param new_val The value of the option while `code` runs.
#' @param code Expression to evaluate.
#' @return The value of `code`.
#' @export
numpy_rank_promotion <- function(new_val, code) {
  if (!new_val %in% c("allow", "warn", "raise"))
    jax_error("jax_numpy_rank_promotion must be one of 'allow', 'warn', 'raise'",
              class = "jax_value_error")
  with_jax_config("jax_numpy_rank_promotion", new_val, code)
}

#' @rdname numpy_rank_promotion
#' @export
default_matmul_precision <- function(new_val, code) {
  with_jax_config("jax_default_matmul_precision", new_val, code)
}

#' @rdname numpy_rank_promotion
#' @export
threefry_partitionable <- function(new_val, code) {
  with_jax_config("jax_threefry_partitionable", isTRUE(new_val), code)
}

#' @rdname numpy_rank_promotion
#' @export
default_prng_impl <- function(new_val, code) {
  with_jax_config("jax_default_prng_impl", new_val, code)
}

# ---- differentiation helpers (jax/_src/custom_derivatives.py, api.py) ------------------------

#' Hoist the closed-over values of a function
#'
#' Port of `jax.closure_convert`: traces `fun` at the types of the example arguments
#' and hoists the closed-over values that may carry derivatives (traced values) out of
#' it. Returns `list(converted, consts)`; `converted(..., <consts>)` takes the original
#' arguments followed by the hoisted values. Useful for writing [custom_vjp()] rules of
#' functions that close over traced values.
#' @param fun A function of pytrees of arrays.
#' @param ... Example arguments (only shapes and dtypes are used).
#' @return `list(converted, consts)`.
#' @export
closure_convert <- function(fun, ...) {
  args <- list(...)
  fl <- tree_flatten(args)
  store <- new.env(parent = emptyenv())
  cj <- trace_to_jaxpr(flat_fun(fun, fl$treedef, store), lapply(fl$leaves, aval_of))
  out_tree <- store$out_tree
  consts <- cj$consts
  perturbed <- vapply(consts, am_maybe_perturbed, NA)
  jaxpr <- cj$jaxpr
  n_args <- length(fl$leaves)
  converted <- function(...) {
    all <- list(...)
    nc <- sum(perturbed)
    nargs <- length(all) - nc
    cargs <- all[seq_len(nargs)]
    hoisted <- all[nargs + seq_len(nc)]
    cs <- consts
    cs[perturbed] <- hoisted
    afl <- tree_flatten(cargs)
    if (!treedef_equal(afl$treedef, fl$treedef))
      jax_error("The inputs to the closure produced by closure_convert must have the same ",
                "Pytree structure as the example arguments passed when closure_convert was ",
                "called. Expected ", format(fl$treedef), ", but got ", format(afl$treedef),
                class = "jax_type_error")
    tree_unflatten(out_tree, eval_jaxpr(jaxpr, cs, lapply(afl$leaves, as_jax)))
  }
  list(converted, consts[perturbed])
}

# Port of custom_derivatives._maybe_perturbed: whether a value may carry a tangent.
am_maybe_perturbed <- function(x) {
  if (!is_tracer(x)) return(FALSE)
  if (x$trace$kind == "jvp") return(TRUE)
  if (x$trace$kind %in% c("jaxpr", "pe")) return(x$aval$dtype != "float0" && is_inexact_dtype(x$aval$dtype))
  if (x$trace$kind == "batch") return(am_maybe_perturbed(x$val))
  if (x$trace$kind == "jet") return(am_maybe_perturbed(x$primal))
  TRUE
}

#' Define a function with a custom gradient
#'
#' Port of `jax.custom_gradient`: `fun` returns `list(value, vjp)` where `vjp(g)` maps
#' the output cotangent to the input cotangents (a list with one entry per argument,
#' or the cotangent itself when `fun` has a single argument). Built on [custom_vjp()];
#' the values `vjp` closes over are saved as residuals.
#' @param fun A function returning `list(value, vjp)`.
#' @return A function returning `value`, differentiated with `vjp`.
#' @examples
#' \dontrun{
#' f <- custom_gradient(function(x) list(x^2, function(g) g * x))
#' grad(f)(3)  # 3, not 6
#' }
#' @export
custom_gradient <- function(fun) {
  force(fun)
  function(...) {
    # One custom_vjp per call: the traced rule (a jaxpr) is shared by fwd and bwd
    # through `store`, its closed-over values travel as residuals.
    store <- new.env(parent = emptyenv())
    nargs <- ...length()
    wrapped <- function(...) am_cg_result(fun(...))[[1L]]
    fwd <- function(...) {
      r <- am_cg_result(fun(...))
      ofl <- tree_flatten(r[[1L]])
      ct_avals <- lapply(ofl$leaves, function(l) tangent_aval(aval_of(l)))
      rule <- r[[2L]]
      cj <- trace_to_jaxpr(function(cts) {
        out <- rule(tree_unflatten(ofl$treedef, cts))
        cfl <- tree_flatten(out)
        store$in_tree <- cfl$treedef
        lapply(cfl$leaves, as_jax)
      }, ct_avals)
      store$jaxpr <- cj$jaxpr
      list(r[[1L]], cj$consts)
    }
    bwd <- function(res, ct) {
      cts <- lapply(tree_leaves(ct), as_jax)
      out <- tree_unflatten(store$in_tree, eval_jaxpr(store$jaxpr, res, cts))
      if (nargs == 1L && (treedef_is_leaf(store$in_tree) || !is.list(out) || length(out) != 1L))
        out <- list(out)
      out
    }
    custom_vjp(wrapped, fwd, bwd)(...)
  }
}

am_cg_result <- function(r) {
  if (!is.list(r) || !is.null(oldClass(r)) || length(r) != 2L || !is.function(r[[2L]]))
    jax_error("custom_gradient: fun must return list(value, vjp_function)", class = "jax_type_error")
  r
}

# Aux outputs are computed under the (finished) JVP trace of linearize: return their
# primal values.
am_unwrap_aux <- function(aux) {
  tree_map(function(l) {
    while (is_tracer(l) && identical(l$trace$kind, "jvp") && !l$trace$alive) l <- l$primal
    l
  }, aux)
}

# Residuals of fwd_and_bwd(): the residual arrays are pytree children, the linear
# jaxpr and the bookkeeping are static.
am_residuals <- function(consts, info) structure(list(consts = consts, info = info),
                                                 class = "jaxr_vjp_residuals")

register_pytree_node("jaxr_vjp_residuals",
                     function(r) list(children = r$consts, aux = r$info),
                     function(aux, children) am_residuals(children, aux))

#' Separate forward and backward functions of a VJP
#'
#' Port of `jax.fwd_and_bwd`. `fwd(...)` returns `list(primals_out, residuals)` (plus
#' `aux` with `has_aux`); `bwd(residuals, ct)` returns the cotangents of the arguments
#' in `argnums` (one value for a single argnum, a list otherwise). The residuals are a
#' pytree, so `bwd` can be jitted once and reused.
#' @param fun Function.
#' @param argnums 0-based positions of the differentiated arguments; negative values
#'   count from the end.
#' @param has_aux Whether `fun` returns `list(output, aux)`.
#' @param jitted Return jitted functions.
#' @return `list(fwd, bwd)`.
#' @export
fwd_and_bwd <- function(fun, argnums = 0L, has_aux = FALSE, jitted = TRUE) {
  check_callable(fun, "fwd_and_bwd")
  force(argnums)
  fwd <- function(...) {
    ap <- argnums_partial(fun, argnums, list(...))
    store <- new.env(parent = emptyenv())
    f <- if (has_aux) aux_fun(ap$f, store, "fwd_and_bwd") else ap$f
    fl <- tree_flatten(ap$dyn)
    lin <- linearize_flat(flat_fun(f, fl$treedef, store), lapply(fl$leaves, as_jax))
    out <- tree_unflatten(store$out_tree, lin$primals_out)
    info <- list(jaxpr = lin$jaxpr, diff_in = lin$diff_in, in_avals = lin$in_avals,
                 in_tree = fl$treedef, out_tree = store$out_tree,
                 out_avals = lapply(lin$primals_out, aval_of), single = ap$single)
    res <- am_residuals(lin$consts, info)
    if (has_aux) list(out, res, am_unwrap_aux(store$aux)) else list(out, res)
  }
  bwd <- function(residuals, ct) {
    info <- residuals$info
    cts <- match_leaves(info$out_tree, ct, function(got)
      jax_error("unexpected tree structure of argument to vjp function: got ", format(got),
                ", but expected to match ", format(info$out_tree), class = "jax_value_error"))
    check_ct_avals(cts, info$out_avals)
    cts <- Map(function(c, a) if (is_inexact_dtype(a$dtype)) c else zero_of(tangent_aval(a)),
               cts, info$out_avals)
    args <- lapply(info$jaxpr$invars, function(v) undefined_primal(v$aval))
    cts_lin <- backward_pass(info$jaxpr, residuals$consts, args, cts)
    k <- 0L
    g <- lapply(seq_along(info$diff_in), function(i) {
      if (!info$diff_in[i]) return(zeros_aval(tangent_aval(info$in_avals[[i]])))
      k <<- k + 1L
      instantiate_zero(cts_lin[[k]])
    })
    g <- tree_unflatten(info$in_tree, g)
    if (info$single) g[[1L]] else g
  }
  if (jitted) list(jit(fwd), jit(bwd)) else list(fwd, bwd)
}

#' The float0 dtype
#'
#' Port of `jax.float0` (and `jax.dtypes.float0`): the dtype of tangents of integer and
#' boolean values.
#' @export
float0 <- "float0"

# ---- jax.dtypes ---------------------------------------------------------------------------------

#' The jax.dtypes module
#'
#' Ports of `jax.dtypes`: `canonicalize_dtype` (64-bit types map to 32-bit ones
#' unless x64 is enabled), `issubdtype` (also for the extended categories
#' `dtypes_prng_key` and `dtypes_extended`, which typed PRNG key arrays belong to),
#' `result_type`, `itemsize_bits`, `scalar_type_of` (the R type of scalars of the
#' dtype: `"logical"`, `"integer"`, `"double"` or `"complex"`, standing for Python's
#' bool, int, float and complex), `finfo` and `iinfo`, the dtype names `float0` and
#' `bfloat16`, and `TypePromotionError` (a condition constructor; jaxR signals it with
#' class `jax_type_promotion_error`).
#' @param dtype A dtype name or a value.
#' @param allow_extended_dtype Accept extended dtypes (key types).
#' @param arg1,arg2 Dtypes, values or categories.
#' @param ... Values or dtypes.
#' @param return_weak_type_flag Also return whether the result is weakly typed.
#' @param x A value or dtype.
#' @param message Error message.
#' @return See the descriptions above.
#' @export
dtypes_canonicalize_dtype <- function(dtype, allow_extended_dtype = FALSE) {
  if (is.character(dtype) && grepl("^key<", dtype)) {
    if (!allow_extended_dtype)
      jax_error(dtype, " is not a valid dtype for canonicalize_dtype.", class = "jax_value_error")
    return(dtype)
  }
  canonicalize_dtype(dtype(dtype))
}

am_is_extended <- function(x) {
  inherits(x, "jax_prng_key") || (is.character(x) && length(x) == 1L && grepl("^key<", x))
}

#' @rdname dtypes_canonicalize_dtype
#' @export
dtypes_issubdtype <- function(arg1, arg2) {
  if (identical(arg2, "prng_key") || identical(arg2, "extended")) return(am_is_extended(arg1))
  if (am_is_extended(arg1)) return(FALSE)
  issubdtype(arg1, arg2)
}

#' @rdname dtypes_canonicalize_dtype
#' @export
dtypes_result_type <- function(..., return_weak_type_flag = FALSE) {
  avs <- lapply(list(...), function(a) {
    if (is.character(a)) list(dtype = dtype(a), weak = FALSE) else aval_of(a)
  })
  r <- promote_types(vapply(avs, `[[`, "", "dtype"), vapply(avs, `[[`, NA, "weak"))
  if (return_weak_type_flag) list(r$dtype, r$weak) else r$dtype
}

#' @rdname dtypes_canonicalize_dtype
#' @export
dtypes_itemsize_bits <- function(dtype) {
  dt <- if (is_jax_value(dtype)) aval_of(dtype)$dtype else dtype(dtype)
  if (dt == "float0") jax_error("unexpected input: dtype=float0", class = "jax_value_error")
  dtype_bits(dt)
}

#' @rdname dtypes_canonicalize_dtype
#' @export
dtypes_scalar_type_of <- function(x) {
  dt <- if (is.character(x)) dtype(x) else jax_typeof(x)$dtype
  switch(dtype_kind(dt), b = "logical", i = , u = "integer", f = "double", c = "complex",
         jax_error("Invalid scalar value ", format(x), class = "jax_type_error"))
}

#' @rdname dtypes_canonicalize_dtype
#' @export
dtypes_finfo <- function(dtype = "float") finfo(dtype)

#' @rdname dtypes_canonicalize_dtype
#' @export
dtypes_iinfo <- function(dtype = "int") iinfo(dtype)

#' @rdname dtypes_canonicalize_dtype
#' @export
dtypes_float0 <- "float0"

#' @rdname dtypes_canonicalize_dtype
#' @export
dtypes_bfloat16 <- "bfloat16"

#' @rdname dtypes_canonicalize_dtype
#' @export
dtypes_prng_key <- "prng_key"

#' @rdname dtypes_canonicalize_dtype
#' @export
dtypes_extended <- "extended"

#' @rdname dtypes_canonicalize_dtype
#' @export
dtypes_TypePromotionError <- function(message) {
  structure(class = c("jax_type_promotion_error", "jax_value_error", "jax_error", "error", "condition"),
            list(message = message, call = NULL))
}
