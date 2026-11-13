# Eager dispatch and jit (port of jax/_src/dispatch.py and the jit path of pjit.py).

# Run a closed jaxpr on concrete arguments through XLA. `entry` (an environment) caches
# the compiled executable across calls.
execute_jaxpr <- function(jaxpr, consts, args, name = "jit_fun", entry = NULL) {
  exe <- if (!is.null(entry)) entry$exe else NULL
  if (is.null(exe)) {
    exe <- compile_mlir(mon_elapsed("/jax/core/compile/jaxpr_to_mlir_module_duration", name,
                                    lower_jaxpr_module(jaxpr, name)))
    if (!is.null(entry)) entry$exe <- exe
  }
  bufs <- c(lapply(consts, function(c) array_buf(as_jax(c))),
            lapply(args, function(a) array_buf(as_jax(a))))
  outs <- execute_exe(exe, bufs)
  res <- Map(function(b, v) new_array(v$aval, buf = b), outs, jaxpr$outvars)
  check_debug_values(res, name)
  res
}

# Cache of eager executables keyed by (primitive, input avals, params), compared with
# identical() so closures and jaxprs in params are matched by identity.
eager_cache <- function() {
  if (is.null(.jx$eager_cache)) .jx$eager_cache <- utils::hashtab("identical")
  .jx$eager_cache
}

lower_primitive_module <- function(prim, avals, out_avals, params) {
  ctx <- new_lowering_ctx()
  arg_names <- sprintf("%%arg%d", seq_along(avals) - 1L)
  outs <- prim$lowering(ctx, arg_names, avals, out_avals, params)
  in_types <- vapply(avals, tensor_type, "")
  out_types <- vapply(out_avals, tensor_type, "")
  emit_line(ctx, paste0("\"func.return\"(", paste(outs, collapse = ", "), ") : (",
                        paste(out_types, collapse = ", "), ") -> ()"))
  ftype <- paste0("(", paste(in_types, collapse = ", "), ") -> ",
                  if (length(out_types) == 1L) out_types else
                    paste0("(", paste(out_types, collapse = ", "), ")"))
  paste0(
    "\"builtin.module\"() <{sym_name = \"jit_", prim$name, "\"}> ({\n",
    "  \"func.func\"() <{function_type = ", ftype,
    ", sym_name = \"main\", sym_visibility = \"public\"}> ({\n",
    if (length(arg_names)) paste0("  ^bb0(", paste(paste0(arg_names, ": ", in_types), collapse = ", "), "):\n"),
    ctx_text(ctx), "\n  }) : () -> ()\n",
    "}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()\n")
}

# Eager evaluation of one primitive: a cached single-equation jit (dispatch.apply_primitive).
apply_primitive <- function(prim, args, params) {
  avals <- lapply(args, aval_of)
  key <- list(prim$name, vapply(avals, aval_key, ""), params, x64_enabled())
  hit <- utils::gethash(eager_cache(), key)
  if (is.null(hit)) {
    out_avals <- abstract_eval(prim, avals, params)
    if (!prim$multiple_results) out_avals <- list(out_avals)
    if (is.null(prim$lowering))
      jax_error("MLIR translation rule for primitive '", prim$name, "' not found")
    exe <- compile_mlir(lower_primitive_module(prim, avals, out_avals, params))
    hit <- list(exe = exe, out_avals = out_avals)
    utils::sethash(eager_cache(), key, hit)
  }
  res <- execute_exe(hit$exe, lapply(args, array_buf))
  res <- Map(function(b, a) new_array(a, buf = b), res, hit$out_avals)
  check_debug_values(res, prim$name)
  if (prim$multiple_results) res else res[[1L]]
}

# ---- jit --------------------------------------------------------------------------------

aval_key <- function(a) paste0(aval_str(a), if (a$weak) "w")

# Options that change what tracing produces; they are part of the jit cache key (the
# options JAX marks include_in_jit_key).
trace_config_names <- c(
  "jax_enable_x64", "jax_numpy_dtype_promotion", "jax_numpy_rank_promotion",
  "jax_default_matmul_precision", "jax_threefry_partitionable", "jax_default_prng_impl",
  "jax_random_seed_offset", "jax_use_high_dynamic_range_gumbel", "jax_softmax_custom_jvp",
  "jax_accurate_erf_inv", "jax_enable_checks", "jax_check_static_indices", "jax_debug_key_reuse",
  "jax_legacy_prng_key", "jax_enable_custom_prng", "jax_allow_f16_reductions")

trace_config_key <- function() .jx$config[trace_config_names]

# Port of api_util.resolve_argnums: complete static_argnums from static_argnames or
# the reverse (when only one is given) using the parameters of `fun`, and validate.
resolve_static_args <- function(fun, static_argnums, static_argnames) {
  infer_nums <- is.null(static_argnums) && !is.null(static_argnames)
  infer_names <- is.null(static_argnames) && !is.null(static_argnums)
  nums <- as.integer(static_argnums)
  nms <- as.character(static_argnames)
  fn <- tryCatch(names(formals(args(fun))), error = function(e) NULL)
  if (is.null(fn)) return(list(nums = nums, names = nms, formals = NULL))
  np <- length(fn)
  idx <- seq_along(fn) - 1L
  if (any(nums < -np | nums >= np) && !("..." %in% fn))
    jax_error("jitted function has static_argnums=(", paste(nums, collapse = ", "),
              "), but only accepts ", np, " positional arguments.", class = "jax_value_error")
  bad <- setdiff(nms, fn)
  if (length(bad) && !("..." %in% fn))
    jax_error("jitted function has static_argnames=(", paste(nms, collapse = ", "),
              "), but only accepts (", paste(fn, collapse = ", "), ") as parameters.",
              class = "jax_value_error")
  if (infer_names) {
    nms <- fn[(idx %in% nums | (idx - np) %in% nums) & fn != "..."]
  } else if (infer_nums) {
    nums <- idx[fn %in% nms & fn != "..."]
  }
  if (!("..." %in% fn) && np > 0L) nums <- ifelse(nums < 0L, nums %% np, nums)
  list(nums = as.integer(nums), names = nms, formals = fn)
}

# The parameter of `fun` each supplied argument binds to ("" for an argument taken by
# `...` or when matching fails; the call itself reports such errors).
match_formals <- function(fn, args) {
  nms <- names(args) %||% rep("", length(args))
  if (is.null(fn) || !length(args)) return(nms)
  def <- function() NULL
  formals(def) <- stats::setNames(rep(list(quote(expr = )), length(fn)), fn)
  cl <- as.call(c(list(as.name("def")), stats::setNames(as.list(seq_along(args)), nms)))
  m <- tryCatch(as.list(match.call(def, cl))[-1L], error = function(e) NULL)
  if (is.null(m)) return(nms)
  out <- character(length(args))
  mn <- names(m) %||% rep("", length(m))
  for (i in seq_along(m)) if (mn[i] %in% fn && is.numeric(m[[i]])) out[m[[i]]] <- mn[i]
  out
}

#' Just-in-time compilation
#'
#' Port of `jax.jit`. The function is traced once per combination of argument shapes,
#' dtypes, weak types, pytree structure and static arguments, lowered to StableHLO and
#' compiled by XLA. Under other transformations (grad, vmap, ...) the cached jaxpr is
#' evaluated under the active transformation.
#' @param fun Function of arrays (or pytrees of arrays).
#' @param static_argnums 0-based positions of arguments treated as compile time
#'   constants (negative positions count from the end). Static values are compared with
#'   [identical()], so two closures are the same static argument only when they are the
#'   same function with the same environment.
#' @param static_argnames Names of arguments treated as static. As in JAX, when only one
#'   of `static_argnums` and `static_argnames` is given the other is inferred from the
#'   parameters of `fun`, so a static parameter is static however it is passed.
#' @param donate_argnums Accepted for compatibility; ignored.
#' @param inline Ignored.
#' @return The compiled function.
#' @export
jit <- function(fun, static_argnums = NULL, static_argnames = NULL,
                donate_argnums = integer(0), inline = FALSE) {
  force(fun)
  if (!is.function(fun)) jax_error("Expected a callable value, got ", class(fun)[1L], class = "jax_type_error")
  static <- resolve_static_args(fun, static_argnums, static_argnames)
  # Keys are compared with identical(), which keeps closures passed as static
  # arguments apart by their environments (and keeps them alive while cached).
  cache <- utils::hashtab("identical")
  cache_keys <- 0L
  f <- function(...) {
    if (isTRUE(.jx$config$jax_disable_jit)) return(fun(...))
    args <- list(...)
    nms <- names(args) %||% rep("", length(args))
    bound <- match_formals(static$formals, args)
    pos <- ifelse(nzchar(bound), match(bound, static$formals), seq_along(args)) - 1L
    # As in JAX, a named argument is static by its name and a positional one by its
    # position.
    # A negative position of a function with `...` counts from the end of the call.
    snums <- ifelse(static$nums < 0L, static$nums + length(args), static$nums)
    is_static <- ifelse(nzchar(nms), bound %in% static$names | nms %in% static$names,
                        pos %in% snums)
    dyn <- args[!is_static]
    fl <- tree_flatten(dyn, is_leaf = is_ref)
    ref_pos <- which(vapply(fl$leaves, is_ref, NA))
    call_refs <- fl$leaves[ref_pos]
    for (i in seq_along(call_refs)) for (j in seq_len(i - 1L)) {
      if (identical(call_refs[[i]], call_refs[[j]]))
        jax_error("only one reference to a mutable array may be passed as an argument to a ",
                  "function, but the same Ref appeared at both flat index ", ref_pos[j] - 1L,
                  " and flat index ", ref_pos[i] - 1L, ".", class = "jax_value_error")
    }
    leaves <- lapply(fl$leaves, function(l) if (is_ref(l)) l else as_jax(l))
    avals <- lapply(leaves, function(l) aval_of(if (is_ref(l)) ref_read(l) else l))
    # The full treedef (with aux data of registered nodes) is part of the key.
    key <- list(paste(paste(vapply(avals, aval_key, ""), collapse = ";"),
                      paste(nms, collapse = ","), paste(which(is_static), collapse = ","),
                      paste(ref_pos, collapse = ","), sep = "|"),
                treedef_key(fl$treedef), unname(args[is_static]), trace_config_key(),
                .jx$user_contexts)
    entry <- utils::gethash(cache, key)
    if (is.null(entry)) {
      if (isTRUE(.jx$config$jax_no_tracing))
        jax_error("re-tracing function for jit under jax_no_tracing (no_tracing)",
                  class = "jax_runtime_error")
      if (isTRUE(.jx$config$jax_explain_cache_misses))
        message("TRACING CACHE MISS: tracing jit function for new arguments ",
                paste(vapply(avals, aval_key, ""), collapse = ", "),
                if (cache_keys) " (previous keys differ)" else " (first call)")
      cache_keys <<- cache_keys + 1L
      store <- new.env(parent = emptyenv())
      fun_leaves <- function(ls) {
        a <- args
        full <- fl$leaves
        nonref <- setdiff(seq_along(full), ref_pos)
        full[nonref] <- ls[nonref]
        full[ref_pos] <- .jx$ref_ctx$proxies
        a[!is_static] <- tree_unflatten(fl$treedef, full)
        out <- do.call(fun, a)
        ofl <- tree_flatten(out)
        store$out_tree <- ofl$treedef
        # Port of partial_eval._check_no_returned_refs.
        for (l in ofl$leaves) if (is_ref(l))
          jax_error("function traced for jit returned a mutable array reference of type Ref{",
                    aval_long(l$aval), "}, but mutable array references cannot be returned.",
                    class = "jax_value_error")
        lapply(ofl$leaves, as_jax)
      }
      # The bodies of the internal jit functions of random are traced quietly (R/key_reuse.R).
      old_quiet <- .jx$kr_quiet
      if (isTRUE(.jx$kr_prim)) .jx$kr_quiet <- TRUE
      rt <- tryCatch(mon_elapsed("/jax/core/compile/jaxpr_trace_duration", "jit_fun",
                                 ref_jit_trace(fun_leaves, avals, call_refs)),
                     finally = .jx$kr_quiet <- old_quiet)
      cj <- rt$cj
      entry <- new.env(parent = emptyenv())
      entry$jaxpr <- cj$jaxpr
      entry$consts <- cj$consts
      entry$out_tree <- store$out_tree
      entry$ref_where <- rt$where
      entry$n_out <- rt$n_out
      # The keys from outside consumed while tracing are consumed by the call below.
      kr_revert(rt$kr)
      entry$kr <- rt$kr
      if (!any(vapply(cj$consts, is_tracer, NA))) utils::sethash(cache, key, entry)
    }
    # Refs: the arguments that are refs are not inputs themselves (their position is a
    # dummy); every ref the trace captured contributes its current value.
    refs <- lapply(entry$ref_where, function(w) if (!is.null(w$arg)) call_refs[[w$arg]] else w$ref)
    for (w in entry$ref_where) if (is.null(w$arg)) ref_check_alias(w$ref, call_refs)
    ins <- c(lapply(seq_along(leaves), function(i) if (i %in% ref_pos) ref_read(leaves[[i]]) else leaves[[i]]),
             lapply(refs, ref_read))
    kr_apply_in(entry$kr, leaves, "jit-compiled function")
    trace <- current_trace()
    outs <- if (inherits(trace, "eval_trace")) {
      execute_jaxpr(entry$jaxpr, entry$consts, ins, entry = entry)
    } else {
      eval_jaxpr(entry$jaxpr, entry$consts, ins)
    }
    n_out <- entry$n_out %||% length(outs)
    kr_apply_out(entry$kr, leaves, outs[seq_len(n_out)])
    for (i in seq_along(refs)) ref_write(refs[[i]], outs[[n_out + i]])
    tree_unflatten(entry$out_tree, outs[seq_len(n_out)])
  }
  class(f) <- c("jax_jitted", "function")
  attr(f, "fun") <- fun
  attr(f, "static") <- static
  f
}

#' Disable jit compilation temporarily
#' @param code Expression evaluated with jit disabled.
#' @export
disable_jit <- function(code) with_jax_config("jax_disable_jit", TRUE, code)

#' Evaluate a function abstractly
#'
#' Port of `jax.eval_shape`.
#' @param fun Function.
#' @param ... Arguments (arrays or [ShapeDtypeStruct()] values).
#' @return A pytree of [ShapeDtypeStruct()] values.
#' @export
eval_shape <- function(fun, ...) {
  args <- list(...)
  fl <- tree_flatten(args)
  store <- new.env(parent = emptyenv())
  cj <- trace_to_jaxpr(function(ls) {
    out <- do.call(fun, tree_unflatten(fl$treedef, ls))
    ofl <- tree_flatten(out)
    store$out_tree <- ofl$treedef
    lapply(ofl$leaves, as_jax)
  }, lapply(fl$leaves, aval_of))
  kr_trace_only(cj)
  tree_unflatten(store$out_tree, jaxpr_out_avals(cj$jaxpr))
}

# jax_debug_nans / jax_debug_infs: check floating outputs after execution.
check_debug_values <- function(arrays, name) {
  dn <- isTRUE(.jx$config$jax_debug_nans)
  di <- isTRUE(.jx$config$jax_debug_infs)
  if (!dn && !di) return(invisible())
  for (a in arrays) {
    if (!is_inexact_dtype(a$aval$dtype)) next
    h <- array_host(a)
    v <- if (is.complex(h)) c(Re(h), Im(h)) else as.vector(h)
    if (dn && anyNA(v) && any(is.nan(v)))
      jax_error("invalid value (nan) encountered in ", name, class = "jax_floating_point_error")
    if (di && any(is.infinite(v)))
      jax_error("divide by zero or overflow (inf) encountered in ", name,
                class = "jax_floating_point_error")
  }
  invisible()
}
