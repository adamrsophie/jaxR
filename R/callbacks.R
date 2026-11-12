# Host callbacks (port of jax/_src/callback.py): pure_callback and io_callback, plus the
# machinery shared with the debug callbacks of R/debug.R.
#
# A callback primitive lowers to a stablehlo.custom_call to the typed FFI target
# "jaxr_r_callback" (src/callbacks.cpp, the counterpart of jaxlib's
# xla_ffi_python_cpu_callback) with the attribute `index`, a position in the callback
# registry below. The handler copies the operands to R vectors on the R main thread and
# calls callback_dispatch(). In eager mode the primitives call the R function directly.
#
# Callbacks receive R arrays (the host values of the operands, as as.array() returns
# them) and may return R data or jax arrays.

.jx$ffi_hooks$callbacks <- function() {
  .jx$callbacks_ffi_error <- .Call("jaxr_callbacks_register", PACKAGE = "jaxR")
}

.jx$cb_registry <- list()

# ---- registry and dispatch --------------------------------------------------------------

# Index of a callback entry (a spec list, see cb_call(), and its result avals). Equal
# entries share an index, so re-lowering a program yields the same module text (and a
# compile cache hit) and the registry grows only with distinct callbacks.
# ponytail: entries are never released and lookup is a linear scan; key by a hash and
# drop entries with their executables if programs with many distinct callbacks appear.
cb_register <- function(spec, result_avals) {
  entry <- list(spec = spec, result_avals = lapply(result_avals, function(a) new_aval(a$shape, a$dtype)))
  reg <- .jx$cb_registry
  for (i in seq_along(reg)) if (identical(reg[[i]], entry)) return(i - 1L)
  .jx$cb_registry[[length(reg) + 1L]] <- entry
  length(reg)
}

# Run a callback spec on host values (R arrays). Returns a list of results.
cb_call <- function(spec, args) {
  switch(spec$kind,
    pure = , io = tree_leaves(do.call(spec$fun, tree_unflatten(spec$in_tree, args))),
    debug = {
      do.call(spec$fun, tree_unflatten(spec$in_tree, cb_merge_static(spec$static, args)))
      list()
    },
    print = {
      dbg_format_print(spec, args)
      list()
    },
    jax_error("internal: unknown callback kind ", spec$kind))
}

# Called by the FFI handler on the R main thread. `ins` holds the operands as flat
# row-major vectors with a "shape" attribute. Returns a list of flat row-major result
# vectors, or a character scalar with an error message.
callback_dispatch <- function(index, ins) {
  tryCatch({
    entry <- .jx$cb_registry[[index + 1L]]
    if (is.null(entry)) jax_error("internal: unknown host callback ", index)
    args <- lapply(ins, function(v) c_unravel(v, attr(v, "shape", exact = TRUE)))
    cb_flatten_results(cb_call(entry$spec, args), entry$result_avals)
  }, error = function(e) {
    msg <- conditionMessage(e)
    if (!nzchar(msg)) msg <- "host callback failed"
    msg
  })
}

cb_py_tuple <- function(s) {
  if (length(s) == 1L) paste0("(", s, ",)") else paste0("(", paste(s, collapse = ", "), ")")
}

# Check callback results against the expected avals (port of the checks of
# callback.py:emit_python_callback._wrapped_callback) and return flat row-major vectors
# of the R storage type of each dtype.
cb_flatten_results <- function(outs, result_avals) {
  if (length(outs) != length(result_avals))
    jax_error("Mismatched number of outputs from callback. Expected: ", length(result_avals),
              ", Actual: ", length(outs))
  Map(function(v, a, i) {
    if (is_jax_value(v)) {
      va <- aval_of(v)
      shape <- va$shape
      v <- array_host(concrete_or_error(v, "a callback result"))
      ok_dtype <- va$dtype == a$dtype
      got <- va$dtype
    } else {
      if (!(is.numeric(v) || is.logical(v) || is.complex(v)))
        jax_error("Incorrect output dtype for return value #", i, ": Expected: ", a$dtype,
                  ", Actual: ", class(v)[1L])
      shape <- r_shape(v)
      if (is.null(dim(v)) && length(a$shape) <= 1L && length(v) == prod_int(a$shape))
        shape <- a$shape
      k <- dtype_kind(a$dtype)
      ok_dtype <- if (is.logical(v)) k == "b" else if (is.complex(v)) k == "c" else k %in% c("i", "u", "f")
      got <- if (is.logical(v)) "bool" else if (is.complex(v)) "complex" else typeof(v)
    }
    if (!identical(as.integer(shape), a$shape))
      jax_error("Incorrect output shape for return value #", i, ": Expected: ", cb_py_tuple(a$shape),
                ", Actual: ", cb_py_tuple(shape))
    if (!ok_dtype)
      jax_error("Incorrect output dtype for return value #", i, ": Expected: ", a$dtype,
                ", Actual: ", got)
    flat <- c_ravel(v)
    switch(dtype_kind(a$dtype), b = as.logical(flat), c = as.complex(flat), as.double(flat))
  }, outs, result_avals, seq_along(outs) - 1L)
}

# Eager evaluation: call the R function on the host values of the operands.
cb_impl <- function(args, params) {
  host <- lapply(args, function(a) array_host(concrete_or_error(a, "a host callback")))
  flat <- cb_flatten_results(cb_call(params$callback, host), params$result_avals)
  Map(function(v, a) as_jax(c_unravel(v, a$shape), a$dtype), flat, params$result_avals)
}

# ---- lowering ---------------------------------------------------------------------------

cb_layout <- function(rank) {
  if (rank == 0L) return("dense<> : tensor<0xindex>")
  if (rank == 1L) return("dense<0> : tensor<1xindex>")
  paste0("dense<[", paste(seq.int(rank - 1L, 0L), collapse = ", "), "]> : tensor<", rank, "xindex>")
}

# Port of callback.py:emit_python_callback (CPU branch): a typed FFI custom call with
# the callback index as attribute. With `ordered`, a token threads the ordered
# callbacks of one region (the token of jax's ordered effects; jaxR has no
# function level tokens, so ordering is enforced within a region only).
cb_emit <- function(ctx, spec, ins, avals_in, avals_out, has_side_effect, ordered = FALSE) {
  index <- cb_register(spec, avals_out)
  types_in <- vapply(avals_in, tensor_type, "")
  types_out <- vapply(avals_out, tensor_type, "")
  lay_in <- vapply(avals_in, function(a) cb_layout(length(a$shape)), "")
  lay_out <- vapply(avals_out, function(a) cb_layout(length(a$shape)), "")
  if (ordered) {
    tok <- ctx$cb_token %||%
      emit_op(ctx, "stablehlo.create_token", character(0), character(0), "!stablehlo.token")
    ins <- c(tok, ins)
    types_in <- c("!stablehlo.token", types_in)
    lay_in <- c(cb_layout(0L), lay_in)
    types_out <- c("!stablehlo.token", types_out)
    lay_out <- c(cb_layout(0L), lay_out)
  }
  nres <- length(types_out)
  lhs <- ""
  outs <- character(0)
  if (nres >= 1L) {
    v <- ssa_name(ctx)
    lhs <- if (nres == 1L) paste0(v, " = ") else paste0(v, ":", nres, " = ")
    outs <- if (nres == 1L) v else paste0(v, "#", seq_len(nres) - 1L)
  }
  rt <- if (nres == 1L) types_out else paste0("(", paste(types_out, collapse = ", "), ")")
  emit_line(ctx, paste0(
    lhs, "\"stablehlo.custom_call\"(", paste(ins, collapse = ", "), ") <{api_version = 1 : i32, ",
    "backend_config = \"\", call_target_name = \"jaxr_r_callback\", called_computations = [], ",
    "has_side_effect = ", attr_bool(has_side_effect),
    ", operand_layouts = [", paste(lay_in, collapse = ", "), "], result_layouts = [",
    paste(lay_out, collapse = ", "), "]}> {mhlo.backend_config = {index = ", index, " : ui64}} : (",
    paste(types_in, collapse = ", "), ") -> ", rt))
  if (ordered) {
    ctx$cb_token <- outs[1L]
    outs <- outs[-1L]
  }
  outs
}

cb_check_backend <- function() {
  ensure_backend()
  err <- .jx$callbacks_ffi_error
  if (!is.null(err) && nzchar(err))
    jax_error("host callbacks are unavailable: the XLA runtime rejected the FFI handler (", err, ")")
}

# ---- shared argument handling --------------------------------------------------------------

# Flatten the (args, kwargs) of a callback; every leaf must be an array.
cb_flatten_args <- function(args) {
  fl <- tree_flatten(args)
  list(leaves = lapply(fl$leaves, as_jax), in_tree = fl$treedef)
}

# Port of callback.py:_check_shape_dtype and the ShapedArray construction.
cb_result_avals <- function(result_shape_dtypes) {
  fl <- tree_flatten(result_shape_dtypes)
  avals <- lapply(fl$leaves, function(x) {
    a <- if (inherits(x, "jax_aval")) x else aval_of(x)
    if (canonicalize_dtype(a$dtype) != a$dtype)
      jax_error("result_shape_dtypes cannot specify 64-bit types when `jax_enable_x64` is disabled",
                class = "jax_value_error")
    new_aval(a$shape, a$dtype)
  })
  list(avals = avals, out_tree = fl$treedef)
}

# ---- pure_callback ------------------------------------------------------------------------

pure_callback_p <- new_primitive("pure_callback", multiple_results = TRUE)
pure_callback_p$impl <- cb_impl
pure_callback_p$abstract_eval <- function(avals, params) params$result_avals
pure_callback_p$jvp <- function(primals, tangents, params) {
  jax_error("Pure callbacks do not support JVP. Please use `jax.custom_jvp` to use callbacks ",
            "while taking gradients.", class = "jax_value_error")
}
pure_callback_p$transpose <- function(ct, args, params) {
  jax_error("Pure callbacks do not support transpose. Please use `jax.custom_vjp` to use ",
            "callbacks while taking gradients.", class = "jax_value_error")
}
pure_callback_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  cb_check_backend()
  cb_emit(ctx, params$callback, ins, avals_in, avals_out, has_side_effect = FALSE)
}

# Port of jax/_src/ffi.py:ffi_batching_rule (the rule of pure_callback_p).
cb_ffi_batching_rule <- function(prim) {
  function(vals, dims, params, axis) {
    size <- axis$size
    new_args <- Map(function(v, d) if (is.na(d)) v else moveaxis(v, d, 0L), vals, as.list(dims))
    batched_avals <- lapply(params$result_avals, function(a) new_aval(c(size, a$shape), a$dtype))
    vm <- params$vmap_method
    is_vm <- function(...) !is.null(vm) && vm %in% c(...)
    if (is_vm("legacy_vectorized")) {
      outs <- bind(prim, new_args, update_params(params, list(result_avals = batched_avals)))
    } else if (is_vm("expand_dims", "broadcast_all")) {
      s <- if (vm == "broadcast_all") size else 1L
      bcast <- Map(function(x, d) if (is.na(d)) lax_broadcast(x, s) else x, new_args, as.list(dims))
      outs <- bind(prim, bcast, update_params(params, list(result_avals = batched_avals)))
    } else if (is_vm("sequential", "sequential_unrolled")) {
      is_b <- !is.na(dims)
      g <- function(carry, xs) {
        merged <- new_args
        merged[is_b] <- xs
        list(carry, bind(prim, merged, params))
      }
      outs <- lax_scan(g, NULL, new_args[is_b])[[2L]]
    } else {
      jax_error("vmap is only supported for the ", prim$name, " primitive when vmap_method is one ",
                "of 'sequential', 'sequential_unrolled', 'expand_dims', 'broadcast_all', or ",
                "'legacy_vectorized'. Got vmap_method=", if (is.null(vm)) "None" else paste0("'", vm, "'"),
                ".", class = "jax_not_implemented_error")
    }
    list(outs, rep(0L, length(outs)))
  }
}
pure_callback_p$batch <- cb_ffi_batching_rule(pure_callback_p)

cb_vmap_methods <- c("sequential", "sequential_unrolled", "expand_dims", "broadcast_all",
                     "legacy_vectorized")

#' Call a pure R function from compiled code
#'
#' Port of `jax.pure_callback`. `callback` is called with R arrays (the host values of
#' the array arguments, as [as.array()] returns them) and must return R data or jax
#' arrays matching `result_shape_dtypes`. It works under [jit()], [vmap()] and the
#' other transformations: inside compiled code the call goes through an XLA FFI
#' handler that runs the function on the R main thread. The callback is assumed pure:
#' it may be called several times, or not at all when its result is unused.
#'
#' Under [vmap()] the behavior follows `vmap_method`: `"sequential"` (and
#' `"sequential_unrolled"`) loops over the batch with [lax_scan()]; `"expand_dims"`
#' adds a leading axis of size one to unbatched arguments; `"broadcast_all"` broadcasts
#' them to the batch size; `"legacy_vectorized"` passes batched arguments unchanged.
#' With `vmap_method = NULL`, vmap raises an error, as in JAX. Differentiation raises
#' an error unless the callback is wrapped in [custom_jvp()] or [custom_vjp()].
#' @param callback An R function.
#' @param result_shape_dtypes A pytree of [ShapeDtypeStruct()] values (or arrays)
#'   describing the results.
#' @param ... Arguments (pytrees of arrays) passed to `callback`; named arguments are
#'   passed by name.
#' @param sharding Ignored (single device).
#' @param vmap_method One of `"sequential"`, `"sequential_unrolled"`, `"expand_dims"`,
#'   `"broadcast_all"`, `"legacy_vectorized"` or `NULL`.
#' @return A pytree of arrays with the structure of `result_shape_dtypes`.
#' @examples
#' \dontrun{
#' f <- function(x) pure_callback(function(v) besselJ(v, 0), ShapeDtypeStruct(dim(x), "float32"), x)
#' jit(f)(as_jax(c(0.5, 1, 2)))
#' }
#' @export
pure_callback <- function(callback, result_shape_dtypes, ..., sharding = NULL, vmap_method = NULL) {
  if (!is.null(vmap_method) && !(is.character(vmap_method) && length(vmap_method) == 1L &&
                                 vmap_method %in% cb_vmap_methods))
    jax_error("vmap_method must be on of the allowed methods ['sequential', ",
              "'sequential_unrolled', 'expand_dims', 'broadcast_all', 'legacy_vectorized', ",
              "None], but got: ", paste(format(vmap_method), collapse = " "),
              class = "jax_value_error")
  if (!is.function(callback))
    jax_error("pure_callback: callback must be a function", class = "jax_type_error")
  fl <- cb_flatten_args(list(...))
  ra <- cb_result_avals(result_shape_dtypes)
  out <- bind(pure_callback_p, fl$leaves, list(
    callback = list(kind = "pure", fun = callback, in_tree = fl$in_tree),
    result_avals = ra$avals, sharding = NULL, vmap_method = vmap_method))
  tree_unflatten(ra$out_tree, out)
}

# ---- io_callback ----------------------------------------------------------------------------

io_callback_p <- new_primitive("io_callback", multiple_results = TRUE)
io_callback_p$effectful <- TRUE
io_callback_p$impl <- cb_impl
io_callback_p$abstract_eval <- function(avals, params) params$result_avals
io_callback_p$jvp <- function(primals, tangents, params) {
  jax_error("IO callbacks do not support JVP.", class = "jax_value_error")
}
io_callback_p$transpose <- function(ct, args, params) {
  jax_error("IO callbacks do not support transpose.", class = "jax_value_error")
}
io_callback_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  cb_check_backend()
  cb_emit(ctx, params$callback, ins, avals_in, avals_out, has_side_effect = TRUE,
          ordered = isTRUE(params$ordered))
}

# Port of callback.py:io_callback_batching_rule.
io_callback_p$batch <- function(vals, dims, params, axis) {
  if (isTRUE(params$ordered))
    jax_error("Cannot `vmap` ordered IO callback.", class = "jax_value_error")
  is_b <- !is.na(dims)
  new_args <- Map(function(v, d) if (is.na(d)) v else moveaxis(v, d, 0L), vals, as.list(dims))
  outs <- lax_map(function(xs) {
    merged <- new_args
    merged[is_b] <- xs
    bind(io_callback_p, merged, update_params(params, list(ordered = FALSE)))
  }, new_args[is_b])
  list(outs, rep(0L, length(outs)))
}

#' Call an impure R function from compiled code
#'
#' Port of `jax.experimental.io_callback`. Like [pure_callback()], but the callback is
#' treated as having side effects: it is not removed when its results are unused, and
#' it cannot be differentiated. Under [vmap()] it is called once per batch element
#' (an `ordered` callback cannot be vmapped). `ordered = TRUE` orders the ordered
#' callbacks of one compiled region with a token.
#' @inheritParams pure_callback
#' @param ordered Whether calls must keep their program order.
#' @return A pytree of arrays with the structure of `result_shape_dtypes`.
#' @export
experimental_io_callback <- function(callback, result_shape_dtypes, ..., sharding = NULL,
                                     ordered = FALSE) {
  if (!is.function(callback))
    jax_error("io_callback: callback must be a function", class = "jax_type_error")
  fl <- cb_flatten_args(list(...))
  ra <- cb_result_avals(result_shape_dtypes)
  out <- bind(io_callback_p, fl$leaves, list(
    callback = list(kind = "io", fun = callback, in_tree = fl$in_tree),
    result_avals = ra$avals, sharding = NULL, ordered = isTRUE(ordered)))
  tree_unflatten(ra$out_tree, out)
}

#' @rdname experimental_io_callback
#' @export
io_callback <- experimental_io_callback
