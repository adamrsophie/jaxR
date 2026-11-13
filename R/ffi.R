# Port of jax.ffi (jax/_src/ffi.py): calling compiled foreign functions (XLA FFI
# handlers) from jaxR programs.

#' Foreign function interface
#'
#' Ports of `jax.ffi.register_ffi_target`, `jax.ffi.register_ffi_type`,
#' `jax.ffi.register_ffi_type_id`, `jax.ffi.register_ffi_target_as_batch_partitionable`
#' and `jax.ffi.include_dir`.
#'
#' A foreign function is a typed XLA FFI handler written in C or C++ with the headers
#' in `ffi_include_dir()` (`#include "xla/ffi/api/ffi.h"`, then
#' `XLA_FFI_DEFINE_HANDLER_SYMBOL`), compiled into a shared library and loaded with
#' [dyn.load()]. `ffi_register_target()` registers it under a name; [ffi_call()]
#' calls it.
#'
#' @param name The name of the target (or of the type).
#' @param fn The handler: the address of the symbol defined with
#'   `XLA_FFI_DEFINE_HANDLER_SYMBOL`, as an external pointer or the result of
#'   [getNativeSymbolInfo()]; or a named list with element `execute` holding it (the
#'   other FFI stages, `instantiate`, `prepare` and `initialize`, cannot be registered
#'   through the PJRT plugin interface jaxR uses).
#' @param platform The target platform: `"cpu"` (the default), `"cuda"`, `"rocm"`, or
#'   the platform name XLA uses.
#' @param api_version The custom call API of the handler: 1 (the default) for the typed
#'   FFI. Handlers of the earlier custom call API (0) cannot be registered through the
#'   PJRT plugin interface.
#' @param ... Not used (JAX passes extra arguments to
#'   `xla_client.register_custom_call_target`).
#' @return `ffi_register_target()`, `ffi_register_type()` and
#'   `ffi_register_target_as_batch_partitionable()` return `NULL` invisibly;
#'   `ffi_include_dir()` returns the directory of the XLA FFI headers.
#' @examples
#' \dontrun{
#' dir <- tempfile("ffi")
#' dir.create(dir)
#' writeLines(c(
#'   '#include "xla/ffi/api/ffi.h"',
#'   'namespace ffi = xla::ffi;',
#'   'static ffi::Error AddOne(ffi::Buffer<ffi::F32> x, float y,',
#'   '                         ffi::ResultBuffer<ffi::F32> out) {',
#'   '  for (size_t i = 0; i < x.element_count(); ++i)',
#'   '    out->typed_data()[i] = x.typed_data()[i] + y;',
#'   '  return ffi::Error::Success();',
#'   '}',
#'   'XLA_FFI_DEFINE_HANDLER_SYMBOL(add_one, AddOne,',
#'   '  ffi::Ffi::Bind().Arg<ffi::Buffer<ffi::F32>>().Attr<float>("y")',
#'   '    .Ret<ffi::Buffer<ffi::F32>>());'), file.path(dir, "add_one.cc"))
#' writeLines(c("CXX_STD = CXX17", paste0("PKG_CPPFLAGS = -I", shQuote(ffi_include_dir()))),
#'            file.path(dir, "Makevars"))
#' system2(file.path(R.home("bin"), "R"), c("CMD", "SHLIB", file.path(dir, "add_one.cc")),
#'         env = paste0("R_MAKEVARS_USER=", file.path(dir, "Makevars")))
#' dll <- dyn.load(file.path(dir, paste0("add_one", .Platform$dynlib.ext)))
#' ffi_register_target("add_one", getNativeSymbolInfo("add_one", dll))
#' x <- jnp_arange(4, dtype = "float32")
#' ffi_call("add_one", ShapeDtypeStruct(4L, "float32"))(x, y = as_jax(1, "float32"))
#' }
#' @export
ffi_register_target <- function(name, fn, platform = "cpu", api_version = 1L, ...) {
  if (!is.character(name) || length(name) != 1L || !nzchar(name))
    jax_error("register_ffi_target: name must be a string", class = "jax_type_error")
  if (!identical(as.integer(api_version), 1L))
    jax_error("register_ffi_target: only typed FFI handlers (api_version = 1) can be ",
              "registered in jaxR; the custom call API (api_version = 0) is not available ",
              "through the PJRT plugin interface", class = "jax_not_implemented_error")
  if (is.list(fn) && !inherits(fn, "NativeSymbolInfo")) {
    other <- setdiff(names(fn), "execute")
    if (length(other) || is.null(fn$execute))
      jax_error("register_ffi_target: only the 'execute' stage of an FFI handler can be ",
                "registered in jaxR, got stages ", paste(names(fn), collapse = ", "),
                class = "jax_not_implemented_error")
    fn <- fn$execute
  }
  ptr <- ffi_pointer(fn, "fn")
  ensure_backend()
  msg <- .Call("jaxr_ffi_register_target", name, ptr, ffi_platform(platform), 0L,
               PACKAGE = "jaxR")
  if (nzchar(msg))
    jax_error("register_ffi_target: the XLA runtime rejected the handler for '", name,
              "': ", msg)
  invisible(NULL)
}

#' @rdname ffi_register_target
#' @param type_registration For `ffi_register_type()`: a list with element `type_id`,
#'   the address of the `XLA_FFI_TypeId` of the type (filled in by XLA when it holds
#'   0), and optionally `type_info`, the address of its `XLA_FFI_TypeInfo`.
#' @export
ffi_register_type <- function(name, type_registration, platform = "cpu") {
  if (!is.list(type_registration) || is.null(type_registration$type_id))
    jax_error("register_ffi_type: type_registration must be a list with element type_id",
              class = "jax_type_error")
  id <- ffi_pointer(type_registration$type_id, "type_id")
  info <- if (!is.null(type_registration$type_info)) ffi_pointer(type_registration$type_info, "type_info")
  ffi_platform(platform)
  ensure_backend()
  msg <- .Call("jaxr_ffi_register_type", name, id, info, PACKAGE = "jaxR")
  if (nzchar(msg))
    jax_error("register_ffi_type: the XLA runtime rejected the type '", name, "': ", msg)
  invisible(NULL)
}

#' @rdname ffi_register_target
#' @param obj Not used.
#' @export
ffi_register_type_id <- function(name, obj, platform = "cpu") {
  jax_error("register_ffi_type_id is not supported after jaxlib version 381.",
            class = "jax_value_error")
}

#' @rdname ffi_register_target
#' @export
ffi_register_target_as_batch_partitionable <- function(name) {
  # Batch partitioning splits a call across devices; jaxR runs on one device.
  invisible(NULL)
}

#' @rdname ffi_register_target
#' @export
ffi_include_dir <- function() system.file("include", package = "jaxR", mustWork = TRUE)

ffi_pointer <- function(x, what) {
  if (inherits(x, "NativeSymbolInfo")) x <- x$address
  if (!is.null(x) && typeof(x) == "externalptr") return(x)
  jax_error("register_ffi_target: ", what, " must be an external pointer or the result ",
            "of getNativeSymbolInfo(), got ", class(x)[1L], class = "jax_type_error")
}

# Platform names as xla_client.register_custom_call_target maps them.
ffi_platform <- function(platform) {
  if (!is.character(platform) || length(platform) != 1L)
    jax_error("platform must be a string", class = "jax_type_error")
  switch(tolower(platform), cpu = , host = "Host", cuda = , gpu = "CUDA", rocm = "ROCM", platform)
}

# ---- ffi_call -------------------------------------------------------------------------

ffi_call_p <- new_primitive("ffi_call", multiple_results = TRUE)
ffi_call_p$effectful <- function(params) isTRUE(params$has_side_effect)
ffi_call_p$abstract_eval <- function(avals, params) params$result_avals
ffi_call_p$jvp <- function(primals, tangents, params) {
  jax_error("The FFI call to `", params$target_name, "` cannot be differentiated. You can use ",
            "`jax.custom_jvp` or `jax.custom_vjp` to add support.", class = "jax_value_error")
}
ffi_call_p$transpose <- function(ct, args, params) {
  jax_error("The FFI call to `", params$target_name, "` cannot be differentiated. You can use ",
            "`jax.custom_jvp` or `jax.custom_vjp` to add support.", class = "jax_value_error")
}

# Port of ffi_call_lowering and build_ffi_lowering_function: a typed FFI call
# (api_version 4) carries its attributes in mhlo.backend_config with api_version 1.
ffi_call_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  typed <- params$custom_call_api_version >= 4L
  props <- c(
    api_version = paste0(if (typed) 1L else params$custom_call_api_version, " : i32"),
    backend_config = if (!typed && !is.null(params$legacy_backend_config))
      ffi_attr(params$legacy_backend_config) else "\"\"",
    call_target_name = ffi_attr(params$target_name), called_computations = "[]",
    has_side_effect = attr_bool(params$has_side_effect),
    operand_layouts = paste0("[", paste(vapply(params$input_layouts, layout_attr, ""),
                                        collapse = ", "), "]"))
  nres <- length(avals_out)
  if (length(params$input_output_aliases)) {
    props[["output_operand_aliases"]] <- paste0("[", paste(vapply(params$input_output_aliases, function(al) {
      paste0("#stablehlo.output_operand_alias<output_tuple_indices = [",
             if (nres > 1L) al[2L] else "", "], operand_index = ", al[1L],
             ", operand_tuple_indices = []>")
    }, ""), collapse = ", "), "]")
  }
  props[["result_layouts"]] <- paste0(
    "[", paste(vapply(params$output_layouts, layout_attr, ""), collapse = ", "), "]")
  attrs <- params$attributes
  extra <- if (typed) {
    paste0(" {mhlo.backend_config = ", ffi_dict_attr(attrs), "}")
  } else ""
  lhs <- ""
  outs <- character(0)
  if (nres >= 1L) {
    v <- ssa_name(ctx)
    lhs <- if (nres == 1L) paste0(v, " = ") else paste0(v, ":", nres, " = ")
    outs <- if (nres == 1L) v else paste0(v, "#", seq_len(nres) - 1L)
  }
  emit_line(ctx, paste0(
    lhs, "\"stablehlo.custom_call\"(", paste(ins, collapse = ", "), ") <{",
    paste(names(props), props, sep = " = ", collapse = ", "), "}>", extra, " : (",
    paste(types_of(avals_in), collapse = ", "), ") -> ", fmt_types(types_of(avals_out))))
  outs
}

# Port of ffi_batching_rule: the layouts gain the batch dimension as the most major.
ffi_call_batch_base <- cb_ffi_batching_rule(ffi_call_p)
ffi_call_p$batch <- function(vals, dims, params, axis) {
  vm <- params$vmap_method
  shift <- function(l) c(l + 1L, 0L)
  if (!is.null(vm) && !vm %in% c("sequential", "sequential_unrolled")) {
    params$output_layouts <- lapply(params$output_layouts, shift)
    if (vm == "legacy_vectorized") {
      params$input_layouts <- Map(function(l, d) if (is.na(d)) l else shift(l),
                                  params$input_layouts, as.list(dims))
    } else if (vm %in% c("expand_dims", "broadcast_all")) {
      params$input_layouts <- lapply(params$input_layouts, shift)
    }
  }
  ffi_call_batch_base(vals, dims, params, axis)
}

#' Call a foreign function
#'
#' Port of `jax.ffi.ffi_call`. Returns a function that calls the XLA FFI target
#' `target_name` (registered with [ffi_register_target()]). Its unnamed arguments are
#' the input arrays and its named arguments are the attributes of the handler. It
#' works eagerly and under [jit()] and [vmap()] (see `vmap_method`); it cannot be
#' differentiated (use [custom_jvp()] or [custom_vjp()]).
#'
#' Attributes convert as in JAX: an R integer scalar is a 64-bit integer (a Python
#' `int`), a double scalar a 64-bit float, a logical scalar a boolean, a string a
#' string, a jax scalar an attribute of its dtype, a jax array or an R vector of
#' length other than one a dense array (of int64 for R integers, as for Python
#' integers, and float64 for R doubles; pass a jax array for another dtype), a named
#' list a dictionary and an unnamed list an array of attributes.
#' @param target_name The name of the registered target.
#' @param result_shape_dtypes A [ShapeDtypeStruct()] (or array) describing the output,
#'   or a list of them for several outputs (the function then returns a list).
#' @param has_side_effect Whether the call has side effects: it then runs even when its
#'   outputs are unused.
#' @param vmap_method How the call behaves under [vmap()], as for [pure_callback()]:
#'   `"sequential"`, `"sequential_unrolled"`, `"expand_dims"`, `"broadcast_all"`,
#'   `"legacy_vectorized"` or `NULL` (vmap is then an error).
#' @param input_layouts,output_layouts `NULL` (row major), or a list with one entry per
#'   input (output): `NULL` or a major to minor order of the axes (0-based, so
#'   `c(1, 0)` is a column major matrix).
#' @param input_output_aliases `NULL`, or a named integer vector or list mapping input
#'   indices (names) to output indices, 0-based: the output reuses the buffer of the
#'   input.
#' @param custom_call_api_version The custom call API of the target; 4 (the default)
#'   is the typed FFI.
#' @param legacy_backend_config For `custom_call_api_version < 4`: the opaque backend
#'   configuration string.
#' @return A function of the inputs and attributes.
#' @seealso [ffi_register_target()]
#' @export
ffi_call <- function(target_name, result_shape_dtypes, has_side_effect = FALSE,
                     vmap_method = NULL, input_layouts = NULL, output_layouts = NULL,
                     input_output_aliases = NULL, custom_call_api_version = 4L,
                     legacy_backend_config = NULL) {
  if (!is.null(vmap_method) && !(is.character(vmap_method) && length(vmap_method) == 1L &&
                                 vmap_method %in% cb_vmap_methods))
    jax_error("vmap_method must be on of the allowed methods ['sequential', ",
              "'sequential_unrolled', 'expand_dims', 'broadcast_all', 'legacy_vectorized', ",
              "None], but got: ", paste(format(vmap_method), collapse = " "),
              class = "jax_value_error")
  multiple <- is.list(result_shape_dtypes) && !inherits(result_shape_dtypes, "jax_aval") &&
    !inherits(result_shape_dtypes, "jax_ShapeDtypeStruct")
  rs <- if (multiple) result_shape_dtypes else list(result_shape_dtypes)
  result_avals <- lapply(seq_along(rs), function(i) {
    x <- rs[[i]]
    if (inherits(x, "jax_aval")) return(new_aval(x$shape, x$dtype))
    a <- tryCatch(aval_of(x), error = function(e) NULL)
    if (is.null(a))
      jax_error("All elements of result_shape_dtypes must have 'shape' and 'dtype' ",
                "attributes. Got ", class(x)[1L], " at position ", i - 1L, ".",
                class = "jax_value_error")
    new_aval(a$shape, a$dtype)
  })
  custom_call_api_version <- as.integer(custom_call_api_version)
  if (custom_call_api_version >= 4L && !is.null(legacy_backend_config))
    jax_error("The use of the legacy_backend_config parameter requires ",
              "custom_call_api_version < 4; got ", custom_call_api_version, ".",
              class = "jax_value_error")
  out_layouts <- if (multiple) output_layouts else if (!is.null(output_layouts)) list(output_layouts)
  force(target_name); force(has_side_effect); force(vmap_method); force(input_layouts)
  force(input_output_aliases); force(legacy_backend_config)
  function(...) {
    a <- list(...)
    nm <- names(a) %||% character(length(a))
    args <- lapply(unname(a[!nzchar(nm)]), as_jax)
    attrs <- a[nzchar(nm)]
    if (length(attrs) && custom_call_api_version < 4L)
      jax_error("The use of ffi_call attributes requires a custom call API version of at ",
                "least 4; got api_version=", custom_call_api_version, ".",
                class = "jax_value_error")
    if (length(attrs)) attrs <- lapply(attrs[order(names(attrs))], ffi_attr)
    in_avals <- lapply(args, aval_of)
    in_l <- ffi_layouts(in_avals, input_layouts, "input arguments", "input layouts")
    out_l <- ffi_layouts(result_avals, out_layouts, "outputs", "output layouts")
    aliases <- list()
    if (length(input_output_aliases)) {
      io <- unlist(input_output_aliases)
      ii <- as.integer(names(io))
      oo <- as.integer(io)
      for (k in order(ii)) {
        i <- ii[k]; o <- oo[k]
        map <- paste0("'", i, ":", o, "'")
        if (i >= length(args))
          jax_error("input_output_aliases contains the mapping ", map, " with input index ", i,
                    " outside the range [0, ", length(args), ").", class = "jax_value_error")
        if (o >= length(result_avals))
          jax_error("input_output_aliases contains the mapping ", map, " with output index ", o,
                    " outside the range [0, ", length(result_avals), ").", class = "jax_value_error")
        ia <- in_avals[[i + 1L]]; oa <- result_avals[[o + 1L]]
        if (!identical(as.integer(ia$shape), as.integer(oa$shape)) || ia$dtype != oa$dtype)
          jax_error("input_output_aliases contains the mapping ", map, " referring to an ",
                    "input with abstract value ", aval_str(ia), " and an output with a ",
                    "different abstract value ", aval_str(oa), ".", class = "jax_value_error")
        if (!identical(in_l[[i + 1L]], out_l[[o + 1L]]))
          jax_error("input_output_aliases contains the mapping ", map, " referring to an ",
                    "input with layout (", paste(in_l[[i + 1L]], collapse = ", "), ") and an ",
                    "output with a different layout (", paste(out_l[[o + 1L]], collapse = ", "),
                    ").", class = "jax_value_error")
        aliases[[length(aliases) + 1L]] <- c(i, o)
      }
    }
    outs <- bind(ffi_call_p, args, list(
      result_avals = result_avals, vmap_method = vmap_method, target_name = target_name,
      has_side_effect = isTRUE(has_side_effect), input_layouts = in_l, output_layouts = out_l,
      input_output_aliases = aliases, custom_call_api_version = custom_call_api_version,
      legacy_backend_config = legacy_backend_config, attributes = attrs))
    if (multiple) outs else outs[[1L]]
  }
}

# Port of _convert_layout_for_lowering: layouts are given major to minor and passed
# to XLA minor to major.
ffi_layouts <- function(avals, layouts, what, what_l) {
  if (is.null(layouts)) return(lapply(avals, function(a) rev(seq_along(a$shape)) - 1L))
  if (length(layouts) != length(avals))
    jax_error("The number of ", what, " (", length(avals), ") must equal the number of ",
              what_l, " (", length(layouts), ").", class = "jax_value_error")
  Map(function(a, l) if (is.null(l)) rev(seq_along(a$shape)) - 1L else rev(as.integer(l)),
      avals, layouts)
}

# Port of mlir.ir_attribute for the values an R user passes.
ffi_attr <- function(v) {
  if (is_jax_array(v)) {
    dt <- v$aval$dtype
    data <- as.array(v)
    if (!length(v$aval$shape)) return(ffi_scalar_attr(as.vector(data), dt))
    return(ffi_dense_attr(data, v$aval$shape, dt))
  }
  if (is.list(v) && is.null(oldClass(v))) {
    if (!is.null(names(v)) && all(nzchar(names(v)))) return(ffi_dict_attr(lapply(v, ffi_attr)))
    return(paste0("[", paste(vapply(v, ffi_attr, ""), collapse = ", "), "]"))
  }
  if (is.character(v) && length(v) == 1L) return(mlir_string(v))
  if ((is.numeric(v) || is.logical(v)) && is.null(oldClass(v))) {
    dt <- if (is.logical(v)) "bool" else if (is.integer(v)) "int64" else "float64"
    if (length(v) == 1L && is.null(dim(v))) return(ffi_scalar_attr(v, dt))
    shape <- if (is.null(dim(v))) length(v) else dim(v)
    return(ffi_dense_attr(v, shape, dt))
  }
  jax_error("Non-hashable keyword argument to ffi_call: unsupported attribute of class ",
            class(v)[1L], class = "jax_type_error")
}

# An MLIR string literal: quotes, backslashes and control or non ASCII bytes are
# written as \XX escapes.
mlir_string <- function(s) {
  b <- as.integer(charToRaw(enc2utf8(s)))
  esc <- b < 32L | b > 126L | b == 34L | b == 92L
  out <- ifelse(esc, sprintf("\\%02X", b), vapply(b, function(i) rawToChar(as.raw(i)), ""))
  paste0("\"", paste(out, collapse = ""), "\"")
}

ffi_dict_attr <- function(attrs) {
  if (!length(attrs)) return("{}")
  keys <- names(attrs)
  keys <- ifelse(grepl("^[A-Za-z_][A-Za-z0-9_$.]*$", keys), keys, vapply(keys, mlir_string, ""))
  paste0("{", paste(keys, unlist(attrs), sep = " = ", collapse = ", "), "}")
}

ffi_scalar_attr <- function(x, dt) {
  k <- dtype_kind(dt)
  ty <- mlir_elem_type(dt)
  if (k == "b") return(if (isTRUE(as.logical(x))) "true" else "false")
  if (k == "f") {
    lit <- if (is.finite(x)) sprintf("%.17e", x) else float_hex(x, if (dt == "float64") "float64" else "float32")
    if (!is.finite(x) && !dt %in% c("float32", "float64"))
      jax_error("non-finite attributes of dtype ", dt, " are not supported", class = "jax_type_error")
    return(paste0(lit, " : ", ty))
  }
  if (k == "c") jax_error("Unsupported scalar attribute type: ", dt, class = "jax_type_error")
  paste0(int_literal(x), " : ", ty)
}

ffi_dense_attr <- function(data, shape, dt) {
  aval <- new_aval(as.integer(shape), dt)
  ty <- tensor_type(aval)
  flat <- c_ravel(data)
  if (!length(flat)) return(paste0("dense<> : ", ty))
  k <- dtype_kind(dt)
  vals <- if (k == "b") ifelse(as.logical(flat), "true", "false")
          else if (k == "f") vapply(flat, function(z) {
            if (is.finite(z)) sprintf("%.17e", z) else float_hex(z, if (dt == "float64") "float64" else "float32")
          }, "")
          else int_literal(flat)
  nest <- function(v, s) {
    if (length(s) <= 1L) return(paste0("[", paste(v, collapse = ", "), "]"))
    n <- prod(s[-1L])
    paste0("[", paste(vapply(seq_len(s[1L]), function(i) nest(v[(i - 1L) * n + seq_len(n)], s[-1L]), ""),
                      collapse = ", "), "]")
  }
  paste0("dense<", nest(vals, shape), "> : ", ty)
}
