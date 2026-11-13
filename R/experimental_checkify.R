# Runtime error checks: port of jax/_src/checkify.py (jax.experimental.checkify).
#
# A checked function is traced to a jaxpr, and the jaxpr is evaluated with a rule per
# primitive that threads a functional error value (`Error`) next to the results. Rules
# are the `checkify` field of the primitive objects (the counterpart of
# `checkify.error_checks[prim]`): `function(ctx, error, enabled, invals, params)`
# returning `list(error, outvals)`.

# ---- error categories -------------------------------------------------------------

#' @rdname experimental_checkify_checkify
#' @export
experimental_checkify_user_checks <- "FailedCheckError"
#' @rdname experimental_checkify_checkify
#' @export
experimental_checkify_nan_checks <- "NaNError"
#' @rdname experimental_checkify_checkify
#' @export
experimental_checkify_index_checks <- "OOBError"
#' @rdname experimental_checkify_checkify
#' @export
experimental_checkify_div_checks <- "DivisionByZeroError"
#' @rdname experimental_checkify_checkify
#' @export
experimental_checkify_float_checks <- c("NaNError", "DivisionByZeroError")
#' @rdname experimental_checkify_checkify
#' @export
experimental_checkify_automatic_checks <- c("NaNError", "DivisionByZeroError", "OOBError")
#' @rdname experimental_checkify_checkify
#' @export
experimental_checkify_all_checks <- c("NaNError", "DivisionByZeroError", "OOBError", "FailedCheckError")

#' @rdname experimental_checkify_checkify
#' @export
experimental_checkify_JaxRuntimeError <- "jax_checkify_runtime_error"

# ---- exceptions -------------------------------------------------------------------

# The counterpart of the JaxException classes: the category, the static description
# and the payload (arrays that are only known when the function runs).
ck_exception <- function(category, prim = NULL, operand_shape = NULL, fmt = NULL,
                         payload_tree = NULL, payload = list()) {
  structure(list(category = category, prim = prim, operand_shape = operand_shape, fmt = fmt,
                 payload_tree = payload_tree, payload = payload),
            class = "jax_checkify_exception")
}

# tree_flatten of an exception: the payload is the list of leaves, the rest is static.
ck_exc_metadata <- function(e) unclass(e)[c("category", "prim", "operand_shape", "fmt", "payload_tree")]
ck_exc_unflatten <- function(metadata, payload) do.call(ck_exception, c(metadata, list(payload = payload)))

# The effect of an exception (port of get_effect_type): the category with the types of
# the payload. Errors of one effect share one slot of the error value.
ck_effect <- function(category, payload) {
  paste0(category, "(", paste(vapply(payload, function(x) aval_str(strip_weak(aval_of(x))), ""),
                              collapse = ","), ")")
}
ck_effect_category <- function(effect) sub("\\(.*$", "", effect)

ck_host <- function(x) if (is_jax_value(x)) as.array(x) else x

# repr() of a Python float: the shortest decimal that reads back as the same double.
ck_py_float <- function(x) {
  if (is.nan(x)) return("nan")
  if (is.infinite(x)) return(if (x > 0) "inf" else "-inf")
  if (x == 0) return(if (identical(1 / x, -Inf)) "-0.0" else "0.0")
  for (d in 1:17) {
    s <- sprintf("%.*e", d - 1L, x)
    if (as.numeric(s) == x) break
  }
  e <- as.integer(sub("^.*e", "", s))
  if (e < -4L || e >= 16L) return(sub("e([+-])0*(\\d{2,})$", "e\\1\\2", sub("\\.?0*e", "e", s)))
  s <- format(x, scientific = FALSE, digits = d, trim = TRUE)
  if (grepl(".", s, fixed = TRUE)) s else paste0(s, ".0")
}

# str() of a value in a message, as Python formats it: a scalar is formatted as the
# Python number, an array like a NumPy array.
# ponytail: array elements are printed with up to 8 significant digits and padded to
# one width; NumPy aligns the decimal points and summarizes long arrays.
ck_str <- function(x, spec = "") {
  h <- ck_host(x)
  shape <- if (is_jax_value(x)) aval_of(x)$shape else r_shape(h)
  one <- function(v) {
    if (nzchar(spec)) {
      if (grepl("[dxXob]$", spec) && is.numeric(v)) v <- as.integer(v)
      return(sprintf(paste0("%", spec), v))
    }
    if (is.logical(v)) return(if (v) "True" else "False")
    if (is.complex(v)) return(format(v))
    if (is.integer(v) || (is_jax_value(x) && !is_inexact_dtype(aval_of(x)$dtype)))
      return(format(v, scientific = FALSE, trim = TRUE))
    if (length(shape)) {
      if (is.finite(v) && v == round(v) && abs(v) < 1e16) paste0(format(v, scientific = FALSE, trim = TRUE), ".")
      else format(v, digits = 8L, trim = TRUE)
    } else ck_py_float(v)
  }
  if (!length(shape)) return(one(as.vector(h)))
  v <- c_ravel(h)
  s <- formatC(vapply(seq_along(v), function(i) one(v[i]), ""), width = 0L)
  # NumPy prints booleans in the width of "False".
  s <- formatC(s, width = if (is.logical(v)) 5L else max(nchar(s)))
  nest <- function(s, shape) {
    if (length(shape) <= 1L) return(paste0("[", paste(s, collapse = " "), "]"))
    rows <- split(s, rep(seq_len(shape[1L]), each = prod_int(shape[-1L])))
    paste0("[", paste(vapply(rows, nest, "", shape = shape[-1L]),
                      collapse = paste0(strrep("\n", length(shape) - 1L), " ")), "]")
  }
  nest(s, shape)
}

# Port of str.format for the messages of `check`: `{}`, `{0}`, `{name}`, an optional
# conversion and a printf like format specification (`{x:.3f}`), `{{` and `}}`.
ck_format <- function(fmt, args, kwargs) {
  auto <- 0L
  field <- function(body) {
    spec <- if (grepl(":", body, fixed = TRUE)) sub("^[^:]*:", "", body) else ""
    name <- sub("![rsa]$", "", sub(":.*$", "", body))
    val <- if (!nzchar(name)) {
      auto <<- auto + 1L
      if (auto > length(args))
        jax_error("Replacement index ", auto - 1L, " out of range for positional args tuple",
                  class = "jax_index_error")
      args[[auto]]
    } else if (grepl("^[0-9]+$", name)) {
      i <- as.integer(name) + 1L
      if (i > length(args))
        jax_error("Replacement index ", i - 1L, " out of range for positional args tuple",
                  class = "jax_index_error")
      args[[i]]
    } else {
      if (is.null(kwargs[[name]])) jax_error("'", name, "'", class = "jax_key_error")
      kwargs[[name]]
    }
    ck_str(val, spec)
  }
  m <- gregexpr("\\{\\{|\\}\\}|\\{[^{}]*\\}", fmt)
  toks <- regmatches(fmt, m)[[1L]]
  regmatches(fmt, m) <- list(vapply(toks, function(tok) {
    switch(tok, "{{" = "{", "}}" = "}", field(substr(tok, 2L, nchar(tok) - 1L)))
  }, ""))
  fmt
}

ck_exc_message <- function(e) {
  e <- unclass(e)
  switch(e$category,
    DivisionByZeroError = "division by zero",
    NaNError = paste0("nan generated by primitive: ", e$prim, "."),
    OOBError = {
      p <- as.vector(ck_host(e$payload[[1L]]))
      paste0("out-of-bounds indexing for array of shape ", cb_py_tuple(e$operand_shape),
             ": index ", p[1L], " is out of bounds for axis ", p[2L], " with size ", p[3L], ". ")
    },
    FailedCheckError = {
      a <- tree_unflatten(e$payload_tree, e$payload)
      paste0(ck_format(e$fmt, a[[1L]], a[[2L]]), " (`check` failed)")
    },
    BatchedError = paste(vapply(e$payload, function(m) {
      paste0("at mapped index ", paste(m$index, collapse = ", "), ": ", ck_exc_message(m$error))
    }, ""), collapse = "\n"))
}

#' @export
format.jax_checkify_exception <- function(x, ...) ck_exc_message(x)
#' @export
print.jax_checkify_exception <- function(x, ...) {
  cat(.subset2(x, "category"), ": ", ck_exc_message(x), "\n", sep = "")
  invisible(x)
}

# ---- the error value ----------------------------------------------------------------

# Port of checkify.Error. `pred`, `code` and `payload` are named by effect; `metadata`
# maps a code to the static description of the exception with that code.
ck_new_error <- function(pred = list(), code = list(), metadata = list(), payload = list()) {
  o <- function(l) if (length(l)) l[order(names(l))] else list()
  structure(list(pred = o(pred), code = o(code), metadata = metadata, payload = o(payload)),
            class = "jax_checkify_error")
}

register_pytree_node("jax_checkify_error",
  function(x) {
    x <- unclass(x)
    list(children = list(x$pred, x$code, x$payload), aux = x$metadata)
  },
  function(aux, children) ck_new_error(children[[1L]], children[[2L]], aux, children[[3L]]))

#' @rdname experimental_checkify_checkify
#' @export
experimental_checkify_init_error <- ck_new_error()

#' @rdname experimental_checkify_checkify
#' @export
experimental_checkify_Error <- "jax_checkify_error"

ck_is_batched <- function(error) {
  any(vapply(unclass(error)$pred, function(p) length(aval_of(as_jax(p))$shape) > 0L, NA))
}

# Port of Error.get_exception.
ck_get_exception <- function(error) {
  if (ck_is_batched(error)) return(ck_batched_exception(error))
  e <- unclass(error)
  best <- NULL
  min_code <- Inf
  for (k in names(e$pred)) {
    if (isTRUE(as.logical(ck_host(e$pred[[k]])))) {
      code <- as.integer(ck_host(e$code[[k]]))
      if (code < min_code) {
        min_code <- code
        best <- k
      }
    }
  }
  if (is.null(best)) return(NULL)
  ck_exc_unflatten(e$metadata[[as.character(min_code)]], e$payload[[best]])
}

# Port of Error._get_batched_exception: one exception per mapped index with an error.
ck_batched_exception <- function(error) {
  e <- unclass(error)
  shape <- aval_of(as_jax(e$pred[[1L]]))$shape
  n <- prod_int(shape)
  flat <- function(x) c_ravel(ck_host(x))
  preds <- lapply(e$pred, flat)
  codes <- lapply(e$code, flat)
  mapping <- list()
  for (i in seq_len(n)) {
    best <- NULL
    min_code <- Inf
    for (k in names(preds)) {
      if (isTRUE(as.logical(preds[[k]][i])) && codes[[k]][i] < min_code) {
        min_code <- codes[[k]][i]
        best <- k
      }
    }
    if (is.null(best)) next
    payload <- lapply(e$payload[[best]], function(x) {
      rest <- drop_first(aval_of(as_jax(x))$shape, length(shape))
      m <- prod_int(rest)
      c_unravel(flat(x)[(i - 1L) * m + seq_len(m)], rest)
    })
    # np.ndindex order: the last index varies fastest.
    idx <- rev(as.vector(arrayInd(i, rev(shape)))) - 1L
    mapping[[length(mapping) + 1L]] <- list(
      index = idx, error = ck_exc_unflatten(e$metadata[[as.character(as.integer(min_code))]], payload))
  }
  if (length(mapping)) ck_exception("BatchedError", payload = mapping) else NULL
}

#' @export
`$.jax_checkify_error` <- function(x, name) {
  switch(name,
    get = function() {
      exc <- ck_get_exception(x)
      if (is.null(exc)) NULL else ck_exc_message(exc)
    },
    get_exception = function() ck_get_exception(x),
    throw = function() ck_check_error(x),
    .subset2(x, name))
}

#' @export
format.jax_checkify_error <- function(x, ...) {
  paste0("Error(", x$get() %||% "NULL", ")")
}
#' @export
print.jax_checkify_error <- function(x, ...) {
  cat(format(x), "\n", sep = "")
  invisible(x)
}

ck_replace_metadata <- function(error, metadata) {
  e <- unclass(error)
  ck_new_error(e$pred, e$code, metadata, e$payload)
}

# The effects of an error value with the types of their payloads.
ck_effects_of <- function(error) {
  lapply(unclass(error)$payload, function(p) lapply(p, function(x) strip_weak(aval_of(x))))
}

# Port of Error._add_placeholder_effects: slots that never fire, so that two error
# values (the branches of a cond, the carry of a loop) have the same structure.
ck_add_placeholders <- function(error, effects) {
  e <- unclass(error)
  for (k in sort(setdiff(names(effects), names(e$pred)))) {
    e$pred[[k]] <- as_jax(FALSE)
    e$payload[[k]] <- lapply(effects[[k]], function(a) jnp_ones(a$shape, a$dtype))
    e$code[[k]] <- ck_i32(-2L)
  }
  ck_new_error(e$pred, e$code, e$metadata, e$payload)
}

ck_i32 <- function(v) lax_convert_element_type(as_jax(v), "int32", weak_type = FALSE)

# Port of CheckifyContext: the counter of error codes. Codes order the checks of one
# checked function; the error with the smallest code is the one reported.
ck_ctx <- function(start = 0L) {
  ctx <- new.env(parent = emptyenv())
  ctx$code <- start
  ctx
}

# Port of update_error: the first error of an effect is kept.
ck_update_error <- function(error, pred, code, metadata, payload, effect) {
  e <- unclass(error)
  cur <- e$pred[[effect]]
  if (!is.null(cur)) {
    cur <- as_jax(cur)
    payload <- Map(function(old, new) lax_select(cur, old, new), e$payload[[effect]], payload)
    code <- lax_select(cur, e$code[[effect]], code)
    pred <- lax_bitwise_or(cur, pred)
  }
  e$pred[[effect]] <- pred
  e$code[[effect]] <- code
  e$payload[[effect]] <- payload
  e$metadata[names(metadata)] <- metadata
  ck_new_error(e$pred, e$code, e$metadata, e$payload)
}

# Port of assert_func: add a check with the next code.
ck_assert <- function(ctx, error, pred, exc) {
  code <- ctx$code
  ctx$code <- code + 1L
  ck_update_error(error, as_jax(pred), ck_i32(code),
                  stats::setNames(list(ck_exc_metadata(exc)), code),
                  .subset2(exc, "payload"), ck_effect(.subset2(exc, "category"), .subset2(exc, "payload")))
}

# ---- the interpreter ----------------------------------------------------------------

# Port of checkify_jaxpr_flat: evaluate a jaxpr with the rules. Returns list(error, outs).
ck_eval <- function(ctx, jaxpr, consts, enabled, error, args) {
  env <- vector("list", jaxpr$nvars)
  for (i in seq_along(jaxpr$constvars)) env[[jaxpr$constvars[[i]]$id]] <- consts[[i]]
  for (i in seq_along(jaxpr$invars)) env[[jaxpr$invars[[i]]$id]] <- args[[i]]
  rd <- function(a) if (is_lit(a)) a$val else env[[a$id]]
  for (e in jaxpr$eqns) {
    invals <- lapply(e$invars, rd)
    rule <- e$prim$checkify
    if (is.null(rule)) {
      out <- bind(e$prim, invals, e$params)
    } else {
      r <- rule(ctx, error, enabled, invals, e$params)
      error <- r[[1L]]
      out <- r[[2L]]
    }
    if (e$prim$multiple_results) {
      for (j in seq_along(e$outvars)) env[[e$outvars[[j]]$id]] <- out[[j]]
    } else {
      env[[e$outvars[[1L]]$id]] <- out
    }
  }
  list(error, lapply(jaxpr$outvars, rd))
}

ck_eval_closed <- function(ctx, cj, enabled, error, args) {
  ck_eval(ctx, cj$jaxpr, cj$consts, enabled, error, args)
}

# Port of jaxpr_to_checkify_jaxpr: the jaxpr of `fun(error, args) -> list(error, outs)`.
# Its inputs are the first `front` arguments, the leaves of the error, then the other
# arguments; its outputs are the leaves of the error followed by the outputs.
ck_trace <- function(fun, error, avals, front = 0L) {
  efl <- tree_flatten(error)
  ne <- length(efl$leaves)
  store <- new.env(parent = emptyenv())
  in_avals <- c(avals[seq_len(front)], lapply(efl$leaves, aval_of), drop_first(avals, front))
  cj <- trace_to_jaxpr(function(a) {
    err <- tree_unflatten(efl$treedef, a[front + seq_len(ne)])
    r <- fun(err, c(a[seq_len(front)], drop_first(a, front + ne)))
    fl <- tree_flatten(list(r[[1L]], r[[2L]]))
    store$out_tree <- fl$treedef
    store$effects <- ck_effects_of(r[[1L]])
    lapply(fl$leaves, as_jax)
  }, in_avals)
  list(cj = cj, out_tree = store$out_tree, effects = store$effects, err_vals = efl$leaves)
}

# The jaxpr of a function without arguments, with the equations whose results are not
# used: JAX checks them too (pe.trace_to_jaxpr removes no dead code).
ck_trace_all <- function(fun) {
  trace <- new_jaxpr_trace(current_trace())
  on.exit(trace$alive <- FALSE, add = TRUE)
  outs <- with_trace(trace, fun())
  outvars <- lapply(outs, function(o) to_atom(trace, o))
  closed_jaxpr(renumber_jaxpr(list(constvars = trace$constvars, invars = list(), outvars = outvars,
                                   eqns = trace$eqns[seq_len(trace$neqns)])), trace$consts)
}

ck_hoist <- function(cj) list(jaxpr = closed_jaxpr(convert_constvars(cj$jaxpr)), consts = cj$consts)

# ---- rules: values ------------------------------------------------------------------

# Port of check_nans.
ck_check_nans <- function(prim, ctx, error, enabled, out) {
  if (!"NaNError" %in% enabled) return(error)
  isnan <- function(x) jnp_any(jnp_isnan(x))
  any_nans <- if (prim$multiple_results) {
    if (length(out)) jnp_any(jnp_stack(lapply(out, isnan))) else as_jax(FALSE)
  } else isnan(out)
  ck_assert(ctx, error, any_nans, ck_exception("NaNError", prim = prim$name))
}

# Port of nan_error_check.
ck_nan_rule <- function(prim) {
  force(prim)
  function(ctx, error, enabled, invals, params) {
    out <- bind(prim, invals, params)
    list(ck_check_nans(prim, ctx, error, enabled, out), out)
  }
}

# All primitives which can generate a NaN (checkify.nan_primitives).
for (ck_p in list(acos_p, acosh_p, add_p, asin_p, asinh_p, atan2_p, atan_p, atanh_p, bessel_i0e_p,
                  bessel_i1e_p, cbrt_p, conv_general_dilated_p, cos_p, cosh_p, cumlogsumexp_p,
                  cummax_p, cummin_p, cumprod_p, cumsum_p, digamma_p, dot_general_p, erf_inv_p,
                  erf_p, erfc_p, exp_p, exp2_p, expm1_p, fft_p, igamma_grad_a_p, igamma_p,
                  igammac_p, integer_pow_p, lgamma_p, linear_solve_p, log1p_p, log2_p, log_p,
                  logistic_p, mul_p, pad_p, pow_p, psum_p, reduce_p, reduce_prod_p, reduce_sum_p,
                  reduce_window_p, reduce_window_sum_p, regularized_incomplete_beta_p, rem_p,
                  rng_uniform_p, rsqrt_p, sin_p, sinh_p, sqrt_p, sub_p, tan_p, tanh_p)) {
  ck_p$checkify <- ck_nan_rule(ck_p)
}
rm(ck_p)

# Port of div_error_check: division by zero, then NaN.
div_p$checkify <- function(ctx, error, enabled, invals, params) {
  if ("DivisionByZeroError" %in% enabled) {
    any_zero <- jnp_any(jnp_equal(invals[[2L]], 0L))
    error <- ck_assert(ctx, error, any_zero, ck_exception("DivisionByZeroError"))
  }
  ck_nan_rule(div_p)(ctx, error, enabled, invals, params)
}

# Port of oob_payload: the first index out of bounds, its axis and the size of the axis.
ck_oob_payload <- function(oob_mask, indices, dims_map, operand_shape) {
  flat_idx <- jnp_argmin(jnp_logical_not(oob_mask))
  multi_idx <- jnp_unravel_index(flat_idx, aval_of(indices)$shape)
  oob_axis <- jnp_array(as.integer(dims_map), ndmin = 1L)[multi_idx[[length(multi_idx)]]]
  oob_axis_size <- jnp_array(as.integer(operand_shape), ndmin = 1L)[oob_axis]
  oob_index <- jnp_ravel(indices)[flat_idx]
  jnp_stack(lapply(list(oob_index, oob_axis, oob_axis_size), jnp_astype, "int32"))
}

ck_oob_assert <- function(ctx, error, oob_mask, indices, dims_map, operand_shape, name) {
  payload <- ck_oob_payload(oob_mask, indices, dims_map, operand_shape)
  ck_assert(ctx, error, jnp_any(oob_mask),
            ck_exception("OOBError", prim = name, operand_shape = operand_shape, payload = list(payload)))
}

# Port of dynamic_slice_error_check.
dynamic_slice_p$checkify <- function(ctx, error, enabled, invals, params) {
  out <- bind(dynamic_slice_p, invals, params)
  if (!"OOBError" %in% enabled) return(list(error, out))
  shape <- aval_of(invals[[1L]])$shape
  start <- jnp_stack(invals[-1L])
  dt <- aval_of(start)$dtype
  dims <- jnp_array(shape, dtype = dt, ndmin = 1L)
  sizes <- jnp_array(as.integer(params$slice_sizes), dtype = dt, ndmin = 1L)
  oob_mask <- jnp_logical_or(start < 0L, start + sizes > dims)
  list(ck_oob_assert(ctx, error, oob_mask, start, seq_along(shape) - 1L, shape, "dynamic_slice"), out)
}

# Port of dynamic_update_slice_error_check.
dynamic_update_slice_p$checkify <- function(ctx, error, enabled, invals, params) {
  out <- bind(dynamic_update_slice_p, invals, params)
  if (!"OOBError" %in% enabled) return(list(error, out))
  shape <- aval_of(invals[[1L]])$shape
  start <- jnp_stack(invals[-(1:2)])
  dt <- aval_of(start)$dtype
  dims <- jnp_array(shape, dtype = dt, ndmin = 1L)
  upd <- jnp_array(aval_of(invals[[2L]])$shape, dtype = dt, ndmin = 1L)
  oob_mask <- jnp_logical_or(start < 0L, start + upd > dims)
  list(ck_oob_assert(ctx, error, oob_mask, start, seq_along(shape) - 1L, shape, "dynamic_update_slice"), out)
}

# Port of gather_error_check.
gather_p$checkify <- function(ctx, error, enabled, invals, params) {
  out <- bind(gather_p, invals, params)
  if (!"OOBError" %in% enabled) return(list(error, out))
  operand <- invals[[1L]]
  start <- invals[[2L]]
  shape <- aval_of(operand)$shape
  sim <- params$dimension_numbers$start_index_map
  upper <- shape[sim + 1L] - as.integer(params$slice_sizes)[sim + 1L]
  upper <- jnp_array(upper, dtype = aval_of(start)$dtype, ndmin = 1L)
  oob_mask <- jnp_logical_or(start < 0L, start > upper)
  list(ck_oob_assert(ctx, error, oob_mask, start, sim, shape, "gather"), out)
}

# Port of scatter_oob.
ck_scatter_oob <- function(operand, indices, updates, dnums) {
  shape <- aval_of(operand)$shape
  ushape <- aval_of(updates)$shape
  slice_sizes <- integer(0)
  pos <- 0L
  for (i in seq_along(shape) - 1L) {
    if (i %in% dnums$inserted_window_dims) {
      slice_sizes <- c(slice_sizes, 1L)
    } else {
      pos <- pos + 1L
      slice_sizes <- c(slice_sizes, ushape[dnums$update_window_dims[pos] + 1L])
    }
  }
  sd <- dnums$scatter_dims_to_operand_dims
  ia <- aval_of(indices)
  upper <- pmin(shape[sd + 1L] - slice_sizes[sd + 1L], iinfo(ia$dtype)$max)
  upper <- jnp_array(upper, dtype = ia$dtype, ndmin = 1L)
  oob_mask <- jnp_logical_or(indices < 0L, indices > upper)
  list(jnp_any(oob_mask), ck_oob_payload(oob_mask, indices, sd, shape))
}

# Port of scatter_error_check: out of bounds indices, then NaN in the result.
ck_scatter_rule <- function(prim) {
  force(prim)
  function(ctx, error, enabled, invals, params) {
    out <- bind(prim, invals, params)
    if (!"OOBError" %in% enabled) return(list(error, out))
    shape <- aval_of(invals[[1L]])$shape
    oob <- ck_scatter_oob(invals[[1L]], invals[[2L]], invals[[3L]], params$dimension_numbers)
    error <- ck_assert(ctx, error, oob[[1L]],
                       ck_exception("OOBError", prim = prim$name, operand_shape = shape,
                                    payload = list(oob[[2L]])))
    list(ck_check_nans(prim, ctx, error, enabled, out), out)
  }
}
for (ck_p in list(scatter_p, scatter_add_p, scatter_mul_p, scatter_min_p, scatter_max_p)) {
  ck_p$checkify <- ck_scatter_rule(ck_p)
}
rm(ck_p)

# ---- rules: control flow --------------------------------------------------------------

# Port of cond_error_check. The error enters every branch with the effects of all
# branches, so the branches return error values of one structure.
cond_p$checkify <- function(ctx, error, enabled, invals, params) {
  index <- invals[[1L]]
  ops <- invals[-1L]
  op_avals <- lapply(ops, aval_of)
  run <- function(ctx, b) function(err, a) ck_eval_closed(ctx, b, enabled, err, a)
  effects <- list()
  for (b in params$branches) {
    new <- ck_trace(run(ck_ctx(), b), error, op_avals)$effects
    effects[names(new)] <- new
  }
  merged <- ck_add_placeholders(error, effects)
  traced <- lapply(params$branches, function(b) ck_trace(run(ctx, b), merged, op_avals))
  jc <- join_consts(lapply(traced, `[[`, "cj"))
  out <- bind(cond_p, c(list(index), jc$consts, traced[[1L]]$err_vals, ops), list(branches = jc$jaxprs))
  res <- tree_unflatten(traced[[1L]]$out_tree, out)
  metadata <- unclass(res[[1L]])$metadata
  for (t in traced[-1L]) {
    m <- unclass(tree_unflatten(t$out_tree, out)[[1L]])$metadata
    metadata[names(m)] <- m
  }
  list(ck_replace_metadata(res[[1L]], metadata), res[[2L]])
}

# Port of scan_error_check: the error is one more carry.
scan_p$checkify <- function(ctx, error, enabled, invals, params) {
  nc <- params$num_consts
  nk <- params$num_carry
  consts <- invals[seq_len(nc)]
  carry <- invals[nc + seq_len(nk)]
  xs <- drop_first(invals, nc + nk)
  avals <- c(lapply(c(consts, carry), aval_of), lapply(xs, function(x) {
    a <- aval_of(x)
    new_aval(a$shape[-1L], a$dtype, a$weak)
  }))
  run <- function(ctx) function(err, a) ck_eval_closed(ctx, params$jaxpr, enabled, err, a)
  merged <- ck_add_placeholders(error, ck_trace(run(ck_ctx()), error, avals, front = nc)$effects)
  t <- ck_trace(run(ctx), merged, avals, front = nc)
  h <- ck_hoist(t$cj)
  out <- bind(scan_p, c(h$consts, consts, t$err_vals, carry, xs),
              update_params(params, list(jaxpr = h$jaxpr, num_consts = length(h$consts) + nc,
                                         num_carry = length(t$err_vals) + nk)))
  tree_unflatten(t$out_tree, out)
}

# Port of while_loop_error_check. The first test of the condition is checked before the
# loop; the body checks itself and the test that follows it.
while_p$checkify <- function(ctx, error, enabled, invals, params) {
  cj <- params$cond_jaxpr
  bj <- params$body_jaxpr
  # A while loop with a batched predicate keeps a scalar test in jaxR (the any() of the
  # predicates); the batching rule marks its test.
  if (length(jaxpr_out_avals(cj$jaxpr)[[1L]]$shape) || isTRUE(cj$batched_predicate))
    jax_error("Checkify does not support batched while-loops (checkify-of-vmap-of-while). \n",
              "Hint: if possible, move the vmap to the outer level to get ",
              "vmap-of-checkify-of-while.", class = "jax_value_error")
  nc <- params$cond_nconsts
  nb <- params$body_nconsts
  cc <- invals[seq_len(nc)]
  bc <- invals[nc + seq_len(nb)]
  carry <- drop_first(invals, nc + nb)
  error <- ck_eval_closed(ctx, cj, enabled, error, c(cc, carry))[[1L]]
  avals <- lapply(c(cc, bc, carry), aval_of)
  body <- function(ctx) function(err, a) {
    r <- ck_eval_closed(ctx, bj, enabled, err, drop_first(a, nc))
    r2 <- ck_eval_closed(ctx, cj, enabled, r[[1L]], c(a[seq_len(nc)], r[[2L]]))
    list(r2[[1L]], r[[2L]])
  }
  error <- ck_add_placeholders(error, ck_trace(body(ck_ctx()), error, avals, front = nc + nb)$effects)
  t <- ck_trace(body(ctx), error, avals, front = nc + nb)
  ne <- length(t$err_vals)
  hb <- ck_hoist(t$cj)
  # The loop test itself ignores the error values.
  hc <- ck_hoist(trace_to_jaxpr(
    function(a) eval_closed(cj, c(a[seq_len(nc)], drop_first(a, nc + ne))),
    c(avals[seq_len(nc)], lapply(t$err_vals, aval_of), drop_first(avals, nc + nb))))
  out <- bind(while_p, c(hc$consts, cc, hb$consts, cc, bc, t$err_vals, carry),
              list(cond_jaxpr = hc$jaxpr, body_jaxpr = hb$jaxpr,
                   cond_nconsts = length(hc$consts) + nc,
                   body_nconsts = length(hb$consts) + nc + nb))
  tree_unflatten(t$out_tree, out)
}

# Port of remat_error_check.
remat_p$checkify <- function(ctx, error, enabled, invals, params) {
  t <- ck_trace(function(err, a) ck_eval_closed(ctx, params$jaxpr, enabled, err, a),
                error, lapply(invals, aval_of))
  h <- ck_hoist(t$cj)
  out <- bind(remat_p, c(h$consts, t$err_vals, invals), update_params(params, list(jaxpr = h$jaxpr)))
  tree_unflatten(t$out_tree, out)
}

# The checked function of a custom derivative call. Every run numbers its checks from
# the same code, because the function is traced and may also be run.
ck_custom_fun <- function(ctx, error, enabled, call_jaxpr) {
  efl <- tree_flatten(error)
  ne <- length(efl$leaves)
  st <- new.env(parent = emptyenv())
  st$start <- ctx$code
  st$ne <- ne
  st$err_vals <- efl$leaves
  st$err_tree <- efl$treedef
  st$rule_ran <- FALSE
  st$fun <- function(args) {
    sub <- ck_ctx(st$start)
    r <- ck_eval_closed(sub, call_jaxpr, enabled, tree_unflatten(efl$treedef, args[seq_len(ne)]),
                        drop_first(args, ne))
    st$ncodes <- sub$code - st$start
    fl <- tree_flatten(list(r[[1L]], r[[2L]]))
    st$out_tree <- fl$treedef
    lapply(fl$leaves, as_jax)
  }
  st
}

# Port of custom_jvp_call_rule. The function is checked; its derivative rule is not
# (use checkify of jvp for that) and passes the error it receives through.
custom_jvp_call_p$checkify <- function(ctx, error, enabled, invals, params) {
  st <- ck_custom_fun(ctx, error, enabled, params$call_jaxpr)
  ne <- st$ne
  jvp <- function(primals, tangents) {
    st$rule_ran <- TRUE
    out <- params$jvp(drop_first(primals, ne), drop_first(tangents, ne))
    n <- length(out) %/% 2L
    c(primals[seq_len(ne)], out[seq_len(n)], tangents[seq_len(ne)], out[n + seq_len(n)])
  }
  cj <- trace_to_jaxpr(st$fun, lapply(c(st$err_vals, invals), aval_of))
  out <- bind(custom_jvp_call_p, c(st$err_vals, invals),
              update_params(params, list(fun = st$fun, jvp = jvp, call_jaxpr = cj)))
  ctx$code <- st$start + st$ncodes
  if (st$rule_ran) return(list(tree_unflatten(st$err_tree, out[seq_len(ne)]), drop_first(out, ne)))
  tree_unflatten(st$out_tree, out)
}

# Port of custom_vjp_call_rule.
custom_vjp_call_p$checkify <- function(ctx, error, enabled, invals, params) {
  st <- ck_custom_fun(ctx, error, enabled, params$call_jaxpr)
  ne <- st$ne
  fwd <- function(primals, nz) {
    st$rule_ran <- TRUE
    params$fwd(drop_first(primals, ne), drop_first(nz, ne))
  }
  bwd <- function(res, cts, state) c(rep(list(NULL), ne), params$bwd(res, cts, state))
  cj <- trace_to_jaxpr(st$fun, lapply(c(st$err_vals, invals), aval_of))
  out <- bind(custom_vjp_call_p, c(st$err_vals, invals),
              update_params(params, list(fun = st$fun, fwd = fwd, bwd = bwd, call_jaxpr = cj)))
  ctx$code <- st$start + st$ncodes
  if (st$rule_ran) return(list(error, out))
  tree_unflatten(st$out_tree, out)
}

# ---- the check primitive ----------------------------------------------------------------

ck_functionalization_error <- function() {
  jax_error("Cannot abstractly evaluate a checkify.check which was not functionalized. This ",
            "probably means you tried to stage (jit/scan/pmap/...) a `check` without ",
            "functionalizing it through `checkify.checkify`.", class = "jax_value_error")
}

ck_raise <- function(error) {
  exc <- ck_get_exception(error)
  if (!is.null(exc))
    jax_error(ck_exc_message(exc), class = c("jax_checkify_runtime_error", "jax_value_error"))
  invisible(NULL)
}

check_p <- new_primitive("check", multiple_results = TRUE)
check_p$effectful <- TRUE
check_p$abstract_eval <- function(avals, params) list()
# Port of check_impl. A debug check only runs when it is discharged by checkify.
check_p$impl <- function(args, params) {
  if (!isTRUE(params$debug)) ck_raise(tree_unflatten(params$err_tree, args))
  list()
}
# Port of check_lowering_rule: with jax_experimental_unsafe_xla_runtime_errors the
# compiled code calls back to raise the error.
check_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  if (isTRUE(params$debug)) return(character(0))
  if (!isTRUE(.jx$config$jax_experimental_unsafe_xla_runtime_errors)) ck_functionalization_error()
  cb_check_backend()
  spec <- list(kind = "io", in_tree = tree_structure(as.list(seq_along(ins))),
               fun = function(...) {
                 ck_raise(tree_unflatten(params$err_tree, list(...)))
                 list()
               })
  cb_emit(ctx, spec, ins, avals_in, list(), has_side_effect = TRUE)
}
# Port of check_batching_rule.
check_p$batch <- function(vals, dims, params, axis) {
  vals <- Map(function(v, d) bdim_at_front(v, d, axis$size), vals, as.list(dims))
  ck_check_error(tree_unflatten(params$err_tree, vals), debug = isTRUE(params$debug))
  list(list(), integer(0))
}
# Port of check_jvp_rule: check the primals, discard the tangents.
check_p$jvp <- function(primals, tangents, params) {
  bind(check_p, primals, params)
  list(list(), list())
}

# Port of check_discharge_rule: a check met by checkify joins the error value. Its codes
# are local (a `check` has the code 0, an inner checkify counts from 0) and move to a
# free range of the codes of the outer function.
check_p$checkify <- function(ctx, error, enabled, invals, params) {
  new <- unclass(tree_unflatten(params$err_tree, invals))
  codes <- as.integer(names(new$metadata))
  offset <- ctx$code
  ctx$code <- offset + if (length(codes)) max(codes) + 1L else 0L
  if (offset) {
    new$code <- lapply(new$code, function(c) lax_add(c, lax_full_like(c, offset)))
    names(new$metadata) <- as.character(codes + offset)
  }
  for (k in names(new$pred)) {
    # An effect that is not enabled is dropped, as in JAX.
    if (ck_effect_category(k) %in% enabled)
      error <- ck_update_error(error, as_jax(new$pred[[k]]), new$code[[k]], new$metadata,
                               new$payload[[k]], k)
  }
  list(error, list())
}

# Port of _reduce_any_error: one element of a batch of errors that failed, if any.
ck_reduce_any <- function(error) {
  e <- unclass(error)
  out <- list(pred = list(), code = list(), payload = list())
  for (k in names(e$pred)) {
    idx <- jnp_argsort(e$pred[[k]])[-1L]
    out$pred[[k]] <- e$pred[[k]][idx]
    out$code[[k]] <- e$code[[k]][idx]
    out$payload[[k]] <- lapply(e$payload[[k]], function(x) x[idx])
  }
  ck_new_error(out$pred, out$code, e$metadata, out$payload)
}

# Port of _check_error.
ck_check_error <- function(error, debug = FALSE) {
  if (ck_is_batched(error)) error <- ck_reduce_any(error)
  fl <- tree_flatten(error)
  bind(check_p, lapply(fl$leaves, as_jax), list(err_tree = fl$treedef, debug = debug))
  invisible(NULL)
}

# Port of _check.
ck_check <- function(pred, msg, debug, fmt) {
  scalar_pred <- if (is_jax_value(pred)) {
    !length(aval_of(pred)$shape) && aval_of(pred)$dtype == "bool"
  } else is.logical(pred) && length(pred) == 1L && is.null(dim(pred)) && !is.na(pred)
  if (!scalar_pred)
    jax_error(if (debug) "debug_check" else "check", " takes a scalar pred as argument, got ",
              paste(format(pred), collapse = " "), class = "jax_type_error")
  nms <- names(fmt) %||% rep("", length(fmt))
  fl <- tree_flatten(list(unname(fmt[!nzchar(nms)]), fmt[nzchar(nms)]))
  for (leaf in fl$leaves) {
    if (!(is_jax_value(leaf) || is.numeric(leaf) || is.logical(leaf) || is.complex(leaf)))
      jax_error("Formatting arguments to checkify.check need to be PyTrees of arrays, but got ",
                paste(format(leaf), collapse = " "), " of type ", class(leaf)[1L], ".",
                class = "jax_type_error")
  }
  leaves <- lapply(fl$leaves, as_jax)
  exc <- ck_exception("FailedCheckError", fmt = msg, payload_tree = fl$treedef, payload = leaves)
  k <- ck_effect("FailedCheckError", leaves)
  one <- function(v) stats::setNames(list(v), k)
  error <- ck_new_error(one(jnp_logical_not(pred)), one(ck_i32(0L)),
                        list("0" = ck_exc_metadata(exc)), one(leaves))
  ck_check_error(error, debug = debug)
}

# ---- public functions -------------------------------------------------------------------

#' Runtime error checks that work under jit, vmap and control flow
#'
#' Port of `jax.experimental.checkify`. A compiled function cannot raise an R error
#' that depends on the values it computes. `experimental_checkify_checkify()` turns
#' such checks into data: the checked function returns an error value next to its
#' result, and the error is raised (or inspected) outside the compiled code.
#'
#' * `experimental_checkify_check(pred, msg, ...)` adds a check to a function: an error
#'   with the message `msg` if `pred` is `FALSE`. Called outside any transformation it
#'   raises at once. Inside `jit()`, `lax_scan()` and the other staging functions it
#'   must be functionalized by `experimental_checkify_checkify()` first.
#' * `experimental_checkify_debug_check()` is a check that only runs under
#'   `experimental_checkify_checkify()` and is dropped otherwise.
#' * `experimental_checkify_checkify(f, errors)` returns a function with the arguments
#'   of `f` that returns `list(error, value)`. `errors` selects the checks: the user
#'   checks (the default) and the automatic checks listed below. Sets are combined
#'   with `c()`.
#' * `experimental_checkify_check_error(error)` raises the error of an error value. It
#'   is itself a check, so it can be functionalized again by an outer
#'   `experimental_checkify_checkify()`.
#'
#' The error value (`Error` in JAX) is a pytree, so it passes through `jit()` and
#' `vmap()`. Its methods are `error$get()` (the message of the first error, or `NULL`),
#' `error$get_exception()` and `error$throw()` (raise the error, like
#' `experimental_checkify_check_error()`). After `vmap()` the message lists the mapped
#' indices that failed. `experimental_checkify_init_error` is the empty error value.
#'
#' The sets of checks:
#'
#' | set | checks |
#' |---|---|
#' | `experimental_checkify_user_checks` | calls of `experimental_checkify_check()` |
#' | `experimental_checkify_nan_checks` | a floating point operation produced a NaN |
#' | `experimental_checkify_div_checks` | a division by zero |
#' | `experimental_checkify_index_checks` | an index out of bounds |
#' | `experimental_checkify_float_checks` | NaN and division checks |
#' | `experimental_checkify_automatic_checks` | float and index checks |
#' | `experimental_checkify_all_checks` | automatic and user checks |
#'
#' Raised errors have the class `experimental_checkify_JaxRuntimeError`
#' (`"jax_checkify_runtime_error"`, a `jax_value_error`). Error values have the class
#' `experimental_checkify_Error` (`"jax_checkify_error"`).
#'
#' With `jax_config_update("jax_experimental_unsafe_xla_runtime_errors", TRUE)` a check
#' that was not functionalized is compiled to a host callback that raises the error
#' while the compiled code runs.
#'
#' Messages are format strings in the style of Python: `{}` and `{0}` take the unnamed
#' arguments in `...`, `{name}` a named one, and a printf like specification may
#' follow a colon (`{x:.3f}`). The values may be traced arrays; they are formatted by R
#' when the message is built.
#' @param f Function to check.
#' @param errors Character vector of error categories, one of the sets above or several
#'   combined with `c()`.
#' @return `experimental_checkify_checkify()`: a function returning
#'   `list(error, value)`. The check functions return `NULL` invisibly.
#' @examples
#' \dontrun{
#' f <- function(x) {
#'   experimental_checkify_check(x > 0, "{x} needs to be positive!", x = x)
#'   1 / x
#' }
#' checked <- jit(experimental_checkify_checkify(f))
#' res <- checked(as_jax(-3))
#' res[[1]]$get()
#'
#' g <- function(x) x + sin(x)
#' res <- experimental_checkify_checkify(g, errors = experimental_checkify_float_checks)(Inf)
#' res[[1]]$get()
#' }
#' @export
experimental_checkify_checkify <- function(f, errors = experimental_checkify_user_checks) {
  bad <- setdiff(errors, experimental_checkify_all_checks)
  if (!is.character(errors) || length(bad))
    jax_error("errors must be a set of error categories (", paste(experimental_checkify_all_checks, collapse = ", "),
              "), got ", paste(format(errors), collapse = ", "), class = "jax_type_error")
  function(...) {
    # The arguments are closed over, so they are not turned into abstract values.
    args <- list(...)
    store <- new.env(parent = emptyenv())
    cj <- ck_trace_all(function() {
      fl <- tree_flatten(do.call(f, args))
      store$tree <- fl$treedef
      lapply(fl$leaves, as_jax)
    })
    r <- ck_eval_closed(ck_ctx(), cj, errors, ck_new_error(), list())
    list(r[[1L]], tree_unflatten(store$tree, r[[2L]]))
  }
}

#' @rdname experimental_checkify_checkify
#' @param pred Scalar boolean: the check fails if it is `FALSE`.
#' @param msg Message of the error, a format string.
#' @param ... Formatting arguments of `msg`: arrays or pytrees of arrays, unnamed for
#'   `{}` and `{0}`, named for `{name}`.
#' @param debug If `TRUE` the check only runs under `experimental_checkify_checkify()`.
#' @export
experimental_checkify_check <- function(pred, msg, ..., debug = FALSE) {
  ck_check(pred, msg, isTRUE(debug), list(...))
}

#' @rdname experimental_checkify_checkify
#' @export
experimental_checkify_debug_check <- function(pred, msg, ...) ck_check(pred, msg, TRUE, list(...))

#' @rdname experimental_checkify_checkify
#' @param error An error value returned by a checked function.
#' @export
experimental_checkify_check_error <- function(error) {
  if (!inherits(error, "jax_checkify_error"))
    jax_error("check_error takes an Error as argument, got type ", class(error)[1L], " instead.",
              class = "jax_type_error")
  ck_check_error(error, debug = FALSE)
}
