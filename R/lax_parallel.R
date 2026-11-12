# Named axis collectives (port of jax/_src/lax/parallel.py, vmap semantics).
#
# On one device collectives are meaningful under vmap(..., axis_name = name): each
# primitive's batching rule handles the axis it names and forwards the others outward.

find_axis_trace <- function(name, trace = current_trace()) {
  t <- trace
  while (!is.null(t)) {
    if (identical(t$kind, "batch") && identical(t$axis_name, name)) return(t)
    t <- t$parent
  }
  NULL
}

axis_size_of <- function(name) {
  t <- find_axis_trace(name)
  if (is.null(t)) jax_error("unbound axis name: ", name, class = "jax_name_error")
  t$axis_size
}

unbound_axis <- function(trace, args, params) {
  ax <- params$axes %||% params$axis_name
  jax_error("unbound axis name: ", paste(ax, collapse = ", "),
            ". The axis must be bound by an enclosing vmap(..., axis_name = ...).",
            class = "jax_name_error")
}

# Generic batch override for collectives: forward when the axis is not ours.
collective_batch <- function(prim, own_rule, unmapped_rule) {
  function(trace, args, params) {
    names <- params$axes %||% params$axis_name
    v <- args[[1L]]
    own <- is_own(v, trace)
    if (!(identical(trace$axis_name, names[1L]) || trace$axis_name %in% names)) {
      if (!own) return(with_trace(trace$parent, bind(prim, args, params)))
      out <- with_trace(trace$parent, bind(prim, c(list(v$val), args[-1L]), params))
      return(batch_tracer(trace, out, v$bdim))
    }
    with_trace(trace$parent, {
      if (own) own_rule(trace, v$val, v$bdim, params) else unmapped_rule(trace, v, params)
    })
  }
}

remaining_axes <- function(trace, params) setdiff(params$axes, trace$axis_name)

rebind_rest <- function(prim, x, rest, params) {
  if (!length(rest)) return(x)
  bind(prim, list(x), update_params(params, list(axes = rest)))
}

def_reduction_collective <- function(name, reducer, const_rule) {
  p <- new_primitive(name)
  p$abstract_eval <- function(avals, params) avals[[1L]]
  p$override$eval <- unbound_axis
  p$override$batch <- collective_batch(p,
    function(trace, val, d, params) rebind_rest(p, reducer(val, d), remaining_axes(trace, params), params),
    function(trace, v, params) rebind_rest(p, const_rule(v, trace$axis_size), remaining_axes(trace, params), params))
  p
}

psum_p <- def_reduction_collective("psum", function(v, d) lax_reduce_sum(v, d),
                                   function(v, n) lax_mul(v, lax_full_like(v, n)))
pmax_p <- def_reduction_collective("pmax", function(v, d) lax_reduce_max(v, d), function(v, n) v)
pmin_p <- def_reduction_collective("pmin", function(v, d) lax_reduce_min(v, d), function(v, n) v)
# Transpose of psum is psum (cotangents are per lane).
deflinear(psum_p, function(ct, args, params) list(bind(psum_p, list(ct), params)))

collective_names <- function(axis_name) {
  if (is.list(axis_name)) unlist(axis_name) else as.character(axis_name)
}

#' Named axis collectives
#'
#' Ports of `jax.lax.psum`, `pmean`, `pmax`, `pmin`, `axis_index`, `all_gather`,
#' `ppermute`, `pshuffle`, `pbroadcast`, `all_to_all`, `pswapaxes` and `psum_scatter`.
#' On one device they apply to axes named by an enclosing
#' `vmap(..., axis_name = name)`.
#' @param x A pytree of arrays.
#' @param axis_name Axis name (or several names for reductions).
#' @param axis_index_groups Not supported (must be `NULL`).
#' @name lax_collectives
NULL

collective_tree <- function(prim, x, params) {
  tree_map(function(l) bind(prim, list(as_jax(l)), params), x)
}

# Collectives run under vmap here, where JAX does not implement index groups either.
check_no_index_groups <- function(axis_index_groups) {
  if (!is.null(axis_index_groups))
    jax_error("axis_index_groups not supported in vmap collectives. Please open a feature ",
              "request!", class = "jax_not_implemented_error")
}

#' @rdname lax_collectives
#' @export
lax_psum <- function(x, axis_name, axis_index_groups = NULL) {
  check_no_index_groups(axis_index_groups)
  # Booleans are counted, as in JAX.
  x <- tree_map(function(l) {
    l <- as_jax(l)
    if (aval_of(l)$dtype == "bool") lax_convert_element_type(l, "int32") else l
  }, x)
  collective_tree(psum_p, x, list(axes = collective_names(axis_name)))
}
#' @rdname lax_collectives
#' @export
lax_pmax <- function(x, axis_name, axis_index_groups = NULL) {
  check_no_index_groups(axis_index_groups)
  collective_tree(pmax_p, x, list(axes = collective_names(axis_name)))
}
#' @rdname lax_collectives
#' @export
lax_pmin <- function(x, axis_name, axis_index_groups = NULL) {
  check_no_index_groups(axis_index_groups)
  collective_tree(pmin_p, x, list(axes = collective_names(axis_name)))
}
#' @rdname lax_collectives
#' @export
lax_pmean <- function(x, axis_name, axis_index_groups = NULL) {
  check_no_index_groups(axis_index_groups)
  names <- collective_names(axis_name)
  n <- prod(vapply(names, axis_size_of, 1L))
  # True division: an integer mean is a float.
  tree_map(function(s) jnp_true_divide(s, n), lax_psum(x, names))
}

# ---- axis_index -----------------------------------------------------------------------------

axis_index_p <- new_primitive("axis_index")
axis_index_p$abstract_eval <- function(avals, params) new_aval(integer(0), "int32")
axis_index_p$override$eval <- unbound_axis
axis_index_p$override$batch <- function(trace, args, params) {
  if (!identical(trace$axis_name, params$axis_name))
    return(with_trace(trace$parent, bind(axis_index_p, args, params)))
  batch_tracer(trace, with_trace(trace$parent, lax_iota("int32", trace$axis_size)), 0L)
}

#' @rdname lax_collectives
#' @export
lax_axis_index <- function(axis_name) bind(axis_index_p, list(), list(axis_name = axis_name))

#' @rdname lax_collectives
#' @export
lax_axis_size <- function(axis_name) axis_size_of(axis_name)

# ---- all_gather / psum_scatter -----------------------------------------------------------------

all_gather_p <- new_primitive("all_gather")
all_gather_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  n <- axis_size_of(params$axis_name)
  s <- a$shape
  if (isTRUE(params$tiled)) s[params$axis + 1L] <- s[params$axis + 1L] * n
  else s <- append(s, n, after = params$axis)
  new_aval(s, a$dtype, a$weak)
}
all_gather_p$override$eval <- unbound_axis
all_gather_p$override$batch <- function(trace, args, params) {
  v <- args[[1L]]
  own <- is_own(v, trace)
  if (!identical(trace$axis_name, params$axis_name)) {
    if (!own) return(with_trace(trace$parent, bind(all_gather_p, args, params)))
    d <- v$bdim
    out <- with_trace(trace$parent, bind(all_gather_p, list(v$val),
      update_params(params, list(axis = params$axis + (params$axis >= d)))))
    return(batch_tracer(trace, out, d + (!isTRUE(params$tiled) && params$axis <= d)))
  }
  with_trace(trace$parent, {
    n <- trace$axis_size
    full <- if (own) moveaxis(v$val, v$bdim, params$axis) else {
      moveaxis(lax_broadcast(v, n), 0L, params$axis)
    }
    if (isTRUE(params$tiled)) {
      s <- aval_of(full)$shape
      ax <- params$axis
      lax_reshape(full, c(s[seq_len(ax)], s[ax + 1L] * s[ax + 2L], drop_first(s, ax + 2L)))
    } else full
  })
}

#' @rdname lax_collectives
#' @param axis 0-based position of the gathered axis.
#' @param tiled Concatenate instead of stacking.
#' @export
lax_all_gather <- function(x, axis_name, axis_index_groups = NULL, axis = 0L, tiled = FALSE) {
  check_no_index_groups(axis_index_groups)
  tree_map(function(l) {
    l <- as_jax(l)
    r <- length(aval_of(l)$shape) + if (tiled) 0L else 1L
    bind(all_gather_p, list(l), list(axis_name = axis_name, axis = norm_axis(axis, r),
                                     tiled = isTRUE(tiled)))
  }, x)
}

#' @rdname lax_collectives
#' @param scatter_dimension 0-based dimension split across the axis.
#' @export
lax_psum_scatter <- function(x, axis_name, scatter_dimension = 0L, axis_index_groups = NULL,
                             tiled = FALSE) {
  check_no_index_groups(axis_index_groups)
  tree_map(function(l) {
    s <- lax_psum(l, axis_name)
    n <- axis_size_of(axis_name)
    sd <- as.integer(scatter_dimension)
    sh <- aval_of(s)$shape
    if (sd < 0L || sd >= length(sh))
      jax_error("psum_scatter: scatter_dimension ", sd, " is out of bounds for an array of rank ",
                length(sh), class = "jax_value_error")
    if (isTRUE(tiled)) {
      s <- lax_reshape(s, c(sh[seq_len(sd)], n, sh[sd + 1L] %/% n, drop_first(sh, sd + 1L)))
    } else if (sh[sd + 1L] != n) {
      jax_error("psum_scatter requires scatter_dimension of size equal to the axis size")
    }
    idx <- lax_axis_index(axis_name)
    out <- psum_scatter_slice(s, idx, sd)
    out
  }, x)
}

#' @rdname lax_collectives
#' @return `lax_psum_scatter_start()` returns a future: call `$done()` on it to get the
#'   result of [lax_psum_scatter()]. With one device the reduction runs eagerly, so
#'   there is no communication to overlap.
#' @export
lax_psum_scatter_start <- function(x, axis_name, scatter_dimension = 0L,
                                   axis_index_groups = NULL, tiled = FALSE) {
  val <- lax_psum_scatter(x, axis_name, scatter_dimension, axis_index_groups, tiled)
  structure(list(done = function() val), class = "jax_future")
}

#' @export
print.jax_future <- function(x, ...) {
  cat("Future\n")
  invisible(x)
}

psum_scatter_slice <- function(s, idx, d) {
  sh <- aval_of(s)$shape
  starts <- lapply(seq_along(sh), function(k) if (k == d + 1L) idx else lax_full_like(idx, 0))
  sizes <- sh
  sizes[d + 1L] <- 1L
  out <- bind(dynamic_slice_p, c(list(s), starts), list(slice_sizes = sizes))
  lax_squeeze(out, d)
}

# ---- ppermute / pshuffle / pbroadcast ----------------------------------------------------------

ppermute_p <- new_primitive("ppermute")
ppermute_p$abstract_eval <- function(avals, params) avals[[1L]]
ppermute_p$override$eval <- unbound_axis
ppermute_p$override$batch <- function(trace, args, params) {
  v <- args[[1L]]
  own <- is_own(v, trace)
  if (!identical(trace$axis_name, params$axis_name)) {
    if (!own) return(with_trace(trace$parent, bind(ppermute_p, args, params)))
    out <- with_trace(trace$parent, bind(ppermute_p, list(v$val), params))
    return(batch_tracer(trace, out, v$bdim))
  }
  with_trace(trace$parent, {
    n <- trace$axis_size
    full <- if (own) moveaxis(v$val, v$bdim, 0L) else lax_broadcast(v, n)
    src <- rep(NA_integer_, n)
    for (pr in params$perm) src[pr[2L] + 1L] <- pr[1L]
    lanes <- lapply(seq_len(n), function(j) {
      if (is.na(src[j])) lax_full_like(lax_index_in_dim(full, 0L, 0L, keepdims = FALSE), 0)
      else lax_index_in_dim(full, src[j], 0L, keepdims = FALSE)
    })
    batch_tracer(trace, lax_stack0(lanes), 0L)
  })
}
deflinear(ppermute_p, function(ct, args, params) {
  inv <- lapply(params$perm, function(p) c(p[2L], p[1L]))
  list(bind(ppermute_p, list(ct), update_params(params, list(perm = inv))))
})

#' @rdname lax_collectives
#' @param perm List of `c(source, destination)` index pairs (0-based).
#' @export
lax_ppermute <- function(x, axis_name, perm) {
  perm <- lapply(perm, as.integer)
  tree_map(function(l) bind(ppermute_p, list(as_jax(l)), list(axis_name = axis_name, perm = perm)), x)
}

#' @rdname lax_collectives
#' @export
lax_pshuffle <- function(x, axis_name, perm) {
  lax_ppermute(x, axis_name, lapply(seq_along(perm), function(i) c(perm[i], i - 1L)))
}

#' @rdname lax_collectives
#' @param source 0-based source index.
#' @export
lax_pbroadcast <- function(x, axis_name, source) {
  n <- axis_size_of(axis_name)
  lax_ppermute(x, axis_name, lapply(seq_len(n) - 1L, function(i) c(as.integer(source), i)))
}

# ---- all_to_all ---------------------------------------------------------------------------------

all_to_all_p <- new_primitive("all_to_all")
all_to_all_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  n <- axis_size_of(params$axis_name)
  s <- a$shape
  if (isTRUE(params$tiled)) {
    s[params$split_axis + 1L] <- s[params$split_axis + 1L] %/% n
    s[params$concat_axis + 1L] <- s[params$concat_axis + 1L] * n
  } else {
    s <- append(s[-(params$split_axis + 1L)], n, after = params$concat_axis)
  }
  new_aval(s, a$dtype, a$weak)
}
all_to_all_p$override$eval <- unbound_axis
# Lane i receives chunk i of the split axis from every lane, concatenated (stacked) along
# concat_axis: the batch dimension and the split axis exchange roles.
all_to_all_p$override$batch <- function(trace, args, params) {
  v <- args[[1L]]
  own <- is_own(v, trace)
  if (!identical(trace$axis_name, params$axis_name)) {
    if (!own) return(with_trace(trace$parent, bind(all_to_all_p, args, params)))
    d <- v$bdim
    sh <- function(a) a + (a >= d)
    out <- with_trace(trace$parent, bind(all_to_all_p, list(v$val),
      update_params(params, list(split_axis = sh(params$split_axis), concat_axis = sh(params$concat_axis)))))
    return(batch_tracer(trace, out, d))
  }
  with_trace(trace$parent, {
    n <- trace$axis_size
    full <- if (own) moveaxis(v$val, v$bdim, 0L) else lax_broadcast(v, n)
    s <- aval_of(full)$shape
    sa <- params$split_axis + 1L
    ca <- params$concat_axis + 1L
    if (isTRUE(params$tiled)) {
      full <- lax_reshape(full, c(s[seq_len(sa)], n, s[sa + 1L] %/% n, drop_first(s, sa + 1L)))
    }
    # full: (lanes, ..., chunks at axis sa, ...). Move chunks to the front (new lanes) and
    # old lanes to the concat position.
    out <- moveaxis(full, sa, 0L)
    out <- moveaxis(out, 1L, ca)
    if (isTRUE(params$tiled)) {
      so <- aval_of(out)$shape
      out <- lax_reshape(out, c(so[seq_len(ca)], so[ca + 1L] * so[ca + 2L], drop_first(so, ca + 2L)))
    }
    batch_tracer(trace, out, 0L)
  })
}
deflinear(all_to_all_p, function(ct, args, params) {
  list(bind(all_to_all_p, list(ct), update_params(params, list(split_axis = params$concat_axis,
                                                               concat_axis = params$split_axis))))
})

#' @rdname lax_collectives
#' @param split_axis,concat_axis 0-based axes.
#' @export
lax_all_to_all <- function(x, axis_name, split_axis, concat_axis, axis_index_groups = NULL,
                           tiled = FALSE) {
  tree_map(function(l) {
    l <- as_jax(l)
    r <- length(aval_of(l)$shape)
    n <- axis_size_of(axis_name)
    if (!isTRUE(tiled) && aval_of(l)$shape[norm_axis(split_axis, r) + 1L] != n)
      jax_error("all_to_all requires the split axis size to equal the axis size")
    bind(all_to_all_p, list(l), list(axis_name = axis_name, split_axis = norm_axis(split_axis, r),
                                     concat_axis = norm_axis(concat_axis, r), tiled = isTRUE(tiled)))
  }, x)
}

#' @rdname lax_collectives
#' @export
lax_pswapaxes <- function(x, axis_name, axis, axis_index_groups = NULL) {
  lax_all_to_all(x, axis_name, axis, axis)
}

# ---- ragged_all_to_all -----------------------------------------------------------------

ragged_all_to_all_p <- new_primitive("ragged_all_to_all")
ragged_all_to_all_p$abstract_eval <- function(avals, params) {
  op <- avals[[1L]]; out <- avals[[2L]]
  if (!length(op$shape)) jax_error("ragged_all_to_all input must be at least 1D, but got shape ()",
                                   class = "jax_value_error")
  if (length(op$shape) != length(out$shape) || !identical(op$shape[-1L], out$shape[-1L]))
    jax_error("ragged_all_to_all input and output shapes must be equal, except for the outermost dimension.",
              class = "jax_value_error")
  if (op$dtype != out$dtype)
    jax_error("ragged_all_to_all input and output dtypes must be equal", class = "jax_type_error")
  for (a in avals[3:6]) {
    if (!is_int_dtype(a$dtype) || length(a$shape) != 1L)
      jax_error("ragged_all_to_all offsets and sizes must be 1D integer arrays", class = "jax_value_error")
  }
  out
}
ragged_all_to_all_p$override$eval <- function(trace, args, params) {
  jax_error("ragged_all_to_all must be used within a mapped context like vmap or shard_map.",
            class = "jax_runtime_error")
}
ragged_all_to_all_p$override$batch <- function(trace, args, params) {
  names <- params$axis_name
  if (trace$axis_name %in% names)
    jax_error("Please open a feature request!", class = "jax_not_implemented_error")
  if (!is.null(params$axis_index_groups))
    jax_error("Please open a feature request!", class = "jax_not_implemented_error")
  own <- vapply(args, is_own, NA, trace = trace)
  if (!any(own)) return(with_trace(trace$parent, bind(ragged_all_to_all_p, args, params)))
  size <- trace$axis_size
  vals <- lapply(seq_along(args), function(i) {
    if (own[i]) moveaxis(args[[i]]$val, args[[i]]$bdim, 0L) else lax_broadcast(args[[i]], size)
  })
  with_trace(trace$parent, {
    N <- aval_of(vals[[1L]])$shape[2L]; M <- aval_of(vals[[2L]])$shape[2L]
    second <- lapply(vals[3:6], function(x) lax_transpose(x, c(1L, 0L)))
    lane <- function(x, n) lax_add(x, lax_mul(lax_expand_dims(lax_iota(aval_of(x)$dtype, size), 0L),
                                              lax_full(integer(0), n, aval_of(x)$dtype)))
    second[[1L]] <- lane(second[[1L]], N)
    second[[3L]] <- lane(second[[3L]], M)
    merge <- function(x) { s <- aval_of(x)$shape; lax_reshape(x, c(prod_int(s[1:2]), s[-(1:2)])) }
    ins <- c(lapply(vals[1:2], merge), lapply(second, merge))
    res <- bind(ragged_all_to_all_p, ins, params)
    s <- aval_of(res)$shape
    batch_tracer(trace, lax_reshape(res, c(size, s[1L] %/% size, s[-1L])), 0L)
  })
}

#' Ragged all to all collective
#'
#' Port of `jax.lax.ragged_all_to_all`. As in JAX, it needs a multi device mapped
#' context (shard_map); on one device it raises an error, and under `vmap` over its own
#' axis it raises `NotImplementedError` like JAX's batching rule.
#' @param operand,output Data to send and the array receiving it.
#' @param input_offsets,send_sizes,output_offsets,recv_sizes 1D integer arrays.
#' @param axis_name Axis name(s).
#' @param axis_index_groups Not supported.
#' @return An array shaped like `output`.
#' @export
lax_ragged_all_to_all <- function(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes,
                                  axis_name, axis_index_groups = NULL) {
  args <- lapply(list(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes), as_jax)
  bind(ragged_all_to_all_p, args, list(axis_name = collective_names(axis_name),
                                       axis_index_groups = axis_index_groups))
}
