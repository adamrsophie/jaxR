# Vectorization: batch trace and vmap. Batch dims are 0-based integers; NA means
# "not mapped".

new_batch_trace <- function(parent, axis_size, axis_name = NULL) {
  t <- new_trace("batch", parent)
  t$axis_size <- axis_size
  t$axis_name <- axis_name
  t
}

batch_tracer <- function(trace, val, bdim) {
  if (is.na(bdim)) return(val)
  a <- aval_of(val)
  a$shape <- a$shape[-(bdim + 1L)]
  new_tracer(trace, a, val = val, bdim = as.integer(bdim))
}

# Primitives allowed to use the slow per-element fallback batching rule.
.fallback_ok <- new.env(parent = emptyenv())
allow_fallback_batching <- function(...) for (n in c(...)) assign(n, TRUE, envir = .fallback_ok)

process_batch <- function(trace, prim, args, params) {
  ov <- prim$override$batch
  if (!is.null(ov)) return(ov(trace, args, params))
  own <- vapply(args, is_own, NA, trace = trace)
  if (!any(own)) return(with_trace(trace$parent, bind(prim, args, params)))
  vals <- args
  dims <- rep(NA_integer_, length(args))
  for (i in which(own)) {
    vals[[i]] <- args[[i]]$val
    dims[i] <- args[[i]]$bdim
  }
  axis <- list(size = trace$axis_size, name = trace$axis_name, trace = trace)
  rule <- prim$batch
  if (is.null(rule)) {
    if (!isTRUE(.fallback_ok[[prim$name]]))
      jax_error("Batching rule for '", prim$name, "' not implemented")
    rule <- fallback_batch_rule(prim)
  }
  res <- with_trace(trace$parent, rule(vals, dims, params, axis))
  if (prim$multiple_results) {
    Map(function(v, d) batch_tracer(trace, v, d), res[[1L]], as.list(res[[2L]]))
  } else {
    batch_tracer(trace, res[[1L]], res[[2L]])
  }
}

# ---- helpers used by batching rules (run under the parent trace) -----------------

moveaxis <- function(x, from, to) {
  r <- length(aval_of(x)$shape)
  from <- as.integer(from)
  to <- as.integer(to)
  if (from < 0L) from <- from + r
  if (to < 0L) to <- to + r
  if (from == to) return(x)
  perm <- seq_len(r) - 1L
  perm <- perm[perm != from]
  perm <- append(perm, from, after = to)
  lax_transpose(x, perm)
}

# Put the batch dim of x at position 0, broadcasting unmapped values.
bdim_at_front <- function(x, d, size) {
  if (is.na(d)) return(lax_broadcast(x, size))
  moveaxis(x, d, 0L)
}

bdim_to <- function(x, d, to, size) {
  if (is.na(d)) {
    x <- lax_broadcast(x, size)
    return(moveaxis(x, 0L, to))
  }
  moveaxis(x, d, to)
}

# Elementwise primitives: move batch dims to the front and broadcast unmapped args.
broadcast_batcher <- function(prim) {
  function(vals, dims, params, axis) {
    mapped <- !is.na(dims)
    shapes <- lapply(vals, function(v) aval_of(v)$shape)
    if (all(mapped) && length(unique(dims)) == 1L && length(unique(shapes)) == 1L) {
      return(list(bind(prim, vals, params), dims[1L]))
    }
    vals <- Map(function(v, d) bdim_at_front(v, d, axis$size), vals, dims)
    out <- bind(prim, vals, params)
    if (prim$multiple_results) list(out, rep(0L, length(out))) else list(out, 0L)
  }
}

# Slice-and-stack fallback: correct for pure per-lane primitives only.
fallback_batch_rule <- function(prim) {
  function(vals, dims, params, axis) {
    cnt <- .jx$fallback_hits[[prim$name]] %||% 0L
    assign(prim$name, cnt + 1L, envir = .jx$fallback_hits)
    n <- axis$size
    if (n == 0L) {
      avals <- lapply(seq_along(vals), function(i) aval_of(vals[[i]]))
      for (i in which(!is.na(dims))) avals[[i]]$shape <- avals[[i]]$shape[-(dims[i] + 1L)]
      oa <- abstract_eval(prim, avals, params)
      mk <- function(a) { a$shape <- c(0L, a$shape); zeros_aval(a) }
      if (prim$multiple_results) return(list(lapply(oa, mk), rep(0L, length(oa))))
      return(list(mk(oa), 0L))
    }
    per <- lapply(seq_len(n) - 1L, function(i) {
      a <- Map(function(v, d) if (is.na(d)) v else lax_index_in_dim(v, i, d, keepdims = FALSE),
               vals, dims)
      bind(prim, a, params)
    })
    if (prim$multiple_results) {
      k <- length(per[[1L]])
      outs <- lapply(seq_len(k), function(j) lax_stack0(lapply(per, `[[`, j)))
      list(outs, rep(0L, k))
    } else {
      list(lax_stack0(per), 0L)
    }
  }
}

# ---- vmap -------------------------------------------------------------------------

# Expand an axis specification to one entry per leaf of `tree` (port of
# api_util.flatten_axes). The specification is a pytree prefix of `tree` whose leaves are
# integers or NULL (not mapped); it may use registered pytree nodes, and named lists
# match by name.
broadcast_prefix_axes <- function(spec, tree, name = "vmap in_axes") {
  sfl <- tree_flatten(spec, is_leaf = is.null)
  ok <- vapply(sfl$leaves, function(a) {
    is.null(a) || (is.numeric(a) && length(a) == 1L && !is.na(a) && a == trunc(a))
  }, NA)
  if (!all(ok))
    jax_error(name, " must be an integer, NULL, or a (nested) list of these, one entry per ",
              "argument (use list(0L, 1L), not c(0L, 1L))", class = "jax_type_error")
  subs <- tryCatch(flatten_up_to(sfl$treedef, tree), error = function(e)
    jax_error(name, " specification must be a tree prefix of the corresponding value, got ",
              "specification ", format(sfl$treedef), " for value tree ",
              format(tree_structure(tree)), ".", class = "jax_value_error"))
  as.integer(unlist(Map(function(a, sub) {
    rep(if (is.null(a)) NA_integer_ else as.integer(a), length(tree_leaves(sub)))
  }, sfl$leaves, subs)))
}

# Port of batching.register_vmappable for pytree classes whose static data depends on
# the mapped axis (the shape of a sparse array). `axes` has one entry per array leaf
# of `tree`; f(handler, node, axis) rebuilds a registered node whose leaves are mapped
# like any others.
.jx$vmappable <- list()
vmap_elts <- function(tree, axes, f) {
  cls <- names(.jx$vmappable)
  is_vm <- function(x) inherits(x, cls)
  fl <- tree_flatten(tree, is_leaf = is_vm)
  if (!any(vapply(fl$leaves, is_vm, NA))) return(tree)
  at <- 0L
  tree_unflatten(fl$treedef, lapply(fl$leaves, function(x) {
    if (!is_vm(x)) { at <<- at + 1L; return(x) }
    axis <- axes[at + 1L]
    at <<- at + length(tree_leaves(x))
    if (is.na(axis)) x else f(.jx$vmappable[[cls[inherits(x, cls, which = TRUE) > 0L][1L]]], x, axis)
  }))
}

# Port of the error of api._mapped_axis_size.
vmap_size_error <- function(args, leaves, axes, sizes, axis_size) {
  paths <- tree_leaves_with_path(args)
  desc <- vapply(seq_along(leaves), function(i) {
    paste0("axis ", axes[i], " of args", keystr(paths[[i]][[1L]]), " of type ",
           aval_long(aval_of(leaves[[i]])))
  }, "")
  tab <- sort(table(sizes[!is.na(sizes)]), decreasing = TRUE)
  msg <- "vmap got inconsistent sizes for array axes to be mapped:\n"
  if (!is.null(axis_size))
    msg <- paste0(msg, "  * the `axis_size` argument was ", axis_size, ";\n")
  for (k in seq_along(tab)) {
    sz <- as.integer(names(tab)[k])
    ex <- desc[match(sz, sizes)]
    msg <- paste0(msg, if (tab[[k]] == 1L) paste0("  * one axis had size ", sz, ": ", ex)
                       else paste0("  * ", if (k == 1L) "most" else "some", " axes (", tab[[k]],
                                   " of them) had size ", sz, ", e.g. ", ex),
                  if (k < length(tab)) ";\n")
  }
  jax_error(msg, class = "jax_value_error")
}

#' Vectorizing map
#'
#' Port of `jax.vmap`. Axes are 0-based; `NULL` means not mapped.
#' @param fun Function to vectorize.
#' @param in_axes Integer, `NULL`, or a list (one entry per argument, each possibly a
#'   nested list or a registered pytree node) giving the mapped axis of each input.
#' @param out_axes Like `in_axes` for the outputs.
#' @param axis_name Optional name for collectives (`lax_psum` and friends).
#' @param axis_size Size of the mapped axis; needed when no input is mapped. When given
#'   it must agree with the sizes of the mapped input axes.
#' @return The vectorized function.
#' @export
vmap <- function(fun, in_axes = 0L, out_axes = 0L, axis_name = NULL, axis_size = NULL) {
  force(fun)
  function(...) {
    args <- list(...)
    if (is.list(in_axes) && length(in_axes) != length(args))
      jax_error("vmap in_axes must have one entry per positional argument (got ",
                length(in_axes), " for ", length(args), " arguments)")
    axes <- broadcast_prefix_axes(in_axes, args, "vmap in_axes")
    args <- vmap_elts(args, axes, function(h, x, a) h$to_elt(x, a))
    fl <- tree_flatten(args)
    # A Ref argument is mapped through its value (batching of core.ref_p in JAX): the
    # function sees the ref holding one element, and the ref keeps the stacked result.
    refs <- fl$leaves
    ref_at <- which(vapply(refs, is_ref, NA))
    for (i in ref_at) {
      for (j in ref_at[ref_at < i]) if (identical(refs[[j]], refs[[i]]))
        jax_error("only one reference to a mutable array may be passed as an argument to a ",
                  "function, but when tracing a function for vmap the mutable array reference ",
                  "of type Ref{", aval_long(refs[[i]]$aval), "} appeared at both flat index ",
                  j - 1L, " and flat index ", i - 1L, ".", class = "jax_value_error")
    }
    leaves <- lapply(fl$leaves, function(l) as_jax(if (is_ref(l)) ref_read(l) else l))
    axes0 <- axes
    sizes <- rep(NA_integer_, length(leaves))
    for (i in seq_along(leaves)) {
      if (is.na(axes[i])) next
      sh <- aval_of(leaves[[i]])$shape
      ax <- axes[i]
      if (ax < 0L) ax <- ax + length(sh)
      if (ax < 0L || ax >= length(sh))
        jax_error("vmap was requested to map its argument along axis ", axes[i],
                  ", which implies that its rank should be at least ",
                  if (axes[i] >= 0L) axes[i] + 1L else -axes[i], ", but is only ", length(sh),
                  " (its shape is ", py_tuple(sh), ")", class = "jax_value_error")
      axes[i] <- ax
      sizes[i] <- sh[ax + 1L]
    }
    if (!is.null(axis_size)) axis_size <- as.integer(axis_size)
    size <- unique(c(sizes[!is.na(sizes)], axis_size))
    if (!length(size))
      jax_error("vmap must have at least one non-None value in in_axes or axis_size must be ",
                "specified", class = "jax_value_error")
    if (length(size) > 1L) vmap_size_error(args, leaves, axes0, sizes, axis_size)
    trace <- new_batch_trace(current_trace(), as.integer(size), axis_name)
    on.exit(trace$alive <- FALSE, add = TRUE)
    ins <- Map(function(l, a) batch_tracer(trace, l, a), leaves, axes)
    mapped_refs <- ref_at[!is.na(axes[ref_at])]
    ref_set <- function(ref, v) {
      assign("value", v, envir = ref)
      assign("aval", new_aval(aval_of(v)$shape, aval_of(v)$dtype), envir = ref)
    }
    for (i in mapped_refs) ref_set(refs[[i]], ins[[i]])
    ins[ref_at] <- refs[ref_at]
    ref_done <- FALSE
    on.exit(if (!ref_done) for (i in mapped_refs) ref_set(refs[[i]], leaves[[i]]), add = TRUE)
    store <- new.env(parent = emptyenv())
    outs <- with_trace(trace, flat_fun(fun, fl$treedef, store)(ins))
    for (i in mapped_refs) {
      v <- ref_value(refs[[i]])
      ref_set(refs[[i]], if (is_own(v, trace)) moveaxis(v$val, v$bdim, axes[i])
                         else moveaxis(lax_broadcast(v, size), 0L, axes[i]))
    }
    ref_done <- TRUE
    out_tree <- store$out_tree
    oax <- broadcast_prefix_axes(out_axes, tree_unflatten(out_tree, outs), "vmap out_axes")
    res <- Map(function(o, ax) {
      if (is_own(o, trace)) {
        if (is.na(ax)) jax_error("vmap has mapped output but out_axes is None")
        r <- length(o$aval$shape) + 1L
        if (ax < 0L) ax <- ax + r
        return(moveaxis(o$val, o$bdim, ax))
      }
      if (is.na(ax)) return(o)
      r <- length(aval_of(o)$shape) + 1L
      if (ax < 0L) ax <- ax + r
      moveaxis(lax_broadcast(o, size), 0L, ax)
    }, outs, oax)
    kr_vmap(ins, leaves, outs, res, axes)
    vmap_elts(tree_unflatten(out_tree, res), oax,
              function(h, x, a) h$from_elt(x, as.integer(size), a))
  }
}
