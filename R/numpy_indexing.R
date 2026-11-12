# NumPy style indexing, the functional update API and segment reductions (ports of
# jax/_src/numpy/indexing.py, the `at` helper of jax/_src/numpy/array_methods.py and
# jax/_src/ops/scatter.py).
#
# Indices are 0-based. Not ported: sharding (out_sharding). Dynamic slices (jax.ds) are in
# R/refs.R.

# ---- index objects ------------------------------------------------------------------------

#' Index helpers
#'
#' `sl()` builds a Python style half-open slice for use inside `x[...]`: `sl(1, 4)`
#' is `1:4` in NumPy, `sl(2)` is `2:`, `sl(stop = 3)` is `:3` and `sl(step = -1)` is
#' `::-1`. Negative bounds count from the end. `newaxis` inserts a dimension of size
#' one (`None` in NumPy) and `ellipsis` stands for as many full slices as needed
#' (`...` in NumPy).
#' @param start First index (inclusive); `NULL` for the beginning.
#' @param stop Last index (exclusive); `NULL` for the end.
#' @param step Step; `NULL` for 1. May be negative.
#' @return An index object.
#' @export
sl <- function(start = NULL, stop = NULL, step = NULL) {
  structure(list(start = start, stop = stop, step = step), class = "jax_slice")
}

#' @rdname sl
#' @format NULL
#' @export
newaxis <- structure(list(), class = "jax_newaxis")

#' @rdname sl
#' @format NULL
#' @export
ellipsis <- structure(list(), class = "jax_ellipsis")

idx_error <- function(...) jax_error(..., class = "jax_index_error")

# Collect the index arguments of `[`: a missing argument is a full slice.
idx_collect <- function(...) {
  # alist() does not evaluate its arguments: this only captures the argument
  # expressions so that missing ones can be detected. Values come from ...elt().
  exprs <- eval(substitute(alist(...)))
  n <- length(exprs)
  if (!is.null(names(exprs)) && any(nzchar(names(exprs))))
    jax_error("unsupported named index argument(s): ",
              paste(names(exprs)[nzchar(names(exprs))], collapse = ", "),
              class = "jax_type_error")
  out <- vector("list", n)
  for (i in seq_len(n)) {
    if (identical(exprs[[i]], quote(expr = ))) {
      if (n == 1L) return(list())
      out[[i]] <- sl()
    } else {
      out[i] <- list(...elt(i))
    }
  }
  # An index tuple from jnp_s_ / jnp_index_exp stands for its elements.
  if (n == 1L && inherits(out[[1L]], "jax_index_tuple")) return(unclass(out[[1L]]))
  out
}

idx_format_slice <- function(s) {
  f <- function(v) if (is.null(v)) "NULL" else if (is_jax_value(v)) format(v) else format(v)
  paste0("sl(", f(s$start), ", ", f(s$stop), ", ", f(s$step), ")")
}

# Static value of a slice bound: NULL, an integer, or NA when it is not static.
idx_slice_bound <- function(v) {
  if (is.null(v)) return(NULL)
  if (is_jax_value(v)) {
    cv <- tracer_concrete_value(v)
    if (is.null(cv)) return(NA_integer_)
    v <- as.vector(array_host(cv))
  }
  if (!is.numeric(v) || length(v) != 1L || is.na(v) || v != round(v))
    jax_error("slice indices must be integers or NULL", class = "jax_type_error")
  as.integer(v)
}

# Port of Python's slice.indices(): (start, stop, step) clipped to an axis of `size`.
idx_slice_indices <- function(s, size) {
  step <- s$step %||% 1L
  if (step == 0L) jax_error("slice step cannot be zero", class = "jax_value_error")
  if (step > 0L) { lower <- 0L; upper <- size } else { lower <- -1L; upper <- size - 1L }
  bound <- function(v, default) {
    if (is.null(v)) default else if (v < 0L) max(v + size, lower) else min(v, upper)
  }
  c(bound(s$start, if (step < 0L) upper else lower),
    bound(s$stop, if (step < 0L) lower else upper), step)
}

# Port of jax/_src/core.py:canonicalize_slice: (start, step, size) of x[s].
idx_canonicalize_slice <- function(s, axis_size) {
  step <- s$step %||% 1L
  if (step == 0L) jax_error("slice step cannot be zero", class = "jax_value_error")
  pos <- step > 0L
  clamp <- function(i) {
    if (i >= 0L) (if (pos) min(axis_size, i) else min(axis_size - 1L, i))
    else (if (pos) max(0L, axis_size + i) else max(-1L, axis_size + i))
  }
  start <- if (is.null(s$start)) (if (pos) 0L else axis_size - 1L) else clamp(s$start)
  stop <- if (is.null(s$stop)) (if (pos) axis_size else -1L) else clamp(s$stop)
  gap <- abs(step)
  distance <- if (pos) stop - start else start - stop
  c(start, step, max(0L, distance + gap - 1L) %/% gap)
}

# ---- parsing (ports of IndexType.from_index and _parse_indices) -----------------------------

# Classify one raw index. R numeric vectors become integer arrays; an R numeric
# scalar is a static integer index.
idx_classify <- function(i) {
  out <- function(index, typ, ndim = 0L) list(index = index, typ = typ, ndim = ndim)
  if (inherits(i, "jax_newaxis")) return(out(NULL, "none"))
  if (inherits(i, "jax_ellipsis")) return(out(NULL, "ellipsis"))
  if (inherits(i, "jax_slice")) return(out(i, "slice"))
  bad_dtype <- function(dt) {
    jax_error("Indexer must have integer or boolean type, got indexer with type ", dt,
              class = "jax_type_error")
  }
  if (is_jax_value(i)) {
    a <- aval_of(i)
    if (a$dtype == "bool") return(out(i, "boolean", length(a$shape)))
    if (is_int_dtype(a$dtype)) return(out(i, "array", length(a$shape)))
    bad_dtype(a$dtype)
  }
  if (is.null(i))
    idx_error("NULL is not a valid index; use `newaxis` to insert a dimension")
  if (is.character(i))
    jax_error("JAX does not support string indexing; got idx=", deparse(i),
              class = "jax_type_error")
  if (is.list(i))
    jax_error("Using a list for multidimensional indexing is not allowed; pass one index ",
              "argument per dimension, for example do.call(`[`, c(list(x), indices)).",
              class = "jax_type_error")
  if (is.logical(i)) {
    if (anyNA(i)) idx_error("NA is not a valid index")
    return(out(i, "boolean", length(r_shape(i))))
  }
  if (is.numeric(i)) {
    if (anyNA(i) || any(i != round(i))) bad_dtype("float64")
    if (length(i) == 1L && is.null(dim(i))) return(out(as.numeric(i), "integer"))
    arr <- gs_as_indices(i)
    return(out(arr, "array", length(aval_of(arr)$shape)))
  }
  if (is.complex(i)) bad_dtype("complex128")
  idx_error("only integers, slices (`sl()`), `ellipsis`, `newaxis` and integer or boolean ",
            "arrays are valid indices. Got an object of class ", class(i)[1L])
}

# An indexer is list(shape, indices); each index is list(index, typ, axes) where
# `axes` are the 0-based axes of the array that it consumes.
idx_parse <- function(indices, shape) {
  cls <- lapply(indices, idx_classify)
  typs <- vapply(cls, `[[`, "", "typ")
  consumed <- vapply(cls, function(c) {
    switch(c$typ, none = 0L, ellipsis = 0L, boolean = c$ndim, 1L)
  }, 1L)
  ell <- which(typs == "ellipsis")
  if (length(ell) > 1L) idx_error("an index can only have a single ellipsis ('...')")
  total <- sum(consumed)
  if (total > length(shape))
    idx_error("Too many indices: array is ", length(shape), "-dimensional, but ", total,
              " were indexed")
  if (length(ell)) consumed[ell] <- length(shape) - total
  ends <- cumsum(consumed)
  parsed <- lapply(seq_along(cls), function(k) {
    list(index = cls[[k]]$index, typ = cls[[k]]$typ,
         axes = ends[k] - consumed[k] + seq_len(consumed[k]) - 1L)
  })
  list(shape = as.integer(shape), indices = parsed)
}

idx_pi <- function(index, typ, axes) list(index = index, typ = typ, axes = as.integer(axes))

idx_ndim <- function(p) {
  if (p$typ == "boolean") return(length(if (is_jax_value(p$index)) aval_of(p$index)$shape
                                        else r_shape(p$index)))
  if (p$typ == "array") return(length(aval_of(p$index)$shape))
  0L
}

# ---- NDIndexer methods ----------------------------------------------------------------------

# Port of NDIndexer.validate_static_indices.
idx_validate_static_indices <- function(ix, normalize_indices = TRUE) {
  for (p in ix$indices) {
    if (p$typ != "integer") next
    i <- p$index
    size <- ix$shape[p$axes + 1L]
    normed <- if (normalize_indices && i < 0) i + size else i
    if (!(0 <= normed && normed < size))
      idx_error("index ", i, " out of bounds for axis ", p$axes, " with size ", size,
                " (normalize_indices=", if (normalize_indices) "True" else "False", ")")
  }
  invisible()
}

# Port of NDIndexer.validate_slices; returns the indexer with static slice bounds.
idx_validate_slices <- function(ix) {
  for (k in seq_along(ix$indices)) {
    p <- ix$indices[[k]]
    if (p$typ != "slice") next
    s <- lapply(unclass(p$index), idx_slice_bound)
    if (any(vapply(s, function(v) !is.null(v) && is.na(v), NA)))
      idx_error("Array slice indices must have static start/stop/step to be used with ",
                "NumPy indexing syntax. Got ", idx_format_slice(p$index), " at position ",
                k - 1L, ". To index an array at a dynamic position with a static slice ",
                "size, use lax_dynamic_slice/lax_dynamic_update_slice instead (jaxR does ",
                "not support dynamically sized arrays within traced functions).")
    ix$indices[[k]]$index <- structure(list(start = s$start, stop = s$stop, step = s$step),
                                       class = "jax_slice")
  }
  ix
}

# Row-major coordinates of the TRUE entries of a logical array (np.where order).
idx_where <- function(m) {
  shape <- r_shape(m)
  if (length(shape) <= 1L) return(list(which(as.vector(m)) - 1L))
  ind <- which(m, arr.ind = TRUE)
  ind <- ind[do.call(order, lapply(seq_len(ncol(ind)), function(j) ind[, j])), , drop = FALSE]
  lapply(seq_len(ncol(ind)), function(j) unname(ind[, j]) - 1L)
}

# Port of NDIndexer.expand_bool_indices: boolean arrays become integer index arrays;
# scalar booleans stay in place.
idx_expand_bool_indices <- function(ix) {
  out <- list()
  for (k in seq_along(ix$indices)) {
    p <- ix$indices[[k]]
    if (p$typ != "boolean") {
      out[[length(out) + 1L]] <- p
      next
    }
    m <- p$index
    if (is_jax_value(m)) {
      cv <- tracer_concrete_value(m)
      if (is.null(cv))
        jax_error("Array boolean indices must be concrete; got ", aval_str(aval_of(m)),
                  class = c("jax_non_concrete_boolean_index_error", "jax_index_error"))
      m <- array_host(cv)
    }
    mshape <- r_shape(m)
    if (!length(mshape)) {
      out[[length(out) + 1L]] <- idx_pi(isTRUE(as.vector(m)), "boolean", integer(0))
      next
    }
    expected <- ix$shape[p$axes + 1L]
    if (!all(mshape == 0L | mshape == expected))
      idx_error("boolean index did not match shape of indexed array in index ", k - 1L,
                ": got ", gs_tuple(mshape), ", expected [", paste(expected, collapse = ", "),
                "]")
    coords <- idx_where(m)
    for (j in seq_along(coords)) {
      out[[length(out) + 1L]] <- idx_pi(as_jax(array(as.integer(coords[[j]]),
                                                     length(coords[[j]]))), "array", p$axes[j])
    }
  }
  ix$indices <- out
  ix
}

# Port of NDIndexer.expand_ellipses: the ellipsis and implicit trailing slices become
# explicit full slices.
idx_expand_ellipses <- function(ix) {
  out <- list()
  consumed <- 0L
  for (p in ix$indices) {
    consumed <- consumed + length(p$axes)
    if (p$typ == "ellipsis") {
      for (axis in p$axes) out[[length(out) + 1L]] <- idx_pi(sl(), "slice", axis)
    } else {
      out[[length(out) + 1L]] <- p
    }
  }
  for (axis in consumed + seq_len(length(ix$shape) - consumed) - 1L)
    out[[length(out) + 1L]] <- idx_pi(sl(), "slice", axis)
  ix$indices <- out
  ix
}

# Port of NDIndexer.expand_scalar_bool_indices: a scalar boolean indexes a new axis
# of size one with arange(as.integer(index)).
idx_expand_scalar_bool_indices <- function(ix) {
  new_shape <- ix$shape
  cur <- 0L
  for (k in seq_along(ix$indices)) {
    p <- ix$indices[[k]]
    if (p$typ == "boolean") {
      new_shape <- append(new_shape, 1L, after = k - 1L)
      n <- as.integer(isTRUE(p$index))
      ix$indices[[k]] <- idx_pi(as_jax(array(seq_len(n) - 1L, n)), "array", cur)
      cur <- cur + 1L
    } else {
      n <- length(p$axes)
      ix$indices[[k]]$axes <- cur + seq_len(n) - 1L
      cur <- cur + n
    }
  }
  ix$shape <- as.integer(new_shape)
  ix
}

# Port of NDIndexer.normalize_indices: negative indices count from the end.
idx_normalize_indices <- function(ix) {
  for (k in seq_along(ix$indices)) {
    p <- ix$indices[[k]]
    size <- ix$shape[p$axes + 1L]
    if (p$typ == "integer") {
      if (p$index < 0) ix$indices[[k]]$index <- p$index + size
    } else if (p$typ == "array") {
      i <- p$index
      if (!is_unsigned_dtype(aval_of(i)$dtype))
        ix$indices[[k]]$index <- lax_select(lax_lt(i, lax_full_like(i, 0)),
                                           lax_add(i, lax_full_like(i, size)), i)
    }
  }
  ix
}

# Port of NDIndexer.is_advanced_int_indexer.
idx_is_advanced <- function(ix) {
  any(vapply(ix$indices, function(p) p$typ %in% c("array", "boolean") && idx_ndim(p) > 0L, NA))
}

idx_gather_mode <- function(mode) {
  if (is.null(mode)) "promise_in_bounds" else gs_mode_from_any(mode)
}

# Port of NDIndexer.to_static_slice. Signals a type, value or index error if the
# index cannot be computed with a static slice.
idx_to_static_slice <- function(ix, normalize_indices = TRUE, mode = NULL) {
  parsed_mode <- idx_gather_mode(mode)
  if (!(parsed_mode %in% c("promise_in_bounds", "clip")))
    jax_error("static_slice requires mode='promise_in_bounds' or mode='clip'",
              class = "jax_value_error")
  if (parsed_mode == "promise_in_bounds") idx_validate_static_indices(ix, normalize_indices)
  ix <- idx_validate_slices(ix)
  for (k in seq_along(ix$indices)) {
    if (ix$indices[[k]]$typ %in% c("array", "boolean"))
      jax_error("static_slice: indices must be static scalars or slices. Got an array ",
                "index at position ", k - 1L, class = "jax_type_error")
  }
  start_indices <- limit_indices <- strides <- integer(0)
  rev_axes <- squeeze_axes <- newaxis_dims <- integer(0)
  for (p in idx_expand_ellipses(ix)$indices) {
    if (p$typ == "none") {
      # Positions refer to the array after slicing and squeezing; expand_dims inserts
      # dimensions in order, so earlier new axes count too.
      newaxis_dims <- c(newaxis_dims,
                        length(start_indices) - length(squeeze_axes) + length(newaxis_dims))
    } else if (p$typ == "integer") {
      size <- ix$shape[p$axes + 1L]
      if (size == 0L) idx_error("index is out of bounds for axis ", p$axes, " with size 0")
      start <- p$index
      if (normalize_indices && start < 0) start <- start + size
      start <- as.integer(min(max(start, 0), size - 1L))
      start_indices <- c(start_indices, start)
      limit_indices <- c(limit_indices, start + 1L)
      strides <- c(strides, 1L)
      squeeze_axes <- c(squeeze_axes, p$axes)
    } else {
      size <- ix$shape[p$axes + 1L]
      r <- idx_slice_indices(p$index, size)
      start <- r[1L]; stop <- r[2L]; stride <- r[3L]
      if (stride < 0L) {
        new_start <- min(size, stop + 1L + abs(start - stop - 1L) %% abs(stride))
        start_indices <- c(start_indices, new_start)
        limit_indices <- c(limit_indices, max(new_start, start + 1L))
        strides <- c(strides, abs(stride))
        rev_axes <- c(rev_axes, p$axes)
      } else {
        start_indices <- c(start_indices, start)
        limit_indices <- c(limit_indices, max(start, stop))
        strides <- c(strides, stride)
      }
    }
  }
  list(start_indices = start_indices, limit_indices = limit_indices,
       strides = if (all(strides == 1L)) NULL else strides, rev_axes = rev_axes,
       squeeze_axes = squeeze_axes, newaxis_dims = newaxis_dims)
}

# Port of NDIndexer.to_dynamic_slice.
idx_to_dynamic_slice <- function(ix, normalize_indices = TRUE, mode = NULL) {
  if (!is.null(mode) && !(gs_mode_from_any(mode) %in% c("promise_in_bounds", "clip")))
    jax_error("dynamic_slice requires mode='promise_in_bounds' or mode='clip'",
              class = "jax_value_error")
  ix <- idx_validate_slices(ix)
  for (k in seq_along(ix$indices)) {
    p <- ix$indices[[k]]
    if (p$typ == "slice") {
      if (!is.null(p$index$step) && !(p$index$step %in% c(-1L, 1L)))
        jax_error("dynamic_slice: only unit steps supported in slice. Got ",
                  idx_format_slice(p$index), " at position ", k - 1L, class = "jax_type_error")
    } else if (p$typ == "array") {
      if (idx_ndim(p) != 0L)
        jax_error("dynamic_slice: only scalar indices allowed. Got an array index at ",
                  "position ", k - 1L, class = "jax_type_error")
    } else if (p$typ == "boolean") {
      jax_error("dynamic_slice: indices must be scalars or slices. Got a boolean index at ",
                "position ", k - 1L, class = "jax_type_error")
    }
  }
  start_indices <- list()
  slice_sizes <- rev_axes <- squeeze_axes <- newaxis_dims <- integer(0)
  trivial_slicing <- TRUE
  for (p in idx_expand_ellipses(ix)$indices) {
    if (p$typ == "none") {
      newaxis_dims <- c(newaxis_dims,
                        length(start_indices) - length(squeeze_axes) + length(newaxis_dims))
    } else if (p$typ %in% c("integer", "array")) {
      trivial_slicing <- FALSE
      if (ix$shape[p$axes + 1L] == 0L)
        idx_error("index is out of bounds for axis ", p$axes, " with size 0")
      # As upstream, scalar indices are arrays here (and are normalized dynamically).
      start_indices[[length(start_indices) + 1L]] <- gs_as_indices(p$index)
      slice_sizes <- c(slice_sizes, 1L)
      squeeze_axes <- c(squeeze_axes, p$axes)
    } else {
      s <- p$index
      if (!(is.null(s$start) && is.null(s$stop) && is.null(s$step))) trivial_slicing <- FALSE
      r <- idx_slice_indices(s, ix$shape[p$axes + 1L])
      start <- r[1L]; stop <- r[2L]; stride <- r[3L]
      if (stride < 0L) {
        new_start <- stop + 1L
        start_indices[[length(start_indices) + 1L]] <- new_start
        slice_sizes <- c(slice_sizes, max(0L, start + 1L - new_start))
        rev_axes <- c(rev_axes, p$axes)
      } else {
        start_indices[[length(start_indices) + 1L]] <- start
        slice_sizes <- c(slice_sizes, max(0L, stop - start))
      }
    }
  }
  if (length(start_indices) > 1L) {
    # dynamic_slice requires all start indices to have the same type.
    start_indices <- lapply(start_indices, function(i) {
      lax_convert_element_type(gs_as_indices(i), "int32")
    })
  }
  list(start_indices = start_indices, slice_sizes = slice_sizes, rev_axes = rev_axes,
       squeeze_axes = squeeze_axes, newaxis_dims = newaxis_dims,
       normalize_indices = normalize_indices, trivial_slicing = trivial_slicing)
}

# Port of jax/_src/numpy/indexing.py:_index_to_gather.
idx_to_gather <- function(ix, normalize_indices = TRUE) {
  ix <- idx_validate_slices(ix)
  is_advanced <- which(vapply(ix$indices, function(p) p$typ %in% c("array", "integer"), NA))
  advanced_axes_are_contiguous <- all(diff(is_advanced) == 1L)

  ix <- idx_expand_ellipses(ix)
  scalar_bool_dims <- which(vapply(ix$indices, function(p) p$typ == "boolean", NA)) - 1L
  ix <- idx_expand_scalar_bool_indices(ix)
  if (normalize_indices) ix <- idx_normalize_indices(ix)

  # With array indices, scalar integer indices take part in advanced indexing too.
  advanced_indexes <- list()
  idx_advanced_axes <- x_advanced_axes <- integer(0)
  if (idx_is_advanced(ix)) {
    j <- 0L
    for (k in seq_along(ix$indices)) {
      p <- ix$indices[[k]]
      if (p$typ == "none") next
      if (p$typ %in% c("array", "integer")) {
        advanced_indexes[[length(advanced_indexes) + 1L]] <- gs_as_indices(p$index)
        idx_advanced_axes <- c(idx_advanced_axes, k - 1L)
        x_advanced_axes <- c(x_advanced_axes, j)
      }
      j <- j + 1L
    }
  }

  x_axis <- 0L            # current axis in x
  y_axis <- 0L            # current axis in y, before collapsing
  collapsed_y_axis <- 0L  # current axis in y, after collapsing
  offset_dims <- collapsed_slice_dims <- start_index_map <- integer(0)
  index_dtype <- "int32"
  gather_indices <- list()          # pairs (array, start_dim)
  gather_indices_shape <- integer(0)
  slice_shape <- newaxis_dims <- reversed_y_dims <- gather_slice_shape <- integer(0)
  add_index <- function(g) {
    gather_indices[[length(gather_indices) + 1L]] <<- list(g, length(gather_indices_shape))
  }

  for (k in seq_along(ix$indices)) {
    idx_pos <- k - 1L
    p <- ix$indices[[k]]
    # The advanced indices are handled at the first advanced index if they are
    # contiguous, and at the start otherwise.
    if (length(advanced_indexes) &&
        ((advanced_axes_are_contiguous && idx_pos == idx_advanced_axes[1L]) ||
         (!advanced_axes_are_contiguous && idx_pos == 0L))) {
      arrs <- do.call(jnp_broadcast_arrays, advanced_indexes)
      shape <- aval_of(arrs[[1L]])$shape
      for (a in arrs) add_index(lax_convert_element_type(a, index_dtype))
      gather_indices_shape <- c(gather_indices_shape, shape)
      start_index_map <- c(start_index_map, x_advanced_axes)
      collapsed_slice_dims <- c(collapsed_slice_dims, x_advanced_axes)
      slice_shape <- c(slice_shape, shape)
      y_axis <- y_axis + length(shape)
      collapsed_y_axis <- collapsed_y_axis + length(shape)
    }
    if (idx_pos %in% idx_advanced_axes) {
      x_axis <- x_axis + 1L
      gather_slice_shape <- c(gather_slice_shape, 1L)
      next
    }
    if (p$typ %in% c("integer", "array") && idx_ndim(p) == 0L) {
      # Basic scalar integer index.
      if (ix$shape[x_axis + 1L] == 0L)
        idx_error("index is out of bounds for axis ", x_axis, " with size 0")
      add_index(lax_convert_element_type(gs_as_indices(p$index), index_dtype))
      collapsed_slice_dims <- c(collapsed_slice_dims, x_axis)
      gather_slice_shape <- c(gather_slice_shape, 1L)
      start_index_map <- c(start_index_map, x_axis)
      x_axis <- x_axis + 1L
    } else if (p$typ == "none") {
      slice_shape <- c(slice_shape, 1L)
      newaxis_dims <- c(newaxis_dims, y_axis)
      y_axis <- y_axis + 1L
    } else if (p$typ == "slice") {
      size <- ix$shape[x_axis + 1L]
      r <- idx_canonicalize_slice(p$index, size)
      start <- r[1L]; step <- r[2L]; slice_size <- r[3L]
      slice_shape <- c(slice_shape, slice_size)
      if (step == 1L) {
        # Avoid generating a trivial gather.
        if (slice_size != size) {
          add_index(new_array(new_aval(integer(0), index_dtype), host = as.double(start)))
          start_index_map <- c(start_index_map, x_axis)
        }
        gather_slice_shape <- c(gather_slice_shape, slice_size)
        offset_dims <- c(offset_dims, collapsed_y_axis)
      } else {
        vals <- start + step * (seq_len(slice_size) - 1L)
        if (step < 0L) {
          reversed_y_dims <- c(reversed_y_dims, collapsed_y_axis)
          vals <- rev(vals)
        }
        gather_slice_shape <- c(gather_slice_shape, 1L)
        add_index(as_jax(array(as.integer(vals), slice_size), index_dtype))
        start_index_map <- c(start_index_map, x_axis)
        gather_indices_shape <- c(gather_indices_shape, slice_size)
        collapsed_slice_dims <- c(collapsed_slice_dims, x_axis)
      }
      collapsed_y_axis <- collapsed_y_axis + 1L
      y_axis <- y_axis + 1L
      x_axis <- x_axis + 1L
    } else {
      idx_error("Got unsupported indexer at position ", idx_pos)
    }
  }

  if (!length(gather_indices)) {
    gather_indices_array <- as_jax(integer(0))
  } else if (length(gather_indices) == 1L) {
    g <- gather_indices[[1L]][[1L]]
    gather_indices_array <- lax_expand_dims(g, length(aval_of(g)$shape))
  } else {
    last_dim <- length(gather_indices_shape)
    full_shape <- c(gather_indices_shape, 1L)
    gather_indices_array <- lax_concatenate(lapply(gather_indices, function(gi) {
      nd <- length(aval_of(gi[[1L]])$shape)
      lax_broadcast_in_dim(gi[[1L]], full_shape, gi[[2L]] + seq_len(nd) - 1L)
    }), last_dim)
  }
  list(slice_shape = slice_shape, newaxis_dims = newaxis_dims,
       gather_slice_shape = gather_slice_shape, reversed_y_dims = reversed_y_dims,
       dnums = lax_GatherDimensionNumbers(offset_dims = offset_dims,
                                      collapsed_slice_dims = sort(collapsed_slice_dims),
                                      start_index_map = start_index_map),
       gather_indices = gather_indices_array,
       unique_indices = !length(advanced_indexes),
       indices_are_sorted = !length(advanced_indexes),
       scalar_bool_dims = scalar_bool_dims)
}

# ---- take (ports of rewriting_take, _static_slice, _dynamic_slice, _gather) ---------------------

idx_finish_slice <- function(result, s) {
  if (length(s$rev_axes)) result <- lax_rev(result, s$rev_axes)
  if (length(s$squeeze_axes)) result <- lax_squeeze(result, s$squeeze_axes)
  if (length(s$newaxis_dims)) result <- lax_expand_dims(result, s$newaxis_dims)
  result
}

idx_static_slice <- function(arr, s) {
  shape <- aval_of(arr)$shape
  trivial <- is.null(s$strides) && length(shape) == length(s$start_indices) &&
    all(s$start_indices == 0L) && all(s$limit_indices == shape)
  result <- if (trivial) arr else lax_slice(arr, s$start_indices, s$limit_indices, s$strides)
  idx_finish_slice(result, s)
}

idx_dynamic_slice <- function(arr, s) {
  result <- arr
  if (!s$trivial_slicing) {
    shape <- aval_of(arr)$shape
    start <- gs_dynamic_slice_indices(shape, s$start_indices, s$normalize_indices)
    result <- bind(dynamic_slice_p, c(list(arr), start), list(slice_sizes = s$slice_sizes))
  }
  idx_finish_slice(result, s)
}

idx_gather <- function(arr, ix, indices_are_sorted, unique_indices, mode, fill_value,
                       normalize_indices) {
  g <- idx_to_gather(ix, normalize_indices)
  y <- arr
  if (!is.null(fill_value)) {
    if (is_jax_value(fill_value))
      fill_value <- as.vector(array_host(concrete_or_error(
        fill_value, "fill_value argument to indexed get()")))
    if (length(fill_value) != 1L || !is.null(dim(fill_value)))
      jax_error("fill_value argument to indexed get() must be a scalar",
                class = "jax_value_error")
  }
  if (length(g$scalar_bool_dims)) y <- lax_expand_dims(y, g$scalar_bool_dims)
  # No gather if the result is empty (also handles zeros(0)[integer(0)]).
  if (any(g$slice_shape == 0L)) return(lax_full_like(y, 0, shape = g$slice_shape))
  if (!any(aval_of(g$gather_indices)$shape == 0L))
    y <- lax_gather(y, g$gather_indices, g$dnums, g$gather_slice_shape,
                    unique_indices = unique_indices || g$unique_indices,
                    indices_are_sorted = indices_are_sorted || g$indices_are_sorted,
                    mode = mode, fill_value = fill_value)
  if (length(g$reversed_y_dims)) y <- lax_rev(y, g$reversed_y_dims)
  lax_expand_dims(y, g$newaxis_dims)
}

idx_try <- function(expr) {
  tryCatch(expr, jax_type_error = function(e) NULL, jax_value_error = function(e) NULL,
           jax_index_error = function(e) NULL)
}

# Port of jax/_src/numpy/indexing.py:rewriting_take: computes arr[idx] for a list of
# raw indices.
idx_rewriting_take <- function(arr, idx, indices_are_sorted = FALSE, unique_indices = FALSE,
                               mode = NULL, fill_value = NULL, normalize_indices = TRUE,
                               strategy = "auto") {
  arr <- as_jax(arr)
  ix <- idx_parse(idx, aval_of(arr)$shape)
  if (!is.character(strategy) || length(strategy) != 1L ||
      !(strategy %in% c("auto", "gather", "scatter", "static_slice", "dynamic_slice")))
    jax_error("Unknown indexing strategy: ", deparse(strategy), class = "jax_value_error")
  if (strategy == "scatter")
    jax_error("Indexing strategy scatter is not supported for take.",
              class = "jax_value_error")
  if (isTRUE(.jx$config$jax_check_static_indices) &&
      idx_gather_mode(mode) == "promise_in_bounds")
    idx_validate_static_indices(ix, normalize_indices)
  if (strategy == "static_slice")
    return(idx_static_slice(arr, idx_to_static_slice(ix, normalize_indices, mode)))
  if (strategy == "dynamic_slice")
    return(idx_dynamic_slice(arr, idx_to_dynamic_slice(ix, normalize_indices, mode)))
  if (strategy == "auto") {
    s <- idx_try(idx_to_static_slice(ix, normalize_indices, mode))
    if (!is.null(s)) return(idx_static_slice(arr, s))
    s <- idx_try(idx_to_dynamic_slice(ix, normalize_indices, mode))
    if (!is.null(s)) return(idx_dynamic_slice(arr, s))
  }
  idx_gather(arr, idx_expand_bool_indices(ix), indices_are_sorted, unique_indices, mode,
             fill_value, normalize_indices)
}

# ---- scatter updates (ports of ops/scatter.py:_scatter_update and _scatter_impl) ----------------

idx_safe_to_cast <- function(ya, xa) {
  if (ya$dtype == xa$dtype) return(TRUE)
  promote_types(c(ya$dtype, xa$dtype), c(ya$weak, FALSE))$dtype == xa$dtype
}

idx_scatter_update <- function(x, idx, y, scatter_op, indices_are_sorted = FALSE,
                               unique_indices = FALSE, mode = NULL, normalize_indices = TRUE) {
  x <- as_jax(x)
  xa <- aval_of(x)
  # For integer arrays, integer valued R numerics are integer updates, as Python ints
  # are upstream: a scalar takes the array dtype if it is in range and is weakly typed
  # otherwise; vectors and arrays are int32.
  if (!is_jax_value(y) && is.numeric(y) && is_int_dtype(xa$dtype) && !anyNA(y) &&
      all(y == round(y))) {
    in_range <- length(y) == 1L && is.null(dim(y)) && y >= iinfo(xa$dtype)$min &&
      y <= iinfo(xa$dtype)$max
    y <- if (in_range) lax_full_like(x, y, shape = integer(0)) else gs_as_indices(y)
  } else {
    y <- as_jax(y)
  }
  ix <- idx_expand_bool_indices(idx_parse(idx, xa$shape))
  ya <- aval_of(y)
  if (!idx_safe_to_cast(ya, xa))
    warning("scatter inputs have incompatible types: cannot safely cast value from dtype=",
            ya$dtype, " to dtype=", xa$dtype, " with jax_numpy_dtype_promotion=",
            .jx$config$jax_numpy_dtype_promotion %||% "standard",
            ". In future JAX releases this will result in an error.", call. = FALSE)
  g <- idx_to_gather(ix, normalize_indices)
  # No scatter if the slice is empty (also handles zeros(0)[integer(0)]).
  if (any(g$slice_shape == 0L)) return(x)
  xy <- promote_dtypes(list(x, y))
  x <- xy[[1L]]
  # Broadcast y to the slice shape, then drop the newaxis dimensions.
  y <- jnp_broadcast_to(xy[[2L]], g$slice_shape)
  if (length(g$newaxis_dims)) y <- lax_squeeze(y, g$newaxis_dims)
  if (length(g$reversed_y_dims)) y <- lax_rev(y, g$reversed_y_dims)
  if (length(g$scalar_bool_dims)) x <- lax_expand_dims(x, g$scalar_bool_dims)
  # The scatter dimension numbers are the transpose of the gather ones.
  out <- scatter_op(x, g$gather_indices, y, gs_gather_to_scatter_dnums(g$dnums),
                    indices_are_sorted = g$indices_are_sorted || indices_are_sorted,
                    unique_indices = g$unique_indices || unique_indices, mode = mode)
  if (length(g$scalar_bool_dims)) out <- lax_squeeze(out, g$scalar_bool_dims)
  lax_convert_element_type(out, xa$dtype, xa$weak)
}

# ---- `[`, `[<-` and the at_* functions ------------------------------------------------------------

#' NumPy style indexing of jax arrays
#'
#' `x[...]` follows NumPy semantics with 0-based indices (port of
#' `jax.numpy.ndarray.__getitem__`). Give one argument per dimension; trailing
#' dimensions that are not indexed are kept in full, and a missing argument
#' (`x[, 1]`) is a full slice. Index arguments can be:
#' * an R numeric scalar or a rank 0 integer jax value: selects one index and drops
#'   the dimension (negative values count from the end);
#' * [sl()]: a half-open slice with optional step;
#' * an integer vector or array (R or jax): advanced indexing, with NumPy
#'   broadcasting and placement rules;
#' * a logical vector or a boolean array: a mask (concrete arrays only);
#' * [newaxis] or [ellipsis].
#'
#' Out of bounds indices are clamped when reading (use [at_get()] with
#' `mode = "fill"` for a fill value) and dropped when writing. `x[...] <- value`
#' returns an updated copy, as [at_set()] does; arrays are immutable.
#' @param x A jax array or tracer.
#' @param ... Index arguments.
#' @param value Replacement values, broadcast to the indexed shape.
#' @return The indexed array, or the updated array.
#' @name jax_indexing
#' @export
`[.jax_value` <- function(x, ...) idx_rewriting_take(x, idx_collect(...))

#' @rdname jax_indexing
#' @export
`[<-.jax_value` <- function(x, ..., value) {
  idx_scatter_update(x, idx_collect(...), value, lax_scatter)
}

#' Functional indexed updates
#'
#' Ports of the `x.at[idx]` methods of JAX arrays: `at_get` is `x.at[idx].get()`,
#' `at_set` is `x.at[idx].set(value)`, and likewise `add`, `subtract`, `multiply`,
#' `divide`, `power`, `min`, `max` and `apply`. The index arguments in `...` follow
#' the rules of [`[.jax_value`][jax_indexing]. None of these modify `x`; they return
#' a new array. With repeated indices all updates are applied (in an unspecified
#' order for `at_set`).
#' @param x Array.
#' @param ... Index arguments (see [jax_indexing]).
#' @param value Update values, broadcast to the indexed shape.
#' @param func Unary function applied at the indices (not differentiable).
#' @param mode Out of bounds mode: `"clip"`, `"fill"` or `"drop"`,
#'   `"promise_in_bounds"`. By default reads clamp indices and updates drop out of
#'   bounds indices.
#' @param indices_are_sorted Whether the indices are known to be sorted.
#' @param unique_indices Whether the indices are known to be free of duplicates.
#' @param fill_value Value returned for out of bounds indices with `mode = "fill"`.
#' @param wrap_negative_indices Whether negative indices count from the end.
#' @param strategy One of `"auto"`, `"gather"`, `"static_slice"`, `"dynamic_slice"`.
#' @return An array.
#' @name at
#' @export
at_get <- function(x, ..., indices_are_sorted = FALSE, unique_indices = FALSE, mode = NULL,
                   fill_value = NULL, wrap_negative_indices = TRUE, strategy = "auto") {
  idx_rewriting_take(x, idx_collect(...), indices_are_sorted = indices_are_sorted,
                     unique_indices = unique_indices, mode = mode, fill_value = fill_value,
                     normalize_indices = wrap_negative_indices, strategy = strategy)
}

#' @rdname at
#' @export
at_set <- function(x, ..., value, mode = NULL, indices_are_sorted = FALSE,
                   unique_indices = FALSE, wrap_negative_indices = TRUE) {
  idx_scatter_update(x, idx_collect(...), value, lax_scatter, indices_are_sorted,
                     unique_indices, mode, wrap_negative_indices)
}

#' @rdname at
#' @export
at_add <- function(x, ..., value, mode = NULL, indices_are_sorted = FALSE,
                   unique_indices = FALSE, wrap_negative_indices = TRUE) {
  idx_scatter_update(x, idx_collect(...), value, lax_scatter_add, indices_are_sorted,
                     unique_indices, mode, wrap_negative_indices)
}

#' @rdname at
#' @export
at_subtract <- function(x, ..., value, mode = NULL, indices_are_sorted = FALSE,
                        unique_indices = FALSE, wrap_negative_indices = TRUE) {
  idx_scatter_update(x, idx_collect(...), value, lax_scatter_sub, indices_are_sorted,
                     unique_indices, mode, wrap_negative_indices)
}

#' @rdname at
#' @export
at_multiply <- function(x, ..., value, mode = NULL, indices_are_sorted = FALSE,
                        unique_indices = FALSE, wrap_negative_indices = TRUE) {
  idx_scatter_update(x, idx_collect(...), value, lax_scatter_mul, indices_are_sorted,
                     unique_indices, mode, wrap_negative_indices)
}

#' @rdname at
#' @export
at_divide <- function(x, ..., value, mode = NULL, indices_are_sorted = FALSE,
                      unique_indices = FALSE, wrap_negative_indices = TRUE) {
  x <- as_jax(x)
  jnp_true_divide(x, idx_scatter_update(jnp_ones_like(x), idx_collect(...), value,
                                        lax_scatter_mul, indices_are_sorted, unique_indices,
                                        mode, wrap_negative_indices))
}

#' @rdname at
#' @export
at_power <- function(x, ..., value, mode = NULL, indices_are_sorted = FALSE,
                     unique_indices = FALSE, wrap_negative_indices = TRUE) {
  x <- as_jax(x)
  jnp_power(x, idx_scatter_update(jnp_ones_like(x), idx_collect(...), value, lax_scatter_mul,
                                  indices_are_sorted, unique_indices, mode,
                                  wrap_negative_indices))
}

#' @rdname at
#' @export
at_min <- function(x, ..., value, mode = NULL, indices_are_sorted = FALSE,
                   unique_indices = FALSE, wrap_negative_indices = TRUE) {
  idx_scatter_update(x, idx_collect(...), value, lax_scatter_min, indices_are_sorted,
                     unique_indices, mode, wrap_negative_indices)
}

#' @rdname at
#' @export
at_max <- function(x, ..., value, mode = NULL, indices_are_sorted = FALSE,
                   unique_indices = FALSE, wrap_negative_indices = TRUE) {
  idx_scatter_update(x, idx_collect(...), value, lax_scatter_max, indices_are_sorted,
                     unique_indices, mode, wrap_negative_indices)
}

#' @rdname at
#' @export
at_apply <- function(x, ..., func, mode = NULL, indices_are_sorted = FALSE,
                     unique_indices = FALSE, wrap_negative_indices = TRUE) {
  x <- as_jax(x)
  scatter_apply <- function(x, indices, y, dims, ...) {
    lax_scatter_apply(x, indices, func, dims, update_shape = aval_of(y)$shape, ...)
  }
  idx_scatter_update(x, idx_collect(...), lax_full(integer(0), 0, dtype(x)), scatter_apply,
                     indices_are_sorted, unique_indices, mode, wrap_negative_indices)
}

# ---- take, take_along_axis, put_along_axis --------------------------------------------------------

#' Take elements from an array
#'
#' Port of `jnp.take`.
#' @param a Array.
#' @param indices Integer array of 0-based indices.
#' @param axis Axis along which to take; `NULL` takes from the flattened array.
#' @param out Not supported.
#' @param mode Out of bounds mode: `"fill"` (default: out of bounds indices give
#'   `fill_value`), `"clip"` or `"wrap"`.
#' @param unique_indices Whether the indices are known to be free of duplicates.
#' @param indices_are_sorted Whether the indices are known to be sorted.
#' @param fill_value Fill value for `mode = "fill"`; defaults to NaN for inexact
#'   types, the minimum for signed integers, the maximum for unsigned integers and
#'   `TRUE` for booleans.
#' @return An array.
#' @export
jnp_take <- function(a, indices, axis = NULL, out = NULL, mode = NULL, unique_indices = FALSE,
                     indices_are_sorted = FALSE, fill_value = NULL) {
  if (!is.null(out))
    jax_error("The 'out' argument to jnp.take is not supported.",
              class = "jax_not_implemented_error")
  a <- as_jax(a)
  indices <- gs_as_indices(indices)
  if (is.null(axis)) {
    a <- jnp_ravel(a)
    axis_idx <- 0L
  } else {
    axis_idx <- norm_axis(axis, length(aval_of(a)$shape))
  }
  slice_sizes <- aval_of(a)$shape
  if (is.null(mode) || identical(mode, "fill")) {
    gather_mode <- "fill_or_drop"
    # lax_gather does not support negative indices, so they are wrapped here.
    indices <- lax_select(lax_lt(indices, lax_full_like(indices, 0)),
                         lax_add(indices, lax_full_like(indices, slice_sizes[axis_idx + 1L])),
                         indices)
  } else if (identical(mode, "raise")) {
    jax_error("The 'raise' mode to jnp.take is not supported.",
              class = "jax_not_implemented_error")
  } else if (identical(mode, "wrap")) {
    indices <- jnp_remainder(indices, lax_full_like(indices, slice_sizes[axis_idx + 1L]))
    gather_mode <- "promise_in_bounds"
  } else if (identical(mode, "clip")) {
    gather_mode <- "clip"
  } else {
    jax_error("Invalid mode '", mode, "' for np.take", class = "jax_value_error")
  }
  ishape <- aval_of(indices)$shape
  index_dims <- length(ishape)
  if (slice_sizes[axis_idx + 1L] == 0L) {
    if (prod(ishape) != 0)
      idx_error("Cannot do a non-empty jnp.take() from an empty axis.")
    return(a)
  }
  if (prod(ishape) == 0) {
    out_shape <- c(slice_sizes[seq_len(axis_idx)], ishape, drop_first(slice_sizes, axis_idx + 1L))
    return(lax_full_like(a, 0, shape = out_shape))
  }
  nd <- length(slice_sizes)
  slice_sizes[axis_idx + 1L] <- 1L
  dnums <- lax_GatherDimensionNumbers(
    offset_dims = c(seq_len(axis_idx) - 1L,
                    axis_idx + index_dims + seq_len(nd - axis_idx - 1L) - 1L),
    collapsed_slice_dims = axis_idx, start_index_map = axis_idx)
  lax_gather(a, lax_expand_dims(indices, index_dims), dnums, slice_sizes, mode = gather_mode,
             unique_indices = unique_indices, indices_are_sorted = indices_are_sorted,
             fill_value = fill_value)
}

# Port of jax/_src/numpy/indexing.py:_normalize_index.
idx_normalize_index <- function(index, axis_size) {
  if (is_unsigned_dtype(aval_of(index)$dtype)) return(index)
  lax_select(lax_lt(index, lax_full_like(index, 0)),
            lax_add(index, lax_full_like(index, axis_size)), index)
}

#' Take values along an axis
#'
#' Port of `jnp.take_along_axis`.
#' @param arr Array.
#' @param indices Integer array with the same number of dimensions as `arr`
#'   (broadcast against it along the other axes).
#' @param axis 0-based axis; `NULL` flattens `arr` (then `indices` must be 1-D).
#' @param mode Out of bounds mode: `"fill"` (default), `"clip"`,
#'   `"promise_in_bounds"` or `"one_hot"` (a sum over a one hot mask; out of bounds
#'   indices give zero).
#' @param fill_value Fill value for `mode = "fill"`.
#' @param wrap_negative_indices Whether negative indices count from the end.
#' @return An array.
#' @export
jnp_take_along_axis <- function(arr, indices, axis = -1L, mode = NULL, fill_value = NULL,
                                wrap_negative_indices = TRUE) {
  a <- as_jax(arr)
  indices <- gs_as_indices(indices)
  ia <- aval_of(indices)
  idx_shape <- ia$shape
  if (!is_int_dtype(ia$dtype))
    jax_error("take_along_axis indices must be of integer type, got ", ia$dtype,
              class = "jax_type_error")
  if (is.null(axis)) {
    if (length(idx_shape) != 1L)
      jax_error("take_along_axis indices must be 1D if axis=None, got shape ",
                gs_tuple(idx_shape), class = "jax_value_error")
    a <- jnp_ravel(a)
    axis <- 0L
  }
  ashape <- aval_of(a)$shape
  rank <- length(ashape)
  if (rank != length(idx_shape))
    jax_error("indices and arr must have the same number of dimensions; ", length(idx_shape),
              " vs. ", rank, class = "jax_value_error")
  axis_int <- norm_axis(axis, rank)
  axis_size <- ashape[axis_int + 1L]
  arr_shape <- ashape
  arr_shape[axis_int + 1L] <- 1L
  out_shape <- broadcast_shapes_impl(list(idx_shape, arr_shape))
  if (axis_size == 0L) return(lax_full(out_shape, 0, aval_of(a)$dtype))
  if (wrap_negative_indices) {
    indices <- lax_convert_element_type(indices, "int32")
    indices <- idx_normalize_index(indices, axis_size)
  }
  if (identical(mode, "one_hot")) return(idx_one_hot_take(a, indices, axis_int))

  index_dims <- which(seq_along(idx_shape) - 1L == axis_int | idx_shape != 1L)
  gather_indices <- lax_reshape(indices, c(out_shape[index_dims], 1L))
  slice_sizes <- offset_dims <- start_index_map <- collapsed_slice_dims <- integer(0)
  operand_batching_dims <- start_indices_batching_dims <- dims_to_squeeze <- integer(0)
  # i indexes the unsqueezed shape, new_i the squeezed shape and j the gather indices.
  new_i <- 0L
  j <- 0L
  for (i in seq_len(rank) - 1L) {
    if (i == axis_int) {
      slice_sizes <- c(slice_sizes, 1L)
      start_index_map <- c(start_index_map, new_i)
      collapsed_slice_dims <- c(collapsed_slice_dims, new_i)
      new_i <- new_i + 1L
      j <- j + 1L
    } else if (idx_shape[i + 1L] == 1L) {
      # The whole axis of arr is taken; no iota index is needed.
      offset_dims <- c(offset_dims, i)
      slice_sizes <- c(slice_sizes, arr_shape[i + 1L])
      new_i <- new_i + 1L
    } else if (arr_shape[i + 1L] == 1L) {
      # The array dimension is 1 but the index dimension is not: squeeze it.
      dims_to_squeeze <- c(dims_to_squeeze, i)
      j <- j + 1L
    } else {
      # Equal sizes: batching dimensions of both the array and the indices.
      slice_sizes <- c(slice_sizes, if (arr_shape[i + 1L] == 0L) 0L else 1L)
      operand_batching_dims <- c(operand_batching_dims, new_i)
      start_indices_batching_dims <- c(start_indices_batching_dims, j)
      new_i <- new_i + 1L
      j <- j + 1L
    }
  }
  if (length(dims_to_squeeze)) a <- lax_squeeze(a, dims_to_squeeze)
  dnums <- lax_GatherDimensionNumbers(
    offset_dims = offset_dims, collapsed_slice_dims = collapsed_slice_dims,
    start_index_map = start_index_map, operand_batching_dims = operand_batching_dims,
    start_indices_batching_dims = start_indices_batching_dims)
  lax_gather(a, gather_indices, dnums, slice_sizes, mode = if (is.null(mode)) "fill" else mode,
             fill_value = fill_value)
}

# The one_hot mode of take_along_axis: sum_z one_hot(indices)[..., z] * a[..., z, ...]
# (upstream writes the same contraction with einsum).
idx_one_hot_take <- function(a, indices, axis_int) {
  aa <- aval_of(a)
  rank <- length(aa$shape)
  axis_size <- aa$shape[axis_int + 1L]
  indices <- lax_convert_element_type(indices, "int32")
  ishape <- aval_of(indices)$shape
  classes <- lax_broadcast_in_dim(lax_iota("int32", axis_size), c(rep(1L, rank), axis_size), rank)
  hot <- lax_eq(lax_expand_dims(indices, rank), classes)
  # Move the taken axis of a to the end and give it a unit dimension at `axis_int`.
  perm <- c(setdiff(seq_len(rank) - 1L, axis_int), axis_int)
  am <- lax_expand_dims(lax_transpose(a, perm), axis_int)
  full <- c(broadcast_shapes_impl(list(ishape, aval_of(am)$shape[seq_len(rank)])), axis_size)
  hot <- jnp_broadcast_to(hot, full)
  am <- jnp_broadcast_to(am, full)
  if (aa$dtype == "bool") return(lax_reduce_or(lax_bitwise_and(hot, am), rank))
  lax_reduce_sum(lax_mul(lax_convert_element_type(hot, aa$dtype), am), rank)
}

#' Put values along an axis
#'
#' Port of `jnp.put_along_axis`. Arrays are immutable, so `inplace = FALSE` must be
#' passed; the updated array is returned.
#' @param arr Array.
#' @param indices Integer array with the same number of dimensions as `arr`.
#' @param values Values, broadcastable to the shape of `indices`.
#' @param axis 0-based axis; `NULL` flattens `arr`.
#' @param inplace Must be `FALSE`.
#' @param mode Out of bounds mode (see [at_set()]).
#' @return The updated array.
#' @export
jnp_put_along_axis <- function(arr, indices, values, axis, inplace = TRUE, mode = NULL) {
  if (inplace)
    jax_error("jax.numpy.put_along_axis cannot modify arrays in-place, because JAX arrays ",
              "are immutable. Pass inplace=FALSE to instead return an updated array.",
              class = "jax_value_error")
  arr <- as_jax(arr)
  indices <- gs_as_indices(indices)
  values <- as_jax(values)
  original_shape <- aval_of(arr)$shape
  flatten <- is.null(axis)
  if (flatten) {
    arr <- jnp_ravel(arr)
    axis <- 0L
  }
  shape <- aval_of(arr)$shape
  ishape <- aval_of(indices)$shape
  if (length(shape) != length(ishape))
    jax_error("put_along_axis arguments 'arr' and 'indices' must have same ndim. Got ",
              "arr.ndim=", length(shape), " and indices.ndim=", length(ishape), ".",
              class = "jax_value_error")
  values <- tryCatch(jnp_broadcast_to(values, ishape), jax_error = function(e) {
    jax_error("put_along_axis argument 'values' must be broadcastable to 'indices'. Got ",
              "values.shape=", gs_tuple(aval_of(values)$shape), " and indices.shape=",
              gs_tuple(ishape), ".", class = "jax_value_error")
  })
  ax <- norm_axis(axis, length(shape))
  # Sparse index grids (jnp.indices(shape, sparse = TRUE)) with `indices` along axis.
  idx <- lapply(seq_along(shape), function(k) {
    s <- rep(1L, length(shape))
    s[k] <- shape[k]
    lax_reshape(lax_iota("int32", shape[k]), s)
  })
  idx[[ax + 1L]] <- indices
  result <- idx_scatter_update(arr, idx, values, lax_scatter, mode = mode)
  if (flatten) jnp_reshape(result, original_shape) else result
}

# ---- segment reductions (port of jax/_src/ops/scatter.py) ------------------------------------------

idx_segment_update <- function(name, data, segment_ids, scatter_op, identity,
                               num_segments = NULL, indices_are_sorted = FALSE,
                               unique_indices = FALSE, bucket_size = NULL, reducer = NULL,
                               mode = NULL) {
  mode <- gs_mode_from_any(mode)
  data <- as_jax(data)
  segment_ids <- gs_as_indices(segment_ids)
  da <- aval_of(data)
  if (is.null(num_segments)) {
    ids <- concrete_or_error(segment_ids, paste0(name, "() `num_segments` argument."))
    num_segments <- max(as.vector(array_host(ids))) + 1
  }
  if (is_jax_value(num_segments))
    num_segments <- as.vector(array_host(concrete_or_error(
      num_segments, "segment_sum() `num_segments` argument.")))
  num_segments <- as.integer(num_segments)
  if (num_segments < 0L)
    jax_error("num_segments must be non-negative.", class = "jax_value_error")
  full <- function(shape) {
    z <- lax_full(as.integer(shape), 0, da$dtype)
    if (is.character(identity)) gs_int_extreme_like(z, identity) else lax_full_like(z, identity)
  }
  if (is.null(bucket_size)) {
    return(idx_scatter_update(full(c(num_segments, da$shape[-1L])), list(segment_ids), data,
                              scatter_op, indices_are_sorted, unique_indices, mode,
                              normalize_indices = FALSE))
  }
  # Bucketize the indices and reduce each bucket first, which improves the numerical
  # stability of sums and products.
  n <- aval_of(segment_ids)$shape[1L]
  num_buckets <- as.integer(ceiling(prod(aval_of(segment_ids)$shape) / bucket_size))
  buckets <- as_jax(array(as.integer((seq_len(n) - 1L) %/% as.integer(bucket_size)), n))
  out <- idx_scatter_update(full(c(num_buckets, num_segments, da$shape[-1L])),
                            list(buckets, lax_expand_dims(segment_ids, 0L)), data, scatter_op,
                            indices_are_sorted, unique_indices, mode,
                            normalize_indices = FALSE)
  lax_convert_element_type(reducer(out, axis = 0L), da$dtype)
}

idx_segment_identity <- function(dt, op) {
  if (op == "max") return(if (dt == "bool") FALSE else if (is_int_dtype(dt)) "min" else -Inf)
  if (dt == "bool") TRUE else if (is_int_dtype(dt)) "max" else Inf
}

#' Segment reductions
#'
#' Ports of `jax.ops.segment_sum`, `segment_prod`, `segment_max` and `segment_min`:
#' reduce the entries of `data` (along its first axis) that share a segment id.
#' Segments without entries hold the identity of the reduction. Ids outside
#' `[0, num_segments)` are dropped (negative ids do not wrap).
#' @param data Array.
#' @param segment_ids Integer array of 0-based segment ids, one per entry of the
#'   first axis of `data`.
#' @param num_segments Number of segments. `NULL` uses `max(segment_ids) + 1`, which
#'   requires concrete ids (so it cannot be used under [jit()]).
#' @param indices_are_sorted Whether `segment_ids` is known to be sorted.
#' @param unique_indices Whether `segment_ids` is known to be free of duplicates.
#' @param bucket_size Optional size of the buckets used to group the reduction.
#' @param mode Out of bounds mode (see [lax_GatherScatterMode]); the default drops.
#' @return An array with leading dimension `num_segments`.
#' @export
ops_segment_sum <- function(data, segment_ids, num_segments = NULL, indices_are_sorted = FALSE,
                            unique_indices = FALSE, bucket_size = NULL, mode = NULL) {
  idx_segment_update("segment_sum", data, segment_ids, lax_scatter_add, 0, num_segments,
                     indices_are_sorted, unique_indices, bucket_size, jnp_sum, mode)
}

#' @rdname ops_segment_sum
#' @export
ops_segment_prod <- function(data, segment_ids, num_segments = NULL, indices_are_sorted = FALSE,
                             unique_indices = FALSE, bucket_size = NULL, mode = NULL) {
  idx_segment_update("segment_prod", data, segment_ids, lax_scatter_mul, 1, num_segments,
                     indices_are_sorted, unique_indices, bucket_size, jnp_prod, mode)
}

#' @rdname ops_segment_sum
#' @export
ops_segment_max <- function(data, segment_ids, num_segments = NULL, indices_are_sorted = FALSE,
                            unique_indices = FALSE, bucket_size = NULL, mode = NULL) {
  idx_segment_update("segment_max", data, segment_ids, lax_scatter_max,
                     idx_segment_identity(dtype(as_jax(data)), "max"), num_segments,
                     indices_are_sorted, unique_indices, bucket_size, jnp_max, mode)
}

#' @rdname ops_segment_sum
#' @export
ops_segment_min <- function(data, segment_ids, num_segments = NULL, indices_are_sorted = FALSE,
                            unique_indices = FALSE, bucket_size = NULL, mode = NULL) {
  idx_segment_update("segment_min", data, segment_ids, lax_scatter_min,
                     idx_segment_identity(dtype(as_jax(data)), "min"), num_segments,
                     indices_are_sorted, unique_indices, bucket_size, jnp_min, mode)
}
