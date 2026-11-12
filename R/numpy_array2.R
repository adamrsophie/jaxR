# jax.numpy array functions m to z (ports of jax/_src/numpy/lax_numpy.py), jnp.place and
# jnp.put (jax/_src/numpy/indexing.py), window functions (window_functions.py),
# jnp.vectorize (vectorize.py) and the index tricks mgrid, ogrid, r_ and c_
# (index_tricks.py). Axes and indices are 0-based. Private helpers use the prefix `a2_`.
#
# Deviations: lax.split, lax.tile, lax.unstack, lax._tri and lax._delta have no R
# primitives, so they are composed from slices, broadcast_in_dim, reshape and iota
# comparisons exactly as their upstream lowerings do (values are identical).

# ---- private helpers -------------------------------------------------------------------------

# A static integer (R value or concrete jax value), as operator.index() upstream.
a2_int <- function(x, what = "this argument") {
  if (is_jax_value(x)) x <- as.vector(array_host(concrete_or_error(x, what)))
  if (!(is.numeric(x) || is.logical(x)) || length(x) != 1L || is.na(x) || x != round(x))
    jax_error(what, " must be an integer, got ", a2_fmt(x), class = "jax_type_error")
  as.integer(x)
}

a2_ints <- function(x, what = "this argument") {
  if (is_jax_value(x)) x <- as.vector(array_host(concrete_or_error(x, what)))
  x <- unlist(x)
  if (length(x) && (!(is.numeric(x) || is.logical(x)) || anyNA(x) || any(x != round(x))))
    jax_error(what, " must be integers", class = "jax_type_error")
  as.integer(x)
}

# Integer valued R doubles stand for Python ints in index like arguments: a scalar is a
# weakly typed int (so fill_value = -1 keeps an integer result) and a vector is a list of
# Python ints (default int dtype), as upstream.
a2_pyint <- function(x) {
  if (is.double(x) && !is_jax_value(x) && length(x) && all(is.finite(x)) &&
      all(x == round(x)) && all(abs(x) <= .Machine$integer.max)) {
    if (!length(r_shape(x))) return(as.integer(x))
    return(as_jax(x, default_int()))
  }
  x
}

a2_shape <- function(x) aval_of(x)$shape
a2_ndim <- function(x) length(aval_of(x)$shape)
a2_dtype <- function(x) aval_of(x)$dtype
a2_weak_int <- function() new_aval(integer(0), default_int(), TRUE)

# One line rendering of an R value for error messages.
a2_fmt <- function(x) paste(if (is_jax_value(x)) format(x) else deparse(x), collapse = " ")

a2_fmt_shape <- function(s) {
  if (length(s) == 1L) paste0("(", s, ",)") else paste0("(", paste(s, collapse = ", "), ")")
}

# Python slice semantics along one axis (negative bounds count from the end, clipped).
a2_slice <- function(x, start = NULL, stop = NULL, axis = 0L) {
  s <- a2_shape(x)
  ax <- norm_axis(axis, length(s))
  n <- s[ax + 1L]
  st <- if (is.null(start)) 0L else as.integer(start)
  sp <- if (is.null(stop)) n else as.integer(stop)
  if (st < 0L) st <- st + n
  if (sp < 0L) sp <- sp + n
  st <- min(max(st, 0L), n)
  sp <- min(max(sp, st), n)
  lax_slice_in_dim(x, st, sp, 1L, ax)
}

a2_flip <- function(x, axis) lax_rev(x, axis)

# Port of jnp.diff(a, axis=axis) with n = 1.
a2_diff <- function(x, axis = -1L) {
  hi <- a2_slice(x, 1L, NULL, axis)
  lo <- a2_slice(x, NULL, -1L, axis)
  if (a2_dtype(x) == "bool") lax_ne(hi, lo) else jnp_subtract(hi, lo)
}

# Port of lax._tri: boolean mask i + offset >= j, converted to dtype.
a2_tri <- function(dtype, shape, offset) {
  shape <- as.integer(shape)
  r <- lax_broadcasted_iota("int32", shape, 0L)
  c <- lax_broadcasted_iota("int32", shape, 1L)
  m <- lax_ge(lax_add(r, lax_full_like(r, offset)), c)
  lax_convert_element_type(m, dtype)
}

# Port of lax._delta: boolean mask that is true where all `axes` indices agree.
a2_delta <- function(shape, axes) {
  base <- shape[axes + 1L]
  iotas <- lapply(seq_along(base) - 1L, function(i) lax_broadcasted_iota("uint32", base, i))
  eyes <- Map(lax_eq, iotas[-length(iotas)], iotas[-1L])
  res <- Reduce(lax_bitwise_and, eyes)
  lax_broadcast_in_dim(res, shape, axes)
}

# Port of lax.tile via its lowering: reshape, broadcast_in_dim, reshape.
a2_tile <- function(x, reps) {
  s <- a2_shape(x)
  if (!length(s)) return(x)
  reps <- as.integer(reps)
  inter <- as.vector(rbind(1L, s))
  y <- lax_reshape(x, inter)
  y <- lax_broadcast_in_dim(y, as.vector(rbind(reps, s)), seq_along(inter) - 1L)
  lax_reshape(y, reps * s)
}

a2_tile_to_size <- function(arr, size) {
  n <- a2_shape(arr)[1L]
  if (n < size) arr <- a2_tile(arr, ceiling(size / n))
  if (a2_shape(arr)[1L] > size) a2_slice(arr, 0L, size) else arr
}

a2_atleast_2d <- function(x) {
  x <- as_jax(x)
  s <- a2_shape(x)
  if (length(s) >= 2L) x else lax_reshape(x, c(rep(1L, 2L - length(s)), s))
}

# ndmin for np.array(x, ndmin=n): prepend unit dimensions.
a2_ndmin <- function(x, n) {
  s <- a2_shape(x)
  if (length(s) >= n) x else lax_reshape(x, c(rep(1L, n - length(s)), s))
}

a2_vmap <- function(f, in_axes, out_axes = 0L) {
  force(f); force(in_axes); force(out_axes)
  vmap(f, in_axes = in_axes, out_axes = out_axes)
}

# Port of jnp.apply_along_axis (used by jnp.pad with a callable mode).
a2_apply_along_axis <- function(func1d, axis, arr, ...) {
  arr <- as_jax(arr)
  nd <- a2_ndim(arr)
  axis <- norm_axis(axis, nd)
  extra <- list(...)
  func <- function(a) do.call(func1d, c(list(a), extra))
  for (i in seq_len(nd - axis - 1L)) func <- a2_vmap(func, i, -1L)
  for (i in seq_len(axis)) func <- a2_vmap(func, 0L, 0L)
  func(arr)
}

# Port of jax/_src/numpy/array_creation.py:_linspace (array bounds and `axis`).
a2_linspace <- function(start, stop, num = 50L, endpoint = TRUE, dtype = NULL, axis = 0L) {
  num <- a2_int(num, "num")
  if (num < 0L) jax_error("Number of samples, ", num, ", must be non-negative.",
                          class = "jax_value_error")
  start <- as_jax(start); stop <- as_jax(stop)
  dt <- if (is.null(dtype)) to_inexact_dtype(result_type(start, stop)) else dtype(dtype)
  cdt <- to_inexact_dtype(dt)
  start <- lax_convert_element_type(start, cdt)
  stop <- lax_convert_element_type(stop, cdt)
  bshape <- broadcast_shapes_impl(list(a2_shape(start), a2_shape(stop)))
  bstart <- jnp_broadcast_to(start, bshape)
  bstop <- jnp_broadcast_to(stop, bshape)
  axis <- as.integer(axis)
  if (axis < 0L) axis <- length(bshape) + axis + 1L
  bounds <- append(bshape, 1L, after = axis)
  div <- if (endpoint) num - 1L else num
  if (num > 1L) {
    ishape <- rep(1L, length(bounds))
    ishape[axis + 1L] <- div
    rdt <- complex_component(cdt)
    iota <- lax_reshape(lax_iota(rdt, div), ishape)
    step <- lax_convert_element_type(lax_div(iota, lax_full_like(iota, div)), cdt)
    dv <- as_jax(div, cdt)
    out <- jnp_add(jnp_multiply(lax_reshape(bstart, bounds), lax_sub(lax_full_like(step, 1), step)),
                   jnp_multiply(jnp_true_divide(lax_reshape(bstop, bounds), dv),
                                lax_convert_element_type(iota, cdt)))
    if (endpoint) out <- lax_concatenate(list(out, lax_expand_dims(bstop, axis)), axis)
  } else if (num == 1L) {
    out <- lax_reshape(bstart, bounds)
  } else {
    bounds[axis + 1L] <- 0L
    out <- jnp_zeros(bounds, dt)
  }
  if (is_int_dtype(dt) && !is_int_dtype(a2_dtype(out))) out <- lax_floor(out)
  lax_convert_element_type(out, dt)
}

# ---- transposes ------------------------------------------------------------------------------

#' Permute, transpose and rotate axes
#'
#' Ports of `jnp.permute_dims`, `jnp.matrix_transpose`, `jnp.rot90` and
#' `jnp.rollaxis`. Axes are 0-based.
#' @param a,x,m Arrays.
#' @param axes For `jnp_permute_dims`, a permutation of the axes; for `jnp_rot90`, the
#'   two axes of the rotation plane.
#' @param k Number of 90 degree rotations.
#' @param axis Axis to roll.
#' @param start Position before which the axis is placed.
#' @return An array.
#' @name jnp_permute_dims
NULL

#' @rdname jnp_permute_dims
#' @export
jnp_permute_dims <- function(a, axes) {
  # Port of lax_numpy.py:permute_dims
  a <- as_jax(a)
  nd <- a2_ndim(a)
  ax <- norm_axis(axes, nd)
  if (length(ax) != nd || anyDuplicated(ax))
    jax_error("axes ", a2_fmt_shape(axes), " must be a permutation of the ", nd, " axes",
              class = "jax_value_error")
  lax_transpose(a, ax)
}

#' @rdname jnp_permute_dims
#' @export
jnp_matrix_transpose <- function(x) {
  x <- as_jax(x)
  nd <- a2_ndim(x)
  if (nd < 2L)
    jax_error("x must be at least two-dimensional for matrix_transpose; got ndim=", nd,
              class = "jax_value_error")
  lax_transpose(x, c(seq_len(nd - 2L) - 1L, nd - 1L, nd - 2L))
}

#' @rdname jnp_permute_dims
#' @export
jnp_rot90 <- function(m, k = 1L, axes = c(0L, 1L)) {
  # Port of lax_numpy.py:rot90
  m <- as_jax(m)
  nd <- a2_ndim(m)
  if (nd < 2L)
    jax_error("rot90 requires its first argument to have ndim at least two, but got first ",
              "argument of shape ", a2_fmt_shape(a2_shape(m)), ", which has ndim ", nd,
              class = "jax_value_error")
  ax1 <- norm_axis(axes[[1L]], nd)
  ax2 <- norm_axis(axes[[2L]], nd)
  if (ax1 == ax2) jax_error("Axes must be different", class = "jax_value_error")
  k <- a2_int(k, "k") %% 4L
  if (k == 0L) return(m)
  if (k == 2L) return(a2_flip(a2_flip(m, ax1), ax2))
  perm <- seq_len(nd) - 1L
  perm[c(ax1, ax2) + 1L] <- perm[c(ax2, ax1) + 1L]
  if (k == 1L) lax_transpose(a2_flip(m, ax2), perm) else a2_flip(lax_transpose(m, perm), ax2)
}

#' @rdname jnp_permute_dims
#' @export
jnp_rollaxis <- function(a, axis, start = 0L) {
  # Port of lax_numpy.py:rollaxis
  a <- as_jax(a)
  start <- a2_int(start, "'start' argument of jnp.rollaxis()")
  nd <- a2_ndim(a)
  axis <- norm_axis(axis, nd)
  if (!(-nd <= start && start <= nd))
    jax_error("start=", start, " must satisfy ", -nd, "<=start<=", nd, class = "jax_value_error")
  if (start < 0L) start <- start + nd
  if (start > axis) start <- start - 1L
  jnp_moveaxis(a, axis, start)
}

# ---- index conversion ------------------------------------------------------------------------

#' Convert between flat and multi-dimensional indices
#'
#' Ports of `jnp.ravel_multi_index` and `jnp.unravel_index`. Indices are 0-based.
#' @param multi_index List of integer index arrays, one per dimension.
#' @param dims,shape Integer vector giving the array shape.
#' @param mode `"raise"` (requires concrete indices), `"wrap"`, `"clip"` or `"ignore"`.
#' @param order `"C"` or `"F"`.
#' @param dtype Optional integer dtype of the result.
#' @param indices Integer array of flat indices.
#' @return `jnp_ravel_multi_index` returns an integer array; `jnp_unravel_index` returns
#'   a list of integer arrays, one per dimension (out of bound indices are clipped).
#' @name jnp_ravel_multi_index
NULL

#' @rdname jnp_ravel_multi_index
#' @export
jnp_ravel_multi_index <- function(multi_index, dims, mode = "raise", order = "C", dtype = NULL) {
  # Port of lax_numpy.py:ravel_multi_index
  if (!is.list(multi_index)) multi_index <- list(multi_index)
  dims <- vapply(dims, a2_int, 1L, what = "in `dims` argument of ravel_multi_index()")
  if (length(multi_index) != length(dims))
    jax_error("len(multi_index)=", length(multi_index), " != len(dims)=", length(dims),
              class = "jax_assertion_error")
  arrs <- lapply(multi_index, function(i) as_jax(a2_pyint(i)))
  dt <- if (is.null(dtype)) do.call(result_type, c(list(a2_weak_int()), arrs)) else dtype(dtype)
  if (!is_int_dtype(dt)) jax_error("dtype=", dt, " is not an integer type.", class = "jax_value_error")
  if (prod(dims) - 1 > iinfo(dt)$max)
    jax_error("jnp.ravel_multi_index: dtype=", dt, " not large enough to hold flattened index for ",
              "dims=", a2_fmt_shape(dims), ". Use the dtype argument to pass a suitable dtype ",
              "(e.g. dtype=np.min_scalar_type(prod(dims))).", class = "jax_value_error")
  for (index in arrs) {
    if (identical(mode, "raise"))
      concrete_or_error(index, paste0("The error occurred because ravel_multi_index was jit-compiled ",
                                      "with mode='raise'. Use mode='wrap' or mode='clip' instead."))
    if (!is_int_dtype(a2_dtype(index)))
      jax_error("only int indices permitted", class = "jax_type_error")
  }
  if (identical(mode, "raise")) {
    for (k in seq_along(arrs)) {
      v <- as.vector(array_host(arrs[[k]]))
      if (any(v < 0 | v >= dims[k]))
        jax_error("invalid entry in coordinates array", class = "jax_value_error")
    }
  } else if (identical(mode, "clip")) {
    arrs <- Map(function(i, d) jnp_minimum(jnp_maximum(0L, i), d - 1L), arrs, dims)
  } else if (identical(mode, "wrap")) {
    arrs <- Map(function(i, d) jnp_remainder(i, d), arrs, dims)
  } else if (!identical(mode, "ignore")) {
    jax_error("invalid mode='", mode, "'. Expected 'raise', 'wrap', 'clip', or 'ignore'.",
              class = "jax_value_error")
  }
  strides <- if (identical(order, "F")) cumprod(c(1, dims[-length(dims)]))
             else if (identical(order, "C")) rev(cumprod(c(1, rev(dims[-1L]))))
             else jax_error("invalid order='", order, "'. Expected 'C' or 'F'", class = "jax_value_error")
  result <- as_jax(0, dt)
  for (k in seq_along(arrs)) {
    result <- jnp_add(result, jnp_multiply(lax_convert_element_type(arrs[[k]], dt),
                                           as_jax(strides[k], dt)))
  }
  result
}

#' @rdname jnp_ravel_multi_index
#' @export
jnp_unravel_index <- function(indices, shape) {
  # Port of lax_numpy.py:unravel_index
  idx <- as_jax(a2_pyint(indices))
  shape <- as.list(if (is_jax_value(shape)) as.vector(array_host(shape)) else shape)
  if (any(vapply(shape, function(s) length(s) != 1L, NA)))
    jax_error("unravel_index: shape should be a scalar or 1D sequence.", class = "jax_value_error")
  shape <- vapply(shape, a2_int, 1L, what = "shape")
  out <- vector("list", length(shape))
  for (i in rev(seq_along(shape))) {
    s <- shape[i]
    q <- jnp_floor_divide(idx, s)
    out[[i]] <- jnp_remainder(idx, s)
    idx <- q
  }
  oob_pos <- jnp_greater(idx, 0L)
  oob_neg <- if (is_unsigned_dtype(a2_dtype(idx))) as_jax(FALSE) else jnp_less(idx, -1L)
  Map(function(s, i) jnp_where(oob_pos, s - 1L, jnp_where(oob_neg, 0L, i)), shape, out)
}

# ---- shape manipulation -----------------------------------------------------------------------

#' Resize, tile, repeat, roll and split arrays
#'
#' Ports of `jnp.resize`, `jnp.tile`, `jnp.repeat`, `jnp.roll`, `jnp.split`,
#' `jnp.vsplit`, `jnp.unstack` and `jnp.vstack`. Axes are 0-based.
#' @param a,A,ary,x Arrays.
#' @param new_shape Integer vector.
#' @param reps Integer repetitions per axis.
#' @param repeats Integer repetitions (scalar or one per element along `axis`).
#' @param axis Axis (0-based); `NULL` flattens first where JAX does.
#' @param total_repeat_length Static output length along `axis`, needed when
#'   `repeats` is traced.
#' @param shift Integer shift(s); a traced value selects the dynamic implementation.
#' @param indices_or_sections A number of equal sections, or the split indices. Pass a
#'   list (or an array) to split at a single index, since a single R number means a
#'   number of sections.
#' @param tup List of arrays (or an array, split along its first axis).
#' @param dtype Optional result dtype.
#' @return An array (`jnp_split`, `jnp_vsplit` and `jnp_unstack` return lists).
#' @name jnp_tile
NULL

#' @rdname jnp_tile
#' @export
jnp_resize <- function(a, new_shape) {
  # Port of lax_numpy.py:resize
  new_shape <- a2_ints(new_shape, "new_shape")
  if (any(new_shape < 0L))
    jax_error("all elements of `new_shape` must be non-negative", class = "jax_value_error")
  arr <- jnp_ravel(a)
  new_size <- prod(new_shape)
  n <- a2_shape(arr)
  if (n == 0L || new_size == 0) return(jnp_zeros_like(arr, shape = new_shape))
  arr <- a2_slice(jnp_tile(arr, ceiling(new_size / n)), NULL, new_size)
  lax_reshape(arr, new_shape)
}

#' @rdname jnp_tile
#' @export
jnp_tile <- function(A, reps) {
  # Port of lax_numpy.py:tile
  A <- as_jax(A)
  reps <- a2_ints(reps, "reps")
  nd <- a2_ndim(A)
  reps <- c(rep(1L, max(nd - length(reps), 0L)), reps)
  if (length(reps) > nd) A <- lax_expand_dims(A, seq_len(length(reps) - nd) - 1L)
  a2_tile(A, reps)
}

#' @rdname jnp_tile
#' @export
jnp_repeat <- function(a, repeats, axis = NULL, total_repeat_length = NULL) {
  # Port of lax_numpy.py:repeat and _repeat
  arr <- as_jax(a)
  if (is.null(axis)) { arr <- jnp_ravel(arr); axis <- 0L }
  axis <- a2_int(axis, "'axis' argument of jnp.repeat()")
  nd <- a2_ndim(arr)
  if (is.null(total_repeat_length)) {
    rep_val <- if (is_jax_value(repeats)) {
      array_host(concrete_or_error(repeats, paste0(
        "When jit-compiling jnp.repeat, the total number of repeats must be static. To fix this, ",
        "either specify a static value for `repeats`, or pass a static value to `total_repeat_length`.")))
    } else repeats
    rshape <- r_shape(rep_val)
    rv <- a2_ints(as.vector(rep_val), "repeats")
    if (!length(rshape) && nd != 0L) {
      s <- a2_shape(arr)
      axis <- norm_axis(axis, nd)
      aux <- append(s, rv, after = axis + 1L)
      out <- lax_broadcast_in_dim(arr, aux, setdiff(seq_along(aux) - 1L, axis + 1L))
      s[axis + 1L] <- s[axis + 1L] * rv
      return(lax_reshape(out, s))
    }
    if (nd != 0L) {
      n <- a2_shape(arr)[norm_axis(axis, nd) + 1L]
      if (length(rv) != 1L && length(rv) != n)
        jax_error("operands could not be broadcast together with remapped shapes [original->remapped]: (",
                  length(rv), ",) and requested shape (", n, ",)", class = "jax_value_error")
      rv <- rep_len(rv, n)
    }
    total_repeat_length <- sum(rv)
    repeats <- as_jax(rv)
  } else {
    total_repeat_length <- a2_int(total_repeat_length, "total_repeat_length")
    repeats <- jnp_ravel(a2_pyint(repeats))
    if (nd != 0L) repeats <- jnp_broadcast_to(repeats, a2_shape(arr)[norm_axis(axis, nd) + 1L])
  }
  if (nd == 0L) {
    if (identical(a2_shape(repeats), 1L)) return(jnp_full(total_repeat_length, arr))
    jax_error("`repeat` with a scalar parameter `a` is only implemented for scalar values of ",
              "the parameter `repeats`.", class = "jax_value_error")
  }
  axis <- norm_axis(axis, nd)
  if (total_repeat_length == 0L) {
    s <- a2_shape(arr)
    s[axis + 1L] <- 0L
    return(jnp_zeros(s, a2_dtype(arr)))
  }
  if (a2_shape(arr)[axis + 1L] == 0L) return(arr)
  exclusive <- at_set(jnp_roll(repeats, 1L), 0L, value = 0L)
  scatter_indices <- jnp_cumsum(exclusive)
  block <- at_add(jnp_zeros(total_repeat_length, "int32"), scatter_indices, value = 1L)
  gather_indices <- jnp_subtract(jnp_cumsum(block), 1L)
  jnp_take(arr, gather_indices, axis = axis)
}

#' @rdname jnp_tile
#' @export
jnp_roll <- function(a, shift, axis = NULL) {
  # Port of lax_numpy.py:roll, _roll_static and _roll_dynamic
  arr <- as_jax(a)
  if (is.null(axis)) return(lax_reshape(jnp_roll(jnp_ravel(arr), shift, 0L), a2_shape(arr)))
  s <- a2_shape(arr)
  axis <- norm_axis(a2_ints(axis, "axis"), length(s))
  static <- !is_tracer(shift)
  if (static) {
    sv <- if (is_jax_value(shift)) as.vector(array_host(shift)) else unlist(shift)
    static <- is.numeric(sv) && all(sv == round(sv))
  }
  if (static) {
    sv <- as.integer(sv)
    n <- max(length(axis), length(sv))
    if (!(length(axis) %in% c(1L, n) && length(sv) %in% c(1L, n)))
      jax_error("shape mismatch: objects cannot be broadcast to a single shape", class = "jax_value_error")
    axis <- rep_len(axis, n); sv <- rep_len(sv, n)
    for (k in seq_len(n)) {
      ax <- axis[k]
      len <- a2_shape(arr)[ax + 1L]
      if (len == 0L) next
      i <- (-sv[k]) %% len
      arr <- lax_concatenate(list(lax_slice_in_dim(arr, i, len, 1L, ax),
                                  lax_slice_in_dim(arr, 0L, i, 1L, ax)), ax)
    }
    return(arr)
  }
  shift <- as_jax(shift)
  ss <- a2_shape(shift)
  if (length(ss) > 1L)
    jax_error("'shift' and 'axis' arguments to roll must be scalars or 1D arrays",
              class = "jax_value_error")
  n <- max(if (length(ss)) ss else 1L, length(axis))
  shift <- jnp_broadcast_to(shift, n)
  axis <- rep_len(axis, n)
  for (k in seq_len(n)) {
    i <- axis[k]
    len <- a2_shape(arr)[i + 1L]
    x <- lax_index_in_dim(shift, k - 1L, 0L, keepdims = FALSE)
    x <- jnp_remainder(lax_convert_element_type(x, "int32"), as_jax(max(len, 1L), "int32"))
    cat2 <- lax_concatenate(list(arr, arr), i)
    arr <- lax_dynamic_slice_in_dim(cat2, jnp_subtract(as_jax(len, "int32"), x), len, axis = i)
  }
  arr
}

a2_split <- function(op, ary, ios, axis) {
  # Port of lax_numpy.py:_split
  ary <- as_jax(ary)
  axis <- a2_int(axis, paste0("in jax.numpy.", op, " argument `axis`"))
  s <- a2_shape(ary)
  axis <- norm_axis(axis, length(s))
  size <- s[axis + 1L]
  is_seq <- is.list(ios) || (is_jax_value(ios) && length(a2_shape(ios)) > 0L) ||
    (!is_jax_value(ios) && (length(ios) != 1L || !is.null(dim(ios))))
  if (is_seq) {
    idx <- vapply(as.list(if (is_jax_value(ios)) as.vector(array_host(concrete_or_error(
      ios, paste0("in jax.numpy.", op, " argument 1")))) else unlist(ios)),
      a2_int, 1L, what = paste0("in jax.numpy.", op, " argument 1"))
    idx <- ifelse(idx < 0L, idx + size, idx)
    idx <- pmin(pmax(idx, 0L), size)
    sizes <- diff(c(0L, idx, size))
  } else {
    num <- a2_int(ios, paste0("in jax.numpy.", op, " argument 1"))
    part <- size %/% num
    r <- size %% num
    if (r != 0L)
      jax_error("array split does not result in an equal division: rest is ", r,
                class = "jax_value_error")
    sizes <- rep(part, num)
  }
  if (any(sizes < 0L))
    jax_error("split sizes must be nonnegative, got sizes=", a2_fmt_shape(sizes),
              class = "jax_value_error")
  starts <- c(0L, cumsum(sizes))
  lapply(seq_along(sizes), function(k)
    lax_slice_in_dim(ary, starts[k], starts[k] + sizes[k], 1L, axis))
}

#' @rdname jnp_tile
#' @export
jnp_split <- function(ary, indices_or_sections, axis = 0L) {
  a2_split("split", ary, indices_or_sections, axis)
}

#' @rdname jnp_tile
#' @export
jnp_vsplit <- function(ary, indices_or_sections) {
  a2_split("vsplit", ary, indices_or_sections, 0L)
}

#' @rdname jnp_tile
#' @export
jnp_unstack <- function(x, axis = 0L) {
  # Port of lax.unstack via its lowering (one slice per index).
  x <- as_jax(x)
  s <- a2_shape(x)
  axis <- norm_axis(axis, length(s))
  lapply(seq_len(s[axis + 1L]) - 1L, function(i) lax_index_in_dim(x, i, axis, keepdims = FALSE))
}

#' @rdname jnp_tile
#' @export
jnp_vstack <- function(tup, dtype = NULL) {
  # Port of lax_numpy.py:vstack
  if (!is.list(tup)) tup <- jnp_unstack(tup)
  arrs <- lapply(tup, a2_atleast_2d)
  if (!length(arrs)) jax_error("Need at least one array to concatenate.", class = "jax_value_error")
  arrs <- if (is.null(dtype)) promote_dtypes(arrs) else lapply(arrs, as_jax, dtype = dtype)
  lax_concatenate(arrs, 0L)
}

# ---- creation --------------------------------------------------------------------------------

#' Coordinate grids
#'
#' Port of `jnp.meshgrid`.
#' @param ... One dimensional arrays.
#' @param copy Must be `TRUE`.
#' @param sparse Return broadcastable (sparse) grids.
#' @param indexing `"xy"` (Cartesian) or `"ij"` (matrix).
#' @return A list of arrays.
#' @export
jnp_meshgrid <- function(..., copy = TRUE, sparse = FALSE, indexing = "xy") {
  # Port of lax_numpy.py:meshgrid
  args <- lapply(list(...), as_jax)
  if (!isTRUE(copy)) jax_error("jax.numpy.meshgrid only supports copy=True", class = "jax_value_error")
  if (!indexing %in% c("xy", "ij"))
    jax_error("Valid values for indexing are 'xy' and 'ij', got ", indexing, class = "jax_value_error")
  if (any(vapply(args, a2_ndim, 1L) != 1L))
    jax_error("Arguments to jax.numpy.meshgrid must be 1D, got shapes [",
              paste(vapply(args, function(a) a2_fmt_shape(a2_shape(a)), ""), collapse = ", "), "]",
              class = "jax_value_error")
  swap <- indexing == "xy" && length(args) >= 2L
  if (swap) args[1:2] <- args[2:1]
  shape <- vapply(args, function(a) if (sparse) 1L else a2_shape(a), 1L)
  out <- lapply(seq_along(args), function(i) {
    sh <- shape
    if (sparse) sh[i] <- a2_shape(args[[i]])
    lax_broadcast_in_dim(args[[i]], sh, i - 1L)
  })
  if (swap) out[1:2] <- out[2:1]
  out
}

#' Triangular matrices
#'
#' Ports of `jnp.tri`, `jnp.tril` and `jnp.triu`.
#' @param N,M Number of rows and columns (`M` defaults to `N`).
#' @param k Diagonal offset.
#' @param dtype Dtype (default float).
#' @param m Array of rank at least two.
#' @return An array.
#' @name jnp_tri
NULL

#' @rdname jnp_tri
#' @export
jnp_tri <- function(N, M = NULL, k = 0L, dtype = NULL) {
  # Port of lax_numpy.py:tri
  dt <- if (is.null(dtype)) default_float() else dtype(dtype)
  N <- a2_int(N, "N")
  M <- if (is.null(M)) N else a2_int(M, "M")
  if (!is.numeric(k) || length(k) != 1L || k != round(k))
    jax_error("offset must be an integer, got ", a2_fmt(k), class = "jax_type_error")
  a2_tri(dt, c(N, M), as.integer(k))
}

a2_tri_mask <- function(name, m, k) {
  m <- as_jax(m)
  s <- a2_shape(m)
  if (length(s) < 2L)
    jax_error("Argument to jax.numpy.", name, " must be at least 2D", class = "jax_value_error")
  k <- a2_int(k, "k")
  nm <- utils::tail(s, 2L)
  mask <- a2_tri("bool", nm, if (name == "tril") k else k - 1L)
  list(m = m, which = lax_broadcast(mask, utils::head(s, -2L)))
}

#' @rdname jnp_tri
#' @export
jnp_tril <- function(m, k = 0L) {
  r <- a2_tri_mask("tril", m, k)
  lax_select(r$which, r$m, jnp_zeros_like(r$m))
}

#' @rdname jnp_tri
#' @export
jnp_triu <- function(m, k = 0L) {
  r <- a2_tri_mask("triu", m, k)
  lax_select(r$which, jnp_zeros_like(r$m), r$m)
}

#' Vandermonde matrix
#'
#' Port of `jnp.vander`.
#' @param x One dimensional array.
#' @param N Number of columns (default `length(x)`).
#' @param increasing Powers increase from left to right.
#' @return An array of shape `(length(x), N)`.
#' @export
jnp_vander <- function(x, N = NULL, increasing = FALSE) {
  # Port of lax_numpy.py:vander
  x <- as_jax(x)
  if (a2_ndim(x) != 1L) jax_error("x must be a one-dimensional array", class = "jax_value_error")
  N <- if (is.null(N)) a2_shape(x) else a2_int(N, "'N' argument of jnp.vander()")
  if (N < 0L) jax_error("N must be nonnegative", class = "jax_value_error")
  iota <- lax_iota(a2_dtype(x), N)
  if (!increasing) iota <- lax_sub(lax_full_like(iota, N - 1L), iota)
  # Upstream passes the (n, 1) and (1, N) operands to pow directly; the pow JVP in
  # R/lax.R needs equal shapes, so both are broadcast first (values are identical).
  shape <- c(a2_shape(x), N)
  jnp_power(lax_broadcast_in_dim(x, shape, 0L), lax_broadcast_in_dim(iota, shape, 1L))
}

# ---- indices of triangles and masks ----------------------------------------------------------

a2_triu_size <- function(n, m, k) {
  # Port of lax_numpy.py:_triu_size
  if (k < 0L) return(n * m - a2_triu_size(m, n, 1L - k))
  if (k >= m) return(0L)
  mk <- min(n, m - k)
  as.integer((mk * (mk + 1L)) %/% 2L + mk * (m - k - mk))
}

#' Indices of triangles and masks
#'
#' Ports of `jnp.triu_indices`, `jnp.tril_indices`, `jnp.triu_indices_from`,
#' `jnp.tril_indices_from` and `jnp.mask_indices`. Indices are 0-based.
#' @param n,m Number of rows and columns (`m` defaults to `n`).
#' @param k Diagonal offset.
#' @param arr Two dimensional array (or anything with a shape).
#' @param mask_func Function `(array, k)` returning a mask array, such as [jnp_triu()].
#' @param size Static number of indices to return (see [jnp_nonzero()]).
#' @return A list of two integer arrays.
#' @name jnp_triu_indices
NULL

#' @rdname jnp_triu_indices
#' @export
jnp_triu_indices <- function(n, k = 0L, m = NULL) {
  n <- a2_int(n, "n argument of jnp.triu_indices")
  k <- a2_int(k, "k argument of jnp.triu_indices")
  m <- if (is.null(m)) n else a2_int(m, "m argument of jnp.triu_indices")
  jnp_nonzero(jnp_triu(jnp_ones(c(n, m)), k = k), size = a2_triu_size(n, m, k))
}

#' @rdname jnp_triu_indices
#' @export
jnp_tril_indices <- function(n, k = 0L, m = NULL) {
  n <- a2_int(n, "n argument of jnp.triu_indices")
  k <- a2_int(k, "k argument of jnp.triu_indices")
  m <- if (is.null(m)) n else a2_int(m, "m argument of jnp.triu_indices")
  jnp_nonzero(jnp_tril(jnp_ones(c(n, m)), k = k), size = a2_triu_size(m, n, -k))
}

a2_shape_of <- function(arr) {
  if (inherits(arr, "jax_aval")) return(arr$shape)
  if (is.list(arr) && !is.null(arr$shape)) return(as.integer(arr$shape))
  a2_shape(as_jax(arr))
}

#' @rdname jnp_triu_indices
#' @export
jnp_triu_indices_from <- function(arr, k = 0L) {
  s <- a2_shape_of(arr)
  if (length(s) != 2L) jax_error("Only 2-D inputs are accepted", class = "jax_value_error")
  jnp_triu_indices(s[1L], k = k, m = s[2L])
}

#' @rdname jnp_triu_indices
#' @export
jnp_tril_indices_from <- function(arr, k = 0L) {
  s <- a2_shape_of(arr)
  if (length(s) != 2L) jax_error("Only 2-D inputs are accepted", class = "jax_value_error")
  jnp_tril_indices(s[1L], k = k, m = s[2L])
}

#' @rdname jnp_triu_indices
#' @export
jnp_mask_indices <- function(n, mask_func, k = 0L, size = NULL) {
  jnp_nonzero(mask_func(jnp_ones(c(n, n)), k), size = size)
}

# ---- nonzero, place, put -----------------------------------------------------------------------

#' Indices of nonzero elements
#'
#' Port of `jnp.nonzero` (and of the one argument form of `jnp.where`). Without `size`
#' the number of nonzero elements must be concrete, so `size` is required under
#' [jit()] and other transformations; extra slots are filled with `fill_value`
#' (default 0) and extra nonzero entries are dropped.
#' @param a Array of rank at least one.
#' @param size Optional static number of indices per dimension.
#' @param fill_value Scalar, or a list with one scalar per dimension.
#' @return A list of integer arrays, one per dimension (0-based).
#' @export
jnp_nonzero <- function(a, size = NULL, fill_value = NULL) {
  # Port of lax_numpy.py:nonzero and hijax.py:nonzero / _nonzero_impl (all axes).
  arr <- as_jax(a)
  s <- a2_shape(arr)
  nd <- length(s)
  if (!nd)
    jax_error("Calling nonzero on 0d arrays is not allowed. Use jnp.atleast_1d(scalar).nonzero() instead.",
              class = "jax_value_error")
  mask <- if (a2_dtype(arr) == "bool") arr else lax_ne(arr, lax_full_like(arr, 0))
  what <- paste0("The size argument of jnp.nonzero must be statically specified to use ",
                 "jnp.nonzero within JAX transformations.")
  size <- if (is.null(size)) a2_int(jnp_sum(mask), what) else a2_int(size, what)
  if (size < 0L) jax_error("size must be a positive integer; got size=", size, class = "jax_value_error")
  fvs <- NULL
  if (!is.null(fill_value)) {
    fvs <- if (is.list(fill_value)) fill_value else rep(list(fill_value), nd)
    fvs <- lapply(fvs, function(v) as_jax(a2_pyint(v)))
  }
  dt <- do.call(result_type, c(list(default_int()), fvs))
  if (!is.null(fill_value)) {
    if (length(fvs) != nd)
      jax_error("fill_value tuple must have length equal to number of axes (", nd, "); got ",
                length(fvs), class = "jax_value_error")
    fvs <- lapply(fvs, as_jax, dtype = dt)
    if (any(vapply(fvs, a2_ndim, 1L) != 0L))
      jax_error("fill_value must be a scalar or tuple of scalars; got ", a2_fmt(fill_value),
                class = "jax_value_error")
  }
  if (prod(s) == 0 || size == 0L) return(rep(list(jnp_zeros(size, dt)), nd))
  strides <- rev(cumprod(rev(s))) %/% s
  flat <- lax_reshape(mask, prod_int(s))
  cs <- jnp_cumsum(flat, axis = -1L)
  bincount <- at_add(jnp_zeros(size, a2_dtype(cs)), cs, value = 1L, mode = "drop")
  flat_indices <- jnp_cumsum(bincount, axis = -1L)
  counts <- jnp_sum(flat, axis = -1L, keepdims = TRUE)
  fill_mask <- jnp_greater_equal(jnp_arange(size), counts)
  lapply(seq_len(nd), function(i) {
    entry <- lax_convert_element_type(
      jnp_remainder(jnp_floor_divide(flat_indices, as.integer(strides[i])), s[i]), dt)
    fv <- if (is.null(fvs)) 0L else lax_expand_dims(fvs[[i]], 0L)
    jnp_where(fill_mask, fv, entry)
  })
}

#' Replace elements by mask or flat index
#'
#' Ports of `jnp.place` and `jnp.put`. JAX arrays are immutable, so `inplace` must be
#' `FALSE` and the updated array is returned.
#' @param arr,a Array to update.
#' @param mask Boolean array with as many elements as `arr`.
#' @param vals,v Values (repeated as needed).
#' @param ind Flat 0-based indices into `a`.
#' @param mode `NULL` (out of bound indices are dropped), `"clip"` or `"wrap"`.
#' @param inplace Must be `FALSE`.
#' @return An array.
#' @name jnp_place
NULL

#' @rdname jnp_place
#' @export
jnp_place <- function(arr, mask, vals, inplace = TRUE) {
  # Port of indexing.py:place
  data <- as_jax(arr)
  mask_arr <- as_jax(mask)
  vals_arr <- jnp_ravel(vals)
  if (inplace)
    jax_error("jax.numpy.place cannot modify arrays in-place, because JAX arrays are immutable. ",
              "Pass inplace=False to instead return an updated array.", class = "jax_value_error")
  n <- prod_int(a2_shape(data))
  if (n != prod_int(a2_shape(mask_arr)))
    jax_error("place: arr and mask must be the same size", class = "jax_value_error")
  if (!prod_int(a2_shape(vals_arr)))
    jax_error("Cannot place values from an empty array", class = "jax_value_error")
  if (!n) return(data)
  indices <- jnp_nonzero(jnp_ravel(mask_arr), size = n, fill_value = n)[[1L]]
  vals_arr <- a2_tile_to_size(vals_arr, n)
  lax_reshape(at_set(jnp_ravel(data), indices, value = vals_arr, mode = "drop"), a2_shape(data))
}

#' @rdname jnp_place
#' @export
jnp_put <- function(a, ind, v, mode = NULL, inplace = TRUE) {
  # Port of indexing.py:put
  if (inplace)
    jax_error("jax.numpy.put cannot modify arrays in-place, because JAX arrays are immutable. ",
              "Pass inplace=False to instead return an updated array.", class = "jax_value_error")
  arr <- as_jax(a)
  ind_arr <- jnp_ravel(a2_pyint(ind))
  v_arr <- jnp_ravel(v)
  n <- prod_int(a2_shape(arr))
  ni <- prod_int(a2_shape(ind_arr))
  if (!n || !ni || !prod_int(a2_shape(v_arr))) return(arr)
  v_arr <- a2_tile_to_size(v_arr, ni)
  if (is.null(mode)) {
    smode <- "drop"
  } else if (identical(mode, "clip")) {
    ind_arr <- jnp_minimum(jnp_maximum(0L, ind_arr), n - 1L)
    smode <- "promise_in_bounds"
  } else if (identical(mode, "wrap")) {
    ind_arr <- jnp_remainder(ind_arr, n)
    smode <- "promise_in_bounds"
  } else if (identical(mode, "raise")) {
    jax_error("The 'raise' mode to jnp.put is not supported.", class = "jax_not_implemented_error")
  } else {
    jax_error("mode should be one of 'wrap' or 'clip'; got mode='", mode, "'", class = "jax_value_error")
  }
  idx <- jnp_unravel_index(ind_arr, a2_shape(arr))
  do.call(at_set, c(list(arr), idx, list(value = v_arr, mode = smode)))
}

# ---- elementwise and selection ----------------------------------------------------------------

#' Select from several choices
#'
#' Port of `jnp.select`: where several conditions hold, the first wins.
#' @param condlist List of boolean arrays.
#' @param choicelist List of arrays, one per condition.
#' @param default Value where no condition holds.
#' @return An array.
#' @export
jnp_select <- function(condlist, choicelist, default = 0L) {
  # Port of lax_numpy.py:select
  if (!is.list(condlist)) condlist <- jnp_unstack(condlist)
  if (!is.list(choicelist)) choicelist <- jnp_unstack(choicelist)
  if (length(condlist) != length(choicelist))
    jax_error("condlist must have length equal to choicelist (", length(condlist), " vs ",
              length(choicelist), ")", class = "jax_value_error")
  if (!length(condlist)) jax_error("condlist must be non-empty", class = "jax_value_error")
  conds <- lapply(condlist, as_jax)
  choices <- promote_dtypes(c(list(as_jax(default)), lapply(choicelist, as_jax)))
  conditions <- jnp_stack(do.call(jnp_broadcast_arrays, c(list(FALSE), conds)))
  idx <- jnp_argmax(lax_convert_element_type(conditions, "bool"), axis = 0L)
  # lax_select_n in R/lax_shape.R only accepts int32 `which` (upstream accepts any
  # integer type), so the default int index is narrowed; the values are unchanged.
  idx <- lax_convert_element_type(idx, "int32")
  do.call(lax_select_n, do.call(jnp_broadcast_arrays, c(list(idx), choices)))
}

#' Piecewise defined function
#'
#' Port of `jnp.piecewise`. Each function is evaluated on every element (through
#' [jnp_vectorize()] and [lax_switch()]); where several conditions hold the last wins.
#' @param x Array.
#' @param condlist List of boolean arrays (or a boolean array whose first axis indexes
#'   the conditions).
#' @param funclist List of functions or constants: one per condition, plus optionally a
#'   default as the last entry.
#' @param ... Extra arguments passed to the functions.
#' @return An array with the dtype of `x`.
#' @export
jnp_piecewise <- function(x, condlist, funclist, ...) {
  # Port of lax_numpy.py:piecewise and _piecewise
  x <- as_jax(x)
  nc <- if (is.list(condlist)) length(condlist) else a2_shape(as_jax(condlist))[1L]
  nf <- length(funclist)
  if (nf == nc + 1L) {
    funclist <- c(funclist[nf], funclist[-nf])
  } else if (nf == nc) {
    funclist <- c(list(0L), as.list(funclist))
  } else {
    jax_error("with ", nc, " condition(s), either ", nc, " or ", nc + 1L,
              " functions are expected; got ", nf, class = "jax_value_error")
  }
  cl <- jnp_asarray(if (is.list(condlist)) condlist else as_jax(condlist), dtype = "bool")
  dt <- a2_dtype(x)
  zero <- jnp_zeros_like(a2_slice(cl, NULL, 1L, 0L))
  indices <- jnp_argmax(jnp_cumsum(lax_concatenate(list(zero, cl), 0L), axis = 0L), axis = 0L)
  extra <- list(...)
  funcs <- lapply(funclist, function(f) {
    if (is.function(f)) {
      force(f)
      function(xv) lax_convert_element_type(do.call(f, c(list(xv), extra)), dt)
    } else {
      v <- f
      function(xv) as_jax(v, dt)
    }
  })
  jnp_vectorize(function(i, fl, xv) lax_switch(i, fl, xv), excluded = 1L)(indices, funcs, x)
}

#' Replace NaN and infinities
#'
#' Port of `jnp.nan_to_num`.
#' @param x Array.
#' @param copy Ignored.
#' @param nan Replacement for NaN.
#' @param posinf,neginf Replacements for infinities (default: the largest and smallest
#'   finite values of the dtype).
#' @return An array.
#' @export
jnp_nan_to_num <- function(x, copy = TRUE, nan = 0, posinf = NULL, neginf = NULL) {
  # Port of lax_numpy.py:nan_to_num
  x <- as_jax(x)
  dt <- a2_dtype(x)
  if (!is_inexact_dtype(dt)) return(x)
  if (is_complex_dtype(dt))
    return(lax_complex(jnp_nan_to_num(lax_real(x), nan = nan, posinf = posinf, neginf = neginf),
                       jnp_nan_to_num(lax_imag(x), nan = nan, posinf = posinf, neginf = neginf)))
  info <- finfo(dt)
  posinf <- posinf %||% info$max
  neginf <- neginf %||% info$min
  out <- jnp_where(jnp_isnan(x), as_jax(nan, dt), x)
  out <- jnp_where(lax_eq(out, lax_full_like(out, Inf)), as_jax(posinf, dt), out)
  jnp_where(lax_eq(out, lax_full_like(out, -Inf)), as_jax(neginf, dt), out)
}

#' Unwrap a phase signal
#'
#' Port of `jnp.unwrap`.
#' @param p Array.
#' @param discont Maximum discontinuity (default `period / 2`).
#' @param axis Axis (0-based).
#' @param period Period of the signal.
#' @return An array.
#' @export
jnp_unwrap <- function(p, discont = NULL, axis = -1L, period = 2 * pi) {
  # Port of lax_numpy.py:unwrap
  pp <- promote_dtypes(list(p, period))
  p <- pp[[1L]]; period <- pp[[2L]]
  dt <- a2_dtype(p)
  if (is_complex_dtype(dt)) jax_error("jnp.unwrap does not support complex inputs.", class = "jax_value_error")
  axis <- norm_axis(axis, a2_ndim(p))
  if (a2_shape(p)[axis + 1L] == 0L) return(p)
  if (is.null(discont)) discont <- jnp_true_divide(period, 2L)
  interval <- if (is_int_dtype(dt)) jnp_floor_divide(period, 2L) else jnp_true_divide(period, 2L)
  dd <- a2_diff(p, axis)
  ddmod <- jnp_subtract(jnp_remainder(jnp_add(dd, interval), period), interval)
  ddmod <- jnp_where(jnp_logical_and(jnp_equal(ddmod, jnp_negative(interval)), jnp_greater(dd, 0L)),
                     interval, ddmod)
  ph_correct <- jnp_where(jnp_less(jnp_abs(dd), discont), 0L, jnp_subtract(ddmod, dd))
  jnp_concatenate(list(a2_slice(p, 0L, 1L, axis),
                       jnp_add(a2_slice(p, 1L, NULL, axis), jnp_cumsum(ph_correct, axis = axis))),
                  axis)
}

#' Trim leading and trailing zeros
#'
#' Port of `jnp.trim_zeros`. The input must be concrete (not traced).
#' @param filt Array.
#' @param trim `"f"`, `"b"` or `"fb"` (front, back or both).
#' @param axis Axes to trim (default all).
#' @return An array.
#' @export
jnp_trim_zeros <- function(filt, trim = "fb", axis = NULL) {
  # Port of lax_numpy.py:trim_zeros
  filt <- as_jax(filt)
  h <- array_host(concrete_or_error(filt, "Error arose in the `filt` argument of trim_zeros()"))
  s <- a2_shape(filt)
  nd <- length(s)
  axes <- if (is.null(axis)) seq_len(nd) - 1L else unique(norm_axis(axis, nd))
  tr <- tolower(trim)
  front <- grepl("f", tr, fixed = TRUE)
  back <- grepl("b", tr, fixed = TRUE)
  if (!length(axes) || !(front || back)) return(filt)
  # NaN differs from zero, so it is kept.
  nz <- array(is.na(as.vector(h)) | as.vector(h) != 0, dim = s)
  starts <- integer(nd)
  stops <- s
  for (ax in seq_len(nd) - 1L) {
    if (!ax %in% axes) next
    mask <- if (nd == 1L) as.vector(nz) else apply(nz, ax + 1L, any)
    if (!any(mask)) {
      starts[ax + 1L] <- 0L; stops[ax + 1L] <- 0L
      next
    }
    if (front) starts[ax + 1L] <- which(mask)[1L] - 1L
    if (back) stops[ax + 1L] <- s[ax + 1L] - (which(rev(mask))[1L] - 1L)
  }
  lax_slice(filt, starts, stops)
}

# ---- reductions over a structure --------------------------------------------------------------

#' Sum along diagonals
#'
#' Port of `jnp.trace`.
#' @param a Array of rank at least two.
#' @param offset Diagonal offset.
#' @param axis1,axis2 Axes of the matrices (0-based).
#' @param dtype Optional accumulation dtype.
#' @param out Not supported.
#' @return An array.
#' @export
jnp_trace <- function(a, offset = 0L, axis1 = 0L, axis2 = 1L, dtype = NULL, out = NULL) {
  # Port of lax_numpy.py:trace
  a <- as_jax(a)
  if (!is.null(out))
    jax_error("The 'out' argument to jnp.trace is not supported.", class = "jax_not_implemented_error")
  nd <- a2_ndim(a)
  if (norm_axis(axis1, nd) == norm_axis(axis2, nd))
    jax_error("axis1 and axis2 can not be same. axis1=", axis1, " and axis2=", axis2,
              class = "jax_value_error")
  s <- a2_shape(a)
  a <- jnp_moveaxis(a, c(axis1, axis2), c(-2L, -1L))
  eye <- jnp_eye(s[norm_axis(axis1, nd) + 1L], s[norm_axis(axis2, nd) + 1L],
                 k = a2_int(offset, "offset"), dtype = "bool")
  a <- jnp_where(eye, a, jnp_zeros_like(a))
  jnp_sum(a, axis = c(-2L, -1L), dtype = dtype)
}

#' Trapezoidal integration
#'
#' Port of `jnp.trapezoid`.
#' @param y Array of values.
#' @param x Optional sample points (default spacing `dx`).
#' @param dx Spacing when `x` is `NULL`.
#' @param axis Axis of integration (0-based).
#' @return An array.
#' @export
jnp_trapezoid <- function(y, x = NULL, dx = 1.0, axis = -1L) {
  # Port of lax_numpy.py:trapezoid
  if (is.null(x)) {
    y_arr <- promote_dtypes_inexact(list(y))[[1L]]
    dx_array <- as_jax(dx)
    if (a2_ndim(dx_array) > 0L) {
      dx_array <- lax_expand_dims(dx_array, seq_len(a2_ndim(y_arr) - a2_ndim(dx_array)) - 1L)
      dx_array <- jnp_moveaxis(dx_array, axis, -1L)
    }
  } else {
    yx <- promote_dtypes_inexact(list(y, x))
    y_arr <- yx[[1L]]; x_arr <- yx[[2L]]
    dx_array <- if (a2_ndim(x_arr) == 1L) a2_diff(x_arr) else jnp_moveaxis(a2_diff(x_arr, axis), axis, -1L)
  }
  y_arr <- jnp_moveaxis(y_arr, axis, -1L)
  last <- function(v, i) lax_index_in_dim(v, i, -1L, keepdims = FALSE)
  dnd <- a2_ndim(dx_array)
  if (dnd == 0L || utils::tail(a2_shape(dx_array), 1L) == 1L) {
    dx_reduced <- if (dnd == 0L) dx_array else last(dx_array, 0L)
    n <- utils::tail(a2_shape(y_arr), 1L)
    return(jnp_multiply(dx_reduced, jnp_subtract(
      jnp_sum(y_arr, axis = -1L),
      jnp_multiply(0.5, jnp_add(last(y_arr, 0L), last(y_arr, n - 1L))))))
  }
  jnp_multiply(0.5, jnp_sum(jnp_multiply(dx_array, jnp_add(a2_slice(y_arr, 1L, NULL, -1L),
                                                         a2_slice(y_arr, NULL, -1L, -1L))),
                            axis = -1L))
}

# ---- bits ------------------------------------------------------------------------------------

#' Pack and unpack bits
#'
#' Ports of `jnp.packbits` and `jnp.unpackbits` (uint8 results).
#' @param a Integer or boolean array (`jnp_unpackbits` requires uint8).
#' @param axis Axis (0-based) or `NULL` for the flattened array.
#' @param count Number of bits to keep (padding with zeros if larger).
#' @param bitorder `"big"` or `"little"`.
#' @return A uint8 array.
#' @name jnp_packbits
NULL

#' @rdname jnp_packbits
#' @export
jnp_packbits <- function(a, axis = NULL, bitorder = "big") {
  # Port of lax_numpy.py:packbits
  arr <- as_jax(a)
  dt <- a2_dtype(arr)
  if (!(is_int_dtype(dt) || dt == "bool"))
    jax_error("Expected an input array of integer or boolean data type", class = "jax_type_error")
  if (!bitorder %in% c("little", "big"))
    jax_error("'order' must be either 'little' or 'big'", class = "jax_value_error")
  arr <- lax_convert_element_type(lax_ne(arr, lax_full_like(arr, 0)), "uint8")
  bits <- lax_iota("uint8", 8L)
  if (bitorder == "big") bits <- lax_rev(bits, 0L)
  if (is.null(axis)) { arr <- jnp_ravel(arr); axis <- 0L }
  arr <- jnp_swapaxes(arr, axis, -1L)
  s <- a2_shape(arr)
  nd <- length(s)
  rem <- s[nd] %% 8L
  if (rem) arr <- lax_pad(arr, as_jax(0L, "uint8"), c(rep(list(c(0L, 0L, 0L)), nd - 1L),
                                                       list(c(0L, 8L - rem, 0L))))
  s <- a2_shape(arr)
  arr <- lax_reshape(arr, c(s[-nd], s[nd] %/% 8L, 8L))
  bits <- lax_reshape(bits, c(rep(1L, nd), 8L))
  packed <- lax_convert_element_type(jnp_sum(lax_shift_left(arr, jnp_broadcast_to(bits, a2_shape(arr))),
                                             axis = -1L), "uint8")
  jnp_swapaxes(packed, axis, -1L)
}

#' @rdname jnp_packbits
#' @export
jnp_unpackbits <- function(a, axis = NULL, count = NULL, bitorder = "big") {
  # Port of lax_numpy.py:unpackbits
  arr <- as_jax(a)
  if (a2_dtype(arr) != "uint8")
    jax_error("Expected an input array of unsigned byte data type", class = "jax_type_error")
  if (!bitorder %in% c("little", "big"))
    jax_error("'order' must be either 'little' or 'big'", class = "jax_value_error")
  bits <- lax_shift_left(as_jax(rep(1L, 8L), "uint8"), lax_iota("uint8", 8L))
  if (bitorder == "big") bits <- lax_rev(bits, 0L)
  if (is.null(axis)) { arr <- jnp_ravel(arr); axis <- 0L }
  arr <- jnp_swapaxes(arr, axis, -1L)
  nd <- a2_ndim(arr)
  ex <- lax_expand_dims(arr, nd)
  b <- jnp_broadcast_to(lax_reshape(bits, c(rep(1L, nd), 8L)), a2_shape(ex) * c(rep(1L, nd), 8L))
  unpacked <- lax_convert_element_type(
    lax_gt(lax_bitwise_and(jnp_broadcast_to(ex, a2_shape(b)), b), lax_full_like(b, 0)), "uint8")
  s <- a2_shape(unpacked)
  unpacked <- lax_reshape(unpacked, c(utils::head(s, -2L), prod_int(utils::tail(s, 2L))))
  if (!is.null(count)) {
    count <- a2_int(count, "count")
    s <- a2_shape(unpacked)
    n <- s[length(s)]
    if (count > n) {
      unpacked <- lax_pad(unpacked, as_jax(0L, "uint8"),
                          c(rep(list(c(0L, 0L, 0L)), length(s) - 1L), list(c(0L, count - n, 0L))))
    } else {
      unpacked <- a2_slice(unpacked, NULL, count, -1L)
    }
  }
  jnp_swapaxes(unpacked, axis, -1L)
}

# ---- pad -------------------------------------------------------------------------------------

# Port of lax_numpy.py:_broadcast_to_pairs: a list of nd (before, after) pairs.
a2_broadcast_to_pairs <- function(nvals, nd, name) {
  conc <- function(v) {
    if (is_jax_value(v)) as.vector(array_host(concrete_or_error(v, paste0(name, " argument of jnp.pad"))))
    else v
  }
  if (is.list(nvals)) {
    parts <- lapply(nvals, function(p) as.vector(unlist(lapply(if (is.list(p)) p else list(p), conc))))
    lens <- vapply(parts, length, 1L)
    if (length(unique(lens)) > 1L)
      jax_error("`", name, "` entries must be the same shape: ", a2_fmt(nvals), class = "jax_type_error")
    v <- unlist(parts)
    # A list of scalars is a vector (a Python list of ints), a list of pairs a matrix.
    shp <- if (!length(parts) || all(lens == 1L)) length(parts) else c(length(parts), lens[1L])
    mat <- if (length(shp) == 2L) matrix(v, shp[1L], shp[2L], byrow = TRUE) else v
  } else {
    v <- conc(nvals)
    shp <- if (is.null(dim(v))) (if (length(v) == 1L) integer(0) else length(v)) else dim(v)
    mat <- v
  }
  pair <- function(a, b) c(a, b)
  if (identical(as.integer(shp), c(nd, 2L))) {
    m <- matrix(as.vector(if (is.matrix(mat)) t(mat) else mat), ncol = 2L, byrow = TRUE)
    return(lapply(seq_len(nd), function(i) m[i, ]))
  }
  if (identical(as.integer(shp), c(1L, 2L))) return(rep(list(as.vector(mat)[1:2]), nd))
  if (identical(as.integer(shp), 2L)) return(rep(list(as.vector(mat)[1:2]), nd))
  if (identical(as.integer(shp), 1L) || !length(shp)) return(rep(list(rep(as.vector(mat)[1L], 2L)), nd))
  jax_error("jnp.pad: ", name, " with nd=", nd, " has unsupported shape ", a2_fmt_shape(shp),
            ". Valid shapes are (", nd, ", 2), (1, 2), (2,), (1,), or ().", class = "jax_value_error")
}

a2_check_no_padding <- function(p, mode) {
  if (p[1L] > 0 || p[2L] > 0)
    jax_error("Cannot apply '", mode, "' padding to empty axis", class = "jax_value_error")
}

a2_pad_constant <- function(array, pad_width, constant_values) {
  nd <- a2_ndim(array)
  cv <- lax_convert_element_type(constant_values, a2_dtype(array), weak_type = aval_of(array)$weak)
  cnd <- a2_ndim(cv)
  if (cnd == 0L) return(lax_pad(array, cv, lapply(pad_width, function(p) c(p, 0L))))
  if (cnd == 1L) {
    last <- utils::tail(a2_shape(cv), 1L)
    if (last == 1L) return(lax_pad(array, lax_reshape(cv, integer(0)),
                                   lapply(pad_width, function(p) c(p, 0L))))
    if (last != 2L)
      jax_error("jnp.pad: constant_values has unsupported shape ", a2_fmt_shape(a2_shape(cv)),
                ". If the shape is 1D or 2D, the last dimension must be of size 1 or 2.",
                class = "jax_value_error")
  }
  cv <- jnp_broadcast_to(cv, c(nd, 2L))
  for (i in seq_len(nd)) {
    widths <- rep(list(c(0L, 0L, 0L)), nd)
    if (pad_width[[i]][1L] != 0L) {
      widths[[i]] <- c(pad_width[[i]][1L], 0L, 0L)
      array <- lax_pad(array, lax_reshape(lax_slice(cv, c(i - 1L, 0L), c(i, 1L)), integer(0)), widths)
    }
    if (pad_width[[i]][2L] != 0L) {
      widths[[i]] <- c(0L, pad_width[[i]][2L], 0L)
      array <- lax_pad(array, lax_reshape(lax_slice(cv, c(i - 1L, 1L), c(i, 2L)), integer(0)), widths)
    }
  }
  array
}

a2_pad_wrap <- function(array, pad_width) {
  for (i in seq_len(a2_ndim(array))) {
    size <- a2_shape(array)[i]
    if (size == 0L) { a2_check_no_padding(pad_width[[i]], "wrap"); next }
    lrep <- pad_width[[i]][1L] %/% size; lrem <- pad_width[[i]][1L] %% size
    rrep <- pad_width[[i]][2L] %/% size; rrem <- pad_width[[i]][2L] %% size
    parts <- list()
    if (lrem > 0L) parts <- c(parts, list(lax_slice_in_dim(array, size - lrem, size, 1L, i - 1L)))
    parts <- c(parts, rep(list(array), lrep + rrep + 1L))
    if (rrem > 0L) parts <- c(parts, list(lax_slice_in_dim(array, 0L, rrem, 1L, i - 1L)))
    array <- lax_concatenate(parts, i - 1L)
  }
  array
}

a2_pad_symmetric_or_reflect <- function(array, pad_width, mode, reflect_type) {
  if (!reflect_type %in% c("even", "odd"))
    jax_error("reflect_type must be 'even' or 'odd', got '", reflect_type, "'", class = "jax_value_error")
  for (i in seq_len(a2_ndim(array))) {
    ax <- i - 1L
    axis_size <- a2_shape(array)[i]
    if (axis_size == 0L) { a2_check_no_padding(pad_width[[i]], mode); next }
    if (pad_width[[i]][1L] == 0L && pad_width[[i]][2L] == 0L) next
    build <- function(array, padding, before) {
      edge <- if (before) a2_slice(array, 0L, 1L, ax) else a2_slice(array, -1L, NULL, ax)
      offset <- if (mode == "reflect" && axis_size > 1L) 1L else 0L
      while (padding > 0L) {
        curr <- min(padding, axis_size - offset)
        padding <- padding - curr
        x <- if (before) {
          a2_slice(array, offset, offset + curr, ax)
        } else {
          a2_slice(array, -(curr + offset), if (mode == "symmetric" || axis_size == 1L) NULL else -1L, ax)
        }
        x <- a2_flip(x, ax)
        if (reflect_type == "odd") {
          x <- jnp_subtract(jnp_multiply(2L, edge), x)
          if (axis_size > 1L)
            edge <- if (before) a2_slice(x, 0L, 1L, ax) else a2_slice(x, -1L, NULL, ax)
        }
        array <- lax_concatenate(if (before) list(x, array) else list(array, x), ax)
      }
      array
    }
    array <- build(array, pad_width[[i]][1L], TRUE)
    array <- build(array, pad_width[[i]][2L], FALSE)
  }
  array
}

a2_pad_edge <- function(array, pad_width) {
  for (i in seq_len(a2_ndim(array))) {
    ax <- i - 1L
    n <- a2_shape(array)[i]
    if (n == 0L) { a2_check_no_padding(pad_width[[i]], "edge"); next }
    before <- jnp_repeat(lax_slice_in_dim(array, 0L, 1L, 1L, ax), pad_width[[i]][1L], axis = ax)
    after <- jnp_repeat(lax_slice_in_dim(array, n - 1L, n, 1L, ax), pad_width[[i]][2L], axis = ax)
    array <- lax_concatenate(list(before, array, after), ax)
  }
  array
}

a2_pad_linear_ramp <- function(array, pad_width, end_values) {
  weak <- aval_of(array)$weak
  for (i in seq_len(a2_ndim(array))) {
    ax <- i - 1L
    eb <- a2_slice(array, 0L, 1L, ax)
    ea <- a2_slice(array, -1L, NULL, ax)
    rb <- a2_linspace(end_values[[i]][1L], lax_squeeze(eb, ax), pad_width[[i]][1L],
                      endpoint = FALSE, dtype = a2_dtype(array), axis = ax)
    rb <- lax_convert_element_type(rb, a2_dtype(rb), weak_type = weak)
    ra <- a2_linspace(end_values[[i]][2L], lax_squeeze(ea, ax), pad_width[[i]][2L],
                      endpoint = FALSE, dtype = a2_dtype(array), axis = ax)
    ra <- lax_convert_element_type(ra, a2_dtype(ra), weak_type = weak)
    ra <- a2_flip(ra, ax)
    array <- lax_concatenate(list(rb, array, ra), ax)
  }
  array
}

a2_pad_stats <- function(array, pad_width, stat_length, stat_func) {
  dt <- a2_dtype(array)
  weak <- aval_of(array)$weak
  for (i in seq_len(a2_ndim(array))) {
    ax <- i - 1L
    if (is.null(stat_length)) {
      sb <- stat_func(array, axis = ax, keepdims = TRUE)
      sa <- sb
    } else {
      len <- a2_shape(array)[i]
      lb <- stat_length[[i]][1L]; la <- stat_length[[i]][2L]
      if (lb == 0L || la == 0L)
        jax_error("stat_length of 0 yields no value for padding", class = "jax_value_error")
      lb <- min(lb, len); la <- min(la, len)
      sb <- stat_func(lax_slice_in_dim(array, 0L, lb, 1L, ax), axis = ax, keepdims = TRUE)
      sa <- stat_func(lax_slice_in_dim(array, len - la, len, 1L, ax), axis = ax, keepdims = TRUE)
    }
    if (is_int_dtype(dt)) { sb <- jnp_round(sb); sa <- jnp_round(sa) }
    sb <- lax_convert_element_type(sb, dt, weak_type = weak)
    sa <- lax_convert_element_type(sa, dt, weak_type = weak)
    array <- lax_concatenate(list(jnp_repeat(sb, pad_width[[i]][1L], axis = ax), array,
                                  jnp_repeat(sa, pad_width[[i]][2L], axis = ax)), ax)
  }
  array
}

a2_pad_empty <- function(array, pad_width) {
  for (i in seq_len(a2_ndim(array))) {
    s <- a2_shape(array)
    sb <- s; sb[i] <- pad_width[[i]][1L]
    sa <- s; sa[i] <- pad_width[[i]][2L]
    array <- lax_concatenate(list(jnp_zeros_like(array, shape = sb), array,
                                  jnp_zeros_like(array, shape = sa)), i - 1L)
  }
  array
}

#' Pad an array
#'
#' Port of `jnp.pad` with all NumPy modes: `"constant"`, `"edge"`, `"linear_ramp"`,
#' `"maximum"`, `"mean"`, `"median"`, `"minimum"`, `"reflect"`, `"symmetric"`,
#' `"wrap"`, `"empty"`, or a function `(vector, iaxis_pad_width, iaxis, kwargs)` that
#' returns the padded vector (`iaxis` is 0-based and `kwargs` the named list of extra
#' arguments).
#' @param array Array.
#' @param pad_width Padding: a number, a `c(before, after)` pair, a list of pairs or an
#'   `(ndim, 2)` matrix.
#' @param mode Padding mode (string or function).
#' @param ... Mode options: `constant_values`, `end_values`, `stat_length`,
#'   `reflect_type`.
#' @return An array.
#' @export
jnp_pad <- function(array, pad_width, mode = "constant", ...) {
  # Port of lax_numpy.py:pad and _pad
  array <- as_jax(array)
  kwargs <- list(...)
  nd <- a2_ndim(array)
  pw <- a2_broadcast_to_pairs(pad_width, nd, "pad_width")
  if (length(pw) && any(vapply(pw, function(p) any(p != round(p)), NA)))
    jax_error("`pad_width` must be of integral type.", class = "jax_type_error")
  pw <- lapply(pw, as.integer)
  if (is.function(mode)) {
    padded <- a2_pad_constant(array, pw, as_jax(0L))
    for (ax in seq_len(a2_ndim(padded)) - 1L)
      padded <- a2_apply_along_axis(mode, ax, padded, pw[[ax + 1L]], ax, kwargs)
    return(padded)
  }
  allowed <- list(empty = character(0), edge = character(0), wrap = character(0),
                  constant = "constant_values", linear_ramp = "end_values",
                  maximum = "stat_length", mean = "stat_length", median = "stat_length",
                  minimum = "stat_length", reflect = "reflect_type", symmetric = "reflect_type")
  if (!is.character(mode) || length(mode) != 1L || !mode %in% names(allowed))
    jax_error("Unimplemented padding mode '", a2_fmt(mode), "' for np.pad.",
              class = "jax_not_implemented_error")
  bad <- setdiff(names(kwargs), allowed[[mode]])
  if (length(bad))
    jax_error("unsupported keyword arguments for mode '", mode, "': {'",
              paste(bad, collapse = "', '"), "'}", class = "jax_value_error")
  if (nd == 0L) return(array)
  if (any(unlist(pw) < 0L)) jax_error("index can't contain negative values", class = "jax_value_error")
  switch(mode,
    constant = a2_pad_constant(array, pw, jnp_asarray(kwargs$constant_values %||% 0L)),
    wrap = a2_pad_wrap(array, pw),
    symmetric = , reflect = a2_pad_symmetric_or_reflect(array, pw, mode, kwargs$reflect_type %||% "even"),
    edge = a2_pad_edge(array, pw),
    linear_ramp = a2_pad_linear_ramp(array, pw,
                                     a2_broadcast_to_pairs(kwargs$end_values %||% 0L, nd, "end_values")),
    maximum = , minimum = , mean = , median = {
      sl_ <- kwargs$stat_length
      if (!is.null(sl_)) sl_ <- lapply(a2_broadcast_to_pairs(sl_, nd, "stat_length"), as.integer)
      f <- switch(mode, maximum = jnp_max, minimum = jnp_min, mean = jnp_mean,
                  median = function(...) jnp_median(...))
      a2_pad_stats(array, pw, sl_, f)
    },
    empty = a2_pad_empty(array, pw))
}

# ---- window functions --------------------------------------------------------------------------

#' Window functions
#'
#' Ports of `jnp.bartlett`, `jnp.blackman`, `jnp.hamming`, `jnp.hanning` and
#' `jnp.kaiser` (default float dtype).
#' @param M Number of points (static).
#' @param beta Shape parameter of the Kaiser window.
#' @return A one dimensional array.
#' @name jnp_window
NULL

a2_window_n <- function(M, name) {
  M <- a2_int(M, paste0("M argument of jnp.", name))
  list(M = M, dt = default_float())
}

#' @rdname jnp_window
#' @export
jnp_blackman <- function(M) {
  w <- a2_window_n(M, "blackman")
  if (w$M <= 1L) return(lax_full(w$M, 1, w$dt))
  n <- lax_iota(w$dt, w$M)
  c2 <- jnp_cos(jnp_true_divide(jnp_multiply(2 * pi, n), w$M - 1L))
  c4 <- jnp_cos(jnp_true_divide(jnp_multiply(4 * pi, n), w$M - 1L))
  jnp_add(jnp_subtract(0.42, jnp_multiply(0.5, c2)), jnp_multiply(0.08, c4))
}

#' @rdname jnp_window
#' @export
jnp_bartlett <- function(M) {
  w <- a2_window_n(M, "bartlett")
  if (w$M <= 1L) return(lax_full(w$M, 1, w$dt))
  n <- lax_iota(w$dt, w$M)
  jnp_subtract(1L, jnp_true_divide(jnp_abs(jnp_subtract(jnp_add(jnp_multiply(2L, n), 1L), w$M)),
                                   w$M - 1L))
}

#' @rdname jnp_window
#' @export
jnp_hamming <- function(M) {
  w <- a2_window_n(M, "hamming")
  if (w$M <= 1L) return(lax_full(w$M, 1, w$dt))
  n <- lax_iota(w$dt, w$M)
  jnp_subtract(0.54, jnp_multiply(0.46, jnp_cos(jnp_true_divide(jnp_multiply(2 * pi, n), w$M - 1L))))
}

#' @rdname jnp_window
#' @export
jnp_hanning <- function(M) {
  w <- a2_window_n(M, "hanning")
  if (w$M <= 1L) return(lax_full(w$M, 1, w$dt))
  n <- lax_iota(w$dt, w$M)
  jnp_multiply(0.5, jnp_subtract(1L, jnp_cos(jnp_true_divide(jnp_multiply(2 * pi, n), w$M - 1L))))
}

#' @rdname jnp_window
#' @export
jnp_kaiser <- function(M, beta) {
  w <- a2_window_n(M, "kaiser")
  if (w$M <= 1L) return(lax_full(w$M, 1, w$dt))
  n <- lax_iota(w$dt, w$M)
  alpha <- 0.5 * (w$M - 1L)
  r <- jnp_power(jnp_true_divide(jnp_subtract(n, alpha), alpha), 2L)
  jnp_true_divide(jnp_i0(jnp_multiply(beta, jnp_sqrt(jnp_subtract(1L, r)))), jnp_i0(beta))
}

# ---- vectorize -------------------------------------------------------------------------------

a2_parse_gufunc_signature <- function(signature) {
  # Port of vectorize.py:_parse_gufunc_signature
  dim <- "\\w+"
  arg <- sprintf("\\((?:%s(?:,%s)*)?\\)", dim, dim)
  arglist <- sprintf("%s(?:,%s)*", arg, arg)
  if (!grepl(sprintf("^%s->%s$", arglist, arglist), signature, perl = TRUE))
    jax_error("not a valid gufunc signature: ", signature, class = "jax_value_error")
  lapply(strsplit(signature, "->", fixed = TRUE)[[1L]], function(side) {
    args <- regmatches(side, gregexpr(arg, side, perl = TRUE))[[1L]]
    lapply(args, function(a) regmatches(a, gregexpr(dim, a, perl = TRUE))[[1L]])
  })
}

a2_fmt_dims <- function(d) {
  if (!length(d)) "()" else if (length(d) == 1L) paste0("('", d, "',)")
  else paste0("(", paste0("'", d, "'", collapse = ", "), ")")
}

a2_update_dim_sizes <- function(dim_sizes, shape, core_dims, error_context, is_input) {
  nc <- length(core_dims)
  if (is_input) {
    if (length(shape) < nc)
      jax_error("input with shape ", a2_fmt_shape(shape), " does not have enough dimensions for ",
                "all core dimensions ", a2_fmt_dims(core_dims), " ", error_context,
                class = "jax_value_error")
  } else if (length(shape) != nc) {
    jax_error("output shape ", a2_fmt_shape(shape), " does not match core dimensions ",
              a2_fmt_dims(core_dims), " ", error_context, class = "jax_value_error")
  }
  core_shape <- if (nc) utils::tail(shape, nc) else integer(0)
  for (k in seq_len(nc)) {
    d <- core_dims[[k]]
    if (is.null(dim_sizes[[d]])) {
      dim_sizes[[d]] <- core_shape[k]
    } else if (core_shape[k] != dim_sizes[[d]]) {
      jax_error("inconsistent size for core dimension '", d, "': ", core_shape[k], " vs ",
                dim_sizes[[d]], " ", error_context, class = "jax_value_error")
    }
  }
  dim_sizes
}

#' Vectorize a function with broadcasting
#'
#' Port of `jnp.vectorize`: the function is mapped with [vmap()] over the broadcast
#' leading dimensions of its arguments. With a generalized ufunc `signature` such as
#' `"(n,m),(m)->(n)"` the trailing core dimensions are passed whole.
#' @param pyfunc Function to vectorize.
#' @param excluded Positions (0-based) or names of arguments passed through unchanged.
#'   Named arguments that are not excluded are ignored, as upstream.
#' @param signature Optional generalized ufunc signature.
#' @return The vectorized function. Multiple outputs are returned (and must be
#'   returned by `pyfunc`) as an unnamed list.
#' @export
jnp_vectorize <- function(pyfunc, excluded = integer(0), signature = NULL) {
  # Port of vectorize.py:vectorize
  force(pyfunc)
  ex <- as.list(excluded)
  if (any(vapply(ex, function(e) !(is.character(e) || (is.numeric(e) && e == round(e))), NA)))
    jax_error("jax.numpy.vectorize can only exclude integer or string arguments, but excluded=",
              a2_fmt(excluded), class = "jax_type_error")
  ex_int <- as.integer(unlist(Filter(is.numeric, ex)))
  ex_chr <- as.character(unlist(Filter(is.character, ex)))
  if (any(ex_int < 0L))
    jax_error("excluded=", a2_fmt(excluded), " contains negative numbers", class = "jax_value_error")
  function(...) {
    error_context <- paste0("on vectorized function with excluded=", a2_fmt(excluded),
                            " and signature=", if (is.null(signature)) "None" else signature)
    all_args <- list(...)
    nms <- names(all_args) %||% rep("", length(all_args))
    pos <- all_args[!nzchar(nms)]
    kw <- all_args[nzchar(nms)]
    static_kw <- kw[names(kw) %in% ex_chr]
    keep <- !((seq_along(pos) - 1L) %in% ex_int)
    static_pos <- pos[!keep]
    static_at <- which(!keep) - 1L
    args <- pos[keep]
    excluded_func <- function(...) {
      a <- list(...)
      for (k in seq_along(static_at)) a <- append(a, static_pos[k], after = static_at[k])
      do.call(pyfunc, c(a, static_kw))
    }
    if (!is.null(signature)) {
      sig <- a2_parse_gufunc_signature(signature)
      input_core_dims <- sig[[1L]]
      output_core_dims <- sig[[2L]]
    } else {
      input_core_dims <- rep(list(character(0)), length(args))
      output_core_dims <- NULL
    }
    none_args <- which(vapply(args, is.null, NA)) - 1L
    if (any(none_args != 0L)) {
      if (any(vapply(input_core_dims[none_args + 1L], length, 1L) > 0L))
        jax_error("Cannot pass None at locations {", paste(none_args, collapse = ", "),
                  "} with signature='", signature, "'", class = "jax_value_error")
      inner <- excluded_func
      none_vals <- args[none_args + 1L]
      excluded_func <- function(...) {
        a <- list(...)
        for (k in seq_along(none_args)) a <- append(a, none_vals[k], after = none_args[k])
        do.call(inner, a)
      }
      args <- args[-(none_args + 1L)]
      input_core_dims <- input_core_dims[-(none_args + 1L)]
    }
    args <- lapply(args, jnp_asarray)
    if (length(args) != length(input_core_dims))
      jax_error("wrong number of positional arguments: expected ", length(input_core_dims),
                ", got ", length(args), " ", error_context, class = "jax_type_error")
    dim_sizes <- list()
    noncore <- list()
    for (k in seq_along(args)) {
      s <- a2_shape(args[[k]])
      dim_sizes <- a2_update_dim_sizes(dim_sizes, s, input_core_dims[[k]], error_context, TRUE)
      noncore[[k]] <- s[seq_len(length(s) - length(input_core_dims[[k]]))]
    }
    broadcast_shape <- broadcast_shapes_impl(c(noncore, list(integer(0))))
    checked_func <- if (is.null(output_core_dims)) excluded_func else function(...) {
      out <- excluded_func(...)
      is_tuple <- is.list(out) && !is_jax_value(out)
      if (length(output_core_dims) > 1L && !is_tuple)
        jax_error("output must be a tuple when multiple outputs are expected, got: ", a2_fmt(out),
                  "\n", error_context, class = "jax_type_error")
      outs <- if (is_tuple) out else list(out)
      if (length(outs) != length(output_core_dims))
        jax_error("wrong number of output arguments: expected ", length(output_core_dims), ", got ",
                  length(outs), " ", error_context, class = "jax_type_error")
      sizes <- dim_sizes
      for (k in seq_along(outs))
        sizes <- a2_update_dim_sizes(sizes, a2_shape(outs[[k]]), output_core_dims[[k]],
                                     error_context, FALSE)
      out
    }
    rp <- .jx$config$jax_numpy_rank_promotion %||% "allow"
    if (!identical(rp, "allow")) {
      ranks <- unlist(lapply(seq_along(args), function(k) {
        nd <- a2_ndim(args[[k]])
        if (nd != 0L) nd - length(input_core_dims[[k]])
      }))
      if (length(unique(ranks)) > 1L) {
        msg <- paste0("operands with shapes [", paste(vapply(args, function(a) a2_fmt_shape(a2_shape(a)), ""),
                                                      collapse = ", "),
                      "] require rank promotion for jnp.vectorize function with signature ",
                      if (is.null(signature)) "None" else signature,
                      ". Set the jax_numpy_rank_promotion config option to 'allow' to disable this message;",
                      " for more information, see https://docs.jax.dev/en/latest/101/rank_promotion_warning.html.")
        if (identical(rp, "warn")) warning(msg, call. = FALSE)
        else if (identical(rp, "raise")) jax_error(msg, class = "jax_value_error")
      }
    }
    nb <- length(broadcast_shape)
    squeezed <- vector("list", length(args))
    rev_filled <- vector("list", length(args))
    for (k in seq_along(args)) {
      nc <- noncore[[k]]
      filled <- c(rep(1L, nb - length(nc)), nc)
      rev_filled[[k]] <- rev(filled)
      sq <- which(nc == 1L) - 1L
      squeezed[[k]] <- if (length(sq)) lax_squeeze(args[[k]], sq) else args[[k]]
    }
    vf <- checked_func
    dims_to_expand <- integer(0)
    for (negdim in seq_len(nb) - 1L) {
      sizes <- vapply(rev_filled, function(r) r[negdim + 1L], 1L)
      if (all(sizes == 1L)) {
        dims_to_expand <- c(dims_to_expand, nb - 1L - negdim)
      } else {
        vf <- a2_vmap(vf, lapply(sizes, function(sz) if (sz == 1L) NULL else 0L))
      }
    }
    result <- do.call(vf, squeezed)
    if (!length(dims_to_expand)) return(result)
    if (is.list(result) && !is_jax_value(result))
      return(lapply(result, lax_expand_dims, dimensions = dims_to_expand))
    lax_expand_dims(result, dims_to_expand)
  }
}

# ---- index tricks ----------------------------------------------------------------------------

a2_grid_from_slice <- function(s, op_name) {
  # Port of index_tricks.py:_make_1d_grid_from_slice
  conc <- function(v, what) {
    if (is.null(v)) return(NULL)
    if (is_jax_value(v)) v <- as.vector(array_host(concrete_or_error(v, paste0("slice ", what, " of jnp.", op_name))))
    v
  }
  start <- conc(s$start, "start")
  stop <- conc(s$stop, "stop")
  step <- conc(s$step, "step")
  if (is.null(start) || isTRUE(start == 0)) start <- 0L
  if (is.null(step) || isTRUE(step == 0)) step <- 1L
  if (is.complex(step) && Im(step) != 0) return(a2_linspace(Re(start), Re(stop), as.integer(Mod(step))))
  jnp_arange(start, stop, step)
}

a2_grid_keys <- function(...) {
  key <- list(...)
  if (length(key) == 1L && is.list(key[[1L]]) && !inherits(key[[1L]], "jax_slice")) key[[1L]] else NULL
}

#' Grids and concatenation helpers
#'
#' Ports of `jnp.mgrid`, `jnp.ogrid`, `jnp.r_` and `jnp.c_`. NumPy's slice syntax is
#' written with [sl()]: `jnp.mgrid[0:3, 0:2]` is `jnp_mgrid(sl(0, 3), sl(0, 2))` and a
#' complex step gives a number of points, as in `jnp_mgrid(sl(0, 1, 5i))`. A single
#' slice gives a one dimensional result; pass a list of slices for the tuple form.
#' `jnp_r_` and `jnp_c_` concatenate arrays, scalars and slices; an optional leading
#' string is a directive (`"r"`, `"c"` or `"axis,ndmin,trans1d"`).
#' @param ... Slices (or, for `jnp_r_` and `jnp_c_`, items and a leading directive).
#' @return An array (`jnp_ogrid` with several slices returns a list).
#' @name jnp_mgrid
NULL

#' @rdname jnp_mgrid
#' @export
jnp_mgrid <- function(...) {
  key <- a2_grid_keys(...)
  if (is.null(key)) {
    if (...length() == 1L) return(a2_grid_from_slice(..1, "mgrid"))
    key <- list(...)
  }
  out <- lapply(key, a2_grid_from_slice, op_name = "mgrid")
  if (!length(out)) return(jnp_arange(0L))
  out <- promote_dtypes(out)
  jnp_stack(do.call(jnp_meshgrid, c(out, list(indexing = "ij", sparse = FALSE))), 0L)
}

#' @rdname jnp_mgrid
#' @export
jnp_ogrid <- function(...) {
  key <- a2_grid_keys(...)
  if (is.null(key)) {
    if (...length() == 1L) return(a2_grid_from_slice(..1, "ogrid"))
    key <- list(...)
  }
  out <- promote_dtypes(lapply(key, a2_grid_from_slice, op_name = "ogrid"))
  do.call(jnp_meshgrid, c(out, list(indexing = "ij", sparse = TRUE)))
}

a2_axis_concat <- function(key, axis, ndmin, trans1d, op_name) {
  # Port of index_tricks.py:_AxisConcat.__getitem__
  params <- c(axis, ndmin, trans1d, -1L)
  if (length(key) && is.character(key[[1L]])) {
    directive <- key[[1L]]
    key <- key[-1L]
    if (directive == "r") {
      params[4L] <- 0L
    } else if (directive == "c") {
      params[4L] <- 1L
    } else {
      vec <- strsplit(directive, ",", fixed = TRUE)[[1L]]
      k <- length(vec)
      vec <- if (k < 4L) c(vec, as.character(params[-seq_len(k)])) else c(vec[1:3], params[4L])
      v <- suppressWarnings(as.integer(trimws(vec)))
      if (anyNA(v) || any(suppressWarnings(as.numeric(trimws(vec))) != v))
        jax_error("could not understand directive '", directive, "'", class = "jax_value_error")
      params <- v
    }
  }
  axis <- params[1L]; ndmin <- params[2L]; trans1d <- params[3L]; matrix <- params[4L]
  output <- lapply(key, function(item) {
    if (inherits(item, "jax_slice")) {
      newobj <- a2_grid_from_slice(item, op_name)
      item_ndim <- 0L
    } else if (is.character(item)) {
      jax_error("string directive must be placed at the beginning", class = "jax_value_error")
    } else {
      newobj <- as_jax(item)
      item_ndim <- a2_ndim(newobj)
    }
    newobj <- a2_ndmin(newobj, ndmin)
    if (trans1d != -1L && ndmin - item_ndim > 0L) {
      shape_obj <- seq_len(ndmin) - 1L
      num_lshifts <- ndmin - abs(ndmin + trans1d + 1L) %% ndmin
      shape_obj <- c(shape_obj[-seq_len(num_lshifts)], shape_obj[seq_len(num_lshifts)])
      newobj <- lax_transpose(newobj, shape_obj)
    }
    newobj
  })
  res <- jnp_concatenate(output, axis = axis)
  if (matrix != -1L && a2_ndim(res) == 1L) res <- lax_expand_dims(res, matrix)
  res
}

#' @rdname jnp_mgrid
#' @export
jnp_r_ <- function(...) a2_axis_concat(list(...), 0L, 1L, -1L, "r_")

#' @rdname jnp_mgrid
#' @export
jnp_c_ <- function(...) a2_axis_concat(list(...), -1L, 2L, 0L, "c_")
