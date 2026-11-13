# jax.numpy array functions a to l (ports of jax/_src/numpy/lax_numpy.py and
# jax/_src/numpy/array_creation.py).
#
# Axes and indices are 0-based. Functions defined elsewhere are not redefined here:
# arange, argmax, argmin, astype, broadcast_arrays, broadcast_to, concatenate,
# expand_dims, eye, identity, linspace, zeros, ones, full, *_like, array and asarray
# (R/numpy_core.R), gcd and lcm (R/numpy_ufuncs.R), cov and corrcoef
# (R/numpy_reductions.R), can_cast, finfo, iinfo and issubdtype (R/numpy_misc.R), load
# and save (R/numpy_io.R). Not ported: jnp.from_dlpack (dlpack is not ported).

# ---- private helpers ---------------------------------------------------------------------

npa_shape <- function(x) aval_of(x)$shape
npa_ndim <- function(x) length(aval_of(x)$shape)
npa_dtype <- function(x) aval_of(x)$dtype

# np.ndim of R data or a jax value (an R list counts as a sequence).
npa_r_ndim <- function(x) {
  if (is_jax_value(x)) return(npa_ndim(x))
  if (is.list(x)) return(1L)
  length(r_shape(x))
}

# Port of core.concrete_or_error(operator.index, x, what).
npa_index <- function(x, what) {
  if (is_jax_value(x)) {
    a <- aval_of(x)
    if (length(a$shape) || !(is_int_dtype(a$dtype) || a$dtype == "bool"))
      jax_error("only integer scalar arrays can be converted to a scalar index (", what, ")",
                class = "jax_type_error")
    x <- as.vector(as.array(concrete_or_error(x, what)))
  }
  if (!(is.numeric(x) || is.logical(x)) || length(x) != 1L || is.na(x) || x != round(x))
    jax_error("'", paste(format(x), collapse = " "), "' cannot be interpreted as an integer (",
              what, ")", class = "jax_type_error")
  as.integer(x)
}

# R doubles with whole values become integer index arrays (R has no int literals).
npa_as_index <- function(x) {
  if (!is_jax_value(x) && is.double(x) && length(x) && all(!is.na(x) & x == round(x)))
    storage.mode(x) <- "integer"
  as_jax(x)
}

# NumPy dtype of a dtype specifier, without x64 canonicalization (np.dtype semantics).
npa_np_dtype <- function(dt) {
  if (is_jax_value(dt)) return(npa_dtype(dt))
  if (!is.character(dt) || length(dt) != 1L) jax_error("dtype must be a string", class = "jax_type_error")
  if (dt %in% names(.dtypes)) return(dt)
  if (!is.na(dtype_aliases[dt])) return(unname(dtype_aliases[dt]))
  switch(dt, float = , float_ = , numeric = "float64", int = , int_ = , integer = "int64",
         uint = "uint64", complex = , complex_ = "complex128", dtype(dt))
}

# Python style slice of one axis: NULL bounds and negative bounds as in
# jax/_src/lax/slicing.py:slice_in_dim, clamped like a Python slice.
npa_slice_axis <- function(x, start, stop, axis) {
  n <- npa_shape(x)[axis + 1L]
  fix <- function(i, dflt) {
    if (is.null(i)) return(dflt)
    if (i < 0L) i <- i + n
    max(0L, min(as.integer(i), n))
  }
  st <- fix(start, 0L)
  lax_slice_in_dim(x, st, max(st, fix(stop, n)), axis = axis)
}

npa_take0 <- function(x, i) lax_index_in_dim(x, i, 0L, keepdims = FALSE)
npa_last <- function(x) npa_take0(x, npa_shape(x)[1L] - 1L)

# vmap with its axes forced (vmap keeps in_axes as a lazy promise).
npa_vmap <- function(f, in_axes = 0L, out_axes = 0L) {
  force(f); force(in_axes); force(out_axes)
  vmap(f, in_axes = in_axes, out_axes = out_axes)
}

# Port of jax/_src/numpy/lax_numpy.py:_atleast_nd.
npa_atleast_nd <- function(x, n) {
  x <- as_jax(x)
  m <- npa_ndim(x)
  if (m < n) lax_broadcast(x, rep(1L, n - m)) else x
}

# Port of jax/_src/numpy/lax_numpy.py:_concatenate_array.
npa_concatenate_array <- function(arr, axis, dtype = NULL) {
  arr <- jnp_asarray(arr, dtype = dtype)
  s <- npa_shape(arr)
  nd <- length(s)
  if (nd == 0L || s[1L] == 0L)
    jax_error("Need at least one array to concatenate.", class = "jax_value_error")
  if (is.null(axis)) return(lax_reshape(arr, prod_int(s)))
  if (nd == 1L) jax_error("Zero-dimensional arrays cannot be concatenated.", class = "jax_value_error")
  axis <- norm_axis(axis, nd - 1L)
  shape <- c(s[1L + seq_len(axis)], s[1L] * s[axis + 2L], drop_first(s, axis + 2L))
  dims <- c(seq_len(axis), 0L, seq.int(axis + 1L, length.out = nd - axis - 1L))
  lax_reshape(arr, shape, dims)
}

# Port of jax/_src/numpy/lax_numpy.py:concatenate (with the dtype argument).
npa_concatenate <- function(arrays, axis = 0L, dtype = NULL) {
  if (is_jax_value(arrays) || !is.list(arrays)) return(npa_concatenate_array(arrays, axis, dtype))
  if (!length(arrays)) jax_error("Need at least one array to concatenate.", class = "jax_value_error")
  arrays <- lapply(arrays, as_jax)
  if (is.null(axis)) return(npa_concatenate(lapply(arrays, jnp_ravel), 0L, dtype))
  if (npa_ndim(arrays[[1L]]) == 0L)
    jax_error("Zero-dimensional arrays cannot be concatenated.", class = "jax_value_error")
  axis <- norm_axis(axis, npa_ndim(arrays[[1L]]))
  xs <- if (is.null(dtype)) promote_dtypes(arrays) else lapply(arrays, jnp_asarray, dtype = dtype)
  lax_concatenate(xs, axis)
}

# Port of jax/_src/numpy/lax_numpy.py:_split (lax.split as consecutive slices).
npa_split <- function(op, ary, indices_or_sections, axis = 0L) {
  ary <- as_jax(ary)
  axis <- npa_index(axis, paste0("in jax.numpy.", op, " argument `axis`"))
  axis <- norm_axis(axis, npa_ndim(ary))
  size <- npa_shape(ary)[axis + 1L]
  ios <- indices_or_sections
  is_seq <- is.list(ios) || (is_jax_value(ios) && npa_ndim(ios) > 0L) ||
    (!is_jax_value(ios) && (length(ios) != 1L || !is.null(dim(ios))))
  if (is_seq) {
    if (is_jax_value(ios)) ios <- as.vector(as.array(concrete_or_error(ios, paste0("in jax.numpy.", op, " argument 1"))))
    resolved <- vapply(as.list(ios), function(i) {
      i <- npa_index(i, paste0("in jax.numpy.", op, " argument 1"))
      if (i < 0L) i <- i + size
      max(0L, min(i, size))
    }, 1L)
    sizes <- diff(c(0L, resolved, size))
  } else {
    num_sections <- npa_index(ios, paste0("in jax.numpy.", op, " argument 1"))
    if (num_sections <= 0L) jax_error("number sections must be larger than 0.", class = "jax_value_error")
    part_size <- size %/% num_sections
    r <- size %% num_sections
    if (r == 0L) {
      sizes <- rep(part_size, num_sections)
    } else if (op == "array_split") {
      sizes <- c(rep(part_size + 1L, r), rep(part_size, num_sections - r))
    } else {
      jax_error("array split does not result in an equal division: rest is ", r,
                class = "jax_value_error")
    }
  }
  if (any(sizes < 0L))
    jax_error("Sizes passed to split must be nonnegative, got ", gs_tuple(sizes),
              class = "jax_value_error")
  starts <- c(0L, cumsum(sizes))
  lapply(seq_along(sizes), function(k)
    lax_slice_in_dim(ary, starts[k], starts[k] + sizes[k], axis = axis))
}

# Port of jax/_src/numpy/lax_numpy.py:nonzero and hijax.py:_nonzero_impl (all axes).
npa_nonzero <- function(a, size = NULL, fill_value = NULL) {
  arr <- as_jax(a)
  nd <- npa_ndim(arr)
  if (nd == 0L)
    jax_error("Calling nonzero on 0d arrays is not allowed. Use jnp.atleast_1d(scalar).nonzero() instead.",
              class = "jax_value_error")
  is_mask <- function(x) if (npa_dtype(x) == "bool") x else jnp_not_equal(x, 0L)
  if (is.null(size)) {
    s <- concrete_or_error(jnp_sum(is_mask(arr)), paste(
      "The size argument of jnp.nonzero must be statically specified",
      "to use jnp.nonzero within JAX transformations."))
    size <- as.integer(as.vector(as.array(s)))
  } else {
    size <- npa_index(size, "The size argument of jnp.nonzero")
  }
  fills <- if (is.null(fill_value)) list() else if (is.list(fill_value)) fill_value else rep(list(fill_value), nd)
  fills <- lapply(fills, as_jax)
  out_dtype <- promote_types(c(default_int(), vapply(fills, npa_dtype, "")),
                             c(FALSE, vapply(fills, function(f) aval_of(f)$weak, NA)))$dtype
  if (length(fills)) {
    if (length(fills) != nd)
      jax_error("fill_value tuple must have length equal to number of axes (", nd, "); got ",
                length(fills), class = "jax_value_error")
    fills <- lapply(fills, jnp_asarray, dtype = out_dtype)
    if (any(vapply(fills, npa_ndim, 1L) != 0L))
      jax_error("fill_value must be a scalar or tuple of scalars", class = "jax_value_error")
  }
  sub_shape <- npa_shape(arr)
  if (prod_int(sub_shape) == 0L || size == 0L) return(rep(list(jnp_zeros(size, out_dtype)), nd))
  strides <- rev(cumprod(rev(sub_shape))) %/% sub_shape
  mask <- is_mask(jnp_reshape(arr, -1L))
  cs_mask <- jnp_cumsum(mask, axis = -1L)
  counts <- at_add(jnp_zeros(size, dtype = npa_dtype(cs_mask)), cs_mask, value = 1L, mode = "drop")
  flat_indices <- jnp_cumsum(counts, axis = -1L)
  out <- lapply(seq_along(strides), function(k)
    lax_convert_element_type(jnp_remainder(jnp_floor_divide(flat_indices, as.integer(strides[k])),
                                           sub_shape[k]), out_dtype))
  fill_mask <- jnp_greater_equal(jnp_arange(size), jnp_sum(mask, axis = -1L, keepdims = TRUE))
  if (length(fills)) Map(function(fv, entry) jnp_where(fill_mask, lax_expand_dims(fv, 0L), entry), fills, out)
  else lapply(out, function(entry) jnp_where(fill_mask, 0L, entry))
}

# Port of jax/_src/numpy/lax_numpy.py:_tile_to_size for 1-D arrays.
npa_tile_to_size <- function(arr, size) {
  n <- prod_int(npa_shape(arr))
  if (n == 0L) jax_error("division by zero", class = "jax_value_error")
  if (n < size) arr <- lax_concatenate(rep(list(arr), ceiling(size / n)), 0L)
  if (prod_int(npa_shape(arr)) > size) lax_slice(arr, 0L, size) else arr
}

# Port of jax/_src/numpy/lax_numpy.py:ravel_multi_index for mode = "clip", order = "C".
npa_ravel_multi_index_clip <- function(multi_index, dims) {
  idx <- lapply(multi_index, as_jax)
  dt <- do.call(result_type, c(list(0L), idx))
  if (prod(dims) - 1 > iinfo(dt)$max)
    jax_error("jnp.ravel_multi_index: dtype=", dt, " not large enough to hold flattened index for dims=",
              gs_tuple(dims), class = "jax_value_error")
  idx <- Map(function(i, d) jnp_clip(i, 0L, d - 1L), idx, dims)
  strides <- rev(cumprod(c(1, rev(dims)[-length(dims)])))
  result <- as_jax(0L, dtype = dt)
  for (k in seq_along(idx))
    result <- jnp_add(result, jnp_multiply(jnp_astype(idx[[k]], dt), as.integer(strides[k])))
  result
}

# Port of jax/_src/numpy/reductions.py:ptp (no axis).
npa_ptp <- function(x) lax_sub(jnp_max(x), jnp_min(x))

# Port of jax/_src/numpy/lax_numpy.py:_i1_maclaurin, _i1, i1_impl and i0_impl.
npa_i1_maclaurin <- custom_jvp(function(k, x) {
  if (k %% 2L == 0L) {
    lax_mul(lax_full_like(x, choose(k + 1L, (k + 1L) %/% 2L) / 2^(k + 1L)), x)
  } else {
    lax_full_like(x, choose(k, k %/% 2L) / 2^k)
  }
}, nondiff_argnums = 0L)
npa_i1_maclaurin <- custom_jvp_defjvp(npa_i1_maclaurin, function(k, primals, tangents) {
  x <- primals[[1L]]
  list(npa_i1_maclaurin(k, x), lax_mul(npa_i1_maclaurin(k + 1L, x), tangents[[1L]]))
})
npa_i1_raw <- function(x) lax_mul(lax_exp(lax_abs(x)), lax_bessel_i1e(x))
# The Bessel recurrence i1'(x) = i0(x) - i1(x) / x is singular at 0 and autodiff of
# exp(abs(x)) * i1e(x) is inaccurate there, so |x| <= sqrt(eps) uses a Maclaurin series.
npa_i1_impl <- custom_jvp(npa_i1_raw, function(primals, tangents) {
  out <- jvp(npa_i1_raw, primals, tangents)
  x <- primals[[1L]]
  use_series <- lax_le(lax_abs(x), lax_full_like(x, sqrt(finfo(npa_dtype(x))$eps)))
  list(out[[1L]], jnp_where(use_series, lax_mul(npa_i1_maclaurin(1L, x), tangents[[1L]]), out[[2L]]))
})
npa_i0_impl <- custom_jvp(function(x) {
  abs_x <- lax_abs(x)
  lax_mul(lax_exp(abs_x), lax_bessel_i0e(abs_x))
}, function(primals, tangents) {
  x <- primals[[1L]]
  list(npa_i0_impl(x), lax_mul(tangents[[1L]], npa_i1_impl(x)))
})

# Port of jax/_src/numpy/array_creation.py:_linspace. Like JAX, the computation runs
# under jit, so XLA contracts the multiply and add into an FMA as in the wheel.
npa_linspace_impl <- function(start, stop, num, endpoint, retstep, dtype, axis) {
  if (num < 0L)
    jax_error("Number of samples, ", num, ", must be non-negative.", class = "jax_value_error")
  start <- as_jax(start)
  stop <- as_jax(stop)
  dt <- if (is.null(dtype)) to_inexact_dtype(result_type(start, stop)) else dtype(dtype)
  cdt <- to_inexact_dtype(dt)
  start <- jnp_astype(start, cdt)
  stop <- jnp_astype(stop, cdt)
  bounds_shape <- broadcast_shapes_impl(list(npa_shape(start), npa_shape(stop)))
  bstart <- jnp_broadcast_to(start, bounds_shape)
  bstop <- jnp_broadcast_to(stop, bounds_shape)
  if (axis < 0L) axis <- length(bounds_shape) + axis + 1L
  bshape <- append(bounds_shape, 1L, after = axis)
  div <- if (endpoint) num - 1L else num
  if (num > 1L) {
    delta <- jnp_true_divide(jnp_subtract(stop, start), as_jax(div, dtype = cdt))
    iota_shape <- rep(1L, length(bshape))
    iota_shape[axis + 1L] <- div
    real_dt <- complex_component(cdt)
    iota <- lax_reshape(lax_iota(real_dt, div), iota_shape)
    step <- jnp_astype(lax_div(iota, lax_full_like(iota, div)), cdt)
    out <- jnp_add(jnp_multiply(lax_reshape(bstart, bshape), jnp_subtract(1, step)),
                   jnp_multiply(jnp_true_divide(lax_reshape(bstop, bshape), as_jax(div, dtype = cdt)),
                                jnp_astype(iota, cdt)))
    if (endpoint) out <- lax_concatenate(list(out, lax_expand_dims(bstop, axis)), axis)
  } else if (num == 1L) {
    delta <- if (endpoint) as_jax(NaN, dtype = cdt) else jnp_astype(jnp_subtract(stop, start), cdt)
    out <- lax_reshape(bstart, bshape)
  } else {
    delta <- as_jax(NaN, dtype = cdt)
    out <- jnp_zeros(append(bounds_shape, 0L, after = axis), dt)
  }
  if (is_int_dtype(dt) && !is_int_dtype(npa_dtype(out))) out <- lax_floor(out)
  result <- lax_convert_element_type(out, dt)
  if (retstep) list(result, delta) else result
}
npa_linspace_jit <- jit(npa_linspace_impl, static_argnames = c("num", "endpoint", "retstep", "dtype", "axis"))
npa_linspace <- function(start, stop, num = 50L, endpoint = TRUE, retstep = FALSE, dtype = NULL,
                         axis = 0L) {
  npa_linspace_jit(start, stop, num = npa_index(num, "'num' argument of jnp.linspace"),
                   endpoint = isTRUE(endpoint), retstep = isTRUE(retstep), dtype = dtype,
                   axis = npa_index(axis, "'axis' argument of jnp.linspace"))
}

# Port of jax/_src/numpy/ufuncs.py:log10 (multiplication by 1 / log(10)).
npa_log10 <- function(x) {
  x <- promote_args_inexact(x)[[1L]]
  dt <- npa_dtype(x)
  one_over_log10 <- function(v) lax_full_like(v, 0.4342944819032518, dtype = complex_component(dt), shape = integer(0))
  r <- lax_log(x)
  if (is_complex_dtype(dt)) {
    re <- lax_real(r)
    im <- lax_imag(r)
    return(lax_complex(lax_mul(re, one_over_log10(re)), lax_mul(im, one_over_log10(im))))
  }
  lax_mul(r, one_over_log10(r))
}

# Port of jax/_src/numpy/lax_numpy.py:round (integer, float and float16 paths).
npa_round <- function(a, decimals = 0L, out = NULL) {
  a <- as_jax(a)
  decimals <- npa_index(decimals, "'decimals' argument of jnp.round")
  if (!is.null(out))
    jax_error("The 'out' argument to jnp.round is not supported.", class = "jax_not_implemented_error")
  dt <- npa_dtype(a)
  if (is_int_dtype(dt)) {
    if (decimals < 0L)
      jax_error("integer np.round not implemented for decimals < 0", class = "jax_not_implemented_error")
    return(a)
  }
  if (dt == "bool")   # dtypes.finfo(dtype) upstream
    jax_error("data type dtype('bool') not compatible with finfo", class = "jax_value_error")
  # Rounding beyond the precision of the dtype changes nothing.
  if (decimals > log10(finfo(dt)$max)) return(a)
  round_float <- function(x) {
    if (decimals == 0L) return(lax_round(x, 1L))
    if (dt == "float16") x <- lax_convert_element_type(x, "float32")
    factor <- lax_full_like(x, 10^decimals)
    r <- lax_div(lax_round(lax_mul(x, factor), 1L), factor)
    if (dt == "float16") lax_convert_element_type(r, dt) else r
  }
  if (is_complex_dtype(dt)) return(lax_complex(round_float(lax_real(a)), round_float(lax_imag(a))))
  round_float(a)
}

# Port of jax/_src/numpy/lax_numpy.py:iscomplexobj.
npa_iscomplexobj <- function(x) {
  if (is.null(x)) return(FALSE)
  if (is_jax_value(x)) return(is_complex_dtype(npa_dtype(x)))
  is.complex(x)
}

# ---- exported functions --------------------------------------------------------------------

#' Compare arrays elementwise within a tolerance
#'
#' Ports of `jnp.isclose` and `jnp.allclose`: `abs(a - b) <= atol + rtol * abs(b)`,
#' with infinities equal only to themselves and NaN equal to NaN when `equal_nan`.
#' @param a,b Arrays.
#' @param rtol Relative tolerance.
#' @param atol Absolute tolerance.
#' @param equal_nan Whether NaN values compare equal.
#' @return A boolean array (`jnp_isclose`) or boolean scalar (`jnp_allclose`).
#' @export
jnp_isclose <- function(a, b, rtol = 1e-05, atol = 1e-08, equal_nan = FALSE) {
  # Port of jax/_src/numpy/lax_numpy.py:isclose.
  ab <- promote_args_inexact(a, b)
  a <- ab[[1L]]
  b <- ab[[2L]]
  dt <- complex_component(npa_dtype(a))
  rtol <- lax_convert_element_type(rtol, dt)
  atol <- lax_convert_element_type(atol, dt)
  both_nan <- jnp_logical_and(jnp_isnan(a), jnp_isnan(b))
  check_fin <- jnp_isfinite(b)
  in_range <- lax_le(lax_abs(lax_sub(a, b)), lax_add(atol, lax_mul(rtol, lax_abs(b))))
  out <- jnp_logical_or(lax_eq(a, b), jnp_logical_and(check_fin, in_range))
  if (isTRUE(equal_nan)) jnp_logical_or(out, both_nan) else out
}

#' @rdname jnp_isclose
#' @export
jnp_allclose <- function(a, b, rtol = 1e-05, atol = 1e-08, equal_nan = FALSE) {
  # Port of jax/_src/numpy/lax_numpy.py:allclose.
  jnp_all(jnp_isclose(a, b, rtol, atol, equal_nan))
}

#' Angle of complex values
#'
#' Port of `jnp.angle`: `atan2(imag(z), real(z))`, in radians or degrees.
#' @param z Array.
#' @param deg Return degrees instead of radians.
#' @return A floating point array.
#' @export
jnp_angle <- function(z, deg = FALSE) {
  # Port of jax/_src/numpy/lax_numpy.py:angle.
  z <- as_jax(z)
  re <- jnp_real(z)
  im <- jnp_imag(z)
  dt <- npa_dtype(re)
  if (!is_inexact_dtype(dt) || (is_float_dtype(npa_dtype(z)) && npa_ndim(z) == 0L)) {
    dt <- default_float()
    re <- lax_convert_element_type(re, dt)
    im <- lax_convert_element_type(im, dt)
  }
  result <- lax_atan2(im, re)
  if (isTRUE(deg)) lax_mul(result, lax_full_like(result, 180 / pi)) else result
}

#' Append values to an array
#'
#' Port of `jnp.append`.
#' @param arr Array.
#' @param values Values to append.
#' @param axis 0-based axis; `NULL` flattens both inputs.
#' @return An array.
#' @export
jnp_append <- function(arr, values, axis = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:append.
  if (is.null(axis)) return(npa_concatenate(list(jnp_ravel(arr), jnp_ravel(values)), 0L))
  npa_concatenate(list(arr, values), axis = axis)
}

#' Apply a function along axes
#'
#' Ports of `jnp.apply_along_axis` (`func1d` is applied to 1-D slices along `axis`
#' by nested `vmap`) and `jnp.apply_over_axes` (`func(a, axis)` is applied
#' repeatedly, restoring reduced axes).
#' @param func1d Function of a 1-D array.
#' @param axis 0-based axis.
#' @param arr,a Array.
#' @param ... Further arguments to `func1d`.
#' @param func Function of an array and an axis.
#' @param axes 0-based axes.
#' @return An array.
#' @export
jnp_apply_along_axis <- function(func1d, axis, arr, ...) {
  # Port of jax/_src/numpy/lax_numpy.py:apply_along_axis.
  arr <- as_jax(arr)
  num_dims <- npa_ndim(arr)
  axis <- norm_axis(axis, num_dims)
  extra <- list(...)
  func <- function(arr) do.call(func1d, c(list(arr), extra))
  for (i in seq_len(num_dims - axis - 1L)) func <- npa_vmap(func, in_axes = i, out_axes = -1L)
  for (i in seq_len(axis)) func <- npa_vmap(func, in_axes = 0L, out_axes = 0L)
  func(arr)
}

#' @rdname jnp_apply_along_axis
#' @export
jnp_apply_over_axes <- function(func, a, axes) {
  # Port of jax/_src/numpy/lax_numpy.py:apply_over_axes.
  a_arr <- as_jax(a)
  for (axis in axes) {
    b <- as_jax(func(a_arr, axis))
    if (npa_ndim(b) == npa_ndim(a_arr)) {
      a_arr <- b
    } else if (npa_ndim(b) == npa_ndim(a_arr) - 1L) {
      a_arr <- jnp_expand_dims(b, axis)
    } else {
      jax_error("function is not returning an array of the correct shape", class = "jax_value_error")
    }
  }
  a_arr
}

#' Indices of nonzero elements
#'
#' Ports of `jnp.argwhere` (a `(N, ndim)` array of coordinates) and `jnp.flatnonzero`
#' (indices into the flattened array). The number of results depends on the data,
#' so inside `jit`, `vmap` or `grad` a static `size` is required; missing entries are
#' padded with `fill_value` (default 0) and extra entries are dropped.
#' @param a Array.
#' @param size Optional static number of results.
#' @param fill_value Optional padding value.
#' @return An integer array.
#' @export
jnp_argwhere <- function(a, size = NULL, fill_value = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:argwhere.
  a <- as_jax(a)
  result <- jnp_transpose(jnp_stack(npa_nonzero(jnp_atleast_1d(a), size = size, fill_value = fill_value)))
  n <- npa_shape(result)[1L]
  if (npa_ndim(a) == 0L)
    return(lax_reshape(lax_slice(result, c(0L, 0L), c(0L, npa_shape(result)[2L])), c(n, 0L)))
  lax_reshape(result, c(n, npa_ndim(a)))
}

#' @rdname jnp_argwhere
#' @export
jnp_flatnonzero <- function(a, size = NULL, fill_value = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:flatnonzero.
  npa_nonzero(jnp_ravel(a), size = size, fill_value = fill_value)[[1L]]
}

#' Round to a number of decimals
#'
#' Port of `jnp.around` (an alias of `jnp.round`): rounds half to even; integers are
#' returned unchanged.
#' @param a Array.
#' @param decimals Number of decimals (static integer).
#' @param out Not supported (must be `NULL`).
#' @return An array.
#' @export
jnp_around <- function(a, decimals = 0L, out = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:around.
  npa_round(a, decimals, out)
}

#' Test arrays for equality
#'
#' Ports of `jnp.array_equal` (same shape and elements) and `jnp.array_equiv`
#' (broadcastable and equal elements).
#' @param a1,a2 Arrays.
#' @param equal_nan Whether NaN values compare equal.
#' @return A boolean scalar array.
#' @export
jnp_array_equal <- function(a1, a2, equal_nan = FALSE) {
  # Port of jax/_src/numpy/lax_numpy.py:array_equal.
  a1 <- jnp_asarray(a1)
  a2 <- jnp_asarray(a2)
  if (!identical(npa_shape(a1), npa_shape(a2))) return(as_jax(FALSE))
  eq <- jnp_equal(a1, a2)
  if (isTRUE(equal_nan)) eq <- jnp_logical_or(eq, jnp_logical_and(jnp_isnan(a1), jnp_isnan(a2)))
  jnp_all(eq)
}

#' @rdname jnp_array_equal
#' @export
jnp_array_equiv <- function(a1, a2) {
  # Port of jax/_src/numpy/lax_numpy.py:array_equiv.
  a1 <- jnp_asarray(a1)
  a2 <- jnp_asarray(a2)
  eq <- tryCatch(jnp_equal(a1, a2), jax_value_error = function(e) NULL)
  if (is.null(eq)) return(as_jax(FALSE))
  jnp_all(eq)
}

#' Split an array into sub-arrays
#'
#' Ports of `jnp.array_split` (sections may differ in size by one), `jnp.hsplit`
#' (axis 1, or 0 for 1-D input) and `jnp.dsplit` (axis 2). A single number gives the
#' number of sections; a vector of length other than one, a list or a 1-D array
#' gives the 0-based split points (out of range points give empty sections).
#' @param ary Array.
#' @param indices_or_sections Number of sections or split points.
#' @param axis 0-based axis.
#' @return A list of arrays.
#' @export
jnp_array_split <- function(ary, indices_or_sections, axis = 0L) {
  # Port of jax/_src/numpy/lax_numpy.py:array_split.
  npa_split("array_split", ary, indices_or_sections, axis = axis)
}

#' @rdname jnp_array_split
#' @export
jnp_hsplit <- function(ary, indices_or_sections) {
  # Port of jax/_src/numpy/lax_numpy.py:hsplit.
  a <- as_jax(ary)
  npa_split("hsplit", a, indices_or_sections, axis = if (npa_ndim(a) == 1L) 0L else 1L)
}

#' @rdname jnp_array_split
#' @export
jnp_dsplit <- function(ary, indices_or_sections) {
  # Port of jax/_src/numpy/lax_numpy.py:dsplit.
  npa_split("dsplit", ary, indices_or_sections, axis = 2L)
}

#' Ensure a minimum number of dimensions
#'
#' Ports of `jnp.atleast_1d`, `jnp.atleast_2d` and `jnp.atleast_3d`. Leading unit
#' dimensions are added (`atleast_3d` gives `(1, n, 1)` for 1-D and `(m, n, 1)` for
#' 2-D input).
#' @param ... Arrays.
#' @return An array for one input, otherwise a list of arrays.
#' @export
jnp_atleast_1d <- function(...) {
  # Port of jax/_src/numpy/lax_numpy.py:atleast_1d.
  out <- lapply(list(...), npa_atleast_nd, n = 1L)
  if (length(out) == 1L) out[[1L]] else out
}

#' @rdname jnp_atleast_1d
#' @export
jnp_atleast_2d <- function(...) {
  # Port of jax/_src/numpy/lax_numpy.py:atleast_2d.
  out <- lapply(list(...), npa_atleast_nd, n = 2L)
  if (length(out) == 1L) out[[1L]] else out
}

#' @rdname jnp_atleast_1d
#' @export
jnp_atleast_3d <- function(...) {
  # Port of jax/_src/numpy/lax_numpy.py:atleast_3d.
  out <- lapply(list(...), function(x) {
    arr <- as_jax(x)
    switch(as.character(npa_ndim(arr)),
           "0" = lax_expand_dims(arr, c(0L, 1L, 2L)),
           "1" = lax_expand_dims(arr, c(0L, 2L)),
           "2" = lax_expand_dims(arr, 2L),
           arr)
  })
  if (length(out) == 1L) out[[1L]] else out
}

#' Count occurrences of non-negative integers
#'
#' Port of `jnp.bincount`. Negative values count toward bin 0 and values at or above
#' `length` are dropped. The output length depends on the data unless `length` is
#' given, which is required inside `jit` and other transformations.
#' @param x 1-D integer (or boolean) array.
#' @param weights Optional weights with the shape of `x`.
#' @param minlength Minimum number of bins.
#' @param length Optional static number of bins.
#' @return An array of counts (the dtype of `weights`, default integer).
#' @export
jnp_bincount <- function(x, weights = NULL, minlength = 0L, length = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:bincount.
  n_bins <- length
  x <- as_jax(x)
  if (npa_dtype(x) == "bool") x <- lax_convert_element_type(x, "int32")
  if (!is_int_dtype(npa_dtype(x)))
    jax_error("x argument to bincount must have an integer type; got ", npa_dtype(x),
              class = "jax_type_error")
  if (npa_ndim(x) != 1L) jax_error("only 1-dimensional input supported.", class = "jax_value_error")
  minlength <- npa_index(minlength, "The error occurred because of argument 'minlength' of jnp.bincount.")
  if (is.null(n_bins)) {
    x_arr <- as.vector(as.array(concrete_or_error(x, paste(
      "argument 'x' of jnp.bincount. To avoid this error, pass a static `length` argument."))))
    n_bins <- max(minlength, if (base::length(x_arr)) as.integer(max(0, x_arr)) + 1L else 0L)
  } else {
    n_bins <- npa_index(n_bins, "The error occurred because of argument 'length' of jnp.bincount.")
  }
  if (is.null(weights)) {
    weights <- as_jax(1L, dtype = default_int())
  } else {
    weights <- as_jax(weights)
    if (!identical(npa_shape(x), npa_shape(weights)))
      jax_error("type of weights must match type of x. Got typeof(x)=", aval_str(aval_of(x)),
                " and typeof(weights)=", aval_str(aval_of(weights)), class = "jax_value_error")
  }
  at_add(jnp_zeros(n_bins, npa_dtype(weights)), jnp_clip(x, 0L), value = weights, mode = "drop")
}

#' Assemble an array from nested lists of blocks
#'
#' Port of `jnp.block`: the innermost lists are concatenated along the last axis, the
#' next level along the second to last axis, and so on.
#' @param arrays A (nested) list of arrays.
#' @return An array.
#' @export
jnp_block <- function(arrays) {
  # Port of jax/_src/numpy/lax_numpy.py:block and _block.
  block <- function(xs) {
    if (is.list(xs)) {
      if (!length(xs))
        jax_error("jax.numpy.block does not allow empty list arguments", class = "jax_value_error")
      parts <- lapply(xs, block)
      depths <- vapply(parts, `[[`, 1L, 2L)
      if (any(depths != depths[1L]))
        jax_error("Mismatched list depths in jax.numpy.block", class = "jax_value_error")
      arrs <- lapply(parts, `[[`, 1L)
      rank <- max(depths[1L], vapply(arrs, npa_ndim, 1L))
      arrs <- lapply(arrs, npa_atleast_nd, n = rank)
      return(list(npa_concatenate(arrs, axis = -depths[1L]), depths[1L] + 1L))
    }
    list(as_jax(xs), 1L)
  }
  block(arrays)[[1L]]
}

#' Broadcast shapes
#'
#' Port of `jnp.broadcast_shapes`.
#' @param ... Integer shape vectors.
#' @return The broadcast shape (an integer vector).
#' @export
jnp_broadcast_shapes <- function(...) {
  # Port of jax/_src/numpy/lax_numpy.py:broadcast_shapes.
  shapes <- list(...)
  if (!length(shapes)) return(integer(0))
  broadcast_shapes_impl(lapply(shapes, as.integer))
}

#' Dtype and scalar queries
#'
#' `jnp_iscomplexobj` and `jnp_isrealobj` port `jnp.iscomplexobj` and `jnp.isrealobj`
#' (test the dtype of a value), `jnp_isscalar` ports `jnp.isscalar` (a length one R
#' vector without `dim`, or a rank 0 array) and `jnp_iterable` ports `numpy.iterable`
#' (a list, an R vector that is not a scalar, or an array of rank at least one).
#' @param x,element,y Values.
#' @return A logical scalar.
#' @export
jnp_iscomplexobj <- function(x) npa_iscomplexobj(x)

#' @rdname jnp_iscomplexobj
#' @export
jnp_isrealobj <- function(x) !jnp_iscomplexobj(x)

#' @rdname jnp_iscomplexobj
#' @export
jnp_isscalar <- function(element) {
  # Port of jax/_src/numpy/lax_numpy.py:isscalar.
  if (is_jax_value(element)) return(npa_ndim(element) == 0L)
  is.atomic(element) && !is.null(element) && length(element) == 1L && is.null(dim(element))
}

#' @rdname jnp_iscomplexobj
#' @export
jnp_iterable <- function(y) {
  # Port of numpy.iterable (re-exported by jax.numpy).
  if (is_jax_value(y)) return(npa_ndim(y) > 0L)
  if (is.list(y)) return(TRUE)
  is.atomic(y) && !is.null(y) && (length(y) != 1L || !is.null(dim(y)))
}

#' Construct an array from an index array and choices
#'
#' Port of `jnp.choose`: `out[i] = choices[a[i]][i]` after broadcasting.
#' @param a Integer array of 0-based choice indices.
#' @param choices A list of arrays, or an array whose leading axis indexes the choices.
#' @param out Not supported (must be `NULL`).
#' @param mode `"raise"` (needs concrete `a`), `"wrap"` or `"clip"`.
#' @return An array.
#' @export
jnp_choose <- function(a, choices, out = NULL, mode = "raise") {
  # Port of jax/_src/numpy/lax_numpy.py:choose.
  if (!is.null(out))
    jax_error("The 'out' argument to jnp.choose is not supported.", class = "jax_not_implemented_error")
  a <- npa_as_index(a)
  if (is.list(choices)) {
    choices <- lapply(choices, as_jax)
  } else {
    ch <- as_jax(choices)
    choices <- lapply(seq_len(npa_shape(ch)[1L]) - 1L, function(i) npa_take0(ch, i))
  }
  if (!is_int_dtype(npa_dtype(a))) jax_error("`a` array must be integer typed", class = "jax_value_error")
  n <- length(choices)
  if (identical(mode, "raise")) {
    arr <- concrete_or_error(a, paste("jnp.choose, which was jit-compiled with mode='raise'.",
                                      "Use mode='wrap' or mode='clip' instead."))
    v <- as.vector(as.array(arr))
    if (any(v < 0 | v >= n)) jax_error("invalid entry in choice array", class = "jax_value_error")
  } else if (identical(mode, "wrap")) {
    arr <- jnp_remainder(a, n)
  } else if (identical(mode, "clip")) {
    arr <- jnp_clip(a, 0L, n - 1L)
  } else {
    jax_error("mode='", mode, "' not understood. Must be 'raise', 'wrap', or 'clip'",
              class = "jax_value_error")
  }
  b <- do.call(jnp_broadcast_arrays, c(list(arr), choices))
  arr <- b[[1L]]
  # jnp.array of a list is never weakly typed.
  stacked <- jnp_stack(b[-1L])
  if (aval_of(stacked)$weak) stacked <- lax_convert_element_type(stacked, npa_dtype(stacked))
  idx_rewriting_take(stacked, c(list(arr), jnp_indices(npa_shape(arr), sparse = TRUE)))
}

#' Clip values to an interval
#'
#' Port of `jnp.clip`: `minimum(max, maximum(min, arr))`; either bound may be `NULL`.
#' Complex values are rejected.
#' @param arr Array.
#' @param min,max Lower and upper bounds (arrays or scalars).
#' @return An array.
#' @export
jnp_clip <- function(arr = NULL, min = NULL, max = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:clip.
  if (is.null(arr)) jax_error("No input was provided to the clip function.", class = "jax_value_error")
  if (npa_iscomplexobj(arr) || npa_iscomplexobj(min) || npa_iscomplexobj(max))
    jax_error("Clip received a complex value either through the input or the min/max keywords. ",
              "Complex values have no ordering and cannot be clipped. Please convert to a real ",
              "value or array by taking the real or imaginary components via ",
              "jax.numpy.real/imag respectively.", class = "jax_value_error")
  if (!is.null(min)) arr <- jnp_maximum(min, arr)
  if (!is.null(max)) arr <- jnp_minimum(max, arr)
  jnp_asarray(arr)
}

#' Stack arrays
#'
#' Ports of `jnp.hstack` (along axis 1, or 0 for 1-D inputs), `jnp.dstack` (along
#' axis 2 after [jnp_atleast_3d()]), `jnp.column_stack` (1-D inputs become columns)
#' and `jnp.concat` (the array API name of `jnp.concatenate`).
#' @param tup,arrays A list of arrays, or an array whose leading axis is iterated over.
#' @param dtype Optional result dtype.
#' @param axis 0-based axis; `NULL` flattens the inputs.
#' @return An array.
#' @export
jnp_hstack <- function(tup, dtype = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:hstack.
  if (!is.list(tup)) {
    arrs <- npa_vmap(function(x) jnp_atleast_1d(x))(as_jax(tup))
    arr0_ndim <- npa_ndim(arrs) - 1L
  } else {
    arrs <- lapply(tup, jnp_atleast_1d)
    arr0_ndim <- npa_ndim(arrs[[1L]])
  }
  npa_concatenate(arrs, axis = if (arr0_ndim == 1L) 0L else 1L, dtype = dtype)
}

#' @rdname jnp_hstack
#' @export
jnp_dstack <- function(tup, dtype = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:dstack.
  arrs <- if (!is.list(tup)) npa_vmap(function(x) jnp_atleast_3d(x))(as_jax(tup)) else lapply(tup, jnp_atleast_3d)
  npa_concatenate(arrs, axis = 2L, dtype = dtype)
}

#' @rdname jnp_hstack
#' @export
jnp_column_stack <- function(tup) {
  # Port of jax/_src/numpy/lax_numpy.py:column_stack.
  if (!is.list(tup)) {
    t <- as_jax(tup)
    arrs <- if (npa_ndim(t) < 3L) npa_vmap(function(x) jnp_transpose(jnp_atleast_2d(x)))(t) else t
  } else {
    arrs <- lapply(lapply(tup, as_jax), function(arr)
      if (npa_ndim(arr) < 2L) jnp_transpose(jnp_atleast_2d(arr)) else arr)
  }
  npa_concatenate(arrs, axis = 1L)
}

#' @rdname jnp_hstack
#' @export
jnp_concat <- function(arrays, axis = 0L) {
  # Port of jax/_src/numpy/lax_numpy.py:concat.
  npa_concatenate(arrays, axis = axis)
}

#' Select slices with a boolean condition
#'
#' Ports of `jnp.compress` (slices along `axis` where `condition` is true) and
#' `jnp.extract` (elements of the flattened array). The result size depends on the
#' data, so inside transformations a static `size` is required; the result is then
#' padded with `fill_value` or truncated.
#' @param condition Boolean (or numeric) array; 1-D for `jnp_compress`.
#' @param a,arr Array.
#' @param axis 0-based axis; `NULL` flattens `a`.
#' @param size Optional static result length.
#' @param fill_value Padding value when `size` is given.
#' @param out Not supported (must be `NULL`).
#' @return An array.
#' @export
jnp_compress <- function(condition, a, axis = NULL, size = NULL, fill_value = 0, out = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:compress.
  condition_arr <- jnp_astype(as_jax(condition), "bool")
  arr <- as_jax(a)
  if (!is.null(out))
    jax_error("The 'out' argument to jnp.compress is not supported.", class = "jax_not_implemented_error")
  if (npa_ndim(condition_arr) != 1L) jax_error("condition must be a 1D array", class = "jax_value_error")
  if (is.null(axis)) {
    axis <- 0L
    arr <- jnp_ravel(arr)
  } else {
    arr <- jnp_moveaxis(arr, axis, 0L)
  }
  nc <- npa_shape(condition_arr)[1L]
  n0 <- npa_shape(arr)[1L]
  extra <- npa_slice_axis(condition_arr, n0, NULL, 0L)
  condition_arr <- npa_slice_axis(condition_arr, NULL, n0, 0L)
  arr <- npa_slice_axis(arr, NULL, npa_shape(condition_arr)[1L], 0L)
  if (is.null(size)) {
    msg <- paste("The size argument of jnp.compress must be specified in order to use",
                 "jnp.compress within JAX transformations like jax.jit, jax.vmap, and",
                 "jax.grad. For more information, refer to the jnp.compress documentation.")
    condition_arr <- concrete_or_error(condition_arr, msg)
    extra <- concrete_or_error(extra, msg)
    if (any(as.vector(as.array(extra))))
      jax_error("condition contains entries that are out of bounds", class = "jax_value_error")
    result <- idx_rewriting_take(arr, list(condition_arr))
  } else {
    size <- npa_index(size, "size argument of jnp.compress")
    if (size < 0L || size > npa_shape(arr)[1L])
      jax_error("size must be positive and not greater than the size of the array axis; got size=",
                size, " for a.shape[axis]=", npa_shape(arr)[1L], class = "jax_value_error")
    mask <- if (npa_ndim(arr) > 1L) lax_expand_dims(condition_arr, seq_len(npa_ndim(arr) - 1L)) else condition_arr
    arr <- jnp_where(mask, arr, jnp_asarray(fill_value, dtype = npa_dtype(arr)))
    perm <- jnp_argsort(condition_arr, stable = TRUE, descending = TRUE)
    result <- npa_slice_axis(idx_rewriting_take(arr, list(perm)), NULL, size, 0L)
  }
  jnp_moveaxis(result, 0L, axis)
}

#' @rdname jnp_compress
#' @export
jnp_extract <- function(condition, arr, size = NULL, fill_value = 0) {
  # Port of jax/_src/numpy/lax_numpy.py:extract.
  jnp_compress(jnp_ravel(condition), jnp_ravel(arr), size = size, fill_value = fill_value)
}

#' Copy an array
#'
#' Port of `jnp.copy`. Arrays are immutable, so the copy is the same value.
#' @param a Array.
#' @param order Memory layout; only `NULL` or `"K"` is supported.
#' @return An array.
#' @export
jnp_copy <- function(a, order = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:copy (array(a, copy=True, order=order)).
  if (!is.null(order) && !identical(order, "K"))
    jax_error("Only implemented for order='K'", class = "jax_not_implemented_error")
  jnp_asarray(a)
}

#' Cross product of 3-vectors
#'
#' Port of `jnp.cross` (only vectors of length 3 are supported, as in JAX).
#' @param a,b Arrays whose vector axes have length 3.
#' @param axisa,axisb,axisc 0-based vector axes of `a`, `b` and the result.
#' @param axis If given, overrides the three axes.
#' @return An array.
#' @export
jnp_cross <- function(a, b, axisa = -1L, axisb = -1L, axisc = -1L, axis = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:cross.
  if (!is.null(axis)) {
    axisa <- axis
    axisb <- axis
    axisc <- axis
  }
  a <- jnp_moveaxis(a, axisa, -1L)
  b <- jnp_moveaxis(b, axisb, -1L)
  if (utils::tail(npa_shape(a), 1L) != 3L || utils::tail(npa_shape(b), 1L) != 3L)
    jax_error("Dimension must be 3 for cross product", class = "jax_value_error")
  comp <- function(x, k) lax_index_in_dim(x, k, npa_ndim(x) - 1L, keepdims = FALSE)
  a0 <- comp(a, 0L); a1 <- comp(a, 1L); a2 <- comp(a, 2L)
  b0 <- comp(b, 0L); b1 <- comp(b, 1L); b2 <- comp(b, 2L)
  c <- jnp_stack(list(jnp_subtract(jnp_multiply(a1, b2), jnp_multiply(a2, b1)),
                      jnp_subtract(jnp_multiply(a2, b0), jnp_multiply(a0, b2)),
                      jnp_subtract(jnp_multiply(a0, b1), jnp_multiply(a1, b0))))
  jnp_moveaxis(c, 0L, axisc)
}

#' Delete or insert entries along an axis
#'
#' Ports of `jnp.delete` and `jnp.insert`. `obj` is a 0-based index, a vector or
#' array of indices, a boolean mask (delete only) or a slice built with [sl()].
#' Deleting with an index array needs concrete indices unless
#' `assume_unique_indices = TRUE`.
#' @param arr Array.
#' @param obj Indices.
#' @param values Values to insert (broadcast to the inserted slots).
#' @param axis 0-based axis; `NULL` flattens `arr`.
#' @param assume_unique_indices Whether the integer indices are known to be unique
#'   (enables use under `jit`).
#' @return An array.
#' @export
jnp_delete <- function(arr, obj, axis = NULL, assume_unique_indices = FALSE) {
  # Port of jax/_src/numpy/lax_numpy.py:delete.
  a <- as_jax(arr)
  if (is.null(axis)) {
    a <- jnp_ravel(a)
    axis <- 0L
  }
  axis <- norm_axis(axis, npa_ndim(a))
  n <- npa_shape(a)[axis + 1L]
  lead <- rep(list(sl()), axis)
  static_int <- if (is_jax_array(obj)) {
    npa_ndim(obj) == 0L && is_int_dtype(npa_dtype(obj))
  } else {
    !is_jax_value(obj) && (is.numeric(obj) || is.logical(obj)) && length(obj) == 1L &&
      is.null(dim(obj)) && !is.na(obj) && obj == round(obj)
  }
  if (static_int) {
    o <- norm_axis(npa_index(obj, "obj argument of jnp.delete"), n)
    return(npa_concatenate(list(npa_slice_axis(a, 0L, o, axis), npa_slice_axis(a, o + 1L, NULL, axis)),
                           axis = axis))
  }
  if (inherits(obj, "jax_slice")) {
    obj <- idx_rewriting_take(jnp_arange(n), list(obj))
    assume_unique_indices <- TRUE
  }
  obj <- npa_as_index(obj)
  if (is_int_dtype(npa_dtype(obj)) && assume_unique_indices) {
    oa <- jnp_ravel(obj)
    oa <- jnp_clip(jnp_where(jnp_less(oa, 0L), jnp_add(oa, n), oa), 0L, n)
    oa <- jnp_sort(oa)
    oa <- jnp_subtract(oa, jnp_arange(prod_int(npa_shape(oa)), dtype = npa_dtype(oa)))
    i <- jnp_arange(n - prod_int(npa_shape(oa)), dtype = npa_dtype(oa))
    i <- jnp_add(i, jnp_sum(jnp_greater_equal(lax_expand_dims(i, 0L), lax_expand_dims(oa, 1L)),
                            axis = 0L, dtype = npa_dtype(i)))
    return(idx_rewriting_take(a, c(lead, list(i))))
  }
  v <- as.array(concrete_or_error(obj, "'obj' array argument of jnp.delete()"))
  if (is_int_dtype(npa_dtype(obj))) {
    v <- as.vector(v)
    bad <- v < -n | v >= n
    if (any(bad))
      jax_error("index ", v[bad][1L], " is out of bounds for axis 0 with size ", n,
                class = "jax_index_error")
    mask <- rep(TRUE, n)
    mask[ifelse(v < 0, v + n, v) + 1L] <- FALSE
  } else if (npa_dtype(obj) == "bool") {
    if (!identical(npa_shape(obj), n))
      jax_error("np.delete(arr, obj): for boolean indices, obj must be one-dimensional ",
                "with length matching specified axis.", class = "jax_value_error")
    mask <- !as.vector(v)
  } else {
    jax_error("np.delete(arr, obj): got obj.dtype=", npa_dtype(obj), "; must be integer or bool.",
              class = "jax_value_error")
  }
  idx_rewriting_take(a, c(lead, list(mask)))
}

#' @rdname jnp_delete
#' @export
jnp_insert <- function(arr, obj, values, axis = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:insert.
  a <- as_jax(arr)
  values_arr <- as_jax(values)
  if (is.null(axis)) {
    a <- jnp_ravel(a)
    axis <- 0L
  }
  axis <- norm_axis(npa_index(axis, "axis argument of jnp.insert()"), npa_ndim(a))
  n_input <- npa_shape(a)[axis + 1L]
  indices <- if (inherits(obj, "jax_slice")) idx_rewriting_take(jnp_arange(n_input), list(obj)) else npa_as_index(obj)
  if (npa_ndim(indices) > 1L)
    jax_error("jnp.insert(): obj must be a slice, a one-dimensional array, or a scalar; got ",
              format(obj), class = "jax_value_error")
  if (!is_int_dtype(npa_dtype(indices))) {
    if (prod_int(npa_shape(indices)) == 0L && !is_jax_value(obj)) {
      indices <- jnp_astype(indices, "int")
    } else {
      jax_error("jnp.insert(): index array must be integer typed; got ", format(obj),
                class = "jax_value_error")
    }
  }
  values_arr <- npa_atleast_nd(jnp_asarray(values_arr, dtype = npa_dtype(a)), npa_ndim(a))
  if (prod_int(npa_shape(indices)) == 1L) {
    index <- npa_take0(jnp_ravel(indices), 0L)
    if (npa_ndim(indices) == 0L) values_arr <- jnp_moveaxis(values_arr, 0L, axis)
    indices <- jnp_full(npa_shape(values_arr)[axis + 1L], index)
  }
  n_insert <- jnp_broadcast_shapes(npa_shape(indices), npa_shape(values_arr)[axis + 1L])[1L]
  out_shape <- npa_shape(a)
  out_shape[axis + 1L] <- out_shape[axis + 1L] + n_insert
  out <- jnp_zeros_like(a, shape = out_shape)
  indices <- jnp_where(jnp_less(indices, 0L), jnp_add(indices, n_input), indices)
  indices <- jnp_clip(indices, 0L, n_input)
  values_ind <- at_add(indices, jnp_argsort(indices), value = jnp_arange(n_insert, dtype = npa_dtype(indices)))
  arr_mask <- at_set(jnp_ones(n_input + n_insert, dtype = "bool"), values_ind, value = FALSE)
  arr_ind <- npa_nonzero(arr_mask, size = n_input)[[1L]]
  lead <- rep(list(sl()), axis)
  out <- idx_scatter_update(out, c(lead, list(values_ind)), values_arr, lax_scatter)
  idx_scatter_update(out, c(lead, list(arr_ind)), a, lax_scatter)
}

#' Diagonals
#'
#' Ports of `jnp.diagonal` (diagonal of two axes, moved to the end), `jnp.diag`
#' (extract a diagonal of a 2-D array or build a 2-D array from a 1-D diagonal),
#' `jnp.diagflat` (2-D array with the flattened input on a diagonal),
#' `jnp.diag_indices`, `jnp.diag_indices_from` (lists of index arrays for the main
#' diagonal) and `jnp.fill_diagonal`.
#' @param a,v,arr Array.
#' @param offset,k 0-based diagonal offset (positive above the main diagonal).
#' @param axis1,axis2 0-based axes.
#' @param n Size of each dimension.
#' @param ndim Number of dimensions.
#' @param val Value(s) written on the diagonal (repeated as needed).
#' @param wrap Not supported (must be `FALSE`).
#' @param inplace Must be `FALSE` (arrays are immutable).
#' @return An array, or a list of index arrays.
#' @export
jnp_diagonal <- function(a, offset = 0L, axis1 = 0L, axis2 = 1L) {
  # Port of jax/_src/numpy/lax_numpy.py:diagonal (the default, non Mosaic path).
  a <- as_jax(a)
  nd <- npa_ndim(a)
  if (nd < 2L) jax_error("diagonal requires an array of at least two dimensions.", class = "jax_value_error")
  offset <- npa_index(offset, "'offset' argument of jnp.diagonal()")
  a_shape <- npa_shape(a)
  ax1 <- norm_axis(axis1, nd)
  ax2 <- norm_axis(axis2, nd)
  a <- jnp_moveaxis(a, c(ax1, ax2), c(-2L, -1L))
  diag_size <- max(0L, min(a_shape[ax1 + 1L] + min(offset, 0L), a_shape[ax2 + 1L] - max(offset, 0L)))
  i <- jnp_arange(diag_size)
  j <- jnp_arange(abs(offset), abs(offset) + diag_size)
  idx_rewriting_take(a, if (offset >= 0L) list(ellipsis, i, j) else list(ellipsis, j, i))
}

#' @rdname jnp_diagonal
#' @export
jnp_diag <- function(v, k = 0L) {
  # Port of jax/_src/numpy/lax_numpy.py:diag and _diag.
  v <- as_jax(v)
  k <- npa_index(k, "k argument of jnp.diag")
  s <- npa_shape(v)
  if (length(s) == 1L) {
    n <- s[1L] + abs(k)
    zero <- lax_full_like(v, 0, shape = integer(0))
    v <- lax_pad(v, zero, list(c(max(0L, -k), max(0L, k), 0L)))
    return(jnp_where(jnp_eye(n, k = k, dtype = "bool"), lax_expand_dims(v, 1L), zero))
  }
  if (length(s) == 2L) return(jnp_diagonal(v, offset = k))
  jax_error("diag input must be 1d or 2d", class = "jax_value_error")
}

#' @rdname jnp_diagonal
#' @export
jnp_diagflat <- function(v, k = 0L) {
  # Port of jax/_src/numpy/lax_numpy.py:diagflat.
  v_ravel <- jnp_ravel(v)
  k <- npa_index(k, "k argument of jnp.diagflat")
  v_length <- prod_int(npa_shape(v_ravel))
  adj_length <- v_length + abs(k)
  res <- jnp_zeros(adj_length * adj_length, dtype = npa_dtype(v_ravel))
  i <- jnp_arange(0L, adj_length - abs(k))
  fi <- if (k >= 0L) {
    jnp_add(jnp_add(i, k), jnp_multiply(i, adj_length))
  } else {
    jnp_add(i, jnp_multiply(jnp_subtract(i, k), adj_length))
  }
  jnp_reshape(at_set(res, fi, value = v_ravel), c(adj_length, adj_length))
}

#' @rdname jnp_diagonal
#' @export
jnp_diag_indices <- function(n, ndim = 2L) {
  # Port of jax/_src/numpy/lax_numpy.py:diag_indices.
  n <- npa_index(n, "'n' argument of jnp.diag_indices()")
  ndim <- npa_index(ndim, "'ndim' argument of jnp.diag_indices()")
  if (n < 0L)
    jax_error("n argument to diag_indices must be nonnegative, got ", n, class = "jax_value_error")
  if (ndim < 0L)
    jax_error("ndim argument to diag_indices must be nonnegative, got ", ndim, class = "jax_value_error")
  rep(list(lax_iota(default_int(), n)), ndim)
}

#' @rdname jnp_diagonal
#' @export
jnp_diag_indices_from <- function(arr) {
  # Port of jax/_src/numpy/lax_numpy.py:diag_indices_from.
  s <- npa_shape(as_jax(arr))
  if (length(s) < 2L) jax_error("input array must be at least 2-d", class = "jax_value_error")
  if (length(unique(s)) != 1L)
    jax_error("All dimensions of input must be of equal length", class = "jax_value_error")
  jnp_diag_indices(s[1L], ndim = length(s))
}

#' @rdname jnp_diagonal
#' @export
jnp_fill_diagonal <- function(a, val, wrap = FALSE, inplace = TRUE) {
  # Port of jax/_src/numpy/lax_numpy.py:fill_diagonal.
  if (inplace)
    jax_error("JAX arrays are immutable, must use inplace=False", class = "jax_not_implemented_error")
  if (wrap)
    jax_error("wrap=True is not implemented, must use wrap=False", class = "jax_not_implemented_error")
  a <- as_jax(a)
  val <- as_jax(val)
  s <- npa_shape(a)
  if (length(s) < 2L) jax_error("array must be at least 2-d", class = "jax_value_error")
  if (length(s) > 2L && !all(s[-1L] == s[1L]))
    jax_error("All dimensions of input must be of equal length", class = "jax_value_error")
  n <- min(s)
  idx <- jnp_diag_indices(n, length(s))
  idx_scatter_update(a, idx, if (npa_ndim(val) == 0L) val else npa_tile_to_size(jnp_ravel(val), n),
                     lax_scatter)
}

#' Differences between consecutive elements
#'
#' Ports of `jnp.diff` (the `n`th discrete difference along `axis`; boolean input gives
#' `!=`) and `jnp.ediff1d` (differences of the flattened array with optional values
#' before and after).
#' @param a,ary Array.
#' @param n Order of the difference.
#' @param axis 0-based axis.
#' @param prepend,append Optional values joined before or after `a` along `axis`.
#' @param to_end,to_begin Optional values joined after or before the differences.
#' @return An array.
#' @export
jnp_diff <- function(a, n = 1L, axis = -1L, prepend = NULL, append = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:diff.
  arr <- as_jax(a)
  n <- npa_index(n, "'n' argument of jnp.diff")
  axis <- npa_index(axis, "'axis' argument of jnp.diff")
  if (n == 0L) return(arr)
  if (n < 0L) jax_error("order must be non-negative but got ", n, class = "jax_value_error")
  if (npa_ndim(arr) == 0L)
    jax_error("diff requires input that is at least one dimensional; got ", format(a),
              class = "jax_value_error")
  axis <- norm_axis(axis, npa_ndim(arr))
  edge <- function(v) {
    v <- as_jax(v)
    if (npa_ndim(v) == 0L) {
      s <- npa_shape(arr)
      s[axis + 1L] <- 1L
      v <- jnp_broadcast_to(v, s)
    }
    v
  }
  combined <- c(if (!is.null(prepend)) list(edge(prepend)), list(arr),
                if (!is.null(append)) list(edge(append)))
  if (length(combined) > 1L) arr <- npa_concatenate(combined, axis = axis)
  op <- if (npa_dtype(arr) == "bool") jnp_not_equal else jnp_subtract
  for (k in seq_len(n))
    arr <- op(npa_slice_axis(arr, 1L, NULL, axis), npa_slice_axis(arr, NULL, -1L, axis))
  arr
}

#' @rdname jnp_diff
#' @export
jnp_ediff1d <- function(ary, to_end = NULL, to_begin = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:ediff1d.
  arr <- jnp_ravel(ary)
  result <- lax_sub(npa_slice_axis(arr, 1L, NULL, 0L), npa_slice_axis(arr, NULL, -1L, 0L))
  if (!is.null(to_begin))
    result <- npa_concatenate(list(jnp_ravel(jnp_astype(to_begin, npa_dtype(arr))), result))
  if (!is.null(to_end))
    result <- npa_concatenate(list(result, jnp_ravel(jnp_astype(to_end, npa_dtype(arr)))))
  result
}

#' Bin indices of values
#'
#' Port of `jnp.digitize`: the index of the bin of each value of `x`, for increasing
#' or decreasing `bins`.
#' @param x Array.
#' @param bins 1-D monotonic array of bin edges.
#' @param right Whether bins include their right edge.
#' @param method Search method passed to [jnp_searchsorted()].
#' @return An int32 array.
#' @export
jnp_digitize <- function(x, bins, right = FALSE, method = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:digitize.
  x <- as_jax(x)
  bins_arr <- as_jax(bins)
  right <- as.logical(as.vector(as.array(concrete_or_error(as_jax(right), "right argument of jnp.digitize()"))))
  if (npa_ndim(bins_arr) != 1L)
    jax_error("digitize: bins must be a 1-dimensional array; got bins=", format(bins),
              class = "jax_value_error")
  nb <- npa_shape(bins_arr)[1L]
  if (nb == 0L) return(jnp_zeros_like(x, dtype = "int32"))
  side <- if (!right) "right" else "left"
  m <- method %||% "scan"
  jnp_where(jnp_greater_equal(npa_last(bins_arr), npa_take0(bins_arr, 0L)),
            jnp_searchsorted(bins_arr, x, side = side, method = m),
            jnp_subtract(nb, jnp_searchsorted(jnp_flip(bins_arr), x, side = side, method = m)))
}

#' Uninitialized arrays
#'
#' Ports of `jnp.empty` and `jnp.empty_like`. On the CPU backend the contents are
#' zeros (JAX lowers `empty` to a zero broadcast there).
#' @param shape Integer vector.
#' @param dtype Optional dtype (default float).
#' @param prototype Array whose shape and dtype are used.
#' @return An array.
#' @export
jnp_empty <- function(shape, dtype = NULL) {
  # Port of jax/_src/numpy/array_creation.py:empty.
  jnp_zeros(shape, dtype)
}

#' @rdname jnp_empty
#' @export
jnp_empty_like <- function(prototype, dtype = NULL, shape = NULL) {
  # Port of jax/_src/numpy/array_creation.py:empty_like (lax.empty_like).
  lax_full_like(prototype, 0, dtype, shape)
}

#' Reverse the order of elements
#'
#' Ports of `jnp.flip` (along the given axes, all by default), `jnp.fliplr` (axis 1)
#' and `jnp.flipud` (axis 0).
#' @param m Array.
#' @param axis 0-based axis or axes; `NULL` flips all axes.
#' @return An array.
#' @export
jnp_flip <- function(m, axis = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:flip and _flip.
  m <- as_jax(m)
  nd <- npa_ndim(m)
  if (is.null(axis)) return(lax_rev(m, seq_len(nd) - 1L))
  lax_rev(m, norm_axis(axis, nd))
}

#' @rdname jnp_flip
#' @export
jnp_fliplr <- function(m) jnp_flip(m, 1L)

#' @rdname jnp_flip
#' @export
jnp_flipud <- function(m) jnp_flip(m, 0L)

#' Elementwise maximum and minimum ignoring NaN
#'
#' Ports of `jnp.fmax` and `jnp.fmin`: when one value is NaN the other is returned.
#' @param x1,x2 Arrays.
#' @return An array.
#' @export
jnp_fmax <- function(x1, x2) {
  # Port of jax/_src/numpy/lax_numpy.py:fmax.
  jnp_where(jnp_bitwise_or(jnp_greater(x1, x2), jnp_isnan(x2)), x1, x2)
}

#' @rdname jnp_fmax
#' @export
jnp_fmin <- function(x1, x2) {
  # Port of jax/_src/numpy/lax_numpy.py:fmin.
  jnp_where(jnp_bitwise_or(jnp_less(x1, x2), jnp_isnan(x2)), x1, x2)
}

#' Arrays from raw bytes, text, functions or iterators
#'
#' `jnp_frombuffer` ports `jnp.frombuffer` for an R raw vector (little endian, as
#' NumPy on common platforms); `jnp_fromstring` ports `jnp.fromstring` (text mode,
#' `sep` is required; whitespace around separators is ignored). `jnp_fromfunction`
#' ports `jnp.fromfunction`: `function` is called once with index arrays built by
#' nested `vmap` (further arguments in `...` are passed unmapped). `jnp_fromfile` and
#' `jnp_fromiter` raise errors, as in JAX.
#' @param buffer Raw vector.
#' @param string Character scalar.
#' @param dtype Dtype of the data (`"float"` is float64 before canonicalization).
#' @param count Number of items (`-1` for all).
#' @param offset Number of bytes to skip.
#' @param sep Separator.
#' @param function Function of the index arrays.
#' @param shape Integer vector.
#' @param ... Further arguments to `function`; ignored by `jnp_fromfile` and
#'   `jnp_fromiter`.
#' @return An array.
#' @export
jnp_frombuffer <- function(buffer, dtype = "float", count = -1L, offset = 0L) {
  # Port of jax/_src/numpy/lax_numpy.py:frombuffer (numpy.frombuffer on a raw vector).
  if (!is.raw(buffer)) jax_error("buffer must be a raw vector", class = "jax_type_error")
  dt <- npa_np_dtype(dtype)
  isz <- max(1L, dtype_bits(dt) %/% 8L)
  nb <- base::length(buffer)
  if (offset < 0L || offset > nb)
    jax_error("offset must be non-negative and no greater than buffer length (", nb, ")",
              class = "jax_value_error")
  avail <- nb - offset
  if (count < 0L) {
    if (avail %% isz != 0L)
      jax_error("buffer size must be a multiple of element size", class = "jax_value_error")
    count <- avail %/% isz
  } else if (count * isz > avail) {
    jax_error("buffer is smaller than requested size", class = "jax_value_error")
  }
  bytes <- buffer[offset + seq_len(count * isz)]
  if (grepl("^u?int[24]$", dt)) {
    # The narrow integers of ml_dtypes are stored one per byte, in its low bits.
    bits <- as.integer(sub("^u?int", "", dt))
    v <- as.integer(bytes) %% 2^bits
    if (startsWith(dt, "int")) v <- ifelse(v >= 2^(bits - 1L), v - 2^bits, v)
    return(as_jax(array(as.integer(v), dim = count), dt))
  }
  if (grepl("^float8_", dt))
    return(lax_bitcast_convert_type(as_jax(array(as.integer(bytes), dim = count), "uint8"), dt))
  if (!dt %in% names(npy_descr))
    jax_error("unsupported dtype for frombuffer: ", dt, class = "jax_type_error")
  # The bytes are reinterpreted on the device, so every bit pattern survives.
  npy_from_bytes(bytes, dt, count, FALSE)
}

#' @rdname jnp_frombuffer
#' @export
jnp_fromstring <- function(string, dtype = "float", count = -1L, sep) {
  # Port of jax/_src/numpy/lax_numpy.py:fromstring (numpy.fromstring text mode).
  if (missing(sep)) jax_error("fromstring() missing required keyword argument 'sep'", class = "jax_type_error")
  if (!nzchar(sep))
    jax_error("The binary mode of fromstring is removed, use frombuffer instead", class = "jax_value_error")
  dt <- npa_np_dtype(dtype)
  s <- trimws(string)
  toks <- if (!nzchar(s)) character(0)
          else if (!nzchar(trimws(sep))) strsplit(s, "[[:space:]]+")[[1L]]
          else trimws(strsplit(s, sep, fixed = TRUE)[[1L]])
  if (count >= 0L) {
    if (count > base::length(toks)) jax_error("string is smaller than requested size", class = "jax_value_error")
    toks <- toks[seq_len(count)]
  }
  vals <- if (is_complex_dtype(dt)) suppressWarnings(as.complex(toks)) else suppressWarnings(as.numeric(toks))
  if (anyNA(vals)) jax_error("string could not be parsed as ", dt, class = "jax_value_error")
  if (dt == "bool") vals <- vals != 0
  as_jax(array(vals, dim = base::length(vals)), dtype = dt)
}

#' @rdname jnp_frombuffer
#' @export
jnp_fromfunction <- function(`function`, shape, ..., dtype = "float") {
  # The body cannot create closures: `function` would shadow the keyword.
  npa_fromfunction(`function`, shape, dtype, ...)
}

# Port of jax/_src/numpy/lax_numpy.py:fromfunction.
npa_fromfunction <- function(fun, shape, dtype, ...) {
  extra <- list(...)
  shape <- as.integer(shape)
  nd <- length(shape)
  f <- function(...) do.call(fun, c(list(...), extra))
  for (i in seq_len(nd) - 1L) {
    in_axes <- lapply(seq_len(nd) - 1L, function(j) if (i == j) 0L else NULL)
    f <- npa_vmap(f, in_axes = rev(in_axes))
  }
  do.call(f, lapply(shape, function(s) jnp_arange(s, dtype = dtype)))
}

#' @rdname jnp_frombuffer
#' @export
jnp_fromfile <- function(...) {
  # Port of jax/_src/numpy/lax_numpy.py:fromfile.
  jax_error("jnp.fromfile() is not implemented because it may be non-pure and thus unsafe for ",
            "use with JIT and other JAX transformations. Consider using jnp_asarray() on data ",
            "read with readBin() instead.", class = "jax_not_implemented_error")
}

#' @rdname jnp_frombuffer
#' @export
jnp_fromiter <- function(...) {
  # Port of jax/_src/numpy/lax_numpy.py:fromiter.
  jax_error("jnp.fromiter() is not implemented because it may be non-pure and thus unsafe for ",
            "use with JIT and other JAX transformations. Consider using jnp_asarray() on the ",
            "collected values instead.", class = "jax_not_implemented_error")
}

#' Numerical gradient
#'
#' Port of `jnp.gradient`: second order central differences in the interior and first
#' order differences at the edges, with scalar or 1-D coordinate spacings.
#' @param f Array.
#' @param ... Spacings: none (unit spacing), one for all axes, or one per axis; each a
#'   scalar or a 1-D array of coordinates.
#' @param axis 0-based axis or axes; `NULL` for all.
#' @param edge_order Not supported (must be `NULL`).
#' @return An array for one axis, otherwise a list of arrays.
#' @export
jnp_gradient <- function(f, ..., axis = NULL, edge_order = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:gradient.
  if (!is.null(edge_order))
    jax_error("The 'edge_order' argument to jnp.gradient is not supported.",
              class = "jax_not_implemented_error")
  args <- promote_dtypes_inexact(c(list(f), list(...)))
  a <- args[[1L]]
  spacing <- args[-1L]
  nd <- npa_ndim(a)
  sub <- function(x, y) jnp_subtract(x, y)
  div <- function(x, y) jnp_true_divide(x, y)
  mul <- function(x, y) jnp_multiply(x, y)
  along_axis <- function(h, ax) {
    sliced <- function(start, stop) npa_slice_axis(a, start, stop, ax)
    upper_edge <- sub(sliced(1L, 2L), sliced(0L, 1L))
    lower_edge <- sub(sliced(-1L, NULL), sliced(-2L, -1L))
    hn <- npa_r_ndim(h)
    if (hn == 0L) {
      inner <- div(mul(sub(sliced(2L, NULL), sliced(NULL, -2L)), 0.5), h)
      lower_edge <- div(lower_edge, h)
      upper_edge <- div(upper_edge, h)
    } else if (hn == 1L) {
      h <- as_jax(h)
      if (npa_shape(h)[1L] != npa_shape(a)[ax + 1L])
        jax_error("Spacing arrays must have the same length as the dimension along which the ",
                  "gradient is calculated.", class = "jax_value_error")
      h_shape <- rep(1L, nd)
      h_shape[ax + 1L] <- npa_shape(h)[1L]
      h <- lax_reshape(h, h_shape)
      sliced_x <- function(start, stop) npa_slice_axis(h, start, stop, ax)
      upper_edge <- div(upper_edge, sub(sliced_x(1L, 2L), sliced_x(0L, 1L)))
      lower_edge <- div(lower_edge, sub(sliced_x(-1L, NULL), sliced_x(-2L, -1L)))
      dx1 <- sub(sliced_x(1L, -1L), sliced_x(0L, -2L))
      dx2 <- sub(sliced_x(2L, NULL), sliced_x(1L, -1L))
      ca <- div(jnp_negative(dx2), mul(dx1, jnp_add(dx1, dx2)))
      cb <- div(sub(dx2, dx1), mul(dx1, dx2))
      cc <- div(dx1, mul(dx2, jnp_add(dx1, dx2)))
      inner <- jnp_add(jnp_add(mul(ca, sliced(0L, -2L)), mul(cb, sliced(1L, -1L))), mul(cc, sliced(2L, NULL)))
    } else {
      jax_error("Spacing arrays must be 1D arrays or scalars.", class = "jax_value_error")
    }
    npa_concatenate(list(upper_edge, inner, lower_edge), axis = ax)
  }
  axis_tuple <- if (is.null(axis)) seq_len(nd) - 1L else norm_axis(axis, nd)
  if (!length(axis_tuple)) return(list())
  if (min(npa_shape(a)[axis_tuple + 1L]) < 2L)
    jax_error("Shape of array too small to calculate a numerical gradient, at least 2 elements ",
              "are required.", class = "jax_value_error")
  dx <- if (!length(spacing)) {
    rep(list(1.0), length(axis_tuple))
  } else if (length(spacing) == 1L) {
    rep(spacing, length(axis_tuple))
  } else if (length(spacing) == length(axis_tuple)) {
    spacing
  } else {
    jax_error("Invalid number of spacing arguments ", length(spacing), " for axis=",
              format(axis), class = "jax_type_error")
  }
  a_grad <- Map(along_axis, dx, axis_tuple)
  if (length(axis_tuple) == 1L) a_grad[[1L]] else unname(a_grad)
}

#' Histograms
#'
#' Ports of `jnp.histogram_bin_edges`, `jnp.histogram`, `jnp.histogram2d` and
#' `jnp.histogramdd`. `bins` is a number of equal width bins or an array of edges;
#' the last bin includes its right edge. String `bins` are not supported.
#' @param a,x,y Arrays of values.
#' @param sample Array of shape `(N, D)`.
#' @param bins Number of bins, or bin edges; for 2-D and D-D histograms one per
#'   dimension may be given in a list.
#' @param range Optional `c(lower, upper)`; for 2-D and D-D histograms a list with one
#'   pair (or `NULL`) per dimension.
#' @param weights Optional weights.
#' @param density Normalize to a probability density.
#' @return `jnp_histogram_bin_edges`: the edges. `jnp_histogram`: a list of counts and
#'   edges. `jnp_histogram2d`: a list of counts, x edges and y edges.
#'   `jnp_histogramdd`: a list of counts and a list of edges.
#' @export
jnp_histogram_bin_edges <- function(a, bins = 10L, range = NULL, weights = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:histogram_bin_edges.
  if (is.character(bins))
    jax_error("string values for `bins` not implemented.", class = "jax_not_implemented_error")
  arr <- as_jax(a)
  dt <- to_inexact_dtype(npa_dtype(arr))
  if (npa_r_ndim(bins) == 1L) return(jnp_asarray(bins, dtype = dt))
  bins_int <- npa_index(bins, "bins argument of histogram_bin_edges")
  if (is.null(range))
    range <- if (prod_int(npa_shape(arr)) == 0L) c(0, 1) else list(jnp_min(arr), jnp_max(arr))
  range <- jnp_asarray(range, dtype = dt)
  if (!identical(npa_shape(range), 2L))
    jax_error("`range` must be either None or a sequence of scalars, got ", format(range),
              class = "jax_value_error")
  is_zero <- jnp_equal(npa_ptp(range), 0L)
  r0 <- npa_take0(range, 0L)
  r1 <- npa_take0(range, 1L)
  npa_linspace(jnp_where(is_zero, jnp_subtract(r0, 0.5), r0), jnp_where(is_zero, jnp_add(r1, 0.5), r1),
               bins_int + 1L, dtype = dt)
}

#' @rdname jnp_histogram_bin_edges
#' @export
jnp_histogram <- function(a, bins = 10L, range = NULL, weights = NULL, density = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:histogram and _histogram.
  if (is.null(weights)) {
    a <- promote_dtypes_inexact(list(a))[[1L]]
    weights <- jnp_ones_like(a)
  } else {
    a <- as_jax(a)
    weights <- as_jax(weights)
    if (!identical(npa_shape(a), npa_shape(weights)))
      jax_error("weights should have the same shape as a.", class = "jax_value_error")
    aw <- promote_dtypes_inexact(list(a, weights))
    a <- aw[[1L]]
    weights <- aw[[2L]]
  }
  bin_edges <- jnp_histogram_bin_edges(a, bins, range, weights)
  ne <- npa_shape(bin_edges)[1L]
  bin_idx <- jnp_searchsorted(bin_edges, a, side = "right")
  bin_idx <- jnp_where(jnp_equal(a, npa_last(bin_edges)), ne - 1L, bin_idx)
  counts <- at_add(jnp_zeros(ne, npa_dtype(weights)), bin_idx, value = weights)
  counts <- npa_slice_axis(counts, 1L, NULL, 0L)
  if (isTRUE(density))
    counts <- jnp_true_divide(jnp_true_divide(counts, jnp_diff(bin_edges)), jnp_sum(counts))
  list(counts, bin_edges)
}

#' @rdname jnp_histogram_bin_edges
#' @export
jnp_histogram2d <- function(x, y, bins = 10L, range = NULL, weights = NULL, density = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:histogram2d.
  x <- as_jax(x)
  y <- as_jax(y)
  N <- if (is.list(bins)) length(bins) else if (npa_r_ndim(bins) == 0L) 1L else npa_shape(as_jax(bins))[1L]
  if (N != 1L && N != 2L) {
    edges <- jnp_asarray(bins)
    bins <- list(edges, edges)
  }
  sample <- jnp_transpose(jnp_stack(list(x, y)))
  r <- jnp_histogramdd(sample, bins, range, weights, density)
  list(r[[1L]], r[[2L]][[1L]], r[[2L]][[2L]])
}

#' @rdname jnp_histogram_bin_edges
#' @export
jnp_histogramdd <- function(sample, bins = 10L, range = NULL, weights = NULL, density = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:histogramdd.
  if (is.null(weights)) {
    sample <- promote_dtypes_inexact(list(sample))[[1L]]
  } else {
    sample <- as_jax(sample)
    weights <- as_jax(weights)
    if (!identical(npa_shape(weights), npa_shape(sample)[1L]))
      jax_error("should have one weight for each sample.", class = "jax_value_error")
    sw <- promote_dtypes_inexact(list(sample, weights))
    sample <- sw[[1L]]
    weights <- sw[[2L]]
  }
  s <- npa_shape(sample)
  if (length(s) != 2L) jax_error("sample must be a 2-D array of shape (N, D)", class = "jax_value_error")
  D <- s[2L]
  if (!is.null(range) && (length(range) != D ||
                          any(vapply(range, function(r) !is.null(r) && length(r) != 2L, NA))))
    jax_error("For sample.shape=", gs_tuple(s), ", range must be a sequence of ", D,
              " pairs or Nones", class = "jax_value_error")
  bins_per_dimension <- if (is.list(bins)) {
    bins
  } else if (npa_r_ndim(bins) == 0L) {
    rep(list(bins), D)
  } else if (is_jax_value(bins)) {
    lapply(seq_len(npa_shape(bins)[1L]) - 1L, function(i) npa_take0(bins, i))
  } else {
    as.list(bins)
  }
  if (length(bins_per_dimension) != D)
    jax_error("should be a bin for each dimension.", class = "jax_value_error")
  bin_idx_by_dim <- list()
  bin_edges_by_dim <- list()
  for (i in seq_len(D)) {
    col <- lax_index_in_dim(sample, i - 1L, 1L, keepdims = FALSE)
    bin_edges <- jnp_histogram_bin_edges(col, bins_per_dimension[[i]], range[[i]], weights)
    bin_idx <- jnp_searchsorted(bin_edges, col, side = "right")
    bin_idx <- jnp_where(jnp_equal(col, npa_last(bin_edges)), jnp_subtract(bin_idx, 1L), bin_idx)
    bin_idx_by_dim[[i]] <- bin_idx
    bin_edges_by_dim[[i]] <- bin_edges
  }
  nbins <- vapply(bin_edges_by_dim, function(e) npa_shape(e)[1L] + 1L, 1L)
  dedges <- lapply(bin_edges_by_dim, jnp_diff)
  xy <- npa_ravel_multi_index_clip(bin_idx_by_dim, nbins)
  hist <- jnp_reshape(jnp_bincount(xy, weights, length = prod_int(nbins)), nbins)
  hist <- lax_slice(hist, rep(1L, D), nbins - 1L)
  if (isTRUE(density)) {
    hist <- jnp_astype(hist, npa_dtype(sample))
    hist <- jnp_true_divide(hist, jnp_sum(hist))
    for (norm in do.call(jnp_ix_, dedges)) hist <- jnp_true_divide(hist, norm)
  }
  list(hist, bin_edges_by_dim)
}

#' Modified Bessel function of the first kind of order zero
#'
#' Port of `jnp.i0`, with the derivative `i1` (accurate near zero through a series).
#' @param x Array of floats.
#' @return An array.
#' @export
jnp_i0 <- function(x) {
  # Port of jax/_src/numpy/lax_numpy.py:i0.
  x <- promote_args_inexact(x)[[1L]]
  if (!is_float_dtype(npa_dtype(x)))
    jax_error("Unsupported input type to jax.numpy.i0: ", npa_dtype(x), class = "jax_value_error")
  npa_i0_impl(x)
}

#' Index grids
#'
#' Ports of `jnp.indices` (a dense `(ndim, ...)` grid of 0-based indices, or a list of
#' broadcastable index arrays with `sparse = TRUE`) and `jnp.ix_` (open mesh: the k-th
#' 1-D input is reshaped to lie along axis k).
#' @param dimensions Integer vector giving the grid shape.
#' @param dtype Optional integer dtype.
#' @param sparse Return a list of sparse index arrays.
#' @param ... 1-D arrays.
#' @return An array or a list of arrays.
#' @export
jnp_indices <- function(dimensions, dtype = NULL, sparse = FALSE) {
  # Port of jax/_src/numpy/lax_numpy.py:indices.
  dt <- dtype(dtype %||% "int")
  dims <- vapply(as.list(dimensions), npa_index, 1L, what = "dimensions argument of jnp.indices")
  N <- length(dims)
  out <- lapply(seq_len(N), function(i) {
    s <- dims
    if (sparse) {
      s <- rep(1L, N)
      s[i] <- dims[i]
    }
    lax_broadcast_in_dim(lax_iota(dt, dims[i]), s, i - 1L)
  })
  if (sparse) return(out)
  if (N) jnp_stack(out, 0L) else as_jax(numeric(0), dtype = dt)
}

#' @rdname jnp_indices
#' @export
jnp_ix_ <- function(...) {
  # Port of jax/_src/numpy/lax_numpy.py:ix_.
  args <- lapply(list(...), npa_as_index)
  n <- length(args)
  lapply(seq_len(n), function(i) {
    a <- args[[i]]
    if (npa_ndim(a) != 1L)
      jax_error("Arguments to jax.numpy.ix_ must be 1-dimensional, got shape ",
                gs_tuple(npa_shape(a)), class = "jax_value_error")
    if (npa_dtype(a) == "bool")
      jax_error("Boolean arguments to jax.numpy.ix_ are not implemented",
                class = "jax_not_implemented_error")
    shape <- rep(1L, n)
    shape[i] <- npa_shape(a)[1L]
    if (shape[i] == 0L) lax_full(shape, 0L, default_int()) else lax_broadcast_in_dim(a, shape, i - 1L)
  })
}

#' One-dimensional linear interpolation
#'
#' Port of `jnp.interp` for increasing `xp`.
#' @param x Points to evaluate.
#' @param xp,fp 1-D arrays of coordinates and values.
#' @param left,right Values outside the range (default `fp[0]` and `fp[-1]`), or
#'   `"extrapolate"`.
#' @param period Optional period of `xp`.
#' @return An array.
#' @export
jnp_interp <- function(x, xp, fp, left = NULL, right = NULL, period = NULL) {
  # Port of jax/_src/numpy/lax_numpy.py:interp and _interp.
  x <- as_jax(x)
  xp <- as_jax(xp)
  fp <- as_jax(fp)
  if (!identical(npa_shape(xp), npa_shape(fp)) || npa_ndim(xp) != 1L)
    jax_error("xp and fp must be one-dimensional arrays of equal size", class = "jax_value_error")
  p <- promote_dtypes_inexact(list(x, xp))
  x_arr <- p[[1L]]
  xp_arr <- p[[2L]]
  fp_arr <- promote_dtypes_inexact(list(fp))[[1L]]
  extrapolate <- function(side, v) {
    if (!is.character(v)) return(FALSE)
    if (!identical(v, "extrapolate"))
      jax_error("the only valid string value of `", side, "` is 'extrapolate', but got: '", v, "'",
                class = "jax_value_error")
    TRUE
  }
  extrapolate_left <- extrapolate("left", left)
  extrapolate_right <- extrapolate("right", right)
  if (is_complex_dtype(npa_dtype(x_arr)))
    jax_error("jnp.interp: complex x values not supported.", class = "jax_value_error")
  if (!is.null(period)) {
    if (npa_r_ndim(period) != 0L)
      jax_error("period must be a scalar; got ", format(period), class = "jax_value_error")
    period <- jnp_abs(period)
    x_arr <- jnp_remainder(x_arr, period)
    xp_arr <- jnp_remainder(xp_arr, period)
    kv <- lax_sort_key_val(xp_arr, fp_arr)
    xp_arr <- kv[[1L]]
    fp_arr <- kv[[2L]]
    xp_arr <- npa_concatenate(list(jnp_subtract(npa_slice_axis(xp_arr, -1L, NULL, 0L), period), xp_arr,
                                   jnp_add(npa_slice_axis(xp_arr, NULL, 1L, 0L), period)))
    fp_arr <- npa_concatenate(list(npa_slice_axis(fp_arr, -1L, NULL, 0L), fp_arr,
                                   npa_slice_axis(fp_arr, NULL, 1L, 0L)))
  }
  take <- function(v, i) idx_rewriting_take(v, list(i))
  i <- jnp_clip(jnp_searchsorted(xp_arr, x_arr, side = "right"), 1L, npa_shape(xp_arr)[1L] - 1L)
  im1 <- jnp_subtract(i, 1L)
  df <- jnp_subtract(take(fp_arr, i), take(fp_arr, im1))
  dx <- jnp_subtract(take(xp_arr, i), take(xp_arr, im1))
  delta <- jnp_subtract(x_arr, take(xp_arr, im1))
  epsilon <- finfo(npa_dtype(xp_arr))$eps^2
  dx0 <- lax_le(lax_abs(dx), lax_full_like(dx, epsilon))
  f <- jnp_where(dx0, take(fp_arr, im1),
                 jnp_add(take(fp_arr, im1), jnp_multiply(jnp_true_divide(delta, jnp_where(dx0, 1L, dx)), df)))
  if (!extrapolate_left && is.null(period))
    f <- jnp_where(jnp_less(x_arr, npa_take0(xp_arr, 0L)), left %||% npa_take0(fp_arr, 0L), f)
  if (!extrapolate_right && is.null(period))
    f <- jnp_where(jnp_greater(x_arr, npa_last(xp_arr)), right %||% npa_last(fp_arr), f)
  f
}

#' Elementwise complex and real tests
#'
#' Ports of `jnp.iscomplex` (nonzero imaginary part) and `jnp.isreal` (zero imaginary
#' part).
#' @param x Array.
#' @return A boolean array.
#' @export
jnp_iscomplex <- function(x) {
  # Port of jax/_src/numpy/lax_numpy.py:iscomplex.
  i <- jnp_imag(x)
  lax_ne(i, lax_full_like(i, 0))
}

#' @rdname jnp_iscomplex
#' @export
jnp_isreal <- function(x) {
  # Port of jax/_src/numpy/lax_numpy.py:isreal.
  i <- jnp_imag(x)
  lax_eq(i, lax_full_like(i, 0))
}

#' Kronecker product
#'
#' Port of `jnp.kron`.
#' @param a,b Arrays.
#' @return An array whose shape is the elementwise product of the (rank promoted)
#'   input shapes.
#' @export
jnp_kron <- function(a, b) {
  # Port of jax/_src/numpy/lax_numpy.py:kron.
  ab <- promote_dtypes(list(a, b))
  a <- ab[[1L]]
  b <- ab[[2L]]
  na <- npa_ndim(a)
  nb <- npa_ndim(b)
  if (na < nb) a <- lax_expand_dims(a, seq_len(nb - na) - 1L)
  if (nb < na) b <- lax_expand_dims(b, seq_len(na - nb) - 1L)
  n <- npa_ndim(a)
  if (n == 0L) return(lax_mul(a, b))
  a_r <- lax_expand_dims(a, 2L * seq_len(n) - 1L)
  b_r <- lax_expand_dims(b, 2L * (seq_len(n) - 1L))
  lax_reshape(lax_mul(a_r, b_r), npa_shape(a) * npa_shape(b))
}

#' Numbers spaced evenly on a log scale
#'
#' Ports of `jnp.logspace` (`base^linspace(start, stop)`) and `jnp.geomspace` (a
#' geometric progression from `start` to `stop`).
#' @param start,stop Interval ends (exponents for `jnp_logspace`).
#' @param num Number of samples.
#' @param endpoint Include `stop`.
#' @param base Base of the log scale.
#' @param dtype Optional dtype.
#' @param axis 0-based axis of the samples in the result.
#' @return An array.
#' @export
jnp_logspace <- function(start, stop, num = 50L, endpoint = TRUE, base = 10.0, dtype = NULL, axis = 0L) {
  # Port of jax/_src/numpy/array_creation.py:logspace.
  npa_logspace_impl(start, stop, base, num = npa_index(num, "'num' argument of jnp.logspace"),
                    endpoint = isTRUE(endpoint), dtype = dtype,
                    axis = npa_index(axis, "'axis' argument of jnp.logspace"))
}

# Port of jax/_src/numpy/array_creation.py:_logspace. Unlike JAX it is not jitted as a
# whole: the CPU plugin's XLA evaluates pow inside a fusion differently from jaxlib's in
# the last bit, while op by op dispatch (with the jitted linspace) matches the wheel.
npa_logspace_impl <- function(start, stop, base, num, endpoint, dtype, axis) {
  start <- as_jax(start)
  stop <- as_jax(stop)
  dt <- if (is.null(dtype)) to_inexact_dtype(result_type(start, stop)) else dtype(dtype)
  cdt <- to_inexact_dtype(dt)
  lin <- npa_linspace(jnp_astype(start, cdt), jnp_astype(stop, cdt), num, endpoint = endpoint,
                      retstep = FALSE, dtype = NULL, axis = axis)
  lax_convert_element_type(jnp_power(base, lin), dt)
}

#' @rdname jnp_logspace
#' @export
jnp_geomspace <- function(start, stop, num = 50L, endpoint = TRUE, dtype = NULL, axis = 0L) {
  # Port of jax/_src/numpy/array_creation.py:geomspace.
  npa_geomspace_impl(start, stop, num = npa_index(num, "'num' argument of jnp.geomspace"),
                     endpoint = isTRUE(endpoint), dtype = dtype,
                     axis = npa_index(axis, "'axis' argument of jnp.geomspace"))
}

# Port of jax/_src/numpy/array_creation.py:_geomspace (not jitted as a whole, see
# npa_logspace_impl).
npa_geomspace_impl <- function(start, stop, num, endpoint, dtype, axis) {
  start <- as_jax(start)
  stop <- as_jax(stop)
  dt <- if (is.null(dtype)) to_inexact_dtype(result_type(start, stop)) else dtype(dtype)
  cdt <- to_inexact_dtype(dt)
  start <- jnp_astype(start, cdt)
  stop <- jnp_astype(stop, cdt)
  sign <- jnp_sign(start)
  res <- jnp_multiply(sign, jnp_logspace(npa_log10(jnp_true_divide(start, sign)),
                                         npa_log10(jnp_true_divide(stop, sign)), num,
                                         endpoint = endpoint, base = 10.0, dtype = cdt, axis = 0L))
  nd <- npa_ndim(res)
  axis <- norm_axis(axis, nd)
  if (axis != 0L)
    res <- lax_transpose(res, c(seq_len(axis), 0L, seq.int(axis + 1L, length.out = nd - axis - 1L)))
  lax_convert_element_type(res, dt)
}

#' Greatest common divisor and least common multiple
#'
#' Ports of `jnp.gcd` and `jnp.lcm` (integer arrays, elementwise with broadcasting).
#' @param x1,x2 Integer arrays.
#' @return An integer array.
#' @export
jnp_gcd <- function(x1, x2) {
  v <- promote_dtypes(list(x1, x2))
  if (!is_int_dtype(aval_of(v[[1L]])$dtype))
    jax_error("Arguments to jax.numpy.gcd must be integers.", class = "jax_value_error")
  v <- jnp_broadcast_arrays(v[[1L]], v[[2L]])
  cond <- function(xs) jnp_any(jnp_not_equal(xs[[2L]], 0L))
  body <- function(xs) {
    a <- xs[[1L]]; b <- xs[[2L]]
    nz <- jnp_not_equal(b, 0L)
    a2 <- jnp_where(nz, b, a)
    b2 <- jnp_where(nz, lax_rem(a, jnp_where(nz, b, jnp_ones_like(b))), jnp_zeros_like(b))
    list(jnp_where(a2 < b2, b2, a2), jnp_where(a2 < b2, a2, b2))
  }
  lax_while_loop(cond, body, list(jnp_abs(v[[1L]]), jnp_abs(v[[2L]])))[[1L]]
}

#' @rdname jnp_gcd
#' @export
jnp_lcm <- function(x1, x2) {
  v <- promote_dtypes(list(x1, x2))
  a <- jnp_abs(v[[1L]]); b <- jnp_abs(v[[2L]])
  if (!is_int_dtype(aval_of(a)$dtype))
    jax_error("Arguments to jax.numpy.lcm must be integers.", class = "jax_value_error")
  d <- jnp_gcd(a, b)
  jnp_where(jnp_equal(d, 0L), jnp_zeros_like(d), jnp_multiply(a, jnp_floor_divide(b, jnp_where(jnp_equal(d, 0L), jnp_ones_like(d), d))))
}
