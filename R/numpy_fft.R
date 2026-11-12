# jax.numpy.fft and the DCTs of jax.scipy.fft (ports of jax/_src/numpy/fft.py and
# jax/_src/scipy/fft.py).

# ---- private helpers (lax compositions; `fft_` prefix avoids clashes with jnp) --------

# Python range(a, b).
fft_range <- function(a, b) if (b > a) seq.int(a, b - 1L) else integer(0)

fft_ndim <- function(x) length(aval_of(x)$shape)

# Static-shift path of jnp.roll (port of jax/_src/numpy/lax_numpy.py:_roll_static).
fft_roll <- function(a, shift, axis) {
  a <- as_jax(a)
  axis <- norm_axis(axis, fft_ndim(a))
  shift <- as.integer(shift)
  if (length(shift) == 1L) shift <- rep(shift, length(axis))
  if (length(axis) == 1L) axis <- rep(axis, length(shift))
  if (length(axis) != length(shift))
    jax_error("shape mismatch: shift and axis cannot be broadcast to a single shape",
              class = "jax_value_error")
  for (j in seq_along(axis)) {
    ax <- axis[j]
    n <- aval_of(a)$shape[ax + 1L]
    if (n == 0L) next
    i <- (-shift[j]) %% n
    a <- lax_concatenate(list(lax_slice_in_dim(a, i, n, axis = ax),
                              lax_slice_in_dim(a, 0L, i, axis = ax)), ax)
  }
  a
}

# ---- jax.numpy.fft ----------------------------------------------------------------------

# Port of jax/_src/numpy/fft.py:_fft_norm.
fft_norm <- function(s, func_name, norm) {
  if (identical(norm, "backward")) return(jnp_array(1L))
  # Avoid potential integer overflow
  s <- promote_dtypes_inexact(list(s))[[1L]]
  inverse <- startsWith(func_name, "i")
  if (identical(norm, "ortho")) {
    r <- jnp_sqrt(jnp_prod(s))
    return(if (inverse) r else jnp_true_divide(1, r))
  }
  if (identical(norm, "forward")) {
    r <- jnp_prod(s)
    return(if (inverse) r else jnp_true_divide(1, r))
  }
  jax_error("Invalid norm value ", paste(format(norm), collapse = ", "),
            "; should be \"backward\",\"ortho\" or \"forward\".", class = "jax_value_error")
}

# Port of jax/_src/numpy/fft.py:_fft_core.
fft_core <- function(func_name, fft_type, a, s, axes, norm) {
  full_name <- paste0("jax.numpy.fft.", func_name)
  arr <- as_jax(a)
  nd <- fft_ndim(arr)
  if (!is.null(s)) {
    s <- vapply(s, npa_index, 1L, what = paste0("the shape argument of ", full_name))
    if (any(s <= 0L)) jax_error("Shape should be positive.", class = "jax_value_error")
  }
  if (!is.null(s) && !is.null(axes) && length(s) != length(axes))
    jax_error("Shape and axes have different lengths.", class = "jax_value_error")
  orig_axes <- if (is.null(axes)) NULL else as.integer(axes)
  axes <- if (!is.null(orig_axes)) orig_axes
          else if (is.null(s)) fft_range(0L, nd) else fft_range(nd - length(s), nd)
  if (anyDuplicated(axes))
    jax_error(full_name, " does not support repeated axes. Got axes ", fft_tuple_str(axes), ".",
              class = "jax_value_error")
  # XLA only supports FFTs over the innermost axes, so rearrange if necessary.
  if (!is.null(orig_axes)) {
    axes <- fft_range(nd - length(axes), nd)
    arr <- jnp_moveaxis(arr, orig_axes, axes)
  }
  shape <- aval_of(arr)$shape
  if (!is.null(s)) {
    in_s <- shape
    in_s[axes + 1L] <- s
    if (fft_type == "IRFFT") in_s[nd] <- in_s[nd] %/% 2L + 1L
    # Cropping
    lim <- pmin(in_s, shape)
    if (any(lim != shape)) arr <- lax_slice(arr, rep(0L, nd), lim)
    # Padding
    if (any(in_s != lim))
      arr <- lax_pad(arr, lax_full(integer(0), 0, aval_of(arr)$dtype),
                     lapply(in_s - lim, function(h) c(0L, h, 0L)))
  } else if (fft_type == "IRFFT") {
    s <- shape[utils::head(axes, -1L) + 1L]
    if (length(axes)) s <- c(s, max(0L, 2L * (shape[axes[length(axes)] + 1L] - 1L)))
  } else {
    s <- shape[axes + 1L]
  }
  transformed <- fft_core_nd(arr, fft_type, s)
  if (!is.null(norm)) {
    transformed <- jnp_multiply(transformed, fft_norm(
      jnp_array(as.numeric(s), dtype = aval_of(transformed)$dtype), func_name, norm))
  }
  if (!is.null(orig_axes)) transformed <- jnp_moveaxis(transformed, axes, orig_axes)
  transformed
}

# Port of jax/_src/numpy/fft.py:_fft_core_nd. XLA supports N-D transforms up to N = 3;
# larger N repeatedly applies N <= 3 transforms (two 2-D transforms for N = 4).
fft_core_nd <- function(arr, fft_type, s) {
  if (length(s) <= 3L) return(lax_fft(arr, fft_type, s))
  n <- if (length(s) == 4L) 2L else 3L
  nd <- fft_ndim(arr)
  src <- fft_range(nd - length(s), nd - n)
  dst <- fft_range(nd - length(s) + n, nd)
  inner <- utils::tail(s, n)
  outer <- utils::head(s, -n)
  if (fft_type %in% c("RFFT", "FFT")) {
    arr <- lax_fft(arr, fft_type, inner)
    arr <- jnp_moveaxis(arr, src, dst)
    arr <- fft_core_nd(arr, "FFT", outer)
    arr <- jnp_moveaxis(arr, dst, src)
  } else {
    arr <- jnp_moveaxis(arr, src, dst)
    arr <- fft_core_nd(arr, "IFFT", outer)
    arr <- jnp_moveaxis(arr, dst, src)
    arr <- lax_fft(arr, fft_type, inner)
  }
  arr
}

# Port of jax/_src/numpy/fft.py:_axis_check_1d.
fft_axis_check_1d <- function(func_name, axis) {
  if (is.list(axis) || length(axis) > 1L) {
    full_name <- paste0("jax.numpy.fft.", func_name)
    jax_error(full_name, " does not support multiple axes. Please use ", full_name, "n. ",
              "Got axis = ", fft_tuple_str(unlist(axis)), ".", class = "jax_value_error")
  }
}

# Port of jax/_src/numpy/fft.py:_fft_core_1d.
fft_core_1d <- function(func_name, fft_type, a, n, axis, norm) {
  fft_axis_check_1d(func_name, axis)
  fft_core(func_name, fft_type, a, n, axis, norm)
}

# Port of jax/_src/numpy/fft.py:_fft_core_2d.
fft_core_2d <- function(func_name, fft_type, a, s, axes, norm) {
  if (length(axes) != 2L)
    jax_error("jax.numpy.fft.", func_name, " only supports 2 axes. Got axes = ",
              fft_tuple_str(axes), ".", class = "jax_value_error")
  fft_core(func_name, fft_type, a, s, axes, norm)
}

#' Discrete Fourier transforms
#'
#' Ports of `jax.numpy.fft`. The one dimensional transforms (`fft`, `ifft`, `rfft`,
#' `irfft`, `hfft`, `ihfft`) act along `axis`; the `2` variants along two `axes`
#' (default the last two) and the `n` variants along `axes` (default all, or the last
#' `length(s)`). Axes are 0-based; negative values count from the end. Inputs are
#' cropped or zero padded to the transform lengths `n` or `s`.
#' @param a Array.
#' @param n Length of the transformed axis (`NULL` uses the input length; for the
#'   inverse real transforms the default is `2 * (m - 1)`).
#' @param axis 0-based axis.
#' @param s Transform lengths along `axes`.
#' @param axes 0-based axes.
#' @param norm `NULL` or `"backward"` (no scaling of the forward transform),
#'   `"ortho"` (both directions scaled by `1 / sqrt(n)`) or `"forward"` (forward
#'   transform scaled by `1 / n`).
#' @return An array.
#' @name jnp_fft
NULL

#' @rdname jnp_fft
#' @export
jnp_fft_fftn <- function(a, s = NULL, axes = NULL, norm = NULL) {
  fft_core("fftn", "FFT", a, s, axes, norm)
}

#' @rdname jnp_fft
#' @export
jnp_fft_ifftn <- function(a, s = NULL, axes = NULL, norm = NULL) {
  fft_core("ifftn", "IFFT", a, s, axes, norm)
}

#' @rdname jnp_fft
#' @export
jnp_fft_rfftn <- function(a, s = NULL, axes = NULL, norm = NULL) {
  fft_core("rfftn", "RFFT", a, s, axes, norm)
}

#' @rdname jnp_fft
#' @export
jnp_fft_irfftn <- function(a, s = NULL, axes = NULL, norm = NULL) {
  fft_core("irfftn", "IRFFT", a, s, axes, norm)
}

#' @rdname jnp_fft
#' @export
jnp_fft_fft <- function(a, n = NULL, axis = -1L, norm = NULL) {
  fft_core_1d("fft", "FFT", a, n, axis, norm)
}

#' @rdname jnp_fft
#' @export
jnp_fft_ifft <- function(a, n = NULL, axis = -1L, norm = NULL) {
  fft_core_1d("ifft", "IFFT", a, n, axis, norm)
}

#' @rdname jnp_fft
#' @export
jnp_fft_rfft <- function(a, n = NULL, axis = -1L, norm = NULL) {
  fft_core_1d("rfft", "RFFT", a, n, axis, norm)
}

#' @rdname jnp_fft
#' @export
jnp_fft_irfft <- function(a, n = NULL, axis = -1L, norm = NULL) {
  fft_core_1d("irfft", "IRFFT", a, n, axis, norm)
}

#' @rdname jnp_fft
#' @export
jnp_fft_hfft <- function(a, n = NULL, axis = -1L, norm = NULL) {
  conj_a <- jnp_conj(a)
  fft_axis_check_1d("hfft", axis)
  nn <- if (is.null(n)) {
    sh <- aval_of(conj_a)$shape
    (sh[norm_axis(axis, length(sh)) + 1L] - 1L) * 2L
  } else n
  jnp_multiply(fft_core_1d("hfft", "IRFFT", conj_a, n, axis, norm), nn)
}

#' @rdname jnp_fft
#' @export
jnp_fft_ihfft <- function(a, n = NULL, axis = -1L, norm = NULL) {
  fft_axis_check_1d("ihfft", axis)
  arr <- jnp_asarray(a)
  sh <- aval_of(arr)$shape
  nn <- if (is.null(n)) sh[norm_axis(axis, length(sh)) + 1L] else n
  output <- fft_core_1d("ihfft", "RFFT", arr, n, axis, norm)
  jnp_multiply(jnp_conj(output), 1 / nn)
}

#' @rdname jnp_fft
#' @export
jnp_fft_fft2 <- function(a, s = NULL, axes = c(-2L, -1L), norm = NULL) {
  fft_core_2d("fft2", "FFT", a, s, axes, norm)
}

#' @rdname jnp_fft
#' @export
jnp_fft_ifft2 <- function(a, s = NULL, axes = c(-2L, -1L), norm = NULL) {
  fft_core_2d("ifft2", "IFFT", a, s, axes, norm)
}

#' @rdname jnp_fft
#' @export
jnp_fft_rfft2 <- function(a, s = NULL, axes = c(-2L, -1L), norm = NULL) {
  fft_core_2d("rfft2", "RFFT", a, s, axes, norm)
}

#' @rdname jnp_fft
#' @export
jnp_fft_irfft2 <- function(a, s = NULL, axes = c(-2L, -1L), norm = NULL) {
  fft_core_2d("irfft2", "IRFFT", a, s, axes, norm)
}

fft_freq_args_check <- function(func_name, n, d) {
  if (is.list(n) || length(n) != 1L)
    jax_error("The n argument of jax.numpy.fft.", func_name, " only takes an int. Got n = [",
              paste(unlist(n), collapse = ", "), "].", class = "jax_value_error")
  if (is.list(d) || (!is_jax_value(d) && length(d) != 1L))
    jax_error("The d argument of jax.numpy.fft.", func_name, " only takes a single value. ",
              "Got d = [", paste(format(unlist(d)), collapse = ", "), "].",
              class = "jax_value_error")
}

#' Sample frequencies of the discrete Fourier transform
#'
#' Ports of `jax.numpy.fft.fftfreq` and `jax.numpy.fft.rfftfreq`.
#' @param n Window length (a single integer).
#' @param d Sample spacing.
#' @param dtype Optional dtype (default the default float type).
#' @param device Ignored (single device).
#' @return A one dimensional array.
#' @export
jnp_fft_fftfreq <- function(n, d = 1.0, dtype = NULL, device = NULL) {
  dtype <- if (is.null(dtype)) default_float() else dtype(dtype)
  fft_freq_args_check("fftfreq", n, d)
  n <- as.integer(n)
  out_dtype <- dtype
  dtype <- complex_component(to_inexact_dtype(dtype))
  i <- jnp_arange(n, dtype = dtype)
  k <- jnp_subtract(jnp_remainder(jnp_add(i, n %/% 2L), n), n %/% 2L)
  result <- jnp_true_divide(jnp_astype(k, dtype), jnp_array(d * n, dtype = dtype))
  jnp_astype(result, out_dtype)
}

#' @rdname jnp_fft_fftfreq
#' @export
jnp_fft_rfftfreq <- function(n, d = 1.0, dtype = NULL, device = NULL) {
  dtype <- if (is.null(dtype)) default_float() else dtype(dtype)
  fft_freq_args_check("rfftfreq", n, d)
  n <- as.integer(n)
  k <- if (n %% 2L == 0L) jnp_arange(0L, n %/% 2L + 1L, dtype = dtype)
       else jnp_arange(0L, (n - 1L) %/% 2L + 1L, dtype = dtype)
  jnp_true_divide(k, jnp_array(d * n, dtype = dtype))
}

#' Shift the zero frequency term to the center
#'
#' Ports of `jax.numpy.fft.fftshift` and `jax.numpy.fft.ifftshift` (its inverse).
#' @param x Array.
#' @param axes 0-based axis or axes to shift (`NULL` shifts all).
#' @return An array.
#' @export
jnp_fft_fftshift <- function(x, axes = NULL) {
  x <- as_jax(x)
  shape <- aval_of(x)$shape
  if (is.null(axes)) {
    axes <- fft_range(0L, length(shape))
    shift <- shape %/% 2L
  } else {
    shift <- shape[norm_axis(axes, length(shape)) + 1L] %/% 2L
  }
  fft_roll(x, shift, axes)
}

#' @rdname jnp_fft_fftshift
#' @export
jnp_fft_ifftshift <- function(x, axes = NULL) {
  x <- as_jax(x)
  shape <- aval_of(x)$shape
  if (is.null(axes)) {
    axes <- fft_range(0L, length(shape))
    shift <- -(shape %/% 2L)
  } else {
    shift <- -(shape[norm_axis(axes, length(shape)) + 1L] %/% 2L)
  }
  fft_roll(x, shift, axes)
}

# ---- jax.scipy.fft (discrete cosine transforms) ---------------------------------------------

# Port of jax/_src/scipy/fft.py:_W4.
fft_W4 <- function(N, k) {
  dt <- complex_of(to_inexact_dtype(result_type(N, k)))
  N_arr <- lax_convert_element_type(N, dt)
  k <- lax_convert_element_type(k, dt)
  jnp_exp(jnp_true_divide(jnp_multiply(-0.5i * pi, k), N_arr))
}

# Port of jax/_src/scipy/fft.py:_dct_interleave.
fft_dct_interleave <- function(x, axis) {
  n <- aval_of(x)$shape[axis + 1L]
  v0 <- lax_slice_in_dim(x, 0L, n, 2L, axis)
  v1 <- lax_rev(lax_slice_in_dim(x, 1L, n, 2L, axis), axis)
  lax_concatenate(list(v0, v1), axis)
}

fft_other_axes <- function(nd, axis) setdiff(fft_range(0L, nd), axis)

# Port of jax/_src/scipy/fft.py:_dct_ortho_norm.
fft_dct_ortho_norm <- function(out, axis) {
  a <- aval_of(out)
  n <- a$shape[axis + 1L]
  factor <- lax_concatenate(list(lax_full(1L, 4, a$dtype), lax_full(n - 1L, 2, a$dtype)), 0L)
  factor <- lax_expand_dims(factor, fft_other_axes(length(a$shape), axis))
  jnp_true_divide(out, lax_sqrt(jnp_multiply(factor, n)))
}

# Port of jax/_src/scipy/fft.py:_dct_deinterleave. The scatters into the even and odd
# positions are written as two interior paddings that are added together.
fft_dct_deinterleave <- function(x, axis) {
  a <- aval_of(x)
  nd <- length(a$shape)
  n <- a$shape[axis + 1L]
  h <- ceiling(n / 2)
  v0 <- lax_slice_in_dim(x, 0L, h, 1L, axis)
  v1 <- lax_rev(lax_slice_in_dim(x, h, n, 1L, axis), axis)
  cfg <- function(lo, m) {
    out <- rep(list(c(0L, 0L, 0L)), nd)
    out[[axis + 1L]] <- as.integer(c(lo, n - lo - max(0, 2 * m - 1), 1))
    out
  }
  jnp_add(lax_pad(v0, 0, cfg(0L, h)), lax_pad(v1, 0, cfg(1L, n - h)))
}

fft_dct_checks <- function(func_name, type, norm) {
  if (type != 2L)
    jax_error("Only DCT type 2 is implemented.", class = "jax_not_implemented_error")
  if (!is.null(norm) && !norm %in% c("backward", "ortho"))
    jax_error("jax.scipy.fft.", func_name, ": norm='", norm, "' is not implemented",
              class = "jax_value_error")
}

fft_pad_axis <- function(x, axis, n) {
  s <- aval_of(x)$shape
  lax_pad(x, 0, lapply(seq_along(s) - 1L, function(a) c(0L, if (a == axis) n - s[a + 1L] else 0L, 0L)))
}

#' Discrete cosine transforms
#'
#' Ports of `jax.scipy.fft.dct`, `dctn`, `idct` and `idctn` (type 2 only).
#' @param x Array.
#' @param type DCT type (only 2 is implemented).
#' @param n Length of the transform (the input is zero padded or cropped).
#' @param axis 0-based axis.
#' @param s Transform lengths along `axes`.
#' @param axes 0-based axes (default all, or the last `length(s)`).
#' @param norm `NULL`, `"backward"` or `"ortho"`.
#' @return An array.
#' @name scipy_fft_dct
NULL

#' @rdname scipy_fft_dct
#' @export
scipy_fft_dct <- function(x, type = 2L, n = NULL, axis = -1L, norm = NULL) {
  x <- as_jax(x)
  fft_dct_checks("dct", type, norm)
  if (is_complex_dtype(aval_of(x)$dtype)) {
    return(lax_complex(scipy_fft_dct(lax_real(x), type = type, n = n, axis = axis, norm = norm),
                       scipy_fft_dct(lax_imag(x), type = type, n = n, axis = axis, norm = norm)))
  }
  nd <- fft_ndim(x)
  axis <- norm_axis(axis, nd)
  if (!is.null(n)) x <- fft_pad_axis(x, axis, n)
  N <- aval_of(x)$shape[axis + 1L]
  v <- fft_dct_interleave(x, axis)
  V <- jnp_fft_fft(v, axis = axis)
  k <- lax_expand_dims(jnp_arange(N, dtype = complex_component(aval_of(V)$dtype)),
                       fft_other_axes(nd, axis))
  out <- jnp_multiply(V, fft_W4(N, k))
  out <- jnp_multiply(2, jnp_real(out))
  if (identical(norm, "ortho")) out <- fft_dct_ortho_norm(out, axis)
  out
}

# Port of jax/_src/scipy/fft.py:_dct2.
fft_dct2 <- function(x, axes, norm) {
  nd <- fft_ndim(x)
  axes <- norm_axis(axes, nd)
  axis1 <- axes[1L]
  axis2 <- axes[2L]
  N1 <- aval_of(x)$shape[axis1 + 1L]
  N2 <- aval_of(x)$shape[axis2 + 1L]
  v <- fft_dct_interleave(fft_dct_interleave(x, axis1), axis2)
  V <- jnp_fft_fftn(v, axes = axes)
  vdt <- aval_of(V)$dtype
  k1 <- lax_expand_dims(jnp_arange(N1, dtype = vdt), fft_other_axes(nd, axis1))
  k2 <- lax_expand_dims(jnp_arange(N2, dtype = vdt), fft_other_axes(nd, axis2))
  out <- jnp_multiply(fft_W4(N1, k1), jnp_add(
    jnp_multiply(fft_W4(N2, k2), V),
    jnp_multiply(fft_W4(N2, jnp_negative(k2)), fft_roll(lax_rev(V, axis2), 1L, axis2))))
  out <- jnp_multiply(2, jnp_real(out))
  if (identical(norm, "ortho"))
    return(fft_dct_ortho_norm(fft_dct_ortho_norm(out, axis1), axis2))
  out
}

# Shared s/axes handling of dctn and idctn (port of jax/_src/scipy/fft.py:dctn).
fft_dctn_axes <- function(x, s, axes) {
  nd <- fft_ndim(x)
  if (!is.null(s)) {
    s <- vapply(s, npa_index, 1L, what = "the 's' argument")
    if (length(s) > nd)
      jax_error("s must have at most x.ndim (", nd, ") elements, got ", length(s),
                class = "jax_value_error")
  }
  if (is.null(axes)) {
    axes <- if (!is.null(s)) fft_range(nd - length(s), nd) else fft_range(0L, nd)
  } else {
    axes <- norm_axis(axes, nd)
    if (anyDuplicated(axes))
      jax_error("repeated axis: ", fft_tuple_str(axes), class = "jax_value_error")
  }
  list(s = s, axes = axes)
}

fft_dctn_pad <- function(x, s, axes) {
  sh <- aval_of(x)$shape
  pads <- rep(0L, length(sh))
  for (i in seq_len(min(length(axes), length(s)))) pads[axes[i] + 1L] <- s[i] - sh[axes[i] + 1L]
  lax_pad(x, 0, lapply(pads, function(p) c(0L, p, 0L)))
}

#' @rdname scipy_fft_dct
#' @export
scipy_fft_dctn <- function(x, type = 2L, s = NULL, axes = NULL, norm = NULL) {
  x <- as_jax(x)
  fft_dct_checks("dctn", type, norm)
  if (is_complex_dtype(aval_of(x)$dtype)) {
    return(lax_complex(scipy_fft_dctn(lax_real(x), type = type, s = s, axes = axes, norm = norm),
                       scipy_fft_dctn(lax_imag(x), type = type, s = s, axes = axes, norm = norm)))
  }
  sa <- fft_dctn_axes(x, s, axes)
  s <- sa$s
  axes <- sa$axes
  if (length(axes) == 1L)
    return(scipy_fft_dct(x, n = if (!is.null(s)) s[1L] else NULL, axis = axes[1L], norm = norm))
  if (!is.null(s)) x <- fft_dctn_pad(x, s, axes)
  if (length(axes) == 2L) return(fft_dct2(x, axes = axes, norm = norm))
  # No axes: nothing to transform.
  if (!length(axes)) return(as_jax(x))
  # compose high-D DCTs from 2D and 1D DCTs:
  for (start in seq(1L, length(axes), by = 2L)) {
    blk <- axes[start:min(start + 1L, length(axes))]
    x <- scipy_fft_dctn(x, axes = blk, norm = norm)
  }
  x
}

#' @rdname scipy_fft_dct
#' @export
scipy_fft_idct <- function(x, type = 2L, n = NULL, axis = -1L, norm = NULL) {
  x <- as_jax(x)
  fft_dct_checks("idct", type, norm)
  if (is_complex_dtype(aval_of(x)$dtype)) {
    return(lax_complex(scipy_fft_idct(lax_real(x), type = type, n = n, axis = axis, norm = norm),
                       scipy_fft_idct(lax_imag(x), type = type, n = n, axis = axis, norm = norm)))
  }
  nd <- fft_ndim(x)
  axis <- norm_axis(axis, nd)
  if (!is.null(n)) x <- fft_pad_axis(x, axis, n)
  N <- aval_of(x)$shape[axis + 1L]
  x <- promote_dtypes_inexact(list(x))[[1L]]
  if (is.null(norm) || norm == "backward") x <- fft_dct_ortho_norm(x, axis)
  x <- fft_dct_ortho_norm(x, axis)
  k <- lax_expand_dims(jnp_arange(N, dtype = aval_of(x)$dtype), fft_other_axes(nd, axis))
  # everything is complex from here...
  w4 <- fft_W4(N, k)
  x <- jnp_astype(x, aval_of(w4)$dtype)
  x <- jnp_true_divide(x, fft_W4(N, k))
  x <- jnp_multiply(jnp_multiply(x, 2), N)
  x <- jnp_fft_ifft(x, axis = axis)
  # convert back to reals..
  fft_dct_deinterleave(jnp_real(x), axis)
}

#' @rdname scipy_fft_dct
#' @export
scipy_fft_idctn <- function(x, type = 2L, s = NULL, axes = NULL, norm = NULL) {
  x <- as_jax(x)
  fft_dct_checks("idctn", type, norm)
  if (is_complex_dtype(aval_of(x)$dtype)) {
    return(lax_complex(scipy_fft_idctn(lax_real(x), type = type, s = s, axes = axes, norm = norm),
                       scipy_fft_idctn(lax_imag(x), type = type, s = s, axes = axes, norm = norm)))
  }
  sa <- fft_dctn_axes(x, s, axes)
  s <- sa$s
  axes <- sa$axes
  if (length(axes) == 1L)
    return(scipy_fft_idct(x, n = if (!is.null(s)) s[1L] else NULL, axis = axes[1L], norm = norm))
  if (!is.null(s)) x <- fft_dctn_pad(x, s, axes)
  # compose high-D DCTs from 1D DCTs:
  for (axis in axes) x <- scipy_fft_idct(x, axis = axis, norm = norm)
  x
}
