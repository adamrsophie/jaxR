# jax.scipy.signal (port of jax/_src/scipy/signal.py and
# jax/_src/third_party/scipy/signal_helper.py).

sig_shape <- function(x) aval_of(x)$shape
sig_ndim <- function(x) length(aval_of(x)$shape)
sig_last <- function(s) s[length(s)]

# Static integer argument (core.concrete_or_error(int, ...)).
sig_int <- function(v, what) {
  as.integer(if (is_jax_value(v)) as.vector(concrete_or_error(v, what)) else v)
}

# jnp.arange that stays one dimensional: jnp_arange gives a rank 0 array when the result
# has a single element and does not start at 0 with step 1.
sig_arange <- function(...) jnp_ravel(jnp_arange(...))

# Python `a or b` for an optional integer argument.
sig_or <- function(a, b) if (is.null(a) || identical(as.numeric(a), 0)) b else a

# Port of jax/_src/dtypes.py:to_complex_dtype.
sig_complex_dtype <- function(dt) {
  ftype <- to_inexact_dtype(dt)
  if (ftype %in% c("float64", "complex128")) "complex128" else "complex64"
}

# Port of jax/_src/numpy/util.py:promote_dtypes_complex for one argument.
sig_promote_complex <- function(x) {
  x <- as_jax(x)
  av <- aval_of(x)
  lax_convert_element_type(x, sig_complex_dtype(av$dtype), weak_type = av$weak)
}

# jnp.flip(x) with axis=None is lax.rev over every dimension.
sig_flip <- function(x) {
  x <- as_jax(x)
  nd <- sig_ndim(x)
  if (nd == 0L) x else lax_rev(x, seq_len(nd) - 1L)
}

# jnp.pad(x, pad_width) in "constant" mode with a zero fill is lax.pad
# (lax_numpy.py:_pad_constant).
sig_pad_zeros <- function(x, pad_width) {
  lax_pad(x, lax_full_like(x, 0, shape = integer(0)),
          lapply(pad_width, function(p) c(as.integer(p), 0L)))
}

sig_mode_check <- function(mode) {
  if (!(is.character(mode) && length(mode) == 1L && mode %in% c("full", "same", "valid")))
    jax_error("mode must be one of ['full', 'same', 'valid']", class = "jax_value_error")
}

#' Convolution and correlation of N-dimensional arrays
#'
#' Ports of `jax.scipy.signal.convolve`, `convolve2d`, `correlate`, `correlate2d` and
#' `fftconvolve`. The direct methods use [lax_conv_general_dilated()]; `fftconvolve`
#' (and `method = "fft"`) multiplies Fourier transforms. Inputs are promoted to an
#' inexact dtype.
#' @param in1 Left-hand input array.
#' @param in2 Right-hand input array, with the same number of dimensions as `in1`.
#' @param mode Output size: `"full"` (the full convolution), `"same"` (the size of
#'   `in1`, centered) or `"valid"` (only positions without zero padding).
#' @param method `"auto"` or `"direct"` (lax convolution) or `"fft"`.
#' @param precision Optional precision of the convolution (see
#'   [lax_conv_general_dilated()]).
#' @param boundary Only `"fill"` is supported.
#' @param fillvalue Only `0` is supported.
#' @param axes Optional 0-based axes over which to convolve; the other axes are
#'   mapped over (they must have equal sizes). By default all axes are convolved.
#' @return An array.
#' @examples
#' \dontrun{
#' scipy_signal_convolve(c(1, 2, 3), c(0, 1, 0.5))
#' scipy_signal_correlate2d(matrix(1:6, 2), matrix(c(1, -1), 1), mode = "same")
#' }
#' @name scipy_signal_convolve
NULL

#' @rdname scipy_signal_convolve
#' @export
scipy_signal_fftconvolve <- function(in1, in2, mode = "full", axes = NULL) {
  # Port of jax/_src/scipy/signal.py:fftconvolve
  a <- promote_dtypes_inexact(list(in1, in2))
  in1 <- a[[1L]]
  in2 <- a[[2L]]
  s1 <- sig_shape(in1)
  s2 <- sig_shape(in2)
  if (length(s1) != length(s2))
    jax_error("in1 and in2 should have the same dimensionality", class = "jax_value_error")
  if (!(is.character(mode) && length(mode) == 1L && mode %in% c("same", "full", "valid")))
    jax_error("mode must be one of ['same', 'full', 'valid']", class = "jax_value_error")
  fftconvolve <- function(x, y) sig_fftconvolve_unbatched(x, y, mode)
  if (is.null(axes)) return(fftconvolve(in1, in2))
  axes <- norm_axis(axes, length(s1))
  mapped_axes <- setdiff(seq_along(s1) - 1L, axes)
  if (any(s1[mapped_axes + 1L] != s2[mapped_axes + 1L]))
    jax_error("mapped axes must have same shape; got in1.shape=", gs_tuple(s1), " in2.shape=",
              gs_tuple(s2), " axes=", gs_tuple(axes), class = "jax_value_error")
  vmap_axis <- function(f, ax) {
    force(f)
    force(ax)
    vmap(f, in_axes = ax, out_axes = ax)
  }
  for (ax in sort(mapped_axes)) fftconvolve <- vmap_axis(fftconvolve, ax)
  fftconvolve(in1, in2)
}

# Port of jax/_src/scipy/signal.py:_fftconvolve_unbatched
sig_fftconvolve_unbatched <- function(in1, in2, mode) {
  s1 <- sig_shape(in1)
  s2 <- sig_shape(in2)
  full_shape <- s1 + s2 - 1L
  fft_shape <- full_shape
  if (mode == "valid") {
    no_swap <- all(s1 >= s2)
    swap <- all(s1 <= s2)
    if (!(no_swap || swap))
      jax_error("For 'valid' mode, One input must be at least as large as the other in every ",
                "dimension.", class = "jax_value_error")
    if (swap) {
      tmp <- in1
      in1 <- in2
      in2 <- tmp
      s1 <- sig_shape(in1)
      s2 <- sig_shape(in2)
    }
  }
  if (all(s1 == 1L | s2 == 1L)) {
    conv <- in1 * in2
  } else {
    if (is_complex_dtype(aval_of(in1)$dtype)) {
      fft <- jnp_fft_fftn
      ifft <- jnp_fft_ifftn
    } else {
      fft <- jnp_fft_rfftn
      ifft <- jnp_fft_irfftn
    }
    sp1 <- fft(in1, fft_shape)
    sp2 <- fft(in2, fft_shape)
    conv <- ifft(sp1 * sp2, fft_shape)
  }
  out_shape <- switch(mode, full = full_shape, same = s1, valid = s1 - s2 + 1L)
  start_indices <- (full_shape - out_shape) %/% 2L
  lax_dynamic_slice(conv, as.list(start_indices), out_shape)
}

# Port of jax/_src/scipy/signal.py:_convolve_nd. (jnp.convolve is not reused: its
# padding for mode = "same" differs.)
sig_convolve_nd <- function(in1, in2, mode, precision = NULL) {
  sig_mode_check(mode)
  in1 <- as_jax(in1)
  in2 <- as_jax(in2)
  s1 <- sig_shape(in1)
  s2 <- sig_shape(in2)
  if (length(s1) != length(s2))
    jax_error("in1 and in2 must have the same number of dimensions", class = "jax_value_error")
  if (prod_int(s1) == 0L || prod_int(s2) == 0L)
    jax_error("zero-size arrays not supported in convolutions, got shapes ", gs_tuple(s1),
              " and ", gs_tuple(s2), ".", class = "jax_value_error")
  a <- promote_dtypes_inexact(list(in1, in2))
  in1 <- a[[1L]]
  in2 <- a[[2L]]
  no_swap <- all(s1 >= s2)
  swap <- all(s1 <= s2)
  if (!(no_swap || swap))
    jax_error("One input must be smaller than the other in every dimension.",
              class = "jax_value_error")
  shape_o <- s2
  if (swap) {
    tmp <- in1
    in1 <- in2
    in2 <- tmp
  }
  shape <- sig_shape(in2)
  in2 <- sig_flip(in2)
  padding <- switch(mode,
    valid = lapply(shape, function(s) c(0L, 0L)),
    same = Map(function(s, s_o) c(s - 1L - (s_o - 1L) %/% 2L, s - s_o + (s_o - 1L) %/% 2L),
               shape, shape_o),
    full = lapply(shape, function(s) c(s - 1L, s - 1L)))
  strides <- rep(1L, length(shape))
  lhs <- lax_reshape(in1, c(1L, 1L, sig_shape(in1)))
  rhs <- lax_reshape(in2, c(1L, 1L, shape))
  result <- lax_conv_general_dilated(lhs, rhs, strides, unname(padding), precision = precision)
  lax_reshape(result, drop_first(sig_shape(result), 2L))
}

#' @rdname scipy_signal_convolve
#' @export
scipy_signal_convolve <- function(in1, in2, mode = "full", method = "auto", precision = NULL) {
  # Port of jax/_src/scipy/signal.py:convolve
  if (identical(method, "fft")) return(scipy_signal_fftconvolve(in1, in2, mode = mode))
  if (identical(method, "direct") || identical(method, "auto"))
    return(sig_convolve_nd(in1, in2, mode, precision = precision))
  jax_error("Got method='", method, "'; expected 'auto', 'fft', or 'direct'.",
            class = "jax_value_error")
}

#' @rdname scipy_signal_convolve
#' @export
scipy_signal_convolve2d <- function(in1, in2, mode = "full", boundary = "fill", fillvalue = 0,
                                    precision = NULL) {
  # Port of jax/_src/scipy/signal.py:convolve2d
  if (!identical(boundary, "fill") || fillvalue != 0)
    jax_error("convolve2d() only supports boundary='fill', fillvalue=0",
              class = "jax_not_implemented_error")
  in1 <- as_jax(in1)
  in2 <- as_jax(in2)
  if (sig_ndim(in1) != 2L || sig_ndim(in2) != 2L)
    jax_error("convolve2d() only supports 2-dimensional inputs.", class = "jax_value_error")
  sig_convolve_nd(in1, in2, mode, precision = precision)
}

#' @rdname scipy_signal_convolve
#' @export
scipy_signal_correlate <- function(in1, in2, mode = "full", method = "auto", precision = NULL) {
  # Port of jax/_src/scipy/signal.py:correlate
  scipy_signal_convolve(in1, sig_flip(jnp_conjugate(in2)), mode, precision = precision,
                        method = method)
}

#' @rdname scipy_signal_convolve
#' @export
scipy_signal_correlate2d <- function(in1, in2, mode = "full", boundary = "fill", fillvalue = 0,
                                     precision = NULL) {
  # Port of jax/_src/scipy/signal.py:correlate2d
  if (!identical(boundary, "fill") || fillvalue != 0)
    jax_error("correlate2d() only supports boundary='fill', fillvalue=0",
              class = "jax_not_implemented_error")
  in1 <- as_jax(in1)
  in2 <- as_jax(in2)
  if (sig_ndim(in1) != 2L || sig_ndim(in2) != 2L)
    jax_error("correlate2d() only supports 2-dimensional inputs.", class = "jax_value_error")
  s1 <- sig_shape(in1)
  s2 <- sig_shape(in2)
  swap <- all(s1 <= s2)
  same_shape <- all(s1 == s2)
  if (identical(mode, "same")) {
    result <- sig_flip(sig_convolve_nd(sig_flip(in1), jnp_conjugate(in2), mode, precision))
  } else if (identical(mode, "valid")) {
    if (swap && !same_shape) {
      result <- sig_convolve_nd(sig_flip(in2), jnp_conjugate(in1), mode, precision)
    } else {
      result <- sig_flip(sig_convolve_nd(sig_flip(in1), jnp_conjugate(in2), mode, precision))
    }
  } else if (swap) {
    result <- jnp_conjugate(sig_convolve_nd(sig_flip(in2), jnp_conjugate(in1), mode, precision))
  } else {
    result <- sig_flip(sig_convolve_nd(sig_flip(in1), jnp_conjugate(in2), mode, precision))
  }
  result
}

#' Remove a linear or constant trend
#'
#' Port of `jax.scipy.signal.detrend`.
#' @param data Array.
#' @param axis 0-based axis along which to detrend.
#' @param type `"linear"` (least squares line, piecewise between the breakpoints) or
#'   `"constant"` (the mean).
#' @param bp Static breakpoints (0-based positions along `axis`) for `"linear"`.
#' @param overwrite_data Not supported (must be `NULL`).
#' @return The detrended array (inexact dtype).
#' @examples
#' \dontrun{
#' scipy_signal_detrend(c(1, 3, 2, 5, 4))
#' }
#' @export
scipy_signal_detrend <- function(data, axis = -1L, type = "linear", bp = 0L,
                                 overwrite_data = NULL) {
  # Port of jax/_src/scipy/signal.py:detrend
  if (!is.null(overwrite_data))
    jax_error("overwrite_data argument not implemented.", class = "jax_not_implemented_error")
  if (!(is.character(type) && length(type) == 1L && type %in% c("constant", "linear")))
    jax_error("Trend type must be 'linear' or 'constant'.", class = "jax_value_error")
  data_arr <- promote_dtypes_inexact(list(jnp_asarray(data)))[[1L]]
  if (type == "constant") return(data_arr - jnp_mean(data_arr, axis, keepdims = TRUE))
  s <- sig_shape(data_arr)
  N <- s[norm_axis(axis, length(s)) + 1L]
  # bp is static, so R operations are used.
  bp <- if (is_jax_value(bp)) as.vector(concrete_or_error(bp, "detrend breakpoints")) else bp
  bp_arr <- sort(unique(c(0, as.numeric(bp), N)))
  if (bp_arr[1L] < 0 || bp_arr[length(bp_arr)] > N)
    jax_error("Breakpoints must be non-negative and less than length of data along given axis.",
              class = "jax_value_error")
  data_arr <- jnp_moveaxis(data_arr, axis, 0L)
  shape <- sig_shape(data_arr)
  data_arr <- jnp_reshape(data_arr, c(N, -1L))
  dt <- aval_of(data_arr)$dtype
  for (m in seq_len(length(bp_arr) - 1L)) {
    Npts <- bp_arr[m + 1L] - bp_arr[m]
    A <- jnp_transpose(jnp_stack(list(
      jnp_ones(Npts, dtype = dt),
      sig_arange(1L, Npts + 1L, dtype = dt) / as_jax(Npts, dt))))
    lo <- as.integer(bp_arr[m])
    hi <- as.integer(bp_arr[m + 1L])
    coef <- jnp_linalg_lstsq(A, data_arr[sl(lo, hi)])[[1L]]
    data_arr <- at_add(data_arr, sl(lo, hi), value = -jnp_matmul(A, coef, precision = "highest"))
  }
  jnp_moveaxis(jnp_reshape(data_arr, shape), 0L, axis)
}

# Port of jax/_src/third_party/scipy/signal_helper.py:_triage_segments (and the Hann
# default of istft). Windows other than "hann" go through sig_get_window().
sig_hann <- function(n) {
  if (n == 1L) return(jnp_ones(1L))
  jnp_sin(jnp_linspace(0, pi, n, endpoint = FALSE))^2
}

# Port of scipy.signal.get_window(window, n) with fftbins = TRUE (periodic windows: the
# symmetric window of length n + 1 without its last sample), which JAX calls for every
# window but "hann". `window` is a name, a number (the beta of a Kaiser window), or a
# list of a name and its parameters. Returns R doubles.
sig_get_window <- function(window, n) {
  args <- list()
  if (is.list(window)) {
    args <- window[-1L]
    window <- window[[1L]]
  } else if (is.numeric(window)) {
    args <- list(window)
    window <- "kaiser"
  }
  unsupported <- function()
    jax_error("window=", if (is.character(window)) paste0("'", window, "'") else format(window),
              " is not one of the windows jaxR implements (boxcar, triang, bartlett, ",
              "blackman, hamming, hann, flattop, parzen, bohman, blackmanharris, nuttall, ",
              "barthann, cosine, lanczos, exponential, tukey, kaiser, gaussian, ",
              "general_gaussian, general_hamming, general_cosine); pass the window values ",
              "as an array", class = "jax_not_implemented_error")
  if (!is.character(window) || length(window) != 1L) unsupported()
  name <- switch(window,
    boxcar = , box = , ones = , rect = , rectangular = "boxcar",
    triangle = , triang = , tri = "triang",
    blackman = , black = , blk = "blackman",
    hamming = , hamm = , ham = "hamming",
    hann = , han = "hann",
    bartlett = , bart = , brt = "bartlett",
    flattop = , flat = , flt = "flattop",
    parzen = , parz = , par = "parzen",
    bohman = , bman = , bmn = "bohman",
    blackmanharris = , blackharr = , bkh = "blackmanharris",
    nuttall = , nutl = , nut = "nuttall",
    barthann = , brthan = , bth = "barthann",
    cosine = , halfcosine = "cosine",
    lanczos = , sinc = "lanczos",
    exponential = , poisson = "exponential",
    tukey = , tuk = "tukey",
    kaiser = , ksr = "kaiser",
    gaussian = , gauss = , gss = "gaussian",
    "general gaussian" = , general_gaussian = , "general gauss" = , general_gauss = , ggs = "general_gaussian",
    general_hamming = "general_hamming",
    "general cosine" = , general_cosine = "general_cosine",
    unsupported())
  needs <- c(kaiser = 1L, gaussian = 1L, general_gaussian = 2L, general_hamming = 1L,
             general_cosine = 1L)
  if (name %in% names(needs) && length(args) < needs[[name]])
    jax_error("The '", window, "' window needs one or more parameters -- pass a list.",
              class = "jax_value_error")
  if (n <= 1L) return(rep(1, n))
  M <- n + 1L
  k <- seq_len(M) - 1
  gcos <- function(a) {
    fac <- seq(-pi, pi, length.out = M)
    Reduce(`+`, lapply(seq_along(a), function(i) a[i] * cos((i - 1) * fac)))
  }
  w <- switch(name,
    boxcar = rep(1, M),
    triang = {
      h <- seq_len((M + 1L) %/% 2L)
      if (M %% 2L == 0L) { v <- (2 * h - 1) / M; c(v, rev(v)) }
      else { v <- 2 * h / (M + 1); c(v, rev(v)[-1L]) }
    },
    bartlett = ifelse(k <= (M - 1) / 2, 2 * k / (M - 1), 2 - 2 * k / (M - 1)),
    blackman = gcos(c(0.42, 0.5, 0.08)),
    hamming = gcos(c(0.54, 0.46)),
    hann = gcos(c(0.5, 0.5)),
    general_hamming = gcos(c(args[[1L]], 1 - args[[1L]])),
    general_cosine = gcos(unlist(args[[1L]])),
    nuttall = gcos(c(0.3635819, 0.4891775, 0.1365995, 0.0106411)),
    blackmanharris = gcos(c(0.35875, 0.48829, 0.14128, 0.01168)),
    flattop = gcos(c(0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368)),
    cosine = sin(pi / M * (k + 0.5)),
    bohman = {
      fac <- abs(seq(-1, 1, length.out = M)[-c(1L, M)])
      c(0, (1 - fac) * cos(pi * fac) + sin(pi * fac) / pi, 0)
    },
    parzen = {
      x <- k - (M - 1) / 2
      xa <- x[x < -(M - 1) / 4]
      xb <- abs(x[abs(x) <= (M - 1) / 4])
      wa <- 2 * (1 - abs(xa) / (M / 2))^3
      c(wa, 1 - 6 * (xb / (M / 2))^2 + 6 * (xb / (M / 2))^3, rev(wa))
    },
    barthann = {
      fac <- abs(k / (M - 1) - 0.5)
      0.62 - 0.48 * fac + 0.38 * cos(2 * pi * fac)
    },
    lanczos = {
      sinc <- function(v) ifelse(v == 0, 1, sin(pi * v) / (pi * v))
      right <- function(from) if (from < M) sinc(2 * seq(from, M - 1) / (M - 1) - 1) else numeric(0)
      if (M %% 2L == 0L) { h <- right(M / 2); c(rev(h), h) }
      else { h <- right((M + 1) / 2); c(rev(h), 1, h) }
    },
    exponential = {
      if (length(args) && !is.null(args[[1L]]))
        jax_error("If sym==True, center must be None.", class = "jax_value_error")
      tau <- if (length(args) > 1L) args[[2L]] else 1
      exp(-abs(k - (M - 1) / 2) / tau)
    },
    tukey = {
      alpha <- if (length(args)) args[[1L]] else 0.5
      if (alpha <= 0) rep(1, M)
      else if (alpha >= 1) gcos(c(0.5, 0.5))
      else {
        width <- floor(alpha * (M - 1) / 2)
        n1 <- k[seq_len(width + 1L)]
        n3 <- k[seq.int(M - width, M)]
        c(0.5 * (1 + cos(pi * (-1 + 2 * n1 / alpha / (M - 1)))),
          rep(1, M - 2 * width - 2),
          0.5 * (1 + cos(pi * (-2 / alpha + 1 + 2 * n3 / alpha / (M - 1)))))
      }
    },
    kaiser = {
      alpha <- (M - 1) / 2
      besselI(args[[1L]] * sqrt(pmax(1 - ((k - alpha) / alpha)^2, 0)), 0) / besselI(args[[1L]], 0)
    },
    gaussian = exp(-(k - (M - 1) / 2)^2 / (2 * args[[1L]]^2)),
    general_gaussian = exp(-0.5 * abs((k - (M - 1) / 2) / args[[2L]])^(2 * args[[1L]])))
  w[-M]
}

sig_triage_segments <- function(window, nperseg, input_length, dtype) {
  if (is.character(window) || is.list(window)) {
    nperseg_int <- if (is.null(nperseg)) input_length else as.integer(nperseg)
    if (nperseg_int > input_length) {
      warning("nperseg=", nperseg_int, " is greater than input_length=", input_length,
              ", using nperseg=", input_length, call. = FALSE)
      nperseg_int <- input_length
    }
    win <- if (identical(window, "hann")) jnp_array(sig_hann(nperseg_int), dtype = dtype)
           else jnp_array(array(sig_get_window(window, nperseg_int), dim = nperseg_int),
                          dtype = dtype)
  } else {
    win <- jnp_asarray(window, dtype = dtype)
    nperseg_int <- if (is.null(nperseg)) prod_int(sig_shape(win)) else as.integer(nperseg)
    if (sig_ndim(win) != 1L) jax_error("window must be 1-D", class = "jax_value_error")
    if (input_length < prod_int(sig_shape(win)))
      jax_error("window is longer than input signal", class = "jax_value_error")
    if (nperseg_int != prod_int(sig_shape(win)))
      jax_error("value specified for nperseg is different from length of window",
                class = "jax_value_error")
  }
  list(win, nperseg_int)
}

# Port of jax/_src/third_party/scipy/signal_helper.py:_median_bias
sig_median_bias <- function(n) {
  ii_2 <- jnp_arange(2, n, 2, dtype = default_float())
  1L + jnp_sum(1 / (ii_2 + 1L) - 1 / ii_2)
}

# Port of jax/_src/scipy/signal.py:_fft_helper
sig_fft_helper <- function(x, win, detrend_func, nperseg, noverlap, nfft, sides) {
  if (dtype_kind(aval_of(x)$dtype) == "i") x <- lax_convert_element_type(x, aval_of(win)$dtype)
  s <- sig_shape(x)
  batch_shape <- s[-length(s)]
  signal_length <- sig_last(s)
  if (nperseg == 1L && noverlap == 0L) {
    result <- lax_expand_dims(x, length(s))
  } else {
    step <- nperseg - noverlap
    starts <- sig_arange(0L, signal_length - nperseg + 1L, step)
    slice_func <- function(start_index) lax_dynamic_slice_in_dim(x, start_index, nperseg, axis = -1L)
    result <- vmap(slice_func, out_axes = -2L)(starts)
  }
  result <- detrend_func(result)
  if (is_complex_dtype(aval_of(win)$dtype)) result <- sig_promote_complex(result)
  result <- jnp_reshape(win, c(rep(1L, length(batch_shape)), 1L, nperseg)) * result
  if (sides == "twosided") jnp_fft_fft(result, n = nfft)
  else jnp_fft_rfft(jnp_real(result), n = nfft)
}

# Port of jax/_src/scipy/signal.py:odd_ext
sig_odd_ext <- function(x, n, axis = -1L) {
  if (n < 1L) return(x)
  s <- sig_shape(x)
  ax <- norm_axis(axis, length(s))
  size <- s[ax + 1L]
  if (n > size - 1L)
    jax_error("The extension length n (", n, ") is too big. It must not exceed x.shape[axis]-1, ",
              "which is ", size - 1L, ".", class = "jax_value_error")
  left_end <- lax_slice_in_dim(x, 0L, 1L, axis = ax)
  left_ext <- lax_rev(lax_slice_in_dim(x, 1L, n + 1L, axis = ax), ax)
  right_end <- lax_slice_in_dim(x, size - 1L, size, axis = ax)
  right_ext <- lax_rev(lax_slice_in_dim(x, size - (n + 1L), size - 1L, axis = ax), ax)
  jnp_concatenate(list(2L * left_end - left_ext, x, 2L * right_end - right_ext), axis = ax)
}

# Port of jax/_src/scipy/signal.py:_spectral_helper
sig_spectral_helper <- function(x, y, fs = 1.0, window = "hann", nperseg = NULL, noverlap = NULL,
                                nfft = NULL, detrend_type = "constant", return_onesided = TRUE,
                                scaling = "density", axis = -1L, mode = "psd", boundary = NULL,
                                padded = FALSE) {
  if (!(is.character(mode) && length(mode) == 1L && mode %in% c("psd", "stft")))
    jax_error("Unknown value for mode ", mode, ", must be one of: ('psd', 'stft')",
              class = "jax_value_error")
  make_pad <- function(pad_mode) {
    function(x, n, axis = -1L) {
      nd <- sig_ndim(x)
      pad_width <- rep(list(c(0L, 0L)), nd)
      pad_width[[norm_axis(axis, nd) + 1L]] <- c(n, n)
      jnp_pad(x, pad_width, pad_mode)
    }
  }
  pad_zeros <- function(x, n, axis = -1L) {
    nd <- sig_ndim(x)
    pad_width <- rep(list(c(0L, 0L)), nd)
    pad_width[[norm_axis(axis, nd) + 1L]] <- c(n, n)
    sig_pad_zeros(x, pad_width)
  }
  boundary_funcs <- list(even = make_pad("reflect"), odd = sig_odd_ext,
                         constant = make_pad("edge"), zeros = pad_zeros)
  if (!is.null(boundary) && !(is.character(boundary) && length(boundary) == 1L &&
                              boundary %in% names(boundary_funcs)))
    jax_error("Unknown boundary option '", format(boundary), "', must be one of: ",
              "['even', 'odd', 'constant', 'zeros', None]", class = "jax_value_error")
  x <- as_jax(x)
  axis <- norm_axis(sig_int(axis, "axis of windowed-FFT"), sig_ndim(x))
  if (is.null(y)) {
    x <- promote_dtypes_inexact(list(x))[[1L]]
    y_arr <- x
    outershape <- sig_shape(x)[-(axis + 1L)]
  } else {
    if (mode != "psd")
      jax_error("two-argument mode is available only when mode=='psd'", class = "jax_value_error")
    a <- promote_dtypes_inexact(list(x, y))
    x <- a[[1L]]
    y_arr <- a[[2L]]
    if (sig_ndim(x) != sig_ndim(y_arr))
      jax_error("two-arguments must have the same rank ({x.ndim} vs {y.ndim}).",
                class = "jax_value_error")
    outershape <- tryCatch(
      broadcast_shapes_impl(list(sig_shape(x)[-(axis + 1L)], sig_shape(y_arr)[-(axis + 1L)])),
      jax_error = function(e) jax_error("x and y cannot be broadcast together.",
                                        class = "jax_value_error"))
  }
  result_dtype <- sig_complex_dtype(aval_of(x)$dtype)
  freq_dtype <- complex_component(result_dtype)

  if (!is.null(nperseg)) {
    nperseg <- sig_int(nperseg, "nperseg of windowed-FFT")
    if (nperseg < 1L) jax_error("nperseg must be a positive integer", class = "jax_value_error")
  }
  # Parse the window; for an array the segment length is its length.
  tw <- sig_triage_segments(window, nperseg, input_length = sig_shape(x)[axis + 1L],
                            dtype = aval_of(x)$dtype)
  win <- tw[[1L]]
  nperseg_int <- tw[[2L]]
  noverlap_int <- if (is.null(noverlap)) nperseg_int %/% 2L
                  else sig_int(noverlap, "noverlap of windowed-FFT")
  nfft_int <- if (is.null(nfft)) nperseg_int else sig_int(nfft, "nfft of windowed-FFT")

  # Special cases for size == 0.
  if (is.null(y)) {
    if (prod_int(sig_shape(x)) == 0L)
      return(list(jnp_zeros(sig_shape(x), freq_dtype), jnp_zeros(sig_shape(x), freq_dtype),
                  jnp_zeros(sig_shape(x), result_dtype)))
  } else if (prod_int(sig_shape(x)) == 0L || prod_int(sig_shape(y_arr)) == 0L) {
    shape <- append(outershape, min(sig_shape(x)[axis + 1L], sig_shape(y_arr)[axis + 1L]),
                    after = axis)
    return(list(jnp_zeros(shape, freq_dtype), jnp_zeros(shape, freq_dtype),
                jnp_zeros(shape, result_dtype)))
  }

  # Move the time axis to the end.
  x <- jnp_moveaxis(x, axis, -1L)
  if (!is.null(y) && sig_ndim(y_arr) > 1L) y_arr <- jnp_moveaxis(y_arr, axis, -1L)

  # Zero pad x or y to the same length.
  if (!is.null(y) && sig_last(sig_shape(x)) != sig_last(sig_shape(y_arr))) {
    sx <- sig_shape(x)
    sy <- sig_shape(y_arr)
    if (sig_last(sx) < sig_last(sy)) {
      pad_shape <- sx
      pad_shape[length(sx)] <- sig_last(sy) - sig_last(sx)
      x <- jnp_concatenate(list(x, jnp_zeros_like(x, shape = pad_shape)), -1L)
    } else {
      pad_shape <- sy
      pad_shape[length(sy)] <- sig_last(sx) - sig_last(sy)
      y_arr <- jnp_concatenate(list(y_arr, jnp_zeros_like(x, shape = pad_shape)), -1L)
    }
  }

  if (nfft_int < nperseg_int)
    jax_error("nfft must be greater than or equal to nperseg.", class = "jax_value_error")
  if (noverlap_int >= nperseg_int)
    jax_error("noverlap must be less than nperseg.", class = "jax_value_error")
  nstep <- nperseg_int - noverlap_int

  if (!is.null(boundary)) {
    ext_func <- boundary_funcs[[boundary]]
    x <- ext_func(x, nperseg_int %/% 2L, axis = -1L)
    if (!is.null(y)) y_arr <- ext_func(y_arr, nperseg_int %/% 2L, axis = -1L)
  }

  if (isTRUE(padded)) {
    # Pad to an integer number of windowed segments, i.e. make
    # x.shape[-1] = nperseg + (nseg - 1) * nstep with integer nseg.
    sx <- sig_shape(x)
    nadd <- (-(sig_last(sx) - nperseg_int) %% nstep) %% nperseg_int
    x <- jnp_concatenate(list(x, jnp_zeros_like(x, shape = c(sx[-length(sx)], nadd))), -1L)
    if (!is.null(y)) {
      sy <- sig_shape(y_arr)
      y_arr <- jnp_concatenate(list(y_arr, jnp_zeros_like(x, shape = c(sy[-length(sy)], nadd))),
                               -1L)
    }
  }

  # Detrending function. The callable branch mirrors upstream, where `axis` is already
  # canonicalized (so the `axis != -1` test there always holds).
  if (is.character(detrend_type)) {
    detrend_func <- function(d) scipy_signal_detrend(d, type = detrend_type, axis = -1L)
  } else if (is.function(detrend_type)) {
    detrend_func <- function(d) {
      d <- jnp_moveaxis(d, axis, -1L)
      d <- detrend_type(d)
      jnp_moveaxis(d, -1L, axis)
    }
  } else if (is.null(detrend_type) || isFALSE(detrend_type) || identical(detrend_type, 0)) {
    detrend_func <- function(d) d
  } else {
    jax_error("Unsupported detrend type: ", format(detrend_type), class = "jax_value_error")
  }

  # Scale.
  if (identical(scaling, "density")) {
    scale <- 1.0 / (fs * jnp_sum(win * win))
  } else if (identical(scaling, "spectrum")) {
    scale <- 1.0 / jnp_sum(win)^2L
  } else {
    jax_error("Unknown scaling: ", scaling, class = "jax_value_error")
  }
  if (mode == "stft") scale <- jnp_sqrt(scale)
  scale <- sig_promote_complex(scale)

  # One sided or two sided.
  if (isTRUE(return_onesided)) {
    sides <- "onesided"
    if (is_complex_dtype(aval_of(x)$dtype) ||
        (!is.null(y) && is_complex_dtype(aval_of(y_arr)$dtype))) {
      sides <- "twosided"
      warning("Input data is complex, switching to return_onesided=False", call. = FALSE)
    }
  } else {
    sides <- "twosided"
  }
  freqs <- if (sides == "twosided") jnp_fft_fftfreq(nfft_int, 1 / fs, dtype = freq_dtype)
           else jnp_fft_rfftfreq(nfft_int, 1 / fs, dtype = freq_dtype)

  # The windowed FFTs.
  result <- sig_fft_helper(x, win, detrend_func, nperseg_int, noverlap_int, nfft_int, sides)
  if (!is.null(y)) {
    result_y <- sig_fft_helper(y_arr, win, detrend_func, nperseg_int, noverlap_int, nfft_int,
                               sides)
    result <- jnp_conjugate(result) * result_y
  } else if (mode == "psd") {
    result <- jnp_conjugate(result) * result
  }
  result <- result * scale
  if (sides == "onesided" && mode == "psd") {
    end <- if (nfft_int %% 2L) NULL else -1L
    result <- at_multiply(result, ellipsis, sl(1L, end), value = 2L)
  }

  time <- sig_arange(nperseg_int / 2, sig_last(sig_shape(x)) - nperseg_int / 2 + 1,
                   nperseg_int - noverlap_int, dtype = freq_dtype) / fs
  if (!is.null(boundary)) time <- time - (nperseg_int / 2) / fs

  result <- lax_convert_element_type(result, result_dtype)
  # All imaginary parts are zero.
  if (is.null(y) && mode != "stft") result <- jnp_real(result)
  # Move the frequency axis back to where the data came from.
  result <- jnp_moveaxis(result, -1L, axis)
  list(freqs, time, result)
}

#' Spectral analysis: STFT, inverse STFT, cross and power spectral densities
#'
#' Ports of `jax.scipy.signal.stft`, `istft`, `csd` and `welch`. Signals are split in
#' overlapping segments (`nperseg` long, `noverlap` overlap) that are detrended,
#' windowed and Fourier transformed. Each function returns an unnamed list, like the
#' tuples of JAX: `stft` gives `list(f, t, Zxx)`, `istft` `list(t, x)`, `csd`
#' `list(f, Pxy)` and `welch` `list(f, Pxx)`.
#' @param x Signal array (time along `axis`).
#' @param y Second signal for `csd` (or `NULL`, giving the power spectral density of
#'   `x`).
#' @param fs Sampling frequency.
#' @param window `"hann"` (periodic Hann window) or a 1-D array of window values.
#'   Other window names need `scipy.signal.get_window` upstream and are not supported.
#' @param nperseg Segment length (static integer).
#' @param noverlap Overlap of the segments (default `nperseg %/% 2`).
#' @param nfft FFT length (default `nperseg`).
#' @param detrend Detrending of each segment: `"constant"`, `"linear"`, a function, or
#'   `FALSE` for none.
#' @param return_onesided One sided spectrum for real data.
#' @param scaling `"density"` or `"spectrum"`.
#' @param axis 0-based axis of the time samples.
#' @param average Averaging over segments: `"mean"` or `"median"`.
#' @param boundary Signal extension at both ends for `stft`: `"zeros"`, `"even"`,
#'   `"odd"`, `"constant"` or `NULL`; for `istft`, whether the input was extended.
#' @param padded Zero pad the signal to a whole number of segments.
#' @param Zxx STFT array.
#' @param input_onesided Whether `Zxx` is a one sided spectrum.
#' @param time_axis,freq_axis 0-based axes of `Zxx`.
#' @return An unnamed list of arrays (see Description).
#' @examples
#' \dontrun{
#' x <- sin(seq(0, 20, length.out = 64))
#' res <- scipy_signal_stft(x, nperseg = 16L)
#' scipy_signal_istft(res[[3L]], nperseg = 16L)
#' scipy_signal_welch(x, nperseg = 16L)
#' }
#' @name scipy_signal_spectral
NULL

#' @rdname scipy_signal_spectral
#' @export
scipy_signal_stft <- function(x, fs = 1.0, window = "hann", nperseg = 256L, noverlap = NULL,
                              nfft = NULL, detrend = FALSE, return_onesided = TRUE,
                              boundary = "zeros", padded = TRUE, axis = -1L) {
  # Port of jax/_src/scipy/signal.py:stft
  sig_spectral_helper(x, NULL, fs, window, nperseg, noverlap, nfft, detrend, return_onesided,
                      scaling = "spectrum", axis = axis, mode = "stft", boundary = boundary,
                      padded = padded)
}

#' @rdname scipy_signal_spectral
#' @export
scipy_signal_csd <- function(x, y, fs = 1.0, window = "hann", nperseg = NULL, noverlap = NULL,
                             nfft = NULL, detrend = "constant", return_onesided = TRUE,
                             scaling = "density", axis = -1L, average = "mean") {
  # Port of jax/_src/scipy/signal.py:csd
  res <- sig_spectral_helper(x, y, fs, window, nperseg, noverlap, nfft, detrend,
                             return_onesided, scaling, axis, mode = "psd")
  freqs <- res[[1L]]
  Pxy <- res[[3L]]
  # Complex output when x is not y.
  if (!is.null(y)) Pxy <- Pxy + 0i
  s <- sig_shape(Pxy)
  # Average over the segments.
  if (length(s) >= 2L && prod_int(s) > 0L) {
    if (sig_last(s) > 1L) {
      if (identical(average, "median")) {
        bias <- lax_convert_element_type(sig_median_bias(sig_last(s)), aval_of(Pxy)$dtype)
        if (is_complex_dtype(aval_of(Pxy)$dtype)) {
          Pxy <- jnp_median(jnp_real(Pxy), axis = -1L) + 1i * jnp_median(jnp_imag(Pxy), axis = -1L)
        } else {
          Pxy <- jnp_median(Pxy, axis = -1L)
        }
        Pxy <- Pxy / bias
      } else if (identical(average, "mean")) {
        Pxy <- jnp_mean(Pxy, axis = -1L)
      } else {
        jax_error("average must be \"median\" or \"mean\", got ", format(average),
                  class = "jax_value_error")
      }
    } else {
      Pxy <- jnp_reshape(Pxy, s[-length(s)])
    }
  }
  list(freqs, Pxy)
}

#' @rdname scipy_signal_spectral
#' @export
scipy_signal_welch <- function(x, fs = 1.0, window = "hann", nperseg = NULL, noverlap = NULL,
                               nfft = NULL, detrend = "constant", return_onesided = TRUE,
                               scaling = "density", axis = -1L, average = "mean") {
  # Port of jax/_src/scipy/signal.py:welch
  res <- scipy_signal_csd(x, NULL, fs = fs, window = window, nperseg = nperseg,
                          noverlap = noverlap, nfft = nfft, detrend = detrend,
                          return_onesided = return_onesided, scaling = scaling, axis = axis,
                          average = average)
  list(res[[1L]], jnp_real(res[[2L]]))
}

# Port of jax/_src/scipy/signal.py:_overlap_and_add
sig_overlap_and_add <- function(x, step_size) {
  x <- as_jax(x)
  step_size <- sig_int(step_size, "step_size for overlap_and_add")
  s <- sig_shape(x)
  nd <- length(s)
  if (nd < 2L)
    jax_error("Input must have (..., frames, frame_length) shape.", class = "jax_value_error")
  batch_shape <- s[seq_len(nd - 2L)]
  nframes <- s[nd - 1L]
  segment_len <- s[nd]
  flat_batchsize <- prod_int(batch_shape)
  x <- jnp_reshape(x, c(flat_batchsize, nframes, segment_len))
  output_size <- step_size * (nframes - 1L) + segment_len
  nstep_per_segment <- 1L + (segment_len - 1L) %/% step_size
  # Axes: B batch, N frames, S steps per segment, T segment length divided by S.
  padded_segment_len <- nstep_per_segment * step_size
  x <- sig_pad_zeros(x, list(c(0L, 0L), c(0L, 0L), c(0L, padded_segment_len - segment_len)))
  x <- jnp_reshape(x, c(flat_batchsize, nframes, nstep_per_segment, step_size))
  # Shifted signals come from reinterpreting the flattened array with a shrunken axis
  # (as in TensorFlow's overlap_and_add).
  x <- jnp_transpose(x, c(0L, 2L, 1L, 3L))
  x <- sig_pad_zeros(x, list(c(0L, 0L), c(0L, 0L), c(0L, nframes), c(0L, 0L)))
  shrunken <- sig_shape(x)[3L] - 1L
  x <- jnp_reshape(x, c(flat_batchsize, -1L))
  x <- x[, sl(NULL, nstep_per_segment * shrunken * step_size)]
  x <- jnp_reshape(x, c(flat_batchsize, nstep_per_segment, shrunken * step_size))
  # Sum the shifted segments and truncate to output_size.
  x <- jnp_sum(x, axis = 1L)[, sl(NULL, output_size)]
  jnp_reshape(x, c(batch_shape, -1L))
}

#' @rdname scipy_signal_spectral
#' @export
scipy_signal_istft <- function(Zxx, fs = 1.0, window = "hann", nperseg = NULL, noverlap = NULL,
                               nfft = NULL, input_onesided = TRUE, boundary = TRUE,
                               time_axis = -1L, freq_axis = -2L) {
  # Port of jax/_src/scipy/signal.py:istft
  Zxx <- jnp_asarray(Zxx)
  nd <- sig_ndim(Zxx)
  if (nd < 2L) jax_error("Input stft must be at least 2d!", class = "jax_value_error")
  freq_axis <- norm_axis(freq_axis, nd)
  time_axis <- norm_axis(time_axis, nd)
  if (freq_axis == time_axis)
    jax_error("Must specify differing time and frequency axes!", class = "jax_value_error")
  Zxx <- jnp_asarray(Zxx, dtype = sig_complex_dtype(aval_of(Zxx)$dtype))
  nfreq <- sig_shape(Zxx)[freq_axis + 1L]
  n_default <- if (isTRUE(input_onesided)) 2L * (nfreq - 1L) else nfreq
  nperseg_int <- sig_int(sig_or(nperseg, n_default), "nperseg: segment length of STFT")
  if (nperseg_int < 1L) jax_error("nperseg must be a positive integer", class = "jax_value_error")
  if (is.null(nfft)) {
    nfft_int <- n_default
    # Odd nperseg, no FFT padding.
    if (isTRUE(input_onesided) && nperseg_int == n_default + 1L) nfft_int <- nfft_int + 1L
  } else {
    nfft_int <- sig_int(nfft, "nfft of STFT")
  }
  if (nfft_int < nperseg_int)
    jax_error("FFT length (", nfft_int, ") must be longer than nperseg (", nperseg_int, ").",
              class = "jax_value_error")
  noverlap_int <- sig_int(sig_or(noverlap, nperseg_int %/% 2L), "noverlap of STFT")
  if (noverlap_int >= nperseg_int)
    jax_error("noverlap must be less than nperseg.", class = "jax_value_error")
  nstep <- nperseg_int - noverlap_int

  # Rearrange the axes if necessary.
  if (time_axis != nd - 1L || freq_axis != nd - 2L) {
    outer_idxs <- setdiff(seq_len(nd) - 1L, c(time_axis, freq_axis))
    Zxx <- jnp_transpose(Zxx, c(outer_idxs, freq_axis, time_axis))
  }

  # Inverse FFT; xsubs is [..., T, N] with N frames of length T. As upstream, the FFT
  # length is the `nfft` argument itself.
  ifunc <- if (isTRUE(input_onesided)) jnp_fft_irfft else jnp_fft_ifft
  xsubs <- ifunc(Zxx, axis = -2L, n = if (is.null(nfft)) NULL else sig_int(nfft, "nfft of STFT"))
  xsubs <- xsubs[ellipsis, sl(NULL, nperseg_int), sl()]
  xdt <- aval_of(xsubs)$dtype

  # The window as an array.
  if (identical(window, "hann")) {
    win <- lax_convert_element_type(sig_hann(nperseg_int), xdt)
  } else if (is.character(window) || is.list(window)) {
    win <- jnp_array(array(sig_get_window(window, nperseg_int), dim = nperseg_int), dtype = xdt)
  } else {
    win <- jnp_asarray(window)
    if (sig_ndim(win) != 1L) jax_error("window must be 1-D", class = "jax_value_error")
    if (sig_shape(win)[1L] != nperseg_int)
      jax_error("window must have length of ", nperseg_int, class = "jax_value_error")
  }
  # The 'spectrum' scaling.
  xsubs <- xsubs * jnp_sum(win)

  # Make the window broadcastable over xsubs.
  nds <- sig_ndim(xsubs)
  win <- lax_expand_dims(win, c(seq_len(nds - 2L) - 1L, -1L))
  x <- sig_overlap_and_add(jnp_swapaxes(xsubs * win, -2L, -1L), nstep)
  # jnp.repeat(win * win, xsubs.shape[-1], axis=-1) of a size one axis is a broadcast.
  win2 <- win * win
  sw <- sig_shape(win2)
  win_squared <- jnp_broadcast_to(win2, c(sw[-length(sw)], sig_last(sig_shape(xsubs))))
  norm <- sig_overlap_and_add(jnp_swapaxes(win_squared, -2L, -1L), nstep)

  # Remove the extension points.
  if (isTRUE(boundary)) {
    h <- nperseg_int %/% 2L
    x <- x[ellipsis, sl(h, -h)]
    norm <- norm[ellipsis, sl(h, -h)]
  }
  x <- x / jnp_where(norm > 1e-10, norm, 1.0)

  # Put the axes back.
  if (sig_ndim(x) > 1L && time_axis != nd - 1L) {
    if (freq_axis < time_axis) time_axis <- time_axis - 1L
    x <- jnp_moveaxis(x, -1L, time_axis)
  }
  dt <- aval_of(x)$dtype
  real_dt <- if (is_complex_dtype(dt)) complex_component(dt) else dt
  time <- jnp_arange(sig_shape(x)[1L], dtype = real_dt) / fs
  list(time, x)
}
