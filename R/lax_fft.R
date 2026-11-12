# FFT primitive (port of jax/_src/lax/fft.py).

#' FFT types
#'
#' Port of `jax.lax.FftType`: `FFT` (forward complex to complex), `IFFT` (inverse
#' complex to complex), `RFFT` (forward real to complex) and `IRFFT` (inverse complex
#' to real). R has no enums, so the values are integer codes; [lax_fft()] also accepts
#' the names as strings.
#' @format A named integer vector.
#' @export
lax_FftType <- c(FFT = 0L, IFFT = 1L, RFFT = 2L, IRFFT = 3L)

# Python tuple repr of an integer vector, for error messages.
fft_tuple_str <- function(x) {
  if (length(x) == 1L) return(paste0("(", x, ",)"))
  paste0("(", paste(x, collapse = ", "), ")")
}

# Port of jax/_src/lax/fft.py:_str_to_fft_type (plus the FftType values).
fft_type_name <- function(fft_type) {
  if (is.character(fft_type) && length(fft_type) == 1L) {
    out <- switch(fft_type, fft = , FFT = "FFT", ifft = , IFFT = "IFFT", rfft = , RFFT = "RFFT",
                  irfft = , IRFFT = "IRFFT", NULL)
    if (is.null(out)) jax_error("Unknown FFT type '", fft_type, "'", class = "jax_value_error")
    return(out)
  }
  if (is.numeric(fft_type) && length(fft_type) == 1L && fft_type %in% lax_FftType)
    return(names(lax_FftType)[match(fft_type, lax_FftType)])
  jax_error("Unknown FFT type value '", paste(format(fft_type), collapse = ", "), "'",
            class = "jax_type_error")
}

fft_p <- new_primitive("fft")

# Port of jax/_src/lax/fft.py:fft_abstract_eval.
fft_p$abstract_eval <- function(avals, params) {
  x <- avals[[1L]]
  typ <- params$fft_type
  ln <- params$fft_lengths
  k <- length(ln)
  if (k > length(x$shape))
    jax_error("FFT input shape ", fft_tuple_str(x$shape), " must have at least as many ",
              "input dimensions as fft_lengths ", fft_tuple_str(ln), ".",
              class = "jax_value_error")
  minor <- utils::tail(x$shape, k)
  if (typ == "RFFT") {
    if (!x$dtype %in% c("float32", "float64"))
      jax_error("RFFT input must be float32 or float64, got ", x$dtype, class = "jax_value_error")
    if (!identical(minor, ln))
      jax_error("RFFT input shape ", fft_tuple_str(x$shape), " minor dimensions must ",
                "be equal to fft_lengths ", fft_tuple_str(ln), class = "jax_value_error")
    shape <- c(utils::head(x$shape, length(x$shape) - k), utils::head(ln, -1L),
               ln[k] %/% 2L + 1L)
    dt <- complex_of(x$dtype)
  } else if (typ == "IRFFT") {
    if (!is_complex_dtype(x$dtype))
      jax_error("IRFFT input must be complex64 or complex128, got ", x$dtype,
                class = "jax_value_error")
    if (!identical(utils::head(minor, -1L), utils::head(ln, -1L)))
      jax_error("IRFFT input shape ", fft_tuple_str(x$shape), " minor dimensions must ",
                "be equal to all except the last fft_length, got fft_lengths=",
                fft_tuple_str(ln), class = "jax_value_error")
    shape <- c(utils::head(x$shape, length(x$shape) - k), ln)
    dt <- complex_component(x$dtype)
  } else {
    if (!is_complex_dtype(x$dtype))
      jax_error("FFT input must be complex64 or complex128, got ", x$dtype,
                class = "jax_value_error")
    if (!identical(minor, ln))
      jax_error("FFT input shape ", fft_tuple_str(x$shape), " minor dimensions must ",
                "be equal to fft_lengths ", fft_tuple_str(ln), class = "jax_value_error")
    shape <- x$shape
    dt <- x$dtype
  }
  new_aval(shape, dt, x$weak)
}

# Port of jax/_src/lax/fft.py:_fft_lowering (the CPU lowering; the cuda/rocm variants
# _fft_lowering_gpu and _fft_lowering_1d_real are not used on CPU).
fft_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  emit_op(ctx, "stablehlo.fft", ins, tensor_type(avals_in[[1L]]), tensor_type(avals_out[[1L]]),
          attrs = list(fft_length = attr_i64_array(params$fft_lengths),
                       fft_type = paste0("#stablehlo<fft_type ", params$fft_type, ">")))
}

# Port of jax/_src/lax/fft.py:_rfft_transpose. The mask is 1 for the DC and Nyquist
# entries and 2 for the others, accounting for the Hermitian redundancy.
fft_rfft_transpose <- function(t, fft_lengths) {
  n <- fft_lengths[length(fft_lengths)]
  is_odd <- n %% 2L
  ts <- aval_of(t)$shape
  m <- ts[length(ts)]
  mask <- lax_full_like(t, 2.0, shape = m)
  mask <- lax_dynamic_update_slice(mask, lax_full_like(t, 1.0, shape = 1L), list(0L))
  if (!is_odd)
    mask <- lax_dynamic_update_slice(mask, lax_full_like(t, 1.0, shape = 1L), list(m - 1L))
  N <- prod(fft_lengths)
  mask <- lax_expand_dims(mask, seq_len(length(ts) - 1L) - 1L)
  jnp_multiply(N, lax_fft(jnp_true_divide(lax_conj(t), mask), "IRFFT", fft_lengths))
}

# Port of jax/_src/lax/fft.py:_irfft_transpose: the RFFT of the cotangent times a
# scaling factor and a mask doubling the de-duplicated Hermitian components.
fft_irfft_transpose <- function(t, fft_lengths) {
  x <- lax_fft(t, "RFFT", fft_lengths)
  xa <- aval_of(x)
  n <- xa$shape[length(xa$shape)]
  is_odd <- fft_lengths[length(fft_lengths)] %% 2L
  mask <- lax_full_like(t, 2.0, shape = n, dtype = xa$dtype)
  mask <- lax_dynamic_update_slice(mask, lax_full_like(t, 1.0, shape = 1L, dtype = xa$dtype),
                                   list(0L))
  if (!is_odd)
    mask <- lax_dynamic_update_slice(mask, lax_full_like(t, 1.0, shape = 1L, dtype = xa$dtype),
                                     list(n - 1L))
  scale <- 1 / prod(fft_lengths)
  out <- jnp_multiply(jnp_multiply(scale, lax_expand_dims(mask, seq_len(length(xa$shape) - 1L) - 1L)), x)
  # JAX's convention for complex gradients (jax-ml/jax#6223).
  lax_conj(out)
}

# Port of jax/_src/lax/fft.py:_fft_transpose_rule (registered with ad.deflinear2).
deflinear(fft_p, function(ct, args, params) {
  if (is_zero(ct)) return(list(zero_of(args[[1L]]$aval)))
  typ <- params$fft_type
  ln <- params$fft_lengths
  res <- if (typ == "RFFT") fft_rfft_transpose(ct, ln)
         else if (typ == "IRFFT") fft_irfft_transpose(ct, ln)
         else lax_fft(ct, typ, ln)
  list(res)
})

# Port of jax/_src/lax/fft.py:_fft_batching_rule.
fft_p$batch <- function(vals, dims, params, axis) {
  x <- moveaxis(vals[[1L]], dims[1L], 0L)
  list(lax_fft(x, params$fft_type, params$fft_lengths), 0L)
}

#' Fast Fourier transform
#'
#' Port of `jax.lax.fft`: an FFT over the trailing `length(fft_lengths)` dimensions of
#' `x`. `RFFT` converts `x` to an inexact real dtype and returns the
#' `fft_lengths[n] %/% 2 + 1` non-negative frequency terms of the last dimension; the
#' other types convert `x` to a complex dtype. `IRFFT` returns a real array whose
#' trailing dimensions are `fft_lengths`.
#' @param x Array.
#' @param fft_type One of `"FFT"`, `"IFFT"`, `"RFFT"`, `"IRFFT"` (or lower case), or a
#'   value of `lax_FftType`.
#' @param fft_lengths Integer vector of transform lengths (at most 3 for XLA).
#' @return An array.
#' @export
lax_fft <- function(x, fft_type, fft_lengths) {
  typ <- fft_type_name(fft_type)
  x <- as_jax(x)
  dt <- aval_of(x)$dtype
  if (typ == "RFFT") {
    if (is_complex_dtype(dt))
      jax_error("only real valued inputs supported for rfft", class = "jax_value_error")
    x <- lax_convert_element_type(x, to_inexact_dtype(dt))
  } else {
    x <- lax_convert_element_type(x, complex_of(to_inexact_dtype(dt)))
  }
  fft_lengths <- as.integer(fft_lengths)
  # XLA FFT doesn't support 0-rank.
  if (!length(fft_lengths)) return(x)
  bind(fft_p, list(x), list(fft_type = typ, fft_lengths = fft_lengths))
}
