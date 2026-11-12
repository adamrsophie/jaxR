# Convolutions (port of jax/_src/lax/convolution.py) and jnp.convolve / jnp.correlate
# (port of jax/_src/numpy/lax_numpy.py:_conv).

# ---- dimension numbers --------------------------------------------------------------

#' Convolution dimension numbers
#'
#' Port of `jax.lax.ConvDimensionNumbers`: 0-based dimension numbers of
#' `(batch, feature, spatial...)` for the input and output and
#' `(out feature, in feature, spatial...)` for the kernel.
#' @param lhs_spec,rhs_spec,out_spec Integer vectors.
#' @return An object of class `ConvDimensionNumbers`.
#' @export
lax_ConvDimensionNumbers <- function(lhs_spec, rhs_spec, out_spec) {
  structure(list(lhs_spec = as.integer(lhs_spec), rhs_spec = as.integer(rhs_spec),
                 out_spec = as.integer(out_spec)), class = "ConvDimensionNumbers")
}

#' @export
format.ConvDimensionNumbers <- function(x, ...) {
  paste0("ConvDimensionNumbers(lhs_spec=", fmt_tuple(x$lhs_spec), ", rhs_spec=",
         fmt_tuple(x$rhs_spec), ", out_spec=", fmt_tuple(x$out_spec), ")")
}

#' @export
print.ConvDimensionNumbers <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' Convolution dimension permutations
#'
#' Port of `jax.lax.conv_general_permutations`: the 0-based permutations (relative to
#' the canonical `NC...`, `OI...`, `NC...` layouts) described by three spec strings.
#' @param dimension_numbers Character vector (or list) of three spec strings such as
#'   `c("NHWC", "HWIO", "NHWC")`.
#' @return A list of three integer vectors.
#' @export
lax_conv_general_permutations <- function(dimension_numbers) {
  specs <- vapply(dimension_numbers, as.character, "")
  charpairs <- list(c("N", "C"), c("O", "I"), c("N", "C"))
  chars <- lapply(specs, function(s) strsplit(s, "")[[1L]])
  for (i in 1:3) {
    a <- charpairs[[i]][1L]
    b <- charpairs[[i]][2L]
    if (!(sum(chars[[i]] == a) == 1L && sum(chars[[i]] == b) == 1L))
      jax_error("convolution dimension_numbers[", i - 1L, "] must contain the characters '", a,
                "' and '", b, "' exactly once, got ", specs[i], ".", class = "jax_type_error")
    if (anyDuplicated(chars[[i]]))
      jax_error("convolution dimension_numbers[", i - 1L, "] cannot have duplicate characters, ",
                "got ", specs[i], ".", class = "jax_type_error")
  }
  spatial <- lapply(1:3, function(i) sort(setdiff(chars[[i]], charpairs[[i]])))
  if (!(identical(spatial[[1L]], spatial[[2L]]) && identical(spatial[[2L]], spatial[[3L]])))
    jax_error("convolution dimension_numbers elements must each have the same set of spatial ",
              "characters, got (", paste0("'", specs, "'", collapse = ", "), ").",
              class = "jax_type_error")
  rhs_chars <- chars[[2L]]
  lapply(1:3, function(i) {
    ch <- chars[[i]]
    sp <- which(!ch %in% charpairs[[i]])
    if (i != 2L) sp <- sp[order(match(ch[sp], rhs_chars))]
    as.integer(c(match(charpairs[[i]][1L], ch), match(charpairs[[i]][2L], ch), sp) - 1L)
  })
}

#' Convert convolution dimension numbers to canonical form
#'
#' Port of `jax.lax.conv_dimension_numbers`.
#' @param lhs_shape,rhs_shape Integer vectors.
#' @param dimension_numbers `NULL`, three spec strings, or a `ConvDimensionNumbers`.
#' @return A `ConvDimensionNumbers` object.
#' @export
lax_conv_dimension_numbers <- function(lhs_shape, rhs_shape, dimension_numbers) {
  if (inherits(dimension_numbers, "ConvDimensionNumbers")) return(dimension_numbers)
  if (length(lhs_shape) != length(rhs_shape))
    jax_error("convolution requires lhs and rhs ndim to be equal, got ", length(lhs_shape),
              " and ", length(rhs_shape), ".", class = "jax_type_error")
  if (is.null(dimension_numbers)) {
    iota <- seq_along(lhs_shape) - 1L
    return(lax_ConvDimensionNumbers(iota, iota, iota))
  }
  if (is.character(dimension_numbers) || is.list(dimension_numbers)) {
    if (length(dimension_numbers) != 3L)
      jax_error("convolution dimension_numbers list/tuple must be length 3, got ",
                length(dimension_numbers), ".", class = "jax_type_error")
    if (!all(vapply(dimension_numbers, function(e) is.character(e) && length(e) == 1L, NA)))
      jax_error("convolution dimension_numbers elements must be strings.",
                class = "jax_type_error")
    for (i in 1:3) {
      if (nchar(dimension_numbers[[i]]) != length(lhs_shape))
        jax_error("convolution dimension_numbers[", i - 1L, "] must have len equal to the ndim ",
                  "of lhs and rhs, got ", nchar(dimension_numbers[[i]]), " for lhs and rhs shapes ",
                  fmt_tuple(lhs_shape), " and ", fmt_tuple(rhs_shape), ".", class = "jax_type_error")
    }
    p <- lax_conv_general_permutations(dimension_numbers)
    return(lax_ConvDimensionNumbers(p[[1L]], p[[2L]], p[[3L]]))
  }
  jax_error("convolution dimension_numbers must be tuple/list or None, got ",
            class(dimension_numbers)[1L], ".", class = "jax_type_error")
}

# ---- shape helpers --------------------------------------------------------------------

#' Convolution output shapes
#'
#' Ports of `jax.lax.conv_shape_tuple` (shapes in canonical order),
#' `jax.lax.conv_general_shape_tuple` and `jax.lax.conv_transpose_shape_tuple`.
#' @param lhs_shape,rhs_shape Integer vectors.
#' @param strides,window_strides Integer vectors.
#' @param pads,padding A padding string or a list of `c(low, high)` pairs.
#' @param batch_group_count Batch group count.
#' @param dimension_numbers Three spec strings or a `ConvDimensionNumbers`.
#' @return An integer vector.
#' @name lax_conv_shapes
NULL

#' @rdname lax_conv_shapes
#' @export
lax_conv_shape_tuple <- function(lhs_shape, rhs_shape, strides, pads, batch_group_count = 1L) {
  lhs_shape <- as.integer(lhs_shape)
  rhs_shape <- as.integer(rhs_shape)
  sp <- drop_first(lhs_shape, 2L)
  if (is.character(pads)) pads <- lax_padtype_to_pads(sp, drop_first(rhs_shape, 2L), strides, pads)
  pads <- normalize_pairs(pads)
  if (length(pads) != length(lhs_shape) - 2L)
    jax_error("Wrong number of explicit pads for convolution: expected ", length(lhs_shape) - 2L,
              ", got ", length(pads), ".", class = "jax_type_error")
  lhs_padded <- sp + vapply(pads, function(p) sum(p), 1L)
  if (any(lhs_padded < 0L))
    jax_error("Negative padding is larger than the size of the corresponding dimension: got ",
              "padding=", fmt_tuple(pads), " for lhs_shape[2:]=", fmt_tuple(sp),
              class = "jax_value_error")
  out_space <- stride_dim(lhs_padded, drop_first(rhs_shape, 2L), strides)
  out0 <- if (batch_group_count > 1L) lhs_shape[1L] %/% batch_group_count else lhs_shape[1L]
  as.integer(c(out0, rhs_shape[1L], out_space))
}

conv_perms <- function(dimension_numbers) {
  if (inherits(dimension_numbers, "ConvDimensionNumbers"))
    return(list(dimension_numbers$lhs_spec, dimension_numbers$rhs_spec, dimension_numbers$out_spec))
  lax_conv_general_permutations(dimension_numbers)
}

#' @rdname lax_conv_shapes
#' @export
lax_conv_general_shape_tuple <- function(lhs_shape, rhs_shape, window_strides, padding,
                                         dimension_numbers) {
  p <- conv_perms(dimension_numbers)
  out_trans <- lax_conv_shape_tuple(as.integer(lhs_shape)[p[[1L]] + 1L],
                                    as.integer(rhs_shape)[p[[2L]] + 1L], window_strides, padding)
  out_trans[order(p[[3L]])]
}

#' @rdname lax_conv_shapes
#' @export
lax_conv_transpose_shape_tuple <- function(lhs_shape, rhs_shape, window_strides, padding,
                                           dimension_numbers) {
  p <- conv_perms(dimension_numbers)
  lhs_trans <- as.integer(lhs_shape)[p[[1L]] + 1L]
  rhs_trans <- as.integer(rhs_shape)[p[[2L]] + 1L]
  ks <- drop_first(rhs_trans, 2L)
  if (is.character(padding)) {
    padding <- Map(function(k, s) conv_transpose_padding(k, s, padding), ks, window_strides)
  }
  padding <- vapply(padding, function(p) sum(p), 1L)
  unpad_out_space <- (drop_first(lhs_trans, 2L) - 1L) * window_strides - ks + 2L
  out_trans <- as.integer(c(lhs_trans[1L], rhs_trans[1L], unpad_out_space + padding))
  out_trans[order(p[[3L]])]
}

# Port of convolution._conv_transpose_padding.
conv_transpose_padding <- function(k, s, padding) {
  if (identical(padding, "SAME")) {
    pad_len <- k + s - 2L
    pad_a <- if (s > k - 1L) k - 1L else as.integer(ceiling(pad_len / 2))
  } else if (identical(padding, "VALID")) {
    pad_len <- k + s - 2L + max(k - s, 0L)
    pad_a <- k - 1L
  } else if (is.numeric(padding) && length(padding) == 2L) {
    pads <- k - as.integer(padding) - 1L
    pad_a <- pads[1L]
    pad_len <- sum(pads)
  } else {
    jax_error("Invalid padding mode: ", paste(format(padding), collapse = " "),
              class = "jax_value_error")
  }
  as.integer(c(pad_a, pad_len - pad_a))
}

# Port of convolution._conv_general_vjp_rhs_padding.
conv_general_vjp_rhs_padding <- function(in_shape, window_dimensions, window_strides,
                                         out_shape, padding, lhs_dilation, rhs_dilation) {
  if (!length(in_shape)) return(list())
  lhs_dilated_shape <- dilate_shape(in_shape, lhs_dilation)
  rhs_dilated_shape <- dilate_shape(window_dimensions, rhs_dilation)
  out_dilated_shape <- dilate_shape(out_shape, window_strides)
  pads_lo <- vapply(padding, `[`, 1L, 1L)
  pads_from_lhs <- out_dilated_shape - lhs_dilated_shape
  pads_from_rhs <- rhs_dilated_shape - pads_lo - 1L
  pads_hi <- pads_from_lhs + pads_from_rhs
  lapply(seq_along(pads_lo), function(i) as.integer(c(pads_lo[i], pads_hi[i])))
}

# Ports of convolution._reshape_axis_into and _reshape_axis_out_of.
reshape_axis_into <- function(src, dst, x) {
  s <- aval_of(x)$shape
  perm <- setdiff(seq_along(s) - 1L, src)
  perm <- append(perm, src, after = dst)
  new_shape <- s[-(src + 1L)]
  new_shape[dst + 1L] <- new_shape[dst + 1L] * s[src + 1L]
  lax_reshape(x, new_shape, perm)
}

reshape_axis_out_of <- function(src, size1, x) {
  s <- aval_of(x)$shape
  size2 <- s[src + 1L] %/% size1
  new_shape <- append(s[-(src + 1L)], c(size1, size2), after = src)
  lax_reshape(x, new_shape)
}

conv_spec_transpose <- function(spec) c(spec[2L], spec[1L], drop_first(spec, 2L))
conv_sdims <- function(spec) drop_first(spec, 2L)

# Port of lax.canonicalize_precision for Precision values given by name (lax._precision_strings).
canonicalize_precision <- function(precision) {
  if (is.null(precision)) {
    precision <- .jx$config$jax_default_matmul_precision
    if (is.null(precision)) return(NULL)
  }
  names_map <- c(highest = "HIGHEST", float32 = "HIGHEST", high = "HIGH", bfloat16_3x = "HIGH",
                 tensorfloat32 = "HIGH", default = "DEFAULT", bfloat16 = "DEFAULT",
                 fastest = "DEFAULT")
  p <- tolower(as.character(unlist(precision)))
  if (length(p) == 1L) p <- c(p, p)
  if (length(p) != 2L || anyNA(names_map[p]))
    jax_error("Precision argument must be one of: None, a precision name (",
              paste(names(names_map), collapse = ", "), ") or a pair of them, got ",
              paste(p, collapse = ", "), class = "jax_value_error")
  unname(names_map[p])
}

# Port of lax._validate_preferred_element_type.
validate_preferred_element_type <- function(input_dtype, pet) {
  ki <- dtype_kind(input_dtype)
  kp <- dtype_kind(pet)
  if (!(ki %in% c("i", "u") && kp == "f")) {
    grp <- function(k) if (k %in% c("i", "u")) "int" else k
    if (ki %in% c("i", "u", "f", "c") && grp(ki) != grp(kp))
      jax_error("Input type is incompatible with `preferred_element_type`. The compatible ",
                "combinations of (input_type, preferred_element_type) are (integral, integral), ",
                "(integral, floating), (floating, floating), (complex, complex.",
                class = "jax_type_error")
    if (ki == "i" && kp != "i")
      jax_error("`preferred_element_type` must have the same signedness as the original type.",
                class = "jax_type_error")
  }
  if (dtype_bits(pet) < dtype_bits(input_dtype))
    jax_error("`preferred_element_type` must not be narrower than the original type.",
              class = "jax_type_error")
}

# ---- conv_general_dilated -------------------------------------------------------------

conv_general_dilated_p <- new_primitive("conv_general_dilated")

# Port of convolution._conv_general_dilated_shape_rule.
conv_general_dilated_shape_rule <- function(lhs, rhs, params) {
  dn <- params$dimension_numbers
  fgc <- params$feature_group_count
  bgc <- params$batch_group_count
  if (length(lhs$shape) != length(rhs$shape))
    jax_error("conv_general_dilated lhs and rhs must have the same number of dimensions, but ",
              "got ", fmt_tuple(lhs$shape), " and ", fmt_tuple(rhs$shape), ".",
              class = "jax_value_error")
  if (!(fgc > 0L))
    jax_error("conv_general_dilated feature_group_count must be a positive integer, got ", fgc,
              ".", class = "jax_value_error")
  lhs_feature_count <- lhs$shape[dn$lhs_spec[2L] + 1L]
  if (lhs_feature_count %% fgc)
    jax_error("conv_general_dilated feature_group_count must divide lhs feature dimension ",
              "size, but ", fgc, " does not divide ", lhs_feature_count, ".",
              class = "jax_value_error")
  if (lhs_feature_count %/% fgc != rhs$shape[dn$rhs_spec[2L] + 1L])
    jax_error("conv_general_dilated lhs feature dimension size divided by feature_group_count ",
              "must equal the rhs input feature dimension size, but ", lhs_feature_count, " // ",
              fgc, " != ", rhs$shape[dn$rhs_spec[2L] + 1L], ".", class = "jax_value_error")
  if (rhs$shape[dn$rhs_spec[1L] + 1L] %% fgc)
    jax_error("conv_general_dilated rhs output feature dimension size must be a multiple of ",
              "feature_group_count, but ", rhs$shape[dn$rhs_spec[1L] + 1L], " is not a multiple ",
              "of ", fgc, ".", class = "jax_value_error")
  if (!(bgc > 0L))
    jax_error("conv_general_dilated batch_group_count must be a positive integer, got ", bgc,
              ".", class = "jax_value_error")
  lhs_batch_count <- lhs$shape[dn$lhs_spec[1L] + 1L]
  if (bgc > 1L && lhs_batch_count %% bgc != 0L)
    jax_error("conv_general_dilated batch_group_count must divide lhs batch dimension size, ",
              "but ", bgc, " does not divide ", lhs_batch_count, ".", class = "jax_value_error")
  if (rhs$shape[dn$rhs_spec[1L] + 1L] %% bgc)
    jax_error("conv_general_dilated rhs output feature dimension size must be a multiple of ",
              "batch_group_count, but ", rhs$shape[dn$rhs_spec[1L] + 1L], " is not a multiple ",
              "of ", bgc, ".", class = "jax_value_error")
  if (bgc > 1L && fgc > 1L)
    jax_error("At most one of batch_group_count and feature_group_count may be > 1, got ",
              "batch_group_count=", bgc, " and feature_group_count=", fgc,
              class = "jax_value_error")
  if (length(conv_sdims(dn$rhs_spec)) != length(params$window_strides))
    jax_error("conv_general_dilated window and window_strides must have the same number of ",
              "dimensions, but got ", length(conv_sdims(dn$rhs_spec)), " and ",
              length(params$window_strides), class = "jax_value_error")
  lhs_trans <- dilate_shape(lhs$shape[dn$lhs_spec + 1L], params$lhs_dilation)
  rhs_trans <- dilate_shape(rhs$shape[dn$rhs_spec + 1L], params$rhs_dilation)
  out_trans <- lax_conv_shape_tuple(lhs_trans, rhs_trans, params$window_strides, params$padding,
                                    bgc)
  out_trans[order(dn$out_spec)]
}

conv_general_dilated_p$abstract_eval <- function(avals, params) {
  lhs <- avals[[1L]]
  rhs <- avals[[2L]]
  shape <- conv_general_dilated_shape_rule(lhs, rhs, params)
  # Port of convolution._conv_general_dilated_dtype_rule.
  if (lhs$dtype != rhs$dtype)
    jax_error("lax.conv_general_dilated requires arguments to have the same dtypes, got ",
              lhs$dtype, ", ", rhs$dtype, ".", class = "jax_type_error")
  dt <- lhs$dtype
  pet <- params$preferred_element_type
  if (!is.null(pet)) {
    validate_preferred_element_type(dt, pet)
    dt <- pet
  }
  new_aval(shape, dt, lhs$weak && rhs$weak)
}

conv_dn_attr <- function(dn) {
  lab <- function(spec, a, b) {
    out <- character(length(spec))
    out[spec[1L] + 1L] <- a
    out[spec[2L] + 1L] <- b
    sp <- drop_first(spec, 2L)
    out[sp + 1L] <- as.character(seq_along(sp) - 1L)
    paste0("[", paste(out, collapse = ", "), "]")
  }
  paste0("#stablehlo.conv<", lab(dn$lhs_spec, "b", "f"), "x", lab(dn$rhs_spec, "o", "i"), "->",
         lab(dn$out_spec, "b", "f"), ">")
}

# Port of convolution._complex_mul (Gauss's three multiplication trick).
conv_complex_mul <- function(mul, x, y) {
  x_re <- lax_real(x)
  x_im <- lax_imag(x)
  y_re <- lax_real(y)
  y_im <- lax_imag(y)
  k1 <- mul(lax_add(x_re, x_im), y_re)
  k2 <- mul(x_re, lax_sub(y_im, y_re))
  k3 <- mul(x_im, lax_add(y_re, y_im))
  lax_complex(lax_sub(k1, k3), lax_add(k1, k2))
}

# Port of convolution._conv_general_dilated_lower with expand_complex_convolutions = True
# (the CPU registration).
conv_general_dilated_p$lowering <- function(ctx, ins, ai, ao, params) {
  if (is_complex_dtype(ai[[1L]]$dtype)) {
    p2 <- params
    if (!is.null(p2$preferred_element_type))
      p2$preferred_element_type <- complex_component(p2$preferred_element_type)
    rule <- lower_fun(function(args, p) {
      conv_complex_mul(function(a, b) bind(conv_general_dilated_p, list(a, b), p2),
                       args[[1L]], args[[2L]])
    })
    return(rule(ctx, ins, ai, ao, params))
  }
  dn <- params$dimension_numbers
  nsp <- length(dn$rhs_spec) - 2L
  prec <- params$precision %||% c("DEFAULT", "DEFAULT")
  emit_op(ctx, "stablehlo.convolution", ins, types_of(ai), tensor_type(ao[[1L]]),
          attrs = list(batch_group_count = attr_i64(params$batch_group_count),
                       dimension_numbers = conv_dn_attr(dn),
                       feature_group_count = attr_i64(params$feature_group_count),
                       lhs_dilation = attr_i64_array(params$lhs_dilation),
                       padding = attr_i64_pairs(params$padding),
                       precision_config = paste0("[", paste0("#stablehlo<precision ", prec, ">",
                                                             collapse = ", "), "]"),
                       rhs_dilation = attr_i64_array(params$rhs_dilation),
                       window_reversal = attr_bool_array(rep(FALSE, nsp)),
                       window_strides = attr_i64_array(params$window_strides)))
}

# Bind with some params replaced (not modifyList, which would merge the padding lists).
conv_bind <- function(lhs, rhs, params, ...) {
  upd <- list(...)
  params[names(upd)] <- upd
  bind(conv_general_dilated_p, list(lhs, rhs), params)
}

# Port of convolution._conv_general_dilated_transpose_lhs.
conv_general_dilated_transpose_lhs <- function(g, lhs_aval, rhs, params) {
  dn <- params$dimension_numbers
  fgc <- params$feature_group_count
  bgc <- params$batch_group_count
  rhs_shape <- aval_of(rhs)$shape
  lhs_shape <- lhs_aval$shape
  lhs_spec <- dn$lhs_spec
  rhs_spec <- dn$rhs_spec
  out_spec <- dn$out_spec
  t_rhs_spec <- conv_spec_transpose(rhs_spec)
  if (fgc > 1L) {
    rhs <- reshape_axis_out_of(rhs_spec[1L], fgc, rhs)
    rhs <- reshape_axis_into(rhs_spec[1L], rhs_spec[2L], rhs)
  } else if (bgc > 1L) {
    rhs <- reshape_axis_out_of(rhs_spec[1L], bgc, rhs)
    rhs <- reshape_axis_into(rhs_spec[1L], rhs_spec[2L], rhs)
    fgc <- bgc
  }
  trans_dn <- lax_ConvDimensionNumbers(out_spec, t_rhs_spec, lhs_spec)
  padding <- conv_general_vjp_lhs_padding(lhs_shape[conv_sdims(lhs_spec) + 1L],
                                          rhs_shape[conv_sdims(rhs_spec) + 1L],
                                          params$window_strides,
                                          aval_of(g)$shape[conv_sdims(out_spec) + 1L],
                                          params$padding, params$lhs_dilation, params$rhs_dilation)
  revd_weights <- lax_rev(rhs, conv_sdims(rhs_spec))
  if (aval_of(g)$dtype != aval_of(revd_weights)$dtype)
    revd_weights <- lax_convert_element_type(revd_weights, aval_of(g)$dtype)
  out <- conv_bind(g, revd_weights, params, window_strides = params$lhs_dilation,
                   padding = padding, lhs_dilation = params$window_strides,
                   rhs_dilation = params$rhs_dilation, dimension_numbers = trans_dn,
                   feature_group_count = fgc, batch_group_count = 1L)
  if (aval_of(out)$dtype != lhs_aval$dtype) out <- lax_convert_element_type(out, lhs_aval$dtype)
  if (bgc > 1L) {
    out <- reshape_axis_out_of(lhs_spec[2L], bgc, out)
    out <- reshape_axis_into(lhs_spec[2L], lhs_spec[1L], out)
  }
  out
}

# Port of convolution._conv_general_dilated_transpose_rhs.
conv_general_dilated_transpose_rhs <- function(g, lhs, rhs_aval, params) {
  dn <- params$dimension_numbers
  if (prod_int(aval_of(g)$shape) == 0L) return(zero_of(rhs_aval))
  lhs_shape <- aval_of(lhs)$shape
  rhs_shape <- rhs_aval$shape
  fgc <- params$feature_group_count
  bgc <- params$batch_group_count
  if (bgc > 1L) {
    fgc <- bgc
    bgc <- 1L
  } else if (fgc > 1L) {
    bgc <- fgc
    fgc <- 1L
  }
  trans_dn <- lax_ConvDimensionNumbers(conv_spec_transpose(dn$lhs_spec),
                                       conv_spec_transpose(dn$out_spec),
                                       conv_spec_transpose(dn$rhs_spec))
  padding <- conv_general_vjp_rhs_padding(lhs_shape[conv_sdims(dn$lhs_spec) + 1L],
                                          rhs_shape[conv_sdims(dn$rhs_spec) + 1L],
                                          params$window_strides,
                                          aval_of(g)$shape[conv_sdims(dn$out_spec) + 1L],
                                          params$padding, params$lhs_dilation, params$rhs_dilation)
  if (aval_of(g)$dtype != aval_of(lhs)$dtype) lhs <- lax_convert_element_type(lhs, aval_of(g)$dtype)
  out <- conv_bind(lhs, g, params, window_strides = params$rhs_dilation, padding = padding,
                   lhs_dilation = params$lhs_dilation, rhs_dilation = params$window_strides,
                   dimension_numbers = trans_dn, feature_group_count = fgc,
                   batch_group_count = bgc)
  if (aval_of(out)$dtype != rhs_aval$dtype) out <- lax_convert_element_type(out, rhs_aval$dtype)
  out
}

# ad.defbilinear(conv_general_dilated_p, transpose_lhs, transpose_rhs).
conv_general_dilated_p$jvp <- function(primals, tangents, params) {
  ans <- bind(conv_general_dilated_p, primals, params)
  t1 <- tangents[[1L]]
  t2 <- tangents[[2L]]
  out <- add_tangents(
    if (is_zero(t1)) t1 else bind(conv_general_dilated_p, list(t1, primals[[2L]]), params),
    if (is_zero(t2)) t2 else bind(conv_general_dilated_p, list(primals[[1L]], t2), params))
  if (is_zero(out)) out <- zero_of(tangent_aval(aval_of(ans)))
  list(ans, out)
}

conv_general_dilated_p$transpose <- function(ct, args, params) {
  lhs <- args[[1L]]
  rhs <- args[[2L]]
  if (is_undefined(lhs) && is_undefined(rhs))
    jax_error("Transpose rule for conv_general_dilated: both arguments are linear")
  if (is_undefined(lhs)) {
    if (is_zero(ct)) return(list(zero_of(lhs$aval), NULL))
    return(list(conv_general_dilated_transpose_lhs(ct, lhs$aval, rhs, params), NULL))
  }
  if (is_zero(ct)) return(list(NULL, zero_of(rhs$aval)))
  list(NULL, conv_general_dilated_transpose_rhs(ct, lhs, rhs$aval, params))
}

# Port of convolution._conv_general_dilated_batch_rule.
conv_general_dilated_p$batch <- function(vals, dims, params, axis) {
  lhs <- vals[[1L]]
  rhs <- vals[[2L]]
  lhs_bdim <- dims[1L]
  rhs_bdim <- dims[2L]
  dn <- params$dimension_numbers
  lhs_spec <- dn$lhs_spec
  rhs_spec <- dn$rhs_spec
  out_spec <- dn$out_spec
  fgc <- params$feature_group_count
  bgc <- params$batch_group_count
  ls <- aval_of(lhs)$shape
  rs <- aval_of(rhs)$shape
  if ((!is.na(lhs_bdim) && ls[lhs_bdim + 1L] == 0L) || (!is.na(rhs_bdim) && rs[rhs_bdim + 1L] == 0L)) {
    lu <- if (is.na(lhs_bdim)) ls else ls[-(lhs_bdim + 1L)]
    ru <- if (is.na(rhs_bdim)) rs else rs[-(rhs_bdim + 1L)]
    shape <- conv_general_dilated_shape_rule(new_aval(lu, aval_of(lhs)$dtype),
                                             new_aval(ru, aval_of(rhs)$dtype), params)
    dt <- params$preferred_element_type %||% aval_of(lhs)$dtype
    return(list(lax_full(c(0L, shape), 0, dt), 0L))
  }
  if (!is.na(lhs_bdim) && !is.na(rhs_bdim)) {
    n <- ls[lhs_bdim + 1L]
    if (bgc > 1L) {
      new_lhs <- reshape_axis_into(lhs_bdim, lhs_spec[1L], lhs)
      bgc <- bgc * n
    } else {
      new_lhs <- reshape_axis_into(lhs_bdim, lhs_spec[2L], lhs)
      fgc <- fgc * n
    }
    new_rhs <- reshape_axis_into(rhs_bdim, rhs_spec[1L], rhs)
    out <- conv_bind(new_lhs, new_rhs, params, feature_group_count = fgc, batch_group_count = bgc)
    return(list(reshape_axis_out_of(out_spec[2L], n, out), out_spec[2L]))
  }
  if (!is.na(lhs_bdim)) {
    n <- ls[lhs_bdim + 1L]
    if (bgc == 1L) {
      new_lhs <- reshape_axis_into(lhs_bdim, lhs_spec[1L], lhs)
      out <- conv_bind(new_lhs, rhs, params)
    } else {
      new_lhs <- reshape_axis_out_of(lhs_spec[1L] + as.integer(lhs_bdim <= lhs_spec[1L]), bgc, lhs)
      new_lhs <- reshape_axis_into(lhs_bdim + as.integer(lhs_spec[1L] < lhs_bdim),
                                   lhs_spec[1L] + 1L, new_lhs)
      new_lhs <- reshape_axis_into(lhs_spec[1L], lhs_spec[1L], new_lhs)
      out <- conv_bind(new_lhs, rhs, params)
    }
    return(list(reshape_axis_out_of(out_spec[1L], n, out), out_spec[1L]))
  }
  n <- rs[rhs_bdim + 1L]
  if (fgc == 1L && bgc == 1L) {
    new_rhs <- reshape_axis_into(rhs_bdim, rhs_spec[1L], rhs)
    out <- conv_bind(lhs, new_rhs, params)
    return(list(reshape_axis_out_of(out_spec[2L], n, out), out_spec[2L]))
  }
  group_count <- if (fgc > 1L) fgc else bgc
  new_rhs <- reshape_axis_out_of(rhs_spec[1L] + as.integer(rhs_bdim <= rhs_spec[1L]),
                                 group_count, rhs)
  new_rhs <- reshape_axis_into(rhs_bdim + as.integer(rhs_spec[1L] < rhs_bdim),
                               rhs_spec[1L] + 1L, new_rhs)
  new_rhs <- reshape_axis_into(rhs_spec[1L], rhs_spec[1L], new_rhs)
  out <- conv_bind(lhs, new_rhs, params)
  out <- reshape_axis_out_of(out_spec[2L], group_count, out)
  out <- reshape_axis_out_of(out_spec[2L] + 1L, n, out)
  out <- reshape_axis_into(out_spec[2L], out_spec[2L] + 1L, out)
  list(out, out_spec[2L])
}

#' General n dimensional convolution
#'
#' Port of `jax.lax.conv_general_dilated` (XLA Conv). Arrays have rank `n + 2`.
#' @param lhs Input array.
#' @param rhs Kernel array.
#' @param window_strides Integer vector of `n` strides.
#' @param padding `"SAME"`, `"SAME_LOWER"`, `"VALID"` or a list of `n` `c(low, high)`
#'   pairs.
#' @param lhs_dilation,rhs_dilation Optional integer vectors of `n` dilation factors.
#' @param dimension_numbers `NULL` (`NCHW`, `OIHW`, `NCHW` layouts), three spec strings
#'   such as `c("NHWC", "HWIO", "NHWC")`, or a `ConvDimensionNumbers`.
#' @param feature_group_count,batch_group_count Group counts.
#' @param precision Optional precision (`"default"`, `"high"`, `"highest"`, or a pair).
#' @param preferred_element_type Optional accumulation and result dtype.
#' @return The convolution.
#' @export
lax_conv_general_dilated <- function(lhs, rhs, window_strides, padding, lhs_dilation = NULL,
                                     rhs_dilation = NULL, dimension_numbers = NULL,
                                     feature_group_count = 1L, batch_group_count = 1L,
                                     precision = NULL, preferred_element_type = NULL) {
  lhs <- as_jax(lhs)
  rhs <- as_jax(rhs)
  ls <- aval_of(lhs)$shape
  rs <- aval_of(rhs)$shape
  dnums <- lax_conv_dimension_numbers(ls, rs, dimension_numbers)
  if (is.null(lhs_dilation)) {
    lhs_dilation <- rep(1L, length(ls) - 2L)
  } else if (is.character(padding) && !all(lhs_dilation == 1L)) {
    jax_error("String padding is not implemented for transposed convolution using this op. ",
              "Please either exactly specify the required padding or use conv_transpose.",
              class = "jax_value_error")
  }
  if (is.null(rhs_dilation)) rhs_dilation <- rep(1L, length(rs) - 2L)
  if (is.character(padding)) {
    rhs_shape <- drop_first(rs[dnums$rhs_spec + 1L], 2L)
    effective_rhs_shape <- dilate_shape(rhs_shape, rhs_dilation)
    padding <- lax_padtype_to_pads(drop_first(ls[dnums$lhs_spec + 1L], 2L), effective_rhs_shape,
                                   window_strides, padding)
  } else {
    padding <- normalize_pairs(padding, "padding argument to conv_general_dilated")
  }
  pet <- if (is.null(preferred_element_type)) NULL else dtype(preferred_element_type)
  bind(conv_general_dilated_p, list(lhs, rhs),
       list(window_strides = as.integer(window_strides), padding = padding,
            lhs_dilation = as.integer(lhs_dilation), rhs_dilation = as.integer(rhs_dilation),
            dimension_numbers = dnums, feature_group_count = as.integer(feature_group_count),
            batch_group_count = as.integer(batch_group_count),
            precision = canonicalize_precision(precision), preferred_element_type = pet))
}

#' Convolution wrappers
#'
#' Ports of `jax.lax.conv`, `jax.lax.conv_with_general_padding` and
#' `jax.lax.conv_transpose`, convenience wrappers around [lax_conv_general_dilated()].
#' @inheritParams lax_conv_general_dilated
#' @param strides Integer vector of `n` (fractional) strides of the transposed
#'   convolution.
#' @param transpose_kernel Flip the spatial axes and swap the feature axes of the kernel
#'   (gradient of a forward convolution, as in Keras `Conv2DTranspose`).
#' @param use_consistent_padding Interpret explicit padding as the padding of the
#'   corresponding forward convolution.
#' @return The convolution.
#' @name lax_conv_wrappers
NULL

#' @rdname lax_conv_wrappers
#' @export
lax_conv <- function(lhs, rhs, window_strides, padding, precision = NULL,
                     preferred_element_type = NULL) {
  lax_conv_general_dilated(lhs, rhs, window_strides, padding, precision = precision,
                           preferred_element_type = preferred_element_type)
}

#' @rdname lax_conv_wrappers
#' @export
lax_conv_with_general_padding <- function(lhs, rhs, window_strides, padding, lhs_dilation,
                                          rhs_dilation, precision = NULL,
                                          preferred_element_type = NULL) {
  lax_conv_general_dilated(lhs, rhs, window_strides, padding, lhs_dilation = lhs_dilation,
                           rhs_dilation = rhs_dilation, precision = precision,
                           preferred_element_type = preferred_element_type)
}

#' @rdname lax_conv_wrappers
#' @export
lax_conv_transpose <- function(lhs, rhs, strides, padding, rhs_dilation = NULL,
                               dimension_numbers = NULL, transpose_kernel = FALSE,
                               precision = NULL, preferred_element_type = NULL,
                               use_consistent_padding = FALSE) {
  lhs <- as_jax(lhs)
  rhs <- as_jax(rhs)
  ls <- aval_of(lhs)$shape
  rs <- aval_of(rhs)$shape
  ndims <- length(ls)
  if (!(length(ls) == length(rs) && ndims >= 2L))
    jax_error("conv_transpose requires lhs and rhs of equal rank of at least 2",
              class = "jax_value_error")
  one <- rep(1L, ndims - 2L)
  if (is.null(dimension_numbers)) {
    dimension_numbers <- switch(as.character(ndims),
      "2" = c("NC", "IO", "NC"), "3" = c("NHC", "HIO", "NHC"),
      "4" = c("NHWC", "HWIO", "NHWC"), "5" = c("NHWDC", "HWDIO", "NHWDC"),
      jax_error("No 4+ dimensional dimension_number defaults.", class = "jax_value_error"))
  }
  dn <- lax_conv_dimension_numbers(ls, rs, dimension_numbers)
  k_sdims <- drop_first(rs[dn$rhs_spec + 1L], 2L)
  if (is.null(rhs_dilation)) rhs_dilation <- rep(1L, length(rs) - 2L)
  if (use_consistent_padding || (is.character(padding) && padding %in% c("SAME", "VALID"))) {
    effective_k_size <- dilate_shape(k_sdims, rhs_dilation)
    replicated <- if (is.character(padding)) rep(list(padding), length(strides))
                  else normalize_pairs(padding)
    pads <- Map(conv_transpose_padding, effective_k_size, as.integer(strides), replicated)
  } else {
    pads <- padding
  }
  if (transpose_kernel) {
    rhs <- lax_rev(rhs, drop_first(dn$rhs_spec, 2L))
    rhs <- jnp_swapaxes(rhs, dn$rhs_spec[1L], dn$rhs_spec[2L])
  }
  lax_conv_general_dilated(lhs, rhs, one, pads, strides, rhs_dilation, dn,
                           precision = precision, preferred_element_type = preferred_element_type)
}

# ---- jnp.convolve / jnp.correlate -------------------------------------------------------

# Port of lax_numpy._conv.
jnp_conv_impl <- function(x, y, mode, op, precision, preferred_element_type) {
  x <- as_jax(x)
  y <- as_jax(y)
  if (length(aval_of(x)$shape) != 1L || length(aval_of(y)$shape) != 1L)
    jax_error(op, "() only support 1-dimensional inputs.", class = "jax_value_error")
  xy <- if (is.null(preferred_element_type)) promote_dtypes_inexact(list(x, y))
        else promote_dtypes(list(x, y))
  x <- xy[[1L]]
  y <- xy[[2L]]
  nx <- aval_of(x)$shape
  ny <- aval_of(y)$shape
  if (nx == 0L || ny == 0L)
    jax_error(op, ": inputs cannot be empty, got shapes ", fmt_tuple(nx), " and ", fmt_tuple(ny),
              ".", class = "jax_value_error")
  reverse_out <- FALSE
  if (op == "correlate") {
    if (is_complex_dtype(aval_of(y)$dtype)) y <- lax_conj(y)
    if (nx < ny) {
      tmp <- x; x <- y; y <- tmp
      reverse_out <- TRUE
    }
  } else {
    if (nx < ny) {
      tmp <- x; x <- y; y <- tmp
    }
    y <- lax_rev(y, 0L)
  }
  n <- aval_of(y)$shape[1L]
  padding <- switch(mode,
    valid = list(c(0L, 0L)),
    same = list(c(n %/% 2L, n - n %/% 2L - 1L)),
    full = list(c(n - 1L, n - 1L)),
    jax_error("mode must be one of ['full', 'same', 'valid']", class = "jax_value_error"))
  result <- lax_conv_general_dilated(lax_reshape(x, c(1L, 1L, aval_of(x)$shape)),
                                     lax_reshape(y, c(1L, 1L, n)), 1L, padding,
                                     precision = precision,
                                     preferred_element_type = preferred_element_type)
  out <- lax_reshape(result, aval_of(result)$shape[3L])
  if (reverse_out) lax_rev(out, 0L) else out
}

#' Convolution and correlation of one dimensional arrays
#'
#' Ports of `jnp.convolve` and `jnp.correlate`.
#' @param a,v One dimensional arrays.
#' @param mode `"full"`, `"same"` or `"valid"`.
#' @param precision Optional precision.
#' @param preferred_element_type Optional result dtype (inputs are then promoted without
#'   conversion to an inexact dtype).
#' @return A one dimensional array.
#' @export
jnp_convolve <- function(a, v, mode = "full", precision = NULL, preferred_element_type = NULL) {
  jnp_conv_impl(a, v, mode, "convolve", precision, preferred_element_type)
}

#' @rdname jnp_convolve
#' @export
jnp_correlate <- function(a, v, mode = "valid", precision = NULL, preferred_element_type = NULL) {
  jnp_conv_impl(a, v, mode, "correlate", precision, preferred_element_type)
}
