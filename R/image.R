# Image resampling (port of jax/_src/image/scale.py): jax.image.resize,
# jax.image.scale_and_translate and jax.image.ResizeMethod.
#
# As in JAX, every resampling method except nearest neighbor builds one weight matrix
# per resized dimension and contracts it with the image (JAX uses one einsum; here each
# dimension is one dot_general). Gradients and vmap follow from those primitives.

#' Image resize methods
#'
#' Port of `jax.image.ResizeMethod`: the method names accepted by [image_resize()] and
#' [image_scale_and_translate()]. Strings work too: `"nearest"`, `"linear"`
#' (`"bilinear"`, `"trilinear"`, `"triangle"`), `"lanczos3"`, `"lanczos5"`, `"cubic"`
#' (`"bicubic"`, `"tricubic"`), `"cubic-pytorch"` (`"bicubic-pytorch"`) and `"area"`.
#' @export
image_ResizeMethod <- list(NEAREST = "nearest", LINEAR = "linear", LANCZOS3 = "lanczos3",
                           LANCZOS5 = "lanczos5", CUBIC = "cubic", CUBIC_PYTORCH = "cubic-pytorch",
                           AREA = "area")

# Port of ResizeMethod.from_string.
img_method <- function(s) {
  if (!is.character(s) || length(s) != 1L)
    jax_error("method must be a string or a member of image_ResizeMethod", class = "jax_type_error")
  switch(s,
    nearest = "nearest",
    linear = , bilinear = , trilinear = , triangle = "linear",
    lanczos3 = "lanczos3",
    lanczos5 = "lanczos5",
    cubic = , bicubic = , tricubic = "cubic",
    "cubic-pytorch" = , "bicubic-pytorch" = "cubic-pytorch",
    area = "area",
    jax_error("Unknown resize method \"", s, "\"", class = "jax_value_error"))
}

# ---- kernels (scale.py:_fill_*_kernel) ------------------------------------------------------

img_lanczos_kernel <- function(radius, x) {
  y <- radius * jnp_sin(pi * x) * jnp_sin(pi * x / radius)
  out <- jnp_where(x > 1e-3, jnp_divide(y, jnp_where(x != 0, pi^2 * x^2, 1)), 1)
  jnp_where(x > radius, 0, out)
}

img_keys_cubic_kernel <- function(x) {
  out <- ((1.5 * x - 2.5) * x) * x + 1
  out <- jnp_where(x >= 1, ((-0.5 * x + 2.5) * x - 4) * x + 2, out)
  jnp_where(x >= 2, 0, out)
}

img_opencv_cubic_kernel <- function(x) {
  a <- -0.75
  out <- ((a + 2) * x - (a + 3)) * x * x + 1
  out <- jnp_where(x >= 1, ((a * x - 5 * a) * x + 8 * a) * x - 4 * a, out)
  jnp_where(x >= 2, 0, out)
}

img_triangle_kernel <- function(x) jnp_maximum(0, 1 - jnp_abs(x))

img_kernels <- list(
  linear = list(radius = 1L, kernel = img_triangle_kernel),
  lanczos3 = list(radius = 3L, kernel = function(x) img_lanczos_kernel(3, x)),
  lanczos5 = list(radius = 5L, kernel = function(x) img_lanczos_kernel(5, x)),
  cubic = list(radius = 2L, kernel = img_keys_cubic_kernel),
  "cubic-pytorch" = list(radius = 2L, kernel = img_opencv_cubic_kernel),
  area = list(radius = 1L, kernel = "area"))

# Elementwise helpers that keep R scalars as R scalars (Python floats in JAX).
img_max1 <- function(v) if (is_jax_value(v)) jnp_maximum(v, 1) else max(v, 1)

# Port of scale.py:compute_weight_mat. Returns the (input_size, output_size) weights.
img_weight_mat <- function(input_size, output_size, scale, translation, kernel, antialias,
                           edge_padding, radius) {
  dt <- dtypes_result_type(scale, translation)
  inv_scale <- 1 / scale
  kernel_scale <- if (antialias) img_max1(inv_scale) else 1
  sample_f <- (jnp_arange(output_size, dtype = dt) + 0.5) * inv_scale - translation * inv_scale - 0.5
  k <- if (edge_padding) radius else 0L
  if (identical(kernel, "area")) {
    L_i <- jnp_arange(output_size, dtype = dt) * inv_scale - translation * inv_scale
    R_i <- L_i + inv_scale
    L_j <- jnp_arange(input_size, dtype = dt)
    R_j <- L_j + 1
    weights <- jnp_maximum(0, jnp_minimum(jnp_expand_dims(R_i, 0L), jnp_expand_dims(R_j, 1L)) -
                              jnp_maximum(jnp_expand_dims(L_i, 0L), jnp_expand_dims(L_j, 1L)))
  } else {
    expanded <- jnp_arange(-k, input_size + k, dtype = dt)
    x <- jnp_abs(jnp_expand_dims(sample_f, 0L) - jnp_expand_dims(expanded, 1L))
    weights <- kernel(x / kernel_scale)
  }
  eps_thresh <- 1000 * 2^-23
  normalize <- function(w) {
    total <- jnp_sum(w, axis = 0L, keepdims = TRUE)
    jnp_where(jnp_abs(total) > eps_thresh, jnp_divide(w, jnp_where(total != 0, total, 1)), 0)
  }
  if (edge_padding) {
    # The scatter-add of JAX onto the clamped indices: the k rows below the image add to
    # row 0, the k rows above it to the last row.
    weights <- img_fold_edges(weights, input_size, k)
    return(normalize(weights))
  }
  weights <- normalize(weights)
  inside <- jnp_logical_and(sample_f >= -0.5, sample_f <= input_size - 0.5)
  jnp_where(jnp_expand_dims(inside, 0L), weights, 0)
}

img_fold_edges <- function(w, m, k) {
  if (k == 0L) return(w)
  lo <- jnp_sum(lax_slice_in_dim(w, 0L, k + 1L, axis = 0L), axis = 0L, keepdims = TRUE)
  hi <- jnp_sum(lax_slice_in_dim(w, m + k - 1L, m + 2L * k, axis = 0L), axis = 0L, keepdims = TRUE)
  if (m == 1L) return(jnp_sum(w, axis = 0L, keepdims = TRUE))
  mid <- lax_slice_in_dim(w, k + 1L, m + k - 1L, axis = 0L)
  lax_concatenate(list(lo, mid, hi), 0L)
}

# Port of scale.py:_scale_and_translate (one dot_general per spatial dimension).
img_scale_and_translate <- function(x, output_shape, spatial_dims, scale, translation, kernel,
                                    antialias, precision, edge_padding = FALSE, radius = NULL) {
  in_shape <- aval_of(x)$shape
  r <- length(in_shape)
  for (i in seq_along(spatial_dims)) {
    d <- norm_axis(spatial_dims[i], r)
    w <- img_weight_mat(in_shape[d + 1L], output_shape[d + 1L], scale[[i]], translation[[i]],
                        kernel, antialias, edge_padding, radius)
    w <- lax_convert_element_type(w, aval_of(x)$dtype)
    y <- lax_dot_general(x, w, list(list(d, 0L), list(integer(0), integer(0))),
                         precision = precision)
    x <- moveaxis(y, r - 1L, d)
  }
  x
}

img_shape_str <- function(s) cb_py_tuple(s)

img_check_shape <- function(shape, image) {
  if (length(shape) != length(aval_of(image)$shape))
    jax_error("shape must have length equal to the number of dimensions of x;  ",
              img_shape_str(shape), " vs ", img_shape_str(aval_of(image)$shape),
              class = "jax_value_error")
}

# Port of scale.py:_resize_nearest.
img_resize_nearest <- function(x, output_shape) {
  in_shape <- aval_of(x)$shape
  for (d in which(in_shape != output_shape) - 1L) {
    m <- in_shape[d + 1L]
    n <- output_shape[d + 1L]
    offsets <- (jnp_arange(n, dtype = "float32") + 0.5) * m / n
    offsets <- lax_convert_element_type(jnp_floor(offsets), "int32")
    x <- jnp_take(x, offsets, axis = d)
  }
  x
}

#' Resize an image
#'
#' Port of `jax.image.resize`. `shape` gives the size of every dimension (batch and
#' channel dimensions keep their size). Methods: see [image_ResizeMethod]. Except
#' for `"nearest"` (which keeps the dtype), the result is floating point; with
#' `antialias` a downsampling kernel is widened to low-pass filter the image.
#' @param image An array.
#' @param shape Integer vector, the output shape.
#' @param method A method name (see [image_ResizeMethod]).
#' @param antialias Antialias when downsampling.
#' @param precision Precision of the contractions (`"highest"` as in JAX).
#' @return The resized array.
#' @examples
#' \dontrun{
#' img <- jnp_reshape(jnp_arange(16, dtype = "float32"), c(4L, 4L))
#' image_resize(img, c(2L, 2L), "linear")
#' }
#' @export
image_resize <- function(image, shape, method, antialias = TRUE, precision = "highest") {
  image <- as_jax(image)
  shape <- canon_shape(shape)
  img_check_shape(shape, image)
  method <- img_method(method)
  if (method == "nearest") return(img_resize_nearest(image, shape))
  image <- promote_dtypes_inexact(list(image))[[1L]]
  in_shape <- aval_of(image)$shape
  spatial_dims <- which(in_shape != shape) - 1L
  if (method == "cubic-pytorch" && antialias) method <- "cubic"
  kr <- img_kernels[[method]]
  scale <- lapply(spatial_dims, function(d) {
    if (shape[d + 1L] == 0L) return(1)
    if (in_shape[d + 1L] == 0L) jax_error("division by zero", class = "jax_value_error")
    shape[d + 1L] / in_shape[d + 1L]
  })
  edge_padding <- method == "cubic-pytorch" && !antialias
  img_scale_and_translate(image, shape, spatial_dims, scale, rep(list(0), length(spatial_dims)),
                          kr$kernel, antialias, precision, edge_padding, kr$radius)
}

#' Scale and translate an image
#'
#' Port of `jax.image.scale_and_translate`: resamples `image` to `shape` so that a
#' location `x` of the input maps to `x * scale + translation` along each dimension in
#' `spatial_dims` (pixels are centered at half integers). Output locations whose
#' source lies outside the input are zero.
#' @param image An array.
#' @param shape Integer vector, the output shape.
#' @param spatial_dims 0-based dimensions to transform.
#' @param scale,translation Arrays (or numbers) with one entry per spatial dimension.
#' @param method A method name (see [image_ResizeMethod]); `"nearest"` is not supported.
#' @param antialias Antialias when downsampling.
#' @param precision Precision of the contractions.
#' @return The resampled array.
#' @export
image_scale_and_translate <- function(image, shape, spatial_dims, scale, translation, method,
                                      antialias = TRUE, precision = "highest") {
  image <- as_jax(image)
  shape <- canon_shape(shape)
  img_check_shape(shape, image)
  method <- img_method(method)
  if (method == "nearest")
    jax_error("Nearest neighbor resampling is not currently supported for scale_and_translate.",
              class = "jax_value_error")
  if (method == "cubic-pytorch" && antialias) method <- "cubic"
  kr <- img_kernels[[method]]
  edge_padding <- method == "cubic-pytorch" && !antialias
  image <- promote_dtypes_inexact(list(image))[[1L]]
  st <- promote_dtypes_inexact(list(scale, translation))
  spatial_dims <- as.integer(spatial_dims)
  # A length 1 R vector is a rank 0 array: one entry.
  st <- lapply(st, function(v) if (length(aval_of(v)$shape) == 0L) lax_reshape(v, 1L) else v)
  for (v in st)
    if (length(aval_of(v)$shape) != 1L || aval_of(v)$shape != length(spatial_dims))
      jax_error("scale and translation must have one entry per spatial dimension (",
                length(spatial_dims), "), got shape ", img_shape_str(aval_of(v)$shape),
                class = "jax_assertion_error")
  pick <- function(v, i) lax_index_in_dim(v, i, 0L, keepdims = FALSE)
  img_scale_and_translate(image, shape, spatial_dims,
                          lapply(seq_along(spatial_dims) - 1L, function(i) pick(st[[1L]], i)),
                          lapply(seq_along(spatial_dims) - 1L, function(i) pick(st[[2L]], i)),
                          kr$kernel, antialias, precision, edge_padding, kr$radius)
}
