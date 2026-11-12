# jax.scipy.ndimage (port of jax/_src/scipy/ndimage.py).

# Port of jax/_src/scipy/ndimage.py:_mirror_index_fixer
ndi_mirror_index_fixer <- function(index, size) {
  s <- size - 1L # Half-wavelength of the triangular wave.
  # Scaled, integer valued version of the triangular wave |x - round(x)|.
  jnp_abs((index + s) %% (2L * s) - s)
}

# Port of jax/_src/scipy/ndimage.py:_reflect_index_fixer
ndi_reflect_index_fixer <- function(index, size) {
  jnp_floor_divide(ndi_mirror_index_fixer(2L * index + 1L, 2L * size + 1L) - 1L, 2L)
}

# Port of jax/_src/scipy/ndimage.py:_INDEX_FIXERS ("nearest" is jnp.clip, which is
# maximum then minimum).
ndi_index_fixers <- list(
  constant = function(index, size) index,
  nearest = function(index, size) jnp_minimum(size - 1L, jnp_maximum(0L, index)),
  wrap = function(index, size) index %% size,
  mirror = ndi_mirror_index_fixer,
  reflect = ndi_reflect_index_fixer
)

# Port of jax/_src/scipy/ndimage.py:_round_half_away_from_zero
ndi_round_half_away_from_zero <- function(a) {
  if (is_int_dtype(aval_of(a)$dtype)) a else lax_round(a)
}

# Port of jax/_src/scipy/ndimage.py:_nearest_indices_and_weights
ndi_nearest_indices_and_weights <- function(coordinate) {
  index <- lax_convert_element_type(ndi_round_half_away_from_zero(coordinate), "int32")
  # coordinate.dtype.type(1): a strongly typed scalar.
  weight <- lax_full(integer(0), 1, aval_of(coordinate)$dtype)
  list(list(index, weight))
}

# Port of jax/_src/scipy/ndimage.py:_linear_indices_and_weights
ndi_linear_indices_and_weights <- function(coordinate) {
  lower <- jnp_floor(coordinate)
  upper_weight <- coordinate - lower
  lower_weight <- 1L - upper_weight
  index <- lax_convert_element_type(lower, "int32")
  list(list(index, lower_weight), list(index + 1L, upper_weight))
}

# itertools.product over a list of lists (the last list varies fastest).
ndi_product <- function(lists) {
  out <- list(list())
  for (l in lists) {
    out <- unlist(lapply(out, function(p) lapply(l, function(e) c(p, list(e)))),
                  recursive = FALSE)
  }
  out
}

#' Map an array to new coordinates by interpolation
#'
#' Port of `jax.scipy.ndimage.map_coordinates`: the value of `input` at each point of
#' `coordinates` (0-based, fractional array coordinates) by nearest (`order = 0`) or
#' linear (`order = 1`) interpolation. Higher orders are not implemented upstream.
#' Integer inputs give rounded integer outputs.
#' @param input Array.
#' @param coordinates A list with one coordinate array per dimension of `input` (the
#'   arrays broadcast together), or an array whose first axis indexes the dimensions.
#' @param order Interpolation order, `0` or `1`.
#' @param mode Handling of points outside the array: `"constant"` (use `cval`),
#'   `"nearest"`, `"wrap"`, `"mirror"` or `"reflect"`.
#' @param cval Fill value for `mode = "constant"`.
#' @return An array with the broadcast shape of the coordinates and the dtype of
#'   `input`.
#' @examples
#' \dontrun{
#' x <- matrix(as.numeric(0:11), 3, 4, byrow = TRUE)
#' scipy_ndimage_map_coordinates(x, list(c(0.5, 2), c(0.5, 1)), order = 1L)
#' }
#' @export
scipy_ndimage_map_coordinates <- function(input, coordinates, order, mode = "constant",
                                          cval = 0.0) {
  # Port of jax/_src/scipy/ndimage.py:map_coordinates and _map_coordinates
  input_arr <- jnp_asarray(input)
  if (!is.list(coordinates)) {
    # A sequence given as one array: iterate over its first axis.
    coordinates <- jnp_asarray(coordinates)
    n <- aval_of(coordinates)$shape[1L]
    coordinates <- lapply(seq_len(n) - 1L, function(i) coordinates[i])
  }
  coordinate_arrs <- lapply(coordinates, jnp_asarray)
  in_dtype <- aval_of(input_arr)$dtype
  cval <- jnp_asarray(cval, in_dtype)
  shape <- aval_of(input_arr)$shape

  if (length(coordinates) != length(shape))
    jax_error("coordinates must be a sequence of length input.ndim, but ", length(coordinates),
              " != ", length(shape), class = "jax_value_error")

  index_fixer <- if (is.character(mode) && length(mode) == 1L) ndi_index_fixers[[mode]]
  if (is.null(index_fixer))
    jax_error("jax.scipy.ndimage.map_coordinates does not yet support mode ", format(mode),
              ". Currently supported modes are {'constant', 'nearest', 'wrap', 'mirror', ",
              "'reflect'}.", class = "jax_not_implemented_error")

  is_valid <- if (identical(mode, "constant")) {
    function(index, size) jnp_logical_and(jnp_less_equal(0L, index), jnp_less(index, size))
  } else {
    function(index, size) TRUE
  }

  interp_fun <- if (isTRUE(order == 0)) ndi_nearest_indices_and_weights
    else if (isTRUE(order == 1)) ndi_linear_indices_and_weights
    else jax_error("jax.scipy.ndimage.map_coordinates currently requires order<=1",
                   class = "jax_not_implemented_error")

  valid_1d_interpolations <- Map(function(coordinate, size) {
    lapply(interp_fun(coordinate), function(node) {
      index <- node[[1L]]
      list(index = index_fixer(index, size), valid = is_valid(index, size), weight = node[[2L]])
    })
  }, coordinate_arrs, as.list(shape))

  outputs <- lapply(ndi_product(valid_1d_interpolations), function(items) {
    indices <- lapply(items, `[[`, "index")
    validities <- lapply(items, `[[`, "valid")
    weights <- lapply(items, `[[`, "weight")
    gathered <- do.call(`[`, c(list(input_arr), indices))
    if (all(vapply(validities, isTRUE, NA))) {
      # Fast path.
      contribution <- gathered
    } else {
      all_valid <- Reduce(`&`, validities)
      contribution <- jnp_where(all_valid, gathered, cval)
    }
    Reduce(`*`, weights) * contribution
  })
  result <- Reduce(`+`, outputs)
  if (is_int_dtype(in_dtype)) result <- ndi_round_half_away_from_zero(result)
  lax_convert_element_type(result, in_dtype)
}
