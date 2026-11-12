# jax.scipy.interpolate, jax.scipy.cluster.vq and jax.scipy.spatial.transform (ports of
# jax/_src/third_party/scipy/interpolate.py, jax/_src/scipy/cluster/vq.py and
# jax/_src/scipy/spatial/transform.py).

smisc_shape <- function(x) aval_of(as_jax(x))$shape
smisc_ndim <- function(x) length(aval_of(as_jax(x))$shape)

# ---- jax.scipy.interpolate ---------------------------------------------------------------

# Port of jax/_src/third_party/scipy/interpolate.py:_ndim_coords_from_arrays. A list
# plays the role of the tuple of coordinate arrays.
rgi_ndim_coords_from_arrays <- function(points, ndim = NULL) {
  if (is.list(points) && length(points) == 1L) points <- points[[1L]]
  if (is.list(points)) {
    p <- do.call(jnp_broadcast_arrays, unname(points))
    # empty(..., dtype=float) filled with .at[..., j].set(item).
    return(jnp_stack(lapply(p, lax_convert_element_type, new_dtype = default_float()),
                     axis = -1L))
  }
  points <- jnp_asarray(points)
  if (smisc_ndim(points) == 1L) points <- jnp_reshape(points, c(-1L, ndim %||% 1L))
  points
}

# numpy.can_cast(from, to, casting = "same_kind") for a target of inexact kind.
rgi_can_cast_same_kind <- function(from, to) {
  rank <- c(b = 0L, u = 1L, i = 1L, f = 2L, c = 3L, V = 4L)
  rank[[dtype_kind(from)]] <= rank[[dtype_kind(to)]]
}

rgi_check_method <- function(method) {
  if (!(is.character(method) && length(method) == 1L && method %in% c("linear", "nearest")))
    jax_error("method ", if (is.character(method)) paste0("'", method, "'") else format(method),
              " is not defined", class = "jax_value_error")
}

#' Interpolation on a regular grid
#'
#' Port of `jax.scipy.interpolate.RegularGridInterpolator`: linear or nearest
#' neighbor interpolation of values given on a rectilinear grid in any number of
#' dimensions. The constructor returns a callable object: `interp(xi, method = NULL)`
#' evaluates the interpolant at the points `xi` (an array whose last axis has one
#' coordinate per grid dimension, or a list of coordinate arrays that broadcast
#' together). Its fields `interp$grid`, `interp$values`, `interp$method`,
#' `interp$bounds_error` and `interp$fill_value` are read with `$`. The object is a
#' pytree, so it can be passed through [jit()] and [vmap()].
#' @param points List of 1-D arrays of strictly ascending grid coordinates, one per
#'   dimension (a single vector is taken as a one dimensional grid).
#' @param values Array of values on the grid; trailing dimensions beyond the grid are
#'   carried along.
#' @param method `"linear"` or `"nearest"`.
#' @param bounds_error Must be `FALSE` (bounds cannot be checked under jit).
#' @param fill_value Value for points outside the grid, or `NULL` to extrapolate.
#' @param x Interpolator object.
#' @param name Field name.
#' @return A callable interpolator; for `$`, the field.
#' @examples
#' \dontrun{
#' interp <- scipy_interpolate_RegularGridInterpolator(
#'   list(c(0, 1, 2), c(0, 1)), matrix(c(0, 1, 2, 3, 4, 5), 3, 2, byrow = TRUE))
#' interp(matrix(c(0.5, 0.5, 1.5, 0.25), 2, 2, byrow = TRUE))
#' }
#' @export
scipy_interpolate_RegularGridInterpolator <- function(points, values, method = "linear",
                                                      bounds_error = FALSE, fill_value = NaN) {
  # Port of jax/_src/third_party/scipy/interpolate.py:RegularGridInterpolator.__init__
  # (based on SciPy's implementation, originally by Johannes Buchner).
  rgi_check_method(method)
  if (isTRUE(bounds_error))
    jax_error("`bounds_error` takes no effect under JIT", class = "jax_not_implemented_error")
  values <- jnp_asarray(values)
  if (!is.list(points)) points <- list(points)
  if (length(points) > smisc_ndim(values))
    jax_error("there are ", length(points), " point arrays, but values has ", smisc_ndim(values),
              " dimensions", class = "jax_value_error")
  values <- promote_dtypes_inexact(list(values))[[1L]]
  if (!is.null(fill_value)) {
    fill_value <- jnp_asarray(fill_value)
    if (!rgi_can_cast_same_kind(aval_of(fill_value)$dtype, aval_of(values)$dtype))
      jax_error("fill_value must be either 'None' or of a type compatible with values",
                class = "jax_value_error")
  }
  self <- list(grid = lapply(unname(points), jnp_asarray), values = values, method = method,
               bounds_error = bounds_error, fill_value = fill_value)
  interp <- function(xi, method = NULL) rgi_call(self, xi, method)
  class(interp) <- "scipy_interpolate_RegularGridInterpolator"
  interp
}

#' @rdname scipy_interpolate_RegularGridInterpolator
#' @export
`$.scipy_interpolate_RegularGridInterpolator` <- function(x, name) environment(x)$self[[name]]

register_pytree_node("scipy_interpolate_RegularGridInterpolator",
  function(x) {
    s <- environment(x)$self
    list(children = list(s$grid, s$values, s$fill_value), aux = list(s$method, s$bounds_error))
  },
  function(aux, children) {
    scipy_interpolate_RegularGridInterpolator(children[[1L]], children[[2L]], aux[[1L]],
                                              aux[[2L]], children[[3L]])
  })

# Port of RegularGridInterpolator.__call__
rgi_call <- function(self, xi, method = NULL) {
  method <- method %||% self$method
  rgi_check_method(method)
  ndim <- length(self$grid)
  xi <- rgi_ndim_coords_from_arrays(xi, ndim = ndim)
  xi_shape <- smisc_shape(xi)
  if (xi_shape[length(xi_shape)] != ndim)
    jax_error("the requested sample points xi have dimension ", xi_shape[2L], ", but this ",
              "RegularGridInterpolator has dimension ", ndim, class = "jax_value_error")
  xi <- jnp_reshape(xi, c(-1L, xi_shape[length(xi_shape)]))
  found <- rgi_find_indices(self, jnp_transpose(xi))
  result <- if (method == "linear") rgi_evaluate_linear(self, found$indices, found$norm_distances)
            else rgi_evaluate_nearest(self, found$indices, found$norm_distances)
  if (!isTRUE(self$bounds_error) && !is.null(self$fill_value)) {
    rs <- smisc_shape(result)
    bc_shp <- c(rs[1L], rep(1L, length(rs) - 1L))
    result <- jnp_where(jnp_reshape(found$out_of_bounds, bc_shp), self$fill_value, result)
  }
  jnp_reshape(result, c(xi_shape[-length(xi_shape)], drop_first(smisc_shape(self$values), ndim)))
}

# Port of RegularGridInterpolator._evaluate_linear
rgi_evaluate_linear <- function(self, indices, norm_distances) {
  # Slice for broadcasting over the trailing dimensions of the values.
  vslice <- c(list(sl()), rep(list(newaxis), smisc_ndim(self$values) - length(indices)))
  # Each i and i + 1 is an edge.
  edges <- ndi_product(lapply(indices, function(i) list(i, i + 1L)))
  values <- jnp_asarray(0)
  for (edge_indices in edges) {
    weight <- jnp_asarray(1)
    for (k in seq_along(indices)) {
      weight <- weight * jnp_where(edge_indices[[k]] == indices[[k]], 1L - norm_distances[[k]],
                                   norm_distances[[k]])
    }
    values <- values + do.call(`[`, c(list(self$values), edge_indices)) *
      do.call(`[`, c(list(weight), vslice))
  }
  values
}

# Port of RegularGridInterpolator._evaluate_nearest
rgi_evaluate_nearest <- function(self, indices, norm_distances) {
  idx_res <- Map(function(i, yi) jnp_where(yi <= 0.5, i, i + 1L), indices, norm_distances)
  do.call(`[`, c(list(self$values), unname(idx_res)))
}

# Port of RegularGridInterpolator._find_indices
rgi_find_indices <- function(self, xi) {
  indices <- list()
  # Distance to the lower edge in unity units.
  norm_distances <- list()
  out_of_bounds <- jnp_zeros(smisc_shape(xi)[2L], dtype = "bool")
  for (d in seq_along(self$grid)) {
    x <- xi[d - 1L]
    g <- self$grid[[d]]
    gsize <- prod_int(smisc_shape(g))
    i <- jnp_searchsorted(g, x) - 1L
    i <- jnp_where(i < 0L, 0L, i)
    i <- jnp_where(i > gsize - 2L, gsize - 2L, i)
    indices[[d]] <- i
    norm_distances[[d]] <- (x - g[i]) / (g[i + 1L] - g[i])
    if (!isTRUE(self$bounds_error)) {
      out_of_bounds <- out_of_bounds + (x < g[0L])
      out_of_bounds <- out_of_bounds + (x > g[-1L])
    }
  }
  list(indices = indices, norm_distances = norm_distances, out_of_bounds = out_of_bounds)
}

# ---- jax.scipy.cluster.vq ------------------------------------------------------------------

#' Vector quantization
#'
#' Port of `jax.scipy.cluster.vq.vq`: assigns each observation to the nearest code in
#' the code book (Euclidean distance).
#' @param obs Array of observations, shape `(M, N)` (or `(M,)` for scalar features).
#' @param code_book Array of codes, shape `(K, N)` (or `(K,)`).
#' @param check_finite Unused (accepted for compatibility).
#' @return A list of the 0-based code index of each observation and its distance to
#'   that code.
#' @examples
#' \dontrun{
#' scipy_cluster_vq_vq(matrix(c(0, 0, 1, 1, 5, 5), 3, byrow = TRUE),
#'                     matrix(c(0, 0, 4, 4), 2, byrow = TRUE))
#' }
#' @export
scipy_cluster_vq_vq <- function(obs, code_book, check_finite = TRUE) {
  # Port of jax/_src/scipy/cluster/vq.py:vq
  a <- promote_dtypes_inexact(list(obs, code_book))
  obs_arr <- a[[1L]]
  cb_arr <- a[[2L]]
  if (smisc_ndim(obs_arr) != smisc_ndim(cb_arr))
    jax_error("Observation and code_book should have the same rank", class = "jax_value_error")
  if (smisc_ndim(obs_arr) == 1L) {
    obs_arr <- obs_arr[ellipsis, newaxis]
    cb_arr <- cb_arr[ellipsis, newaxis]
  }
  if (smisc_ndim(obs_arr) != 2L)
    jax_error("ndim different than 1 or 2 are not supported", class = "jax_value_error")
  dist <- vmap(function(ob) jnp_linalg_norm(ob[newaxis] - cb_arr, axis = -1L))(obs_arr)
  code <- jnp_argmin(dist, axis = -1L)
  dist_min <- vmap(function(d, i) d[i])(dist, code)
  list(code, dist_min)
}

# ---- jax.scipy.spatial.transform -----------------------------------------------------------

# The private helpers of transform.py are jnp.vectorize'd kernels; rot_vec() builds the
# vectorized function when it is called.
rot_vec <- function(kernel, signature) {
  force(kernel)
  function(...) jnp_vectorize(kernel, signature = signature)(...)
}

# Port of transform.py:_apply
rot_apply <- rot_vec(function(matrix, vector, inverse) {
  jnp_matmul(jnp_where(inverse, jnp_transpose(matrix), matrix), vector)
}, "(m,m),(m),()->(m)")

# Port of transform.py:_as_matrix
rot_as_matrix <- rot_vec(function(quat) {
  x <- quat[0L]
  y <- quat[1L]
  z <- quat[2L]
  w <- quat[3L]
  x2 <- x * x
  y2 <- y * y
  z2 <- z * z
  w2 <- w * w
  xy <- x * y
  zw <- z * w
  xz <- x * z
  yw <- y * w
  yz <- y * z
  xw <- x * w
  jnp_array(list(list(x2 - y2 - z2 + w2, 2L * (xy - zw), 2L * (xz + yw)),
                 list(2L * (xy + zw), -x2 + y2 - z2 + w2, 2L * (yz - xw)),
                 list(2L * (xz - yw), 2L * (yz + xw), -x2 - y2 + z2 + w2)))
}, "(m)->(n,n)")

# Port of transform.py:_as_mrp
rot_as_mrp <- rot_vec(function(quat) {
  sign <- jnp_where(quat[3L] < 0L, -1, 1)
  denominator <- 1 + sign * quat[3L]
  sign * quat[sl(NULL, 3L)] / denominator
}, "(m)->(n)")

# Port of transform.py:_as_rotvec
rot_as_rotvec <- rot_vec(function(quat, degrees) {
  # w > 0 to ensure 0 <= angle <= pi.
  quat <- jnp_where(quat[3L] < 0L, -quat, quat)
  angle <- 2 * jnp_arctan2(rot_vector_norm(quat[sl(NULL, 3L)]), quat[3L])
  angle2 <- angle * angle
  small_scale <- 2L + angle2 / 12L + 7L * angle2 * angle2 / 2880L
  large_scale <- angle / jnp_sin(angle / 2L)
  scale <- jnp_where(angle <= 1e-3, small_scale, large_scale)
  scale <- jnp_where(degrees, jnp_rad2deg(scale), scale)
  scale * jnp_array(quat[sl(NULL, 3L)])
}, "(m),()->(n)")

# Port of transform.py:_compose_quat
rot_compose_quat <- rot_vec(function(p, q) {
  cross <- jnp_cross(p[sl(NULL, 3L)], q[sl(NULL, 3L)])
  jnp_array(list(p[3L] * q[0L] + q[3L] * p[0L] + cross[0L],
                 p[3L] * q[1L] + q[3L] * p[1L] + cross[1L],
                 p[3L] * q[2L] + q[3L] * p[2L] + cross[2L],
                 p[3L] * q[3L] - p[0L] * q[0L] - p[1L] * q[1L] - p[2L] * q[2L]))
}, "(n),(n)->(n)")

# Port of transform.py:_compute_euler_from_quat
rot_compute_euler_from_quat <- rot_vec(function(quat, axes, extrinsic, degrees) {
  dt <- aval_of(quat)$dtype
  angle_first <- jnp_where(extrinsic, 0L, 2L)
  angle_third <- jnp_where(extrinsic, 2L, 0L)
  axes <- jnp_where(extrinsic, axes, axes[sl(step = -1L)])
  i <- axes[0L]
  j <- axes[1L]
  k <- axes[2L]
  symmetric <- i == k
  k <- jnp_where(symmetric, 3L - i - j, k)
  sign <- jnp_array(((i - j) * (j - k) * (k - i)) %/% 2L, dtype = dt)
  eps <- 1e-7
  a <- jnp_where(symmetric, quat[3L], quat[3L] - quat[j])
  b <- jnp_where(symmetric, quat[i], quat[i] + quat[k] * sign)
  cc <- jnp_where(symmetric, quat[j], quat[j] + quat[3L])
  d <- jnp_where(symmetric, quat[k] * sign, quat[k] * sign - quat[i])
  angles <- jnp_zeros(3L, dtype = dt)
  angles <- at_set(angles, 1L, value = 2L * jnp_arctan2(jnp_hypot(cc, d), jnp_hypot(a, b)))
  case <- jnp_where(jnp_abs(angles[1L] - pi) <= eps, 2L, 0L)
  case <- jnp_where(jnp_abs(angles[1L]) <= eps, 1L, case)
  half_sum <- jnp_arctan2(b, a)
  half_diff <- jnp_arctan2(d, cc)
  # Any degenerate case.
  angles <- at_set(angles, 0L, value = jnp_where(case == 1L, 2L * half_sum,
                                                 2L * half_diff * jnp_where(extrinsic, -1L, 1L)))
  angles <- at_set(angles, angle_first,
                   value = jnp_where(case == 0L, half_sum - half_diff, angles[angle_first]))
  angles <- at_set(angles, angle_third,
                   value = jnp_where(case == 0L, half_sum + half_diff, angles[angle_third]))
  angles <- at_set(angles, angle_third,
                   value = jnp_where(symmetric, angles[angle_third], angles[angle_third] * sign))
  angles <- at_set(angles, 1L, value = jnp_where(symmetric, angles[1L], angles[1L] - pi / 2))
  angles <- (angles + pi) %% (2 * pi) - pi
  jnp_where(degrees, jnp_rad2deg(angles), angles)
}, "(m),(l),(),()->(n)")

# Port of transform.py:_elementary_basis_index
rot_elementary_basis_index <- function(axis) {
  switch(axis, x = 0L, y = 1L, z = 2L,
         jax_error("Expected axis to be from ['x', 'y', 'z'], got ", axis,
                   class = "jax_value_error"))
}

# Port of transform.py:_elementary_quat_compose
rot_elementary_quat_compose <- rot_vec(function(angles, axes, intrinsic, degrees) {
  angles <- jnp_where(degrees, jnp_deg2rad(angles), angles)
  result <- rot_make_elementary_quat(axes[0L], angles[0L])
  for (idx in seq_len(smisc_shape(axes)[1L] - 1L)) {
    quat <- rot_make_elementary_quat(axes[idx], angles[idx])
    result <- jnp_where(intrinsic, rot_compose_quat(result, quat), rot_compose_quat(quat, result))
  }
  result
}, "(m),(m),(),()->(n)")

# Port of transform.py:_from_rotvec
rot_from_rotvec <- rot_vec(function(rotvec, degrees) {
  rotvec <- jnp_where(degrees, jnp_deg2rad(rotvec), rotvec)
  angle <- rot_vector_norm(rotvec)
  angle2 <- angle * angle
  small_scale <- 0.5 - angle2 / 48L + angle2 * angle2 / 3840L
  large_scale <- jnp_sin(angle / 2L) / angle
  scale <- jnp_where(angle <= 1e-3, small_scale, large_scale)
  jnp_hstack(list(scale * rotvec, jnp_cos(angle / 2L)))
}, "(m),()->(n)")

# Port of transform.py:_from_matrix
rot_from_matrix <- rot_vec(function(matrix) {
  dt <- aval_of(matrix)$dtype
  matrix_trace <- matrix[0L, 0L] + matrix[1L, 1L] + matrix[2L, 2L]
  decision <- jnp_array(list(matrix[0L, 0L], matrix[1L, 1L], matrix[2L, 2L], matrix_trace),
                        dtype = dt)
  choice <- jnp_argmax(decision)
  i <- choice
  j <- (i + 1L) %% 3L
  k <- (j + 1L) %% 3L
  # jnp.empty is zeros.
  quat_012 <- jnp_zeros(4L, dtype = dt)
  quat_012 <- at_set(quat_012, i, value = 1L - decision[3L] + 2L * matrix[i, i])
  quat_012 <- at_set(quat_012, j, value = matrix[j, i] + matrix[i, j])
  quat_012 <- at_set(quat_012, k, value = matrix[k, i] + matrix[i, k])
  quat_012 <- at_set(quat_012, 3L, value = matrix[k, j] - matrix[j, k])
  quat_3 <- jnp_zeros(4L, dtype = dt)
  quat_3 <- at_set(quat_3, 0L, value = matrix[2L, 1L] - matrix[1L, 2L])
  quat_3 <- at_set(quat_3, 1L, value = matrix[0L, 2L] - matrix[2L, 0L])
  quat_3 <- at_set(quat_3, 2L, value = matrix[1L, 0L] - matrix[0L, 1L])
  quat_3 <- at_set(quat_3, 3L, value = 1L + decision[3L])
  quat <- jnp_where(choice != 3L, quat_012, quat_3)
  rot_normalize_quaternion(quat)
}, "(m,m)->(n)")

# Port of transform.py:_from_mrp
rot_from_mrp <- rot_vec(function(mrp) {
  mrp_squared_plus_1 <- jnp_dot(mrp, mrp) + 1L
  jnp_hstack(list(2L * mrp[sl(NULL, 3L)], 2L - mrp_squared_plus_1)) / mrp_squared_plus_1
}, "(m)->(n)")

# Port of transform.py:_inv
rot_inv <- rot_vec(function(quat) {
  quat * jnp_array(c(-1, -1, -1, 1), dtype = aval_of(quat)$dtype)
}, "(n)->(n)")

# Port of transform.py:_magnitude
rot_magnitude <- rot_vec(function(quat) {
  2 * jnp_arctan2(rot_vector_norm(quat[sl(NULL, 3L)]), jnp_abs(quat[3L]))
}, "(n)->()")

# Port of transform.py:_make_elementary_quat
rot_make_elementary_quat <- rot_vec(function(axis, angle) {
  quat <- jnp_zeros(4L, dtype = aval_of(angle)$dtype)
  quat <- at_set(quat, 3L, value = jnp_cos(angle / 2))
  at_set(quat, axis, value = jnp_sin(angle / 2))
}, "(),()->(n)")

# Port of transform.py:_normalize_quaternion
rot_normalize_quaternion <- rot_vec(function(quat) quat / rot_vector_norm(quat), "(n)->(n)")

# Port of transform.py:_vector_norm
rot_vector_norm <- rot_vec(function(vector) jnp_sqrt(jnp_dot(vector, vector)), "(n)->()")

# Port of transform.py:_make_canonical
rot_make_canonical <- rot_vec(function(quat) {
  is_neg <- quat < 0L
  is_zero <- quat == 0L
  neg <- is_neg[3L] |
    (is_zero[3L] & is_neg[0L]) |
    (is_zero[3L] & is_zero[0L] & is_neg[1L]) |
    (is_zero[3L] & is_zero[0L] & is_zero[1L] & is_neg[2L])
  jnp_where(neg, -quat, quat)
}, "(n)->(n)")

rot_check_seq <- function(seq, n_expected = NULL) {
  intrinsic <- grepl("^[XYZ]{1,3}$", seq)
  extrinsic <- grepl("^[xyz]{1,3}$", seq)
  if (!(intrinsic || extrinsic))
    jax_error("Expected axes from `seq` to be from ['x', 'y', 'z'] or ['X', 'Y', 'Z'], got ",
              seq, class = "jax_value_error")
  chars <- strsplit(seq, "")[[1L]]
  if (any(chars[-1L] == chars[-length(chars)]))
    jax_error("Expected consecutive axes to be different, got ", seq, class = "jax_value_error")
  axes <- unname(vapply(tolower(chars), rot_elementary_basis_index, 1L))
  # array() keeps a one character sequence one dimensional.
  list(intrinsic = intrinsic, extrinsic = extrinsic,
       axes = as_jax(array(axes, length(axes)), default_int()))
}

#' Rotations in three dimensions
#'
#' Port of `jax.scipy.spatial.transform.Rotation`, a pytree holding unit quaternions
#' in scalar last `(x, y, z, w)` order: `quat` has shape `(4,)` for a single rotation
#' or `(N, 4)` for `N` rotations. The class methods of JAX are functions:
#' `scipy_spatial_transform_Rotation_from_quat(quat)` and the other `_from_*`
#' constructors, `_identity`, `_concatenate`; instance methods take the rotation as
#' first argument (`scipy_spatial_transform_Rotation_apply(r, v)` is `r.apply(v)`).
#' `r[i]` indexes the rotations (0-based) and `r1 * r2` composes them (`r2` is
#' applied first). `scipy_spatial_transform_Rotation()` itself stores `quat` as given,
#' like the JAX named tuple; `_from_quat` normalizes.
#' @param quat Quaternion array (scalar last).
#' @param rotations List of rotations.
#' @param seq Axis sequence for Euler angles: up to three characters from `"xyz"`
#'   (extrinsic) or `"XYZ"` (intrinsic).
#' @param angles Euler angles, shape `(..., length(seq))`.
#' @param degrees Whether angles are in degrees.
#' @param matrix Rotation matrices, shape `(..., 3, 3)`.
#' @param mrp Modified Rodrigues parameters, shape `(..., 3)`.
#' @param rotvec Rotation vectors (axis times angle), shape `(..., 3)`.
#' @param num Must be `NULL` (one identity rotation).
#' @param dtype Dtype of the identity quaternion.
#' @param random_key Unused (`random` is not implemented upstream).
#' @param ... Index arguments (see [jax_indexing]); 0-based.
#' @param self,x,e1,e2 Rotations.
#' @param vectors Vectors to rotate, shape `(..., 3)`.
#' @param inverse Apply the inverse rotation.
#' @param canonical Return the quaternion with non-negative `w` (or the first nonzero
#'   component positive).
#' @param scalar_first Return quaternions in `(w, x, y, z)` order.
#' @param weights Optional weights of the rotations for the mean.
#' @return A rotation, or an array for the `as_*`, `apply` and `magnitude` methods.
#' @examples
#' \dontrun{
#' r <- scipy_spatial_transform_Rotation_from_euler("xyz", c(90, 0, 0), degrees = TRUE)
#' scipy_spatial_transform_Rotation_apply(r, c(0, 1, 0))
#' scipy_spatial_transform_Rotation_as_rotvec(r)
#' }
#' @name scipy_spatial_transform_Rotation
NULL

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation <- function(quat) {
  structure(list(quat = quat), class = "scipy_spatial_transform_Rotation")
}

register_pytree_node("scipy_spatial_transform_Rotation",
  function(x) list(children = list(x$quat), aux = NULL),
  function(aux, children) scipy_spatial_transform_Rotation(children[[1L]]))

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_concatenate <- function(rotations) {
  scipy_spatial_transform_Rotation(jnp_concatenate(lapply(rotations, function(r) r$quat)))
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_from_euler <- function(seq, angles, degrees = FALSE) {
  # Port of transform.py:Rotation.from_euler
  num_axes <- nchar(seq)
  if (num_axes < 1L || num_axes > 3L)
    jax_error("Expected axis specification to be a non-empty string of upto 3 characters, got ",
              seq, class = "jax_value_error")
  s <- rot_check_seq(seq)
  angles <- jnp_atleast_1d(angles)
  scipy_spatial_transform_Rotation(rot_elementary_quat_compose(angles, s$axes, s$intrinsic,
                                                               degrees))
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_from_matrix <- function(matrix) {
  scipy_spatial_transform_Rotation(rot_from_matrix(matrix))
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_from_mrp <- function(mrp) {
  scipy_spatial_transform_Rotation(rot_from_mrp(mrp))
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_from_quat <- function(quat) {
  scipy_spatial_transform_Rotation(rot_normalize_quaternion(quat))
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_from_rotvec <- function(rotvec, degrees = FALSE) {
  scipy_spatial_transform_Rotation(rot_from_rotvec(rotvec, degrees))
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_identity <- function(num = NULL, dtype = "float") {
  stopifnot(is.null(num))
  scipy_spatial_transform_Rotation(jnp_array(c(0, 0, 0, 1), dtype = dtype))
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_random <- function(random_key, num = NULL) {
  # Upstream needs scipy.stats.special_ortho_group for this.
  jax_error("Rotation.random is not implemented", class = "jax_not_implemented_error")
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
`[.scipy_spatial_transform_Rotation` <- function(x, ...) {
  if (scipy_spatial_transform_Rotation_single(x))
    jax_error("Single rotation is not subscriptable.", class = "jax_type_error")
  scipy_spatial_transform_Rotation(as_jax(x$quat)[...])
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_len <- function(self) {
  if (scipy_spatial_transform_Rotation_single(self))
    jax_error("Single rotation has no len().", class = "jax_type_error")
  smisc_shape(self$quat)[1L]
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
`*.scipy_spatial_transform_Rotation` <- function(e1, e2) {
  scipy_spatial_transform_Rotation_from_quat(rot_compose_quat(e1$quat, e2$quat))
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_apply <- function(self, vectors, inverse = FALSE) {
  rot_apply(scipy_spatial_transform_Rotation_as_matrix(self), vectors, inverse)
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_as_euler <- function(self, seq, degrees = FALSE) {
  # Port of transform.py:Rotation.as_euler
  if (nchar(seq) != 3L) jax_error("Expected 3 axes, got ", seq, ".", class = "jax_value_error")
  s <- rot_check_seq(seq)
  with_jax_config("jax_numpy_rank_promotion", "allow",
                  rot_compute_euler_from_quat(self$quat, s$axes, s$extrinsic, degrees))
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_as_matrix <- function(self) rot_as_matrix(self$quat)

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_as_mrp <- function(self) rot_as_mrp(self$quat)

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_as_rotvec <- function(self, degrees = FALSE) {
  rot_as_rotvec(self$quat, degrees)
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_as_quat <- function(self, canonical = FALSE,
                                                     scalar_first = FALSE) {
  quat <- if (isTRUE(canonical)) rot_make_canonical(self$quat) else self$quat
  if (isTRUE(scalar_first)) return(jnp_roll(quat, shift = 1L, axis = -1L))
  quat
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_inv <- function(self) {
  scipy_spatial_transform_Rotation(rot_inv(self$quat))
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_magnitude <- function(self) rot_magnitude(self$quat)

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_mean <- function(self, weights = NULL) {
  # Port of transform.py:Rotation.mean
  quat <- as_jax(self$quat)
  dt <- aval_of(quat)$dtype
  w <- if (is.null(weights)) jnp_ones(smisc_shape(quat)[1L], dtype = dt)
       else jnp_asarray(weights, dtype = dt)
  if (smisc_ndim(w) != 1L)
    jax_error("Expected `weights` to be 1 dimensional, got shape ", gs_tuple(smisc_shape(w)), ".",
              class = "jax_value_error")
  n <- scipy_spatial_transform_Rotation_len(self)
  if (smisc_shape(w)[1L] != n)
    jax_error("Expected `weights` to have number of values equal to number of rotations, got ",
              smisc_shape(w)[1L], " values and ", n, " rotations.", class = "jax_value_error")
  K <- jnp_dot(w[newaxis] * jnp_transpose(quat), quat)
  v <- jnp_linalg_eigh(K)[[2L]]
  scipy_spatial_transform_Rotation(v[, -1L])
}

#' @rdname scipy_spatial_transform_Rotation
#' @export
scipy_spatial_transform_Rotation_single <- function(self) smisc_ndim(self$quat) == 1L

#' Spherical linear interpolation of rotations
#'
#' Port of `jax.scipy.spatial.transform.Slerp`. `scipy_spatial_transform_Slerp_init()`
#' (the `Slerp.init` class method) builds the interpolator from increasing `times`
#' and a rotation holding one rotation per time; `scipy_spatial_transform_Slerp()` is
#' the raw constructor of the named tuple. The result is callable: `slerp(times)`
#' returns the interpolated rotations (a single rotation for a scalar time). Its
#' fields `times`, `timedelta`, `rotations` and `rotvecs` are read with `$`, and it is
#' a pytree.
#' @param times Times of the key rotations (1-D, increasing); for the raw constructor
#'   the stored times.
#' @param rotations A [scipy_spatial_transform_Rotation()] with at least two rotations.
#' @param timedelta,rotvecs Stored fields of the raw constructor.
#' @param x Slerp object.
#' @param name Field name.
#' @return A callable Slerp object; for `$`, the field.
#' @examples
#' \dontrun{
#' rots <- scipy_spatial_transform_Rotation_from_euler(
#'   "z", matrix(c(0, 90), 2), degrees = TRUE)
#' slerp <- scipy_spatial_transform_Slerp_init(c(0, 1), rots)
#' scipy_spatial_transform_Rotation_as_euler(slerp(0.5), "zyx", degrees = TRUE)
#' }
#' @export
scipy_spatial_transform_Slerp <- function(times, timedelta, rotations, rotvecs) {
  self <- list(times = times, timedelta = timedelta, rotations = rotations, rotvecs = rotvecs)
  slerp <- function(times) slerp_call(self, times)
  class(slerp) <- "scipy_spatial_transform_Slerp"
  slerp
}

#' @rdname scipy_spatial_transform_Slerp
#' @export
`$.scipy_spatial_transform_Slerp` <- function(x, name) environment(x)$self[[name]]

register_pytree_node("scipy_spatial_transform_Slerp",
  function(x) {
    s <- environment(x)$self
    list(children = list(s$times, s$timedelta, s$rotations, s$rotvecs), aux = NULL)
  },
  function(aux, children) {
    scipy_spatial_transform_Slerp(children[[1L]], children[[2L]], children[[3L]], children[[4L]])
  })

#' @rdname scipy_spatial_transform_Slerp
#' @export
scipy_spatial_transform_Slerp_init <- function(times, rotations) {
  # Port of transform.py:Slerp.init
  if (!inherits(rotations, "scipy_spatial_transform_Rotation"))
    jax_error("`rotations` must be a `Rotation` instance.", class = "jax_type_error")
  if (scipy_spatial_transform_Rotation_single(rotations) ||
      scipy_spatial_transform_Rotation_len(rotations) == 1L)
    jax_error("`rotations` must be a sequence of at least 2 rotations.", class = "jax_value_error")
  times <- jnp_asarray(times, dtype = aval_of(as_jax(rotations$quat))$dtype)
  if (smisc_ndim(times) != 1L)
    jax_error("Expected times to be specified in a 1 dimensional array, got ", smisc_ndim(times),
              " dimensions.", class = "jax_value_error")
  n <- scipy_spatial_transform_Rotation_len(rotations)
  if (smisc_shape(times)[1L] != n)
    jax_error("Expected number of rotations to be equal to number of timestamps given, got ",
              n, " rotations and ", smisc_shape(times)[1L], " timestamps.",
              class = "jax_value_error")
  timedelta <- jnp_diff(times)
  # Upstream does not check that the times increase (it would need concrete values).
  quat <- as_jax(scipy_spatial_transform_Rotation_as_quat(rotations))
  new_rotations <- scipy_spatial_transform_Rotation(quat[sl(NULL, -1L)])
  rel <- scipy_spatial_transform_Rotation_inv(new_rotations) *
    scipy_spatial_transform_Rotation(quat[sl(1L)])
  scipy_spatial_transform_Slerp(times, timedelta, new_rotations,
                                scipy_spatial_transform_Rotation_as_rotvec(rel))
}

# Port of transform.py:Slerp.__call__
slerp_call <- function(self, times) {
  self_times <- as_jax(self$times)
  compute_times <- jnp_asarray(times, dtype = aval_of(self_times)$dtype)
  if (smisc_ndim(compute_times) > 1L)
    jax_error("`times` must be at most 1-dimensional.", class = "jax_value_error")
  single_time <- smisc_ndim(compute_times) == 0L
  compute_times <- jnp_atleast_1d(compute_times)
  ind <- jnp_maximum(jnp_searchsorted(self_times, compute_times) - 1L, 0L)
  alpha <- (compute_times - self_times[ind]) / as_jax(self$timedelta)[ind]
  result <- self$rotations[ind] *
    scipy_spatial_transform_Rotation_from_rotvec(as_jax(self$rotvecs)[ind] * alpha[, newaxis])
  if (single_time) return(result[0L])
  result
}
