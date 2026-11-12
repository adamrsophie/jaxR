# jax.nn.initializers (port of jax/_src/nn/initializers.py). An initializer is a function
# `init(key, shape, dtype = NULL, out_sharding = NULL)` returning an array (the
# `Initializer` protocol); the builders below return such functions.

# random_split(key) as a list of keys, for typed and raw (uint32) keys alike.
nni_split <- function(key, num = 2L) {
  keys <- random_split(key, num)
  if (is_prng_key(keys)) return(as.list(keys))
  lapply(seq_len(num) - 1L, function(i) lax_index_in_dim(keys, i, 0L, keepdims = FALSE))
}

nni_dtype <- function(d) if (is.null(d)) default_float() else dtype(d)

# jnp.array(value, dtype): a strongly typed array of the given dtype.
nni_array <- function(value, dt) as_jax(value, dt)

#' Neural network initializers
#'
#' Ports of `jax.nn.initializers`. `nn_initializers_zeros` and `nn_initializers_ones` are
#' initializers; the other functions build initializers. An initializer is a function
#' `init(key, shape, dtype = NULL, out_sharding = NULL)` (the `Initializer` protocol of
#' JAX) returning an array of the given shape and dtype (default float). Random
#' initializers give the same values as JAX for the same key.
#'
#' `nn_initializers_variance_scaling` draws from a (truncated) normal distribution with
#' standard deviation `sqrt(scale / n)` or a uniform distribution with that standard
#' deviation, where `n` is the number of inputs (`"fan_in"`), outputs (`"fan_out"`),
#' their mean (`"fan_avg"`) or their geometric mean (`"fan_geo_avg"`). Axes that are not
#' input, output or batch axes form the receptive field of a convolution kernel. Complex
#' dtypes draw from a disk (`"uniform"`) or a truncated complex normal distribution. The
#' Glorot (Xavier), LeCun and He (Kaiming) initializers are its specializations.
#' @param key A PRNG key ([random_key()] or [random_PRNGKey()]); ignored by the constant
#'   initializers.
#' @param shape Integer vector.
#' @param dtype Dtype (default float).
#' @param out_sharding Accepted for compatibility (jaxR runs on one device); the
#'   orthogonal initializers require `NULL`.
#' @param value Fill value.
#' @param scale Upper bound of `uniform`, scaling factor of `variance_scaling` and
#'   `orthogonal`.
#' @param stddev Standard deviation.
#' @param lower,upper Truncation bounds of `truncated_normal` (in units of `stddev`).
#' @param mode One of `"fan_in"`, `"fan_out"`, `"fan_avg"` and `"fan_geo_avg"`.
#' @param distribution One of `"truncated_normal"`, `"normal"` and `"uniform"`.
#' @param in_axis,out_axis,batch_axis 0-based axis or axes of the input dimension, the
#'   output dimension and the ignored (batch) dimensions of the weights.
#' @param column_axis 0-based axis holding the columns that should be orthogonal.
#' @return An array (`zeros`, `ones`) or an initializer function.
#' @examples
#' \dontrun{
#' init <- nn_initializers_glorot_normal()
#' init(random_key(42), c(2, 3))
#' nn_initializers_zeros(random_key(0), c(2, 2))
#' }
#' @name nn_initializers
NULL

#' @rdname nn_initializers
#' @export
nn_initializers_zeros <- function(key, shape, dtype = NULL, out_sharding = NULL) {
  jnp_zeros(canon_shape(shape), nni_dtype(dtype))
}

#' @rdname nn_initializers
#' @export
nn_initializers_ones <- function(key, shape, dtype = NULL, out_sharding = NULL) {
  jnp_ones(canon_shape(shape), nni_dtype(dtype))
}

#' @rdname nn_initializers
#' @export
nn_initializers_constant <- function(value, dtype = NULL) {
  force(value)
  default_dtype <- dtype
  function(key, shape, dtype = default_dtype, out_sharding = NULL) {
    jnp_full(canon_shape(shape), value, dtype = nni_dtype(dtype))
  }
}

#' @rdname nn_initializers
#' @export
nn_initializers_uniform <- function(scale = 1e-2, dtype = NULL) {
  force(scale)
  default_dtype <- dtype
  function(key, shape, dtype = default_dtype, out_sharding = NULL) {
    dt <- nni_dtype(dtype)
    random_uniform(key, shape, dt, out_sharding = out_sharding) * nni_array(scale, dt)
  }
}

# Port of initializers._scaled_normal: sqrt(2) * scale * erf_inv(u), u uniform on
# (nextafter(-1, 0), 1), as random.normal draws but with the scale folded in first.
nni_scaled_normal <- function(key, shape, dt, scale) {
  if (is_float_dtype(dt)) {
    lo <- nni_array(-1 + 2^-(finfo(dt)$nmant + 1), dt)
    hi <- nni_array(1, dt)
    u <- random_uniform(key, shape, dt, lo, hi)
    return((nni_array(sqrt(2), dt) * scale) * lax_erf_inv(u))
  }
  random_normal(key, shape, dt) * scale
}

#' @rdname nn_initializers
#' @export
nn_initializers_normal <- function(stddev = 1e-2, dtype = NULL) {
  force(stddev)
  default_dtype <- dtype
  function(key, shape, dtype = default_dtype, out_sharding = NULL) {
    dt <- nni_dtype(dtype)
    nni_scaled_normal(key, shape, dt, nni_array(stddev, dt))
  }
}

#' @rdname nn_initializers
#' @export
nn_initializers_truncated_normal <- function(stddev = 1e-2, dtype = NULL, lower = -2.0,
                                             upper = 2.0) {
  force(stddev); force(lower); force(upper)
  default_dtype <- dtype
  function(key, shape, dtype = default_dtype, out_sharding = NULL) {
    dt <- nni_dtype(dtype)
    random_truncated_normal(key, lower, upper, shape, dt, out_sharding = out_sharding) *
      nni_array(stddev, dt)
  }
}

# Port of initializers._compute_fans.
nni_compute_fans <- function(shape, in_axis = -2L, out_axis = -1L, batch_axis = integer(0)) {
  nd <- length(shape)
  if (length(in_axis) == 1L && in_axis == -2L && nd <= 1L)
    jax_error("Can't compute input and output sizes of a ", nd, "-dimensional weights tensor ",
              "with default in_axis. Must be at least 2D or specify in_axis explicitly.",
              class = "jax_value_error")
  size <- function(axes) if (length(axes)) prod(as.numeric(shape[norm_axis(axes, nd) + 1L])) else 1
  in_size <- size(in_axis)
  out_size <- size(out_axis)
  batch_size <- size(batch_axis)
  denominator <- in_size * out_size * batch_size
  receptive_field_size <- if (denominator != 0) prod(as.numeric(shape)) / denominator else 0
  c(in_size * receptive_field_size, out_size * receptive_field_size)
}

# Split key, real dtype and the uniform draws of _complex_uniform and
# _complex_truncated_normal.
nni_complex_parts <- function(key, shape, dt) {
  ks <- nni_split(key)
  real_dtype <- complex_component(dt)
  dt <- if (real_dtype == "float64") "complex128" else "complex64"
  list(ks = ks, real = real_dtype, dt = dt,
       theta = 2 * pi * jnp_astype(random_uniform(ks[[2L]], shape, real_dtype), dt))
}

# Port of initializers._complex_uniform: uniform on the unit variance disk.
nni_complex_uniform <- function(key, shape, dt) {
  p <- nni_complex_parts(key, shape, dt)
  r <- jnp_astype(jnp_sqrt(2L * random_uniform(p$ks[[1L]], shape, p$real)), p$dt)
  r * jnp_exp(1i * p$theta)
}

# Port of initializers._complex_truncated_normal: complex normal with modulus truncated
# at `upper`.
nni_complex_truncated_normal <- function(key, upper, shape, dt) {
  p <- nni_complex_parts(key, shape, dt)
  t <- (1L - jnp_exp(nni_array(-(upper^2), p$dt))) *
    jnp_astype(random_uniform(p$ks[[1L]], shape, p$real), p$dt)
  r <- jnp_sqrt(-jnp_log(1L - t))
  r * jnp_exp(1i * p$theta)
}

#' @rdname nn_initializers
#' @export
nn_initializers_variance_scaling <- function(scale, mode, distribution, in_axis = -2L,
                                             out_axis = -1L, batch_axis = integer(0),
                                             dtype = NULL) {
  force(scale); force(mode); force(distribution); force(in_axis); force(out_axis)
  force(batch_axis)
  default_dtype <- dtype
  function(key, shape, dtype = default_dtype, out_sharding = NULL) {
    shape <- canon_shape(shape)
    dt <- nni_dtype(dtype)
    fans <- nni_compute_fans(shape, in_axis, out_axis, batch_axis)
    denominator <- switch(mode,
      fan_in = fans[1L],
      fan_out = fans[2L],
      fan_avg = (fans[1L] + fans[2L]) / 2,
      fan_geo_avg = (fans[1L] * fans[2L])^0.5,
      jax_error("invalid mode for variance scaling initializer: ", mode,
                class = "jax_value_error"))
    variance <- nni_array(if (denominator != 0) scale / denominator else 0, dt)
    floating <- is_float_dtype(dt)
    if (identical(distribution, "truncated_normal")) {
      if (floating) {
        # 0.87962566103423978: stddev of a standard normal truncated to (-2, 2).
        stddev <- jnp_sqrt(variance) / nni_array(0x1.c25e4b6dba981p-1, dt)
        return(random_truncated_normal(key, -2L, 2L, shape, dt, out_sharding = out_sharding) *
                 stddev)
      }
      # 0.95311164380491208: stddev of a complex standard normal truncated to 2.
      stddev <- jnp_sqrt(variance) / nni_array(0x1.e7fe3fd728656p-1, dt)
      return(nni_complex_truncated_normal(key, 2L, shape, dt) * stddev)
    }
    if (identical(distribution, "normal"))
      return(nni_scaled_normal(key, shape, dt, jnp_sqrt(variance)))
    if (identical(distribution, "uniform")) {
      if (floating)
        return(random_uniform(key, shape, dt, -1L, out_sharding = out_sharding) *
                 jnp_sqrt(3L * variance))
      return(nni_complex_uniform(key, shape, dt) * jnp_sqrt(variance))
    }
    jax_error("invalid distribution for variance scaling initializer: ", distribution,
              class = "jax_value_error")
  }
}

#' @rdname nn_initializers
#' @export
nn_initializers_glorot_uniform <- function(in_axis = -2L, out_axis = -1L,
                                           batch_axis = integer(0), dtype = NULL) {
  nn_initializers_variance_scaling(1.0, "fan_avg", "uniform", in_axis, out_axis, batch_axis,
                                   dtype)
}

#' @rdname nn_initializers
#' @export
nn_initializers_xavier_uniform <- nn_initializers_glorot_uniform

#' @rdname nn_initializers
#' @export
nn_initializers_glorot_normal <- function(in_axis = -2L, out_axis = -1L,
                                          batch_axis = integer(0), dtype = NULL) {
  nn_initializers_variance_scaling(1.0, "fan_avg", "truncated_normal", in_axis, out_axis,
                                   batch_axis, dtype)
}

#' @rdname nn_initializers
#' @export
nn_initializers_xavier_normal <- nn_initializers_glorot_normal

#' @rdname nn_initializers
#' @export
nn_initializers_lecun_uniform <- function(in_axis = -2L, out_axis = -1L,
                                          batch_axis = integer(0), dtype = NULL) {
  nn_initializers_variance_scaling(1.0, "fan_in", "uniform", in_axis, out_axis, batch_axis,
                                   dtype)
}

#' @rdname nn_initializers
#' @export
nn_initializers_lecun_normal <- function(in_axis = -2L, out_axis = -1L,
                                         batch_axis = integer(0), dtype = NULL) {
  nn_initializers_variance_scaling(1.0, "fan_in", "truncated_normal", in_axis, out_axis,
                                   batch_axis, dtype)
}

#' @rdname nn_initializers
#' @export
nn_initializers_he_uniform <- function(in_axis = -2L, out_axis = -1L,
                                       batch_axis = integer(0), dtype = NULL) {
  nn_initializers_variance_scaling(2.0, "fan_in", "uniform", in_axis, out_axis, batch_axis,
                                   dtype)
}

#' @rdname nn_initializers
#' @export
nn_initializers_kaiming_uniform <- nn_initializers_he_uniform

#' @rdname nn_initializers
#' @export
nn_initializers_he_normal <- function(in_axis = -2L, out_axis = -1L,
                                      batch_axis = integer(0), dtype = NULL) {
  nn_initializers_variance_scaling(2.0, "fan_in", "truncated_normal", in_axis, out_axis,
                                   batch_axis, dtype)
}

#' @rdname nn_initializers
#' @export
nn_initializers_kaiming_normal <- nn_initializers_he_normal

#' @rdname nn_initializers
#' @export
nn_initializers_orthogonal <- function(scale = 1.0, column_axis = -1L, dtype = NULL) {
  force(scale); force(column_axis)
  default_dtype <- dtype
  function(key, shape, dtype = default_dtype, out_sharding = NULL) {
    if (!is.null(out_sharding)) jax_error("orthogonal initializer does not support out_sharding",
                                          class = "jax_not_implemented_error")
    dt <- nni_dtype(dtype)
    shape <- canon_shape(shape)
    if (length(shape) < 2L)
      jax_error("orthogonal initializer requires at least a 2D shape", class = "jax_value_error")
    if (any(shape == 0L)) return(jnp_zeros(shape, dt))
    ca <- norm_axis(column_axis, length(shape))
    n_cols <- shape[ca + 1L]
    n_rows <- prod(shape) %/% n_cols
    q <- random_orthogonal(key, n_rows, integer(0), dt, n_cols)
    q <- jnp_reshape(q, c(shape[-(ca + 1L)], n_cols))
    q <- jnp_moveaxis(q, -1L, column_axis)
    nni_array(scale, dt) * q
  }
}

#' @rdname nn_initializers
#' @export
nn_initializers_delta_orthogonal <- function(scale = 1.0, column_axis = -1L, dtype = NULL) {
  force(scale); force(column_axis)
  default_dtype <- dtype
  function(key, shape, dtype = default_dtype, out_sharding = NULL) {
    if (!is.null(out_sharding)) jax_error("delta_orthogonal initializer does not support ",
                                          "out_sharding", class = "jax_not_implemented_error")
    dt <- nni_dtype(dtype)
    shape <- canon_shape(shape)
    nd <- length(shape)
    if (!nd %in% 3:5)
      jax_error("Delta orthogonal initializer requires a 3D, 4D or 5D shape.",
                class = "jax_value_error")
    if (shape[nd] < shape[nd - 1L])
      jax_error("`fan_in` must be less or equal than `fan_out`. ", class = "jax_value_error")
    ortho_init <- nn_initializers_orthogonal(scale = scale, column_axis = column_axis, dtype = dt)
    ortho_matrix <- ortho_init(key, shape[nd - 1:0])
    w <- jnp_zeros(shape, dt)
    # W.at[(k1 - 1) // 2, ..., :, :].set(ortho_matrix) as a dynamic_update_slice.
    centers <- (shape[seq_len(nd - 2L)] - 1L) %/% 2L
    update <- jnp_reshape(ortho_matrix, c(rep(1L, nd - 2L), shape[nd - 1:0]))
    lax_dynamic_update_slice(w, update, c(centers, 0L, 0L))
  }
}
