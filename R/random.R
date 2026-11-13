# jax.random: port of jax/_src/random/core.py. The samplers validate their arguments
# and call jitted implementation functions, as in JAX (the jit caches are also keyed on
# jax_threefry_partitionable, see prng_jit()).

# ---- utilities (core.py) ----------------------------------------------------------------

rnd_shape <- function(x) if (is_jax_value(x)) aval_of(x)$shape else r_shape(x)

rnd_ndim <- function(x) length(rnd_shape(x))

# Port of core.py:_check_shape.
rnd_check_shape <- function(name, shape, ...) {
  ps <- list(...)
  if (!length(ps)) return(invisible())
  out <- broadcast_shapes_impl(c(list(shape), ps))
  if (!identical(as.integer(out), as.integer(shape)))
    jax_error(name, " parameter shapes must be broadcast-compatible with shape argument, and ",
              "the result of broadcasting the shapes must equal the shape argument, but got ",
              "result ", prng_shape_str(out), " for shape argument ", prng_shape_str(shape), ".",
              class = "jax_value_error")
  invisible()
}

# Port of core.py:_check_broadcast_shapes.
rnd_check_broadcast_shapes <- function(name, shape, ...) {
  shapes <- lapply(list(...), rnd_shape)
  if (is.null(shape)) {
    if (length(shapes)) return(broadcast_shapes_impl(shapes))
    return(integer(0))
  }
  shape <- canon_shape(shape)
  do.call(rnd_check_shape, c(list(name, shape), shapes))
  shape
}

# Port of core.py:_check_all_safe_to_cast (dtypes.safe_to_cast).
rnd_check_safe_cast <- function(name, dtype, ...) {
  for (a in list(...)) {
    if (result_type(a, dtype) != dtype)
      jax_error("In arguments to ", name, ", cannot safely cast argument of type ",
                aval_of(a)$dtype, " to ", dtype, class = "jax_type_promotion_error")
  }
}

# check_and_canonicalize_user_dtype(float if dtype is None else dtype) plus the float check.
rnd_float_dtype <- function(name, dtype, kind = "a float") {
  dt <- dtype(dtype %||% "float")
  ok <- if (kind == "a float or complex") is_inexact_dtype(dt) else is_float_dtype(dt)
  if (!ok)
    jax_error("dtype argument to `", name, "` must be ", kind, " dtype, got ", dt,
              class = "jax_value_error")
  dt
}

# Port of core.py:_check_prng_key: returns list(key = typed key, wrapped = was raw).
rnd_check_key <- function(name, key, allow_batched = FALSE) {
  if (is_prng_key(key)) {
    k <- key
    wrapped <- FALSE
  } else if (is_jax_value(key) || (is.atomic(key) && (is.numeric(key) || is.logical(key)))) {
    msg <- paste0("Legacy uint32 key array passed as key to jax.random function. Please create keys ",
                  "using jax.random.key(). If use of a raw key array was intended, set ",
                  "jax_legacy_prng_key=\"allow\".")
    mode <- .jx$config$jax_legacy_prng_key %||% "allow"
    if (identical(mode, "error")) jax_error(msg, class = "jax_value_error")
    if (identical(mode, "warn")) warning(msg, call. = FALSE)
    k <- prng_random_wrap(key, prng_resolve_impl(NULL))
    wrapped <- TRUE
  } else {
    jax_error("unexpected PRNG key type ", class(key)[1L], class = "jax_type_error")
  }
  if (!allow_batched && length(prng_keys_shape(k)))
    jax_error(name, " accepts a single key, but was given a key array of shape ",
              prng_shape_str(if (wrapped) rnd_shape(key) else prng_keys_shape(k)),
              " != (). Use jax.vmap for batching.", class = "jax_value_error")
  list(key = k, wrapped = wrapped)
}

# Port of core.py:_return_prng_keys (jax_enable_custom_prng is off).
rnd_return_keys <- function(wrapped, key) if (wrapped) key$data else key

# core.py:_split followed by unpacking along the first axis.
rnd_split <- function(key, num = 2L) as.list.jax_prng_key(prng_random_split(key, num))

rnd_bits <- function(key, bit_width, shape) prng_random_bits(key, bit_width, shape)

# jnp.clip(x, lo, hi) (lax_numpy.py:clip): maximum(lo, x), then minimum(hi, .).
rnd_clip <- function(x, lo = NULL, hi = NULL) {
  if (!is.null(lo)) x <- jnp_maximum(lo, x)
  if (!is.null(hi)) x <- jnp_minimum(hi, x)
  x
}

# Port of lax.broadcast_to_rank: leading unit dimensions up to rank `rank`.
rnd_broadcast_to_rank <- function(x, rank) {
  x <- as_jax(x)
  nd <- length(aval_of(x)$shape)
  if (nd == rank) return(x)
  lax_broadcast(x, rep(1L, rank - nd))
}

# Exact iinfo(dt) bounds as constants (int64 and uint64 bounds are not doubles).
rnd_iinfo_const <- function(dt, which = c("max", "min")) {
  which <- match.arg(which)
  if (which == "max" && dt == "int64")
    return(lax_bitwise_not(prng_full(-2^63, "int64")))
  if (which == "max" && dt == "uint64")
    return(lax_bitwise_not(prng_full(0, "uint64")))
  prng_full(iinfo(dt)[[which]], dt)
}

# Port of core.py:_convert_and_clip_integer. Clipping to the input dtype's own bound is
# the identity and is skipped; other bounds are exact constants of the target dtype.
rnd_convert_and_clip_integer <- function(val, dtype) {
  vdt <- aval_of(val)$dtype
  if (!(is_int_dtype(dtype) && is_int_dtype(vdt)))
    jax_error("_convert_and_clip_integer only accepts integer dtypes.", class = "jax_type_error")
  ti <- iinfo(dtype)
  vi <- iinfo(vdt)
  lo <- if (ti$min > vi$min) lax_convert_element_type(rnd_iinfo_const(dtype, "min"), vdt)
  hi <- if (ti$max < vi$max) lax_convert_element_type(rnd_iinfo_const(dtype, "max"), vdt)
  lax_convert_element_type(rnd_clip(val, lo, hi), dtype)
}

# An R whole number given for an integer parameter is a Python int (default int dtype).
rnd_int_arg <- function(x) {
  if (!is_jax_value(x) && is.numeric(x) && !anyNA(x) && all(x == round(x)))
    return(as_jax(x, default_int()))
  x
}

# finfo(dtype).minexp and the bit pattern of 1.0.
rnd_minexp <- function(fi) 2 - 2^(fi$nexp - 1)
rnd_one_bits <- function(fi) (2^(fi$nexp - 1) - 1) * 2^fi$nmant

# ---- keys -------------------------------------------------------------------------------

#' PRNG keys
#'
#' Ports of the key functions of `jax.random`. `random_key()` returns a typed key array
#' (a scalar array of keys whose dtype names the PRNG implementation, `key<fry>` for
#' threefry2x32); `random_PRNGKey()` returns a legacy raw key, a `uint32` array of
#' shape `2`. Both are accepted by all samplers. Typed key arrays are pytrees: they
#' pass through [jit()], [vmap()] and [grad()] (with float0 tangents), can be indexed
#' with `[` and split into a list with `as.list()`.
#'
#' The implementations are `"threefry2x32"` (the default, see the option
#' `jax_default_prng_impl`), `"rbg"` and `"unsafe_rbg"`. The option
#' `jax_threefry_partitionable` (default `TRUE`) selects the threefry split and bits
#' algorithms, as in JAX.
#' @param seed An integer scalar (an R whole number or an integer jax scalar). R numbers
#'   are treated as Python ints: they wrap to 32 bits unless x64 is enabled.
#' @param impl Optional PRNG implementation name.
#' @param dtype Optional key dtype (as returned by `random_key_dtype()`) or
#'   implementation name; an alternative to `impl`.
#' @param key,keys A key (typed or raw) or key array.
#' @param num Number of keys (or shape of the key array) to produce.
#' @param data A 32-bit integer scalar to fold into the key.
#' @param key_bits_array A `uint32` array whose trailing dimensions are the key shape of
#'   the implementation.
#' @param impl_spec Optional implementation name.
#' @param x A value (returned unchanged by `random_Key()`).
#' @return `random_key()`, `random_split()`, `random_fold_in()`, `random_clone()` and
#'   `random_wrap_key_data()` return keys (raw keys stay raw); `random_key_data()` the
#'   `uint32` key data; `random_key_impl()` the implementation name;
#'   `random_key_dtype()` the key dtype name.
#' @name random_key
#' @export
random_key <- function(seed, impl = NULL, dtype = NULL) {
  if (!is.null(dtype)) {
    if (!is.null(impl))
      jax_error("Cannot specify both `impl` and `dtype` arguments to jax.random.key",
                class = "jax_value_error")
    impl <- dtype
  }
  rnd_key("key", seed, impl)
}

# Port of core.py:_key.
rnd_key <- function(ctor_name, seed, impl_spec) {
  impl <- prng_resolve_impl(impl_spec)
  if (is_prng_key(seed))
    jax_error(ctor_name, " accepts a scalar seed, but was given a PRNG key.", class = "jax_type_error")
  if (length(rnd_shape(seed)))
    jax_error(ctor_name, " accepts a scalar seed, but was given an array of shape ",
              prng_shape_str(rnd_shape(seed)), " != (). Use jax.vmap for batching",
              class = "jax_type_error")
  prng_random_seed(seed, impl)
}

#' @rdname random_key
#' @export
random_PRNGKey <- function(seed, impl = NULL) rnd_return_keys(TRUE, rnd_key("PRNGKey", seed, impl))

#' @rdname random_key
#' @export
random_fold_in <- function(key, data) {
  ck <- rnd_check_key("fold_in", key)
  if (length(rnd_shape(data)))
    jax_error("fold_in accepts a scalar, but was given an array of shape ",
              prng_shape_str(rnd_shape(data)), " != (). Use jax.vmap for batching.",
              class = "jax_type_error")
  if (!is_jax_value(data) && is.numeric(data) && !is.na(data) && data == round(data)) {
    if (data < 0 || data >= 2^32)
      jax_error("Python integer ", data, " out of bounds for uint32", class = "jax_overflow_error")
    data <- prng_u32(data)
  }
  rnd_return_keys(ck$wrapped, prng_random_fold_in(ck$key, as_jax(data, "uint32")))
}

#' @rdname random_key
#' @export
random_split <- function(key, num = 2L) {
  ck <- rnd_check_key("split", key)
  rnd_return_keys(ck$wrapped, prng_random_split(ck$key, canon_shape(num)))
}

#' @rdname random_key
#' @export
random_key_impl <- function(keys) prng_key_impl(rnd_check_key("key_impl", keys, TRUE)$key)$name

#' @rdname random_key
#' @export
random_key_data <- function(keys) rnd_check_key("key_data", keys, TRUE)$key$data

#' @rdname random_key
#' @export
random_wrap_key_data <- function(key_bits_array, impl = NULL, dtype = NULL) {
  if (!is.null(dtype)) {
    if (!is.null(impl))
      jax_error("Cannot specify both `impl` and `dtype` arguments to jax.random.wrap_key_data",
                class = "jax_value_error")
    impl <- dtype
  }
  prng_random_wrap(key_bits_array, prng_resolve_impl(impl))
}

#' @rdname random_key
#' @export
random_key_dtype <- function(impl_spec = NULL) paste0("key<", prng_resolve_impl(impl_spec)$tag, ">")

#' @rdname random_key
#' @export
random_clone <- function(key) {
  if (!is_prng_key(key)) return(key)
  prng_key_array(kr_copy(key$data), attr(key, "impl", exact = TRUE))
}

#' @rdname random_key
#' @export
random_Key <- function(x) x

# ---- bits, uniform, randint -------------------------------------------------------------

#' Uniform random bits, floats and integers
#'
#' Ports of `jax.random.bits`, `jax.random.uniform` and `jax.random.randint`. For a given
#' key the results equal those of JAX bit for bit.
#' @param key A PRNG key from [random_key()] or [random_PRNGKey()].
#' @param shape Integer vector, the result shape.
#' @param dtype Result dtype (`bits`: an unsigned integer dtype, default `uint32`, or
#'   `uint64` with x64; `uniform`: a float dtype; `randint`: an integer dtype).
#' @param minval,maxval Bounds of the range `[minval, maxval)`, broadcast-compatible with
#'   `shape`.
#' @param out_sharding Ignored (one device).
#' @return An array of shape `shape`.
#' @name random_uniform
#' @export
random_bits <- function(key, shape = integer(0), dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("bits", key)$key
  dt <- if (is.null(dtype)) default_uint() else dtype(dtype)
  if (!is_unsigned_dtype(dt))
    jax_error("dtype argument to `bits` must be an unsigned int dtype, got ", dt,
              class = "jax_value_error")
  rnd_bits(key, dtype_bits(dt), canon_shape(shape))
}

#' @rdname random_uniform
#' @export
random_uniform <- function(key, shape = integer(0), dtype = NULL, minval = 0, maxval = 1,
                           out_sharding = NULL) {
  key <- rnd_check_key("uniform", key)$key
  dt <- rnd_float_dtype("uniform", dtype)
  rnd_uniform(key, minval, maxval, canon_shape(shape), dt)
}

# Port of core.py:_uniform.
rnd_uniform <- prng_jit(function(key, minval, maxval, shape, dtype) {
  if (!is_float_dtype(dtype))
    jax_error("uniform only accepts floating point dtypes.", class = "jax_type_error")
  minval <- rnd_broadcast_to_rank(lax_convert_element_type(minval, dtype), length(shape))
  maxval <- rnd_broadcast_to_rank(lax_convert_element_type(maxval, dtype), length(shape))
  fi <- finfo(dtype)
  nbits <- fi$bits
  nmant <- fi$nmant
  rng_bits <- if (nmant < 8) 8L else nbits
  bits <- rnd_bits(key, rng_bits, shape)
  uint_dtype <- prng_uint_dtypes[[as.character(nbits)]]
  if (rng_bits != nbits) bits <- lax_convert_element_type(bits, uint_dtype)
  float_bits <- lax_shift_right_logical(bits, prng_full(rng_bits - nmant, uint_dtype))
  float_bits <- lax_bitwise_or(float_bits, prng_full(rnd_one_bits(fi), uint_dtype))
  floats <- lax_bitcast_convert_type(float_bits, dtype) - prng_full(1, dtype)
  lax_max(minval, lax_reshape(floats * (maxval - minval) + minval, shape))
}, static_argnums = 3:4)

#' @rdname random_uniform
#' @export
random_randint <- function(key, shape, minval, maxval, dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("randint", key)$key
  dt <- dtype(dtype %||% "int")
  shape <- canon_shape(shape)
  if (!is_int_dtype(dt))
    jax_error("randint only accepts integer dtypes, got ", dt, class = "jax_type_error")
  minval <- rnd_int_arg(minval)
  maxval <- rnd_int_arg(maxval)
  info <- iinfo(dt)
  dt_sampling <- dt
  if (info$bits < 32L) {
    # Sample in 32 bits to avoid biased results.
    dt_sampling <- "int32"
    minval <- rnd_clip(jnp_astype(minval, "int32"), as.integer(info$min), as.integer(info$max))
    maxval <- rnd_clip(jnp_astype(maxval, "int32"), as.integer(info$min), as.integer(info$max) + 1L)
  }
  jnp_astype(rnd_randint(key, minval, maxval, shape, dt_sampling), dt)
}

# Port of core.py:_randint.
rnd_randint <- prng_jit(function(key, minval, maxval, shape, dtype) {
  rnd_check_shape("randint", shape, rnd_shape(minval), rnd_shape(maxval))
  minval <- as_jax(minval)
  maxval <- as_jax(maxval)
  if (!is_int_dtype(aval_of(minval)$dtype)) minval <- jnp_astype(minval, "int")
  if (!is_int_dtype(aval_of(maxval)$dtype)) maxval <- jnp_astype(maxval, "int")
  # Flag where maxval is greater than the maximum value of dtype, to handle cases like
  # randint(key, shape, 0, 256, 'uint8').
  maxval_out_of_range <- lax_gt(
    maxval, rnd_convert_and_clip_integer(rnd_iinfo_const(dtype, "max"), aval_of(maxval)$dtype))
  minval <- rnd_broadcast_to_rank(rnd_convert_and_clip_integer(minval, dtype), length(shape))
  maxval <- rnd_broadcast_to_rank(rnd_convert_and_clip_integer(maxval, dtype), length(shape))
  nbits <- iinfo(dtype)$bits
  if (!nbits %in% c(8L, 16L, 32L, 64L))
    jax_error("randint only accepts 8-, 16-, 32-, or 64-bit dtypes, got ", dtype,
              class = "jax_type_error")
  ks <- rnd_split(key)
  higher_bits <- rnd_bits(ks[[1L]], nbits, shape)
  lower_bits <- rnd_bits(ks[[2L]], nbits, shape)
  unsigned_dtype <- prng_uint_dtypes[[as.character(nbits)]]
  span <- lax_convert_element_type(maxval - minval, unsigned_dtype)
  # Ensure that span=1 when maxval <= minval, so minval is always returned.
  span <- lax_select(maxval <= minval, lax_full_like(span, 1), span)
  # When maxval is out of range, the span has to be one larger.
  span <- lax_select(maxval_out_of_range & (maxval > minval), lax_add(span, prng_const(span, 1)), span)
  # (a * b) % N = [(a % N) * (b % N)] % N, with multiplier = 2**nbits % span.
  multiplier <- lax_rem(prng_const(span, 2^(nbits %/% 2L)), span)
  multiplier <- lax_rem(lax_mul(multiplier, multiplier), span)
  random_offset <- lax_add(lax_mul(lax_rem(higher_bits, span), multiplier), lax_rem(lower_bits, span))
  random_offset <- lax_rem(random_offset, span)
  lax_add(minval, lax_convert_element_type(random_offset, dtype))
}, static_argnums = 3:4)

# ---- permutation and choice -------------------------------------------------------------

#' Random permutations and choices
#'
#' Ports of `jax.random.permutation` and `jax.random.choice`. Axes are 0-based.
#' @inheritParams random_uniform
#' @param x An integer `n` (permute `0:(n - 1)`) or an array.
#' @param axis Axis along which to shuffle or select.
#' @param independent Whether each slice along `axis` is shuffled independently.
#' @param a An integer `n` (choose from `0:(n - 1)`) or an array.
#' @param replace Whether to sample with replacement.
#' @param p Optional probabilities of the entries of `a` along `axis`.
#' @param mode `"low"` or `"high"`: bits used by the Gumbel sampler (`p` given,
#'   `replace = FALSE`).
#' @return An array.
#' @name random_permutation
#' @export
random_permutation <- function(key, x, axis = 0L, independent = FALSE, out_sharding = NULL) {
  key <- rnd_check_key("permutation", key)$key
  axis <- norm_axis(axis, max(rnd_ndim(x), 1L))
  if (!rnd_ndim(x)) {
    if (is_jax_value(x)) {
      if (!is_int_dtype(aval_of(x)$dtype))
        jax_error("x must be an integer or at least 1-dimensional", class = "jax_type_error")
      x <- as.numeric(concrete_or_error(x, "argument x of jax.random.permutation()"))
    } else if (!is.numeric(x) || x != round(x)) {
      jax_error("x must be an integer or at least 1-dimensional", class = "jax_type_error")
    }
    return(rnd_shuffle(key, jnp_arange(as.integer(x)), axis))
  }
  rnd_permutation(key, as_jax(x), axis, independent)
}

# Port of core.py:_permutation.
rnd_permutation <- function(key, x, axis, independent) {
  if (independent || rnd_ndim(x) == 1L) return(rnd_shuffle(key, x, axis))
  ind <- rnd_shuffle(key, jnp_arange(aval_of(x)$shape[axis + 1L]), 0L)
  jnp_take(x, ind, axis, unique_indices = TRUE)
}

# Port of core.py:_shuffle: repeated sorts by fresh 32 bit keys.
rnd_shuffle <- prng_jit(function(key, x, axis) {
  exponent <- 3
  uint32max <- 2^32 - 1
  num_rounds <- as.integer(ceiling(exponent * log(max(1, length(x))) / log(uint32max)))
  for (i in seq_len(num_rounds)) {
    ks <- rnd_split(key)
    key <- ks[[1L]]
    sort_keys <- rnd_bits(ks[[2L]], 32L, aval_of(x)$shape)
    x <- lax_sort_key_val(sort_keys, x, axis)[[2L]]
  }
  x
}, static_argnums = 2L)

#' @rdname random_permutation
#' @export
random_choice <- function(key, a, shape = integer(0), replace = TRUE, p = NULL, axis = 0L,
                          mode = NULL) {
  key <- rnd_check_key("choice", key)$key
  shape <- canon_shape(shape)
  arr <- if (!is_jax_value(a) && length(a) == 1L) as_jax(rnd_int_arg(a)) else as_jax(a)
  ad <- aval_of(arr)
  if (!length(ad$shape)) {
    n_inputs <- as.integer(as.numeric(concrete_or_error(arr, "The error occurred in jax.random.choice()")))
  } else {
    axis <- norm_axis(axis, length(ad$shape))
    n_inputs <- ad$shape[axis + 1L]
  }
  n_draws <- prod_int(shape)
  if (n_draws == 0L) return(jnp_zeros(shape, dtype = ad$dtype))
  if (n_inputs <= 0L)
    jax_error("a must be greater than 0 unless no samples are taken", class = "jax_value_error")
  if (!replace && n_draws > n_inputs)
    jax_error("Cannot take a larger sample (size ", n_draws, ") than population (size ",
              n_inputs, ") when 'replace=False'", class = "jax_value_error")
  if (is.null(p)) {
    if (replace) {
      ind <- random_randint(key, shape, 0L, n_inputs)
      result <- if (!length(ad$shape)) ind else jnp_take(arr, ind, axis)
    } else {
      perm <- random_permutation(key, if (!length(ad$shape)) n_inputs else arr, axis)
      result <- lax_slice_in_dim(perm, 0L, n_draws, 1L, if (!length(ad$shape)) 0L else axis)
    }
  } else {
    p_arr <- promote_dtypes_inexact(list(p))[[1L]]
    if (!identical(aval_of(p_arr)$shape, n_inputs))
      jax_error("p must be None or a 1D vector with the same size as a.shape[axis]. p has shape ",
                prng_shape_str(aval_of(p_arr)$shape), " and a.shape[axis] is ", n_inputs, ".",
                class = "jax_value_error")
    if (replace) {
      p_cuml <- jnp_cumsum(p_arr)
      r <- lax_index_in_dim(p_cuml, -1L, 0L, keepdims = FALSE) *
        (1 - random_uniform(key, shape, dtype = aval_of(p_cuml)$dtype))
      ind <- jnp_astype(jnp_searchsorted(p_cuml, r), "int")
    } else {
      g <- random_gumbel(key, n_inputs, dtype = aval_of(p_arr)$dtype, mode = mode) + jnp_log(p_arr)
      ind <- jnp_astype(lax_top_k(g, k = n_draws)[[2L]], "int")
    }
    result <- if (!length(ad$shape)) ind else jnp_take(arr, ind, axis)
  }
  out_shape <- if (!length(ad$shape)) shape
               else c(ad$shape[seq_len(axis)], shape, drop_first(ad$shape, axis + 1L))
  lax_reshape(result, out_shape)
}

# ---- normal family ----------------------------------------------------------------------

#' Normal random values
#'
#' Ports of `jax.random.normal`, `jax.random.multivariate_normal`,
#' `jax.random.truncated_normal` and `jax.random.lognormal`.
#' @inheritParams random_uniform
#' @param dtype A float dtype (`normal` also accepts complex dtypes).
#' @param mean Mean vector of shape `(..., n)`.
#' @param cov Positive definite covariance of shape `(..., n, n)`.
#' @param method `"cholesky"`, `"svd"` or `"eigh"`: factorization of `cov`.
#' @param lower,upper Truncation bounds, broadcast-compatible with each other.
#' @param sigma Standard deviation of the underlying normal distribution.
#' @return An array.
#' @name random_normal
#' @export
random_normal <- function(key, shape = integer(0), dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("normal", key)$key
  dt <- rnd_float_dtype("normal", dtype, kind = "a float or complex")
  rnd_normal(key, canon_shape(shape), dt)
}

# Port of core.py:_normal.
rnd_normal <- prng_jit(function(key, shape, dtype) {
  if (is_complex_dtype(dtype)) {
    sqrt2 <- prng_full(complex(real = sqrt(2)), dtype)
    ks <- rnd_split(key)
    real_dtype <- complex_component(dtype)
    re <- jnp_astype(rnd_normal_real(ks[[1L]], shape, real_dtype), dtype)
    im <- jnp_astype(rnd_normal_real(ks[[2L]], shape, real_dtype), dtype)
    return((re + 1i * im) / sqrt2)
  }
  rnd_normal_real(key, shape, dtype)
}, static_argnums = 1:2)

# Port of core.py:_normal_real.
rnd_normal_real <- prng_jit(function(key, shape, dtype) {
  fi <- finfo(dtype)
  lo <- prng_full(-1 + 2^-(fi$nmant + 1), dtype)  # nextafter(-1, 0)
  hi <- prng_full(1, dtype)
  u <- random_uniform(key, shape, dtype, lo, hi)
  lax_mul(prng_full(sqrt(2), dtype), lax_erf_inv(u))
}, static_argnums = 1:2)

#' @rdname random_normal
#' @export
random_multivariate_normal <- function(key, mean, cov, shape = NULL, dtype = NULL,
                                       method = "cholesky", out_sharding = NULL) {
  key <- rnd_check_key("multivariate_normal", key)$key
  mc <- promote_dtypes_inexact(list(mean, cov))
  if (!method %in% c("svd", "eigh", "cholesky"))
    jax_error("method must be one of {'svd', 'eigh', 'cholesky'}", class = "jax_value_error")
  dt <- if (is.null(dtype)) aval_of(mc[[1L]])$dtype else dtype(dtype)
  if (!is_float_dtype(dt))
    jax_error("dtype argument to `multivariate_normal` must be a float dtype, got ", dt,
              class = "jax_value_error")
  if (!is.null(shape)) shape <- canon_shape(shape)
  rnd_multivariate_normal(key, mc[[1L]], mc[[2L]], shape, dt, method)
}

# jnp.einsum('...ij,...j->...i', factor, z) as a dot_general.
rnd_matvec <- function(factor, z) {
  fs <- aval_of(factor)$shape
  zs <- aval_of(z)$shape
  if (length(fs) == 2L) {
    if (length(zs) == 1L) return(lax_dot_general(factor, z, list(list(1L, 0L), list(integer(0), integer(0)))))
    return(lax_dot_general(z, factor, list(list(length(zs) - 1L, 1L), list(integer(0), integer(0)))))
  }
  batch <- broadcast_shapes_impl(list(utils::head(fs, -2L), utils::head(zs, -1L)))
  n <- length(batch)
  factor <- jnp_broadcast_to(factor, c(batch, utils::tail(fs, 2L)))
  z <- jnp_broadcast_to(z, c(batch, utils::tail(zs, 1L)))
  b <- seq_len(n) - 1L
  lax_dot_general(z, factor, list(list(n, n + 1L), list(b, b)))
}

# Port of core.py:_multivariate_normal.
rnd_multivariate_normal <- prng_jit(function(key, mean, cov, shape, dtype, method) {
  if (rnd_ndim(mean) < 1L)
    jax_error("multivariate_normal requires mean.ndim >= 1, got mean.ndim == ", rnd_ndim(mean),
              class = "jax_value_error")
  if (rnd_ndim(cov) < 2L)
    jax_error("multivariate_normal requires cov.ndim >= 2, got cov.ndim == ", rnd_ndim(cov),
              class = "jax_value_error")
  ms <- aval_of(mean)$shape
  cs <- aval_of(cov)$shape
  n <- utils::tail(ms, 1L)
  if (!identical(utils::tail(cs, 2L), c(n, n)))
    jax_error("multivariate_normal requires cov.shape == (..., n, n) for n=", n,
              ", but got cov.shape == ", prng_shape_str(cs), ".", class = "jax_value_error")
  if (is.null(shape)) {
    shape <- broadcast_shapes_impl(list(utils::head(ms, -1L), utils::head(cs, -2L)))
  } else {
    rnd_check_shape("normal", shape, utils::head(ms, -1L), utils::head(cs, -2L))
  }
  factor <- switch(method,
    svd = {
      r <- jnp_linalg_svd(cov)
      r[[1L]] * jnp_sqrt(lax_expand_dims(r[[2L]], rnd_ndim(r[[2L]]) - 1L))
    },
    eigh = {
      r <- jnp_linalg_eigh(cov)
      r[[2L]] * jnp_sqrt(lax_expand_dims(r[[1L]], rnd_ndim(r[[1L]]) - 1L))
    },
    jnp_linalg_cholesky(cov))
  normal_samples <- random_normal(key, c(shape, n), dtype)
  mean + rnd_matvec(factor, normal_samples)
}, static_argnums = 3:5)

#' @rdname random_normal
#' @export
random_truncated_normal <- function(key, lower, upper, shape = NULL, dtype = NULL,
                                    out_sharding = NULL) {
  if (!is.null(shape)) shape <- canon_shape(shape)
  key <- rnd_check_key("truncated_normal", key)$key
  dt <- rnd_float_dtype("truncated_normal", dtype)
  rnd_truncated_normal(key, lower, upper, shape, dt)
}

# Port of core.py:_truncated_normal.
rnd_truncated_normal <- prng_jit(function(key, lower, upper, shape, dtype) {
  if (is.null(shape)) {
    shape <- broadcast_shapes_impl(list(rnd_shape(lower), rnd_shape(upper)))
  } else {
    rnd_check_shape("truncated_normal", shape, rnd_shape(lower), rnd_shape(upper))
  }
  sqrt2 <- prng_full(sqrt(2), dtype)
  lower <- lax_convert_element_type(lower, dtype)
  upper <- lax_convert_element_type(upper, dtype)
  a <- lax_erf(lower / sqrt2)
  b <- lax_erf(upper / sqrt2)
  u <- random_uniform(key, shape, dtype, minval = a, maxval = b)
  out <- sqrt2 * lax_erf_inv(u)
  # Clamp to the open interval (lower, upper).
  rnd_clip(out,
           lax_nextafter(lax_stop_gradient(lower), prng_full(Inf, dtype)),
           lax_nextafter(lax_stop_gradient(upper), prng_full(-Inf, dtype)))
}, static_argnums = 3:4)

#' @rdname random_normal
#' @export
random_lognormal <- function(key, sigma = 1, shape = NULL, dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("lognormal", key)$key
  dt <- rnd_float_dtype("lognormal", dtype, kind = "a float or complex")
  shape <- rnd_check_broadcast_shapes("lognormal", shape, sigma)
  rnd_check_safe_cast("lognormal", dt, sigma)
  rnd_lognormal(key, sigma, shape, dt)
}

# Port of core.py:_lognormal.
rnd_lognormal <- prng_jit(function(key, sigma, shape, dtype) {
  sigma <- lax_convert_element_type(sigma, dtype)
  lax_exp(random_normal(key, shape, dtype) * sigma)
}, static_argnums = 2:3)

# ---- bernoulli, rademacher, categorical, geometric -------------------------------------

#' Discrete random values
#'
#' Ports of `jax.random.bernoulli`, `rademacher`, `categorical`, `geometric`, `poisson`,
#' `binomial` and `multinomial`. Axes are 0-based.
#' @inheritParams random_uniform
#' @param p Probability (of success, or of each outcome for `multinomial`).
#' @param shape Integer vector; `NULL` (where allowed) broadcasts the parameters.
#' @param mode `"low"` or `"high"` (also `"highest"` for the Gumbel sampler of
#'   `categorical`): how many random bits to use.
#' @param dtype Result dtype.
#' @param logits Unnormalized log probabilities.
#' @param axis Axis of `logits` holding the categories.
#' @param replace Whether `categorical` samples with replacement.
#' @param lam Poisson rate.
#' @param method `"exact"` or `"approximate"`.
#' @param n Number of trials.
#' @param unroll Unroll parameter of the scan in `multinomial`.
#' @return An array.
#' @name random_discrete
#' @export
random_bernoulli <- function(key, p = 0.5, shape = NULL, mode = "low", out_sharding = NULL) {
  if (!is.null(shape)) shape <- canon_shape(shape)
  if (!mode %in% c("high", "low"))
    jax_error("got mode='", mode, "', expected 'high' or 'low'", class = "jax_value_error")
  key <- rnd_check_key("bernoulli", key)$key
  dt <- aval_of(p)$dtype
  if (!is_float_dtype(dt))
    jax_error("bernoulli probability `p` must have a floating dtype, got ", dt, ".",
              class = "jax_type_error")
  rnd_bernoulli(key, lax_convert_element_type(p, dt), shape, mode)
}

# Port of core.py:_bernoulli.
rnd_bernoulli <- prng_jit(function(key, p, shape, mode) {
  if (is.null(shape)) shape <- rnd_shape(p) else rnd_check_shape("bernoulli", shape, rnd_shape(p))
  dt <- aval_of(p)$dtype
  if (mode == "high") {
    u <- random_uniform(key, c(2L, shape), dt)
    u1 <- lax_index_in_dim(u, 0L, 0L, keepdims = FALSE)
    u2 <- lax_index_in_dim(u, 1L, 0L, keepdims = FALSE) * 2^-finfo(dt)$nmant
    return(u2 < p - u1)
  }
  random_uniform(key, shape, dt) < p
}, static_argnums = 2:3)

#' @rdname random_discrete
#' @export
random_rademacher <- function(key, shape = integer(0), dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("rademacher", key)$key
  rnd_rademacher(key, canon_shape(shape), dtype(dtype %||% "int"))
}

# Port of core.py:_rademacher.
rnd_rademacher <- prng_jit(function(key, shape, dtype) {
  b <- jnp_astype(random_bernoulli(key, 0.5, shape), dtype)
  jnp_astype(2L * b - 1L, dtype)
}, static_argnums = 1:2)

#' @rdname random_discrete
#' @export
random_categorical <- function(key, logits, axis = -1L, shape = NULL, replace = TRUE, mode = NULL,
                               out_sharding = NULL) {
  key <- rnd_check_key("categorical", key)$key
  logits <- as_jax(logits)
  ls <- aval_of(logits)$shape
  batch_shape <- ls[-(norm_axis(axis, length(ls)) + 1L)]
  if (is.null(shape)) {
    shape <- batch_shape
  } else {
    shape <- canon_shape(shape)
    rnd_check_shape("categorical", shape, batch_shape)
  }
  rnd_categorical(key, logits, shape, batch_shape, as.integer(axis), replace, mode)
}

# Port of core.py:_categorical.
rnd_categorical <- function(key, logits, shape, batch_shape, axis, replace, mode) {
  ls <- aval_of(logits)$shape
  nd <- length(ls)
  shape_prefix <- shape[seq_len(length(shape) - length(batch_shape))]
  if (replace) {
    if (axis >= 0L) axis <- axis - nd
    logits_shape <- utils::tail(shape, length(batch_shape))
    pos <- axis %% nd
    logits_shape <- append(logits_shape, ls[axis + nd + 1L], after = pos)
    g <- random_gumbel(key, c(shape_prefix, logits_shape), aval_of(logits)$dtype, mode = mode)
    lead <- seq_along(shape_prefix) - 1L
    return(jnp_argmax(g + (if (length(lead)) lax_expand_dims(logits, lead) else logits), axis = axis))
  }
  logits <- logits + random_gumbel(key, ls, aval_of(logits)$dtype, mode = mode)
  k <- prod_int(shape_prefix)
  ax <- norm_axis(axis, nd)
  if (k > ls[ax + 1L])
    jax_error("Number of samples without replacement (", k, ") cannot exceed number of ",
              "categories (", ls[ax + 1L], ").", class = "jax_value_error")
  indices <- lax_top_k(jnp_moveaxis(logits, ax, -1L), k)[[2L]]
  r <- length(aval_of(indices)$shape)
  lax_reshape(indices, shape, dimensions = c(r - 1L, seq_len(r - 1L) - 1L))
}

#' @rdname random_discrete
#' @export
random_geometric <- function(key, p, shape = NULL, dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("geometric", key)$key
  dt <- dtype(dtype %||% "int")
  if (!is_int_dtype(dt))
    jax_error("dtype argument to `geometric` must be an int dtype, got ", dt,
              class = "jax_value_error")
  shape <- rnd_check_broadcast_shapes("geometric", shape, p)
  rnd_geometric(key, p, shape, dt)
}

# Port of core.py:_geometric.
rnd_geometric <- prng_jit(function(key, p, shape, dtype) {
  p <- promote_dtypes_inexact(list(p))[[1L]]
  u <- random_uniform(key, shape, aval_of(p)$dtype)
  # u = 0 is mapped to 1 to avoid inf in log(u).
  u <- jnp_where(u == 0, 1, u)
  log_u <- lax_log(u)
  log_one_minus_p <- jnp_broadcast_to(lax_log1p(-p), shape)
  g <- lax_floor(lax_div(log_u, log_one_minus_p)) + 1
  jnp_astype(g, dtype)
}, static_argnums = 2:3)

# ---- poisson ----------------------------------------------------------------------------

#' @rdname random_discrete
#' @export
random_poisson <- function(key, lam, shape = NULL, dtype = NULL, method = "exact",
                           out_sharding = NULL) {
  key <- rnd_check_key("poisson", key)$key
  if (!method %in% c("exact", "approximate"))
    jax_error("method argument to `poisson` must be one of {'exact', 'approximate'}, got '",
              method, "'", class = "jax_value_error")
  dt <- dtype(dtype %||% "int")
  shape <- if (is.null(shape)) rnd_shape(lam) else canon_shape(shape)
  if (method == "approximate") {
    if (!identical(broadcast_shapes_impl(list(rnd_shape(lam), shape)), shape))
      jax_error("lam shape must be broadcastable to shape argument; got lam.shape ",
                prng_shape_str(rnd_shape(lam)), ", shape ", prng_shape_str(shape),
                class = "jax_value_error")
    return(rnd_poisson_approx(key, lam, shape, dt))
  }
  lam <- jnp_broadcast_to(lam, shape)
  if (attr(key, "impl", exact = TRUE) != "threefry2x32")
    jax_error("`poisson` with method='exact' is only implemented for the threefry2x32 RNG, not ",
              attr(key, "impl", exact = TRUE), class = "jax_not_implemented_error")
  rnd_poisson(key, lax_convert_element_type(lam, "float32"), shape, dt)
}

# Port of core.py:_poisson_knuth (Knuth's algorithm).
rnd_poisson_knuth <- function(key, lam, shape, dtype, max_iters) {
  body_fn <- function(carry) {
    ks <- rnd_split(carry[[3L]])
    k <- lax_select(carry[[4L]] > -lam, carry[[2L]] + 1L, carry[[2L]])
    u <- random_uniform(ks[[2L]], shape, "float32")
    list(carry[[1L]] + 1L, k, ks[[1L]], carry[[4L]] + jnp_log(u))
  }
  cond_fn <- function(carry) jnp_any(carry[[4L]] > -lam) & jnp_less(carry[[1L]], max_iters)
  k_init <- lax_full_like(lam, 0, dtype, shape)
  log_rate_init <- lax_full_like(lam, 0, "float32", shape)
  k <- lax_while_loop(cond_fn, body_fn, list(as_jax(0L), k_init, key, log_rate_init))[[2L]]
  jnp_astype(k - 1L, dtype)
}

# Port of core.py:_poisson_rejection (transformed rejection due to Hormann).
rnd_poisson_rejection <- function(key, lam, shape, dtype, max_iters) {
  log_lam <- lax_log(lam)
  b <- 0.931 + 2.53 * lax_sqrt(lam)
  a <- -0.059 + 0.02483 * b
  inv_alpha <- 1.1239 + 1.1328 / (b - 3.4)
  v_r <- 0.9277 - 3.6224 / (b - 2)
  body_fn <- function(carry) {
    ks <- rnd_split(carry[[4L]], 3L)
    u <- random_uniform(ks[[2L]], shape, aval_of(lam)$dtype) - 0.5
    v <- random_uniform(ks[[3L]], shape, aval_of(lam)$dtype)
    u_shifted <- 0.5 - abs(u)
    k <- lax_floor((2 * a / u_shifted + b) * u + lam + 0.43)
    s <- lax_log(v * inv_alpha / (a / (u_shifted * u_shifted) + b))
    t <- -lam + k * log_lam - lax_lgamma(k + 1)
    accept1 <- (u_shifted >= 0.07) & (v <= v_r)
    reject <- (k < 0) | ((u_shifted < 0.013) & (v > u_shifted))
    accept2 <- s <= t
    accept <- accept1 | (!reject & accept2)
    list(carry[[1L]] + 1L, lax_select(accept, k, carry[[2L]]), carry[[3L]] | accept, ks[[1L]])
  }
  cond_fn <- function(carry) jnp_any(!carry[[3L]]) & jnp_less(carry[[1L]], max_iters)
  k_init <- lax_full_like(lam, -1, aval_of(lam)$dtype, shape)
  accepted <- lax_full_like(lam, FALSE, "bool", shape)
  k <- lax_while_loop(cond_fn, body_fn, list(as_jax(0L), k_init, accepted, key))[[2L]]
  jnp_astype(k, dtype)
}

# Port of core.py:_poisson (Knuth for lam < 10, transformed rejection otherwise).
rnd_poisson <- prng_jit(function(key, lam, shape, dtype) {
  use_knuth <- lax_ne(lam, lam) | (lam < 10)
  lam_knuth <- lax_select(use_knuth, lam, lax_full_like(lam, 0))
  lam_rejection <- lax_select(use_knuth, lax_full_like(lam, 1e5), lam)
  max_iters <- rnd_iinfo_const(dtype, "max")
  result <- lax_select(use_knuth, rnd_poisson_knuth(key, lam_knuth, shape, dtype, max_iters),
                       rnd_poisson_rejection(key, lam_rejection, shape, dtype, max_iters))
  lax_select(lam == 0, jnp_zeros_like(result), result)
}, static_argnums = 2:3)

# Port of core.py:_poisson_approx.
rnd_poisson_approx <- prng_jit(function(key, lam, shape, dtype) {
  ld <- aval_of(lam)$dtype
  compute_dtype <- if ((is_float_dtype(ld) && dtype_bits(ld) > 32L) || dtype_bits(dtype) > 32L)
    "float64" else "float32"
  lam <- lax_convert_element_type(lam, compute_dtype)
  extra <- length(shape) - rnd_ndim(lam)
  if (extra > 0L) lam <- lax_expand_dims(lam, seq_len(extra) - 1L)
  z <- rnd_normal(key, shape, compute_dtype)
  lax_convert_element_type(rnd_poisson_from_normal(z, lam), dtype)
}, static_argnums = 2:3)

# Port of core.py:_poisson_from_normal (_POISSON_SPLIT = 7).
rnd_poisson_from_normal <- function(z, lam) {
  small <- rnd_poisson_cdf_inversion(scipy_special_ndtr(z), lam)
  large <- rnd_poisson_peizer_pratt(z, lam)
  jnp_where(lam < 7, small, large)
}

# Port of core.py:_poisson_cdf_inversion (_POISSON_NUM_TERMS = 20).
rnd_poisson_cdf_inversion <- function(u, lam) {
  pmf <- lax_exp(-lam)
  cdf <- pmf
  dt <- aval_of(u)$dtype
  count <- lax_convert_element_type(cdf < u, dt)
  for (k in 1:18) {
    pmf <- pmf * (lam / k)
    cdf <- cdf + pmf
    count <- count + lax_convert_element_type(cdf < u, dt)
  }
  count
}

# Port of core.py:_poisson_peizer_pratt.
rnd_poisson_peizer_pratt <- function(z, lam) {
  x <- rnd_peizer_pratt_root(z, lam) + 0.6
  c <- jnp_ceil(x)
  zint <- function(c) jnp_where(c < 0, -Inf, rnd_peizer_pratt_z(jnp_maximum(c, 0), lam))
  root_ge <- function(c) lax_convert_element_type(zint(c) >= z, aval_of(x)$dtype)
  out <- c + 1 - root_ge(c) - root_ge(c - 1)
  jnp_maximum(out, 0)
}

# Port of core.py:_peizer_pratt_root.
rnd_peizer_pratt_root <- function(z, lam) {
  s <- jnp_sqrt(lam)
  lam + s * z - 2 / 3 + jnp_square(z) / 6 + z * (1 - jnp_square(z)) / (72 * s)
}

# Port of core.py:_peizer_pratt_z (_PEIZER_PRATT_EPS = 0.02).
rnd_peizer_pratt_z <- function(x, lam) {
  y <- (x + 0.5) / lam
  t <- rnd_peizer_pratt_t(y)
  (x - lam + 2 / 3 + 0.02 / (x + 1)) * jnp_sqrt((1 + t) / lam)
}

# Port of core.py:_peizer_pratt_t.
rnd_peizer_pratt_t <- function(y) {
  d <- y - 1
  small <- jnp_abs(d) < 0.25
  series <- jnp_zeros_like(d)
  for (k in 9:1) series <- d * (2 * (-1)^k / ((k + 1) * (k + 2)) + series)
  y_safe <- jnp_where(small, 2, y)
  d_safe <- jnp_where(small, 1, d)
  direct <- (1 - jnp_square(y_safe) + 2 * y_safe * jnp_log(y_safe)) / jnp_square(d_safe)
  jnp_where(small, series, direct)
}

# ---- binomial and multinomial -----------------------------------------------------------

#' @rdname random_discrete
#' @export
random_binomial <- function(key, n, p, shape = NULL, dtype = NULL) {
  key <- rnd_check_key("binomial", key)$key
  dt <- rnd_float_dtype("binomial", dtype)
  if (!is.null(shape)) shape <- canon_shape(shape)
  rnd_binomial(key, n, p, shape, dt)
}

# Port of core.py:_stirling_approx_tail. The table lookup at the clamped index is a
# select_n over the ten tabulated values.
rnd_stirling_approx_tail <- function(k) {
  vals <- c(0.0810614667953272, 0.0413406959554092, 0.0276779256849983, 0.02079067210376509,
            0.0166446911898211, 0.0138761288230707, 0.0118967099458917, 0.0104112652619720,
            0.00925546218271273, 0.00833056343336287)
  use_tail_values <- k <= 9
  k <- lax_clamp(prng_const(k, 0), k, prng_const(k, 9))
  kp1sq <- (k + 1) * (k + 1)
  approx <- (1 / 12 - (1 / 360 - 1 / 1260 / kp1sq) / kp1sq) / (k + 1)
  idx <- jnp_astype(jnp_floor(k), "int32")
  table <- do.call(lax_select_n, c(list(idx), lapply(vals, function(v) lax_full_like(k, v))))
  lax_select(use_tail_values, table, approx)
}

# Port of core.py:_binomial_inversion.
rnd_binomial_inversion <- function(key, count, prob, shape, dtype, max_iters) {
  log1minusprob <- jnp_log1p(-prob)
  body_fn <- function(carry) {
    ks <- as.list.jax_prng_key(random_split(carry[[4L]]))
    num_geom_out <- lax_select(carry[[3L]] <= count, carry[[2L]] + 1, carry[[2L]])
    u <- random_uniform(ks[[1L]], shape, aval_of(prob)$dtype)
    geom <- jnp_ceil(jnp_log(u) / log1minusprob)
    list(carry[[1L]] + 1L, num_geom_out, carry[[3L]] + geom, ks[[2L]])
  }
  cond_fn <- function(carry) jnp_any(carry[[3L]] <= count) & jnp_less(carry[[1L]], max_iters)
  num_geom_init <- lax_full_like(prob, 0, aval_of(prob)$dtype, shape)
  geom_sum_init <- lax_full_like(prob, 0, aval_of(prob)$dtype, shape)
  k <- lax_while_loop(cond_fn, body_fn, list(as_jax(0L), num_geom_init, geom_sum_init, key))[[2L]]
  jnp_astype(k - 1, dtype)
}

# Port of core.py:_btrs (transformed rejection).
rnd_btrs <- function(key, count, prob, shape, dtype, max_iters) {
  stddev <- jnp_sqrt(count * prob * (1 - prob))
  b <- 1.15 + 2.53 * stddev
  a <- -0.0873 + 0.0248 * b + 0.01 * prob
  c <- count * prob + 0.5
  v_r <- 0.92 - 4.2 / b
  r <- prob / (1 - prob)
  alpha <- (2.83 + 5.1 / b) * stddev
  m <- jnp_floor((count + 1) * prob)
  body_fn <- function(carry) {
    ks <- as.list.jax_prng_key(random_split(carry[[4L]], 3L))
    u <- random_uniform(ks[[2L]], shape, aval_of(prob)$dtype)
    v <- random_uniform(ks[[3L]], shape, aval_of(prob)$dtype)
    u <- u - 0.5
    us <- 0.5 - jnp_abs(u)
    accept1 <- (us >= 0.07) & (v <= v_r)
    k <- jnp_floor((2 * a / us + b) * u + c)
    reject <- (k < 0) | (k > count)
    v <- jnp_log(v * alpha / (a / (us * us) + b))
    ub <- (m + 0.5) * jnp_log((m + 1) / (r * (count - m + 1))) +
      (count + 1) * jnp_log((count - m + 1) / (count - k + 1)) +
      (k + 0.5) * jnp_log(r * (count - k + 1) / (k + 1)) +
      rnd_stirling_approx_tail(m) +
      rnd_stirling_approx_tail(count - m) -
      rnd_stirling_approx_tail(k) -
      rnd_stirling_approx_tail(count - k)
    accept2 <- v <= ub
    accept <- accept1 | (!reject & accept2)
    list(carry[[1L]] + 1L, lax_select(accept, k, carry[[2L]]), carry[[3L]] | accept, ks[[1L]])
  }
  cond_fn <- function(carry) jnp_any(!carry[[3L]]) & jnp_less(carry[[1L]], max_iters)
  k_init <- lax_full_like(prob, -1, aval_of(prob)$dtype, shape)
  carry <- list(as_jax(0L), k_init, jnp_full(shape, FALSE, "bool"), key)
  jnp_astype(lax_while_loop(cond_fn, body_fn, carry)[[2L]], dtype)
}

# Port of core.py:_binomial (inversion for n * p <= 10, BTRS otherwise).
rnd_binomial <- prng_jit(function(key, count, prob, shape, dtype) {
  if (is.null(shape)) {
    shape <- broadcast_shapes_impl(list(rnd_shape(count), rnd_shape(prob)))
  } else {
    rnd_check_shape("binomial", shape, rnd_shape(count), rnd_shape(prob))
  }
  prob <- promote_dtypes_inexact(list(prob))[[1L]]
  pdt <- aval_of(prob)$dtype
  count <- jnp_broadcast_to(lax_convert_element_type(count, pdt), shape)
  prob <- jnp_broadcast_to(prob, shape)
  p_lt_half <- prob < 0.5
  q <- lax_select(p_lt_half, prob, 1.0 - prob)
  count_nan_or_neg <- lax_ne(count, count) | (count < 0.0)
  count_inf <- jnp_isinf(count)
  q_is_nan <- lax_ne(q, q)
  q_l_0 <- q < 0.0
  q <- lax_select(q_is_nan | q_l_0, lax_full_like(q, 0.01), q)
  use_inversion <- count_nan_or_neg | (count * q <= 10.0)
  # Consistent with np.random.binomial for float count input.
  count <- jnp_floor(count)
  count_inv <- lax_select(use_inversion, count, lax_full_like(count, 0.0))
  count_btrs <- lax_select(use_inversion, lax_full_like(count, 1e4), count)
  q_btrs <- lax_select(use_inversion, lax_full_like(q, 0.5), q)
  max_iters <- prng_full(finfo(dtype)$max, dtype)
  samples <- lax_select(use_inversion,
                        rnd_binomial_inversion(key, count_inv, q, shape, dtype, max_iters),
                        rnd_btrs(key, count_btrs, q_btrs, shape, dtype, max_iters))
  # NaN q leads to NaN output, NaN or negative count leads to NaN.
  invalid <- q_l_0 | q_is_nan | count_nan_or_neg
  samples <- lax_select(invalid, jnp_full_like(samples, NaN, dtype), samples)
  # +inf count leads to inf.
  samples <- lax_select(count_inf & !invalid, jnp_full_like(samples, Inf, dtype), samples)
  lax_select(p_lt_half | count_nan_or_neg | q_is_nan | count_inf, samples,
             jnp_astype(count, dtype) - samples)
}, static_argnums = 3:4)

#' @rdname random_discrete
#' @export
random_multinomial <- function(key, n, p, shape = NULL, dtype = NULL, unroll = 1L) {
  key <- rnd_check_key("multinomial", key)$key
  np_ <- promote_dtypes_inexact(list(n, p))
  n <- np_[[1L]]
  p <- np_[[2L]]
  ps <- aval_of(p)$shape
  if (is.null(shape)) {
    shape <- utils::head(ps, -1L)
  } else {
    shape <- canon_shape(shape)
    rnd_check_shape("multinomial", shape, aval_of(n)$shape, utils::head(ps, -1L))
  }
  n <- jnp_broadcast_to(n, shape)
  p <- jnp_moveaxis(jnp_broadcast_to(p, c(shape, utils::tail(ps, 1L))), -1L, 0L)
  f <- function(remainder, ratio_key) {
    count <- random_binomial(ratio_key[[2L]], remainder, rnd_clip(ratio_key[[1L]], 0, 1),
                             dtype = aval_of(remainder)$dtype)
    list(remainder - count, count)
  }
  remaining_probs <- lax_cumsum(p, 0L, reverse = TRUE)
  ratios <- p / jnp_where(remaining_probs == 0, 1, remaining_probs)
  keys <- random_split(key, aval_of(ratios)$shape[1L])
  counts <- lax_scan(f, n, list(ratios, keys), unroll = unroll)[[2L]]
  jnp_astype(jnp_moveaxis(counts, 0L, -1L), dtype(dtype %||% "float"))
}

# ---- gamma family -----------------------------------------------------------------------

# Port of core.py:_gamma_one (Marsaglia and Tsang).
rnd_gamma_one <- function(key, alpha, log_space) {
  zero <- prng_const(alpha, 0)
  one <- prng_const(alpha, 1)
  two <- prng_const(alpha, 2)
  minus_one <- prng_const(alpha, -1)
  one_over_two <- prng_const(alpha, 0.5)
  one_over_three <- prng_const(alpha, 1 / 3)
  squeeze_const <- prng_const(alpha, 0.0331)
  dtype <- aval_of(alpha)$dtype
  # For alpha < 1, Gamma(alpha) ~ Gamma(alpha + 1) * Uniform()^(1 / alpha).
  boost_mask <- lax_ge(alpha, one)
  alpha_orig <- alpha
  alpha <- lax_select(boost_mask, alpha, lax_add(alpha, one))
  d <- lax_sub(alpha, one_over_three)
  c <- lax_div(one_over_three, lax_sqrt(d))
  cond_fn <- function(kXVU) {
    X <- kXVU[[2L]]
    V <- kXVU[[3L]]
    U <- kXVU[[4L]]
    lax_bitwise_and(
      lax_ge(U, lax_sub(one, lax_mul(squeeze_const, lax_mul(X, X)))),
      lax_ge(lax_log(U), lax_add(lax_mul(X, one_over_two),
                                 lax_mul(d, lax_add(lax_sub(one, V), lax_log(V))))))
  }
  body_fn <- function(kXVU) {
    next_kxv <- function(kxv) {
      ks <- rnd_split(kxv[[1L]])
      x <- random_normal(ks[[2L]], integer(0), dtype = dtype)
      list(ks[[1L]], x, lax_add(one, lax_mul(x, c)))
    }
    ks <- rnd_split(kXVU[[1L]], 3L)
    r <- lax_while_loop(function(kxv) lax_le(kxv[[3L]], zero), next_kxv,
                        list(ks[[2L]], zero, minus_one))
    x <- r[[2L]]
    v <- r[[3L]]
    list(ks[[1L]], lax_mul(x, x), lax_mul(lax_mul(v, v), v),
         random_uniform(ks[[3L]], integer(0), dtype = dtype))
  }
  ks <- rnd_split(key)
  V <- lax_while_loop(cond_fn, body_fn, list(ks[[1L]], zero, one, two))[[3L]]
  subkey <- ks[[2L]]
  if (log_space) {
    log_samples <- lax_neg(random_exponential(subkey, integer(0), dtype = dtype))
    log_boost <- lax_select(boost_mask | (log_samples == 0), zero,
                            lax_mul(log_samples, lax_div(one, alpha_orig)))
    return(lax_add(lax_add(lax_log(d), lax_log(V)), log_boost))
  }
  samples <- 1 - random_uniform(subkey, integer(0), dtype = dtype)
  boost <- lax_select(boost_mask, one, lax_pow(samples, lax_div(one, alpha_orig)))
  lax_mul(lax_mul(d, V), boost)
}

# Port of core.py:_gamma_grad (the CPU path maps over the samples).
rnd_gamma_grad <- function(sample, a, log_space) {
  samples <- lax_reshape(sample, length(sample))
  alphas <- lax_reshape(a, length(a))
  if (log_space) {
    # d[log(sample)] = d[sample] / sample; exp(log_sample) may round to zero.
    samples <- lax_exp(samples)
    tiny <- lax_full_like(samples, finfo(aval_of(samples)$dtype)$tiny)
    samples <- lax_select(lax_eq(samples, prng_const(samples, 0)), tiny, samples)
    gamma_grad <- function(alpha, s) lax_random_gamma_grad(alpha, s) / s
  } else {
    gamma_grad <- function(alpha, s) lax_random_gamma_grad(alpha, s)
  }
  grads <- lax_map(function(args) gamma_grad(args[[1L]], args[[2L]]), list(alphas, samples))
  lax_reshape(grads, aval_of(a)$shape)
}

# Port of core.py:_gamma_impl with use_vmap=True (the lowering JAX registers). `key`
# is the key data, with any leading key dimensions.
rnd_gamma_impl <- function(key, a, log_space, impl) {
  info <- prng_impls[[impl]]
  a_shape <- aval_of(a)$shape
  ks <- aval_of(key)$shape
  key_ndim <- length(ks) - length(info$key_shape)
  split_count <- prod_int(drop_first(a_shape, key_ndim))
  keys <- lax_reshape(key, c(prod_int(ks[seq_len(key_ndim)]), info$key_shape))
  keys <- vmap(function(k) prng_random_split(prng_key_array(k, impl), split_count)$data)(keys)
  keys <- lax_reshape(keys, c(prod_int(aval_of(keys)$shape) %/% prod_int(info$key_shape),
                              info$key_shape))
  alphas <- lax_reshape(a, prod_int(a_shape))
  one <- function(k, alpha) rnd_gamma_one(prng_key_array(k, impl), alpha, log_space)
  samples <- if (impl == "threefry2x32") vmap(one)(keys, alphas)
             else lax_map(function(args) one(args[[1L]], args[[2L]]), list(keys, alphas))
  lax_reshape(samples, a_shape)
}

random_gamma_p <- new_primitive("random_gamma")
random_gamma_p$abstract_eval <- function(avals, params) avals[[2L]]
random_gamma_p$lowering <- lower_fun(function(args, params) {
  rnd_gamma_impl(args[[1L]], args[[2L]], params$log_space, params$impl)
})
defjvp2(random_gamma_p, NULL, function(t, ans, primals, params) {
  t * rnd_gamma_grad(ans, primals[[2L]], params$log_space)
})
# Port of core.py:_gamma_batching_rule.
random_gamma_p$batch <- function(vals, dims, params, axis) {
  k <- bdim_at_front(vals[[1L]], dims[1L], axis$size)
  a <- bdim_at_front(vals[[2L]], dims[2L], axis$size)
  list(bind(random_gamma_p, list(k, a), params), 0L)
}

#' Gamma family random values
#'
#' Ports of `jax.random.gamma`, `loggamma`, `beta`, `dirichlet`, `chisquare`, `f` and
#' `t`. The exact samplers are rejection samplers (Marsaglia and Tsang); their gradients
#' with respect to the shape parameters use implicit reparameterization through
#' [lax_random_gamma_grad()]. `random_random_gamma_p` is the underlying primitive.
#' @inheritParams random_uniform
#' @param a,b Shape parameters (`a` for gamma, `a` and `b` for beta).
#' @param shape Integer vector; `NULL` broadcasts the parameters.
#' @param dtype A float dtype.
#' @param method `"exact"` (rejection sampler) or `"approximate"` (loop free, slightly
#'   biased).
#' @param alpha Dirichlet concentration of shape `(..., n)`.
#' @param df Degrees of freedom.
#' @param dfnum,dfden Numerator and denominator degrees of freedom.
#' @return An array.
#' @name random_gamma
#' @export
random_gamma <- function(key, a, shape = NULL, dtype = NULL, method = "exact",
                         out_sharding = NULL) {
  key <- rnd_check_key("gamma", key)$key
  rnd_check_method("gamma", method)
  dt <- rnd_float_dtype("gamma", dtype)
  if (!is.null(shape)) shape <- canon_shape(shape)
  if (method == "approximate") return(rnd_gamma_approx(key, a, shape, dt, FALSE))
  rnd_gamma(key, a, shape, dt, FALSE)
}

rnd_check_method <- function(name, method) {
  if (!method %in% c("exact", "approximate"))
    jax_error("method argument to `", name, "` must be one of {'exact', 'approximate'}, got '",
              method, "'", class = "jax_value_error")
}

#' @rdname random_gamma
#' @export
random_loggamma <- function(key, a, shape = NULL, dtype = NULL, method = "exact",
                            out_sharding = NULL) {
  key <- rnd_check_key("loggamma", key)$key
  rnd_check_method("loggamma", method)
  dt <- rnd_float_dtype("gamma", dtype)
  if (!is.null(shape)) shape <- canon_shape(shape)
  if (method == "approximate") return(rnd_gamma_approx(key, a, shape, dt, TRUE))
  rnd_gamma(key, a, shape, dt, TRUE)
}

#' @rdname random_gamma
#' @format NULL
#' @export
random_random_gamma_p <- random_gamma_p

# Port of core.py:_gamma.
rnd_gamma <- prng_jit(function(key, a, shape, dtype, log_space) {
  kr_sink(key, "random_gamma")
  if (is.null(shape)) shape <- rnd_shape(a) else rnd_check_shape("gamma", shape, rnd_shape(a))
  a <- lax_convert_element_type(a, dtype)
  if (!identical(aval_of(a)$shape, as.integer(shape))) a <- jnp_broadcast_to(a, shape)
  bind(random_gamma_p, list(key$data, a),
       list(log_space = isTRUE(log_space), impl = attr(key, "impl", exact = TRUE)))
}, static_argnums = 2:4)

# Coefficients of the chi-square quantile series used by rnd_gamma_approx (rows k:
# coefficients of x^k nu^(-k/2) as a quadratic in 1 / nu), core.py:_CHI2_QUANTILE_COEF.
rnd_chi2_quantile_coef <- list(
  c(1.0, -0.2237368, -0.01513904),
  c(0.4713941, 0.02607083, -0.008986007),
  c(0.0001348028, 0.01128186, 0.02277679),
  c(-0.008553069, -0.01153761, -0.01323293),
  c(0.00312558, 0.005169654, -0.006950356),
  c(-0.0008426812, 0.00253001, 0.001060438),
  c(0.00009780499, -0.001450117, 0.001565326)
)

# lax.polynomial by Horner's rule, as lax.py:_polynomial_lower (at most 8 terms).
rnd_horner <- function(x, coeffs) {
  acc <- coeffs[[length(coeffs)]]
  for (i in rev(seq_len(length(coeffs) - 1L))) acc <- lax_add(lax_mul(acc, x), coeffs[[i]])
  acc
}

# Port of core.py:_loggamma_chisquare.
rnd_loggamma_chisquare <- function(key, a) {
  nu <- 2 * a
  x <- rnd_normal(key, aval_of(a)$shape, aval_of(a)$dtype)
  u <- 1 / nu
  t <- x * lax_rsqrt(nu)
  row_coeffs <- lapply(rnd_chi2_quantile_coef, function(row) {
    rnd_horner(u, lapply(row, function(cf) prng_const(u, cf)))
  })
  poly <- rnd_horner(t, row_coeffs)
  poly <- jnp_maximum(poly, finfo(aval_of(a)$dtype)$tiny)
  jnp_log(a) + 3 * jnp_log(poly)
}

# Port of core.py:_gamma_approx (_LOGGAMMA_NUM_BOOSTS = 4).
rnd_gamma_approx <- prng_jit(function(key, a, shape, dtype, log_space) {
  compute_dtype <- if (finfo(dtype)$bits < 32L) "float32" else dtype
  shape <- rnd_check_broadcast_shapes("gamma", shape, a)
  a <- lax_convert_element_type(a, compute_dtype)
  if (!identical(aval_of(a)$shape, shape)) a <- jnp_broadcast_to(a, shape)
  ks <- rnd_split(key)
  n <- 4L
  log_base <- rnd_loggamma_chisquare(ks[[1L]], a + n)
  k <- lax_reshape(jnp_arange(n, dtype = compute_dtype), c(n, rep(1L, length(shape))))
  log_u <- -rnd_exponential(ks[[2L]], c(n, shape), compute_dtype)
  log_sample <- log_base + jnp_sum(log_u / (lax_expand_dims(a, 0L) + k), axis = 0L)
  out <- if (log_space) log_sample else lax_exp(log_sample)
  lax_convert_element_type(out, dtype)
}, static_argnums = 2:4)

#' @rdname random_gamma
#' @export
random_beta <- function(key, a, b, shape = NULL, dtype = NULL, method = "exact",
                        out_sharding = NULL) {
  key <- rnd_check_key("beta", key)$key
  if (!method %in% c("exact", "approximate"))
    jax_error("method argument to `beta` must be one of {'exact', 'approximate'}, got '",
              method, "'", class = "jax_value_error")
  dt <- rnd_float_dtype("beta", dtype)
  rnd_check_safe_cast("beta", dt, a, b)
  shape <- rnd_check_broadcast_shapes("beta", shape, a, b)
  rnd_beta(key, a, b, method, shape, dt)
}

# Port of core.py:_beta.
rnd_beta <- prng_jit(function(key, a, b, method, shape, dtype) {
  rnd_check_shape("beta", shape, rnd_shape(a), rnd_shape(b))
  a <- lax_convert_element_type(a, dtype)
  b <- lax_convert_element_type(b, dtype)
  ks <- rnd_split(key)
  a <- jnp_broadcast_to(a, shape)
  b <- jnp_broadcast_to(b, shape)
  log_gamma_a <- random_loggamma(ks[[1L]], a, shape, dtype, method = method)
  log_gamma_b <- random_loggamma(ks[[2L]], b, shape, dtype, method = method)
  # gamma_a / (gamma_a + gamma_b) without losing precision.
  log_max <- lax_max(log_gamma_a, log_gamma_b)
  gamma_a_scaled <- jnp_exp(log_gamma_a - log_max)
  gamma_b_scaled <- jnp_exp(log_gamma_b - log_max)
  gamma_a_scaled / (gamma_a_scaled + gamma_b_scaled)
}, static_argnums = 3:5)

#' @rdname random_gamma
#' @export
random_dirichlet <- function(key, alpha, shape = NULL, dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("dirichlet", key)$key
  dt <- rnd_float_dtype("dirichlet", dtype)
  if (!is.null(shape)) shape <- canon_shape(shape)
  rnd_dirichlet(key, alpha, shape, dt)
}

# Port of nn/functions.py:_softmax (custom_jvp; where = None).
rnd_softmax_fun <- function(x, axis) {
  unnormalized <- jnp_exp(x - jnp_max(x, axis, keepdims = TRUE))
  unnormalized / jnp_sum(unnormalized, axis, keepdims = TRUE)
}
rnd_softmax <- custom_jvp(rnd_softmax_fun, jvp = function(axis, primals, tangents) {
  y <- rnd_softmax_fun(primals[[1L]], axis)
  x_dot <- tangents[[1L]]
  list(y, y * (x_dot - jnp_sum(y * x_dot, axis, keepdims = TRUE)))
}, nondiff_argnums = 1L)

# Port of core.py:_dirichlet.
rnd_dirichlet <- prng_jit(function(key, alpha, shape, dtype) {
  if (rnd_ndim(alpha) < 1L)
    jax_error("dirichlet requires alpha.ndim >= 1, got alpha.ndim == ", rnd_ndim(alpha),
              class = "jax_value_error")
  as_ <- rnd_shape(alpha)
  if (is.null(shape)) shape <- utils::head(as_, -1L)
  else rnd_check_shape("dirichlet", shape, utils::head(as_, -1L))
  alpha <- lax_convert_element_type(alpha, dtype)
  # Gamma in log space: small alpha behaves poorly otherwise.
  log_gamma_samples <- random_loggamma(key, alpha, c(shape, utils::tail(as_, 1L)), dtype)
  rnd_softmax(log_gamma_samples, -1L)
}, static_argnums = 2:3)

#' @rdname random_gamma
#' @export
random_chisquare <- function(key, df, shape = NULL, dtype = NULL, method = "exact",
                             out_sharding = NULL) {
  key <- rnd_check_key("chisquare", key)$key
  if (!method %in% c("exact", "approximate"))
    jax_error("method argument to `chisquare` must be one of {'exact', 'approximate'}, got '",
              method, "'", class = "jax_value_error")
  dt <- rnd_float_dtype("chisquare", dtype)
  shape <- rnd_check_broadcast_shapes("chisquare", shape, df)
  rnd_check_safe_cast("chisquare", dt, df)
  rnd_chisquare(key, df, method, shape, dt)
}

# Port of core.py:_chisquare.
rnd_chisquare <- prng_jit(function(key, df, method, shape, dtype) {
  df <- lax_convert_element_type(df, dtype)
  two <- prng_const(df, 2)
  half_df <- lax_div(df, two)
  log_g <- random_loggamma(key, half_df, shape, dtype, method = method)
  lax_mul(jnp_exp(log_g), two)
}, static_argnums = 2:4)

#' @rdname random_gamma
#' @export
random_f <- function(key, dfnum, dfden, shape = NULL, dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("f", key)$key
  dt <- rnd_float_dtype("f", dtype)
  shape <- rnd_check_broadcast_shapes("f", shape, dfnum, dfden)
  rnd_check_safe_cast("f", dt, dfnum, dfden)
  rnd_f(key, dfnum, dfden, shape, dt)
}

# Port of core.py:_f.
rnd_f <- prng_jit(function(key, dfnum, dfden, shape, dtype) {
  dfden <- lax_convert_element_type(dfden, dtype)
  dfnum <- lax_convert_element_type(dfnum, dtype)
  ks <- rnd_split(key)
  chi2_dfn <- random_chisquare(ks[[2L]], dfnum, shape, dtype)
  chi2_dfd <- random_chisquare(ks[[1L]], dfden, shape, dtype)
  lax_div(chi2_dfn / dfnum, chi2_dfd / dfden)
}, static_argnums = 3:4)

#' @rdname random_gamma
#' @export
random_t <- function(key, df, shape = NULL, dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("t", key)$key
  dt <- rnd_float_dtype("t", dtype)
  shape <- rnd_check_broadcast_shapes("t", shape, df)
  rnd_check_safe_cast("t", dt, df)
  rnd_t(key, df, shape, dt)
}

# Port of core.py:_t.
rnd_t <- prng_jit(function(key, df, shape, dtype) {
  rnd_check_shape("t", shape, rnd_shape(df))
  df <- lax_convert_element_type(df, dtype)
  ks <- rnd_split(key)
  n <- random_normal(ks[[1L]], shape, dtype)
  half_df <- lax_div(df, prng_const(n, 2))
  g <- random_gamma(ks[[2L]], half_df, shape, dtype)
  n * jnp_sqrt(half_df / g)
}, static_argnums = 2:3)

# ---- other continuous distributions -----------------------------------------------------

#' Continuous random values
#'
#' Ports of `jax.random.cauchy`, `exponential`, `gumbel`, `laplace`, `logistic`,
#' `pareto`, `rayleigh`, `wald`, `maxwell`, `double_sided_maxwell`, `weibull_min`,
#' `triangular`, `generalized_normal`, `ball` and `orthogonal`.
#' @inheritParams random_uniform
#' @param shape Integer vector; `NULL` (where allowed) broadcasts the parameters.
#' @param dtype A float dtype.
#' @param mode For `random_gumbel()`, the bits used: `"low"` (default), `"high"` or
#'   `"highest"`; for `random_triangular()`, the peak of the distribution.
#' @param b Pareto shape parameter.
#' @param scale Scale parameter.
#' @param mean Wald mean.
#' @param loc Location parameter.
#' @param concentration Weibull concentration.
#' @param left,right Triangular limits.
#' @param p Shape parameter (`generalized_normal`), or the `p` of the Lp ball.
#' @param d Dimension of the ball.
#' @param n,m Rows and columns of the orthogonal matrix (`m` defaults to `n`).
#' @return An array.
#' @name random_continuous
#' @export
random_cauchy <- function(key, shape = integer(0), dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("cauchy", key)$key
  rnd_cauchy(key, canon_shape(shape), rnd_float_dtype("cauchy", dtype))
}

# Port of core.py:_cauchy.
rnd_cauchy <- prng_jit(function(key, shape, dtype) {
  u <- random_uniform(key, shape, dtype, minval = finfo(dtype)$eps, maxval = 1)
  lax_tan(lax_mul(prng_const(u, pi), lax_sub(u, prng_const(u, 0.5))))
}, static_argnums = 1:2)

#' @rdname random_continuous
#' @export
random_exponential <- function(key, shape = integer(0), dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("exponential", key)$key
  rnd_exponential(key, canon_shape(shape), rnd_float_dtype("exponential", dtype))
}

# Port of core.py:_exponential (1 - u moves the domain of log to (0, 1]).
rnd_exponential <- prng_jit(function(key, shape, dtype) {
  u <- random_uniform(key, shape, dtype)
  lax_neg(lax_log1p(lax_neg(u)))
}, static_argnums = 1:2)

#' @rdname random_continuous
#' @export
random_gumbel <- function(key, shape = integer(0), dtype = NULL, mode = NULL, out_sharding = NULL) {
  key <- rnd_check_key("gumbel", key)$key
  dt <- rnd_float_dtype("gumbel", dtype)
  shape <- canon_shape(shape)
  mode <- mode %||% (if (isTRUE(.jx$config$jax_use_high_dynamic_range_gumbel)) "high" else "low")
  if (!mode %in% c("highest", "high", "low"))
    jax_error("Must provide valid mode for gumbel got: ", mode, class = "jax_value_error")
  rnd_gumbel(key, shape, dt, mode)
}

# Port of core.py:_safe_int_to_float: u32[2, ...] bits to f32 in (0, 1).
rnd_safe_int_to_float <- function(bits) {
  fi <- finfo("float32")
  clz <- lax_clz(bits)
  hiclz <- lax_index_in_dim(clz, 0L, 0L, keepdims = FALSE)
  loclz <- lax_index_in_dim(clz, 1L, 0L, keepdims = FALSE)
  hi <- lax_index_in_dim(bits, 0L, 0L, keepdims = FALSE)
  lo <- lax_index_in_dim(bits, 1L, 0L, keepdims = FALSE)
  mantissa <- lax_bitwise_or(
    lax_shift_left(hi, hiclz),
    jnp_where(hiclz == 32L, lax_shift_left(lo, loclz), lax_shift_right_logical(lo, 32L - hiclz)))
  mantissa <- lax_shift_right_logical(mantissa, prng_u32(fi$bits - fi$nmant - 1))
  mantissa <- lax_bitwise_and(mantissa, prng_u32(2^fi$nmant - 1))
  ex <- lax_shift_left(as.integer(-rnd_minexp(fi)) - jnp_where(hiclz == 32L, 32L + loclz, hiclz),
                       prng_u32(fi$nmant))
  ex <- lax_bitwise_and(ex, prng_u32(2^(fi$bits - 2) - 2^(fi$nmant - 1)))
  lax_bitcast_convert_type(lax_bitwise_or(ex, mantissa), "float32")
}

# Port of core.py:_gumbel.
rnd_gumbel <- prng_jit(function(key, shape, dtype, mode) {
  fi <- finfo(dtype)
  if (dtype == "float32" && mode == "highest") {
    bits <- rnd_bits(key, 32L, c(2L, shape))
    neg <- lax_bitwise_not(bits)
    lo_bits <- lax_index_in_dim(neg, 1L, 0L, keepdims = FALSE)
    # 1 - bits in u64 fixed point.
    carry <- jnp_astype(jnp_stack(list(lo_bits == prng_u32(2^32 - 1), jnp_ones_like(lo_bits, dtype = "bool"))),
                        "uint32")
    neg <- neg + carry
    flip_mask <- lax_index_in_dim(bits, 0L, 0L, keepdims = FALSE) < prng_u32(2^31)
    x <- rnd_safe_int_to_float(jnp_where(flip_mask, bits, neg))
    # log1p on (0, 0.5) and log on [0.5, 1).
    return(jnp_where(flip_mask, -jnp_log(-jnp_log1p(-x)), -jnp_log(-jnp_log(x))))
  }
  if (mode %in% c("high", "highest")) {
    u <- rnd_uniform(key, 0, 1, c(2L, shape), dtype)
    high <- lax_index_in_dim(u, 0L, 0L, keepdims = FALSE)
    low <- lax_index_in_dim(u, 1L, 0L, keepdims = FALSE)
    x <- jnp_where(high >= 0.5, high,
                   high + 2^-fi$nmant * low + max(2^-(2 * fi$nmant + 1), fi$tiny))
    return(-jnp_log(-jnp_log1p(-x)))
  }
  -jnp_log(-jnp_log(rnd_uniform(key, fi$tiny, 1, shape, dtype)))
}, static_argnums = 1:3)

#' @rdname random_continuous
#' @export
random_laplace <- function(key, shape = integer(0), dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("laplace", key)$key
  rnd_laplace(key, canon_shape(shape), rnd_float_dtype("laplace", dtype))
}

# Port of core.py:_laplace.
rnd_laplace <- prng_jit(function(key, shape, dtype) {
  epsneg <- 2^-(finfo(dtype)$nmant + 1)
  u <- random_uniform(key, shape, dtype, minval = -1 + epsneg, maxval = 1)
  lax_mul(lax_sign(u), lax_log1p(lax_neg(lax_abs(u))))
}, static_argnums = 1:2)

#' @rdname random_continuous
#' @export
random_logistic <- function(key, shape = integer(0), dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("logistic", key)$key
  rnd_logistic(key, canon_shape(shape), rnd_float_dtype("logistic", dtype))
}

# Port of core.py:_logistic.
rnd_logistic <- prng_jit(function(key, shape, dtype) {
  x <- random_uniform(key, shape, dtype, minval = finfo(dtype)$tiny, maxval = 1)
  lax_sub(lax_log(x), lax_log1p(lax_neg(x)))
}, static_argnums = 1:2)

#' @rdname random_continuous
#' @export
random_pareto <- function(key, b, shape = NULL, dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("pareto", key)$key
  dt <- rnd_float_dtype("pareto", dtype)
  shape <- rnd_check_broadcast_shapes("pareto", shape, b)
  rnd_check_safe_cast("pareto", dt, b)
  rnd_pareto(key, b, shape, dt)
}

# Port of core.py:_pareto.
rnd_pareto <- prng_jit(function(key, b, shape, dtype) {
  b <- lax_convert_element_type(b, dtype)
  lax_exp(random_exponential(key, shape, dtype) / b)
}, static_argnums = 2:3)

#' @rdname random_continuous
#' @export
random_rayleigh <- function(key, scale, shape = NULL, dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("rayleigh", key)$key
  dt <- rnd_float_dtype("rayleigh", dtype)
  shape <- rnd_check_broadcast_shapes("rayleigh", shape, scale)
  rnd_check_safe_cast("rayleigh", dt, scale)
  rnd_rayleigh(key, scale, shape, dt)
}

# Port of core.py:_rayleigh.
rnd_rayleigh <- prng_jit(function(key, scale, shape, dtype) {
  u <- random_uniform(key, shape, dtype)
  scale <- jnp_broadcast_to(jnp_astype(scale, dtype), shape)
  log_u <- lax_log(u)
  sqrt_u <- lax_sqrt(lax_mul(log_u, prng_const(scale, -2)))
  lax_mul(scale, sqrt_u)
}, static_argnums = 2:3)

#' @rdname random_continuous
#' @export
random_wald <- function(key, mean, shape = NULL, dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("wald", key)$key
  dt <- rnd_float_dtype("wald", dtype)
  shape <- rnd_check_broadcast_shapes("wald", shape, mean)
  rnd_check_safe_cast("wald", dt, mean)
  rnd_wald(key, mean, shape, dt)
}

# Port of core.py:_wald.
rnd_wald <- prng_jit(function(key, mean, shape, dtype) {
  ks <- rnd_split(key, 2L)
  mean <- jnp_broadcast_to(jnp_astype(mean, dtype), shape)
  v <- random_normal(ks[[1L]], shape, dtype)
  z <- random_uniform(ks[[2L]], shape, dtype)
  y <- lax_integer_pow(v, 2L)
  y_sq <- lax_integer_pow(y, 2L)
  mean_sq <- lax_integer_pow(mean, 2L)
  sqrt_term <- lax_sqrt(4 * mean * y + mean_sq * y_sq)
  x <- mean + mean_sq * y / 2 - mean / 2 * sqrt_term
  lax_select(lax_le(z, mean / (mean + x)), x, mean_sq / x)
}, static_argnums = 2:3)

#' @rdname random_continuous
#' @export
random_maxwell <- function(key, shape = integer(0), dtype = NULL, out_sharding = NULL) {
  key <- rnd_check_key("maxwell", key)$key
  rnd_maxwell(key, canon_shape(shape), rnd_float_dtype("maxwell", dtype))
}

# Port of core.py:_maxwell: the norm of three standard normals (jnp.linalg.norm with
# ord=None is sqrt(sum(x * conj(x)))).
rnd_maxwell <- prng_jit(function(key, shape, dtype) {
  norm_rvs <- random_normal(key, c(shape, 3L), dtype)
  jnp_sqrt(jnp_sum(norm_rvs * norm_rvs, axis = -1L))
}, static_argnums = 1:2)

#' @rdname random_continuous
#' @export
random_double_sided_maxwell <- function(key, loc, scale, shape = integer(0), dtype = NULL) {
  key <- rnd_check_key("double_sided_maxwell", key)$key
  dt <- rnd_float_dtype("double_sided_maxwell", dtype)
  rnd_double_sided_maxwell(key, loc, scale, canon_shape(shape), dt)
}

# Port of core.py:_double_sided_maxwell.
rnd_double_sided_maxwell <- prng_jit(function(key, loc, scale, shape, dtype) {
  params_shapes <- broadcast_shapes_impl(list(rnd_shape(loc), rnd_shape(scale)))
  if (!length(shape)) shape <- params_shapes
  shape <- c(shape, params_shapes)
  ks <- rnd_split(key)
  maxwell_rvs <- random_maxwell(ks[[1L]], shape = shape, dtype = dtype)
  random_sign <- random_rademacher(ks[[2L]], shape = shape, dtype = dtype)
  random_sign * maxwell_rvs * scale + loc
}, static_argnums = 3:4)

#' @rdname random_continuous
#' @export
random_weibull_min <- function(key, scale, concentration, shape = integer(0), dtype = NULL) {
  key <- rnd_check_key("weibull_min", key)$key
  dt <- rnd_float_dtype("weibull_min", dtype)
  rnd_weibull_min(key, scale, concentration, canon_shape(shape), dt)
}

# Port of core.py:_weibull_min (inverse Weibull CDF).
rnd_weibull_min <- prng_jit(function(key, scale, concentration, shape, dtype) {
  u <- random_uniform(key, shape, dtype, minval = 0L, maxval = 1L)
  jnp_power(-jnp_log1p(-u), 1.0 / as_jax(concentration)) * scale
}, static_argnums = 3:4)

#' @rdname random_continuous
#' @export
random_triangular <- function(key, left, mode, right, shape = NULL, dtype = NULL,
                              out_sharding = NULL) {
  key <- rnd_check_key("triangular", key)$key
  dt <- rnd_float_dtype("triangular", dtype)
  shape <- rnd_check_broadcast_shapes("triangular", shape, left, mode, right)
  rnd_check_safe_cast("triangular", dt, left, mode, right)
  rnd_triangular(key, left, mode, right, shape, dt)
}

# Port of core.py:_triangular.
rnd_triangular <- prng_jit(function(key, left, mode, right, shape, dtype) {
  left <- jnp_broadcast_to(lax_convert_element_type(left, dtype), shape)
  right <- jnp_broadcast_to(lax_convert_element_type(right, dtype), shape)
  mode <- jnp_broadcast_to(lax_convert_element_type(mode, dtype), shape)
  fc <- (mode - left) / (right - left)
  u <- random_uniform(key, shape, dtype)
  out1 <- left + lax_sqrt(u * (right - left) * (mode - left))
  out2 <- right - lax_sqrt((1 - u) * (right - left) * (right - mode))
  lax_select(u < fc, out1, out2)
}, static_argnums = 4:5)

#' @rdname random_continuous
#' @export
random_generalized_normal <- function(key, p, shape = integer(0), dtype = NULL,
                                      out_sharding = NULL) {
  shape <- canon_shape(shape)
  key <- rnd_check_key("generalized_normal", key)$key
  dt <- dtype(dtype %||% "float")
  rnd_generalized_normal(key, p, shape, dt)
}

# Port of core.py:_generalized_normal.
rnd_generalized_normal <- prng_jit(function(key, p, shape, dtype) {
  ks <- as.list.jax_prng_key(random_split(key))
  g <- random_gamma(ks[[1L]], 1 / p, shape, dtype)
  r <- random_rademacher(ks[[2L]], shape, dtype)
  r * jnp_power(g, 1 / p)
}, static_argnums = 2:3)

#' @rdname random_continuous
#' @export
random_ball <- function(key, d, p = 2, shape = integer(0), dtype = NULL, out_sharding = NULL) {
  shape <- canon_shape(shape)
  key <- rnd_check_key("ball", key)$key
  dt <- dtype(dtype %||% "float")
  d <- as.integer(as.numeric(concrete_or_error(as_jax(d), "The error occurred in jax.random.ball()")))
  rnd_ball(key, p, d, shape, dt)
}

# Port of core.py:_ball.
rnd_ball <- prng_jit(function(key, p, d, shape, dtype) {
  ks <- as.list.jax_prng_key(random_split(key))
  g <- random_generalized_normal(ks[[1L]], p, c(shape, d), dtype)
  e <- random_exponential(ks[[2L]], shape, dtype)
  g / lax_expand_dims(jnp_power(jnp_sum(jnp_power(jnp_abs(g), p), axis = -1L) + e, 1 / p),
                      length(shape))
}, static_argnums = 2:4)

#' @rdname random_continuous
#' @export
random_orthogonal <- function(key, n, shape = integer(0), dtype = NULL, m = NULL,
                              out_sharding = NULL) {
  m <- m %||% n
  shape <- canon_shape(shape)
  key <- rnd_check_key("orthogonal", key)$key
  dt <- dtype(dtype %||% "float")
  n <- as.integer(as.numeric(concrete_or_error(as_jax(n), "The error occurred in jax.random.orthogonal()")))
  m <- as.integer(as.numeric(concrete_or_error(as_jax(m), "The error occurred in jax.random.orthogonal()")))
  rnd_orthogonal(key, n, m, shape, dt)
}

# Port of core.py:_orthogonal.
rnd_orthogonal <- prng_jit(function(key, n, m, shape, dtype) {
  z <- random_normal(key, c(shape, max(n, m), min(n, m)), dtype)
  qr <- jnp_linalg_qr(z)
  d <- jnp_linalg_diagonal(qr[[2L]])
  x <- qr[[1L]] * lax_expand_dims(jnp_sign(d), rnd_ndim(d) - 1L)
  if (n < m) jnp_swapaxes(x, -1L, -2L) else x
}, static_argnums = 1:4)
