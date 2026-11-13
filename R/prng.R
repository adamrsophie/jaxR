# PRNG implementations and key arrays: port of jax/_src/random/prng.py,
# jax/_src/random/threefry2x32.py, jax/_src/random/rbg.py and of the rng_bit_generator
# and rng_uniform primitives of jax/_src/lax/lax.py.
#
# A typed key array (jax.random.key) is a classed list holding the key data, a uint32
# array whose trailing dimensions are the key shape of the implementation, and the
# implementation name. Each implementation has its own class (jax_key_<impl>), which is
# registered as a pytree node, so keys flow through jit, vmap, grad and control flow and
# jit caches distinguish implementations with equal key shapes. The key data is integer,
# so keys have float0 (symbolic zero) tangents.

prng_uint_dtypes <- c("8" = "uint8", "16" = "uint16", "32" = "uint32", "64" = "uint64")

# uint32 constant from R numbers (host data, exact).
prng_u32 <- function(v, shape = integer(0)) {
  new_array(new_aval(shape, "uint32"), host = with_shape(as.double(v), shape))
}

# Port of lax._const: a strongly typed scalar with the dtype of x.
prng_const <- function(x, v) lax_full_like(x, v, dtype = aval_of(x)$dtype, shape = integer(0))

# A strongly typed scalar constant, np.array(v, dtype).
prng_full <- function(v, dtype) lax_full_like(0L, v, dtype = dtype, shape = integer(0))

prng_shape_str <- function(s) {
  if (length(s) == 1L) return(paste0("(", s, ",)"))
  paste0("(", paste(s, collapse = ", "), ")")
}

prng_unpack2 <- function(key) {
  list(lax_index_in_dim(key, 0L, 0L, keepdims = FALSE),
       lax_index_in_dim(key, 1L, 0L, keepdims = FALSE))
}

# jit whose cache is also keyed on jax_threefry_partitionable (JAX's jit cache includes
# that option; the jaxR jit cache keys on x64 only).
prng_jit <- function(fun, static_argnums = integer(0)) {
  force(fun)
  j <- jit(function(partitionable, ...) fun(...), static_argnums = c(0L, static_argnums + 1L))
  function(...) {
    old <- .jx$kr_prim
    .jx$kr_prim <- TRUE
    on.exit(.jx$kr_prim <- old)
    j(isTRUE(.jx$config$jax_threefry_partitionable), ...)
  }
}

# ---- threefry2x32 primitive (threefry2x32.py) -------------------------------------------

# Port of threefry2x32.py:_make_rotate_left for uint32.
prng_rotate_left <- function(x, d) {
  lax_bitwise_or(lax_shift_left(x, d), lax_shift_right_logical(x, lax_sub(prng_u32(32), d)))
}

# Port of threefry2x32.py:apply_round.
prng_apply_round <- function(v, rot) {
  v[[1L]] <- jnp_add(v[[1L]], v[[2L]])
  v[[2L]] <- prng_rotate_left(v[[2L]], rot)
  v[[2L]] <- jnp_bitwise_xor(v[[1L]], v[[2L]])
  v
}

prng_rotate_list <- function(xs) c(drop_first(xs, 1L), xs[1L])

# Port of threefry2x32.py:rolled_loop_step.
prng_rolled_loop_step <- function(i, state) {
  x <- state[[1L]]
  ks <- state[[2L]]
  rotations <- state[[3L]]
  for (j in 0:3) x <- prng_apply_round(x, lax_index_in_dim(rotations[[1L]], j, 0L, keepdims = FALSE))
  step <- lax_convert_element_type(lax_add(i, 1L), "uint32")
  new_x <- list(jnp_add(x[[1L]], ks[[1L]]), jnp_add(jnp_add(x[[2L]], ks[[2L]]), step))
  list(new_x, prng_rotate_list(ks), prng_rotate_list(rotations))
}

# Port of threefry2x32.py:_threefry2x32_lowering with use_rolled_loops=True, the
# lowering JAX registers for CPU.
prng_threefry2x32_lowering <- function(args, params) {
  key1 <- args[[1L]]
  key2 <- args[[2L]]
  rotations <- list(prng_u32(c(13, 15, 26, 6), 4L), prng_u32(c(17, 29, 16, 24), 4L))
  ks <- list(key1, key2, lax_bitwise_xor(lax_bitwise_xor(key1, key2), prng_u32(0x1BD11BDA)))
  x <- list(jnp_add(args[[3L]], ks[[1L]]), jnp_add(args[[4L]], ks[[2L]]))
  out <- lax_fori_loop(0L, 5L, prng_rolled_loop_step, list(x, prng_rotate_list(ks), rotations))
  out[[1L]]
}

threefry2x32_p <- new_primitive("threefry2x32", multiple_results = TRUE)
# Port of threefry2x32.py:_threefry2x32_abstract_eval.
threefry2x32_p$abstract_eval <- function(avals, params) {
  if (any(vapply(avals, function(a) a$dtype != "uint32", NA)))
    jax_error("Arguments to threefry2x32 must have uint32 type, got ",
              paste(vapply(avals, aval_str, ""), collapse = ", "), class = "jax_type_error")
  a <- new_aval(broadcasting_shape_rule("threefry2x32", avals), "uint32")
  list(a, a)
}
threefry2x32_p$lowering <- lower_fun(prng_threefry2x32_lowering, multiple_results = TRUE)
# batching.defbroadcasting(threefry2x32_p)
threefry2x32_p$batch <- lax_broadcast_batcher(threefry2x32_p)

# Port of threefry2x32.py:threefry_2x32 (the Threefry 2x32 hash of `count`).
prng_threefry_2x32 <- function(keypair, count) {
  kp <- if (is.list(keypair) && is.null(oldClass(keypair))) keypair else prng_unpack2(keypair)
  count <- as_jax(count)
  dts <- vapply(list(kp[[1L]], kp[[2L]], count), function(v) aval_of(v)$dtype, "")
  if (!all(dts == "uint32"))
    jax_error("threefry_2x32 requires uint32 arguments, got [", paste(dts, collapse = ", "), "]",
              class = "jax_type_error")
  shape <- aval_of(count)$shape
  n <- prod_int(shape)
  flat <- lax_reshape(count, n)
  odd <- n %% 2L
  if (odd) flat <- lax_concatenate(list(flat, prng_u32(0, 1L)), 0L)
  half <- (n + odd) %/% 2L
  x <- bind(threefry2x32_p, list(kp[[1L]], kp[[2L]], lax_slice(flat, 0L, half),
                                 lax_slice(flat, half, 2L * half)))
  out <- lax_concatenate(x, 0L)
  if (odd) out <- lax_slice(out, 0L, n)
  lax_reshape(out, shape)
}

# ---- iota_2x32_shape (prng.py) -----------------------------------------------------------

iota_2x32_shape_p <- new_primitive("iota_2x32_shape", multiple_results = TRUE)
iota_2x32_shape_p$abstract_eval <- function(avals, params) {
  a <- new_aval(params$shape, "uint32")
  list(a, a)
}
# Port of prng.py:iota_2x32_shape_lowering.
iota_2x32_shape_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  shape <- params$shape
  a64 <- new_aval(shape, "uint64")
  t64 <- tensor_type(a64)
  strides <- c(rev(cumprod(rev(drop_first(shape, 1L)))), 1)
  terms <- lapply(seq_along(shape), function(d) {
    iota <- emit_op(ctx, "stablehlo.iota", character(0), character(0), t64,
                    attrs = list(iota_dimension = attr_i64(d - 1L)))
    s <- full_constant(ctx, strides[d], a64)
    emit_op(ctx, "stablehlo.multiply", c(s, iota), c(t64, t64), t64)
  })
  counts <- Reduce(function(a, b) emit_op(ctx, "stablehlo.add", c(a, b), c(t64, t64), t64), terms)
  shift <- full_constant(ctx, 32, a64)
  shifted <- emit_op(ctx, "stablehlo.shift_right_logical", c(counts, shift), c(t64, t64), t64)
  t32 <- tensor_type(avals_out[[1L]])
  lo <- emit_op(ctx, "stablehlo.convert", counts, t64, t32)
  hi <- emit_op(ctx, "stablehlo.convert", shifted, t64, t32)
  c(hi, lo)
}

# Port of prng.py:iota_2x32_shape: the uint64 iota of `shape` as (hi, lo) uint32 halves.
prng_iota_2x32_shape <- function(shape) {
  shape <- as.integer(shape)
  if (!length(shape)) return(list(prng_u32(0), prng_u32(0)))
  bind(iota_2x32_shape_p, list(), list(shape = shape))
}

# ---- threefry2x32 PRNG implementation (threefry2x32.py) -----------------------------------

# Port of threefry2x32.py:_threefry_seed.
prng_threefry_seed <- jit(function(seed) {
  a <- aval_of(seed)
  if (length(a$shape))
    jax_error("PRNG key seed must be a scalar; got ", format(seed), ".", class = "jax_type_error")
  if (!is_int_dtype(a$dtype))
    jax_error("PRNG key seed must be an integer; got ", format(seed), class = "jax_type_error")
  convert <- function(k) lax_expand_dims(lax_convert_element_type(k, "uint32"), 0L)
  k1 <- convert(lax_shift_right_logical(seed, prng_const(seed, 32)))
  k2 <- convert(jnp_bitwise_and(seed, prng_u32(0xFFFFFFFF)))
  lax_concatenate(list(k1, k2), 0L)
})

# Port of threefry2x32.py:_threefry_split_original.
prng_threefry_split_original <- function(key, shape) {
  num <- prod_int(shape)
  counts <- lax_iota("uint32", num * 2L)
  lax_reshape(prng_threefry_2x32(key, counts), c(shape, 2L))
}

# Port of threefry2x32.py:_threefry_split_foldlike.
prng_threefry_split_foldlike <- function(key, shape) {
  k <- prng_unpack2(key)
  counts <- prng_iota_2x32_shape(shape)
  bits <- bind(threefry2x32_p, list(k[[1L]], k[[2L]], counts[[1L]], counts[[2L]]))
  jnp_stack(bits, axis = length(aval_of(bits[[1L]])$shape))
}

# Port of threefry2x32.py:_threefry_split.
prng_threefry_split <- function(key, shape) {
  if (isTRUE(.jx$config$jax_threefry_partitionable)) prng_threefry_split_foldlike(key, shape)
  else prng_threefry_split_original(key, shape)
}

# Port of threefry2x32.py:_threefry_fold_in (data is a uint32 scalar).
prng_threefry_fold_in <- function(key, data) prng_threefry_2x32(key, prng_threefry_seed(data))

prng_check_bit_width <- function(bit_width) {
  if (!bit_width %in% c(8L, 16L, 32L, 64L))
    jax_error("requires 8-, 16-, 32- or 64-bit field width.", class = "jax_type_error")
}

# Port of threefry2x32.py:threefry_random_bits.
prng_threefry_random_bits <- function(key, bit_width, shape) {
  a <- aval_of(key)
  if (!identical(a$shape, 2L) || a$dtype != "uint32")
    jax_error("threefry_random_bits got invalid prng key.", class = "jax_type_error")
  prng_check_bit_width(bit_width)
  if (isTRUE(.jx$config$jax_threefry_partitionable))
    prng_threefry_random_bits_partitionable(key, bit_width, shape)
  else prng_threefry_random_bits_original(key, bit_width, shape)
}

# Port of threefry2x32.py:_threefry_random_bits_partitionable.
prng_threefry_random_bits_partitionable <- function(key, bit_width, shape) {
  if (prod(as.numeric(shape)) > 2^64)
    jax_error("random bits array of size exceeding 2 ** 64", class = "jax_not_implemented_error")
  k <- prng_unpack2(key)
  counts <- prng_iota_2x32_shape(shape)
  bits <- bind(threefry2x32_p, list(k[[1L]], k[[2L]], counts[[1L]], counts[[2L]]))
  dt <- prng_uint_dtypes[[as.character(bit_width)]]
  if (bit_width == 64L) {
    hi <- lax_convert_element_type(bits[[1L]], dt)
    lo <- lax_convert_element_type(bits[[2L]], dt)
    return(lax_bitwise_or(lax_shift_left(hi, prng_const(hi, 32)), lo))
  }
  x <- lax_bitwise_xor(bits[[1L]], bits[[2L]])
  if (bit_width == 32L) x else lax_convert_element_type(x, dt)
}

# Port of threefry2x32.py:_threefry_random_bits_original.
prng_threefry_random_bits_original <- function(key, bit_width, shape) {
  size <- prod(as.numeric(shape))
  max_count <- ceiling(bit_width * size / 32)
  u32max <- 2^32 - 1
  nblocks <- max_count %/% u32max
  rem <- max_count %% u32max
  if (!nblocks) {
    bits <- prng_threefry_2x32(key, lax_iota("uint32", rem))
  } else {
    keys <- prng_threefry_split(key, nblocks + 1)
    subkeys <- lax_slice_in_dim(keys, 0L, nblocks, 1L, 0L)
    last_key <- lax_index_in_dim(keys, nblocks, 0L, keepdims = FALSE)
    blocks <- vmap(prng_threefry_2x32, in_axes = list(0L, NULL))(subkeys, lax_iota("uint32", u32max))
    last <- prng_threefry_2x32(last_key, lax_iota("uint32", rem))
    bits <- lax_concatenate(list(lax_reshape(blocks, prod_int(aval_of(blocks)$shape)), last), 0L)
  }
  dt <- prng_uint_dtypes[[as.character(bit_width)]]
  if (bit_width == 64L) {
    h <- max_count %/% 2
    parts <- lapply(list(lax_slice(bits, 0L, h), lax_slice(bits, h, max_count)),
                    lax_convert_element_type, new_dtype = dt)
    bits <- lax_bitwise_or(lax_shift_left(parts[[1L]], prng_const(parts[[1L]], 32)), parts[[2L]])
  } else if (bit_width %in% c(8L, 16L)) {
    # This is essentially bits.view(dtype)[:size].
    per <- 32L %/% bit_width
    sh <- lax_mul(prng_u32(bit_width), lax_broadcasted_iota("uint32", c(per, 1L), 0L))
    bits <- lax_bitwise_and(prng_u32(2^bit_width - 1),
                            lax_shift_right_logical(lax_broadcast(bits, 1L), sh))
    bits <- lax_reshape(bits, max_count * per, dimensions = c(1L, 0L))
    bits <- lax_slice(lax_convert_element_type(bits, dt), 0L, size)
  }
  lax_reshape(bits, shape)
}

# ---- rng_bit_generator and rng_uniform (lax.py) ------------------------------------------

#' Random algorithms of rng_bit_generator
#'
#' Port of `jax.lax.RandomAlgorithm`: `RNG_DEFAULT` (the platform default),
#' `RNG_THREE_FRY` and `RNG_PHILOX`.
#' @format A named integer vector.
#' @export
lax_RandomAlgorithm <- c(RNG_DEFAULT = 0L, RNG_THREE_FRY = 1L, RNG_PHILOX = 2L)

rng_bit_generator_p <- new_primitive("rng_bit_generator", multiple_results = TRUE)
rng_bit_generator_p$abstract_eval <- function(avals, params) {
  k <- avals[[1L]]
  list(new_aval(k$shape, k$dtype, k$weak), new_aval(params$shape, params$dtype))
}
# Port of lax.py:_rng_bit_generator_lowering. The u32[4] key is bitcast to the u64[2]
# key of RngBitGenerator.
rng_bit_generator_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  ka <- avals_in[[1L]]
  u32 <- ka$dtype == "uint32"
  if (!((u32 && identical(ka$shape, 4L)) || (ka$dtype == "uint64" && identical(ka$shape, 2L))))
    jax_error("rng_bit_generator: key must be uint32[4] or uint64[2], got ", aval_str(ka))
  kt <- tensor_type(ka)
  t22 <- "tensor<2x2xui32>"
  t64 <- "tensor<2xui64>"
  key <- ins[[1L]]
  if (u32) key <- emit_op(ctx, "stablehlo.bitcast_convert",
                          emit_op(ctx, "stablehlo.reshape", key, kt, t22), t22, t64)
  alg <- c("DEFAULT", "THREE_FRY", "PHILOX")[params$algorithm + 1L]
  res <- emit_op(ctx, "stablehlo.rng_bit_generator", key, t64, c(t64, tensor_type(avals_out[[2L]])),
                 attrs = list(rng_algorithm = paste0("#stablehlo<rng_algorithm ", alg, ">")))
  out_key <- res[1L]
  if (u32) out_key <- emit_op(ctx, "stablehlo.reshape",
                              emit_op(ctx, "stablehlo.bitcast_convert", out_key, t64, t22), t22, kt)
  c(out_key, res[2L])
}
# Port of lax/control_flow/loops.py:_rng_bit_generator_batching_rule (all lanes draw from
# the first key, as in JAX).
rng_bit_generator_p$batch <- function(vals, dims, params, axis) {
  keys <- vals[[1L]]
  if (is.na(dims[1L])) return(list(bind(rng_bit_generator_p, vals, params), c(NA_integer_, NA_integer_)))
  keys <- moveaxis(keys, dims[1L], 0L)
  b <- aval_of(keys)$shape[1L]
  key <- lax_index_in_dim(keys, 0L, 0L, keepdims = FALSE)
  out <- bind(rng_bit_generator_p, list(key), update_params(params, list(shape = c(b, params$shape))))
  new_keys <- lax_dynamic_update_slice(keys, lax_expand_dims(out[[1L]], 0L),
                                       rep(0L, length(aval_of(keys)$shape)))
  list(list(new_keys, out[[2L]]), c(0L, 0L))
}

#' Stateless random bit generator
#'
#' Port of `jax.lax.rng_bit_generator`: uniformly distributed random bits from the XLA
#' `RngBitGenerator` operation. The output is not guaranteed to be stable across
#' backends or compiler versions; most users should use the `random_*` functions.
#' @param key A `uint32[4]` (or `uint64[2]`) key.
#' @param shape Integer vector, the shape of the bits.
#' @param dtype An unsigned integer dtype.
#' @param algorithm One of [lax_RandomAlgorithm].
#' @param out_sharding Ignored (one device).
#' @return `list(new_key, bits)`.
#' @export
lax_rng_bit_generator <- function(key, shape, dtype = "uint32",
                                  algorithm = lax_RandomAlgorithm[["RNG_DEFAULT"]],
                                  out_sharding = NULL) {
  shape <- as.integer(shape)
  dt <- dtype(dtype)
  if (!dt %in% prng_uint_dtypes)
    jax_error("rng_bit_generator: unsupported dtype ", dt, class = "jax_type_error")
  bind(rng_bit_generator_p, list(as_jax(key)),
       list(shape = shape, dtype = dt, algorithm = as.integer(algorithm)))
}

#' @rdname lax_rng_bit_generator
#' @format NULL
#' @export
lax_rng_bit_generator_p <- rng_bit_generator_p

rng_uniform_p <- new_primitive("rng_uniform")
# Port of lax.py:_rng_uniform_abstract_eval.
rng_uniform_p$abstract_eval <- function(avals, params) {
  a <- avals[[1L]]
  b <- avals[[2L]]
  if (a$dtype != b$dtype)
    jax_error("Arguments to rng_uniform must have identical dtypes, got ", a$dtype, " and ",
              b$dtype, ".", class = "jax_value_error")
  if (length(a$shape) || length(b$shape))
    jax_error("Arguments to rng_uniform must be scalars; got shapes ", prng_shape_str(a$shape),
              " and ", prng_shape_str(b$shape), ".", class = "jax_value_error")
  new_aval(params$shape, a$dtype, a$weak && b$weak)
}
rng_uniform_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  s <- ir_constant(ctx, params$shape, new_aval(length(params$shape), "int64"))
  emit_op(ctx, "stablehlo.rng", c(ins, s),
          c(tensor_type(avals_in[[1L]]), tensor_type(avals_in[[2L]]),
            tensor_type(new_aval(length(params$shape), "int64"))),
          tensor_type(avals_out[[1L]]),
          attrs = list(rng_distribution = "#stablehlo<rng_distribution UNIFORM>"))
}

#' Stateful uniform random numbers
#'
#' Port of `jax.lax.rng_uniform`: uniform numbers in `[a, b)` from the XLA `Rng`
#' operation. Experimental and nondeterministic; use [random_uniform()] instead.
#' @param a,b Scalars of the same dtype.
#' @param shape Integer vector.
#' @return An array of shape `shape`.
#' @export
lax_rng_uniform <- function(a, b, shape) {
  args <- lax_args(a, b)
  bind(rng_uniform_p, args, list(shape = as.integer(shape)))
}

#' @rdname lax_rng_uniform
#' @format NULL
#' @export
lax_rng_uniform_p <- rng_uniform_p

# ---- rbg and unsafe_rbg PRNG implementations (rbg.py) -------------------------------------

# Port of rbg.py:_rbg_seed.
prng_rbg_seed <- function(seed) {
  halfkey <- prng_threefry_seed(seed)
  lax_concatenate(list(halfkey, halfkey), 0L)
}

# Port of rbg.py:_rbg_split.
prng_rbg_split <- function(key, shape) {
  sp <- if (isTRUE(.jx$config$jax_threefry_partitionable)) prng_threefry_split_foldlike
        else prng_threefry_split_original
  halfkeys <- lax_reshape(key, c(2L, 2L))
  out <- vmap(function(k) sp(k, shape), in_axes = 0L, out_axes = length(shape))(halfkeys)
  lax_reshape(out, c(shape, 4L))
}

# Port of rbg.py:_rbg_fold_in.
prng_rbg_fold_in <- function(key, data) {
  out <- vmap(function(k) prng_threefry_fold_in(k, data), in_axes = 0L)(lax_reshape(key, c(2L, 2L)))
  lax_reshape(out, 4L)
}

# Port of rbg.py:_rbg_random_bits.
prng_rbg_random_bits <- function(key, bit_width, shape) {
  a <- aval_of(key)
  if (!identical(a$shape, 4L) || a$dtype != "uint32")
    jax_error("_rbg_random_bits got invalid prng key.", class = "jax_type_error")
  prng_check_bit_width(bit_width)
  lax_rng_bit_generator(key, shape, dtype = prng_uint_dtypes[[as.character(bit_width)]])[[2L]]
}

# Port of rbg.py:_unsafe_rbg_split.
prng_unsafe_rbg_split <- function(key, shape) {
  num <- prod_int(shape)
  keys <- lax_rng_bit_generator(key, c(10L * num, 4L), dtype = "uint32")[[2L]]
  keys <- lax_slice_in_dim(keys, 0L, 10L * num, 10L, 0L)
  lax_reshape(keys, c(shape, 4L))
}

# Port of rbg.py:_unsafe_rbg_fold_in.
prng_unsafe_rbg_fold_in <- function(key, data) {
  bits <- lax_rng_bit_generator(prng_rbg_seed(data), c(10L, 4L), dtype = "uint32")[[2L]]
  lax_bitwise_xor(key, lax_index_in_dim(bits, 9L, 0L, keepdims = FALSE))
}

# ---- implementation registry (prng.py:PRNGImpl, register_prng) ----------------------------

prng_impls <- list(
  threefry2x32 = list(key_shape = 2L, seed = prng_threefry_seed, split = prng_threefry_split,
                      random_bits = prng_threefry_random_bits, fold_in = prng_threefry_fold_in,
                      name = "threefry2x32", tag = "fry"),
  rbg = list(key_shape = 4L, seed = prng_rbg_seed, split = prng_rbg_split,
             random_bits = prng_rbg_random_bits, fold_in = prng_rbg_fold_in,
             name = "rbg", tag = "rbg"),
  unsafe_rbg = list(key_shape = 4L, seed = prng_rbg_seed, split = prng_unsafe_rbg_split,
                    random_bits = prng_rbg_random_bits, fold_in = prng_unsafe_rbg_fold_in,
                    name = "unsafe_rbg", tag = "urbg")
)

# Port of random/core.py:resolve_prng_impl. Accepts an implementation name or a key
# dtype name such as "key<fry>".
prng_resolve_impl <- function(impl_spec) {
  if (is.null(impl_spec)) impl_spec <- .jx$config$jax_default_prng_impl
  if (!is.character(impl_spec) || length(impl_spec) != 1L)
    jax_error("unrecognized type ", class(impl_spec)[1L], " for specifying PRNG implementation.",
              class = "jax_type_error")
  if (!is.null(prng_impls[[impl_spec]])) return(prng_impls[[impl_spec]])
  for (impl in prng_impls) if (identical(impl_spec, paste0("key<", impl$tag, ">"))) return(impl)
  jax_error("unrecognized PRNG implementation \"", impl_spec, "\". Did you mean one of: ",
            paste0("\"", names(prng_impls), "\"", collapse = ", "), "?", class = "jax_value_error")
}

# ---- key arrays (prng.py:PRNGKeyArray, random_wrap, random_seed, ...) ---------------------

prng_key_array <- function(data, impl) {
  kr_init(data)
  structure(list(data = data), impl = impl, class = c(paste0("jax_key_", impl), "jax_prng_key"))
}

is_prng_key <- function(x) inherits(x, "jax_prng_key")

prng_key_impl <- function(keys) prng_impls[[attr(keys, "impl", exact = TRUE)]]

# Shape of a key array: the key data shape without the trailing key dimensions.
prng_keys_shape <- function(keys) {
  s <- aval_of(keys$data)$shape
  s[seq_len(length(s) - length(prng_key_impl(keys)$key_shape))]
}

invisible(lapply(names(prng_impls), function(nm) {
  register_pytree_node(paste0("jax_key_", nm),
                       function(x) list(children = list(x$data), aux = attr(x, "impl", exact = TRUE)),
                       function(aux, children) prng_key_array(children[[1L]], aux))
}))

# Port of prng.py:_check_prng_key_data.
prng_check_key_data <- function(impl, key_data) {
  a <- aval_of(key_data)
  nd <- length(impl$key_shape)
  if (length(a$shape) < 1L)
    jax_error("JAX encountered invalid PRNG key data: expected key_data.ndim >= 1; got ndim=",
              length(a$shape), class = "jax_type_error")
  if (length(a$shape) < nd || !identical(utils::tail(a$shape, nd), impl$key_shape))
    jax_error("JAX encountered invalid PRNG key data: expected key_data.shape to end with ",
              prng_shape_str(impl$key_shape), "; got shape=", prng_shape_str(a$shape),
              " for impl=", impl$name, class = "jax_type_error")
  if (!a$dtype %in% c("uint32", "float0"))
    jax_error("JAX encountered invalid PRNG key data: expected key_data.dtype = uint32; got dtype=",
              a$dtype, class = "jax_type_error")
}

# Port of prng.py:random_wrap.
prng_random_wrap <- function(base_arr, impl) {
  base_arr <- as_jax(base_arr)
  prng_check_key_data(impl, base_arr)
  prng_key_array(kr_copy(base_arr), impl$name)
}

# Port of prng.py:random_seed. An R number is treated as a Python int: converted to
# int64, which wraps to int32 when x64 is disabled. The option jax_random_seed_offset,
# if set, is added to the seed.
prng_random_seed <- function(seed, impl) {
  if (!is_jax_value(seed) && is.numeric(seed) && length(seed) == 1L && !is.na(seed) &&
      seed == round(seed)) {
    v <- as.numeric(seed)
    dt <- canonicalize_dtype("int64")
    if (dt == "int32") {
      v <- v %% 2^32
      if (v >= 2^31) v <- v - 2^32
    }
    seed <- new_array(new_aval(integer(0), dt), host = v)
  }
  seed <- as_jax(seed)
  offset <- .jx$config$jax_random_seed_offset
  if (!is.null(offset) && offset != 0) seed <- seed + as.integer(offset)
  prng_key_array(impl$seed(seed), impl$name)
}

# Key operations on single keys (the public functions take one key; batched keys come
# from vmap). These are jitted, as threefry2x32.py jits its implementation functions.
prng_split_jit <- prng_jit(function(key, shape) {
  kr_sink(key, "random_split")
  prng_key_array(prng_key_impl(key)$split(key$data, shape), attr(key, "impl", exact = TRUE))
}, static_argnums = 1L)

prng_fold_in_jit <- prng_jit(function(key, data) {
  prng_key_array(prng_key_impl(key)$fold_in(key$data, data), attr(key, "impl", exact = TRUE))
})

prng_bits_jit <- prng_jit(function(key, bit_width, shape) {
  kr_sink(key, "random_bits")
  prng_key_impl(key)$random_bits(key$data, bit_width, shape)
}, static_argnums = 1:2)

prng_random_split <- function(key, shape) prng_split_jit(key, as.integer(shape))
prng_random_fold_in <- function(key, data) prng_fold_in_jit(key, data)
prng_random_bits <- function(key, bit_width, shape) {
  prng_bits_jit(key, as.integer(bit_width), as.integer(shape))
}

#' @export
format.jax_prng_key <- function(x, ...) {
  paste0("Array(", prng_shape_str(prng_keys_shape(x)), ", dtype=key<", prng_key_impl(x)$tag, ">)")
}

#' @export
print.jax_prng_key <- function(x, ...) {
  cat(format(x), " overlaying:\n", sep = "")
  print(x$data)
  invisible(x)
}

#' @export
dim.jax_prng_key <- function(x) prng_keys_shape(x)

#' Index and iterate key arrays
#'
#' `keys[i]` indexes the leading (key array) dimensions of a typed key array with the
#' rules of `[` for jax arrays; `as.list(keys)` splits a key array along its first
#' dimension (port of `PRNGKeyArray.__iter__`).
#' @param x A key array from [random_key()] or [random_split()].
#' @param ... Index arguments.
#' @return A key array, or a list of key arrays.
#' @name key_array_indexing
#' @export
`[.jax_prng_key` <- function(x, ...) {
  kr_slice(x, prng_key_array(x$data[..., ], attr(x, "impl", exact = TRUE)), ...)
}

#' @rdname key_array_indexing
#' @export
as.list.jax_prng_key <- function(x, ...) {
  s <- prng_keys_shape(x)
  if (!length(s)) jax_error("iteration over a 0-d key array", class = "jax_type_error")
  # Unstacking consumes the key array and returns new keys (_unstack_signature).
  if (kr_on()) kr_consume(x$data, TRUE, "unstack", 0L)
  lapply(seq_len(s[1L]) - 1L, function(i) {
    k <- prng_key_array(lax_index_in_dim(x$data, i, 0L, keepdims = FALSE), attr(x, "impl", exact = TRUE))
    if (kr_on()) kr_set(k$data, FALSE)
    k
  })
}
