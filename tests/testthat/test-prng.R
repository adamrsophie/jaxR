# Keys, threefry2x32, split, fold_in, bits and the rbg implementations (R/prng.R).
# Expected values were produced by jax 0.12.0.1 on CPU; arrays are listed in C order as
# list(shape, values, dtype).

rp_c_ravel <- function(x) {
  d <- dim(x)
  if (length(d) <= 1L) return(as.vector(x))
  as.vector(aperm(x, length(d):1))
}

rp_expect <- function(x, ref) {
  expect_identical(jaxR::dtype(x), ref[[3L]])
  expect_identical(as.integer(dim(x)), as.integer(ref[[1L]]))
  expect_identical(as.numeric(rp_c_ravel(as.array(x))), as.numeric(ref[[2L]]))
}

# Compare a uint64 array through its (hi, lo) uint32 halves.
rp_expect_u64 <- function(x, shape, hi, lo) {
  expect_identical(jaxR::dtype(x), "uint64")
  expect_identical(as.integer(dim(x)), as.integer(shape))
  h <- lax_convert_element_type(lax_shift_right_logical(x, lax_full(integer(0), 32, "uint64")), "uint32")
  expect_identical(as.numeric(rp_c_ravel(as.array(h))), as.numeric(hi))
  expect_identical(as.numeric(rp_c_ravel(as.array(lax_convert_element_type(x, "uint32")))), as.numeric(lo))
}

rp_ref <- list(
  PRNGKey_0 = list(c(2), c(0, 0), "uint32"),
  PRNGKey_42 = list(c(2), c(0, 42), "uint32"),
  PRNGKey_m1 = list(c(2), c(0, 4294967295), "uint32"),
  PRNGKey_big = list(c(2), c(0, 5), "uint32"),
  split_0 = list(c(2, 2), c(1797259609, 2579123966, 928981903, 3453687069), "uint32"),
  split_42_3 = list(c(3, 2), c(1832780943, 270669613, 64467757, 2916123636, 2465931498, 255383827), "uint32"),
  split_0_23 = list(c(2, 3, 2), c(1797259609, 2579123966, 928981903, 3453687069, 4146024105, 2718843009, 2467461003, 3840466878, 2285895361, 433833334, 1524306142, 1887795613), "uint32"),
  fold_in_0_7 = list(c(2), c(2716826189, 292468403), "uint32"),
  fold_in_42_max = list(c(2), c(2398536845, 3890976714), "uint32"),
  bits_0_5 = list(c(5), c(4070199207, 4202968722, 1427181096, 2012915765, 2447653815), "uint32"),
  bits_42_23 = list(c(2, 3), c(2098992034, 2919706841, 2646866425, 2409546199, 1935504149, 2516274904), "uint32"),
  bits_0_u8 = list(c(7), c(167, 146, 40, 53, 183, 67, 125), "uint8"),
  bits_0_u16 = list(c(5), c(20391, 13970, 3624, 43061, 15287), "uint16"),
  np_split_0 = list(c(2, 2), c(4146024105, 967050713, 2718843009, 1272950319), "uint32"),
  np_split_42_3 = list(c(3, 2), c(3134548294, 3733159049, 3746501087, 894150801, 801545058, 2363201431), "uint32"),
  np_bits_0_5 = list(c(5), c(2467461003, 428148500, 1688610540, 3840466878, 2562233961), "uint32"),
  np_bits_0_u8 = list(c(7), c(143, 35, 95, 55, 29, 21, 219), "uint8"),
  np_bits_0_u16 = list(c(5), c(20137, 63263, 64300, 20622, 16513), "uint16"),
  vmap_split = list(c(3, 2, 2), c(4165894930, 804218099, 1353695780, 2116000888, 346279018, 360566543, 3968330031, 3923691647, 2799984767, 1105366846, 3777617834, 145086855), "uint32"),
  vmap_fold_in = list(c(3, 2), c(1797259609, 2579123966, 928981903, 3453687069, 4146024105, 2718843009), "uint32"),
  vmap_key = list(c(3, 2), c(0, 0, 0, 1, 0, 2), "uint32"),
  rbg_split = list(c(3, 4), c(1797259609, 2579123966, 1797259609, 2579123966, 928981903, 3453687069, 928981903, 3453687069, 4146024105, 2718843009, 4146024105, 2718843009), "uint32"),
  rbg_fold_in = list(c(4), c(1524306142, 1887795613, 1524306142, 1887795613), "uint32"),
  rbg_bits = list(c(4), c(1713891541, 3781805453, 3159862348, 2600524760), "uint32"),
  unsafe_rbg_split = list(c(3, 4), c(1713891541, 3781805453, 3159862348, 2600524760, 3614953066, 3503513258, 724076422, 3372229774, 1393889421, 2738237077, 2849386114, 3441723459), "uint32"),
  unsafe_rbg_fold_in = list(c(4), c(3810769997, 2355009021, 1837142617, 748193198), "uint32"),
  rbg_bits_u8 = list(c(5), c(211, 175, 250, 104, 232), "uint8"),
  rbg_vmap_bits = list(c(2, 3), c(132991142, 1711370325, 2756298552, 622686569, 1540221606, 2998426742), "uint32")
)

test_that("threefry2x32 matches the known answer vectors", {
  skip_if_no_backend()
  tf <- get("prng_threefry_2x32", asNamespace("jaxR"))
  u <- function(...) jnp_asarray(c(...), dtype = "uint32")
  rp_expect(tf(u(0, 0), u(0, 0)), list(2, c(0x6b200159, 0x99ba4efe), "uint32"))
  rp_expect(tf(u(2^32 - 1, 2^32 - 1), u(2^32 - 1, 2^32 - 1)), list(2, c(0x1cb996fc, 0xbb002be7), "uint32"))
  rp_expect(tf(u(0x13198a2e, 0x03707344), u(0x243f6a88, 0x85a308d3)), list(2, c(0xc4923a9c, 0x483df7a0), "uint32"))
  # Odd counts are padded with a zero, as in JAX.
  rp_expect(tf(u(0x13198a2e, 0x03707344), jnp_arange(5L, dtype = "uint32")),
            list(5, c(2319929243, 3637777754, 547490127, 4059551135, 3871567085), "uint32"))
  rp_expect(jit(tf)(u(0x13198a2e, 0x03707344), jnp_arange(5L, dtype = "uint32")),
            list(5, c(2319929243, 3637777754, 547490127, 4059551135, 3871567085), "uint32"))
  expect_error(tf(u(0, 0), jnp_arange(3L)), "threefry_2x32 requires uint32 arguments")
  p <- get("threefry2x32_p", asNamespace("jaxR"))
  expect_error(jaxR:::bind(p, list(u(0), u(0), jnp_arange(2L), jnp_arange(2L))),
               "Arguments to threefry2x32 must have uint32 type")
})

test_that("raw keys, split, fold_in and bits match JAX", {
  skip_if_no_backend()
  k0 <- random_PRNGKey(0)
  k42 <- random_PRNGKey(42)
  rp_expect(k0, rp_ref$PRNGKey_0)
  rp_expect(k42, rp_ref$PRNGKey_42)
  rp_expect(random_PRNGKey(-1), rp_ref$PRNGKey_m1)
  rp_expect(random_PRNGKey(2^40 + 5), rp_ref$PRNGKey_big)
  rp_expect(random_PRNGKey(jnp_asarray(42L)), rp_ref$PRNGKey_42)
  rp_expect(random_split(k0), rp_ref$split_0)
  rp_expect(random_split(k42, 3), rp_ref$split_42_3)
  rp_expect(random_split(k0, c(2, 3)), rp_ref$split_0_23)
  rp_expect(random_fold_in(k0, 7), rp_ref$fold_in_0_7)
  rp_expect(random_fold_in(k42, 2^32 - 1), rp_ref$fold_in_42_max)
  rp_expect(random_fold_in(k0, jnp_asarray(7L)), rp_ref$fold_in_0_7)
  rp_expect(random_bits(k0, 5), rp_ref$bits_0_5)
  rp_expect(random_bits(k42, c(2, 3)), rp_ref$bits_42_23)
  rp_expect(random_bits(k0, 7, "uint8"), rp_ref$bits_0_u8)
  rp_expect(random_bits(k0, 5, "uint16"), rp_ref$bits_0_u16)
  expect_identical(dim(random_bits(k0)), integer(0))
  expect_identical(dim(random_bits(k0, c(0, 3))), c(0L, 3L))
})

test_that("the original (non-partitionable) threefry algorithms match JAX", {
  skip_if_no_backend()
  k0 <- random_PRNGKey(0)
  with_jax_config("jax_threefry_partitionable", FALSE, {
    rp_expect(random_split(k0), rp_ref$np_split_0)
    rp_expect(random_split(random_PRNGKey(42), 3), rp_ref$np_split_42_3)
    rp_expect(random_bits(k0, 5), rp_ref$np_bits_0_5)
    rp_expect(random_bits(k0, 7, "uint8"), rp_ref$np_bits_0_u8)
    rp_expect(random_bits(k0, 5, "uint16"), rp_ref$np_bits_0_u16)
    rp_expect(random_fold_in(k0, 7), rp_ref$fold_in_0_7)
  })
  # The jit caches are keyed on the option.
  rp_expect(random_split(k0), rp_ref$split_0)
})

test_that("64-bit seeds and bits match JAX with x64", {
  skip_if_no_backend()
  old <- enable_x64(TRUE)
  on.exit(jax_config_update("jax_enable_x64", old), add = TRUE)
  k0 <- random_PRNGKey(0)
  rp_expect(random_PRNGKey(-1), list(2, c(4294967295, 4294967295), "uint32"))
  rp_expect(random_PRNGKey(2^40 + 5), list(2, c(256, 5), "uint32"))
  rp_expect(random_split(k0), rp_ref$split_0)
  rp_expect(random_fold_in(k0, 7), rp_ref$fold_in_0_7)
  rp_expect_u64(random_bits(k0, 5), 5,
                hi = c(1797259609, 928981903, 4146024105, 2467461003, 2285895361),
                lo = c(2579123966, 3453687069, 2718843009, 3840466878, 433833334))
  rp_expect_u64(random_bits(random_PRNGKey(42), c(2, 3)), c(2, 3),
                hi = c(1832780943, 64467757, 2465931498, 3134548294, 2954079971, 2765691542),
                lo = c(270669613, 2916123636, 255383827, 894150801, 3276725750, 824333390))
  rp_expect(random_bits(k0, 5, "uint32"), rp_ref$bits_0_5)
})

test_that("typed keys wrap the same data and are pytrees", {
  skip_if_no_backend()
  K <- random_key(0)
  expect_s3_class(K, "jax_prng_key")
  expect_identical(dim(K), integer(0))
  expect_identical(random_key_impl(K), "threefry2x32")
  expect_identical(random_key_dtype(), "key<fry>")
  expect_identical(random_key_dtype("rbg"), "key<rbg>")
  rp_expect(random_key_data(K), rp_ref$PRNGKey_0)
  expect_output(print(K), "Array\\(\\(\\), dtype=key<fry>\\) overlaying")
  ks <- random_split(K, 2)
  expect_s3_class(ks, "jax_prng_key")
  expect_identical(dim(ks), 2L)
  rp_expect(random_key_data(ks), rp_ref$split_0)
  kl <- as.list(ks)
  expect_length(kl, 2L)
  rp_expect(random_key_data(kl[[2L]]), list(2, c(928981903, 3453687069), "uint32"))
  rp_expect(random_key_data(random_fold_in(K, 7)), rp_ref$fold_in_0_7)
  # wrap_key_data round trip, with dtype or impl
  w <- random_wrap_key_data(random_key_data(ks), dtype = random_key_dtype())
  rp_expect(random_key_data(w), rp_ref$split_0)
  expect_identical(random_key_impl(random_wrap_key_data(random_PRNGKey(0), impl = "threefry2x32")),
                   "threefry2x32")
  # key_data and key_impl accept raw keys and key arrays
  rp_expect(random_key_data(random_PRNGKey(42)), rp_ref$PRNGKey_42)
  expect_identical(random_key_impl(random_split(random_PRNGKey(0))), "threefry2x32")
  # clone is the identity; Key is the NewType constructor
  rp_expect(random_key_data(random_clone(K)), rp_ref$PRNGKey_0)
  expect_identical(random_Key(K), K)
  # raw keys stay raw, typed keys stay typed
  expect_false(inherits(random_split(random_PRNGKey(0)), "jax_prng_key"))
  expect_true(inherits(random_fold_in(K, 1), "jax_prng_key"))
  # pytree with one uint32 leaf
  fl <- tree_flatten(list(a = K, b = 1))
  expect_length(fl$leaves, 2L)
  expect_identical(jaxR::dtype(fl$leaves[[1L]]), "uint32")
  expect_s3_class(tree_unflatten(fl$treedef, fl$leaves)$a, "jax_key_threefry2x32")
})

test_that("key indexing uses the jax array indexing rules", {
  skip_if_no_backend()
  skip_if_not(exists("[.jax_value", envir = asNamespace("jaxR"), inherits = FALSE),
              "jax array indexing not available")
  ks <- random_split(random_key(0), 3)
  rp_expect(random_key_data(ks[1]), list(2, c(928981903, 3453687069), "uint32"))
  expect_identical(dim(ks[jaxR::sl(0, 2)]), 2L)
})

test_that("keys flow through jit and vmap", {
  skip_if_no_backend()
  k0 <- random_PRNGKey(0)
  K <- random_key(0)
  rp_expect(jit(function(k) random_split(k))(k0), rp_ref$split_0)
  out <- jit(function(k) random_split(k, 2))(K)
  expect_s3_class(out, "jax_key_threefry2x32")
  rp_expect(random_key_data(out), rp_ref$split_0)
  rp_expect(jit(function(k) random_bits(k, 5))(K), rp_ref$bits_0_5)
  rp_expect(jit(function(k, d) random_fold_in(k, d))(k0, jnp_asarray(7L)), rp_ref$fold_in_0_7)
  # vmap over raw and typed key arrays and over seeds
  rp_expect(vmap(function(k) random_split(k))(random_split(k0, 3)), rp_ref$vmap_split)
  vk <- vmap(function(k) random_split(k))(random_split(K, 3))
  expect_identical(dim(vk), c(3L, 2L))
  rp_expect(random_key_data(vk), rp_ref$vmap_split)
  rp_expect(vmap(function(d) random_fold_in(k0, d))(jnp_arange(3L)), rp_ref$vmap_fold_in)
  rp_expect(vmap(function(s) random_key_data(random_key(s)))(jnp_arange(3L)), rp_ref$vmap_key)
  vb <- vmap(function(k) random_bits(k, 2))(random_split(K, 3))
  for (i in 1:3) {
    ki <- as.list(random_split(K, 3))[[i]]
    expect_identical(as.numeric(as.array(vb)[i, ]), as.numeric(as.array(random_bits(ki, 2))))
  }
})

test_that("keys have float0 tangents under grad and jvp", {
  skip_if_no_backend()
  K <- random_key(3)
  f <- function(x, k) jnp_sum(x * random_uniform(k, 3))
  u <- as.array(random_uniform(K, 3))
  g <- grad(f)(jnp_ones(3), K)
  expect_equal(as.vector(as.array(g)), as.vector(u), tolerance = 0)
  g2 <- vjp(function(k) random_uniform(k, 3), K)
  expect_identical(jaxR::dtype(random_key_data(g2[[2L]](jnp_ones(3))[[1L]])), "float0")
  r <- jvp(function(x) random_split(random_fold_in(K, 1)), list(jnp_ones(2)), list(jnp_ones(2)))
  expect_s3_class(r[[1L]], "jax_prng_key")
})

test_that("rbg and unsafe_rbg keys match JAX", {
  skip_if_no_backend()
  for (impl in c("rbg", "unsafe_rbg")) {
    kk <- random_key(0, impl = impl)
    expect_identical(random_key_impl(kk), impl)
    rp_expect(random_key_data(kk), list(4, c(0, 0, 0, 0), "uint32"))
    rp_expect(random_bits(kk, 4), rp_ref$rbg_bits)
  }
  rp_expect(random_key_data(random_split(random_key(0, impl = "rbg"), 3)), rp_ref$rbg_split)
  rp_expect(random_key_data(random_fold_in(random_key(0, impl = "rbg"), 5)), rp_ref$rbg_fold_in)
  rp_expect(random_key_data(random_split(random_key(0, impl = "unsafe_rbg"), 3)), rp_ref$unsafe_rbg_split)
  rp_expect(random_key_data(random_fold_in(random_key(0, impl = "unsafe_rbg"), 5)),
            rp_ref$unsafe_rbg_fold_in)
  rp_expect(random_bits(random_key(3, impl = "rbg"), 5, "uint8"), rp_ref$rbg_bits_u8)
  # rng_bit_generator's batching rule draws every lane from the first key, as in JAX
  rp_expect(vmap(function(k) random_bits(k, 3))(random_split(random_key(0, impl = "rbg"), 2)),
            rp_ref$rbg_vmap_bits)
  # jit caches distinguish implementations with equal key shapes
  f <- jit(function(k) random_key_data(random_fold_in(k, 5)))
  rp_expect(f(random_key(0, impl = "rbg")), rp_ref$rbg_fold_in)
  rp_expect(f(random_key(0, impl = "unsafe_rbg")), rp_ref$unsafe_rbg_fold_in)
  expect_error(random_PRNGKey(0, impl = "philox"), "unrecognized PRNG implementation")
})

test_that("lax_rng_bit_generator and lax_rng_uniform", {
  skip_if_no_backend()
  key <- jnp_asarray(c(0, 0, 0, 0), dtype = "uint32")
  r <- lax_rng_bit_generator(key, c(2, 3), dtype = "uint16")
  expect_identical(jaxR::dtype(r[[1L]]), "uint32")
  expect_identical(dim(r[[1L]]), 4L)
  expect_identical(jaxR::dtype(r[[2L]]), "uint16")
  expect_identical(dim(r[[2L]]), c(2L, 3L))
  r2 <- lax_rng_bit_generator(key, 4)
  rp_expect(r2[[2L]], rp_ref$rbg_bits)
  rp_expect(jit(function(k) lax_rng_bit_generator(k, 4)[[2L]])(key), rp_ref$rbg_bits)
  expect_error(lax_rng_bit_generator(key, 3, dtype = "int32"), "unsupported dtype")
  u <- lax_rng_uniform(0, 1, c(2, 2))
  expect_identical(dim(u), c(2L, 2L))
  v <- as.array(u)
  expect_true(all(v >= 0 & v < 1))
  expect_error(lax_rng_uniform(jnp_zeros(2), 1, 3), "must be scalars")
  expect_identical(lax_RandomAlgorithm[["RNG_PHILOX"]], 2L)
})

test_that("key errors follow JAX", {
  skip_if_no_backend()
  k0 <- random_PRNGKey(0)
  expect_error(random_split(random_split(k0)), "split accepts a single key, but was given a key array of shape \\(2, 2\\)")
  expect_error(random_uniform(random_split(random_key(0))), "uniform accepts a single key")
  expect_error(random_split(c(0, 0)), "expected key_data.dtype = uint32; got dtype=float32")
  expect_error(random_split(jnp_zeros(3, dtype = "uint32")), "expected key_data.shape to end with \\(2,\\)")
  expect_error(random_split(jnp_zeros(integer(0), dtype = "uint32")), "expected key_data.ndim >= 1")
  expect_error(random_split("a"), "unexpected PRNG key type")
  expect_error(random_key(c(1, 2)), "key accepts a scalar seed, but was given an array of shape \\(2,\\)")
  expect_error(random_PRNGKey(random_key(0)), "accepts a scalar seed, but was given a PRNG key")
  expect_error(random_key(0.5), "PRNG key seed must be an integer")
  expect_error(random_key(0, impl = "rbg", dtype = "rbg"), "Cannot specify both")
  expect_error(random_fold_in(k0, c(1, 2)), "fold_in accepts a scalar")
  expect_error(random_fold_in(k0, -1), "out of bounds for uint32")
  expect_error(random_bits(k0, 3, "int32"), "must be an unsigned int dtype")
  expect_error(as.list(random_key(0)), "iteration over a 0-d key array")
  expect_error(random_wrap_key_data(jnp_zeros(c(2, 2), dtype = "uint32"), impl = "rbg"),
               "expected key_data.shape to end with \\(4,\\)")
})
