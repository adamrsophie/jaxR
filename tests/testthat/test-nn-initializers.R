# jax.nn.initializers (R/nn_initializers.R). Expected values were produced by jax 0.12.0.1
# on CPU with the keys below; arrays are listed in C order as list(shape, values, dtype).

nit_c_ravel <- function(x) {
  d <- dim(x)
  if (length(d) <= 1L) return(as.vector(x))
  as.vector(aperm(x, length(d):1))
}

# Exact comparison (tol = 0) unless a relative tolerance is given.
nit_expect <- function(x, ref, tol = 0) {
  expect_identical(jaxR::dtype(x), ref[[3L]])
  expect_identical(as.integer(dim(x)), as.integer(ref[[1L]]))
  got <- nit_c_ravel(as.array(x))
  if (tol == 0) expect_identical(got + 0, ref[[2L]] + 0)
  else expect_equal(got, ref[[2L]], tolerance = tol)
}

init_ref <- list(
  zeros = list(c(2, 3), c(0, 0, 0, 0, 0, 0), "float32"),
  ones_i32 = list(c(2), c(1, 1), "int32"),
  constant = list(c(2, 3), c(-0x1.c000000000000p+2, -0x1.c000000000000p+2, -0x1.c000000000000p+2, -0x1.c000000000000p+2, -0x1.c000000000000p+2, -0x1.c000000000000p+2), "float32"),
  constant_i = list(c(2), c(3, 3), "int32"),
  uniform = list(c(2, 3), c(0x1.4047dc0000000p-8, 0x1.bd830a0000000p-8, 0x1.93e1320000000p-8, 0x1.6faae00000000p-8, 0x1.27559e0000000p-8, 0x1.7ff3fa0000000p-8), "float32"),
  uniform_s = list(c(2, 3), c(0x1.7b26300000000p-2, 0x1.2c34c80000000p+2, 0x1.9f551e0000000p+1, 0x1.3832e00000000p+2, 0x1.6452d40000000p+0, 0x1.7514940000000p+1), "float32"),
  normal = list(c(2, 3), c(-0x1.28cba00000000p-12, 0x1.3223b80000000p-8, 0x1.8395720000000p-9, 0x1.9282ec0000000p-10, -0x1.4525040000000p-10, 0x1.1c53540000000p-9), "float32"),
  normal_s = list(c(2, 3), c(-0x1.cecd5e0000000p+2, 0x1.ec9a1a0000000p+2, 0x1.e99ba60000000p+0, 0x1.3b4ff20000000p+3, -0x1.781b6e0000000p+1, 0x1.0c0d500000000p+0), "float32"),
  truncated_normal = list(c(2, 3), c(-0x1.14a6200000000p-3, 0x1.1c675a0000000p+1, 0x1.68cd7a0000000p+0, 0x1.770f840000000p-1, -0x1.2f01ea0000000p-1, 0x1.08d7720000000p+0), "float32"),
  truncated_normal_b = list(c(4), c(-0x1.b25fe40000000p-1, 0x1.a2304c0000000p-2, 0x1.6b670c0000000p-7, 0x1.da8e560000000p-2), "float32"),
  glorot_uniform = list(c(2, 3), c(-0x1.9546b40000000p-6, 0x1.935eb00000000p-2, 0x1.04da100000000p-2, 0x1.11c6940000000p-3, -0x1.bae8e00000000p-4, 0x1.8146760000000p-3), "float32"),
  glorot_normal = list(c(2, 3), c(-0x1.3e42600000000p-6, 0x1.472e320000000p-2, 0x1.9f12080000000p-3, 0x1.af79100000000p-4, -0x1.5c95160000000p-4, 0x1.30ad0e0000000p-3), "float32"),
  lecun_uniform = list(c(2, 3), c(-0x1.c51cd00000000p-6, 0x1.c2fb320000000p-2, 0x1.23a4240000000p-2, 0x1.32172e0000000p-3, -0x1.ef30240000000p-4, 0x1.aec0340000000p-3), "float32"),
  lecun_normal = list(c(2, 3), c(-0x1.63d3200000000p-6, 0x1.6dcc820000000p-2, 0x1.d010160000000p-3, 0x1.e266c00000000p-4, -0x1.85ba1a0000000p-4, 0x1.54a35c0000000p-3), "float32"),
  he_uniform = list(c(2, 3), c(-0x1.4066220000000p-5, 0x1.3ee4500000000p-1, 0x1.9c71620000000p-2, 0x1.b0e09c0000000p-3, -0x1.5e269c0000000p-3, 0x1.3096380000000p-2), "float32"),
  he_normal = list(c(2, 3), c(-0x1.f736500000000p-6, 0x1.02a8ac0000000p-1, 0x1.48245a0000000p-2, 0x1.551bfc0000000p-3, -0x1.13941e0000000p-3, 0x1.e1bc2e0000000p-3), "float32"),
  he_normal_raw = list(c(3, 4), c(-0x1.39a54c0000000p+0, 0x1.4b0b480000000p+0, 0x1.5a42f40000000p-2, 0x1.9056c40000000p+0, -0x1.0923240000000p-1, 0x1.7bbd860000000p-3, -0x1.1209820000000p+0, -0x1.110c3e0000000p-2, -0x1.687e740000000p-1, -0x1.f71b4c0000000p-1, 0x1.1db3aa0000000p-1, -0x1.c2260a0000000p-1), "float32"),
  vs_fan_out_normal = list(c(3, 2, 4), c(-0x1.7aa4fa0000000p-7, 0x1.8690c00000000p-3, 0x1.ee78560000000p-4, 0x1.00c1d60000000p-4, -0x1.9ecfc80000000p-5, 0x1.6abc520000000p-4, -0x1.2d2d520000000p-1, 0x1.3bfc000000000p-2, 0x1.b3f25a0000000p-3, 0x1.7c7e4a0000000p-2, -0x1.4179860000000p-3, 0x1.dc7fb60000000p-2, 0x1.2e33e80000000p-1, 0x1.c3de760000000p-2, -0x1.7887d80000000p-6, 0x1.7c40a80000000p-2, 0x1.d1fe320000000p-3, 0x1.6e4ddc0000000p-4, -0x1.2ec5960000000p-1, 0x1.3f773a0000000p-2, -0x1.93e9460000000p-4, -0x1.ed092a0000000p-2, -0x1.95475e0000000p-1, 0x1.29dec40000000p-3), "float32"),
  vs_geo_uniform = list(c(3, 2, 4), c(-0x1.371a240000000p-7, 0x1.35a3860000000p-3, 0x1.9079ac0000000p-4, 0x1.a4511c0000000p-5, -0x1.53fd9e0000000p-5, 0x1.27bfb00000000p-4, -0x1.6e1f4a0000000p-2, 0x1.d9d2920000000p-3, 0x1.56a38a0000000p-3, 0x1.125e520000000p-2, -0x1.01cc3c0000000p-3, 0x1.4106f00000000p-2, 0x1.6eb7920000000p-2, 0x1.3619c00000000p-2, -0x1.353e320000000p-6, 0x1.123cd60000000p-2, 0x1.6bfc420000000p-3, 0x1.2a9cbc0000000p-4, -0x1.6f0b9c0000000p-2, 0x1.de1bd60000000p-3, -0x1.48b4b80000000p-4, -0x1.47f6f00000000p-2, -0x1.97ee4c0000000p-2, 0x1.df60020000000p-4), "float32"),
  vs_conv = list(c(3, 3, 2, 4), c(-0x1.835f740000000p-8, 0x1.8e3b3a0000000p-4, 0x1.f935280000000p-5, 0x1.0696080000000p-5, -0x1.a847ee0000000p-6, 0x1.72d6fe0000000p-5, -0x1.26bdc60000000p-2, 0x1.4041900000000p-3, 0x1.bc1e4c0000000p-4, 0x1.7fc5160000000p-3, -0x1.4826440000000p-4, 0x1.dbcdf60000000p-3, 0x1.279f980000000p-2, 0x1.c48a6c0000000p-3, -0x1.8132400000000p-7, 0x1.7f88f40000000p-3, 0x1.da6d080000000p-4, 0x1.767b940000000p-5, -0x1.281cb80000000p-2, 0x1.43b5d60000000p-3, -0x1.9cdd7c0000000p-5, -0x1.eb441e0000000p-3, -0x1.75593a0000000p-2, 0x1.3027560000000p-4, -0x1.9c22d60000000p-5, 0x1.f91e720000000p-3, -0x1.1e74ba0000000p-2, -0x1.c764740000000p-4, 0x1.ce8c5e0000000p-5, 0x1.383d520000000p-2, 0x1.2650ae0000000p-3, -0x1.635b700000000p-6, -0x1.abfbc80000000p-4, 0x1.1f7ec40000000p-3, 0x1.7b341a0000000p-6, -0x1.28e9660000000p-4, 0x1.83e03e0000000p-4, 0x1.8578a20000000p-5, -0x1.da2c2a0000000p-4, -0x1.74ac440000000p-3, -0x1.6cff480000000p-5, 0x1.0716980000000p-4, -0x1.401e080000000p-5, 0x1.4054cc0000000p-6, 0x1.3f21340000000p-4, -0x1.bb1c680000000p-3, 0x1.7ce6e80000000p-4, 0x1.f772360000000p-3, 0x1.7210640000000p-4, -0x1.2a97040000000p-3, 0x1.2d9faa0000000p-5, -0x1.53bb520000000p-5, -0x1.74dc300000000p-5, 0x1.09f7f20000000p-2, 0x1.4061120000000p-4, -0x1.30804a0000000p-5, -0x1.99de700000000p-5, -0x1.5da5c60000000p-4, 0x1.3961920000000p-4, 0x1.ee93960000000p-3, -0x1.b3a1f80000000p-3, -0x1.502e1e0000000p-3, 0x1.bba5c40000000p-3, -0x1.3528200000000p-4, -0x1.7927b80000000p-7, -0x1.83630c0000000p-2, 0x1.316eaa0000000p-2, -0x1.2a9e060000000p-2, 0x1.13e6a60000000p-6, -0x1.5d0f900000000p-3, -0x1.93e55c0000000p-6, -0x1.7c96200000000p-3), "float32"),
  vs_axes = list(c(2, 3, 4, 2), c(-0x1.7aa4fa0000000p-7, 0x1.8690c00000000p-3, 0x1.ee78560000000p-4, 0x1.00c1d60000000p-4, -0x1.9ecfc80000000p-5, 0x1.6abc520000000p-4, -0x1.2d2d520000000p-1, 0x1.3bfc000000000p-2, 0x1.b3f25a0000000p-3, 0x1.7c7e4a0000000p-2, -0x1.4179860000000p-3, 0x1.dc7fb60000000p-2, 0x1.2e33e80000000p-1, 0x1.c3de760000000p-2, -0x1.7887d80000000p-6, 0x1.7c40a80000000p-2, 0x1.d1fe320000000p-3, 0x1.6e4ddc0000000p-4, -0x1.2ec5960000000p-1, 0x1.3f773a0000000p-2, -0x1.93e9460000000p-4, -0x1.ed092a0000000p-2, -0x1.95475e0000000p-1, 0x1.29dec40000000p-3, -0x1.93325a0000000p-4, 0x1.fbfac00000000p-2, -0x1.23a2380000000p-1, -0x1.bf1dc20000000p-3, 0x1.c49ea00000000p-4, 0x1.41ebec0000000p-1, 0x1.21ec420000000p-2, -0x1.5b67000000000p-5, -0x1.a3f9a20000000p-3, 0x1.1b18120000000p-2, 0x1.72b9400000000p-5, -0x1.22c0300000000p-3, 0x1.7c57420000000p-3, 0x1.7cfce20000000p-4, -0x1.d1bdc60000000p-3, -0x1.7123b60000000p-2, -0x1.6503360000000p-4, 0x1.0186a40000000p-3, -0x1.39109c0000000p-4, 0x1.3926c80000000p-5, 0x1.38996c0000000p-3, -0x1.b9f6b60000000p-2, 0x1.7575500000000p-3, 0x1.fa2b160000000p-2), "float32"),
  glorot_axes = list(c(2, 3, 2), c(-0x1.c2162a0000000p-7, 0x1.ceb3f80000000p-3, 0x1.257fc40000000p-3, 0x1.3118f00000000p-4, -0x1.ecf84c0000000p-5, 0x1.aee07c0000000p-4, -0x1.56755c0000000p-1, 0x1.741a9c0000000p-2, 0x1.02026a0000000p-2, 0x1.bde6780000000p-2, -0x1.7d46740000000p-3, 0x1.146ade0000000p-1), "float32"),
  vs_c_uniform = list(c(2, 3), complex(real = c(-0x1.a136d80000000p-4, 0x1.0e34be0000000p-3, 0x1.9496420000000p-2, -0x1.0f591e0000000p-4, 0x1.34f7c20000000p-1, 0x1.d99b940000000p-3), imaginary = c(-0x1.712af40000000p-1, -0x1.1689720000000p-1, 0x1.ba0cc60000000p-1, 0x1.aa83540000000p-1, 0x1.01fe8e0000000p-1, 0x1.81ffee0000000p-1)), "complex64"),
  vs_c_tn = list(c(2, 3), complex(real = c(-0x1.6c72120000000p-4, 0x1.b241540000000p-4, 0x1.d0ee920000000p-2, -0x1.031c6c0000000p-4, 0x1.19aca40000000p-1, 0x1.b0a3720000000p-3), imaginary = c(-0x1.4279da0000000p-1, -0x1.bfa4da0000000p-2, 0x1.fbfb840000000p-1, 0x1.9747440000000p-1, 0x1.d668560000000p-2, 0x1.609be40000000p-1)), "complex64"),
  vs_c_normal = list(c(2, 3), complex(real = c(0x1.36fdb20000000p-5, -0x1.f203d00000000p-3, 0x1.4a52740000000p-1, 0x1.0a0a920000000p-2, 0x1.339d340000000p-3, 0x1.3dcc0a0000000p-3), imaginary = c(0x1.3626b40000000p-2, 0x1.991c500000000p-2, -0x1.d15ee20000000p-2, -0x1.45407a0000000p-2, -0x1.3900260000000p-1, -0x1.aa1e1c0000000p-2)), "complex64"),
  orth_34 = list(c(3, 4), c(-0x1.0ef8800000000p-6, 0x1.6f7d380000000p-4, -0x1.af10e60000000p-1, 0x1.104b680000000p-1, 0x1.f5b1820000000p-1, -0x1.9f9be40000000p-4, 0x1.1bf2740000000p-4, 0x1.42381c0000000p-3, -0x1.2a65380000000p-3, 0x1.aa34ea0000000p-3, 0x1.11907c0000000p-1, 0x1.9cc7f60000000p-1), "float32"),
  orth_43 = list(c(4, 3), c(-0x1.0ef8800000000p-5, 0x1.f5b1820000000p+0, -0x1.2a65380000000p-2, 0x1.6f7d380000000p-3, -0x1.9f9be40000000p-3, 0x1.aa34ea0000000p-2, -0x1.af10e60000000p+0, 0x1.1bf2740000000p-3, 0x1.11907c0000000p+0, 0x1.104b680000000p+0, 0x1.42381c0000000p-2, 0x1.9cc7f60000000p+0), "float32"),
  orth_col0 = list(c(3, 4), c(-0x1.0ef8800000000p-6, 0x1.6f7d380000000p-4, -0x1.af10e60000000p-1, 0x1.104b680000000p-1, 0x1.f5b1820000000p-1, -0x1.9f9be40000000p-4, 0x1.1bf2740000000p-4, 0x1.42381c0000000p-3, -0x1.2a65380000000p-3, 0x1.aa34ea0000000p-3, 0x1.11907c0000000p-1, 0x1.9cc7f60000000p-1), "float32"),
  orth_234 = list(c(2, 3, 4), c(-0x1.17c8800000000p-6, 0x1.8e93e80000000p-2, -0x1.a9123c0000000p-4, 0x1.451ace0000000p-6, -0x1.3281ec0000000p-4, 0x1.07b7760000000p-2, -0x1.797d220000000p-1, -0x1.031f940000000p-2, 0x1.421fd60000000p-2, 0x1.6fc6dc0000000p-2, -0x1.4d06aa0000000p-2, 0x1.7673b80000000p-1, 0x1.be99900000000p-1, -0x1.2aa7ac0000000p-3, 0x1.3f9ac00000000p-3, -0x1.f6e3200000000p-6, 0x1.5853640000000p-2, -0x1.b366820000000p-3, -0x1.b5fbb40000000p-2, -0x1.eeb6b20000000p-2, -0x1.2a73fe0000000p-3, -0x1.882d7c0000000p-1, -0x1.74da600000000p-2, 0x1.a1a6520000000p-2), "float32"),
  delta3 = list(c(3, 2, 2), c(0, 0, 0, 0, -0x1.8648c00000000p-4, 0x1.fdaba20000000p-1, 0x1.fdaba20000000p-1, 0x1.8648d20000000p-4, 0, 0, 0, 0), "float32"),
  delta4 = list(c(2, 3, 2, 3), c(0, 0, 0, 0, 0, 0, -0x1.6826800000000p-5, 0x1.d6519a0000000p-2, -0x1.8a8d120000000p-3, 0x1.be994e0000000p-2, 0x1.07fd400000000p-3, 0x1.a98a0e0000000p-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), "float32"),
  normal_bf16 = list(c(4), c(0x1.c400000000000p-9, 0x1.4e00000000000p-7, 0x1.3600000000000p-6, 0x1.4200000000000p-7), "float32"),
  uniform_f16 = list(c(4), c(0x1.8f80000000000p-11, 0x1.f340000000000p-10, 0x1.3fc0000000000p-7, 0x1.f000000000000p-8), "float32"),
  normal_f64 = list(c(2, 3), c(-0x1.e435f434f86d8p-10, -0x1.638109cffa6b1p-6, 0x1.ea0a5341ae6e2p-10, 0x1.914114d99efcap-8, 0x1.40e16f7ef90afp-8, 0x1.e3a94c6a85eb3p-9), "float64"),
  glorot_uniform_f64 = list(c(2, 3), c(-0x1.48c4ed4cdc2c2p-3, -0x1.1003e99b15084p+0, 0x1.4cae89dbbb19dp-3, 0x1.01cbcc238d797p-1, 0x1.a553219536092p-2, 0x1.42eb948e06e8fp-2), "float64"),
  he_normal_f64 = list(c(3, 2), c(-0x1.4efe39427f900p-3, -0x1.a8524a82823f1p+0, 0x1.53059e94a62b3p-3, 0x1.141918e477b9fp-1, 0x1.ba92ec6c146d5p-2, 0x1.4e17829345883p-2), "float64"),
  orth_f64 = list(c(3, 3), c(-0x1.ffb32ea4feb08p-3, -0x1.ef95d29f5514ep-1, 0x1.a6abc84fa8a26p-6, 0x1.a808ab572b221p-1, -0x1.993dbf48dd2f4p-3, 0x1.0c16ddc673a27p-1, 0x1.00dadff3ed31ep-1, -0x1.37b05202f9a90p-3, -0x1.b4007c28a6e49p-1), "float64")
)

test_that("constant initializers match jax.nn.initializers", {
  skip_if_no_backend()
  key <- random_key(42)
  nit_expect(nn_initializers_zeros(key, c(2, 3)), init_ref$zeros)
  nit_expect(nn_initializers_ones(key, 2, "int32"), init_ref$ones_i32)
  nit_expect(nn_initializers_constant(-7)(key, c(2, 3)), init_ref$constant)
  nit_expect(nn_initializers_constant(3L, "int32")(key, 2), init_ref$constant_i)
  expect_identical(jaxR::dtype(nn_initializers_constant(1, "float16")(key, 2, "bfloat16")), "bfloat16")
  # builder arguments are captured when the initializer is built
  inits <- list()
  for (v in c(1, 2)) inits[[length(inits) + 1L]] <- nn_initializers_constant(v)
  expect_identical(as.vector(as.array(inits[[1L]](key, 1))), 1)
})

test_that("random initializers match jax.nn.initializers bit for bit", {
  skip_if_no_backend()
  key <- random_key(42)
  raw <- random_PRNGKey(3)
  nit_expect(nn_initializers_uniform()(key, c(2, 3)), init_ref$uniform)
  nit_expect(nn_initializers_uniform(5)(raw, c(2, 3)), init_ref$uniform_s)
  nit_expect(nn_initializers_normal()(key, c(2, 3)), init_ref$normal)
  nit_expect(nn_initializers_normal(5)(raw, c(2, 3)), init_ref$normal_s)
  nit_expect(nn_initializers_truncated_normal(5)(key, c(2, 3)), init_ref$truncated_normal)
  nit_expect(nn_initializers_truncated_normal(1, lower = -1, upper = 0.5)(raw, 4),
             init_ref$truncated_normal_b)
  for (nm in c("glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal", "he_uniform",
               "he_normal")) {
    nit_expect(get(paste0("nn_initializers_", nm))()(key, c(2, 3)), init_ref[[nm]])
  }
  nit_expect(nn_initializers_xavier_uniform()(key, c(2, 3)), init_ref$glorot_uniform)
  nit_expect(nn_initializers_xavier_normal()(key, c(2, 3)), init_ref$glorot_normal)
  nit_expect(nn_initializers_kaiming_uniform()(key, c(2, 3)), init_ref$he_uniform)
  nit_expect(nn_initializers_kaiming_normal()(key, c(2, 3)), init_ref$he_normal)
  nit_expect(nn_initializers_he_normal()(raw, c(3, 4)), init_ref$he_normal_raw)
  nit_expect(jnp_astype(nn_initializers_normal(dtype = "bfloat16")(key, 4), "float32"),
             init_ref$normal_bf16)
  nit_expect(jnp_astype(nn_initializers_uniform(dtype = "float16")(key, 4), "float32"),
             init_ref$uniform_f16)
})

test_that("variance_scaling covers modes, distributions, axes and complex dtypes", {
  skip_if_no_backend()
  key <- random_key(42)
  vs <- nn_initializers_variance_scaling
  nit_expect(vs(2, "fan_out", "normal")(key, c(3, 2, 4)), init_ref$vs_fan_out_normal)
  nit_expect(vs(0.5, "fan_geo_avg", "uniform")(key, c(3, 2, 4)), init_ref$vs_geo_uniform)
  nit_expect(vs(1, "fan_avg", "truncated_normal")(key, c(3, 3, 2, 4)), init_ref$vs_conv)
  nit_expect(vs(1, "fan_in", "normal", in_axis = c(0L, 1L), out_axis = 2L, batch_axis = 3L)(
    key, c(2, 3, 4, 2)), init_ref$vs_axes)
  nit_expect(nn_initializers_glorot_normal(in_axis = 0L, out_axis = 1L)(key, c(2, 3, 2)),
             init_ref$glorot_axes)
  nit_expect(vs(1, "fan_in", "uniform", dtype = "complex64")(key, c(2, 3)), init_ref$vs_c_uniform)
  nit_expect(vs(1, "fan_in", "truncated_normal")(key, c(2, 3), "complex64"), init_ref$vs_c_tn)
  # complex normals divide by sqrt(2) in complex arithmetic (last ulp may differ)
  nit_expect(vs(1, "fan_in", "normal")(key, c(2, 3), "complex64"), init_ref$vs_c_normal, tol = 1e-6)
  expect_error(vs(1, "fan_max", "normal")(key, c(2, 3)),
               "invalid mode for variance scaling initializer: fan_max")
  expect_error(vs(1, "fan_in", "cauchy")(key, c(2, 3)),
               "invalid distribution for variance scaling initializer: cauchy")
  expect_error(nn_initializers_lecun_normal()(key, 3), "Can't compute input and output sizes")
  # jit and vmap over keys
  f <- jit(function(k) nn_initializers_he_normal()(k, c(3, 4)))
  nit_expect(f(random_PRNGKey(3)), init_ref$he_normal_raw)
  keys <- random_split(random_key(0), 3L)
  out <- vmap(function(k) nn_initializers_glorot_uniform()(k, c(2, 3)))(keys)
  first <- nn_initializers_glorot_uniform()(as.list(keys)[[1L]], c(2, 3))
  expect_identical(as.array(lax_index_in_dim(out, 0L, keepdims = FALSE)), as.array(first))
})

test_that("orthogonal and delta_orthogonal match jax.nn.initializers", {
  skip_if_no_backend()
  key <- random_key(42)
  # the QR factorization runs on R's LAPACK, so the last bits may differ from the wheel
  orth <- nn_initializers_orthogonal
  nit_expect(orth()(key, c(3, 4)), init_ref$orth_34, tol = 1e-6)
  nit_expect(orth(scale = 2)(key, c(4, 3)), init_ref$orth_43, tol = 1e-6)
  nit_expect(orth(column_axis = 0L)(key, c(3, 4)), init_ref$orth_col0, tol = 1e-6)
  nit_expect(orth()(key, c(2, 3, 4)), init_ref$orth_234, tol = 1e-6)
  q <- as.array(orth()(key, c(3, 4)))
  expect_equal(q %*% t(q), diag(3), tolerance = 1e-6)
  nit_expect(nn_initializers_delta_orthogonal()(key, c(3, 2, 2)), init_ref$delta3, tol = 1e-6)
  nit_expect(nn_initializers_delta_orthogonal(scale = 0.5)(key, c(2, 3, 2, 3)), init_ref$delta4,
             tol = 1e-6)
  expect_identical(dim(orth()(key, c(0, 3))), c(0L, 3L))
  expect_error(orth()(key, 3), "requires at least a 2D shape")
  expect_error(nn_initializers_delta_orthogonal()(key, c(2, 2)), "requires a 3D, 4D or 5D shape")
  expect_error(nn_initializers_delta_orthogonal()(key, c(3, 3, 2)),
               "`fan_in` must be less or equal than `fan_out`")
})

test_that("float64 initializers match JAX with x64", {
  skip_if_no_backend()
  old <- enable_x64(TRUE)
  on.exit(jax_config_update("jax_enable_x64", old), add = TRUE)
  key <- random_key(42)
  nit_expect(nn_initializers_normal()(key, c(2, 3), "float64"), init_ref$normal_f64)
  nit_expect(nn_initializers_glorot_uniform()(key, c(2, 3), "float64"), init_ref$glorot_uniform_f64)
  nit_expect(nn_initializers_he_normal(dtype = "float64")(key, c(3, 2)), init_ref$he_normal_f64)
  nit_expect(nn_initializers_orthogonal()(key, c(3, 3), "float64"), init_ref$orth_f64, tol = 1e-12)
  expect_identical(jaxR::dtype(nn_initializers_zeros(key, 2)), "float64")
})
