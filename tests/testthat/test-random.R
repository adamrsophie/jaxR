# jax.random samplers (R/random.R). Expected values were produced by jax 0.12.0.1 on CPU
# with the keys below; arrays are listed in C order as list(shape, values, dtype).

rr_c_ravel <- function(x) {
  d <- dim(x)
  if (length(d) <= 1L) return(as.vector(x))
  as.vector(aperm(x, length(d):1))
}

# With tol = 0 integer, bit and logical samples are compared exactly and float samples
# to a few units in the last place (ulp_tol); otherwise tol is a relative tolerance.
rr_expect <- function(x, ref, tol = 0) {
  expect_identical(jaxR::dtype(x), ref[[3L]])
  expect_identical(as.integer(dim(x)), as.integer(ref[[1L]]))
  got <- rr_c_ravel(as.array(x))
  exp <- ref[[2L]]
  if (is.complex(got)) {
    exp <- complex(real = exp[["re"]], imaginary = exp[["im"]])
    expect_equal(got, exp, tolerance = tol)
  } else if (is.logical(got)) {
    expect_identical(got, as.logical(exp))
  } else if (tol == 0 && !grepl("float", ref[[3L]])) {
    expect_identical(as.numeric(got), as.numeric(exp))
  } else {
    expect_equal(as.numeric(got), as.numeric(exp), tolerance = if (tol == 0) ulp_tol(ref[[3L]]) else tol)
  }
}

rr_skip_missing <- function(...) {
  for (f in c(...)) {
    skip_if_not(exists(f, envir = asNamespace("jaxR"), inherits = FALSE), paste(f, "is not available"))
  }
}

rr_ref <- list(
  uniform_0_5 = list(c(5), c(0x1.e5349c0000000p-1, 0x1.f5086c0000000p-1, 0x1.5444380000000p-2, 0x1.dfeaa00000000p-2, 0x1.23c8740000000p-1), "float32"),
  uniform_42_23_m = list(c(2, 3), c(0x1.c631680000000p-2, 0x1.6623f00000000p+0, 0x1.14d3d80000000p+0, 0x1.9c33800000000p-1, 0x1.034ce80000000p-2, 0x1.dbd10c0000000p-1), "float32"),
  uniform_0_minvec = list(c(2, 3), c(0x1.2f40e20000000p+2, 0x1.3a84360000000p+2, 0x1.7f99940000000p+1, 0x1.2bf2a40000000p+1, 0x1.a3c8740000000p+1, 0x1.3f8d9c0000000p+1), "float32"),
  normal_0_5 = list(c(5), c(0x1.9f657a0000000p+0, 0x1.033be00000000p+1, -0x1.bc002e0000000p-2, -0x1.4204440000000p-4, 0x1.68a2580000000p-3), "float32"),
  normal_42_23 = list(c(2, 3), c(-0x1.cfbe2a0000000p-6, 0x1.de57d00000000p-2, 0x1.2eccc20000000p-2, 0x1.3a76480000000p-3, -0x1.fc09d60000000p-4, 0x1.bc42340000000p-3), "float32"),
  normal_0_c64 = list(c(3), list(re = c(0x1.6b7df40000000p-1, -0x1.4821100000000p-1, -0x1.0ede080000000p-1), im = c(-0x1.ba21ba0000000p+0, -0x1.707f600000000p+0, 0x1.29a8ec0000000p-3)), "complex64"),
  randint_0_6 = list(c(6), c(9, 0, 2, 3, 1, 7), "int32"),
  randint_42_neg = list(c(5), c(-1, -1, -4, 4, 4), "int32"),
  randint_u8 = list(c(6), c(101, 80, 200, 61, 3, 211), "uint8"),
  randint_i8 = list(c(6), c(61, -119, -122, -109, 66, 31), "int8"),
  randint_vec = list(c(2, 3), c(4, 10, 912, 3, 11, 247), "int32"),
  randint_empty = list(c(4), c(5, 5, 5, 5), "int32"),
  unif_bf16 = list(c(4), c(0.6484375, 0.5703125, 0.15625, 0.203125), "float32"),
  unif_f16 = list(c(4), c(0.310546875, 0.212890625, 0.0546875, 0.65625), "float32"),
  vmap_uniform = list(c(4, 3), c(0x1.af43cc0000000p-1, 0x1.7582f00000000p-3, 0x1.d142c00000000p-3, 0x1.de02000000000p-8, 0x1.5648000000000p-6, 0x1.29b0bc0000000p-1, 0x1.ce0ddc0000000p-1, 0x1.d3180c0000000p-1, 0x1.5d3b980000000p-2, 0x1.1165b00000000p-2, 0x1.77c8500000000p-1, 0x1.e84e640000000p-1), "float32"),
  np_uniform_0_5 = list(c(5), c(0x1.2624ec0000000p-1, 0x1.9850600000000p-4, 0x1.9298980000000p-2, 0x1.c9d1bc0000000p-1, 0x1.31712c0000000p-1), "float32"),
  np_normal_42 = list(c(2, 3), c(0x1.397ad40000000p-1, 0x1.1f61f20000000p+0, 0x1.23282c0000000p+0, -0x1.a01e7a0000000p-1, -0x1.c7e32a0000000p-1, 0x1.0285a20000000p-3), "float32"),
  rbg_uniform = list(c(4), c(0x1.989fa00000000p-2, 0x1.c2d3880000000p-1, 0x1.78af580000000p-1, 0x1.3601b40000000p-1), "float32"),
  bernoulli = list(c(8), c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE), "bool"),
  bernoulli_high = list(c(8), c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE), "bool"),
  bernoulli_vec = list(c(3), c(FALSE, FALSE, TRUE), "bool"),
  cauchy = list(c(4), c(0x1.37be7a0000000p-1, 0x1.906fbc0000000p+3, -0x1.6d0bd40000000p-1, -0x1.6c577e0000000p-3), "float32"),
  exponential = list(c(4), c(0x1.1f02a00000000p+0, 0x1.d63ada0000000p+1, 0x1.7167420000000p-2, 0x1.2c7fc80000000p-1), "float32"),
  gumbel = list(c(4), c(0x1.dc5ec40000000p-1, 0x1.d496820000000p+1, -0x1.6c0d880000000p-3, 0x1.aa72000000000p-3), "float32"),
  gumbel_high = list(c(4), c(-0x1.d456b40000000p-4, -0x1.4d1ae20000000p+0, 0x1.05035a0000000p+0, 0x1.10d5960000000p-1), "float32"),
  gumbel_highest = list(c(4), c(-0x1.d456ec0000000p-4, -0x1.4d1ae20000000p+0, 0x1.0503580000000p+0, 0x1.10d5960000000p-1), "float32"),
  laplace = list(c(4), c(-0x1.b642200000000p-2, -0x1.7d81d80000000p+1, 0x1.00b6660000000p-1, 0x1.e6eb760000000p-4), "float32"),
  logistic = list(c(4), c(0x1.7417880000000p-1, 0x1.d2f05c0000000p+1, -0x1.aae6f80000000p-1, -0x1.cd07600000000p-3), "float32"),
  pareto = list(c(4), c(0x1.c06d220000000p+0, 0x1.91b48a0000000p+2, 0x1.329a160000000p+0, 0x1.574f580000000p+0), "float32"),
  rayleigh = list(c(4), c(0x1.550b360000000p+0, 0x1.5c4f020000000p-2, 0x1.28c47c0000000p+1, 0x1.e95ccc0000000p+0), "float32"),
  wald = list(c(4), c(0x1.619e880000000p-2, 0x1.26ea820000000p+0, 0x1.31678a0000000p-1, 0x1.4d26620000000p-1), "float32"),
  maxwell = list(c(4), c(0x1.08ff280000000p+1, 0x1.6dcf0e0000000p-1, 0x1.7ffeaa0000000p+0, 0x1.b4a6fc0000000p+0), "float32"),
  dsmaxwell = list(c(4), c(-0x1.7a29340000000p+0, -0x1.0ae3020000000p+2, 0x1.e152f80000000p+1, 0x1.77d5900000000p+2), "float32"),
  weibull = list(c(4), c(0x1.1446f20000000p+1, 0x1.30bff60000000p+2, 0x1.0375c80000000p+0, 0x1.66e8f20000000p+0), "float32"),
  triangular = list(c(4), c(0x1.0b732c0000000p-1, 0x1.bbc0740000000p-1, 0x1.34a84c0000000p-2, 0x1.8125040000000p-2), "float32"),
  truncnorm = list(c(4), c(0x1.1c06100000000p-1, 0x1.b60c8a0000000p+0, -0x1.e425660000000p-3, 0x1.c592c60000000p-5), "float32"),
  lognormal = list(c(4), c(0x1.40caec0000000p+0, 0x1.53ef780000000p+1, 0x1.8b857a0000000p-1, 0x1.dd29480000000p-1), "float32"),
  rademacher = list(c(8), c(-1, -1, 1, 1, -1, -1, 1, 1), "int32"),
  geometric = list(c(6), c(2, 1, 4, 3, 1, 2), "int32"),
  poisson_small = list(c(6), c(4, 1, 1, 3, 2, 7), "int32"),
  poisson_large = list(c(6), c(28, 30, 23, 35, 24, 25), "int32"),
  poisson_vec = list(c(4), c(0, 1, 11, 98), "int32"),
  gamma_half = list(c(4), c(0x1.ea0fdc0000000p-13, 0x1.355d360000000p-2, 0x1.8eabee0000000p-9, 0x1.2b69ea0000000p+1), "float32"),
  gamma_2 = list(c(4), c(0x1.0945060000000p+1, 0x1.ba31a00000000p-1, 0x1.dfb1e00000000p-1, 0x1.cf80720000000p+1), "float32"),
  gamma_vec = list(c(3), c(0x1.32de780000000p-21, 0x1.be4b2e0000000p-3, 0x1.ad1a0a0000000p+1), "float32"),
  loggamma = list(c(4), c(-0x1.0b91e40000000p+3, -0x1.326b800000000p+0, -0x1.72e73e0000000p+2, 0x1.b3187c0000000p-1), "float32"),
  beta = list(c(4), c(0x1.97ca220000000p-2, 0x1.605b5c0000000p-2, 0x1.11f9820000000p-2, 0x1.6cf3a40000000p-2), "float32"),
  dirichlet = list(c(3), c(0x1.10f4040000000p-2, 0x1.f8950c0000000p-3, 0x1.f2c1780000000p-2), "float32"),
  chisquare = list(c(4), c(0x1.82d3140000000p+1, 0x1.0b35e20000000p+0, 0x1.27dacc0000000p+0, 0x1.71b7160000000p+2), "float32"),
  f = list(c(4), c(0x1.ef74900000000p-2, 0x1.1b8c1c0000000p+0, 0x1.a5cfba0000000p+0, 0x1.4f19ee0000000p-1), "float32"),
  t = list(c(4), c(0x1.fd40de0000000p+0, 0x1.2bf6020000000p-3, -0x1.fc16720000000p-2, -0x1.6099ba0000000p-1), "float32"),
  gennorm = list(c(4), c(-0x1.cf751e0000000p-4, -0x1.9277180000000p-1, 0x1.e0a7e80000000p-5, 0x1.447b5c0000000p-2), "float32"),
  ball = list(c(2, 3), c(-0x1.749e0c0000000p-2, 0x1.940e320000000p-2, 0x1.ea87920000000p-4, -0x1.f2f4ba0000000p-2, 0x1.10da420000000p-3, -0x1.282d8e0000000p-3), "float32"),
  categorical = list(c(2), c(1, 0), "int32"),
  categorical_shape = list(c(5), c(1, 2, 1, 2, 2), "int32"),
  categorical_ax0 = list(c(3), c(1, 0, 0), "int32"),
  binomial_small = list(c(6), c(2.0, 2.0, 2.0, 2.0, 3.0, 1.0), "float32"),
  binomial_large = list(c(6), c(43.0, 45.0, 29.0, 49.0, 39.0, 40.0), "float32"),
  binomial_highp = list(c(4), c(18.0, 17.0, 17.0, 17.0), "float32"),
  gamma_approx = list(c(4), c(0x1.9e92720000000p+0, 0x1.688a240000000p+0, 0x1.b001ca0000000p+0, 0x1.31fc660000000p+0), "float32"),
  loggamma_approx = list(c(4), c(-0x1.12d48a0000000p+1, -0x1.7c38260000000p+1, -0x1.3248d40000000p+0, -0x1.c870e00000000p+0), "float32"),
  poisson_approx = list(c(3, 2), c(2, 41, 1, 29, 3, 32), "int32"),
  gamma_vmap = list(c(3, 2), c(0x1.0154100000000p-1, 0x1.0e2d740000000p+0, 0x1.63a7860000000p-2, 0x1.5694e00000000p+0, 0x1.aa8edc0000000p+0, 0x1.cee6cc0000000p-1), "float32"),
  gamma_grad = list(integer(0), c(0x1.428e920000000p+1), "float32"),
  permutation = list(c(10), c(9, 4, 2, 3, 5, 0, 7, 6, 1, 8), "int32"),
  perm_arr = list(c(4, 3), c(6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0), "float32"),
  choice = list(c(4), c(9, 2, 0, 7), "int32"),
  choice_norep = list(c(4), c(9, 4, 2, 3), "int32"),
  choice_p = list(c(6), c(2, 0, 3, 2, 1, 2), "int32"),
  multinomial = list(c(3), c(2.0, 4.0, 4.0), "float32"),
  mvn = list(c(3, 2), c(0x1.46baa80000000p-1, 0x1.fca05a0000000p+0, -0x1.75cc1c0000000p-1, -0x1.41e67c0000000p-2, 0x1.bdaace0000000p-1, 0x1.0f881e0000000p-1), "float32"),
  orthogonal = list(c(3, 3), c(0x1.d764f40000000p-1, 0x1.5cf59c0000000p-2, -0x1.85a1100000000p-3, -0x1.26794a0000000p-2, 0x1.d802440000000p-1, 0x1.09e1320000000p-2, -0x1.0e34660000000p-2, 0x1.798a4a0000000p-3, -0x1.e4c0120000000p-1), "float32")
)

rr_k0 <- function() random_PRNGKey(0)
rr_k42 <- function() random_PRNGKey(42)
rr_kd <- function() random_PRNGKey(7)

test_that("uniform, normal and randint match JAX bit for bit", {
  skip_if_no_backend()
  k0 <- rr_k0()
  k42 <- rr_k42()
  rr_expect(random_uniform(k0, 5), rr_ref$uniform_0_5)
  rr_expect(random_uniform(k42, c(2, 3), minval = -2, maxval = 3), rr_ref$uniform_42_23_m)
  rr_expect(random_uniform(k0, c(2, 3), minval = c(0, 1, 2), maxval = 5), rr_ref$uniform_0_minvec)
  rr_expect(jnp_astype(random_uniform(k0, 4, "bfloat16"), "float32"), rr_ref$unif_bf16)
  rr_expect(jnp_astype(random_uniform(k0, 4, "float16"), "float32"), rr_ref$unif_f16)
  rr_expect(random_normal(k0, 5), rr_ref$normal_0_5)
  rr_expect(random_normal(k42, c(2, 3)), rr_ref$normal_42_23)
  # complex normals divide by sqrt(2) in complex arithmetic (last ulp may differ)
  rr_expect(random_normal(k0, 3, "complex64"), rr_ref$normal_0_c64, tol = 1e-6)
  rr_expect(random_randint(k0, 6, 0, 10), rr_ref$randint_0_6)
  rr_expect(random_randint(k42, 5, -5, 5), rr_ref$randint_42_neg)
  rr_expect(random_randint(k0, 6, 0, 256, "uint8"), rr_ref$randint_u8)
  rr_expect(random_randint(k42, 6, -128, 127, "int8"), rr_ref$randint_i8)
  rr_expect(random_randint(k0, c(2, 3), jnp_asarray(c(0L, 10L, 100L)), jnp_asarray(c(5L, 20L, 1000L))),
            rr_ref$randint_vec)
  rr_expect(random_randint(k0, 4, 5, 5), rr_ref$randint_empty)
  rr_expect(random_uniform(random_key(0, impl = "rbg"), 4), rr_ref$rbg_uniform)
  with_jax_config("jax_threefry_partitionable", FALSE, {
    rr_expect(random_uniform(k0, 5), rr_ref$np_uniform_0_5)
    rr_expect(random_normal(k42, c(2, 3)), rr_ref$np_normal_42)
  })
  # typed and raw keys give the same samples
  rr_expect(random_uniform(random_key(0), 5), rr_ref$uniform_0_5)
  expect_identical(jaxR::dtype(random_normal(k0, c(2, 2), "float16")), "float16")
  expect_identical(dim(random_uniform(k0)), integer(0))
})

test_that("float64 and int64 samplers match JAX with x64", {
  skip_if_no_backend()
  old <- enable_x64(TRUE)
  on.exit(jax_config_update("jax_enable_x64", old), add = TRUE)
  k0 <- random_PRNGKey(0)
  k42 <- random_PRNGKey(42)
  kd <- random_PRNGKey(7)
  ref <- list(
    uniform_0_5 = list(c(5), c(0x1.ac80056666e90p-2, 0x1.baf91c7e6ed88p-3, 0x1.ee3e9d53441c8p-1, 0x1.2624ef17c9d1ap-1, 0x1.107ffd8233b78p-1), "float64"),
    uniform_42_23_m = list(c(2, 3), c(0x1.11b0b65a85528p-3, -0x1.ecc9801b9aed8p+0, 0x1.bdcf492498382p-1, 0x1.a62b0e5f0a7a2p+0, 0x1.70626a72d08a1p+0, 0x1.383d84eef5ab9p+0), "float64"),
    normal_0_5 = list(c(5), c(-0x1.a5909049a141dp-3, -0x1.91ccd1e1b7b4ap-1, 0x1.d0eb0e61cf797p+0, 0x1.80b45ca1069abp-3, 0x1.4b3c1e85a437cp-4), "float64"),
    normal_0_c128 = list(c(3), list(re = c(0x1.54525f0caf907p+0, -0x1.5c6fd60ee6c42p-2, 0x1.2cd2bb18883d6p-2), im = c(-0x1.fb2c99b1646acp-1, 0x1.033ef5b249fe3p+0, 0x1.c46948f233f50p-2)), "complex128"),
    randint_0_6 = list(c(6), c(5, 1, 5, 7, 3, 5), "int64"),
    randint_42_neg = list(c(5), c(0, 3, -2, -4, 0), "int64"),
    np_uniform_0_5 = list(c(5), c(0x1.6b6c4379c215cp-2, 0x1.3559954e8082cp-1, 0x1.b5d8ab6d1bf60p-2, 0x1.d84d30a6247f0p-3, 0x1.51c675f6dbe64p-2), "float64"),
    exponential = list(c(4), c(0x1.dbcc3ae02525bp+0, 0x1.7cbb5bbb546a6p-5, 0x1.04fd4608f0bbcp-2, 0x1.1098ae889eb5cp-4), "float64"),
    gumbel = list(c(4), c(0x1.c6693fe6e4da2p+0, -0x1.20f98127f8ecfp+0, -0x1.9987c40dc880dp-2, -0x1.024e065e624afp+0), "float64"),
    cauchy = list(c(4), c(0x1.e03ca55593e57p+0, -0x1.bd8bf53d919c3p+2, -0x1.2bc44d666f5b9p+0, -0x1.381398b0cb071p+2), "float64"),
    truncnorm = list(c(4), c(0x1.08eda6db214acp+0, -0x1.b695cfe936c7dp-1, -0x1.9e79c4ed6d620p-2, -0x1.9a7976409d3c8p-1), "float64"),
    gamma_2 = list(c(4), c(0x1.00e703d8530b8p+2, 0x1.78fbd85624388p-1, 0x1.d93c6074d0144p-1, 0x1.9bfdda4ac2ac4p+0), "float64"),
    gamma_half = list(c(4), c(0x1.5e5cc1d122b38p-11, 0x1.7dcab9d42e6dap-6, 0x1.fe1fd97a51ebdp-4, 0x1.1a247080c1c4ep-2), "float64"),
    beta = list(c(4), c(0x1.c9060e7129be0p-3, 0x1.369538362ada2p-1, 0x1.00521470f77d1p-1, 0x1.584f73d289d9dp-2), "float64"),
    dirichlet = list(c(3), c(0x1.0005eb3ba755bp-1, 0x1.36b9f7ad72415p-3, 0x1.64972db1f833ep-2), "float64"),
    gamma_approx = list(c(4), c(0x1.01e354fbf2db3p+2, 0x1.29de8a75422e7p+0, 0x1.43382b4d745d0p+0, 0x1.71ce005782730p+1), "float64"),
    poisson_small = list(c(6), c(4, 1, 1, 3, 2, 7), "int64"),
    poisson_large = list(c(6), c(28, 30, 23, 35, 24, 25), "int64"),
    binomial_large = list(c(6), c(37.0, 38.0, 44.0, 42.0, 39.0, 37.0), "float64"),
    geometric = list(c(6), c(1, 9, 5, 8, 13, 1), "int64"),
    rademacher = list(c(8), c(-1, 1, 1, 1, 1, -1, -1, -1), "int64"),
    categorical = list(c(2), c(0, 0), "int64")
  )
  rr_expect(random_uniform(k0, 5), ref$uniform_0_5)
  rr_expect(random_uniform(k42, c(2, 3), minval = -2, maxval = 3), ref$uniform_42_23_m)
  rr_expect(random_normal(k0, 5), ref$normal_0_5)
  rr_expect(random_normal(k0, 3, "complex128"), ref$normal_0_c128)
  rr_expect(random_randint(k0, 6, 0, 10), ref$randint_0_6)
  rr_expect(random_randint(k42, 5, -5, 5), ref$randint_42_neg)
  with_jax_config("jax_threefry_partitionable", FALSE, rr_expect(random_uniform(k0, 5), ref$np_uniform_0_5))
  rr_expect(random_exponential(kd, 4), ref$exponential)
  rr_expect(random_gumbel(kd, 4), ref$gumbel)
  rr_expect(random_cauchy(kd, 4), ref$cauchy)
  # erf in float64 may differ from the wheel in the last ulp
  rr_expect(random_truncated_normal(kd, -1, 2, 4), ref$truncnorm, tol = 1e-14)
  rr_expect(random_gamma(kd, 2, 4), ref$gamma_2)
  rr_expect(random_gamma(kd, 0.5, 4), ref$gamma_half)
  rr_expect(random_beta(kd, 2, 3, 4), ref$beta)
  rr_expect(random_dirichlet(kd, c(1, 2, 3)), ref$dirichlet)
  rr_expect(random_gamma(kd, 2, 4, method = "approximate"), ref$gamma_approx, tol = 1e-14)
  rr_expect(random_poisson(kd, 3, 6), ref$poisson_small)
  rr_expect(random_poisson(kd, 25, 6), ref$poisson_large)
  rr_expect(random_binomial(kd, 100, 0.4, 6), ref$binomial_large)
  rr_expect(random_geometric(kd, 0.3, 6), ref$geometric)
  rr_expect(random_rademacher(kd, 8), ref$rademacher)
  rr_expect(random_categorical(kd, rbind(c(0.1, 1, 2), c(3, 0, -1))), ref$categorical)
  # float32 can still be requested
  rr_expect(random_uniform(k0, 5, "float32"), rr_ref$uniform_0_5)
})

test_that("continuous distributions match JAX", {
  skip_if_no_backend()
  kd <- rr_kd()
  rr_expect(random_cauchy(kd, 4), rr_ref$cauchy)
  rr_expect(random_exponential(kd, 4), rr_ref$exponential)
  rr_expect(random_gumbel(kd, 4), rr_ref$gumbel)
  rr_expect(random_gumbel(kd, 4, mode = "high"), rr_ref$gumbel_high)
  rr_expect(random_gumbel(kd, 4, mode = "highest"), rr_ref$gumbel_highest)
  rr_expect(random_laplace(kd, 4), rr_ref$laplace)
  rr_expect(random_logistic(kd, 4), rr_ref$logistic)
  rr_expect(random_pareto(kd, 2, 4), rr_ref$pareto)
  rr_expect(random_rayleigh(kd, 1.5, 4), rr_ref$rayleigh)
  rr_expect(random_wald(kd, 1, 4), rr_ref$wald)
  rr_expect(random_maxwell(kd, 4), rr_ref$maxwell)
  rr_expect(random_double_sided_maxwell(kd, 1, 2, 4), rr_ref$dsmaxwell)
  rr_expect(random_weibull_min(kd, 2, 1.5, 4), rr_ref$weibull)
  rr_expect(random_triangular(kd, 0, 0.3, 1, 4), rr_ref$triangular)
  rr_expect(random_truncated_normal(kd, -1, 2, 4), rr_ref$truncnorm)
  rr_expect(random_lognormal(kd, 0.5, 4), rr_ref$lognormal)
  rr_expect(random_generalized_normal(kd, 1.5, 4), rr_ref$gennorm)
  rr_expect(random_ball(kd, 3, 2, 2), rr_ref$ball)
  tn <- as.array(random_truncated_normal(kd, c(-0.5, 0), c(0.5, 0.1), c(50, 2)))
  expect_true(all(tn[, 1] > -0.5 & tn[, 1] < 0.5 & tn[, 2] > 0 & tn[, 2] < 0.1))
})

test_that("discrete distributions match JAX", {
  skip_if_no_backend()
  kd <- rr_kd()
  rr_expect(random_bernoulli(kd, 0.3, 8), rr_ref$bernoulli)
  rr_expect(random_bernoulli(kd, 0.3, 8, mode = "high"), rr_ref$bernoulli_high)
  rr_expect(random_bernoulli(kd, c(0.1, 0.5, 0.9)), rr_ref$bernoulli_vec)
  rr_expect(random_rademacher(kd, 8), rr_ref$rademacher)
  rr_expect(random_geometric(kd, 0.3, 6), rr_ref$geometric)
  rr_expect(random_poisson(kd, 3, 6), rr_ref$poisson_small)
  rr_expect(random_poisson(kd, 25, 6), rr_ref$poisson_large)
  rr_expect(random_poisson(kd, c(0, 1.5, 12, 100)), rr_ref$poisson_vec)
  rr_expect(random_categorical(kd, rbind(c(0.1, 1, 2), c(3, 0, -1))), rr_ref$categorical)
  rr_expect(random_categorical(kd, c(0.1, 1, 2), shape = 5), rr_ref$categorical_shape)
  rr_expect(random_categorical(kd, rbind(c(0.1, 1, 2), c(3, 0, -1)), axis = 0), rr_ref$categorical_ax0)
  rr_expect(random_binomial(kd, 5, 0.3, 6), rr_ref$binomial_small)
  rr_expect(random_binomial(kd, 100, 0.4, 6), rr_ref$binomial_large)
  rr_expect(random_binomial(kd, 20, 0.9, 4), rr_ref$binomial_highp)
  rr_expect(random_choice(kd, 10, 4), rr_ref$choice)
  expect_identical(jaxR::dtype(random_rademacher(kd, 3, "float32")), "float32")
  b <- as.array(random_binomial(kd, c(NaN, -1, Inf, 3), c(0.5, 0.5, 0.5, NaN)))
  expect_true(is.nan(b[1]) && is.nan(b[2]) && b[3] == Inf && is.nan(b[4]))
  expect_error(random_poisson(random_key(0, impl = "rbg"), 3), "only implemented for the threefry2x32 RNG")
})

test_that("gamma family samplers match JAX", {
  skip_if_no_backend()
  kd <- rr_kd()
  rr_expect(random_gamma(kd, 0.5, 4), rr_ref$gamma_half)
  rr_expect(random_gamma(kd, 2, 4), rr_ref$gamma_2)
  rr_expect(random_gamma(kd, c(0.3, 1, 5)), rr_ref$gamma_vec)
  rr_expect(random_loggamma(kd, 0.5, 4), rr_ref$loggamma)
  rr_expect(random_beta(kd, 2, 3, 4), rr_ref$beta)
  rr_expect(random_dirichlet(kd, c(1, 2, 3)), rr_ref$dirichlet)
  rr_expect(random_chisquare(kd, 3, 4), rr_ref$chisquare)
  rr_expect(random_f(kd, 3, 5, 4), rr_ref$f)
  rr_expect(random_t(kd, 4, 4), rr_ref$t)
  rr_expect(random_gamma(kd, 2, 4, method = "approximate"), rr_ref$gamma_approx)
  rr_expect(random_loggamma(kd, 0.5, 4, method = "approximate"), rr_ref$loggamma_approx)
  rr_expect(vmap(function(k) random_gamma(k, 1.5, 2))(random_split(kd, 3)), rr_ref$gamma_vmap)
  # log space keeps tiny samples representable
  rr_expect(random_loggamma(kd, c(0.01, 0.1), 2), list(2, c(-0x1.b6c5e20000000p+8, -0x1.01bc2c0000000p+2), "float32"))
  # the primitive is staged under jit and batched by vmap
  expect_s3_class(random_random_gamma_p, "jax_primitive")
  expect_match(paste(capture.output(print(make_jaxpr(function(k) random_gamma(k, 2, 3))(kd))),
                     collapse = "\n"), "random_gamma")
})

test_that("samplers are the same under jit and match a loop under vmap", {
  skip_if_no_backend()
  kd <- rr_kd()
  rr_expect(jit(function(k) random_normal(k, 5))(rr_k0()), rr_ref$normal_0_5)
  rr_expect(jit(function(k, a) random_gamma(k, a, 4))(kd, 2), rr_ref$gamma_2)
  rr_expect(jit(function(k) random_poisson(k, 3, 6))(kd), rr_ref$poisson_small)
  # With the parameters constant inside an outer jit, XLA folds them; JAX's
  # jit(lambda k: beta(k, 2., 3., (4,))) gives these values too.
  rr_expect(jit(function(k) random_beta(k, 2, 3, 4))(random_key(7)),
            list(4, c(0x1.97ca200000000p-2, 0x1.605b5a0000000p-2, 0x1.11f9820000000p-2,
                      0x1.6cf3a40000000p-2), "float32"))
  rr_expect(vmap(function(k) random_uniform(k, 3))(random_split(rr_k0(), 4)), rr_ref$vmap_uniform)
  keys <- random_split(random_key(7), 3)
  v <- as.array(vmap(function(k) random_normal(k, 2))(keys))
  expected <- c(0x1.1d32320000000p+0, 0x1.220f960000000p-3, 0x1.d108600000000p-2, 0x1.e0d9f20000000p-1,
                -0x1.a409a80000000p-1, -0x1.4bdb600000000p-1)
  expect_identical(as.numeric(rr_c_ravel(v)), expected)
  kl <- as.list(keys)
  for (i in 1:3) expect_identical(v[i, ], as.vector(as.array(random_normal(kl[[i]], 2))))
  # vmap over a parameter
  vp <- as.array(vmap(function(p) random_exponential(kd, 4) / p)(jnp_asarray(c(1, 2))))
  expect_equal(vp[2, ], vp[1, ] / 2)
})

test_that("reparameterized samples have the JAX gradients", {
  skip_if_no_backend()
  kd <- rr_kd()
  # d/ds sum(2 + s * z) = sum(z)
  g <- grad(function(s) jnp_sum(2 + s * random_normal(kd, 3)))(1.5)
  expect_identical(as.numeric(as.array(g)), 0x1.e3713c0000000p+0)
  old <- enable_x64(TRUE)
  on.exit(jax_config_update("jax_enable_x64", old), add = TRUE)
  k <- random_PRNGKey(3)
  check_grad_fd(function(b) jnp_sum(random_uniform(k, 4, minval = 0.5, maxval = b)), as_jax(2, "float64"))
  check_grad_fd(function(u) jnp_sum(random_truncated_normal(k, -1, u, 4)), as_jax(1.5, "float64"))
  check_grad_fd(function(b) jnp_sum(random_pareto(k, b, 3)), as_jax(2.5, "float64"))
  check_grad_fd(function(s) jnp_sum(random_lognormal(k, s, 3)), as_jax(0.7, "float64"))
  # second order: d2/db2 sum(exp(e / b)) = sum(exp(e / b) (e^2 / b^4 + 2 e / b^3))
  e <- as.array(random_exponential(k, 3))
  h <- grad(grad(function(b) jnp_sum(random_pareto(k, b, 3))))(as_jax(2.5, "float64"))
  expect_equal(as.numeric(as.array(h)), sum(exp(e / 2.5) * (e^2 / 2.5^4 + 2 * e / 2.5^3)),
               tolerance = 1e-10)
})

test_that("gamma gradients use random_gamma_grad", {
  skip_if_no_backend()
  rr_skip_missing("lax_random_gamma_grad")
  kd <- rr_kd()
  rr_expect(grad(function(a) jnp_sum(random_gamma(kd, a, 3)))(2), rr_ref$gamma_grad, tol = 1e-6)
  g <- grad(function(a) random_dirichlet(kd, a)[0])(jnp_asarray(c(1, 2, 3)))
  expect_equal(as.array(g)[2], -0x1.ae3b820000000p-5, tolerance = 1e-5)
})

test_that("permutation and choice match JAX", {
  skip_if_no_backend()
  rr_skip_missing("lax_sort_key_val")
  kd <- rr_kd()
  rr_expect(random_permutation(kd, 10), rr_ref$permutation)
  rr_expect(random_choice(kd, 10, 4, replace = FALSE), rr_ref$choice_norep)
  rr_skip_missing("jnp_take")
  rr_expect(random_permutation(kd, jnp_reshape(jnp_arange(12, dtype = "float32"), c(4, 3))), rr_ref$perm_arr)
  rr_expect(random_choice(kd, jnp_arange(5L) * 10L, c(2, 2)), list(c(2, 2), c(40, 20, 0, 20), "int32"))
  rr_skip_missing("jnp_cumsum", "jnp_searchsorted")
  rr_expect(random_choice(kd, 5, 6, p = c(0.1, 0.2, 0.3, 0.2, 0.2)), rr_ref$choice_p)
})

test_that("samplers built on cumsum, top_k, linalg and ndtr match JAX", {
  skip_if_no_backend()
  kd <- rr_kd()
  if (exists("lax_cumsum", envir = asNamespace("jaxR"), inherits = FALSE))
    rr_expect(random_multinomial(kd, 10, c(0.2, 0.3, 0.5)), rr_ref$multinomial)
  if (exists("lax_top_k", envir = asNamespace("jaxR"), inherits = FALSE)) {
    rr_expect(random_categorical(kd, c(0.1, 1, 2, 0.5), shape = 2, replace = FALSE), list(2, c(1, 2), "int32"))
    rr_expect(random_categorical(kd, rbind(c(0.1, 1, 2, 0.5), c(1, 1, 1, 1)), shape = c(3, 2), replace = FALSE),
              list(c(3, 2), c(1, 0, 2, 1, 0, 2), "int32"))
  }
  if (exists("scipy_special_ndtr", envir = asNamespace("jaxR"), inherits = FALSE))
    rr_expect(random_poisson(kd, c(2, 30), c(3, 2), method = "approximate"), rr_ref$poisson_approx)
  if (exists("jnp_linalg_cholesky", envir = asNamespace("jaxR"), inherits = FALSE)) {
    cov <- rbind(c(2, 0.5), c(0.5, 1))
    rr_expect(random_multivariate_normal(kd, c(0, 0), cov, 3), rr_ref$mvn, tol = 1e-6)
    # LAPACK factorizations may differ from the wheel in the last bits
    rr_expect(random_multivariate_normal(kd, c(0, 0), cov, 3, method = "svd"),
              list(c(3, 2), c(-1.2849966287612915, 1.3504947423934937, 0.7565892338752747,
                              0.17754970490932465, -0.9587190747261047, -0.07489647716283798), "float32"),
              tol = 1e-5)
    rr_expect(random_multivariate_normal(kd, c(0, 0), cov, 3, method = "eigh"),
              list(c(3, 2), c(-2.527439594268799, -1.4818052053451538, 0.017533985897898674,
                              0.5048202276229858, -0.2491290122270584, -0.6964152455329895), "float32"),
              tol = 1e-5)
  }
  if (exists("jnp_linalg_qr", envir = asNamespace("jaxR"), inherits = FALSE)) {
    rr_expect(random_orthogonal(kd, 3), rr_ref$orthogonal, tol = 1e-5)
    rr_expect(random_orthogonal(kd, 2, m = 3),
              list(c(2, 3), c(0.48973512649536133, -0.5602856278419495, 0.6680116653442383,
                              0.8697620034217834, 0.36721742153167725, -0.3296443819999695), "float32"),
              tol = 1e-5)
  }
  succeed()
})

test_that("argument errors follow JAX", {
  skip_if_no_backend()
  k <- rr_k0()
  expect_error(random_uniform(k, 3, dtype = "int32"), "dtype argument to `uniform` must be a float dtype")
  expect_error(random_normal(k, 3, dtype = "int32"), "must be a float or complex dtype")
  expect_error(random_randint(k, 3, 0, 5, dtype = "float32"), "randint only accepts integer dtypes")
  expect_error(random_bernoulli(k, 1L), "bernoulli probability `p` must have a floating dtype")
  expect_error(random_bernoulli(k, 0.5, mode = "mid"), "expected 'high' or 'low'")
  expect_error(random_normal(k, -1), "Shapes must be 1D sequences")
  expect_error(random_beta(k, matrix(1, 3, 2), 1, shape = 2), "beta parameter shapes must be broadcast-compatible")
  expect_error(random_gamma(k, 1, method = "fast"), "must be one of \\{'exact', 'approximate'\\}")
  expect_error(random_gumbel(k, 2, mode = "medium"), "Must provide valid mode for gumbel")
  expect_error(random_choice(k, 3, 5, replace = FALSE), "Cannot take a larger sample \\(size 5\\)")
  expect_error(random_choice(k, 0, 2), "a must be greater than 0")
  expect_error(random_choice(k, 5, 2, p = c(0.5, 0.5)), "p must be None or a 1D vector")
  expect_error(random_categorical(k, c(0, 1), shape = 3, replace = FALSE),
               "Number of samples without replacement \\(3\\) cannot exceed number of categories \\(2\\)")
  expect_error(random_dirichlet(k, 1), "dirichlet requires alpha.ndim >= 1")
  expect_error(random_multivariate_normal(k, c(0, 0), diag(2), method = "qr"), "method must be one of")
  expect_error(random_permutation(k, 2.5), "x must be an integer or at least 1-dimensional")
  expect_error(random_poisson(k, 3, method = "slow"), "method argument to `poisson`")
  old <- enable_x64(TRUE)
  on.exit(jax_config_update("jax_enable_x64", old), add = TRUE)
  expect_error(random_pareto(k, jnp_asarray(2, dtype = "float64"), dtype = "float32"),
               "cannot safely cast argument of type float64 to float32")
})
