# jax.example_libraries.optimizers (R/example_optimizers.R). Expected values were produced
# by jax 0.12.0.1 on CPU: four steps of each optimizer on the gradient of a small loss.
# Arrays are listed in C order as list(shape, values, dtype).

opt_t_c_ravel <- function(x) {
  d <- dim(x)
  if (length(d) <= 1L) return(as.vector(x))
  as.vector(aperm(x, length(d):1))
}

opt_t_expect <- function(x, ref) {
  if (inherits(x, "jax_value")) {
    expect_identical(jaxR::dtype(x), ref[[3L]])
    expect_identical(as.integer(dim(x)), as.integer(ref[[1L]]))
    x <- opt_t_c_ravel(as.array(x))
  }
  expect_identical(as.numeric(x), as.numeric(ref[[2L]]))
}

opt_ref <- list(
  sgd_w = list(c(3), c(0x1.9724740000000p-1, 0x1.7247460000000p-3, 0x1.d495100000000p-3), "float32"),
  sgd_m = list(c(2, 3), c(0x1.4fec560000000p-1, -0x1.4fec560000000p+0, 0x1.4fec560000000p-2, 0, 0x1.f7e2820000000p+0, -0x1.4fec560000000p-1), "float32"),
  sgd_b = list(integer(0), c(0x1.6d330a0000000p-1), "float32"),
  momentum_w = list(c(3), c(0x1.2780340000000p+0, 0x1.9e00d20000000p+0, -0x1.ed013a0000000p+0), "float32"),
  momentum_m = list(c(2, 3), c(0x1.d07c800000000p-3, -0x1.d07c800000000p-2, 0x1.d07c800000000p-4, 0, 0x1.5c5d6a0000000p-1, -0x1.d07c800000000p-3), "float32"),
  momentum_b = list(integer(0), c(0x1.374d160000000p+0), "float32"),
  nesterov_w = list(c(3), c(0x1.2993cc0000000p+0, 0x1.a64f320000000p+0, -0x1.f976cc0000000p+0), "float32"),
  nesterov_m = list(c(2, 3), c(0x1.80909c0000000p-4, -0x1.80909c0000000p-3, 0x1.80909c0000000p-5, 0, 0x1.206c700000000p-2, -0x1.80909c0000000p-4), "float32"),
  nesterov_b = list(integer(0), c(0x1.3a35520000000p+0), "float32"),
  adagrad_w = list(c(3), c(0x1.2332120000000p-1, -0x1.dc80940000000p-1, 0x1.ee3c1a0000000p+0), "float32"),
  adagrad_m = list(c(2, 3), c(0x1.dc99f60000000p-1, -0x1.ee404a0000000p+0, 0x1.b99bdc0000000p-2, 0, 0x1.771e0c0000000p+1, -0x1.dc99f60000000p-1), "float32"),
  adagrad_b = list(integer(0), c(0x1.79d3200000000p-2), "float32"),
  rmsprop_w = list(c(3), c(0x1.fc07c40000000p-1, -0x1.5bd4f60000000p-3, 0x1.241c4e0000000p+0), "float32"),
  rmsprop_m = list(c(2, 3), c(0x1.175f6a0000000p-2, -0x1.2b7a9e0000000p+0, 0x1.fc1df80000000p-8, 0, 0x1.120e280000000p+1, -0x1.175f6a0000000p-2), "float32"),
  rmsprop_b = list(integer(0), c(0x1.d988100000000p-1), "float32"),
  rmsprop_momentum_w = list(c(3), c(0x1.71471c0000000p+0, 0x1.fdb3120000000p-1, -0x1.7ef5900000000p-4), "float32"),
  rmsprop_momentum_m = list(c(2, 3), c(-0x1.4063220000000p-1, -0x1.2676000000000p-8, -0x1.c51c700000000p-2, 0, 0x1.d021560000000p-1, 0x1.4063220000000p-1), "float32"),
  rmsprop_momentum_b = list(integer(0), c(0x1.99aeea0000000p+0), "float32"),
  adam_w = list(c(3), c(0x1.c62f7a0000000p-1, -0x1.33f8f60000000p-1, 0x1.99d5ba0000000p+0), "float32"),
  adam_m = list(c(2, 3), c(0x1.3538020000000p-1, -0x1.99fc7c0000000p+0, 0x1.ce842e0000000p-4, 0, 0x1.4ceade0000000p+1, -0x1.3538020000000p-1), "float32"),
  adam_b = list(integer(0), c(0x1.62d02e0000000p-1), "float32"),
  adamax_w = list(c(3), c(0x1.ad93be0000000p-1, -0x1.3b139a0000000p-1, 0x1.9c2f7e0000000p+0), "float32"),
  adamax_m = list(c(2, 3), c(0x1.430eec0000000p-1, -0x1.9d89cc0000000p+0, 0x1.49b1060000000p-3, 0, 0x1.4e17c00000000p+1, -0x1.430eec0000000p-1), "float32"),
  adamax_b = list(integer(0), c(0x1.4fdc1e0000000p-1), "float32"),
  sm3_w = list(c(3), c(0x1.2332120000000p-1, -0x1.dc80940000000p-1, 0x1.ee3c1a0000000p+0), "float32"),
  sm3_m = list(c(2, 3), c(0x1.dc99f60000000p-1, -0x1.ee404a0000000p+0, 0x1.c7aada0000000p-2, 0, 0x1.771e0c0000000p+1, -0x1.dc99f60000000p-1), "float32"),
  sm3_b = list(integer(0), c(0x1.79d3200000000p-2), "float32"),
  sgd_expdecay_w = list(c(3), c(0x1.6cefde0000000p-1, -0x1.3102220000000p-3, 0x1.7260cc0000000p-1), "float32"),
  sgd_expdecay_m = list(c(2, 3), c(0x1.886c440000000p-1, -0x1.886c440000000p+0, 0x1.886c440000000p-2, 0, 0x1.2651340000000p+1, -0x1.886c440000000p-1), "float32"),
  sgd_expdecay_b = list(integer(0), c(0x1.321c9c0000000p-1), "float32"),
  adam_poly_w = list(c(3), c(0x1.5b755a0000000p-1, -0x1.a3f0580000000p-1, 0x1.d1f3680000000p+0), "float32"),
  adam_poly_m = list(c(2, 3), c(0x1.a414fe0000000p-1, -0x1.d1f82c0000000p+0, 0x1.4915500000000p-2, 0, 0x1.68f9b20000000p+1, -0x1.a414fe0000000p-1), "float32"),
  adam_poly_b = list(integer(0), c(0x1.eab4800000000p-2), "float32"),
  momentum_pw_w = list(c(3), c(0x1.996d0a0000000p-1, 0x1.96d08e0000000p-3, 0x1.9dc7200000000p-3), "float32"),
  momentum_pw_m = list(c(2, 3), c(0x1.587d2c0000000p-1, -0x1.587d2c0000000p+0, 0x1.587d2c0000000p-2, 0, 0x1.025de00000000p+1, -0x1.587d2c0000000p-1), "float32"),
  momentum_pw_b = list(integer(0), c(0x1.7065720000000p-1), "float32"),
  sched_const = list(integer(0), c(0x1.999999999999ap-4), "float64"),
  sched_exp = list(integer(0), c(0x1.21a1851ff630bp-5), "float64"),
  sched_inv = list(integer(0), c(0x1.d41d41d41d41ep-5), "float64"),
  sched_inv_stair = list(integer(0), c(0x1.1111120000000p-4), "float32"),
  sched_poly = list(integer(0), c(0x1.3d70a40000000p-4), "float32"),
  sched_poly_end = list(integer(0), c(0x1.47ae140000000p-7), "float32"),
  sched_poly_int = list(integer(0), c(0x1.f0a3d80000000p-5), "float32"),
  sched_pw = list(c(4), c(0x1p+0, 0x1p+0, 0x1.0000000000000p-1, 0x1.99999a0000000p-4), "float32"),
  l2_norm = list(integer(0), c(0x1.2268600000000p+2), "float32"),
  clip_w = list(c(3), c(0x1.c3567e0000000p-4, -0x1.c3567e0000000p-3, 0x1.c3567e0000000p-2), "float32"),
  clip_b = list(integer(0), c(0x1.0ecd800000000p-4), "float32"),
  clip2_w = list(c(3), c(0x1.0000000000000p-1, -0x1p+0, 0x1.0000000000000p+1), "float32")
)

opt_t_params <- function() {
  list(w = jnp_array(c(0.5, -1, 2), "float32"),
       m = jnp_reshape(jnp_array(c(1, -2, 0.5, 0, 3, -1), "float32"), c(2, 3)),
       b = as_jax(0.3, "float32"))
}

opt_t_loss <- function(p) {
  jnp_sum((p$w - jnp_array(c(1, 1, -1), "float32"))^2L) + jnp_sum(p$m^2L * 0.5) + (p$b - 1)^2L
}

opt_t_run <- function(opt, steps = 4L) {
  st <- opt$init_fn(opt_t_params())
  for (i in seq_len(steps) - 1L) st <- opt$update_fn(i, grad(opt_t_loss)(opt$params_fn(st)), st)
  opt$params_fn(st)
}

test_that("optimizers match jax.example_libraries.optimizers bit for bit", {
  skip_if_no_backend()
  opts <- list(
    sgd = optimizers_sgd(0.1), momentum = optimizers_momentum(0.1, 0.9),
    nesterov = optimizers_nesterov(0.1, 0.9), adagrad = optimizers_adagrad(0.1),
    rmsprop = optimizers_rmsprop(0.1), rmsprop_momentum = optimizers_rmsprop_momentum(0.1),
    adam = optimizers_adam(0.1), adamax = optimizers_adamax(0.1), sm3 = optimizers_sm3(0.1),
    sgd_expdecay = optimizers_sgd(optimizers_exponential_decay(0.1, 2, 0.5)),
    adam_poly = optimizers_adam(optimizers_polynomial_decay(0.1, 3, 0.01, 2L)),
    momentum_pw = optimizers_momentum(optimizers_piecewise_constant(c(1L, 2L), c(0.1, 0.05, 0.01)),
                                      0.5))
  for (nm in names(opts)) {
    p <- opt_t_run(opts[[nm]])
    expect_identical(names(p), c("w", "m", "b"))
    opt_t_expect(p$w, opt_ref[[paste0(nm, "_w")]])
    opt_t_expect(p$m, opt_ref[[paste0(nm, "_m")]])
    opt_t_expect(p$b, opt_ref[[paste0(nm, "_b")]])
  }
})

test_that("optimizer states are pytrees usable under jit", {
  skip_if_no_backend()
  opt <- optimizers_adam(0.1)
  st <- opt$init_fn(opt_t_params())
  expect_s3_class(st, "OptimizerState")
  expect_identical(length(tree_leaves(st)), 9L)
  step <- jit(function(i, s) opt$update_fn(i, grad(opt_t_loss)(opt$params_fn(s)), s))
  for (i in 0:3) st <- step(i, st)
  p <- opt$params_fn(st)
  # a traced step uses lax.pow for the bias correction, as JAX does under jit
  expect_equal(as.vector(as.array(p$w)), opt_ref$adam_w[[2L]], tolerance = 1e-6)
  expect_error(opt$update_fn(0L, list(w = 1, b = 2), st), "did not match the parameter tree")
  bad <- optimizers_optimizer(function() list(function(x) list(x, x), function(i, g, s) list(s[[1L]]),
                                              function(s) s[[1L]]))()
  st2 <- bad$init_fn(list(a = 1))
  expect_error(bad$update_fn(0L, list(a = 1), st2), "did not match its input structure")
})

test_that("pack and unpack optimizer states", {
  skip_if_no_backend()
  opt <- optimizers_momentum(0.1, 0.9)
  st <- opt$init_fn(opt_t_params())
  marked <- optimizers_unpack_optimizer_state(st)
  expect_identical(names(marked), c("w", "m", "b"))
  expect_s3_class(marked$w, "JoinPoint")
  expect_length(marked$w$subtree, 2L)
  st2 <- optimizers_pack_optimizer_state(marked)
  expect_true(jaxR:::treedef_equal(tree_structure(st2), tree_structure(st)))
  expect_identical(st2$tree_def, st$tree_def)
  expect_identical(lapply(tree_leaves(st2), as.array), lapply(tree_leaves(st), as.array))
  expect_error(optimizers_pack_optimizer_state(list(1, 2)), "JoinPoint")
})

test_that("schedules and gradient utilities match JAX", {
  skip_if_no_backend()
  opt_t_expect(optimizers_constant(0.1)(3L), opt_ref$sched_const)
  opt_t_expect(optimizers_exponential_decay(0.1, 2, 0.5)(3L), opt_ref$sched_exp)
  opt_t_expect(optimizers_inverse_time_decay(0.1, 2, 0.5)(3L), opt_ref$sched_inv)
  opt_t_expect(optimizers_inverse_time_decay(0.1, 2, 0.5, staircase = TRUE)(3L), opt_ref$sched_inv_stair)
  opt_t_expect(optimizers_polynomial_decay(0.1, 4L, 0.01)(1L), opt_ref$sched_poly)
  opt_t_expect(optimizers_polynomial_decay(0.1, 4L, 0.01)(9L), opt_ref$sched_poly_end)
  opt_t_expect(optimizers_polynomial_decay(0.1, 4L, 0.01, 2L)(1L), opt_ref$sched_poly_int)
  pw <- optimizers_piecewise_constant(c(2L, 5L), c(1, 0.5, 0.1))
  opt_t_expect(jnp_stack(list(pw(0L), pw(2L), pw(3L), pw(7L))), opt_ref$sched_pw)
  opt_t_expect(jit(pw)(3L), list(integer(0), opt_ref$sched_pw[[2L]][3L], "float32"))
  expect_error(optimizers_piecewise_constant(c(1, 2), c(1, 2)), "one shorter than values")
  f <- function(i) 0.5
  expect_identical(optimizers_make_schedule(f), f)
  expect_identical(optimizers_make_schedule(0.3)(10L), 0.3)
  expect_error(optimizers_make_schedule(c(1, 2)), class = "jax_type_error")
  p <- opt_t_params()
  opt_t_expect(optimizers_l2_norm(p), opt_ref$l2_norm)
  cg <- optimizers_clip_grads(p, 1)
  opt_t_expect(cg$w, opt_ref$clip_w)
  opt_t_expect(cg$b, opt_ref$clip_b)
  opt_t_expect(optimizers_clip_grads(p, 100)$w, opt_ref$clip2_w)
})
