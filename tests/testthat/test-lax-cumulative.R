# Cumulative reductions and associative_scan. Expected values from jax 0.12.0.1 (x64).

cum_x <- function() {
  matrix(c(0.3, 1.7, -2.1, 4.4, 0.9, 2.2, -0.5, 3.3, 1.1, -1.8, 0.6, 2.9, -0.7, 1.4, 3.8),
         3, byrow = TRUE)
}

cum_expected <- list(
  cumsum = list(
    a1 = c(0.3, 2.0, -0.1, 4.3, 5.2, 2.2, 1.7, 5.0, 6.1, 4.3, 0.6, 3.5, 2.8, 4.2, 8.0),
    a0r = c(3.1, 4.1, 0.5, 6.9, 2.9, 2.8, 2.4, 2.6, 2.5, 2.0, 0.6, 2.9, -0.7, 1.4, 3.8),
    g = c(0.4685166713003779, 0.6550290407229536, -0.34326573507179947, 0.21110860110736207,
          0.8327185693780265, -0.40079917207997545, -0.9056452766798326, -1.7625340300487802,
          -0.9976918427642917, -1.2248939374573788, -0.14550003380861265, 0.2930472937657777,
          0.08225149433499801, 0.5507681656353743, -0.2401995462790425)),
  cumprod = list(
    a1 = c(0.3, 0.51, -1.071, -4.7124, -4.24116, 2.2, -1.1, -3.63, -3.993, 7.1874,
           0.6, 1.74, -1.218, -1.7052, -6.47976),
    a0r = c(0.396, -2.465, 4.851, 6.776, -6.156, 1.32, -1.45, -2.31, 1.54, -6.84,
            0.6, 2.9, -0.7, 1.4, 3.8),
    g = c(6.418029566488728, 1.1328683702724829, -2.6822767417678857, 0.6651312126187612,
          3.873362563295719, 2.0199958184255964, -2.405294415704725, -1.553608325243204,
          -3.944643002907762, 2.1834130737505455, -10.5916140736604, -1.4662227127189333,
          1.6314014165893207, 1.3537485676062606, -0.29221823963842614)),
  cummax = list(
    a1 = c(0.3, 1.7, 1.7, 4.4, 4.4, 2.2, 2.2, 3.3, 3.3, 3.3, 0.6, 2.9, 2.9, 2.9, 3.8),
    a0r = c(2.2, 2.9, 3.3, 4.4, 3.8, 2.2, 2.9, 3.3, 1.4, 3.8, 0.6, 2.9, -0.7, 1.4, 3.8),
    g = c(0, 0, 0, -1.2293314799136774, 0.6216099682706644, 0, 0, -2.9624393097265944,
          0.4535961214255773, -0.2272020946930871, 0, 0, 0, 0, -3.9548385595720843)),
  cummin = list(
    a1 = c(0.3, 0.3, -2.1, -2.1, -2.1, 2.2, -0.5, -0.5, -0.5, -1.8, 0.6, 0.6, -0.7, -0.7, -0.7),
    a0r = c(0.3, -0.5, -2.1, 1.1, -1.8, 0.6, -0.5, -0.7, 1.1, -1.8, 0.6, 2.9, -0.7, 1.4, 3.8),
    g = c(0, 0, -1.5145383137995727, 0, 1.2432199365413288, 0, 0, 0, 0, -1.1360104734654355,
          0, 0, 2.2945265618534654, 0.16996714290024104, -0.7909677119144168)),
  cumlogsumexp = list(
    a1 = c(0.3, 1.9204174099184508, 1.9382037635057792, 4.481839517675006, 4.5092839187177365,
           2.2, 2.265043561776591, 3.603979929587953, 3.6825683082567595, 3.6867183203450233,
           0.6, 2.9955454645979627, 3.0200760191327074, 3.2006323985573157, 4.237712082678035),
    a0r = c(2.501189290779966, 3.188607255742099, 3.3225754655337107, 4.483118191155208,
            3.8570616549444647, 2.383900740888339, 2.932828470424865, 3.3181499279178097,
            1.9543552444685273, 3.803691043426946, 0.6, 2.9, -0.7, 1.4, 3.8),
    g = c(-0.002996828956682902, -0.025383948501552147, -0.0009722286176004262,
          -0.9173853161558573, 0.5939073321327011, -0.193341067766925, -0.031816455542455345,
          -2.2853783185276715, 0.13087140716981577, -0.22000112845729067,
          -0.012026145820928289, -0.2494914612743352, -0.01415711305959519,
          -0.1767349013267109, -2.7391477018540407)))

cum_fun <- function(name) switch(name, cumsum = lax_cumsum, cumprod = lax_cumprod,
                                 cummax = lax_cummax, cummin = lax_cummin,
                                 cumlogsumexp = lax_cumlogsumexp)

cum_stack_loop <- function(f, xs, axis) {
  n <- dim(xs)[axis + 1L]
  jnp_stack(lapply(seq_len(n) - 1L, function(i) f(lax_index_in_dim(xs, i, axis, keepdims = FALSE))))
}

test_that("cumulative reductions match JAX values, gradients and finite differences", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  x <- as_jax(cum_x(), "float64")
  for (name in names(cum_expected)) {
    f <- cum_fun(name)
    e <- cum_expected[[name]]
    expect_jax_equal(f(x, axis = 1L), matrix(e$a1, 3, byrow = TRUE), dtype = "float64")
    expect_jax_equal(f(x, axis = 0L, reverse = TRUE), matrix(e$a0r, 3, byrow = TRUE))
    expect_jax_equal(jit(function(x) f(x, axis = 1L))(x), matrix(e$a1, 3, byrow = TRUE))
    loss <- function(x) jnp_sum(jnp_sin(f(x, axis = 1L, reverse = TRUE)))
    expect_jax_equal(grad(loss)(x), matrix(e$g, 3, byrow = TRUE), tolerance = 1e-10)
    check_grad_fd(loss, x)
    expect_jax_equal(jit(grad(loss))(x), matrix(e$g, 3, byrow = TRUE), tolerance = 1e-10)
  }
})

test_that("cumulative reductions on integer, complex and float32 inputs", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  xi <- as_jax(c(3, -5, 7, 1, -9), "int64")
  expect_jax_equal(lax_cummax(xi), c(3, 3, 7, 7, 7), dtype = "int64")
  expect_jax_equal(lax_cummin(xi, reverse = TRUE), c(-9, -9, -9, -9, -9), dtype = "int64")
  xu <- as_jax(c(3, 5, 1, 7), "uint64")
  expect_jax_equal(lax_cummin(xu), c(3, 3, 1, 1), dtype = "uint64")
  expect_jax_equal(lax_cummax(xu, reverse = TRUE), c(7, 7, 7, 7), dtype = "uint64")
  expect_jax_equal(lax_cumprod(as_jax(c(1L, 2L, 3L, -4L))), c(1, 2, 6, -24), dtype = "int32")
  expect_jax_equal(lax_cumsum(as_jax(c(1 + 2i, 3 - 1i, -2 + 0.5i))), c(1 + 2i, 4 + 1i, 2 + 1.5i),
                   dtype = "complex128")
  expect_jax_equal(lax_cummax(as_jax(c(1 + 2i, 1 + 1i, 0 + 5i, 2 - 1i))),
                   c(1 + 2i, 1 + 2i, 1 + 2i, 2 - 1i))
  xinf <- as_jax(c(-Inf, 0, -Inf, 1), "float64")
  expect_jax_equal(lax_cumlogsumexp(xinf), c(-Inf, 0, 0, 1.3132616875182228))
  expect_jax_equal(grad(function(x) jnp_sum(lax_cumlogsumexp(x)))(xinf),
                   c(1, 2.268941421369995, 0, 0.731058578630005))
  enable_x64(FALSE)
  xf <- as_jax(cum_x(), "float32")
  expect_jax_equal(lax_cumsum(xf, 1L), matrix(cum_expected$cumsum$a1, 3, byrow = TRUE),
                   dtype = "float32")
  expect_jax_equal(lax_cumsum(as_jax(c(100L, 100L), "int8")), c(100, -56), dtype = "int8")
})

test_that("cumulative reductions: errors and dtype rules match JAX", {
  skip_if_no_backend()
  x <- as_jax(c(1, 2, 3))
  for (f in list(lax_cumsum, lax_cummax)) {
    expect_error(f(x, axis = -1L), "XLA operations do not allow negative axes")
    expect_error(f(x, axis = 1L), "axis 1 is out of bounds for array of shape \\(3,\\)")
  }
  expect_error(lax_cumsum(as_jax(c(TRUE, FALSE))),
               "cumsum does not accept dtype bool. Accepted dtypes are subtypes of number.")
  expect_error(lax_cummax(as_jax(c(TRUE, FALSE))), "cummax does not accept dtype bool")
  cj <- make_jaxpr(function(x) lax_cumsum(x))(x)
  expect_identical(cj$jaxpr$eqns[[1L]]$prim$name, "cumsum")
  expect_identical(cj$jaxpr$eqns[[1L]]$params$reverse, FALSE)
})

test_that("cumulative reductions: second order derivatives", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  x <- as_jax(cum_x(), "float64")
  w <- as_jax(matrix(0:14, 3, byrow = TRUE) / 10, "float64")
  f2 <- function(x) jnp_sum(lax_cumprod(x, axis = 1L)^2)
  expect_jax_equal(jvp(grad(f2), list(x), list(w))[[2L]], matrix(c(
    123.26407627200007, 18.87350184000001, -21.329778960000013, 7.182745038000004, 20.304977616000013,
    -87.36222304000002, -79.29268547199997, -56.57303696000003, -191.51130768000007, 32.52585996000003,
    252.4213141119998, 95.67563865599995, -641.6776120319998, 146.29824787199993, 63.84159667199997),
    3, byrow = TRUE), tolerance = 1e-9)
  f3 <- function(x) jnp_sum(lax_cumlogsumexp(x, axis = 1L)^2)
  expect_jax_equal(grad(function(x) jnp_sum(grad(f3)(x) * w))(x), matrix(c(
    -0.1191441179825619, 0.24027189258285178, 0.010423513078013065, 1.2881630459385816, 0.043304652833609354,
    2.0379721660179504, 0.1326263862655145, 3.4183151941657712, 0.3538408873584112, 0.012759524141549492,
    2.4010985074690616, 5.777170098589833, 0.13597952334452634, 0.7057688125559954, 2.2110026373165983),
    3, byrow = TRUE), tolerance = 1e-9)
  check_grad_fd(function(x) jnp_sum(grad(function(y) jnp_sum(lax_cumsum(y, 0L)^2))(x) * w), x)
})

test_that("cumulative reductions batch like a stacked loop", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  xs <- as_jax(array(sin(seq_len(2 * 3 * 4)) * 2, c(2L, 3L, 4L)), "float64")
  for (name in names(cum_expected)) {
    f <- cum_fun(name)
    for (ax in 0:2) {
      xa <- jnp_moveaxis(xs, 0L, ax)
      g1 <- function(x) f(x, axis = 1L, reverse = TRUE)
      g0 <- function(x) f(x, axis = 0L)
      expect_jax_equal(vmap(g1, in_axes = ax)(xa), cum_stack_loop(g1, xa, ax))
      expect_jax_equal(vmap(g0, in_axes = ax)(xa), cum_stack_loop(g0, xa, ax))
    }
    gr <- grad(function(x) jnp_sum(jnp_sin(f(x, axis = 0L))))
    expect_jax_equal(vmap(gr, in_axes = 2L)(xs), cum_stack_loop(gr, xs, 2L))
  }
})

test_that("associative_scan matches JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  expect_jax_equal(lax_associative_scan(jnp_add, jnp_arange(0L, 7L)), c(0, 1, 3, 6, 10, 15, 21))
  expect_jax_equal(lax_associative_scan(jnp_add, jnp_arange(0L, 7L), reverse = TRUE),
                   c(21, 21, 20, 18, 15, 11, 6))
  m <- jnp_reshape(jnp_arange(1, 17) / 10, c(4L, 2L, 2L))
  out <- lax_associative_scan(jnp_matmul, m)
  expect_jax_equal(lax_index_in_dim(out, 3L, 0L, keepdims = FALSE), matrix(c(1.2179, 1.3046, 2.7631, 2.9598), 2, byrow = TRUE))
  expect_jax_equal(lax_index_in_dim(out, 1L, 0L, keepdims = FALSE), matrix(c(0.19, 0.22, 0.43, 0.5), 2, byrow = TRUE))
  tr <- lax_associative_scan(function(a, b) list(a[[1L]] + b[[1L]], a[[2L]] * b[[2L]]),
                             list(jnp_arange(5), jnp_arange(1, 6)), axis = 0L)
  expect_jax_equal(tr[[1L]], c(0, 1, 3, 6, 10))
  expect_jax_equal(tr[[2L]], c(1, 2, 6, 24, 120))
  expect_error(lax_associative_scan(1, jnp_arange(3)), "fn argument should be callable")
  expect_error(lax_associative_scan(jnp_add, list(jnp_arange(3), jnp_arange(4))),
               "must have the same first dimension")
})

test_that("jnp cumulative functions match JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  a <- as_jax(matrix(1:6, 2, byrow = TRUE), "int64")
  expect_jax_equal(jnp_cumsum(a), c(1, 3, 6, 10, 15, 21), dtype = "int64")
  expect_jax_equal(jnp_cumsum(a, axis = 1L), matrix(c(1, 3, 6, 4, 9, 15), 2, byrow = TRUE))
  expect_jax_equal(jnp_cumprod(a), c(1, 2, 6, 24, 120, 720))
  expect_jax_equal(jnp_cumprod(a, axis = 0L), matrix(c(1, 2, 3, 4, 10, 18), 2, byrow = TRUE))
  expect_jax_equal(jnp_cumsum(c(TRUE, TRUE, FALSE)), c(1, 2, 2), dtype = "int64")
  expect_jax_equal(jnp_cumsum(as_jax(c(1, -1, 2), "int64"), dtype = "bool"), c(TRUE, TRUE, TRUE),
                   dtype = "bool")
  expect_jax_equal(jnp_cumprod(as_jax(c(2, 0.5), "float64"), dtype = "int32"), c(2, 0), dtype = "int32")
  xn <- as_jax(matrix(c(1, 2, NaN, 4, NaN, 6), 2, byrow = TRUE), "float64")
  expect_jax_equal(jnp_nancumsum(xn), c(1, 3, 3, 7, 7, 13))
  expect_jax_equal(jnp_nancumprod(xn, axis = 1L), matrix(c(1, 2, 2, 4, 4, 24), 2, byrow = TRUE))
  expect_jax_equal(jnp_cumsum(as_jax(c(100L, 100L), "int8")), c(100, -56), dtype = "int8")
  expect_jax_equal(jnp_cumulative_sum(as_jax(c(100L, 100L), "int8")), c(100, 200), dtype = "int64")
  expect_jax_equal(jnp_cumulative_sum(a, axis = 1L, include_initial = TRUE),
                   matrix(c(0, 1, 3, 6, 0, 4, 9, 15), 2, byrow = TRUE))
  expect_jax_equal(jnp_cumulative_prod(a, axis = 0L, include_initial = TRUE),
                   matrix(c(1, 1, 1, 1, 2, 3, 4, 10, 18), 3, byrow = TRUE))
  expect_jax_equal(jnp_cumulative_prod(as_jax(c(20L, 20L), "uint8")), c(20, 144), dtype = "uint8")
  expect_jax_equal(jnp_cumulative_sum(as_jax(c(1.5, 2.5), "float64"), dtype = "int32"), c(1, 3),
                   dtype = "int32")
  expect_jax_equal(jnp_cumsum(5), 5, shape = 1L)
  expect_error(jnp_cumulative_sum(as_jax(3)), "The input must be non-scalar to take a cumulative sum")
  expect_error(jnp_cumulative_sum(a), "The input array has rank 2, however axis was not set")
  expect_error(jnp_cumsum(a, out = 1), "The 'out' argument to jnp.cumsum is not supported")
  # base generics on jax values
  x <- as_jax(c(3, 1, 4, 1, 5), "float64")
  expect_jax_equal(cumsum(x), c(3, 4, 8, 9, 14))
  expect_jax_equal(cumprod(x), c(3, 3, 12, 12, 60))
  expect_jax_equal(cummax(x), c(3, 3, 4, 4, 5))
  expect_jax_equal(cummin(x), c(3, 1, 1, 1, 1))
  check_grad_fd(function(x) jnp_sum(jnp_sin(jnp_cumsum(x * x))), x)
})
