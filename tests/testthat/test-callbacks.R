# Host callbacks: jax.pure_callback and jax.experimental.io_callback, eager and inside
# compiled code (through the jaxr_r_callback FFI handler), under vmap and AD.

f32 <- function(shape) ShapeDtypeStruct(shape, "float32")

sin2 <- function(v) sin(v) * 2

test_that("pure_callback runs eagerly and under jit with R arrays", {
  skip_if_no_backend()
  seen <- list()
  cb <- function(v, w) {
    seen[[length(seen) + 1L]] <<- list(class = class(v), dim = dim(v), w = w)
    sin(v) * w
  }
  f <- function(x, w) pure_callback(cb, f32(dim(x)), x, w)
  x <- as_jax(matrix(c(0.5, 1, 2, 3, 4, 5), 2))
  expected <- sin(c(0.5, 1, 2, 3, 4, 5)) * 2
  expect_jax_equal(f(x, 2), expected, dtype = "float32", shape = c(2L, 3L))
  expect_jax_equal(jit(f)(x, 2), expected, dtype = "float32", shape = c(2L, 3L))
  expect_identical(seen[[1L]]$dim, c(2L, 3L))
  expect_identical(seen[[2L]]$dim, c(2L, 3L))
  expect_true(is.numeric(seen[[2L]]$w) && is.null(dim(seen[[2L]]$w)))
  expect_false(inherits(seen[[2L]]$w, "jax_value"))
  # the compiled program calls the R function on every run
  n <- length(seen)
  jf <- jit(f)
  jf(x, 2)
  jf(x, 3)
  expect_length(seen, n + 2L)
})

test_that("pure_callback passes pytrees, named arguments and multiple results", {
  skip_if_no_backend()
  cb <- function(p, scale) list(a = p$u * scale, b = as.integer(round(sum(p$v))))
  res_type <- list(a = f32(2L), b = ShapeDtypeStruct(integer(0), "int32"))
  f <- function(u, v) pure_callback(cb, res_type, list(u = u, v = v), scale = as_jax(10))
  for (g in list(f, jit(f))) {
    r <- g(as_jax(c(1, 2)), as_jax(c(1.4, 2.4)))
    expect_named(r, c("a", "b"))
    expect_jax_equal(r$a, c(10, 20), dtype = "float32")
    expect_jax_equal(r$b, 4, dtype = "int32")
  }
  # bool and complex values cross the boundary
  h <- jit(function(z) pure_callback(function(z) list(Mod(z) > 1, Conj(z)),
                                     list(ShapeDtypeStruct(2L, "bool"), ShapeDtypeStruct(2L, "complex64")), z))
  r <- h(as_jax(c(1 + 1i, 0.5 + 0i)))
  expect_identical(as.vector(as.array(r[[1L]])), c(TRUE, FALSE))
  expect_equal(as.vector(as.array(r[[2L]])), c(1 - 1i, 0.5 + 0i))
  # jax array results must have the declared dtype
  expect_jax_equal(jit(function(x) pure_callback(function(v) as_jax(v * 2, "float32"), f32(3L), x))(
    as_jax(c(1, 2, 3))), c(2, 4, 6))
})

test_that("compiled callbacks reuse the registry and the compile cache", {
  skip_if_no_backend()
  f <- jit(function(x) pure_callback(sin2, f32(3L), x))
  f(as_jax(c(1, 2, 3)))
  n <- length(jaxR:::.jx$cb_registry)
  for (i in 1:3) f(as_jax(c(1, 2, 3) + i))
  g <- jit(function(x) pure_callback(sin2, f32(3L), x) + 1)
  g(as_jax(c(1, 2, 3)))
  expect_identical(length(jaxR:::.jx$cb_registry), n)
})

test_that("pure_callback result checks and errors", {
  skip_if_no_backend()
  x <- as_jax(c(1, 2, 3))
  for (wrap in list(identity, jit)) {
    expect_error(wrap(function(x) pure_callback(function(v) list(v, v), f32(3L), x))(x),
                 "Mismatched number of outputs from callback. Expected: 1, Actual: 2", fixed = TRUE)
    expect_error(wrap(function(x) pure_callback(function(v) v[1:2], f32(3L), x))(x),
                 "Incorrect output shape for return value #0: Expected: (3,), Actual: (2,)", fixed = TRUE)
    expect_error(wrap(function(x) pure_callback(function(v) v > 1, f32(3L), x))(x),
                 "Incorrect output dtype for return value #0: Expected: float32, Actual: bool", fixed = TRUE)
    expect_error(wrap(function(x) pure_callback(function(v) as_jax(v, "int32"), f32(3L), x))(x),
                 "Incorrect output dtype for return value #0: Expected: float32, Actual: int32", fixed = TRUE)
    expect_error(wrap(function(x) pure_callback(function(v) stop("boom in R"), f32(3L), x))(x),
                 "boom in R")
  }
  # the session keeps working after a failing compiled callback
  expect_jax_equal(jit(function(x) pure_callback(sin2, f32(3L), x))(x), 2 * sin(c(1, 2, 3)))
  expect_error(pure_callback(sin2, f32(3L), x, vmap_method = "bad"),
               "vmap_method must be on of the allowed methods")
  expect_error(pure_callback(1, f32(3L), x), "callback must be a function")
  expect_error(pure_callback(sin2, jaxR:::new_aval(3L, "float64"), x),
               "cannot specify 64-bit types when `jax_enable_x64` is disabled", fixed = TRUE)
  with_jax_config("jax_enable_x64", TRUE, {
    expect_jax_equal(jit(function(x) pure_callback(sin2, ShapeDtypeStruct(3L, "float64"), x))(
      as_jax(c(1, 2, 3), "float64")), 2 * sin(c(1, 2, 3)), dtype = "float64")
  })
})

test_that("pure_callback batches according to vmap_method", {
  skip_if_no_backend()
  shapes <- list()
  cb <- function(x, y) {
    shapes[[length(shapes) + 1L]] <<- list(r_shape(x), r_shape(y))
    out <- as.vector(x) + as.vector(y)
    dim(out) <- dim(x)
    out
  }
  r_shape <- function(v) if (is.null(dim(v))) (if (length(v) == 1L) integer(0) else length(v)) else dim(v)
  fun <- function(x, y, vmap_method) {
    shape <- broadcast_shapes(dim(x), dim(y))
    pure_callback(cb, f32(shape), x, y, vmap_method = vmap_method)
  }
  x <- jnp_arange(4, dtype = "float32")
  y <- as_jax(1)
  shapes <- list()
  expect_jax_equal(vmap(function(x) fun(x, y, "expand_dims"))(x), c(1, 2, 3, 4))
  expect_identical(shapes, list(list(4L, 1L)))
  shapes <- list()
  expect_jax_equal(vmap(function(x) fun(x, y, "broadcast_all"))(x), c(1, 2, 3, 4))
  expect_identical(shapes, list(list(4L, 4L)))
  shapes <- list()
  expect_jax_equal(vmap(function(x) fun(x, y, "sequential"))(x), c(1, 2, 3, 4))
  expect_identical(shapes, rep(list(list(integer(0), integer(0))), 4L))
  shapes <- list()
  expect_jax_equal(vmap(function(x) fun(x, y, "sequential_unrolled"))(x), c(1, 2, 3, 4))
  expect_length(shapes, 4L)
  shapes <- list()
  expect_jax_equal(vmap(function(x) fun(x, y, "legacy_vectorized"))(x), c(1, 2, 3, 4))
  expect_identical(shapes, list(list(4L, integer(0))))
  expect_error(vmap(function(x) fun(x, y, NULL))(x),
               "vmap is only supported for the pure_callback primitive when vmap_method is one of",
               class = "jax_not_implemented_error")
  # batch dims in other positions, under jit, and mixed mapped arguments
  m <- as_jax(matrix(1:6, 2))
  g <- function(a, b) pure_callback(function(a, b) a * b, f32(integer(0)), a, b, vmap_method = "sequential")
  expect_jax_equal(jit(vmap(vmap(g, in_axes = list(0L, NULL)), in_axes = list(1L, NULL)))(m, as_jax(10)),
                   t(matrix(1:6, 2)) * 10)
  g2 <- function(a, b) pure_callback(function(a, b) sum(a) * b, f32(integer(0)), a, b,
                                     vmap_method = "sequential")
  expect_jax_equal(vmap(g2, in_axes = list(1L, 0L))(as_jax(matrix(1:4, 2)), as_jax(c(10, 20))), c(30, 140))
  expect_jax_equal(vmap(g2, in_axes = list(1L, NULL))(as_jax(matrix(1:4, 2)), as_jax(2)), c(6, 14))
})

test_that("pure_callback has no derivative unless wrapped in custom_jvp", {
  skip_if_no_backend()
  f <- function(x) pure_callback(function(v) sin(v), f32(dim(x)), x)
  expect_error(grad(function(x) jnp_sum(f(x)))(as_jax(c(1, 2))),
               "Pure callbacks do not support JVP", class = "jax_value_error")
  # a callback on a constant does not need a derivative
  expect_jax_equal(grad(function(x) jnp_sum(x * f(as_jax(c(1, 2)))))(as_jax(c(1, 2))), sin(c(1, 2)))
  fs <- custom_jvp(f, jvp = function(primals, tangents) {
    x <- primals[[1L]]
    list(f(x), jnp_cos(x) * tangents[[1L]])
  })
  expect_jax_equal(grad(function(x) jnp_sum(fs(x)))(as_jax(c(1, 2))), cos(c(1, 2)))
  expect_jax_equal(jit(grad(function(x) jnp_sum(fs(x))))(as_jax(c(1, 2))), cos(c(1, 2)))
})

test_that("callbacks run on XLA worker threads are serviced by the main thread", {
  skip_if_no_backend()
  big <- as_jax(matrix(seq(0, 1, length.out = 640000), 800))
  g <- jit(function(a) {
    s <- jnp_sum(jnp_matmul(a, a))
    pure_callback(function(v) v / 1e6, f32(integer(0)), s)
  })
  ref <- sum(as.array(big) %*% as.array(big)) / 1e6
  before <- .Call("jaxr_callbacks_stats", PACKAGE = "jaxR")
  expect_equal(as.numeric(as.array(g(big))), ref, tolerance = 1e-4)
  after <- .Call("jaxr_callbacks_stats", PACKAGE = "jaxR")
  # this program runs on XLA's thread pool: the handler queued the call for the main thread
  expect_identical(after[2L] - before[2L], 1)
})

test_that("io_callback runs eagerly, under jit and once per element under vmap", {
  skip_if_no_backend()
  log <- c()
  cb <- function(v) {
    log <<- c(log, sum(v))
    v * 2
  }
  f <- function(x) experimental_io_callback(cb, f32(dim(x)), x)
  expect_jax_equal(f(as_jax(c(1, 2))), c(2, 4))
  expect_jax_equal(jit(f)(as_jax(c(1, 2))), c(2, 4))
  expect_identical(log, c(3, 3))
  log <- c()
  expect_jax_equal(vmap(f)(as_jax(matrix(1:6, 3))), matrix(1:6, 3) * 2)
  expect_identical(log, c(5, 7, 9))
  expect_identical(io_callback, experimental_io_callback)
  expect_error(vmap(function(x) experimental_io_callback(cb, f32(integer(0)), x, ordered = TRUE))(
    as_jax(c(1, 2))), "Cannot `vmap` ordered IO callback.", fixed = TRUE)
  expect_error(grad(function(x) jnp_sum(f(x)))(as_jax(c(1, 2))), "IO callbacks do not support JVP.",
               fixed = TRUE)
})

test_that("io_callback with unused results is kept, ordered callbacks keep their order", {
  skip_if_no_backend()
  log <- c()
  g <- jit(function(x) {
    experimental_io_callback(function(v) { log <<- c(log, "a"); v }, f32(integer(0)), x, ordered = TRUE)
    experimental_io_callback(function(v) { log <<- c(log, "b"); v }, f32(integer(0)), x + 1, ordered = TRUE)
    x * 3
  })
  expect_jax_equal(g(as_jax(2)), 6)
  expect_identical(log, c("a", "b"))
  txt <- lower_text(g, as_jax(2))
  expect_match(txt, "stablehlo.create_token")
  expect_match(txt, "has_side_effect = true")
})

test_that("compiled side effects happen before execution returns", {
  skip_if_no_backend()
  called <- FALSE
  big <- as_jax(matrix(seq(0, 1, length.out = 640000), 800))
  h <- jit(function(a) {
    experimental_io_callback(function(v) { called <<- TRUE; v }, f32(integer(0)), jnp_sum(jnp_matmul(a, a)))
    jnp_sum(a)
  })
  out <- h(big)
  expect_true(called)
})

test_that("callbacks may run jax operations themselves", {
  skip_if_no_backend()
  f <- jit(function(x) pure_callback(function(v) as.array(jnp_sin(as_jax(v)) * 2), f32(3L), x))
  expect_jax_equal(f(as_jax(c(1, 2, 3))), 2 * sin(c(1, 2, 3)))
  g <- jit(function(x) pure_callback(function(v) as.array(f(as_jax(v))) + 1, f32(3L), x))
  expect_jax_equal(g(as_jax(c(1, 2, 3))), 2 * sin(c(1, 2, 3)) + 1)
})
