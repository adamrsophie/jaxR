# Tests for jax.scipy.ndimage (port of jax/_src/scipy/ndimage.py).
# Expected values come from jax 0.12.0.1 (float32, x64 disabled).

# R array from C order values.
carr <- function(v, shape) {
  if (length(shape) <= 1L) return(array(v, shape))
  aperm(array(v, rev(shape)), rev(seq_along(shape)))
}

mc_constant_0 <- c(-1.5, -1.5, 0, 0.69693846, 1.4828727, -1.5, -1.5, -1.5, -1.5)
mc_constant_1 <- c(-1.5, -1.5, 0.2, 2.1764264, 2.1012526, -0.7025087, 0.52641225, -1.5, -1.5)
mc_nearest_0 <- c(0, 5.196152, 0, 0.69693846, 1.4828727, 1.627417, 1.4828727, 1.4828727, 1.4828727)
mc_nearest_1 <- c(0, 5.196152, 0.2, 2.1764264, 3.001566, 1.627417, 2.5528247, 1.4828728, 1.4828727)
mc_wrap_0 <- c(0.69693846, 6, 0, 0.69693846, 1.4828727, 1.4828727, 5.196152, 1, 1.4828727)
mc_wrap_1 <- c(2.7795525, 2.1568542, 0.2, 2.1764264, 2.6639943, 2.0661812, 3.282557, 1.2052472, 2.3493657)
mc_mirror_0 <- c(3.6227765, 4.18034, 0, 0.69693846, 1.4828727, 6, 4.5202594, 2.828427, 4.18034)
mc_mirror_1 <- c(2.5260572, 2.1764264, 0.2, 2.1764264, 2.8332243, 3.2141747, 2.5807118, 2.142005, 4.202393)
mc_reflect_0 <- c(4.18034, 2.828427, 0, 0.69693846, 1.4828727, 1.627417, 1.4828727, 4.5202594, 0)
mc_reflect_1 <- c(2.500603, 4.0122895, 0.2, 2.1764264, 3.001566, 1.627417, 2.5528247, 3.2021732, 0.099999905)
mc_int1 <- c(-7, 2, -5, 4, 20, 17, 25, 26, 26)
mc_int0 <- c(0, 0, -7, 11, 26, 0, 0, 0, 0)
mc1_constant <- c(0.25, 1.9000001, 3, 5, 5.2999997, 2.5, 0)
mc1_nearest <- c(1, 1.9000001, 3, 5, 5.2999997, 5, 5)
mc1_wrap <- c(4, 1.9000001, 3, 5, 5.2999997, 3, 3.8000002)
mc1_mirror <- c(3.25, 1.9000001, 3, 5, 5.2999997, 6.5, 2.1999998)
mc1_reflect <- c(1, 1.9000001, 3, 5, 5.2999997, 5, 7.4000006)
mc_2d <- carr(c(0.709017, 2.7389476, 3.860499, 2.5494373), c(2L, 2L))
mc_arr <- c(2.7795525, 2.1568542, 0.2, 2.1764264, 2.6639943, 2.0661812, 3.282557, 1.2052472, 2.3493657)
mc_ic <- c(5.196152, 6, 1)

inp <- carr(((0:11)^1.5) %% 7, c(3L, 4L))
inpi <- carr((0:11) * 3L - 7L, c(3L, 4L))
c0 <- c(-1.6, -0.5, 0.2, 0.5, 1.5, 2.49, 2.5, 3.7, 5.2)
c1 <- c(-2.3, 4.5, 0.0, 1.5, 3.2, -0.5, 2.5, 4.2, 6.9)
v1 <- c(1, 4, 2, 8, 5)
cc <- c(-0.75, 0.3, 1.5, 2.5, 3.9, 4.5, 6.1)
C2 <- carr(c(0.5, 1.2, 2.0, -0.4, 0.1, 3.3, 1.9, 2.6), c(2L, 2L, 2L))
all_modes <- c("constant", "nearest", "wrap", "mirror", "reflect")
with_x64 <- function(code) with_jax_config("jax_enable_x64", TRUE, code)

test_that("map_coordinates matches jax for every mode and order", {
  skip_if_no_backend()
  for (m in all_modes) {
    for (o in 0:1) {
      expect_jax_equal(scipy_ndimage_map_coordinates(inp, list(c0, c1), o, mode = m, cval = -1.5),
                       get(paste0("mc_", m, "_", o)), dtype = "float32", shape = 9L)
    }
    expect_jax_equal(scipy_ndimage_map_coordinates(v1, list(cc), 1L, mode = m),
                     get(paste0("mc1_", m)))
  }
})

test_that("map_coordinates handles integer inputs and coordinate layouts", {
  skip_if_no_backend()
  storage.mode(inpi) <- "integer"
  expect_jax_equal(scipy_ndimage_map_coordinates(inpi, list(c0, c1), 1L, mode = "nearest"),
                   mc_int1, dtype = "int32")
  expect_jax_equal(scipy_ndimage_map_coordinates(inpi, list(c0, c1), 0L), mc_int0,
                   dtype = "int32")
  expect_jax_equal(scipy_ndimage_map_coordinates(inp, list(C2[1, , ], C2[2, , ]), 1L), mc_2d,
                   shape = c(2L, 2L))
  expect_jax_equal(scipy_ndimage_map_coordinates(inp, rbind(c0, c1), 1L, mode = "wrap"), mc_arr)
  expect_jax_equal(scipy_ndimage_map_coordinates(inp, list(as_jax(c(0L, 2L, 1L)),
                                                           as_jax(c(3L, 1L, 0L))), 1L), mc_ic)
})

test_that("map_coordinates works under jit and vmap", {
  skip_if_no_backend()
  f <- jit(function(x, a, b) scipy_ndimage_map_coordinates(x, list(a, b), 1L, mode = "mirror"))
  expect_jax_equal(f(inp, c0, c1), mc_mirror_1)
  cs <- rbind(c0, c1, c0 + 0.25)
  got <- vmap(function(a) scipy_ndimage_map_coordinates(inp, list(a, c1), 1L, mode = "reflect"))(
    unname(cs))
  want <- t(sapply(1:3, function(i) {
    as.array(scipy_ndimage_map_coordinates(inp, list(cs[i, ], c1), 1L, mode = "reflect"))
  }))
  expect_jax_equal(got, want)
  got <- vmap(function(x) scipy_ndimage_map_coordinates(x, list(c0, c1), 0L, mode = "wrap"))(
    jnp_stack(list(inp, inp * 2)))
  expect_jax_equal(got, rbind(mc_wrap_0, 2 * mc_wrap_0))
})

test_that("map_coordinates gradients match finite differences", {
  skip_if_no_backend()
  with_x64({
    x <- as_jax(inp, "float64")
    a <- as_jax(c(0.3, 1.6, 2.2, -0.4), "float64")
    b <- as_jax(c(0.7, 2.4, 3.6, 1.1), "float64")
    for (m in all_modes) {
      check_grad_fd(function(v) jnp_sum(scipy_ndimage_map_coordinates(v, list(a, b), 1L, mode = m)^2L), x)
      check_grad_fd(function(v) jnp_sum(scipy_ndimage_map_coordinates(x, list(v, b), 1L, mode = m)), a)
    }
    check_grad_fd(function(v) jnp_sum(scipy_ndimage_map_coordinates(v, list(a, b), 0L)^2L), x)
  })
})

test_that("map_coordinates raises jax errors", {
  skip_if_no_backend()
  expect_error(scipy_ndimage_map_coordinates(inp, list(c0), 1L),
               "coordinates must be a sequence of length input.ndim, but 1 != 2")
  expect_error(scipy_ndimage_map_coordinates(inp, list(c0, c1), 2L), "requires order<=1",
               class = "jax_not_implemented_error")
  expect_error(scipy_ndimage_map_coordinates(inp, list(c0, c1), 1L, mode = "grid-wrap"),
               "does not yet support mode grid-wrap", class = "jax_not_implemented_error")
})
