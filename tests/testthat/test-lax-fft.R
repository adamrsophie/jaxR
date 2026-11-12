# Tests for lax_fft (port of jax/_src/lax/fft.py). Expected values come from jax
# 0.12.0.1 (jax.lax.fft and jax.vjp).

# R array from C order values.
carr <- function(v, shape) {
  if (length(shape) <= 1L) return(array(v, shape))
  aperm(array(v, rev(shape)), rev(seq_along(shape)))
}
with_x64 <- function(code) with_jax_config("jax_enable_x64", TRUE, code)
# |z|^2 through real, mul and conj only.
abs2 <- function(z) lax_real(z * lax_conj(z))

xr <- carr(c(3, 1, 4, 1, 5, 9, 2, 6), c(2L, 4L))
xi <- carr(c(2, 7, 1, 8, 2, 8, 1, 8), c(2L, 4L))
xc <- xr + 1i * xi
x3 <- carr((0:23 %% 7) - 3, c(2L, 3L, 4L))
zc <- carr(c(1, -2, 3, 0, 5, -1), c(2L, 3L)) + 1i * carr(c(2, 1, -1, 3, 0, 4), c(2L, 3L))
z3 <- x3[, , 1:3] + 1i * x3[, , 2:4]
ct5 <- carr(c(1, 2, 3, 4, 5, -1, 0, 2, -3, 1), c(2L, 5L))

test_that("lax_fft matches jax for every fft type", {
  skip_if_no_backend()
  expect_jax_equal(lax_fft(xc, "FFT", 4L), carr(complex(real = c(9, -2, 5, 0, 22, 3, -8, 3), imaginary = c(18, 1, -12, 1, 19, -2, -13, 4)), c(2L, 4L)), dtype = "complex64")
  expect_jax_equal(lax_fft(xc, "ifft", 4L), carr(complex(real = c(2.25, 0, 1.25, -0.5, 5.5, 0.75, -2, 0.75), imaginary = c(4.5, 0.25, -3, 0.25, 4.75, 1, -3.25, -0.5)), c(2L, 4L)), dtype = "complex64")
  expect_jax_equal(lax_fft(xr, "RFFT", 4L), carr(complex(real = c(9, -1, 5, 22, 3, -8), imaginary = c(0, 0, 0, 0, -3, 0)), c(2L, 3L)), dtype = "complex64", shape = c(2L, 3L))
  expect_jax_equal(lax_fft(zc, "irfft", 4L), carr(c(0, -1, 2, 0, 2.25, 0.25, -2.75, 0.25), c(2L, 4L)), dtype = "float32", shape = c(2L, 4L))
  expect_jax_equal(lax_fft(zc, "IRFFT", 5L), carr(c(0.600000024, -1.16334248, 0.60249728, 1.83357072, -0.872725487, 1.60000002, 0.00118436816, -0.219950393, -3.26333117, 1.88209724), c(2L, 5L)), dtype = "float32", shape = c(2L, 5L))
  expect_jax_equal(lax_fft(xc, lax_FftType[["FFT"]], 4L), carr(complex(real = c(9, -2, 5, 0, 22, 3, -8, 3), imaginary = c(18, 1, -12, 1, 19, -2, -13, 4)), c(2L, 4L)))
  expect_jax_equal(lax_fft(xc, lax_FftType["IFFT"], 4L), carr(complex(real = c(2.25, 0, 1.25, -0.5, 5.5, 0.75, -2, 0.75), imaginary = c(4.5, 0.25, -3, 0.25, 4.75, 1, -3.25, -0.5)), c(2L, 4L)))
})

test_that("multi dimensional lax_fft matches jax", {
  skip_if_no_backend()
  expect_jax_equal(lax_fft(xc, "FFT", c(2L, 4L)), carr(complex(real = c(31, 1, -3, 3, -13, -5, 13, -3), imaginary = c(37, -1, -25, 5, -1, 3, 1, -3)), c(2L, 4L)))
  expect_jax_equal(lax_fft(x3, "IFFT", c(3L, 4L)), carr(complex(real = c(-0.416666687, -0.5, 0.0833333284, -0.5, -0.541666687, -0.505181491, -0.291666687, 0.505181491, -0.541666687, 0.505181491, -0.291666687, -0.505181491, -0.0833333284, 0.666666687, 0.0833333284, 0.666666687, 0.0416666642, 0.291666687, -0.291666687, 0.291666687, 0.0416666642, 0.291666687, -0.291666687, 0.291666687), imaginary = c(0, 0.0833333284, 0, -0.0833333284, 0.360843927, -0.291666687, 0.505181491, 0.291666687, -0.360843927, -0.291666687, -0.505181491, 0.291666687, 0, 0.0833333433, 0, -0.0833333433, 0.360843927, 0.0781518817, -0.505181491, -1.0885148, -0.360843927, 1.0885148, 0.505181491, -0.0781518817)), c(2L, 3L, 4L)), dtype = "complex64")
  expect_jax_equal(lax_fft(x3, "RFFT", c(3L, 4L)), carr(complex(real = c(-5, -6, 1, -6.5, -6.06217766, -3.5, -6.5, 6.06217766, -3.5, -1, 8, 1, 0.5, 3.5, -3.5, 0.5, 3.5, -3.5), imaginary = c(0, -1, 0, -4.33012676, 3.5, -6.06217766, 4.33012676, 3.5, 6.06217766, 0, -1, 0, -4.33012676, -0.937822342, 6.06217766, 4.33012676, -13.0621777, -6.06217766)), c(2L, 3L, 3L)), shape = c(2L, 3L, 3L))
  expect_jax_equal(lax_fft(z3, "IRFFT", c(3L, 4L)), carr(c(-0.333333343, -0.833333373, 0, -0.166666672, -1.19417715, -0.521527529, 0.505181491, -0.488835454, -0.472489476, 1.35486078, -0.505181551, -0.344497859, 0, 1.33333337, 0, 0, -0.241025329, 0.275320649, -0.875, 0.308012664, 1.49102533, 1.14134598, -0.875, -0.558012664), c(2L, 3L, 4L)), shape = c(2L, 3L, 4L))
  expect_jax_equal(lax_fft(x3, "FFT", c(2L, 3L, 4L)), carr(complex(real = c(-6, 2, 2, 2, -6, -2.56217766, -7, 9.56217766, -6, 9.56217766, -7, -2.56217766, -4, -14, 0, -14, -7, -9.56217766, 0, 2.56217766, -7, 2.56217766, 0, -9.56217766), imaginary = c(0, -2, 0, 2, -8.66025352, 2.56217766, 0, 9.56217766, 8.66025352, -9.56217766, 0, -2.56217766, 0, 0, 0, 0, 0, 4.43782234, -12.1243553, -16.5621777, 0, 16.5621777, 12.1243553, -4.43782234)), c(2L, 3L, 4L)))
})

test_that("lax_fft converts dtypes like jax", {
  skip_if_no_backend()
  xint <- as_jax(xr, "int32")
  expect_jax_equal(lax_fft(xint, "RFFT", 4L), carr(complex(real = c(9, -1, 5, 22, 3, -8), imaginary = c(0, 0, 0, 0, -3, 0)), c(2L, 3L)), dtype = "complex64")
  expect_identical(dtype(lax_fft(xint, "FFT", 4L)), "complex64")
  expect_identical(dtype(lax_fft(as_jax(xr, "bool"), "IFFT", 4L)), "complex64")
  # IRFFT of a real input converts it to complex first.
  expect_identical(dtype(lax_fft(xr, "IRFFT", 6L)), "float32")
  # Empty fft_lengths only converts.
  e <- lax_fft(xr, "FFT", integer(0))
  expect_jax_equal(e, xr + 0i, dtype = "complex64", shape = c(2L, 4L))
  expect_identical(dtype(lax_fft(xr, "RFFT", integer(0))), "float32")
  with_x64({
    # dtypes.to_complex_dtype: only 64 bit inputs become complex128.
    expect_identical(dtype(lax_fft(xint, "FFT", 4L)), "complex64")
    expect_identical(dtype(lax_fft(as_jax(xr, "int64"), "FFT", 4L)), "complex128")
    expect_identical(dtype(lax_fft(as_jax(xr, "float32"), "RFFT", 4L)), "complex64")
    expect_identical(dtype(lax_fft(as_jax(xr, "float32"), "FFT", 4L)), "complex64")
  })
})

test_that("lax_fft in 64 bit matches jax", {
  skip_if_no_backend()
  with_x64({
    expect_jax_equal(lax_fft(xc, "FFT", 4L), carr(complex(real = c(9, -2, 5, 0, 22, 3, -8, 3), imaginary = c(18, 1, -12, 1, 19, -2, -13, 4)), c(2L, 4L)), dtype = "complex128")
    expect_jax_equal(lax_fft(xr, "RFFT", 4L), carr(complex(real = c(9, -1, 5, 22, 3, -8), imaginary = c(0, 0, 0, 0, -3, 0)), c(2L, 3L)), dtype = "complex128")
    expect_jax_equal(lax_fft(zc, "IRFFT", 4L), carr(c(0, -1, 2, 0, 2.25, 0.25, -2.75, 0.25), c(2L, 4L)), dtype = "float64")
    expect_jax_equal(lax_fft(z3, "IRFFT", c(3L, 4L)), carr(c(-0.33333333333333331, -0.83333333333333326, 0, -0.16666666666666666, -1.1941772515768494, -0.52152752076647513, 0.50518148554092246, -0.48883545031536985, -0.47248941508981718, 1.3548608540998084, -0.50518148554092246, -0.34449788301796347, 0, 1.3333333333333333, 0, 0, -0.24102540378443865, 0.27532063144111396, -0.875, 0.3080127018922193, 1.4910254037844386, 1.1413460352255527, -0.875, -0.5580127018922193), c(2L, 3L, 4L)), dtype = "float64")
  })
})

test_that("lax_fft under jit matches eager", {
  skip_if_no_backend()
  f <- jit(function(x, y) list(lax_fft(x, "RFFT", 4L), lax_fft(y, "IRFFT", c(3L, 4L))))
  r <- f(xr, z3)
  expect_jax_equal(r[[1L]], carr(complex(real = c(9, -1, 5, 22, 3, -8), imaginary = c(0, 0, 0, 0, -3, 0)), c(2L, 3L)), dtype = "complex64")
  expect_jax_equal(r[[2L]], carr(c(-0.333333343, -0.833333373, 0, -0.166666672, -1.19417715, -0.521527529, 0.505181491, -0.488835454, -0.472489476, 1.35486078, -0.505181551, -0.344497859, 0, 1.33333337, 0, 0, -0.241025329, 0.275320649, -0.875, 0.308012664, 1.49102533, 1.14134598, -0.875, -0.558012664), c(2L, 3L, 4L)), dtype = "float32")
  g <- jit(function(x) lax_fft(x, "FFT", c(2L, 4L)))
  expect_jax_equal(g(xc), carr(complex(real = c(31, 1, -3, 3, -13, -5, 13, -3), imaginary = c(37, -1, -25, 5, -1, 3, 1, -3)), c(2L, 4L)))
  with_x64(expect_jax_equal(jit(function(x) lax_fft(x, "FFT", 4L))(xc), carr(complex(real = c(9, -2, 5, 0, 22, 3, -8, 3), imaginary = c(18, 1, -12, 1, 19, -2, -13, 4)), c(2L, 4L)),
                            dtype = "complex128"))
})

test_that("lax_fft lowers to stablehlo.fft", {
  skip_if_no_backend()
  txt <- lower_text(function(x) lax_fft(x, "RFFT", 4L), as_jax(xr))
  expect_match(txt, "\"stablehlo.fft\"(%arg0) <{fft_length = array<i64: 4>, fft_type = #stablehlo<fft_type RFFT>}> : (tensor<2x4xf32>) -> tensor<2x3xcomplex<f32>>",
               fixed = TRUE)
  txt <- lower_text(function(x) lax_fft(x, "IRFFT", c(3L, 4L)), as_jax(z3))
  expect_match(txt, "fft_length = array<i64: 3, 4>, fft_type = #stablehlo<fft_type IRFFT>",
               fixed = TRUE)
})

test_that("vjp of lax_fft matches jax", {
  skip_if_no_backend()
  with_x64({
    vj <- function(f, x, ct) vjp(f, as_jax(x))[[2L]](as_jax(ct))[[1L]]
    expect_jax_equal(vj(function(v) lax_fft(v, "RFFT", 4L), xr, zc), carr(c(2, -1, 6, -3, 4, 1, -6, 1), c(2L, 4L)),
                     dtype = "float64")
    expect_jax_equal(vj(function(v) lax_fft(v, "RFFT", 5L), ct5, zc), carr(c(2, -1.6818137078720572, 5.0839267404623634, 2.0062432032871107, -2.4083562358774175, 4, 4.7052429754195773, -8.1583280314302993, -0.54987590106907058, 0.0029609570797921596), c(2L, 5L)),
                     dtype = "float64")
    expect_jax_equal(vj(function(v) lax_fft(v, "IRFFT", 4L), zc, xr), carr(complex(real = c(2.25, -0.5, 1.25, 5.5, 1.5, -2), imaginary = c(-0, -0, -0, -0, 1.5, -0)), c(2L, 3L)),
                     dtype = "complex128")
    expect_jax_equal(vj(function(v) lax_fft(v, "IRFFT", 5L), zc, ct5), carr(complex(real = c(3, -1, -1, -0.20000000000000001, 0.047213595499957961, -0.84721359549995801), imaginary = c(-0, -1.3763819204711734, -0.3249196962329064, -0, 0.79514789806688491, -2.1372271335072965)), c(2L, 3L)),
                     dtype = "complex128")
    expect_jax_equal(vj(function(v) lax_fft(v, "FFT", 4L), xc, xi + 1i * xr), carr(complex(real = c(18, 1, -12, 1, 19, 4, -13, -2), imaginary = c(9, 0, 5, -2, 22, 3, -8, 3)), c(2L, 4L)),
                     dtype = "complex128")
    expect_jax_equal(vj(function(v) lax_fft(v, "IFFT", 4L), xc, xi + 1i * xr), carr(complex(real = c(4.5, 0.25, -3, 0.25, 4.75, -0.5, -3.25, 1), imaginary = c(2.25, -0.5, 1.25, 0, 5.5, 0.75, -2, 0.75)), c(2L, 4L)),
                     dtype = "complex128")
    expect_jax_equal(vj(function(v) lax_fft(v, "RFFT", c(3L, 4L)), x3, z3), carr(c(-3, -4, -1, -8, -5.7679491924311224, 0.96410161513775428, -5.9641016151377544, 11.160254037844386, -9.2320508075688785, -5.9641016151377544, 0.96410161513775428, -6.1602540378443855, 0, 4, 0, 12, 10.794228634059948, -1.598076211353316, -3.4019237886466835, 8.598076211353316, -4.794228634059948, 3.5980762113533156, -8.598076211353316, 3.4019237886466835), c(2L, 3L, 4L)))
    expect_jax_equal(vj(function(v) lax_fft(v, "IRFFT", c(3L, 4L)), z3, x3), carr(complex(real = c(-0.41666666666666663, -1, 0.083333333333333329, -0.54166666666666663, -1.0103629710818449, -0.29166666666666663, -0.54166666666666663, 1.0103629710818449, -0.29166666666666663, -0.083333333333333329, 1.3333333333333333, 0.083333333333333329, 0.041666666666666664, 0.58333333333333326, -0.29166666666666663, 0.041666666666666664, 0.58333333333333326, -0.29166666666666663), imaginary = c(-0, 0.16666666666666666, -0, 0.36084391824351603, -0.58333333333333326, 0.50518148554092246, -0.36084391824351603, -0.58333333333333326, -0.50518148554092246, -0, 0.16666666666666666, -0, 0.36084391824351603, 0.1563036955848216, -0.50518148554092246, -0.36084391824351603, 2.1770296377485114, 0.50518148554092246)), c(2L, 3L, 3L)))
  })
})

test_that("jvp of lax_fft is the fft of the tangent", {
  skip_if_no_backend()
  r <- jvp(function(v) lax_fft(v, "IRFFT", 4L), list(as_jax(zc)), list(as_jax(Conj(zc))))
  expect_jax_equal(r[[1L]], as.array(lax_fft(zc, "IRFFT", 4L)))
  expect_jax_equal(r[[2L]], as.array(lax_fft(Conj(zc), "IRFFT", 4L)))
})

test_that("gradients of lax_fft match finite differences", {
  skip_if_no_backend()
  with_x64({
    w23 <- carr((1:6) / 7, c(2L, 3L))
    w24 <- carr((1:8) / 5, c(2L, 4L))
    w25 <- carr((1:10) / 9, c(2L, 5L))
    x0 <- as_jax(carr(c(0.3, -1.2, 0.5, 0.9, -0.4, 1.1), c(2L, 3L)), "float64")
    check_grad_fd(function(x) sum(w23 * abs2(lax_fft(x, "RFFT", 4L))), as_jax(xr, "float64"))
    check_grad_fd(function(x) sum(lax_real(lax_fft(x, "RFFT", 5L) * (w23 + 2i * w23[, 3:1]))),
                  as_jax(ct5, "float64"))
    check_grad_fd(function(x) sum(w24 * lax_fft(lax_complex(x, sin(x)), "IRFFT", 4L)^2), x0)
    check_grad_fd(function(x) sum(w25 * lax_fft(lax_complex(x, cos(x)), "IRFFT", 5L)^2), x0)
    check_grad_fd(function(x) sum(w24 * abs2(lax_fft(lax_complex(x, cos(x)), "FFT", 4L))),
                  as_jax(xr, "float64"))
    check_grad_fd(function(x) sum(w24 * abs2(lax_fft(lax_complex(x, x * x), "IFFT", c(2L, 4L)))),
                  as_jax(xr / 3, "float64"))
    check_grad_fd(function(x) sum(abs2(lax_fft(x, "RFFT", c(3L, 4L))) * (1:3)),
                  as_jax(x3 / 4, "float64"))
    check_grad_fd(function(x) sum(lax_fft(lax_complex(x, sin(x)), "IRFFT", c(3L, 4L))^3),
                  as_jax(Re(z3) / 5, "float64"))
  })
})

test_that("second order derivatives of lax_fft", {
  skip_if_no_backend()
  with_x64({
    # Complex inputs are built without lax_complex, whose second order transpose goes
    # through the imag transpose rule.
    f <- function(x) sum(abs2(lax_fft(x + 1i * x * x, "FFT", 4L)))
    g <- function(x) sum(grad(f)(x) * carr(c(1, -1, 2, 0.5, 3, 1, -2, 1), c(2L, 4L)))
    check_grad_fd(g, as_jax(xr / 4, "float64"))
    h <- function(x) sum(lax_fft(x * (1 + 2i) + cos(x), "IRFFT", 5L)^2)
    check_grad_fd(function(x) sum(grad(h)(x)^2), as_jax(Re(zc) / 3, "float64"))
  })
})

test_that("vmap of lax_fft matches a stacked loop", {
  skip_if_no_backend()
  x43 <- t(xr) + 0  # shape (4, 2): batch along axis 1
  out <- vmap(function(v) lax_fft(v, "RFFT", 4L), in_axes = 1L)(x43)
  loop <- do.call(rbind, lapply(1:2, function(i) as.array(lax_fft(x43[, i], "RFFT", 4L))))
  expect_jax_equal(out, loop, shape = c(2L, 3L))
  out <- vmap(function(v) lax_fft(v, "FFT", 4L))(xc)
  expect_jax_equal(out, carr(complex(real = c(9, -2, 5, 0, 22, 3, -8, 3), imaginary = c(18, 1, -12, 1, 19, -2, -13, 4)), c(2L, 4L)))
  zt <- t(zc)  # shape (3, 2)
  out <- vmap(function(v) lax_fft(v, "IRFFT", 4L), in_axes = 1L)(zt)
  expect_jax_equal(out, carr(c(0, -1, 2, 0, 2.25, 0.25, -2.75, 0.25), c(2L, 4L)), dtype = "float32")
  x3t <- aperm(x3, c(2L, 3L, 1L))  # batch along the last axis
  out <- vmap(function(v) lax_fft(v, "RFFT", c(3L, 4L)), in_axes = 2L)(x3t)
  expect_jax_equal(out, carr(complex(real = c(-5, -6, 1, -6.5, -6.06217766, -3.5, -6.5, 6.06217766, -3.5, -1, 8, 1, 0.5, 3.5, -3.5, 0.5, 3.5, -3.5), imaginary = c(0, -1, 0, -4.33012676, 3.5, -6.06217766, 4.33012676, 3.5, 6.06217766, 0, -1, 0, -4.33012676, -0.937822342, 6.06217766, 4.33012676, -13.0621777, -6.06217766)), c(2L, 3L, 3L)))
  with_x64({
    f <- function(x) sum(abs2(lax_fft(x, "RFFT", 4L)) * c(1, 2, 3))
    out <- vmap(grad(f))(as_jax(xr, "float64"))
    loop <- do.call(rbind, lapply(1:2, function(i) as.array(grad(f)(as_jax(xr[i, ], "float64")))))
    expect_jax_equal(out, loop)
  })
})

test_that("lax_fft reports jax errors", {
  skip_if_no_backend()
  expect_error(lax_fft(xc, "foo", 4L), "Unknown FFT type 'foo'", fixed = TRUE)
  expect_error(lax_fft(xc, 7, 4L), "Unknown FFT type value '7'", fixed = TRUE)
  expect_error(lax_fft(xc, "RFFT", 4L), "only real valued inputs supported for rfft")
  expect_error(lax_fft(xc, "FFT", 3L),
               "FFT input shape (2, 4) minor dimensions must be equal to fft_lengths (3,)",
               fixed = TRUE)
  expect_error(lax_fft(xr, "RFFT", 3L),
               "RFFT input shape (2, 4) minor dimensions must be equal to fft_lengths (3,)",
               fixed = TRUE)
  expect_error(lax_fft(xc, "IRFFT", c(3L, 4L)),
               "IRFFT input shape (2, 4) minor dimensions must be equal to all except the last fft_length, got fft_lengths=(3, 4)",
               fixed = TRUE)
  expect_error(lax_fft(xc[1, ], "FFT", c(4L, 4L)),
               "FFT input shape (4,) must have at least as many input dimensions as fft_lengths (4, 4).",
               fixed = TRUE)
  expect_error(lax_fft(as_jax(xr, "float16"), "RFFT", 4L),
               "RFFT input must be float32 or float64, got float16", fixed = TRUE)
  bind <- jaxR:::bind
  expect_error(bind(jaxR:::fft_p, list(as_jax(xr)), list(fft_type = "FFT", fft_lengths = 4L)),
               "FFT input must be complex64 or complex128, got float32", fixed = TRUE)
  expect_error(bind(jaxR:::fft_p, list(as_jax(xr)), list(fft_type = "IRFFT", fft_lengths = 4L)),
               "IRFFT input must be complex64 or complex128, got float32", fixed = TRUE)
})
