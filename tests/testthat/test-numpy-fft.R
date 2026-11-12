# Tests for jnp_fft_* (port of jax/_src/numpy/fft.py) and the DCTs of jax.scipy.fft.
# Expected values come from jax 0.12.0.1.

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
x4 <- carr((0:23 %% 5) - 2, c(2L, 2L, 2L, 3L))
x5 <- carr((0:23 %% 5) - 2, c(2L, 2L, 1L, 2L, 3L))
zc <- carr(c(1, -2, 3, 0, 5, -1), c(2L, 3L)) + 1i * carr(c(2, 1, -1, 3, 0, 4), c(2L, 3L))
z3 <- x3[, , 1:3] + 1i * x3[, , 2:4]
v5 <- c(1, -2, 3, 0.5, 4)
v7 <- as.numeric(0:6)

test_that("one dimensional transforms match jax", {
  skip_if_no_backend()
  expect_jax_equal(jnp_fft_fft(xr), carr(complex(real = c(9, -1, 5, -1, 22, 3, -8, 3), imaginary = c(0, 0, 0, 0, 0, -3, 0, 3)), c(2L, 4L)), dtype = "complex64")
  expect_jax_equal(jnp_fft_fft(xc), carr(complex(real = c(9, -2, 5, 0, 22, 3, -8, 3), imaginary = c(18, 1, -12, 1, 19, -2, -13, 4)), c(2L, 4L)))
  expect_jax_equal(jnp_fft_fft(xc, n = 3L), carr(complex(real = c(8, 5.69615221, -4.69615221, 16, 5.56217766, -6.56217766), imaginary = c(10, 0.598076105, -4.59807587, 11, -8.56217766, 3.56217766)), c(2L, 3L)), shape = c(2L, 3L))
  expect_jax_equal(jnp_fft_fft(xc, n = 6L, axis = 0L), carr(complex(real = c(8, 10, 6, 7, 7.2320509, 12.4282036, 5.86602545, 10.9282036, 2.2320509, 3.42820311, 3.86602545, 4.92820311, -1.99999976, -8, 2, -5, -1.23205078, -10.4282036, 2.13397455, -8.92820358, 3.76794887, -1.42820358, 4.13397455, -2.92820358), imaginary = c(4, 15, 2, 16, -1.33012676, 3.20577145, -0.232050776, 6.80384779, -3.33012676, -4.79422855, -1.23205078, -1.19615221, -2.38418579e-07, -1.00000095, 0, -9.53674316e-07, 5.33012676, 10.7942286, 2.2320509, 9.19615173, 7.33012676, 18.7942276, 3.2320509, 17.1961517)), c(6L, 4L)), shape = c(6L, 4L))
  expect_jax_equal(jnp_fft_fft(xc, norm = "ortho"), carr(complex(real = c(4.5, -1, 2.5, 0, 11, 1.5, -4, 1.5), imaginary = c(9, 0.5, -6, 0.5, 9.5, -1, -6.5, 2)), c(2L, 4L)))
  expect_jax_equal(jnp_fft_fft(xc, norm = "forward"), carr(complex(real = c(2.25, -0.5, 1.25, 0, 5.5, 0.75, -2, 0.75), imaginary = c(4.5, 0.25, -3, 0.25, 4.75, -0.5, -3.25, 1)), c(2L, 4L)))
  expect_jax_equal(jnp_fft_fft(xc, norm = "backward"), carr(complex(real = c(9, -2, 5, 0, 22, 3, -8, 3), imaginary = c(18, 1, -12, 1, 19, -2, -13, 4)), c(2L, 4L)))
  expect_jax_equal(jnp_fft_ifft(xc), carr(complex(real = c(2.25, 0, 1.25, -0.5, 5.5, 0.75, -2, 0.75), imaginary = c(4.5, 0.25, -3, 0.25, 4.75, 1, -3.25, -0.5)), c(2L, 4L)))
  expect_jax_equal(jnp_fft_ifft(xc, axis = 0L, norm = "ortho"), carr(complex(real = c(5.65685415, 7.07106781, 4.2426405, 4.94974756, -1.41421354, -5.65685415, 1.41421354, -3.53553391), imaginary = c(2.82842708, 10.6066017, 1.41421354, 11.3137083, 0, -0.707106769, 0, 0)), c(2L, 4L)))
  expect_jax_equal(jnp_fft_ifft(xc, norm = "forward"), carr(complex(real = c(9, 0, 5, -2, 22, 3, -8, 3), imaginary = c(18, 1, -12, 1, 19, 4, -13, -2)), c(2L, 4L)))
  expect_jax_equal(jnp_fft_rfft(xr), carr(complex(real = c(9, -1, 5, 22, 3, -8), imaginary = c(0, 0, 0, 0, -3, 0)), c(2L, 3L)), dtype = "complex64")
  expect_jax_equal(jnp_fft_rfft(xr, n = 6L, axis = 0L), carr(complex(real = c(8, 10, 6, 7, 5.5, 5.5, 5, 4, 0.5, -3.5, 3, -2, -2, -8, 2, -5), imaginary = c(0, 0, 0, 0, -4.33012676, -7.79422855, -1.73205078, -5.19615221, -4.33012676, -7.79422855, -1.73205078, -5.19615221, 0, 0, 0, 0)), c(4L, 4L)), shape = c(4L, 4L))
  expect_jax_equal(jnp_fft_rfft(x3, axis = 1L, norm = "ortho"), carr(complex(real = c(-2.30940104, -0.577350259, 1.15470052, -1.15470052, -1.44337559, -1.44337559, -1.44337559, 0.577350259, 2.30940104, 0, -2.30940104, -0.577350259, 0.577350259, 2.59807611, -1.44337559, -1.44337559), imaginary = c(0, 0, 0, 0, -1.49999988, -1.49999988, -1.49999988, 1.99999988, 0, 0, 0, 0, 1.99999988, -1.49999988, -1.49999988, -1.49999988)), c(2L, 2L, 4L)))
  expect_jax_equal(jnp_fft_irfft(zc), carr(c(0, -1, 2, 0, 2.25, 0.25, -2.75, 0.25), c(2L, 4L)), dtype = "float32", shape = c(2L, 4L))
  expect_jax_equal(jnp_fft_irfft(zc, n = 5L), carr(c(0.600000024, -1.16334248, 0.60249728, 1.83357072, -0.872725487, 1.60000002, 0.00118436816, -0.219950393, -3.26333117, 1.88209724), c(2L, 5L)))
  expect_jax_equal(jnp_fft_irfft(zc, axis = 0L), carr(c(0.5, 1.5, 1, 0.5, -3.5, 2), c(2L, 3L)), shape = c(2L, 3L))
  expect_jax_equal(jnp_fft_irfft(zc, n = 7L, norm = "forward"), carr(c(3.00000024, -2.44289207, -6.33335304, 5.91338444, 10.7762451, -0.698106289, -3.21527791, 8, -1.11948359, 3.0477984, -4.00201607, -16.5113201, -3.89434147, 14.4793634), c(2L, 7L)))
  expect_jax_equal(jnp_fft_hfft(zc), carr(c(0, 0, 8, -4, 9, 1, -11, 1), c(2L, 4L)), dtype = "float32")
  expect_jax_equal(jnp_fft_hfft(zc, n = 5L, norm = "ortho"), carr(c(1.34164083, -1.95147347, 4.09998846, 1.34722483, -2.60131264, 3.57770896, 4.20849752, -7.29702997, -0.491824031, 0.00264832773), c(2L, 5L)))
  expect_jax_equal(jnp_fft_ihfft(xr), carr(complex(real = c(2.25, -0.25, 1.25, 5.5, 0.75, -2), imaginary = c(0, -0, 0, 0, 0.75, -0)), c(2L, 3L)), dtype = "complex64")
  expect_jax_equal(jnp_fft_ihfft(xr, n = 5L, norm = "forward"), carr(complex(real = c(9, -0.73606807, 3.73606801, 22, 1.30901718, 0.190983057), imaginary = c(0, 2.71441245, -2.26538444, 0, 6.20836782, 9.09429359)), c(2L, 3L)))
})

test_that("two and n dimensional transforms match jax", {
  skip_if_no_backend()
  expect_jax_equal(jnp_fft_fft2(x3), carr(complex(real = c(-5, -6, 1, -6, -6.5, -6.06217766, -3.5, 6.06217766, -6.5, 6.06217766, -3.5, -6.06217766, -1, 8, 1, 8, 0.5, 3.5, -3.5, 3.5, 0.5, 3.5, -3.5, 3.5), imaginary = c(0, -1, 0, 1, -4.33012676, 3.5, -6.06217766, -3.5, 4.33012676, 3.5, 6.06217766, -3.5, 0, -1, 0, 1, -4.33012676, -0.937822342, 6.06217766, 13.0621777, 4.33012676, -13.0621777, -6.06217766, 0.937822342)), c(2L, 3L, 4L)))
  expect_jax_equal(jnp_fft_fft2(x3, s = c(3L, 3L), axes = c(0L, 2L)), carr(complex(real = c(-4, 0.5, 0.5, 6, -3, -3, -5, 4, 4, -7, -6.99999952, 1.99999952, 6, 0, -1.5, -2, -3.5, -5, -7, 1.99999952, -6.99999952, 6, -1.5, 0, -2, -5, -3.5), imaginary = c(0, -4.33012676, 4.33012676, 0, 1.73205078, -1.73205078, 0, 1.73205078, -1.73205078, -1.73205078, 1.73205078, -5.19615221, 0, 1.73205078, 0.866025329, 1.73205078, -4.33012724, -5.19615221, 1.73205078, 5.19615221, -1.73205078, 0, -0.866025329, -1.73205078, -1.73205078, 5.19615221, 4.33012724)), c(3L, 3L, 3L)),
                   shape = c(3L, 3L, 3L))
  expect_jax_equal(jnp_fft_ifft2(x3, norm = "ortho"), carr(complex(real = c(-1.44337571, -1.73205078, 0.2886751, -1.73205078, -1.87638843, -1.75, -1.01036298, 1.75, -1.87638843, 1.75, -1.01036298, -1.75, -0.2886751, 2.30940104, 0.2886751, 2.30940104, 0.14433755, 1.01036298, -1.01036298, 1.01036298, 0.14433755, 1.01036298, -1.01036298, 1.01036298), imaginary = c(0, 0.2886751, 0, -0.2886751, 1.25, -1.01036298, 1.75, 1.01036298, -1.25, -1.01036298, -1.75, 1.01036298, 0, 0.288675159, 0, -0.288675159, 1.25, 0.270726055, -1.75, -3.77072573, -1.25, 3.77072573, 1.75, -0.270726055)), c(2L, 3L, 4L)))
  expect_jax_equal(jnp_fft_rfft2(x3), carr(complex(real = c(-5, -6, 1, -6.5, -6.06217766, -3.5, -6.5, 6.06217766, -3.5, -1, 8, 1, 0.5, 3.5, -3.5, 0.5, 3.5, -3.5), imaginary = c(0, -1, 0, -4.33012676, 3.5, -6.06217766, 4.33012676, 3.5, 6.06217766, 0, -1, 0, -4.33012676, -0.937822342, 6.06217766, 4.33012676, -13.0621777, -6.06217766)), c(2L, 3L, 3L)))
  expect_jax_equal(jnp_fft_irfft2(z3), carr(c(-0.333333343, -0.833333373, 0, -0.166666672, -1.19417715, -0.521527529, 0.505181491, -0.488835454, -0.472489476, 1.35486078, -0.505181551, -0.344497859, 0, 1.33333337, 0, 0, -0.241025329, 0.275320649, -0.875, 0.308012664, 1.49102533, 1.14134598, -0.875, -0.558012664), c(2L, 3L, 4L)), shape = c(2L, 3L, 4L))
  expect_jax_equal(jnp_fft_irfft2(z3, s = c(3L, 5L)), carr(c(-0.13333334, -0.620479167, -0.48675102, 0.333964616, -0.426734507, -0.891068339, -0.948343575, 0.698352337, -0.198801339, -0.359497696, -0.775598347, 1.42563868, -0.0528804176, -0.446670562, -0.117797159, -0.266666681, 1.28374398, 0.288535535, -0.0848203376, 0.112540826, -0.532692075, 0.501626015, -0.516998768, -0.526499629, 0.541872382, 1.1993587, 1.5677669, -0.768680513, -0.274817854, -0.524268448), c(2L, 3L, 5L)), shape = c(2L, 3L, 5L))
  expect_jax_equal(jnp_fft_fftn(x3), carr(complex(real = c(-6, 2, 2, 2, -6, -2.56217766, -7, 9.56217766, -6, 9.56217766, -7, -2.56217766, -4, -14, 0, -14, -7, -9.56217766, 0, 2.56217766, -7, 2.56217766, 0, -9.56217766), imaginary = c(0, -2, 0, 2, -8.66025352, 2.56217766, 0, 9.56217766, 8.66025352, -9.56217766, 0, -2.56217766, 0, 0, 0, 0, 0, 4.43782234, -12.1243553, -16.5621777, 0, 16.5621777, 12.1243553, -4.43782234)), c(2L, 3L, 4L)))
  expect_jax_equal(jnp_fft_fftn(x3, s = c(2L, 5L)), carr(complex(real = c(-3, -1.19098294, -2.30901718, -2.30901718, -1.19098294, -9, -4.42705107, -1.07294905, -1.07294905, -4.42705107, 2, 3.54508495, -2.04508495, -2.04508495, 3.54508495, -2, 10.3991871, -1.89918709, -1.89918709, 10.3991871), imaginary = c(0, -2.93892622, 4.75528288, -4.75528288, 2.93892622, 0, 7.91872263, -4.30625486, 4.30625486, -7.91872263, 0, -1.67759919, -3.66546869, 3.66546869, 1.67759919, 0, -2.85316968, -1.76335573, 1.76335573, 2.85316968)), c(2L, 2L, 5L)), shape = c(2L, 2L, 5L))
  expect_jax_equal(jnp_fft_fftn(x3, axes = c(-1L, 0L)), carr(complex(real = c(-6, 3, -4, 3, 5, -4, 3, -4, -5, 3, 3, 3, -6, -7, 0, -7, 1, 0, 7, 0, 1, -7, -7, -7), imaginary = c(0, -3, 0, 3, 0, -3, 0, 3, 0, 4, 0, -4, 0, 7, 0, -7, 0, -7, 0, 7, 0, 0, 0, 0)), c(2L, 3L, 4L)))
  expect_jax_equal(jnp_fft_ifftn(x3, axes = 1L, norm = "forward"), carr(complex(real = c(-4, -1, 2, -2, -2.5, -2.5, -2.5, 1, -2.5, -2.5, -2.5, 1, 4, 0, -4, -1, 1, 4.5, -2.5, -2.5, 1, 4.5, -2.5, -2.5), imaginary = c(0, 0, 0, 0, 2.59807611, 2.59807611, 2.59807611, -3.46410155, -2.59807611, -2.59807611, -2.59807611, 3.46410155, 0, 0, 0, 0, -3.46410155, 2.59807611, 2.59807611, 2.59807611, 3.46410155, -2.59807611, -2.59807611, -2.59807611)), c(2L, 3L, 4L)))
  expect_jax_equal(jnp_fft_rfftn(x3), carr(complex(real = c(-6, 2, 2, -6, -2.56217766, -7, -6, 9.56217766, -7, -4, -14, 0, -7, -9.56217766, 0, -7, 2.56217766, 0), imaginary = c(0, -2, 0, -8.66025352, 2.56217766, 0, 8.66025352, -9.56217766, 0, 0, 0, 0, 0, 4.43782234, -12.1243553, 0, 16.5621777, 12.1243553)), c(2L, 3L, 3L)))
  expect_jax_equal(jnp_fft_irfftn(z3), carr(c(-0.166666672, 0.25, 0, -0.0833333358, -0.717601299, -0.123103425, -0.184909269, -0.0904113874, 0.509267986, 1.24810338, -0.690090775, -0.451255262, -0.166666672, -1.08333337, 0, -0.0833333358, -0.476575941, -0.398424089, 0.690090775, -0.398424089, -0.981757402, 0.106757402, 0.184909269, 0.106757402), c(2L, 3L, 4L)))
  expect_jax_equal(jnp_fft_irfftn(z3, axes = c(0L, 2L)), carr(c(-0.375, 1.375, -0.875, -0.625, 1, -1.5, 0, 0.5, -1.125, 0.875, 0.875, -0.125, -1.625, -1.375, 0.875, -0.375, 1, -0.5, 0, 0.5, 0.125, -1.375, -0.875, -0.375), c(2L, 3L, 4L)))
})

test_that("transforms over more than three axes match jax", {
  skip_if_no_backend()
  expect_jax_equal(jnp_fft_fftn(x4), carr(complex(real = c(-2, -2, -2, 4, -5, -5, -2, -5, -5, 0, 0, 0, -4, 5, 5, -10, -10, -10, 0, 0, 0, -10, 5, 5), imaginary = c(0, -1.73205066, 1.73205066, 0, 0, 0, 0, 0, 0, 0, 8.66025352, -8.66025352, 0, 0, 0, 0, 8.66025352, -8.66025352, 0, -8.66025352, 8.66025352, 0, 0, 0)), c(2L, 2L, 2L, 3L)))
  expect_jax_equal(jnp_fft_ifftn(x4, norm = "ortho"), carr(complex(real = c(-0.408248276, -0.408248365, -0.408248365, 0.816496611, -1.02062082, -1.02062082, -0.408248305, -1.02062082, -1.02062082, 0, 0, 0, -0.81649673, 1.0206207, 1.0206207, -2.04124165, -2.04124165, -2.04124165, 0, 0, 0, -2.04124165, 1.02062082, 1.02062082), imaginary = c(0, 0.353553385, -0.353553385, 0, 0, 0, 0, 0, 0, 0, -1.76776695, 1.76776695, 0, 0, 0, 0, -1.76776695, 1.76776695, 0, 1.76776695, -1.76776695, 0, 0, 0)), c(2L, 2L, 2L, 3L)))
  expect_jax_equal(jnp_fft_rfftn(x4), carr(complex(real = c(-2, -2, 4, -5, -2, -5, 0, 0, -4, 5, -10, -10, 0, 0, -10, 5), imaginary = c(0, -1.73205066, 0, 0, 0, 0, 0, 8.66025352, 0, 0, 0, 8.66025352, 0, -8.66025352, 0, 0)), c(2L, 2L, 2L, 2L)), shape = c(2L, 2L, 2L, 2L))
  expect_jax_equal(jnp_fft_irfftn(jnp_fft_rfftn(x4), s = dim(x4)), carr(c(-2, -1, -3.9736431e-08, 1, 1.99999988, -1.99999988, -1, 3.9736431e-08, 1, 2, -2, -1, 0, 1, 2, -2, -1, -3.9736431e-08, 1, 1.99999988, -1.99999988, -1, 3.9736431e-08, 1), c(2L, 2L, 2L, 3L)))
  expect_jax_equal(jnp_fft_fftn(x5), carr(complex(real = c(-2, -2, -2, 4, -5, -5, -2, -5, -5, 0, 0, 0, -4, 5, 5, -10, -10, -10, 0, 0, 0, -10, 5, 5), imaginary = c(0, -1.73205066, 1.73205066, 0, 0, 0, 0, 0, 0, 0, 8.66025352, -8.66025352, 0, 0, 0, 0, 8.66025352, -8.66025352, 0, -8.66025352, 8.66025352, 0, 0, 0)), c(2L, 2L, 1L, 2L, 3L)))
  expect_jax_equal(jnp_fft_rfftn(x5), carr(complex(real = c(-2, -2, 4, -5, -2, -5, 0, 0, -4, 5, -10, -10, 0, 0, -10, 5), imaginary = c(0, -1.73205066, 0, 0, 0, 0, 0, 8.66025352, 0, 0, 0, 8.66025352, 0, -8.66025352, 0, 0)), c(2L, 2L, 1L, 2L, 2L)))
  expect_jax_equal(jnp_fft_irfftn(jnp_fft_rfftn(x5) * (1 + 0.5i), s = dim(x5)), carr(c(-2.28867531, -0.422649682, -0.288675189, 2.15470052, 1.13397455, -2.28867507, -1.28867519, 0.577350378, 0.711324871, 1.71132493, -2.86602545, 0.154700607, -0.288675159, 1.5773505, 1.71132493, -2.28867531, -0.422649682, -0.288675189, 2.15470076, 1.13397443, -2.28867507, -1.28867531, 0.577350438, 0.71132493), c(2L, 2L, 1L, 2L, 3L)),
                   tolerance = 1e-4)
})

test_that("fftfreq and rfftfreq match jax", {
  skip_if_no_backend()
  expect_jax_equal(jnp_fft_fftfreq(8L), carr(c(0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125), c(8L)), dtype = "float32")
  expect_jax_equal(jnp_fft_fftfreq(7L, d = 0.5), carr(c(0, 0.285714298, 0.571428597, 0.857142925, -0.857142925, -0.571428597, -0.285714298), c(7L)))
  expect_jax_equal(jnp_fft_rfftfreq(8L), carr(c(0, 0.125, 0.25, 0.375, 0.5), c(5L)), dtype = "float32")
  expect_jax_equal(jnp_fft_rfftfreq(7L, d = 0.1), carr(c(0, 1.42857146, 2.85714293, 4.28571415), c(4L)))
  expect_identical(dtype(jnp_fft_fftfreq(4L, dtype = "complex64")), "complex64")
  with_x64(expect_jax_equal(jnp_fft_fftfreq(7L, d = 0.5, dtype = "float64"), carr(c(0, 0.2857142857142857, 0.5714285714285714, 0.8571428571428571, -0.8571428571428571, -0.5714285714285714, -0.2857142857142857), c(7L)),
                            dtype = "float64"))
})

test_that("fftshift and ifftshift match jax", {
  skip_if_no_backend()
  expect_jax_equal(jnp_fft_fftshift(as_jax(v7, "float32")), carr(c(4, 5, 6, 0, 1, 2, 3), c(7L)))
  expect_jax_equal(jnp_fft_ifftshift(as_jax(v7, "float32")), carr(c(3, 4, 5, 6, 0, 1, 2), c(7L)))
  expect_jax_equal(jnp_fft_fftshift(x3), carr(c(-2, -1, 3, -3, -3, -2, 2, 3, 1, 2, -1, 0, 0, 1, -2, -1, -1, 0, -3, -2, 3, -3, 1, 2), c(2L, 3L, 4L)))
  expect_jax_equal(jnp_fft_fftshift(x3, axes = 1L), carr(c(-2, -1, 0, 1, -3, -2, -1, 0, 1, 2, 3, -3, 3, -3, -2, -1, 2, 3, -3, -2, -1, 0, 1, 2), c(2L, 3L, 4L)))
  expect_jax_equal(jnp_fft_ifftshift(x3, axes = c(-1L, 0L)), carr(c(-3, -2, 2, 3, 1, 2, -1, 0, -2, -1, 3, -3, -1, 0, -3, -2, 3, -3, 1, 2, 0, 1, -2, -1), c(2L, 3L, 4L)))
  expect_jax_equal(jnp_fft_ifftshift(jnp_fft_fftshift(x3)), x3)
})

test_that("scipy dct and idct match jax", {
  skip_if_no_backend()
  expect_jax_equal(scipy_fft_dct(xr), carr(c(18, 1.3994174, -1.41421354, 7.07401037, 44, 3.50980926, 0, -13.6996794), c(2L, 4L)), dtype = "float32")
  expect_jax_equal(scipy_fft_dct(xr, axis = 0L, norm = "ortho"), carr(c(5.65685415, 7.07106781, 4.2426405, 4.94974756, -1.41421354, -5.65685415, 1.41421354, -3.53553391), c(2L, 4L)))
  expect_jax_equal(scipy_fft_dct(xr, n = 6L), carr(c(18, 8.76268291, -3.46410155, -1.41421366, 6, 5.93425465, 44, 20.3166275, -5.19615221, 3.37174782e-07, -5, -17.8671417), c(2L, 6L)), shape = c(2L, 6L))
  expect_jax_equal(scipy_fft_dct(xr, n = 3L), carr(c(16, -1.73205066, 5, 32, 5.19615221, -11), c(2L, 3L)), shape = c(2L, 3L))
  expect_jax_equal(scipy_fft_dct(v5), carr(c(13, -8.64526653, 3.01722121, 1.22857082, 11.5172215), c(5L)))
  expect_jax_equal(scipy_fft_dct(xc), carr(complex(real = c(18, 1.3994174, -1.41421354, 7.07401037, 44, 3.50980926, 0, -13.6996794), imaginary = c(36, -6.49435282, 2.82842708, -15.6787548, 38, -5.72898579, 1.41421354, -17.5265141)), c(2L, 4L)), dtype = "complex64")
  expect_jax_equal(scipy_fft_dct(as_jax(xr, "int32")), carr(c(18, 1.3994174, -1.41421354, 7.07401037, 44, 3.50980926, 0, -13.6996794), c(2L, 4L)), dtype = "float32")
  expect_jax_equal(scipy_fft_idct(xr), carr(c(1.40874755, -0.467405766, -0.196807772, 0.755466104, 3.63130713, -0.25333491, 0.79622817, -1.67420042), c(2L, 4L)))
  expect_jax_equal(scipy_fft_idct(xr, norm = "ortho"), carr(c(4.42387962, -0.882683396, -0.117316604, 2.57612038, 11.0031204, 0.0156936646, 2.98430634, -4.0031209), c(2L, 4L)))
  expect_jax_equal(scipy_fft_idct(v5), carr(c(0.510979652, -1.06284356, 0.300000012, -0.40240398, 1.15426791), c(5L)))
  expect_jax_equal(scipy_fft_idct(xr, n = 5L, axis = 0L), carr(c(1.25105655, 1.81190145, 0.780422628, 1.24126768, 0.887785137, 1.15801334, 0.635114133, 0.805342197, 0.299999982, 0.0999999568, 0.400000006, 0.100000001, -0.287785292, -0.958013356, 0.164885879, -0.605342209, -0.651056528, -1.61190152, 0.019577384, -1.04126775), c(5L, 4L)), shape = c(5L, 4L))
  expect_jax_equal(scipy_fft_idct(scipy_fft_dct(v5, norm = "ortho"), norm = "ortho"), v5,
                   tolerance = 1e-5)
})

test_that("scipy dctn and idctn match jax", {
  skip_if_no_backend()
  expect_jax_equal(scipy_fft_dctn(x3), carr(c(-48, 25.2345753, 0, 1.79336619, -6.92820549, 18.5591602, -34.2928543, -44.8057747, -84, 10.7151365, 59.3969688, -25.8686256, -22.6274166, -51.7372551, -55.9999962, 21.4302692, -34.2928543, -13.1233082, 24.2487087, 31.6824646, -19.7989883, -80.7442703, 13.999999, -12.0151024), c(2L, 3L, 4L)))
  expect_jax_equal(scipy_fft_dctn(x3, axes = c(0L, 2L), norm = "ortho"), carr(c(-2.12132025, 1.41864836, 0, -2.11835694, 1.76776695, 0.0792562962, -2.47487378, 1.11522114, -1.76776695, 0.0792562962, 2.47487378, 1.11522114, -2.12132025, -4.57296991, 0, 1.89418626, 0.353553385, 3.23357797, -2.47487378, 1.33939195, 0.353553385, -3.23357797, -2.47487378, -1.33939195), c(2L, 3L, 4L)))
  expect_jax_equal(scipy_fft_dctn(xr, s = c(3L, 5L)), carr(c(124, 37.4872322, -19.1246128, 7.39645052, -21.1246109, 31.1769142, 9.88367081, -7.58978748, 6.10844374, 11.4627714, -70, -20.3682175, 5.97871494, 3.18368363, 40.978714), c(3L, 5L)), shape = c(3L, 5L))
  expect_jax_equal(scipy_fft_dctn(x3, axes = 1L), carr(c(-8, -2, 4, -4, -1.7320509, -1.7320509, -1.7320509, -1.73205078, -7, -7, -7, 7, 8, 0, -8, -2, -1.73205078, 10.3923044, -1.7320509, -1.7320509, 7, -2.68220901e-07, -7, -7), c(2L, 3L, 4L)))
  expect_jax_equal(scipy_fft_dctn(x4), carr(c(-32, -13.8564062, -40, 45.2548332, -48.9897919, -28.2842674, -22.6274185, -48.9897957, -28.2842693, 0, -34.6410179, 60.0000038, -45.2548294, 48.9897957, 28.2842693, -79.9999924, -103.92305, 20.0000076, 0, 34.6410141, -60, -56.5685425, 24.4948959, 14.1421337), c(2L, 2L, 2L, 3L)))
  expect_jax_equal(scipy_fft_idctn(x3), carr(c(-0.005727713, 0.046100013, -0.0257325973, 0.0135608977, 0.093314223, 0.042679213, -0.0972900316, -0.135721192, -0.175191343, 0.132866636, 0.127709284, -0.0272906721, 0.0089804111, -0.0574908294, -0.154589176, 0.0387344547, -0.113574095, 0.0209638141, 0.0839431435, 0.147351623, -0.0612305552, -0.282510102, 0.064238973, -0.0590943098), c(2L, 3L, 4L)))
  expect_jax_equal(scipy_fft_idctn(xr, norm = "ortho"), carr(c(10.9085369, -0.613054037, 2.02726769, -1.00904274, -4.65222645, -0.635248423, -2.19317889, 4.65222645), c(2L, 4L)))
  expect_jax_equal(scipy_fft_idctn(xc), carr(complex(real = c(1.63604784, -0.206418902, 0.232307285, -0.403052807, -0.931674242, -0.0272840261, -0.330711067, 0.780785799), imaginary = c(1.77700102, -0.633004844, 0.721393168, -1.26183593, -0.372534692, 0.080585137, -0.0957501829, 0.284146309)), c(2L, 4L)))
  expect_jax_equal(scipy_fft_idctn(xr, s = c(3L, 3L)), carr(c(1.32934463, -0.0907762945, -0.266880363, 0.24255687, -0.138888866, 0.146331936, -0.844230652, -0.187001407, 0.559544265), c(3L, 3L)), shape = c(3L, 3L))
})

test_that("64 bit transforms match jax", {
  skip_if_no_backend()
  with_x64({
    expect_jax_equal(jnp_fft_fft(xc, norm = "ortho"), carr(complex(real = c(4.5, -1, 2.5, 0, 11, 1.5, -4, 1.5), imaginary = c(9, 0.5, -6, 0.5, 9.5, -1, -6.5, 2)), c(2L, 4L)), dtype = "complex128")
    expect_jax_equal(jnp_fft_rfft(xr, n = 5L), carr(complex(real = c(9, -0.73606797749978981, 3.7360679774997898, 22, 1.3090169943749475, 0.19098300562505255), imaginary = c(0, -2.7144122731725728, 2.2653842965929876, 0, -6.2083676374864885, -9.094293335812873)), c(2L, 3L)), dtype = "complex128")
    expect_jax_equal(jnp_fft_irfftn(z3), carr(c(-0.16666666666666666, 0.25, 0, -0.083333333333333329, -0.71760132768064411, -0.1231034446626806, -0.18490925722953877, -0.09041137421157526, 0.50926799434731074, 1.2481034446626804, -0.69009074277046123, -0.45125529245509144, -0.16666666666666666, -1.0833333333333333, 0, -0.083333333333333329, -0.4765759238962054, -0.39842407610379454, 0.69009074277046123, -0.39842407610379454, -0.98175740943712786, 0.10675740943712793, 0.18490925722953877, 0.10675740943712793), c(2L, 3L, 4L)), dtype = "float64")
    expect_jax_equal(jnp_fft_fftn(x5), carr(complex(real = c(-2, -2, -2, 4, -5, -5, -2, -5, -5, 0, 0, 0, -4, 5, 5, -10, -10, -10, 0, 0, 0, -10, 5, 5), imaginary = c(0, -1.7320508075688776, 1.7320508075688776, 0, 0, 0, 0, 0, 0, 0, 8.6602540378443855, -8.6602540378443855, 0, 0, 0, 0, 8.6602540378443855, -8.6602540378443855, 0, -8.6602540378443855, 8.6602540378443855, 0, 0, 0)), c(2L, 2L, 1L, 2L, 3L)), dtype = "complex128")
    expect_jax_equal(scipy_fft_dct(xr, norm = "ortho"), carr(c(4.5, 0.494768814657081, -0.5, 2.5010405474607613, 11, 1.2409048680735013, 0, -4.8435684271404158), c(2L, 4L)), dtype = "float64")
    expect_jax_equal(scipy_fft_idctn(x3), carr(c(-0.005727718391150753, 0.046100012012224953, -0.025732599558714698, 0.01356090753454001, 0.093314228262754886, 0.042679204403944834, -0.097290044894770708, -0.13572118463635363, -0.17519137204345697, 0.13286664949073052, 0.12770930001906289, -0.027290686902174371, 0.0089804102277215529, -0.0574908264188995, -0.15458918725143497, 0.038734452161083281, -0.11357409588602005, 0.020963834058745169, 0.083943122314853541, 0.14735160304351261, -0.061230567557918725, -0.2825101287682229, 0.064239012240799379, -0.059094323460856329), c(2L, 3L, 4L)), dtype = "float64")
    expect_jax_equal(scipy_fft_dctn(x3, norm = "ortho"), carr(c(-1.2247448713915889, 0.91057433653646269, -1.1102230246251565e-16, 0.064712525638503368, -0.25000000000000033, 0.94709317525584491, -1.7499999999999996, -2.2864851885336592, -3.0310889132455361, 0.54680449968161937, 3.0310889132455352, -1.3201028390980001, -0.81649658092772603, -2.6402056781960006, -2.8577380332470415, 1.093608999363239, -1.7500000000000007, -0.94709317525584547, 1.7500000000000004, 2.2864851885336592, -1.0103629710818456, -5.8272158560736198, 1.0103629710818456, -0.86711515962847696), c(2L, 3L, 4L)), dtype = "float64")
  })
})

test_that("transforms under jit match eager", {
  skip_if_no_backend()
  f <- jit(function(a, b) list(jnp_fft_rfft2(a, norm = "ortho"), jnp_fft_irfftn(b)))
  r <- f(x3, z3)
  expect_jax_equal(r[[1L]], as.array(jnp_fft_rfft2(x3, norm = "ortho")))
  expect_jax_equal(r[[2L]], carr(c(-0.166666672, 0.25, 0, -0.0833333358, -0.717601299, -0.123103425, -0.184909269, -0.0904113874, 0.509267986, 1.24810338, -0.690090775, -0.451255262, -0.166666672, -1.08333337, 0, -0.0833333358, -0.476575941, -0.398424089, 0.690090775, -0.398424089, -0.981757402, 0.106757402, 0.184909269, 0.106757402), c(2L, 3L, 4L)))
  expect_jax_equal(jit(function(a) jnp_fft_fftn(a))(x5), carr(complex(real = c(-2, -2, -2, 4, -5, -5, -2, -5, -5, 0, 0, 0, -4, 5, 5, -10, -10, -10, 0, 0, 0, -10, 5, 5), imaginary = c(0, -1.73205066, 1.73205066, 0, 0, 0, 0, 0, 0, 0, 8.66025352, -8.66025352, 0, 0, 0, 0, 8.66025352, -8.66025352, 0, -8.66025352, 8.66025352, 0, 0, 0)), c(2L, 2L, 1L, 2L, 3L)))
  expect_jax_equal(jit(function(a) jnp_fft_fftshift(a, axes = 1L))(x3), carr(c(-2, -1, 0, 1, -3, -2, -1, 0, 1, 2, 3, -3, 3, -3, -2, -1, 2, 3, -3, -2, -1, 0, 1, 2), c(2L, 3L, 4L)))
  expect_jax_equal(jit(function(a) scipy_fft_dctn(a))(x3), carr(c(-48, 25.2345753, 0, 1.79336619, -6.92820549, 18.5591602, -34.2928543, -44.8057747, -84, 10.7151365, 59.3969688, -25.8686256, -22.6274166, -51.7372551, -55.9999962, 21.4302692, -34.2928543, -13.1233082, 24.2487087, 31.6824646, -19.7989883, -80.7442703, 13.999999, -12.0151024), c(2L, 3L, 4L)))
  expect_jax_equal(jit(function(a) scipy_fft_idct(a, n = 5L, axis = 0L))(xr), carr(c(1.25105655, 1.81190145, 0.780422628, 1.24126768, 0.887785137, 1.15801334, 0.635114133, 0.805342197, 0.299999982, 0.0999999568, 0.400000006, 0.100000001, -0.287785292, -0.958013356, 0.164885879, -0.605342209, -0.651056528, -1.61190152, 0.019577384, -1.04126775), c(5L, 4L)))
  expect_jax_equal(jit(function(a) jnp_fft_hfft(a, n = 5L, norm = "ortho"))(zc), carr(c(1.34164083, -1.95147347, 4.09998846, 1.34722483, -2.60131264, 3.57770896, 4.20849752, -7.29702997, -0.491824031, 0.00264832773), c(2L, 5L)))
  # A module larger than 10000 bytes (the compile cache key limit).
  expect_jax_equal(jit(function(a) scipy_fft_idctn(a, norm = "ortho"))(x5), carr(c(-0.30258584, -0.119573116, -0.80258584, 0.774764299, -1.27427363, -0.725235581, -0.422158957, -1.10517168, -0.922159076, -0.528312266, -1.44337547, 1.97168779, -0.774764419, 1.27427375, 0.72523582, -2.56955338, -3.48461699, -0.0695536956, 0.528312266, 1.44337547, -1.97168779, -1.97168767, 1.44337559, 0.528312147), c(2L, 2L, 1L, 2L, 3L)))
})

test_that("vjp of numpy and scipy transforms matches jax", {
  skip_if_no_backend()
  with_x64({
    vj <- function(f, x, ct) vjp(f, as_jax(x))[[2L]](as_jax(ct))[[1L]]
    expect_jax_equal(vj(function(v) jnp_fft_hfft(v, n = 5L), zc, carr(as.numeric(0:9), c(2L, 5L))),
                     carr(complex(real = c(10, -5, -5, 35, -5, -5), imaginary = c(0, 6.8819096023558677, 1.624598481164532, 0, 6.8819096023558677, 1.624598481164532)), c(2L, 3L)), dtype = "complex128")
    expect_jax_equal(vj(function(v) jnp_fft_irfftn(v, norm = "ortho"), z3, x3), carr(complex(real = c(-1.2247448713915889, 0.81649658092772626, 0.40824829046386313, -1.2247448713915889, -1.0460047175293954, -1.4288690166235203, -1.2247448713915889, 3.9037427507764364, -1.4288690166235203, -0.81649658092772592, -5.7154760664940811, 0, -1.4288690166235203, -3.9037427507764364, 0, -1.4288690166235203, 1.0460047175293954, 0), imaginary = c(-0, 0.81649658092772626, -0, 1.7677669529663684, -1.0460047175293954, -0, -1.7677669529663684, 3.9037427507764364, -0, -0, -0, -0, -0, -1.8117333157176452, 2.4748737341529159, -0, -6.761480784023477, -2.4748737341529159)), c(2L, 3L, 3L)))
    expect_jax_equal(vj(function(v) scipy_fft_dct(v, norm = "ortho"), xr, xi), carr(c(8.2377547776521052, -2.832065508993816, 3.832065508993816, -5.2377547776521052, 8.8910362600902939, -2.5614674589207178, 3.5614674589207178, -5.8910362600902939), c(2L, 4L)))
  })
})

test_that("gradients match finite differences", {
  skip_if_no_backend()
  with_x64({
    x0 <- as_jax(carr(c(0.3, -1.2, 0.5, 0.9, -0.4, 1.1, 0.7, -0.2), c(2L, 4L)), "float64")
    w <- carr((1:8) / 5, c(2L, 4L))
    check_grad_fd(function(x) sum(abs2(jnp_fft_rfft(x, n = 6L, norm = "ortho")) * (1:4)), x0)
    check_grad_fd(function(x) sum(w * jnp_fft_irfft(x + 1i * cos(x), n = 4L)^2),
                  as_jax(xr[, 1:3] / 4, "float64"))
    check_grad_fd(function(x) sum(abs2(jnp_fft_fftn(x * x, s = c(3L, 3L), axes = c(1L, 0L)))), x0)
    check_grad_fd(function(x) sum(jnp_fft_hfft(sin(x) + 1i * x, n = 5L, norm = "forward")^2),
                  as_jax(xr[, 1:3] / 4, "float64"))
    check_grad_fd(function(x) sum(abs2(jnp_fft_ihfft(x * x, n = 5L)) * (1:3)), x0)
    check_grad_fd(function(x) sum(jnp_fft_irfftn(jnp_fft_rfftn(sin(x)) * (1 + 2i), s = c(2L, 2L, 2L, 3L))^2),
                  as_jax(x4 / 3, "float64"))
    check_grad_fd(function(x) sum(w * scipy_fft_dct(x * x, norm = "ortho")), x0)
    check_grad_fd(function(x) sum(w * scipy_fft_idct(x, n = 4L)^2), x0)
    check_grad_fd(function(x) sum(scipy_fft_dctn(x, norm = "ortho")^3), as_jax(x3 / 5, "float64"))
    check_grad_fd(function(x) sum(scipy_fft_idctn(x)^2 * w), x0)
    check_grad_fd(function(x) sum(jnp_fft_fftshift(x) * w), x0)
    # The complex input branch of the DCTs transposes lax_imag.
    check_grad_fd(function(x) sum(abs2(scipy_fft_dct(x + 1i * x * x, norm = "ortho")) * (1:4)), x0)
  })
})

test_that("vmap matches a stacked loop", {
  skip_if_no_backend()
  out <- vmap(function(v) jnp_fft_fft(v, n = 6L))(xr)
  loop <- do.call(rbind, lapply(1:2, function(i) as.array(jnp_fft_fft(xr[i, ], n = 6L))))
  expect_jax_equal(out, loop, shape = c(2L, 6L))
  x3t <- aperm(x3, c(2L, 1L, 3L))  # shape (3, 2, 4), batch along axis 1
  out <- vmap(function(v) jnp_fft_rfft2(v), in_axes = 1L)(x3t)
  loop <- lapply(1:2, function(i) as.array(jnp_fft_rfft2(x3t[, i, ])))
  expect_jax_equal(out, aperm(simplify2array(loop), c(3L, 1L, 2L)), shape = c(2L, 3L, 3L))
  out <- vmap(function(v) scipy_fft_dct(v, norm = "ortho"), in_axes = 1L)(xr)
  loop <- do.call(rbind, lapply(1:4, function(i) as.array(scipy_fft_dct(xr[, i], norm = "ortho"))))
  expect_jax_equal(out, loop, shape = c(4L, 2L))
  out <- vmap(function(v) scipy_fft_idctn(v))(x3)
  loop <- lapply(1:2, function(i) as.array(scipy_fft_idctn(x3[i, , ])))
  expect_jax_equal(out, aperm(simplify2array(loop), c(3L, 1L, 2L)))
  out <- vmap(function(v) jnp_fft_fftshift(v))(x3)
  expect_jax_equal(out, carr(c(0, 1, -2, -1, -1, 0, -3, -2, 3, -3, 1, 2, -2, -1, 3, -3, -3, -2, 2, 3, 1, 2, -1, 0), c(2L, 3L, 4L)))
})

test_that("numpy fft functions report jax errors", {
  skip_if_no_backend()
  expect_error(jnp_fft_fft(xr, norm = "bad"),
               "Invalid norm value bad; should be \"backward\",\"ortho\" or \"forward\".",
               fixed = TRUE)
  expect_error(jnp_fft_fft(xr, n = 0L), "Shape should be positive.", fixed = TRUE)
  expect_error(jnp_fft_fftn(xr, s = c(2L, 2L), axes = 0L), "Shape and axes have different lengths.",
               fixed = TRUE)
  expect_error(jnp_fft_fftn(xr, axes = c(0L, 0L)),
               "jax.numpy.fft.fftn does not support repeated axes. Got axes (0, 0).", fixed = TRUE)
  expect_error(jnp_fft_fft(xr, axis = c(0L, 1L)),
               "jax.numpy.fft.fft does not support multiple axes. Please use jax.numpy.fft.fftn. Got axis = (0, 1).",
               fixed = TRUE)
  expect_error(jnp_fft_fft2(xr, axes = 0L),
               "jax.numpy.fft.fft2 only supports 2 axes. Got axes = (0,).", fixed = TRUE)
  expect_error(jnp_fft_rfft(xc), "only real valued inputs supported for rfft", fixed = TRUE)
  expect_error(jnp_fft_fftfreq(c(1L, 2L)),
               "The n argument of jax.numpy.fft.fftfreq only takes an int. Got n = [1, 2].",
               fixed = TRUE)
  expect_error(jnp_fft_rfftfreq(4L, d = c(1, 2)),
               "The d argument of jax.numpy.fft.rfftfreq only takes a single value.", fixed = TRUE)
  expect_error(scipy_fft_dct(xr, type = 3L), "Only DCT type 2 is implemented.", fixed = TRUE)
  expect_error(scipy_fft_idct(xr, norm = "forward"),
               "jax.scipy.fft.idct: norm='forward' is not implemented", fixed = TRUE)
  expect_error(scipy_fft_dctn(xr, s = c(2L, 2L, 2L)),
               "s must have at most x.ndim (2) elements, got 3", fixed = TRUE)
  expect_error(scipy_fft_idctn(xr, axes = c(0L, -2L)), "repeated axis: (0, 0)", fixed = TRUE)
})
