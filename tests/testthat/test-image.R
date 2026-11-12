# jax.image: resize and scale_and_translate. Expected values from jax 0.12.0.1 (row-major,
# as JAX prints them), for x = (arange(20).reshape(4, 5) ** 1.5) / 7 in float32.

ref <- list(
  down_linear = c(1.4908929, 2.1675954, 2.9753978, 6.0343108, 7.2211246, 8.4765797),
  down_cubic = c(1.0818665, 1.7431922, 2.5577614, 6.3366156, 7.6214995, 8.977766),
  down_lanczos3 = c(0.90360051, 1.5577445, 2.3822627, 6.4208798, 7.7495956, 9.1525898),
  down_lanczos5 = c(0.82763952, 1.4692781, 2.2913272, 6.4769859, 7.8116102, 9.2223511),
  down_area = c(0.92764139, 1.5366653, 2.2949591, 6.7159762, 7.9815521, 9.3155174),
  up_linear = c(0, 0.081632651, 0.21748686, 0.40406102, 0.64566571, 0.91397166, 1.1428572, 1.1180341, 1.3434724, 1.6441778, 1.9732444, 2.3390942, 2.7243297, 3.0428574, 3.0573654, 3.3992712, 3.8375311, 4.2921057, 4.7722135, 5.2668161, 5.670229, 5.6520529, 6.0743914, 6.6110744, 7.1608992, 7.7317648, 8.3147955, 8.7877092, 8.2992506, 8.781312, 9.3915434, 10.013257, 10.653537, 11.304641, 11.831297),
  up_area = c(0, 0.11428572, 0.29957947, 0.53935963, 0.82241738, 1.1428572, 0.79859585, 1.0566872, 1.3634278, 1.709903, 2.0899184, 2.5000002, 1.5971915, 1.9990885, 2.4272759, 2.8804462, 3.357419, 3.8571429, 4.5175395, 5.0729795, 5.6478114, 6.2414856, 6.8534822, 7.483315, 6.4083958, 7.0235586, 7.656455, 8.3066502, 8.973731, 9.6573067, 8.2992506, 8.9741364, 9.6650972, 10.371814, 11.093978, 11.831297),
  noaa_cubic = c(0.27119666, 1.1067338, 3.3526134, 5.0607157, 8.4873486, 10.729759),
  noaa_cubicpt = c(0.2948145, 1.1760832, 3.2620668, 5.0365114, 8.355731, 10.662878),
  up_cubicpt_noaa = c(-0.14743398, -0.098687619, 0.033753023, 0.20946966, 0.4112674, 0.71937662, 0.93724084, 0.50083411, 0.66095144, 0.94526446, 1.2162148, 1.5074284, 1.9183539, 2.1991971, 2.0453382, 2.3509724, 2.8381436, 3.2421625, 3.6595166, 4.2179751, 4.5900493, 3.745914, 4.1080508, 4.6812449, 5.1510973, 5.6326771, 6.2699957, 6.6921091, 6.6202316, 7.0531602, 7.7334442, 8.2845497, 8.8456945, 9.5811491, 10.065722, 8.5600977, 9.0318193, 9.7709589, 10.366972, 10.972295, 11.762589, 12.282194),
  noaa_lanczos3 = c(0.21113622, 1.0605974, 3.1687605, 4.9499769, 8.5109234, 10.820602),
  nearest_down = c(0.14285715, 0.74230748, 5.2118392, 6.6960239, 9.1428576, 10.909648),
  nearest_up = c(0, 0, 1, 2, 2, 3, 3, 0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 9, 10, 10, 11, 11, 8, 8, 9, 10, 10, 11, 11),
  int_linear = c(0.30000001, 1.5, 2.7000003, 1.9000002, 3.1000001, 4.3000007, 4.3000002, 5.5, 6.7000003, 6.7000008, 7.9000006, 9.1000013, 8.3000002, 9.5, 10.700001),
  sat_linear = c(0.071428575, 0.39963543, 0.89296693, 0, 0.60451299, 1.0648608, 1.6646507, 0, 1.7891455, 2.5431399, 3.3795042, 0, 3.7587085, 4.6940827, 5.6967769, 0, 6.0215983, 7.1159897, 8.2676811, 0),
  sat_cubicpt = c(-0.14611939, 0.1296524, 0.63009679, 0.85362619, 0.39353329, 0.83728307, 1.4728099, 1.7411281, 1.6819197, 2.4554858, 3.3606699, 3.7190046, 3.4311082, 4.4077172, 5.4998345, 5.923336, 6.1994152, 7.3583779, 8.6308575, 9.118988),
  sat_lanczos3_dim1 = c(0.021788707, 0.33298543, 0.96737742, 1.7098467, 2.5096395, 3.5912757, 4.6758866, 5.7594948, 7.1505609, 8.4928112, 9.7999029, 11.442974),
  grad_cubic = c(0.20103006, 0.56749672, 1.2103231, 1.9638919, 2.842943, 3.5561059, 4.2806082, 5.5529761, 6.6436777, 8.1112213, 11.115704, 12.072494, 14.140006, 15.485971, 17.740232, 15.942232, 16.792389, 19.090927, 20.376028, 22.89325),
  grad_scale = c(-501.87534, -241.75301),
  batch_linear = c(0.76341474, 1.5098619, 3.6694388, 4.9766421, 7.973855, 9.6552725, 2.5268295, 4.0197239, 8.3388786, 10.953285, 16.94771, 20.310545, -0.76341474, -1.5098619, -3.6694388, -4.9766421, -7.973855, -9.6552725),
  x64_cubic = c(0.2148033500545089, 0.3298092815240151, 0.5308618172194337, 0.7697013762998707, 1.042055907228806, 1.382222791439334, 1.632235015678473, 2.945182355564048, 3.276771002406427, 3.765845362762402, 4.227578162077308, 4.707019031639693, 5.260863715533934, 5.654029477079749, 7.48407535692895, 7.933060213900716, 8.588095248317027, 9.194583021380557, 9.81467110940752, 10.51980683161603, 11.01648149468259),
  x64_cubicpt_noaa = c(-0.116633205247959, 0.2094698465323685, 0.8048604868030845, 0.5992244286845426, 1.216214733136475, 2.028275906077894, 2.232121912605707, 3.242164145963097, 4.363329265126442, 3.967149125738252, 5.151098502344476, 6.43482513716128, 6.884618410335289, 8.284549646817212, 9.770292542463167, 8.848134862165626, 10.36697150266381, 11.96537447792657),
  end = NULL
)

# Row-major values of an array.
rm_vals <- function(a) {
  a <- as.array(a)
  if (length(dim(a)) > 1L) as.vector(aperm(a, length(dim(a)):1)) else as.vector(a)
}
expect_rm <- function(object, expected, shape, dtype = "float32", tolerance = 2e-6) {
  expect_identical(jaxR::dtype(object), dtype)
  expect_identical(as.integer(dim(object)), as.integer(shape))
  expect_equal(rm_vals(object), expected, tolerance = tolerance)
}
img_x <- function(dt = "float32") as_jax(matrix((0:19)^1.5 / 7, 4, byrow = TRUE), dt)
has_take <- function() is.function(get0("jnp_take", envir = asNamespace("jaxR"), inherits = FALSE))

test_that("image_resize matches jax for every method", {
  skip_if_no_backend()
  x <- img_x()
  expect_rm(image_resize(x, c(2L, 3L), "linear"), ref$down_linear, c(2, 3))
  expect_rm(image_resize(x, c(2L, 3L), "bilinear"), ref$down_linear, c(2, 3))
  expect_rm(image_resize(x, c(2L, 3L), image_ResizeMethod$CUBIC), ref$down_cubic, c(2, 3))
  expect_rm(image_resize(x, c(2L, 3L), "lanczos3"), ref$down_lanczos3, c(2, 3))
  expect_rm(image_resize(x, c(2L, 3L), "lanczos5"), ref$down_lanczos5, c(2, 3))
  expect_rm(image_resize(x, c(2L, 3L), "area"), ref$down_area, c(2, 3))
  expect_rm(image_resize(x, c(5L, 7L), "triangle"), ref$up_linear, c(5, 7))
  expect_rm(image_resize(x, c(6L, 6L), "area"), ref$up_area, c(6, 6))
  expect_rm(image_resize(x, c(3L, 2L), "bicubic", antialias = FALSE), ref$noaa_cubic, c(3, 2))
  expect_rm(image_resize(x, c(3L, 2L), "cubic-pytorch", antialias = FALSE), ref$noaa_cubicpt, c(3, 2))
  expect_rm(image_resize(x, c(6L, 7L), "bicubic-pytorch", antialias = FALSE), ref$up_cubicpt_noaa, c(6, 7))
  expect_rm(image_resize(x, c(3L, 2L), "lanczos3", antialias = FALSE), ref$noaa_lanczos3, c(3, 2))
  # cubic-pytorch with antialias is the Keys cubic
  expect_rm(image_resize(x, c(2L, 3L), "cubic-pytorch"), ref$down_cubic, c(2, 3))
  # integer images are promoted to float; identity resize returns the (promoted) input
  expect_rm(image_resize(as_jax(matrix(0:11, 3, byrow = TRUE)), c(5L, 3L), "linear"), ref$int_linear, c(5, 3))
  expect_rm(image_resize(x, c(4L, 5L), "cubic"), rm_vals(x), c(4, 5))
})

test_that("image_resize under jit, vmap and with float64", {
  skip_if_no_backend()
  x <- img_x()
  xb <- jnp_stack(list(x, x * 2 + 1, -x))
  expect_rm(jit(function(a) image_resize(a, c(2L, 3L), "lanczos3"))(x), ref$down_lanczos3, c(2, 3))
  expect_rm(image_resize(xb, c(3L, 3L, 2L), "linear"), ref$batch_linear, c(3, 3, 2))
  expect_rm(vmap(function(a) image_resize(a, c(3L, 2L), "linear"))(xb), ref$batch_linear, c(3, 3, 2))
  bl <- aperm(array(ref$batch_linear, c(2L, 3L, 3L)), 3:1)
  expect_rm(jit(vmap(function(a) image_resize(a, c(3L, 2L), "linear"), in_axes = 2L, out_axes = 2L))(
    jnp_moveaxis(xb, 0L, 2L)), rm_vals(aperm(bl, c(2L, 3L, 1L))), c(3, 2, 3))
  with_jax_config("jax_enable_x64", TRUE, {
    x64 <- img_x("float64")
    expect_rm(image_resize(x64, c(3L, 7L), "cubic"), ref$x64_cubic, c(3, 7), "float64", 1e-13)
    expect_rm(image_resize(x64, c(6L, 3L), "cubic-pytorch", antialias = FALSE), ref$x64_cubicpt_noaa,
              c(6, 3), "float64", 1e-13)
  })
})

test_that("image_resize nearest keeps the dtype", {
  skip_if_no_backend()
  skip_if_not(has_take(), "jnp_take (R/numpy_indexing.R) is not in this build")
  expect_rm(image_resize(img_x(), c(3L, 2L), "nearest"), ref$nearest_down, c(3, 2))
  expect_rm(image_resize(as_jax(matrix(0:11, 3, byrow = TRUE)), c(5L, 7L), image_ResizeMethod$NEAREST),
            ref$nearest_up, c(5, 7), "int32", 0)
  expect_rm(jit(function(a) image_resize(a, c(3L, 2L), "nearest"))(img_x()), ref$nearest_down, c(3, 2))
})

test_that("image_scale_and_translate matches jax", {
  skip_if_no_backend()
  x <- img_x()
  s <- as_jax(c(1.5, 0.7))
  t0 <- as_jax(c(0.3, -0.2))
  expect_rm(image_scale_and_translate(x, c(5L, 4L), c(0L, 1L), s, t0, "linear"), ref$sat_linear, c(5, 4))
  expect_rm(jit(function(a, s, t) image_scale_and_translate(a, c(5L, 4L), c(0L, -1L), s, t, "linear"))(x, s, t0),
            ref$sat_linear, c(5, 4))
  expect_rm(image_scale_and_translate(x, c(5L, 4L), c(0L, 1L), s, t0, "cubic-pytorch", antialias = FALSE),
            ref$sat_cubicpt, c(5, 4))
  expect_rm(image_scale_and_translate(x, c(4L, 3L), 1L, as_jax(0.6), as_jax(0.1), "lanczos3"),
            ref$sat_lanczos3_dim1, c(4, 3))
  expect_rm(image_scale_and_translate(x, c(4L, 3L), 1L, 0.6, 0.1, "lanczos3"), ref$sat_lanczos3_dim1, c(4, 3))
})

test_that("gradients of resize and scale_and_translate", {
  skip_if_no_backend()
  x <- img_x()
  expect_rm(grad(function(a) jnp_sum(image_resize(a, c(3L, 7L), "cubic")^2))(x), ref$grad_cubic, c(4, 5),
            tolerance = 1e-5)
  expect_rm(grad(function(s) jnp_sum(image_scale_and_translate(x, c(5L, 4L), c(0L, 1L), s,
                                                                 as_jax(c(0.3, -0.2)), "linear")^2))(
    as_jax(c(1.5, 0.7))), ref$grad_scale, 2, tolerance = 1e-5)
  with_jax_config("jax_enable_x64", TRUE, {
    x64 <- img_x("float64")
    check_grad_fd(function(a) jnp_sum(jnp_sin(image_resize(a, c(3L, 7L), "lanczos3"))), x64)
    check_grad_fd(function(a) jnp_sum(image_resize(a, c(6L, 3L), "cubic-pytorch", antialias = FALSE)^2), x64)
    check_grad_fd(function(t) jnp_sum(image_scale_and_translate(x64, c(5L, 4L), c(0L, 1L),
                                                                 as_jax(c(1.5, 0.7), "float64"), t, "cubic")^2),
                  as_jax(c(0.3, -0.2), "float64"))
    g2 <- jacfwd(grad(function(a) jnp_sum(image_resize(a, c(2L, 3L), "linear")^2)))(x64)
    expect_identical(as.integer(dim(g2)), c(4L, 5L, 4L, 5L))
  })
})

test_that("image errors", {
  skip_if_no_backend()
  x <- img_x()
  expect_error(image_resize(x, c(2L, 3L, 1L), "linear"),
               "shape must have length equal to the number of dimensions of x;  (2, 3, 1) vs (4, 5)",
               fixed = TRUE)
  expect_error(image_resize(x, c(2L, 3L), "gaussian"), 'Unknown resize method "gaussian"', fixed = TRUE)
  expect_error(image_scale_and_translate(x, c(2L, 3L), 0:1, c(1, 1), c(0, 0), "nearest"),
               "Nearest neighbor resampling is not currently supported for scale_and_translate.",
               fixed = TRUE)
  expect_identical(image_ResizeMethod$LANCZOS5, "lanczos5")
})
