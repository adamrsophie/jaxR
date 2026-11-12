# jax.numpy ufuncs of R/numpy_ufuncs.R and the jnp_ufunc API. Expected values from jax
# 0.12.0.1 (float32 values computed in float32, float64 values with x64 enabled).

ufn_xs <- c(-2.5, -1.0, -0.3, 0.0, 0.4, 1.0, 2.7, 9.5)
ufn_ys <- c(1.5, -2.0, 0.7, -0.0, 3.0, -1.0, 0.5, 2.0)
ufn_ns <- c(3L, -2L, 1L, 5L, -1L, 0L, 2L, -3L)
ufn_xa <- c(-0.9, -0.5, 0.0, 0.25, 0.6, 0.99)
ufn_xc <- c(1.0, 1.5, 3.0, 10.0)

# Row-major comparison helper: `v` is the flattened NumPy result.
expect_rm <- function(object, v, shape = NULL, dtype = NULL, tolerance = NULL) {
  if (!is.null(shape)) expect_identical(as.integer(dim(object)), as.integer(shape))
  expect_jax_equal(jnp_ravel(object), v, dtype = dtype, tolerance = tolerance)
}

ufn_cases <- function(X, Y, XA, XC, N) {
  list(fabs = jnp_fabs(X), spacing = jnp_spacing(X), deg2rad = jnp_deg2rad(X),
       rad2deg = jnp_rad2deg(X), degrees = jnp_degrees(X), radians = jnp_radians(X),
       sinc = jnp_sinc(X), asinh = jnp_asinh(X), atan = jnp_atan(X), asin = jnp_asin(XA),
       acos = jnp_acos(XA), atanh = jnp_atanh(XA), acosh = jnp_acosh(XC), signbit = jnp_signbit(Y),
       modf_f = jnp_modf(X)[[1L]], modf_w = jnp_modf(X)[[2L]], frexp_m = jnp_frexp(X)[[1L]],
       frexp_e = jnp_frexp(X)[[2L]], copysign = jnp_copysign(X, Y), nextafter = jnp_nextafter(X, Y),
       float_power = jnp_float_power(jnp_abs(X), Y), logaddexp = jnp_logaddexp(X, Y),
       logaddexp2 = jnp_logaddexp2(X, Y), hypot = jnp_hypot(X, Y), heaviside = jnp_heaviside(X, Y),
       fmod = jnp_fmod(X, Y), atan2 = jnp_atan2(X, Y), divmod_q = jnp_divmod(X, Y)[[1L]],
       divmod_r = jnp_divmod(X, Y)[[2L]], ldexp = jnp_ldexp(X, N))
}

ufn_expected <- list(
  float32 = list(
    fabs = c(2.5, 1.0, 0.30000001192092896, 0.0, 0.4000000059604645, 1.0, 2.700000047683716, 9.5),
    spacing = c(-2.384185791015625e-07, -1.1920928955078125e-07, -2.9802322387695312e-08,
                1.401298464324817e-45, 2.9802322387695312e-08, 1.1920928955078125e-07,
                2.384185791015625e-07, 9.5367431640625e-07),
    deg2rad = c(-0.04363323003053665, -0.01745329238474369, -0.005235987715423107, 0.0,
                0.006981316953897476, 0.01745329238474369, 0.04712389037013054, 0.16580627858638763),
    rad2deg = c(-143.2394561767578, -57.295780181884766, -17.18873405456543, 0.0, 22.918312072753906,
                57.295780181884766, 154.6986083984375, 544.3099365234375),
    sinc = c(0.1273239552974701, 0.0, 0.858393669128418, 1.0, 0.7568266987800598, 0.0,
             0.09537706524133682, -0.033506304025650024),
    asinh = c(-1.647231101989746, -0.8813736438751221, -0.29567307233810425, 0.0, 0.3900352716445923,
              0.8813736438751221, 1.7190518379211426, 2.947197675704956),
    atan = c(-1.1902899742126465, -0.7853981256484985, -0.29145678877830505, 0.0, 0.38050636649131775,
             0.7853981256484985, 1.2160906791687012, 1.4659193754196167),
    asin = c(-1.1197694540023804, -0.5235987901687622, 0.0, 0.252680242061615, 0.6435011029243469,
             1.4292569160461426),
    acos = c(2.690565824508667, 2.094395160675049, 1.5707963705062866, 1.3181160688400269,
             0.9272952079772949, 0.14153940975666046),
    atanh = c(-1.4722193479537964, -0.5493061542510986, 0.0, 0.2554128170013428, 0.6931471824645996,
              2.646652936935425),
    acosh = c(0.0, 0.9624236226081848, 1.7627471685409546, 2.993222951889038),
    signbit = c(FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE),
    modf_f = c(-0.5, 0.0, -0.30000001192092896, 0.0, 0.4000000059604645, 0.0, 0.7000000476837158, 0.5),
    modf_w = c(-2.0, -1.0, -0.0, 0.0, 0.0, 1.0, 2.0, 9.0),
    frexp_m = c(-0.625, -0.5, -0.6000000238418579, 0.0, 0.800000011920929, 0.5, 0.675000011920929,
                0.59375),
    frexp_e = c(2, 1, -1, 0, -1, 1, 2, 4),
    copysign = c(2.5, -1.0, 0.30000001192092896, -0.0, 0.4000000059604645, -1.0, 2.700000047683716, 9.5),
    nextafter = c(-2.499999761581421, -1.0000001192092896, -0.29999998211860657, -0.0,
                  0.40000003576278687, 0.9999999403953552, 2.6999998092651367, 9.499999046325684),
    float_power = c(3.9528470039367676, 1.0, 0.4305116534233093, 1.0, 0.06400000303983688, 1.0,
                    1.6431677341461182, 90.25),
    logaddexp = c(1.518149971961975, -0.6867383122444153, 1.0132616758346558, 0.6931471824645996,
                  3.0716447830200195, 1.1269279718399048, 2.8050832748413086, 9.500553131103516),
    logaddexp2 = c(1.5874627828598022, -0.41503751277923584, 1.2849624156951904, 1.0,
                   3.2202537059783936, 1.3219281435012817, 2.9840848445892334, 9.50794792175293),
    hypot = c(2.915475845336914, 2.2360680103302, 0.761577308177948, 0.0, 3.0265491008758545,
              1.4142135381698608, 2.745906114578247, 9.708244323730469),
    heaviside = c(0.0, 0.0, 0.0, -0.0, 1.0, 1.0, 1.0, 1.0),
    fmod = c(-1.0, -1.0, -0.30000001192092896, NaN, 0.4000000059604645, 0.0, 0.20000004768371582, 1.5),
    atan2 = c(-1.0303767919540405, -2.677945137023926, -0.40489181876182556, 3.141592502593994,
              0.13255153596401215, 2.356194496154785, 1.3876855373382568, 1.363300085067749),
    divmod_q = c(-2.0, -0.0, -1.0, NaN, 0.0, -1.0, 5.0, 4.0),
    divmod_r = c(0.5, -1.0, 0.3999999761581421, NaN, 0.4000000059604645, 0.0, 0.20000004768371582, 1.5),
    ldexp = c(-20.0, -0.25, -0.6000000238418579, 0.0, 0.20000000298023224, 1.0, 10.800000190734863,
              1.1875)),
  float64 = list(
    fabs = c(2.5, 1.0, 0.3, 0.0, 0.4, 1.0, 2.7, 9.5),
    spacing = c(-0x1.0000000000000p-51, -0x1.0000000000000p-52, -0x1.0000000000000p-54,
                0x0.0000000000001p-1022, 0x1.0000000000000p-54, 0x1.0000000000000p-52,
                0x1.0000000000000p-51, 0x1.0000000000000p-49),
    deg2rad = c(-0.04363323129985824, -0.017453292519943295, -0.005235987755982988, 0.0,
                0.006981317007977318, 0.017453292519943295, 0.0471238898038469, 0.16580627893946132),
    rad2deg = c(-143.2394487827058, -57.29577951308232, -17.188733853924695, 0.0, 22.918311805232932,
                57.29577951308232, 154.6986046853223, 544.309905374282),
    sinc = c(0.12732395447351624, 0.0, 0.8583936913341398, 1.0, 0.7568267286406569, 0.0,
             0.09537707681490437, -0.03350630380882007),
    asinh = c(-1.6472311463710958, -0.8813735870195429, -0.29567304756342244, 0.0, 0.3900353197707153,
              0.8813735870195429, 1.7190518512089272, 2.9471976225700334),
    atan = c(-1.1902899496825317, -0.7853981633974484, -0.2914567944778671, 0.0, 0.3805063771123649,
             0.7853981633974484, 1.2160906747839562, 1.4659193880646626),
    asin = c(-1.1197695149986342, -0.5235987755982989, 0.0, 0.25268025514207865, 0.6435011087932844,
             1.4292568534704693),
    acos = c(2.6905658417935308, 2.0943951023931957, 1.5707963267948966, 1.318116071652818,
             0.9272952180016123, 0.1415394733244273),
    atanh = c(-1.4722194895832204, -0.5493061443340548, 0.0, 0.25541281188299536, 0.6931471805599453,
              2.646652412362246),
    acosh = c(0.0, 0.9624236501192069, 1.762747174039086, 2.993222846126381),
    signbit = c(FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE),
    modf_f = c(-0.5, 0.0, -0.3, 0.0, 0.4, 0.0, 0.7000000000000002, 0.5),
    modf_w = c(-2.0, -1.0, -0.0, 0.0, 0.0, 1.0, 2.0, 9.0),
    frexp_m = c(-0x1.4000000000000p-1, -0x1.0000000000000p-1, -0x1.3333333333333p-1, 0x0.0p+0,
                0x1.999999999999ap-1, 0x1.0000000000000p-1, 0x1.599999999999ap-1, 0x1.3000000000000p-1),
    frexp_e = c(2, 1, -1, 0, -1, 1, 2, 4),
    copysign = c(0x1.4000000000000p+1, -0x1.0000000000000p+0, 0x1.3333333333333p-2, -0x0.0p+0,
                 0x1.999999999999ap-2, -0x1.0000000000000p+0, 0x1.599999999999ap+1, 0x1.3000000000000p+3),
    nextafter = c(-0x1.3ffffffffffffp+1, -0x1.0000000000001p+0, -0x1.3333333333332p-2, -0x0.0p+0,
                  0x1.999999999999bp-2, 0x1.fffffffffffffp-1, 0x1.5999999999999p+1, 0x1.2ffffffffffffp+3),
    float_power = c(3.952847075210474, 1.0, 0.4305116202499342, 1.0, 0.06400000000000002, 1.0,
                    1.6431676725154984, 90.25),
    logaddexp = c(1.5181499279178097, -0.6867383124817772, 1.0132616875182228, 0.6931471805599453,
                  3.07164469196767, 1.1269280110429725, 2.805083319768696, 9.500552931475362),
    logaddexp2 = c(1.5874628412503393, -0.4150374992788438, 1.284962500721156, 1.0, 3.220253779618794,
                   1.3219280948873624, 2.9840848626653336, 9.5079479063862),
    hypot = c(2.9154759474226504, 2.23606797749979, 0.7615773105863908, 0.0, 3.026549190084311,
              1.4142135623730951, 2.7459060435491964, 9.7082439194738),
    heaviside = c(0.0, 0.0, 0.0, -0.0, 1.0, 1.0, 1.0, 1.0),
    fmod = c(-1.0, -1.0, -0.3, NaN, 0.4, 0.0, 0.20000000000000018, 1.5),
    atan2 = c(-1.0303768265243125, -2.677945044588987, -0.40489178628508343, 3.141592653589793,
              0.13255153229667402, 2.356194490192345, 1.3876855095324125, 1.3633001003596938),
    divmod_q = c(-2.0, -0.0, -1.0, NaN, 0.0, -1.0, 5.0, 4.0),
    divmod_r = c(0.5, -1.0, 0.39999999999999997, NaN, 0.4, 0.0, 0.20000000000000018, 1.5),
    ldexp = c(-0x1.4000000000000p+4, -0x1.0000000000000p-2, -0x1.3333333333333p-1, 0x0.0p+0,
              0x1.999999999999ap-3, 0x1.0000000000000p+0, 0x1.599999999999ap+3, 0x1.3000000000000p+0)))

ufn_bit_exact <- c("spacing", "frexp_m", "copysign", "nextafter", "ldexp", "frexp_e", "modf_w",
                   "divmod_q", "heaviside")

test_that("elementwise ufuncs match JAX in float32 and float64, eagerly and under jit", {
  skip_if_no_backend()
  run <- function(dt) {
    args <- list(as_jax(ufn_xs, dt), as_jax(ufn_ys, dt), as_jax(ufn_xa, dt), as_jax(ufn_xc, dt),
                 as_jax(ufn_ns, "int32"))
    eager <- do.call(ufn_cases, args)
    jitted <- do.call(jit(ufn_cases), args)
    e <- ufn_expected[[dt]]
    for (nm in names(e)) {
      out_dt <- if (nm == "signbit") "bool" else if (nm == "frexp_e") "int32" else dt
      expect_jax_equal(eager[[nm]], e[[nm]], dtype = out_dt, shape = length(e[[nm]]))
      expect_jax_equal(jitted[[nm]], e[[nm]], dtype = out_dt)
      if (dt == "float64" && nm %in% ufn_bit_exact)
        expect_identical(as.vector(as.array(eager[[nm]])), as.numeric(e[[nm]]), label = nm)
    }
    # signed zeros
    expect_identical(1 / as.vector(as.array(eager$copysign))[4L], -Inf)
    expect_identical(1 / as.vector(as.array(eager$modf_w))[3L], -Inf)
    expect_identical(1 / as.vector(as.array(eager$heaviside))[4L], -Inf)
    expect_identical(1 / as.vector(as.array(eager$divmod_q))[2L], -Inf)
  }
  run("float32")
  enable_x64()
  on.exit(enable_x64(FALSE))
  run("float64")
})

test_that("dtypes and weak types follow JAX promotion", {
  skip_if_no_backend()
  f32 <- as_jax(3, "float32")
  i32 <- as_jax(3L, "int32")
  chk <- function(x, dt, weak) {
    expect_identical(jaxR:::aval_of(x)$dtype, dt)
    expect_identical(jaxR:::aval_of(x)$weak, weak)
  }
  chk(jnp_fabs(2), "float32", TRUE); chk(jnp_fabs(3L), "float32", TRUE)
  chk(jnp_bitwise_invert(3L), "int32", TRUE)
  chk(jnp_spacing(2), "float32", FALSE); chk(jnp_spacing(3L), "float32", FALSE)
  chk(jnp_acos(0.5), "float32", TRUE); chk(jnp_acosh(2), "float32", TRUE)
  chk(jnp_asin(0.5), "float32", TRUE); chk(jnp_asinh(2), "float32", TRUE)
  chk(jnp_atan(2), "float32", TRUE); chk(jnp_atanh(0.5), "float32", TRUE)
  chk(jnp_bitwise_count(3L), "uint8", FALSE); chk(jnp_signbit(2), "bool", FALSE)
  fr <- jnp_frexp(2)
  chk(fr[[1L]], "float32", FALSE); chk(fr[[2L]], "int32", FALSE)
  chk(jnp_deg2rad(2), "float32", TRUE); chk(jnp_rad2deg(3L), "float32", TRUE)
  chk(jnp_degrees(2), "float32", TRUE); chk(jnp_radians(3L), "float32", TRUE)
  mf <- jnp_modf(2)
  chk(mf[[1L]], "float32", TRUE); chk(mf[[2L]], "float32", TRUE)
  chk(jnp_isposinf(2), "bool", FALSE); chk(jnp_isneginf(3L), "bool", FALSE)
  chk(jnp_sinc(2), "float32", TRUE); chk(jnp_sinc(3L), "float32", TRUE)
  chk(jnp_left_shift(3L, 2L), "int32", TRUE); chk(jnp_left_shift(i32, 2L), "int32", FALSE)
  chk(jnp_right_shift(3L, 2L), "int32", TRUE); chk(jnp_bitwise_right_shift(i32, 2L), "int32", FALSE)
  chk(jnp_float_power(3L, 2L), "float32", TRUE); chk(jnp_float_power(f32, 2L), "float32", FALSE)
  chk(jnp_float_power(i32, 2L), "float32", FALSE)
  chk(jnp_nextafter(3L, 2L), "float32", TRUE); chk(jnp_nextafter(i32, 2L), "float32", FALSE)
  chk(jnp_atan2(3L, 2L), "float32", TRUE); chk(jnp_atan2(i32, 2L), "float32", FALSE)
  chk(jnp_pow(3L, 2L), "int32", TRUE); chk(jnp_pow(3, 2L), "float32", TRUE)
  chk(jnp_pow(i32, 2L), "int32", FALSE)
  chk(jnp_copysign(3L, 2L), "float32", TRUE); chk(jnp_copysign(i32, 2L), "float32", FALSE)
  dm <- jnp_divmod(3L, 2L)
  chk(dm[[1L]], "int32", TRUE); chk(dm[[2L]], "int32", TRUE)
  dm <- jnp_divmod(3, 2L)
  chk(dm[[1L]], "float32", TRUE); chk(dm[[2L]], "float32", TRUE)
  dm <- jnp_divmod(f32, 2L)
  chk(dm[[1L]], "float32", FALSE); chk(dm[[2L]], "float32", FALSE)
  chk(jnp_logaddexp(3L, 2L), "float32", TRUE); chk(jnp_logaddexp(i32, 2L), "float32", FALSE)
  chk(jnp_logaddexp2(3L, 2L), "float32", TRUE); chk(jnp_logaddexp2(i32, 2L), "float32", FALSE)
  chk(jnp_ldexp(3L, 2L), "float32", FALSE); chk(jnp_ldexp(3, 2L), "float32", FALSE)
  chk(jnp_fmod(3L, 2L), "int32", TRUE); chk(jnp_fmod(3, 2L), "float32", TRUE)
  chk(jnp_fmod(i32, 2L), "int32", FALSE)
  chk(jnp_heaviside(3, 2L), "float32", TRUE); chk(jnp_heaviside(i32, 2L), "float32", FALSE)
  chk(jnp_hypot(3L, 2L), "float32", TRUE); chk(jnp_hypot(f32, 2L), "float32", FALSE)
  expect_jax_equal(jnp_hypot(3L, 2L), 3.605551242828369)
  expect_jax_equal(jnp_logaddexp2(3L, 2L), 3.5849623680114746)
  expect_jax_equal(jnp_spacing(3L), 2.384185791015625e-07)
  enable_x64()
  on.exit(enable_x64(FALSE))
  chk(jnp_sinc(2), "float64", TRUE); chk(jnp_frexp(2)[[2L]], "int32", FALSE)
  chk(jnp_ldexp(as_jax(2, "float32"), 3L), "float32", FALSE)
  chk(jnp_deg2rad(as_jax(2, "float32")), "float32", FALSE)
})

test_that("integer ufuncs match JAX", {
  skip_if_no_backend()
  a <- as_jax(c(5L, -7L, 12L, -1L, 0L, 255L), "int32")
  b <- as_jax(c(1L, 2L, 3L, 4L, 0L, 31L), "int32")
  expect_jax_equal(jnp_left_shift(a, b), c(10, -28, 96, -16, 0, -2147483648), dtype = "int32")
  expect_jax_equal(jnp_bitwise_left_shift(a, b), c(10, -28, 96, -16, 0, -2147483648))
  expect_jax_equal(jnp_right_shift(a, b), c(2, -2, 1, -1, 0, 0), dtype = "int32")
  expect_jax_equal(jnp_bitwise_right_shift(a, b), c(2, -2, 1, -1, 0, 0))
  au <- as_jax(c(5, 4294967289, 12, 1, 0, 255), "uint32")
  expect_jax_equal(jnp_right_shift(au, as_jax(c(1, 2, 3, 4, 0, 31), "uint32")),
                   c(2, 1073741822, 1, 0, 0, 0), dtype = "uint32")
  expect_jax_equal(jnp_bitwise_count(a), c(2, 3, 2, 1, 0, 8), dtype = "uint8")
  expect_jax_equal(jnp_bitwise_count(as_jax(c(0, 255, 7), "uint8")), c(0, 8, 3), dtype = "uint8")
  expect_jax_equal(jnp_bitwise_invert(a), c(-6, 6, -13, 0, -1, -256), dtype = "int32")
  expect_jax_equal(jnp_fmod(a, as_jax(c(3L, 2L, -5L, 4L, 0L, 7L))), c(2, -1, 2, -1, 0, 3), dtype = "int32")
  d <- as_jax(c(3L, 2L, -5L, 4L, 1L, 7L))
  dm <- jnp_divmod(a, d)
  expect_jax_equal(dm[[1L]], c(1, -4, -3, -1, 0, 36), dtype = "int32")
  expect_jax_equal(dm[[2L]], c(2, 1, -3, 3, 0, 3), dtype = "int32")
  expect_jax_equal(jnp_signbit(a), c(FALSE, TRUE, FALSE, TRUE, FALSE, FALSE), dtype = "bool")
  expect_jax_equal(jnp_signbit(c(TRUE, FALSE)), c(FALSE, FALSE), dtype = "bool")
  s <- c(Inf, -Inf, NaN, 1, -0)
  expect_jax_equal(jnp_isposinf(s), c(TRUE, FALSE, FALSE, FALSE, FALSE), dtype = "bool")
  expect_jax_equal(jnp_isneginf(s), c(FALSE, TRUE, FALSE, FALSE, FALSE), dtype = "bool")
  expect_jax_equal(jnp_isposinf(a), rep(FALSE, 6), dtype = "bool")
  expect_jax_equal(jnp_fabs(c(-5L, 2L)), c(5, 2), dtype = "float32")
  expect_jax_equal(jnp_fabs(c(TRUE, FALSE)), c(1, 0), dtype = "float32")
  enable_x64()
  on.exit(enable_x64(FALSE))
  expect_jax_equal(jnp_bitwise_count(as_jax(c(-1, 2^40, 2^40 - 1), "int64")), c(1, 1, 40), dtype = "uint8")
  expect_jax_equal(jnp_left_shift(as_jax(1, "int64"), 40L), 2^40, dtype = "int64")
})

test_that("special values, subnormals and other dtypes match JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  same <- function(x, e) expect_identical(as.vector(as.array(x)), e)
  # Smallest subnormal, smallest normal and largest finite double as hex literals.
  sp <- c(0, -0, Inf, -Inf, NaN, 0x0.0000000000001p-1022, 0x1p-1022, 0x1.fffffffffffffp+1023)
  fr <- jnp_frexp(sp)
  same(fr[[1L]], c(0x0.0p+0, -0x0.0p+0, Inf, -Inf, NaN, 0x0.0p+0, 0x1.0000000000000p-1,
                   0x1.fffffffffffffp-1))
  expect_jax_equal(fr[[2L]], c(0, 0, 0, 0, 0, 0, -1021, 1024), dtype = "int32")
  same(jnp_ldexp(sp, c(1L, 2L, 3L, -4L, 5L, 1L, -1L, 1L)),
       c(0x0.0p+0, -0x0.0p+0, Inf, -Inf, NaN, 0x0.0p+0, 0x0.0p+0, Inf))
  same(jnp_ldexp(c(1, 1, 1.5, -3, 1), c(1023L, 1024L, -1074L, -1080L, 2000L)),
       c(0x1.0000000000000p+1023, Inf, 0x0.0p+0, -0x0.0p+0, Inf))
  expect_identical(1 / as.vector(as.array(jnp_ldexp(-3, -1080L))), -Inf)
  same(jnp_spacing(sp), c(0x0.0000000000001p-1022, -0x0.0000000000001p-1022, NaN, NaN, NaN,
                          0x0.0000000000001p-1022, 0x0.0000000000001p-1022, Inf))
  expect_jax_equal(jnp_sinc(c(Inf, -Inf, NaN, 1e-8, 0.5, 1.5, -3.0, 1e6 + 0.5)),
                   c(0.0, 0.0, NaN, 0.9999999999999999, 0.6366197723675813, -0.21220659078919377, 0.0,
                     3.1830972702892714e-07))
  x1 <- c(Inf, -Inf, Inf, NaN, 1e300, -800)
  x2 <- c(Inf, -Inf, 1, 1, 1e300, -801)
  expect_jax_equal(jnp_logaddexp(x1, x2), c(Inf, -Inf, Inf, NaN, 1e300, -799.6867383124818))
  expect_jax_equal(jnp_logaddexp2(x1, x2), c(Inf, -Inf, Inf, NaN, 1e300, -799.4150374992788))
  expect_jax_equal(jnp_hypot(c(Inf, NaN, 1e300, 1e-300, 3e-320, 0), c(NaN, Inf, 1e300, 1e-300, 4e-320, 0)),
                   c(Inf, Inf, 1.4142135623730952e+300, 1.414213562373095e-300, 0.0, 0.0))
  expect_jax_equal(jnp_heaviside(c(NaN, 0, -0), c(0.5, 0.5, 0.25)), c(NaN, 0.5, 0.25))
  expect_jax_equal(jnp_spacing(as_jax(c(1, -1, 0), "float16")),
                   c(0.0009765625, 0.00048828125, 5.960464477539063e-08), dtype = "float16")
  expect_jax_equal(jnp_sinc(as_jax(c(0.3, 1.5), "float16")), c(0.8583984375, -0.212158203125),
                   dtype = "float16", tolerance = 1e-3)
  fr16 <- jnp_frexp(as_jax(c(3, -0.1), "float16"))
  expect_jax_equal(fr16[[1L]], c(0.75, -0.7998046875), dtype = "float16")
  expect_jax_equal(fr16[[2L]], c(2, -3), dtype = "int32")
  same(jnp_ldexp(c(3, 3), as_jax(c(-3, 100), "int8")), c(0x1.8000000000000p-2, 0x1.8000000000000p+101))
  same(jnp_ldexp(c(3, 3), c(TRUE, FALSE)), c(0x1.8000000000000p+2, 0x1.8000000000000p+1))
  same(jnp_ldexp(c(3, 3), as_jax(c(3, 1), "uint32")), c(0x1.8000000000000p+4, 0x1.8000000000000p+2))
  sp32 <- as_jax(c(0, -0, Inf, NaN, 1e-45, 1.1754944e-38, 3.4028235e38), "float32")
  fr <- jnp_frexp(sp32)
  same(fr[[1L]], c(0x0.0p+0, -0x0.0p+0, Inf, NaN, 0x0.0p+0, 0x1.0000000000000p-1, 0x1.fffffe0000000p-1))
  expect_jax_equal(fr[[2L]], c(0, 0, 0, 0, 0, -125, 128), dtype = "int32")
  same(jnp_ldexp(sp32, c(1L, 2L, 3L, 5L, 1L, -1L, 1L)),
       c(0x0.0p+0, -0x0.0p+0, Inf, NaN, 0x0.0p+0, 0x0.0p+0, Inf))
  same(jnp_spacing(sp32)[c(2L, 3L, 6L)], c(NaN, NaN, Inf))
})

test_that("complex sinc, logaddexp and logaddexp2 match JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  z <- c(0.3 + 0.2i, 1.5 - 0.5i, 0i, 2 + 0i, complex(real = Inf, imaginary = 1))
  r <- jnp_sinc(z)
  expect_jax_equal(jnp_real(r), c(0.9084815996554283, -0.47921778957387784, 1.0, -3.8981718325193755e-17, 0.0),
                   dtype = "float64", tolerance = 1e-10)
  expect_jax_equal(jnp_imag(r), c(-0.187500669562326, -0.1597392631912925, 0.0, 0.0, 0.0), tolerance = 1e-10)
  r <- jnp_sinc(as_jax(z, "complex64"))
  expect_jax_equal(jnp_real(r), c(0.9084815979003906, -0.4792178273200989, 1.0, 2.7827534054836178e-08, 0.0),
                   dtype = "float32", tolerance = 1e-6)
  expect_jax_equal(jnp_imag(r), c(-0.18750068545341492, -0.15973928570747375, 0.0, 0.0, 0.0), tolerance = 1e-6)
  a <- c(1 + 2i, -1 + 3i)
  b <- c(0.5 - 1i, 2 + 2i)
  r <- jnp_logaddexp(a, b)
  expect_jax_equal(jnp_real(r), c(0.10499301432446884, 2.0273761243841655), tolerance = 1e-10)
  expect_jax_equal(jnp_imag(r), c(1.7889591478473275, 2.0407743217569454), tolerance = 1e-10)
  r <- jnp_logaddexp2(a, b)
  expect_jax_equal(jnp_real(r), c(0.8491411504763546, 2.136271248038769), tolerance = 1e-10)
  expect_jax_equal(jnp_imag(r), c(0.9100140045266683, 2.1049349983225003), tolerance = 1e-10)
})

test_that("custom JVP rules of logaddexp, logaddexp2 and hypot match jax.grad", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  x1 <- c(0.3, -1.2, 2.0, -Inf, Inf, 5.0)
  x2 <- c(1.1, -1.2, -Inf, -Inf, 1.0, -3.0)
  e <- list(
    logaddexp = list(g1 = c(0.3100255188723876, 0.5, 1.0, NaN, 1.0, 0.9996646498695333),
                     g2 = c(0.6899744811276125, 0.5, 0.0, NaN, 2.7182818284590455, 0.00033535013046647805),
                     gg11 = c(0.21390969652029443, 0.25, 0.0, NaN, 0.0, 0.000335237670756701),
                     gg12 = c(-0.21390969652029448, -0.25, -0.0, NaN, 0.0, -0.0003352376707564741)),
    logaddexp2 = list(g1 = c(0.36481689431254416, 0.5, 1.0, NaN, 1.0, 0.9961089494163425),
                      g2 = c(0.635183105687456, 0.5, 0.0, NaN, 2.0, 0.0038910505836575863),
                      gg11 = c(0.16061989635308438, 0.17328679513998632, 0.0, NaN, 0.0, 0.0026865763027955052),
                      gg12 = c(-0.1606198963530844, -0.17328679513998632, -0.0, NaN, 0.0, -0.0026865763027955907)),
    hypot = list(g1 = c(0.26311740579210874, -0.7071067811865475, 0.0, NaN, NaN, 0.8574929257125441),
                 g2 = c(0.9647638212377322, -0.7071067811865475, NaN, NaN, 0.0, -0.5144957554275265),
                 gg11 = c(0.8163386179703888, 0.29462782549439487, 0.0, NaN, NaN, 0.045396684302428825),
                 gg12 = c(-0.22263780490101512, -0.29462782549439476, NaN, NaN, NaN, 0.075661140504048)))
  fns <- list(logaddexp = jnp_logaddexp, logaddexp2 = jnp_logaddexp2, hypot = jnp_hypot)
  for (nm in names(fns)) {
    f <- fns[[nm]]
    g <- vmap(grad(f, argnums = c(0L, 1L)))(x1, x2)
    expect_jax_equal(g[[1L]], e[[nm]]$g1, tolerance = 1e-12)
    expect_jax_equal(g[[2L]], e[[nm]]$g2, tolerance = 1e-12)
    expect_jax_equal(vmap(grad(grad(f, 0L), 0L))(x1, x2), e[[nm]]$gg11, tolerance = 1e-12)
    expect_jax_equal(vmap(grad(grad(f, 0L), 1L))(x1, x2), e[[nm]]$gg12, tolerance = 1e-12)
    expect_jax_equal(jit(vmap(grad(f)))(x1, x2), e[[nm]]$g1, tolerance = 1e-12)
    check_grad_fd(function(x) jnp_sum(f(x, as_jax(c(0.4, -2.0, 1.5), "float64"))),
                  as_jax(c(0.3, 1.7, -0.6), "float64"))
  }
  g <- vmap(grad(jnp_hypot, argnums = c(0L, 1L)))(c(0, 1e-300, 1e300), c(0, 2e-300, -1e300))
  expect_jax_equal(g[[1L]], c(NaN, 0.447213595499958, 0.7071067811865475))
  expect_jax_equal(g[[2L]], c(NaN, 0.894427190999916, -0.7071067811865475))
})

test_that("sinc gradients match jax.grad at zero, integers and generic points", {
  skip_if_no_backend()
  xs <- c(0.0, 0.3, -1.0, 2.0, 2.5, -0.7, 1e-9, 3.0)
  expect_jax_equal(vmap(grad(jnp_sinc))(as_jax(xs, "float32")),
                   c(0.0, -0.9020282030105591, 1.0, 0.5, -0.050929583609104156, 1.3652403354644775,
                     -3.2898679336312853e-09, -0.3333333432674408), tolerance = 1e-6)
  expect_jax_equal(vmap(grad(grad(jnp_sinc)))(as_jax(xs, "float32")),
                   c(-3.289868116378784, -2.4584856033325195, 2.0, -0.5, -1.2158931493759155,
                     0.26982641220092773, -3.289868116378784, 0.2222222238779068), tolerance = 1e-5)
  enable_x64()
  on.exit(enable_x64(FALSE))
  x <- as_jax(xs, "float64")
  g <- c(0.0, -0.9020281301388888, 1.0, 0.5, -0.050929581789406556, 1.3652403755203535,
         -3.289868133696453e-09, -0.3333333333333333)
  expect_jax_equal(vmap(grad(jnp_sinc))(x), g, tolerance = 1e-12)
  expect_jax_equal(jit(vmap(grad(jnp_sinc)))(x), g, tolerance = 1e-12)
  expect_jax_equal(vmap(grad(grad(jnp_sinc)))(x),
                   c(-3.289868133696453, -2.4584852862661744, 2.0, -0.5, -1.215893396004392,
                     0.2698270069758233, -3.289868133696453, 0.2222222222222222), tolerance = 1e-12)
  check_grad_fd(function(x) jnp_sum(jnp_sinc(x)), as_jax(c(0.3, -1.7, 2.6), "float64"))
})

test_that("gradients of the other ufuncs match jax.grad and finite differences", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  xv <- as_jax(c(-2.5, -0.3, 0.4, 2.7), "float64")
  yv <- as_jax(c(1.5, 0.7, -2.0, 0.5), "float64")
  nv <- c(3L, -2L, 1L, 0L)
  gr <- function(f, ...) vmap(grad(f))(...)
  expect_jax_equal(gr(function(x, n) jnp_ldexp(x, n), xv, nv), c(8.0, 0.25, 2.0, 1.0))
  expect_jax_equal(gr(function(x) jnp_frexp(x)[[1L]], xv), c(0.25, 2.0, 2.0, 0.25))
  expect_jax_equal(gr(function(x, y) jnp_copysign(x, y), xv, yv), c(-1.0, -1.0, -1.0, 1.0))
  expect_jax_equal(gr(function(y, x) jnp_copysign(x, y), yv, xv), c(0, 0, 0, 0))
  expect_jax_equal(gr(function(x, y) jnp_fmod(x, y), xv, yv), c(1, 1, 1, 1))
  expect_jax_equal(gr(function(y, x) jnp_fmod(x, y), yv, xv), c(1.0, 0.0, 0.0, -5.0))
  expect_jax_equal(gr(function(x, y) jnp_divmod(x, y)[[2L]], xv, yv), c(1, 1, 1, 1))
  expect_jax_equal(gr(function(y, x) jnp_divmod(x, y)[[2L]], yv, xv), c(2.0, 1.0, 1.0, -5.0))
  expect_jax_equal(gr(function(x) jnp_modf(x)[[1L]] + 3 * jnp_modf(x)[[2L]], xv), c(1, 1, 1, 1))
  expect_jax_equal(gr(function(x, y) jnp_float_power(x, y), jnp_abs(xv), yv),
                   c(2.3717082451262845, 1.004527113916513, -31.249999999999996, 0.3042903097250923))
  expect_jax_equal(gr(function(y, x) jnp_float_power(x, y), yv, jnp_abs(xv)),
                   c(3.6219571395312187, -0.5183242827272158, -5.726817074213468, 1.6320792040791996))
  expect_jax_equal(gr(function(y, x) jnp_heaviside(x, y), yv, c(0.0, -0.3, 0.0, 1.0)), c(1, 0, 1, 0))
  expect_jax_equal(gr(jnp_deg2rad, xv), rep(0.017453292519943295, 4))
  expect_jax_equal(gr(jnp_rad2deg, xv), rep(57.29577951308232, 4))
  expect_jax_equal(gr(jnp_acosh, c(1.5, 3.0)), c(0.8944271909999159, 0.35355339059327373))
  expect_jax_equal(gr(jnp_atanh, c(0.3, -0.6)), c(1.098901098901099, 1.5624999999999998))
  x <- as_jax(c(0.3, -1.7, 2.6), "float64")
  check_grad_fd(function(x) jnp_sum(jnp_ldexp(x, c(2L, -1L, 3L))^2), x)
  check_grad_fd(function(x) jnp_sum(jnp_frexp(x)[[1L]]^2), x)
  check_grad_fd(function(x) jnp_sum(jnp_float_power(jnp_abs(x), 1.5)), x)
  check_grad_fd(function(x) jnp_sum(jnp_copysign(x, -1)), x)
  check_grad_fd(function(x) jnp_sum(jnp_fmod(x, 0.7)), x)
  check_grad_fd(function(x) jnp_sum(jnp_deg2rad(x) * jnp_atan(x)), x)
  expect_jax_equal(jit(grad(function(x) jnp_sum(jnp_ldexp(x, c(2L, -1L, 3L)))))(x), c(4, 0.5, 8))
})

test_that("nextafter and spacing gradients match jax.grad", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  xv <- as_jax(c(-2.5, -0.3, 0.4, 2.7), "float64")
  expect_jax_equal(vmap(grad(jnp_spacing))(xv), c(0, 0, 0, 0))
  expect_jax_equal(vmap(grad(function(x, y) jnp_nextafter(x, y)))(xv, c(1.5, 0.7, -2.0, 0.5)), c(1, 1, 1, 1))
})

test_that("ldexp and frexp primitives batch, linearize and transpose", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  X <- as_jax(matrix(c(0.5, -1.5, 3.0, 2.0, 0.25, -7.0), 2, byrow = TRUE), "float64")
  N <- matrix(c(1L, -2L, 3L, 0L, 4L, -1L), 2, byrow = TRUE)
  expect_rm(vmap(jnp_ldexp, in_axes = list(0L, NULL))(X, N[1L, ]), c(1.0, -0.375, 24.0, 4.0, 0.0625, -56.0),
            shape = c(2, 3))
  expect_rm(vmap(jnp_ldexp, in_axes = list(NULL, 1L))(jnp_ravel(X[0L]), t(N)),
            c(1.0, -0.375, 24.0, 0.5, -24.0, 1.5), shape = c(2, 3))
  expect_rm(vmap(jnp_ldexp, in_axes = list(1L, 0L))(X, t(N)), c(1.0, 2.0, -0.375, 4.0, 24.0, -3.5),
            shape = c(3, 2))
  fr <- vmap(jnp_frexp, in_axes = 1L)(X)
  expect_rm(fr[[1L]], c(0.5, 0.5, -0.75, 0.5, 0.75, -0.875), shape = c(3, 2))
  expect_rm(fr[[2L]], c(0, 2, 1, -1, 2, 3), dtype = "int32")
  loop <- jnp_stack(lapply(0:1, function(i) jnp_ldexp(X[i], N[i + 1L, ])))
  expect_jax_equal(vmap(jnp_ldexp)(X, N), as.array(loop))
  expect_rm(jacfwd(function(x) jnp_ldexp(x, N[1L, ]))(X[0L]), c(2, 0, 0, 0, 0.25, 0, 0, 0, 8), shape = c(3, 3))
  expect_rm(jacrev(function(x) jnp_frexp(x)[[1L]])(X[1L]), c(0.25, 0, 0, 0, 2, 0, 0, 0, 0.125), shape = c(3, 3))
  expect_rm(grad(function(x) jnp_sum(jnp_ldexp(x, N)^2))(X), c(4.0, -0.1875, 384.0, 4.0, 128.0, -3.5))
  expect_rm(vmap(jnp_hypot, in_axes = list(0L, NULL))(X, c(1, 2, -3)),
            c(1.118033988749895, 2.5, 4.242640687119285, 2.23606797749979, 2.0155644370746373, 7.615773105863909))
  expect_rm(vmap(grad(function(a, b) jnp_sum(jnp_logaddexp(a, b))))(X, X[c(1L, 0L)]),
            c(0.18242552380635635, 0.14804719803168948, 0.9999546021312976, 0.8175744761936437,
              0.8519528019683105, 4.5397868702434354e-05))
  xs <- as_jax(matrix(c(0.3, 1.5, -2.0, 0.0, 4.25, -0.75), 3), "float64")
  expect_jax_equal(vmap(jnp_sinc, in_axes = 1L)(xs), as.array(jnp_stack(list(jnp_sinc(xs[, 0L]), jnp_sinc(xs[, 1L])))))
  expect_jax_equal(vmap(function(x) jnp_spacing(x))(xs), as.array(jnp_spacing(xs)))
})

test_that("ufuncs raise JAX's errors", {
  skip_if_no_backend()
  expect_error(jnp_fabs(1i), "does not support complex", class = "jax_type_error")
  expect_error(jnp_spacing(1i), "not defined for complex", class = "jax_value_error")
  expect_error(jnp_copysign(1i, 1), "complex-valued", class = "jax_type_error")
  expect_error(jnp_signbit(1i), "not well defined", class = "jax_value_error")
  expect_error(jnp_ldexp(1, 2.5), "ldexp not supported", class = "jax_value_error")
  expect_error(jnp_ldexp(1i, 2L), "ldexp not supported", class = "jax_value_error")
  expect_error(jnp_frexp(1i), "complex-valued", class = "jax_type_error")
  expect_error(jnp_hypot(1i, 1), "not well defined for complex", class = "jax_value_error")
  expect_error(jnp_isposinf(1i), "not well defined for complex", class = "jax_value_error")
  expect_error(jnp_isneginf(1, out = 1), "not supported", class = "jax_not_implemented_error")
  expect_error(jnp_modf(1, out = 1), "not supported", class = "jax_not_implemented_error")
  expect_error(jnp_left_shift(1.5, 2L), "shift_left")
  expect_error(jnp_bitwise_count(1.5), "population_count")
  expect_error(jnp_bitwise_invert(1.5))
})

ufn_mat <- function(v, dtype = NULL) as_jax(matrix(v, 2, byrow = TRUE), dtype)

test_that("standard ufunc objects reduce like JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  u <- jnp_ufuncs
  A <- ufn_mat(c(0.5, -1.0, 2.0, 3.0, 0.25, -4.0))
  I <- ufn_mat(c(3L, 5L, 12L, 6L, 10L, 7L), "int32")
  U <- ufn_mat(c(3, 5, 12, 6, 10, 255), "uint8")
  B <- ufn_mat(c(TRUE, FALSE, TRUE, TRUE, TRUE, FALSE))
  W <- ufn_mat(c(TRUE, FALSE, TRUE, FALSE, TRUE, TRUE))
  expect_rm(u$add$reduce(A), c(3.5, -0.75, -2.0), shape = 3, dtype = "float64")
  expect_rm(u$add$reduce(A, axis = 1L), c(1.5, -0.75), shape = 2)
  expect_rm(u$add$reduce(A, axis = NULL), 0.75, shape = integer(0))
  expect_rm(u$add$reduce(A, axis = 1L, keepdims = TRUE), c(1.5, -0.75), shape = c(2, 1))
  expect_rm(u$add$reduce(A, axis = 1L, where = W, initial = 10), c(12.5, 6.25))
  expect_rm(u$add$reduce(I), c(9, 15, 19), dtype = "int64")
  expect_rm(u$add$reduce(B), c(2, 1, 1), dtype = "int64")
  expect_rm(u$add$reduce(I, dtype = "float32"), c(9, 15, 19), dtype = "float32")
  expect_false(jaxR:::aval_of(u$add$reduce(2 * jnp_ones(3L)))$weak)
  expect_rm(u$multiply$reduce(A, axis = 1L), c(-1, -3))
  expect_rm(jnp_ufunc_reduce(u$multiply, as_jax(c(2L, 3L), "int8")), 6, dtype = "int64")
  expect_rm(u$maximum$reduce(A), c(3.0, 0.25, 2.0))
  expect_rm(u$minimum$reduce(A, axis = 1L, where = W, initial = 0), c(0, -4))
  expect_rm(u$maximum$reduce(I, axis = 1L), c(12, 10), dtype = "int32")
  expect_rm(u$maximum$reduce(B, axis = 1L), c(TRUE, TRUE), dtype = "bool")
  expect_rm(u$bitwise_and$reduce(U), c(2, 0, 12), dtype = "uint8")
  expect_rm(u$bitwise_or$reduce(I, axis = 1L), c(15, 15), dtype = "int32")
  expect_rm(u$bitwise_xor$reduce(I, axis = NULL), 1, dtype = "int32")
  expect_rm(u$bitwise_and$reduce(B), c(TRUE, FALSE, FALSE), dtype = "bool")
  expect_rm(u$logical_and$reduce(A), c(TRUE, TRUE, TRUE), dtype = "bool")
  expect_rm(u$logical_or$reduce(jnp_zeros(c(2L, 2L)), axis = 1L), c(FALSE, FALSE), dtype = "bool")
  expect_rm(u$logical_xor$reduce(B, axis = 1L), c(FALSE, FALSE), dtype = "bool")
  expect_rm(u$logaddexp$reduce(A, axis = 1L), c(2.241311296657157, 3.0628243119441656))
  expect_rm(u$logaddexp$reduce(A, axis = 1L, where = W, initial = 1), c(2.4643687841079447, 1.391436836905488))
  expect_rm(u$logaddexp2$reduce(A), c(3.234840581079202, 0.756346228949416, 2.0223678130284544))
  expect_rm(u$logaddexp2$reduce(A, axis = NULL, initial = 2), 4.260471397736514)
  expect_rm(u$subtract$reduce(A, axis = 1L), c(-0.5, 6.75))
  expect_rm(u$subtract$reduce(A, axis = NULL), 0.25, shape = integer(0))
  expect_rm(u$subtract$reduce(A, axis = 0L, where = W, initial = 1), c(0.5, 0.75, 3.0))
  expect_rm(u$subtract$reduce(I, axis = 0L, keepdims = TRUE), c(-3, -5, 5), shape = c(1, 3), dtype = "int32")
  expect_rm(u$add$reduce(jnp_zeros(c(0L, 2L))), c(0, 0))
  expect_rm(jit(function(a) u$add$reduce(a, axis = 1L))(A), c(1.5, -0.75))
  expect_rm(jit(function(a) u$subtract$reduce(a, axis = 1L))(A), c(-0.5, 6.75))
  expect_rm(grad(function(a) u$subtract$reduce(a, axis = NULL))(A), c(1, -1, -1, -1, -1, -1))
  expect_rm(grad(function(a) jnp_sum(u$logaddexp$reduce(a, axis = 1L)))(A),
            c(0.1752903921400367, 0.039112573270687456, 0.7855970345892759, 0.939108449334265,
              0.060035194607087256, 0.0008563560586477833))
  check_grad_fd(function(a) jnp_sum(u$logaddexp2$reduce(a, axis = 0L)), A)
})

test_that("standard ufunc objects accumulate, reduceat, outer and at like JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  u <- jnp_ufuncs
  A <- ufn_mat(c(0.5, -1.0, 2.0, 3.0, 0.25, -4.0))
  I <- ufn_mat(c(3L, 5L, 12L, 6L, 10L, 7L), "int32")
  expect_rm(u$add$accumulate(A, axis = 1L), c(0.5, -0.5, 1.5, 3.0, 3.25, -0.75), shape = c(2, 3))
  expect_rm(u$multiply$accumulate(A), c(0.5, -1.0, 2.0, 1.5, -0.25, -8.0))
  expect_rm(u$subtract$accumulate(A, axis = 1L), c(0.5, 1.5, -0.5, 3.0, 2.75, 6.75))
  expect_rm(u$maximum$accumulate(A, axis = 1L), c(0.5, 0.5, 2.0, 3.0, 3.0, 3.0))
  expect_rm(u$logaddexp$accumulate(A, axis = 0L),
            c(0.5, -1.0, 2.0, 3.0788897342925496, 0.501929081345373, 2.0024756851377306))
  expect_rm(u$subtract$accumulate(I, axis = 0L, dtype = "float32"), c(3, 5, 12, -3, -5, 5), dtype = "float32")
  expect_identical(as.integer(dim(u$add$accumulate(jnp_zeros(c(0L, 2L))))), c(0L, 2L))
  expect_rm(u$add$reduceat(A, c(0L, 2L), axis = 1L), c(-0.5, 2.0, 3.25, -4.0), shape = c(2, 2))
  expect_rm(u$maximum$reduceat(jnp_arange(8, dtype = "float64"), c(0L, 4L, 1L, 5L)), c(3, 4, 4, 7))
  expect_rm(u$multiply$reduceat(A, c(1L, 0L), axis = 0L), c(3.0, 0.25, -4.0, 1.5, -0.25, -8.0))
  expect_rm(u$add$outer(c(1, 2), c(10, 20, 30)), c(11, 21, 31, 12, 22, 32), shape = c(2, 3))
  expect_rm(u$subtract$outer(A, c(1, 2)),
            c(-0.5, -1.5, -2.0, -3.0, 1.0, 0.0, 2.0, 1.0, -0.75, -1.75, -5.0, -6.0), shape = c(2, 3, 2))
  expect_rm(u$maximum$outer(2, c(1, 3)), c(2, 3), shape = 2)
  v <- as_jax(c(1, 2, 3, 4))
  expect_rm(u$add$at(v, c(0L, 0L, 2L), 5, inplace = FALSE), c(11, 2, 8, 4))
  expect_rm(u$multiply$at(v, c(1L, 3L, 1L), c(2, 3, 4), inplace = FALSE), c(1, 16, 3, 12))
  expect_rm(u$subtract$at(v, 1L, 10, inplace = FALSE), c(1, -8, 3, 4))
  expect_rm(u$add$at(c(TRUE, FALSE, FALSE), c(1L, 1L), TRUE, inplace = FALSE), c(TRUE, TRUE, FALSE), dtype = "bool")
  expect_rm(u$add$at(A, list(c(0L, 1L, 1L), c(2L, 0L, 0L)), 1, inplace = FALSE), c(0.5, -1, 3, 5, 0.25, -4))
  expect_rm(u$maximum$at(v, c(0L, 0L, 3L), c(7, 0.5, 2), inplace = FALSE), c(7, 2, 3, 4))
  expect_rm(u$negative$at(v, c(1L, 3L, 3L), inplace = FALSE), c(1, -2, 3, 4))
  expect_rm(u$logaddexp$at(v, c(2L, 2L), 0, inplace = FALSE), c(1, 2, 3.094922956420961, 4))
  expect_rm(jit(function(v) u$maximum$at(v, c(0L, 0L, 3L), c(7, 0.5, 2), inplace = FALSE))(v), c(7, 2, 3, 4))
  expect_rm(u$negative(A), c(-0.5, 1.0, -2.0, -3.0, -0.25, 4.0))
  expect_rm(u$add(A, 1), c(1.5, 0.0, 3.0, 4.0, 1.25, -3.0))
})

test_that("jnp_frompyfunc builds ufuncs with generic methods", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  A <- ufn_mat(c(0.5, -1.0, 2.0, 3.0, 0.25, -4.0))
  I <- ufn_mat(c(3L, 5L, 12L, 6L, 10L, 7L), "int32")
  W <- ufn_mat(c(TRUE, FALSE, TRUE, FALSE, TRUE, TRUE))
  v <- as_jax(c(1, 2, 3, 4))
  f <- jnp_frompyfunc(function(x, y) x * y + 1, nin = 2, nout = 1, identity = 0)
  expect_rm(f(A, c(1, 2, 3)), c(1.5, -1.0, 7.0, 4.0, 1.5, -11.0), shape = c(2, 3))
  expect_rm(f$reduce(A, axis = 1L), c(1, -4))
  expect_rm(f$reduce(A, axis = NULL), -7, shape = integer(0))
  expect_rm(jnp_frompyfunc(function(x, y) x * y + 1, 2, 1)$reduce(A, axis = 0L), c(2.5, 0.75, -7.0))
  expect_rm(f$accumulate(A, axis = 1L), c(0.5, 0.5, 2.0, 3.0, 1.75, -6.0))
  expect_rm(f$reduceat(A, c(0L, 1L), axis = 1L), c(0.5, -1.0, 3.0, 0.0))
  expect_rm(f$outer(c(1, 2), c(3, 4, 5)), c(4, 5, 6, 7, 9, 11), shape = c(2, 3))
  expect_rm(f$at(v, c(0L, 0L, 2L), 2, inplace = FALSE), c(7, 2, 7, 4))
  expect_rm(f$at(v, matrix(c(0, 1, 2, 0), 2, byrow = TRUE), matrix(c(10, 20, 30, 40), 2, byrow = TRUE),
                 inplace = FALSE), c(441, 41, 91, 4))
  expect_rm(f$reduce(A, axis = 1L, where = W), c(3, -3))
  expect_rm(jnp_frompyfunc(function(x, y) x + y, 2, 1, identity = 0L)$reduce(I, axis = 1L), c(20, 23),
            dtype = "int32")
  expect_rm(f$reduce(jnp_zeros(c(0L, 2L))), c(0, 0))
  expect_identical(as.integer(dim(f$accumulate(jnp_zeros(c(2L, 0L)), axis = 1L))), c(2L, 0L))
  expect_rm(jit(function(a) f$reduce(a, axis = 1L))(A), c(1, -4))
  expect_rm(jit(function(a) f$accumulate(a, axis = 1L))(A), c(0.5, 0.5, 2.0, 3.0, 1.75, -6.0))
  expect_rm(grad(function(a) jnp_sum(f$reduce(a, axis = 1L)))(A), c(0, 2, 0, 0, -4, 1.25))
  g <- jnp_frompyfunc(function(x) jnp_sin(x), 1, 1)
  expect_rm(g(A), c(0.479425538604203, -0.8414709848078965, 0.9092974268256817, 0.1411200080598672,
                    0.24740395925452294, 0.7568024953079283))
  expect_rm(g$at(v, c(1L, 2L), inplace = FALSE), c(1.0, 0.9092974268256817, 0.1411200080598672, 4.0))
  h <- jnp_frompyfunc(function(x) list(x + 1, x * 2), 1, 2)
  r <- h(c(1, 2))
  expect_rm(r[[1L]], c(2, 3))
  expect_rm(r[[2L]], c(2, 4))
  expect_identical(c(f$nin, f$nout, f$nargs), c(2L, 1L, 2L))
  expect_identical(f$identity, 0)
  expect_identical(f$name, "<lambda>")
  myfun <- function(x, y) x - y
  expect_identical(jnp_frompyfunc(myfun, 2, 1)$name, "myfun")
  expect_identical(jnp_ufuncs$add$identity, 0)
  expect_identical(jnp_ufuncs$logaddexp$identity, -Inf)
  expect_null(jnp_ufuncs$subtract$identity)
  expect_identical(jnp_ufuncs$negative$nin, 1L)
  expect_identical(sort(names(jnp_ufuncs)),
                   c("add", "bitwise_and", "bitwise_or", "bitwise_xor", "logaddexp", "logaddexp2",
                     "logical_and", "logical_or", "logical_xor", "maximum", "minimum", "multiply",
                     "negative", "subtract"))
  expect_output(print(jnp_ufuncs$add), "<jnp.ufunc 'add'>", fixed = TRUE)
  expect_true(inherits(jnp_ufunc(function(x) x, 1, 1), "jnp_ufunc"))
})

test_that("ufunc methods raise JAX's errors", {
  skip_if_no_backend()
  u <- jnp_ufuncs
  A <- ufn_mat(c(0.5, -1.0, 2.0, 3.0, 0.25, -4.0))
  W <- ufn_mat(c(TRUE, FALSE, TRUE, FALSE, TRUE, TRUE))
  expect_error(u$add$at(c(1, 2), 0L, 1), "cannot operate inplace", class = "jax_not_implemented_error")
  expect_error(u$negative$reduce(A), "only supported for binary ufuncs", class = "jax_value_error")
  expect_error(u$negative$accumulate(A), "only supported for binary ufuncs", class = "jax_value_error")
  expect_error(u$negative$outer(A, A), "only supported for binary ufuncs", class = "jax_value_error")
  expect_error(u$negative$reduceat(A, 0L), "only supported for binary ufuncs", class = "jax_value_error")
  h <- jnp_frompyfunc(function(x, y) list(x, y), 2, 2)
  expect_error(h$reduce(A), "returning a single value", class = "jax_value_error")
  expect_error(u$add$reduce(A, out = A), "out argument", class = "jax_not_implemented_error")
  expect_error(u$minimum$reduce(A, where = W), "does not have an identity", class = "jax_value_error")
  expect_error(u$add$reduce(A, where = ufn_mat(1:6)), "dtype=bool", class = "jax_value_error")
  expect_error(u$add$reduce(A, initial = c(1, 2)), "initial value must be a scalar", class = "jax_value_error")
  expect_error(u$maximum$reduce(jnp_zeros(c(0L, 2L))), "zero-size array", class = "jax_value_error")
  f <- jnp_frompyfunc(function(x, y) x * y, 2, 1)
  expect_error(f$reduce(jnp_zeros(c(0L, 2L))), "zero-size array", class = "jax_value_error")
  expect_error(f$reduce(A, axis = c(0L, 1L)), "tuple of axes", class = "jax_not_implemented_error")
  expect_error(f$accumulate(A, axis = NULL), "multiple axes", class = "jax_value_error")
  expect_error(f$reduceat(as_jax(1), 0L), "1 or more dimension", class = "jax_value_error")
  expect_error(f$reduceat(A, matrix(0L, 1, 1)), "one-dimensional", class = "jax_value_error")
  expect_error(f$reduceat(A, matrix(0, 2, 2)), "one-dimensional", class = "jax_value_error")
  expect_error(u$bitwise_and$reduce(A), "integer argument required", class = "jax_value_error")
  expect_error(u$add(A, A, out = A), "out argument", class = "jax_not_implemented_error")
  expect_error(u$add(A, A, where = W), "where argument", class = "jax_not_implemented_error")
  expect_error(u$add$foo, "no attribute")
  expect_error(jnp_ufunc_reduce(function(x, y) x, A), "expected a jnp_ufunc", class = "jax_type_error")
  expect_error(jnp_ufunc(function(x) x, 1.5, 1), "must be an integer", class = "jax_type_error")
})
