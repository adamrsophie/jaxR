# Tests for jax.scipy.interpolate, jax.scipy.cluster.vq and jax.scipy.spatial.transform
# (ports of jax/_src/third_party/scipy/interpolate.py, jax/_src/scipy/cluster/vq.py and
# jax/_src/scipy/spatial/transform.py). Expected values come from jax 0.12.0.1 (float32).

# R array from C order values.
carr <- function(v, shape) {
  if (length(shape) <= 1L) return(array(v, shape))
  aperm(array(v, rev(shape)), rev(seq_along(shape)))
}

rgi_lin <- c(1.5585084, 3.1870441, 2.8433514, NaN, NaN, 2.5495298, 3.5210757)
rgi_near <- c(0, 2.5495298, 2.5845006, NaN, NaN, 2.5495298, 4.1711674)
rgi_none <- c(1.5585084, 3.1870441, 2.8433514, 5.7536545, 1.3448324, 2.5495298, 3.5210757)
rgi_fill0 <- c(0, 2.5495298, 2.5845006, 0, 0, 2.5495298, 4.1711674)
rgi_trail <- carr(c(1.5585084, -2.1170168, 3.1870441, -5.3740883, 2.8433514, -4.6867027, NaN, NaN, NaN, NaN, 2.5495298, -4.0990596, 3.5210757, -6.042152), c(7L, 2L))
rgi_tuple <- carr(c(1.0314331, 1.5584207, 3.350561, 1.9071095, 2.635059, 3.7488732), c(2L, 3L))
rgi_1d <- c(1.6, 0.6666665, -1.1666666, NaN)
rgi_3d <- carr(c(1.5585084, 3.1870441, 2.8433514, NaN, NaN, 2.5495298, 3.5210757), c(7L, 1L))
vq2_0 <- c(0, 0, 1, 1, 2)
vq2_1 <- c(0, 1.8027756, 0.19999981, 1.4142135, 1)
vq1_0 <- c(0, 2, 1, 2)
vq1_1 <- c(0.2, 0.099999905, 0.39999998, 3)
eul_xyz <- carr(c(0.39190382, 0.20056215, -0.5319757, 0.7233174, 0.7109782, 0.4906241, 0.36310402, 0.34921843, 0, 0.70710677, 0, 0.70710677), c(3L, 4L))
eulm_xyz <- carr(c(0.35355335, 0.92677665, -0.12682644, -0.6123724, 0.12682647, -0.78033006, -0.7071068, 0.3535533, 0.6123724, 0.254887, 0.44404086, 0.85898805, 0.95125127, -0.27466902, -0.14027822, 0.17364812, 0.85286856, -0.49240395, 0, 0, 0.99999994, 0, 0.99999994, 0, -0.99999994, 0, 0), c(3L, 3L, 3L))
ase_xyz_xyz <- carr(c(30.000008, 45.000004, -60.000004, 119.99999, -9.999998, 75, 0, 90, 0), c(3L, 3L))
ase_xyz_ZXY <- carr(c(-82.20765, 20.704803, 49.106617, -121.739555, 58.525036, -160.57462, 0, 0, 90), c(3L, 3L))
ase_xyz_zyz <- carr(c(26.565052, 52.23875, -99.231514, 101.50838, 119.49871, -9.274892, 0, 90, 0), c(3L, 3L))
ase_xyz_YXY <- carr(c(69.1188, 82.71377, -38.123425, 27.503464, 105.94228, 81.61121, 90, 0, 0), c(3L, 3L))
eul_ZYX <- carr(c(-0.5319757, 0.20056215, 0.39190382, 0.7233174, 0.36310402, 0.4906241, 0.7109782, 0.34921843, 0, 0.70710677, 0, 0.70710677), c(3L, 4L))
eulm_ZYX <- carr(c(0.6123724, -0.78033006, -0.12682644, 0.3535533, 0.12682647, 0.92677665, -0.7071068, -0.6123724, 0.35355335, -0.49240395, -0.14027822, 0.85898805, 0.85286856, -0.27466902, 0.44404086, 0.17364812, 0.95125127, 0.254887, 0, 0, 0.99999994, 0, 0.99999994, 0, -0.99999994, 0, 0), c(3L, 3L, 3L))
ase_ZYX_xyz <- carr(c(-60.000004, 45.000004, 30.000008, 75, -9.999998, 119.99999, 0, 90, 0), c(3L, 3L))
ase_ZYX_ZXY <- carr(c(80.76848, -37.76125, 63.43496, 152.94582, 72.03613, -34.265625, 0, 0, 90), c(3L, 3L))
ase_ZYX_zyz <- carr(c(-40.893394, 69.29518, 97.79234, 100.345276, 75.23312, 27.336016, 0, 90, 0), c(3L, 3L))
ase_ZYX_YXY <- carr(c(-128.12343, 82.71377, 159.11877, -8.38877, 105.94228, 117.50347, 90, 0, 0), c(3L, 3L))
eul_zxz <- carr(c(0.27059808, -0.27059805, -0.23911762, 0.8923991, -0.0805214, 0.033353057, 0.9876721, -0.13002956, 0.70710677, 0, 0, 0.70710677), c(3L, 4L))
eulm_zxz <- carr(c(0.73919886, 0.28033006, -0.61237246, -0.5732233, 0.73919886, -0.35355344, 0.35355335, 0.6123725, 0.7071067, -0.9532172, 0.25148186, -0.16773124, -0.2622244, -0.96395975, 0.044943437, -0.1503837, 0.08682409, 0.9848077, 0.99999994, 0, 0, 0, 0, -0.99999994, 0, 0.99999994, 0), c(3L, 3L, 3L))
ase_zxz_xyz <- carr(c(40.893394, -20.704803, -37.79235, 5.038371, 8.649164, -164.61874, 90, 0, 0), c(3L, 3L))
ase_zxz_ZXY <- carr(c(-20.768475, 37.761234, -26.565052, -165.37834, 4.9809155, 8.682195, 0, 90, 0), c(3L, 3L))
ase_zxz_zyz <- carr(c(119.99999, 45.000004, -150, 30.000008, 9.999998, 165.00002, 90, 90, -90), c(3L, 3L))
ase_zxz_YXY <- carr(c(24.59722, 42.336777, -58.334488, 70.95278, 164.57074, -99.725586, 0, 90, 0), c(3L, 3L))
eul_XYX <- carr(c(-0.23911762, 0.27059808, 0.27059805, 0.8923991, 0.9876721, -0.0805214, -0.033353057, -0.13002956, 0, 0.70710677, 0, 0.70710677), c(3L, 4L))
eulm_XYX <- carr(c(0.7071067, -0.61237246, 0.35355344, 0.35355335, 0.73919886, 0.5732233, -0.6123725, -0.28033006, 0.73919886, 0.9848078, -0.16773124, -0.044943437, -0.1503837, -0.9532172, 0.2622244, -0.08682409, -0.25148186, -0.96395975, 0, 0, 0.99999994, 0, 0.99999994, 0, -0.99999994, 0, 0), c(3L, 3L, 3L))
ase_XYX_xyz <- carr(c(-20.768475, 37.761234, 26.565052, -165.37834, 4.9809155, -8.682195, 0, 90, 0), c(3L, 3L))
ase_XYX_ZXY <- carr(c(39.63928, -16.279907, 39.63928, 170.0202, -14.565228, 174.85324, 0, 0, 90), c(3L, 3L))
ase_XYX_zyz <- carr(c(-24.59722, 42.336777, 58.334473, -70.952774, 164.57074, 99.725586, 0, 90, 0), c(3L, 3L))
ase_XYX_YXY <- carr(c(-114.59721, 42.336777, 148.33452, -146.29785, 162.40503, -150.16606, 90, 0, 0), c(3L, 3L))
rot_single_x <- c(0.3428978, 0, 0, 0.9393727)
rot_rad <- c(0.12333662, -0.55830216, -0.08437912, 0.816068)
rv_r <- carr(c(0.8651789, 0.44276714, -1.1744058, 1.8423303, 1.2713352, 0.9408974, 0, 1.5707964, 0), c(3L, 3L))
rvd_r <- carr(c(49.5711, 25.36869, -67.2885, 105.557755, 72.84215, 53.90945, 0, 90, 0), c(3L, 3L))
mrp_r <- carr(c(0.22741245, 0.11638144, -0.3086928, 0.5269556, 0.3636358, 0.26912177, 0, 0.41421354, 0), c(3L, 3L))
mag_r <- c(1.5244036, 2.428119, 1.5707964)
inv_r <- carr(c(-0.39190382, -0.20056215, 0.5319757, 0.7233174, -0.7109782, -0.4906241, -0.36310402, 0.34921843, -0, -0.70710677, -0, 0.70710677), c(3L, 4L))
canon_r <- carr(c(-0.10259783, 0.20519567, -0.3077935, 0.9233805, -0, -0, 1, -0, -0, 0.6, -0.8, -0, -0.5, 0.5, 0.5, 0.5), c(4L, 4L))
sfirst_r <- carr(c(0.7233174, 0.39190382, 0.20056215, -0.5319757, 0.34921843, 0.7109782, 0.4906241, 0.36310402, 0.70710677, 0, 0.70710677, 0), c(3L, 4L))
apply_r <- carr(c(1.8266273, -2.6997097, 1.837117, 1.6851095, -1.3691422, -0.73202175, 0.99999994, 0, 0), c(3L, 3L))
apply_inv_r <- carr(c(-2.992512, 2.2410893, 0.14963067, 0.5680349, 1.1243618, -1.9139351, -0.99999994, 0, 0), c(3L, 3L))
apply_one_r <- carr(c(1.8266273, -2.6997097, 1.837117, -0.1438179, -0.88487446, 2.1086283, -0.12682644, -0.78033006, 0.6123724), c(3L, 3L))
compose_r <- carr(c(0.98494804, -0.09560807, 0.12654588, 0.06872255, 0.4965734, 0.34266996, 0.25360522, -0.756093, 0.75949085, 0.59385836, -0.24598423, -0.09998891), c(3L, 4L))
compose2_r <- carr(c(-0.09904577, 0.6532815, -0.6532815, 0.3696438, 0.75949085, 0.59385836, -0.24598423, -0.09998891, 0, 1, 0, 0), c(3L, 4L))
fquat_r <- carr(c(0.31622776, -0.10540926, 0.42163703, 0.84327406, 0.18257418, 0.36514837, 0.5477226, 0.73029673, 0, 0, 0, 1), c(3L, 4L))
fmat_r <- carr(c(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.7109782, 0.4906241, 0.36310402, 0.34921843), c(5L, 4L))
fmrp_r <- carr(c(0.1754386, -0.3508772, 0.5263158, 0.75438595, 0.8571429, 0, -0.2857143, -0.42857146), c(2L, 4L))
frv_r <- carr(c(0.049708843, -0.09941769, 0.14912653, 0.982551, 0.67428035, 0, -0.22476012, 0.7034407, 5e-05, 0, 0, 1), c(3L, 4L))
frvd_r <- c(0.2521877, 0, 0.37828156, 0.8906764)
ident_r <- c(0, 0, 0, 1)
mean_r <- carr(c(0.64756906, 0.65676194, 0.38641775, 0.40458673, 0.13338482, -0.90472025, -0.6457279, 0.74220806, -0.17934123), c(3L, 3L))
meanw_r <- carr(c(0.4225394, 0.5202293, 0.742174, 0.79976314, 0.17125712, -0.57536966, -0.42642665, 0.83667964, -0.3436972), c(3L, 3L))
concat_r <- carr(c(0.39190382, 0.20056215, -0.5319757, 0.7233174, 0.7109782, 0.4906241, 0.36310402, 0.34921843, 0, 0.70710677, 0, 0.70710677), c(3L, 4L))
sl_rotvecs <- carr(c(0.10840969, 2.0035174, 0.9919373, -0.5716921, -0.2323843, -1.7651333), c(2L, 3L))
sl_q <- carr(c(0.39190388, 0.20056218, -0.53197575, 0.72331744, 0.54238087, 0.31648, -0.32857978, 0.70547765, 0.71097827, 0.49062416, 0.36310413, 0.34921828, 0.3228271, 0.6903355, 0.16487119, 0.62612873, -8.940697e-08, 0.7071067, -4.0978193e-08, 0.7071068), c(5L, 4L))
sl_one <- c(0.65067774, 0.40778574, -0.09963077, 0.63277406)

with_x64 <- function(code) with_jax_config("jax_enable_x64", TRUE, code)
gx <- c(0, 1, 2.5, 4)
gy <- c(-1, 0, 2)
vals <- carr(((0:11)^1.3) %% 5, c(4L, 3L))
xi <- matrix(c(0.5, -0.5, 2.0, 1.0, 3.9, 1.9, 4.5, 0.0, -0.1, 0.5, 2.5, 0.0, 1.25, -1.0), 7L,
             byrow = TRUE)
obs <- matrix(c(0, 0, 1, 1.5, 4, 4.2, 5, 3, -1, 2), 5L, byrow = TRUE)
cb <- matrix(c(0, 0, 4, 4, -1, 3), 3L, byrow = TRUE)
ang <- matrix(c(30, 45, -60, 120, -10, 75, 0, 90, 0), 3L, byrow = TRUE)
vecs <- matrix(c(1, 2, 3, -1, 0.5, 2, 0, 0, 1), 3L, byrow = TRUE)
# The rotation helpers use jnp functions of other files (jnp_vectorize, jnp_cross, ...).
rot_deps <- c("jnp_vectorize", "jnp_cross", "jnp_hypot", "jnp_hstack", "jnp_roll", "jnp_diff",
              "jnp_atleast_1d", "jnp_deg2rad", "jnp_rad2deg")
skip_if_no_rot <- function() {
  skip_if_not(all(vapply(rot_deps, exists, NA)), "jnp helpers of Rotation are not available")
}
Rot <- scipy_spatial_transform_Rotation

test_that("RegularGridInterpolator matches jax", {
  skip_if_no_backend()
  f <- scipy_interpolate_RegularGridInterpolator(list(gx, gy), vals)
  expect_jax_equal(f(xi), rgi_lin, dtype = "float32")
  expect_jax_equal(f(xi, method = "nearest"), rgi_near)
  expect_jax_equal(scipy_interpolate_RegularGridInterpolator(list(gx, gy), vals,
                                                             fill_value = NULL)(xi), rgi_none)
  expect_jax_equal(scipy_interpolate_RegularGridInterpolator(list(gx, gy), vals, method = "nearest",
                                                             fill_value = 0L)(xi), rgi_fill0)
  vals3 <- array(c(vals, -2 * vals + 1), c(4L, 3L, 2L))
  expect_jax_equal(scipy_interpolate_RegularGridInterpolator(list(gx, gy), vals3)(xi), rgi_trail,
                   shape = c(7L, 2L))
  expect_jax_equal(f(list(c(0.5, 1.5, 3.0), matrix(c(0, 1), 2L))), rgi_tuple, shape = c(2L, 3L))
  expect_jax_equal(scipy_interpolate_RegularGridInterpolator(gx, c(1, 3, -2, 0.5))(
    c(0.3, 1.7, 3.0, 5.0)), rgi_1d)
  expect_jax_equal(f(array(xi, c(7L, 1L, 2L))), rgi_3d,
                   shape = c(7L, 1L))
  # Fields.
  expect_identical(f$method, "linear")
  expect_false(f$bounds_error)
  expect_jax_equal(f$grid[[2L]], gy)
  expect_jax_equal(f$values, vals)
  expect_true(is.nan(as.array(f$fill_value)))
})

test_that("RegularGridInterpolator is a pytree usable under jit, grad and vmap", {
  skip_if_no_backend()
  f <- scipy_interpolate_RegularGridInterpolator(list(gx, gy), vals)
  expect_jax_equal(jit(function(interp, x) interp(x))(f, xi), rgi_lin)
  expect_jax_equal(jit(function(v, x) scipy_interpolate_RegularGridInterpolator(list(gx, gy), v)(x))(
    vals, xi), rgi_lin)
  leaves <- tree_leaves(f)
  expect_length(leaves, 4L)
  got <- vmap(function(x) f(x))(array(xi, c(7L, 1L, 2L)))
  expect_jax_equal(got, rgi_lin, shape = c(7L, 1L))
  with_x64({
    v64 <- as_jax(vals, "float64")
    x64 <- as_jax(xi[1:3, ], "float64")
    check_grad_fd(function(v) jnp_sum(scipy_interpolate_RegularGridInterpolator(list(gx, gy), v)(x64)^2L), v64)
    check_grad_fd(function(x) jnp_sum(jnp_sin(scipy_interpolate_RegularGridInterpolator(list(gx, gy), v64)(x))), x64)
  })
})

test_that("RegularGridInterpolator raises jax errors", {
  skip_if_no_backend()
  expect_error(scipy_interpolate_RegularGridInterpolator(list(gx, gy), vals, method = "cubic"),
               "method 'cubic' is not defined")
  expect_error(scipy_interpolate_RegularGridInterpolator(list(gx, gy), vals, bounds_error = TRUE),
               "takes no effect under JIT", class = "jax_not_implemented_error")
  expect_error(scipy_interpolate_RegularGridInterpolator(list(gx, gy, gy), vals),
               "there are 3 point arrays, but values has 2 dimensions")
  expect_error(scipy_interpolate_RegularGridInterpolator(list(gx, gy), vals, fill_value = 1i),
               "fill_value must be either 'None'")
  f <- scipy_interpolate_RegularGridInterpolator(list(gx, gy), vals)
  expect_error(f(matrix(1, 2L, 3L)), "have dimension 3, but this RegularGridInterpolator has dimension 2")
  expect_error(f(xi, method = "spline"), "method 'spline' is not defined")
})

test_that("vq matches jax", {
  skip_if_no_backend()
  res <- scipy_cluster_vq_vq(obs, cb)
  expect_jax_equal(res[[1L]], vq2_0, dtype = "int32")
  expect_jax_equal(res[[2L]], vq2_1, dtype = "float32")
  res <- scipy_cluster_vq_vq(c(0.2, 3.9, 1.6, 7.0), c(0, 2, 4))
  expect_jax_equal(res[[1L]], vq1_0)
  expect_jax_equal(res[[2L]], vq1_1)
  res <- jit(scipy_cluster_vq_vq)(obs, cb)
  expect_jax_equal(res[[1L]], vq2_0)
  expect_jax_equal(res[[2L]], vq2_1)
  with_x64({
    check_grad_fd(function(o) jnp_sum(scipy_cluster_vq_vq(o, cb)[[2L]]), as_jax(obs + 0.1, "float64"))
  })
  expect_error(scipy_cluster_vq_vq(obs, c(1, 2)), "should have the same rank")
  expect_error(scipy_cluster_vq_vq(array(1, c(2L, 2L, 2L)), array(1, c(2L, 2L, 2L))),
               "ndim different than 1 or 2 are not supported")
})

test_that("Rotation constructors and conversions match jax", {
  skip_if_no_backend()
  skip_if_no_rot()
  for (s in c("xyz", "ZYX", "zxz", "XYX")) {
    r <- scipy_spatial_transform_Rotation_from_euler(s, ang, degrees = TRUE)
    expect_jax_equal(r$quat, get(paste0("eul_", s)), dtype = "float32")
    expect_jax_equal(scipy_spatial_transform_Rotation_as_matrix(r), get(paste0("eulm_", s)),
                     shape = c(3L, 3L, 3L))
    for (s2 in c("xyz", "ZXY", "zyz", "YXY")) {
      expect_jax_equal(scipy_spatial_transform_Rotation_as_euler(r, s2, degrees = TRUE),
                       get(paste0("ase_", s, "_", s2)), tolerance = 1e-4)
    }
  }
  expect_jax_equal(scipy_spatial_transform_Rotation_from_euler("x", 0.7)$quat, rot_single_x)
  expect_jax_equal(scipy_spatial_transform_Rotation_from_euler("XY", c(0.3, -1.2))$quat, rot_rad)
  r <- scipy_spatial_transform_Rotation_from_euler("xyz", ang, degrees = TRUE)
  expect_jax_equal(scipy_spatial_transform_Rotation_as_rotvec(r), rv_r)
  expect_jax_equal(scipy_spatial_transform_Rotation_as_rotvec(r, degrees = TRUE), rvd_r, tolerance = 1e-5)
  expect_jax_equal(scipy_spatial_transform_Rotation_as_mrp(r), mrp_r)
  expect_jax_equal(scipy_spatial_transform_Rotation_magnitude(r), mag_r)
  expect_jax_equal(scipy_spatial_transform_Rotation_inv(r)$quat, inv_r)
  qc <- matrix(c(0.1, -0.2, 0.3, -0.9, 0, 0, -1, 0, 0, -0.6, 0.8, 0, -0.5, 0.5, 0.5, 0.5), 4L,
               byrow = TRUE)
  expect_jax_equal(scipy_spatial_transform_Rotation_as_quat(
    scipy_spatial_transform_Rotation_from_quat(qc), canonical = TRUE), canon_r)
  expect_jax_equal(scipy_spatial_transform_Rotation_as_quat(r, scalar_first = TRUE), sfirst_r)
  q <- matrix(c(0.3, -0.1, 0.4, 0.8, 1, 2, 3, 4, 0, 0, 0, 2), 3L, byrow = TRUE)
  expect_jax_equal(scipy_spatial_transform_Rotation_from_quat(q)$quat, fquat_r)
  mats <- array(0, c(5L, 3L, 3L))
  mats[1L, , ] <- diag(c(1, -1, -1))
  mats[2L, , ] <- diag(c(-1, 1, -1))
  mats[3L, , ] <- diag(c(-1, -1, 1))
  mats[4L, , ] <- diag(3)
  mats[5L, , ] <- eulm_xyz[2L, , ]
  expect_jax_equal(scipy_spatial_transform_Rotation_from_matrix(mats)$quat, fmat_r)
  expect_jax_equal(scipy_spatial_transform_Rotation_from_mrp(
    matrix(c(0.1, -0.2, 0.3, 1.5, 0, -0.5), 2L, byrow = TRUE))$quat, fmrp_r)
  expect_jax_equal(scipy_spatial_transform_Rotation_from_rotvec(
    matrix(c(0.1, -0.2, 0.3, 1.5, 0, -0.5, 1e-4, 0, 0), 3L, byrow = TRUE))$quat, frv_r)
  expect_jax_equal(scipy_spatial_transform_Rotation_from_rotvec(c(30, 0, 45), degrees = TRUE)$quat,
                   frvd_r)
  expect_jax_equal(scipy_spatial_transform_Rotation_identity()$quat, ident_r, dtype = "float32")
  expect_jax_equal(scipy_spatial_transform_Rotation_concatenate(list(r[sl(NULL, 2L)], r[sl(2L)]))$quat,
                   concat_r)
  expect_identical(scipy_spatial_transform_Rotation_len(r), 3L)
  expect_true(scipy_spatial_transform_Rotation_single(r[0L]))
  expect_false(scipy_spatial_transform_Rotation_single(r))
})

test_that("Rotation apply, composition and mean match jax", {
  skip_if_no_backend()
  skip_if_no_rot()
  r <- scipy_spatial_transform_Rotation_from_euler("xyz", ang, degrees = TRUE)
  expect_jax_equal(scipy_spatial_transform_Rotation_apply(r, vecs), apply_r)
  expect_jax_equal(scipy_spatial_transform_Rotation_apply(r, vecs, inverse = TRUE), apply_inv_r)
  expect_jax_equal(scipy_spatial_transform_Rotation_apply(r[0L], vecs), apply_one_r)
  expect_jax_equal((r * r[1L])$quat, compose_r)
  expect_jax_equal((r[2L] * r)$quat, compose2_r)
  # Eigenvectors are defined up to sign: compare the rotation matrices.
  r2 <- scipy_spatial_transform_Rotation_from_euler("xyz", ang[1:2, ], degrees = TRUE)
  expect_jax_equal(scipy_spatial_transform_Rotation_as_matrix(
    scipy_spatial_transform_Rotation_mean(r2)), mean_r, tolerance = 1e-5)
  expect_jax_equal(scipy_spatial_transform_Rotation_as_matrix(
    scipy_spatial_transform_Rotation_mean(r, weights = c(1, 2, 0.5))), meanw_r, tolerance = 1e-5)
})

test_that("Slerp matches jax", {
  skip_if_no_backend()
  skip_if_no_rot()
  r <- scipy_spatial_transform_Rotation_from_euler("xyz", ang, degrees = TRUE)
  s <- scipy_spatial_transform_Slerp_init(c(0, 1, 3), r)
  expect_jax_equal(s$rotvecs, sl_rotvecs)
  expect_jax_equal(s$timedelta, c(1, 2))
  expect_jax_equal(s(c(0, 0.25, 1, 2.2, 3))$quat, sl_q, tolerance = 1e-5)
  expect_jax_equal(s(0.5)$quat, sl_one, tolerance = 1e-5)
  expect_true(scipy_spatial_transform_Rotation_single(s(0.5)))
  expect_jax_equal(jit(function(sp, t) sp(t)$quat)(s, c(0, 0.25, 1, 2.2, 3)), sl_q, tolerance = 1e-5)
})

test_that("Rotation is a pytree usable under jit, vmap and grad", {
  skip_if_no_backend()
  skip_if_no_rot()
  r <- scipy_spatial_transform_Rotation_from_euler("xyz", ang, degrees = TRUE)
  expect_jax_equal(jit(function(rot, v) scipy_spatial_transform_Rotation_apply(rot, v))(r, vecs),
                   apply_r)
  expect_jax_equal(jit(function(a) scipy_spatial_transform_Rotation_from_euler("xyz", a,
                                                                               degrees = TRUE)$quat)(ang),
                   eul_xyz)
  got <- vmap(function(rot) scipy_spatial_transform_Rotation_as_matrix(rot))(r)
  expect_jax_equal(got, eulm_xyz)
  got <- vmap(function(q) scipy_spatial_transform_Rotation_as_rotvec(Rot(q)))(r$quat)
  expect_jax_equal(got, rv_r)
  with_x64({
    rvec <- as_jax(c(0.4, -0.3, 0.9), "float64")
    v64 <- as_jax(c(1, 2, 3), "float64")
    check_grad_fd(function(p) jnp_sum(scipy_spatial_transform_Rotation_apply(
      scipy_spatial_transform_Rotation_from_rotvec(p), v64)^2L * c(1, 2, 3)), rvec)
    check_grad_fd(function(p) jnp_sum(scipy_spatial_transform_Rotation_as_euler(
      scipy_spatial_transform_Rotation_from_rotvec(p), "xyz")), rvec)
    a64 <- as_jax(c(0.3, 0.5, -0.2), "float64")
    check_grad_fd(function(p) jnp_sum(scipy_spatial_transform_Rotation_as_rotvec(
      scipy_spatial_transform_Rotation_from_euler("zxz", p))), a64)
    m64 <- scipy_spatial_transform_Rotation_as_matrix(scipy_spatial_transform_Rotation_from_rotvec(rvec))
    check_grad_fd(function(m) jnp_sum(scipy_spatial_transform_Rotation_from_matrix(m)$quat *
                                        c(1, -2, 3, 0.5)), m64)
  })
})

test_that("Rotation and Slerp raise jax errors", {
  skip_if_no_backend()
  skip_if_no_rot()
  r <- scipy_spatial_transform_Rotation_from_euler("xyz", ang, degrees = TRUE)
  expect_error(scipy_spatial_transform_Rotation_from_euler("xyzx", c(1, 2, 3, 4)),
               "non-empty string of upto 3 characters")
  expect_error(scipy_spatial_transform_Rotation_from_euler("xYz", c(1, 2, 3)),
               "Expected axes from `seq`")
  expect_error(scipy_spatial_transform_Rotation_from_euler("xxz", c(1, 2, 3)),
               "consecutive axes to be different")
  expect_error(scipy_spatial_transform_Rotation_as_euler(r, "xy"), "Expected 3 axes, got xy.")
  expect_error(r[0L][0L], "Single rotation is not subscriptable.", class = "jax_type_error")
  expect_error(scipy_spatial_transform_Rotation_len(r[0L]), "Single rotation has no len()")
  expect_error(scipy_spatial_transform_Rotation_mean(r, weights = c(1, 2)),
               "got 2 values and 3 rotations")
  expect_error(scipy_spatial_transform_Rotation_mean(r, weights = diag(3)),
               "Expected `weights` to be 1 dimensional, got shape \\(3, 3\\)")
  expect_error(scipy_spatial_transform_Rotation_random(NULL), class = "jax_not_implemented_error")
  expect_error(scipy_spatial_transform_Rotation_identity(num = 2L))
  expect_error(scipy_spatial_transform_Slerp_init(c(0, 1), r$quat), "must be a `Rotation` instance")
  expect_error(scipy_spatial_transform_Slerp_init(c(0, 1), r[0L]), "at least 2 rotations")
  expect_error(scipy_spatial_transform_Slerp_init(c(0, 1), r), "got 3 rotations and 2 timestamps")
  expect_error(scipy_spatial_transform_Slerp_init(diag(3), r), "1 dimensional array, got 2")
  s <- scipy_spatial_transform_Slerp_init(c(0, 1, 3), r)
  expect_error(s(diag(2)), "`times` must be at most 1-dimensional.")
})
