# jax.numpy reductions (R/numpy_reductions.R). Expected values from jax 0.12.0.1: float32
# by default, float64 (hex literals, bit exact up to accumulation order) under x64.

red_exp <- list(
  nansum_f32 = 9,
  nansum_f64 = 9,
  nansum0_f32 = c(1.2, 2.2, -2.1, 7.7),
  nansum0_f64 = c(0x1.3333333333333p+0, 0x1.199999999999ap+1, -0x1.0cccccccccccdp+1, 0x1.ecccccccccccdp+2),
  nansum1k_f32 = array(c(2.6000001, 6.3999996, 0), dim = c(3, 1)),
  nansum1k_f64 = array(c(0x1.4cccccccccccep+1, 0x1.999999999999ap+2, 0), dim = c(3, 1)),
  nanprod1_f32 = c(-2.772, 6.534, 1),
  nanprod1_f64 = c(-0x1.62d0e56041894p+1, 0x1.a22d0e560418ap+2, 1),
  nanmax1_f32 = c(4.4, 3.3, NaN),
  nanmax1_f64 = c(0x1.199999999999ap+2, 0x1.a666666666666p+1, NaN),
  nanmin0_f32 = c(0.3, 2.2, -2.1, 3.3),
  nanmin0_f64 = c(0x1.3333333333333p-2, 0x1.199999999999ap+1, -0x1.0cccccccccccdp+1, 0x1.a666666666666p+1),
  nanmax_init_f32 = c(4.4, 3.3, -5),
  nanmax_init_f64 = c(0x1.199999999999ap+2, 0x1.a666666666666p+1, -5),
  nanmin_where_f32 = c(-2.1, 2.2, 10),
  nanmin_where_f64 = c(-0x1.0cccccccccccdp+1, 0x1.199999999999ap+1, 10),
  nansum_where_f32 = c(0.3, 2.2, -2.1, 7.7),
  nansum_where_f64 = c(0x1.3333333333333p-2, 0x1.199999999999ap+1, -0x1.0cccccccccccdp+1, 0x1.ecccccccccccdp+2),
  nanmean1_f32 = c(0.86666673, 2.1333332, NaN),
  nanmean1_f64 = c(0x1.bbbbbbbbbbbbdp-1, 0x1.1111111111111p+1, NaN),
  nanmean_f32 = 1.5,
  nanmean_f64 = 0x1.8000000000000p+0,
  nanmean_where_f32 = c(0.86666673, 2.75, NaN),
  nanmean_where_f64 = c(0x1.bbbbbbbbbbbbdp-1, 0x1.6000000000000p+1, NaN),
  nanvar1_f32 = c(7.2022223, 0.9622223, NaN),
  nanvar1_f64 = c(0x1.ccf13579be025p+2, 0x1.eca8641fdb975p-1, NaN),
  nanvar1d_f32 = c(10.803333, 1.4433334, NaN),
  nanvar1d_f64 = c(0x1.59b4e81b4e81cp+3, 0x1.717e4b17e4b18p+0, NaN),
  nanvar0_f32 = c(0.17999998, NaN, NaN, 0.60500014),
  nanvar0_f64 = c(0x1.70a3d70a3d70bp-3, NaN, NaN, 0x1.35c28f5c28f61p-1),
  nanstd_f32 = 2.1173882,
  nanstd_f64 = 0x1.0f0694abc0052p+1,
  nanstd1k_f32 = array(c(2.6836956, 0.9809293, NaN), dim = c(3, 1)),
  nanstd1k_f64 = array(c(0x1.5783569cbab3bp+1, 0x1.f63c5c4e9d55cp-1, NaN), dim = c(3, 1)),
  nanargmax1_f32 = c(3, 3, -1),
  nanargmax1_f64 = c(3, 3, -1),
  nanargmin0_f32 = c(0, 1, 0, 1),
  nanargmin0_f64 = c(0, 1, 0, 1),
  nanargmax_f32 = 3,
  nanargmax_f64 = 3,
  nanargmink_f32 = array(c(2, 0, -1), dim = c(3, 1)),
  nanargmink_f64 = array(c(2, 0, -1), dim = c(3, 1)),
  q_lin_f32 = array(c(-1.3799999, 1, 2.7800004, -0.07999998, 1.55, 2.64, -1.0799999, 0.85, 1.8200003), dim = c(3, 3)),
  q_lin_f64 = array(c(-0x1.6147ae147ae14p+0, 1, 0x1.63d70a3d70a40p+1, -0x1.47ae147ae1474p-4, 0x1.8cccccccccccdp+0, 0x1.51eb851eb8520p+1, -0x1.147ae147ae147p+0, 0x1.b333333333334p-1, 0x1.d1eb851eb8522p+0), dim = c(3, 3)),
  q_lin_all_f32 = 0.39000008,
  q_lin_all_f64 = 0x1.8f5c28f5c28f5p-2,
  q_lower_f32 = c(0.3, -1.8, -2.1, 2.9),
  q_lower_f64 = c(0x1.3333333333333p-2, -0x1.ccccccccccccdp+0, -0x1.0cccccccccccdp+1, 0x1.7333333333333p+1),
  q_higher_f32 = c(0.9, 1.7, -0.5, 3.3),
  q_higher_f64 = c(0x1.ccccccccccccdp-1, 0x1.b333333333333p+0, -0x1.0000000000000p-1, 0x1.a666666666666p+1),
  q_nearest_f32 = array(c(0.3, 1.7, 0.9, 2.2, 0.6, 1.1), dim = c(2, 3)),
  q_nearest_f64 = array(c(0x1.3333333333333p-2, 0x1.b333333333333p+0, 0x1.ccccccccccccdp-1, 0x1.199999999999ap+1, 0x1.3333333333333p-1, 0x1.199999999999ap+0), dim = c(2, 3)),
  q_mid_f32 = array(c(-0.9, 0.19999999, -0.59999996), dim = c(3, 1)),
  q_mid_f64 = array(c(-0x1.ccccccccccccdp-1, 0x1.999999999999ap-3, -0x1.3333333333334p-1), dim = c(3, 1)),
  q_inv_f32 = array(c(-2.1, 0.3, 4.4, -0.5, 0.9, 3.3, -1.8, 0.6, 2.9), dim = c(3, 3)),
  q_inv_f64 = array(c(-0x1.0cccccccccccdp+1, 0x1.3333333333333p-2, 0x1.199999999999ap+2, -0x1.0000000000000p-1, 0x1.ccccccccccccdp-1, 0x1.a666666666666p+1, -0x1.ccccccccccccdp+0, 0x1.3333333333333p-1, 0x1.7333333333333p+1), dim = c(3, 3)),
  q_inv_w_f32 = c(1.7, 2.2, -1.8),
  q_inv_w_f64 = c(0x1.b333333333333p+0, 0x1.199999999999ap+1, -0x1.ccccccccccccdp+0),
  q_tuple_f32 = array(c(0.10000001, 2.375), dim = c(2, 1, 1)),
  q_tuple_f64 = array(c(0x1.9999999999999p-4, 0x1.3000000000000p+1), dim = c(2, 1, 1)),
  q_nan_f32 = c(NaN, NaN, NaN),
  q_nan_f64 = c(NaN, NaN, NaN),
  nq_lin_f32 = array(c(-1.6199999, 0.3, 2.7600002, 1.16, 2.2, 2.8600001, NaN, NaN, NaN), dim = c(3, 3)),
  nq_lin_f64 = array(c(-0x1.9eb851eb851ecp+0, 0x1.3333333333333p-2, 0x1.6147ae147ae16p+1, 0x1.28f5c28f5c290p+0, 0x1.199999999999ap+1, 0x1.6e147ae147ae1p+1, NaN, NaN, NaN), dim = c(3, 3)),
  nq_k_f32 = array(c(0.54, 2.2, -2.1, 3.74), dim = c(1, 4)),
  nq_k_f64 = array(c(0x1.147ae147ae148p-1, 0x1.199999999999ap+1, -0x1.0cccccccccccdp+1, 0x1.deb851eb851ecp+1), dim = c(1, 4)),
  nq_inv_f32 = c(0.3, 2.2, NaN),
  nq_inv_f64 = c(0x1.3333333333333p-2, 0x1.199999999999ap+1, NaN),
  pct_f32 = c(-1.6700001, 1, 3.7949998),
  pct_f64 = c(-0x1.ab851eb851eb8p+0, 1, 0x1.e5c28f5c28f5fp+1),
  npct_f32 = c(0.3, 2.2, NaN),
  npct_f64 = c(0x1.3333333333333p-2, 0x1.199999999999ap+1, NaN),
  median0_f32 = c(0.9, 1.7, -0.5, 3.3),
  median0_f64 = c(0x1.ccccccccccccdp-1, 0x1.b333333333333p+0, -0x1.0000000000000p-1, 0x1.a666666666666p+1),
  median_f32 = 1,
  median_f64 = 1,
  nanmedian1_f32 = c(0.3, 2.2, NaN),
  nanmedian1_f64 = c(0x1.3333333333333p-2, 0x1.199999999999ap+1, NaN),
  nanmedian_k_f32 = array(c(0.6, 2.2, -2.1, 3.85), dim = c(1, 4)),
  nanmedian_k_f64 = array(c(0x1.3333333333333p-1, 0x1.199999999999ap+1, -0x1.0cccccccccccdp+1, 0x1.ecccccccccccdp+1), dim = c(1, 4)),
  avg_f32 = 1.0833335,
  avg_f64 = 0x1.1555555555555p+0,
  avg_w_f32 = c(1.85, 2, 0.43000004),
  avg_w_f64 = c(0x1.d99999999999ap+0, 2, 0x1.b851eb851eb84p-2),
  avg_w_ret_f32 = array(c(0.90000004, 6, 0.11666676, 6, -0.21666664, 6, 3.2833335, 6), dim = c(2, 4)),
  avg_w_ret_f64 = array(c(0x1.ccccccccccccdp-1, 6, 0x1.ddddddddddde2p-4, 6, -0x1.bbbbbbbbbbbbdp-3, 6, 0x1.a444444444444p+1, 6), dim = c(2, 4)),
  avg_full_w_f32 = array(c(1.8788991), dim = c(1, 1)),
  avg_full_w_f64 = array(c(0x1.e0ff87c01e0fdp+0), dim = c(1, 1)),
  avg_ret_f32 = c(4, 4, 4),
  avg_ret_f64 = c(4, 4, 4),
  ptp1_f32 = c(6.5, 3.8, 4.7),
  ptp1_f64 = c(0x1.a000000000000p+2, 0x1.e666666666666p+1, 0x1.2cccccccccccdp+2),
  ptp_f32 = array(c(6.5), dim = c(1, 1)),
  ptp_f64 = array(c(0x1.a000000000000p+2), dim = c(1, 1)),
  cnz_f32 = c(2, 3, 2, 3),
  cnz_f64 = c(2, 3, 2, 3),
  minmax_min_f32 = c(-2.1, -0.5, -1.8),
  minmax_min_f64 = c(-0x1.0cccccccccccdp+1, -0x1.0000000000000p-1, -0x1.ccccccccccccdp+0),
  minmax_max_f32 = c(4.4, 3.3, 2.9),
  minmax_max_f64 = c(0x1.199999999999ap+2, 0x1.a666666666666p+1, 0x1.7333333333333p+1),
  minmax_w_f32 = array(c(0, 1.1, -1.8, 2.2, -2.1, 1, 0, 4.4), dim = c(2, 4)),
  minmax_w_f64 = array(c(0, 0x1.199999999999ap+0, -0x1.ccccccccccccdp+0, 0x1.199999999999ap+1, -0x1.0cccccccccccdp+1, 1, 0, 0x1.199999999999ap+2), dim = c(2, 4)),
  cov_f32 = array(c(7.375833, 4.4125, 1.9200001, 4.4125, 2.6958332, 0.72333336, 1.9200001, 0.72333336, 3.7533336), dim = c(3, 3)),
  cov_f64 = array(c(0x1.d80da740da742p+2, 0x1.1a66666666666p+2, 0x1.eb851eb851ebap+0, 0x1.1a66666666666p+2, 0x1.5911111111111p+1, 0x1.7258bf258bf22p-1, 0x1.eb851eb851ebap+0, 0x1.7258bf258bf22p-1, 0x1.e06d3a06d3a08p+1), dim = c(3, 3)),
  cov_T_f32 = array(c(0.115555555, -0.36666673, 0.37111112, -0.21555558, -0.36666673, 3.1666667, -1.45, 0.7, 0.37111112, -1.45, 1.2288889, -0.6944444, -0.21555558, 0.7, -0.6944444, 0.40222225), dim = c(4, 4)),
  cov_T_f64 = array(c(0x1.d950c83fb72ecp-4, -0x1.7777777777778p-2, 0x1.7c048d159e26dp-2, -0x1.b97530eca8645p-3, -0x1.7777777777778p-2, 0x1.9555555555555p+1, -0x1.7333333333332p+0, 0x1.6666666666666p-1, 0x1.7c048d159e26dp-2, -0x1.7333333333332p+0, 0x1.3a987654320ffp+0, -0x1.638e38e38e390p-1, -0x1.b97530eca8645p-3, 0x1.6666666666666p-1, -0x1.638e38e38e390p-1, 0x1.9be02468acf18p-2), dim = c(4, 4)),
  cov_y_f32 = array(c(7.375833, 4.4125, 4.4125, 2.6958332), dim = c(2, 2)),
  cov_y_f64 = array(c(0x1.d80da740da742p+2, 0x1.1a66666666666p+2, 0x1.1a66666666666p+2, 0x1.5911111111111p+1), dim = c(2, 2)),
  cov_fw_f32 = array(c(6.178096, 3.5466666, 2.8088098, 3.5466666, 2.0900002, 1.2, 2.8088093, 1.1999999, 4.4457145), dim = c(3, 3)),
  cov_fw_f64 = array(c(0x1.8b65e991cc500p+2, 0x1.c5f92c5f92c60p+1, 0x1.6787120aba454p+1, 0x1.c5f92c5f92c60p+1, 0x1.0b851eb851eb8p+1, 0x1.3333333333331p+0, 0x1.6787120aba455p+1, 0x1.3333333333331p+0, 0x1.1c869536202ecp+2), dim = c(3, 3)),
  cov_aw_f32 = array(c(18.59125, 11.0725, 5.035, 11.0725, 6.684, 2.3069997, 5.035, 2.307, 6.741), dim = c(3, 3)),
  cov_aw_f64 = array(c(0x1.2975c28f5c290p+4, 0x1.6251eb851eb86p+3, 0x1.423d70a3d70a5p+2, 0x1.6251eb851eb86p+3, 0x1.abc6a7ef9db22p+2, 0x1.274bc6a7ef9dap+1, 0x1.423d70a3d70a5p+2, 0x1.274bc6a7ef9dap+1, 0x1.af6c8b4395811p+2), dim = c(3, 3)),
  cov_faw_f32 = array(c(8.07592, 4.6675005, 3.3238974, 4.6675005, 2.7444117, 1.5576471, 3.3238974, 1.5576471, 4.197206), dim = c(3, 3)),
  cov_faw_f64 = array(c(0x1.026dededededfp+3, 0x1.2ab851eb851ecp+2, 0x1.a975757575758p+1, 0x1.2ab851eb851ebp+2, 0x1.5f48e27c15af4p+1, 0x1.8ec1f5285b8eap+0, 0x1.a975757575758p+1, 0x1.8ec1f5285b8eap+0, 0x1.0c9f056bd238ap+2), dim = c(3, 3)),
  cov_1_f32 = 7.3758335,
  cov_1_f64 = 0x1.d80da740da742p+2,
  corr_f32 = array(c(1, 0.98953843, 0.36491117, 0.98953843, 0.99999994, 0.2273963, 0.36491117, 0.22739631, 1), dim = c(3, 3)),
  corr_f64 = array(c(0x1.fffffffffffffp-1, 0x1.faa4c6ea0d2a6p-1, 0x1.75ab4399f77bap-2, 0x1.faa4c6ea0d2a7p-1, 1, 0x1.d1b526968bd71p-3, 0x1.75ab4399f77bap-2, 0x1.d1b526968bd71p-3, 0x1.fffffffffffffp-1), dim = c(3, 3)),
  corr_y_f32 = array(c(1, 0.36491117, 0.36491117, 1), dim = c(2, 2)),
  corr_y_f64 = array(c(0x1.fffffffffffffp-1, 0x1.75ab4399f77bap-2, 0x1.75ab4399f77bap-2, 0x1.fffffffffffffp-1), dim = c(2, 2)),
  corr_T_f32 = array(c(0.9999999, -0.6061427, 0.9848095, -0.99984056, -0.6061427, 0.99999994, -0.7350392, 0.62024575, 0.9848096, -0.73503923, 0.99999994, -0.9877525, -0.9998406, 0.62024575, -0.98775244, 0.9999999), dim = c(4, 4)),
  corr_T_f64 = array(c(1, -0x1.3658542f9ae16p-1, 0x1.f838f3ba4856cp-1, -0x1.ffeb1bf84b14bp-1, -0x1.3658542f9ae16p-1, 0x1.ffffffffffffep-1, -0x1.78570fd6c00a0p-1, 0x1.3d90db9e3441ap-1, 0x1.f838f3ba4856cp-1, -0x1.78570fd6c00a0p-1, 0x1.fffffffffffffp-1, -0x1.f9bab47680030p-1, -0x1.ffeb1bf84b14bp-1, 0x1.3d90db9e3441ap-1, -0x1.f9bab47680030p-1, 1), dim = c(4, 4)),
  sum_where_init_f32 = c(4.1000004, 6.5, 3.7000003),
  sum_where_init_f64 = c(0x1.0666666666667p+2, 0x1.a000000000000p+2, 0x1.d99999999999ap+1),
  prod_where_f32 = c(0.33, -3.96, 1.05, 42.108),
  prod_where_f64 = c(0x1.51eb851eb851fp-2, -0x1.fae147ae147afp+1, 0x1.0cccccccccccdp+0, 0x1.50dd2f1a9fbe7p+5),
  max_where_f32 = c(4.4, 3.3, 2.9),
  max_where_f64 = c(0x1.199999999999ap+2, 0x1.a666666666666p+1, 0x1.7333333333333p+1),
  min_init_f32 = c(0, -1.8, -2.1, 0),
  min_init_f64 = c(0, -0x1.ccccccccccccdp+0, -0x1.0cccccccccccdp+1, 0),
  mean_where_f32 = c(0.86666673, 1.6666666, 0.7333334),
  mean_where_f64 = c(0x1.bbbbbbbbbbbbdp-1, 0x1.aaaaaaaaaaaabp+0, 0x1.7777777777778p-1),
  var_corr_f32 = c(8.851, 3.235, 4.504),
  var_corr_f64 = c(0x1.1b3b645a1cac2p+3, 0x1.9e147ae147ae2p+1, 0x1.204189374bc6bp+2),
  var_where_f32 = array(c(0.16, 4, 0.6399999, 0.40222225), dim = c(1, 4)),
  var_where_f64 = array(c(0x1.47ae147ae147cp-3, 4, 0x1.47ae147ae147cp-1, 0x1.9be02468acf18p-2), dim = c(1, 4)),
  var_mean_f32 = c(5.594375, 2.084375, 2.8775),
  var_mean_f64 = c(0x1.660a3d70a3d71p+2, 0x1.0accccccccccdp+1, 0x1.7051eb851eb85p+1),
  std_ddof_f32 = c(0.4163332, 2.1794496, 1.3576941, 0.7767454),
  std_ddof_f64 = c(0x1.aa5340171f2b6p-2, 0x1.16f8334644df9p+1, 0x1.5b91d792f3dc7p+0, 0x1.8db190e9352b1p-1),
  var_big_ddof_f32 = c(NaN, NaN, NaN, NaN),
  var_big_ddof_f64 = c(NaN, NaN, NaN, NaN),
  all_where_f32 = c(FALSE, FALSE, FALSE),
  all_where_f64 = c(FALSE, FALSE, FALSE),
  any_where_f32 = c(FALSE, TRUE, TRUE, FALSE),
  any_where_f64 = c(FALSE, TRUE, TRUE, FALSE),
  argmax_k_f32 = array(c(3), dim = c(1, 1)),
  argmax_k_f64 = array(c(3), dim = c(1, 1)),
  argmin_k_f32 = array(c(2, 2, 1), dim = c(3, 1)),
  argmin_k_f64 = array(c(2, 2, 1), dim = c(3, 1)),
  i_nansum = c(4, 4, -5),
  i_nanmax = c(4, 5),
  i_nanmean = 0.5,
  i_nanvar = c(4.6666665, 34.666668),
  i_nanargmin = c(1, 2),
  i_median = c(3, 1),
  i_quantile = -0.5,
  i_ptp = c(2, 6, 13),
  i_cov = array(c(7, -16, -16, 52), dim = c(2, 2)),
  i_avg = c(2, 2, -2.5),
  b_nansum = c(2, 2),
  b_cnz = 4,
  i8_sum = c(4, 4, -5),
  u8_prod = c(12, 45),
  g_nansum = array(c(0x1.e921dd42f09bap-1, 0x1.3e43a9692e21cp-1, NaN, NaN, -0x1.2d5004b88ad70p-1, NaN, -0x1.027b304989ecap-1, NaN, NaN, -0x1.3ab577c62c2aap-2, -0x1.f996f2ca70bb6p-1, NaN), dim = c(3, 4)),
  g_nanmean = array(c(0x1.27d27d27d27d3p-1, 0x1.6c16c16c16c17p+0, 0, 0, 0x1.6c16c16c16c17p+0, 0, 0x1.27d27d27d27d3p-1, 0, 0, 0x1.27d27d27d27d3p-1, 0x1.6c16c16c16c17p+0, 0), dim = c(3, 4)),
  g_nanvar = array(c(-0x1.82d82d82d82dap-2, -0x1.a4fa4fa4fa4fap-1, 0, 0, 0x1.6c16c16c16c25p-5, 0, -0x1.fa4fa4fa4fa50p+0, 0, 0, 0x1.2d82d82d82d83p+1, 0x1.8e38e38e38e38p-1, 0), dim = c(3, 4)),
  g_nanmax = array(c(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0), dim = c(3, 4)),
  g_quantile = array(c(0x1.604189374bc68p-3, 0x1.8083126e978d5p+0, 0x1.845a1cac08312p+1, 0x1.15810624dd2f1p+2, 0x1.2df3b645a1cabp+2, -0x1.83126e978d4fdp+0, -0x1.ee978d4fdf3b5p+0, -0x1.cac083126e96fp-4, 0x1.9db22d0e56044p-3, 0x1.1cac083126e9ep+1, 0x1.0e56041893751p+1, 0x1.74bc6a7ef9dbbp+0), dim = c(3, 4)),
  g_nanquantile = array(c(0x1.999999999999ap-1, 0x1.9999999999998p-3, 0, 0, 0x1.999999999999ap-1, 0, 0x1.9999999999998p-3, 0, 0, 0, 0, 0), dim = c(3, 4)),
  g_median = array(c(0, 0x1.0000000000000p-1, 0x1.0000000000000p-1, 0, 0, 0, 0, 0, 0, 0, 0, 0), dim = c(3, 4)),
  g_avg = array(c(0x1.999999999999ap-3, 0x1.999999999999ap-3, 0x1.999999999999ap-3, 0x1.999999999999ap-2, 0x1.999999999999ap-2, 0x1.999999999999ap-2, 0x1.999999999999ap-4, 0x1.999999999999ap-4, 0x1.999999999999ap-4, 0x1.3333333333334p-2, 0x1.3333333333334p-2, 0x1.3333333333334p-2), dim = c(3, 4)),
  g_cov = array(c(-0x1.3f612a8d8a206p+3, -0x1.8f662c2551b14p+2, -0x1.12d489718cdb0p-1, 0x1.00c3b2a1907f3p+2, 0x1.ef9be02468ad0p+1, -0x1.46c7cb35053bfp+3, -0x1.58ccafac42724p+5, -0x1.9e003a4114b52p+4, -0x1.5110280cbe3c9p+3, 0x1.888c83fb72ea7p+5, 0x1.c3e64945dc0bdp+4, 0x1.54829dec6e232p+4), dim = c(3, 4)),
  g_corr = array(c(0x1.1631ed571c6d9p-2, 0x1.7e4d797af40d8p-1, -0x1.5105238bdd65cp+0, -0x1.3e993ffefffdcp+0, -0x1.c7348888c0a9cp+1, 0x1.3dc0f038310f8p+1, 0x1.61e313a3f1fddp-2, 0x1.1da01b791c85cp-1, -0x1.02b8ff2ebe768p+2, 0x1.4127ff8078c5ep-1, 0x1.2039234bbc850p+1, 0x1.70339feb3a905p+1), dim = c(3, 4)),
  g_ptp = array(c(0, 0, 0, 0, 0, -0x1.2cccccccccccdp+3, -13, -0x1.e666666666666p+2, 0, 13, 0x1.e666666666666p+2, 0x1.2cccccccccccdp+3), dim = c(3, 4)),
  g_var_where = array(c(-0x1.999999999999ap-2, 0, 0x1.999999999999ap-2, 0, 2, -2, -0x1.999999999999ap-1, 0x1.999999999999ap-1, 0, 0x1.27d27d27d27d4p-1, -0x1.3e93e93e93e98p-3, -0x1.b05b05b05b05cp-2), dim = c(3, 4)),
  g2_nanvar = array(c(0x1.c71c71c71c71dp-2, 0, -0x1.c71c71c71c71dp-3, -0x1.c71c71c71c71dp-3, 0, 0, 0, 0, -0x1.c71c71c71c71dp-3, 0, 0x1.c71c71c71c71dp-2, -0x1.c71c71c71c71dp-3, -0x1.c71c71c71c71dp-3, 0, -0x1.c71c71c71c71dp-3, 0x1.c71c71c71c71dp-2), dim = c(4, 4)),
  nq_empty = c(NaN, NaN),
  cov_empty_obs = array(c(NaN, NaN, NaN, NaN), dim = c(2, 2)),
  cov_empty_var = c(0, 0),
  end = NULL
)

red_X <- function(dt = "float32") {
  as_jax(matrix(c(0.3, NaN, -2.1, 4.4, 0.9, 2.2, NaN, 3.3, NaN, NaN, NaN, NaN), 3, byrow = TRUE), dt)
}
red_Y <- function(dt = "float32") {
  as_jax(matrix(c(0.3, 1.7, -2.1, 4.4, 0.9, 2.2, -0.5, 3.3, 1.1, -1.8, 0.6, 2.9), 3, byrow = TRUE), dt)
}
red_W <- function() {
  as_jax(matrix(c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE), 3,
                byrow = TRUE))
}
red_fn <- function(name) get(name, envir = asNamespace("jaxR"))
aval_of_weak <- function(x) red_fn("aval_of")(x)$weak
red_flip_rows <- function(y) y[sl(step = -1L)]

# Each case runs on (X with NaNs, Y without) in float32 and, under x64, float64.
red_cases <- list(
  nansum = function(x, y) jnp_nansum(x),
  nansum0 = function(x, y) jnp_nansum(x, axis = 0L),
  nansum1k = function(x, y) jnp_nansum(x, axis = 1L, keepdims = TRUE),
  nanprod1 = function(x, y) jnp_nanprod(x, axis = 1L),
  nanmax1 = function(x, y) jnp_nanmax(x, axis = 1L),
  nanmin0 = function(x, y) jnp_nanmin(x, axis = 0L),
  nanmax_init = function(x, y) jnp_nanmax(x, axis = 1L, initial = -5),
  nanmin_where = function(x, y) jnp_nanmin(x, axis = 1L, initial = 10, where = red_W()),
  nansum_where = function(x, y) jnp_nansum(x, axis = 0L, where = red_W()),
  nanmean1 = function(x, y) jnp_nanmean(x, axis = 1L),
  nanmean = function(x, y) jnp_nanmean(x),
  nanmean_where = function(x, y) jnp_nanmean(x, axis = 1L, where = red_W()),
  nanvar1 = function(x, y) jnp_nanvar(x, axis = 1L),
  nanvar1d = function(x, y) jnp_nanvar(x, axis = 1L, ddof = 1L),
  nanvar0 = function(x, y) jnp_nanvar(x, axis = 0L, ddof = 1L),
  nanstd = function(x, y) jnp_nanstd(x),
  nanstd1k = function(x, y) jnp_nanstd(x, axis = 1L, keepdims = TRUE),
  nanargmax1 = function(x, y) jnp_nanargmax(x, axis = 1L),
  nanargmin0 = function(x, y) jnp_nanargmin(x, axis = 0L),
  nanargmax = function(x, y) jnp_nanargmax(x),
  nanargmink = function(x, y) jnp_nanargmin(x, axis = 1L, keepdims = TRUE),
  q_lin = function(x, y) jnp_quantile(y, c(0.1, 0.5, 0.8), axis = 1L),
  q_lin_all = function(x, y) jnp_quantile(y, 0.3),
  q_lower = function(x, y) jnp_quantile(y, 0.3, axis = 0L, method = "lower"),
  q_higher = function(x, y) jnp_quantile(y, 0.3, axis = 0L, method = "higher"),
  q_nearest = function(x, y) jnp_quantile(y, c(0.2, 0.6), axis = 1L, method = "nearest"),
  q_mid = function(x, y) jnp_quantile(y, 0.3, axis = 1L, method = "midpoint", keepdims = TRUE),
  q_inv = function(x, y) jnp_quantile(y, c(0.25, 0.5, 0.9), axis = 1L, method = "inverted_cdf"),
  q_inv_w = function(x, y) jnp_quantile(y, 0.4, axis = 1L, method = "inverted_cdf",
                                        weights = c(1, 2, 0.5, 1.5)),
  q_tuple = function(x, y) jnp_quantile(y, c(0.25, 0.75), axis = c(0L, 1L), keepdims = TRUE),
  q_nan = function(x, y) jnp_quantile(x, 0.5, axis = 1L),
  nq_lin = function(x, y) jnp_nanquantile(x, c(0.1, 0.5, 0.8), axis = 1L),
  nq_k = function(x, y) jnp_nanquantile(x, 0.4, axis = 0L, keepdims = TRUE),
  nq_inv = function(x, y) jnp_nanquantile(x, 0.5, axis = 1L, method = "inverted_cdf"),
  pct = function(x, y) jnp_percentile(y, c(10, 50, 95)),
  npct = function(x, y) jnp_nanpercentile(x, 30, axis = 1L, method = "nearest"),
  median0 = function(x, y) jnp_median(y, axis = 0L),
  median = function(x, y) jnp_median(y),
  nanmedian1 = function(x, y) jnp_nanmedian(x, axis = 1L),
  nanmedian_k = function(x, y) jnp_nanmedian(x, axis = 0L, keepdims = TRUE),
  avg = function(x, y) jnp_average(y),
  avg_w = function(x, y) jnp_average(y, axis = 1L, weights = c(1, 2, 0.5, 1.5)),
  avg_w_ret = function(x, y) jnp_stack(jnp_average(y, axis = 0L, weights = c(1, 2, 3), returned = TRUE)),
  avg_full_w = function(x, y) jnp_average(y, weights = jnp_abs(y), keepdims = TRUE),
  avg_ret = function(x, y) jnp_average(y, axis = 1L, returned = TRUE)[[2L]],
  ptp1 = function(x, y) jnp_ptp(y, axis = 1L),
  ptp = function(x, y) jnp_ptp(y, keepdims = TRUE),
  cnz = function(x, y) jnp_count_nonzero(jnp_round(y), axis = 0L),
  minmax_min = function(x, y) jnp_minmax(y, axis = 1L)[[1L]],
  minmax_max = function(x, y) jnp_minmax(y, axis = 1L)[[2L]],
  minmax_w = function(x, y) jnp_stack(jnp_minmax(y, axis = 0L, initial = list(0, 1), where = red_W())),
  cov = function(x, y) jnp_cov(y),
  cov_T = function(x, y) jnp_cov(y, rowvar = FALSE, bias = TRUE),
  cov_y = function(x, y) jnp_cov(y[0], y[1]),
  cov_fw = function(x, y) jnp_cov(y, fweights = c(1L, 2L, 1L, 3L)),
  cov_aw = function(x, y) jnp_cov(y, aweights = c(0.5, 1, 2, 1.5), ddof = 2L),
  cov_faw = function(x, y) jnp_cov(y, fweights = c(1L, 2L, 1L, 3L), aweights = c(0.5, 1, 2, 1.5)),
  cov_1 = function(x, y) jnp_cov(y[0]),
  corr = function(x, y) jnp_corrcoef(y),
  corr_y = function(x, y) jnp_corrcoef(y[0], y[2]),
  corr_T = function(x, y) jnp_corrcoef(y, rowvar = FALSE),
  # complete ports of the core reductions (proposed for R/numpy_core.R)
  sum_where_init = function(x, y) red_fn("red_sum")(y, axis = 1L, where = red_W(), initial = 1.5),
  prod_where = function(x, y) red_fn("red_prod")(y, axis = 0L, where = red_W()),
  max_where = function(x, y) red_fn("red_max")(y, axis = 1L, where = red_W(), initial = -1),
  min_init = function(x, y) red_fn("red_min")(y, axis = 0L, initial = 0),
  mean_where = function(x, y) red_fn("red_mean")(y, axis = 1L, where = red_W()),
  var_corr = function(x, y) red_fn("red_var")(y, axis = 1L, correction = 1.5),
  var_where = function(x, y) red_fn("red_var")(y, axis = 0L, where = red_W(), keepdims = TRUE),
  var_mean = function(x, y) red_fn("red_var")(y, axis = 1L, mean = jnp_mean(y, 1L, keepdims = TRUE) + 0.25),
  std_ddof = function(x, y) red_fn("red_std")(y, axis = 0L, ddof = 1L),
  var_big_ddof = function(x, y) red_fn("red_var")(y, axis = 0L, ddof = 3L),
  all_where = function(x, y) red_fn("red_all")(y > 0, axis = 1L, where = red_W()),
  any_where = function(x, y) red_fn("red_any")(y < 0, axis = 0L, where = red_W()),
  argmax_k = function(x, y) red_fn("red_arg_extreme")(y, NULL, TRUE, "argmax"),
  argmin_k = function(x, y) red_fn("red_arg_extreme")(y, -1L, TRUE, "argmin")
)

red_int_cases <- c("nanargmax1", "nanargmin0", "nanargmax", "nanargmink", "cnz", "argmax_k", "argmin_k")
red_bool_cases <- c("all_where", "any_where")

red_case_dtype <- function(nm, x64) {
  if (nm %in% red_int_cases) return(if (x64) "int64" else "int32")
  if (nm %in% red_bool_cases) return("bool")
  if (x64) "float64" else "float32"
}

red_expect <- function(got, exp, dt, tol, info) {
  expect_identical(dtype(got), dt, info = info)
  g <- as.array(got)
  expect_equal(as.vector(g), as.vector(exp), tolerance = tol, info = info)
  if (length(dim(exp)) > 1L) expect_identical(as.integer(dim(g)), as.integer(dim(exp)), info = info)
}

test_that("nan reductions, quantiles and statistics match JAX in float32", {
  skip_if_no_backend()
  for (nm in names(red_cases)) {
    got <- red_cases[[nm]](red_X(), red_Y())
    red_expect(got, red_exp[[paste0(nm, "_f32")]], red_case_dtype(nm, FALSE), 1e-6, nm)
  }
})

test_that("nan reductions, quantiles and statistics match JAX in float64 and under jit", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  for (nm in names(red_cases)) {
    got <- red_cases[[nm]](red_X("float64"), red_Y("float64"))
    red_expect(got, red_exp[[paste0(nm, "_f64")]], red_case_dtype(nm, TRUE), 1e-14, nm)
    jitted <- jit(red_cases[[nm]])(red_X("float64"), red_Y("float64"))
    expect_equal(as.array(jitted), as.array(got), tolerance = 1e-14, info = nm)
  }
})

test_that("integer and boolean inputs follow JAX dtype rules", {
  skip_if_no_backend()
  I <- as_jax(matrix(c(3L, -1L, 4L, 1L, 5L, -9L), 2, byrow = TRUE))
  expect_jax_equal(jnp_nansum(I, axis = 0L), red_exp$i_nansum, dtype = "int32")
  expect_jax_equal(jnp_nanmax(I, axis = 1L), red_exp$i_nanmax, dtype = "int32")
  expect_jax_equal(jnp_nanmean(I), red_exp$i_nanmean, dtype = "float32")
  expect_jax_equal(jnp_nanvar(I, axis = 1L), red_exp$i_nanvar, dtype = "float32")
  expect_jax_equal(jnp_nanargmin(I, axis = 1L), red_exp$i_nanargmin, dtype = "int32")
  expect_jax_equal(jnp_median(I, axis = 1L), red_exp$i_median, dtype = "float32")
  expect_jax_equal(jnp_quantile(I, 0.25), red_exp$i_quantile, dtype = "float32")
  expect_jax_equal(jnp_ptp(I, axis = 0L), red_exp$i_ptp, dtype = "int32")
  expect_jax_equal(jnp_cov(I), red_exp$i_cov, dtype = "float32")
  expect_jax_equal(jnp_average(I, axis = 0L), red_exp$i_avg, dtype = "float32")
  expect_jax_equal(jnp_nansum(I > 0, axis = 1L), red_exp$b_nansum, dtype = "int32")
  expect_jax_equal(jnp_count_nonzero(I > 0), red_exp$b_cnz, dtype = "int32")
  expect_jax_equal(red_fn("red_sum")(as_jax(I, "int8"), axis = 0L), red_exp$i8_sum, dtype = "int32")
  expect_jax_equal(red_fn("red_sum")(as_jax(I, "int8"), axis = 0L, promote_integers = FALSE),
                   red_exp$i8_sum, dtype = "int8")
  expect_jax_equal(red_fn("red_prod")(as_jax(abs(as.array(I)), "uint8"), axis = 1L),
                   red_exp$u8_prod, dtype = "uint32")
  # reductions of R scalars are strongly typed
  expect_false(aval_of_weak(jnp_nanmean(2)))
  expect_false(aval_of_weak(jnp_median(2)))
  expect_false(aval_of_weak(red_fn("red_sum")(2)))
  # dtype argument and empty inputs
  expect_jax_equal(jnp_nansum(red_X(), dtype = "float16"), 9, dtype = "float16", tolerance = 1e-3)
  expect_jax_equal(jnp_nanquantile(jnp_zeros(c(2L, 0L)), 0.5, axis = 1L), red_exp$nq_empty)
  expect_jax_equal(jnp_cov(jnp_zeros(c(2L, 0L))), red_exp$cov_empty_obs)
  expect_identical(jnp_shape(jnp_cov(jnp_zeros(c(0L, 3L)))), c(0L, 0L))
})


test_that("reduction gradients match JAX and finite differences", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  X <- red_X("float64")
  Y <- red_Y("float64")
  rows2 <- function(v) v[sl(0L, 2L)]
  expect_jax_equal(grad(function(x) jnp_nansum(jnp_sin(x)))(X), red_exp$g_nansum)
  expect_jax_equal(grad(function(x) jnp_sum(rows2(jnp_nanmean(x, axis = 1L))^2))(X), red_exp$g_nanmean)
  expect_jax_equal(grad(function(x) jnp_sum(rows2(jnp_nanvar(x, axis = 1L))))(X), red_exp$g_nanvar)
  expect_jax_equal(grad(function(x) jnp_sum(rows2(jnp_nanmax(x, axis = 1L))))(X), red_exp$g_nanmax)
  fq <- function(x) jnp_sum(jnp_quantile(x, c(0.1, 0.5, 0.8), axis = 1L)^2)
  expect_jax_equal(grad(fq)(Y), red_exp$g_quantile)
  check_grad_fd(fq, Y)
  expect_jax_equal(grad(function(x) jnp_sum(rows2(jnp_nanquantile(x, 0.4, axis = 1L))))(X),
                   red_exp$g_nanquantile)
  expect_jax_equal(grad(function(x) jnp_median(x))(Y), red_exp$g_median)
  fa <- function(x) jnp_sum(jnp_average(x, axis = 1L, weights = c(1, 2, 0.5, 1.5)))
  expect_jax_equal(grad(fa)(Y), red_exp$g_avg)
  fc <- function(x) jnp_sum(jnp_cov(x)^2)
  expect_jax_equal(grad(fc)(Y), red_exp$g_cov, tolerance = 1e-12)
  check_grad_fd(fc, Y)
  fr <- function(x) jnp_sum(jnp_corrcoef(x) * jnp_reshape(jnp_arange(9, dtype = "float64"), c(3L, 3L)))
  expect_jax_equal(grad(fr)(Y), red_exp$g_corr, tolerance = 1e-12)
  check_grad_fd(fr, Y)
  fp <- function(x) jnp_sum(jnp_ptp(x, axis = 1L)^2)
  expect_jax_equal(grad(fp)(Y), red_exp$g_ptp)
  fv <- function(x) jnp_sum(red_fn("red_var")(x, axis = 0L, where = red_W()))
  expect_jax_equal(grad(fv)(Y), red_exp$g_var_where)
  check_grad_fd(fv, Y)
  check_grad_fd(function(x) jnp_nanstd(x), X)
  check_grad_fd(function(x) jnp_sum(jnp_nanpercentile(x, c(20, 70), axis = 0L)), Y)
  # second order
  h <- jacfwd(grad(function(x) jnp_nanvar(x)))(as_jax(c(0.5, NaN, 1.5, -2), "float64"))
  expect_jax_equal(h, red_exp$g2_nanvar)
})

test_that("reductions batch under vmap", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  X <- red_X("float64")
  Y <- red_Y("float64")
  expect_jax_equal(vmap(function(r) jnp_nanmean(r))(X), red_exp$nanmean1_f64)
  expect_jax_equal(vmap(function(r) jnp_nanvar(r, ddof = 1L), in_axes = 1L)(X), red_exp$nanvar0_f64)
  expect_jax_equal(vmap(function(r) jnp_quantile(r, c(0.1, 0.5, 0.8)))(Y),
                   t(red_exp$q_lin_f64))
  expect_jax_equal(vmap(function(r) jnp_median(r), in_axes = 1L)(Y), red_exp$median0_f64)
  expect_jax_equal(vmap(function(r) jnp_nanargmax(r))(X), red_exp$nanargmax1_f64)
  expect_jax_equal(vmap(function(q) jnp_quantile(Y, q, axis = 1L))(c(0.1, 0.5, 0.8)),
                   red_exp$q_lin_f64)
  B <- jnp_stack(list(Y, Y * 2, red_flip_rows(Y)))
  loop <- jnp_stack(lapply(0:2, function(i) jnp_cov(B[i])))
  expect_jax_equal(vmap(jnp_cov)(B), loop)
  loop <- jnp_stack(lapply(0:2, function(i) jnp_corrcoef(B[i])))
  expect_jax_equal(vmap(jnp_corrcoef)(B), loop)
})


test_that("reduction errors match JAX", {
  skip_if_no_backend()
  z <- jnp_ones(c(2L, 3L))
  expect_error(jnp_nansum(z, out = z), "The 'out' argument to jnp.sum is not supported.", fixed = TRUE,
               class = "jax_not_implemented_error")
  expect_error(jnp_nanmax(z, where = jnp_ones(c(2L, 3L)), initial = 0),
               "jnp.nanmax: where must be None or a boolean array; got where.dtype=dtype('float32').",
               fixed = TRUE)
  expect_error(jnp_nanmax(z, where = z > 0), paste0("reduction operation max does not have an ",
               "identity, so to use a where mask one has to specify 'initial'"), fixed = TRUE)
  expect_error(jnp_minmax(z, where = z > 0), "reduction operation minmax does not have an identity",
               fixed = TRUE)
  expect_error(jnp_minmax(jnp_zeros(0L)),
               "zero-size array to reduction operation minmax which has no identity", fixed = TRUE)
  expect_error(jnp_ptp(jnp_zeros(c(0L, 2L)), axis = 0L),
               "zero-size array to reduction operation max which has no identity", fixed = TRUE)
  expect_error(jnp_nansum(z, axis = c(1L, -1L)), "duplicate value in 'axis': (1, -1)", fixed = TRUE)
  expect_error(jnp_nanmax(z, initial = jnp_ones(2L)),
               "initial value must be a scalar. Got array of shape (2,)", fixed = TRUE)
  expect_error(jnp_quantile(z, 0.5, method = "foo"),
               "method can only be 'linear', 'lower', 'higher', 'midpoint', 'nearest' or 'inverted_cdf'",
               fixed = TRUE)
  expect_error(jnp_quantile(z, jnp_ones(c(2L, 2L))), "q must be have rank <= 1, got shape (2, 2)",
               fixed = TRUE)
  expect_error(jnp_quantile(jnp_ones(3L, "complex64"), 0.5),
               "quantile does not support complex input, as the operation is poorly defined.", fixed = TRUE)
  expect_error(jnp_quantile(z, 0.5, weights = z),
               "linear doesn't support weights. Only method 'inverted_cdf' supports weights.", fixed = TRUE,
               class = "jax_not_implemented_error")
  expect_error(jnp_quantile(z, 0.5, weights = jnp_ones(3L), method = "inverted_cdf"),
               "Weights shape must match 'a' shape when axis is None.", fixed = TRUE)
  expect_error(jnp_quantile(z, 0.5, axis = 1L, weights = jnp_ones(2L), method = "inverted_cdf"),
               "Weights shape (2,) must match reduction axes (3,)", fixed = TRUE)
  expect_error(jnp_quantile(z, 0.5, overwrite_input = TRUE),
               "jax.numpy.quantile does not support overwrite_input=True or out != None", fixed = TRUE)
  expect_error(jnp_median(jnp_zeros(c(0L, 3L)), axis = 0L),
               "jax.numpy.median reduction axis must be non-empty", fixed = TRUE)
  expect_error(jnp_average(z, weights = jnp_ones(3L)),
               "Axis must be specified when shapes of a and weights differ.", fixed = TRUE)
  expect_error(jnp_average(z, axis = 1L, weights = jnp_ones(2L)),
               "Shape of weights must be consistent with shape of a along specified axis.", fixed = TRUE)
  expect_error(jnp_cov(jnp_ones(c(2L, 2L, 2L))), "m has more than 2 dimensions", fixed = TRUE)
  expect_error(jnp_cov(z, fweights = jnp_ones(3L)), "fweights must be integer.", fixed = TRUE,
               class = "jax_type_error")
  expect_error(jnp_cov(z, fweights = c(1L, 1L)), "incompatible numbers of samples and fweights",
               fixed = TRUE, class = "jax_runtime_error")
  expect_error(jnp_cov(z, aweights = jnp_ones(c(3L, 1L))), "cannot handle multidimensional aweights",
               fixed = TRUE)
  expect_error(jnp_cov(z, dtype = "int32"),
               "cov: dtype must be a subclass of float or complex; got dtype='int32'", fixed = TRUE)
  expect_error(jnp_corrcoef(z, dtype = "int32"),
               "corrcoef: dtype must be a subclass of float or complex; got dtype='int32'", fixed = TRUE)
  expect_error(jnp_nanargmax(jnp_zeros(0L)), "attempt to get argmax of an empty sequence", fixed = TRUE)
  expect_error(red_fn("red_std")(z, dtype = "int32"), "dtype argument to jnp.std must be inexact; got int32",
               fixed = TRUE)
  expect_error(red_fn("red_var")(z, ddof = 1L, correction = 1),
               "ddof and correction can't be provided simultaneously.", fixed = TRUE)
  expect_error(red_fn("red_var")(jnp_ones(3L, "complex64"), dtype = "float32"),
               "jax.numpy.var does not yet support real dtype parameters", fixed = TRUE)
  # axes must be static
  expect_error(jit(function(x, ax) jnp_nansum(x, axis = ax))(z, 0L),
               class = "jax_concretization_error")
})
