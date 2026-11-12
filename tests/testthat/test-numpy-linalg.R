# Tests of R/numpy_linalg.R (port of jax/_src/numpy/linalg.py). Expected values come from
# jax 0.12.0.1 (reference wheel, x64 enabled); arrays are stored in R (column major) order.

fx <- list()
fx$A <- array(c(1.764, 1.868, -0.103, 0.761, 0.4, -0.977, 0.411, 0.122, 0.979, 0.95, 0.144,
  0.444, 2.241, -0.151, 1.454, 0.334), dim = c(4L, 4L))
fx$S <- array(c(13.252, 3.496, 3.382, 2.574, 3.496, 9.369, -0.677, 1.674, 3.382, -0.677, 6.314,
  0.521, 2.574, 1.674, 0.521, 4.903), dim = c(4L, 4L))
fx$B <- array(c(1.494, 0.313, -2.553, 0.864, -0.205, -0.854, 0.654, -0.742), dim = c(4L, 2L))
fx$M <- array(c(2.27, 1.469, -0.348, -0.302, -1.454, 0.155, 0.156, -1.049, 0.046, 0.378, 1.23,
  -1.42, -0.187, -0.888, 1.202, -1.706, 1.533, -1.981, -0.387, 1.951), dim = c(4L, 5L))
fx$T <- array(c(-0.51, 0.777, -0.895, -1.181, 0.067, -0.438, -1.614, 0.387, -0.028, 0.302,
  -1.253, -0.213, -0.511, 0.428, -0.634), dim = c(5L, 3L))
fx$Zc <- array(c(-0.363-0.907i, -0.813+0.129i, -0.402+0.402i, -0.672+0.052i, -1.726+1.139i,
  -1.63-0.685i, -0.36+0.729i, 0.177-1.235i, 0.463-0.871i), dim = c(3L, 3L))
fx$Hc <- array(c(5.07+0.0i, 0.433-1.144i, 0.039+0.011i, 0.433+1.144i, 9.511+0.0i, 3.569+3.182i,
  0.039-0.011i, 3.569-3.182i, 7.422+0.0i), dim = c(3L, 3L))
fx$A3 <- array(c(-0.579, -1.165, -1.536, -0.312, 0.901, 1.488, 0.056, 0.466, 1.896), dim = c(3L,
  3L))
fx$A2 <- array(c(1.179, -1.071, -0.18, 1.054), dim = c(2L, 2L))
fx$Ab <- array(c(-0.403, 0.402, 0.977, -1.27, 0.011, 1.944, 1.222, 1.883, 0.356, 0.969, 1.786,
  -0.414, 0.208, -1.348, 0.707, -1.173, 0.127, -0.747), dim = c(2L, 3L, 3L))
fx$R2 <- array(c(1.0, 4.0, 2.0, 3.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 0.0, -2.0, 1.0, 1.0, -1.0,
  -3.0, 3.0, 7.0, 1.0, -1.0), dim = c(4L, 5L))
fx$chol_S <- array(c(3.640329655402104, 0.9603525864236157, 0.9290367412141499,
  0.7070788207821471, 0.0, 2.9063246394285467, -0.5399268946142641, 0.3423413242135734, 0.0, 0.0,
  2.2714245930574912, 0.021544666192905463, 0.0, 0.0, 0.0, 2.070115404100859), dim = c(4L, 4L))
fx$chol_S_upper <- array(c(3.640329655402104, 0.0, 0.0, 0.0, 0.9603525864236157,
  2.9063246394285467, 0.0, 0.0, 0.9290367412141499, -0.5399268946142641, 2.2714245930574912, 0.0,
  0.7070788207821471, 0.3423413242135734, 0.021544666192905463, 2.070115404100859), dim = c(4L,
  4L))
fx$chol_Hc_upper <- array(c(2.2516660498395407+0.0i, 0.0+0.0i, 0.0+0.0i,
  0.1923020511993138+0.5080682368868705i, 3.0357678744876395+0.0i, 0.0+0.0i,
  0.01732050807568877-0.004885271508527602i, 1.1753702620143498-1.0449615001620114i,
  2.2244631587135277+0.0i), dim = c(3L, 3L))
fx$svd_M_s <- c(4.024020139383975, 2.9152690821120437, 2.219500993133564, 0.5019994404308582)
fx$svd_herm_s <- c(3.2749993465577103, 1.8637893285868568, 0.3713347308566767,
  0.2251247128858228)
fx$matpow_0 <- array(c(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), dim = c(3L, 3L))
fx$matpow_2 <- array(c(0.6127049999999999, -1.0909060000000002, -3.756432, -0.017136000000000023,
  1.868689, 4.6411679999999995, -0.07164000000000001, 1.238162, 4.202208), dim = c(3L, 3L))
fx$matpow_3 <- array(c(-0.22475371499999988, -3.447204943, -9.686578079999999, -0.313203816,
  3.866436517, 11.606584655999999, -0.10950333600000002, 3.1572734899999997, 9.919810464),
  dim = c(3L, 3L))
fx$matpow_5 <- array(c(0.6153100326053732, -18.1901216702029, -55.85980112414918,
  -1.0896525249934321, 21.93767538145578, 67.8945931677968, -0.8318515016494801,
  18.301752450744193, 56.74988541479797), dim = c(3L, 3L))
fx$matpow_m2 <- array(c(1.8291671303654287, -0.058066241074241264, 1.6992583712674199,
  -0.22623273024996327, 2.002435454887668, -2.413842252132312, 0.0978424444889769,
  -0.5909986733653317, 0.9781673397181307), dim = c(3L, 3L))
fx$matpow_b4 <- array(c(3.2387020433690004, 11.152140402693997, 0.7984472514759999,
  23.18321373429, 4.5646584497000005, -13.419936475307999, 1.9445755055219998, -24.486527530581,
  7.138823722316, -6.511272362824998, 3.932614497924, -9.532525750452, 2.3001118132980003,
  15.764493758511003, 2.074191588815, 3.3199246196859993, 3.5233752256770003, 7.919295146037002),
  dim = c(2L, 3L, 3L))
fx$rank_M <- 4L
fx$rank_R2 <- 2L
fx$rank_tol <- 3L
fx$rank_b <- c(2L, 4L)
fx$slogdet_A_sign <- 1.0
fx$slogdet_A_log <- -1.2646914632639306
fx$det_A <- 0.28232639076700006
fx$slogdet_A3_sign <- -1.0
fx$slogdet_A3_log <- 0.07048905404739902
fx$det_A3 <- -1.073032824
fx$slogdet_A2_sign <- 1.0
fx$slogdet_A2_log <- 0.04868158684655648
fx$det_A2 <- 1.049886
fx$slogdet_Zc_sign <- -0.9499072331488717-0.3125319958203561i
fx$slogdet_Zc_log <- 0.8950927882064609
fx$det_Zc <- -2.324957687-0.7649417130000007i
fx$slogdet_Ab_sign <- c(1.0, -1.0)
fx$slogdet_Ab_log <- c(-0.3415658434389778, 1.5551698838033758)
fx$det_Ab <- c(0.7106566740000001, -4.735891008)
fx$slogdet_S_sign <- 1.0
fx$slogdet_S_log <- 7.813950159616843
fx$det_S <- 2474.887351983407
fx$slogdet_qr_sign <- 1.0
fx$slogdet_qr_log <- -1.2646914632639303
fx$slogdet_1_sign <- 1.0
fx$slogdet_1_log <- 0.9162907318741551
fx$Sing <- array(c(1.0, 2.0, 0.5, 1.0, 2.0, 4.0, 1.0, 0.0, 3.0, 6.0, -1.0, 2.0, 4.0, 8.0, 2.0,
  1.0), dim = c(4L, 4L))
fx$slogdet_sing_sign <- 0.0
fx$slogdet_sing_log <- -Inf
fx$det_grad_A <- array(c(0.5969439440000002, -0.16985636200000012, -0.7766830240000002,
  -0.700905905, 0.008788724000000228, -0.2850806920000001, -0.19321162100000022,
  0.6532544369999996, -1.1404624570000004, 0.3434829130000002, 1.3382339690000007, 1.981587111,
  0.15275079399999997, 0.034533300999999975, 0.06123268500000003, -0.4305566889999999),
  dim = c(4L, 4L))
fx$det_grad_A3 <- array(c(1.014888, 0.6748799999999999, -0.19584800000000002, 1.493064,
  -1.0117679999999998, 0.20457399999999987, -0.34958399999999984, 1.3407839999999998,
  -0.8851589999999998), dim = c(3L, 3L))
fx$det_grad_A2 <- array(c(1.0539999999999998, 0.18, 1.071, 1.179), dim = c(2L, 2L))
fx$det_grad_sing <- array(c(-10.0, 5.0, 0.0, 0.0, -15.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0,
  -5.0, 0.0, 0.0), dim = c(4L, 4L))
fx$slogdet_grad_A <- array(c(2.1143752887509883, -0.6016311884218434, -2.7510110616650985,
  -2.482608526591649, 0.03112965803913597, -1.0097557342249068, -0.6843555095047952,
  2.313827039779364, -4.039517715299977, 1.2166163852654917, 4.740024357497724,
  7.018781013055828, 0.5410432711763847, 0.12231694283408251, 0.21688615376567644,
  -1.5250316763881), dim = c(4L, 4L))
fx$eigvals_A <- c(-1.37571204738413+0.0i, -0.47227674979961287+0.0i, 0.14648120564204295+0.0i,
  2.9665075915416987+0.0i)
fx$eigvalsh_S <- c(4.009147183475617, 4.140762348394433, 8.858043221651155, 16.830047246478802)
fx$pinv_T <- array(c(-0.17304737423794297, -0.14629442993450578, -0.45288855678496565,
  0.02644865716073921, -0.5218310243506576, 0.013524749231446661, -0.24923378959628992,
  0.04762017909942534, -0.17832076123452759, -0.5567618780289842, -0.3036940312588679,
  0.3210994834612567, 0.15813750572902868, 0.22104411986754596, -0.3262759022436512), dim = c(3L,
  5L))
fx$pinv_M <- array(c(0.31680795625138275, -0.05537410385959183, -0.019467439080914778,
  0.16767651458984864, 0.1517175490694968, -0.039597239839048934, -0.2870342792507744,
  0.3807190597243188, -0.4540855394605924, -0.2804234704647322, -0.5429098579908496,
  -0.7950346919349578, 1.1822668468671063, -0.44469576251987086, -0.03986787148160675,
  -0.42695853791016425, -0.5532772418672932, 0.5481753029122629, -0.6073004023825984,
  0.016927287802686427), dim = c(5L, 4L))
fx$pinv_R2 <- array(c(-0.0018726591760299875, -0.007490636704119851, 0.02621722846441947,
  0.02059925093632959, 0.03183520599250937, 0.04494382022471913, 0.013108614232209732,
  0.03745318352059925, 0.005617977528089894, 0.0692883895131086, 0.04868913857677906,
  0.028089887640449417, -0.014981273408239687, -0.035580524344569264, 0.005617977528089857,
  0.09925093632958809, 0.06367041198501867, -0.05617977528089883, -0.0917602996254681,
  -0.02059925093632965), dim = c(5L, 4L))
fx$pinv_Zc <- array(c(-0.5329105689315369+0.11559243636002342i,
  -0.24198202271364183+0.16573555085511454i, -0.6483509469263353-0.43115378900785845i,
  -0.32666285359598224+0.774610244682349i, 0.2716080050621176-0.23302544200527156i,
  0.14947440315452964+0.8188423405063548i, -0.38003873118331394-0.9534412322241497i,
  -0.4184057386447092+0.536327611216399i, -0.598239992578516-0.354144197933475i), dim = c(3L,
  3L))
fx$pinv_rtol <- array(c(0.2563797540307132, -0.14282264294361793, 0.09751312325591577,
  0.09562414089992056, 0.14097633101861895, 0.17538196667223938, 0.024072403589194182,
  -0.035450668055769026, -0.1977522124066017, -0.242210542581986, 0.001925251544434739,
  -0.00657776917962662, 0.1275421462787843, 0.204945654384659, 0.05697751690048652,
  -0.06410207783218215, -0.028170318858965364, -0.1542641407181644, -0.1746436577409313,
  0.0814256547180952), dim = c(5L, 4L))
fx$pinv_grad <- array(c(-6.276252680158379, -2.0326136236883228, -1.5654994975242544,
  1.298820337232336, 0.3133017897748023, -5.493960892926251, -0.9338315705464186,
  -0.769425120776869, 2.034318688113326, 1.3590782529394905, -6.119376275333965,
  -0.2312051638666326, -0.17794085446803032, 2.2100169568272032, 2.90237917905269), dim = c(5L,
  3L))
fx$inv_A <- array(c(2.114375288750989, 0.0311296580391361, -4.039517715299977,
  0.5410432711763847, -0.6016311884218437, -1.0097557342249066, 1.2166163852654914,
  0.12231694283408248, -2.751011061665099, -0.6843555095047952, 4.740024357497724,
  0.21688615376567644, -2.4826085265916493, 2.313827039779364, 7.018781013055829,
  -1.5250316763881), dim = c(4L, 4L))
fx$inv_Ab <- array(c(-1.7131901304004356, 0.2553827775928411, -0.16365427111995287,
  0.6818150997447956, 2.4498552728711864, 0.28673717315413355, 0.3043579381061297,
  -0.41484759608724514, -0.07523886280986365, -0.4899221701007526, 1.0317218240885753,
  -0.8080802521712088, 1.1115156289941488, 0.1905759652144426, 0.6868816094450693,
  -0.4610549517105781, -1.8818679243136183, -0.5872069258566858), dim = c(2L, 3L, 3L))
fx$inv_Zc <- array(c(-0.532910568931537+0.11559243636002364i,
  -0.24198202271364172+0.16573555085511446i, -0.648350946926335-0.43115378900785845i,
  -0.3266628535959821+0.774610244682349i, 0.2716080050621175-0.2330254420052716i,
  0.1494744031545296+0.8188423405063548i, -0.3800387311833139-0.9534412322241497i,
  -0.4184057386447092+0.5363276112163989i, -0.5982399925785159-0.35414419793347507i), dim = c(3L,
  3L))
fx$inv_grad <- array(c(-113.95790310954422, 22.95568748484468, 161.2139640558853,
  247.43368469491367, 22.89976415913159, -3.4179576807223095, -31.535287017516133,
  -54.95609455719658, 272.6055772681911, -55.34514516032148, -385.96036522393536,
  -590.011081339162, -22.007996826349423, 3.5227737624918944, 30.47856898683923,
  51.77380209959109), dim = c(4L, 4L))
fx$solve_vec <- c(7.848923592873751, 3.476760342433186, -11.691293927481517,
  -1.0247338687185037)
fx$solve_mat <- array(c(7.848923592873751, 3.476760342433186, -11.691293927481517,
  -1.0247338687185037, 0.12328039312033118, -1.3084783496023766, -2.3188488432641483,
  1.0580445086712573), dim = c(4L, 2L))
fx$solve_c <- array(c(-1.9614811371350287+1.2537975587068217i,
  -1.3980411221508213+0.4449739823684332i, -2.3050970203446313-2.0227819576771418i,
  -1.7923866547506115+2.19560487082674i, -0.9110004899810056-1.8385827353124164i,
  0.6259233273202122+3.692066127822401i), dim = c(3L, 2L))
fx$solve_bcast <- array(c(-1.0699279185268018, 0.04270613864600178, -0.8726410539556826,
  0.8841687912425882, 0.270127003690111, 1.6773425086390843, 2.4626770788618524,
  -0.16988007444448366, 1.0053497520520014, -1.3401959545264943, -2.634992953010668,
  -1.6913062109050971), dim = c(2L, 3L, 2L))
fx$solve_bcast_vec <- array(c(-1.0699279185268018, 0.04270613864600178, -0.8726410539556826,
  0.8841687912425882, 0.270127003690111, 1.6773425086390843), dim = c(2L, 3L))
fx$solve_grad_a <- array(c(102.60604363093344, -28.513724134676767, -150.80406135721776,
  -188.42250064954553, 26.091120744948046, -8.120804627189132, -38.973801130505,
  -44.10104746036623, -183.1613886660394, 49.53652131081301, 268.2177014341948,
  342.32291706870046, 1.8595287445502053, 0.16898743364636934, -2.2391892489968805,
  -6.418604773660865), dim = c(4L, 4L))
fx$solve_grad_b <- array(c(-12.849551918063327, 3.5808557296166486, 18.892703333575362,
  23.552588073453446, -14.202521415396792, 3.308402135069473, 20.41424727366887,
  28.877555923308883), dim = c(4L, 2L))
fx$X3 <- array(c(1.923, 0.376, -0.861, -0.695, 0.947, 0.672, 1.481, -1.099, 1.91, -0.15, -0.155,
  0.407, 1.868, 0.298, -0.268, -0.435, 0.614, -0.77, 0.906, 1.326, 0.802, 1.849, 0.922, 0.539),
  dim = c(2L, 3L, 4L))
fx$norm_none <- 5.124878047329517
fx$norm_vec2 <- 1.7504582257226249
fx$norm_vec1 <- 3.43
fx$norm_vecinf <- 1.181
fx$norm_vecminf <- 0.067
fx$norm_vec0 <- 2.0
fx$norm_vec3 <- 1.4368093114634075
fx$norm_vecm1 <- 0.06261246187269788
fx$norm_axis1 <- array(c(2.309991125524079, 1.0372969680858033, 2.4218765451607975,
  1.1815032797246057, 1.9845009448221487, 0.9332357687101368, 1.5212245067707793,
  2.3382895458005195), dim = c(2L, 4L))
fx$norm_axis1_keep <- array(c(6.178, 3.099, 3.8409999999999997, 3.129, 2.638, 2.388), dim = c(2L,
  3L, 1L))
fx$mnorm_fro <- 2.926489364409172
fx$mnorm_b_fro <- c(3.660301490314698, 3.0361159398152107, 1.9101225091600802)
fx$mnorm_k_fro <- array(c(4.1777880511103005, 2.968242240788309), dim = c(2L, 1L, 1L))
fx$mnorm_1 <- 3.43
fx$mnorm_b_1 <- c(2.58, 2.651, 1.619)
fx$mnorm_k_1 <- array(c(3.731, 3.714), dim = c(2L, 1L, 1L))
fx$mnorm_m1 <- 2.769
fx$mnorm_b_m1 <- c(2.166, 0.7030000000000001, 0.5619999999999999)
fx$mnorm_k_m1 <- array(c(2.6300000000000003, 1.5030000000000001), dim = c(2L, 1L, 1L))
fx$mnorm_inf <- 2.604
fx$mnorm_b_inf <- c(6.178, 3.8409999999999997, 2.638)
fx$mnorm_k_inf <- array(c(6.178, 3.129), dim = c(2L, 1L, 1L))
fx$mnorm_minf <- 1.0030000000000001
fx$mnorm_b_minf <- c(3.099, 3.129, 2.388)
fx$mnorm_k_minf <- array(c(2.638, 2.388), dim = c(2L, 1L, 1L))
fx$mnorm_2 <- 2.092424169882956
fx$mnorm_b_2 <- c(3.209988461824298, 2.5658413995919283, 1.5823337959037642)
fx$mnorm_k_2 <- array(c(3.4277152885488458, 2.463777110263386), dim = c(2L, 1L, 1L))
fx$mnorm_m2 <- 1.1725359843040122
fx$mnorm_b_m2 <- c(1.7589147435151244, 1.6231013252844484, 1.0699475493409871)
fx$mnorm_k_m2 <- array(c(0.7520366602532359, 1.0449450004334035), dim = c(2L, 1L, 1L))
fx$mnorm_nuc <- 4.941641536800208
fx$mnorm_b_nuc <- c(4.968903205339423, 4.188942724876377, 2.6522813452447513)
fx$mnorm_k_nuc <- array(c(6.446715063253588, 4.792604619733902), dim = c(2L, 1L, 1L))
fx$mnorm_c <- 3.6059188842790126
fx$mnorm_c1 <- 4.510039139729435
fx$matrix_norm <- c(6.446715063253588, 4.792604619733902)
fx$vector_norm_all <- 5.124878047329517
fx$vector_norm_ax <- c(9.277000000000001, 6.97, 5.026)
fx$qr_q <- array(c(-0.29135228279410197, 0.4438837720216022, -0.5112946923543552,
  -0.6746804823134005, 0.03827569205334279, -0.4191660565712894, -0.8139181594492075,
  0.00021655567260190302, -0.3426888645105147, 0.21071625941398883, -0.6776534122642389,
  0.020236970728009836, -0.2668198843131947, 0.48045850879628277, -0.48820394152724006),
  dim = c(5L, 3L))
fx$qr_r <- array(c(1.750458225722625, 0.0, 0.0, -0.7562368416153, 1.570574047723862, 0.0,
  0.2187587195015293, 0.41820403441883797, 1.496291752379147), dim = c(3L, 3L))
fx$qr_qc <- array(c(-0.29135228279410197, 0.4438837720216022, -0.5112946923543552,
  -0.6746804823134005, 0.03827569205334279, -0.4191660565712894, -0.8139181594492075,
  0.00021655567260190302, -0.3426888645105147, 0.21071625941398883, -0.6776534122642389,
  0.020236970728009836, -0.2668198843131947, 0.48045850879628277, -0.48820394152724006,
  -0.5293381557260038, 0.3742921188667952, 0.6220330138245588, 0.028301714034582447,
  0.4381557592247745, 0.0007962840991064369, 0.0005490215359774006, -0.5295845255682378,
  0.44241295554471316, 0.7237472428630334), dim = c(5L, 5L))
fx$qr_rc <- array(c(1.750458225722625, 0.0, 0.0, 0.0, 0.0, -0.7562368416153, 1.570574047723862,
  0.0, 0.0, 0.0, 0.2187587195015293, 0.41820403441883797, 1.496291752379147, 0.0, 0.0),
  dim = c(5L, 3L))
fx$qr_r_only <- array(c(1.750458225722625, 0.0, 0.0, -0.7562368416153, 1.570574047723862, 0.0,
  0.2187587195015293, 0.41820403441883797, 1.496291752379147), dim = c(3L, 3L))
fx$qr_raw_h <- array(c(1.750458225722625, -0.7562368416153, 0.2187587195015293,
  -0.3437356157075665, 1.570574047723862, 0.41820403441883797, 0.395937420924417,
  -0.08487234108296988, 1.496291752379147, 0.522460440348309, 0.06317219245678819,
  -0.8341259912960438, -0.0296400080468558, -0.10127279230714321, 0.5211970760851412),
  dim = c(3L, 5L))
fx$qr_raw_tau <- c(1.291352282794102, 1.9580004619884523, 1.016563601942368)
fx$B2 <- array(c(-0.674, -0.636, 0.577, 0.396, -1.491, 0.032, 0.676, -0.208, -1.093, 0.439),
  dim = c(5L, 2L))
fx$lstsq_x <- array(c(-0.5002560370563753, 0.008124201502220818, 0.807386833225486,
  0.7421455022017971, 0.06163175331616609, -0.4624558415079957), dim = c(3L, 2L))
fx$lstsq_resid <- c(1.4920324466066128, 0.0749425110132463)
fx$lstsq_rank <- 3L
fx$lstsq_s <- c(2.092424169882956, 1.676681382613239, 1.1725359843040117)
fx$lstsq_x_vec <- c(-0.5002560370563753, 0.008124201502220818, 0.807386833225486)
fx$lstsq_resid_vec <- c(1.4920324466066128)
fx$lstsq_wide_x <- array(c(0.37762654565034914, -0.2133884770129067, -0.3243107699873478,
  -0.593152351910984, 0.059694903743147294, -0.15351119285284914, 0.025321324688146317,
  0.2081612363313669, 0.4128974925220922, 0.1547931157582907), dim = c(5L, 2L))
fx$lstsq_wide_resid <- c(2.531439413273707, 0.014792653366495274)
fx$lstsq_wide_rank <- 3L
fx$lstsq_np_resid_len <- 0L
fx$lstsq_np_resid <- c(1.4920324466066128, 0.0749425110132463)
fx$v1 <- array(c(0.167, 0.944, 0.635, -0.913, 2.383, 1.117), dim = c(2L, 3L))
fx$v2 <- array(c(-1.316, 1.713, -0.462, -0.745, -0.068, -0.826), dim = c(2L, 3L))
fx$cross <- array(c(1.057766, 1.586303, -3.1246720000000003, 2.693165, 0.758506,
  0.8606890000000003), dim = c(2L, 3L))
fx$cross_ax0 <- array(c(1.057766, -3.1246720000000003, 0.758506, 1.586303, 2.693165,
  0.8606890000000003), dim = c(3L, 2L))
fx$outer <- array(c(0.286071, 1.087755, 4.082079, -0.12441500000000001, -0.473075,
  -1.7753349999999999, -0.137942, -0.52451, -1.9683579999999998), dim = c(3L, 3L))
fx$vecdot <- c(-0.675186, 1.3746150000000001)
fx$vecdot_c <- c(-2.351271+5.452604999999999i, -24.644613999999997-14.366207999999999i,
  7.975810000000001+10.284592i)
fx$vecdot_b <- c(-0.6751860000000001, -0.8964540000000001)
fx$tensordot_2 <- array(c(17.453913, 3.362299, 3.362299, 8.810462000000001), dim = c(2L, 2L))
fx$tensordot_ax <- array(c(-1.7902029999999998, -0.7293660000000001, -1.397675,
  0.1162190000000001, -1.604638, 1.00336, -1.968602, -2.161489, 2.6821140000000003,
  1.2707460000000002, -1.898988, -0.698514, 1.7532060000000003, 1.0976460000000001, -0.786852,
  -2.068791, -2.5382089999999997, -0.948877, -0.5071200000000001, 0.7175779999999999, -2.08933,
  -0.04158500000000001, -0.9716380000000001, -0.7466360000000001, -1.8416390000000002,
  -0.13698000000000002, -1.8688810000000002, 1.476315, -1.9358120000000005, -0.669318,
  -0.6978260000000002, -1.387086, -0.731579, -0.610746, 0.7743169999999999, -0.376971, -0.345056,
  0.37677600000000006, -0.281642, 0.3055139999999999), dim = c(2L, 4L, 5L))
fx$tensordot_0 <- array(c(-0.21977200000000002, -0.8356600000000001, -3.136028,
  -0.07715400000000001, -0.29337, -1.100946, -0.011356000000000002, -0.04318,
  -0.16204400000000002), dim = c(3L, 3L))
fx$diag_0 <- array(c(1.923, 0.376, 1.91, -0.15, 0.614, -0.77), dim = c(2L, 3L))
fx$diag_1 <- array(c(1.481, -1.099, -0.268, -0.435, 0.922, 0.539), dim = c(2L, 3L))
fx$diag_m1 <- array(c(-0.861, -0.695, -0.155, 0.407), dim = c(2L, 2L))
fx$diag_big <- array(c(0.906, 1.326), dim = c(2L, 1L))
fx$trace_0 <- c(4.447, -0.544)
fx$trace_1 <- c(2.1350000000000002, -0.995)
fx$trace_int <- 12.0
fx$tensorinv_shape <- c(8L, 3L, 4L, 6L)
fx$tensorinv_sum <- 7099.679282077907
fx$ts_a <- array(c(0.9495, -0.1824, 0.0564, -0.0015, 0.1755, -0.0451, -0.0816, 0.0304,
  0.8714999999999999, 0.0821, -0.013300000000000001, 0.026600000000000002, -0.050800000000000005,
  0.07730000000000001, -0.0104, 0.0671, 0.9234, 0.0723, -0.10520000000000002, 0.8338, -0.0988,
  -0.0708, 0.05560000000000001, 0.0025000000000000005, 0.2497, 0.044800000000000006, -0.1178,
  1.004, 0.001, 0.072, -0.22450000000000003, 0.1696, -0.11399999999999999, -0.1567, 0.072,
  0.8896999999999999), dim = c(2L, 3L, 6L))
fx$ts_b <- array(c(-0.102, -0.214, 0.019, -0.499, 1.85, 0.021), dim = c(2L, 3L))
fx$tensorsolve <- c(0.10651814487333215, -0.10181487657662763, 2.0104327544002687,
  -0.36490114727438583, -0.6606148014250732, -0.07684109870925326)
fx$tensorsolve_axes <- array(c(0.10651814487333215, -0.36490114727438583, -0.10181487657662763,
  -0.6606148014250732, 2.0104327544002687, -0.07684109870925326), dim = c(2L, 3L))
fx$multi_dot <- array(c(1.8830476658900002, -5.971760533571001, -9.54841463843,
  -0.7489709255690009, 7.981486377845001, 12.336567781786997), dim = c(3L, 2L))
fx$multi_dot_vec <- -0.8201120589999998
fx$cond_none <- 37.60875191570556
fx$cond_m2 <- 0.02658955559709484
fx$cond_1 <- 59.97775615814398
fx$cond_m1 <- 5.63511167892548
fx$cond_inf <- 91.60843411250481
fx$cond_fro <- 45.80502846779961
fx$cond_nuc <- 86.94221856052252
fx$cond_b <- c(8.456808923956524, 3.9148756283391255)
fx$cond_T <- 1.7845287461475787
fx$cond_sing <- Inf
fx$mT <- array(c(1.923, 0.376, 1.481, -1.099, 1.868, 0.298, 0.906, 1.326, -0.861, -0.695, 1.91,
  -0.15, -0.268, -0.435, 0.802, 1.849, 0.947, 0.672, -0.155, 0.407, 0.614, -0.77, 0.922, 0.539),
  dim = c(2L, 4L, 3L))

x64 <- function(code) with_jax_config("jax_enable_x64", TRUE, code)
host <- function(x) as.array(x)
H <- function(x) Conj(t(x))
resid <- function(x) max(Mod(x))
arange <- function(...) {
  d <- c(...)
  aperm(array(seq_len(prod(d)) - 1, rev(d)))
}
csort <- function(w) w[order(round(Re(w), 8), Im(w))]
# the sort and gather primitives live in files of other work packages
have_sort <- function() {
  ns <- asNamespace("jaxR")
  exists("lax_sort", envir = ns, inherits = FALSE) && exists("jnp_take_along_axis", envir = ns, inherits = FALSE)
}

test_that("jnp_linalg cholesky, svd and svdvals", {
  skip_if_no_backend()
  x64({
    expect_jax_equal(jnp_linalg_cholesky(fx$S), fx$chol_S, tolerance = 1e-10, dtype = "float64")
    expect_jax_equal(jnp_linalg_cholesky(fx$S, upper = TRUE), fx$chol_S_upper, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_cholesky(fx$Hc, upper = TRUE), fx$chol_Hc_upper, tolerance = 1e-10)
    # integer input is promoted to the default float
    expect_jax_equal(jnp_linalg_cholesky(as_jax(4L * diag(2L))), 2 * diag(2), dtype = "float64")
    f <- jnp_linalg_svd(fx$M, full_matrices = FALSE)
    expect_named(f, c("U", "S", "Vh"))
    expect_jax_equal(f$S, fx$svd_M_s, tolerance = 1e-10)
    expect_lt(resid(host(f$U) %*% diag(fx$svd_M_s) %*% host(f$Vh) - fx$M), 1e-11)
    expect_identical(dim(jnp_linalg_svd(fx$M)$Vh), c(5L, 5L))
    expect_jax_equal(jnp_linalg_svd(fx$M, compute_uv = FALSE), fx$svd_M_s, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_svdvals(fx$M), fx$svd_M_s, tolerance = 1e-10)
    expect_jax_equal(jit(jnp_linalg_svdvals)(as_jax(fx$M)), fx$svd_M_s, tolerance = 1e-10)
  })
  expect_identical(dtype(jnp_linalg_svdvals(as_jax(fx$M, "float32"))), "float32")
  expect_error(jnp_linalg_cholesky(matrix(1, 2, 3)), "Argument to linalg.cholesky must have shape \\[..., n, n\\], got \\(2, 3\\)")
})

test_that("jnp_linalg_svd with hermitian = TRUE", {
  skip_if_no_backend()
  skip_if_not(have_sort(), "lax_sort and jnp_take_along_axis are not in this build")
  x64({
    sym <- (fx$A + t(fx$A)) / 2
    f <- jnp_linalg_svd(sym, hermitian = TRUE)
    expect_jax_equal(f$S, fx$svd_herm_s, tolerance = 1e-10)
    expect_lt(resid(host(f$U) %*% diag(fx$svd_herm_s) %*% host(f$Vh) - sym), 1e-11)
    expect_jax_equal(jnp_linalg_svd(sym, compute_uv = FALSE, hermitian = TRUE), fx$svd_herm_s, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_matrix_rank(sym, hermitian = TRUE), 4)
    expect_lt(resid(host(jnp_linalg_pinv(sym, hermitian = TRUE)) - solve(sym)), 1e-10)
  })
})

test_that("jnp_linalg matrix_power and matrix_rank", {
  skip_if_no_backend()
  x64({
    expect_jax_equal(jnp_linalg_matrix_power(fx$A3, 0), fx$matpow_0, dtype = "float64")
    expect_jax_equal(jnp_linalg_matrix_power(fx$A3, 1), fx$A3)
    expect_jax_equal(jnp_linalg_matrix_power(fx$A3, 2), fx$matpow_2, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_matrix_power(fx$A3, 3), fx$matpow_3, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_matrix_power(fx$A3, 5), fx$matpow_5, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_matrix_power(fx$A3, -2), fx$matpow_m2, tolerance = 1e-9)
    expect_jax_equal(jnp_linalg_matrix_power(fx$Ab, 4), fx$matpow_b4, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_matrix_rank(fx$M), fx$rank_M)
    expect_jax_equal(jnp_linalg_matrix_rank(fx$R2), fx$rank_R2)
    expect_jax_equal(jnp_linalg_matrix_rank(fx$M, rtol = 2), fx$rank_tol)
    expect_jax_equal(jnp_linalg_matrix_rank(fx$M, tol = 2), fx$rank_tol)
    two <- aperm(array(c(fx$R2, fx$M), c(4, 5, 2)), c(3, 1, 2))
    expect_jax_equal(jnp_linalg_matrix_rank(two), fx$rank_b)
    expect_jax_equal(jnp_linalg_matrix_rank(c(0, 0, 0)), 0, dtype = "int32")
    expect_jax_equal(jnp_linalg_matrix_rank(c(0, 2, 0)), 1)
  })
  expect_error(jnp_linalg_matrix_power(c(1, 2), 2), "1-dimensional array given. Array must be at least two-dimensional")
  expect_error(jnp_linalg_matrix_power(matrix(1, 2, 3), 2), "Last 2 dimensions of the array must be square")
  expect_error(jnp_linalg_matrix_power(diag(2), 1.5), "exponent must be an integer")
  expect_error(jnp_linalg_matrix_rank(diag(2), rtol = 1, tol = 1), "only one of tol or rtol may be specified")
})

test_that("jnp_linalg slogdet and det match jax", {
  skip_if_no_backend()
  x64({
    for (nm in c("A", "A3", "A2", "Zc", "Ab", "S")) {
      f <- jnp_linalg_slogdet(fx[[nm]])
      expect_named(f, c("sign", "logabsdet"))
      expect_jax_equal(f$sign, fx[[paste0("slogdet_", nm, "_sign")]], tolerance = 1e-10)
      expect_jax_equal(f$logabsdet, fx[[paste0("slogdet_", nm, "_log")]], tolerance = 1e-10)
      expect_jax_equal(jnp_linalg_det(fx[[nm]]), fx[[paste0("det_", nm)]], tolerance = 1e-10)
    }
    expect_identical(dtype(jnp_linalg_slogdet(fx$Zc)$logabsdet), "float64")
    f <- jnp_linalg_slogdet(fx$A, method = "qr")
    expect_jax_equal(f$sign, fx$slogdet_qr_sign)
    expect_jax_equal(f$logabsdet, fx$slogdet_qr_log, tolerance = 1e-10)
    f <- jnp_linalg_slogdet(matrix(2.5))
    expect_jax_equal(f$sign, fx$slogdet_1_sign)
    expect_jax_equal(f$logabsdet, fx$slogdet_1_log, tolerance = 1e-12)
    f <- jnp_linalg_slogdet(fx$Sing)
    expect_jax_equal(f$sign, fx$slogdet_sing_sign)
    expect_jax_equal(f$logabsdet, fx$slogdet_sing_log)
    expect_jax_equal(jit(jnp_linalg_det)(as_jax(fx$A)), fx$det_A, tolerance = 1e-10)
    expect_jax_equal(vmap(jnp_linalg_det)(as_jax(fx$Ab)), fx$det_Ab, tolerance = 1e-10)
    expect_error(jnp_linalg_slogdet(fx$Zc, method = "qr"), "not implemented for complex inputs")
  })
  expect_identical(dtype(jnp_linalg_det(as_jax(fx$A, "float32"))), "float32")
  expect_jax_equal(jnp_linalg_det(as_jax(fx$A, "float32")), fx$det_A, tolerance = 1e-4)
  expect_error(jnp_linalg_det(matrix(1, 2, 3)), "Argument to _det\\(\\) must have shape \\[..., n, n\\], got \\(2, 3\\)")
  expect_error(jnp_linalg_slogdet(matrix(1, 2, 3)), "Argument to slogdet\\(\\) must have shape")
  expect_error(jnp_linalg_slogdet(diag(3), method = "svd"), "Unknown slogdet method 'svd'")
})

test_that("jnp_linalg det and slogdet derivatives", {
  skip_if_no_backend()
  x64({
    expect_jax_equal(grad(jnp_linalg_det)(as_jax(fx$A)), fx$det_grad_A, tolerance = 1e-9)
    expect_jax_equal(grad(jnp_linalg_det)(as_jax(fx$A3)), fx$det_grad_A3, tolerance = 1e-10)
    expect_jax_equal(grad(jnp_linalg_det)(as_jax(fx$A2)), fx$det_grad_A2, tolerance = 1e-10)
    # the cofactor rule gives the gradient of a singular matrix (rank n - 1)
    expect_jax_equal(grad(jnp_linalg_det)(as_jax(fx$Sing)), fx$det_grad_sing, tolerance = 1e-9)
    check_grad_fd(jnp_linalg_det, as_jax(fx$A))
    check_grad_fd(function(x) jnp_sum(jnp_linalg_det(x)), as_jax(fx$Ab))
    ld <- function(x) jnp_linalg_slogdet(x)$logabsdet
    expect_jax_equal(grad(ld)(as_jax(fx$A)), fx$slogdet_grad_A, tolerance = 1e-9)
    check_grad_fd(ld, as_jax(fx$A))
    check_grad_fd(ld, as_jax(fx$A2))
    check_grad_fd(function(x) jnp_linalg_slogdet(x, method = "qr")$logabsdet, as_jax(fx$A))
  })
})

test_that("jnp_linalg_det second order", {
  skip_if_no_backend()
  # the JVP of reduce_prod (used by the cofactor rule) needs lax_cumprod
  skip_if_not(exists("lax_cumprod", envir = asNamespace("jaxR"), inherits = FALSE),
              "lax_cumprod is not in this build")
  x64({
    # the Hessian of det of a 2x2 matrix applied to the identity is the identity
    hv <- host(jvp(grad(jnp_linalg_det), list(as_jax(fx$A2)), list(as_jax(diag(2))))[[2L]])
    expect_equal(hv, diag(2), tolerance = 1e-10)
  })
})

test_that("jnp_linalg eigenvalue functions", {
  skip_if_no_backend()
  x64({
    f <- jnp_linalg_eig(fx$A)
    expect_named(f, c("eigenvalues", "eigenvectors"))
    w <- as.vector(host(f$eigenvalues))
    v <- host(f$eigenvectors)
    expect_equal(csort(w), csort(fx$eigvals_A), tolerance = 1e-10)
    expect_lt(resid(fx$A %*% v - v %*% diag(w)), 1e-11)
    expect_equal(csort(as.vector(host(jnp_linalg_eigvals(fx$A)))), csort(fx$eigvals_A), tolerance = 1e-10)
    f <- jnp_linalg_eigh(fx$S)
    expect_named(f, c("eigenvalues", "eigenvectors"))
    expect_jax_equal(f$eigenvalues, fx$eigvalsh_S, tolerance = 1e-10)
    v <- host(f$eigenvectors)
    expect_lt(resid(fx$S %*% v - v %*% diag(fx$eigvalsh_S)), 1e-11)
    expect_jax_equal(jnp_linalg_eigvalsh(fx$S), fx$eigvalsh_S, tolerance = 1e-10)
    up <- fx$S
    up[lower.tri(up)] <- 0
    expect_jax_equal(jnp_linalg_eigvalsh(up, UPLO = "U", symmetrize_input = FALSE), fx$eigvalsh_S, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_eigh(as_jax(4L * diag(2L)))$eigenvalues, c(4, 4), dtype = "float64")
    check_grad_fd(function(x) jnp_sum(jnp_linalg_eigvalsh(x)^2), as_jax(fx$S))
  })
  expect_error(jnp_linalg_eigh(diag(2), UPLO = "X"), "UPLO must be one of None, 'L', or 'U', got X")
  expect_error(jnp_linalg_eigh(matrix(1, 2, 3)), "Argument to linalg.eigh must have shape")
})

test_that("jnp_linalg pinv and inv", {
  skip_if_no_backend()
  x64({
    expect_jax_equal(jnp_linalg_pinv(fx$T), fx$pinv_T, tolerance = 1e-10, dtype = "float64")
    expect_jax_equal(jnp_linalg_pinv(fx$M), fx$pinv_M, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_pinv(fx$R2), fx$pinv_R2, tolerance = 1e-9)
    expect_jax_equal(jnp_linalg_pinv(fx$Zc), fx$pinv_Zc, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_pinv(fx$M, rtol = 0.3), fx$pinv_rtol, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_pinv(fx$M, rcond = 0.3), fx$pinv_rtol, tolerance = 1e-10)
    expect_identical(dim(jnp_linalg_pinv(matrix(0, 0, 3))), c(3L, 0L))
    w <- as_jax(arange(3, 5))
    f <- function(x) jnp_sum(jnp_linalg_pinv(x) * w)
    expect_jax_equal(grad(f)(as_jax(fx$T)), fx$pinv_grad, tolerance = 1e-9)
    check_grad_fd(f, as_jax(fx$T))
    check_grad_fd(function(x) jnp_sum(jnp_linalg_pinv(x)^2), as_jax(fx$M))
    expect_jax_equal(jnp_linalg_inv(fx$A), fx$inv_A, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_inv(fx$Ab), fx$inv_Ab, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_inv(fx$Zc), fx$inv_Zc, tolerance = 1e-10)
    expect_jax_equal(jit(jnp_linalg_inv)(as_jax(fx$A)), fx$inv_A, tolerance = 1e-10)
    g <- function(x) jnp_sum(jnp_linalg_inv(x) * as_jax(arange(4, 4)))
    expect_jax_equal(grad(g)(as_jax(fx$A)), fx$inv_grad, tolerance = 1e-9)
    check_grad_fd(g, as_jax(fx$A))
    expect_jax_equal(vmap(jnp_linalg_inv)(as_jax(fx$Ab)), fx$inv_Ab, tolerance = 1e-10)
  })
  expect_jax_equal(jnp_linalg_inv(as_jax(2L * diag(2L))), diag(2) / 2, dtype = "float32")
  expect_error(jnp_linalg_inv(matrix(1, 2, 3)), "Argument to inv must have shape \\[..., n, n\\], got \\(2, 3\\)")
  expect_error(jnp_linalg_pinv(diag(2), rtol = 1, rcond = 1), "only one of rtol and rcond may be specified")
})

test_that("jnp_linalg_solve matches jax", {
  skip_if_no_backend()
  x64({
    expect_jax_equal(jnp_linalg_solve(fx$A, fx$B[, 1]), fx$solve_vec, tolerance = 1e-10, shape = 4L)
    expect_jax_equal(jnp_linalg_solve(fx$A, fx$B), fx$solve_mat, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_solve(fx$Zc, fx$Hc[, 1:2]), fx$solve_c, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_solve(fx$Ab, fx$A3[, 1:2]), fx$solve_bcast, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_solve(fx$Ab, fx$A3[, 1]), fx$solve_bcast_vec, tolerance = 1e-10)
    expect_jax_equal(jit(jnp_linalg_solve)(as_jax(fx$A), as_jax(fx$B)), fx$solve_mat, tolerance = 1e-10)
    # mixed dtypes are promoted
    expect_jax_equal(jnp_linalg_solve(as_jax(2L * diag(2L)), c(2, 4)), c(1, 2), dtype = "float64")
    w <- as_jax(arange(4, 2))
    fa <- function(a) jnp_sum(jnp_linalg_solve(a, as_jax(fx$B)) * w)
    fb <- function(b) jnp_sum(jnp_linalg_solve(as_jax(fx$A), b) * w)
    expect_jax_equal(grad(fa)(as_jax(fx$A)), fx$solve_grad_a, tolerance = 1e-9)
    expect_jax_equal(grad(fb)(as_jax(fx$B)), fx$solve_grad_b, tolerance = 1e-9)
    expect_jax_equal(jit(grad(fa))(as_jax(fx$A)), fx$solve_grad_a, tolerance = 1e-9)
    check_grad_fd(fa, as_jax(fx$A))
    check_grad_fd(fb, as_jax(fx$B))
    check_grad_fd(function(a) jnp_sum(jnp_linalg_solve(a, as_jax(fx$B[, 1]))^2), as_jax(fx$A))
    # second order through the custom linear solve
    hv <- host(jvp(grad(fa), list(as_jax(fx$A)), list(as_jax(fx$S)))[[2L]])
    eps <- 1e-5
    fd <- (host(grad(fa)(as_jax(fx$A + eps * fx$S))) - host(grad(fa)(as_jax(fx$A - eps * fx$S)))) / (2 * eps)
    expect_equal(hv, fd, tolerance = 1e-5)
    out <- host(vmap(function(a) jnp_linalg_solve(a, as_jax(fx$A3[, 1:2])))(as_jax(fx$Ab)))
    expect_equal(out, fx$solve_bcast, tolerance = 1e-10)
    bs <- aperm(array(c(fx$B, 2 * fx$B), c(4, 2, 2)), c(3, 1, 2))
    out <- host(vmap(function(b) jnp_linalg_solve(as_jax(fx$A), b))(as_jax(bs)))
    expect_equal(out[2, , ], 2 * fx$solve_mat, tolerance = 1e-10)
  })
  expect_error(jnp_linalg_solve(c(1, 2), c(1, 2)), "left hand array must be at least two dimensional; got a.shape=\\(2,\\)")
  expect_error(jnp_linalg_solve(array(1, c(5, 3, 3)), matrix(1, 5, 3)), "Invalid shapes for solve: \\(5, 3, 3\\), \\(5, 3\\)")
  expect_error(jnp_linalg_solve(diag(3), c(1, 2)), "The arguments to solve must have shapes")
})

test_that("jnp_linalg norms match jax", {
  skip_if_no_backend()
  x64({
    col <- fx$T[, 1]
    expect_jax_equal(jnp_linalg_norm(fx$X3), fx$norm_none, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_norm(col), fx$norm_vec2, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_norm(col, ord = 1), fx$norm_vec1, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_norm(col, ord = Inf), fx$norm_vecinf)
    expect_jax_equal(jnp_linalg_norm(col, ord = -Inf), fx$norm_vecminf)
    expect_jax_equal(jnp_linalg_norm(c(1, 0, 2, 0), ord = 0), fx$norm_vec0)
    expect_jax_equal(jnp_linalg_norm(col, ord = 3), fx$norm_vec3, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_norm(col, ord = -1.5), fx$norm_vecm1, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_norm(fx$X3, axis = 1L), fx$norm_axis1, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_norm(fx$X3, axis = 2L, keepdims = TRUE, ord = 1), fx$norm_axis1_keep, tolerance = 1e-12)
    ords <- list(fro = NULL, "1" = 1, m1 = -1, inf = Inf, minf = -Inf, "2" = 2, m2 = -2, nuc = "nuc")
    for (nm in names(ords)) {
      o <- ords[[nm]]
      expect_jax_equal(jnp_linalg_norm(fx$T, ord = o), fx[[paste0("mnorm_", nm)]], tolerance = 1e-10)
      expect_jax_equal(jnp_linalg_norm(fx$X3, ord = o, axis = c(0L, 2L)), fx[[paste0("mnorm_b_", nm)]], tolerance = 1e-10)
      expect_jax_equal(jnp_linalg_norm(fx$X3, ord = o, axis = c(1L, 2L), keepdims = TRUE),
                       fx[[paste0("mnorm_k_", nm)]], tolerance = 1e-10)
    }
    expect_jax_equal(jnp_linalg_norm(fx$T, ord = "fro"), fx$mnorm_fro, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_norm(fx$Zc), fx$mnorm_c, tolerance = 1e-12, dtype = "float64")
    expect_jax_equal(jnp_linalg_norm(fx$Zc, ord = 1), fx$mnorm_c1, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_matrix_norm(fx$X3, ord = "nuc"), fx$matrix_norm, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_vector_norm(fx$X3), fx$vector_norm_all, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_vector_norm(fx$X3, axis = c(0L, 2L), ord = 1), fx$vector_norm_ax, tolerance = 1e-12)
    expect_jax_equal(jit(function(x) jnp_linalg_norm(x, ord = 2))(as_jax(fx$T)), fx$mnorm_2, tolerance = 1e-10)
    check_grad_fd(function(x) jnp_linalg_norm(x), as_jax(fx$T))
    check_grad_fd(function(x) jnp_linalg_norm(x, ord = "nuc"), as_jax(fx$T))
    check_grad_fd(function(x) jnp_linalg_norm(x, ord = 2), as_jax(fx$T))
    out <- host(vmap(function(x) jnp_linalg_norm(x, ord = 1))(as_jax(fx$X3)))
    for (i in 1:2) expect_equal(out[i], max(colSums(abs(fx$X3[i, , ]))), tolerance = 1e-12)
  })
  expect_identical(dtype(jnp_linalg_norm(as_jax(c(3L, 4L)))), "float32")
  expect_jax_equal(jnp_linalg_norm(as_jax(c(3L, 4L))), 5)
  expect_error(jnp_linalg_norm(diag(2), ord = 3), "Invalid order '3' for matrix norm")
  expect_error(jnp_linalg_norm(array(1, c(2, 2, 2)), axis = c(0L, 1L, 2L)), "Improper number of axes for norm")
  expect_error(jnp_linalg_vector_norm(c(1, 2), ord = "inf"), "Invalid order 'inf' for vector norm.Use 'jax.numpy.inf' instead.")
})

test_that("jnp_linalg_qr modes", {
  skip_if_no_backend()
  x64({
    f <- jnp_linalg_qr(fx$T)
    expect_named(f, c("Q", "R"))
    expect_jax_equal(f$Q, fx$qr_q, tolerance = 1e-10)
    expect_jax_equal(f$R, fx$qr_r, tolerance = 1e-10)
    f <- jnp_linalg_qr(fx$T, mode = "complete")
    expect_jax_equal(f$Q, fx$qr_qc, tolerance = 1e-10)
    expect_jax_equal(f$R, fx$qr_rc, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_qr(fx$T, mode = "r"), fx$qr_r_only, tolerance = 1e-10)
    f <- jnp_linalg_qr(fx$T, mode = "raw")
    expect_jax_equal(f[[1L]], fx$qr_raw_h, tolerance = 1e-10)
    expect_jax_equal(f[[2L]], fx$qr_raw_tau, tolerance = 1e-10)
    # the raw factors feed lax_linalg_ormqr after a matrix transpose
    q <- lax_linalg_ormqr(jnp_linalg_matrix_transpose(f[[1L]]), f[[2L]], diag(5))
    expect_jax_equal(q, fx$qr_qc, tolerance = 1e-10)
  })
  expect_error(jnp_linalg_qr(diag(2), mode = "economic"), "Unsupported QR decomposition mode 'economic'")
})

test_that("jnp_linalg_lstsq matches jax", {
  skip_if_no_backend()
  x64({
    f <- jnp_linalg_lstsq(fx$T, fx$B2)
    expect_jax_equal(f[[1L]], fx$lstsq_x, tolerance = 1e-10)
    expect_jax_equal(f[[2L]], fx$lstsq_resid, tolerance = 1e-10)
    expect_jax_equal(f[[3L]], fx$lstsq_rank)
    expect_jax_equal(f[[4L]], fx$lstsq_s, tolerance = 1e-10)
    f <- jnp_linalg_lstsq(fx$T, fx$B2[, 1])
    expect_jax_equal(f[[1L]], fx$lstsq_x_vec, tolerance = 1e-10, shape = 3L)
    expect_jax_equal(f[[2L]], fx$lstsq_resid_vec, tolerance = 1e-10)
    f <- jnp_linalg_lstsq(fx$M, fx$B, rcond = 0.3)
    expect_jax_equal(f[[1L]], fx$lstsq_wide_x, tolerance = 1e-10)
    expect_jax_equal(f[[2L]], fx$lstsq_wide_resid, tolerance = 1e-10)
    expect_jax_equal(f[[3L]], fx$lstsq_wide_rank)
    expect_identical(length(jnp_linalg_lstsq(fx$M, fx$B, numpy_resid = TRUE)[[2L]]), as.integer(fx$lstsq_np_resid_len))
    expect_jax_equal(jnp_linalg_lstsq(fx$T, fx$B2, numpy_resid = TRUE)[[2L]], fx$lstsq_np_resid, tolerance = 1e-10)
    check_grad_fd(function(b) jnp_sum(jnp_linalg_lstsq(as_jax(fx$T), b)[[1L]]^2), as_jax(fx$B2))
  })
  expect_error(jnp_linalg_lstsq(matrix(1, 3, 2), c(1, 2)), "Leading dimensions of input arrays must match")
  expect_error(jnp_linalg_lstsq(array(1, c(3, 2, 2)), c(1, 2, 3)), "3-dimensional array given. Array must be two-dimensional")
})

test_that("jnp_linalg products, diagonals and traces", {
  skip_if_no_backend()
  x64({
    expect_jax_equal(jnp_linalg_cross(fx$v1, fx$v2), fx$cross, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_cross(t(fx$v1), t(fx$v2), axis = 0L), fx$cross_ax0, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_outer(fx$v1[1, ], fx$v2[2, ]), fx$outer, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_vecdot(fx$v1, fx$v2), fx$vecdot, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_vecdot(fx$Zc, fx$Hc, axis = 0L), fx$vecdot_c, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_vecdot(fx$v1, fx$v2[1, ]), fx$vecdot_b, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_matmul(fx$T, fx$A3), fx$T %*% fx$A3, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_tensordot(fx$X3, aperm(fx$X3, c(2, 3, 1)), axes = 2L), fx$tensordot_2, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_tensordot(fx$X3, fx$T, axes = list(1L, 1L)), fx$tensordot_ax, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_tensordot(fx$v1[1, ], fx$v2[1, ], axes = 0L), fx$tensordot_0, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_diagonal(fx$X3), fx$diag_0)
    expect_jax_equal(jnp_linalg_diagonal(fx$X3, offset = 1L), fx$diag_1)
    expect_jax_equal(jnp_linalg_diagonal(fx$X3, offset = -1L), fx$diag_m1)
    expect_jax_equal(jnp_linalg_diagonal(fx$X3, offset = 3L), fx$diag_big)
    expect_identical(dim(jnp_linalg_diagonal(fx$X3, offset = 7L)), c(2L, 0L))
    expect_jax_equal(jnp_linalg_trace(fx$X3), fx$trace_0, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_trace(fx$X3, offset = 1L), fx$trace_1, tolerance = 1e-12)
    expect_jax_equal(jnp_linalg_trace(as_jax(arange(3, 3), "int32"), dtype = "float32"), fx$trace_int, dtype = "float32")
    expect_jax_equal(jnp_linalg_matrix_transpose(fx$X3), fx$mT)
    expect_jax_equal(jnp_linalg_multi_dot(list(t(fx$T), fx$T, fx$A3, fx$A3[, 1:2])), fx$multi_dot, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_multi_dot(list(fx$T[, 1], fx$T, fx$A3[, 1])), fx$multi_dot_vec, tolerance = 1e-10)
    check_grad_fd(function(x) jnp_sum(jnp_linalg_cross(x, as_jax(fx$v2))^2), as_jax(fx$v1))
    check_grad_fd(function(x) jnp_linalg_trace(x %*% x), as_jax(fx$A3))
    check_grad_fd(function(x) jnp_sum(jnp_linalg_diagonal(x, offset = 1L)^2), as_jax(fx$A3))
  })
  expect_identical(dtype(jnp_linalg_diagonal(as_jax(diag(2) > 0))), "bool")
  expect_error(jnp_linalg_cross(c(1, 2), c(1, 2)), "Both input arrays must be \\(arrays of\\) 3-dimensional vectors")
  expect_error(jnp_linalg_outer(diag(2), c(1, 2)), "Input arrays must be one-dimensional, but they are x1.ndim=2 x2.ndim=1")
  expect_error(jnp_linalg_matrix_transpose(c(1, 2)), "matrix_transpose requires at least 2 dimensions; got ndim=1")
  expect_error(jnp_linalg_vecdot(c(1, 2), c(1, 2, 3)), "axes must match")
  expect_error(jnp_linalg_tensordot(diag(2), diag(2), axes = 3L), "Number of tensordot axes \\(axes 3\\) exceeds input ranks")
  expect_error(jnp_linalg_multi_dot(list(diag(2))), "multi_dot requires at least two arrays")
  expect_error(jnp_linalg_multi_dot(list(diag(2), diag(3))), "last dimension of each array must match first dimension")
})

test_that("jnp_linalg tensorinv, tensorsolve and cond", {
  skip_if_no_backend()
  x64({
    expect_jax_equal(jnp_linalg_tensorsolve(fx$ts_a, fx$ts_b), fx$tensorsolve, tolerance = 1e-10)
    a2 <- aperm(array(aperm(fx$ts_a), c(3, 2, 3, 2)))  # ts_a reshaped to (2, 3, 2, 3)
    expect_jax_equal(jnp_linalg_tensorsolve(aperm(a2, c(3, 1, 4, 2)), fx$ts_b, axes = c(0L, 2L)),
                     fx$tensorsolve_axes, tolerance = 1e-10)
    ident <- aperm(array(diag(24), c(3, 8, 6, 4)))  # eye(24).reshape(4, 6, 8, 3)
    expect_identical(dim(jnp_linalg_tensorinv(ident, ind = 2L)), c(8L, 3L, 4L, 6L))
    expect_jax_equal(jnp_linalg_tensorinv(aperm(array(diag(6), c(3, 2, 6))), ind = 1L),
                     aperm(array(diag(6), c(6, 3, 2)), c(3, 2, 1))[, , ], tolerance = 1e-12)
    ps <- list(none = NULL, m2 = -2, "1" = 1, m1 = -1, inf = Inf, fro = "fro", nuc = "nuc")
    for (nm in names(ps)) expect_jax_equal(jnp_linalg_cond(fx$A, ps[[nm]]), fx[[paste0("cond_", nm)]], tolerance = 1e-9)
    expect_jax_equal(jnp_linalg_cond(fx$Ab), fx$cond_b, tolerance = 1e-9)
    expect_jax_equal(jnp_linalg_cond(fx$T), fx$cond_T, tolerance = 1e-10)
    expect_jax_equal(jnp_linalg_cond(matrix(1, 2, 2), 1), fx$cond_sing)
    expect_true(is.nan(host(jnp_linalg_cond(matrix(NaN, 2, 2)))))
  })
  expect_error(jnp_linalg_tensorinv(diag(2), ind = 0L), "ind must be a positive integer; got ind=0")
  expect_error(jnp_linalg_tensorinv(array(1, c(2, 3, 4)), ind = 1L), "tensorinv is only possible when the product of the first")
  expect_error(jnp_linalg_tensorsolve(array(1, c(2, 3, 6)), matrix(1, 3, 2)), "leading shape of a must match shape of b")
  expect_error(jnp_linalg_cond(c(1, 2)), "input array must be at least 2D; got arr.shape=\\(2,\\)")
  expect_error(jnp_linalg_cond(matrix(1, 2, 3), 1), "for p=1, array must be square")
})
