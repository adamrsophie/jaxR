# Tests of R/lax_linalg.R (port of jax/_src/lax/linalg.py and jax/_src/tpu/linalg/qdwh.py)
# and of the LAPACK FFI kernels in src/lapack_ffi.cpp. Expected values come from
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
fx$Bc <- array(c(-0.579-1.536i, 0.056+1.896i, 0.901-0.18i, -0.312+1.488i, -1.165+1.179i,
  0.466-1.071i), dim = c(3L, 2L))
fx$chol_S <- array(c(3.640329655402104, 0.9603525864236157, 0.9290367412141499,
  0.7070788207821471, 0.0, 2.9063246394285467, -0.5399268946142641, 0.3423413242135734, 0.0, 0.0,
  2.2714245930574912, 0.021544666192905463, 0.0, 0.0, 0.0, 2.070115404100859), dim = c(4L, 4L))
fx$chol_S32 <- array(c(3.6403296, 3.0608823, 0.9603526, -0.22117805, 0.92903674, 0.54690117, 0.0,
  0.0, 2.9063246, 2.5030143, -0.5399269, 0.25647578, 0.0, 0.0, 0.0, 0.0, 2.2714245, 2.1302862),
  dim = c(2L, 3L, 3L))
fx$chol_Hc <- array(c(2.2516660498395407+0.0i, 0.1923020511993138-0.5080682368868705i,
  0.01732050807568877+0.004885271508527602i, 0.0+0.0i, 3.0357678744876395+0.0i,
  1.1753702620143498+1.0449615001620114i, 0.0+0.0i, 0.0+0.0i, 2.2244631587135277+0.0i),
  dim = c(3L, 3L))
fx$chol_grad_c <- array(c(0.07250573711396033+0.0i, -0.10460081645199085-0.28439272931030707i,
  0.09535109273031817+0.06230399359620267i, -0.10460081645199085+0.28439272931030707i,
  4.268388696637239+0.0i, -0.38717395749921213+0.34421653544191844i,
  0.09535109273031817-0.06230399359620267i, -0.38717395749921213-0.34421653544191844i, 8.0+0.0i),
  dim = c(3L, 3L))
fx$chol_grad <- array(c(0.31730980779917733, -0.28194506406534686, -0.5579028656303914,
  -0.5176698280576006, -0.28194506406534686, 5.06268098094982, 0.1869105057997421,
  -0.23734578516756943, -0.5579028656303914, 0.1869105057997421, 10.000089966928126,
  -0.009485089779672086, -0.5176698280576006, -0.23734578516756943, -0.009485089779672086,
  14.999999999999998), dim = c(4L, 4L))
fx$trsm_left_lower <- array(c(0.2591950034698126, -0.05662463330519682, -0.6040130768229182,
  0.2173148624208773, -0.03556557945870922, -0.2605238165964708, 0.182773174212573,
  -0.1763502486750197), dim = c(4L, 2L))
fx$trsm_right_upper_T <- array(c(0.27536891646527584, 0.018939324511498344, 0.32908369650936364,
  -0.3595257166246754, -0.6860184934423512, 0.21788881306647803, 0.19935394554683897,
  -0.17120443008767883), dim = c(2L, 4L))
fx$trsm_left_lower_unit <- array(c(1.494, -2.477792, -1.380745488, 0.6424076206720001, -0.205,
  -0.47106, 0.82649066, -0.8954875330400001), dim = c(4L, 2L))
fx$trsm_vec <- c(0.2591950034698126, -0.05662463330519682, -0.6040130768229182,
  0.2173148624208773)
fx$trsm_c_adjoint <- array(c(-0.4121648692148612-0.19158781340179185i,
  -0.45048996844666434+0.8192246967946732i, 0.23240378605302636-0.11043133053773779i,
  -0.14039844143843697+0.4587400254348811i, -0.807093824802546-0.2377249610421101i,
  0.05340102470314115-0.32270063312631697i), dim = c(3L, 2L))
fx$trsm_c_conj <- array(c(-0.375491003698618-0.45332941207635696i,
  -0.6600674359899541+0.57079123933457i, 0.12656699759931844+0.21920596353390734i,
  0.06775253079245484+0.5409740062841271i, -1.1494064269265887+0.24990425691666995i,
  -0.3622759961670239+0.19749864331075634i), dim = c(3L, 2L))
fx$lu_M <- array(c(2.27, -0.1330396475770925, -0.15330396475770924, 0.647136563876652, -1.454,
  -1.2424396475770925, 0.05384886492328224, -0.8820843459188225, 0.046, -1.4138801762114537,
  1.3131878250053717, -0.6845401969636128, -0.187, -1.7308784140969162, 1.2665379965096382,
  -1.4267700467261462, 1.533, 2.154949779735683, -0.26802662163187535, -1.2556878818402228),
  dim = c(4L, 5L))
fx$lu_M_piv <- c(0L, 3L, 2L, 3L)
fx$lu_M_perm <- c(0L, 3L, 2L, 1L)
fx$X32 <- array(c(1.054, 0.127, 0.208, -1.348, 0.707, -1.173, -0.403, 0.402, 0.977, -1.27, 0.011,
  1.944, 1.222, 1.883, 0.356, 0.969, 1.786, -0.414), dim = c(2L, 3L, 3L))
fx$lu_X32 <- array(c(1.054, -1.348, 0.19734345, 0.870178, 0.670778, -0.09421364, -0.403, -1.27,
  1.0565294, 3.0491261, 0.26627135, 0.09259987, 1.222, 0.969, 0.114846304, -1.2572025, 0.935729,
  2.0907097), dim = c(2L, 3L, 3L))
fx$lu_X32_piv <- array(c(0L, 1L, 1L, 2L, 2L, 2L), dim = c(2L, 3L))
fx$lu_X32_perm <- array(c(0L, 1L, 1L, 2L, 2L, 0L), dim = c(2L, 3L))
fx$lu_grad <- array(c(0.5790586010891199, -7.540642553972066, -18.72512018044646,
  -10.749665776547332, 2.5848882083685565, -4.388556029334367, -10.52647332737089,
  2.695762065660621, 4.915728929114806, -0.5369971551213343, 11.632403861664345,
  5.899931471384696, -1.7712866146221322, 18.0, 25.32172354534503, 22.51397215572083,
  -0.9218527526844521, 19.0, 27.006263742308644, 24.305345924115514), dim = c(4L, 5L))
fx$p2p <- array(c(2L, 0L, 0L, 1L, 3L, 2L, 1L, 3L, 4L, 4L), dim = c(2L, 5L))
fx$qr_T_q <- array(c(-0.29135228279410197, 0.4438837720216022, -0.5112946923543552,
  -0.6746804823134005, 0.03827569205334279, -0.4191660565712894, -0.8139181594492075,
  0.00021655567260190302, -0.3426888645105147, 0.21071625941398883, -0.6776534122642389,
  0.020236970728009836, -0.2668198843131947, 0.48045850879628277, -0.48820394152724006,
  -0.5293381557260038, 0.3742921188667952, 0.6220330138245588, 0.028301714034582447,
  0.4381557592247745, 0.0007962840991064369, 0.0005490215359774006, -0.5295845255682378,
  0.44241295554471316, 0.7237472428630334), dim = c(5L, 5L))
fx$qr_T_r <- array(c(1.750458225722625, 0.0, 0.0, 0.0, 0.0, -0.7562368416153, 1.570574047723862,
  0.0, 0.0, 0.0, 0.2187587195015293, 0.41820403441883797, 1.496291752379147, 0.0, 0.0),
  dim = c(5L, 3L))
fx$qr_T_q_red <- array(c(-0.29135228279410197, 0.4438837720216022, -0.5112946923543552,
  -0.6746804823134005, 0.03827569205334279, -0.4191660565712894, -0.8139181594492075,
  0.00021655567260190302, -0.3426888645105147, 0.21071625941398883, -0.6776534122642389,
  0.020236970728009836, -0.2668198843131947, 0.48045850879628277, -0.48820394152724006),
  dim = c(5L, 3L))
fx$qr_T_r_red <- array(c(1.750458225722625, 0.0, 0.0, -0.7562368416153, 1.570574047723862, 0.0,
  0.2187587195015293, 0.41820403441883797, 1.496291752379147), dim = c(3L, 3L))
fx$Tp <- array(c(-1.253, -0.213, -0.511, 0.428, -0.634, -0.876, -3.228, 0.774, -0.056, 0.604,
  -1.53, 2.331, -2.685, -3.543, 0.201), dim = c(5L, 3L))
fx$qr_Tp_p <- c(2L, 1L, 0L)
fx$qr_Tp_r <- array(c(5.251374677167875, 0.0, 0.0, -1.5124736832306005, 3.1411480954477238, 0.0,
  0.2187587195015293, 0.4182040344188381, 1.496291752379147), dim = c(3L, 3L))
fx$qr_M_q <- array(c(-0.8276094638421747, -0.5355763446626233, 0.1268758120780074,
  0.11010487140102941, 0.40736942831268436, -0.4730787318811276, -0.017477659349066633,
  0.7809873198340795, -0.266667711702297, 0.2588431121374714, -0.8863879644546222,
  0.276052443032914, -0.27928669100142506, 0.6498882486847686, 0.444874629759009,
  0.5493002569157451), dim = c(4L, 4L))
fx$qr_M_r <- array(c(-2.7428395869973876, 0.0, 0.0, 0.0, 1.0246224435883045, -1.4876245655726215,
  0.0, 0.0, -0.24080956215272425, -1.2905842821124276, -1.3966756837362648, 0.0,
  0.5950205793065049, -1.0094566833585485, -1.7163696225783522, -0.9272410869427404,
  -0.042034904464177486, 3.092136416624237, -0.03995934858279798, -0.8160567984238287),
  dim = c(4L, 5L))
fx$qr_Zc_q <- array(c(-0.2596014014031862-0.6486459258200822i,
  -0.581421320498045+0.0922550434738595i, -0.2874924610580739+0.2874924610580739i,
  -0.03897396831031036+0.41257151445928597i, -0.37148270446903453+0.35598531191973104i,
  -0.545677746103691-0.5155352855391154i, -0.4856313284056029+0.3229451399672055i,
  0.11196012485320918-0.6133337129528944i, -0.4480969507001765+0.26526300008490744i), dim = c(3L,
  3L))
fx$qr_Zc_r <- array(c(1.398297536291901+0.0i, 0.0+0.0i, 0.0+0.0i,
  1.5210146229965282-0.28685096668601145i, 2.3368870403898496+0.0i, 0.0+0.0i,
  -0.9797678708875339+0.3962611573137542i, 0.0057834009792049335+1.2298636161775174i,
  0.7490254016098163+0.0i), dim = c(3L, 3L))
fx$geqrf_T_a <- array(c(1.750458225722625, -0.3437356157075665, 0.395937420924417,
  0.522460440348309, -0.0296400080468558, -0.7562368416153, 1.570574047723862,
  -0.08487234108296988, 0.06317219245678819, -0.10127279230714321, 0.2187587195015293,
  0.41820403441883797, 1.496291752379147, -0.8341259912960438, 0.5211970760851412), dim = c(5L,
  3L))
fx$geqrf_T_taus <- c(1.291352282794102, 1.9580004619884523, 1.016563601942368)
fx$householder_T <- array(c(-0.29135228279410197, 0.4438837720216022, -0.5112946923543552,
  -0.6746804823134005, 0.03827569205334279, -0.4191660565712894, -0.8139181594492075,
  0.00021655567260190302, -0.3426888645105147, 0.21071625941398883, -0.6776534122642389,
  0.020236970728009836, -0.2668198843131947, 0.48045850879628277, -0.48820394152724006),
  dim = c(5L, 3L))
fx$C <- array(c(-0.747, 1.481, 0.906, 1.91, 0.802, 1.923, 1.868, -0.861, -0.268, 0.947),
  dim = c(5L, 2L))
fx$C2 <- array(c(-0.155, 0.298, 0.614, 1.326, 0.922, -0.695, 0.376, -0.15, -1.099, -0.435),
  dim = c(2L, 5L))
fx$ormqr_left <- array(c(-2.0274960236354698, -0.8033210140574036, 0.9038753058512528,
  0.8406309850703825, 1.2584888561000631, -0.6171963387522943, -0.784025024512126,
  -1.421304640425447, -1.939847932928335, 1.455528617577916), dim = c(5L, 2L))
fx$ormqr_left_T <- array(c(-0.8461435858536728, -1.3776408416112098, 0.8205754281178678,
  1.9187613335984206, 0.9458687303744814, 0.9261906260749015, -2.0352529897179537,
  -1.6266849430542487, -0.4469613756962835, 1.0253510699713497), dim = c(5L, 2L))
fx$ormqr_right <- array(c(-0.4494543134128282, 0.941668858918085, -0.7950035289550518,
  -1.2445757136488598, 0.5886423766370288, 0.1506332641731752, 0.4144854789569619,
  -0.2885873777650112, -1.1171142060078483, -0.012165453488961567), dim = c(2L, 5L))
fx$eigh_S_w <- c(4.009147183475617, 4.140762348394433, 8.858043221651155, 16.830047246478802)
fx$eigh_Hc_w <- c(3.2510612026843195, 5.27812815434003, 13.47381064297565)
fx$eig_A_w <- c(-1.37571204738413+0.0i, -0.47227674979961287+0.0i, 0.14648120564204295+0.0i,
  2.9665075915416987+0.0i)
fx$Ar <- array(c(0.5, 1.468, -1.082, -1.48, -1.468, 0.5, -0.539, 0.273, 1.082, 0.539, 0.5, -1.01,
  1.48, -0.273, 1.01, 0.5), dim = c(4L, 4L))
fx$eig_Ar_w <- c(0.4999999999999999-0.14853193946405702i,
  0.4999999999999999+0.14853193946405702i, 0.5000000000000001-2.622829819671693i,
  0.5000000000000001+2.622829819671693i)
fx$eig_Zc_w <- c(-1.5584966908080735+0.11035603047833391i,
  -0.2557072375288577+0.9052751404575076i, 0.1882039283369321-1.6546311709358412i)
fx$svd_M_s <- c(4.024020139383975, 2.9152690821120437, 2.219500993133564, 0.5019994404308582)
fx$svd_Zc_s <- c(3.236336482459051, 1.5091928976540196, 0.5011127318031767)
fx$eigh_grad <- array(c(3.2699258511790936, 0.5464456199781222, 0.7835995519596715,
  0.7130577994971335, 0.5464456199781222, 2.971538920769075, -0.27406360595056983,
  0.1986625257749441, 0.7835995519596715, -0.27406360595056983, 2.170333216816435,
  -0.29452432279164037, 0.7130577994971335, 0.1986625257749441, -0.29452432279164037,
  1.5882020112353943), dim = c(4L, 4L))
fx$svd_grad <- array(c(1.5980153462908802, 0.5836007185649341, -1.0298270177447741,
  -1.1806109998412537, -0.5950891243751608, -0.49162572527747517, -1.717510465053374,
  -1.059962194905854, 0.5094094642259368, 0.49529353261830605, 2.888554227786199,
  0.5789359153779337, 0.9392955095165859, -1.8669476485145537, -0.023102058149957624,
  -1.8593038257846042, 0.7643116465832591, -1.5080844265493276, 0.33537412070331923,
  0.046641468789915465), dim = c(4L, 5L))
fx$eig_grad <- array(c(3.528, 0.799999999999999, 1.9580000000000009, 4.481999999999998,
  3.7359999999999998, -1.9540000000000006, 1.9000000000000015, -0.30200000000000177,
  -0.20600000000000068, 0.8219999999999994, 0.2879999999999997, 2.9079999999999995,
  1.5220000000000002, 0.24399999999999936, 0.8880000000000007, 0.6679999999999993), dim = c(4L,
  4L))
fx$hess_A <- array(c(1.764, -2.019691560610184, -0.026493871335778976, 0.19574598142260002,
  -1.1644174020757367, -0.8987331216227684, 1.0509715617204596, 0.30684258361679506,
  -1.9299967725539886, 0.9666692843874054, 0.8096785028154462, 0.34987421675004615,
  1.02944964504068, 0.340906829215866, -0.7734359184443468, -0.4099453811926782), dim = c(4L,
  4L))
fx$hess_A_taus <- c(1.9248936998259503, 1.827898977133872, 0.0)
fx$trid_S_a <- array(c(13.252, -5.503218694545947, 0.37581040252413134, 0.286024830306657, 3.496,
  8.003898577057683, -1.7400966874842525, 0.12241826557368667, 3.382, -0.677, 8.461627446112107,
  0.1620922176633156, 2.574, 1.674, 0.521, 4.120473976830213), dim = c(4L, 4L))
fx$trid_S_d <- c(13.252, 8.003898577057683, 8.461627446112107, 4.120473976830213)
fx$trid_S_e <- c(-5.503218694545947, -1.7400966874842525, 0.1620922176633156)
fx$trid_S_taus <- c(1.63526459587455, 1.9704700787511393, 0.0)
fx$trid_Hc_a <- array(c(5.05523168014803+0.0i, 0.433-1.144i, 0.039+0.011i,
  1.1960020017818331+0.0i, 9.525768319851974+0.0i, 3.569+3.182i,
  0.004516439810285347+0.00040371667946181324i, -4.781686627122275+0.0i, 7.422+0.0i), dim = c(3L,
  3L))
fx$trid_Hc_d <- c(5.05523168014803, 9.525768319851974, 7.422)
fx$trid_Hc_e <- c(1.1960020017818331, -4.781686627122275)
fx$trid_Hc_taus <- c(1.8763961165422938+0.4815909539324697i,
  1.746389355537484-0.6654555699972748i)
fx$wv <- c(1.849, 0.672, 0.407, -0.77)
fx$chol_update <- array(c(4.08298922359587, -1.1873464810485948e-17, 2.5238287816815147e-18,
  9.315565755514022e-17, 1.1605536386468347, 2.910961911778234, 1.7641718372378394e-19,
  6.511637743851095e-18, 1.0126264786853216, -0.5423304709143344, 2.27158849155898,
  1.626713829741918e-18, 0.28172251676602816, 0.2849937351367323, 0.03384904737407772,
  2.309581182048316), dim = c(4L, 4L))
fx$sym_prod <- array(c(59.06438, 10.999662, 6.589502000000001, 15.890581999999998, 10.999662,
  42.18255, -2.676642, -1.9639920000000006, 6.589502000000001, -2.676642, 25.447826, -7.658594,
  15.890581999999998, -1.9639920000000006, -7.658594, 34.558684), dim = c(4L, 4L))
fx$td_dl <- c(0.0, 0.539, -0.674, 0.032, -0.636)
fx$td_d <- c(4.676, 4.577, 3.792, 4.396, 2.907)
fx$td_du <- c(-1.491, 0.439, 0.167, 0.635, 0.0)
fx$td_b <- array(c(2.383, -0.913, -1.316, -0.068, -0.745, 0.944, 1.117, -0.462, 1.713, -0.826),
  dim = c(5L, 2L))
fx$td_x <- array(c(0.44126015090786325, -0.21439807803811622, -0.38619411881293986,
  0.023615532580703362, -0.25111129042954, 0.27254957540993047, 0.22162428881075455,
  -0.10086239415210911, 0.41823334900147446, -0.19263969385451057), dim = c(5L, 2L))
fx$td_x_pert <- array(c(0.44126015090786325, -0.2143980780381162, -0.3861941188129398,
  0.023615532580703362, -0.25111129042954, 0.2725495754099305, 0.22162428881075452,
  -0.10086239415210908, 0.41823334900147446, -0.1926396938545106), dim = c(5L, 2L))
fx$td_sing <- array(c(-4503599627370495.0, 4503599627370496.0, 3.0), dim = c(3L, 1L))
fx$td_sing32 <- array(c(-8388607.0, 8388608.0, 3.0), dim = c(3L, 1L))
fx$td_grad_d <- c(-0.11427575920890298, -0.03928482146854358, -0.08291814504199949,
  -0.0702697143762533, -0.07563977498718015)
fx$qdwh_T_u <- array(c(-0.29849625518505013, 0.25429521681205847, -0.4670679232191541,
  -0.7808172085099525, 0.13566860484828908, -0.2545726281845274, -0.8895152849546619,
  0.14947234111749882, -0.23740262762386813, 0.2554473457473894, -0.7522587622912412,
  -0.0632582711437236, -0.30351397716019773, 0.3707212821526499, -0.447828250912614), dim = c(5L,
  3L))
fx$qdwh_T_h <- array(c(1.6990811846635772, -0.3976116059469612, 0.13831897461481213,
  -0.3976116059469612, 1.6888186490632995, 0.16850395067265345, 0.13831897461481213,
  0.16850395067265345, 1.553741703073331), dim = c(3L, 3L))
fx$qdwh_T_iters <- 6L
fx$qdwh_T_conv <- TRUE
fx$qdwh_Zc_u <- array(c(-0.36283241737683064-0.48052220684295055i,
  -0.3933816722407944-0.30596534335928305i, -0.22246061401767853+0.5827507095235627i,
  -0.3705515607003945+0.008674893317679852i, -0.3372028624150048+0.4242395020946157i,
  -0.5997636597304495-0.4574001683392851i, -0.4592605315756866+0.5377239969133911i,
  0.1344604348770225-0.663228116489639i, -0.13704280601647617-0.15231806704959042i), dim = c(3L,
  3L))
fx$qdwh_Zc_h <- array(c(1.171586531416343+0.0i, 0.5127450347479601+0.21566693889692884i,
  -0.5220175077559662-0.026439020560790782i, 0.5127450347479601-0.21566693889692884i,
  2.605616557330293+0.0i, -0.32319316048525637-0.8085164426218714i,
  -0.5220175077559662+0.026439020560790782i, -0.32319316048525637+0.8085164426218714i,
  1.4694390231696113+0.0i), dim = c(3L, 3L))
fx$qdwh_Zc_iters <- 6L
fx$qdwh_pad_u <- array(c(-0.2984962551850501, 0.2542952168120585, -0.4670679232191541,
  -0.7808172085099525, 0.13566860484828905, 0.0, 0.0, -0.2545726281845274, -0.8895152849546617,
  0.14947234111749882, -0.23740262762386816, 0.25544734574738936, 0.0, 0.0, -0.7522587622912413,
  -0.0632582711437236, -0.3035139771601977, 0.3707212821526499, -0.4478282509126141, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dim = c(7L, 4L))
fx$qdwh_pad_h <- array(c(1.6990811846635772, -0.39761160594696116, 0.13831897461481207, 0.0,
  -0.39761160594696116, 1.6888186490632993, 0.1685039506726535, 0.0, 0.13831897461481207,
  0.1685039506726535, 1.553741703073331, 0.0, 0.0, 0.0, 0.0, 0.0), dim = c(4L, 4L))
fx$qdwh_T32_iters <- 4L
fx$qdwh_T_max2_iters <- 2L
fx$qdwh_T_max2_conv <- FALSE
fx$hess_Zc <- array(c(-0.363-0.907i, 1.0004089163936916+0.0i,
  0.2362561079143985-0.20487544685612702i, 0.39113006050619353-0.5665103446328837i,
  -1.7199628783654966-0.7330571932159495i, -1.7125432122587343+0.0i,
  0.7138156397849443-0.36315222371722233i, 0.805552234476264+1.282104370128418i,
  0.4569628783654971+1.0010571932159493i), dim = c(3L, 3L))
fx$hess_Zc_taus <- c(1.8126676868602194-0.1289472713468245i,
  1.4334192879072165+0.9011923883777545i)

x64 <- function(code) with_jax_config("jax_enable_x64", TRUE, code)
host <- function(x) as.array(x)
H <- function(x) Conj(t(x))
resid <- function(x) max(Mod(x))
# jnp.arange(n).reshape(shape) as an R array
arange <- function(...) {
  d <- c(...)
  aperm(array(seq_len(prod(d)) - 1, rev(d)))
}
# directional second derivative against a finite difference of the gradient
check_hvp_fd <- function(f, x, v, eps = 1e-5, tolerance = 1e-5) {
  g <- grad(f)
  hv <- host(jvp(g, list(x), list(v))[[2L]])
  fd <- (host(g(as_jax(host(x) + eps * host(v)))) - host(g(as_jax(host(x) - eps * host(v))))) / (2 * eps)
  expect_equal(as.vector(hv), as.vector(fd), tolerance = tolerance)
}

test_that("the LAPACK FFI handlers are registered with the XLA runtime", {
  skip_if_no_backend()
  jax_backend_init()
  expect_identical(jaxR:::.jx$lapack_ffi_error, "")
})

# ---- cholesky ------------------------------------------------------------------------------

test_that("cholesky matches jax", {
  skip_if_no_backend()
  x64({
    expect_jax_equal(lax_linalg_cholesky(fx$S), fx$chol_S, dtype = "float64")
    expect_jax_equal(jit(lax_linalg_cholesky)(as_jax(fx$S)), fx$chol_S)
    expect_jax_equal(lax_linalg_cholesky(fx$Hc), fx$chol_Hc, dtype = "complex128")
    # without symmetrization only the lower triangle is read
    low <- fx$S
    low[upper.tri(low)] <- 99
    expect_jax_equal(lax_linalg_cholesky(low, symmetrize_input = FALSE), fx$chol_S)
    # a matrix that is not positive definite gives NaN in the lower triangle
    bad <- host(lax_linalg_cholesky(-diag(2)))
    expect_true(all(is.nan(bad[lower.tri(bad, diag = TRUE)])))
    expect_identical(bad[1, 2], 0)
    # a failure in one batch element leaves the others intact
    two <- array(0, c(2, 2, 2))
    two[1, , ] <- -diag(2)
    two[2, , ] <- 4 * diag(2)
    out <- host(lax_linalg_cholesky(two))
    expect_true(is.nan(out[1, 1, 1]))
    expect_equal(out[2, , ], 2 * diag(2))
    expect_identical(dim(lax_linalg_cholesky(matrix(0, 0, 0))), c(0L, 0L))
  })
  s32 <- array(0, c(2, 3, 3))
  s32[1, , ] <- fx$S[1:3, 1:3]
  s32[2, , ] <- fx$S[2:4, 2:4]
  expect_jax_equal(lax_linalg_cholesky(as_jax(s32, "float32")), fx$chol_S32, dtype = "float32")
  expect_identical(dtype(lax_linalg_cholesky(as_jax(fx$Hc, "complex64"))), "complex64")
})

test_that("cholesky derivatives and vmap", {
  skip_if_no_backend()
  x64({
    w <- as_jax(arange(4, 4))
    f <- function(x) jnp_sum(lax_linalg_cholesky(x)^2 * w)
    S <- as_jax(fx$S)
    expect_jax_equal(grad(f)(S), fx$chol_grad, tolerance = 1e-10)
    check_grad_fd(f, S)
    expect_jax_equal(jit(grad(f))(S), fx$chol_grad, tolerance = 1e-10)
    check_hvp_fd(f, S, as_jax(fx$A))
    xs <- array(0, c(3, 4, 4))
    for (i in 1:3) xs[i, , ] <- fx$S + i * diag(4)
    out <- host(vmap(lax_linalg_cholesky)(as_jax(xs)))
    for (i in 1:3) expect_equal(out[i, , ], host(lax_linalg_cholesky(xs[i, , ])), tolerance = 1e-12)
    # complex: real valued function of a Hermitian matrix
    fc <- function(x) jnp_sum(jnp_abs(lax_linalg_cholesky(x))^2 * as_jax(arange(3, 3)))
    Hc <- as_jax(fx$Hc)
    expect_jax_equal(grad(fc)(Hc), fx$chol_grad_c, tolerance = 1e-10)
    # forward mode agrees with reverse mode: jvp(f, d) = Re(sum(grad * d))
    d <- fx$Zc
    expect_equal(as.numeric(host(jvp(fc, list(Hc), list(as_jax(d)))[[2L]])), sum(Re(fx$chol_grad_c * d)),
                 tolerance = 1e-10)
  })
})

# ---- triangular_solve -----------------------------------------------------------------------

test_that("triangular_solve matches jax", {
  skip_if_no_backend()
  x64({
    L <- fx$A
    L[upper.tri(L)] <- 0
    L <- L + 4 * diag(4)
    U <- fx$A
    U[lower.tri(U)] <- 0
    U <- U + 4 * diag(4)
    expect_jax_equal(lax_linalg_triangular_solve(L, fx$B, left_side = TRUE, lower = TRUE),
                     fx$trsm_left_lower, dtype = "float64")
    # the other triangle is ignored
    expect_jax_equal(lax_linalg_triangular_solve(L + t(U) * 0 + 7 * upper.tri(L), fx$B,
                                                 left_side = TRUE, lower = TRUE), fx$trsm_left_lower)
    expect_jax_equal(lax_linalg_triangular_solve(U, t(fx$B), left_side = FALSE, lower = FALSE,
                                                 transpose_a = TRUE), fx$trsm_right_upper_T)
    expect_jax_equal(lax_linalg_triangular_solve(L, fx$B, left_side = TRUE, lower = TRUE,
                                                 unit_diagonal = TRUE), fx$trsm_left_lower_unit)
    expect_jax_equal(lax_linalg_triangular_solve(L, fx$B[, 1], left_side = TRUE, lower = TRUE),
                     fx$trsm_vec, shape = 4L)
    Lc <- fx$Zc
    Lc[upper.tri(Lc)] <- 0
    Lc <- Lc + 3 * diag(3)
    expect_jax_equal(lax_linalg_triangular_solve(Lc, fx$Bc, left_side = TRUE, lower = TRUE,
                                                 transpose_a = TRUE, conjugate_a = TRUE),
                     fx$trsm_c_adjoint, dtype = "complex128")
    expect_jax_equal(lax_linalg_triangular_solve(Lc, fx$Bc, left_side = TRUE, lower = TRUE,
                                                 conjugate_a = TRUE), fx$trsm_c_conj)
    f <- jit(function(a, b) lax_linalg_triangular_solve(a, b, left_side = TRUE, lower = TRUE))
    expect_jax_equal(f(as_jax(L), as_jax(fx$B)), fx$trsm_left_lower)
  })
  L <- diag(2) * 2
  expect_jax_equal(lax_linalg_triangular_solve(as_jax(L, "float32"), as_jax(matrix(c(2, 4), 2), "float32"),
                                               left_side = TRUE, lower = TRUE),
                   matrix(c(1, 2), 2), dtype = "float32")
})

test_that("triangular_solve derivatives and vmap", {
  skip_if_no_backend()
  x64({
    L <- fx$A
    L[upper.tri(L)] <- 0
    L <- as_jax(L + 4 * diag(4))
    B <- as_jax(fx$B)
    w <- as_jax(arange(4, 2) + 1)
    for (opts in list(list(left_side = TRUE, lower = TRUE),
                      list(left_side = TRUE, lower = TRUE, transpose_a = TRUE, unit_diagonal = TRUE))) {
      solve <- function(a, b) do.call(lax_linalg_triangular_solve, c(list(a, b), opts))
      check_grad_fd(function(a) jnp_sum(solve(a, B) * w), L)
      check_grad_fd(function(b) jnp_sum(solve(L, b) * w), B)
    }
    right <- function(a, b) lax_linalg_triangular_solve(a, b, left_side = FALSE, lower = TRUE)
    Bt <- as_jax(t(fx$B))
    check_grad_fd(function(a) jnp_sum(right(a, Bt)^2), L)
    check_grad_fd(function(b) jnp_sum(right(L, b)^2), Bt)
    check_hvp_fd(function(a) jnp_sum(lax_linalg_triangular_solve(a, B, left_side = TRUE, lower = TRUE)^2),
                 L, as_jax(fx$S))
    solve <- function(a, b) lax_linalg_triangular_solve(a, b, left_side = TRUE, lower = TRUE)
    bs <- array(fx$M[, 1:4], c(4, 2, 2))
    bs <- aperm(array(c(fx$B, 2 * fx$B, fx$B + 1), c(4, 2, 3)), c(3, 1, 2))
    as_ <- aperm(array(c(host(L), host(L) + diag(4), 2 * host(L)), c(4, 4, 3)), c(3, 1, 2))
    ref <- function(i, a, b) host(solve(a, b))
    out <- host(vmap(solve, in_axes = list(NULL, 0L))(L, as_jax(bs)))
    for (i in 1:3) expect_equal(out[i, , ], host(solve(L, bs[i, , ])), tolerance = 1e-12)
    out <- host(vmap(solve)(as_jax(as_), as_jax(bs)))
    for (i in 1:3) expect_equal(out[i, , ], host(solve(as_[i, , ], bs[i, , ])), tolerance = 1e-12)
    out <- host(vmap(solve, in_axes = list(0L, NULL))(as_jax(as_), B))
    for (i in 1:3) expect_equal(out[i, , ], host(solve(as_[i, , ], B)), tolerance = 1e-12)
    # mapped axis in the middle of b, right side solve
    out <- host(vmap(right, in_axes = list(NULL, 1L), out_axes = 1L)(L, as_jax(aperm(bs, c(3, 1, 2)))))
    for (i in 1:3) expect_equal(out[, i, ], host(right(L, t(bs[i, , ]))), tolerance = 1e-12)
  })
})

test_that("triangular_solve errors", {
  expect_error(lax_linalg_triangular_solve(matrix(1, 2, 3), matrix(1, 2, 2)), "must be a square matrix")
  expect_error(lax_linalg_triangular_solve(diag(3), matrix(1, 2, 2), left_side = TRUE),
               "Incompatible shapes for arguments to triangular_solve: \\(3, 3\\) and \\(2, 2\\)")
  expect_error(lax_linalg_triangular_solve(as_jax(diag(2), "float32"), as_jax(diag(2), "int32")),
               "triangular_solve does not accept dtype int32 at position 1")
  expect_error(lax_linalg_triangular_solve(as_jax(diag(2), "float32"), as_jax(diag(2), "complex64")),
               "requires arguments to have the same dtypes")
})

# ---- lu ---------------------------------------------------------------------------------------

test_that("lu matches jax", {
  skip_if_no_backend()
  x64({
    f <- lax_linalg_lu(fx$M)
    expect_jax_equal(f[[1L]], fx$lu_M, dtype = "float64")
    expect_jax_equal(f[[2L]], fx$lu_M_piv, dtype = "int32")
    expect_jax_equal(f[[3L]], fx$lu_M_perm, dtype = "int32")
    lu <- host(f[[1L]])
    l <- diag(4)
    l[lower.tri(l)] <- lu[, 1:4][lower.tri(l)]
    u <- lu
    u[lower.tri(u)] <- 0
    expect_lt(resid(fx$M[host(f[[3L]]) + 1, ] - l %*% u), 1e-12)
    g <- jit(lax_linalg_lu)(as_jax(fx$M))
    expect_jax_equal(g[[1L]], fx$lu_M)
    expect_jax_equal(g[[3L]], fx$lu_M_perm)
    # a singular matrix is not an error for LU
    z <- lax_linalg_lu(matrix(0, 2, 2))
    expect_jax_equal(z[[1L]], matrix(0, 2, 2))
    expect_jax_equal(z[[3L]], c(0, 1))
    tall <- lax_linalg_lu(t(fx$M))
    expect_identical(lapply(tall, dim), list(c(5L, 4L), 4L, 5L))
  })
  f <- lax_linalg_lu(as_jax(fx$X32, "float32"))
  expect_jax_equal(f[[1L]], fx$lu_X32, dtype = "float32")
  expect_jax_equal(f[[2L]], fx$lu_X32_piv)
  expect_jax_equal(f[[3L]], fx$lu_X32_perm)
  expect_identical(dtype(lax_linalg_lu(as_jax(fx$Zc, "complex64"))[[1L]]), "complex64")
})

test_that("lu derivatives and vmap", {
  skip_if_no_backend()
  x64({
    w <- as_jax(arange(4, 5))
    f <- function(x) jnp_sum(lax_linalg_lu(x)[[1L]] * w)
    expect_jax_equal(grad(f)(as_jax(fx$M)), fx$lu_grad, tolerance = 1e-10)
    check_grad_fd(f, as_jax(fx$M))
    check_grad_fd(function(x) jnp_sum(lax_linalg_lu(x)[[1L]]^2), as_jax(t(fx$M)))
    xs <- aperm(array(c(fx$A, t(fx$A), fx$S), c(4, 4, 3)), c(3, 1, 2))
    out <- vmap(lax_linalg_lu)(as_jax(xs))
    for (i in 1:3) {
      one <- lax_linalg_lu(xs[i, , ])
      expect_equal(host(out[[1L]])[i, , ], host(one[[1L]]), tolerance = 1e-12)
      expect_equal(host(out[[2L]])[i, ], as.vector(host(one[[2L]])))
      expect_equal(host(out[[3L]])[i, ], as.vector(host(one[[3L]])))
    }
    # batched input: gradient through the batched rule
    check_grad_fd(function(x) jnp_sum(lax_linalg_lu(x)[[1L]]^2), as_jax(xs[1:2, , ]))
  })
})

test_that("lu_pivots_to_permutation", {
  skip_if_no_backend()
  piv <- as_jax(rbind(c(2L, 2L, 3L), c(0L, 1L, 2L)), "int32")
  expect_jax_equal(lax_linalg_lu_pivots_to_permutation(piv, 5L), fx$p2p, dtype = "int32")
  expect_jax_equal(jit(function(p) lax_linalg_lu_pivots_to_permutation(p, 5L))(piv), fx$p2p)
  expect_jax_equal(vmap(function(p) lax_linalg_lu_pivots_to_permutation(p, 5L))(piv), fx$p2p)
  expect_jax_equal(lax_linalg_lu_pivots_to_permutation(as_jax(integer(0), "int32"), 3L), 0:2)
  expect_error(lax_linalg_lu_pivots_to_permutation(piv, 2L),
               "Output permutation size 2 has to exceed the trailing dimension of the pivots")
  expect_error(lax_linalg_lu_pivots_to_permutation(as_jax(c(1, 2), "float32"), 2L),
               "lu_pivots_to_permutation does not accept dtype float32 at position 0")
})

# ---- qr, householder_product, ormqr ----------------------------------------------------------------

test_that("qr matches jax", {
  skip_if_no_backend()
  x64({
    f <- lax_linalg_qr(fx$T)
    expect_jax_equal(f[[1L]], fx$qr_T_q, tolerance = 1e-10, dtype = "float64")
    expect_jax_equal(f[[2L]], fx$qr_T_r, tolerance = 1e-10)
    f <- lax_linalg_qr(fx$T, full_matrices = FALSE)
    expect_jax_equal(f[[1L]], fx$qr_T_q_red, tolerance = 1e-10)
    expect_jax_equal(f[[2L]], fx$qr_T_r_red, tolerance = 1e-10)
    f <- jit(function(x) lax_linalg_qr(x, full_matrices = FALSE))(as_jax(fx$T))
    expect_jax_equal(f[[1L]], fx$qr_T_q_red, tolerance = 1e-10)
    f <- lax_linalg_qr(fx$M)
    expect_jax_equal(f[[1L]], fx$qr_M_q, tolerance = 1e-10)
    expect_jax_equal(f[[2L]], fx$qr_M_r, tolerance = 1e-10)
    f <- lax_linalg_qr(fx$Zc)
    expect_jax_equal(f[[1L]], fx$qr_Zc_q, tolerance = 1e-10, dtype = "complex128")
    expect_jax_equal(f[[2L]], fx$qr_Zc_r, tolerance = 1e-10)
    f <- lax_linalg_qr(fx$Tp, pivoting = TRUE, full_matrices = FALSE)
    expect_jax_equal(f[[3L]], fx$qr_Tp_p, dtype = "int32")
    expect_jax_equal(f[[2L]], fx$qr_Tp_r, tolerance = 1e-10)
    q <- host(f[[1L]])
    expect_lt(resid(q %*% host(f[[2L]]) - fx$Tp[, host(f[[3L]]) + 1]), 1e-12)
    expect_lt(resid(crossprod(q) - diag(3)), 1e-12)
    # empty matrices
    e <- lax_linalg_qr(matrix(0, 3, 0))
    expect_jax_equal(e[[1L]], diag(3))
    expect_identical(dim(e[[2L]]), c(3L, 0L))
    e <- lax_linalg_qr(matrix(0, 0, 3), pivoting = TRUE, full_matrices = FALSE)
    expect_identical(lapply(e, dim), list(c(0L, 0L), c(0L, 3L), 3L))
  })
  f <- lax_linalg_qr(as_jax(fx$T, "float32"), full_matrices = FALSE)
  expect_jax_equal(f[[1L]], fx$qr_T_q_red, dtype = "float32")
  expect_jax_equal(f[[2L]], fx$qr_T_r_red, dtype = "float32")
})

test_that("qr derivatives and vmap", {
  skip_if_no_backend()
  x64({
    fun <- function(...) function(x) {
      f <- lax_linalg_qr(x, ...)
      jnp_sum(f[[1L]] * as_jax(arange(dim(f[[1L]])))) + jnp_sum(f[[2L]]^2)
    }
    check_grad_fd(fun(full_matrices = FALSE), as_jax(fx$T))      # tall, reduced
    check_grad_fd(fun(), as_jax(fx$A))                           # square
    check_grad_fd(fun(), as_jax(fx$M))                           # wide
    check_grad_fd(fun(), as_jax(fx$T))                           # tall, full matrices
    check_grad_fd(fun(pivoting = TRUE, full_matrices = FALSE), as_jax(fx$Tp))
    xs <- aperm(array(c(fx$T, 2 * fx$T + 1, fx$Tp), c(5, 3, 3)), c(3, 1, 2))
    out <- vmap(function(x) lax_linalg_qr(x, full_matrices = FALSE))(as_jax(xs))
    for (i in 1:3) {
      one <- lax_linalg_qr(xs[i, , ], full_matrices = FALSE)
      expect_equal(host(out[[1L]])[i, , ], host(one[[1L]]), tolerance = 1e-12)
      expect_equal(host(out[[2L]])[i, , ], host(one[[2L]]), tolerance = 1e-12)
    }
  })
})

test_that("householder_product and ormqr match jax", {
  skip_if_no_backend()
  x64({
    a <- as_jax(fx$geqrf_T_a)
    taus <- as_jax(fx$geqrf_T_taus)
    expect_jax_equal(lax_linalg_householder_product(a, taus), fx$householder_T, tolerance = 1e-10)
    expect_jax_equal(lax_linalg_ormqr(a, taus, fx$C), fx$ormqr_left, tolerance = 1e-10)
    expect_jax_equal(lax_linalg_ormqr(a, taus, fx$C, transpose = TRUE), fx$ormqr_left_T, tolerance = 1e-10)
    expect_jax_equal(lax_linalg_ormqr(a, taus, fx$C2, left = FALSE), fx$ormqr_right, tolerance = 1e-10)
    out <- host(vmap(function(c) lax_linalg_ormqr(a, taus, c))(as_jax(aperm(array(c(fx$C, 2 * fx$C), c(5, 2, 2)), c(3, 1, 2)))))
    expect_equal(out[2, , ], 2 * fx$ormqr_left, tolerance = 1e-10)
  })
  expect_error(lax_linalg_householder_product(matrix(1, 2, 3), c(1, 1)),
               "must have at least as many rows as columns, got shape \\(2, 3\\)")
  expect_error(lax_linalg_householder_product(matrix(1, 3, 2), c(1, 1, 1)), "must not have more rows")
  expect_error(lax_linalg_ormqr(matrix(1, 3, 2), c(1, 1), matrix(1, 2, 2)),
               "ormqr with left=True expects c to have the same number of rows")
  expect_error(lax_linalg_ormqr(matrix(1, 3, 2), c(1, 1), matrix(1, 2, 2), left = FALSE),
               "ormqr with left=False expects c to have the same number of columns")
})

# ---- eigh --------------------------------------------------------------------------------------

test_that("eigh matches jax", {
  skip_if_no_backend()
  x64({
    f <- lax_linalg_eigh(fx$S)
    expect_jax_equal(f[[2L]], fx$eigh_S_w, tolerance = 1e-10, dtype = "float64")
    v <- host(f[[1L]])
    expect_lt(resid(fx$S %*% v - v %*% diag(fx$eigh_S_w)), 1e-11)
    expect_lt(resid(crossprod(v) - diag(4)), 1e-12)
    f <- lax_linalg_eigh(fx$Hc)
    expect_identical(c(dtype(f[[1L]]), dtype(f[[2L]])), c("complex128", "float64"))
    expect_jax_equal(f[[2L]], fx$eigh_Hc_w, tolerance = 1e-10)
    v <- host(f[[1L]])
    expect_lt(resid(fx$Hc %*% v - v %*% diag(fx$eigh_Hc_w)), 1e-11)
    expect_lt(resid(H(v) %*% v - diag(3)), 1e-12)
    # the triangle selected by `lower` is the only one read without symmetrization
    up <- fx$S
    up[lower.tri(up)] <- 99
    expect_jax_equal(lax_linalg_eigh(up, lower = FALSE, symmetrize_input = FALSE)[[2L]], fx$eigh_S_w,
                     tolerance = 1e-10)
    expect_jax_equal(jit(function(x) lax_linalg_eigh(x)[[2L]])(as_jax(fx$S)), fx$eigh_S_w, tolerance = 1e-10)
    expect_true(all(is.nan(host(lax_linalg_eigh(matrix(NaN, 2, 2))[[2L]]))))
  })
  f <- lax_linalg_eigh(as_jax(fx$S, "float32"))
  expect_jax_equal(f[[2L]], fx$eigh_S_w, dtype = "float32")
  f <- lax_linalg_eigh(as_jax(fx$Hc, "complex64"))
  expect_identical(c(dtype(f[[1L]]), dtype(f[[2L]])), c("complex64", "float32"))
  expect_error(lax_linalg_eigh(matrix(1, 2, 3)), "must have shape \\[..., n, n\\], got shape \\(2, 3\\)")
  expect_error(lax_linalg_eigh(diag(3), subset_by_index = c(0, 2)), "subset_by_index not supported")
  expect_error(lax_linalg_eigh(diag(3), implementation = "jacobi"), "Jacobi implementation is not supported on CPU")
  expect_error(lax_linalg_eigh(diag(3), implementation = "nope"), "implementation must be NULL or one of")
})

test_that("eigh derivatives and vmap", {
  skip_if_no_backend()
  x64({
    S <- as_jax(fx$S)
    fw <- function(x) jnp_sum(lax_linalg_eigh(x)[[2L]] * as_jax(c(1, 2, 3, 4)))
    expect_jax_equal(grad(fw)(S), fx$eigh_grad, tolerance = 1e-10)
    check_grad_fd(fw, S)
    # eigenvectors: a function that does not depend on their signs
    fv <- function(x) jnp_sum(lax_linalg_eigh(x)[[1L]]^2 * as_jax(arange(4, 4)))
    check_grad_fd(fv, S)
    check_hvp_fd(fw, S, as_jax(fx$A))
    xs <- array(0, c(3, 4, 4))
    for (i in 1:3) xs[i, , ] <- fx$S + i * (fx$A + t(fx$A))
    out <- vmap(function(x) lax_linalg_eigh(x)[[2L]])(as_jax(xs))
    for (i in 1:3) expect_equal(host(out)[i, ], as.vector(host(lax_linalg_eigh(xs[i, , ])[[2L]])), tolerance = 1e-10)
    check_grad_fd(function(x) jnp_sum(lax_linalg_eigh(x)[[2L]]^2), as_jax(xs[1:2, , ]))
  })
})

# ---- eig ---------------------------------------------------------------------------------------

csort <- function(w) w[order(round(Re(w), 8), Im(w))]

test_that("eig matches jax", {
  skip_if_no_backend()
  x64({
    for (nm in c("A", "Ar", "Zc")) {
      a <- fx[[nm]]
      n <- nrow(a)
      f <- lax_linalg_eig(a)
      expect_identical(vapply(f, dtype, ""), rep("complex128", 3L))
      w <- as.vector(host(f[[1L]]))
      expect_equal(csort(w), csort(fx[[paste0("eig_", nm, "_w")]]), tolerance = 1e-10)
      vl <- host(f[[2L]])
      vr <- host(f[[3L]])
      expect_lt(resid(a %*% vr - vr %*% diag(w)), 1e-11)
      expect_lt(resid(H(vl) %*% a - diag(w) %*% H(vl)), 1e-11)
      expect_equal(sqrt(colSums(Mod(vr)^2)), rep(1, n), tolerance = 1e-12)
      only <- lax_linalg_eig(a, compute_left_eigenvectors = FALSE, compute_right_eigenvectors = FALSE)
      expect_length(only, 1L)
      expect_equal(csort(as.vector(host(only[[1L]]))), csort(w), tolerance = 1e-10)
      right <- lax_linalg_eig(a, compute_left_eigenvectors = FALSE)
      expect_length(right, 2L)
      expect_lt(resid(a %*% host(right[[2L]]) - host(right[[2L]]) %*% diag(as.vector(host(right[[1L]])))), 1e-11)
    }
    w <- jit(function(x) lax_linalg_eig(x, compute_left_eigenvectors = FALSE, compute_right_eigenvectors = FALSE)[[1L]])(as_jax(fx$Ar))
    expect_equal(csort(as.vector(host(w))), csort(fx$eig_Ar_w), tolerance = 1e-10)
    # non finite input gives NaN (the kernel reports info = -4)
    bad <- fx$A
    bad[1, 1] <- Inf
    expect_true(all(is.nan(Re(host(lax_linalg_eig(bad)[[1L]])))))
    two <- aperm(array(c(fx$A, fx$Ar), c(4, 4, 2)), c(3, 1, 2))
    out <- host(lax_linalg_eig(two)[[1L]])
    expect_equal(csort(out[2, ]), csort(fx$eig_Ar_w), tolerance = 1e-10)
  })
  f <- lax_linalg_eig(as_jax(fx$Ar, "float32"))
  expect_identical(vapply(f, dtype, ""), rep("complex64", 3L))
  expect_equal(csort(as.vector(host(f[[1L]]))), csort(fx$eig_Ar_w), tolerance = 1e-5)
  expect_error(lax_linalg_eig(matrix(1, 2, 3)), "The input to eig must be a square matrix. Got shape \\(2, 3\\)")
  expect_error(lax_linalg_eig(diag(2), implementation = "magma"), "Only the lapack implementation is supported on CPU")
})

test_that("eig derivatives", {
  skip_if_no_backend()
  x64({
    A <- as_jax(fx$A)
    vals <- function(x) lax_linalg_eig(x, compute_left_eigenvectors = FALSE, compute_right_eigenvectors = FALSE)[[1L]]
    f <- function(x) jnp_sum(jnp_abs(vals(x))^2)
    expect_jax_equal(grad(f)(A), fx$eig_grad, tolerance = 1e-9)
    check_grad_fd(f, A)
    check_grad_fd(f, as_jax(fx$Ar))
    expect_error(jvp(function(x) lax_linalg_eig(x)[[3L]], list(A), list(A)),
                 "Derivatives of non-symmetric eigenvectors are only valid under")
    # eigenvector derivatives under the LAPACK normalization, against finite differences
    Z <- fx$Zc
    V <- fx$Hc
    g <- function(x) lax_linalg_eig(as_jax(x), enable_eigvec_derivs = TRUE)
    out <- jvp(function(x) lax_linalg_eig(x, enable_eigvec_derivs = TRUE), list(as_jax(Z)), list(as_jax(V)))
    eps <- 1e-6
    p <- g(Z + eps * V)
    m <- g(Z - eps * V)
    for (i in 1:3) {
      fd <- (host(p[[i]]) - host(m[[i]])) / (2 * eps)
      expect_equal(as.vector(host(out[[2L]][[i]])), as.vector(fd), tolerance = 1e-5)
    }
  })
})

# ---- svd ---------------------------------------------------------------------------------------

test_that("svd matches jax", {
  skip_if_no_backend()
  x64({
    for (alg in list(NULL, "QR", "divide_and_conquer")) {
      f <- lax_linalg_svd(fx$M, full_matrices = FALSE, algorithm = alg)
      expect_jax_equal(f[[2L]], fx$svd_M_s, tolerance = 1e-10, dtype = "float64")
      u <- host(f[[1L]])
      vt <- host(f[[3L]])
      expect_identical(c(dim(u), dim(vt)), c(4L, 4L, 4L, 5L))
      expect_lt(resid(u %*% diag(fx$svd_M_s) %*% vt - fx$M), 1e-11)
      expect_lt(resid(crossprod(u) - diag(4)), 1e-12)
      expect_lt(resid(tcrossprod(vt) - diag(4)), 1e-12)
      expect_jax_equal(lax_linalg_svd(fx$M, compute_uv = FALSE, algorithm = alg), fx$svd_M_s, tolerance = 1e-10)
    }
    f <- lax_linalg_svd(fx$M)
    vt <- host(f[[3L]])
    expect_identical(dim(vt), c(5L, 5L))
    expect_lt(resid(crossprod(vt) - diag(5)), 1e-12)
    expect_lt(resid(host(f[[1L]]) %*% cbind(diag(fx$svd_M_s), 0) %*% vt - fx$M), 1e-11)
    f <- lax_linalg_svd(t(fx$M), full_matrices = FALSE)
    expect_lt(resid(host(f[[1L]]) %*% diag(fx$svd_M_s) %*% host(f[[3L]]) - t(fx$M)), 1e-11)
    f <- lax_linalg_svd(fx$Zc)
    expect_identical(vapply(f, dtype, ""), c("complex128", "float64", "complex128"))
    expect_jax_equal(f[[2L]], fx$svd_Zc_s, tolerance = 1e-10)
    expect_lt(resid(host(f[[1L]]) %*% diag(fx$svd_Zc_s) %*% host(f[[3L]]) - fx$Zc), 1e-11)
    expect_jax_equal(jit(function(x) lax_linalg_svd(x, compute_uv = FALSE))(as_jax(fx$M)), fx$svd_M_s,
                     tolerance = 1e-10)
    # empty matrices do not reach LAPACK
    e <- lax_linalg_svd(matrix(0, 0, 3))
    expect_identical(lapply(e, dim), list(c(0L, 0L), 0L, c(3L, 3L)))
    expect_identical(dim(lax_linalg_svd(matrix(0, 3, 0), compute_uv = FALSE)), 0L)
    expect_true(all(is.nan(host(lax_linalg_svd(matrix(NaN, 2, 2), compute_uv = FALSE)))))
  })
  f <- lax_linalg_svd(as_jax(fx$M, "float32"), full_matrices = FALSE)
  expect_jax_equal(f[[2L]], fx$svd_M_s, dtype = "float32")
  expect_lt(resid(host(f[[1L]]) %*% diag(as.vector(host(f[[2L]]))) %*% host(f[[3L]]) - fx$M), 1e-5)
  expect_identical(dtype(lax_linalg_svd(as_jax(fx$Zc, "complex64"), compute_uv = FALSE)), "float32")
  expect_error(lax_linalg_svd(diag(3), algorithm = "Jacobi"), "Jacobi and Polar algorithms are not implemented on CPU")
  expect_error(lax_linalg_svd(diag(3), subset_by_index = c(0, 2)), "full_matrices and subset_by_index cannot both be set")
  expect_error(lax_linalg_svd(diag(3), full_matrices = FALSE, subset_by_index = c(0, 2)),
               "subset_by_index not implemented for CPU and GPU")
})

test_that("svd derivatives and vmap", {
  skip_if_no_backend()
  x64({
    M <- as_jax(fx$M)
    fs <- function(x) jnp_sum(lax_linalg_svd(x, compute_uv = FALSE) * as_jax(c(1, 2, 3, 4)))
    expect_jax_equal(grad(fs)(M), fx$svd_grad, tolerance = 1e-10)
    check_grad_fd(fs, M)
    # singular vectors: a function that does not depend on their signs
    fuv <- function(x) {
      f <- lax_linalg_svd(x, full_matrices = FALSE)
      jnp_sum(f[[1L]]^2 * as_jax(arange(dim(f[[1L]])))) + jnp_sum(f[[3L]]^2 * as_jax(arange(dim(f[[3L]]))))
    }
    check_grad_fd(fuv, M)             # wide
    check_grad_fd(fuv, as_jax(fx$T))  # tall
    check_grad_fd(fuv, as_jax(fx$A))  # square
    check_hvp_fd(fs, M, as_jax(fx$M[, 5:1]))
    expect_error(grad(function(x) jnp_sum(lax_linalg_svd(x)[[1L]]))(M),
                 "Singular value decomposition JVP not implemented for full matrices")
    xs <- aperm(array(c(fx$M, 2 * fx$M + 1, fx$M^2), c(4, 5, 3)), c(3, 1, 2))
    out <- host(vmap(function(x) lax_linalg_svd(x, compute_uv = FALSE))(as_jax(xs)))
    for (i in 1:3) expect_equal(out[i, ], svd(xs[i, , ])$d, tolerance = 1e-10)
    check_grad_fd(function(x) jnp_sum(lax_linalg_svd(x, compute_uv = FALSE)^2), as_jax(xs[1:2, , ]))
  })
})

# ---- schur, hessenberg, tridiagonal --------------------------------------------------------------

test_that("schur decomposition", {
  skip_if_no_backend()
  x64({
    f <- lax_linalg_schur(fx$Ar)
    t_ <- host(f[[1L]])
    q <- host(f[[2L]])
    expect_lt(resid(q %*% t_ %*% t(q) - fx$Ar), 1e-11)
    expect_lt(resid(crossprod(q) - diag(4)), 1e-12)
    expect_identical(t_[row(t_) > col(t_) + 1], rep(0, 3))
    only <- lax_linalg_schur(fx$Ar, compute_schur_vectors = FALSE)
    expect_length(only, 1L)
    expect_equal(host(only[[1L]]), t_, tolerance = 1e-10)
    f <- lax_linalg_schur(fx$Zc)
    t_ <- host(f[[1L]])
    q <- host(f[[2L]])
    expect_lt(resid(q %*% t_ %*% H(q) - fx$Zc), 1e-11)
    expect_lt(resid(t_[lower.tri(t_)]), 1e-13)
    expect_equal(csort(diag(t_)), csort(fx$eig_Zc_w), tolerance = 1e-10)
    two <- aperm(array(c(fx$A, fx$Ar), c(4, 4, 2)), c(3, 1, 2))
    out <- vmap(lax_linalg_schur)(as_jax(two))
    expect_lt(resid(host(out[[2L]])[2, , ] %*% host(out[[1L]])[2, , ] %*% t(host(out[[2L]])[2, , ]) - fx$Ar), 1e-11)
    expect_error(lax_linalg_schur(fx$A, sort_eig_vals = TRUE), "sort feature of LAPACK's gees routine is not implemented")
  })
  f <- lax_linalg_schur(as_jax(fx$Zc, "complex64"))
  expect_identical(vapply(f, dtype, ""), rep("complex64", 2L))
  expect_error(lax_linalg_schur(matrix(1, 2, 3)), "The input to schur must be a square matrix")
})

test_that("hessenberg and tridiagonal match jax", {
  skip_if_no_backend()
  x64({
    f <- lax_linalg_hessenberg(fx$A)
    expect_jax_equal(f[[1L]], fx$hess_A, tolerance = 1e-10, dtype = "float64")
    expect_jax_equal(f[[2L]], fx$hess_A_taus, tolerance = 1e-10)
    f <- lax_linalg_tridiagonal(fx$S)
    expect_jax_equal(f[[1L]], fx$trid_S_a, tolerance = 1e-10)
    expect_jax_equal(f[[2L]], fx$trid_S_d, tolerance = 1e-10)
    expect_jax_equal(f[[3L]], fx$trid_S_e, tolerance = 1e-10)
    expect_jax_equal(f[[4L]], fx$trid_S_taus, tolerance = 1e-10)
    f <- lax_linalg_tridiagonal(fx$Hc, lower = FALSE)
    expect_identical(vapply(f, dtype, ""), c("complex128", "float64", "float64", "complex128"))
    expect_jax_equal(f[[1L]], fx$trid_Hc_a, tolerance = 1e-10)
    expect_jax_equal(f[[2L]], fx$trid_Hc_d, tolerance = 1e-10)
    expect_jax_equal(f[[3L]], fx$trid_Hc_e, tolerance = 1e-10)
    expect_jax_equal(f[[4L]], fx$trid_Hc_taus, tolerance = 1e-10)
    two <- aperm(array(c(fx$S, fx$A), c(4, 4, 2)), c(3, 1, 2))
    out <- vmap(lax_linalg_hessenberg)(as_jax(two))
    expect_equal(host(out[[1L]])[2, , ], fx$hess_A, tolerance = 1e-10)
    expect_equal(host(jit(lax_linalg_tridiagonal)(as_jax(two))[[2L]])[1, ], fx$trid_S_d, tolerance = 1e-10)
  })
  expect_identical(vapply(lax_linalg_hessenberg(as_jax(fx$A, "float32")), dtype, ""), rep("float32", 2L))
  expect_error(lax_linalg_hessenberg(matrix(1, 2, 3)), "Argument to Hessenberg reduction must have shape")
  expect_error(lax_linalg_tridiagonal(matrix(1, 2, 3)), "The input to tridiagonal must be a square matrix")
})

# ---- cholesky_update, symmetric_product -------------------------------------------------------------

test_that("cholesky_update and symmetric_product match jax", {
  skip_if_no_backend()
  x64({
    r <- t(fx$chol_S)
    expect_jax_equal(lax_linalg_cholesky_update(r, fx$wv), fx$chol_update, tolerance = 1e-10)
    expect_jax_equal(jit(lax_linalg_cholesky_update)(as_jax(r), as_jax(fx$wv)), fx$chol_update, tolerance = 1e-10)
    # a zero update leaves the factor unchanged (the y == 0 branch of the rotation)
    expect_jax_equal(lax_linalg_cholesky_update(r, rep(0, 4)), r, tolerance = 1e-12)
    expect_jax_equal(lax_linalg_symmetric_product(fx$M, fx$S, alpha = 2, beta = 3, symmetrize_output = TRUE),
                     fx$sym_prod, tolerance = 1e-10)
    out <- host(lax_linalg_symmetric_product(fx$M, fx$S))
    expect_equal(out[lower.tri(out, TRUE)], tcrossprod(fx$M)[lower.tri(out, TRUE)], tolerance = 1e-12)
    ms <- aperm(array(c(fx$M, 2 * fx$M), c(4, 5, 2)), c(3, 1, 2))
    out <- host(vmap(function(m) lax_linalg_symmetric_product(m, as_jax(fx$S), alpha = 2, beta = 3, symmetrize_output = TRUE))(as_jax(ms)))
    expect_equal(out[1, , ], fx$sym_prod, tolerance = 1e-10)
  })
  expect_error(lax_linalg_cholesky_update(matrix(1, 2, 3), c(1, 2)), "Rank-1 update to Cholesky decomposition takes a square matrix")
  expect_error(lax_linalg_cholesky_update(array(1, c(2, 2, 2)), matrix(1, 2, 2)), "must have a rank of exactly 2")
  expect_error(lax_linalg_symmetric_product(matrix(1, 2, 3), matrix(1, 3, 3)), "symmetric_update expects a rectangular matrix")
  expect_error(lax_linalg_symmetric_product(as_jax(diag(2) + 0i), as_jax(diag(2) + 0i)),
               "symmetric_product does not accept dtype complex")
})

# ---- tridiagonal_solve ------------------------------------------------------------------------------

test_that("tridiagonal_solve matches jax", {
  skip_if_no_backend()
  x64({
    expect_jax_equal(lax_linalg_tridiagonal_solve(fx$td_dl, fx$td_d, fx$td_du, fx$td_b), fx$td_x,
                     tolerance = 1e-10, dtype = "float64")
    expect_jax_equal(lax_linalg_tridiagonal_solve(fx$td_dl, fx$td_d, fx$td_du, fx$td_b, perturb_singular = TRUE),
                     fx$td_x_pert, tolerance = 1e-10)
    expect_jax_equal(jit(lax_linalg_tridiagonal_solve)(as_jax(fx$td_dl), as_jax(fx$td_d), as_jax(fx$td_du), as_jax(fx$td_b)),
                     fx$td_x, tolerance = 1e-10)
    # a singular system: NaN without the perturbation, the perturbed pivots of jaxlib with it
    sdl <- c(0, 1, 0)
    sd <- c(1, 1, 1)
    sdu <- c(1, 0, 0)
    sb <- matrix(c(1, 2, 3), 3)
    expect_true(all(is.nan(host(lax_linalg_tridiagonal_solve(sdl, sd, sdu, sb)))))
    expect_jax_equal(lax_linalg_tridiagonal_solve(sdl, sd, sdu, sb, perturb_singular = TRUE), fx$td_sing)
    cplx <- lax_linalg_tridiagonal_solve(fx$td_dl + 0i, fx$td_d + 0i, fx$td_du + 0i, fx$td_b + 0i)
    expect_jax_equal(cplx, fx$td_x + 0i, tolerance = 1e-10, dtype = "complex128")
  })
  f32 <- function(x) as_jax(x, "float32")
  expect_jax_equal(lax_linalg_tridiagonal_solve(f32(c(0, 1, 0)), f32(c(1, 1, 1)), f32(c(1, 0, 0)), f32(matrix(c(1, 2, 3), 3)),
                                                perturb_singular = TRUE), fx$td_sing32, dtype = "float32")
  expect_error(lax_linalg_tridiagonal_solve(c(1, 2), c(1, 2, 3), c(1, 2), matrix(1, 2, 1)),
               "tridiagonal_solve requires that all diagonal arguments have the same shape")
  expect_error(lax_linalg_tridiagonal_solve(c(1, 2), c(1, 2), c(1, 2), matrix(1, 3, 1)),
               "leading ndim-1 dimensions of b equal the dimensions of the diagonal arguments")
})

test_that("tridiagonal_solve derivatives and vmap", {
  skip_if_no_backend()
  x64({
    dl <- as_jax(fx$td_dl)
    d <- as_jax(fx$td_d)
    du <- as_jax(fx$td_du)
    b <- as_jax(fx$td_b)
    loss <- function(dl, d, du, b) jnp_sum(lax_linalg_tridiagonal_solve(dl, d, du, b)^2)
    expect_jax_equal(grad(function(x) loss(dl, x, du, b))(d), fx$td_grad_d, tolerance = 1e-10)
    check_grad_fd(function(x) loss(dl, x, du, b), d)
    check_grad_fd(function(x) loss(dl, d, du, x), b)
    # dl[0] and du[m - 1] are not part of the matrix
    g <- host(grad(function(x) loss(x, d, du, b))(dl))
    expect_identical(g[1], 0)
    check_grad_fd(function(x) loss(x, d, du, b), dl)
    check_grad_fd(function(x) loss(dl, d, x, b), du)
    check_hvp_fd(function(x) loss(dl, x, du, b), d, as_jax(c(1, -1, 2, 0.5, 1)))
    bs <- aperm(array(c(fx$td_b, 2 * fx$td_b, fx$td_b + 1), c(5, 2, 3)), c(3, 1, 2))
    out <- host(vmap(function(b) lax_linalg_tridiagonal_solve(dl, d, du, b))(as_jax(bs)))
    for (i in 1:3) expect_equal(out[i, , ], host(lax_linalg_tridiagonal_solve(dl, d, du, bs[i, , ])), tolerance = 1e-12)
    ds <- rbind(fx$td_d, fx$td_d + 1, fx$td_d + 2)
    out <- host(vmap(function(d, b) lax_linalg_tridiagonal_solve(dl, d, du, b))(as_jax(ds), as_jax(bs)))
    for (i in 1:3) expect_equal(out[i, , ], host(lax_linalg_tridiagonal_solve(dl, ds[i, ], du, bs[i, , ])), tolerance = 1e-12)
  })
})

# ---- shape and dtype rules ---------------------------------------------------------------------------

test_that("shape and dtype rules of the linalg primitives", {
  expect_error(lax_linalg_cholesky(matrix(1, 2, 3)), "The input to linalg.cholesky must have shape \\[..., n, n\\], got shape \\(2, 3\\)")
  expect_error(lax_linalg_cholesky(c(1, 2)), "got shape \\(2,\\)")
  expect_error(lax_linalg_cholesky(as_jax(diag(2), "int32"), symmetrize_input = FALSE),
               "cholesky does not accept dtype int32 at position 0. Accepted dtypes at position 0 are subtypes of float32, float64, complex64, complex128.")
  expect_error(lax_linalg_lu(c(1, 2)), "Input 0 to lu must have rank at least 2, but got shape=\\(2,\\)")
  expect_error(lax_linalg_tridiagonal_solve(matrix(1, 2, 2), matrix(1, 2, 2), matrix(1, 2, 2), matrix(1, 2, 2)),
               "All inputs to tridiagonal_solve must have the same number of batch dimensions")
  # abstract evaluation only: shapes and dtypes of the results
  s <- eval_shape(function(x) lax_linalg_svd(x, full_matrices = FALSE), ShapeDtypeStruct(c(7L, 4L, 6L), "complex64"))
  expect_identical(lapply(s, function(a) a$shape), list(c(7L, 4L, 4L), c(7L, 4L), c(7L, 4L, 6L)))
  expect_identical(vapply(s, function(a) a$dtype, ""), c("complex64", "float32", "complex64"))
  s <- eval_shape(function(x) lax_linalg_qr(x, pivoting = TRUE), ShapeDtypeStruct(c(2L, 5L, 3L), "float32"))
  expect_identical(lapply(s, function(a) a$shape), list(c(2L, 5L, 5L), c(2L, 5L, 3L), c(2L, 3L)))
  expect_identical(vapply(s, function(a) a$dtype, ""), c("float32", "float32", "int32"))
  s <- eval_shape(lax_linalg_tridiagonal, ShapeDtypeStruct(c(4L, 4L), "complex128"))
  expect_identical(lapply(s, function(a) a$shape), list(c(4L, 4L), 4L, 3L, 3L))
  expect_error(eval_shape(function(x) lax_linalg_eigh(x, symmetrize_input = FALSE),
                          ShapeDtypeStruct(c(3L, 3L), "float16")),
               "eigh does not accept dtype float16 at position 0")
})

# ---- qdwh --------------------------------------------------------------------------------------

test_that("qdwh polar decomposition matches jax", {
  skip_if_no_backend()
  x64({
    f <- lax_linalg_qdwh(fx$T)
    expect_jax_equal(f[[1L]], fx$qdwh_T_u, tolerance = 1e-10, dtype = "float64")
    expect_jax_equal(f[[2L]], fx$qdwh_T_h, tolerance = 1e-10)
    expect_jax_equal(f[[3L]], fx$qdwh_T_iters)
    expect_jax_equal(f[[4L]], fx$qdwh_T_conv)
    u <- host(f[[1L]])
    expect_lt(resid(crossprod(u) - diag(3)), 1e-12)
    expect_lt(resid(u %*% host(f[[2L]]) - fx$T), 1e-12)
    g <- jit(lax_linalg_qdwh)(as_jax(fx$T))
    expect_jax_equal(g[[1L]], fx$qdwh_T_u, tolerance = 1e-10)
    expect_jax_equal(g[[3L]], fx$qdwh_T_iters)
    f <- lax_linalg_qdwh(fx$Zc)
    expect_jax_equal(f[[1L]], fx$qdwh_Zc_u, tolerance = 1e-10, dtype = "complex128")
    expect_jax_equal(f[[2L]], fx$qdwh_Zc_h, tolerance = 1e-10)
    expect_jax_equal(f[[3L]], fx$qdwh_Zc_iters)
    # a matrix padded from its true (dynamic) shape
    pad <- matrix(0, 7, 4)
    pad[1:5, 1:3] <- fx$T
    f <- lax_linalg_qdwh(pad, dynamic_shape = list(5L, 3L))
    expect_jax_equal(f[[1L]], fx$qdwh_pad_u, tolerance = 1e-10)
    expect_jax_equal(f[[2L]], fx$qdwh_pad_h, tolerance = 1e-10)
    f <- lax_linalg_qdwh(fx$T, max_iterations = 2L)
    expect_jax_equal(f[[3L]], fx$qdwh_T_max2_iters)
    expect_jax_equal(f[[4L]], fx$qdwh_T_max2_conv)
  })
  f <- lax_linalg_qdwh(as_jax(fx$T, "float32"))
  expect_identical(dtype(f[[1L]]), "float32")
  expect_jax_equal(f[[3L]], fx$qdwh_T32_iters)
  expect_error(lax_linalg_qdwh(matrix(1, 2, 3)), "The input matrix of shape M x N must have M >= N.")
})

# ---- complex kernels and single precision values -----------------------------------------------

test_that("complex LAPACK kernels (geqp3, unmqr, gesvd, gehrd)", {
  skip_if_no_backend()
  x64({
    Zt <- rbind(fx$Zc, t(fx$Bc))  # 5 x 3 complex
    f <- lax_linalg_qr(Zt, pivoting = TRUE, full_matrices = FALSE)
    q <- host(f[[1L]])
    r <- host(f[[2L]])
    p <- as.vector(host(f[[3L]]))
    expect_lt(resid(q %*% r - Zt[, p + 1]), 1e-12)
    expect_lt(resid(H(q) %*% q - diag(3)), 1e-12)
    expect_true(all(diff(Mod(diag(r))) <= 1e-12))
    # Q and Q^H applied from the left and from the right, against the explicit Q
    raw <- jnp_linalg_qr(Zt, mode = "raw")
    a <- jnp_linalg_matrix_transpose(raw[[1L]])
    taus <- raw[[2L]]
    Q <- host(lax_linalg_qr(Zt)[[1L]])
    expect_lt(resid(H(Q) %*% Q - diag(5)), 1e-12)
    Cc <- fx$C + 1i * fx$C[5:1, ]
    expect_lt(resid(host(lax_linalg_ormqr(a, taus, Cc)) - Q %*% Cc), 1e-12)
    expect_lt(resid(host(lax_linalg_ormqr(a, taus, Cc, transpose = TRUE)) - H(Q) %*% Cc), 1e-12)
    expect_lt(resid(host(lax_linalg_ormqr(a, taus, t(Cc), left = FALSE)) - t(Cc) %*% Q), 1e-12)
    expect_lt(resid(host(lax_linalg_ormqr(a, taus, t(Cc), left = FALSE, transpose = TRUE)) - t(Cc) %*% H(Q)), 1e-12)
    f <- lax_linalg_svd(Zt, full_matrices = FALSE, algorithm = "QR")
    u <- host(f[[1L]])
    expect_lt(resid(u %*% diag(as.vector(host(f[[2L]]))) %*% host(f[[3L]]) - Zt), 1e-12)
    expect_lt(resid(H(u) %*% u - diag(3)), 1e-12)
    expect_jax_equal(lax_linalg_svd(fx$Zc, compute_uv = FALSE, algorithm = "QR"), fx$svd_Zc_s, tolerance = 1e-10)
    f <- lax_linalg_hessenberg(fx$Zc)
    expect_jax_equal(f[[1L]], fx$hess_Zc, tolerance = 1e-10, dtype = "complex128")
    expect_jax_equal(f[[2L]], fx$hess_Zc_taus, tolerance = 1e-10)
  })
})

test_that("single precision results match the double precision kernels to float accuracy", {
  skip_if_no_backend()
  c64 <- function(x) as_jax(x, "complex64")
  f32 <- function(x) as_jax(x, "float32")
  expect_jax_equal(lax_linalg_cholesky(c64(fx$Hc)), fx$chol_Hc, tolerance = 1e-5, dtype = "complex64")
  expect_jax_equal(lax_linalg_eigh(c64(fx$Hc))[[2L]], fx$eigh_Hc_w, tolerance = 1e-5, dtype = "float32")
  expect_jax_equal(lax_linalg_svd(c64(fx$Zc), compute_uv = FALSE), fx$svd_Zc_s, tolerance = 1e-5, dtype = "float32")
  f <- lax_linalg_qr(c64(fx$Zc))
  expect_jax_equal(f[[1L]], fx$qr_Zc_q, tolerance = 1e-5, dtype = "complex64")
  expect_jax_equal(f[[2L]], fx$qr_Zc_r, tolerance = 1e-5)
  expect_equal(csort(as.vector(host(lax_linalg_eig(c64(fx$Zc))[[1L]]))), csort(fx$eig_Zc_w), tolerance = 1e-5)
  f <- lax_linalg_lu(c64(fx$Zc))
  lu <- host(f[[1L]])
  l <- diag(3) + 0i
  l[lower.tri(l)] <- lu[lower.tri(lu)]
  u <- lu
  u[lower.tri(u)] <- 0
  expect_lt(resid(fx$Zc[as.vector(host(f[[3L]])) + 1, ] - l %*% u), 1e-5)
  Lc <- fx$Zc
  Lc[upper.tri(Lc)] <- 0
  Lc <- Lc + 3 * diag(3)
  expect_jax_equal(lax_linalg_triangular_solve(c64(Lc), c64(fx$Bc), left_side = TRUE, lower = TRUE,
                                               transpose_a = TRUE, conjugate_a = TRUE),
                   fx$trsm_c_adjoint, tolerance = 1e-5, dtype = "complex64")
  expect_jax_equal(lax_linalg_hessenberg(f32(fx$A))[[1L]], fx$hess_A, tolerance = 1e-5, dtype = "float32")
  expect_jax_equal(lax_linalg_tridiagonal(f32(fx$S))[[2L]], fx$trid_S_d, tolerance = 1e-5, dtype = "float32")
  expect_jax_equal(lax_linalg_tridiagonal_solve(f32(fx$td_dl), f32(fx$td_d), f32(fx$td_du), f32(fx$td_b)),
                   fx$td_x, tolerance = 1e-5, dtype = "float32")
  expect_jax_equal(lax_linalg_tridiagonal_solve(c64(fx$td_dl), c64(fx$td_d), c64(fx$td_du), c64(fx$td_b)),
                   fx$td_x + 0i, tolerance = 1e-5, dtype = "complex64")
  expect_jax_equal(lax_linalg_schur(f32(fx$A))[[1L]], host(with_jax_config("jax_enable_x64", TRUE, lax_linalg_schur(fx$A)[[1L]])),
                   tolerance = 1e-4, dtype = "float32")
  # dtypes without a LAPACK kernel fall back to the StableHLO triangular solve
  half <- lax_linalg_triangular_solve(as_jax(matrix(c(2, 1, 0, 4), 2), "float16"), as_jax(matrix(c(2, 4), 2), "float16"),
                                      left_side = TRUE, lower = TRUE, transpose_a = TRUE)
  expect_jax_equal(half, matrix(c(0.5, 1), 2), tolerance = 1e-3, dtype = "float16")
})
