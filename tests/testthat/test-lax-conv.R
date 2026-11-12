# Convolutions. Expected values from jax 0.12.0.1 (x64 enabled).

conv_expected <- list(
  c_same = list(v = c(1.176004441722903, -1.9959199219285544, -3.332804714015213, -1.6055242220526973, -1.5239092688588285, 3.3321875739426376, 2.0029094241619725, -1.1678344133031544, -3.2648766769216935, -3.081582065282747, 0.714426777473046, 3.1322192510139084, 2.6702637901417394, -0.24671988483434887, -0.22434733778594412, -1.3094173039818082, -0.20981488970101808, 1.082690366559949, 1.3797750928881418, 1.4092537032369759, -3.4273397536379386, -2.43768266377056, 0.7931686252178928, 3.29478433806554, 3.003581530131144, -0.30454549501153, -3.0962837915651615, -3.041313049398077, -0.1901731153481276, -0.14549768784193304, 3.2545636722620377, 0.6810854095110837, -2.518579637758068, -3.40267418109754, -3.086125914357322, 0.16203309047786962, 1.925276746357544, 1.9184298405047115, 0.14778738658434212, -0.31132301217733355, -0.17864834166705232, 2.7052857419151293, 3.1019925904449432, 0.6467417568914446, 0.6498668404968113, -3.2435648143017612, -1.1018906314543866, 2.05285671632315, 3.320217066346977, 3.1239219022828992, -1.66150502616587, -3.3304151512347406, -1.9373569452547694, 1.2368983016131263, 1.1224101875392016, 1.2622656636965197, -0.3504407736709023, -1.6409535798657262, -1.4227812323771587, -1.3186586856425728), shape = c(1, 3, 4, 5L), dtype = "float64"),
  c_valid_s = list(v = c(1.176004441722903, -1.9959199219285542, -3.332804714015213, -1.6055242220526977, 0.7144267774730461, 3.1322192510139084, 2.670263790141739, -0.2467198848343491, -3.4273397536379386, -2.4376826637705604, 0.7931686252178932, 3.29478433806554, 3.2545636722620377, 0.6810854095110837, -2.518579637758068, -3.4026741810975394, -3.1143246809303595, -0.6650951151938742, 2.3956198322085878, 3.253812953845366, 1.9552623716984305, -1.2230603188371192, -3.2769069926654657, -2.317980489668045, 1.7670303965096787, 3.4943898705755374, 2.009023412838787, -1.3234299055757845, -3.1245851575702495, -3.010268749479605, -0.12832513568301399, 2.8716000160588555), shape = c(2, 2, 2, 4L), dtype = "float64"),
  c_pad_dil = list(v = c(-1.3435111301093323, -0.3428209693859263, -0.26324660067948946, 0.9063127918135904, -0.35853210530754714, 0.21635964417249007, 0.1433717552846596, 0.3308185470479918, -0.5517323852590109, 1.6969276832847715, -0.1725249149900342, -0.22177534693415032, 1.4248494209848233, 0.5305023934807312, -0.049764627707329916, 0.05639563861362509, 0.26065451644168636, -0.3421782033589028, 0.6649800461404753, -0.029851610391558613, 0.5234996991505609, -0.18230516464279342, 0.5952051811651304, -0.07791233975704266, -1.1833047625698219, -0.4831776347767371, -0.21994008958035496, -1.9321144073777539, -0.010209339431296179, 0.5713964630287218, 0.6774394211217767, 0.3069313893898568, -0.6684594461670695, -0.9138304260807262, -0.6009971882355652, 0.4020794786944121, 1.567632655998514, 0.6573072921799199, -0.15929324520622254, 1.4136761357680538, -0.3307510123130126, -0.34328697571597844), shape = c(1, 2, 7, 3L), dtype = "float64"),
  c_nhwc = list(v = c(-1.3435111301093323, -0.18230516464279342, -1.5907239715534698, 0.28783538834159295, -1.751975494781941, 1.880448356864911, -0.3024688277568457, 1.7441857782185393, 1.425126284601427, 0.004326838802873918, 1.0496845470982499, -3.1154191699475757, 1.176004441722903, -3.4273397536379386, -1.9959199219285544, -2.43768266377056, -3.332804714015213, 0.7931686252178928, -1.6055242220526973, 3.29478433806554, 3.1217771042695945, -0.23639100495894955, 3.3321875739426376, -0.30454549501153, 2.0029094241619725, -3.0962837915651615, -1.1678344133031544, -3.041313049398077, -3.2648766769216935, -0.1901731153481276, 0.7213756847541004, 2.9813087917665677, 0.714426777473046, 3.2545636722620377, 3.1322192510139084, 0.6810854095110837, 2.6702637901417394, -2.518579637758068, -0.24671988483434887, -3.40267418109754), shape = c(1, 4, 5, 2L), dtype = "float64"),
  c_fgc = list(v = c(-0.15313677367145895, -1.71529485890583, 1.1928591377679858, 0.9205691177040543, 1.6110606464082773, -0.0014254000239985682, -1.7520567436240235, 0.9647182624080383, 0.11108215438928114, 1.9441018124717202, -1.2839940362268867, -1.0807479669305589, -1.4743574986110426, -0.9239896409536682, 2.156981382308192, -0.3368466437467293), shape = c(1, 4, 2, 2L), dtype = "float64"),
  c_bgc = list(v = c(1.176004441722903, -3.332804714015213, 0.7144267774730461, 2.670263790141739, 1.7670303965096787, 2.009023412838787, -3.1245851575702495, -0.12832513568301399), shape = c(1, 2, 2, 2L), dtype = "float64"),
  c_int = list(v = c(7.0, 9.0, 11.0, 15.0, 17.0, 19.0), shape = c(1, 1, 2, 3L), dtype = "int32"),
  c_int_pet = list(v = c(7.0, 9.0, 11.0, 15.0, 17.0, 19.0), shape = c(1, 1, 2, 3L), dtype = "float32"),
  c_complex = list(v = c(complex(real = 0.9408035533783219, imaginary = -0.9996037754644671), complex(real = -1.5967359375428432, imaginary = 1.696531933639271), complex(real = 2.6657500591541097, imaginary = -2.8323594378512427), complex(real = 1.6023275393295782, imaginary = -1.7024730105376766), complex(real = -2.741871802910351, imaginary = 2.913238790592248), complex(real = -1.950146131016448, imaginary = 2.072030264204976), complex(real = -0.2436363960092239, imaginary = 0.25886367075979966), complex(real = -2.477027033252129, imaginary = 2.6318412228303867)), shape = c(1, 2, 2, 2L), dtype = "complex128"),
  conv_gp = list(v = c(-0.40458677813087074, -1.186137193422599, -1.8137413035268855, 0.061765808744944466, -1.5553504389568529, 1.2528816112000072, 0.1330224463240637, 1.2921038382773633, 1.699095107919078, 3.0869801237120624, -1.9109756819891595, 1.1187737600430634, -3.114693681971618, -1.8780280391199902, -1.454776674895223, -3.148179520086167, 1.5426552980334693, -1.5239092688588285, 1.501436136289655, 1.8307514376529823, 3.1463680819609308, -1.143458657798958, 1.8985437232971587, -3.0663781366003127, -1.0947929789830544, -2.1700836979386393, -3.081582065282747, -2.2351788122052905, 2.9496055896783364, 0.6662375327584905, 2.465981718255418, 2.955118162615916, -0.2848543724741774, 2.527076781989909, -2.773796666824258, -0.22434733778594412, -0.873769965850214, -0.4356473381315941, -1.4039854301106847, 1.1941705404096665, -0.6433831647379358, 1.7260735312978848, 0.7087426151813889, 0.671032477706753, 1.4092537032369759), shape = c(1, 1, 5, 9L), dtype = "float64"),
  ct_same = list(v = c(-0.0772955576174739, -0.14111447988036266, -0.0751934001240124, -0.10720716438562058, -0.1351449135818443, -0.03883105248362137, 0.8962988523129701, 0.46083790322296847, -0.39831528882735323, 0.8528179814761836, 0.22754065819028088, -0.6069364968782577, -0.1285787099007546, -0.1184097809460741, 0.0006245545357443771, -0.1397077490994629, -0.09224219304455371, 0.040030409898850025, 0.7414021191561774, -0.023882345224618604, -0.7672094915449782, 0.5709265890705677, -0.27340289471069096, -0.8663670179569887, -0.6686883063642799, -0.2424379912941503, 0.4067086949117411, -0.6025878458980459, -0.05423573660163573, 0.5439804588054042, -0.3397538056538472, -0.25905829695115984, 0.05981421525987734, -0.3512879298517261, -0.18240059776113132, 0.15418500272759347, -0.4884855830916268, 0.13828690577517766, 0.6379190512150241, -0.3354708387852712, 0.31979369251792467, 0.6810413777243143, -0.33483865895124376, -0.09121297152384727, 0.2362735012724046, -0.29171633244195366, 0.007240627732563099, 0.299540588161627, 0.9514632492196822, 0.45805057280918104, -0.45649168783363664, 0.8949625673480265, 0.20604660008684086, -0.6723076610616053, -0.452143036853425, -0.12217429675513461, 0.3201209283441898, -0.3935837526082541, 0.010909316146265328, 0.4053724099467976, 0.767169680444122, -0.062370926671750426, -0.8345679914438808, 0.5782644966149783, -0.32582002116896797, -0.9303471140861772, -0.3036718122721633, 0.1431238991777727, 0.4583321577733428, -0.18956954946574417, 0.2639373144538149, 0.47478142867382517), shape = c(1, 6, 4, 3L), dtype = "float64"),
  ct_valid_tk = list(v = c(0.07624746575887677, -1.4274064235791089, 1.1117738695201562, 0.8932724961381865, -1.174683886856056, 0.0844094707779117, 0.21650548086302587, 0.5064327170461436, -0.6380062271084793, 0.8879249375153375, -1.510052637413241, 0.3688823186430411, 1.2276100466612871, -0.7526081663977305, -0.6012190314493651, -0.16931151898344862, 0.6794772910081656, -0.3962131313338918, -0.4018311210957004, 1.1319659946926877, -0.5402970144449023, -0.7390271383103839, -0.18894499492999997, 0.8962848621534545, 0.35135201951490247, -1.4287427085440523, 0.8377814972859041, -0.908379244645406, 0.9960680816079817, 0.07935808235192114, -0.3306548425408431, -0.6419376462130946, 0.864935484005039, 1.0533050373941708, -1.3491702735873667, 0.06960084523947972, 0.44906123217847943, -0.05239827935170825, -0.4054504757930257, 0.07284825306515878, 1.42168926811626, -1.2561112360243663, -0.6758234933299524, 1.3613443975442936, -0.45721483564851384, 0.2995876596594144, 0.20790554243541526, -0.4726261272296459, -0.7953491895923188, 1.5918050612765133, -0.5295000917075836, -1.2076607180489674, 1.084275794570227, 0.30522483433888903), shape = c(1, 6, 3, 3L), dtype = "float64"),
  ct_pads = list(v = c(-0.10720716438562058, -0.1351449135818443, -0.03883105248362137, 0.8962988523129701, 0.46083790322296847, -0.39831528882735323, 0.8528179814761836, 0.22754065819028088, -0.6069364968782577, 0.0, 0.0, 0.0, -0.1397077490994629, -0.09224219304455371, 0.040030409898850025, 0.7414021191561774, -0.023882345224618604, -0.7672094915449782, 0.5709265890705677, -0.27340289471069096, -0.8663670179569887, 0.0, 0.0, 0.0, -0.6025878458980459, -0.05423573660163573, 0.5439804588054042, -0.3397538056538472, -0.25905829695115984, 0.05981421525987734, -0.3512879298517261, -0.18240059776113132, 0.15418500272759347, 0.0, 0.0, 0.0, -0.3354708387852712, 0.31979369251792467, 0.6810413777243143, -0.33483865895124376, -0.09121297152384727, 0.2362735012724046, -0.29171633244195366, 0.007240627732563099, 0.299540588161627, 0.0, 0.0, 0.0, 0.8949625673480265, 0.20604660008684086, -0.6723076610616053, -0.452143036853425, -0.12217429675513461, 0.3201209283441898, -0.3935837526082541, 0.010909316146265328, 0.4053724099467976, 0.0, 0.0, 0.0), shape = c(1, 5, 4, 3L), dtype = "float64"),
  ct_cons = list(v = c(-0.1285787099007546, -0.1184097809460741, 0.0006245545357443771, -0.1397077490994629, -0.09224219304455371, 0.040030409898850025, 0.7414021191561774, -0.023882345224618604, -0.7672094915449782, -0.6686883063642799, -0.2424379912941503, 0.4067086949117411, -0.6025878458980459, -0.05423573660163573, 0.5439804588054042, -0.3397538056538472, -0.25905829695115984, 0.05981421525987734, -0.4884855830916268, 0.13828690577517766, 0.6379190512150241, -0.3354708387852712, 0.31979369251792467, 0.6810413777243143, -0.33483865895124376, -0.09121297152384727, 0.2362735012724046, 0.9514632492196822, 0.45805057280918104, -0.45649168783363664, 0.8949625673480265, 0.20604660008684086, -0.6723076610616053, -0.452143036853425, -0.12217429675513461, 0.3201209283441898, 0.767169680444122, -0.062370926671750426, -0.8345679914438808, 0.5782644966149783, -0.32582002116896797, -0.9303471140861772, -0.3036718122721633, 0.1431238991777727, 0.4583321577733428), shape = c(1, 5, 3, 3L), dtype = "float64"),
  ct_1d = list(v = c(0.0, -2.1140352825377846, 0.0, -0.10907817447379757, 0.0, 2.6594690704378414, 0.0, -1.698930630848531), shape = c(1, 8, 1L), dtype = "float64"),
  g_l = list(v = c(-1.120694576518412, 0.0, -2.362522811873985, 0.0, -1.0151350230699585, -2.814837710692542, 0.0, -2.011555045161007, 0.0, -0.9112887765045592, -1.0473492057504923, 0.0, -2.3094582054721875, 0.0, -1.096748192692682, -1.3676951478171515, 0.0, -0.7084766307070782, 0.0, 0.22851360009598418, 0.8193602927755025, 0.0, 1.2233750941307724, 0.0, -0.024014283840983926, 1.2781522820562017, 0.0, 0.9729289815911097, 0.0, 0.11452012739534745, 1.3692205379156897, 0.0, 1.337673569615304, 0.0, 0.012906185298117864, -0.229293003354573, 0.0, -1.1136151832015335, 0.0, -1.0277972486352938), shape = c(1, 2, 4, 5L), dtype = "float64"),
  g_r = list(v = c(2.0905006539105364, 0.4275195899745474, 0.2519415623039422, 0.5448671918130236, 3.0480518042736215, -2.0784937959977534, 1.676662419803455, -0.05910594963639888, 2.0322928281747608, 1.5541754670501677, -0.05464312479182744, 2.1148949372400723, 3.0439275485691444, -0.23561052700847696, 1.5349964076764266, 0.3421023496390746), shape = c(2, 2, 2, 2L), dtype = "float64"),
  g_fgc_l = list(v = c(0.5225610178489415, -0.512065666457376, 0.3493535144090371, 1.5787938187020798, -0.3364940410806444, -0.8736904059642533, 1.1777236613230144, 0.8781435057556384, -1.0975376191282877, 0.32522470683694377, 1.135283781803827, -1.030211938180593, 0.4630351906104202, 1.4524201470605451, -0.3379209885730497, -0.501216783403351, 0.7193080379308991, -0.032933709021484445, -1.1955308078205429, -0.34538764924876525, -0.8181371945102199, -1.0727523520373419, 1.0693029022289362, -0.19375067481563213, -0.7756621211671042, 0.6914689767387074, 1.3883318159811981, -1.4042471968891348, -0.664877392118111, 0.9731944049620359, -1.0348306625000032, -1.0767011759796352, 1.535105235338644, -0.6249245264356875, -1.0705259045613966, 0.4215014400245445, 0.8520099646681465, -1.0950110536509856, -0.15518187459448857, 0.8814529900329262), shape = c(1, 2, 4, 5L), dtype = "float64"),
  g_fgc_r = list(v = c(2.555380098436591, 3.287917786443259, -1.448589631213941, 1.8202895271492439, 0.11812800510642414, 0.6346613223124165, -0.6170059199883431, -0.07612668660471529, 2.837685504208977, 0.7162463725807928, 1.7359379747488612, 2.995938329820594, 1.4480567554580854, 1.9829271216849564, -0.9573520662478231, 0.9917349728837695), shape = c(4, 1, 2, 2L), dtype = "float64"),
  g_bgc_l = list(v = c(0.20780896090219153, -0.3828954750480747, -0.3588222374005949, 0.38980260570842307, 0.014448999029651795, -0.9112853967829927, 0.33924169397336607, 1.627689028012277, -0.02328712649143754, 0.4356834472484987, 1.380247026631086, 0.20174222815928106, -0.17974279157444523, 1.6208695033157805, 0.24513718247714303, -0.7479085739041644, 0.4961415581773425, 1.5356642413871904, -0.3776402267263118, -0.6338504562746093, 0.10910104096935665, 0.2523051889371759, -0.6744986055199254, -0.952519815036878, -0.03333799111684462, 0.011437391120784629, -1.4284744061133305, -0.9710266795100335, 0.21168855217454877, -0.9478308393087578, -0.5259524019329804, 0.2688671929297138, -0.8562885322069667, -1.3856435554917308, 1.0754908123830946, 0.5695497307759918, -0.8637898905243441, -0.5262073905496227, 0.8607086662881511, -0.14109410674843, 0.17764950732596352, 1.018613346123814, 1.1740172440665868, 0.13295568971113791, -0.20544783611581094, 0.6666170060413216, -0.15303481795684876, -0.4479621073155171, 1.3041248100768656, 0.9752446037322228, 0.9077562928169653, 1.126950487530929, 0.4786569334978993, -0.7892975913402204, 0.03566767215904783, -0.004425057926018803, -0.8481195072994155, -0.8321985629780222, 0.8326501052898567, -0.8132835433495269, -0.17693131310960988, -0.8782176467747572, -0.513376759284483, 0.16416725892759554, 0.033480298906358466, -0.5166597328619289, 1.3960672694055098, 0.4171071847904718, -0.7455886853055995, -0.3597414629421847, -0.35077994615316177, -0.8342556753077783, 0.8830841100627832, 0.9002391162056366, 0.7454738515462541, 0.7595780437736532, 1.7106675251538808, 0.19597198712872094, -0.2176186499334909, 0.9229662163931017), shape = c(2, 2, 4, 5L), dtype = "float64"),
  g_bgc_r = list(v = c(0.36377682985382387, -0.7352061090238026, 1.1650005943056252, 0.6586827400414156, -0.8624477787590312, -1.125674780048376, 0.507128883385425, -0.6090419520777112, 3.1645072428118586, 1.549159592355183, 1.0807030257270216, 3.091804304798445, 1.1171037313467995, -1.8930049627863723, 3.1619335072879755, 1.9016137679904868), shape = c(2, 2, 2, 2L), dtype = "float64"),
  hvp = list(v = c(0.29437725183022684, 5.256591092721687, -1.642526624254448, -2.1801492600126506, 0.2210206608105876, 2.915261727360206, -13.518617844190826, -6.649559538285898, -1.86726430121744, 0.6322663504652715, -2.761206246461946, -1.1549257315575723, -11.489969878511253, -10.20231351250098, -10.986743168609026, -6.557867763654308, -12.813901902236761, -4.665097917132785, -5.538633462326611, -7.578442068953648), shape = c(1, 1, 4, 5L), dtype = "float64"),
  convolve = list(v = c(4.0, 13.0, 26.0, 31.0, 26.0, 13.0, 4.0), shape = c(7L), dtype = "float64"),
  convolve_same = list(v = c(13.0, 26.0, 30.0, 24.0, 38.0, 37.0), shape = c(6L), dtype = "float64"),
  convolve_valid = list(v = c(13.0, 14.0, 7.0), shape = c(3L), dtype = "float64"),
  correlate_full = list(v = c(complex(real = 3.0, imaginary = 1.0), complex(real = 3.0, imaginary = 17.0), complex(real = 18.0, imaginary = 11.0), complex(real = 27.0, imaginary = 4.0), complex(real = 8.0, imaginary = -12.0)), shape = c(5L), dtype = "complex128"),
  correlate_valid = list(v = c(24.0, 30.0, 19.0), shape = c(3L), dtype = "float64"),
  correlate_swap = list(v = c(2.0, 13.0, 14.0, 8.0), shape = c(4L), dtype = "float64"),
  convolve_pet = list(v = c(1.0, 3.0, 5.0, 3.0), shape = c(4L), dtype = "int64"),
  g_convolve = list(v = c(11.173848945329544, 9.825369798108921, 4.903520584154732, 5.398658500594149), shape = c(4L), dtype = "float64"))

conv_l <- function() array_c(sin(seq_len(2 * 2 * 4 * 5)) * 2, c(2L, 2L, 4L, 5L))
conv_r <- function() array_c(cos(seq_len(3 * 2 * 2 * 2)), c(3L, 2L, 2L, 2L))
# Row major (C order) array from a flat vector.
array_c <- function(v, shape) aperm(array(v, rev(shape)), rev(seq_along(shape)))

ce <- function(name) {
  e <- conv_expected[[name]]
  if (length(e$shape) > 1L) array_c(e$v, e$shape) else e$v
}

conv_stack_loop <- function(f, xs, axis) {
  n <- dim(xs)[axis + 1L]
  jnp_stack(lapply(seq_len(n) - 1L, function(i) f(lax_index_in_dim(xs, i, axis, keepdims = FALSE))))
}

test_that("conv_general_dilated matches JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  l <- as_jax(conv_l(), "float64")
  r <- as_jax(conv_r(), "float64")
  l1 <- lax_slice_in_dim(l, 0L, 1L)
  r2 <- lax_slice_in_dim(r, 0L, 2L)
  expect_jax_equal(lax_conv_general_dilated(l1, r, c(1L, 1L), "SAME"), ce("c_same"), dtype = "float64")
  expect_jax_equal(lax_conv_general_dilated(l, r2, c(2L, 1L), "VALID"), ce("c_valid_s"))
  expect_jax_equal(lax_conv_general_dilated(l1, r2, c(1L, 2L), list(c(1L, 0L), c(2L, 1L)), c(2L, 1L), c(1L, 2L)),
                   ce("c_pad_dil"))
  lN <- jnp_transpose(l1, c(0L, 2L, 3L, 1L))
  rH <- jnp_transpose(r2, c(2L, 3L, 1L, 0L))
  expect_jax_equal(lax_conv_general_dilated(lN, rH, c(1L, 1L), "SAME_LOWER",
                                            dimension_numbers = c("NHWC", "HWIO", "NHWC")), ce("c_nhwc"))
  r4 <- as_jax(array_c(cos(seq_len(16)), c(4L, 1L, 2L, 2L)), "float64")
  expect_jax_equal(lax_conv_general_dilated(l1, r4, c(2L, 2L), "VALID", feature_group_count = 2L), ce("c_fgc"))
  rb <- as_jax(array_c(cos(seq_len(16)), c(2L, 2L, 2L, 2L)), "float64")
  expect_jax_equal(lax_conv_general_dilated(l, rb, c(2L, 2L), "VALID", batch_group_count = 2L), ce("c_bgc"))
  li <- as_jax(array_c(0:11, c(1L, 1L, 3L, 4L)))
  ri <- as_jax(array_c(c(1L, -1L, 2L, 0L), c(1L, 1L, 2L, 2L)))
  expect_jax_equal(lax_conv_general_dilated(li, ri, c(1L, 1L), "VALID"), ce("c_int"), dtype = "int32")
  expect_jax_equal(lax_conv_general_dilated(li, ri, c(1L, 1L), "VALID", preferred_element_type = "float32"),
                   ce("c_int_pet"), dtype = "float32")
  lc <- lax_mul(lax_convert_element_type(lax_slice(l1, c(0L, 0L, 0L, 0L), c(1L, 2L, 3L, 3L)), "complex128"),
                as_jax(1 + 0.5i, "complex128"))
  rc <- lax_mul(lax_convert_element_type(lax_slice(r2, c(0L, 0L, 0L, 0L), c(2L, 2L, 2L, 2L)), "complex128"),
                as_jax(0.3 - 1i, "complex128"))
  expect_jax_equal(lax_conv_general_dilated(lc, rc, c(1L, 1L), "VALID"), ce("c_complex"), dtype = "complex128")
  expect_jax_equal(lax_conv_with_general_padding(l1, lax_slice_in_dim(r, 0L, 1L), c(1L, 1L),
                                                 list(c(1L, 1L), c(0L, 1L)), c(1L, 2L), c(1L, 1L)),
                   ce("conv_gp"))
  expect_jax_equal(lax_conv(l1, r, c(1L, 1L), "SAME", precision = "highest"), ce("c_same"))
  f <- function(l, r) lax_conv_general_dilated(l, r, c(1L, 2L), list(c(1L, 0L), c(2L, 1L)), c(2L, 1L), c(1L, 2L))
  expect_jax_equal(jit(f)(l1, r2), ce("c_pad_dil"))
})

test_that("conv_transpose matches JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  x1 <- as_jax(array_c(sin(seq_len(12)), c(1L, 3L, 2L, 2L)), "float64")
  k1 <- as_jax(array_c(cos(seq_len(24)), c(2L, 2L, 2L, 3L)), "float64")
  k1b <- as_jax(array_c(cos(seq_len(24)), c(2L, 2L, 3L, 2L)), "float64")
  expect_jax_equal(lax_conv_transpose(x1, k1, c(2L, 2L), "SAME"), ce("ct_same"))
  expect_jax_equal(lax_conv_transpose(x1, k1b, c(2L, 1L), "VALID", transpose_kernel = TRUE), ce("ct_valid_tk"))
  expect_jax_equal(lax_conv_transpose(x1, k1, c(2L, 2L), list(c(1L, 0L), c(0L, 2L))), ce("ct_pads"))
  expect_jax_equal(lax_conv_transpose(x1, k1, c(2L, 2L), list(c(1L, 0L), c(0L, 1L)),
                                      use_consistent_padding = TRUE), ce("ct_cons"))
  x3 <- as_jax(array_c(sin(seq_len(8)), c(1L, 4L, 2L)), "float64")
  k3 <- as_jax(array_c(cos(seq_len(6)), c(3L, 2L, 1L)), "float64")
  expect_jax_equal(lax_conv_transpose(x3, k3, 2L, "SAME", rhs_dilation = 2L), ce("ct_1d"))
})

test_that("convolution dimension numbers and shape helpers match JAX", {
  dn <- lax_conv_dimension_numbers(c(1L, 5L, 4L, 2L), c(3L, 2L, 2L, 3L), c("NHWC", "HWIO", "NHWC"))
  expect_identical(dn$lhs_spec, c(0L, 3L, 1L, 2L))
  expect_identical(dn$rhs_spec, c(3L, 2L, 0L, 1L))
  expect_identical(dn$out_spec, c(0L, 3L, 1L, 2L))
  expect_identical(lax_conv_general_permutations(c("NWHC", "HWOI", "CNHW")),
                   list(c(0L, 3L, 2L, 1L), c(2L, 3L, 0L, 1L), c(1L, 0L, 2L, 3L)))
  expect_identical(lax_conv_general_shape_tuple(c(1L, 5L, 4L, 2L), c(3L, 2L, 2L, 3L), c(2L, 1L), "SAME",
                                                c("NHWC", "HWIO", "NHWC")), c(1L, 3L, 4L, 3L))
  expect_identical(lax_conv_transpose_shape_tuple(c(1L, 5L, 4L, 2L), c(3L, 2L, 2L, 3L), c(2L, 2L), "SAME",
                                                  c("NHWC", "HWIO", "NHWC")), c(1L, 10L, 8L, 3L))
  expect_identical(lax_conv_shape_tuple(c(2L, 3L, 7L, 8L), c(4L, 3L, 2L, 3L), c(2L, 3L),
                                        list(c(1L, 0L), c(0L, 2L))), c(2L, 4L, 4L, 3L))
  expect_identical(lax_conv_dimension_numbers(1:3, 1:3, NULL)$lhs_spec, 0:2)
})

test_that("conv_general_dilated errors match JAX", {
  skip_if_no_backend()
  l <- as_jax(conv_l())
  r <- as_jax(conv_r())
  expect_error(lax_conv_general_dilated(l, r, c(1L, 1L), "SAME", lhs_dilation = c(2L, 1L)),
               "String padding is not implemented for transposed convolution")
  r4 <- as_jax(array(1, c(4L, 1L, 2L, 2L)))
  expect_error(lax_conv_general_dilated(l, r4, c(1L, 1L), "VALID"),
               "lhs feature dimension size divided by feature_group_count must equal the rhs input feature dimension size, but 2 // 1 != 1")
  expect_error(lax_conv_general_dilated(l, r, 1L, "VALID"),
               "window and window_strides must have the same number of dimensions, but got 2 and 1")
  expect_error(lax_conv_general_dilated(l, r, c(1L, 1L), "VALID", dimension_numbers = c("NCHW", "OIHW")),
               "convolution dimension_numbers list/tuple must be length 3, got 2.")
  expect_error(lax_conv_general_dilated(l, r, c(1L, 1L), "VALID", dimension_numbers = c("NCHW", "OIHH", "NCHW")),
               "convolution dimension_numbers\\[1\\] cannot have duplicate characters, got OIHH.")
  expect_error(lax_conv_general_dilated(l, as_jax(conv_r(), "float16"), c(1L, 1L), "VALID"),
               "requires arguments to have the same dtypes")
  expect_error(lax_conv_general_dilated(l, r, c(1L, 1L), "VALID", preferred_element_type = "float16"),
               "`preferred_element_type` must not be narrower than the original type.")
  expect_error(jnp_convolve(jnp_ones(c(2L, 2L)), jnp_ones(2L)), "convolve\\(\\) only support 1-dimensional inputs.")
  expect_error(jnp_convolve(jnp_ones(2L), jnp_ones(2L), mode = "foo"), "mode must be one of")
  expect_error(jnp_convolve(jnp_ones(0L), jnp_ones(2L)), "convolve: inputs cannot be empty, got shapes \\(0,\\) and \\(2,\\).")
})

test_that("conv_general_dilated gradients match JAX and finite differences", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  l <- as_jax(conv_l(), "float64")
  r <- as_jax(conv_r(), "float64")
  l1 <- lax_slice_in_dim(l, 0L, 1L)
  r2 <- lax_slice_in_dim(r, 0L, 2L)
  f <- function(l, r) jnp_sum(jnp_sin(lax_conv_general_dilated(l, r, c(1L, 2L), list(c(1L, 0L), c(2L, 1L)),
                                                               c(2L, 1L), c(1L, 2L))))
  g <- grad(f, argnums = c(0L, 1L))(l1, r2)
  expect_jax_equal(g[[1L]], ce("g_l"))
  expect_jax_equal(g[[2L]], ce("g_r"))
  check_grad_fd(function(x) f(x, r2), l1)
  check_grad_fd(function(x) f(l1, x), r2)
  r4 <- as_jax(array_c(cos(seq_len(16)), c(4L, 1L, 2L, 2L)), "float64")
  f2 <- function(l, r) jnp_sum(jnp_sin(lax_conv_general_dilated(l, r, c(1L, 1L), "VALID", feature_group_count = 2L)))
  g2 <- grad(f2, argnums = c(0L, 1L))(l1, r4)
  expect_jax_equal(g2[[1L]], ce("g_fgc_l"))
  expect_jax_equal(g2[[2L]], ce("g_fgc_r"))
  rb <- as_jax(array_c(cos(seq_len(16)), c(2L, 2L, 2L, 2L)), "float64")
  f3 <- function(l, r) jnp_sum(jnp_sin(lax_conv_general_dilated(l, r, c(1L, 1L), "VALID", batch_group_count = 2L)))
  g3 <- grad(f3, argnums = c(0L, 1L))(l, rb)
  expect_jax_equal(g3[[1L]], ce("g_bgc_l"))
  expect_jax_equal(g3[[2L]], ce("g_bgc_r"))
  expect_jax_equal(jit(grad(f3, argnums = c(0L, 1L)))(l, rb)[[2L]], ce("g_bgc_r"))
  # second order
  w <- as_jax(array_c(cos(0:19), c(1L, 1L, 4L, 5L)), "float64")
  r11 <- lax_slice(r, c(0L, 0L, 0L, 0L), c(1L, 1L, 2L, 2L))
  f4 <- function(l) jnp_sum(lax_conv_general_dilated(l, r11, c(1L, 1L), "SAME")^3)
  l11 <- lax_slice(l, c(0L, 0L, 0L, 0L), c(1L, 1L, 4L, 5L))
  expect_jax_equal(jvp(grad(f4), list(l11), list(w))[[2L]], ce("hvp"))
  check_grad_fd(function(x) jnp_sum(grad(f4)(x) * w), l11)
})

test_that("conv_general_dilated batches like a stacked loop", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  ls <- as_jax(array_c(sin(seq_len(3 * 2 * 2 * 4 * 5)), c(3L, 2L, 2L, 4L, 5L)), "float64")
  rs <- as_jax(array_c(cos(seq_len(3 * 2 * 2 * 2 * 2)), c(3L, 2L, 2L, 2L, 2L)), "float64")
  l0 <- lax_index_in_dim(ls, 0L, 0L, keepdims = FALSE)
  r0 <- lax_index_in_dim(rs, 0L, 0L, keepdims = FALSE)
  r4s <- as_jax(array_c(cos(seq_len(3 * 4 * 1 * 2 * 2)), c(3L, 4L, 1L, 2L, 2L)), "float64")
  r40 <- lax_index_in_dim(r4s, 0L, 0L, keepdims = FALSE)
  cases <- list(
    list(function(l, r) lax_conv_general_dilated(l, r, c(1L, 2L), "SAME"), rs),
    list(function(l, r) lax_conv_general_dilated(l, r, c(1L, 1L), "VALID", feature_group_count = 2L), r4s),
    list(function(l, r) lax_conv_general_dilated(l, r, c(1L, 1L), "VALID", batch_group_count = 2L), rs),
    list(function(l, r) lax_conv_general_dilated(l, r, c(1L, 1L), list(c(1L, 0L), c(0L, 1L)), c(1L, 2L),
                                                 dimension_numbers = c("NCHW", "OIHW", "NCHW")), rs))
  for (cs in cases) {
    f <- cs[[1L]]
    rr <- cs[[2L]]
    rfix <- lax_index_in_dim(rr, 0L, 0L, keepdims = FALSE)
    for (ax in c(0L, 2L, 4L)) {
      la <- jnp_moveaxis(ls, 0L, ax)
      ra <- jnp_moveaxis(rr, 0L, ax)
      both <- jnp_stack(lapply(0:2, function(i) f(lax_index_in_dim(ls, i, 0L, keepdims = FALSE),
                                                  lax_index_in_dim(rr, i, 0L, keepdims = FALSE))))
      expect_jax_equal(vmap(f, in_axes = list(ax, ax))(la, ra), both)
      expect_jax_equal(vmap(function(l) f(l, rfix), in_axes = ax)(la), conv_stack_loop(function(l) f(l, rfix), ls, 0L))
      expect_jax_equal(vmap(function(r) f(l0, r), in_axes = ax)(ra), conv_stack_loop(function(r) f(l0, r), rr, 0L))
    }
  }
  gfun <- function(l) grad(function(x) jnp_sum(jnp_sin(lax_conv_general_dilated(x, r0, c(2L, 1L), "SAME"))))(l)
  expect_jax_equal(vmap(gfun, in_axes = 1L)(jnp_moveaxis(ls, 0L, 1L)), conv_stack_loop(gfun, ls, 0L))
})

test_that("jnp_convolve and jnp_correlate match JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  expect_jax_equal(jnp_convolve(as_jax(c(1L, 2L, 3L, 2L, 1L), "int64"), as_jax(c(4L, 5L, 4L), "int64")),
                   ce("convolve"), dtype = "float64")
  a <- as_jax(c(1, 2, 3, 2, 1, 7), "float64")
  v <- as_jax(c(4, 5, 4, -1), "float64")
  expect_jax_equal(jnp_convolve(a, v, mode = "same"), ce("convolve_same"))
  expect_jax_equal(jnp_convolve(as_jax(c(1, 2), "float64"), v, mode = "valid"), ce("convolve_valid"))
  expect_jax_equal(jnp_correlate(c(3 + 1i, 2, 2 - 3i), c(4, 2 - 5i, 1), mode = "full"), ce("correlate_full"),
                   dtype = "complex128")
  expect_jax_equal(jnp_correlate(a, v), ce("correlate_valid"))
  expect_jax_equal(jnp_correlate(as_jax(c(1, 2), "float64"), v, mode = "same"), ce("correlate_swap"))
  expect_jax_equal(jnp_convolve(as_jax(1:3, "int64"), as_jax(c(1L, 1L), "int64"), preferred_element_type = "int64"),
                   ce("convolve_pet"), dtype = "int64")
  g <- grad(function(a, v) jnp_sum(jnp_sin(jnp_convolve(a, v, mode = "same"))), argnums = 1L)(a, v)
  expect_jax_equal(g, ce("g_convolve"))
  check_grad_fd(function(x) jnp_sum(jnp_sin(jnp_correlate(a, x, mode = "full"))), v)
  expect_jax_equal(jit(function(a, v) jnp_convolve(a, v, mode = "same"))(a, v), ce("convolve_same"))
  expect_jax_equal(vmap(function(x) jnp_convolve(x, v), in_axes = 1L)(jnp_stack(list(a, a * 2), 1L)),
                   jnp_stack(list(jnp_convolve(a, v), jnp_convolve(a * 2, v))))
})

test_that("convolutions without spatial dimensions match JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  l <- as_jax(array_c(sin(1:6), c(2L, 3L)), "float64")
  r <- as_jax(array_c(cos(1:12), c(4L, 3L)), "float64")
  expect_jax_equal(lax_conv_general_dilated(l, r, integer(0), "VALID"),
                   array_c(c(-0.06346028334058615, -0.1565896074410973, 0.37350535616517727,
                             -0.5829453926461077, 0.2667704167385043, -0.045617891133211394,
                             -0.17644767687327373, 0.39498164342745334), c(2L, 4L)))
  g <- grad(function(l, r) jnp_sum(jnp_sin(lax_conv_general_dilated(l, r, integer(0), "VALID"))),
            argnums = c(0L, 1L))(l, r)
  expect_jax_equal(g[[2L]], array_c(c(0.10974481646249284, -0.017537457580023934, -0.12869587400159618,
                                      0.07516030347046629, -0.059754628168618464, -0.13973143024206208,
                                      0.038402983818537434, -0.09743046045588336, -0.14368678871075424,
                                      0.00396629306464764, -0.12596828484083986, -0.14008820259616853),
                                    c(4L, 3L)))
  check_grad_fd(function(x) jnp_sum(jnp_sin(lax_conv_general_dilated(x, r, integer(0), "VALID"))), l)
  r2 <- as_jax(array_c(cos(1:12), c(3L, 4L)), "float64")
  expect_jax_equal(lax_conv_transpose(l, r2, integer(0), "VALID"),
                   array_c(c(0.5840032988415473, 0.4044951016455663, -0.1469040265786132,
                             -0.5632402702490447, -0.4263287726976958, -0.37134004204449056,
                             0.0250570107420753, 0.3984167634087027), c(2L, 4L)))
})
