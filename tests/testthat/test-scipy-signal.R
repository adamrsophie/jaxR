# Tests for jax.scipy.signal (port of jax/_src/scipy/signal.py).
# Expected values come from jax 0.12.0.1 (float32, x64 disabled).

# R array from C order values.
carr <- function(v, shape) {
  if (length(shape) <= 1L) return(array(v, shape))
  aperm(array(v, rev(shape)), rev(seq_along(shape)))
}

cv1_full <- c(0.5, 0, 1.5, 3, 4.5, 3, 10)
cv1s_full <- c(0.5, 0, 1.5, 3, 4.5, 3, 10)
cv1f_full <- c(0.5000002, -5.449568e-07, 1.5000001, 3.0000005, 4.5, 3.0000005, 10)
co1_full <- c(2, 3, 4.5, 6, 7.5, -3, 2.5)
co1s_full <- c(2.5, -3, 7.5, 6, 4.5, 3, 2)
cv1_same <- c(0, 1.5, 3, 4.5, 3)
cv1s_same <- c(1.5, 3, 4.5)
cv1f_same <- c(-5.449568e-07, 1.5000001, 3.0000005, 4.5, 3.0000005)
co1_same <- c(3, 4.5, 6, 7.5, -3)
co1s_same <- c(7.5, 6, 4.5)
cv1_valid <- c(1.5, 3, 4.5)
cv1s_valid <- c(1.5, 3, 4.5)
cv1f_valid <- c(1.5000001, 3.0000005, 4.5)
co1_valid <- c(4.5, 6, 7.5)
co1s_valid <- c(7.5, 6, 4.5)
cv2_full <- carr(c(-1.5, 2.5, 1.5, 0.5, -3, 1.75, -11.25, 1.25, 3.75, 3.5, 2.75, 6.25, -11.25, 1.25, 2.5, 0.75, 5.75, 6.75, -4.75, -1.5), c(4L, 5L))
cv2s_full <- carr(c(-1.5, 2.5, 1.5, 0.5, -3, 1.75, -11.25, 1.25, 3.75, 3.5, 2.75, 6.25, -11.25, 1.25, 2.5, 0.75, 5.75, 6.75, -4.75, -1.5), c(4L, 5L))
co2_full <- carr(c(-4.5, -2.25, 1.25, 4.75, 0.75, 10.5, -3.75, -3.75, -1.25, 1.75, -0.5, 13.75, -3.75, -3.75, 0.25, -3, -3.5, 5.5, -0.5, -0.5), c(4L, 5L))
co2s_full <- carr(c(-0.5, -0.5, 5.5, -3.5, -3, 0.25, -3.75, -3.75, 13.75, -0.5, 1.75, -1.25, -3.75, -3.75, 10.5, 0.75, 4.75, 1.25, -2.25, -4.5), c(4L, 5L))
cvn_full <- carr(c(-1.5, 2.5, 1.5, 0.5, -3, 1.75, -11.25, 1.25, 3.75, 3.5, 2.75, 6.25, -11.25, 1.25, 2.5, 0.75, 5.75, 6.75, -4.75, -1.5), c(4L, 5L))
fc2_full <- carr(c(-1.4999995, 2.5, 1.4999995, 0.49999952, -2.9999998, 1.75, -11.25, 1.2500002, 3.7499993, 3.5, 2.7499993, 6.25, -11.25, 1.25, 2.5, 0.74999994, 5.7499995, 6.75, -4.7499995, -1.5000004), c(4L, 5L))
cv2_same <- carr(c(-1.5, 2.5, 1.5, 0.5, 1.75, -11.25, 1.25, 3.75, 2.75, 6.25, -11.25, 1.25), c(3L, 4L))
cv2s_same <- carr(c(-11.25, 1.25, 6.25, -11.25), c(2L, 2L))
co2_same <- carr(c(-3.75, -3.75, -1.25, 1.75, 13.75, -3.75, -3.75, 0.25, -3.5, 5.5, -0.5, -0.5), c(3L, 4L))
co2s_same <- carr(c(-3.75, 13.75, -3.75, -3.75), c(2L, 2L))
cvn_same <- carr(c(-1.5, 2.5, 1.5, 0.5, 1.75, -11.25, 1.25, 3.75, 2.75, 6.25, -11.25, 1.25), c(3L, 4L))
fc2_same <- carr(c(-1.4999995, 2.5, 1.4999995, 0.49999952, 1.75, -11.25, 1.2500002, 3.7499993, 2.7499993, 6.25, -11.25, 1.25), c(3L, 4L))
cv2_valid <- carr(c(-11.25, 1.25, 3.75, 6.25, -11.25, 1.25), c(2L, 3L))
cv2s_valid <- carr(c(-11.25, 1.25, 3.75, 6.25, -11.25, 1.25), c(2L, 3L))
co2_valid <- carr(c(-3.75, -3.75, -1.25, 13.75, -3.75, -3.75), c(2L, 3L))
co2s_valid <- carr(c(-3.75, -3.75, 13.75, -1.25, -3.75, -3.75), c(2L, 3L))
cvn_valid <- carr(c(-11.25, 1.25, 3.75, 6.25, -11.25, 1.25), c(2L, 3L))
fc2_valid <- carr(c(-11.25, 1.2500002, 3.7499993, 6.25, -11.25, 1.25), c(2L, 3L))
co2c_full <- carr(complex(real = c(-4.5, -2.25, 1.25, 4.75, 0.75, 10.5, -3.75, -3.75, -1.25, 1.75, -0.5, 13.75, -3.75, -3.75, 0.25, -3, -3.5, 5.5, -0.5, -0.5), imaginary = c(3, -0.5, -1.5, -2.5, 1.5, -3.5, 6.75, -0.75, -3.25, -0.25, -5.5, -3.25, 6.75, -0.75, -0.75, -1.5, -3.25, 0.25, 1.25, 0.25)), c(4L, 5L))
co2cs_full <- carr(complex(real = c(-0.5, -0.5, 5.5, -3.5, -3, 0.25, -3.75, -3.75, 13.75, -0.5, 1.75, -1.25, -3.75, -3.75, 10.5, 0.75, 4.75, 1.25, -2.25, -4.5), imaginary = c(-0.25, -1.25, -0.25, 3.25, 1.5, 0.75, 0.75, -6.75, 3.25, 5.5, 0.25, 3.25, 0.75, -6.75, 3.5, -1.5, 2.5, 1.5, 0.5, -3)), c(4L, 5L))
co2c_same <- carr(complex(real = c(-3.75, -3.75, -1.25, 1.75, 13.75, -3.75, -3.75, 0.25, -3.5, 5.5, -0.5, -0.5), imaginary = c(6.75, -0.75, -3.25, -0.25, -3.25, 6.75, -0.75, -0.75, -3.25, 0.25, 1.25, 0.25)), c(3L, 4L))
co2cs_valid <- carr(complex(real = c(-3.75, -3.75, 13.75, -1.25, -3.75, -3.75), imaginary = c(-0.75, 6.75, -3.25, -3.25, -0.75, 6.75)), c(2L, 3L))
co1c <- complex(real = c(2, 3, 4.5, 6, 7.5, -3, 2.5), imaginary = c(-4, -6, -9, -12, -15, 6, -5))
fc1c <- complex(real = c(0.5000002, -5.449568e-07, 1.5000001, 3.0000005, 4.5, 3.0000005, 10), imaginary = c(-0.5000002, 5.449568e-07, -1.5000001, -3.0000005, -4.5, -3.0000005, -10))
fca_full <- carr(c(-4.7683716e-07, 1.0000004, 4.828428, 9.853006, 8.563877, -3.1961522, -1.0000005, 2.0901697, 0.3484688, 6.4404693, 1.5106469, 7.52026, 1.6274167, 5.9999995), c(2L, 7L))
fca_same <- carr(c(1.0000004, 4.828428, 9.853006, 8.563877, -3.1961522, 0.3484688, 6.4404693, 1.5106469, 7.52026, 1.6274167), c(2L, 5L))
fca_valid <- carr(c(4.828428, 9.853006, 8.563877, 6.4404693, 1.5106469, 7.52026), c(2L, 3L))
fca0 <- carr(c(-4.7683716e-07, 2.0901697, 1.0000004, 0.3484688, 4.828428, 6.4404693, 9.853006, 1.5106469, 8.563877, 7.52026, -3.1961522, 1.6274167, -1.0000005, 5.9999995), c(7L, 2L))
cvi <- c(1, 3, 5, 3)
dt_lin <- c(-0.08333397, 0.9047613, -1.1071436, 0.8809514, -1.1309533, 0.857142, -1.1547627, 0.83333206)
dt_const <- c(-3.625, -1.625, -2.625, 0.375, -0.625, 2.375, 1.375, 4.375)
dt_bp <- c(-0.49999988, 1, -0.5, -4.7683716e-07, -1.4305115e-06, 0.6666665, -1.3333335, 0.666667)
dt_ax0 <- carr(c(-1.5000006, -0.5000005, 0.9999995, 2.4999995, -0.50000036, -2.5000005, -0.5000001, 2.4999998, 1.9999995, -0.49999988, -1.5, -0.5000006), c(4L, 3L))
dt_ax1 <- carr(c(0.33333334, -0.666667, 0.33333278, 0.33333373, -0.66666603, 0.33333397, -0.49999988, 0.99999976, -0.5000005, 0.3333333, -0.6666667, 0.33333337), c(4L, 3L))
dt_c0 <- carr(c(-1.5, -0.5, 1.75, 2.5, -0.5, -2.25, -0.5, 2.5, 1.75, -0.5, -1.5, -1.25), c(4L, 3L))
stft_def_0 <- c(0, 0.125, 0.25, 0.375, 0.5)
stft_def_1 <- c(0, 4, 8, 12, 16, 20, 24, 28, 32)
stft_def_2 <- carr(complex(real = c(0.35619307, 0.39725074, 0.03468594, 1.0928944, 0.23273216, 1.5782531, 0.6790848, 1.7996585, 0.7655016, -0.12290104, -0.26267505, 0.12636185, -0.74697244, 0.10431677, -0.99668384, -0.1636868, -1.0370616, -0.1876246, -0.042296425, 0.011472128, -0.18248284, 0.28885314, -0.26751643, 0.160853, -0.073417574, 0.07118663, -0.32905304, -0.027098972, 0.073074274, 0.058373258, -0.12998787, 0.07160467, 0.061377913, -0.1411746, 0.090461224, 0.1876246, 0.02839978, -0.040993422, -0.039190456, 0.08331978, -0.049542174, -0.0293473, 0.077473134, -0.048831046, -0.10739553), imaginary = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.23318885, -0.42070758, 0.43594208, -0.2051184, 0.13574211, 0.12618983, -0.17753778, 0.39278203, -0.6376688, -0.06774229, 0.28327546, -0.27233142, 0.10699664, 0.03673929, -0.14026156, 0.101956554, -0.07959932, 0.26534128, -0.0014040023, -0.11817838, 0.119838804, -0.008038536, -0.12119522, 0.12863871, -0.018083349, -0.119203046, 0.0204373, 0, 0, 0, 0, 0, 0, 0, 0, 0)), c(5L, 9L))
stft_b_0 <- c(0, 1, 2, 3, 4, 5)
stft_b_1 <- c(0.4, 1, 1.6, 2.2, 2.8)
stft_b_2 <- carr(complex(real = c(0.3972507, 0.88476145, 0.23273216, 1.279075, 1.7996585, -0.022567857, -0.7751782, -0.072202265, -0.47880065, -1.2530359, -0.32512826, 0.3165038, -0.04026089, -0.29697156, 0.34682375, 0.17293505, 0.04709652, 0.039752647, 0.15459211, 0.030843452, -0.0033675618, -0.009645343, -0.01888448, 0.005562708, -4.506111e-05, -0.040993437, -0.042315036, -0.049542174, -0.047840357, -0.048831046), imaginary = c(0, 0, 0, 0, 0, -0.4930194, -0.08369143, 0.1165972, -0.83032036, -0.4433363, 0.21638316, 0.16115546, -0.28284118, 0.30367392, 0.26858592, 0.109341, 0.051840954, 0.20744687, 0.14228444, 0.08009937, -0.11853182, -0.13753258, -0.119272344, -0.11592695, -0.13479424, 0, 0, 0, 0, 0)), c(6L, 5L))
stft_lin_0 <- c(0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125)
stft_lin_1 <- c(0, 4, 8, 12, 16, 20, 24, 28, 32)
stft_lin_2 <- carr(complex(real = c(-0.12471563, 0.22295393, -0.2990762, 0.37325317, -0.43211403, 0.43731433, -0.36029595, 0.21246132, -0.09262357, 0.117553346, -0.17552663, 0.293243, -0.38715175, 0.4367399, -0.4262144, 0.35600358, -0.24346289, 0.24143797, -0.042296432, 0.011472132, -0.18248285, 0.28885314, -0.26751643, 0.16085297, -0.07341759, 0.07118658, -0.32905307, -0.027098976, 0.07307425, 0.05837322, -0.12998787, 0.07160464, 0.061377943, -0.1411746, 0.090461224, 0.18762463, 0.028399758, -0.040993437, -0.039190486, 0.0833198, -0.04954216, -0.029347375, 0.07747315, -0.048831142, -0.10739551, -0.027098976, 0.07307425, 0.05837322, -0.12998787, 0.07160464, 0.061377943, -0.1411746, 0.090461224, 0.18762463, -0.042296432, 0.011472132, -0.18248285, 0.28885314, -0.26751643, 0.16085297, -0.07341759, 0.07118658, -0.32905307, 0.117553346, -0.17552663, 0.293243, -0.38715175, 0.4367399, -0.4262144, 0.35600358, -0.24346289, 0.24143797), imaginary = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.053021286, -0.2361266, 0.24343863, -0.21347922, 0.14522876, -0.055189177, -0.03280561, 0.103255406, -0.24482185, -0.0287561, 0.2433343, -0.2306759, 0.10880584, 0.034686513, -0.10101317, 0.07063823, -0.016949028, 0.18033373, 0.0066703185, -0.12645048, 0.12846595, -0.007663816, -0.12162034, 0.13676737, -0.024569545, -0.10622768, 0.002831675, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0066703185, 0.12645048, -0.12846595, 0.007663816, 0.12162034, -0.13676737, 0.024569545, 0.10622768, -0.002831675, 0.0287561, -0.2433343, 0.2306759, -0.10880584, -0.034686513, 0.10101317, -0.07063823, 0.016949028, -0.18033373, -0.053021286, 0.2361266, -0.24343863, 0.21347922, -0.14522876, 0.055189177, 0.03280561, -0.103255406, 0.24482185)), c(8L, 9L))
stft_odd_0 <- c(0, 0.14285715, 0.2857143, 0.42857146)
stft_odd_1 <- c(0, 4, 8, 12, 16, 20, 24, 28, 32)
stft_odd_2 <- carr(complex(real = c(0.45517698, 0.22659172, 0.20368089, 1.0371865, 0.26691386, 1.6705459, 0.5629692, 2.024725, 1.1092924, -0.21220309, -0.06879729, -0.06691751, -0.6485754, 0.060621653, -1.0347904, -0.12126967, -1.1284698, -0.42338648, -0.025630245, -0.14262746, 0.02216406, 0.1654729, -0.28162858, 0.24624845, -0.111190155, 0.00830555, -0.23752195, 0.010244826, 0.09812889, -0.05708698, -0.035490744, 0.08755, -0.046731, -0.049024772, 0.107801765, 0.10626222), imaginary = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.24808411, -0.4160125, 0.50829047, -0.3512972, 0.28714454, -0.023849165, -0.052921575, 0.256112, -0.11822894, -0.03170806, 0.16153231, -0.28855145, 0.2501729, -0.11307264, -0.08227258, 0.14049277, -0.10948505, -0.18195683, -0.017497092, 0.01116441, 0.09610067, -0.10775281, 0.0053281104, 0.10211624, -0.11734268, 0.014451794, 0.050811253)), c(4L, 9L))
stft_even_0 <- c(0, 0.125, 0.25, 0.375, 0.5)
stft_even_1 <- c(0, 4, 8, 12, 16, 20, 24, 28, 32)
stft_even_2 <- carr(complex(real = c(0.6373862, 0.39725074, 0.03468594, 1.0928944, 0.23273216, 1.5782531, 0.6790848, 1.7996585, 2.2210307, -0.17080204, -0.26267505, 0.12636185, -0.74697244, 0.10431677, -0.99668384, -0.1636868, -1.0370616, -1.1543924, -0.15959285, 0.011472128, -0.18248284, 0.28885314, -0.26751643, 0.160853, -0.073417574, 0.07118663, 0.092402995, 0.020802036, 0.073074274, 0.058373258, -0.12998787, 0.07160467, 0.061377913, -0.1411746, 0.090461224, -0.1618199, -0.018200487, -0.040993422, -0.039190456, 0.08331978, -0.049542174, -0.0293473, 0.077473134, -0.048831046, 0.22658777), imaginary = c(0, 0, 0, 0, 0, 0, 0, 0, 0, -2.6077032e-08, -0.42070758, 0.43594208, -0.2051184, 0.13574211, 0.12618983, -0.17753778, 0.39278203, -0.004492402, -2.9802322e-08, 0.28327546, -0.27233142, 0.10699664, 0.03673929, -0.14026156, 0.101956554, -0.07959932, -0.17117223, -2.6077032e-08, -0.11817838, 0.119838804, -0.008038536, -0.12119522, 0.12863871, -0.018083349, -0.119203046, 0.18031353, 0, 0, 0, 0, 0, 0, 0, 0, 0)), c(5L, 9L))
stft_const_0 <- c(0, 0.125, 0.25, 0.375, 0.5)
stft_const_1 <- c(0, 4, 8, 12, 16, 20, 24, 28, 32)
stft_const_2 <- carr(complex(real = c(0.46869308, 0.39725074, 0.03468594, 1.0928944, 0.23273216, 1.5782531, 0.6790848, 1.7996585, 1.793249, -0.16040103, -0.26267505, 0.12636185, -0.74697244, 0.10431677, -0.99668384, -0.1636868, -1.0370616, -0.8042731, -0.07979642, 0.011472128, -0.18248284, 0.28885314, -0.26751643, 0.160853, -0.073417574, 0.07118663, -0.123503536, 0.010401025, 0.073074274, 0.058373258, -0.12998787, 0.07160467, 0.061377913, -0.1411746, 0.090461224, -0.017924845, -0.009100229, -0.040993422, -0.039190456, 0.08331978, -0.049542174, -0.0293473, 0.077473134, -0.048831046, 0.09815395), imaginary = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.14265585, -0.42070758, 0.43594208, -0.2051184, 0.13574211, 0.12618983, -0.17753778, 0.39278203, -0.14142847, -0.014709279, 0.28327546, -0.27233142, 0.10699664, 0.03673929, -0.14026156, 0.101956554, -0.07959932, -0.025349647, -0.016937017, -0.11817838, 0.119838804, -0.008038536, -0.12119522, 0.12863871, -0.018083349, -0.119203046, 0.10557866, 0, 0, 0, 0, 0, 0, 0, 0, 0)), c(5L, 9L))
stft_win_0 <- c(0, 0.16666667, 0.33333334, 0.5)
stft_win_1 <- c(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33)
stft_win_2 <- carr(complex(real = c(0.22325183, 0.8011272, -0.15679811, 0.21281624, 1.249754, 0.28535268, 0.6743561, 1.6981678, 0.7276051, 1.1359293, 2.0815911, 0.39667144, -0.10054524, -0.5017078, 0.2682607, -0.27702916, -0.6571791, 0.111234665, -0.44514114, -0.81240433, -0.04577666, -0.61309516, -0.99977005, 0.27576, 0.02072703, 0.20524538, 0.07437908, 0.20774625, 0.21653797, 0.086748116, 0.2213278, 0.22742675, 0.09875207, 0.23450819, 0.2703031, 0.033937037, -0.06361541, -0.10231993, -0.06501065, -0.17016059, -0.0979662, -0.062287416, -0.166996, -0.09335462, -0.059318896, -0.16355921, -0.023717882, -0.086974435), imaginary = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11692775, -0.2722482, -0.10604086, 0.3930126, -0.37178832, -0.19335516, 0.2965334, -0.47130838, -0.28070638, 0.19987214, -0.626898, -0.20942481, -0.021321455, 0.10919559, -0.028190453, -0.048051126, 0.13976265, 0.00036440353, -0.016326388, 0.17029454, 0.028920095, 0.015400526, 0.14467444, -0.20942481, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)), c(4L, 12L))
stft_2d_0 <- c(0, 0.125, 0.25, 0.375, 0.5)
stft_2d_1 <- c(0, 4, 8, 12, 16, 20, 24, 28, 32)
stft_2d_2 <- carr(complex(real = c(0.35619307, 0.39725074, 0.03468594, 1.0928944, 0.23273216, 1.5782531, 0.6790848, 1.7996585, 0.7655016, -0.12290104, -0.26267505, 0.12636185, -0.74697244, 0.10431677, -0.99668384, -0.1636868, -1.0370616, -0.1876246, -0.042296425, 0.011472128, -0.18248284, 0.28885314, -0.26751643, 0.160853, -0.073417574, 0.07118663, -0.32905304, -0.027098972, 0.073074274, 0.058373258, -0.12998787, 0.07160467, 0.061377913, -0.1411746, 0.090461224, 0.1876246, 0.02839978, -0.040993422, -0.039190456, 0.08331978, -0.049542174, -0.0293473, 0.077473134, -0.048831046, -0.10739553, 0.49792463, -0.0075374357, -0.82633394, 0.071745574, 0.8200678, -0.1375333, -0.8267294, 0.18994185, 0.29066038, -0.34695756, -0.05020904, 0.4473452, -0.03661652, -0.4373057, 0.1302594, 0.48561388, -0.17432901, -0.11856378, 0.1757992, 0.079809844, 0.0011650473, -0.0033698585, -0.0109268725, -0.08183161, -0.05477017, 0.10483143, -0.08138685, -0.15304244, -0.023536664, -0.030980453, 0.0036423933, 0.03359276, 0.018746473, -0.014695212, -0.023505427, 0.11856378, 0.15047699, -0.0045908503, -0.008725584, 0.00094239414, 0.009211779, 0.0031848028, -0.0055675805, -0.0039358065, -0.1278867), imaginary = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.23318885, -0.42070758, 0.43594208, -0.2051184, 0.13574211, 0.12618983, -0.17753778, 0.39278203, -0.6376688, -0.06774229, 0.28327546, -0.27233142, 0.10699664, 0.03673929, -0.14026156, 0.101956554, -0.07959932, 0.26534128, -0.0014040023, -0.11817838, 0.119838804, -0.008038536, -0.12119522, 0.12863871, -0.018083349, -0.119203046, 0.0204373, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.19704208, -0.38095337, 0.06323433, 0.4201523, -0.01175794, -0.3911308, -0.014862604, 0.31731617, -0.22936559, -0.1371187, 0.14115797, -0.05750595, -0.18935147, -0.022796303, 0.15662697, 0.07309829, -0.071134545, 0.16767451, 0.04864049, 0.001723811, 0.013377994, 0.011599958, 0.01094155, -0.0029209852, -0.027036734, -0.02005963, -0.06659189, 0, 0, 0, 0, 0, 0, 0, 0, 0)), c(2L, 5L, 9L))
stft_ax0_0 <- c(0, 0.125, 0.25, 0.375, 0.5)
stft_ax0_1 <- c(0, 4, 8, 12, 16, 20, 24, 28, 32)
stft_ax0_2 <- carr(complex(real = c(0.35619307, 0.39725074, 0.03468594, 1.0928944, 0.23273216, 1.5782531, 0.6790848, 1.7996585, 0.7655016, 0.49792463, -0.0075374357, -0.82633394, 0.071745574, 0.8200678, -0.1375333, -0.8267294, 0.18994185, 0.29066038, -0.12290104, -0.26267505, 0.12636185, -0.74697244, 0.10431677, -0.99668384, -0.1636868, -1.0370616, -0.1876246, -0.34695756, -0.05020904, 0.4473452, -0.03661652, -0.4373057, 0.1302594, 0.48561388, -0.17432901, -0.11856378, -0.042296425, 0.011472128, -0.18248284, 0.28885314, -0.26751643, 0.160853, -0.073417574, 0.07118663, -0.32905304, 0.1757992, 0.079809844, 0.0011650473, -0.0033698585, -0.0109268725, -0.08183161, -0.05477017, 0.10483143, -0.08138685, -0.027098972, 0.073074274, 0.058373258, -0.12998787, 0.07160467, 0.061377913, -0.1411746, 0.090461224, 0.1876246, -0.15304244, -0.023536664, -0.030980453, 0.0036423933, 0.03359276, 0.018746473, -0.014695212, -0.023505427, 0.11856378, 0.02839978, -0.040993422, -0.039190456, 0.08331978, -0.049542174, -0.0293473, 0.077473134, -0.048831046, -0.10739553, 0.15047699, -0.0045908503, -0.008725584, 0.00094239414, 0.009211779, 0.0031848028, -0.0055675805, -0.0039358065, -0.1278867), imaginary = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.23318885, -0.42070758, 0.43594208, -0.2051184, 0.13574211, 0.12618983, -0.17753778, 0.39278203, -0.6376688, 0.19704208, -0.38095337, 0.06323433, 0.4201523, -0.01175794, -0.3911308, -0.014862604, 0.31731617, -0.22936559, -0.06774229, 0.28327546, -0.27233142, 0.10699664, 0.03673929, -0.14026156, 0.101956554, -0.07959932, 0.26534128, -0.1371187, 0.14115797, -0.05750595, -0.18935147, -0.022796303, 0.15662697, 0.07309829, -0.071134545, 0.16767451, -0.0014040023, -0.11817838, 0.119838804, -0.008038536, -0.12119522, 0.12863871, -0.018083349, -0.119203046, 0.0204373, 0.04864049, 0.001723811, 0.013377994, 0.011599958, 0.01094155, -0.0029209852, -0.027036734, -0.02005963, -0.06659189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)), c(5L, 2L, 9L))
istft_def_0 <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
istft_def_1 <- c(0.29999998, 0.5427637, 0.9383714, 1.3131669, 0.37920153, -0.2434442, -0.27174535, -0.7925278, -0.36947015, 0.7664321, 0.9926677, 1.404461, 1.7539204, 0.8005824, 0.2043482, 0.16924459, -0.35184285, 0.10726118, 1.2320975, 1.442804, 1.8706118, 2.1936707, 1.2223018, 0.65257293, 0.60972273, 0.089558244, 0.5844841, 1.6969938, 1.8932006, 2.336787, 2.632424, 1.6443958)
istft_7_0 <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
istft_7_1 <- c(0.30000007, 0.5427639, 0.9383715, 1.313167, 0.37920156, -0.24344435, -0.27174532, -0.7925281, -0.36947012, 0.7664323, 0.9926681, 1.4044613, 1.7539203, 0.8005824, 0.20434825, 0.16924457, -0.35184288, 0.10726119, 1.2320977, 1.4428043, 1.8706119, 2.1936717, 1.2223023, 0.65257305, 0.6097229, 0.089558326, 0.5844842, 1.6969938, 1.8932006, 2.3367872, 2.6324246, 1.644396, 3.4823107e-09)
istft_b_0 <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
istft_b_1 <- c(0.29999995, 0.54276377, 0.9383714, 1.3131669, 0.37920153, -0.24344425, -0.27174535, -0.7925278, -0.36947015, 0.76643217, 0.99266773, 1.404461, 1.7539204, 0.80058247, 0.20434822, 0.1692446, -0.3518428, 0.10726115, 1.2320974, 1.442804, 1.8706117, 2.193671, 1.2223018, 0.652573, 0.6097227, 0.08955827, 0.58448416, 1.696994, 1.8932005, 2.3367875, 2.632424, 1.6443958)
istft_two_0 <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
istft_two_1 <- complex(real = c(0.29999998, 0.5427637, 0.9383714, 1.3131666, 0.37920156, -0.24344425, -0.27174535, -0.79252774, -0.36947015, 0.7664322, 0.9926677, 1.404461, 1.7539204, 0.8005824, 0.2043482, 0.16924459, -0.35184285, 0.10726118, 1.2320975, 1.4428039, 1.8706118, 2.1936707, 1.2223018, 0.65257293, 0.60972273, 0.089558244, 0.5844841, 1.6969938, 1.8932006, 2.336787, 2.632424, 1.6443958), imaginary = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
istft_2d_0 <- c(0, 1)
istft_2d_1 <- carr(c(0.29999998, 0.5427637, 0.9383714, 1.3131669, 0.37920153, -0.2434442, -0.27174535, -0.79252785, -0.36947015, 0.76643217, 0.9926677, 1.404461, 1.7539204, 0.8005824, 0.2043482, 0.16924459, -0.35184285, 0.10726118, 1.2320975, 1.442804, 1.8706118, 2.1936707, 1.2223018, 0.65257293, 0.60972273, 0.089558244, 0.5844841, 1.6969938, 1.8932006, 2.3367872, 2.632424, 1.6443958, 0.99999994, 0.7283493, 0.5936063, 0.49991092, 0.14749138, -0.4591708, -0.9371023, -1.006042, -0.8327294, -0.7443616, -0.7376769, -0.5046872, 0.06594824, 0.6543414, 0.89653236, 0.8390622, 0.8074258, 0.89112467, 0.8057344, 0.335132, -0.2980117, -0.68475413, -0.7471048, -0.7751446, -0.94183725, -1.0158455, -0.69840837, -0.091077395, 0.39566883, 0.5681119, 0.65109473, 0.88324416), c(2L, 32L))
istft_ax_0 <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
istft_ax_1 <- carr(c(0.29999998, 0.99999994, 0.5427637, 0.7283493, 0.9383714, 0.5936063, 1.3131669, 0.49991092, 0.37920153, 0.14749138, -0.2434442, -0.4591708, -0.27174535, -0.9371023, -0.79252785, -1.006042, -0.36947015, -0.8327294, 0.76643217, -0.7443616, 0.9926677, -0.7376769, 1.404461, -0.5046872, 1.7539204, 0.06594824, 0.8005824, 0.6543414, 0.2043482, 0.89653236, 0.16924459, 0.8390622, -0.35184285, 0.8074258, 0.10726118, 0.89112467, 1.2320975, 0.8057344, 1.442804, 0.335132, 1.8706118, -0.2980117, 2.1936707, -0.68475413, 1.2223018, -0.7471048, 0.65257293, -0.7751446, 0.60972273, -0.94183725, 0.089558244, -1.0158455, 0.5844841, -0.69840837, 1.6969938, -0.091077395, 1.8932006, 0.39566883, 2.3367872, 0.5681119, 2.632424, 0.65109473, 1.6443958, 0.88324416), c(32L, 2L))
istft_win_0 <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
istft_win_1 <- c(0.30000004, 0.54276395, 0.93837154, 1.3131671, 0.37920156, -0.24344438, -0.2717454, -0.7925281, -0.36947027, 0.76643234, 0.99266785, 1.4044614, 1.7539209, 0.8005828, 0.20434828, 0.16924466, -0.35184294, 0.10726116, 1.2320977, 1.4428043, 1.8706119, 2.1936712, 1.2223022, 0.65257317, 0.6097228, 0.089558326, 0.58448446, 1.6969942, 1.8932011, 2.3367875, 2.632425, 1.6443961, 9.743069e-09)
welch_def_0 <- c(0, 0.125, 0.25, 0.375, 0.5)
welch_def_1 <- c(0.7576688, 2.3317108, 0.6527637, 0.20848271, 0.016655853)
welch_spec_0 <- c(0, 0.5, 1, 1.5, 2)
welch_spec_1 <- c(0.118770346, 0.29187644, 0.104488485, 0.04016825, 0.003122973)
welch_two_0 <- c(0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125)
welch_two_1 <- c(0.7576688, 1.1658556, 0.32638186, 0.10424136, 0.016655853, 0.10424136, 0.32638186, 1.1658556)
welch_med_0 <- c(0, 0.125, 0.25, 0.375, 0.5)
welch_med_1 <- c(1.0013131, 2.9680095, 1.0240076, 0.2782867, 0.016743654)
welch_nfft_0 <- c(0, 0.11111111, 0.22222222, 0.33333334, 0.44444445)
welch_nfft_1 <- c(0.7576688, 2.3424928, 1.0212446, 0.26011702, 0.08166905)
welch_2d_0 <- c(0, 0.125, 0.25, 0.375, 0.5)
welch_2d_1 <- carr(c(0.7576688, 2.3317108, 0.6527637, 0.20848271, 0.016655853, 0.14612058, 1.0595891, 0.18549211, 0.008157735, 0.0001825437), c(2L, 5L))
welch_ax0_0 <- c(0, 0.125, 0.25, 0.375, 0.5)
welch_ax0_1 <- carr(c(0.7576688, 0.14612058, 2.3317108, 1.0595891, 0.6527637, 0.18549211, 0.20848271, 0.008157735, 0.016655853, 0.0001825437), c(5L, 2L))
welch_full_0 <- c(0, 0.03125, 0.0625, 0.09375, 0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.5)
welch_full_1 <- c(0.117779374, 1.5450587, 0.33297962, 6.9180045, 8.375485, 0.44399923, 0.0070173997, 0.0008659585, 0.0005740679, 0.0071479618, 0.5003724, 0.85438716, 0.076672375, 0.0006046788, 4.6927533e-05, 6.236295e-06, 7.7180323e-07)
csd_def_0 <- c(0, 0.125, 0.25, 0.375, 0.5)
csd_def_1 <- complex(real = c(0.04762998, 0.36397174, 0.040639542, 0.0030277509, -0.00013737891), imaginary = c(0, -0.006111903, -0.06216804, 0.0023741191, 0))
csd_med_0 <- c(0, 0.125, 0.25, 0.375, 0.5)
csd_med_1 <- complex(real = c(0.26770356, 0.8314066, 0.16113879, 0.0037193445, 0.0005513724), imaginary = c(0, 0.13021204, 0.046073504, -0.020764934, 0))
csd_len_0 <- c(0, 0.125, 0.25, 0.375, 0.5)
csd_len_1 <- complex(real = c(-0.05829876, 0.025816621, 0.023691354, 0.005417125, 0.00019255796), imaginary = c(0, -0.15534034, -0.10039393, 0.0050880015, 0))
csd_one_0 <- c(0, 0.125, 0.25, 0.375, 0.5)
csd_one_1 <- complex(real = c(-0.052917372, 1.7296027, 0.43629003, -0.020518823, 0.0010037053), imaginary = c(0, 0.45106244, -0.22388044, -0.028325967, 0))
welch_c_0 <- c(0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125)
welch_c_1 <- c(0.947086, 1.4573196, 0.40797728, 0.1303017, 0.020819813, 0.1303017, 0.4079773, 1.4573195)

x1 <- c(1, 2, 3, 4, 5)
h1 <- c(0.5, -1, 2)
A <- carr(((0:11) %% 5) - 1.5, c(3L, 4L))
B <- matrix(c(1, -2, 0.5, 3), 2L, byrow = TRUE)
Bc <- B + 1i * matrix(c(0.5, 1, -1, 2), 2L, byrow = TRUE)
P <- carr(((0:9)^1.5) %% 7, c(2L, 5L))
Q <- matrix(c(1, 2, -1, 0.5, 0, 1), 2L, byrow = TRUE)
d8 <- c(1, 3, 2, 5, 4, 7, 6, 9)
D <- carr(((0:11)^2) %% 5, c(4L, 3L))
tt <- 0:31
sig <- sin(0.7 * tt) + 0.3 * cos(2.1 * tt) + 0.05 * tt
sig2 <- cos(0.4 * tt) - 0.2 * sin(1.3 * tt)
X2 <- unname(rbind(sig, sig2))
win6 <- c(0.2, 0.5, 1, 1, 0.5, 0.2)
modes <- c("full", "same", "valid")
# Compare a list result (a JAX tuple) with the expected elements name_0, name_1, ...
expect_tuple <- function(res, name, tolerance = NULL) {
  for (i in seq_along(res)) {
    expect_jax_equal(res[[i]], get(paste0(name, "_", i - 1L), envir = parent.frame()),
                     tolerance = tolerance)
  }
}
with_x64 <- function(code) with_jax_config("jax_enable_x64", TRUE, code)
f64 <- function(x) as_jax(x, "float64")

test_that("one dimensional convolve and correlate match jax", {
  skip_if_no_backend()
  for (m in modes) {
    expect_jax_equal(scipy_signal_convolve(x1, h1, mode = m), get(paste0("cv1_", m)),
                     dtype = "float32")
    expect_jax_equal(scipy_signal_convolve(h1, x1, mode = m), get(paste0("cv1s_", m)))
    expect_jax_equal(scipy_signal_convolve(x1, h1, mode = m, method = "fft"),
                     get(paste0("cv1f_", m)))
    expect_jax_equal(scipy_signal_correlate(x1, h1, mode = m), get(paste0("co1_", m)))
    expect_jax_equal(scipy_signal_correlate(h1, x1, mode = m), get(paste0("co1s_", m)))
  }
  expect_jax_equal(scipy_signal_correlate(x1, h1 * (1 + 2i)), co1c, dtype = "complex64")
  expect_jax_equal(scipy_signal_fftconvolve(x1 * (1 - 1i), h1), fc1c, dtype = "complex64")
  # Integer inputs are promoted to an inexact dtype.
  expect_jax_equal(scipy_signal_convolve(as_jax(1:3), as_jax(c(1L, 1L))), cvi, dtype = "float32")
})

test_that("two dimensional convolve and correlate match jax", {
  skip_if_no_backend()
  for (m in modes) {
    expect_jax_equal(scipy_signal_convolve2d(A, B, mode = m), get(paste0("cv2_", m)))
    expect_jax_equal(scipy_signal_convolve2d(B, A, mode = m), get(paste0("cv2s_", m)))
    expect_jax_equal(scipy_signal_correlate2d(A, B, mode = m), get(paste0("co2_", m)))
    expect_jax_equal(scipy_signal_correlate2d(B, A, mode = m), get(paste0("co2s_", m)))
    expect_jax_equal(scipy_signal_convolve(A, B, mode = m), get(paste0("cvn_", m)))
    expect_jax_equal(scipy_signal_fftconvolve(A, B, mode = m), get(paste0("fc2_", m)))
  }
  expect_jax_equal(scipy_signal_correlate2d(A, Bc), co2c_full, dtype = "complex64")
  expect_jax_equal(scipy_signal_correlate2d(Bc, A), co2cs_full)
  expect_jax_equal(scipy_signal_correlate2d(A, Bc, mode = "same"), co2c_same)
  expect_jax_equal(scipy_signal_correlate2d(Bc, A, mode = "valid"), co2cs_valid)
})

test_that("fftconvolve maps over the axes it does not convolve", {
  skip_if_no_backend()
  for (m in modes) {
    expect_jax_equal(scipy_signal_fftconvolve(P, Q, mode = m, axes = 1L), get(paste0("fca_", m)))
  }
  expect_jax_equal(scipy_signal_fftconvolve(t(P), t(Q), axes = 0L), fca0)
  expect_jax_equal(scipy_signal_fftconvolve(P, Q, axes = -1L), fca_full)
})

test_that("detrend matches jax", {
  skip_if_no_backend()
  expect_jax_equal(scipy_signal_detrend(d8), dt_lin, tolerance = 1e-5)
  expect_jax_equal(scipy_signal_detrend(d8, type = "constant"), dt_const)
  expect_jax_equal(scipy_signal_detrend(d8, bp = c(3L, 5L)), dt_bp, tolerance = 1e-5)
  expect_jax_equal(scipy_signal_detrend(D, axis = 0L), dt_ax0, tolerance = 1e-5)
  expect_jax_equal(scipy_signal_detrend(D, axis = 1L), dt_ax1, tolerance = 1e-5)
  expect_jax_equal(scipy_signal_detrend(D, axis = 0L, type = "constant"), dt_c0)
})

test_that("stft matches jax", {
  skip_if_no_backend()
  expect_tuple(scipy_signal_stft(sig, nperseg = 8L), "stft_def")
  expect_identical(dtype(scipy_signal_stft(sig, nperseg = 8L)[[3L]]), "complex64")
  expect_tuple(scipy_signal_stft(sig, fs = 10, nperseg = 8L, noverlap = 2L, nfft = 10L,
                                 boundary = NULL, padded = FALSE), "stft_b")
  expect_tuple(scipy_signal_stft(sig, nperseg = 8L, detrend = "linear", return_onesided = FALSE),
               "stft_lin", tolerance = 1e-4)
  expect_tuple(scipy_signal_stft(sig, nperseg = 7L, boundary = "odd"), "stft_odd")
  expect_tuple(scipy_signal_stft(sig, window = win6, nperseg = 6L), "stft_win")
  expect_tuple(scipy_signal_stft(X2, nperseg = 8L), "stft_2d")
  expect_tuple(scipy_signal_stft(t(X2), nperseg = 8L, axis = 0L), "stft_ax0")
})

test_that("stft boundary extensions that use jnp_pad match jax", {
  skip_if_no_backend()
  skip_if_not(exists("jnp_pad"), "jnp_pad is not available in this build")
  expect_tuple(scipy_signal_stft(sig, nperseg = 8L, boundary = "even"), "stft_even")
  expect_tuple(scipy_signal_stft(sig, nperseg = 8L, boundary = "constant"), "stft_const")
})

test_that("istft matches jax", {
  skip_if_no_backend()
  Z <- scipy_signal_stft(sig, nperseg = 8L)[[3L]]
  expect_tuple(scipy_signal_istft(Z, nperseg = 8L), "istft_def")
  Z7 <- scipy_signal_stft(sig, nperseg = 7L)[[3L]]
  expect_tuple(scipy_signal_istft(Z7, nperseg = 7L, nfft = 7L), "istft_7")
  Zb <- scipy_signal_stft(sig, nperseg = 8L, noverlap = 6L, nfft = 12L)[[3L]]
  expect_tuple(scipy_signal_istft(Zb, nperseg = 8L, noverlap = 6L, nfft = 12L), "istft_b")
  Z2 <- scipy_signal_stft(X2, nperseg = 8L)[[3L]]
  # As upstream, the time vector has the length of the first axis.
  expect_tuple(scipy_signal_istft(Z2, nperseg = 8L), "istft_2d")
  expect_tuple(scipy_signal_istft(jnp_transpose(Z2, c(2L, 0L, 1L)), nperseg = 8L, time_axis = 0L,
                                  freq_axis = -1L), "istft_ax")
  Zw <- scipy_signal_stft(sig, window = win6, nperseg = 6L)[[3L]]
  expect_tuple(scipy_signal_istft(Zw, window = win6, nperseg = 6L), "istft_win")
  # Round trip.
  expect_jax_equal(scipy_signal_istft(Z, nperseg = 8L)[[2L]], sig, tolerance = 1e-5)
})

test_that("two sided istft matches jax", {
  skip_if_no_backend()
  # The normalization compares a complex array with `>`, as jnp.greater allows.
  ok <- tryCatch({
    jnp_greater(as_jax(1i), 0)
    TRUE
  }, error = function(e) FALSE)
  skip_if_not(ok, "jnp_greater does not compare complex arrays in this build")
  Zt <- scipy_signal_stft(sig, nperseg = 8L, return_onesided = FALSE)[[3L]]
  expect_tuple(scipy_signal_istft(Zt, nperseg = 8L, input_onesided = FALSE), "istft_two")
})

test_that("welch and csd match jax", {
  skip_if_no_backend()
  expect_tuple(scipy_signal_welch(sig, nperseg = 8L), "welch_def")
  expect_tuple(scipy_signal_welch(sig, fs = 4, nperseg = 8L, noverlap = 4L, scaling = "spectrum",
                                  detrend = "linear"), "welch_spec", tolerance = 1e-4)
  expect_tuple(scipy_signal_welch(sig, nperseg = 8L, return_onesided = FALSE), "welch_two")
  expect_tuple(scipy_signal_welch(sig, nperseg = 8L, nfft = 9L), "welch_nfft")
  expect_tuple(scipy_signal_welch(X2, nperseg = 8L), "welch_2d")
  expect_tuple(scipy_signal_welch(t(X2), nperseg = 8L, axis = 0L), "welch_ax0")
  expect_tuple(scipy_signal_welch(sig), "welch_full")
  expect_tuple(scipy_signal_csd(sig, sig2, nperseg = 8L), "csd_def")
  expect_identical(dtype(scipy_signal_csd(sig, sig2, nperseg = 8L)[[2L]]), "complex64")
  # y shorter than x is zero padded.
  expect_tuple(scipy_signal_csd(sig, sig2[1:20], nperseg = 8L), "csd_len")
  # One segment: no averaging.
  expect_tuple(scipy_signal_csd(sig[1:8], sig2[1:8], nperseg = 8L), "csd_one")
  expect_warning(res <- scipy_signal_welch(sig * (1 + 0.5i), nperseg = 8L),
                 "Input data is complex, switching to return_onesided=False")
  expect_tuple(res, "welch_c")
})

test_that("median averaging matches jax", {
  skip_if_no_backend()
  skip_if_not(exists("jnp_median"), "jnp_median is not available in this build")
  expect_tuple(scipy_signal_welch(sig, nperseg = 8L, average = "median"), "welch_med")
  expect_tuple(scipy_signal_csd(sig, sig2, nperseg = 8L, average = "median"), "csd_med")
})

test_that("signal functions work under jit", {
  skip_if_no_backend()
  expect_jax_equal(jit(function(a, b) scipy_signal_convolve(a, b, mode = "same"))(x1, h1), cv1_same)
  expect_jax_equal(jit(function(a, b) scipy_signal_correlate2d(a, b))(B, A), co2s_full)
  expect_jax_equal(jit(function(a, b) scipy_signal_fftconvolve(a, b, axes = 1L))(P, Q), fca_full)
  expect_jax_equal(jit(function(a) scipy_signal_detrend(a, axis = 0L))(D), dt_ax0, tolerance = 1e-5)
  expect_tuple(jit(function(a) scipy_signal_stft(a, nperseg = 8L))(sig), "stft_def")
  Z <- scipy_signal_stft(sig, nperseg = 8L)[[3L]]
  expect_tuple(jit(function(z) scipy_signal_istft(z, nperseg = 8L))(Z), "istft_def")
  expect_tuple(jit(function(a) scipy_signal_welch(a, nperseg = 8L))(sig), "welch_def")
  expect_tuple(jit(function(a, b) scipy_signal_csd(a, b, nperseg = 8L))(sig, sig2), "csd_def")
})

test_that("signal functions vectorize under vmap", {
  skip_if_no_backend()
  X <- rbind(x1, rev(x1), x1^2)
  got <- vmap(function(v) scipy_signal_convolve(v, h1, mode = "same"))(unname(X))
  want <- t(sapply(1:3, function(i) as.array(scipy_signal_convolve(X[i, ], h1, mode = "same"))))
  expect_jax_equal(got, want)
  got <- vmap(function(v) scipy_signal_welch(v, nperseg = 8L)[[2L]])(X2)
  expect_jax_equal(got, welch_2d_1)
  Z2 <- scipy_signal_stft(X2, nperseg = 8L)[[3L]]
  got <- vmap(function(z) scipy_signal_istft(z, nperseg = 8L)[[2L]])(Z2)
  expect_jax_equal(got, istft_2d_1)
})

test_that("signal gradients match finite differences", {
  skip_if_no_backend()
  with_x64({
    h <- f64(h1)
    x <- f64(c(0.3, -1.2, 2.5, 0.7, -0.4, 1.1))
    check_grad_fd(function(v) jnp_sum(scipy_signal_convolve(v, h, mode = "same")^2L), x)
    check_grad_fd(function(v) jnp_sum(scipy_signal_convolve(x, v)^2L), h)
    check_grad_fd(function(v) jnp_sum(scipy_signal_correlate(v, h, method = "fft")^2L), x)
    check_grad_fd(function(v) jnp_sum(scipy_signal_fftconvolve(v, h, mode = "valid")^3L), x)
    M <- f64(A)
    check_grad_fd(function(m) jnp_sum(jnp_sin(scipy_signal_correlate2d(m, B))), M)
    check_grad_fd(function(v) jnp_sum(scipy_signal_detrend(v)^2L), f64(d8))
    s <- f64(sig[1:16])
    check_grad_fd(function(v) {
      z <- scipy_signal_stft(v, nperseg = 8L)[[3L]]
      jnp_sum(jnp_real(z * jnp_conjugate(z)))
    }, s)
    check_grad_fd(function(v) jnp_sum(jnp_sqrt(scipy_signal_welch(v, nperseg = 8L)[[2L]])), s)
    check_grad_fd(function(v) jnp_sum(jnp_real(scipy_signal_csd(v, sig2[1:16], nperseg = 8L)[[2L]])), s)
    Z <- scipy_signal_stft(s, nperseg = 8L)[[3L]]
    zr <- f64(as.array(jnp_real(Z)))
    check_grad_fd(function(v) jnp_sum(scipy_signal_istft(v + 0.5i * zr, nperseg = 8L)[[2L]]^2L), zr)
    # Second order.
    g <- grad(function(v) jnp_sum(scipy_signal_convolve(v, h)^3L))
    check_grad_fd(function(v) jnp_sum(g(v)), x)
  })
})

test_that("signal functions raise jax errors", {
  skip_if_no_backend()
  expect_error(scipy_signal_convolve(x1, h1, mode = "wrong"), "mode must be one of")
  expect_error(scipy_signal_convolve(x1, h1, method = "wrong"), "Got method='wrong'")
  expect_error(scipy_signal_convolve(x1, A), "same number of dimensions")
  expect_error(scipy_signal_convolve(x1, numeric(0)), "zero-size arrays not supported")
  expect_error(scipy_signal_convolve(matrix(1, 2, 3), matrix(1, 3, 2)),
               "One input must be smaller than the other in every dimension.")
  expect_error(scipy_signal_convolve2d(x1, h1), "only supports 2-dimensional inputs")
  expect_error(scipy_signal_convolve2d(A, B, boundary = "wrap"), "only supports boundary='fill'",
               class = "jax_not_implemented_error")
  expect_error(scipy_signal_correlate2d(A, B, fillvalue = 1), "only supports boundary='fill'")
  expect_error(scipy_signal_fftconvolve(x1, A), "should have the same dimensionality")
  expect_error(scipy_signal_fftconvolve(x1, h1, mode = "wrong"), "mode must be one of")
  expect_error(scipy_signal_fftconvolve(matrix(1, 2, 3), matrix(1, 3, 2), mode = "valid"),
               "One input must be at least as large")
  expect_error(scipy_signal_fftconvolve(P, t(Q), axes = 1L), "mapped axes must have same shape")
  expect_error(scipy_signal_detrend(d8, type = "quadratic"), "Trend type must be")
  expect_error(scipy_signal_detrend(d8, bp = 9L), "Breakpoints must be non-negative")
  expect_error(scipy_signal_detrend(d8, overwrite_data = TRUE), "not implemented")
  expect_error(scipy_signal_stft(sig, window = "hamming", nperseg = 8L), "get_window",
               class = "jax_not_implemented_error")
  expect_error(scipy_signal_stft(sig, nperseg = 0L), "nperseg must be a positive integer")
  expect_error(scipy_signal_stft(sig, nperseg = 8L, noverlap = 8L), "noverlap must be less")
  expect_error(scipy_signal_stft(sig, nperseg = 8L, nfft = 4L), "nfft must be greater")
  expect_error(scipy_signal_stft(sig, nperseg = 8L, boundary = "periodic"),
               "Unknown boundary option 'periodic'")
  expect_error(scipy_signal_stft(sig, window = c(1, 2), nperseg = 3L),
               "different from length of window")
  expect_error(scipy_signal_stft(sig, window = matrix(1, 2, 2)), "window must be 1-D")
  expect_error(scipy_signal_welch(sig, nperseg = 8L, scaling = "power"), "Unknown scaling")
  expect_error(scipy_signal_welch(sig, nperseg = 8L, average = "mode"), "average must be")
  expect_error(scipy_signal_welch(sig, nperseg = 8L, detrend = TRUE), "Unsupported detrend type")
  expect_error(scipy_signal_csd(sig, X2, nperseg = 8L), "same rank")
  expect_warning(scipy_signal_welch(sig[1:6], nperseg = 8L), "nperseg=8 is greater than input_length=6")
  expect_error(scipy_signal_istft(sig), "at least 2d")
  Z <- scipy_signal_stft(sig, nperseg = 8L)[[3L]]
  expect_error(scipy_signal_istft(Z, time_axis = 0L, freq_axis = -2L), "differing time and frequency")
  expect_error(scipy_signal_istft(Z, nperseg = 8L, window = c(1, 1)), "window must have length of 8")
  expect_error(scipy_signal_istft(Z, nperseg = 8L, noverlap = 8L), "noverlap must be less")
  expect_error(scipy_signal_istft(Z, nperseg = 8L, nfft = 6L), "must be longer than nperseg")
})
