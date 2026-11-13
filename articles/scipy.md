# SciPy functions

``` r

library(jaxR)
```

jaxR ports `jax.scipy`. The functions are named after their Python path:
`jax.scipy.special.gammaln` is `scipy_special_gammaln`,
`jax.scipy.stats.norm.logpdf` is `scipy_stats_norm_logpdf`, and so on.
All of them are built from jaxR primitives, so they work under `grad`,
`vmap` and `jit`, which is the main reason to prefer them over the base
R equivalents inside a differentiable program.

## Special functions

``` r

x <- as_jax(c(0.5, 1, 2.5, 6))
scipy_special_gammaln(x)
#> Array f32[4]
#> [1] 0.5723649 0.0000000 0.2846829 4.7874918
scipy_special_digamma(x)
#> Array f32[4]
#> [1] -1.9635100 -0.5772154  0.7031568  1.7061176
scipy_special_erf(x)
#> Array f32[4]
#> [1] 0.5204999 0.8427008 0.9995930 1.0000000
scipy_special_gammainc(2, x)          # regularized lower incomplete gamma
#> Array f32[4]
#> [1] 0.09020401 0.26424113 0.71270251 0.98264873
scipy_special_betainc(2, 3, c(0.1, 0.5, 0.9))
#> Array f32[3]
#> [1] 0.05230001 0.68750000 0.99629998
scipy_special_i0e(x)                  # exponentially scaled Bessel function
#> Array f32[4]
#> [1] 0.6450353 0.4657596 0.2700465 0.1666574
scipy_special_zeta(x + 1, 1)
#> Array f32[4]
#> [1] 2.612375 1.644934 1.126734 1.008349
```

Logistic and log domain helpers:

``` r

scipy_special_expit(c(-2, 0, 2))
#> Array f32[3]
#> [1] 0.1192029 0.5000000 0.8807970
scipy_special_logit(c(0.1, 0.5, 0.9))
#> Array f32[3]
#> [1] -2.197225  0.000000  2.197224
scipy_special_logsumexp(c(1000, 1000.5, 999))     # no overflow
#> Array f32[]
#> [1] 1001.104
scipy_special_xlogy(c(0, 2), c(0, 3))             # 0 * log(0) is 0
#> Array f32[2]
#> [1] 0.000000 2.197225
scipy_special_softmax(c(1, 2, 3))
#> Array f32[3]
#> [1] 0.09003057 0.24472848 0.66524094
scipy_special_ndtr(c(-1.96, 0, 1.96))
#> Array f32[3]
#> [1] 0.02499789 0.50000000 0.97500211
scipy_special_ndtri(c(0.025, 0.5, 0.975))
#> Array f32[3]
#> [1] -1.959964  0.000000  1.959964
```

They differentiate like any other function. The derivative of `gammaln`
is `digamma`:

``` r

grad(scipy_special_gammaln)(3)
#> Array f32[] weak_type
#> [1] 0.9227843
scipy_special_digamma(3)
#> Array f32[] weak_type
#> [1] 0.9227843
vmap(grad(function(a) scipy_special_gammainc(a, 2)))(c(1, 2, 3))
#> Array f32[3]
#> [1] -0.2208254 -0.2940046 -0.2318487
```

The complete list: `bernoulli`, `bessel_jn`, `beta`, `betainc`,
`betaln`, `boxcox`, `boxcox1p`, `comb`, `dawsn`, `digamma`, `entr`,
`erf`, `erfc`, `erfcinv`, `erfcx`, `erfinv`, `exp1`, `expi`, `expit`,
`expn`, `factorial`, `fresnel`, `gamma`, `gammainc`, `gammaincc`,
`gammaln`, `gammasgn`, `hyp1f1`, `hyp2f1`, `i0`, `i0e`, `i1`, `i1e`,
`kl_div`, `log_ndtr`, `log_softmax`, `loggamma`, `logit`, `logsumexp`,
`lpmn`, `lpmn_values`, `multigammaln`, `ndtr`, `ndtri`, `owens_t`,
`poch`, `polygamma`, `rel_entr`, `sici`, `softmax`, `spence`,
`sph_harm_y`, `wofz`, `xlog1py`, `xlogy` and `zeta`, each with the
prefix `scipy_special_`.

## Statistical distributions

Each distribution of `jax.scipy.stats` provides the functions that JAX
provides for it (`logpdf`/`pdf` or `logpmf`/`pmf`, and for most
continuous distributions `cdf`, `logcdf`, `sf`, `logsf`, `ppf`, `isf`).

``` r

x <- as_jax(c(-1, 0, 1.5))
scipy_stats_norm_logpdf(x, loc = 0, scale = 2)
#> Array f32[3]
#> [1] -1.737086 -1.612086 -1.893336
scipy_stats_norm_cdf(x)
#> Array f32[3]
#> [1] 0.1586553 0.5000000 0.9331928
scipy_stats_norm_ppf(c(0.025, 0.975))
#> Array f32[2]
#> [1] -1.959964  1.959964
scipy_stats_gamma_logpdf(c(0.5, 2), a = 3, scale = 0.5)
#> Array f32[2]
#> [1] -1.000000 -1.227411
scipy_stats_poisson_logpmf(0:3, mu = 2)
#> Array f32[4] weak_type
#> [1] -2.000000 -1.306853 -1.306853 -1.712318
scipy_stats_t_logpdf(x, df = 4)
#> Array f32[3]
#> [1] -1.5386882 -0.9808292 -2.0965471
scipy_stats_beta_cdf(c(0.2, 0.5), 2, 3)
#> Array f32[2]
#> [1] 0.1808 0.6875
```

Distributions available: `bernoulli`, `beta`, `betabinom`, `binom`,
`cauchy`, `chi2`, `dirichlet`, `expon`, `gamma`, `gennorm`, `geom`,
`gumbel_l`, `gumbel_r`, `laplace`, `logistic`, `multinomial`,
`multivariate_normal`, `nbinom`, `norm`, `pareto`, `poisson`, `t`,
`truncnorm`, `uniform`, `vonmises` and `wrapcauchy`.

A log likelihood is an ordinary function, so maximum likelihood needs
only `grad`:

``` r

y <- random_normal(random_key(1L), 200L) * 1.5 + 3
nll <- function(theta) {
  -jnp_sum(scipy_stats_norm_logpdf(y, loc = theta[0], scale = jnp_exp(theta[1])))
}
fit <- scipy_optimize_minimize(nll, as_jax(c(0, 0)), method = "BFGS")
fit$x[0]              # mean
#> Array f32[]
#> [1] 3.200885
jnp_exp(fit$x[1])     # standard deviation
#> Array f32[]
#> [1] 1.611081
fit$status
#> Array i32[] weak_type
#> [1] 3
```

The estimates have converged, yet the status is 3 rather than 0: in
single precision the line search of BFGS runs out of digits near the
optimum and reports a failure. JAX behaves the same way. Use
[`enable_x64()`](https://adamrsophie.github.io/jaxR/reference/enable_x64.md)
when the convergence flag matters.

The observed information is the Hessian of the same function:

``` r

se <- jnp_sqrt(jnp_diag(jnp_linalg_inv(hessian(nll)(fit$x))))
se
#> Array f32[2]
#> [1] 0.11392063 0.05000025
```

### Multivariate normal and summary statistics

``` r

Sigma <- matrix(c(2, 0.5, 0.5, 1), 2)
scipy_stats_multivariate_normal_logpdf(c(0.3, -0.2), mean = c(0, 0), cov = Sigma)
#> Array f32[]
#> [1] -2.183399

v <- as_jax(c(3, 1, 4, 1, 5, 9, 2, 6))
scipy_stats_rankdata(v)
#> Array f32[8]
#> [1] 4.0 1.5 5.0 1.5 6.0 8.0 3.0 7.0
scipy_stats_mode(v)
#> $mode
#> Array f32[]
#> [1] 1
#> 
#> $count
#> Array i32[]
#> [1] 2
scipy_stats_sem(v)
#> Array f32[]
#> [1] 0.9716977
```

### Kernel density estimation

``` r

kde <- scipy_stats_gaussian_kde(y)
kde$evaluate(c(1, 3, 5))
#> Array f32[3]
#> [1] 0.1127215 0.2236105 0.1208788
kde$integrate_box_1d(-Inf, 3)
#> Array f32[]
#> [1] 0.4576454
```

## Linear algebra

`scipy_linalg_*` complements `jnp_linalg_*` with factorizations, matrix
functions and structured matrices.

``` r

A <- as_jax(matrix(c(4, 1, 2, 1, 3, 0, 2, 0, 5), 3))
b <- as_jax(c(1, 2, 3))

scipy_linalg_solve(A, b, assume_a = "pos")
#> Array f32[3]
#> [1] -0.3023256  0.7674418  0.7209302
cf <- scipy_linalg_cho_factor(A)
scipy_linalg_cho_solve(cf, b)
#> Array f32[3]
#> [1] -0.3023256  0.7674418  0.7209302

plu <- scipy_linalg_lu(A)             # P, L, U
jnp_allclose(plu[[1]] %*% plu[[2]] %*% plu[[3]], A)
#> Array bool[]
#> [1] FALSE

E <- scipy_linalg_expm(A * 0.1)       # matrix exponential
R <- scipy_linalg_sqrtm(A)            # matrix square root
jnp_allclose(jnp_real(R %*% R), A, atol = 1e-4)
#> Array bool[]
#> [1] TRUE

scipy_linalg_eigh_tridiagonal(c(2, 2, 2), c(-1, -1), eigvals_only = TRUE)
#> Array f32[3]
#> [1] 0.5857865 2.0000000 3.4142137
scipy_linalg_block_diag(jnp_eye(2L), 3 * jnp_ones(c(1L, 1L)))
#> Array f32[3,3]
#>      [,1] [,2] [,3]
#> [1,]    1    0    0
#> [2,]    0    1    0
#> [3,]    0    0    3
scipy_linalg_toeplitz(c(1, 2, 3))
#> Array f32[3,3]
#>      [,1] [,2] [,3]
#> [1,]    1    2    3
#> [2,]    2    1    2
#> [3,]    3    2    1
```

Also available: `cholesky`, `det`, `inv`, `eigh`, `svd`, `qr`,
`qr_multiply`, `lu_factor`, `lu_solve`, `solve_triangular`,
`solve_sylvester`, `schur`, `rsf2csf`, `hessenberg`, `polar`, `funm`,
`expm_frechet`, and the matrix constructors `circulant`, `companion`,
`convolution_matrix`, `dft`, `fiedler`, `fiedler_companion`, `hadamard`,
`hankel`, `helmert`, `hilbert`, `invhilbert`, `invpascal`, `leslie` and
`pascal`.

### Matrix free solvers

`scipy_sparse_linalg_cg`, `scipy_sparse_linalg_bicgstab` and
`scipy_sparse_linalg_gmres` solve `A x = b` when `A` is given as a
matrix or as a function. They return the solution and an info slot, and
they are differentiable through implicit differentiation.

``` r

matvec <- function(v) A %*% v
sol <- scipy_sparse_linalg_cg(matvec, b)
sol[[1]]
#> Array f32[3]
#> [1] -0.3023256  0.7674419  0.7209302
jnp_allclose(A %*% sol[[1]], b, atol = 1e-4)
#> Array bool[]
#> [1] TRUE
scipy_sparse_linalg_gmres(A, b)[[1]]
#> Array f32[3]
#> [1] -0.3023258  0.7674420  0.7209303
```

## Optimization

`scipy_optimize_minimize` implements BFGS, as in JAX. The objective
takes one array.

``` r

rosen <- function(p) (1 - p[0])^2 + 100 * (p[1] - p[0]^2)^2
res <- scipy_optimize_minimize(rosen, as_jax(c(0.8, 0.8)), method = "BFGS")
res$x
#> Array f32[2]
#> [1] 1 1
res$nit
#> Array i32[] weak_type
#> [1] 11
res$success
#> Array bool[]
#> [1] TRUE
```

The result also carries `fun`, `jac`, `hess_inv`, `nfev`, `njev` and
`status`. The line search is the one JAX uses, and it is less robust
than SciPy’s: from the classic starting point `c(-1.2, 1)` it stops
early with status 3, in jaxR exactly as in JAX.

## Integration, interpolation and images

``` r

t <- jnp_linspace(0, pi, 101L)
scipy_integrate_trapezoid(jnp_sin(t), t)
#> Array f32[]
#> [1] 1.999836

interp <- scipy_interpolate_RegularGridInterpolator(
  list(as_jax(c(0, 1, 2)), as_jax(c(0, 1))),
  as_jax(matrix(c(0, 1, 2, 10, 11, 12), 3)))
interp(as_jax(matrix(c(0.5, 0.5, 1.5, 0.25), 2, byrow = TRUE)))
#> Array f32[2]
#> [1] 5.5 4.0

img <- jnp_reshape(jnp_arange(12, dtype = "float32"), c(3L, 4L))
scipy_ndimage_map_coordinates(img, list(as_jax(c(0.5, 1.5)), as_jax(c(1, 2.5))), order = 1L)
#> Array f32[2]
#> [1] 3.0 8.5
```

## Signal processing

``` r

sig <- jnp_sin(2 * pi * 5 * jnp_linspace(0, 1, 256L, endpoint = FALSE))
scipy_signal_convolve(c(1, 2, 3), c(0, 1, 0.5))
#> Array f32[5]
#> [1] 0.0 1.0 2.5 4.0 1.5
scipy_signal_correlate(c(1, 2, 3), c(0, 1, 0.5), mode = "same")
#> Array f32[3]
#> [1] 2.0 3.5 3.0
scipy_signal_detrend(jnp_arange(5, dtype = "float32") + c(0, 1, 0, 1, 0))
#> Array f32[5]
#> [1] -0.3999996  0.6000004 -0.3999996  0.6000004 -0.3999996

pw <- scipy_signal_welch(sig, fs = 256, nperseg = 128L)
pw[[1]][jnp_argmax(pw[[2]])]          # the peak is at 4 or 6 Hz bins around 5 Hz
#> Array f32[]
#> [1] 6

st <- scipy_signal_stft(sig, fs = 256, nperseg = 64L)
dim(st[[3]])
#> [1] 33  9
back <- scipy_signal_istft(st[[3]], fs = 256, nperseg = 64L)
jnp_allclose(back[[2]][sl(0, 256)], sig, atol = 1e-4)
#> Array bool[]
#> [1] TRUE
```

Windows are named as in `scipy.signal.get_window` (`"hann"`,
`"hamming"`, `"blackman"`, `"bartlett"`, `"flattop"`, `"tukey"`, …). A
list gives a name with its parameters, where Python takes a tuple:
`list("kaiser", 8)`, `list("tukey", 0.25)`, `list("chebwin", 60)`,
`list("taylor", 4, 30)`, `list("dpss", 2)`. A 1-D array is used as the
window itself.

``` r

pk <- scipy_signal_welch(sig, fs = 256, nperseg = 128L, window = list("kaiser", 8))
pk[[1]][jnp_argmax(pk[[2]])]
#> Array f32[]
#> [1] 6
```

Discrete cosine transforms live in `scipy_fft_dct`, `scipy_fft_idct`,
`scipy_fft_dctn` and `scipy_fft_idctn`:

``` r

d <- scipy_fft_dct(c(1, 2, 3, 4), norm = "ortho")
d
#> Array f32[4]
#> [1]  5.0000000 -2.2304425  0.0000000 -0.1585127
scipy_fft_idct(d, norm = "ortho")
#> Array f32[4]
#> [1] 1 2 3 4
```

## Rotations

`scipy_spatial_transform_Rotation_*` ports
`jax.scipy.spatial.transform.Rotation`. A rotation is a pytree holding a
quaternion.

``` r

r <- scipy_spatial_transform_Rotation_from_euler("z", 90, degrees = TRUE)
scipy_spatial_transform_Rotation_apply(r, c(1, 0, 0))
#> Array f32[3]
#> [1] 0.0000000 0.9999999 0.0000000
scipy_spatial_transform_Rotation_as_matrix(r)
#> Array f32[3,3]
#>           [,1]       [,2]      [,3]
#> [1,] 0.0000000 -0.9999999 0.0000000
#> [2,] 0.9999999  0.0000000 0.0000000
#> [3,] 0.0000000  0.0000000 0.9999999
scipy_spatial_transform_Rotation_as_rotvec(r)
#> Array f32[3]
#> [1] 0.000000 0.000000 1.570796
r2 <- scipy_spatial_transform_Rotation_inv(r)
scipy_spatial_transform_Rotation_magnitude(r2)
#> Array f32[]
#> [1] 1.570796
```

## Clustering

``` r

obs <- as_jax(matrix(c(0, 0, 0.1, 0.2, 5, 5, 5.2, 4.9), 4, byrow = TRUE))
book <- as_jax(matrix(c(0, 0, 5, 5), 2, byrow = TRUE))
scipy_cluster_vq_vq(obs, book)        # codes and distances
#> [[1]]
#> Array i32[4]
#> [1] 0 0 1 1
#> 
#> [[2]]
#> Array f32[4]
#> [1] 0.0000000 0.2236068 0.0000000 0.2236066
```
