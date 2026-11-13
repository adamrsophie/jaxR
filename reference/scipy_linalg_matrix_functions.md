# Matrix functions (jax.scipy.linalg)

Ports of `jax.scipy.linalg.expm` (Pade approximation with scaling and
squaring), `expm_frechet` (the Frechet derivative, computed as the
forward mode derivative of `expm`, as in JAX), `sqrtm` (via the complex
Schur form), `funm` (Schur-Parlett), `polar`, `rsf2csf` and
`solve_sylvester`.

## Usage

``` r
scipy_linalg_expm(A, upper_triangular = FALSE, max_squarings = 16L)

scipy_linalg_expm_frechet(A, E, method = NULL, compute_expm = TRUE)

scipy_linalg_polar(
  a,
  side = "right",
  method = "qdwh",
  eps = NULL,
  max_iterations = NULL
)

scipy_linalg_sqrtm(A, blocksize = 1L)

scipy_linalg_rsf2csf(T, Z, check_finite = TRUE)

scipy_linalg_funm(A, func, disp = TRUE)

scipy_linalg_solve_sylvester(A, B, C, method = "schur", tol = 1e-08)
```

## Arguments

- A, B, C:

  Square matrices (`C` has the shape of `A %*% X`).

- upper_triangular:

  Whether `A` is upper triangular (a faster solve is used).

- max_squarings:

  Maximum number of squarings; larger norms give `NaN`.

- E:

  Direction of the Frechet derivative (same shape as `A`).

- method:

  For `expm_frechet`: ignored. For `polar`: `"qdwh"` or `"svd"`. For
  `solve_sylvester`: `"schur"` or `"eigen"`.

- compute_expm:

  Whether to also return `expm(A)` (`list(expm_A, frechet)`).

- a:

  Matrix (or batch of matrices) to decompose.

- side:

  `"right"` (`a = u p`) or `"left"` (`a = p u`).

- eps, max_iterations:

  Convergence parameters of the QDWH iteration.

- blocksize:

  Must be 1 (the blocked algorithm is not implemented).

- T, Z:

  Real Schur form and Schur vectors (as returned by
  [`scipy_linalg_schur()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)).

- check_finite:

  Ignored.

- func:

  Function applied elementwise to the eigenvalues (a vector).

- disp:

  If `FALSE`, also return an error estimate (`list(F, err)`).

- tol:

  Threshold on `|eig(A) + eig(B)|` below which the result is `NaN`.

## Value

The matrix function, or the lists described above: `polar` gives
`list(u, p)` and `rsf2csf` gives `list(T, Z)` in complex Schur form.
