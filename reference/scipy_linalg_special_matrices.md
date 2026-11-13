# Special matrices (jax.scipy.linalg)

Ports of the matrix constructors of `jax.scipy.linalg`. `toeplitz`,
`hankel`, `circulant`, `leslie`, `companion`, `fiedler`,
`fiedler_companion` and `convolution_matrix` work on batches of vectors
(leading dimensions).

## Usage

``` r
scipy_linalg_block_diag(...)

scipy_linalg_toeplitz(c, r = NULL)

scipy_linalg_hankel(c, r = NULL)

scipy_linalg_circulant(c)

scipy_linalg_leslie(f, s)

scipy_linalg_companion(a)

scipy_linalg_fiedler(a)

scipy_linalg_fiedler_companion(a)

scipy_linalg_convolution_matrix(a, n, mode = "full")

scipy_linalg_hilbert(n)

scipy_linalg_invhilbert(n)

scipy_linalg_pascal(n, kind = NULL)

scipy_linalg_invpascal(n, kind = NULL)

scipy_linalg_helmert(n, full = FALSE)

scipy_linalg_hadamard(n, dtype = "int")

scipy_linalg_dft(n, scale = NULL, dtype = NULL)
```

## Arguments

- ...:

  Arrays of at most 2 dimensions (for `block_diag`).

- c:

  First column (`toeplitz`, `hankel`, `circulant`).

- r:

  Last (`hankel`) or first (`toeplitz`) row; `NULL` for the default.

- f, s:

  Fecundity coefficients and survival rates (`leslie`).

- a:

  Coefficients (`companion`, `fiedler`, `fiedler_companion`) or the
  filter (`convolution_matrix`).

- n:

  Matrix size (or the input length for `convolution_matrix`).

- mode:

  `"full"`, `"valid"` or `"same"`.

- kind:

  `"symmetric"` (default), `"lower"` or `"upper"`.

- full:

  Whether `helmert` includes the first row.

- dtype:

  Result dtype (`hadamard`: default integer; `dft`: default complex).

- scale:

  `NULL`, `"sqrtn"` or `"n"`.

## Value

The matrix (or batch of matrices).
