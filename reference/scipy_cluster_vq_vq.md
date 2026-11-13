# Vector quantization

Port of `jax.scipy.cluster.vq.vq`: assigns each observation to the
nearest code in the code book (Euclidean distance).

## Usage

``` r
scipy_cluster_vq_vq(obs, code_book, check_finite = TRUE)
```

## Arguments

- obs:

  Array of observations, shape `(M, N)` (or `(M,)` for scalar features).

- code_book:

  Array of codes, shape `(K, N)` (or `(K,)`).

- check_finite:

  Unused (accepted for compatibility).

## Value

A list of the 0-based code index of each observation and its distance to
that code.

## Examples

``` r
if (FALSE) { # \dontrun{
scipy_cluster_vq_vq(matrix(c(0, 0, 1, 1, 5, 5), 3, byrow = TRUE),
                    matrix(c(0, 0, 4, 4), 2, byrow = TRUE))
} # }
```
