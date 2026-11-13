# The NumPy API: jnp

`jax.numpy` is a NumPy compatible array API whose functions can be
differentiated, vectorized and compiled. jaxR ports it as the `jnp_*`
functions: `jax.numpy.sum` is `jnp_sum`, `jax.numpy.linalg.solve` is
`jnp_linalg_solve`, `jax.numpy.fft.fft` is `jnp_fft_fft`. Arguments keep
their NumPy names and meaning; axes and indices are 0-based.

``` r

library(jaxR)
```

## Creating arrays

``` r

jnp_array(c(1, 2, 3))
#> Array f32[3]
#> [1] 1 2 3
jnp_array(list(c(1, 2), c(3, 4)))            # a nested list is a nested sequence
#> Array f32[2,2]
#>      [,1] [,2]
#> [1,]    1    2
#> [2,]    3    4
jnp_zeros(c(2L, 3L)); jnp_ones(2L); jnp_full(3L, 2.5); jnp_empty(2L)
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    0    0    0
#> [2,]    0    0    0
#> Array f32[2]
#> [1] 1 1
#> Array f32[3] weak_type
#> [1] 2.5 2.5 2.5
#> Array f32[2]
#> [1] 0 0
jnp_zeros_like(jnp_ones(c(1L, 2L)))
#> Array f32[1,2]
#>      [,1] [,2]
#> [1,]    0    0
jnp_arange(0, 10, 3)
#> Array i32[4]
#> [1] 0 3 6 9
jnp_linspace(0, 1, 5L, endpoint = FALSE)
#> Array f32[5]
#> [1] 0.0 0.2 0.4 0.6 0.8
jnp_logspace(0, 2, 3L)
#> Array f32[3]
#> [1]   1  10 100
jnp_geomspace(1, 8, 4L)
#> Array f32[4]
#> [1] 1 2 4 8
jnp_eye(2L, 3L, k = 1L)
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    0    1    0
#> [2,]    0    0    1
jnp_identity(2L)
#> Array f32[2,2]
#>      [,1] [,2]
#> [1,]    1    0
#> [2,]    0    1
jnp_tri(3L)
#> Array f32[3,3]
#>      [,1] [,2] [,3]
#> [1,]    1    0    0
#> [2,]    1    1    0
#> [3,]    1    1    1
jnp_diag(c(1, 2, 3))
#> Array f32[3,3]
#>      [,1] [,2] [,3]
#> [1,]    1    0    0
#> [2,]    0    2    0
#> [3,]    0    0    3
jnp_vander(c(1, 2, 3), 3L)
#> Array f32[3,3]
#>      [,1] [,2] [,3]
#> [1,]    1    1    1
#> [2,]    4    2    1
#> [3,]    9    3    1
```

Grids:

``` r

g <- jnp_meshgrid(jnp_arange(3), jnp_arange(2), indexing = "ij")
g[[1]]
#> Array i32[3,2]
#>      [,1] [,2]
#> [1,]    0    0
#> [2,]    1    1
#> [3,]    2    2
jnp_indices(c(2L, 2L))
#> Array i32[2,2,2]
#> [0, , ]
#>      [,1] [,2]
#> [1,]    0    0
#> [2,]    1    1
#> [1, , ]
#>      [,1] [,2]
#> [1,]    0    1
#> [2,]    0    1
```

Functions that build an array from a function, a buffer or a file:

``` r

jnp_fromfunction(function(i, j) i * 10 + j, c(2L, 3L))
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    0    1    2
#> [2,]   10   11   12
jnp_frombuffer(as.raw(c(1, 0, 0, 0, 2, 0, 0, 0)), dtype = "int32")
#> Array i32[2]
#> [1] 1 2
```

## Shape manipulation

``` r

a <- jnp_arange(12)
b <- jnp_reshape(a, c(3L, 4L))
b
#> Array i32[3,4]
#>      [,1] [,2] [,3] [,4]
#> [1,]    0    1    2    3
#> [2,]    4    5    6    7
#> [3,]    8    9   10   11
jnp_reshape(a, c(2L, -1L))
#> Array i32[2,6]
#>      [,1] [,2] [,3] [,4] [,5] [,6]
#> [1,]    0    1    2    3    4    5
#> [2,]    6    7    8    9   10   11
jnp_ravel(b)
#> Array i32[12]
#>  [1]  0  1  2  3  4  5  6  7  8  9 10 11
jnp_transpose(b)
#> Array i32[4,3]
#>      [,1] [,2] [,3]
#> [1,]    0    4    8
#> [2,]    1    5    9
#> [3,]    2    6   10
#> [4,]    3    7   11
t(b)
#> Array i32[4,3]
#>      [,1] [,2] [,3]
#> [1,]    0    4    8
#> [2,]    1    5    9
#> [3,]    2    6   10
#> [4,]    3    7   11
jnp_swapaxes(b, 0L, 1L)
#> Array i32[4,3]
#>      [,1] [,2] [,3]
#> [1,]    0    4    8
#> [2,]    1    5    9
#> [3,]    2    6   10
#> [4,]    3    7   11
jnp_moveaxis(jnp_zeros(c(2L, 3L, 4L)), 0L, -1L) |> dim()
#> [1] 3 4 2
jnp_expand_dims(a, 0L) |> dim()
#> [1]  1 12
jnp_squeeze(jnp_zeros(c(1L, 3L, 1L))) |> dim()
#> [1] 3
jnp_broadcast_to(as_jax(c(1, 2, 3)), c(2L, 3L))
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    1    2    3
#> [2,]    1    2    3
```

Joining, splitting and repeating:

``` r

jnp_concatenate(list(b, b), axis = 0L) |> dim()
#> [1] 6 4
jnp_stack(list(a, a)) |> dim()
#> [1]  2 12
jnp_vstack(list(a, a)) |> dim()
#> [1]  2 12
jnp_hstack(list(a, a)) |> dim()
#> [1] 24
jnp_split(a, 3L)
#> [[1]]
#> Array i32[4]
#> [1] 0 1 2 3
#> 
#> [[2]]
#> Array i32[4]
#> [1] 4 5 6 7
#> 
#> [[3]]
#> Array i32[4]
#> [1]  8  9 10 11
jnp_array_split(a, 5L)[[1]]
#> Array i32[3]
#> [1] 0 1 2
jnp_tile(as_jax(c(1, 2)), 3L)
#> Array f32[6]
#> [1] 1 2 1 2 1 2
jnp_repeat(as_jax(c(1, 2)), 2L)
#> Array f32[4]
#> [1] 1 1 2 2
jnp_roll(a, 2L)
#> Array i32[12]
#>  [1] 10 11  0  1  2  3  4  5  6  7  8  9
jnp_flip(b, axis = 1L)
#> Array i32[3,4]
#>      [,1] [,2] [,3] [,4]
#> [1,]    3    2    1    0
#> [2,]    7    6    5    4
#> [3,]   11   10    9    8
jnp_rot90(b) |> dim()
#> [1] 4 3
jnp_pad(as_jax(c(1, 2, 3)), c(1L, 2L), mode = "constant", constant_values = 9)
#> Array f32[6]
#> [1] 9 1 2 3 9 9
jnp_pad(as_jax(c(1, 2, 3)), 2L, mode = "reflect")
#> Array f32[7]
#> [1] 3 2 1 2 3 2 1
```

## Elementwise functions

Arithmetic, comparison, logic and bit operations follow NumPy’s type
promotion.

``` r

x <- as_jax(c(-1.5, 0.5, 2))
jnp_abs(x); jnp_sign(x); jnp_floor(x); jnp_ceil(x); jnp_round(x); jnp_trunc(x)
#> Array f32[3]
#> [1] 1.5 0.5 2.0
#> Array f32[3]
#> [1] -1  1  1
#> Array f32[3]
#> [1] -2  0  2
#> Array f32[3]
#> [1] -1  1  2
#> Array f32[3]
#> [1] -2  0  2
#> Array f32[3]
#> [1] -1  0  2
jnp_exp(x); jnp_expm1(x); jnp_log1p(jnp_abs(x)); jnp_sqrt(jnp_abs(x)); jnp_cbrt(x)
#> Array f32[3]
#> [1] 0.2231302 1.6487212 7.3890562
#> Array f32[3]
#> [1] -0.7768698  0.6487213  6.3890562
#> Array f32[3]
#> [1] 0.9162908 0.4054651 1.0986123
#> Array f32[3]
#> [1] 1.2247449 0.7071068 1.4142135
#> Array f32[3]
#> [1] -1.1447142  0.7937005  1.2599211
jnp_power(x, 2L); jnp_float_power(as_jax(2L), 0.5)
#> Array f32[3]
#> [1] 2.25 0.25 4.00
#> Array f32[] weak_type
#> [1] 1.414214
jnp_maximum(x, 0); jnp_clip(x, -1, 1)
#> Array f32[3]
#> [1] 0.0 0.5 2.0
#> Array f32[3]
#> [1] -1.0  0.5  1.0
jnp_floor_divide(as_jax(7L), 2L); jnp_remainder(as_jax(-7L), 3L); jnp_fmod(as_jax(-7L), 3L)
#> Array i32[] weak_type
#> [1] 3
#> Array i32[] weak_type
#> [1] 2
#> Array i32[] weak_type
#> [1] -1
jnp_divmod(as_jax(7L), 2L)
#> [[1]]
#> Array i32[] weak_type
#> [1] 3
#> 
#> [[2]]
#> Array i32[] weak_type
#> [1] 1
jnp_hypot(3, 4); jnp_arctan2(1, 1); jnp_deg2rad(180)
#> Array f32[] weak_type
#> [1] 5
#> Array f32[] weak_type
#> [1] 0.7853982
#> Array f32[] weak_type
#> [1] 3.141593
jnp_logaddexp(as_jax(1000), as_jax(1000))
#> Array f32[] weak_type
#> [1] 1000.693
jnp_isfinite(as_jax(c(1, Inf, NaN))); jnp_nan_to_num(as_jax(c(1, Inf, NaN)))
#> Array bool[3]
#> [1]  TRUE FALSE FALSE
#> Array f32[3]
#> [1] 1.000000e+00 3.402823e+38 0.000000e+00
jnp_where(x > 0, x, 0)
#> Array f32[3]
#> [1] 0.0 0.5 2.0
jnp_bitwise_and(as_jax(12L), as_jax(10L)); jnp_left_shift(as_jax(1L), 4L); jnp_bitwise_count(as_jax(255L))
#> Array i32[] weak_type
#> [1] 8
#> Array i32[] weak_type
#> [1] 16
#> Array u8[]
#> [1] 8
```

Complex numbers:

``` r

z <- as_jax(c(1 + 2i, -1i))
jnp_real(z); jnp_imag(z); jnp_conj(z); jnp_abs(z); jnp_angle(z)
#> Array f32[2]
#> [1] 1 0
#> Array f32[2]
#> [1]  2 -1
#> Array c64[2]
#> [1] 1-2i 0+1i
#> Array f32[2]
#> [1] 2.236068 1.000000
#> Array f32[2]
#> [1]  1.107149 -1.570796
```

Comparisons of arrays:

``` r

jnp_allclose(as_jax(c(1, 2)), as_jax(c(1, 2 + 1e-9)))
#> Array bool[]
#> [1] TRUE
jnp_array_equal(as_jax(c(1, 2)), as_jax(c(1, 2)))
#> Array bool[]
#> [1] TRUE
jnp_isclose(as_jax(c(1, 2)), as_jax(c(1.1, 2)))
#> Array bool[2]
#> [1] FALSE  TRUE
```

## Reductions

Every reduction takes `axis` (one or several, 0-based), `keepdims`, and
where NumPy has them `where`, `initial`, `dtype` and `ddof`.

``` r

m <- as_jax(matrix(c(1, 5, 2, 8, 3, 4), 2))
m
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    1    2    3
#> [2,]    5    8    4
jnp_sum(m); jnp_sum(m, axis = 0L); jnp_sum(m, axis = 1L, keepdims = TRUE)
#> Array f32[]
#> [1] 23
#> Array f32[3]
#> [1]  6 10  7
#> Array f32[2,1]
#>      [,1]
#> [1,]    6
#> [2,]   17
jnp_prod(m, axis = 1L)
#> Array f32[2]
#> [1]   6 160
jnp_mean(m, axis = 0L); jnp_var(m, ddof = 1L); jnp_std(m, axis = 1L)
#> Array f32[3]
#> [1] 3.0 5.0 3.5
#> Array f32[]
#> [1] 6.166667
#> Array f32[2]
#> [1] 0.8164966 1.6996732
jnp_min(m, axis = 1L); jnp_max(m, initial = 10)
#> Array f32[2]
#> [1] 1 4
#> Array f32[]
#> [1] 10
jnp_sum(m, where = m > 2)
#> Array f32[]
#> [1] 20
jnp_argmax(m, axis = 1L); jnp_argmin(m)
#> Array i32[2]
#> [1] 2 1
#> Array i32[]
#> [1] 0
jnp_any(m > 7); jnp_all(m > 0, axis = 0L)
#> Array bool[]
#> [1] TRUE
#> Array bool[3]
#> [1] TRUE TRUE TRUE
jnp_cumsum(m, axis = 1L); jnp_cumprod(as_jax(c(1, 2, 3)))
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    1    3    6
#> [2,]    5   13   17
#> Array f32[3]
#> [1] 1 2 6
jnp_median(m); jnp_quantile(m, c(0.25, 0.75)); jnp_percentile(m, 50, axis = 0L)
#> Array f32[]
#> [1] 3.5
#> Array f32[2]
#> [1] 2.25 4.75
#> Array f32[3]
#> [1] 3.0 5.0 3.5
jnp_ptp(m, axis = 1L); jnp_average(m, axis = 0L, weights = as_jax(c(1, 3)))
#> Array f32[2]
#> [1] 2 4
#> Array f32[3]
#> [1] 4.00 6.50 3.75
jnp_count_nonzero(m > 2)
#> Array i32[]
#> [1] 4
```

NaN aware versions ignore missing values:

``` r

v <- as_jax(c(1, NaN, 3))
jnp_nansum(v); jnp_nanmean(v); jnp_nanmax(v); jnp_nanstd(v); jnp_nanargmin(v)
#> Array f32[]
#> [1] 4
#> Array f32[]
#> [1] 2
#> Array f32[]
#> [1] 3
#> Array f32[]
#> [1] 1
#> Array i32[]
#> [1] 0
```

Statistics and histograms:

``` r

jnp_cov(m); jnp_corrcoef(m)
#> Array f32[2,2]
#>      [,1]      [,2]
#> [1,]  1.0 -0.500000
#> [2,] -0.5  4.333333
#> Array f32[2,2]
#>            [,1]       [,2]
#> [1,]  1.0000000 -0.2401922
#> [2,] -0.2401922  1.0000000
jnp_histogram(as_jax(c(1, 2, 2, 3, 5)), bins = 4L)
#> [[1]]
#> Array f32[4]
#> [1] 1 2 1 1
#> 
#> [[2]]
#> Array f32[5]
#> [1] 1 2 3 4 5
jnp_bincount(as_jax(c(0L, 1L, 1L, 3L)))
#> Array i32[4]
#> [1] 1 2 0 1
jnp_digitize(as_jax(c(0.5, 2.5)), as_jax(c(0, 1, 2, 3)))
#> Array i32[2]
#> [1] 1 3
```

## Sorting, searching and sets

``` r

s <- as_jax(c(3, 1, 2, 3, 1))
jnp_sort(s); jnp_argsort(s); jnp_sort(s, descending = TRUE)
#> Array f32[5]
#> [1] 1 1 2 3 3
#> Array i32[5]
#> [1] 1 4 2 0 3
#> Array f32[5]
#> [1] 3 3 2 1 1
jnp_searchsorted(jnp_sort(s), as_jax(c(2, 3)))
#> Array i32[2]
#> [1] 2 3
jnp_partition(s, 2L)
#> Array f32[5]
#> [1] 1 1 2 3 3
jnp_unique(s); jnp_unique_counts(s)
#> Array f32[3]
#> [1] 1 2 3
#> $values
#> Array f32[3]
#> [1] 1 2 3
#> 
#> $counts
#> Array i32[3]
#> [1] 2 1 2
jnp_union1d(s, as_jax(c(9, 1))); jnp_intersect1d(s, as_jax(c(9, 1))); jnp_setdiff1d(s, as_jax(c(1)))
#> Array f32[4]
#> [1] 1 2 3 9
#> Array f32[1]
#> [1] 1
#> Array f32[2]
#> [1] 2 3
jnp_isin(s, as_jax(c(1, 2)))
#> Array bool[5]
#> [1] FALSE  TRUE  TRUE FALSE  TRUE
jnp_nonzero(as_jax(c(0, 3, 0, 4)))
#> [[1]]
#> Array i32[2]
#> [1] 1 3
jnp_argwhere(as_jax(c(0, 3, 0, 4)))
#> Array i32[2,1]
#>      [,1]
#> [1,]    1
#> [2,]    3
jnp_lexsort(list(as_jax(c(2, 1, 2)), as_jax(c(1, 1, 0))))
#> Array i32[3]
#> [1] 2 1 0
```

Under `jit` the size of a result cannot depend on the data. Functions
whose output size is data dependent take a `size` argument (and a
`fill_value`) to make it static.

``` r

nz <- jit(function(v) jnp_nonzero(v, size = 3L, fill_value = -1L)[[1]])
nz(as_jax(c(0, 3, 0, 4)))
#> Array i32[3]
#> [1]  1  3 -1
jit(function(v) jnp_unique(v, size = 4L, fill_value = 0))(s)
#> Array f32[4]
#> [1] 1 2 3 0
```

## Indexing functions

``` r

jnp_take(s, as_jax(c(0L, 0L, 4L)))
#> Array f32[3]
#> [1] 3 3 1
jnp_take_along_axis(m, jnp_argsort(m, axis = 1L), axis = 1L)
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    1    2    3
#> [2,]    4    5    8
jnp_compress(as_jax(c(TRUE, FALSE, TRUE, FALSE, TRUE)), s)
#> Array f32[3]
#> [1] 3 2 1
jnp_choose(as_jax(c(0L, 1L)), list(as_jax(c(10, 20)), as_jax(c(30, 40))))
#> Array f32[2]
#> [1] 10 40
jnp_select(list(s < 2, s > 2), list(s * 10, s * 100), default = 0)
#> Array f32[5]
#> [1] 300  10   0 300  10
jnp_diagonal(m); jnp_tril(m); jnp_triu_indices(3L)
#> Array f32[2]
#> [1] 1 8
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    1    0    0
#> [2,]    5    8    0
#> [[1]]
#> Array i32[6]
#> [1] 0 0 0 1 1 2
#> 
#> [[2]]
#> Array i32[6]
#> [1] 0 1 2 1 2 2
jnp_unravel_index(5L, c(2L, 3L)); jnp_ravel_multi_index(list(1L, 2L), c(2L, 3L))
#> [[1]]
#> Array i32[] weak_type
#> [1] 1
#> 
#> [[2]]
#> Array i32[] weak_type
#> [1] 2
#> Array i32[]
#> [1] 5
jnp_put_along_axis(m, as_jax(matrix(c(0L, 1L), 2)), 99, axis = 1L, inplace = FALSE)
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]   99    2    3
#> [2,]    5   99    4
```

[`ops_segment_sum()`](https://adamrsophie.github.io/jaxR/reference/ops_segment_sum.md)
and its relatives reduce groups of entries:

``` r

ops_segment_sum(as_jax(c(1, 2, 3, 4)), as_jax(c(0L, 0L, 1L, 2L)), num_segments = 3L)
#> Array f32[3]
#> [1] 3 3 4
```

## Linear algebra

Products:

``` r

A <- as_jax(matrix(c(4, 1, 2, 1, 3, 0, 2, 0, 5), 3))
u <- as_jax(c(1, 2, 3))
jnp_dot(A, u); A %*% u; jnp_matmul(A, A) |> dim()
#> Array f32[3]
#> [1] 12  7 17
#> Array f32[3]
#> [1] 12  7 17
#> [1] 3 3
jnp_vdot(u, u); jnp_inner(u, u); jnp_outer(u, u) |> dim(); jnp_cross(u, as_jax(c(0, 1, 0)))
#> Array f32[]
#> [1] 14
#> Array f32[]
#> [1] 14
#> [1] 3 3
#> Array f32[3]
#> [1] -3  0  1
jnp_tensordot(A, A, axes = 2L); jnp_kron(jnp_eye(2L), jnp_ones(c(1L, 2L)))
#> Array f32[]
#> [1] 60
#> Array f32[2,4]
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    1    0    0
#> [2,]    0    0    1    1
jnp_trace(A)
#> Array f32[]
#> [1] 12
jnp_einsum("ij,j->i", A, u)
#> Array f32[3]
#> [1] 12  7 17
jnp_einsum("ii->", A)
#> Array f32[]
#> [1] 12
jnp_einsum("ij,jk,kl->il", A, A, A) |> dim()
#> [1] 3 3
```

`jnp_linalg_*` has the decompositions and solvers.

``` r

jnp_linalg_solve(A, u)
#> Array f32[3]
#> [1] -0.3023256  0.7674419  0.7209302
jnp_linalg_inv(A) %*% A
#> Array f32[3,3]
#>               [,1]          [,2]          [,3]
#> [1,]  9.999999e-01 -2.235174e-08  1.490116e-08
#> [2,] -1.490116e-08  1.000000e+00 -7.450581e-09
#> [3,]  0.000000e+00 -7.450581e-09  1.000000e+00
jnp_linalg_det(A); jnp_linalg_slogdet(A)
#> Array f32[]
#> [1] 43
#> $sign
#> Array f32[]
#> [1] 1
#> 
#> $logabsdet
#> Array f32[]
#> [1] 3.7612
jnp_linalg_cholesky(A)
#> Array f32[3,3]
#>      [,1]       [,2]     [,3]
#> [1,]  2.0  0.0000000 0.000000
#> [2,]  0.5  1.6583124 0.000000
#> [3,]  1.0 -0.3015113 1.977142
qr <- jnp_linalg_qr(A); qr[[1]] %*% qr[[2]]
#> Array f32[3,3]
#>      [,1]          [,2]          [,3]
#> [1,]    4  9.999999e-01  2.000000e+00
#> [2,]    1  3.000000e+00 -6.407687e-08
#> [3,]    2 -5.859445e-08  5.000000e+00
sv <- jnp_linalg_svd(A, full_matrices = FALSE); sv[[2]]
#> Array f32[3]
#> [1] 6.669079 3.476024 1.854897
eg <- jnp_linalg_eigh(A); eg$eigenvalues
#> Array f32[3]
#> [1] 1.854897 3.476024 6.669079
jnp_linalg_eigvals(A)
#> Array c64[3]
#> [1] 6.669079+0i 1.854897+0i 3.476024+0i
jnp_linalg_norm(A); jnp_linalg_norm(A, ord = "nuc"); jnp_linalg_norm(u, ord = 1)
#> Array f32[]
#> [1] 7.745967
#> Array f32[]
#> [1] 12
#> Array f32[]
#> [1] 6
jnp_linalg_cond(A); jnp_linalg_matrix_rank(A)
#> Array f32[]
#> [1] 3.59539
#> Array i32[]
#> [1] 3
jnp_linalg_pinv(A[sl(0, 2), ]) |> dim()
#> [1] 3 2
jnp_linalg_lstsq(A[, sl(0, 2)], u)[[1]]
#> Array f32[2]
#> [1] 0.4409938 0.3913043
jnp_linalg_matrix_power(A, 3L)
#> Array f32[3,3]
#>      [,1] [,2] [,3]
#> [1,]  127   42  132
#> [2,]   42   37   24
#> [3,]  132   24  181
```

All of them are differentiable and batch over leading dimensions.

``` r

grad(function(M) jnp_linalg_slogdet(M)[[2]])(A)     # the inverse transpose
#> Array f32[3,3]
#>            [,1]        [,2]        [,3]
#> [1,]  0.3488372 -0.11627907 -0.13953489
#> [2,] -0.1162791  0.37209302  0.04651163
#> [3,] -0.1395349  0.04651163  0.25581396
jnp_linalg_det(jnp_stack(list(A, 2 * A)))
#> Array f32[2]
#> [1]  43 344
```

## Fast Fourier transforms

``` r

sig <- jnp_sin(2 * pi * 3 * jnp_linspace(0, 1, 16L, endpoint = FALSE))
F <- jnp_fft_fft(sig)
jnp_round(jnp_abs(F), 3L)
#> Array f32[16]
#>  [1] 0 0 0 8 0 0 0 0 0 0 0 0 0 8 0 0
jnp_fft_fftfreq(8L, d = 0.125)
#> Array f32[8]
#> [1]  0  1  2  3 -4 -3 -2 -1
jnp_allclose(jnp_real(jnp_fft_ifft(F)), sig, atol = 1e-5)
#> Array bool[]
#> [1] TRUE
jnp_fft_rfft(sig) |> dim()
#> [1] 9
jnp_fft_fft2(jnp_ones(c(2L, 2L)))
#> Array c64[2,2]
#>      [,1] [,2]
#> [1,] 4+0i 0+0i
#> [2,] 0+0i 0+0i
jnp_fft_fftshift(jnp_arange(6))
#> Array i32[6]
#> [1] 3 4 5 0 1 2
```

## Polynomials, interpolation and convolution

``` r

p <- as_jax(c(1, -3, 2))                 # x^2 - 3 x + 2
jnp_polyval(p, as_jax(c(0, 1, 2, 3)))
#> Array f32[4]
#> [1] 2 0 0 2
jnp_roots(p)
#> Array c64[2]
#> [1] 2+0i 1+0i
jnp_polyder(p); jnp_polyint(p)
#> Array f32[2]
#> [1]  2 -3
#> Array f32[4]
#> [1]  0.3333333 -1.5000000  2.0000000  0.0000000
jnp_polymul(p, as_jax(c(1, 1))); jnp_polyadd(p, as_jax(c(1, 1)))
#> Array f32[4]
#> [1]  1 -2 -1  2
#> Array f32[3]
#> [1]  1 -2  3
jnp_polyfit(as_jax(c(0, 1, 2, 3)), as_jax(c(2, 0, 0, 2)), 2L)
#> Array f32[3]
#> [1]  0.9999998 -2.9999998  1.9999999
jnp_interp(as_jax(c(0.5, 1.5)), as_jax(c(0, 1, 2)), as_jax(c(0, 10, 0)))
#> Array f32[2]
#> [1] 5 5
jnp_convolve(as_jax(c(1, 2, 3)), as_jax(c(0, 1, 0.5)), mode = "same")
#> Array f32[3]
#> [1] 1.0 2.5 4.0
jnp_correlate(as_jax(c(1, 2, 3)), as_jax(c(0, 1, 0.5)), mode = "valid")
#> Array f32[1]
#> [1] 3.5
jnp_gradient(as_jax(c(1, 4, 9, 16)))
#> Array f32[4]
#> [1] 3 4 6 7
jnp_diff(as_jax(c(1, 4, 9, 16)), n = 2L)
#> Array f32[2]
#> [1] 2 2
jnp_trapezoid(as_jax(c(1, 4, 9)))
#> Array f32[]
#> [1] 9
```

## Window functions

``` r

jnp_hanning(5L); jnp_hamming(5L); jnp_blackman(5L); jnp_bartlett(5L); jnp_kaiser(5L, 2)
#> Array f32[5]
#> [1] 0.0 0.5 1.0 0.5 0.0
#> Array f32[5]
#> [1] 0.08000001 0.54000002 1.00000000 0.54000002 0.08000001
#> Array f32[5]
#> [1] -1.490116e-08  3.400000e-01  9.999999e-01  3.400000e-01 -1.490116e-08
#> Array f32[5]
#> [1] 0.0 0.5 1.0 0.5 0.0
#> Array f32[5]
#> [1] 0.4386764 0.8347615 1.0000000 0.8347615 0.4386764
```

## Vectorizing helpers

[`jnp_vectorize()`](https://adamrsophie.github.io/jaxR/reference/jnp_vectorize.md)
gives a function NumPy style broadcasting over a core signature, and
[`jnp_apply_along_axis()`](https://adamrsophie.github.io/jaxR/reference/jnp_apply_along_axis.md)
applies a function to 1-D slices.

``` r

mv <- jnp_vectorize(function(M, v) M %*% v, signature = "(n,m),(m)->(n)")
mv(jnp_stack(list(A, 2 * A)), u)
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]   12    7   17
#> [2,]   24   14   34
jnp_apply_along_axis(function(v) v / jnp_sum(v), 1L, m)
#> Array f32[2,3]
#>           [,1]      [,2]      [,3]
#> [1,] 0.1666667 0.3333333 0.5000000
#> [2,] 0.2941177 0.4705882 0.2352941
```

## Universal functions

`jnp_ufuncs` holds the ufunc objects of `jax.numpy` and
[`jnp_frompyfunc()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufunc.md)
builds one from a scalar function. The `jnp_ufunc_*` functions call
their methods.

``` r

add <- jnp_frompyfunc(function(a, b) a + b, nin = 2L, nout = 1L, identity = 0)
jnp_ufunc_reduce(add, as_jax(c(1, 2, 3)))
#> Array f32[]
#> [1] 6
jnp_ufunc_accumulate(add, as_jax(c(1, 2, 3)))
#> Array f32[3]
#> [1] 1 3 6
jnp_ufunc_outer(add, as_jax(c(1, 2)), as_jax(c(10, 20)))
#> Array f32[2,2]
#>      [,1] [,2]
#> [1,]   11   21
#> [2,]   12   22
jnp_ufunc_at(add, as_jax(c(0, 0, 0)), as_jax(c(0L, 0L, 2L)), 1, inplace = FALSE)
#> Array f32[3]
#> [1] 2 0 1
```

## Saving and loading

[`jnp_save()`](https://adamrsophie.github.io/jaxR/reference/jnp_save.md),
[`jnp_savez()`](https://adamrsophie.github.io/jaxR/reference/jnp_save.md)
and
[`jnp_load()`](https://adamrsophie.github.io/jaxR/reference/jnp_save.md)
read and write the NumPy `.npy` and `.npz` formats, so arrays can be
exchanged with Python.

``` r

f <- tempfile(fileext = ".npy")
jnp_save(f, m)
jnp_load(f)
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    1    2    3
#> [2,]    5    8    4
```

## Printing

``` r

jnp_printoptions(print(as_jax(pi * 1:3)), precision = 3L)
#> Array f32[3]
#> [1] 3.141593 6.283185 9.424778
jnp_array_str(as_jax(c(1.5, 2.5)))
#> [1] "[1.5 2.5]"
```
