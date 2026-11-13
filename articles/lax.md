# lax: primitives and structured control flow

`jax.lax` is the layer under `jax.numpy`: a library of primitive
operations that map one to one to XLA operations. Each primitive carries
its shape rule, its lowering to StableHLO and its differentiation and
batching rules. jaxR ports it as `lax_*`.

Compared with `jnp_*`, the `lax_*` functions are stricter: they do not
promote dtypes implicitly and do not broadcast ranks. They also expose
operations that NumPy does not have, most importantly *structured
control flow*.

``` r

library(jaxR)
```

## Strict typing

``` r

jnp_add(as_jax(1L), as_jax(1.5))               # promotes to float32
#> Array f32[] weak_type
#> [1] 2.5
tryCatch(lax_add(as_jax(1L), as_jax(1.5)), error = function(e) conditionMessage(e))
#> [1] "lax.add requires arguments to have the same dtypes, got int32, float32."
lax_add(lax_convert_element_type(as_jax(1L), "float32"), as_jax(1.5))
#> Array f32[]
#> [1] 2.5
```

## Elementwise primitives

``` r

x <- as_jax(c(0.5, 1, 2))
lax_exp(x); lax_log(x); lax_sin(x); lax_tanh(x); lax_logistic(x)
#> Array f32[3]
#> [1] 1.648721 2.718282 7.389056
#> Array f32[3]
#> [1] -0.6931472  0.0000000  0.6931472
#> Array f32[3]
#> [1] 0.4794255 0.8414710 0.9092974
#> Array f32[3]
#> [1] 0.4621172 0.7615942 0.9640276
#> Array f32[3]
#> [1] 0.6224594 0.7310586 0.8807970
lax_integer_pow(x, 3L); lax_rsqrt(x); lax_erf(x); lax_lgamma(x); lax_digamma(x)
#> Array f32[3]
#> [1] 0.125 1.000 8.000
#> Array f32[3]
#> [1] 1.4142135 1.0000000 0.7071068
#> Array f32[3]
#> [1] 0.5204999 0.8427008 0.9953222
#> Array f32[3]
#> [1] 0.5723649 0.0000000 0.0000000
#> Array f32[3]
#> [1] -1.9635100 -0.5772154  0.4227846
lax_max(x, lax_full_like(x, 1)); lax_clamp(0.75, x, 1.5)
#> Array f32[3]
#> [1] 1 1 2
#> Array f32[3]
#> [1] 0.75 1.00 1.50
lax_select(x > 1, x, lax_neg(x))
#> Array f32[3]
#> [1] -0.5 -1.0  2.0
lax_nextafter(x, lax_full_like(x, Inf)) - x
#> Array f32[3]
#> [1] 5.960464e-08 1.192093e-07 2.384186e-07
lax_bitcast_convert_type(as_jax(1), "int32")
#> Array i32[]
#> [1] 1065353216
lax_population_count(as_jax(255L)); lax_clz(as_jax(1L)); lax_shift_left(as_jax(1L), as_jax(4L))
#> Array i32[] weak_type
#> [1] 8
#> Array i32[] weak_type
#> [1] 31
#> Array i32[] weak_type
#> [1] 16
```

## Shapes

``` r

a <- lax_iota("float32", 6L)
lax_reshape(a, c(2L, 3L))
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    0    1    2
#> [2,]    3    4    5
lax_broadcast_in_dim(as_jax(c(1, 2)), c(2L, 3L), 0L)
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    1    1    1
#> [2,]    2    2    2
lax_transpose(lax_reshape(a, c(2L, 3L)), c(1L, 0L)) |> dim()
#> [1] 3 2
lax_concatenate(list(a, a), 0L) |> dim()
#> [1] 12
lax_pad(a, 0, list(c(1L, 2L, 0L)))               # low, high and interior padding
#> Array f32[9]
#> [1] 0 0 1 2 3 4 5 0 0
lax_pad(a, 0, list(c(0L, 0L, 1L)))
#> Array f32[11]
#>  [1] 0 0 1 0 2 0 3 0 4 0 5
lax_slice(a, 1L, 5L, 2L)
#> Array f32[2]
#> [1] 1 3
lax_slice_in_dim(a, 2L, NULL)
#> Array f32[4]
#> [1] 2 3 4 5
lax_dynamic_slice(a, 2L, 3L)
#> Array f32[3]
#> [1] 2 3 4
lax_rev(a, 0L)
#> Array f32[6]
#> [1] 5 4 3 2 1 0
lax_expand_dims(a, c(0L, 2L)) |> dim()
#> [1] 1 6 1
lax_squeeze(jnp_zeros(c(1L, 3L)), 0L) |> dim()
#> [1] 3
```

## Reductions and windows

``` r

m <- lax_reshape(lax_iota("float32", 12L), c(3L, 4L))
lax_reduce_sum(m, 0L); lax_reduce_max(m, c(0L, 1L)); lax_argmax(m, 1L, "int32")
#> Array f32[4]
#> [1] 12 15 18 21
#> Array f32[]
#> [1] 11
#> Array i32[3]
#> [1] 3 3 3
lax_reduce(m, 0, lax_add, 1L)                    # a general monoid reduction
#> Array f32[3]
#> [1]  6 22 38
lax_cumsum(m, axis = 1L); lax_cummax(m, axis = 0L, reverse = TRUE)
#> Array f32[3,4]
#>      [,1] [,2] [,3] [,4]
#> [1,]    0    1    3    6
#> [2,]    4    9   15   22
#> [3,]    8   17   27   38
#> Array f32[3,4]
#>      [,1] [,2] [,3] [,4]
#> [1,]    8    9   10   11
#> [2,]    8    9   10   11
#> [3,]    8    9   10   11
lax_reduce_window(m, -Inf, lax_max, c(2L, 2L), c(1L, 1L), "VALID")   # 2 x 2 max pooling
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    5    6    7
#> [2,]    9   10   11
lax_top_k(as_jax(c(3, 1, 4, 1, 5)), 2L)
#> [[1]]
#> Array f32[2]
#> [1] 5 4
#> 
#> [[2]]
#> Array i32[2]
#> [1] 4 2
lax_sort(as_jax(c(3, 1, 4, 1, 5)))
#> Array f32[5]
#> [1] 1 1 3 4 5
lax_sort_key_val(as_jax(c(3, 1, 2)), as_jax(c(30, 10, 20)))
#> [[1]]
#> Array f32[3]
#> [1] 1 2 3
#> 
#> [[2]]
#> Array f32[3]
#> [1] 10 20 30
```

## Products and convolutions

``` r

A <- lax_reshape(lax_iota("float32", 6L), c(2L, 3L))
lax_dot(A, lax_transpose(A, c(1L, 0L)))
#> Array f32[2,2]
#>      [,1] [,2]
#> [1,]    5   14
#> [2,]   14   50
lax_dot_general(A, A, list(list(1L, 1L), list(integer(0), integer(0))))
#> Array f32[2,2]
#>      [,1] [,2]
#> [1,]    5   14
#> [2,]   14   50
lax_batch_matmul(jnp_stack(list(A, A)), jnp_stack(list(t(A), t(A)))) |> dim()
#> [1] 2 2 2
```

[`lax_conv_general_dilated()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_general_dilated.md)
is the general N-dimensional convolution: strides, padding, input and
kernel dilation, feature groups and arbitrary layouts.

``` r

img <- jnp_reshape(jnp_arange(25, dtype = "float32"), c(1L, 1L, 5L, 5L))     # NCHW
ker <- jnp_ones(c(1L, 1L, 3L, 3L)) / 9                                       # OIHW
lax_conv(img, ker, c(1L, 1L), "VALID")
#> Array f32[1,1,3,3]
#> [0, 0, , ]
#>      [,1] [,2] [,3]
#> [1,]    6    7    8
#> [2,]   11   12   13
#> [3,]   16   17   18
lax_conv_general_dilated(img, ker, c(2L, 2L), "SAME") |> dim()
#> [1] 1 1 3 3
lax_conv_transpose(img, jnp_ones(c(3L, 3L, 1L, 1L)), c(2L, 2L), "SAME",
                   dimension_numbers = c("NCHW", "HWIO", "NCHW")) |> dim()
#> [1]  1  1 10 10
```

## Gather and scatter

[`lax_gather()`](https://adamrsophie.github.io/jaxR/reference/lax_gather.md)
and the `lax_scatter*()` family are the primitives behind indexing and
`at_*` updates.

``` r

v <- lax_iota("float32", 5L)
lax_index_take(v, list(as_jax(c(0L, 4L))), 0L)
#> Array f32[2]
#> [1] 0 4
lax_scatter_add(v, as_jax(matrix(c(0L, 0L, 2L), 3)), as_jax(c(10, 10, 10)),
                lax_ScatterDimensionNumbers(update_window_dims = integer(0), inserted_window_dims = 0L,
                                            scatter_dims_to_operand_dims = 0L))
#> Array f32[5]
#> [1] 20  1 12  3  4
```

## Structured control flow

R’s `if`, `for` and `while` run at trace time. To put a branch or a loop
*inside* the compiled program, where it can depend on array values, use
the control flow operators. They are differentiable and work under
`vmap` and `jit`.

### cond and switch

``` r

safe_log <- function(x) lax_cond(x > 0, function(v) jnp_log(v), function(v) jnp_zeros_like(v), x)
jit(safe_log)(as_jax(2)); jit(safe_log)(as_jax(-2))
#> Array f32[] weak_type
#> [1] 0.6931472
#> Array f32[] weak_type
#> [1] 0
grad(safe_log)(as_jax(2))
#> Array f32[]
#> [1] 0.5
lax_switch(as_jax(1L), list(function(v) v + 1, function(v) v * 10, function(v) -v), as_jax(3))
#> Array f32[] weak_type
#> [1] 30
```

[`lax_select()`](https://adamrsophie.github.io/jaxR/reference/lax_select.md)
and
[`jnp_where()`](https://adamrsophie.github.io/jaxR/reference/jnp_where.md)
evaluate both sides and pick elementwise;
[`lax_cond()`](https://adamrsophie.github.io/jaxR/reference/lax_cond.md)
evaluates one branch.

### while_loop

``` r

collatz_steps <- function(n) {
  lax_while_loop(function(s) s[[1]] != 1L,
                 function(s) list(jnp_where(s[[1]] %% 2L == 0L, s[[1]] %/% 2L, 3L * s[[1]] + 1L), s[[2]] + 1L),
                 list(n, as_jax(0L)))[[2]]
}
jit(collatz_steps)(as_jax(27L))
#> Array i32[] weak_type
#> [1] 111
vmap(collatz_steps)(as_jax(c(6L, 7L, 27L)))
#> Array i32[3] weak_type
#> [1]   8  16 111
```

[`lax_while_loop()`](https://adamrsophie.github.io/jaxR/reference/lax_while_loop.md)
supports forward mode differentiation. Reverse mode needs a bounded
loop: use
[`lax_scan()`](https://adamrsophie.github.io/jaxR/reference/lax_scan.md)
or
[`lax_fori_loop()`](https://adamrsophie.github.io/jaxR/reference/lax_fori_loop.md)
with fixed bounds.

### fori_loop

``` r

lax_fori_loop(0L, 10L, function(i, acc) acc + i, as_jax(0L))
#> Array i32[]
#> [1] 45
```

### scan

`lax_scan(f, init, xs)` is a loop with a carried state that also stacks
one output per step: `f(carry, x)` returns `list(new_carry, y)`.

``` r

cumulative <- lax_scan(function(carry, x) { s <- carry + x; list(s, s) }, as_jax(0), as_jax(c(1, 2, 3, 4)))
cumulative
#> [[1]]
#> Array f32[]
#> [1] 10
#> 
#> [[2]]
#> Array f32[4]
#> [1]  1  3  6 10
```

A linear recurrence, differentiated through the loop:

``` r

ar1 <- function(phi, eps) lax_scan(function(h, e) { h <- phi * h + e; list(h, h) }, as_jax(0), eps)[[2]]
eps <- as_jax(c(1, 0.5, -0.2, 0.1))
ar1(as_jax(0.9), eps)
#> Array f32[4]
#> [1] 1.000 1.400 1.060 1.054
grad(function(phi) jnp_sum(ar1(phi, eps)))(as_jax(0.9))
#> Array f32[]
#> [1] 6.43
```

`reverse = TRUE` scans from the end, `length` gives the number of steps
when there is no `xs`, and `unroll` unrolls several steps per loop
iteration.

### map and associative_scan

``` r

lax_map(function(row) jnp_sum(row^2), jnp_reshape(jnp_arange(6, dtype = "float32"), c(3L, 2L)))
#> Array f32[3]
#> [1]  1 13 41
lax_associative_scan(lax_add, as_jax(c(1, 2, 3, 4)))        # a parallel prefix sum
#> Array f32[4]
#> [1]  1  3  6 10
```

## Implicit differentiation

[`lax_custom_root()`](https://adamrsophie.github.io/jaxR/reference/lax_custom_root.md)
differentiates the solution of `f(x) = 0` with the implicit function
theorem instead of through the iterations of the solver.

``` r

sqrt_newton <- function(a) {
  f <- function(x) x^2 - a
  solve <- function(f, x0) lax_fori_loop(0L, 20L, function(i, x) x - f(x) / (2 * x), x0)
  tangent_solve <- function(g, y) y / g(jnp_ones_like(y))
  lax_custom_root(f, jnp_ones_like(a), solve, tangent_solve)
}
sqrt_newton(as_jax(2))
#> Array f32[] weak_type
#> [1] 1.414214
grad(sqrt_newton)(as_jax(2))      # 1 / (2 sqrt(2))
#> Array f32[] weak_type
#> [1] 0.3535534
```

[`lax_custom_linear_solve()`](https://adamrsophie.github.io/jaxR/reference/lax_custom_linear_solve.md)
does the same for linear systems, and
[`lax_stop_gradient()`](https://adamrsophie.github.io/jaxR/reference/lax_stop_gradient.md)
blocks differentiation.

## Linear algebra primitives

`lax_linalg_*` are the factorization primitives used by `jnp_linalg_*`
and `scipy_linalg_*`.

``` r

S <- as_jax(matrix(c(4, 1, 1, 3), 2))
lax_linalg_cholesky(S)
#> Array f32[2,2]
#>      [,1]     [,2]
#> [1,]  2.0 0.000000
#> [2,]  0.5 1.658312
lu <- lax_linalg_lu(S); lu[[1]]; lu[[2]]
#> Array f32[2,2]
#>      [,1] [,2]
#> [1,] 4.00 1.00
#> [2,] 0.25 2.75
#> Array i32[2]
#> [1] 0 1
lax_linalg_qr(S, full_matrices = FALSE)[[2]]
#> Array f32[2,2]
#>           [,1]      [,2]
#> [1,] -4.123106 -1.697749
#> [2,]  0.000000  2.667892
lax_linalg_eigh(S)[[2]]
#> Array f32[2]
#> [1] 2.381966 4.618034
lax_linalg_svd(S, compute_uv = FALSE)
#> Array f32[2]
#> [1] 4.618034 2.381966
lax_linalg_triangular_solve(lax_linalg_cholesky(S), as_jax(matrix(c(1, 2), 2)), left_side = TRUE, lower = TRUE)
#> Array f32[2,1]
#>         [,1]
#> [1,] 0.50000
#> [2,] 1.05529
lax_linalg_tridiagonal_solve(as_jax(c(0, 1, 1)), as_jax(c(4, 4, 4)), as_jax(c(1, 1, 0)), as_jax(matrix(c(1, 2, 3), 3)))
#> Array f32[3,1]
#>           [,1]
#> [1,] 0.1785714
#> [2,] 0.2857143
#> [3,] 0.6785714
```

## Named axis collectives

Inside `vmap(..., axis_name = "i")` the parallel operators reduce or
permute over the named axis.

``` r

vmap(function(v) lax_pmean(v, "i"), axis_name = "i")(as_jax(c(1, 2, 6)))
#> Array f32[3]
#> [1] 3 3 3
vmap(function(v) v - lax_pmax(v, "i"), axis_name = "i")(as_jax(c(1, 2, 6)))
#> Array f32[3]
#> [1] -5 -4  0
vmap(function(v) lax_axis_index("i"), axis_name = "i")(as_jax(c(1, 2, 6)))
#> Array i32[3]
#> [1] 0 1 2
```

## Primitives are objects

Every primitive is exported as an object named `lax_<name>_p`. They
appear in the jaxprs printed by
[`make_jaxpr()`](https://adamrsophie.github.io/jaxR/reference/make_jaxpr.md).

``` r

make_jaxpr(function(x) lax_scan(function(c, v) list(c + v, c * v), as_jax(0), x))(as_jax(c(1, 2, 3)))
#> { lambda ; a:f32[3]. let
#>     b:f32[] c:f32[3] = scan[jaxpr={...} num_consts=0 num_carry=1 length=3 reverse=FALSE unroll=1] 0.0 a
#>   in (b, c) }
```
