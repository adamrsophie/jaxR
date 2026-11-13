# Getting started with jaxR

jaxR is an R mirror of [JAX](https://github.com/jax-ml/jax). It gives R
the same three ideas that define JAX:

1.  an array library with the NumPy API (`jnp_*`), running on the XLA
    compiler;
2.  function transformations that compose: automatic differentiation
    (`grad`, `jvp`, `vjp`), vectorization (`vmap`) and just in time
    compilation (`jit`);
3.  explicit, reproducible random numbers (`random_*`).

jaxR does not call Python. R functions are traced to an intermediate
representation (a *jaxpr*), lowered to StableHLO and compiled by XLA,
which is loaded as a shared library (a PJRT plugin) in the same way that
rstan embeds Stan.

This vignette covers installation, arrays, dtypes, indexing and the
conventions that differ from base R. The other vignettes go through each
part of the package:

| Vignette | Topic |
|----|----|
| `transformations` | `grad`, `jvp`, `vjp`, Jacobians, Hessians, `vmap`, `jit`, custom derivatives |
| `numpy` | the `jnp_*` array API, linear algebra, FFT, indexed updates |
| `lax` | `lax_*` primitives and structured control flow |
| `random` | PRNG keys and samplers |
| `scipy` | `scipy_*` special functions, statistics, linear algebra, optimization, signal processing |
| `neural-networks` | `nn_*`, initializers, `stax_*` layers, `optimizers_*`, `image_*` |
| `pytrees` | nested containers of arrays and `tree_*` utilities |
| `sparse` | sparse arrays (`experimental_sparse_*`) |
| `advanced` | callbacks, debugging, configuration, mutable references, errors, monitoring |
| `symbolic` | working with symcompR |
| `differences` | differences from JAX and from base R, and what is not ported |

## Installation

``` r

install.packages("jaxR")
jaxR::install_pjrt()   # downloads the XLA CPU runtime, once
```

[`install_pjrt()`](https://adamrsophie.github.io/jaxR/reference/install_pjrt.md)
stores the runtime in the user data directory of the package. To use
another PJRT plugin, set the environment variable `JAXR_PJRT_PLUGIN` to
its path before loading the package. The runtime is loaded on first use;
[`jax_backend_init()`](https://adamrsophie.github.io/jaxR/reference/jax_backend_init.md)
loads it explicitly and
[`default_backend()`](https://adamrsophie.github.io/jaxR/reference/default_backend.md)
names the platform.

``` r

library(jaxR)
```

## Arrays

[`as_jax()`](https://adamrsophie.github.io/jaxR/reference/as_jax.md) and
[`jnp_array()`](https://adamrsophie.github.io/jaxR/reference/jnp_asarray.md)
move R data to the device. An R vector becomes a rank 1 array and an R
matrix or array keeps its dimensions.

``` r

x <- as_jax(c(1, 2, 3))
x
#> Array f32[3]
#> [1] 1 2 3
m <- as_jax(matrix(1:6, nrow = 2))
m
#> Array i32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    1    3    5
#> [2,]    2    4    6
dim(m)
#> [1] 2 3
dtype(m)
#> [1] "int32"
```

Arrays are immutable values.
[`as.array()`](https://rdrr.io/r/base/array.html) (or
[`as.vector()`](https://rdrr.io/r/base/vector.html),
[`as.numeric()`](https://rdrr.io/r/base/numeric.html)) brings the data
back to R.

``` r

as.array(m)
#>      [,1] [,2] [,3]
#> [1,]    1    3    5
#> [2,]    2    4    6
as.numeric(x * 2)
#> [1] 2 4 6
```

The usual R operators work on arrays and follow the NumPy broadcasting
rules.

``` r

x + 1
#> Array f32[3]
#> [1] 2 3 4
x * x
#> Array f32[3]
#> [1] 1 4 9
m * as_jax(c(10, 100, 1000))     # shape (2, 3) with shape (3)
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]   10  300 5000
#> [2,]   20  400 6000
x %*% x
#> Array f32[]
#> [1] 14
-x > -2
#> Array bool[3]
#> [1]  TRUE FALSE FALSE
```

Most math group generics are also defined, so `sin(x)`, `exp(x)`,
`sum(x)` work, and they are equivalent to the explicit `jnp_sin(x)`,
`jnp_exp(x)`, `jnp_sum(x)`.

``` r

sin(x)
#> Array f32[3]
#> [1] 0.8414710 0.9092974 0.1411200
sum(x)
#> Array f32[]
#> [1] 6
jnp_sum(x)
#> Array f32[]
#> [1] 6
```

### Creating arrays

``` r

jnp_zeros(c(2L, 3L))
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    0    0    0
#> [2,]    0    0    0
jnp_ones(3L, dtype = "int32")
#> Array i32[3]
#> [1] 1 1 1
jnp_full(c(2L, 2L), 7)
#> Array f32[2,2] weak_type
#>      [,1] [,2]
#> [1,]    7    7
#> [2,]    7    7
jnp_arange(5)
#> Array i32[5]
#> [1] 0 1 2 3 4
jnp_arange(1, 2, 0.25)
#> Array f32[4]
#> [1] 1.00 1.25 1.50 1.75
jnp_linspace(0, 1, 5L)
#> Array f32[5]
#> [1] 0.00 0.25 0.50 0.75 1.00
jnp_eye(3L)
#> Array f32[3,3]
#>      [,1] [,2] [,3]
#> [1,]    1    0    0
#> [2,]    0    1    0
#> [3,]    0    0    1
```

## Data types

As in JAX, the default floating type is `float32` and the default
integer type is `int32`.

``` r

dtype(as_jax(1.5))
#> [1] "float32"
dtype(as_jax(1L))
#> [1] "int32"
dtype(as_jax(TRUE))
#> [1] "bool"
dtype(as_jax(1 + 2i))
#> [1] "complex64"
```

Pass `dtype` for another type, or convert with
[`jnp_astype()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md).

``` r

as_jax(c(1, 2, 3), "int8")
#> Array i8[3]
#> [1] 1 2 3
jnp_astype(x, "float16")
#> Array f16[3]
#> [1] 1 2 3
```

64 bit types need
[`enable_x64()`](https://adamrsophie.github.io/jaxR/reference/enable_x64.md)
(or the environment variable `JAX_ENABLE_X64=1`).

``` r

enable_x64(TRUE)
dtype(as_jax(1.5))
#> [1] "float64"
as_jax(2)^62
#> Array f64[] weak_type
#> [1] 4.611686e+18
enable_x64(FALSE)
```

[`finfo()`](https://adamrsophie.github.io/jaxR/reference/finfo.md) and
[`iinfo()`](https://adamrsophie.github.io/jaxR/reference/iinfo.md)
describe a type,
[`result_type()`](https://adamrsophie.github.io/jaxR/reference/result_type.md)
and
[`jnp_promote_types()`](https://adamrsophie.github.io/jaxR/reference/jnp_dtype_functions.md)
give the result of mixing types, and
[`jnp_can_cast()`](https://adamrsophie.github.io/jaxR/reference/jnp_dtype_functions.md)
and
[`issubdtype()`](https://adamrsophie.github.io/jaxR/reference/issubdtype.md)
test relations.

``` r

finfo("float32")$eps
#> [1] 1.192093e-07
iinfo("int16")$max
#> [1] 32767
jnp_promote_types("int32", "float32")
#> [1] "float32"
result_type(as_jax(1L, "int8"), 2.5)
#> [1] "float32"
```

### Weak types

An R number that is not an array is *weakly typed*, like a Python scalar
in JAX: it adopts the type of the array it meets instead of promoting
it.

``` r

h <- as_jax(c(1, 2), "float16")
dtype(h * 2)        # 2 is weak: the result stays float16
#> [1] "float16"
dtype(h * as_jax(2, "float32"))
#> [1] "float32"
```

## Indexing is 0-based

jaxR keeps the JAX convention for every index and axis: **the first
element is `x[0]`** and the first dimension is `axis = 0`. This keeps
code and documentation portable between JAX and jaxR. Negative indices
count from the end.

``` r

v <- jnp_arange(10)
v[0]
#> Array i32[]
#> [1] 0
v[-1]
#> Array i32[]
#> [1] 9
```

`sl(start, stop, step)` is a Python slice (`start:stop:step`, the end is
exclusive). A missing index is a full slice, `newaxis` adds a dimension
and `ellipsis` stands for all remaining dimensions.

``` r

v[sl(2, 5)]
#> Array i32[3]
#> [1] 2 3 4
v[sl(step = 2)]
#> Array i32[5]
#> [1] 0 2 4 6 8
v[sl(step = -1)]
#> Array i32[10]
#>  [1] 9 8 7 6 5 4 3 2 1 0
a <- jnp_reshape(jnp_arange(12), c(3L, 4L))
a
#> Array i32[3,4]
#>      [,1] [,2] [,3] [,4]
#> [1,]    0    1    2    3
#> [2,]    4    5    6    7
#> [3,]    8    9   10   11
a[1, ]
#> Array i32[4]
#> [1] 4 5 6 7
a[, sl(1, 3)]
#> Array i32[3,2]
#>      [,1] [,2]
#> [1,]    1    2
#> [2,]    5    6
#> [3,]    9   10
a[newaxis, 0, ]
#> Array i32[1,4]
#>      [,1] [,2] [,3] [,4]
#> [1,]    0    1    2    3
a[ellipsis, -1]
#> Array i32[3]
#> [1]  3  7 11
```

Integer arrays and logical masks index like in NumPy.

``` r

v[as_jax(c(0L, 3L, 3L))]
#> Array i32[3]
#> [1] 0 3 3
v[v > 6]
#> Array i32[3]
#> [1] 7 8 9
```

### Updates

Arrays cannot be modified in place. The `at_*` functions return an
updated copy, like `x.at[idx].set(y)` in JAX; under `jit` the copy is
optimized away.

``` r

at_set(v, 0, value = 100L)
#> Array i32[10]
#>  [1] 100   1   2   3   4   5   6   7   8   9
at_add(v, sl(0, 3), value = 10L)
#> Array i32[10]
#>  [1] 10 11 12  3  4  5  6  7  8  9
at_multiply(a, 1, sl(), value = 2L)
#> Array i32[3,4]
#>      [,1] [,2] [,3] [,4]
#> [1,]    0    1    2    3
#> [2,]    8   10   12   14
#> [3,]    8    9   10   11
at_max(v, as_jax(c(0L, 1L)), value = 5L)
#> Array i32[10]
#>  [1] 5 5 2 3 4 5 6 7 8 9
```

Assignment syntax is also provided for convenience; it calls
[`at_set()`](https://adamrsophie.github.io/jaxR/reference/at.md) and
rebinds the variable.

``` r

w <- v
w[0] <- 42L
w
#> Array i32[10]
#>  [1] 42  1  2  3  4  5  6  7  8  9
v[0]
#> Array i32[]
#> [1] 0
```

## Axes

Reductions take 0-based axes.

``` r

jnp_sum(a, axis = 0L)
#> Array i32[4]
#> [1] 12 15 18 21
jnp_mean(a, axis = 1L)
#> Array f32[3]
#> [1] 1.5 5.5 9.5
jnp_max(a, axis = c(0L, 1L))
#> Array i32[]
#> [1] 11
jnp_argmax(a, axis = 1L)
#> Array i32[3]
#> [1] 3 3 3
```

## A first transformation

Everything above also works inside the transformations.
[`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md)
differentiates a function,
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) compiles
it and [`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md)
maps it over a new axis.

``` r

f <- function(x) jnp_sum(jnp_sin(x) * x)
grad(f)(x)
#> Array f32[3]
#> [1]  1.38177323  0.07700372 -2.82885742
jit(grad(f))(x)
#> Array f32[3]
#> [1]  1.38177323  0.07700372 -2.82885742
vmap(grad(f))(jnp_stack(list(x, 2 * x)))
#> Array f32[2,3]
#>            [,1]        [,2]      [,3]
#> [1,] 1.38177323  0.07700372 -2.828857
#> [2,] 0.07700372 -3.37137699  5.481606
```

See
[`vignette("transformations", package = "jaxR")`](https://adamrsophie.github.io/jaxR/articles/transformations.md)
for the details.

## Naming

Each jaxR function is named after its JAX module:

| JAX                               | jaxR                       |
|-----------------------------------|----------------------------|
| `jax.numpy.sum`                   | `jnp_sum`                  |
| `jax.numpy.linalg.solve`          | `jnp_linalg_solve`         |
| `jax.lax.scan`                    | `lax_scan`                 |
| `jax.random.normal`               | `random_normal`            |
| `jax.nn.relu`                     | `nn_relu`                  |
| `jax.scipy.special.gammaln`       | `scipy_special_gammaln`    |
| `jax.scipy.stats.norm.logpdf`     | `scipy_stats_norm_logpdf`  |
| `jax.tree.map`                    | `tree_map`                 |
| `jax.experimental.sparse.BCOO`    | `experimental_sparse_BCOO` |
| `jax.grad`, `jax.vmap`, `jax.jit` | `grad`, `vmap`, `jit`      |

No jaxR function masks a function of base R or of the packages attached
by default.
