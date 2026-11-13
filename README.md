# jaxR

<!-- badges: start -->
[![R-CMD-check](https://github.com/adamrsophie/jaxR/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/adamrsophie/jaxR/actions/workflows/R-CMD-check.yaml)
[![test-suite](https://github.com/adamrsophie/jaxR/actions/workflows/test-suite.yaml/badge.svg)](https://github.com/adamrsophie/jaxR/actions/workflows/test-suite.yaml)
<!-- badges: end -->

jaxR is an R mirror of [JAX](https://github.com/jax-ml/jax): composable transformations
of array programs (`grad`, `vmap`, `jit`, `jvp`, `vjp`, `linearize`, `jacfwd`, `jacrev`,
`hessian`, `checkpoint`), a NumPy compatible array API, `lax` primitives with
structured control flow, counter based random numbers whose bits agree with JAX exactly,
and SciPy style special functions, linear algebra and FFTs.

Like JAX, jaxR traces R functions to an intermediate representation (jaxpr), lowers it to
StableHLO and compiles it with XLA. The XLA runtime is a PJRT plugin loaded at run time,
in the way rstan embeds Stan or torch uses libtorch; nothing is translated from Python at
run time and no Python installation is needed.

## Installation

```r
# install.packages("remotes")
remotes::install_github("adamrsophie/jaxR")
jaxR::install_pjrt()   # downloads the XLA CPU runtime (once)
```

Prebuilt runtimes are available for Linux and macOS, on x86-64 and arm64. Set
`JAXR_PJRT_PLUGIN` to use another PJRT plugin (for example a CUDA build). There is no
prebuilt runtime for Windows: the package installs there, and it needs
`JAXR_PJRT_PLUGIN` to name a PJRT plugin built for Windows.

## A quick tour

```r
library(jaxR)

f <- function(x) sum(sin(x) * x)
x <- as_jax(c(1, 2, 3))

grad(f)(x)                      # reverse mode
jit(grad(f))(x)                 # compiled by XLA
vmap(grad(f))(jnp_stack(list(x, 2 * x)))
hessian(f)(x)

# Structured control flow is differentiable and compiles
g <- function(v) lax_scan(function(c, xi) list(c * 0.5 + xi, c * xi), 0, v)[[2]]
grad(function(v) sum(g(v)))(x)

# Random numbers: same keys, same bits as JAX
key <- random_key(0L)
k <- random_split(key)
random_normal(k[1], c(3L, 2L))
```

## Conventions

* Names follow JAX modules: `jax.numpy.sum` is `jnp_sum`, `jax.lax.scan` is `lax_scan`,
  `jax.random.normal` is `random_normal`, `jax.scipy.special.gammaln` is
  `scipy_special_gammaln`, `jax.nn.relu` is `nn_relu`. Transformations keep their JAX
  names (`grad`, `vmap`, `jit`, ...). No jaxR function masks a base R function.
* Axes, indices, `argnums` and `in_axes` are 0-based, as in JAX: `x[0]` is the first
  element and `jnp_sum(x, axis = 0L)` reduces the first dimension. Python style slices are
  written `sl(start, stop, step)`.
* As in JAX, arrays are 32-bit by default. Use `enable_x64()` (or set `JAX_ENABLE_X64=1`)
  for 64-bit floats and integers.
* R scalars are weakly typed, so `x * 2` keeps the dtype of `x`.
* Arrays live on the device; `as.array(x)` returns R data. Use `as.logical(x)` in `if`
  conditions.

## Symbolic computation

jaxR works with [symcompR](https://github.com/adamrsophie/symcompR):
`jax_lambdify(args, expr)` turns a symbolic expression into a jaxR function that can be
differentiated, vectorized and compiled, and `grad()` / `hessian()` forward symbolic
expressions to symcompR.

## Status

jaxR follows JAX 0.12.0.1. The transformations, `jax.numpy`, `jax.lax`, `jax.random`,
`jax.nn`, `jax.scipy`, `jax.image`, `jax.tree_util`, `jax.experimental.sparse` and the
example libraries (`stax`, `optimizers`) are ported and tested against values produced by
JAX itself. Features that need several devices (`pmap` across devices, sharding,
distributed arrays) are not ported; named axis collectives (`lax_psum`, `lax_all_gather`,
...) work under `vmap(..., axis_name = )`. The vignettes describe each area and list the
differences from JAX.

## License

GPL (>= 3). jaxR ports algorithms and rules from JAX (Apache License 2.0) and vendors
OpenXLA headers (Apache License 2.0); see `inst/COPYRIGHTS`.
