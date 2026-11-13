# Function transformations: grad, vmap and jit

A transformation takes an R function and returns a new function.
Transformations compose freely: the gradient of a compiled function, a
compiled gradient, a vectorized Hessian. This vignette covers all of
them.

``` r

library(jaxR)
```

## How tracing works

A transformation calls your function with *tracers*: stand-ins that
record the operations applied to them. The function must therefore be
written with jaxR operations (`jnp_*`, `lax_*`, the R operators) and
must be *pure*: its result depends only on its arguments, and it has no
side effects.

[`make_jaxpr()`](https://adamrsophie.github.io/jaxR/reference/make_jaxpr.md)
shows the recorded program.

``` r

f <- function(x) jnp_sum(jnp_sin(x) * x)
make_jaxpr(f)(as_jax(c(1, 2, 3)))
#> { lambda ; a:f32[3]. let
#>     b:f32[3] = sin a
#>     c:f32[3] = mul b a
#>     d:f32[] = reduce_sum[axes=0] c
#>   in (d) }
```

Ordinary R code that does not depend on array *values* runs once at
trace time: loops over a fixed range, `if` on a shape or on a plain R
value, calls to other R functions.

``` r

poly <- function(x, coefs) {
  out <- 0
  for (cf in coefs) out <- out * x + cf     # an R loop, unrolled in the trace
  out
}
make_jaxpr(function(x) poly(x, c(1, -2, 3)))(as_jax(2))
#> { lambda ; a:f32[]. let
#>     b:f32[] = mul 0.0 a
#>     c:f32[] = add b 1.0
#>     d:f32[] = mul c a
#>     e:f32[] = add d -2.0
#>     f:f32[] = mul e a
#>     g:f32[] = add f 3.0
#>   in (g) }
```

R control flow on an array value does not work under tracing, because
the tracer has no value. Use
[`jnp_where()`](https://adamrsophie.github.io/jaxR/reference/jnp_where.md)
or the structured control flow of `lax` (`lax_cond`, `lax_while_loop`,
`lax_scan`; see
[`vignette("lax", package = "jaxR")`](https://adamrsophie.github.io/jaxR/articles/lax.md)).

``` r

relu_bad <- function(x) if (as.logical(x > 0)) x else 0 * x
tryCatch(grad(relu_bad)(as_jax(1)), error = function(e) class(e)[1])
#> Array f32[] weak_type
#> [1] 1
relu <- function(x) jnp_where(x > 0, x, 0)
grad(relu)(as_jax(1))
#> Array f32[] weak_type
#> [1] 1
```

## Differentiation

### grad and value_and_grad

`grad(f)` returns the gradient of a scalar valued function with respect
to its first argument.

``` r

x <- as_jax(c(1, 2, 3))
grad(f)(x)
#> Array f32[3]
#> [1]  1.38177323  0.07700372 -2.82885742
```

`argnums` selects other arguments (0-based). With several, the result is
a list.

``` r

loss <- function(w, b, x, y) jnp_mean((x %*% w + b - y)^2)
w <- as_jax(c(0.5, -0.5)); b <- as_jax(0.1)
X <- as_jax(matrix(c(1, 2, 3, 4, 5, 6), 3, byrow = TRUE)); y <- as_jax(c(1, 0, 1))
grad(loss)(w, b, X, y)
#> Array f32[2]
#> [1] -6.400000 -8.533334
grad(loss, argnums = c(0L, 1L))(w, b, X, y)
#> [[1]]
#> Array f32[2]
#> [1] -6.400000 -8.533334
#> 
#> [[2]]
#> Array f32[] weak_type
#> [1] -2.133333
```

Arguments can be nested lists of arrays (*pytrees*); the gradient has
the same structure.

``` r

loss2 <- function(params, x, y) jnp_mean((x %*% params$w + params$b - y)^2)
grad(loss2)(list(w = w, b = b), X, y)
#> $w
#> Array f32[2]
#> [1] -6.400000 -8.533334
#> 
#> $b
#> Array f32[] weak_type
#> [1] -2.133333
```

[`value_and_grad()`](https://adamrsophie.github.io/jaxR/reference/value_and_grad.md)
returns the value too, and `has_aux = TRUE` lets the function return
extra data that is not differentiated.

``` r

value_and_grad(loss)(w, b, X, y)
#> [[1]]
#> Array f32[]
#> [1] 1.36
#> 
#> [[2]]
#> Array f32[2]
#> [1] -6.400000 -8.533334
with_aux <- function(w) {
  r <- X %*% w - y
  list(jnp_mean(r^2), list(residual = r))
}
grad(with_aux, has_aux = TRUE)(w)
#> [[1]]
#> Array f32[2]
#> [1] -7.000000 -9.333334
#> 
#> [[2]]
#> [[2]]$residual
#> Array f32[3]
#> [1] -1.5 -0.5 -1.5
```

Gradients nest, so higher derivatives need nothing new.

``` r

g3 <- grad(grad(grad(function(x) x^4)))
g3(as_jax(2))            # 24 x
#> Array f32[] weak_type
#> [1] 48
```

### Forward mode: jvp

`jvp(f, primals, tangents)` evaluates the function and its directional
derivative (Jacobian times vector) in one pass.

``` r

jvp(function(x) jnp_sin(x) * x, list(x), list(jnp_ones_like(x)))
#> [[1]]
#> Array f32[3]
#> [1] 0.841471 1.818595 0.423360
#> 
#> [[2]]
#> Array f32[3]
#> [1]  1.38177323  0.07700372 -2.82885742
```

### Reverse mode: vjp

`vjp(f, ...)` returns the value and a function that maps an output
cotangent to the input cotangents (vector times Jacobian).

``` r

r <- vjp(function(x) jnp_sin(x) * x, x)
r[[1]]
#> Array f32[3]
#> [1] 0.841471 1.818595 0.423360
r[[2]](as_jax(c(1, 0, 0)))
#> [[1]]
#> Array f32[3]
#> [1] 1.381773 0.000000 0.000000
```

Two options of
[`vjp()`](https://adamrsophie.github.io/jaxR/reference/vjp.md) control
what is computed and kept. `in_nzs` marks the inputs that may have a
nonzero tangent: the others are treated as constants, and
`vjp_fun$out_nzs` then tells which outputs still depend on a marked
input. `saveable_args` marks the arguments that may be kept for the
backward pass. An argument array that the backward pass needs as it is,
but that is marked `FALSE`, is not kept: `vjp_fun$args_res` shows
`NotSaveable()` in its place, and the caller puts the value back before
calling the function (useful when the argument lives elsewhere anyway,
so that it is not held twice). Arguments the backward pass does not need
show `NotNeeded()`.

``` r

a <- as_jax(2, "float32")
b <- as_jax(3, "float32")
r <- vjp(function(x, y) list(x * y, y), a, b, in_nzs = list(TRUE, FALSE))
r[[2]]$out_nzs
#> [[1]]
#> [1] TRUE
#> 
#> [[2]]
#> [1] FALSE
r <- vjp(function(x, y) x * y, a, b, saveable_args = list(TRUE, FALSE))
pullback <- r[[2]]
pullback$args_res
#> [[1]]
#> Array f32[]
#> [1] 2
#> 
#> [[2]]
#> NotSaveable()
pullback$args_res[[2]] <- b
pullback(as_jax(1, "float32"))
#> [[1]]
#> Array f32[]
#> [1] 3
#> 
#> [[2]]
#> Array f32[]
#> [1] 2
```

### linearize

[`linearize()`](https://adamrsophie.github.io/jaxR/reference/linearize.md)
evaluates once and returns the linear map of the JVP, which can then be
applied to many tangents without recomputing the function.
[`linear_transpose()`](https://adamrsophie.github.io/jaxR/reference/linear_transpose.md)
transposes a linear function.

``` r

lin <- linearize(function(x) jnp_sin(x) * x, x)
lin[[2]](as_jax(c(1, 0, 0)))
#> Array f32[3]
#> [1] 1.381773 0.000000 0.000000
lt <- linear_transpose(function(v) 3 * v, x)
lt(as_jax(c(1, 1, 1)))
#> [[1]]
#> Array f32[3]
#> [1] 3 3 3
```

### Jacobians and Hessians

[`jacfwd()`](https://adamrsophie.github.io/jaxR/reference/jacfwd.md)
builds the Jacobian with forward mode (one pass per input),
[`jacrev()`](https://adamrsophie.github.io/jaxR/reference/jacrev.md)
with reverse mode (one pass per output);
[`jacobian()`](https://adamrsophie.github.io/jaxR/reference/jacrev.md)
is [`jacrev()`](https://adamrsophie.github.io/jaxR/reference/jacrev.md).
[`hessian()`](https://adamrsophie.github.io/jaxR/reference/hessian.md)
is forward over reverse.

``` r

g <- function(x) jnp_stack(list(x[0] * x[1], jnp_sin(x[2]), jnp_sum(x^2)))
jacfwd(g)(x)
#> Array f32[3,3]
#>      [,1] [,2]       [,3]
#> [1,]    2    1  0.0000000
#> [2,]    0    0 -0.9899925
#> [3,]    2    4  6.0000000
jacrev(g)(x)
#> Array f32[3,3]
#>      [,1] [,2]       [,3]
#> [1,]    2    1  0.0000000
#> [2,]    0    0 -0.9899925
#> [3,]    2    4  6.0000000
hessian(f)(x)
#> Array f32[3,3]
#>           [,1]      [,2]      [,3]
#> [1,] 0.2391336  0.000000  0.000000
#> [2,] 0.0000000 -2.650888  0.000000
#> [3,] 0.0000000  0.000000 -2.403345
```

### Stopping gradients

[`stop_gradient()`](https://adamrsophie.github.io/jaxR/reference/lax_stop_gradient.md)
treats a value as a constant.

``` r

grad(function(x) jnp_sum(x * stop_gradient(x)))(x)   # only one factor is differentiated
#> Array f32[3]
#> [1] 1 2 3
```

### Integer and boolean inputs

Integers have no tangent space. The cotangent of an integer input has
dtype `float0`, and
[`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md) asks
for `allow_int = TRUE` to differentiate with respect to one.

``` r

r <- vjp(function(x, n) x * n, as_jax(c(1, 2)), as_jax(3L))
r[[2]](as_jax(c(1, 1)))
#> [[1]]
#> Array f32[2]
#> [1] 3 3
#> 
#> [[2]]
#> Array float0[] weak_type
#> [1] FALSE
```

### Custom derivatives

[`custom_jvp()`](https://adamrsophie.github.io/jaxR/reference/custom_jvp.md)
and
[`custom_vjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md)
replace the derivative of a function, for numerical stability or to
differentiate through an iterative solver.

``` r

log1pexp <- custom_jvp(
  function(x) jnp_log1p(jnp_exp(x)),
  jvp = function(primals, tangents) {
    x <- primals[[1]]
    list(log1pexp(x), tangents[[1]] * (1 - 1 / (1 + jnp_exp(x))))
  })
grad(log1pexp)(as_jax(100))     # the naive derivative overflows to NaN
#> Array f32[] weak_type
#> [1] 1
grad(function(x) jnp_log1p(jnp_exp(x)))(as_jax(100))
#> Array f32[] weak_type
#> [1] NaN
```

``` r

clip_gradient <- custom_vjp(
  function(x) x,
  fwd = function(x) list(x, NULL),
  bwd = function(res, g) list(jnp_clip(g, -1, 1)))
grad(function(x) jnp_sum(clip_gradient(x) * 10))(x)
#> Array f32[3]
#> [1] 1 1 1
```

[`custom_gradient()`](https://adamrsophie.github.io/jaxR/reference/custom_gradient.md)
is the TensorFlow style shortcut and
[`lax_custom_root()`](https://adamrsophie.github.io/jaxR/reference/lax_custom_root.md),
[`lax_custom_linear_solve()`](https://adamrsophie.github.io/jaxR/reference/lax_custom_linear_solve.md)
define implicit derivatives of root finders and linear solvers.

#### Logs and gradient accumulators

A backward rule can send values out of the backward pass. With
[`custom_vjp_defvjp_with_logs()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp_defvjp_with_logs.md)
(or `custom_gradient(with_logs = TRUE)`) the rule returns
`list(cotangents, logs)`, where `logs` is a named list; the `with_logs`
method of a VJP function returns them next to the cotangents, and a
plain call drops them. Log names must be unique within one backward
pass.

``` r

scaled <- custom_gradient(function(x) {
  list(3 * x, function(g) list(3 * g, list(incoming = g)))
}, with_logs = TRUE)
pullback <- vjp(scaled, as_jax(2))[[2]]
pullback$with_logs(as_jax(5))
#> [[1]]
#> [[1]][[1]]
#> Array f32[] weak_type
#> [1] 15
#> 
#> 
#> [[2]]
#> [[2]]$incoming
#> Array f32[] weak_type
#> [1] 5
grad(scaled)(as_jax(2))
#> Array f32[] weak_type
#> [1] 3
```

A rule defined with
[`custom_vjp_defvjp_with_accums()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp_defvjp_with_logs.md)
or `custom_gradient(with_accums = )` does not return cotangents: it
receives one gradient accumulator per argument and pushes contributions
into it. A `ValAccum` sums what `acc$accum()` receives, a `RefAccum`
exposes a ref, `acc$ref`, for indexed updates, and an argument that is
not differentiated gets a `NullAccum`, which ignores what it receives.
This suits a rule whose cotangent is a sparse update of a large array.

``` r

pick <- custom_gradient(function(x, i) {
  list(x[i], function(g, x_acc, i_acc) {
    if (x_acc$kind == "RefAccum") x_acc$ref[i] <- x_acc$ref[i] + g
    NULL
  })
}, with_accums = list(ad_RefAccum, ad_ValAccum))
grad(function(x) pick(x, 2L) * 4)(jnp_arange(5, dtype = "float32"))
#> Array f32[5]
#> [1] 0 0 4 0 0
```

The `with_refs` method of a VJP function chooses, per argument, whether
the gradient is returned
([`ad_GradValue()`](https://adamrsophie.github.io/jaxR/reference/ad_GradValue.md)),
added into a ref (the ref itself) or not computed
([`ad_DontWant()`](https://adamrsophie.github.io/jaxR/reference/ad_GradValue.md));
the result holds `GradRef()` and `DidntWant()` markers in their places.

``` r

pullback <- vjp(function(x, y) x * y, as_jax(3), as_jax(4))[[2]]
acc <- new_ref(as_jax(10, "float32"))
pullback$with_refs(ad_DontWant(), acc)(as_jax(1))
#> [[1]]
#> DidntWant()
#> 
#> [[2]]
#> GradRef()
acc[]
#> Array f32[]
#> [1] 13
```

### Checkpointing

Reverse mode stores the intermediate values of the forward pass.
[`checkpoint()`](https://adamrsophie.github.io/jaxR/reference/checkpoint.md)
(alias
[`remat()`](https://adamrsophie.github.io/jaxR/reference/checkpoint.md))
trades memory for time: the wrapped function is recomputed during the
backward pass instead of having its intermediates stored.
[`print_saved_residuals()`](https://adamrsophie.github.io/jaxR/reference/saved_residuals.md)
shows what a function stores.

``` r

layer <- function(x) jnp_sin(jnp_sin(x))
deep <- function(x) jnp_sum(checkpoint(layer)(checkpoint(layer)(x)))
grad(deep)(x)
#> Array f32[3]
#> [1]  0.2059352 -0.1366153 -0.9609511
```

### Taylor mode: higher derivatives in one pass

[`experimental_jet_jet()`](https://adamrsophie.github.io/jaxR/reference/experimental_jet_jet.md)
is the port of `jax.experimental.jet`.
[`jvp()`](https://adamrsophie.github.io/jaxR/reference/jvp.md) pushes a
value and one tangent through a function;
[`experimental_jet_jet()`](https://adamrsophie.github.io/jaxR/reference/experimental_jet_jet.md)
pushes a value and a whole truncated Taylor series, so the derivatives
of order 1 to K along a direction come out of one pass instead of K
nested calls of
[`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md) or
[`jvp()`](https://adamrsophie.github.io/jaxR/reference/jvp.md).

The arguments are the function, the list of its arguments (the point of
expansion) and, for each argument, the list of its higher order terms.
With the series `(1, 0, 0)` for a scalar argument the output terms are
the first three derivatives:

``` r

f <- function(x) jnp_sin(x) * jnp_exp(x)
out <- experimental_jet_jet(f, list(0.4), list(list(1, 0, 0)))
out[[1]]              # f(0.4)
#> Array f32[] weak_type
#> [1] 0.5809439
out[[2]]              # f'(0.4), f''(0.4), f'''(0.4)
#> [[1]]
#> Array f32[] weak_type
#> [1] 1.955005
#> 
#> [[2]]
#> Array f32[] weak_type
#> [1] 2.748123
#> 
#> [[3]]
#> Array f32[] weak_type
#> [1] 1.586235
grad(grad(grad(f)))(0.4)
#> Array f32[] weak_type
#> [1] 1.586235
```

A series that is not `(1, 0, ...)` describes a curve through the point,
and the result is the Taylor expansion of the function along that curve
(Faà di Bruno’s formula). With `factorial_scaled = FALSE` the terms are
Taylor coefficients, the derivatives divided by the factorial of their
order.
[`experimental_jet_jet()`](https://adamrsophie.github.io/jaxR/reference/experimental_jet_jet.md)
composes with
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) and
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md). As in
JAX there are rules for arithmetic, the elementary functions, linear
structural operations, reductions, products, convolutions and gathers;
control flow primitives have none, and custom derivative rules are
ignored (the function itself is differentiated).

## Vectorization: vmap

`vmap(f)` maps `f` over a leading axis of its arguments without a loop:
the batch axis is pushed through each primitive.

``` r

dot <- function(a, b) jnp_sum(a * b)
A <- as_jax(matrix(1:6, 2, byrow = TRUE), "float32")
B <- as_jax(matrix(6:1, 2, byrow = TRUE), "float32")
vmap(dot)(A, B)
#> Array f32[2]
#> [1] 28 28
```

`in_axes` gives the mapped axis of each argument (`NULL` for an argument
that is not mapped) and `out_axes` places the batch axis of the result.

``` r

vmap(dot, in_axes = list(0L, NULL))(A, as_jax(c(1, 1, 1)))
#> Array f32[2]
#> [1]  6 15
vmap(dot, in_axes = list(1L, 1L))(A, B)
#> Array f32[3]
#> [1] 18 20 18
vmap(function(a) a * 2, out_axes = 1L)(A)
#> Array f32[3,2]
#>      [,1] [,2]
#> [1,]    2    8
#> [2,]    4   10
#> [3,]    6   12
```

A common use is per example gradients: differentiate the loss of one
example, then map over the batch.

``` r

loss1 <- function(w, x, y) (jnp_sum(x * w) - y)^2
vmap(grad(loss1), in_axes = list(NULL, 0L, 0L))(w, X, y)
#> Array f32[3,2]
#>      [,1] [,2]
#> [1,]   -3   -6
#> [2,]   -3   -4
#> [3,]  -15  -18
```

Nested `vmap` gives all pairs.

``` r

vmap(function(a) vmap(function(b) dot(a, b))(B))(A)
#> Array f32[2,2]
#>      [,1] [,2]
#> [1,]   28   10
#> [2,]   73   28
```

`axis_name` names the mapped axis so that collectives such as
[`lax_psum()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
can reduce over it from inside the function.

``` r

vmap(function(v) v / lax_psum(v, "batch"), axis_name = "batch")(as_jax(c(1, 2, 5)))
#> Array f32[3]
#> [1] 0.125 0.250 0.625
```

## Compilation: jit

`jit(f)` traces `f` once per combination of argument shapes and dtypes,
compiles the program with XLA and reuses the executable afterwards.

``` r

selu <- function(x, alpha = 1.67, lambda = 1.05) lambda * jnp_where(x > 0, x, alpha * jnp_exp(x) - alpha)
selu_jit <- jit(selu)
big <- jnp_linspace(-3, 3, 1000000L)
invisible(selu_jit(big))                 # first call compiles
system.time(for (i in 1:20) block_until_ready(selu(big)))["elapsed"]
#> elapsed 
#>   1.702
system.time(for (i in 1:20) block_until_ready(selu_jit(big)))["elapsed"]
#> elapsed 
#>   0.765
```

Arguments that decide the *structure* of the computation (a shape, a
flag, a function) must be static: name them in `static_argnums` or
`static_argnames`. The function is recompiled for each new static value.

``` r

make <- jit(function(x, n) jnp_reshape(x, c(n, -1L)), static_argnums = 1L)
make(jnp_arange(6), 2L)
#> Array i32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    0    1    2
#> [2,]    3    4    5
make(jnp_arange(6), 3L)
#> Array i32[3,2]
#>      [,1] [,2]
#> [1,]    0    1
#> [2,]    2    3
#> [3,]    4    5
```

A traced value cannot be used where a concrete one is needed; the error
says so.

``` r

bad <- jit(function(x, n) jnp_reshape(x, c(n, -1L)))
tryCatch(bad(jnp_arange(6), 2L), error = function(e) class(e)[1:2])
#> [1] "jax_type_error" "jax_error"
```

Useful tools around compilation:

- `disable_jit(code)` runs `code` with `jit` turned off, to debug with
  ordinary R tools;
- `log_compiles(code)` reports each compilation and
  `explain_cache_misses(code)` explains why a function was traced again;
- [`clear_caches()`](https://adamrsophie.github.io/jaxR/reference/clear_caches.md)
  drops the compiled executables;
- `eval_shape(f, ...)` computes the shapes and dtypes of the result
  without running `f`;
- `lower_text(f, ...)` returns the StableHLO text of the lowered
  function;
- `ensure_compile_time_eval(code)` evaluates constant operations eagerly
  during tracing.

``` r

eval_shape(function(a, b) a %*% b, ShapeDtypeStruct(c(3L, 4L), "float32"), ShapeDtypeStruct(c(4L, 2L), "float32"))
#> ShapedArray(f32[3,2])
cat(lower_text(function(x) x * 2 + 1, as_jax(c(1, 2))))
#> "builtin.module"() <{sym_name = "jit_fun"}> ({
#>   "func.func"() <{function_type = (tensor<2xf32>) -> tensor<2xf32>, sym_name = "main", sym_visibility = "public"}> ({
#>   ^bb0(%arg0: tensor<2xf32>):
#>     %0 = "stablehlo.constant"() <{value = dense<0x40000000> : tensor<f32>}> : () -> tensor<f32>
#>     %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<2xf32>
#>     %2 = "stablehlo.multiply"(%arg0, %1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
#>     %3 = "stablehlo.constant"() <{value = dense<0x3F800000> : tensor<f32>}> : () -> tensor<f32>
#>     %4 = "stablehlo.broadcast_in_dim"(%3) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<2xf32>
#>     %5 = "stablehlo.add"(%2, %4) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
#>     "func.return"(%5) : (tensor<2xf32>) -> ()
#>   }) : () -> ()
#> }) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()
```

### Ahead of time compilation, saved executables and the disk cache

`jit(f)$lower(...)` lowers `f` for the given arguments (or
`ShapeDtypeStruct` values) and `$compile()` compiles it; the compiled
function only accepts arguments of those shapes and dtypes. As in
`jax.experimental.serialize_executable`, a compiled function can be
saved and loaded by a later R session on the same platform with the same
XLA runtime, without tracing or compiling again.

``` r

compiled <- jit(function(x) jnp_tanh(x) * 2)$lower(as_jax(c(0, 1)))$compile()
s <- experimental_serialize_executable_serialize(compiled)
path <- tempfile(fileext = ".rds")
saveRDS(s, path)
s <- readRDS(path)
loaded <- experimental_serialize_executable_deserialize_and_load(s[[1]], s[[2]], s[[3]])
loaded(as_jax(c(0.5, 2)))
#> Array f32[2]
#> [1] 0.9242343 1.9280552
```

The persistent compilation cache does the same automatically: with a
cache directory set, every program whose compilation took at least
`jax_persistent_cache_min_compile_time_secs` seconds (default 1) is
written there, and later sessions read it back instead of compiling.
`jax_enable_compilation_cache = FALSE` turns it off. A cached file is
native code, so use a directory only you can write to.

``` r

experimental_compilation_cache_set_cache_dir(file.path(tempdir(), "jax_cache"))
jax_config_update("jax_persistent_cache_min_compile_time_secs", 0)
invisible(jit(function(x) jnp_cos(x) + 1)(as_jax(1)))
length(list.files(file.path(tempdir(), "jax_cache"))) > 0
#> [1] TRUE
jax_config_update("jax_compilation_cache_dir", NULL)
jax_config_update("jax_persistent_cache_min_compile_time_secs", 1)
```

## Composition

The transformations are designed to be combined in any order.

``` r

per_example_hessian <- jit(vmap(hessian(function(x) jnp_sum(x^3))))
per_example_hessian(as_jax(matrix(c(1, 2, 3, 4), 2)))
#> Array f32[2,2,2]
#> [0, , ]
#>      [,1] [,2]
#> [1,]    6    0
#> [2,]    0   18
#> [1, , ]
#>      [,1] [,2]
#> [1,]   12    0
#> [2,]    0   24
```

## Summary of the transformation API

| Function | Purpose |
|----|----|
| `grad`, `value_and_grad` | gradient of a scalar function |
| `jvp`, `linearize` | forward mode |
| `vjp`, `linear_transpose` | reverse mode |
| `jacfwd`, `jacrev`, `jacobian`, `hessian` | full derivative matrices |
| `stop_gradient` | block differentiation |
| `custom_jvp`, `custom_vjp`, `custom_gradient` | user defined derivative rules |
| `checkpoint` (`remat`), `checkpoint_policies`, `print_saved_residuals` | memory and time trade-off in reverse mode |
| `experimental_jet_jet` | Taylor mode: derivatives of order 1 to K in one pass |
| `fwd_and_bwd`, `closure_convert`, `linear_call` | lower level building blocks |
| `vmap`, `sequential_vmap`, `custom_vmap` | vectorization and user defined batching rules |
| `jit`, `disable_jit`, `clear_caches`, `make_jaxpr`, `eval_shape`, `lower_text` | compilation and inspection |
| `experimental_serialize_executable_serialize`, `experimental_compilation_cache_set_cache_dir` | saved executables and the disk cache |
| `named_call`, `named_scope` | names in the lowered program |
