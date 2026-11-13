# Differences from JAX and from base R

``` r

library(jaxR)
```

jaxR follows JAX 0.12.0.1 closely: the same algorithms, the same dtype
rules, the same random bits. This vignette lists where it differs, first
for readers who know JAX, then for readers who know R, and ends with
what is not ported.

## For JAX users

### Names

Python modules become prefixes. Transformations and core utilities keep
their names.

| JAX | jaxR |
|----|----|
| `jax.grad`, `jax.vmap`, `jax.jit` | `grad`, `vmap`, `jit` |
| `jax.numpy.linalg.solve` | `jnp_linalg_solve` |
| `jax.lax.while_loop` | `lax_while_loop` |
| `jax.random.normal` | `random_normal` |
| `jax.nn.initializers.he_normal` | `nn_initializers_he_normal` |
| `jax.scipy.stats.norm.logpdf` | `scipy_stats_norm_logpdf` |
| `jax.experimental.sparse.BCOO` | `experimental_sparse_BCOO` |
| `jax.tree.map`, `jax.tree_util.tree_map` | `tree_map` |
| `jax.config.update` | `jax_config_update` |
| `jax.tree_util.Partial` | `jax_Partial` |
| `x.at[i].set(v)` | `at_set(x, i, v)` |
| `x.T`, `x.shape`, `x.dtype`, `x.astype(d)` | `t(x)`, `dim(x)`, `dtype(x)`, `jnp_astype(x, d)` |
| `x.sum()`, `x.reshape(s)` (array methods) | `jnp_sum(x)`, `jnp_reshape(x, s)` (functions) |

No jaxR function masks a function of base R or of the recommended
packages. Where the JAX name would (`jax.numpy.sum` against `sum`), the
prefixed name is exported and the base R generic gets a method:
`sum(x)`, `mean(x)`, `sin(x)`, `x %*% y`, `t(x)`, `dim(x)`, `length(x)`,
`rev(x)`, `rep(x, n)`, `c(x, y)` and the arithmetic, comparison and
logical operators all work on jaxR arrays.

### Python syntax that R does not have

| Python | R |
|----|----|
| `x[1:3]`, `x[::2]`, `x[::-1]` | `x[sl(1, 3)]`, `x[sl(step = 2)]`, `x[sl(step = -1)]` |
| `x[..., 0]` | `x[ellipsis, 0]` |
| `x[None]` | `x[newaxis]` |
| tuples `(a, b)` | `list(a, b)` or `c(2L, 3L)` for shapes |
| dicts | named lists |
| keyword only arguments | named arguments |
| decorators | plain function calls: `f <- jit(f)` |
| context managers | functions taking the code: `disable_jit(code)`, `with_jax_config(name, value, code)` |

Indices, axes, `argnums` and `in_axes` are 0-based, as in JAX, and
negative values count from the end.

``` r

x <- jnp_arange(10)
x[0]
#> Array i32[]
#> [1] 0
x[-1]
#> Array i32[]
#> [1] 9
x[sl(2, 8, 2)]
#> Array i32[3]
#> [1] 2 4 6
jnp_sum(jnp_ones(c(2L, 3L)), axis = 0L)
#> Array f32[3]
#> [1] 2 2 2
```

### R values as arguments

- An R double scalar is a weakly typed float and an R integer scalar a
  weakly typed integer, like Python scalars: `x * 2` keeps the dtype of
  `x`.
- A length one R vector is a scalar (rank 0), because R has no separate
  scalar type. Use `jnp_array(5, ndmin = 1L)` or `jnp_reshape(v, 1L)`
  for a vector of length one.
- A whole valued R double is accepted where JAX wants a Python integer
  (shapes, axes, integer exponents in `jnp_power`, the arguments of
  `jnp_arange`), so `x^2` uses the integer power rule and
  `jnp_arange(5)` is an integer range. The same holds for fill values:
  `lax_full_like(x, 128)` of an `int8` array wraps to `-128` like the
  Python integer `128`, where the Python float `128.0` saturates to
  `127`; R cannot tell the two apart. A fill that is not a whole number
  saturates as in JAX.
- R numbers given an integer dtype convert like Python numbers in
  `jnp.array`: they are truncated toward zero, and a value outside the
  range of the dtype, `NaN` or an infinity is an error
  (`as_jax(300, "uint8")`). Conversions of arrays (`jnp_astype`,
  `lax_convert_element_type`) follow the device rules instead.
- R matrices and arrays convert with their R layout:
  `as_jax(matrix(1:6, 2))` is the 2 by 3 array you see when you print
  the matrix. [`as.array()`](https://rdrr.io/r/base/array.html) converts
  back.
- Named lists are the dictionaries of jaxR. Like JAX, pytree flattening
  visits the names in sorted order.

``` r

dtype(as_jax(c(1, 2), "float32") * 2)
#> [1] "float32"
dim(as_jax(3))
#> integer(0)
dtype(jnp_arange(5))
#> [1] "int32"
dtype(as_jax(1:3)^2)
#> [1] "int32"
```

### Where jaxR accepts more than JAX

A few calls that raise in JAX are accepted in jaxR, because the R value
has no exact Python counterpart. Code translated from JAX is not
affected.

- `lax_*` functions take R scalars of any numeric type next to an array:
  the scalar adopts the dtype of the array. In JAX, `lax.add(x, 2.0)`
  raises for a `float32` array when 64 bit mode is on, because the
  Python scalar is a `float64`. The `jnp_*` functions promote like
  `jax.numpy` in both libraries.
- `lax_*` functions accept negative axes (`lax_rev(x, -1L)`,
  `lax_concatenate(xs, -1L)`, `lax_argmax(x, -1L, "int32")`); `jax.lax`
  wants nonnegative axes.
- An R integer vector is an `int32` array in both precision modes,
  because R integers are 32 bit; a Python list of ints is `int64` when
  64 bit mode is on. An R integer scalar is a weakly typed default
  integer, like a Python int.
- `"float0"` is accepted as a dtype name. In Python the dtype is the
  object `jax.dtypes.float0` and the string is not understood.
- [`lax_reshape()`](https://adamrsophie.github.io/jaxR/reference/lax_reshape.md)
  reports a wrong total size when the call is traced (`jax_type_error`);
  JAX reports it later, from the compiler (`ValueError`).
- [`experimental_checkify_check()`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  takes R numbers as formatting arguments; JAX wants arrays there.

Errors have the class of the Python exception (`jax_type_error` for
`TypeError`, `jax_value_error` for `ValueError`, and so on, see
[`vignette("advanced")`](https://adamrsophie.github.io/jaxR/articles/advanced.md)).
Messages follow JAX where the same check exists, but the wording is not
guaranteed to match.

### Semantics that carry over unchanged

Everything below behaves as in JAX and is listed because it differs from
R:

- Arrays are immutable. `x[i] <- v` works for jaxR arrays and returns a
  modified copy through `at_set`; inside `jit` it compiles to an in
  place update where that is safe.
- The default precision is 32 bit.
  [`enable_x64()`](https://adamrsophie.github.io/jaxR/reference/enable_x64.md)
  switches to 64 bit.
- Out of bounds indices are clamped for reads and dropped for updates,
  there is no error.
- Dtype promotion follows the JAX lattice, not R coercion.
- Integer overflow wraps.
- Functions under `jit`, `vmap` and `grad` must be pure and must not
  branch in R on array values; use `jnp_where`, `lax_cond`,
  `lax_while_loop` and friends.

### Numerical agreement

Random bits and integer results agree with JAX exactly. Floating point
results agree to rounding, with these known sources of last digit
differences:

- Elementary functions are evaluated by the XLA build inside the PJRT
  plugin. Its vectorized kernels can round differently from the build
  shipped in `jaxlib` (for example `atan(Inf)` in float32 is one unit in
  the last place below `pi / 2`, so a float32 Cauchy tail does not reach
  exactly zero).
- On the CPU the linear algebra primitives call R’s LAPACK, which has
  double precision routines only: float32 factorizations (QR, SVD,
  eigendecompositions, LU, Cholesky) are computed in double precision
  and rounded, which is at least as accurate as `jaxlib` but not bit
  identical.
- R has no 64 bit integer type. `int64` and `uint64` arrays compute
  exactly on the device, but values beyond 2^53 lose low bits when they
  are converted to R numbers
  ([`as.array()`](https://rdrr.io/r/base/array.html)), and R constants
  beyond 2^53 cannot be written exactly.
- The 6 bit floating point dtypes of `ml_dtypes` cannot be converted to
  R values.

### Behavior that is intentionally different

- `lax_pmin` and `lax_pmax` under `vmap(..., axis_name = )` are
  differentiable; JAX raises an error because the rule is missing there.
- [`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md),
  [`hessian()`](https://adamrsophie.github.io/jaxR/reference/hessian.md)
  and
  [`jacobian()`](https://adamrsophie.github.io/jaxR/reference/jacrev.md)
  forward symbolic expressions of symcompR to symcompR (see the
  `symbolic` vignette).
- [`experimental_checkify_checkify()`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  checks operations whose results are unused at the top level of the
  checked function, as JAX does, but not inside the bodies of
  `lax_cond`, `lax_scan` and `lax_while_loop`: jaxR removes unused code
  when it traces a body. Arrays in check messages are formatted by a
  simpler rule than NumPy’s.
- `experimental_sparse_BCOO$from_scipy_sparse()` takes a sparse matrix
  of the Matrix package.
- In
  [`experimental_jet_jet()`](https://adamrsophie.github.io/jaxR/reference/experimental_jet_jet.md),
  the Taylor terms of an integer constant next to a weakly typed array
  are weakly typed, as in JAX, when the constant meets the array in a
  `jnp_*` ufunc. Inside a function compiled with
  [`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) the
  conversion of the constant has been folded away, and the terms are
  strongly typed; JAX keeps them weak there.
- [`experimental_random_stateful_rng()`](https://adamrsophie.github.io/jaxR/reference/experimental_random_stateful_rng.md)
  without a seed draws the seed from R’s random number generator
  ([`set.seed()`](https://rdrr.io/r/base/Random.html) applies); JAX
  takes it from the operating system.
- Saving and loading (`jnp_save`, `jnp_load`) read and write the NumPy
  `.npy` format, so files are interchangeable with Python.

## For R users

| Base R habit | jaxR |
|----|----|
| 1-based indices | 0-based indices |
| `x[2:4]` (inclusive) | `x[sl(1, 4)]` (end exclusive) |
| double precision everywhere | float32 by default; [`enable_x64()`](https://adamrsophie.github.io/jaxR/reference/enable_x64.md) for doubles |
| vectors recycle silently | broadcasting follows NumPy rules; incompatible shapes are an error |
| column major storage | printing and conversion preserve the R view of a matrix; reshaping is row major, as in NumPy |
| [`set.seed()`](https://rdrr.io/r/base/Random.html) and global random state | explicit keys: [`random_key()`](https://adamrsophie.github.io/jaxR/reference/random_key.md), [`random_split()`](https://adamrsophie.github.io/jaxR/reference/random_key.md) |
| `if (x > 0)` on data | [`jnp_where()`](https://adamrsophie.github.io/jaxR/reference/jnp_where.md) or [`lax_cond()`](https://adamrsophie.github.io/jaxR/reference/lax_cond.md) inside transformed functions |
| `for` loops over elements | `vmap`, `lax_scan`, `lax_fori_loop` |
| modify in place | functional updates (`at_set`, `at_add`, or `x[i] <- v`) |
| `NA` | no missing values; use `NaN` and the `jnp_nan*` reductions |

Reshaping is the point that most often surprises:

``` r

m <- as_jax(matrix(1:6, 2))
m
#> Array i32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    1    3    5
#> [2,]    2    4    6
jnp_reshape(m, 6L)           # row major: 1 3 5 2 4 6
#> Array i32[6]
#> [1] 1 3 5 2 4 6
as.vector(as.array(m))       # R's column major view of the same matrix
#> [1] 1 2 3 4 5 6
```

## What is not ported

jaxR ports the single device, single process functionality of JAX. The
following are outside it:

- **Multiple devices and processes**: `pmap` across devices,
  `jax.sharding`, `shard_map`, `jax.distributed`,
  `jax.experimental.multihost_utils`. The named axis collectives
  (`lax_psum`, `lax_pmean`, `lax_pmax`, `lax_pmin`, `lax_all_gather`, …)
  work under `vmap(..., axis_name = )`.
- **Python interchange**: DLPack (`jax.dlpack`), `jax2tf`, the
  `__array__` protocols. Use `.npy` files (`jnp_save`, `jnp_load`) to
  exchange arrays with Python.
- **The profiler** (`jax.profiler`), which drives TensorFlow’s profiler.
  Use `monitoring_*`,
  [`log_compiles()`](https://adamrsophie.github.io/jaxR/reference/config_contexts.md)
  and R’s own profiler.
- **Exporting** (`jax.export`, shape polymorphism). Compiled functions
  can be saved and loaded on the same platform
  ([`experimental_serialize_executable_serialize()`](https://adamrsophie.github.io/jaxR/reference/experimental_serialize_executable_serialize.md)),
  and the persistent compilation cache is ported
  ([`experimental_compilation_cache_set_cache_dir()`](https://adamrsophie.github.io/jaxR/reference/experimental_compilation_cache_set_cache_dir.md)).
- **Experimental modules**: `rnn`, `pallas`, `fused` and the other small
  experimental modules and `jax.extend`. (`jax.experimental.jet`,
  `jax.experimental.checkify`, `jax.experimental.random` and
  `jax.experimental.key_reuse` are ported:
  [`experimental_jet_jet()`](https://adamrsophie.github.io/jaxR/reference/experimental_jet_jet.md),
  [`experimental_checkify_checkify()`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md),
  [`experimental_random_stateful_rng()`](https://adamrsophie.github.io/jaxR/reference/experimental_random_stateful_rng.md)
  and the option `jax_debug_key_reuse`.)
- **The foreign function interface** (`jax.ffi`) is ported for typed FFI
  handlers:
  [`ffi_register_target()`](https://adamrsophie.github.io/jaxR/reference/ffi_register_target.md),
  [`ffi_register_type()`](https://adamrsophie.github.io/jaxR/reference/ffi_register_target.md),
  [`ffi_call()`](https://adamrsophie.github.io/jaxR/reference/ffi_call.md)
  and
  [`ffi_include_dir()`](https://adamrsophie.github.io/jaxR/reference/ffi_register_target.md).
  A handler is registered from an external pointer
  ([`getNativeSymbolInfo()`](https://rdrr.io/r/base/getNativeSymbolInfo.html))
  instead of a `PyCapsule`, so `pycapsule` has no counterpart. Only the
  `execute` stage of a handler can be registered, and handlers of the
  earlier custom call API (`api_version = 0`) cannot be registered,
  because the PJRT plugin interface does not offer either.
  `ffi_lowering` and `build_ffi_lowering_function` define lowering rules
  of new primitives, which jaxR does not expose (see `jax.extend`), and
  `abstract_token` results are not supported.
- **Key reuse checking** records the uses of keys while a function is
  traced, where JAX reads its jaxpr. The rules are those of JAX, with
  three differences: messages name the random function (`random_bits`)
  where JAX names a jit-compiled function or says “In jit”; a traced key
  that passes on another (`key[newaxis]`, traced indices) consumes all
  of that key when it is consumed, where JAX consumes the forwarded
  elements; and a function mapped by
  [`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) over
  another axis than the first consumes the whole key array.
- **Gradient accumulators** (`vjp_fun$with_refs()`, `vjp_fun$with_logs`,
  [`custom_vjp_defvjp_with_accums()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp_defvjp_with_logs.md),
  [`custom_vjp_defvjp_with_logs()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp_defvjp_with_logs.md)
  and the `with_logs` and `with_accums` arguments of
  [`custom_gradient()`](https://adamrsophie.github.io/jaxR/reference/custom_gradient.md))
  compute the gradients and logs of JAX, with these differences. A rule
  receives the accumulator kind it declares (a `ValAccum` for
  [`custom_vjp_defvjp_with_accums()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp_defvjp_with_logs.md)),
  where JAX gives the kind the caller asks for through `with_refs()` (a
  `RefAccum` for a ref, a `NullAccum` for
  [`ad_DontWant()`](https://adamrsophie.github.io/jaxR/reference/ad_GradValue.md))
  and raises a `TypeError` when it contradicts the declared kind. A
  `RefAccum` is a fresh ref whose value becomes an ordinary cotangent,
  so sparse updates are not done in place, every gradient is computed,
  and a ref passed to `with_refs()` gets its cotangent added after the
  backward pass. Logs of a rule inside the body of
  [`lax_scan()`](https://adamrsophie.github.io/jaxR/reference/lax_scan.md),
  [`lax_cond()`](https://adamrsophie.github.io/jaxR/reference/lax_cond.md),
  [`lax_while_loop()`](https://adamrsophie.github.io/jaxR/reference/lax_while_loop.md)
  or
  [`checkpoint()`](https://adamrsophie.github.io/jaxR/reference/checkpoint.md)
  are an error when requested (JAX stacks or sums them, or raises on a
  repeated key under
  [`checkpoint()`](https://adamrsophie.github.io/jaxR/reference/checkpoint.md));
  gradients are unaffected. Under
  [`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) logs
  are batched, as in JAX.
  [`custom_vjp_defremat()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp_defremat.md)
  runs the rematerialization rule right after the forward rule, also
  under
  [`checkpoint()`](https://adamrsophie.github.io/jaxR/reference/checkpoint.md),
  where JAX defers it to the backward pass. A tangent of dtype `float0`
  (of an integer value) is always a symbolic zero in jaxR, so `out_nzs`
  is `FALSE` for an integer output that a
  [`custom_vjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md)
  rule marks as perturbed, where JAX reports `TRUE`.
- **Accelerators** are not tested. The package talks to any PJRT plugin
  through `JAXR_PJRT_PLUGIN`, but the LAPACK based linear algebra is
  implemented for the CPU plugin only.

Python only helpers with no meaning in R (type annotations such as
`jax.Array` and `jax.typing`, the abstract base classes of key paths and
initializers) have no counterpart.
