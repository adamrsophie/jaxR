# Changelog

## jaxR 0.1.0

First release. jaxR brings the programming model of
[JAX](https://github.com/jax-ml/jax) to R: functions written with jaxR
arrays can be differentiated, vectorized and compiled by XLA. The
reference for behavior is JAX 0.12.0.1 on the CPU; the test suite
compares results against values produced by that version.

### Transformations

- Automatic differentiation in forward and reverse mode:
  [`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md),
  [`value_and_grad()`](https://adamrsophie.github.io/jaxR/reference/value_and_grad.md),
  [`jvp()`](https://adamrsophie.github.io/jaxR/reference/jvp.md),
  [`vjp()`](https://adamrsophie.github.io/jaxR/reference/vjp.md),
  [`linearize()`](https://adamrsophie.github.io/jaxR/reference/linearize.md),
  [`linear_transpose()`](https://adamrsophie.github.io/jaxR/reference/linear_transpose.md),
  [`jacfwd()`](https://adamrsophie.github.io/jaxR/reference/jacfwd.md),
  [`jacrev()`](https://adamrsophie.github.io/jaxR/reference/jacrev.md)
  and
  [`hessian()`](https://adamrsophie.github.io/jaxR/reference/hessian.md),
  composable to any order.
- Vectorization with
  [`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md),
  including `in_axes`, `out_axes`, named axes and the collective
  operations of `lax` under a named axis.
- Staging and compilation:
  [`make_jaxpr()`](https://adamrsophie.github.io/jaxR/reference/make_jaxpr.md),
  [`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) with
  static arguments,
  [`eval_shape()`](https://adamrsophie.github.io/jaxR/reference/eval_shape.md),
  and the ahead of time stages `$trace()`, `$lower()` and `$compile()`.
- Custom derivatives
  ([`custom_jvp()`](https://adamrsophie.github.io/jaxR/reference/custom_jvp.md),
  [`custom_vjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md),
  [`custom_gradient()`](https://adamrsophie.github.io/jaxR/reference/custom_gradient.md)),
  custom batching rules
  ([`custom_vmap()`](https://adamrsophie.github.io/jaxR/reference/custom_vmap.md)),
  and rematerialization with
  [`checkpoint()`](https://adamrsophie.github.io/jaxR/reference/checkpoint.md)
  and its policies.
- Pytrees: nested lists and registered classes work as structured
  arguments and results of every transformation
  ([`tree_map()`](https://adamrsophie.github.io/jaxR/reference/tree_map.md),
  [`tree_flatten()`](https://adamrsophie.github.io/jaxR/reference/tree_flatten.md),
  [`register_pytree_node()`](https://adamrsophie.github.io/jaxR/reference/register_pytree_node.md)
  and the `tree_util` functions with key paths).

### Arrays and numerical libraries

- `jnp_*`: the `jax.numpy` interface, with broadcasting, type promotion,
  weak types, indexing and indexed updates
  ([`at_set()`](https://adamrsophie.github.io/jaxR/reference/at.md),
  [`at_add()`](https://adamrsophie.github.io/jaxR/reference/at.md) and
  related functions), linear algebra (`jnp_linalg_*`), FFTs
  (`jnp_fft_*`), ufunc objects and the extended dtypes (bfloat16, float8
  variants, 2 and 4 bit integers).
- `lax_*`: the primitives of `jax.lax`, including convolutions, windowed
  reductions, scatter and gather, sorting, special functions, linear
  algebra decompositions, and structured control flow
  ([`lax_cond()`](https://adamrsophie.github.io/jaxR/reference/lax_cond.md),
  [`lax_switch()`](https://adamrsophie.github.io/jaxR/reference/lax_switch.md),
  [`lax_while_loop()`](https://adamrsophie.github.io/jaxR/reference/lax_while_loop.md),
  [`lax_fori_loop()`](https://adamrsophie.github.io/jaxR/reference/lax_fori_loop.md),
  [`lax_scan()`](https://adamrsophie.github.io/jaxR/reference/lax_scan.md),
  [`lax_map()`](https://adamrsophie.github.io/jaxR/reference/lax_map.md),
  [`lax_associative_scan()`](https://adamrsophie.github.io/jaxR/reference/lax_associative_scan.md)),
  all differentiable and compilable.
- `random_*`: counter based random numbers with typed and raw keys, the
  threefry and rbg generators, and the samplers of `jax.random`. Bits
  agree with JAX for equal keys.
- `scipy_*`: special functions, statistical distributions, linear
  algebra, signal processing, FFTs, spatial rotations, interpolation,
  integration, optimization
  ([`scipy_optimize_minimize()`](https://adamrsophie.github.io/jaxR/reference/scipy_optimize_minimize.md)
  with BFGS and L-BFGS) and sparse iterative solvers.
- `nn_*`: activation functions, attention, normalization helpers and
  parameter initializers of `jax.nn`.
- `stax_*` and `optimizers_*`: the neural network layers and optimizers
  of `jax.example_libraries`.
- `experimental_*`: sparse arrays (BCOO, BCSR, COO, CSR and CSC formats,
  sparse linear algebra and the `sparsify` transformation), `checkify`,
  Taylor mode differentiation
  ([`experimental_jet_jet()`](https://adamrsophie.github.io/jaxR/reference/experimental_jet_jet.md)),
  ODE integration
  ([`experimental_ode_odeint()`](https://adamrsophie.github.io/jaxR/reference/odeint.md)),
  layouts, executable serialization, the persistent compilation cache
  and stateful random generators.

### Runtime and interoperability

- Programs are lowered to StableHLO and compiled and executed by XLA
  through the PJRT C API. The CPU runtime is installed once with
  [`install_pjrt()`](https://adamrsophie.github.io/jaxR/reference/install_pjrt.md);
  prebuilt runtimes cover Linux and macOS on x86-64 and arm64. Any PJRT
  plugin can be selected with the `JAXR_PJRT_PLUGIN` environment
  variable. No Python installation is needed.
- Host callbacks
  ([`pure_callback()`](https://adamrsophie.github.io/jaxR/reference/pure_callback.md),
  [`io_callback()`](https://adamrsophie.github.io/jaxR/reference/experimental_io_callback.md),
  [`debug_callback()`](https://adamrsophie.github.io/jaxR/reference/debug_callback.md),
  [`debug_print()`](https://adamrsophie.github.io/jaxR/reference/debug_print.md))
  run R functions from compiled programs, with ordered effects.
- Foreign function calls into native code through the XLA FFI
  ([`ffi_call()`](https://adamrsophie.github.io/jaxR/reference/ffi_call.md),
  [`ffi_register_target()`](https://adamrsophie.github.io/jaxR/reference/ffi_register_target.md)),
  with LAPACK kernels provided by R’s LAPACK.
- Configuration
  ([`jax_config_update()`](https://adamrsophie.github.io/jaxR/reference/jax_config_update.md),
  [`enable_x64()`](https://adamrsophie.github.io/jaxR/reference/enable_x64.md)),
  transfer guards, debugging flags for NaNs and infinities, key reuse
  checking, monitoring events and live array inspection.
- Expressions of the symcompR package can be compiled and differentiated
  ([`jax_lambdify()`](https://adamrsophie.github.io/jaxR/reference/jax_lambdify.md),
  and [`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md)
  on symbolic expressions).

### Documentation

- Vignettes cover getting started, transformations, the `jnp` and `lax`
  interfaces, pytrees, random numbers, SciPy functions, sparse arrays,
  neural networks, advanced topics (custom rules, control flow,
  compilation stages and callbacks), symbolic computation, and the
  differences from JAX.

### Known limitations

- Results are tested on the CPU. Other PJRT plugins (for example CUDA)
  can be loaded, but they are untested, and the LAPACK based lowerings
  exist for the CPU only.
- There is no prebuilt runtime for Windows; the package installs there
  and needs `JAXR_PJRT_PLUGIN` to name a PJRT plugin built for Windows.
- The scope is the single device, single process functionality of JAX:
  `pmap`, sharding, `shard_map` and distributed execution are not ported
  (the named axis collectives work under
  [`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md)).
  Neither are DLPack, `jax.export`, the profiler and Pallas. Buffer
  donation (`donate_argnums`) and the `unroll` option of
  [`lax_scan()`](https://adamrsophie.github.io/jaxR/reference/lax_scan.md)
  are accepted and ignored.
- A few results deliberately differ from JAX where its formulas lose
  accuracy or misreport success:
  [`scipy_stats_norm_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_norm.md)
  at extreme scales, the bracket test of
  [`scipy_optimize_line_search()`](https://adamrsophie.github.io/jaxR/reference/scipy_optimize_line_search.md),
  and the derivatives of padded sparse storage. The `differences`
  vignette lists these and every other intentional difference.
