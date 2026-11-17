# jaxR 0.1.0

First release. jaxR brings the programming model of [JAX](https://github.com/jax-ml/jax)
to R: functions written with jaxR arrays can be differentiated, vectorized and compiled
by XLA. The reference for behavior is JAX 0.12.0.1 on the CPU; the test suite compares
results against values produced by that version.

## Transformations

* Automatic differentiation in forward and reverse mode: `grad()`, `value_and_grad()`,
  `jvp()`, `vjp()`, `linearize()`, `linear_transpose()`, `jacfwd()`, `jacrev()` and
  `hessian()`, composable to any order.
* Vectorization with `vmap()`, including `in_axes`, `out_axes`, named axes and the
  collective operations of `lax` under a named axis.
* Staging and compilation: `make_jaxpr()`, `jit()` with static arguments, `eval_shape()`,
  and the ahead of time stages `$trace()`, `$lower()` and `$compile()`.
* Custom derivatives (`custom_jvp()`, `custom_vjp()`, `custom_gradient()`), custom
  batching rules (`custom_vmap()`), and rematerialization with `checkpoint()` and its policies.
* Pytrees: nested lists and registered classes work as structured arguments and results
  of every transformation (`tree_map()`, `tree_flatten()`, `register_pytree_node()` and
  the `tree_util` functions with key paths).

## Arrays and numerical libraries

* `jnp_*`: the `jax.numpy` interface, with broadcasting, type promotion, weak types,
  indexing and indexed updates (`at_set()`, `at_add()` and related functions), linear
  algebra (`jnp_linalg_*`), FFTs (`jnp_fft_*`), ufunc objects and the extended dtypes
  (bfloat16, float8 variants, 2 and 4 bit integers).
* `lax_*`: the primitives of `jax.lax`, including convolutions, windowed reductions,
  scatter and gather, sorting, special functions, linear algebra decompositions, and
  structured control flow (`lax_cond()`, `lax_switch()`, `lax_while_loop()`,
  `lax_fori_loop()`, `lax_scan()`, `lax_map()`, `lax_associative_scan()`), all
  differentiable and compilable.
* `random_*`: counter based random numbers with typed and raw keys, the threefry and rbg
  generators, and the samplers of `jax.random`. Bits agree with JAX for equal keys.
* `scipy_*`: special functions, statistical distributions, linear algebra, signal
  processing, FFTs, spatial rotations, interpolation, integration, optimization
  (`scipy_optimize_minimize()` with BFGS and L-BFGS) and sparse iterative solvers.
* `nn_*`: activation functions, attention, normalization helpers and parameter
  initializers of `jax.nn`.
* `stax_*` and `optimizers_*`: the neural network layers and optimizers of
  `jax.example_libraries`.
* `experimental_*`: sparse arrays (BCOO, BCSR, COO, CSR and CSC formats, sparse linear
  algebra and the `sparsify` transformation), `checkify`, Taylor mode differentiation
  (`experimental_jet_jet()`), ODE integration (`experimental_ode_odeint()`), layouts,
  executable serialization, the persistent compilation cache and stateful random
  generators.

## Runtime and interoperability

* Programs are lowered to StableHLO and compiled and executed by XLA through the PJRT C
  API. The CPU runtime is installed once with `install_pjrt()`; prebuilt runtimes cover
  Linux and macOS on x86-64 and arm64. Any PJRT plugin can be selected with the
  `JAXR_PJRT_PLUGIN` environment variable. No Python installation is needed.
* Host callbacks (`pure_callback()`, `io_callback()`, `debug_callback()`,
  `debug_print()`) run R functions from compiled programs, with ordered effects.
* Foreign function calls into native code through the XLA FFI (`ffi_call()`,
  `ffi_register_target()`), with LAPACK kernels provided by R's LAPACK.
* Configuration (`jax_config_update()`, `enable_x64()`), transfer guards, debugging
  flags for NaNs and infinities, key reuse checking, monitoring events and live array
  inspection.
* Expressions of the symcompR package can be compiled and differentiated
  (`jax_lambdify()`, and `grad()` on symbolic expressions).

## Documentation

* Vignettes cover getting started, transformations, the `jnp` and `lax` interfaces,
  pytrees, random numbers, SciPy functions, sparse arrays, neural networks, advanced
  topics (custom rules, control flow, compilation stages and callbacks), symbolic
  computation, and the differences from JAX.

## Known limitations

* Results are tested on the CPU. Other PJRT plugins (for example CUDA) can be loaded,
  but they are untested, and the LAPACK based lowerings exist for the CPU only.
* There is no prebuilt runtime for Windows; the package installs there and needs
  `JAXR_PJRT_PLUGIN` to name a PJRT plugin built for Windows.
* The scope is the single device, single process functionality of JAX: `pmap`,
  sharding, `shard_map` and distributed execution are not ported (the named axis
  collectives work under `vmap()`). Neither are DLPack, `jax.export`, the profiler and
  Pallas. Buffer donation (`donate_argnums`) and the `unroll` option of `lax_scan()`
  are accepted and ignored.
* A few results deliberately differ from JAX where its formulas lose accuracy or
  misreport success: `scipy_stats_norm_logpdf()` at extreme scales, the bracket test of
  `scipy_optimize_line_search()`, the derivatives of padded sparse storage, the index
  clamp of `lax_switch()` for wide integers, and `NaN` from
  `experimental_ode_odeint()` when its step budget runs out. The
  `differences` vignette lists these and every other intentional difference.
