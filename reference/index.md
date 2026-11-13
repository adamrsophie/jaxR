# Package index

## All functions

- [`CustomVJPPrimal()`](https://adamrsophie.github.io/jaxR/reference/CustomVJPPrimal.md)
  [`custom_vjp_primal_tree_values()`](https://adamrsophie.github.io/jaxR/reference/CustomVJPPrimal.md)
  : Primal values of custom_vjp forward rules

- [`SequenceKey()`](https://adamrsophie.github.io/jaxR/reference/SequenceKey.md)
  [`DictKey()`](https://adamrsophie.github.io/jaxR/reference/SequenceKey.md)
  [`GetAttrKey()`](https://adamrsophie.github.io/jaxR/reference/SequenceKey.md)
  [`FlattenedIndexKey()`](https://adamrsophie.github.io/jaxR/reference/SequenceKey.md)
  : Key path entries

- [`ShapeDtypeStruct()`](https://adamrsophie.github.io/jaxR/reference/ShapeDtypeStruct.md)
  : Shape and dtype of a value without computing it

- [`SymbolicZero()`](https://adamrsophie.github.io/jaxR/reference/SymbolicZero.md)
  [`zero_from_primal()`](https://adamrsophie.github.io/jaxR/reference/SymbolicZero.md)
  : Symbolic zeros

- [`ad_GradValue()`](https://adamrsophie.github.io/jaxR/reference/ad_GradValue.md)
  [`ad_DontWant()`](https://adamrsophie.github.io/jaxR/reference/ad_GradValue.md)
  [`ad_GradRef()`](https://adamrsophie.github.io/jaxR/reference/ad_GradValue.md)
  [`ad_DidntWant()`](https://adamrsophie.github.io/jaxR/reference/ad_GradValue.md)
  :

  Markers for the `with_refs` method of VJP functions

- [`ad_ValAccum`](https://adamrsophie.github.io/jaxR/reference/ad_ValAccum.md)
  [`ad_RefAccum`](https://adamrsophie.github.io/jaxR/reference/ad_ValAccum.md)
  [`ad_NullAccum`](https://adamrsophie.github.io/jaxR/reference/ad_ValAccum.md)
  : Gradient accumulators

- [`all_leaves()`](https://adamrsophie.github.io/jaxR/reference/all_leaves.md)
  : Test whether all elements are leaves

- [`as_jax()`](https://adamrsophie.github.io/jaxR/reference/as_jax.md) :
  Convert R data to a jax array

- [`at_get()`](https://adamrsophie.github.io/jaxR/reference/at.md)
  [`at_set()`](https://adamrsophie.github.io/jaxR/reference/at.md)
  [`at_add()`](https://adamrsophie.github.io/jaxR/reference/at.md)
  [`at_subtract()`](https://adamrsophie.github.io/jaxR/reference/at.md)
  [`at_multiply()`](https://adamrsophie.github.io/jaxR/reference/at.md)
  [`at_divide()`](https://adamrsophie.github.io/jaxR/reference/at.md)
  [`at_power()`](https://adamrsophie.github.io/jaxR/reference/at.md)
  [`at_min()`](https://adamrsophie.github.io/jaxR/reference/at.md)
  [`at_max()`](https://adamrsophie.github.io/jaxR/reference/at.md)
  [`at_apply()`](https://adamrsophie.github.io/jaxR/reference/at.md) :
  Functional indexed updates

- [`block_until_ready()`](https://adamrsophie.github.io/jaxR/reference/block_until_ready.md)
  : Wait for an array computation to finish

- [`broadcast_shapes()`](https://adamrsophie.github.io/jaxR/reference/broadcast_shapes.md)
  : Broadcast shapes

- [`checkpoint()`](https://adamrsophie.github.io/jaxR/reference/checkpoint.md)
  [`remat()`](https://adamrsophie.github.io/jaxR/reference/checkpoint.md)
  : Gradient checkpointing

- [`checkpoint_name()`](https://adamrsophie.github.io/jaxR/reference/checkpoint_name.md)
  : Name an intermediate value for checkpoint policies

- [`checkpoint_name_fwd()`](https://adamrsophie.github.io/jaxR/reference/checkpoint_name_fwd.md)
  : Name a value for checkpoint policies (forward pass)

- [`checkpoint_policies`](https://adamrsophie.github.io/jaxR/reference/checkpoint_policies.md)
  : Checkpoint policies

- [`clear_caches()`](https://adamrsophie.github.io/jaxR/reference/clear_caches.md)
  : Clear the compilation caches

- [`closure_convert()`](https://adamrsophie.github.io/jaxR/reference/closure_convert.md)
  : Hoist the closed-over values of a function

- [`config`](https://adamrsophie.github.io/jaxR/reference/config.md) :
  The configuration object

- [`debug_nans()`](https://adamrsophie.github.io/jaxR/reference/config_contexts.md)
  [`debug_infs()`](https://adamrsophie.github.io/jaxR/reference/config_contexts.md)
  [`numpy_dtype_promotion()`](https://adamrsophie.github.io/jaxR/reference/config_contexts.md)
  [`log_compiles()`](https://adamrsophie.github.io/jaxR/reference/config_contexts.md)
  [`explain_cache_misses()`](https://adamrsophie.github.io/jaxR/reference/config_contexts.md)
  [`check_tracer_leaks()`](https://adamrsophie.github.io/jaxR/reference/config_contexts.md)
  [`checking_leaks()`](https://adamrsophie.github.io/jaxR/reference/config_contexts.md)
  [`no_tracing()`](https://adamrsophie.github.io/jaxR/reference/config_contexts.md)
  [`no_execution()`](https://adamrsophie.github.io/jaxR/reference/config_contexts.md)
  [`transfer_guard_host_to_device()`](https://adamrsophie.github.io/jaxR/reference/config_contexts.md)
  [`transfer_guard_device_to_host()`](https://adamrsophie.github.io/jaxR/reference/config_contexts.md)
  [`transfer_guard_device_to_device()`](https://adamrsophie.github.io/jaxR/reference/config_contexts.md)
  [`transfer_guard()`](https://adamrsophie.github.io/jaxR/reference/config_contexts.md)
  : Debugging and safety configuration contexts

- [`enable_checks()`](https://adamrsophie.github.io/jaxR/reference/config_contexts_more.md)
  [`debug_key_reuse()`](https://adamrsophie.github.io/jaxR/reference/config_contexts_more.md)
  [`legacy_prng_key()`](https://adamrsophie.github.io/jaxR/reference/config_contexts_more.md)
  [`enable_custom_prng()`](https://adamrsophie.github.io/jaxR/reference/config_contexts_more.md)
  [`softmax_custom_jvp()`](https://adamrsophie.github.io/jaxR/reference/config_contexts_more.md)
  [`allow_f16_reductions()`](https://adamrsophie.github.io/jaxR/reference/config_contexts_more.md)
  [`array_garbage_collection_guard()`](https://adamrsophie.github.io/jaxR/reference/config_contexts_more.md)
  [`thread_guard()`](https://adamrsophie.github.io/jaxR/reference/config_contexts_more.md)
  : Further configuration contexts

- [`copy_to_host_async()`](https://adamrsophie.github.io/jaxR/reference/copy_to_host_async.md)
  : Copy arrays to the host

- [`custom_gradient()`](https://adamrsophie.github.io/jaxR/reference/custom_gradient.md)
  : Define a function with a custom gradient

- [`custom_jvp()`](https://adamrsophie.github.io/jaxR/reference/custom_jvp.md)
  [`custom_jvp_defjvp()`](https://adamrsophie.github.io/jaxR/reference/custom_jvp.md)
  [`custom_jvp_defjvps()`](https://adamrsophie.github.io/jaxR/reference/custom_jvp.md)
  : Define a function with a custom JVP rule

- [`custom_vjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md)
  [`custom_vjp_defvjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md)
  : Define a function with a custom VJP rule

- [`custom_vjp_defremat()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp_defremat.md)
  [`custom_vjp_defremat_with_logs()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp_defremat.md)
  : Custom rematerialization rules

- [`custom_vjp_defvjp_with_logs()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp_defvjp_with_logs.md)
  [`custom_vjp_defvjp_with_accums()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp_defvjp_with_logs.md)
  : Custom VJP rules that log or take gradient accumulators

- [`custom_vmap()`](https://adamrsophie.github.io/jaxR/reference/custom_vmap.md)
  [`custom_vmap_def_vmap()`](https://adamrsophie.github.io/jaxR/reference/custom_vmap.md)
  [`sequential_vmap()`](https://adamrsophie.github.io/jaxR/reference/custom_vmap.md)
  : Define a function with a custom vmap rule

- [`debug_DebugEffect`](https://adamrsophie.github.io/jaxR/reference/debug_DebugEffect.md)
  [`debug_OrderedDebugEffect`](https://adamrsophie.github.io/jaxR/reference/debug_DebugEffect.md)
  : Debug effects

- [`debug_breakpoint()`](https://adamrsophie.github.io/jaxR/reference/debug_breakpoint.md)
  : Enter the R browser at a point in a (possibly compiled) program

- [`debug_callback()`](https://adamrsophie.github.io/jaxR/reference/debug_callback.md)
  : Call an R function for debugging, also inside compiled code

- [`debug_print()`](https://adamrsophie.github.io/jaxR/reference/debug_print.md)
  [`debug_log()`](https://adamrsophie.github.io/jaxR/reference/debug_print.md)
  : Print values, also inside compiled code

- [`default_backend()`](https://adamrsophie.github.io/jaxR/reference/default_backend.md)
  [`device_count()`](https://adamrsophie.github.io/jaxR/reference/default_backend.md)
  [`local_device_count()`](https://adamrsophie.github.io/jaxR/reference/default_backend.md)
  [`local_devices()`](https://adamrsophie.github.io/jaxR/reference/default_backend.md)
  [`process_index()`](https://adamrsophie.github.io/jaxR/reference/default_backend.md)
  [`process_count()`](https://adamrsophie.github.io/jaxR/reference/default_backend.md)
  [`process_indices()`](https://adamrsophie.github.io/jaxR/reference/default_backend.md)
  [`host_id()`](https://adamrsophie.github.io/jaxR/reference/default_backend.md)
  [`host_count()`](https://adamrsophie.github.io/jaxR/reference/default_backend.md)
  [`host_ids()`](https://adamrsophie.github.io/jaxR/reference/default_backend.md)
  : Backend, devices and processes

- [`default_device()`](https://adamrsophie.github.io/jaxR/reference/default_device.md)
  : Run code with a default device

- [`device_get()`](https://adamrsophie.github.io/jaxR/reference/device_get.md)
  : Copy an array to the host

- [`device_put()`](https://adamrsophie.github.io/jaxR/reference/device_put.md)
  : Move data to the device

- [`devices()`](https://adamrsophie.github.io/jaxR/reference/devices.md)
  : List devices

- [`disable_jit()`](https://adamrsophie.github.io/jaxR/reference/disable_jit.md)
  : Disable jit compilation temporarily

- [`dtype()`](https://adamrsophie.github.io/jaxR/reference/dtype.md) :
  Canonical dtype name

- [`dtypes_canonicalize_dtype()`](https://adamrsophie.github.io/jaxR/reference/dtypes_canonicalize_dtype.md)
  [`dtypes_issubdtype()`](https://adamrsophie.github.io/jaxR/reference/dtypes_canonicalize_dtype.md)
  [`dtypes_result_type()`](https://adamrsophie.github.io/jaxR/reference/dtypes_canonicalize_dtype.md)
  [`dtypes_itemsize_bits()`](https://adamrsophie.github.io/jaxR/reference/dtypes_canonicalize_dtype.md)
  [`dtypes_scalar_type_of()`](https://adamrsophie.github.io/jaxR/reference/dtypes_canonicalize_dtype.md)
  [`dtypes_finfo()`](https://adamrsophie.github.io/jaxR/reference/dtypes_canonicalize_dtype.md)
  [`dtypes_iinfo()`](https://adamrsophie.github.io/jaxR/reference/dtypes_canonicalize_dtype.md)
  [`dtypes_float0`](https://adamrsophie.github.io/jaxR/reference/dtypes_canonicalize_dtype.md)
  [`dtypes_bfloat16`](https://adamrsophie.github.io/jaxR/reference/dtypes_canonicalize_dtype.md)
  [`dtypes_prng_key`](https://adamrsophie.github.io/jaxR/reference/dtypes_canonicalize_dtype.md)
  [`dtypes_extended`](https://adamrsophie.github.io/jaxR/reference/dtypes_canonicalize_dtype.md)
  [`dtypes_TypePromotionError()`](https://adamrsophie.github.io/jaxR/reference/dtypes_canonicalize_dtype.md)
  : The jax.dtypes module

- [`effects_barrier()`](https://adamrsophie.github.io/jaxR/reference/effects_barrier.md)
  : Wait for pending side effects

- [`enable_x64()`](https://adamrsophie.github.io/jaxR/reference/enable_x64.md)
  : Enable 64-bit types

- [`ensure_compile_time_eval()`](https://adamrsophie.github.io/jaxR/reference/ensure_compile_time_eval.md)
  : Evaluate eagerly at trace time

- [`eval_shape()`](https://adamrsophie.github.io/jaxR/reference/eval_shape.md)
  : Evaluate a function abstractly

- [`experimental_checkify_user_checks`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  [`experimental_checkify_nan_checks`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  [`experimental_checkify_index_checks`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  [`experimental_checkify_div_checks`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  [`experimental_checkify_float_checks`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  [`experimental_checkify_automatic_checks`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  [`experimental_checkify_all_checks`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  [`experimental_checkify_JaxRuntimeError`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  [`experimental_checkify_init_error`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  [`experimental_checkify_Error`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  [`experimental_checkify_checkify()`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  [`experimental_checkify_check()`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  [`experimental_checkify_debug_check()`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  [`experimental_checkify_check_error()`](https://adamrsophie.github.io/jaxR/reference/experimental_checkify_checkify.md)
  : Runtime error checks that work under jit, vmap and control flow

- [`experimental_compilation_cache_set_cache_dir()`](https://adamrsophie.github.io/jaxR/reference/experimental_compilation_cache_set_cache_dir.md)
  [`experimental_compilation_cache_reset_cache()`](https://adamrsophie.github.io/jaxR/reference/experimental_compilation_cache_set_cache_dir.md)
  : Persistent compilation cache

- [`experimental_io_callback()`](https://adamrsophie.github.io/jaxR/reference/experimental_io_callback.md)
  [`io_callback()`](https://adamrsophie.github.io/jaxR/reference/experimental_io_callback.md)
  : Call an impure R function from compiled code

- [`experimental_jet_jet()`](https://adamrsophie.github.io/jaxR/reference/experimental_jet_jet.md)
  : Taylor mode higher order automatic differentiation

- [`experimental_random_stateful_rng()`](https://adamrsophie.github.io/jaxR/reference/experimental_random_stateful_rng.md)
  [`experimental_random_StatefulPRNG()`](https://adamrsophie.github.io/jaxR/reference/experimental_random_stateful_rng.md)
  : Stateful random number generator

- [`experimental_serialize_executable_serialize()`](https://adamrsophie.github.io/jaxR/reference/experimental_serialize_executable_serialize.md)
  [`experimental_serialize_executable_deserialize_and_load()`](https://adamrsophie.github.io/jaxR/reference/experimental_serialize_executable_serialize.md)
  : Serialize a compiled function

- [`experimental_sparse_BCOO()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_BCOO.md)
  : Batched coordinate (BCOO) sparse arrays

- [`experimental_sparse_BCSR()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_BCSR.md)
  : Batched compressed sparse row (BCSR) arrays

- [`experimental_sparse_COO()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_COO.md)
  : Coordinate (COO) sparse matrices

- [`experimental_sparse_CSR()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_CSR.md)
  [`experimental_sparse_CSC()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_CSR.md)
  : Compressed sparse row and column matrices

- [`experimental_sparse_JAXSparse()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_abstract.md)
  [`experimental_sparse_SparseTracer()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_abstract.md)
  : Abstract sparse classes

- [`experimental_sparse_grad()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_ad.md)
  [`experimental_sparse_value_and_grad()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_ad.md)
  [`experimental_sparse_jacfwd()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_ad.md)
  [`experimental_sparse_jacrev()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_ad.md)
  [`experimental_sparse_jacobian()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_ad.md)
  : Differentiation with sparse arguments

- [`experimental_sparse_bcoo_broadcast_in_dim()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_broadcast_in_dim.md)
  : Broadcast a BCOO array

- [`experimental_sparse_bcoo_concatenate()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_concatenate.md)
  : Concatenate BCOO arrays

- [`experimental_sparse_bcoo_conv_general_dilated()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_conv_general_dilated.md)
  : Convolution with BCOO operands

- [`experimental_sparse_bcoo_dot_general()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_dot_general.md)
  : General dot product with BCOO operands

- [`experimental_sparse_bcoo_dot_general_sampled()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_dot_general_sampled.md)
  : Sampled dense dot product

- [`experimental_sparse_bcoo_extract()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_extract.md)
  : Extract values of a dense array at the indices of a BCOO array

- [`experimental_sparse_bcoo_fromdense()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_fromdense.md)
  [`experimental_sparse_bcoo_todense()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_fromdense.md)
  : Convert between dense arrays and BCOO

- [`experimental_sparse_bcoo_gather()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_gather.md)
  : Gather from a BCOO array

- [`experimental_sparse_bcoo_multiply_sparse()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_multiply_sparse.md)
  [`experimental_sparse_bcoo_multiply_dense()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_multiply_sparse.md)
  : Elementwise products with BCOO arrays

- [`experimental_sparse_bcoo_reduce_sum()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_reduce_sum.md)
  : Sum a BCOO array over axes

- [`experimental_sparse_bcoo_reshape()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_reshape.md)
  : Reshape a BCOO array

- [`experimental_sparse_bcoo_rev()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_rev.md)
  : Reverse dimensions of a BCOO array

- [`experimental_sparse_bcoo_slice()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_slice.md)
  [`experimental_sparse_bcoo_dynamic_slice()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_slice.md)
  : Slice a BCOO array

- [`experimental_sparse_bcoo_sort_indices()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_sort_indices.md)
  : Sort the indices of a BCOO array

- [`experimental_sparse_bcoo_squeeze()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_squeeze.md)
  : Remove size one dimensions of a BCOO array

- [`experimental_sparse_bcoo_sum_duplicates()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_sum_duplicates.md)
  : Sum duplicate entries of a BCOO array

- [`experimental_sparse_bcoo_transpose()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_transpose.md)
  : Transpose a BCOO array

- [`experimental_sparse_bcoo_update_layout()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_update_layout.md)
  : Change the batch and dense dimensions of a BCOO array

- [`experimental_sparse_bcsr_extract()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcsr_extract.md)
  : Extract values of a dense array at the indices of a BCSR array

- [`experimental_sparse_bcsr_fromdense()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcsr_fromdense.md)
  [`experimental_sparse_bcsr_todense()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcsr_fromdense.md)
  [`experimental_sparse_bcsr_dot_general()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcsr_fromdense.md)
  [`experimental_sparse_bcsr_broadcast_in_dim()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcsr_fromdense.md)
  [`experimental_sparse_bcsr_concatenate()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcsr_fromdense.md)
  [`experimental_sparse_bcsr_sum_duplicates()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcsr_fromdense.md)
  : Functions on BCSR arrays

- [`experimental_sparse_SparseEfficiencyError`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_conditions.md)
  [`experimental_sparse_SparseEfficiencyWarning`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_conditions.md)
  [`experimental_sparse_CuSparseEfficiencyWarning`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_conditions.md)
  : Sparse efficiency conditions

- [`experimental_sparse_coo_fromdense()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_coo_csr.md)
  [`experimental_sparse_coo_todense()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_coo_csr.md)
  [`experimental_sparse_coo_matvec()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_coo_csr.md)
  [`experimental_sparse_coo_matmat()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_coo_csr.md)
  [`experimental_sparse_csr_fromdense()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_coo_csr.md)
  [`experimental_sparse_csr_todense()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_coo_csr.md)
  [`experimental_sparse_csr_matvec()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_coo_csr.md)
  [`experimental_sparse_csr_matmat()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_coo_csr.md)
  : Functions on COO and CSR matrices

- [`experimental_sparse_empty()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_empty.md)
  [`experimental_sparse_eye()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_empty.md)
  : Empty and identity sparse arrays

- [`experimental_sparse_linalg_lobpcg_standard()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_linalg_lobpcg_standard.md)
  : Top eigenvalues with LOBPCG

- [`experimental_sparse_linalg_spsolve()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_linalg_spsolve.md)
  : Sparse direct solver

- [`experimental_sparse_random_bcoo()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_random_bcoo.md)
  : Random BCOO arrays

- [`experimental_sparse_sparsify()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_sparsify.md)
  : Sparsify a function

- [`experimental_sparse_todense()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_todense.md)
  : Convert a sparse array to a dense array

- [`ffi_call()`](https://adamrsophie.github.io/jaxR/reference/ffi_call.md)
  : Call a foreign function

- [`ffi_register_target()`](https://adamrsophie.github.io/jaxR/reference/ffi_register_target.md)
  [`ffi_register_type()`](https://adamrsophie.github.io/jaxR/reference/ffi_register_target.md)
  [`ffi_register_type_id()`](https://adamrsophie.github.io/jaxR/reference/ffi_register_target.md)
  [`ffi_register_target_as_batch_partitionable()`](https://adamrsophie.github.io/jaxR/reference/ffi_register_target.md)
  [`ffi_include_dir()`](https://adamrsophie.github.io/jaxR/reference/ffi_register_target.md)
  : Foreign function interface

- [`finfo()`](https://adamrsophie.github.io/jaxR/reference/finfo.md) :
  Information about a floating point dtype

- [`flatten_one_level()`](https://adamrsophie.github.io/jaxR/reference/flatten_one_level.md)
  [`flatten_one_level_with_keys()`](https://adamrsophie.github.io/jaxR/reference/flatten_one_level.md)
  : Flatten one level of a pytree

- [`float0`](https://adamrsophie.github.io/jaxR/reference/float0.md) :
  The float0 dtype

- [`fwd_and_bwd()`](https://adamrsophie.github.io/jaxR/reference/fwd_and_bwd.md)
  : Separate forward and backward functions of a VJP

- [`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md) :
  Gradient of a scalar function

- [`hessian()`](https://adamrsophie.github.io/jaxR/reference/hessian.md)
  : Hessian

- [`iinfo()`](https://adamrsophie.github.io/jaxR/reference/iinfo.md) :
  Information about an integer dtype

- [`image_ResizeMethod`](https://adamrsophie.github.io/jaxR/reference/image_ResizeMethod.md)
  : Image resize methods

- [`image_resize()`](https://adamrsophie.github.io/jaxR/reference/image_resize.md)
  : Resize an image

- [`image_scale_and_translate()`](https://adamrsophie.github.io/jaxR/reference/image_scale_and_translate.md)
  : Scale and translate an image

- [`install_pjrt()`](https://adamrsophie.github.io/jaxR/reference/install_pjrt.md)
  : Install the XLA runtime plugin

- [`is_tree_node()`](https://adamrsophie.github.io/jaxR/reference/is_tree_node.md)
  : Test whether a class is a pytree node type

- [`issubdtype()`](https://adamrsophie.github.io/jaxR/reference/issubdtype.md)
  : Test dtype categories

- [`jacfwd()`](https://adamrsophie.github.io/jaxR/reference/jacfwd.md) :
  Jacobian by forward mode

- [`jacrev()`](https://adamrsophie.github.io/jaxR/reference/jacrev.md)
  [`jacobian()`](https://adamrsophie.github.io/jaxR/reference/jacrev.md)
  : Jacobian by reverse mode

- [`jax_Partial()`](https://adamrsophie.github.io/jaxR/reference/jax_Partial.md)
  : A partially applied function that is a pytree

- [`jax_backend_init()`](https://adamrsophie.github.io/jaxR/reference/jax_backend_init.md)
  : Initialize the XLA backend

- [`jax_config()`](https://adamrsophie.github.io/jaxR/reference/jax_config.md)
  : Read a configuration option

- [`jax_config_update()`](https://adamrsophie.github.io/jaxR/reference/jax_config_update.md)
  : Update a configuration option

- [`c(`*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_disabled_base.md)
  [`rev(`*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_disabled_base.md)
  [`rep(`*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_disabled_base.md)
  [`aperm(`*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_disabled_base.md)
  [`` `[[`( ``*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_disabled_base.md)
  [`` `dim<-`( ``*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_disabled_base.md)
  [`` `length<-`( ``*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_disabled_base.md)
  : Base functions that are disabled for jax values

- [`errors_JAXTypeError`](https://adamrsophie.github.io/jaxR/reference/jax_errors.md)
  [`errors_JAXIndexError`](https://adamrsophie.github.io/jaxR/reference/jax_errors.md)
  [`errors_ConcretizationTypeError`](https://adamrsophie.github.io/jaxR/reference/jax_errors.md)
  [`errors_NonConcreteBooleanIndexError`](https://adamrsophie.github.io/jaxR/reference/jax_errors.md)
  [`errors_TracerArrayConversionError`](https://adamrsophie.github.io/jaxR/reference/jax_errors.md)
  [`errors_TracerBoolConversionError`](https://adamrsophie.github.io/jaxR/reference/jax_errors.md)
  [`errors_TracerIntegerConversionError`](https://adamrsophie.github.io/jaxR/reference/jax_errors.md)
  [`errors_UnexpectedTracerError`](https://adamrsophie.github.io/jaxR/reference/jax_errors.md)
  [`errors_KeyReuseError`](https://adamrsophie.github.io/jaxR/reference/jax_errors.md)
  [`errors_JaxRuntimeError`](https://adamrsophie.github.io/jaxR/reference/jax_errors.md)
  : Error classes

- [`` `[`( ``*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_indexing.md)
  [`` `[<-`( ``*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_indexing.md)
  : NumPy style indexing of jax arrays

- [`jax_lambdify()`](https://adamrsophie.github.io/jaxR/reference/jax_lambdify.md)
  : Turn a symcompR expression into a jaxR function

- [`Ops(`*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_operators.md)
  [`matrixOps(`*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_operators.md)
  [`chooseOpsMethod(`*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_operators.md)
  [`Math(`*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_operators.md)
  [`Summary(`*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_operators.md)
  [`mean(`*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_operators.md)
  [`t(`*`<jax_value>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_operators.md)
  : Operators and math functions for jax arrays

- [`print(`*`<jax_primitive>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_primitive.md)
  : Primitive objects

- [`Ops(`*`<jax_sparse>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_sparse_operators.md)
  [`matrixOps(`*`<jax_sparse>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_sparse_operators.md)
  [`chooseOpsMethod(`*`<jax_sparse>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_sparse_operators.md)
  [`Math(`*`<jax_sparse>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_sparse_operators.md)
  [`` `[`( ``*`<jax_sparse>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_sparse_operators.md)
  : Operators for sparse arrays

- [`` `$`( ``*`<jax_jitted>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_stages.md)
  [`` `$`( ``*`<jax_stage>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_stages.md)
  [`print(`*`<jax_stage>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_stages.md)
  [`print(`*`<jax_jitted>`*`)`](https://adamrsophie.github.io/jaxR/reference/jax_stages.md)
  : Ahead of time tracing, lowering and compilation

- [`jax_typeof()`](https://adamrsophie.github.io/jaxR/reference/jax_typeof.md)
  : The abstract type of a value

- [`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) :
  Just-in-time compilation

- [`jnp_ComplexWarning`](https://adamrsophie.github.io/jaxR/reference/jnp_ComplexWarning.md)
  : NumPy warning classes

- [`jnp_angle()`](https://adamrsophie.github.io/jaxR/reference/jnp_angle.md)
  : Angle of complex values

- [`jnp_append()`](https://adamrsophie.github.io/jaxR/reference/jnp_append.md)
  : Append values to an array

- [`jnp_apply_along_axis()`](https://adamrsophie.github.io/jaxR/reference/jnp_apply_along_axis.md)
  [`jnp_apply_over_axes()`](https://adamrsophie.github.io/jaxR/reference/jnp_apply_along_axis.md)
  : Apply a function along axes

- [`jnp_arange()`](https://adamrsophie.github.io/jaxR/reference/jnp_arange.md)
  : Evenly spaced values

- [`jnp_argwhere()`](https://adamrsophie.github.io/jaxR/reference/jnp_argwhere.md)
  [`jnp_flatnonzero()`](https://adamrsophie.github.io/jaxR/reference/jnp_argwhere.md)
  : Indices of nonzero elements

- [`jnp_around()`](https://adamrsophie.github.io/jaxR/reference/jnp_around.md)
  : Round to a number of decimals

- [`jnp_array_api_version`](https://adamrsophie.github.io/jaxR/reference/jnp_array_api.md)
  [`jnp_array_namespace_info()`](https://adamrsophie.github.io/jaxR/reference/jnp_array_api.md)
  : Array API metadata

- [`jnp_array_equal()`](https://adamrsophie.github.io/jaxR/reference/jnp_array_equal.md)
  [`jnp_array_equiv()`](https://adamrsophie.github.io/jaxR/reference/jnp_array_equal.md)
  : Test arrays for equality

- [`jnp_array_repr()`](https://adamrsophie.github.io/jaxR/reference/jnp_array_repr.md)
  [`jnp_array_str()`](https://adamrsophie.github.io/jaxR/reference/jnp_array_repr.md)
  : NumPy style string representations

- [`jnp_array_split()`](https://adamrsophie.github.io/jaxR/reference/jnp_array_split.md)
  [`jnp_hsplit()`](https://adamrsophie.github.io/jaxR/reference/jnp_array_split.md)
  [`jnp_dsplit()`](https://adamrsophie.github.io/jaxR/reference/jnp_array_split.md)
  : Split an array into sub-arrays

- [`jnp_asarray()`](https://adamrsophie.github.io/jaxR/reference/jnp_asarray.md)
  [`jnp_array()`](https://adamrsophie.github.io/jaxR/reference/jnp_asarray.md)
  : Convert to a jax array

- [`jnp_atleast_1d()`](https://adamrsophie.github.io/jaxR/reference/jnp_atleast_1d.md)
  [`jnp_atleast_2d()`](https://adamrsophie.github.io/jaxR/reference/jnp_atleast_1d.md)
  [`jnp_atleast_3d()`](https://adamrsophie.github.io/jaxR/reference/jnp_atleast_1d.md)
  : Ensure a minimum number of dimensions

- [`jnp_bincount()`](https://adamrsophie.github.io/jaxR/reference/jnp_bincount.md)
  : Count occurrences of non-negative integers

- [`jnp_block()`](https://adamrsophie.github.io/jaxR/reference/jnp_block.md)
  : Assemble an array from nested lists of blocks

- [`jnp_broadcast_shapes()`](https://adamrsophie.github.io/jaxR/reference/jnp_broadcast_shapes.md)
  : Broadcast shapes

- [`jnp_choose()`](https://adamrsophie.github.io/jaxR/reference/jnp_choose.md)
  : Construct an array from an index array and choices

- [`jnp_clip()`](https://adamrsophie.github.io/jaxR/reference/jnp_clip.md)
  : Clip values to an interval

- [`jnp_compress()`](https://adamrsophie.github.io/jaxR/reference/jnp_compress.md)
  [`jnp_extract()`](https://adamrsophie.github.io/jaxR/reference/jnp_compress.md)
  : Select slices with a boolean condition

- [`jnp_pi`](https://adamrsophie.github.io/jaxR/reference/jnp_constants.md)
  [`jnp_e`](https://adamrsophie.github.io/jaxR/reference/jnp_constants.md)
  [`jnp_euler_gamma`](https://adamrsophie.github.io/jaxR/reference/jnp_constants.md)
  [`jnp_inf`](https://adamrsophie.github.io/jaxR/reference/jnp_constants.md)
  [`jnp_nan`](https://adamrsophie.github.io/jaxR/reference/jnp_constants.md)
  [`jnp_newaxis`](https://adamrsophie.github.io/jaxR/reference/jnp_constants.md)
  : Constants

- [`jnp_convolve()`](https://adamrsophie.github.io/jaxR/reference/jnp_convolve.md)
  [`jnp_correlate()`](https://adamrsophie.github.io/jaxR/reference/jnp_convolve.md)
  : Convolution and correlation of one dimensional arrays

- [`jnp_copy()`](https://adamrsophie.github.io/jaxR/reference/jnp_copy.md)
  : Copy an array

- [`jnp_cov()`](https://adamrsophie.github.io/jaxR/reference/jnp_cov.md)
  [`jnp_corrcoef()`](https://adamrsophie.github.io/jaxR/reference/jnp_cov.md)
  : Covariance and correlation matrices

- [`jnp_zeros()`](https://adamrsophie.github.io/jaxR/reference/jnp_creation.md)
  [`jnp_ones()`](https://adamrsophie.github.io/jaxR/reference/jnp_creation.md)
  [`jnp_full()`](https://adamrsophie.github.io/jaxR/reference/jnp_creation.md)
  [`jnp_zeros_like()`](https://adamrsophie.github.io/jaxR/reference/jnp_creation.md)
  [`jnp_ones_like()`](https://adamrsophie.github.io/jaxR/reference/jnp_creation.md)
  [`jnp_full_like()`](https://adamrsophie.github.io/jaxR/reference/jnp_creation.md)
  : Arrays of zeros, ones, or a fill value

- [`jnp_cross()`](https://adamrsophie.github.io/jaxR/reference/jnp_cross.md)
  : Cross product of 3-vectors

- [`jnp_cumsum()`](https://adamrsophie.github.io/jaxR/reference/jnp_cumulative.md)
  [`jnp_cumprod()`](https://adamrsophie.github.io/jaxR/reference/jnp_cumulative.md)
  [`jnp_nancumsum()`](https://adamrsophie.github.io/jaxR/reference/jnp_cumulative.md)
  [`jnp_nancumprod()`](https://adamrsophie.github.io/jaxR/reference/jnp_cumulative.md)
  [`jnp_cumulative_sum()`](https://adamrsophie.github.io/jaxR/reference/jnp_cumulative.md)
  [`jnp_cumulative_prod()`](https://adamrsophie.github.io/jaxR/reference/jnp_cumulative.md)
  : Cumulative sums and products

- [`jnp_delete()`](https://adamrsophie.github.io/jaxR/reference/jnp_delete.md)
  [`jnp_insert()`](https://adamrsophie.github.io/jaxR/reference/jnp_delete.md)
  : Delete or insert entries along an axis

- [`jnp_diagonal()`](https://adamrsophie.github.io/jaxR/reference/jnp_diagonal.md)
  [`jnp_diag()`](https://adamrsophie.github.io/jaxR/reference/jnp_diagonal.md)
  [`jnp_diagflat()`](https://adamrsophie.github.io/jaxR/reference/jnp_diagonal.md)
  [`jnp_diag_indices()`](https://adamrsophie.github.io/jaxR/reference/jnp_diagonal.md)
  [`jnp_diag_indices_from()`](https://adamrsophie.github.io/jaxR/reference/jnp_diagonal.md)
  [`jnp_fill_diagonal()`](https://adamrsophie.github.io/jaxR/reference/jnp_diagonal.md)
  : Diagonals

- [`jnp_diff()`](https://adamrsophie.github.io/jaxR/reference/jnp_diff.md)
  [`jnp_ediff1d()`](https://adamrsophie.github.io/jaxR/reference/jnp_diff.md)
  : Differences between consecutive elements

- [`jnp_digitize()`](https://adamrsophie.github.io/jaxR/reference/jnp_digitize.md)
  : Bin indices of values

- [`jnp_dot()`](https://adamrsophie.github.io/jaxR/reference/jnp_dot.md)
  : Dot product with NumPy semantics

- [`jnp_dtype()`](https://adamrsophie.github.io/jaxR/reference/jnp_dtype.md)
  : Data type objects

- [`jnp_issubdtype()`](https://adamrsophie.github.io/jaxR/reference/jnp_dtype_functions.md)
  [`jnp_isdtype()`](https://adamrsophie.github.io/jaxR/reference/jnp_dtype_functions.md)
  [`jnp_promote_types()`](https://adamrsophie.github.io/jaxR/reference/jnp_dtype_functions.md)
  [`jnp_result_type()`](https://adamrsophie.github.io/jaxR/reference/jnp_dtype_functions.md)
  [`jnp_can_cast()`](https://adamrsophie.github.io/jaxR/reference/jnp_dtype_functions.md)
  : Dtype queries and promotion

- [`jnp_einsum()`](https://adamrsophie.github.io/jaxR/reference/jnp_einsum.md)
  [`jnp_einsum_path()`](https://adamrsophie.github.io/jaxR/reference/jnp_einsum.md)
  : Einstein summation

- [`jnp_empty()`](https://adamrsophie.github.io/jaxR/reference/jnp_empty.md)
  [`jnp_empty_like()`](https://adamrsophie.github.io/jaxR/reference/jnp_empty.md)
  : Uninitialized arrays

- [`jnp_eye()`](https://adamrsophie.github.io/jaxR/reference/jnp_eye.md)
  [`jnp_identity()`](https://adamrsophie.github.io/jaxR/reference/jnp_eye.md)
  : Identity matrix

- [`jnp_fft_fftn()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft.md)
  [`jnp_fft_ifftn()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft.md)
  [`jnp_fft_rfftn()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft.md)
  [`jnp_fft_irfftn()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft.md)
  [`jnp_fft_fft()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft.md)
  [`jnp_fft_ifft()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft.md)
  [`jnp_fft_rfft()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft.md)
  [`jnp_fft_irfft()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft.md)
  [`jnp_fft_hfft()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft.md)
  [`jnp_fft_ihfft()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft.md)
  [`jnp_fft_fft2()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft.md)
  [`jnp_fft_ifft2()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft.md)
  [`jnp_fft_rfft2()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft.md)
  [`jnp_fft_irfft2()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft.md)
  : Discrete Fourier transforms

- [`jnp_fft_fftfreq()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft_fftfreq.md)
  [`jnp_fft_rfftfreq()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft_fftfreq.md)
  : Sample frequencies of the discrete Fourier transform

- [`jnp_fft_fftshift()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft_fftshift.md)
  [`jnp_fft_ifftshift()`](https://adamrsophie.github.io/jaxR/reference/jnp_fft_fftshift.md)
  : Shift the zero frequency term to the center

- [`jnp_finfo()`](https://adamrsophie.github.io/jaxR/reference/jnp_finfo.md)
  [`jnp_iinfo()`](https://adamrsophie.github.io/jaxR/reference/jnp_finfo.md)
  : Machine limits of numeric types

- [`jnp_flip()`](https://adamrsophie.github.io/jaxR/reference/jnp_flip.md)
  [`jnp_fliplr()`](https://adamrsophie.github.io/jaxR/reference/jnp_flip.md)
  [`jnp_flipud()`](https://adamrsophie.github.io/jaxR/reference/jnp_flip.md)
  : Reverse the order of elements

- [`jnp_fmax()`](https://adamrsophie.github.io/jaxR/reference/jnp_fmax.md)
  [`jnp_fmin()`](https://adamrsophie.github.io/jaxR/reference/jnp_fmax.md)
  : Elementwise maximum and minimum ignoring NaN

- [`jnp_frombuffer()`](https://adamrsophie.github.io/jaxR/reference/jnp_frombuffer.md)
  [`jnp_fromstring()`](https://adamrsophie.github.io/jaxR/reference/jnp_frombuffer.md)
  [`jnp_fromfunction()`](https://adamrsophie.github.io/jaxR/reference/jnp_frombuffer.md)
  [`jnp_fromfile()`](https://adamrsophie.github.io/jaxR/reference/jnp_frombuffer.md)
  [`jnp_fromiter()`](https://adamrsophie.github.io/jaxR/reference/jnp_frombuffer.md)
  : Arrays from raw bytes, text, functions or iterators

- [`jnp_gcd()`](https://adamrsophie.github.io/jaxR/reference/jnp_gcd.md)
  [`jnp_lcm()`](https://adamrsophie.github.io/jaxR/reference/jnp_gcd.md)
  : Greatest common divisor and least common multiple

- [`jnp_gradient()`](https://adamrsophie.github.io/jaxR/reference/jnp_gradient.md)
  : Numerical gradient

- [`jnp_histogram_bin_edges()`](https://adamrsophie.github.io/jaxR/reference/jnp_histogram_bin_edges.md)
  [`jnp_histogram()`](https://adamrsophie.github.io/jaxR/reference/jnp_histogram_bin_edges.md)
  [`jnp_histogram2d()`](https://adamrsophie.github.io/jaxR/reference/jnp_histogram_bin_edges.md)
  [`jnp_histogramdd()`](https://adamrsophie.github.io/jaxR/reference/jnp_histogram_bin_edges.md)
  : Histograms

- [`jnp_hstack()`](https://adamrsophie.github.io/jaxR/reference/jnp_hstack.md)
  [`jnp_dstack()`](https://adamrsophie.github.io/jaxR/reference/jnp_hstack.md)
  [`jnp_column_stack()`](https://adamrsophie.github.io/jaxR/reference/jnp_hstack.md)
  [`jnp_concat()`](https://adamrsophie.github.io/jaxR/reference/jnp_hstack.md)
  : Stack arrays

- [`jnp_i0()`](https://adamrsophie.github.io/jaxR/reference/jnp_i0.md) :
  Modified Bessel function of the first kind of order zero

- [`jnp_indices()`](https://adamrsophie.github.io/jaxR/reference/jnp_indices.md)
  [`jnp_ix_()`](https://adamrsophie.github.io/jaxR/reference/jnp_indices.md)
  : Index grids

- [`jnp_interp()`](https://adamrsophie.github.io/jaxR/reference/jnp_interp.md)
  : One-dimensional linear interpolation

- [`jnp_isclose()`](https://adamrsophie.github.io/jaxR/reference/jnp_isclose.md)
  [`jnp_allclose()`](https://adamrsophie.github.io/jaxR/reference/jnp_isclose.md)
  : Compare arrays elementwise within a tolerance

- [`jnp_iscomplex()`](https://adamrsophie.github.io/jaxR/reference/jnp_iscomplex.md)
  [`jnp_isreal()`](https://adamrsophie.github.io/jaxR/reference/jnp_iscomplex.md)
  : Elementwise complex and real tests

- [`jnp_iscomplexobj()`](https://adamrsophie.github.io/jaxR/reference/jnp_iscomplexobj.md)
  [`jnp_isrealobj()`](https://adamrsophie.github.io/jaxR/reference/jnp_iscomplexobj.md)
  [`jnp_isscalar()`](https://adamrsophie.github.io/jaxR/reference/jnp_iscomplexobj.md)
  [`jnp_iterable()`](https://adamrsophie.github.io/jaxR/reference/jnp_iscomplexobj.md)
  : Dtype and scalar queries

- [`jnp_kron()`](https://adamrsophie.github.io/jaxR/reference/jnp_kron.md)
  : Kronecker product

- [`jnp_linalg_cholesky()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_svd()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_svdvals()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_matrix_power()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_matrix_rank()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_slogdet()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_det()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_eig()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_eigvals()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_eigh()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_eigvalsh()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_pinv()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_inv()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_solve()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_lstsq()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_norm()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_matrix_norm()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_vector_norm()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_cond()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_qr()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_cross()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_outer()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_matrix_transpose()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_vecdot()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_matmul()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_tensordot()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_diagonal()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_trace()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_tensorinv()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_tensorsolve()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  [`jnp_linalg_multi_dot()`](https://adamrsophie.github.io/jaxR/reference/jnp_linalg.md)
  : Linear algebra (jax.numpy.linalg)

- [`jnp_linspace()`](https://adamrsophie.github.io/jaxR/reference/jnp_linspace.md)
  : Evenly spaced numbers over an interval

- [`jnp_logspace()`](https://adamrsophie.github.io/jaxR/reference/jnp_logspace.md)
  [`jnp_geomspace()`](https://adamrsophie.github.io/jaxR/reference/jnp_logspace.md)
  : Numbers spaced evenly on a log scale

- [`jnp_matmul()`](https://adamrsophie.github.io/jaxR/reference/jnp_matmul.md)
  : Matrix product with NumPy semantics

- [`jnp_meshgrid()`](https://adamrsophie.github.io/jaxR/reference/jnp_meshgrid.md)
  : Coordinate grids

- [`jnp_mgrid()`](https://adamrsophie.github.io/jaxR/reference/jnp_mgrid.md)
  [`jnp_ogrid()`](https://adamrsophie.github.io/jaxR/reference/jnp_mgrid.md)
  [`jnp_r_()`](https://adamrsophie.github.io/jaxR/reference/jnp_mgrid.md)
  [`jnp_c_()`](https://adamrsophie.github.io/jaxR/reference/jnp_mgrid.md)
  : Grids and concatenation helpers

- [`jnp_nanmin()`](https://adamrsophie.github.io/jaxR/reference/jnp_nan_reductions.md)
  [`jnp_nanmax()`](https://adamrsophie.github.io/jaxR/reference/jnp_nan_reductions.md)
  [`jnp_nansum()`](https://adamrsophie.github.io/jaxR/reference/jnp_nan_reductions.md)
  [`jnp_nanprod()`](https://adamrsophie.github.io/jaxR/reference/jnp_nan_reductions.md)
  [`jnp_nanmean()`](https://adamrsophie.github.io/jaxR/reference/jnp_nan_reductions.md)
  [`jnp_nanvar()`](https://adamrsophie.github.io/jaxR/reference/jnp_nan_reductions.md)
  [`jnp_nanstd()`](https://adamrsophie.github.io/jaxR/reference/jnp_nan_reductions.md)
  [`jnp_nanargmax()`](https://adamrsophie.github.io/jaxR/reference/jnp_nan_reductions.md)
  [`jnp_nanargmin()`](https://adamrsophie.github.io/jaxR/reference/jnp_nan_reductions.md)
  : Reductions that ignore NaN

- [`jnp_nan_to_num()`](https://adamrsophie.github.io/jaxR/reference/jnp_nan_to_num.md)
  : Replace NaN and infinities

- [`jnp_nonzero()`](https://adamrsophie.github.io/jaxR/reference/jnp_nonzero.md)
  : Indices of nonzero elements

- [`jnp_packbits()`](https://adamrsophie.github.io/jaxR/reference/jnp_packbits.md)
  [`jnp_unpackbits()`](https://adamrsophie.github.io/jaxR/reference/jnp_packbits.md)
  : Pack and unpack bits

- [`jnp_pad()`](https://adamrsophie.github.io/jaxR/reference/jnp_pad.md)
  : Pad an array

- [`jnp_permute_dims()`](https://adamrsophie.github.io/jaxR/reference/jnp_permute_dims.md)
  [`jnp_matrix_transpose()`](https://adamrsophie.github.io/jaxR/reference/jnp_permute_dims.md)
  [`jnp_rot90()`](https://adamrsophie.github.io/jaxR/reference/jnp_permute_dims.md)
  [`jnp_rollaxis()`](https://adamrsophie.github.io/jaxR/reference/jnp_permute_dims.md)
  : Permute, transpose and rotate axes

- [`jnp_piecewise()`](https://adamrsophie.github.io/jaxR/reference/jnp_piecewise.md)
  : Piecewise defined function

- [`jnp_place()`](https://adamrsophie.github.io/jaxR/reference/jnp_place.md)
  [`jnp_put()`](https://adamrsophie.github.io/jaxR/reference/jnp_place.md)
  : Replace elements by mask or flat index

- [`jnp_roots()`](https://adamrsophie.github.io/jaxR/reference/jnp_polynomial.md)
  [`jnp_polyfit()`](https://adamrsophie.github.io/jaxR/reference/jnp_polynomial.md)
  [`jnp_poly()`](https://adamrsophie.github.io/jaxR/reference/jnp_polynomial.md)
  [`jnp_polyval()`](https://adamrsophie.github.io/jaxR/reference/jnp_polynomial.md)
  [`jnp_polyadd()`](https://adamrsophie.github.io/jaxR/reference/jnp_polynomial.md)
  [`jnp_polysub()`](https://adamrsophie.github.io/jaxR/reference/jnp_polynomial.md)
  [`jnp_polyint()`](https://adamrsophie.github.io/jaxR/reference/jnp_polynomial.md)
  [`jnp_polyder()`](https://adamrsophie.github.io/jaxR/reference/jnp_polynomial.md)
  [`jnp_polymul()`](https://adamrsophie.github.io/jaxR/reference/jnp_polynomial.md)
  [`jnp_polydiv()`](https://adamrsophie.github.io/jaxR/reference/jnp_polynomial.md)
  : Polynomials

- [`jnp_get_printoptions()`](https://adamrsophie.github.io/jaxR/reference/jnp_printoptions.md)
  [`jnp_set_printoptions()`](https://adamrsophie.github.io/jaxR/reference/jnp_printoptions.md)
  [`jnp_printoptions()`](https://adamrsophie.github.io/jaxR/reference/jnp_printoptions.md)
  : Print options

- [`jnp_put_along_axis()`](https://adamrsophie.github.io/jaxR/reference/jnp_put_along_axis.md)
  : Put values along an axis

- [`jnp_quantile()`](https://adamrsophie.github.io/jaxR/reference/jnp_quantile.md)
  [`jnp_nanquantile()`](https://adamrsophie.github.io/jaxR/reference/jnp_quantile.md)
  [`jnp_percentile()`](https://adamrsophie.github.io/jaxR/reference/jnp_quantile.md)
  [`jnp_nanpercentile()`](https://adamrsophie.github.io/jaxR/reference/jnp_quantile.md)
  [`jnp_median()`](https://adamrsophie.github.io/jaxR/reference/jnp_quantile.md)
  [`jnp_nanmedian()`](https://adamrsophie.github.io/jaxR/reference/jnp_quantile.md)
  : Quantiles, percentiles and medians

- [`jnp_ravel_multi_index()`](https://adamrsophie.github.io/jaxR/reference/jnp_ravel_multi_index.md)
  [`jnp_unravel_index()`](https://adamrsophie.github.io/jaxR/reference/jnp_ravel_multi_index.md)
  : Convert between flat and multi-dimensional indices

- [`jnp_sum()`](https://adamrsophie.github.io/jaxR/reference/jnp_reductions.md)
  [`jnp_prod()`](https://adamrsophie.github.io/jaxR/reference/jnp_reductions.md)
  [`jnp_max()`](https://adamrsophie.github.io/jaxR/reference/jnp_reductions.md)
  [`jnp_min()`](https://adamrsophie.github.io/jaxR/reference/jnp_reductions.md)
  [`jnp_amax()`](https://adamrsophie.github.io/jaxR/reference/jnp_reductions.md)
  [`jnp_amin()`](https://adamrsophie.github.io/jaxR/reference/jnp_reductions.md)
  [`jnp_all()`](https://adamrsophie.github.io/jaxR/reference/jnp_reductions.md)
  [`jnp_any()`](https://adamrsophie.github.io/jaxR/reference/jnp_reductions.md)
  [`jnp_mean()`](https://adamrsophie.github.io/jaxR/reference/jnp_reductions.md)
  [`jnp_var()`](https://adamrsophie.github.io/jaxR/reference/jnp_reductions.md)
  [`jnp_std()`](https://adamrsophie.github.io/jaxR/reference/jnp_reductions.md)
  [`jnp_argmax()`](https://adamrsophie.github.io/jaxR/reference/jnp_reductions.md)
  [`jnp_argmin()`](https://adamrsophie.github.io/jaxR/reference/jnp_reductions.md)
  : Reductions

- [`jnp_s_`](https://adamrsophie.github.io/jaxR/reference/jnp_s_.md)
  [`jnp_index_exp`](https://adamrsophie.github.io/jaxR/reference/jnp_s_.md)
  : Index expression builders

- [`jnp_save()`](https://adamrsophie.github.io/jaxR/reference/jnp_save.md)
  [`jnp_savez()`](https://adamrsophie.github.io/jaxR/reference/jnp_save.md)
  [`jnp_load()`](https://adamrsophie.github.io/jaxR/reference/jnp_save.md)
  : Save and load arrays in NumPy formats

- [`jnp_bool_()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_bool()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_int8()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_int16()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_int32()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_int64()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_uint8()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_uint16()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_uint32()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_uint64()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_bfloat16()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float16()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float32()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float64()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_complex64()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_complex128()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_int1()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_uint1()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_int2()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_uint2()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_int4()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_uint4()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float4_e2m1fn()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float6_e2m3fn()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float6_e3m2fn()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float8_e3m4()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float8_e4m3()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float8_e8m0fnu()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float8_e4m3b11fnuz()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float8_e4m3fn()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float8_e4m3fnuz()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float8_e5m2()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float8_e5m2fnuz()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_single()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_double()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_csingle()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_cdouble()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_int_()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_uint()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_float_()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_complex_()`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_generic`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_number`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_integer`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_signedinteger`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_unsignedinteger`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_inexact`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_floating`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_complexfloating`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_character`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  [`jnp_flexible`](https://adamrsophie.github.io/jaxR/reference/jnp_scalar_types.md)
  : Scalar types

- [`jnp_searchsorted()`](https://adamrsophie.github.io/jaxR/reference/jnp_searchsorted.md)
  : Find insertion indices in a sorted array

- [`jnp_select()`](https://adamrsophie.github.io/jaxR/reference/jnp_select.md)
  : Select from several choices

- [`jnp_union1d()`](https://adamrsophie.github.io/jaxR/reference/jnp_setops.md)
  [`jnp_setdiff1d()`](https://adamrsophie.github.io/jaxR/reference/jnp_setops.md)
  [`jnp_setxor1d()`](https://adamrsophie.github.io/jaxR/reference/jnp_setops.md)
  [`jnp_intersect1d()`](https://adamrsophie.github.io/jaxR/reference/jnp_setops.md)
  [`jnp_isin()`](https://adamrsophie.github.io/jaxR/reference/jnp_setops.md)
  : Set operations on 1-D arrays

- [`jnp_shape()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  [`jnp_ndim()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  [`jnp_size()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  [`jnp_reshape()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  [`jnp_ravel()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  [`jnp_transpose()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  [`jnp_moveaxis()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  [`jnp_swapaxes()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  [`jnp_expand_dims()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  [`jnp_squeeze()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  [`jnp_broadcast_to()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  [`jnp_broadcast_arrays()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  [`jnp_concatenate()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  [`jnp_stack()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  [`jnp_astype()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md)
  : Shape helpers

- [`jnp_sort()`](https://adamrsophie.github.io/jaxR/reference/jnp_sorting.md)
  [`jnp_argsort()`](https://adamrsophie.github.io/jaxR/reference/jnp_sorting.md)
  [`jnp_partition()`](https://adamrsophie.github.io/jaxR/reference/jnp_sorting.md)
  [`jnp_argpartition()`](https://adamrsophie.github.io/jaxR/reference/jnp_sorting.md)
  [`jnp_sort_complex()`](https://adamrsophie.github.io/jaxR/reference/jnp_sorting.md)
  [`jnp_lexsort()`](https://adamrsophie.github.io/jaxR/reference/jnp_sorting.md)
  [`jnp_top_k()`](https://adamrsophie.github.io/jaxR/reference/jnp_sorting.md)
  : Sorting functions of jax.numpy

- [`jnp_minmax()`](https://adamrsophie.github.io/jaxR/reference/jnp_statistics.md)
  [`jnp_ptp()`](https://adamrsophie.github.io/jaxR/reference/jnp_statistics.md)
  [`jnp_count_nonzero()`](https://adamrsophie.github.io/jaxR/reference/jnp_statistics.md)
  [`jnp_average()`](https://adamrsophie.github.io/jaxR/reference/jnp_statistics.md)
  : Statistical reductions

- [`jnp_take()`](https://adamrsophie.github.io/jaxR/reference/jnp_take.md)
  : Take elements from an array

- [`jnp_take_along_axis()`](https://adamrsophie.github.io/jaxR/reference/jnp_take_along_axis.md)
  : Take values along an axis

- [`jnp_tensordot()`](https://adamrsophie.github.io/jaxR/reference/jnp_tensordot.md)
  [`jnp_matvec()`](https://adamrsophie.github.io/jaxR/reference/jnp_tensordot.md)
  [`jnp_vecmat()`](https://adamrsophie.github.io/jaxR/reference/jnp_tensordot.md)
  [`jnp_vecdot()`](https://adamrsophie.github.io/jaxR/reference/jnp_tensordot.md)
  : Tensor contractions

- [`jnp_resize()`](https://adamrsophie.github.io/jaxR/reference/jnp_tile.md)
  [`jnp_tile()`](https://adamrsophie.github.io/jaxR/reference/jnp_tile.md)
  [`jnp_repeat()`](https://adamrsophie.github.io/jaxR/reference/jnp_tile.md)
  [`jnp_roll()`](https://adamrsophie.github.io/jaxR/reference/jnp_tile.md)
  [`jnp_split()`](https://adamrsophie.github.io/jaxR/reference/jnp_tile.md)
  [`jnp_vsplit()`](https://adamrsophie.github.io/jaxR/reference/jnp_tile.md)
  [`jnp_unstack()`](https://adamrsophie.github.io/jaxR/reference/jnp_tile.md)
  [`jnp_vstack()`](https://adamrsophie.github.io/jaxR/reference/jnp_tile.md)
  : Resize, tile, repeat, roll and split arrays

- [`jnp_trace()`](https://adamrsophie.github.io/jaxR/reference/jnp_trace.md)
  : Sum along diagonals

- [`jnp_trapezoid()`](https://adamrsophie.github.io/jaxR/reference/jnp_trapezoid.md)
  : Trapezoidal integration

- [`jnp_tri()`](https://adamrsophie.github.io/jaxR/reference/jnp_tri.md)
  [`jnp_tril()`](https://adamrsophie.github.io/jaxR/reference/jnp_tri.md)
  [`jnp_triu()`](https://adamrsophie.github.io/jaxR/reference/jnp_tri.md)
  : Triangular matrices

- [`jnp_trim_zeros()`](https://adamrsophie.github.io/jaxR/reference/jnp_trim_zeros.md)
  : Trim leading and trailing zeros

- [`jnp_triu_indices()`](https://adamrsophie.github.io/jaxR/reference/jnp_triu_indices.md)
  [`jnp_tril_indices()`](https://adamrsophie.github.io/jaxR/reference/jnp_triu_indices.md)
  [`jnp_triu_indices_from()`](https://adamrsophie.github.io/jaxR/reference/jnp_triu_indices.md)
  [`jnp_tril_indices_from()`](https://adamrsophie.github.io/jaxR/reference/jnp_triu_indices.md)
  [`jnp_mask_indices()`](https://adamrsophie.github.io/jaxR/reference/jnp_triu_indices.md)
  : Indices of triangles and masks

- [`jnp_ufunc()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufunc.md)
  [`jnp_frompyfunc()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufunc.md)
  [`jnp_ufunc_reduce()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufunc.md)
  [`jnp_ufunc_accumulate()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufunc.md)
  [`jnp_ufunc_reduceat()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufunc.md)
  [`jnp_ufunc_outer()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufunc.md)
  [`jnp_ufunc_at()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufunc.md)
  [`` `$`( ``*`<jnp_ufunc>`*`)`](https://adamrsophie.github.io/jaxR/reference/jnp_ufunc.md)
  [`print(`*`<jnp_ufunc>`*`)`](https://adamrsophie.github.io/jaxR/reference/jnp_ufunc.md)
  : NumPy style universal functions

- [`jnp_add()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_subtract()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_multiply()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_true_divide()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_divide()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_floor_divide()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_remainder()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_mod()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_power()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_negative()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_positive()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_abs()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_absolute()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_sign()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_exp()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_exp2()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_log()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_log2()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_log10()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_log1p()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_expm1()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_sqrt()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_cbrt()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_square()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_reciprocal()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_sin()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_cos()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_tan()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_arcsin()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_arccos()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_arctan()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_sinh()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_cosh()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_tanh()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_arcsinh()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_arccosh()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_arctanh()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_arctan2()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_floor()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_ceil()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_trunc()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_round()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_rint()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_maximum()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_minimum()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_equal()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_not_equal()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_less()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_less_equal()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_greater()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_greater_equal()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_logical_and()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_logical_or()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_logical_xor()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_logical_not()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_bitwise_and()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_bitwise_or()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_bitwise_xor()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_bitwise_not()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_invert()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_isfinite()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_isnan()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_isinf()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_real()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_imag()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_conjugate()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_conj()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_fabs()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_bitwise_invert()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_left_shift()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_bitwise_left_shift()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_right_shift()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_bitwise_right_shift()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_bitwise_count()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_float_power()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_pow()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_nextafter()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_spacing()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_acos()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_acosh()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_asin()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_asinh()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_atan()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_atanh()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_atan2()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_copysign()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_signbit()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_divmod()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_fmod()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_deg2rad()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_rad2deg()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_degrees()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_radians()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_modf()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_isposinf()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_isneginf()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_heaviside()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_logaddexp()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_logaddexp2()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_ldexp()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_frexp()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_hypot()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_sinc()`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  [`jnp_ufuncs`](https://adamrsophie.github.io/jaxR/reference/jnp_ufuncs.md)
  : jax.numpy elementwise functions

- [`jnp_unique()`](https://adamrsophie.github.io/jaxR/reference/jnp_unique.md)
  [`jnp_unique_all()`](https://adamrsophie.github.io/jaxR/reference/jnp_unique.md)
  [`jnp_unique_counts()`](https://adamrsophie.github.io/jaxR/reference/jnp_unique.md)
  [`jnp_unique_inverse()`](https://adamrsophie.github.io/jaxR/reference/jnp_unique.md)
  [`jnp_unique_values()`](https://adamrsophie.github.io/jaxR/reference/jnp_unique.md)
  : Unique elements of an array

- [`jnp_unwrap()`](https://adamrsophie.github.io/jaxR/reference/jnp_unwrap.md)
  : Unwrap a phase signal

- [`jnp_vander()`](https://adamrsophie.github.io/jaxR/reference/jnp_vander.md)
  : Vandermonde matrix

- [`jnp_vdot()`](https://adamrsophie.github.io/jaxR/reference/jnp_vdot.md)
  [`jnp_inner()`](https://adamrsophie.github.io/jaxR/reference/jnp_vdot.md)
  [`jnp_outer()`](https://adamrsophie.github.io/jaxR/reference/jnp_vdot.md)
  : Inner products

- [`jnp_vectorize()`](https://adamrsophie.github.io/jaxR/reference/jnp_vectorize.md)
  : Vectorize a function with broadcasting

- [`jnp_where()`](https://adamrsophie.github.io/jaxR/reference/jnp_where.md)
  : Select elements

- [`jnp_blackman()`](https://adamrsophie.github.io/jaxR/reference/jnp_window.md)
  [`jnp_bartlett()`](https://adamrsophie.github.io/jaxR/reference/jnp_window.md)
  [`jnp_hamming()`](https://adamrsophie.github.io/jaxR/reference/jnp_window.md)
  [`jnp_hanning()`](https://adamrsophie.github.io/jaxR/reference/jnp_window.md)
  [`jnp_kaiser()`](https://adamrsophie.github.io/jaxR/reference/jnp_window.md)
  : Window functions

- [`jvp()`](https://adamrsophie.github.io/jaxR/reference/jvp.md) :
  Jacobian-vector product

- [`` `[`( ``*`<jax_prng_key>`*`)`](https://adamrsophie.github.io/jaxR/reference/key_array_indexing.md)
  [`as.list(`*`<jax_prng_key>`*`)`](https://adamrsophie.github.io/jaxR/reference/key_array_indexing.md)
  : Index and iterate key arrays

- [`keystr()`](https://adamrsophie.github.io/jaxR/reference/keystr.md) :
  Format a key path

- [`lax_ConvDimensionNumbers()`](https://adamrsophie.github.io/jaxR/reference/lax_ConvDimensionNumbers.md)
  : Convolution dimension numbers

- [`lax_DotAlgorithm()`](https://adamrsophie.github.io/jaxR/reference/lax_DotAlgorithm.md)
  : Dot product algorithm

- [`lax_FftType`](https://adamrsophie.github.io/jaxR/reference/lax_FftType.md)
  : FFT types

- [`lax_GatherDimensionNumbers()`](https://adamrsophie.github.io/jaxR/reference/lax_GatherDimensionNumbers.md)
  [`lax_ScatterDimensionNumbers()`](https://adamrsophie.github.io/jaxR/reference/lax_GatherDimensionNumbers.md)
  : Dimension numbers for gather and scatter

- [`lax_GatherScatterMode`](https://adamrsophie.github.io/jaxR/reference/lax_GatherScatterMode.md)
  : Out of bounds modes for gather and scatter

- [`lax_PrecisionLike()`](https://adamrsophie.github.io/jaxR/reference/lax_PrecisionLike.md)
  : Canonical precision specification

- [`lax_RandomAlgorithm`](https://adamrsophie.github.io/jaxR/reference/lax_RandomAlgorithm.md)
  : Random algorithms of rng_bit_generator

- [`lax_Tolerance()`](https://adamrsophie.github.io/jaxR/reference/lax_Tolerance.md)
  : Tolerance of a unary function

- [`lax_approx_max_k()`](https://adamrsophie.github.io/jaxR/reference/lax_approx_max_k.md)
  [`lax_approx_min_k()`](https://adamrsophie.github.io/jaxR/reference/lax_approx_max_k.md)
  : Approximate top k

- [`lax_argmax()`](https://adamrsophie.github.io/jaxR/reference/lax_argmax.md)
  [`lax_argmin()`](https://adamrsophie.github.io/jaxR/reference/lax_argmax.md)
  : Index of the maximum along an axis

- [`lax_associative_scan()`](https://adamrsophie.github.io/jaxR/reference/lax_associative_scan.md)
  : Parallel prefix scan with an associative operation

- [`lax_batch_matmul()`](https://adamrsophie.github.io/jaxR/reference/lax_batch_matmul.md)
  : Batched matrix multiplication

- [`lax_bitcast_convert_type()`](https://adamrsophie.github.io/jaxR/reference/lax_bitcast_convert_type.md)
  : Bitcast an array to another dtype

- [`lax_broadcast()`](https://adamrsophie.github.io/jaxR/reference/lax_broadcast.md)
  : Add leading dimensions

- [`lax_broadcast_in_dim()`](https://adamrsophie.github.io/jaxR/reference/lax_broadcast_in_dim.md)
  : Broadcast an array into a target shape

- [`lax_broadcast_like()`](https://adamrsophie.github.io/jaxR/reference/lax_broadcast_like.md)
  : Broadcast an array to the shape of another

- [`lax_broadcast_shapes()`](https://adamrsophie.github.io/jaxR/reference/lax_broadcast_shapes.md)
  : Broadcast shapes (lax version)

- [`lax_broadcast_to_rank()`](https://adamrsophie.github.io/jaxR/reference/lax_broadcast_to_rank.md)
  : Add leading dimensions of size one

- [`lax_broadcasted_iota()`](https://adamrsophie.github.io/jaxR/reference/lax_broadcasted_iota.md)
  : Broadcasted iota

- [`lax_clamp()`](https://adamrsophie.github.io/jaxR/reference/lax_clamp.md)
  : Clamp values

- [`lax_collapse()`](https://adamrsophie.github.io/jaxR/reference/lax_collapse.md)
  : Collapse dimensions into one

- [`lax_psum()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
  [`lax_pmax()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
  [`lax_pmin()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
  [`lax_pmean()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
  [`lax_axis_index()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
  [`lax_axis_size()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
  [`lax_all_gather()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
  [`lax_psum_scatter()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
  [`lax_psum_scatter_start()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
  [`lax_ppermute()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
  [`lax_pshuffle()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
  [`lax_pbroadcast()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
  [`lax_all_to_all()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
  [`lax_pswapaxes()`](https://adamrsophie.github.io/jaxR/reference/lax_collectives.md)
  : Named axis collectives

- [`lax_composite()`](https://adamrsophie.github.io/jaxR/reference/lax_composite.md)
  : Composite operation

- [`lax_concatenate()`](https://adamrsophie.github.io/jaxR/reference/lax_concatenate.md)
  : Concatenate arrays

- [`lax_cond()`](https://adamrsophie.github.io/jaxR/reference/lax_cond.md)
  : Conditional execution

- [`lax_conv_dimension_numbers()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_dimension_numbers.md)
  : Convert convolution dimension numbers to canonical form

- [`lax_conv_general_dilated()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_general_dilated.md)
  : General n dimensional convolution

- [`lax_conv_general_permutations()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_general_permutations.md)
  : Convolution dimension permutations

- [`lax_conv_general_dilated_patches()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_patches.md)
  [`lax_conv_general_dilated_local()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_patches.md)
  : Image patches and unshared convolutions

- [`lax_conv_shape_tuple()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_shapes.md)
  [`lax_conv_general_shape_tuple()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_shapes.md)
  [`lax_conv_transpose_shape_tuple()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_shapes.md)
  : Convolution output shapes

- [`lax_conv()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_wrappers.md)
  [`lax_conv_with_general_padding()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_wrappers.md)
  [`lax_conv_transpose()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_wrappers.md)
  : Convolution wrappers

- [`lax_convert_element_type()`](https://adamrsophie.github.io/jaxR/reference/lax_convert_element_type.md)
  : Convert the dtype of an array

- [`lax_cumsum()`](https://adamrsophie.github.io/jaxR/reference/lax_cumulative.md)
  [`lax_cumprod()`](https://adamrsophie.github.io/jaxR/reference/lax_cumulative.md)
  [`lax_cummax()`](https://adamrsophie.github.io/jaxR/reference/lax_cumulative.md)
  [`lax_cummin()`](https://adamrsophie.github.io/jaxR/reference/lax_cumulative.md)
  [`lax_cumlogsumexp()`](https://adamrsophie.github.io/jaxR/reference/lax_cumulative.md)
  : Cumulative reductions

- [`lax_custom_linear_solve()`](https://adamrsophie.github.io/jaxR/reference/lax_custom_linear_solve.md)
  : Matrix free linear solve with implicit differentiation

- [`lax_custom_root()`](https://adamrsophie.github.io/jaxR/reference/lax_custom_root.md)
  : Differentiable root finding

- [`lax_dce_sink()`](https://adamrsophie.github.io/jaxR/reference/lax_dce_sink.md)
  : Mark values as used

- [`lax_DotDimensionNumbers()`](https://adamrsophie.github.io/jaxR/reference/lax_dimension_numbers.md)
  [`lax_ConvGeneralDilatedDimensionNumbers()`](https://adamrsophie.github.io/jaxR/reference/lax_dimension_numbers.md)
  [`lax_RaggedDotDimensionNumbers()`](https://adamrsophie.github.io/jaxR/reference/lax_dimension_numbers.md)
  : Dimension numbers of dot products and convolutions

- [`lax_dot()`](https://adamrsophie.github.io/jaxR/reference/lax_dot.md)
  : Matrix and vector products

- [`lax_dot_general()`](https://adamrsophie.github.io/jaxR/reference/lax_dot_general.md)
  : General dot product

- [`lax_dtype()`](https://adamrsophie.github.io/jaxR/reference/lax_dtype.md)
  : Dtype of a value

- [`lax_dynamic_slice()`](https://adamrsophie.github.io/jaxR/reference/lax_dynamic_slice.md)
  : Dynamic slice

- [`lax_dynamic_slice_in_dim()`](https://adamrsophie.github.io/jaxR/reference/lax_dynamic_slice_in_dim.md)
  [`lax_dynamic_index_in_dim()`](https://adamrsophie.github.io/jaxR/reference/lax_dynamic_slice_in_dim.md)
  [`lax_dynamic_update_slice_in_dim()`](https://adamrsophie.github.io/jaxR/reference/lax_dynamic_slice_in_dim.md)
  [`lax_dynamic_update_index_in_dim()`](https://adamrsophie.github.io/jaxR/reference/lax_dynamic_slice_in_dim.md)
  : Dynamic slices and updates along one axis

- [`lax_dynamic_update_slice()`](https://adamrsophie.github.io/jaxR/reference/lax_dynamic_update_slice.md)
  : Dynamic update slice

- [`lax_neg()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_sign()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_floor()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_ceil()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_round()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_is_finite()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_exp()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_exp2()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_log()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_log1p()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_expm1()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_tanh()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_logistic()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_sin()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_cos()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_tan()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_asin()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_acos()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_atan()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_sinh()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_cosh()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_asinh()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_acosh()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_atanh()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_sqrt()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_rsqrt()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_cbrt()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_abs()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_lgamma()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_digamma()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_erf()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_erfc()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_erf_inv()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_bitwise_not()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_population_count()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_clz()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_square()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_real()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_imag()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_conj()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_integer_pow()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_reduce_precision()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_add()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_sub()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_mul()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_div()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_rem()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_pow()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_max()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_min()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_atan2()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_nextafter()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_bitwise_and()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_bitwise_or()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_bitwise_xor()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_shift_left()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_shift_right_arithmetic()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_shift_right_logical()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_complex()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_eq()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_ne()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_lt()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_le()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_gt()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  [`lax_ge()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)
  : lax elementwise operations

- [`lax_log2()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise_extra.md)
  [`lax_mulhi()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise_extra.md)
  [`lax_one_minus_square()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise_extra.md)
  : More lax elementwise operations

- [`lax_empty()`](https://adamrsophie.github.io/jaxR/reference/lax_empty.md)
  : Array of unspecified values

- [`lax_Precision`](https://adamrsophie.github.io/jaxR/reference/lax_enums.md)
  [`lax_RoundingMethod`](https://adamrsophie.github.io/jaxR/reference/lax_enums.md)
  [`lax_AccuracyMode`](https://adamrsophie.github.io/jaxR/reference/lax_enums.md)
  [`lax_DotAlgorithmPreset`](https://adamrsophie.github.io/jaxR/reference/lax_enums.md)
  : Precision and algorithm settings of lax functions

- [`lax_expand_dims()`](https://adamrsophie.github.io/jaxR/reference/lax_expand_dims.md)
  : Insert size one dimensions

- [`lax_fft()`](https://adamrsophie.github.io/jaxR/reference/lax_fft.md)
  : Fast Fourier transform

- [`lax_fori_loop()`](https://adamrsophie.github.io/jaxR/reference/lax_fori_loop.md)
  : Loop from lower to upper

- [`lax_full()`](https://adamrsophie.github.io/jaxR/reference/lax_full.md)
  : Array filled with a value

- [`lax_full_like()`](https://adamrsophie.github.io/jaxR/reference/lax_full_like.md)
  : Array like x filled with a value

- [`lax_gather()`](https://adamrsophie.github.io/jaxR/reference/lax_gather.md)
  : Gather

- [`lax_index_in_dim()`](https://adamrsophie.github.io/jaxR/reference/lax_index_in_dim.md)
  : Index along one axis

- [`lax_index_take()`](https://adamrsophie.github.io/jaxR/reference/lax_index_take.md)
  : Take elements along several axes

- [`lax_iota()`](https://adamrsophie.github.io/jaxR/reference/lax_iota.md)
  : Iota (0, 1, 2, ...)

- [`lax_linalg_cholesky()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_cholesky_update()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_eig()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_eigh()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_hessenberg()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_householder_product()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_lu()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_lu_pivots_to_permutation()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_ormqr()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_qr()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_schur()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_svd()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_symmetric_product()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_triangular_solve()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_tridiagonal()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  [`lax_linalg_tridiagonal_solve()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg.md)
  : Linear algebra primitives

- [`lax_linalg_EigImplementation`](https://adamrsophie.github.io/jaxR/reference/lax_linalg_enums.md)
  [`lax_linalg_EighImplementation`](https://adamrsophie.github.io/jaxR/reference/lax_linalg_enums.md)
  [`lax_linalg_SvdAlgorithm`](https://adamrsophie.github.io/jaxR/reference/lax_linalg_enums.md)
  : Algorithm selectors of jax.lax.linalg

- [`lax_linalg_qdwh()`](https://adamrsophie.github.io/jaxR/reference/lax_linalg_qdwh.md)
  : QR-based dynamically weighted Halley iteration for the polar
  decomposition

- [`lax_map()`](https://adamrsophie.github.io/jaxR/reference/lax_map.md)
  : Map a function over leading array axes

- [`lax_optimization_barrier()`](https://adamrsophie.github.io/jaxR/reference/lax_optimization_barrier.md)
  : Optimization barrier

- [`lax_pad()`](https://adamrsophie.github.io/jaxR/reference/lax_pad.md)
  : Pad an array

- [`lax_padtype_to_pads()`](https://adamrsophie.github.io/jaxR/reference/lax_padtype_to_pads.md)
  : Convert a padding specification to explicit padding pairs

- [`lax_pcast()`](https://adamrsophie.github.io/jaxR/reference/lax_pcast.md)
  : Cast between replication types

- [`lax_platform_dependent()`](https://adamrsophie.github.io/jaxR/reference/lax_platform_dependent.md)
  : Platform dependent code

- [`lax_polynomial()`](https://adamrsophie.github.io/jaxR/reference/lax_polynomial.md)
  : Evaluate a polynomial

- [`lax_ragged_all_to_all()`](https://adamrsophie.github.io/jaxR/reference/lax_ragged_all_to_all.md)
  : Ragged all to all collective

- [`lax_ragged_dot_general()`](https://adamrsophie.github.io/jaxR/reference/lax_ragged_dot.md)
  [`lax_ragged_dot()`](https://adamrsophie.github.io/jaxR/reference/lax_ragged_dot.md)
  : Ragged dot products

- [`lax_reciprocal()`](https://adamrsophie.github.io/jaxR/reference/lax_reciprocal.md)
  : Elementwise reciprocal

- [`lax_reduce()`](https://adamrsophie.github.io/jaxR/reference/lax_reduce.md)
  : General reduction

- [`lax_reduce_window()`](https://adamrsophie.github.io/jaxR/reference/lax_reduce_window.md)
  : Reduction over padded windows

- [`lax_reduce_window_shape_tuple()`](https://adamrsophie.github.io/jaxR/reference/lax_reduce_window_shape_tuple.md)
  : Shape of a windowed reduction

- [`lax_reduce_sum()`](https://adamrsophie.github.io/jaxR/reference/lax_reductions.md)
  [`lax_reduce_prod()`](https://adamrsophie.github.io/jaxR/reference/lax_reductions.md)
  [`lax_reduce_max()`](https://adamrsophie.github.io/jaxR/reference/lax_reductions.md)
  [`lax_reduce_min()`](https://adamrsophie.github.io/jaxR/reference/lax_reductions.md)
  [`lax_reduce_and()`](https://adamrsophie.github.io/jaxR/reference/lax_reductions.md)
  [`lax_reduce_or()`](https://adamrsophie.github.io/jaxR/reference/lax_reductions.md)
  [`lax_reduce_xor()`](https://adamrsophie.github.io/jaxR/reference/lax_reductions.md)
  : lax reductions

- [`lax_reshape()`](https://adamrsophie.github.io/jaxR/reference/lax_reshape.md)
  : Reshape an array (C order)

- [`lax_rev()`](https://adamrsophie.github.io/jaxR/reference/lax_rev.md)
  : Reverse dimensions

- [`lax_rng_bit_generator()`](https://adamrsophie.github.io/jaxR/reference/lax_rng_bit_generator.md)
  [`lax_rng_bit_generator_p`](https://adamrsophie.github.io/jaxR/reference/lax_rng_bit_generator.md)
  : Stateless random bit generator

- [`lax_rng_uniform()`](https://adamrsophie.github.io/jaxR/reference/lax_rng_uniform.md)
  [`lax_rng_uniform_p`](https://adamrsophie.github.io/jaxR/reference/lax_rng_uniform.md)
  : Stateful uniform random numbers

- [`lax_scaled_dot()`](https://adamrsophie.github.io/jaxR/reference/lax_scaled_dot.md)
  : Scaled dot product

- [`lax_scan()`](https://adamrsophie.github.io/jaxR/reference/lax_scan.md)
  : Scan a function over leading array axes while carrying state

- [`lax_scatter()`](https://adamrsophie.github.io/jaxR/reference/lax_scatter.md)
  [`lax_scatter_add()`](https://adamrsophie.github.io/jaxR/reference/lax_scatter.md)
  [`lax_scatter_sub()`](https://adamrsophie.github.io/jaxR/reference/lax_scatter.md)
  [`lax_scatter_mul()`](https://adamrsophie.github.io/jaxR/reference/lax_scatter.md)
  [`lax_scatter_min()`](https://adamrsophie.github.io/jaxR/reference/lax_scatter.md)
  [`lax_scatter_max()`](https://adamrsophie.github.io/jaxR/reference/lax_scatter.md)
  [`lax_scatter_apply()`](https://adamrsophie.github.io/jaxR/reference/lax_scatter.md)
  : Scatter

- [`lax_select()`](https://adamrsophie.github.io/jaxR/reference/lax_select.md)
  : Elementwise select

- [`lax_select_n()`](https://adamrsophie.github.io/jaxR/reference/lax_select_n.md)
  : Select between cases

- [`lax_shape_as_value()`](https://adamrsophie.github.io/jaxR/reference/lax_shape_as_value.md)
  : Shape as an array

- [`lax_slice()`](https://adamrsophie.github.io/jaxR/reference/lax_slice.md)
  : Static slice

- [`lax_slice_in_dim()`](https://adamrsophie.github.io/jaxR/reference/lax_slice_in_dim.md)
  : Slice along one axis

- [`lax_sort()`](https://adamrsophie.github.io/jaxR/reference/lax_sort.md)
  : Sort arrays

- [`lax_sort_key_val()`](https://adamrsophie.github.io/jaxR/reference/lax_sort_key_val.md)
  : Sort keys and permute values alike

- [`lax_betainc()`](https://adamrsophie.github.io/jaxR/reference/lax_special.md)
  [`lax_polygamma()`](https://adamrsophie.github.io/jaxR/reference/lax_special.md)
  [`lax_igamma()`](https://adamrsophie.github.io/jaxR/reference/lax_special.md)
  [`lax_igammac()`](https://adamrsophie.github.io/jaxR/reference/lax_special.md)
  [`lax_igamma_grad_a()`](https://adamrsophie.github.io/jaxR/reference/lax_special.md)
  [`lax_random_gamma_grad()`](https://adamrsophie.github.io/jaxR/reference/lax_special.md)
  [`lax_zeta()`](https://adamrsophie.github.io/jaxR/reference/lax_special.md)
  [`lax_bessel_i0e()`](https://adamrsophie.github.io/jaxR/reference/lax_special.md)
  [`lax_bessel_i1e()`](https://adamrsophie.github.io/jaxR/reference/lax_special.md)
  : lax special functions

- [`lax_split()`](https://adamrsophie.github.io/jaxR/reference/lax_split.md)
  : Split an array

- [`lax_squeeze()`](https://adamrsophie.github.io/jaxR/reference/lax_squeeze.md)
  : Remove size one dimensions

- [`lax_stack()`](https://adamrsophie.github.io/jaxR/reference/lax_stack.md)
  [`lax_unstack()`](https://adamrsophie.github.io/jaxR/reference/lax_stack.md)
  : Stack and unstack arrays

- [`lax_stage()`](https://adamrsophie.github.io/jaxR/reference/lax_stage.md)
  : Lift a value into the current trace

- [`lax_stop_gradient()`](https://adamrsophie.github.io/jaxR/reference/lax_stop_gradient.md)
  [`stop_gradient()`](https://adamrsophie.github.io/jaxR/reference/lax_stop_gradient.md)
  : Stop gradients

- [`lax_switch()`](https://adamrsophie.github.io/jaxR/reference/lax_switch.md)
  : Select and apply one of several branches

- [`lax_tile()`](https://adamrsophie.github.io/jaxR/reference/lax_tile.md)
  : Tile an array

- [`lax_create_token()`](https://adamrsophie.github.io/jaxR/reference/lax_tokens.md)
  [`lax_after_all()`](https://adamrsophie.github.io/jaxR/reference/lax_tokens.md)
  : Tokens for ordering side effects

- [`lax_top_k()`](https://adamrsophie.github.io/jaxR/reference/lax_top_k.md)
  : Top k values and their indices

- [`lax_transpose()`](https://adamrsophie.github.io/jaxR/reference/lax_transpose.md)
  : Permute dimensions

- [`lax_while_loop()`](https://adamrsophie.github.io/jaxR/reference/lax_while_loop.md)
  : While loop

- [`linear_call()`](https://adamrsophie.github.io/jaxR/reference/linear_call.md)
  : Linear function with a custom transpose

- [`linear_transpose()`](https://adamrsophie.github.io/jaxR/reference/linear_transpose.md)
  : Transpose a linear function

- [`linearize()`](https://adamrsophie.github.io/jaxR/reference/linearize.md)
  : Linearize a function

- [`live_arrays()`](https://adamrsophie.github.io/jaxR/reference/live_arrays.md)
  : Live arrays

- [`lower_text()`](https://adamrsophie.github.io/jaxR/reference/lower_text.md)
  : Lower a function to StableHLO text

- [`make_jaxpr()`](https://adamrsophie.github.io/jaxR/reference/make_jaxpr.md)
  : Trace a function to a jaxpr

- [`make_user_context()`](https://adamrsophie.github.io/jaxR/reference/make_user_context.md)
  : User contexts that affect jit caching

- [`monitoring_record_event()`](https://adamrsophie.github.io/jaxR/reference/monitoring.md)
  [`monitoring_record_event_duration_secs()`](https://adamrsophie.github.io/jaxR/reference/monitoring.md)
  [`monitoring_record_event_time_span()`](https://adamrsophie.github.io/jaxR/reference/monitoring.md)
  [`monitoring_record_scalar()`](https://adamrsophie.github.io/jaxR/reference/monitoring.md)
  [`monitoring_register_event_listener()`](https://adamrsophie.github.io/jaxR/reference/monitoring.md)
  [`monitoring_register_event_duration_secs_listener()`](https://adamrsophie.github.io/jaxR/reference/monitoring.md)
  [`monitoring_register_event_time_span_listener()`](https://adamrsophie.github.io/jaxR/reference/monitoring.md)
  [`monitoring_register_scalar_listener()`](https://adamrsophie.github.io/jaxR/reference/monitoring.md)
  [`monitoring_clear_event_listeners()`](https://adamrsophie.github.io/jaxR/reference/monitoring.md)
  [`monitoring_unregister_event_listener()`](https://adamrsophie.github.io/jaxR/reference/monitoring.md)
  [`monitoring_unregister_event_duration_listener()`](https://adamrsophie.github.io/jaxR/reference/monitoring.md)
  [`monitoring_unregister_event_time_span_listener()`](https://adamrsophie.github.io/jaxR/reference/monitoring.md)
  [`monitoring_unregister_scalar_listener()`](https://adamrsophie.github.io/jaxR/reference/monitoring.md)
  : Instrumentation events

- [`named_call()`](https://adamrsophie.github.io/jaxR/reference/named_call.md)
  [`named_scope()`](https://adamrsophie.github.io/jaxR/reference/named_call.md)
  : Name a function or a scope of staged computations

- [`nn_identity()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_relu()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_squareplus()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_softplus()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_sparse_plus()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_soft_sign()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_sigmoid()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_sparse_sigmoid()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_silu()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_swish()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_mish()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_log_sigmoid()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_elu()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_leaky_relu()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_hard_tanh()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_celu()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_selu()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_gelu()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_glu()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_relu6()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_hard_sigmoid()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_hard_silu()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_hard_swish()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_log1mexp()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  [`nn_tanh()`](https://adamrsophie.github.io/jaxR/reference/nn_activations.md)
  : Activation functions

- [`nn_dot_product_attention()`](https://adamrsophie.github.io/jaxR/reference/nn_dot_product_attention.md)
  : Scaled dot product attention

- [`nn_initializers_zeros()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_ones()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_constant()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_uniform()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_normal()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_truncated_normal()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_variance_scaling()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_glorot_uniform()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_xavier_uniform()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_glorot_normal()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_xavier_normal()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_lecun_uniform()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_lecun_normal()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_he_uniform()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_kaiming_uniform()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_he_normal()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_kaiming_normal()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_orthogonal()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  [`nn_initializers_delta_orthogonal()`](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)
  : Neural network initializers

- [`nn_one_hot()`](https://adamrsophie.github.io/jaxR/reference/nn_one_hot.md)
  : One-hot encoding

- [`nn_scaled_matmul()`](https://adamrsophie.github.io/jaxR/reference/nn_scaled_matmul.md)
  [`nn_scaled_dot_general()`](https://adamrsophie.github.io/jaxR/reference/nn_scaled_matmul.md)
  : Scaled matrix multiplication

- [`nn_logsumexp()`](https://adamrsophie.github.io/jaxR/reference/nn_softmax.md)
  [`nn_logmeanexp()`](https://adamrsophie.github.io/jaxR/reference/nn_softmax.md)
  [`nn_log_softmax()`](https://adamrsophie.github.io/jaxR/reference/nn_softmax.md)
  [`nn_softmax()`](https://adamrsophie.github.io/jaxR/reference/nn_softmax.md)
  [`nn_standardize()`](https://adamrsophie.github.io/jaxR/reference/nn_softmax.md)
  : Softmax, log-sum-exp and standardization

- [`numpy_rank_promotion()`](https://adamrsophie.github.io/jaxR/reference/numpy_rank_promotion.md)
  [`default_matmul_precision()`](https://adamrsophie.github.io/jaxR/reference/numpy_rank_promotion.md)
  [`threefry_partitionable()`](https://adamrsophie.github.io/jaxR/reference/numpy_rank_promotion.md)
  [`default_prng_impl()`](https://adamrsophie.github.io/jaxR/reference/numpy_rank_promotion.md)
  : Run code with a configuration option set

- [`odeint()`](https://adamrsophie.github.io/jaxR/reference/odeint.md)
  [`experimental_ode_odeint()`](https://adamrsophie.github.io/jaxR/reference/odeint.md)
  : Integrate a system of ordinary differential equations

- [`ops_segment_sum()`](https://adamrsophie.github.io/jaxR/reference/ops_segment_sum.md)
  [`ops_segment_prod()`](https://adamrsophie.github.io/jaxR/reference/ops_segment_sum.md)
  [`ops_segment_max()`](https://adamrsophie.github.io/jaxR/reference/ops_segment_sum.md)
  [`ops_segment_min()`](https://adamrsophie.github.io/jaxR/reference/ops_segment_sum.md)
  : Segment reductions

- [`optimizers_sgd()`](https://adamrsophie.github.io/jaxR/reference/optimizers.md)
  [`optimizers_momentum()`](https://adamrsophie.github.io/jaxR/reference/optimizers.md)
  [`optimizers_nesterov()`](https://adamrsophie.github.io/jaxR/reference/optimizers.md)
  [`optimizers_adagrad()`](https://adamrsophie.github.io/jaxR/reference/optimizers.md)
  [`optimizers_rmsprop()`](https://adamrsophie.github.io/jaxR/reference/optimizers.md)
  [`optimizers_rmsprop_momentum()`](https://adamrsophie.github.io/jaxR/reference/optimizers.md)
  [`optimizers_adam()`](https://adamrsophie.github.io/jaxR/reference/optimizers.md)
  [`optimizers_adamax()`](https://adamrsophie.github.io/jaxR/reference/optimizers.md)
  [`optimizers_sm3()`](https://adamrsophie.github.io/jaxR/reference/optimizers.md)
  : Optimizers

- [`optimizers_constant()`](https://adamrsophie.github.io/jaxR/reference/optimizers_schedules.md)
  [`optimizers_exponential_decay()`](https://adamrsophie.github.io/jaxR/reference/optimizers_schedules.md)
  [`optimizers_inverse_time_decay()`](https://adamrsophie.github.io/jaxR/reference/optimizers_schedules.md)
  [`optimizers_polynomial_decay()`](https://adamrsophie.github.io/jaxR/reference/optimizers_schedules.md)
  [`optimizers_piecewise_constant()`](https://adamrsophie.github.io/jaxR/reference/optimizers_schedules.md)
  [`optimizers_make_schedule()`](https://adamrsophie.github.io/jaxR/reference/optimizers_schedules.md)
  : Learning rate schedules

- [`optimizers_JoinPoint()`](https://adamrsophie.github.io/jaxR/reference/optimizers_serialization.md)
  [`optimizers_unpack_optimizer_state()`](https://adamrsophie.github.io/jaxR/reference/optimizers_serialization.md)
  [`optimizers_pack_optimizer_state()`](https://adamrsophie.github.io/jaxR/reference/optimizers_serialization.md)
  : Optimizer state serialization

- [`optimizers_OptimizerState()`](https://adamrsophie.github.io/jaxR/reference/optimizers_state.md)
  [`optimizers_Optimizer()`](https://adamrsophie.github.io/jaxR/reference/optimizers_state.md)
  [`optimizers_optimizer()`](https://adamrsophie.github.io/jaxR/reference/optimizers_state.md)
  : Optimizer state and triple

- [`optimizers_l2_norm()`](https://adamrsophie.github.io/jaxR/reference/optimizers_utils.md)
  [`optimizers_clip_grads()`](https://adamrsophie.github.io/jaxR/reference/optimizers_utils.md)
  : Gradient utilities

- [`print_environment_info()`](https://adamrsophie.github.io/jaxR/reference/print_environment_info.md)
  : Print information about the jaxR installation

- [`pure_callback()`](https://adamrsophie.github.io/jaxR/reference/pure_callback.md)
  : Call a pure R function from compiled code

- [`random_cauchy()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  [`random_exponential()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  [`random_gumbel()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  [`random_laplace()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  [`random_logistic()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  [`random_pareto()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  [`random_rayleigh()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  [`random_wald()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  [`random_maxwell()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  [`random_double_sided_maxwell()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  [`random_weibull_min()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  [`random_triangular()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  [`random_generalized_normal()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  [`random_ball()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  [`random_orthogonal()`](https://adamrsophie.github.io/jaxR/reference/random_continuous.md)
  : Continuous random values

- [`random_bernoulli()`](https://adamrsophie.github.io/jaxR/reference/random_discrete.md)
  [`random_rademacher()`](https://adamrsophie.github.io/jaxR/reference/random_discrete.md)
  [`random_categorical()`](https://adamrsophie.github.io/jaxR/reference/random_discrete.md)
  [`random_geometric()`](https://adamrsophie.github.io/jaxR/reference/random_discrete.md)
  [`random_poisson()`](https://adamrsophie.github.io/jaxR/reference/random_discrete.md)
  [`random_binomial()`](https://adamrsophie.github.io/jaxR/reference/random_discrete.md)
  [`random_multinomial()`](https://adamrsophie.github.io/jaxR/reference/random_discrete.md)
  : Discrete random values

- [`random_gamma()`](https://adamrsophie.github.io/jaxR/reference/random_gamma.md)
  [`random_loggamma()`](https://adamrsophie.github.io/jaxR/reference/random_gamma.md)
  [`random_random_gamma_p`](https://adamrsophie.github.io/jaxR/reference/random_gamma.md)
  [`random_beta()`](https://adamrsophie.github.io/jaxR/reference/random_gamma.md)
  [`random_dirichlet()`](https://adamrsophie.github.io/jaxR/reference/random_gamma.md)
  [`random_chisquare()`](https://adamrsophie.github.io/jaxR/reference/random_gamma.md)
  [`random_f()`](https://adamrsophie.github.io/jaxR/reference/random_gamma.md)
  [`random_t()`](https://adamrsophie.github.io/jaxR/reference/random_gamma.md)
  : Gamma family random values

- [`random_key()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  [`random_PRNGKey()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  [`random_fold_in()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  [`random_split()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  [`random_key_impl()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  [`random_key_data()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  [`random_wrap_key_data()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  [`random_key_dtype()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  [`random_clone()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  [`random_Key()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  : PRNG keys

- [`random_normal()`](https://adamrsophie.github.io/jaxR/reference/random_normal.md)
  [`random_multivariate_normal()`](https://adamrsophie.github.io/jaxR/reference/random_normal.md)
  [`random_truncated_normal()`](https://adamrsophie.github.io/jaxR/reference/random_normal.md)
  [`random_lognormal()`](https://adamrsophie.github.io/jaxR/reference/random_normal.md)
  : Normal random values

- [`random_permutation()`](https://adamrsophie.github.io/jaxR/reference/random_permutation.md)
  [`random_choice()`](https://adamrsophie.github.io/jaxR/reference/random_permutation.md)
  : Random permutations and choices

- [`random_bits()`](https://adamrsophie.github.io/jaxR/reference/random_uniform.md)
  [`random_uniform()`](https://adamrsophie.github.io/jaxR/reference/random_uniform.md)
  [`random_randint()`](https://adamrsophie.github.io/jaxR/reference/random_uniform.md)
  : Uniform random bits, floats and integers

- [`ravel_pytree()`](https://adamrsophie.github.io/jaxR/reference/ravel_pytree.md)
  : Flatten a pytree of arrays into one vector

- [`new_ref()`](https://adamrsophie.github.io/jaxR/reference/refs.md)
  [`empty_ref()`](https://adamrsophie.github.io/jaxR/reference/refs.md)
  [`free_ref()`](https://adamrsophie.github.io/jaxR/reference/refs.md)
  [`freeze()`](https://adamrsophie.github.io/jaxR/reference/refs.md)
  [`Ref()`](https://adamrsophie.github.io/jaxR/reference/refs.md)
  [`ds()`](https://adamrsophie.github.io/jaxR/reference/refs.md) :
  Mutable array references

- [`register_dataclass()`](https://adamrsophie.github.io/jaxR/reference/register_dataclass.md)
  : Register an S3 list class with fixed fields as a pytree node

- [`register_pytree_node()`](https://adamrsophie.github.io/jaxR/reference/register_pytree_node.md)
  : Register a class as a pytree node

- [`register_pytree_node_class()`](https://adamrsophie.github.io/jaxR/reference/register_pytree_node_class.md)
  [`register_pytree_with_keys_class()`](https://adamrsophie.github.io/jaxR/reference/register_pytree_node_class.md)
  : Register a class through its flatten methods

- [`register_pytree_with_keys()`](https://adamrsophie.github.io/jaxR/reference/register_pytree_with_keys.md)
  : Register a pytree node class with key paths

- [`register_static()`](https://adamrsophie.github.io/jaxR/reference/register_static.md)
  : Register a class as a static pytree node

- [`Saveable`](https://adamrsophie.github.io/jaxR/reference/remat_policy_values.md)
  [`Recompute`](https://adamrsophie.github.io/jaxR/reference/remat_policy_values.md)
  [`Offloadable()`](https://adamrsophie.github.io/jaxR/reference/remat_policy_values.md)
  : Rematerialization decisions of checkpoint policies

- [`result_type()`](https://adamrsophie.github.io/jaxR/reference/result_type.md)
  [`promote_types_dtype()`](https://adamrsophie.github.io/jaxR/reference/result_type.md)
  : Result dtype of values under JAX type promotion

- [`saved_residuals()`](https://adamrsophie.github.io/jaxR/reference/saved_residuals.md)
  [`print_saved_residuals()`](https://adamrsophie.github.io/jaxR/reference/saved_residuals.md)
  : Residuals saved for reverse mode differentiation

- [`scipy_cluster_vq_vq()`](https://adamrsophie.github.io/jaxR/reference/scipy_cluster_vq_vq.md)
  : Vector quantization

- [`scipy_fft_dct()`](https://adamrsophie.github.io/jaxR/reference/scipy_fft_dct.md)
  [`scipy_fft_dctn()`](https://adamrsophie.github.io/jaxR/reference/scipy_fft_dct.md)
  [`scipy_fft_idct()`](https://adamrsophie.github.io/jaxR/reference/scipy_fft_dct.md)
  [`scipy_fft_idctn()`](https://adamrsophie.github.io/jaxR/reference/scipy_fft_dct.md)
  : Discrete cosine transforms

- [`scipy_integrate_trapezoid()`](https://adamrsophie.github.io/jaxR/reference/scipy_integrate_trapezoid.md)
  : Integrate along an axis with the trapezoid rule

- [`scipy_interpolate_RegularGridInterpolator()`](https://adamrsophie.github.io/jaxR/reference/scipy_interpolate_RegularGridInterpolator.md)
  [`` `$`( ``*`<scipy_interpolate_RegularGridInterpolator>`*`)`](https://adamrsophie.github.io/jaxR/reference/scipy_interpolate_RegularGridInterpolator.md)
  : Interpolation on a regular grid

- [`scipy_linalg_cholesky()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_cho_factor()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_cho_solve()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_svd()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_det()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_eigh()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_schur()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_inv()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_lu_factor()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_lu_solve()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_lu()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_qr()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_qr_multiply()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_solve()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_solve_triangular()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  [`scipy_linalg_hessenberg()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg.md)
  : Linear algebra (jax.scipy.linalg)

- [`scipy_linalg_eigh_tridiagonal()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_eigh_tridiagonal.md)
  : Eigenvalues of a symmetric tridiagonal matrix

- [`scipy_linalg_expm()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_matrix_functions.md)
  [`scipy_linalg_expm_frechet()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_matrix_functions.md)
  [`scipy_linalg_polar()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_matrix_functions.md)
  [`scipy_linalg_sqrtm()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_matrix_functions.md)
  [`scipy_linalg_rsf2csf()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_matrix_functions.md)
  [`scipy_linalg_funm()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_matrix_functions.md)
  [`scipy_linalg_solve_sylvester()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_matrix_functions.md)
  : Matrix functions (jax.scipy.linalg)

- [`scipy_linalg_block_diag()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_toeplitz()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_hankel()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_circulant()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_leslie()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_companion()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_fiedler()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_fiedler_companion()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_convolution_matrix()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_hilbert()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_invhilbert()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_pascal()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_invpascal()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_helmert()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_hadamard()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  [`scipy_linalg_dft()`](https://adamrsophie.github.io/jaxR/reference/scipy_linalg_special_matrices.md)
  : Special matrices (jax.scipy.linalg)

- [`scipy_ndimage_map_coordinates()`](https://adamrsophie.github.io/jaxR/reference/scipy_ndimage_map_coordinates.md)
  : Map an array to new coordinates by interpolation

- [`scipy_optimize_OptimizeResults()`](https://adamrsophie.github.io/jaxR/reference/scipy_optimize_OptimizeResults.md)
  : Results of an optimization

- [`scipy_optimize_line_search()`](https://adamrsophie.github.io/jaxR/reference/scipy_optimize_line_search.md)
  : Inexact line search satisfying the strong Wolfe conditions

- [`scipy_optimize_minimize()`](https://adamrsophie.github.io/jaxR/reference/scipy_optimize_minimize.md)
  : Minimize a scalar function of one or more variables

- [`scipy_signal_fftconvolve()`](https://adamrsophie.github.io/jaxR/reference/scipy_signal_convolve.md)
  [`scipy_signal_convolve()`](https://adamrsophie.github.io/jaxR/reference/scipy_signal_convolve.md)
  [`scipy_signal_convolve2d()`](https://adamrsophie.github.io/jaxR/reference/scipy_signal_convolve.md)
  [`scipy_signal_correlate()`](https://adamrsophie.github.io/jaxR/reference/scipy_signal_convolve.md)
  [`scipy_signal_correlate2d()`](https://adamrsophie.github.io/jaxR/reference/scipy_signal_convolve.md)
  : Convolution and correlation of N-dimensional arrays

- [`scipy_signal_detrend()`](https://adamrsophie.github.io/jaxR/reference/scipy_signal_detrend.md)
  : Remove a linear or constant trend

- [`scipy_signal_stft()`](https://adamrsophie.github.io/jaxR/reference/scipy_signal_spectral.md)
  [`scipy_signal_csd()`](https://adamrsophie.github.io/jaxR/reference/scipy_signal_spectral.md)
  [`scipy_signal_welch()`](https://adamrsophie.github.io/jaxR/reference/scipy_signal_spectral.md)
  [`scipy_signal_istft()`](https://adamrsophie.github.io/jaxR/reference/scipy_signal_spectral.md)
  : Spectral analysis: STFT, inverse STFT, cross and power spectral
  densities

- [`scipy_sparse_linalg_cg()`](https://adamrsophie.github.io/jaxR/reference/scipy_sparse_linalg.md)
  [`scipy_sparse_linalg_bicgstab()`](https://adamrsophie.github.io/jaxR/reference/scipy_sparse_linalg.md)
  [`scipy_sparse_linalg_gmres()`](https://adamrsophie.github.io/jaxR/reference/scipy_sparse_linalg.md)
  : Iterative linear solvers (jax.scipy.sparse.linalg)

- [`scipy_spatial_transform_Rotation()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_concatenate()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_from_euler()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_from_matrix()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_from_mrp()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_from_quat()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_from_rotvec()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_identity()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_random()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`` `[`( ``*`<scipy_spatial_transform_Rotation>`*`)`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_len()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`` `*`( ``*`<scipy_spatial_transform_Rotation>`*`)`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_apply()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_as_euler()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_as_matrix()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_as_mrp()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_as_rotvec()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_as_quat()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_inv()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_magnitude()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_mean()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  [`scipy_spatial_transform_Rotation_single()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  : Rotations in three dimensions

- [`scipy_spatial_transform_Slerp()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Slerp.md)
  [`` `$`( ``*`<scipy_spatial_transform_Slerp>`*`)`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Slerp.md)
  [`scipy_spatial_transform_Slerp_init()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Slerp.md)
  : Spherical linear interpolation of rotations

- [`scipy_special_gammaln()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_gammasgn()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_loggamma()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_gamma()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_betaln()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_factorial()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_comb()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_beta()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_betainc()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_digamma()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_gammainc()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_gammaincc()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_multigammaln()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_poch()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_polygamma()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_zeta()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_erf()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_erfc()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_erfinv()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_logit()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_expit()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_xlogy()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_xlog1py()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_entr()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_boxcox()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_boxcox1p()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_kl_div()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_rel_entr()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_logsumexp()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_softmax()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_log_softmax()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_ndtr()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_log_ndtr()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_i0e()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_i1e()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_i0()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_i1()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_bessel_jn()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_dawsn()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_wofz()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_expi()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_expn()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_exp1()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_sici()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_fresnel()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_owens_t()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_spence()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_bernoulli()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_erfcx()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_erfcinv()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_ndtri()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_hyp1f1()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_hyp2f1()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_lpmn()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_lpmn_values()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  [`scipy_special_sph_harm_y()`](https://adamrsophie.github.io/jaxR/reference/scipy_special.md)
  : Special functions

- [`scipy_stats_bernoulli_logpmf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_bernoulli.md)
  [`scipy_stats_bernoulli_pmf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_bernoulli.md)
  [`scipy_stats_bernoulli_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_bernoulli.md)
  [`scipy_stats_bernoulli_ppf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_bernoulli.md)
  : Bernoulli distribution

- [`scipy_stats_beta_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_beta.md)
  [`scipy_stats_beta_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_beta.md)
  [`scipy_stats_beta_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_beta.md)
  [`scipy_stats_beta_logcdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_beta.md)
  [`scipy_stats_beta_sf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_beta.md)
  [`scipy_stats_beta_logsf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_beta.md)
  : Beta distribution

- [`scipy_stats_betabinom_logpmf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_betabinom.md)
  [`scipy_stats_betabinom_pmf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_betabinom.md)
  : Beta binomial distribution

- [`scipy_stats_binom_logpmf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_binom.md)
  [`scipy_stats_binom_pmf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_binom.md)
  : Binomial distribution

- [`scipy_stats_cauchy_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_cauchy.md)
  [`scipy_stats_cauchy_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_cauchy.md)
  [`scipy_stats_cauchy_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_cauchy.md)
  [`scipy_stats_cauchy_logcdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_cauchy.md)
  [`scipy_stats_cauchy_sf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_cauchy.md)
  [`scipy_stats_cauchy_logsf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_cauchy.md)
  [`scipy_stats_cauchy_isf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_cauchy.md)
  [`scipy_stats_cauchy_ppf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_cauchy.md)
  : Cauchy distribution

- [`scipy_stats_chi2_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_chi2.md)
  [`scipy_stats_chi2_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_chi2.md)
  [`scipy_stats_chi2_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_chi2.md)
  [`scipy_stats_chi2_logcdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_chi2.md)
  [`scipy_stats_chi2_sf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_chi2.md)
  [`scipy_stats_chi2_logsf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_chi2.md)
  : Chi-squared distribution

- [`scipy_stats_vonmises_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_circular.md)
  [`scipy_stats_vonmises_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_circular.md)
  [`scipy_stats_wrapcauchy_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_circular.md)
  [`scipy_stats_wrapcauchy_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_circular.md)
  : Circular distributions

- [`scipy_stats_dirichlet_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_dirichlet.md)
  [`scipy_stats_dirichlet_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_dirichlet.md)
  : Dirichlet distribution

- [`scipy_stats_expon_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_expon.md)
  [`scipy_stats_expon_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_expon.md)
  [`scipy_stats_expon_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_expon.md)
  [`scipy_stats_expon_logcdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_expon.md)
  [`scipy_stats_expon_logsf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_expon.md)
  [`scipy_stats_expon_sf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_expon.md)
  [`scipy_stats_expon_ppf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_expon.md)
  : Exponential distribution

- [`scipy_stats_gamma_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gamma.md)
  [`scipy_stats_gamma_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gamma.md)
  [`scipy_stats_gamma_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gamma.md)
  [`scipy_stats_gamma_logcdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gamma.md)
  [`scipy_stats_gamma_sf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gamma.md)
  [`scipy_stats_gamma_logsf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gamma.md)
  : Gamma distribution

- [`scipy_stats_gaussian_kde()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gaussian_kde.md)
  : Gaussian kernel density estimation

- [`scipy_stats_gennorm_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gennorm.md)
  [`scipy_stats_gennorm_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gennorm.md)
  [`scipy_stats_gennorm_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gennorm.md)
  : Generalized normal distribution

- [`scipy_stats_geom_logpmf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_geom.md)
  [`scipy_stats_geom_pmf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_geom.md)
  : Geometric distribution

- [`scipy_stats_gumbel_l_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gumbel_l.md)
  [`scipy_stats_gumbel_l_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gumbel_l.md)
  [`scipy_stats_gumbel_l_logcdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gumbel_l.md)
  [`scipy_stats_gumbel_l_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gumbel_l.md)
  [`scipy_stats_gumbel_l_ppf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gumbel_l.md)
  [`scipy_stats_gumbel_l_sf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gumbel_l.md)
  [`scipy_stats_gumbel_l_logsf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gumbel_l.md)
  : Left skewed Gumbel distribution

- [`scipy_stats_gumbel_r_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gumbel_r.md)
  [`scipy_stats_gumbel_r_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gumbel_r.md)
  [`scipy_stats_gumbel_r_logcdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gumbel_r.md)
  [`scipy_stats_gumbel_r_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gumbel_r.md)
  [`scipy_stats_gumbel_r_ppf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gumbel_r.md)
  [`scipy_stats_gumbel_r_sf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gumbel_r.md)
  [`scipy_stats_gumbel_r_logsf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_gumbel_r.md)
  : Right skewed Gumbel distribution

- [`scipy_stats_laplace_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_laplace.md)
  [`scipy_stats_laplace_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_laplace.md)
  [`scipy_stats_laplace_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_laplace.md)
  : Laplace distribution

- [`scipy_stats_logistic_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_logistic.md)
  [`scipy_stats_logistic_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_logistic.md)
  [`scipy_stats_logistic_ppf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_logistic.md)
  [`scipy_stats_logistic_sf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_logistic.md)
  [`scipy_stats_logistic_isf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_logistic.md)
  [`scipy_stats_logistic_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_logistic.md)
  : Logistic distribution

- [`scipy_stats_multinomial_logpmf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_multinomial.md)
  [`scipy_stats_multinomial_pmf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_multinomial.md)
  : Multinomial distribution

- [`scipy_stats_multivariate_normal_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_multivariate_normal.md)
  [`scipy_stats_multivariate_normal_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_multivariate_normal.md)
  : Multivariate normal distribution

- [`scipy_stats_nbinom_logpmf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_nbinom.md)
  [`scipy_stats_nbinom_pmf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_nbinom.md)
  : Negative binomial distribution

- [`scipy_stats_norm_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_norm.md)
  [`scipy_stats_norm_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_norm.md)
  [`scipy_stats_norm_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_norm.md)
  [`scipy_stats_norm_logcdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_norm.md)
  [`scipy_stats_norm_ppf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_norm.md)
  [`scipy_stats_norm_logsf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_norm.md)
  [`scipy_stats_norm_sf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_norm.md)
  [`scipy_stats_norm_isf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_norm.md)
  : Normal distribution

- [`scipy_stats_pareto_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_pareto.md)
  [`scipy_stats_pareto_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_pareto.md)
  [`scipy_stats_pareto_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_pareto.md)
  [`scipy_stats_pareto_logcdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_pareto.md)
  [`scipy_stats_pareto_logsf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_pareto.md)
  [`scipy_stats_pareto_sf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_pareto.md)
  [`scipy_stats_pareto_ppf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_pareto.md)
  : Pareto distribution

- [`scipy_stats_poisson_logpmf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_poisson.md)
  [`scipy_stats_poisson_pmf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_poisson.md)
  [`scipy_stats_poisson_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_poisson.md)
  [`scipy_stats_poisson_entropy()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_poisson.md)
  : Poisson distribution

- [`scipy_stats_mode()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_summary.md)
  [`scipy_stats_rankdata()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_summary.md)
  [`scipy_stats_sem()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_summary.md)
  : Summary statistics

- [`scipy_stats_t_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_t.md)
  [`scipy_stats_t_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_t.md)
  : Student t distribution

- [`scipy_stats_truncnorm_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_truncnorm.md)
  [`scipy_stats_truncnorm_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_truncnorm.md)
  [`scipy_stats_truncnorm_logsf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_truncnorm.md)
  [`scipy_stats_truncnorm_sf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_truncnorm.md)
  [`scipy_stats_truncnorm_logcdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_truncnorm.md)
  [`scipy_stats_truncnorm_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_truncnorm.md)
  : Truncated normal distribution

- [`scipy_stats_uniform_logpdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_uniform.md)
  [`scipy_stats_uniform_pdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_uniform.md)
  [`scipy_stats_uniform_cdf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_uniform.md)
  [`scipy_stats_uniform_ppf()`](https://adamrsophie.github.io/jaxR/reference/scipy_stats_uniform.md)
  : Uniform distribution

- [`sl()`](https://adamrsophie.github.io/jaxR/reference/sl.md)
  [`newaxis`](https://adamrsophie.github.io/jaxR/reference/sl.md)
  [`ellipsis`](https://adamrsophie.github.io/jaxR/reference/sl.md) :
  Index helpers

- [`stax_serial()`](https://adamrsophie.github.io/jaxR/reference/stax_combinators.md)
  [`stax_parallel()`](https://adamrsophie.github.io/jaxR/reference/stax_combinators.md)
  [`stax_shape_dependent()`](https://adamrsophie.github.io/jaxR/reference/stax_combinators.md)
  : Stax combinators

- [`stax_Dense()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_GeneralConv()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_Conv()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_GeneralConvTranspose()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_Conv1DTranspose()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_ConvTranspose()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_BatchNorm()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_elementwise()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_Tanh`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_Relu`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_Exp`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_LogSoftmax`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_Softmax`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_Softplus`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_Sigmoid`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_Elu`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_LeakyRelu`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_Selu`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_Gelu`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_MaxPool()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_SumPool()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_AvgPool()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_Flatten`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_Identity`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_FanOut()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_FanInSum`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_FanInConcat()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  [`stax_Dropout()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
  : Stax layers

- [`test_util_check_grads()`](https://adamrsophie.github.io/jaxR/reference/test_util_check_grads.md)
  [`test_util_check_jvp()`](https://adamrsophie.github.io/jaxR/reference/test_util_check_grads.md)
  [`test_util_check_vjp()`](https://adamrsophie.github.io/jaxR/reference/test_util_check_grads.md)
  : Check derivatives against finite differences

- [`tree_broadcast()`](https://adamrsophie.github.io/jaxR/reference/tree_broadcast.md)
  : Broadcast a pytree prefix over a full pytree

- [`tree_flatten()`](https://adamrsophie.github.io/jaxR/reference/tree_flatten.md)
  [`tree_leaves()`](https://adamrsophie.github.io/jaxR/reference/tree_flatten.md)
  [`tree_structure()`](https://adamrsophie.github.io/jaxR/reference/tree_flatten.md)
  : Flatten a pytree

- [`tree_flatten_with_path()`](https://adamrsophie.github.io/jaxR/reference/tree_flatten_with_path.md)
  [`tree_leaves_with_path()`](https://adamrsophie.github.io/jaxR/reference/tree_flatten_with_path.md)
  [`tree_map_with_path()`](https://adamrsophie.github.io/jaxR/reference/tree_flatten_with_path.md)
  : Flatten a pytree with key paths

- [`tree_map()`](https://adamrsophie.github.io/jaxR/reference/tree_map.md)
  [`tree_all()`](https://adamrsophie.github.io/jaxR/reference/tree_map.md)
  [`tree_reduce()`](https://adamrsophie.github.io/jaxR/reference/tree_map.md)
  : Map and reduce over the leaves of pytrees

- [`tree_reduce_associative()`](https://adamrsophie.github.io/jaxR/reference/tree_reduce_associative.md)
  : Reduce the leaves of a pytree with an associative operation

- [`tree_transpose()`](https://adamrsophie.github.io/jaxR/reference/tree_transpose.md)
  : Transpose a pytree of pytrees

- [`tree_unflatten()`](https://adamrsophie.github.io/jaxR/reference/tree_unflatten.md)
  : Rebuild a pytree from a treedef and leaves

- [`treedef_num_leaves()`](https://adamrsophie.github.io/jaxR/reference/treedef_children.md)
  [`treedef_children()`](https://adamrsophie.github.io/jaxR/reference/treedef_children.md)
  [`treedef_is_leaf()`](https://adamrsophie.github.io/jaxR/reference/treedef_children.md)
  [`treedef_tuple()`](https://adamrsophie.github.io/jaxR/reference/treedef_children.md)
  : Treedef helpers

- [`value_and_grad()`](https://adamrsophie.github.io/jaxR/reference/value_and_grad.md)
  : Value and gradient of a scalar function

- [`vjp()`](https://adamrsophie.github.io/jaxR/reference/vjp.md) :
  Vector-Jacobian product

- [`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) :
  Vectorizing map

- [`with_jax_config()`](https://adamrsophie.github.io/jaxR/reference/with_jax_config.md)
  : Temporarily set a configuration option
