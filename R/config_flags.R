# Configuration context managers of the jax top level namespace (jax.debug_nans,
# jax.transfer_guard, ...). Each runs `code` with the option set and restores it.

flag_context <- function(option, values = NULL) {
  force(option)
  function(new_val = TRUE, code) {
    if (!is.null(values) && !new_val %in% values)
      jax_error(option, " must be one of ", paste0("'", values, "'", collapse = ", "),
                class = "jax_value_error")
    with_jax_config(option, new_val, code)
  }
}

#' Debugging and safety configuration contexts
#'
#' Ports of the configuration context managers of `jax`: each evaluates `code` with the
#' option set and restores the previous value.
#'
#' * `debug_nans()` / `debug_infs()`: raise an error (class `jax_floating_point_error`)
#'   when a primitive or a jitted computation produces NaN / infinite values.
#' * `numpy_dtype_promotion("strict")`: forbid implicit promotion between different
#'   strongly typed dtypes.
#' * `log_compiles()`: report every XLA compilation.
#' * `explain_cache_misses()`: report why `jit` traces again.
#' * `check_tracer_leaks()`: leaked tracers raise errors (jaxR always detects use of a
#'   dead tracer).
#' * `no_tracing()` / `no_execution()`: raise an error on any jit tracing / execution.
#' * `transfer_guard()` and its directional variants: `"allow"`, `"log"`, `"disallow"`,
#'   `"log_explicit"` or `"disallow_explicit"` for implicit (conversions of R data that
#'   is not a scalar, `as.array()`, printing) and explicit (`device_put()`,
#'   `device_get()`) transfers.
#' @param new_val New value of the option.
#' @param code Expression evaluated with the option set.
#' @name config_contexts
NULL

#' @rdname config_contexts
#' @export
debug_nans <- flag_context("jax_debug_nans")
#' @rdname config_contexts
#' @export
debug_infs <- flag_context("jax_debug_infs")
#' @rdname config_contexts
#' @export
numpy_dtype_promotion <- flag_context("jax_numpy_dtype_promotion", c("standard", "strict"))
#' @rdname config_contexts
#' @export
log_compiles <- flag_context("jax_log_compiles")
#' @rdname config_contexts
#' @export
explain_cache_misses <- flag_context("jax_explain_cache_misses")
#' @rdname config_contexts
#' @export
check_tracer_leaks <- flag_context("jax_check_tracer_leaks")
#' @rdname config_contexts
#' @export
checking_leaks <- function(code) with_jax_config("jax_check_tracer_leaks", TRUE, code)
#' @rdname config_contexts
#' @export
no_tracing <- function(code) with_jax_config("jax_no_tracing", TRUE, code)
#' @rdname config_contexts
#' @export
no_execution <- function(code) with_jax_config("jax_no_execution", TRUE, code)

guard_values <- c("allow", "log", "disallow", "log_explicit", "disallow_explicit")

#' @rdname config_contexts
#' @export
transfer_guard_host_to_device <- flag_context("jax_transfer_guard_host_to_device", guard_values)
#' @rdname config_contexts
#' @export
transfer_guard_device_to_host <- flag_context("jax_transfer_guard_device_to_host", guard_values)
#' @rdname config_contexts
#' @export
transfer_guard_device_to_device <- flag_context("jax_transfer_guard_device_to_device", guard_values)
#' @rdname config_contexts
#' @export
transfer_guard <- function(new_val, code) {
  if (!new_val %in% guard_values) jax_error("invalid transfer guard level", class = "jax_value_error")
  with_jax_config("jax_transfer_guard_host_to_device", new_val,
    with_jax_config("jax_transfer_guard_device_to_host", new_val,
      with_jax_config("jax_transfer_guard_device_to_device", new_val, code)))
}

#' Further configuration contexts
#'
#' Ports of `jax.enable_checks`, `jax.debug_key_reuse`, `jax.legacy_prng_key`,
#' `jax.enable_custom_prng`, `jax.softmax_custom_jvp`, `jax.allow_f16_reductions`,
#' `jax.array_garbage_collection_guard` and `jax.thread_guard`. Each evaluates `code`
#' with the option set. `legacy_prng_key("warn")` or `("error")` makes
#' [random_PRNGKey()] warn or fail; `softmax_custom_jvp` selects the custom JVP of
#' [nn_softmax()]. `debug_key_reuse(TRUE, code)` checks that no typed key is used
#' twice (port of `jax.experimental.key_reuse`): a key used by a random function is
#' consumed, and using a consumed key raises an error of class `jax_key_reuse_error`,
#' also inside [jit()], [lax_cond()], [lax_scan()], [lax_while_loop()] and [vmap()].
#' The other options are accepted for compatibility: jaxR has one thread, frees device
#' buffers with R's garbage collector and keeps its checks always on.
#' @param new_val New value of the option.
#' @param code Expression evaluated with the option set.
#' @name config_contexts_more
NULL

#' @rdname config_contexts_more
#' @export
enable_checks <- flag_context("jax_enable_checks")
#' @rdname config_contexts_more
#' @export
debug_key_reuse <- flag_context("jax_debug_key_reuse")
#' @rdname config_contexts_more
#' @export
legacy_prng_key <- flag_context("jax_legacy_prng_key", c("allow", "warn", "error"))
#' @rdname config_contexts_more
#' @export
enable_custom_prng <- flag_context("jax_enable_custom_prng")
#' @rdname config_contexts_more
#' @export
softmax_custom_jvp <- flag_context("jax_softmax_custom_jvp")
#' @rdname config_contexts_more
#' @export
allow_f16_reductions <- flag_context("jax_allow_f16_reductions")
#' @rdname config_contexts_more
#' @export
array_garbage_collection_guard <- flag_context("jax_array_garbage_collection_guard", c("allow", "log", "fatal"))
#' @rdname config_contexts_more
#' @export
thread_guard <- flag_context("jax_thread_guard")

#' User contexts that affect jit caching
#'
#' Port of `jax.make_user_context`: returns a function `ctx(value, code)` that evaluates
#' `code` with the context set to `value`. `jit` caches are keyed on the values of all
#' user contexts, so a function traced under one value is traced again under another.
#' `ctx()` with no arguments returns the current value.
#' @param default_value The value outside any `ctx(value, code)` call.
#' @return A context function.
#' @export
make_user_context <- function(default_value = NULL) {
  id <- paste0("ctx", length(.jx$user_contexts) + 1L)
  .jx$user_contexts[id] <- list(default_value)
  function(value, code) {
    if (missing(value)) return(.jx$user_contexts[[id]])
    old <- .jx$user_contexts[[id]]
    .jx$user_contexts[id] <- list(value)
    on.exit(.jx$user_contexts[id] <- list(old), add = TRUE)
    code
  }
}
