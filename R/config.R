# Configuration (port of the commonly used options of jax._src.config).

.jx$config <- list(
  jax_enable_x64 = FALSE,
  jax_disable_jit = FALSE,
  jax_numpy_rank_promotion = "allow",
  jax_numpy_dtype_promotion = "standard",
  jax_default_matmul_precision = NULL,
  jax_threefry_partitionable = TRUE,
  jax_default_prng_impl = "threefry2x32",
  jax_random_seed_offset = 0L,
  jax_use_high_dynamic_range_gumbel = FALSE,
  jax_debug_nans = FALSE,
  jax_debug_infs = FALSE,
  jax_enable_checks = FALSE,
  jax_check_static_indices = FALSE,
  jax_softmax_custom_jvp = FALSE,
  jax_accurate_erf_inv = FALSE,
  jax_log_compiles = FALSE,
  jax_explain_cache_misses = FALSE,
  jax_check_tracer_leaks = FALSE,
  jax_no_tracing = FALSE,
  jax_no_execution = FALSE,
  jax_transfer_guard_host_to_device = "allow",
  jax_transfer_guard_device_to_host = "allow",
  jax_transfer_guard_device_to_device = "allow",
  jax_debug_key_reuse = FALSE,
  jax_legacy_prng_key = "allow",
  jax_enable_custom_prng = FALSE,
  jax_allow_f16_reductions = TRUE,
  jax_array_garbage_collection_guard = NULL,
  jax_thread_guard = FALSE
)

#' The configuration object
#'
#' Port of `jax.config`: `config$update(name, value)` sets an option (as
#' [jax_config_update()]), `config$read(name)` reads one, `config$values` lists all of
#' them, and `config$<option>` (for example `config$jax_enable_x64`) reads a single
#' option.
#' @format An object of class `jax_config_object`.
#' @export
config <- structure(list(), class = "jax_config_object")

#' @export
`$.jax_config_object` <- function(x, name) {
  switch(name,
    update = function(name, value) jax_config_update(name, value),
    read = function(name) jax_config(name),
    values = jax_config(),
    jax_config(name))
}

#' @export
print.jax_config_object <- function(x, ...) {
  cat("<jax config with", length(.jx$config), "options>\n")
  invisible(x)
}

#' Update a configuration option
#'
#' Port of `jax.config.update`. The most important option is `jax_enable_x64`
#' (default `FALSE`, as in JAX: arrays default to 32-bit types). The environment
#' variable `JAX_ENABLE_X64` sets its initial value.
#' @param name Option name.
#' @param value New value.
#' @return The previous value, invisibly.
#' @export
jax_config_update <- function(name, value) {
  if (!name %in% names(.jx$config)) jax_error("unknown config option '", name, "'")
  old <- .jx$config[[name]]
  .jx$config[name] <- list(value)
  invisible(old)
}

#' Read a configuration option
#' @param name Option name; `NULL` returns all options.
#' @export
jax_config <- function(name = NULL) {
  if (is.null(name)) return(.jx$config)
  if (!name %in% names(.jx$config)) jax_error("unknown config option '", name, "'")
  .jx$config[[name]]
}

#' Temporarily set a configuration option
#' @param name Option name.
#' @param value Value while evaluating `code`.
#' @param code Expression to evaluate.
#' @export
with_jax_config <- function(name, value, code) {
  old <- jax_config_update(name, value)
  on.exit(jax_config_update(name, old), add = TRUE)
  code
}

#' Enable 64-bit types
#'
#' Shortcut for `jax_config_update("jax_enable_x64", value)`.
#' @param value Logical.
#' @export
enable_x64 <- function(value = TRUE) jax_config_update("jax_enable_x64", isTRUE(value))

read_env_config <- function() {
  v <- Sys.getenv("JAX_ENABLE_X64", "")
  if (nzchar(v)) .jx$config$jax_enable_x64 <- tolower(v) %in% c("1", "true", "yes", "on")
  v <- Sys.getenv("JAX_DISABLE_JIT", "")
  if (nzchar(v)) .jx$config$jax_disable_jit <- tolower(v) %in% c("1", "true", "yes", "on")
}
