# XLA backend through the PJRT C API (see src/pjrt.cpp).

pjrt_release <- "manual-2026-09-08T13-23-00Z"

# Functions run after the plugin loads (for example registration of FFI handlers).
.jx$ffi_hooks <- list()

pjrt_platform_tag <- function() {
  sys <- tolower(Sys.info()[["sysname"]])
  arch <- Sys.info()[["machine"]]
  os <- switch(sys, darwin = "darwin", linux = "linux",
               jax_error("no prebuilt PJRT CPU plugin for ", sys, "; set JAXR_PJRT_PLUGIN"))
  cpu <- if (arch %in% c("arm64", "aarch64")) "arm64" else if (arch %in% c("x86_64", "amd64")) "amd64"
         else jax_error("no prebuilt PJRT CPU plugin for architecture ", arch)
  paste0(os, "-", cpu)
}

pjrt_dir <- function() tools::R_user_dir("jaxR", "data")

pjrt_lib_name <- function() if (Sys.info()[["sysname"]] == "Darwin") "libpjrt_cpu.dylib" else "libpjrt_cpu.so"

#' Install the XLA runtime plugin
#'
#' Downloads a prebuilt PJRT CPU plugin (built from OpenXLA by the zml project) into
#' the user data directory of jaxR. Set the environment variable `JAXR_PJRT_PLUGIN` to
#' use a different plugin (for example a CUDA plugin) instead. Prebuilt plugins exist for
#' Linux and macOS on x86-64 and arm64; on other systems, Windows included, the function
#' stops with an error and `JAXR_PJRT_PLUGIN` is the only way to supply a runtime.
#' @param release Release tag of `zml/pjrt-artifacts`.
#' @param force Reinstall even if present.
#' @return The path of the installed plugin, invisibly.
#' @export
install_pjrt <- function(release = pjrt_release, force = FALSE) {
  dest <- file.path(pjrt_dir(), pjrt_platform_tag())
  lib <- file.path(dest, pjrt_lib_name())
  if (file.exists(lib) && !force) return(invisible(lib))
  dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  url <- sprintf("https://github.com/zml/pjrt-artifacts/releases/download/%s/pjrt-cpu_%s.tar.gz",
                 release, pjrt_platform_tag())
  tgz <- tempfile(fileext = ".tar.gz")
  on.exit(unlink(tgz), add = TRUE)
  old <- options(timeout = max(600, getOption("timeout")))
  on.exit(options(old), add = TRUE)
  utils::download.file(url, tgz, mode = "wb", quiet = TRUE)
  utils::untar(tgz, exdir = dest)
  if (!file.exists(lib)) jax_error("plugin archive did not contain ", pjrt_lib_name())
  invisible(lib)
}

find_pjrt_plugin <- function() {
  env <- Sys.getenv("JAXR_PJRT_PLUGIN", "")
  if (nzchar(env)) return(env)
  lib <- tryCatch(file.path(pjrt_dir(), pjrt_platform_tag(), pjrt_lib_name()),
                  error = function(e) "")
  if (nzchar(lib) && file.exists(lib)) return(lib)
  NULL
}

#' Initialize the XLA backend
#'
#' Called automatically on first use.
#' @param plugin Path to a PJRT plugin shared library.
#' @return Backend information (platform, number of devices, PJRT API version).
#' @export
jax_backend_init <- function(plugin = find_pjrt_plugin()) {
  if (!is.null(.jx$backend)) return(invisible(.jx$backend))
  if (is.null(plugin))
    jax_error("No XLA runtime found. Run jaxR::install_pjrt() once, or set the ",
              "JAXR_PJRT_PLUGIN environment variable to a PJRT plugin library.",
              class = "jax_no_backend")
  # jaxlib compiles with xla_cpu_enable_fast_min_max=false (NaN propagating max/min);
  # the plugin's default is true, so set it before XLA reads its flags.
  flags <- Sys.getenv("XLA_FLAGS", "")
  if (!grepl("xla_cpu_enable_fast_min_max", flags, fixed = TRUE))
    Sys.setenv(XLA_FLAGS = trimws(paste(flags, "--xla_cpu_enable_fast_min_max=false")))
  info <- .Call(jaxr_pjrt_load, normalizePath(plugin))
  info$plugin <- plugin
  .jx$backend <- info
  for (hook in .jx$ffi_hooks) hook()
  invisible(info)
}

backend_available <- function() {
  if (!is.null(.jx$backend)) return(TRUE)
  !is.null(find_pjrt_plugin())
}

ensure_backend <- function() if (is.null(.jx$backend)) jax_backend_init()

#' List devices
#' @return A character vector of device descriptions.
#' @export
devices <- function() {
  ensure_backend()
  paste0(.jx$backend$platform, ":", seq_len(.jx$backend$num_devices) - 1L)
}

# ---- compilation cache ----------------------------------------------------------------

# Minimal xla.CompileOptionsProto: executable_build_options {num_replicas: 1,
# num_partitions: 1}.
compile_options_bytes <- as.raw(c(0x1a, 0x04, 0x20, 0x01, 0x28, 0x01))

# Keyed by module text (environments cannot hold names longer than 10000 bytes).
compile_cache <- function() {
  if (is.null(.jx$compile_cache)) .jx$compile_cache <- utils::hashtab()
  .jx$compile_cache
}

compile_mlir <- function(text) {
  ensure_backend()
  exe <- utils::gethash(compile_cache(), text)
  if (!is.null(exe)) return(exe)
  if (isTRUE(.jx$config$jax_log_compiles)) {
    nm <- regmatches(text, regexpr("sym_name = \"[^\"]*\"", text))
    message("Compiling ", sub("sym_name = ", "", nm), " with XLA")
  }
  exe <- pcache_get(text)
  if (is.null(exe)) {
    start <- proc.time()[[3L]]
    exe <- mon_elapsed("/jax/core/compile/backend_compile_duration",
                       sub("sym_name = \"([^\"]*)\"", "\\1", regmatches(text, regexpr("sym_name = \"[^\"]*\"", text))),
                       .Call(jaxr_compile, text, compile_options_bytes))
    pcache_put(text, exe, proc.time()[[3L]] - start)
  }
  utils::sethash(compile_cache(), text, exe)
  exe
}

#' Clear the compilation caches
#' @export
clear_caches <- function() {
  utils::clrhash(compile_cache())
  if (!is.null(.jx$eager_cache)) utils::clrhash(.jx$eager_cache)
  invisible(NULL)
}

# ---- buffers --------------------------------------------------------------------------

pjrt_type_codes <- c(bool = 1L, int8 = 2L, int16 = 3L, int32 = 4L, int64 = 5L, uint8 = 6L,
                     uint16 = 7L, uint32 = 8L, uint64 = 9L, float16 = 10L, float32 = 11L,
                     float64 = 12L, bfloat16 = 13L, complex64 = 14L, complex128 = 15L)

# Host R data (any layout) -> device buffer of dtype `dt` (data already in range).
buffer_from_r <- function(data, dt, shape) {
  ensure_backend()
  # R scalars behave like Python scalars (constants, not guarded transfers).
  if (length(data) > 1L) transfer_guard_check("host_to_device", isTRUE(.jx$explicit_transfer))
  flat <- c_ravel(data)
  if (is_complex_dtype(dt)) flat <- as.complex(flat)
  .Call(jaxr_buffer_from_host, flat, pjrt_type_codes[[dt]], as.integer(shape), 0L)
}

buffer_to_r <- function(buf, dt, shape) {
  transfer_guard_check("device_to_host", isTRUE(.jx$explicit_transfer))
  flat <- .Call(jaxr_buffer_to_host, buf, pjrt_type_codes[[dt]], as.double(prod(shape)))
  c_unravel(flat, shape)
}

execute_exe <- function(exe, bufs) {
  if (isTRUE(.jx$config$jax_no_execution))
    jax_error("executing a computation under jax_no_execution (no_execution)",
              class = "jax_runtime_error")
  .Call(jaxr_execute, exe, bufs)
}

# Transfer guards (jax.transfer_guard): "allow", "log", "disallow", "log_explicit",
# "disallow_explicit".
transfer_guard_check <- function(direction, explicit = FALSE) {
  g <- .jx$config[[paste0("jax_transfer_guard_", direction)]] %||% "allow"
  if (g == "allow") return(invisible())
  what <- switch(direction, host_to_device = "host-to-device", device_to_host = "device-to-host",
                 "device-to-device")
  if (g == "log" && !explicit || g == "log_explicit")
    message(if (explicit) "explicit " else "implicit ", what, " transfer")
  if ((g == "disallow" && !explicit) || g == "disallow_explicit")
    jax_error("Disallowed ", what, " transfer", class = "jax_runtime_error")
  invisible()
}
