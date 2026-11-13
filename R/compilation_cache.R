# Persistent compilation cache and executable serialization (ports of
# jax._src.compilation_cache, jax.experimental.compilation_cache and
# jax.experimental.serialize_executable).

pcache_dir <- function() {
  if (!isTRUE(.jx$config$jax_enable_compilation_cache)) return(NULL)
  d <- .jx$config$jax_compilation_cache_dir
  if (is.null(d) || !nzchar(d)) NULL else path.expand(d)
}

# The key covers the module, the plugin (path, size and time, which change with its
# version) and the version of jaxR, like the cache key of JAX covers the module, the
# compile options and the versions of jaxlib and the backend.
pcache_key <- function(text) {
  pl <- .jx$backend$plugin
  fi <- file.info(pl)
  f <- tempfile("jaxr_key")
  on.exit(unlink(f))
  writeBin(charToRaw(paste(text, .jx$backend$platform, normalizePath(pl), fi$size,
                           as.numeric(fi$mtime), utils::packageVersion("jaxR"),
                           paste(compile_options_bytes, collapse = ""), sep = "\n")), f)
  unname(tools::md5sum(f))
}

pcache_file <- function(dir, text) {
  nm <- sub("sym_name = \"([^\"]*)\"", "\\1", regmatches(text, regexpr("sym_name = \"[^\"]*\"", text)))
  file.path(dir, paste0(if (length(nm)) nm else "jit", "-", pcache_key(text)))
}

pcache_problem <- function(what, e) {
  msg <- paste0("Error ", what, " the persistent compilation cache: ", conditionMessage(e))
  if (isTRUE(.jx$config$jax_raise_persistent_cache_errors)) jax_error(msg, class = "jax_runtime_error")
  warning(msg, call. = FALSE)
  NULL
}

# Programs with host callbacks hold addresses of this process: they are not cached.
pcache_usable <- function(text) !is.null(pcache_dir()) && !grepl("\"jaxr_r_callback\"", text, fixed = TRUE)

pcache_get <- function(text) {
  if (!pcache_usable(text)) return(NULL)
  monitoring_record_event("/jax/compilation_cache/compile_requests_use_cache")
  path <- pcache_file(pcache_dir(), text)
  if (!file.exists(path)) {
    monitoring_record_event("/jax/compilation_cache/cache_misses")
    return(NULL)
  }
  start <- proc.time()[[3L]]
  exe <- tryCatch(.Call(jaxr_executable_deserialize, readBin(path, "raw", file.size(path))),
                  error = function(e) pcache_problem("reading", e))
  if (!is.null(exe)) {
    monitoring_record_event("/jax/compilation_cache/cache_hits")
    monitoring_record_event_duration_secs("/jax/compilation_cache/cache_retrieval_time_sec",
                                          proc.time()[[3L]] - start)
  }
  exe
}

pcache_put <- function(text, exe, compile_time) {
  if (!pcache_usable(text)) return(invisible(NULL))
  if (compile_time < (.jx$config$jax_persistent_cache_min_compile_time_secs %||% 1)) return(invisible(NULL))
  tryCatch({
    bytes <- .Call(jaxr_executable_serialize, exe)
    if (length(bytes) < (.jx$config$jax_persistent_cache_min_entry_size_bytes %||% 0)) return(invisible(NULL))
    dir <- pcache_dir()
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)
    path <- pcache_file(dir, text)
    tmp <- tempfile(basename(path), tmpdir = dir)
    writeBin(bytes, tmp)
    if (!file.rename(tmp, path)) unlink(tmp)
  }, error = function(e) pcache_problem("writing", e))
  invisible(NULL)
}

#' Persistent compilation cache
#'
#' Ports of `jax.experimental.compilation_cache.compilation_cache.set_cache_dir` and
#' `reset_cache`. With a cache directory set (here, or with the option
#' `jax_compilation_cache_dir` of [jax_config_update()]), compiled programs are written
#' to the directory and read back instead of being compiled again, also by later R
#' sessions. As in JAX, only programs whose compilation took at least
#' `jax_persistent_cache_min_compile_time_secs` seconds (default 1) are written, and
#' `jax_enable_compilation_cache = FALSE` turns the cache off. Programs with host
#' callbacks are not cached. Problems with the cache are warnings unless
#' `jax_raise_persistent_cache_errors` is `TRUE`.
#'
#' The cache is trusted: a cached file is native code that is run. Do not use a
#' directory that others can write to.
#' @param path Directory of the cache.
#' @return `NULL`, invisibly.
#' @examples
#' \dontrun{
#' experimental_compilation_cache_set_cache_dir(file.path(tempdir(), "jax_cache"))
#' }
#' @export
experimental_compilation_cache_set_cache_dir <- function(path) {
  jax_config_update("jax_compilation_cache_dir", path)
  invisible(NULL)
}

#' @rdname experimental_compilation_cache_set_cache_dir
#' @export
experimental_compilation_cache_reset_cache <- function() {
  # jaxR reads the options at every compilation, so there is no state to reset beyond
  # the in memory cache of this session, which clear_caches() empties.
  invisible(NULL)
}

#' Serialize a compiled function
#'
#' Ports of `jax.experimental.serialize_executable.serialize` and
#' `deserialize_and_load`. `experimental_serialize_executable_serialize()` takes the
#' result of `jit(f)$lower(...)$compile()` and returns
#' `list(payload, in_tree, out_tree)`: `payload` is a raw vector with the executable
#' and its constants, which can be saved and loaded again by a later R session on the
#' same platform with the same XLA runtime. `deserialize_and_load()` rebuilds the
#' compiled function from the three parts.
#'
#' As with JAX, the payload is trusted: loading it runs native code and R's
#' `unserialize()`. Only load payloads you made yourself.
#' @param compiled A compiled function from `$compile()`.
#' @param serialized The payload.
#' @param in_tree,out_tree The trees returned with the payload.
#' @return `serialize`: `list(payload, in_tree, out_tree)`. `deserialize_and_load`: a
#'   compiled function.
#' @examples
#' \dontrun{
#' f <- jit(function(x) x * 2)
#' compiled <- f$lower(as_jax(c(1, 2)))$compile()
#' s <- experimental_serialize_executable_serialize(compiled)
#' g <- experimental_serialize_executable_deserialize_and_load(s[[1]], s[[2]], s[[3]])
#' g(as_jax(c(3, 4)))
#' }
#' @export
experimental_serialize_executable_serialize <- function(compiled) {
  if (!inherits(compiled, "jax_stages_compiled"))
    jax_error("serialize takes a compiled function (the result of $compile())",
              class = "jax_type_error")
  st <- attr(compiled, "stage")
  exe <- st$runtime_executable()
  if (isTRUE(attr(exe, "host_callbacks")))
    jax_error("Cannot serialize a function with host callbacks", class = "jax_value_error")
  ensure_backend()
  bytes <- .Call(jaxr_executable_serialize, exe)
  consts <- lapply(st$consts, function(c) {
    c <- as_jax(c)
    list(aval = aval_of(c), host = array_host(c))
  })
  payload <- serialize(list(format = "jaxR-executable-1", platform = .jx$backend$platform,
                            exe = bytes, consts = consts, in_avals = st$in_avals,
                            out_avals = st$out_avals, args_info = st$args_info,
                            out_info = st$out_info), NULL)
  list(payload, st$in_tree, st$out_tree)
}

#' @rdname experimental_serialize_executable_serialize
#' @export
experimental_serialize_executable_deserialize_and_load <- function(serialized, in_tree, out_tree) {
  p <- unserialize(serialized)
  if (!is.list(p) || !identical(p$format, "jaxR-executable-1"))
    jax_error("not a serialized jaxR executable", class = "jax_value_error")
  ensure_backend()
  if (!identical(p$platform, .jx$backend$platform))
    jax_error("the executable was compiled for ", p$platform, ", not ", .jx$backend$platform,
              class = "jax_value_error")
  entry <- new.env(parent = emptyenv())
  entry$exe <- .Call(jaxr_executable_deserialize, p$exe)
  consts <- lapply(p$consts, function(c) new_array(c$aval, host = c$host))
  meta <- list(args_info = p$args_info, out_info = p$out_info, in_tree = in_tree,
               out_tree = out_tree, in_avals = p$in_avals)
  stg_compiled(entry, consts, p$out_avals, meta)
}
