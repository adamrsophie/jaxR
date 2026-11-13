test_that("compiled functions serialize and load again", {
  skip_if_no_backend()
  k <- as_jax(c(10, 20), "float32")
  f <- jit(function(x, y) list(x * 2 + k, jnp_sin(y)))
  compiled <- f$lower(as_jax(c(1, 2), "float32"), as_jax(c(0, 0), "float32"))$compile()
  s <- experimental_serialize_executable_serialize(compiled)
  expect_type(s[[1L]], "raw")
  p <- tempfile(fileext = ".rds")
  saveRDS(s, p)
  s <- readRDS(p)
  g <- experimental_serialize_executable_deserialize_and_load(s[[1L]], s[[2L]], s[[3L]])
  out <- g(as_jax(c(5, 6), "float32"), as_jax(c(0, 0), "float32"))
  expect_equal(as.numeric(out[[1L]]), c(20, 32))
  expect_equal(as.numeric(out[[2L]]), c(0, 0))
  expect_error(g(as_jax(c(5, 6, 7), "float32"), as_jax(c(0, 0), "float32")), "compiled with")
  expect_error(experimental_serialize_executable_serialize(f), "compiled function")
  expect_error(experimental_serialize_executable_deserialize_and_load(serialize(1, NULL), NULL, NULL),
               "not a serialized")
})

test_that("the persistent compilation cache is read back by a new compilation", {
  skip_if_no_backend()
  d <- tempfile("jax_cache")
  old_dir <- jax_config_update("jax_compilation_cache_dir", d)
  old_min <- jax_config_update("jax_persistent_cache_min_compile_time_secs", 0)
  hits <- 0
  listener <- function(e, ...) if (e == "/jax/compilation_cache/cache_hits") hits <<- hits + 1
  monitoring_register_event_listener(listener)
  on.exit({
    jax_config_update("jax_compilation_cache_dir", old_dir)
    jax_config_update("jax_persistent_cache_min_compile_time_secs", old_min)
    monitoring_unregister_event_listener(listener)
    unlink(d, recursive = TRUE)
  })
  f <- function(x) jnp_cos(x) * 3 + 0.25
  expect_equal(as.numeric(jit(f)(as_jax(1, "float32"))), cos(1) * 3 + 0.25, tolerance = 1e-6)
  expect_gte(length(list.files(d)), 1L)
  clear_caches()
  expect_equal(as.numeric(jit(f)(as_jax(1, "float32"))), cos(1) * 3 + 0.25, tolerance = 1e-6)
  expect_gte(hits, 1)

  # Turned off, nothing is read.
  jax_config_update("jax_enable_compilation_cache", FALSE)
  on.exit(jax_config_update("jax_enable_compilation_cache", TRUE), add = TRUE)
  hits <- 0
  clear_caches()
  jit(f)(as_jax(1, "float32"))
  expect_identical(hits, 0)
})
