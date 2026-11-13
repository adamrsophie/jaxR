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

test_that("serialized executables keep exact 64 bit constants", {
  skip_if_no_backend()
  with_jax_config("jax_enable_x64", TRUE, {
    for (dt in c("int64", "uint64")) {
      k <- lax_add(as_jax(2^53, dt), as_jax(1L, dt))
      x <- as_jax(2^53, dt)
      f <- jit(function(x) k - x)$lower(x)$compile()
      s <- experimental_serialize_executable_serialize(f)
      g <- experimental_serialize_executable_deserialize_and_load(s[[1]], s[[2]], s[[3]])
      expect_identical(as.numeric(g(x)), 1)
    }
  })
})

test_that("the XLA flags are part of the cache key", {
  skip_if_no_backend()
  old <- Sys.getenv("XLA_FLAGS", NA)
  on.exit(if (is.na(old)) Sys.unsetenv("XLA_FLAGS") else Sys.setenv(XLA_FLAGS = old))
  Sys.setenv(XLA_FLAGS = "--xla_cpu_enable_fast_min_max=true --xla_a=1")
  k1 <- jaxR:::pcache_key("module")
  Sys.setenv(XLA_FLAGS = "--xla_a=1  --xla_cpu_enable_fast_min_max=true")
  expect_identical(jaxR:::pcache_key("module"), k1)
  Sys.setenv(XLA_FLAGS = "--xla_cpu_enable_fast_min_max=false --xla_a=1")
  expect_false(identical(jaxR:::pcache_key("module"), k1))
})

test_that("a saved executable with 64 bit constants loads with x64 off", {
  skip_if_no_backend()
  s <- with_jax_config("jax_enable_x64", TRUE, {
    k <- as_jax(c(2^40, 2^40 + 1), "int64")
    experimental_serialize_executable_serialize(jit(function() k)$lower()$compile())
  })
  with_jax_config("jax_enable_x64", FALSE, {
    l <- experimental_serialize_executable_deserialize_and_load(s[[1]], s[[2]], s[[3]])
    expect_identical(as.numeric(l()), c(2^40, 2^40 + 1))
  })
})
