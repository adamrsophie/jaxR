# jax.monitoring: event listeners and the compile stage events recorded by jit.

test_that("listeners receive recorded events and can be removed", {
  on.exit(monitoring_clear_event_listeners())
  seen <- list()
  listener <- function(event, ...) seen[[length(seen) + 1L]] <<- list(event, ...)
  monitoring_register_event_listener(listener)
  monitoring_record_event("/a", x = 1L)
  expect_identical(seen, list(list("/a", x = 1L)))
  monitoring_unregister_event_listener(listener)
  monitoring_record_event("/a")
  expect_length(seen, 1L)
  expect_error(monitoring_unregister_event_listener(listener), class = "jax_assertion_error")
  expect_error(monitoring_register_event_listener("no"), class = "jax_type_error")
})

test_that("each kind of record reaches its own listeners", {
  on.exit(monitoring_clear_event_listeners())
  got <- list()
  monitoring_register_event_duration_secs_listener(function(event, duration, ...) got$duration <<- c(event, duration))
  monitoring_register_event_time_span_listener(function(event, start_time, end_time, ...) got$span <<- c(start_time, end_time))
  monitoring_register_scalar_listener(function(event, value, ...) got$scalar <<- value)
  monitoring_record_event("/ignored")
  monitoring_record_event_duration_secs("/d", 1.5)
  monitoring_record_event_time_span("/s", 2, 3)
  monitoring_record_scalar("/v", 7)
  expect_identical(got, list(duration = c("/d", "1.5"), span = c(2, 3), scalar = 7))
  monitoring_clear_event_listeners()
  monitoring_record_scalar("/v", 8)
  expect_identical(got$scalar, 7)
})

test_that("jit reports tracing, lowering and compilation", {
  skip_if_no_backend()
  on.exit(monitoring_clear_event_listeners())
  events <- character()
  monitoring_register_event_duration_secs_listener(function(event, duration, fun_name) {
    expect_gte(duration, 0)
    events <<- c(events, event)
  })
  f <- jit(function(x) x * 3.25 + 1.125)
  f(1)
  expect_setequal(events, c("/jax/core/compile/jaxpr_trace_duration",
                            "/jax/core/compile/jaxpr_to_mlir_module_duration",
                            "/jax/core/compile/backend_compile_duration"))
  n <- length(events)
  f(2)
  expect_length(events, n)
})
