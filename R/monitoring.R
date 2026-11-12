# jax.monitoring (port of jax/_src/monitoring.py): named events and their listeners.

.jx$monitoring <- list(event = list(), duration = list(), time_span = list(), scalar = list())

mon_record <- function(kind, ...) {
  for (callback in .jx$monitoring[[kind]]) callback(...)
  invisible(NULL)
}
mon_register <- function(kind, callback) {
  if (!is.function(callback)) jax_error("a listener must be a function", class = "jax_type_error")
  .jx$monitoring[[kind]] <- c(.jx$monitoring[[kind]], list(callback))
  invisible(NULL)
}
mon_unregister <- function(kind, callback) {
  i <- which(vapply(.jx$monitoring[[kind]], identical, NA, callback))
  if (!length(i)) jax_error("the listener is not registered", class = "jax_assertion_error")
  .jx$monitoring[[kind]][[i[1L]]] <- NULL
  invisible(NULL)
}

# Port of dispatch.log_elapsed_time: evaluates `expr` and reports its start time,
# duration and time span to the listeners.
mon_elapsed <- function(event, fun_name, expr) {
  m <- .jx$monitoring
  if (!length(m$scalar) && !length(m$duration) && !length(m$time_span)) return(expr)
  start <- as.numeric(Sys.time())
  mon_record("scalar", event, start, fun_name = fun_name)
  on.exit({
    end <- as.numeric(Sys.time())
    mon_record("duration", event, end - start, fun_name = fun_name)
    mon_record("time_span", event, start, end, fun_name = fun_name)
  })
  expr
}

#' Instrumentation events
#'
#' Port of `jax.monitoring`. Code points are marked as named events; every time an
#' event is reached the registered listeners are called, for example to send it to a
#' metrics collector.
#'
#' jaxR itself records, each with a `fun_name` argument, the events
#' `"/jax/core/compile/jaxpr_trace_duration"` (tracing a function for [jit()]),
#' `"/jax/core/compile/jaxpr_to_mlir_module_duration"` (lowering to StableHLO) and
#' `"/jax/core/compile/backend_compile_duration"` (XLA compilation): the start time as a
#' scalar, then the duration and the time span.
#' @param event Event name.
#' @param duration Duration in seconds.
#' @param start_time,end_time Times in seconds.
#' @param value A number.
#' @param ... Named metadata passed to the listeners.
#' @param callback A listener: a function of the arguments of the matching `record`
#'   function.
#' @return `NULL`, invisibly.
#' @examples
#' monitoring_register_event_listener(function(event, ...) message("event: ", event))
#' monitoring_record_event("/my/event")
#' monitoring_clear_event_listeners()
#' @name monitoring
NULL

#' @rdname monitoring
#' @export
monitoring_record_event <- function(event, ...) mon_record("event", event, ...)

#' @rdname monitoring
#' @export
monitoring_record_event_duration_secs <- function(event, duration, ...) mon_record("duration", event, duration, ...)

#' @rdname monitoring
#' @export
monitoring_record_event_time_span <- function(event, start_time, end_time, ...) {
  mon_record("time_span", event, start_time, end_time, ...)
}

#' @rdname monitoring
#' @export
monitoring_record_scalar <- function(event, value, ...) mon_record("scalar", event, value, ...)

#' @rdname monitoring
#' @export
monitoring_register_event_listener <- function(callback) mon_register("event", callback)

#' @rdname monitoring
#' @export
monitoring_register_event_duration_secs_listener <- function(callback) mon_register("duration", callback)

#' @rdname monitoring
#' @export
monitoring_register_event_time_span_listener <- function(callback) mon_register("time_span", callback)

#' @rdname monitoring
#' @export
monitoring_register_scalar_listener <- function(callback) mon_register("scalar", callback)

#' @rdname monitoring
#' @export
monitoring_clear_event_listeners <- function() {
  .jx$monitoring <- list(event = list(), duration = list(), time_span = list(), scalar = list())
  invisible(NULL)
}

#' @rdname monitoring
#' @export
monitoring_unregister_event_listener <- function(callback) mon_unregister("event", callback)

#' @rdname monitoring
#' @export
monitoring_unregister_event_duration_listener <- function(callback) mon_unregister("duration", callback)

#' @rdname monitoring
#' @export
monitoring_unregister_event_time_span_listener <- function(callback) mon_unregister("time_span", callback)

#' @rdname monitoring
#' @export
monitoring_unregister_scalar_listener <- function(callback) mon_unregister("scalar", callback)
