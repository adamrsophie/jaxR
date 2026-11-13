# Instrumentation events

Port of `jax.monitoring`. Code points are marked as named events; every
time an event is reached the registered listeners are called, for
example to send it to a metrics collector.

## Usage

``` r
monitoring_record_event(event, ...)

monitoring_record_event_duration_secs(event, duration, ...)

monitoring_record_event_time_span(event, start_time, end_time, ...)

monitoring_record_scalar(event, value, ...)

monitoring_register_event_listener(callback)

monitoring_register_event_duration_secs_listener(callback)

monitoring_register_event_time_span_listener(callback)

monitoring_register_scalar_listener(callback)

monitoring_clear_event_listeners()

monitoring_unregister_event_listener(callback)

monitoring_unregister_event_duration_listener(callback)

monitoring_unregister_event_time_span_listener(callback)

monitoring_unregister_scalar_listener(callback)
```

## Arguments

- event:

  Event name.

- ...:

  Named metadata passed to the listeners.

- duration:

  Duration in seconds.

- start_time, end_time:

  Times in seconds.

- value:

  A number.

- callback:

  A listener: a function of the arguments of the matching `record`
  function.

## Value

`NULL`, invisibly.

## Details

jaxR itself records, each with a `fun_name` argument, the events
`"/jax/core/compile/jaxpr_trace_duration"` (tracing a function for
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md)),
`"/jax/core/compile/jaxpr_to_mlir_module_duration"` (lowering to
StableHLO) and `"/jax/core/compile/backend_compile_duration"` (XLA
compilation): the start time as a scalar, then the duration and the time
span.

## Examples

``` r
monitoring_register_event_listener(function(event, ...) message("event: ", event))
monitoring_record_event("/my/event")
#> event: /my/event
monitoring_clear_event_listeners()
```
