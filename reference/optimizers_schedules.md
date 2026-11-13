# Learning rate schedules

Ports of the step size schedules of `jax.example_libraries.optimizers`.
A schedule is a function of the (0-based) iteration index returning the
step size. `optimizers_make_schedule` returns a schedule unchanged and
turns a scalar into a constant schedule.

## Usage

``` r
optimizers_constant(step_size)

optimizers_exponential_decay(step_size, decay_steps, decay_rate)

optimizers_inverse_time_decay(
  step_size,
  decay_steps,
  decay_rate,
  staircase = FALSE
)

optimizers_polynomial_decay(step_size, decay_steps, final_step_size, power = 1)

optimizers_piecewise_constant(boundaries, values)

optimizers_make_schedule(scalar_or_schedule)
```

## Arguments

- step_size:

  Initial (or constant) step size.

- decay_steps, decay_rate:

  Decay parameters: the step size is multiplied by `decay_rate` every
  `decay_steps` iterations (`exponential_decay`) or divided by
  `1 + decay_rate * i / decay_steps` (`inverse_time_decay`).

- staircase:

  Decay in discrete steps (`floor(i / decay_steps)`).

- final_step_size, power:

  Final value and power of `polynomial_decay`.

- boundaries, values:

  Boundaries (one shorter than `values`) and step sizes of
  `piecewise_constant`.

- scalar_or_schedule:

  A scalar or a schedule.

## Value

A schedule function.

## Examples

``` r
if (FALSE) { # \dontrun{
s <- optimizers_piecewise_constant(c(10, 20), c(1, 0.1, 0.01))
s(15L)
} # }
```
