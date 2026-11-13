# Optimizers

Ports of the optimizers of `jax.example_libraries.optimizers`. Each
constructor returns an optimizer `list(init_fn, update_fn, params_fn)`
(see
[`optimizers_Optimizer()`](https://adamrsophie.github.io/jaxR/reference/optimizers_state.md))
that works on pytrees of parameters: `init_fn(params)` creates the
state, `update_fn(step, grads, opt_state)` applies one update (`step` is
the 0-based iteration index) and `params_fn(opt_state)` extracts the
parameters.

## Usage

``` r
optimizers_sgd(step_size)

optimizers_momentum(step_size, mass)

optimizers_nesterov(step_size, mass)

optimizers_adagrad(step_size, momentum = 0.9)

optimizers_rmsprop(step_size, gamma = 0.9, eps = 1e-08)

optimizers_rmsprop_momentum(
  step_size,
  gamma = 0.9,
  eps = 1e-08,
  momentum = 0.9
)

optimizers_adam(step_size, b1 = 0.9, b2 = 0.999, eps = 1e-08)

optimizers_adamax(step_size, b1 = 0.9, b2 = 0.999, eps = 1e-08)

optimizers_sm3(step_size, momentum = 0.9)
```

## Arguments

- step_size:

  Positive scalar, or a schedule (a function of the iteration index
  returning a positive scalar).

- mass:

  Momentum coefficient (`momentum`, `nesterov`).

- momentum:

  Momentum parameter (`adagrad`, `rmsprop_momentum`, `sm3`).

- gamma:

  Decay parameter (`rmsprop`, `rmsprop_momentum`).

- eps:

  Small constant for numerical stability.

- b1, b2:

  Exponential decay rates of the first and second moment estimates
  (`adam`, `adamax`).

## Value

An optimizer `list(init_fn, update_fn, params_fn)`.

## Examples

``` r
if (FALSE) { # \dontrun{
opt <- optimizers_adam(1e-3)
state <- opt$init_fn(list(w = jnp_ones(3), b = 0))
grads <- list(w = jnp_ones(3), b = 1)
state <- opt$update_fn(0L, grads, state)
opt$params_fn(state)
} # }
```
