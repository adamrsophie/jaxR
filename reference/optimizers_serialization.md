# Optimizer state serialization

`optimizers_unpack_optimizer_state` converts an `OptimizerState` into a
pytree with the structure of the parameters whose leaves are `JoinPoint`
objects holding each parameter's state (a pytree), useful for
serialization. `optimizers_pack_optimizer_state` is its inverse.
`optimizers_JoinPoint` marks the boundary between the two nested pytrees
(it is a leaf).

## Usage

``` r
optimizers_JoinPoint(subtree)

optimizers_unpack_optimizer_state(opt_state)

optimizers_pack_optimizer_state(marked_pytree)
```

## Arguments

- subtree:

  The pytree held by a `JoinPoint`.

- opt_state:

  An `OptimizerState`.

- marked_pytree:

  A pytree with `JoinPoint` leaves.

## Value

A pytree with `JoinPoint` leaves, an `OptimizerState`, or a `JoinPoint`.

## Examples

``` r
if (FALSE) { # \dontrun{
opt <- optimizers_momentum(0.1, 0.9)
state <- opt$init_fn(list(a = jnp_ones(2)))
marked <- optimizers_unpack_optimizer_state(state)
state2 <- optimizers_pack_optimizer_state(marked)
} # }
```
