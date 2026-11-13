# Optimizer state and triple

`optimizers_OptimizerState` ports the `OptimizerState` named tuple: a
pytree node with one child (`packed_state`, the list of flattened per
parameter states) and the parameter treedef and per parameter state
treedefs as auxiliary data. `optimizers_Optimizer` ports the `Optimizer`
named tuple `list(init_fn, update_fn, params_fn)`.
`optimizers_optimizer` ports the decorator that turns a constructor of
single array optimizer functions into one that works on pytrees of
parameters (nested lists of arrays).

## Usage

``` r
optimizers_OptimizerState(packed_state, tree_def, subtree_defs)

optimizers_Optimizer(init_fn, update_fn, params_fn)

optimizers_optimizer(opt_maker)
```

## Arguments

- packed_state:

  List of flattened per parameter states.

- tree_def:

  Treedef of the parameters.

- subtree_defs:

  List of treedefs of the per parameter states.

- init_fn, update_fn, params_fn:

  The optimizer functions: `init_fn(params)`,
  `update_fn(step, grads, opt_state)` and `params_fn(opt_state)` (the
  `get_params` function).

- opt_maker:

  A function returning `list(init, update, get_params)` that work on
  single arrays: `init(x0)` returns a state pytree,
  `update(i, g, state)` the new state and `get_params(state)` the array.

## Value

`optimizers_OptimizerState()` an object of class `OptimizerState`,
`optimizers_Optimizer()` a named list and `optimizers_optimizer()` an
optimizer constructor.

## Examples

``` r
if (FALSE) { # \dontrun{
my_sgd <- optimizers_optimizer(function(step_size) {
  list(function(x0) x0, function(i, g, x) x - step_size * g, function(x) x)
})
opt <- my_sgd(0.1)
state <- opt$init_fn(list(w = jnp_ones(2)))
} # }
```
