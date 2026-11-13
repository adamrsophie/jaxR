# Integrate a system of ordinary differential equations

Port of `jax.experimental.ode.odeint`: adaptive step size Dormand-Prince
(Runge-Kutta 4(5)) integration. Reverse mode derivatives use the adjoint
method (an augmented system integrated backwards in time); forward mode
differentiation is not supported, as in JAX. `experimental_ode_odeint`
is an alias.

## Usage

``` r
odeint(
  func,
  y0,
  t,
  ...,
  rtol = 1.4e-08,
  atol = 1.4e-08,
  mxstep = Inf,
  hmax = Inf
)

experimental_ode_odeint(
  func,
  y0,
  t,
  ...,
  rtol = 1.4e-08,
  atol = 1.4e-08,
  mxstep = Inf,
  hmax = Inf
)
```

## Arguments

- func:

  Function `func(y, t, ...)` returning the time derivative of `y` (same
  structure as `y0`).

- y0:

  Initial state (array or pytree of arrays).

- t:

  Strictly increasing float array of evaluation times; `t[0]` is the
  time of `y0`.

- ...:

  Extra arguments passed to `func` (arrays or pytrees of arrays).

- rtol, atol:

  Relative and absolute local error tolerances.

- mxstep:

  Maximum number of steps per output time.

- hmax:

  Maximum step size.

## Value

The solution at the times `t`: the structure of `y0` with a new leading
axis of length `length(t)`.
