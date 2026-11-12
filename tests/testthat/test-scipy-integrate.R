# Tests of R/scipy_integrate.R (ports of jax/_src/scipy/integrate.py and jax/experimental/ode.py).
# Expected values come from jax 0.12.0.1 (reference wheel, x64 enabled), written as hexadecimal
# doubles in R (column major) order.

fx <- list()
fx$ts <- c(0, 0x1.999999999999ap-2, 0x1.999999999999ap-1, 0x1.3333333333334p+0, 0x1.999999999999ap+0, 0x1.0000000000000p+1)
fx$ode_decay <- array(c(0x1.0000000000000p+0, 0x1.82f615fcab737p-1, 0x1.247583bf6728bp-1, 0x1.ba1252f6ee288p-2, 0x1.4e1c4a5bef7aap-2, 0x1.f907d425abae1p-3, 0x1.0000000000000p+1, 0x1.82f615fcab737p+0, 0x1.247583bf6728bp+0, 0x1.ba1252f6ee288p-1, 0x1.4e1c4a5bef7aap-1, 0x1.f907d425abae1p-2), dim = c(6L, 2L))
fx$ode_pend_theta <- c(0x1.8552e8777604bp+1, 0x1.800b5856c211ep+1, 0x1.6ca1d78ba1018p+1, 0x1.3d44b98466585p+1, 0x1.aa47a92029785p+0, 0x1.f960169118335p-3)
fx$ode_pend_omega <- c(0, -0x1.bac5d5543be3dp-3, -0x1.2b747137ccf75p-1, -0x1.5c7e06a08dbcdp+0, -0x1.6871eea654342p+1, -0x1.00384b6b364d4p+2)
fx$ode_grad_y0 <- c(0x1.202512d51c0acp+2, 0x1.202512d51c0acp+3)
fx$ode_grad_k <- c(-0x1.41a29bc8db081p+3)
fx$ode_grad_t <- c(0x1.09241c2d60883p-1, 0, 0, 0, 0, -0x1.09241c2d60883p-1)
fx$ode_rtol <- array(c(0x1.0000000000000p+0, 0x1.d7b779154e390p-1, 0x1.64cd9c897d396p-1, 0x1.72b2adbbb8ae3p-2, -0x1.e31fd0385f0f1p-6, -0x1.aaedb89d8a419p-2, 0, -0x1.8f001c776dd8fp-2, -0x1.6f6796bbaaebdp-1, -0x1.dd57ea11d82dcp-1, -0x1.0042077b89427p+0, -0x1.d1e353eb94c1ep-1), dim = c(6L, 2L))
fx$ode_f32 <- array(c(0x1.0000000000000p+0, 0x1.82f6180000000p-1, 0x1.2475820000000p-1, 0x1.ba123e0000000p-2, 0x1.4e1c480000000p-2, 0x1.f907d20000000p-3, 0x1.0000000000000p+1, 0x1.82f6180000000p+0, 0x1.2475820000000p+0, 0x1.ba123e0000000p-1, 0x1.4e1c480000000p-1, 0x1.f907d20000000p-2), dim = c(6L, 2L))
fx$ode_lotka <- array(c(0x1.0000000000000p+0, 0x1.60ad4d561b902p+0, 0x1.158dd97cde1bbp+1, 0x1.c6e598e400318p+1, 0x1.6916e472e42efp+2, 0x1.b1bd77adb7f3ap+2, 0x1.0000000000000p+0, 0x1.ea4df1c2b29fdp-2, 0x1.274ee4b4e3a9cp-2, 0x1.10dbd90829dadp-2, 0x1.fb6e3cf5a270cp-2, 0x1.01a1eb86893d9p+1), dim = c(6L, 2L))
fx$ode_lotka_grad_p <- c(0x1.b0c940b987ecep+2, -0x1.12ac5bd2dbceap+2, 0x1.ac71060c3f06bp+1, -0x1.e8d862c14d877p+2)
fx$ode_grad_closure <- c(-0x1.758e8819a4048p+1)
fx$ode_mxstep <- array(c(0x1.0000000000000p+0, 0x1.82f689e148322p-1, 0x1.24767cdcffc36p-1, 0x1.ba153331ce8e6p-2, 0x1.4e1fe2b0bce7cp-2, 0x1.f90f9b8f2e0f9p-3, 0x1.0000000000000p+1, 0x1.82f689e148322p+0, 0x1.24767cdcffc36p+0, 0x1.ba153331ce8e6p-1, 0x1.4e1fe2b0bce7cp-1, 0x1.f90f9b8f2e0f9p-2), dim = c(6L, 2L))
fx$ode_hmax <- array(c(0x1.0000000000000p+0, 0x1.82f61628d4724p-1, 0x1.247583b0dc866p-1, 0x1.ba1252cdf6d80p-2, 0x1.4e1c4a5d1a7a8p-2, 0x1.f907d43b8f754p-3, 0x1.0000000000000p+1, 0x1.82f61628d4724p+0, 0x1.247583b0dc866p+0, 0x1.ba1252cdf6d80p-1, 0x1.4e1c4a5d1a7a8p-1, 0x1.f907d43b8f754p-2), dim = c(6L, 2L))
fx$trap_y <- array(c(0x1.0000000000000p+0, 0, 0x1.0000000000000p+1, 0x1.0000000000000p+0, 0x1.0000000000000p+2, 0x1.8000000000000p+1, 0x1.c000000000000p+2, 0x1.8000000000000p+1), dim = c(2L, 4L))
fx$trap_dx <- c(0x1.4000000000000p+2, 0x1.6000000000000p+1)
fx$trap_x <- c(0x1.a000000000000p+3, 0x1.e000000000000p+2)
fx$trap_axis0 <- c(0x1.0000000000000p-1, 0x1.8000000000000p+0, 0x1.c000000000000p+1, 0x1.4000000000000p+2)

x64 <- function(code) with_jax_config("jax_enable_x64", TRUE, code)
decay <- function(y, t, k) -k * y

test_that("trapezoid matches the wheel", {
  skip_if_no_backend()
  skip_if_not(exists("jnp_trapezoid", mode = "function"))
  x64({
    expect_jax_equal(scipy_integrate_trapezoid(fx$trap_y, dx = 0.5), fx$trap_dx)
    expect_jax_equal(scipy_integrate_trapezoid(fx$trap_y, x = c(0, 1, 3, 4)), fx$trap_x)
    expect_jax_equal(scipy_integrate_trapezoid(fx$trap_y, axis = 0L), fx$trap_axis0)
    expect_jax_equal(jit(scipy_integrate_trapezoid)(fx$trap_y), fx$trap_dx * 2)
  })
})

test_that("odeint matches the wheel", {
  skip_if_no_backend()
  x64({
    expect_jax_equal(odeint(decay, c(1, 2), fx$ts, 0.7), fx$ode_decay, tolerance = 1e-12)
    out <- odeint(function(y, t, b, c) list(theta = y$omega,
                                            omega = -b * y$omega - c * jnp_sin(y$theta)),
                  list(theta = pi - 0.1, omega = 0), fx$ts, 0.25, 5.0)
    expect_named(out, c("theta", "omega"))
    expect_jax_equal(out$theta, fx$ode_pend_theta, tolerance = 1e-10)
    expect_jax_equal(out$omega, fx$ode_pend_omega, tolerance = 1e-10)
    expect_jax_equal(odeint(function(y, t) jnp_stack(list(y[1L], -y[0L])), c(1, 0), fx$ts,
                            rtol = 1e-3, atol = 1e-3), fx$ode_rtol, tolerance = 1e-12)
    lotka <- function(y, t, p) {
      jnp_stack(list(p[0L] * y[0L] - p[1L] * y[0L] * y[1L], -p[2L] * y[1L] + p[3L] * y[0L] * y[1L]))
    }
    expect_jax_equal(odeint(lotka, c(1, 1), fx$ts, c(1.5, 1, 3, 1)), fx$ode_lotka,
                     tolerance = 1e-10)
    expect_jax_equal(odeint(decay, c(1, 2), fx$ts, 0.7, mxstep = 2), fx$ode_mxstep,
                     tolerance = 1e-12)
    expect_jax_equal(odeint(decay, c(1, 2), fx$ts, 0.7, hmax = 0.05, rtol = 1e-4), fx$ode_hmax,
                     tolerance = 1e-12)
    expect_jax_equal(experimental_ode_odeint(decay, c(1, 2), fx$ts, 0.7), fx$ode_decay,
                     tolerance = 1e-12)
    expect_jax_equal(jit(function(y0, k) odeint(decay, y0, fx$ts, k))(c(1, 2), 0.7), fx$ode_decay,
                     tolerance = 1e-12)
  })
  expect_jax_equal(odeint(decay, c(1, 2), fx$ts, 0.7), fx$ode_f32, dtype = "float32",
                   tolerance = 1e-5)
})

test_that("odeint gradients use the adjoint method", {
  skip_if_no_backend()
  x64({
    loss <- function(y0, k) jnp_sum(odeint(decay, y0, fx$ts, k)^2)
    g <- grad(loss, argnums = c(0L, 1L))(as_jax(c(1, 2)), as_jax(0.7))
    expect_jax_equal(g[[1L]], fx$ode_grad_y0, tolerance = 1e-7)
    expect_jax_equal(g[[2L]], fx$ode_grad_k, tolerance = 1e-7)
    g <- grad(function(t) jnp_sum(odeint(decay, c(1, 2), t, 0.7)[-1L]))(as_jax(fx$ts))
    expect_jax_equal(g, fx$ode_grad_t, tolerance = 1e-7)
    lotka <- function(y, t, p) {
      jnp_stack(list(p[0L] * y[0L] - p[1L] * y[0L] * y[1L], -p[2L] * y[1L] + p[3L] * y[0L] * y[1L]))
    }
    g <- grad(function(p) jnp_sum(odeint(lotka, c(1, 1), fx$ts, p)[-1L]))(as_jax(c(1.5, 1, 3, 1)))
    expect_jax_equal(g, fx$ode_lotka_grad_p, tolerance = 1e-6)
    # A function closing over the differentiated value (closure conversion).
    g <- grad(function(k) jnp_sum(odeint(function(y, t) -k * y * t, 1, fx$ts)))(as_jax(0.3))
    expect_jax_equal(g, fx$ode_grad_closure, tolerance = 1e-7)
    check_grad_fd(function(y0) jnp_sum(odeint(decay, y0, fx$ts, 0.7)),
                  as_jax(c(1, 2), "float64"), tolerance = 1e-6)
    ys <- vmap(function(y0) odeint(decay, y0, fx$ts, 0.7))(rbind(c(1, 2), c(2, 4)))
    expect_equal(as.array(ys)[2L, , ], 2 * fx$ode_decay, tolerance = 1e-8)
  })
  expect_error(jvp(function(y0) odeint(decay, y0, fx$ts, 0.7), list(c(1, 2)), list(c(1, 1))),
               "forward-mode")
})

test_that("odeint checks its arguments", {
  expect_error(odeint(decay, c(1, 2), 0:3, 0.7), "t must be an array of floats")
  expect_error(odeint(decay, c(1, 2), fx$ts, "k"), "must be arrays or scalars")
})
