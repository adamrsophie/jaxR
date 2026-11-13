# Ports of tests/key_reuse_test.py.

kr_with <- function(code) with_jax_config("jax_debug_key_reuse", TRUE, code)
consumed <- function(key) jaxR:::kr_assert_consumed(key)
unconsumed <- function(key) jaxR:::kr_assert_unconsumed(key)

test_that("random functions consume their keys and sources make new ones", {
  skip_if_no_backend()
  kr_with({
    key <- random_key(0)
    unconsumed(key)
    k2 <- random_clone(key)
    random_uniform(key)
    consumed(key)
    unconsumed(k2)
    expect_error(random_uniform(key), class = "jax_key_reuse_error")
    expect_error(random_bits(key), "Previously-consumed key passed to random_bits at index 0")

    key <- random_key(1)
    ks <- random_split(key)
    consumed(key)
    unconsumed(ks)
    expect_error(random_split(key), "random_split")

    # fold_in does not consume its key.
    key <- random_key(2)
    f <- random_fold_in(key, 3L)
    unconsumed(key)
    unconsumed(f)

    key <- random_key(3)
    random_gamma(key, 1.5)
    consumed(key)

    # Wrapping key data makes a new key; unwrapping does not consume it.
    data <- random_key_data(random_key(4))
    w1 <- random_wrap_key_data(data)
    random_normal(w1)
    w2 <- random_wrap_key_data(data)
    unconsumed(w2)
    random_key_data(w2)
    unconsumed(w2)

    # The consume and assert helpers.
    key <- random_key(5)
    expect_error(consumed(key), "Expected key to be consumed")
    jaxR:::kr_consume_key(key)
    expect_error(unconsumed(key), "Expected key to not be consumed")
  })
  # Without the option nothing is tracked.
  key <- random_key(0)
  random_uniform(key)
  expect_silent(random_uniform(key))
})

test_that("slicing consumes the selected keys, unstacking all of them", {
  skip_if_no_backend()
  kr_with({
    keys <- random_split(random_key(0), 3L)
    k0 <- keys[0]
    unconsumed(k0)
    jaxR:::kr_assert_consumed(keys, c(TRUE, FALSE, FALSE))
    expect_error(keys[0], "slice")
    random_normal(keys[1])
    jaxR:::kr_assert_consumed(keys, c(TRUE, TRUE, FALSE))

    keys <- random_split(random_key(1), 2L)
    parts <- as.list(keys)
    consumed(keys)
    unconsumed(parts[[1L]])
    expect_error(as.list(keys), "unstack")
  })
})

test_that("jit applies the key reuse signature of the function", {
  skip_if_no_backend()
  kr_with({
    key <- random_key(0)
    jit(function(k) random_bits(k))(key)
    expect_error(jit(function(k) random_bits(k))(key),
                 "Previously-consumed key passed to jit-compiled function at index 0")

    # A key the function closes over is consumed by each call.
    key <- random_key(1)
    f <- jit(function() random_uniform(key))
    f()
    consumed(key)
    expect_error(f(), "jit-compiled function")

    # Reuse inside the traced function.
    g <- jit(function() {
      k <- random_key(0)
      random_bits(k) + random_bits(k)
    })
    expect_error(g(), "In random_bits, argument 0 is already consumed.")

    # Consumed outputs stay consumed, forwarded outputs keep the state of the input,
    # new keys are not consumed.
    h <- jit(function(k) {
      ks <- random_split(k)
      list(k, ks, random_key(7))
    })
    key <- random_key(2)
    out <- h(key)
    consumed(key)
    consumed(out[[1L]])
    unconsumed(out[[2L]])
    unconsumed(out[[3L]])
    pass <- jit(function(k) k)
    key <- random_key(3)
    unconsumed(pass(key))
    random_normal(key)
    consumed(pass(key))
  })
})

test_that("loops and conditionals follow the rules of JAX", {
  skip_if_no_backend()
  kr_with({
    # A scan body that splits its carried key is fine.
    r <- lax_scan(function(k, x) {
      ks <- as.list(random_split(k))
      list(ks[[1L]], random_uniform(ks[[2L]]))
    }, random_key(0), NULL, length = 3L)
    unconsumed(r[[1L]])
    expect_identical(dim(r[[2L]]), 3L)
    # Consuming the carried key without making a new one is reuse.
    expect_error(lax_scan(function(k, x) list(k, random_uniform(k)), random_key(0), NULL, length = 3L),
                 "scan body function leads to key reuse")
    # So is consuming a key the body closes over.
    key <- random_key(1)
    expect_error(lax_scan(function(c, x) list(c + random_uniform(key), NULL), 0, NULL, length = 2L),
                 "key constants are repeatedly consumed")
    unconsumed(key)
    expect_error(lax_fori_loop(0L, 3L, function(i, k) { random_uniform(k); k }, random_key(0)),
                 class = "jax_key_reuse_error")
    # A while loop with a traced bound.
    w <- jit(function(n) lax_while_loop(function(s) s[[1L]] < n,
                                        function(s) {
                                          ks <- as.list(random_split(s[[2L]]))
                                          list(s[[1L]] + 1L, ks[[1L]])
                                        }, list(0L, random_key(0))))
    expect_no_error(w(3L))
    expect_error(lax_while_loop(function(s) random_uniform(s) < 2, function(s) s, random_key(0)),
                 "while_loop cond function")

    # cond consumes a key that either branch consumes.
    key <- random_key(2)
    lax_cond(TRUE, function(k) random_uniform(k), function(k) 0, key)
    consumed(key)
    key <- random_key(3)
    lax_cond(FALSE, function(k) random_uniform(k), function(k) random_normal(k), key)
    consumed(key)
    expect_error(lax_cond(TRUE, function(k) random_uniform(k), function(k) 0, key),
                 "Previously-consumed key passed to cond at index 1")
  })
})

test_that("the samplers do not report the uses of keys inside them", {
  skip_if_no_backend()
  kr_with({
    expect_no_error(random_poisson(random_key(0), 3, 6L))
    expect_no_error(random_multinomial(random_key(0), 10, c(0.2, 0.3, 0.5)))
    expect_no_error(random_binomial(random_key(0), 10, 0.3, 4L))
    expect_no_error(random_permutation(random_key(0), 5L))
    expect_no_error(random_choice(random_key(0), 5L, 3L, replace = FALSE))
    expect_no_error(random_dirichlet(random_key(0), jnp_ones(3)))
    expect_no_error(random_beta(random_key(0), 2, 3))
    expect_no_error(random_t(random_key(0), 3))
    expect_no_error(jit(function(k) random_poisson(k, 3, 6L))(random_key(1)))
  })
})

test_that("vmap consumes the mapped keys", {
  skip_if_no_backend()
  kr_with({
    keys <- random_split(random_key(0), 3L)
    vmap(function(k) random_uniform(k))(keys)
    consumed(keys)
    expect_error(vmap(function(k) random_uniform(k))(keys), class = "jax_key_reuse_error")
    # A zero sized key axis consumes nothing (JAX issue 37859).
    f <- function(key) {
      ks <- as.list(random_split(key))
      random_bits(ks[[1L]]) + random_bits(ks[[2L]])
    }
    expect_no_error(vmap(f)(random_split(random_key(1), 0L)))
  })
})

test_that("tracing consumes nothing; boundaries that run a function do", {
  skip_if_no_backend()
  kr_with({
    # make_jaxpr and eval_shape only trace (as in JAX).
    key <- random_key(0)
    make_jaxpr(function() random_normal(key))()
    eval_shape(function() random_normal(key))
    unconsumed(key)
    expect_no_error(random_normal(key))
    # checkpoint applies the signature of its function (remat).
    key <- random_key(1)
    ck <- checkpoint(function(k) random_normal(k))
    ck(key)
    expect_error(ck(key), "remat")
    # grad and custom_jvp run the function.
    key <- random_key(2)
    grad(function(x) x * random_normal(key))(1)
    consumed(key)
    key <- random_key(3)
    cj <- custom_jvp(function(x, k) x * random_normal(k), nondiff_argnums = 1L)
    cj <- custom_jvp_defjvp(cj, function(k, p, t) list(cj(p[[1L]], k), t[[1L]]))
    expect_no_error(cj(1, key))
    expect_error(cj(1, key), class = "jax_key_reuse_error")
    # Nested jit consumes the outer key once.
    key <- random_key(4)
    expect_no_error(jit(function(k) jit(function(k) random_normal(k))(k))(key))
    consumed(key)
    # lax_map closes over a key: reuse in the loop.
    key <- random_key(5)
    expect_error(lax_map(function(x) x + random_normal(key), jnp_ones(3)), "key constants")
  })
})

test_that("initializers and stax layers use each key once", {
  skip_if_no_backend()
  kr_with({
    inits <- list(nn_initializers_glorot_normal(), nn_initializers_glorot_uniform(),
                  nn_initializers_he_normal(), nn_initializers_he_uniform(),
                  nn_initializers_lecun_normal(), nn_initializers_lecun_uniform(),
                  nn_initializers_normal(), nn_initializers_uniform(),
                  nn_initializers_truncated_normal(), nn_initializers_orthogonal(),
                  nn_initializers_delta_orthogonal(), nn_initializers_variance_scaling(1, "fan_in", "normal"),
                  nn_initializers_zeros, nn_initializers_ones, nn_initializers_constant(2))
    for (i in seq_along(inits)) {
      shape <- if (i == 11L) c(3L, 3L, 4L) else c(4L, 3L)
      expect_no_error(inits[[i]](random_key(i), shape))
    }
    layers <- list(stax_Dense(3L), stax_serial(stax_Dense(5L), stax_Relu, stax_Dense(2L)),
                   stax_BatchNorm(), stax_Conv(2L, c(2L, 2L)), stax_ConvTranspose(2L, c(2L, 2L)),
                   stax_serial(stax_FanOut(2L), stax_parallel(stax_Dense(2L), stax_Dense(2L)),
                               stax_FanInSum))
    shapes <- list(c(-1L, 4L), c(-1L, 4L), c(2L, 3L, 3L, 2L), c(1L, 4L, 4L, 2L),
                   c(1L, 4L, 4L, 2L), c(-1L, 4L))
    for (i in seq_along(layers)) expect_no_error(layers[[i]]$init_fun(random_key(10L + i), shapes[[i]]))
    drop <- stax_Dropout(0.5)
    p <- drop$init_fun(random_key(20), c(2L, 3L))$params
    expect_no_error(drop$apply_fun(p, jnp_ones(c(2L, 3L)), rng = random_key(21)))
  })
})

test_that("slices, vmap masks, clones, closed over keys and cond outputs follow JAX", {
  skip_if_no_backend()
  # Static slices inside jit are sinks (also across cached calls).
  kr_with({
    f <- jit(function(keys) random_uniform(keys[0]) + random_uniform(keys[0]))
    expect_error(f(random_split(random_key(0), 2L)), "In slice, argument 0 is already consumed")
    g <- jit(function(keys) random_uniform(keys[0]))
    keys <- random_split(random_key(0), 2L)
    g(keys)
    expect_error(g(keys), "Previously-consumed key passed to jit-compiled function at index 0")
    # Adding an axis forwards the key.
    key <- random_key(0)
    b <- key[newaxis]
    expect_no_error(random_uniform(key))
    # Masks of a 2 by 3 key array are row major positions.
    k2 <- random_split(random_key(0), c(2L, 3L))
    random_uniform(k2[1, 2])
    expect_no_error(random_uniform(k2[2, 1]))
    expect_error(random_uniform(k2[1, 2]), class = "jax_key_reuse_error")
    # vmap consumes only the mapped elements that were used.
    keys <- random_split(random_key(0), c(2L, 3L))
    vmap(function(k) random_uniform(k[0]))(keys)
    expect_no_error(vmap(function(k) random_uniform(k[1]))(keys))
    expect_error(vmap(function(k) random_uniform(k[0]))(keys), class = "jax_key_reuse_error")
    # Lowering only traces.
    key <- random_key(0)
    jit(function() random_uniform(key))$lower()
    unconsumed(key)
    # A closed over key returned by jit has its current state.
    key <- random_key(0)
    f <- jit(function() key)
    f()
    random_uniform(key)
    expect_error(random_uniform(f()), class = "jax_key_reuse_error")
    # A cond output that one branch makes and the other forwards is new.
    key <- random_key(0)
    out <- lax_cond(TRUE, function(k) random_split(k)[0], function(k) k, key)
    expect_no_error(random_uniform(out))
  })
  # Clones and wrapped keys made with the check off have their own state.
  for (m in c("clone", "wrap")) {
    key <- random_key(0)
    other <- if (m == "clone") random_clone(key) else random_wrap_key_data(random_key_data(key))
    kr_with({
      random_uniform(key)
      expect_no_error(random_uniform(other))
    })
  }
})

test_that("aliases, gathers, transposes and custom rules under jit follow JAX", {
  skip_if_no_backend()
  kr_with({
    key <- random_key(0)
    random_uniform(key)
    out <- lax_cond(TRUE, function() key, function() key)
    expect_error(random_uniform(out), class = "jax_key_reuse_error")
    key <- random_key(0)
    tr <- linear_transpose(function(x) x * random_uniform(key), 1)
    tr(1)
    expect_error(random_uniform(key), class = "jax_key_reuse_error")
    f <- jit(function(keys) {
      a <- keys[as_jax(c(0L, 2L))]
      random_key_data(keys[as_jax(c(0L, 2L))])
    })
    expect_identical(dim(f(random_split(random_key(0), 4L))), c(2L, 2L))
    keys <- random_split(random_key(0), c(2L, 3L))
    vmap(function(k) random_uniform(k[0]), in_axes = 1L)(keys)
    expect_error(vmap(function(k) random_uniform(k[1]), in_axes = 1L)(keys), class = "jax_key_reuse_error")
    expect_error(jit(function(key) { e <- key[newaxis]; random_uniform(e[0]) + random_uniform(key) })(random_key(0)),
                 class = "jax_key_reuse_error")
    expect_error(jit(function(k, i) random_uniform(k[i]) + random_uniform(k[i]))(random_split(random_key(0), 3L), as_jax(0L)),
                 class = "jax_key_reuse_error")
    key <- random_key(1)
    cj <- custom_jvp(function(x, k) x * random_uniform(k), nondiff_argnums = 1L)
    cj <- custom_jvp_defjvp(cj, function(k, p, t) list(cj(p[[1L]], k), t[[1L]]))
    jit(function(x) cj(x, key))(1)
    expect_error(random_uniform(key), class = "jax_key_reuse_error")
  })
})

test_that("custom derivatives under jit(grad) and jit(vmap) consume a key once", {
  skip_if_no_backend()
  kr_with({
    for (tr in c("grad", "vmap")) {
      key <- random_key(0)
      cj <- custom_jvp(function(x, k) x * random_uniform(k), nondiff_argnums = 1L)
      cj <- custom_jvp_defjvp(cj, function(k, p, t) list(cj(p[[1L]], k), t[[1L]]))
      f <- function(x) cj(x, key)
      expect_no_error(if (tr == "grad") jit(grad(f))(1) else jit(vmap(f))(jnp_ones(3)))
      consumed(key)
    }
    key <- random_key(1)
    cv <- custom_vjp(function(x) x * random_uniform(key))
    cv <- custom_vjp_defvjp(cv, function(x) list(x * random_uniform(key), NULL), function(r, g) list(g))
    expect_no_error(jit(grad(cv))(1))
    consumed(key)
  })
})

test_that("jit of vmap of custom derivative functions applies their key signature", {
  skip_if_no_backend()
  with_jax_config("jax_debug_key_reuse", TRUE, {
    for (kind in c("jvp", "vjp")) for (reuse in c(TRUE, FALSE)) {
      key <- random_key(0)
      if (kind == "jvp") {
        f <- custom_jvp(function(x) x * random_uniform(key))
        f <- custom_jvp_defjvp(f, function(p, t) list(f(p[[1]]), t[[1]]))
      } else {
        f <- custom_vjp(function(x) x * random_uniform(key))
        f <- custom_vjp_defvjp(f, function(x) list(x * random_uniform(key), NULL), function(r, g) list(g))
      }
      h <- jit(function(x) if (reuse) vmap(f)(x) + random_uniform(key) else vmap(f)(x))
      if (reuse) expect_error(h(jnp_ones(3)), class = "jax_key_reuse_error")
      else expect_equal(dim(h(jnp_ones(3))), 3L)
    }
  })
})

test_that("vmap of a custom derivative consumes its mapped keys", {
  skip_if_no_backend()
  with_jax_config("jax_debug_key_reuse", TRUE, {
    fj <- custom_jvp(function(k) random_uniform(k),
                     function(p, t) list(random_uniform(p[[1]]), as_jax(0)))
    fv <- custom_vjp(function(k) random_uniform(k), function(k) list(random_uniform(k), NULL),
                     function(r, ct) list(NULL))
    for (f in list(fj, fv)) {
      keys <- random_split(random_key(0), 3L)
      h <- vmap(f)
      expect_length(as.numeric(h(keys)), 3L)
      expect_error(h(keys), class = "jax_key_reuse_error")
    }
  })
})
