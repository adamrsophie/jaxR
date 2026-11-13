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
