# jax.experimental.random (stateful generator) and vmap over Ref arguments: values
# recorded from JAX by tools/gen_stateful_rng_fixture.py.

srng_ref <- function() {
  skip_if_no_backend()
  testthat::skip_if_not_installed("jsonlite")
  jsonlite::fromJSON(testthat::test_path("fixtures", "stateful_rng.json"), simplifyVector = FALSE)
}

# Shape, dtype and values of `x` against a recorded entry. Integers and key data must
# agree exactly; floats to a few units in the last place.
srng_expect <- function(ref, name, x) {
  e <- ref[[name]]
  expect_false(is.null(e), info = name)
  x <- as_jax(x)
  expect_identical(as.integer(dim(x)), as.integer(unlist(e$shape)), info = name)
  expect_identical(dtype(x), e$dtype, info = name)
  got <- as.numeric(as.array(jnp_ravel(x)))
  want <- as.numeric(unlist(e$value))
  if (grepl("float", e$dtype)) expect_equal(got, want, tolerance = if (e$dtype == "float32") 2e-6 else 1e-12, info = name)
  else expect_identical(got, want, info = name)
}

stateful_rng <- experimental_random_stateful_rng

test_that("the stateful generator draws the values of JAX", {
  ref <- srng_ref()
  rng <- stateful_rng(42)
  expect_s3_class(rng, "jax_stateful_prng")
  srng_expect(ref, "u1", rng$uniform())
  srng_expect(ref, "u2", rng$uniform())
  ju <- jit(function() rng$uniform())
  srng_expect(ref, "ju1", ju())
  srng_expect(ref, "ju2", ju())
  srng_expect(ref, "k1", random_key_data(rng$key()))
  srng_expect(ref, "k2", random_key_data(rng$key()))
  srng_expect(ref, "k3", random_key_data(rng$key(3L)))
  srng_expect(ref, "k23", random_key_data(rng$key(c(2L, 3L))))
  srng_expect(ref, "counter", rng$counter$get())

  rng <- stateful_rng(0)
  srng_expect(ref, "key0a", random_key_data(rng$key()))
  srng_expect(ref, "key0b", random_key_data(rng$key()))

  rng <- stateful_rng(7)
  srng_expect(ref, "random", rng$random())
  srng_expect(ref, "random3", rng$random(3L))
  srng_expect(ref, "random23", rng$random(c(2L, 3L)))
  srng_expect(ref, "unif_lh", rng$uniform(-1, jnp_array(c(1, 2, 3))))
  srng_expect(ref, "unif_size", rng$uniform(2, 5, size = 4L))
  srng_expect(ref, "normal", rng$normal())
  srng_expect(ref, "normal_ls", rng$normal(jnp_array(c(0, 10)), 2))
  srng_expect(ref, "normal_size", rng$normal(1, 3, c(2L, 2L)))
  srng_expect(ref, "int1", rng$integers(10L))
  srng_expect(ref, "int2", rng$integers(5L, 10L, 6L))
  srng_expect(ref, "int3", rng$integers(jnp_array(c(0L, 100L)), jnp_array(c(10L, 200L))))

  srng_expect(ref, "rbg", stateful_rng(5, impl = "rbg")$uniform(size = 3L))
  with_jax_config("jax_enable_x64", TRUE, {
    r64 <- stateful_rng(42)
    srng_expect(ref, "x64_u", r64$uniform())
    srng_expect(ref, "x64_i", r64$integers(10L))
    srng_expect(ref, "x64_n", r64$normal(size = 2L))
  })
})

test_that("the stateful generator works under jit, grad, vmap and scan", {
  ref <- srng_ref()
  rng <- stateful_rng(9)
  f <- jit(function(r, x) x + r$normal(size = 2L) + r$uniform())
  srng_expect(ref, "jit_arg1", f(rng, jnp_ones(2L)))
  srng_expect(ref, "jit_arg2", f(rng, jnp_ones(2L)))
  srng_expect(ref, "jit_arg_counter", rng$counter$get())

  g <- function(x) {
    r <- stateful_rng(3)
    x * r$normal() + r$uniform()
  }
  srng_expect(ref, "inner", jit(g)(2))
  srng_expect(ref, "inner_grad", grad(g)(2))

  rng <- stateful_rng(123)
  srng_expect(ref, "vmap_split", vmap(function(r, x) x + r$uniform())(rng$split(3L), jnp_zeros(3L)))
  sp <- stateful_rng(123)$split(3L)
  vf <- vmap(function(r, x) x + r$uniform())
  srng_expect(ref, "vmap_split_1", vf(sp, jnp_zeros(3L)))
  srng_expect(ref, "vmap_split_2", vf(sp, jnp_zeros(3L)))
  srng_expect(ref, "vmap_split_counter", sp$counter$get())
  srng_expect(ref, "vmap_vmap",
              vmap(vmap(function(r) r$uniform()))(stateful_rng(1)$split(c(2L, 3L))))

  kids <- stateful_rng(123)$spawn(2L)
  expect_length(kids, 2L)
  srng_expect(ref, "spawn", jnp_stack(lapply(kids, function(r) r$integers(0L, 10L, 2L))))

  rs <- stateful_rng(4)
  out <- lax_scan(function(carry, x) list(carry, rs$uniform()), 0, jnp_arange(3))
  srng_expect(ref, "scan", out[[2L]])
  srng_expect(ref, "scan_counter", rs$counter$get())
  srng_expect(ref, "after_scan", rs$uniform())
})

test_that("the stateful generator reports the errors of JAX", {
  ref <- srng_ref()
  errs <- ref$errors
  cls <- c(ValueError = "jax_value_error", TypeError = "jax_type_error")
  sp <- stateful_rng(1)$split(2L)
  e <- expect_error(sp$key(), class = cls[[errs$split_key[[1L]]]])
  expect_identical(conditionMessage(e), errs$split_key[[2L]])
  e <- expect_error(jit(function() stateful_rng()$uniform())(), class = cls[[errs$no_seed[[1L]]]])
  expect_identical(conditionMessage(e), errs$no_seed[[2L]])
  expect_error(experimental_random_StatefulPRNG(jnp_zeros(2L), new_ref(0L)),
               "Expected base_key to be a typed PRNG key", class = cls[[errs$bad_key[[1L]]]])
  expect_error(experimental_random_StatefulPRNG(random_key(0), 0L),
               "Expected counter to be a scalar integer ref", class = cls[[errs$bad_counter[[1L]]]])
  expect_s3_class(experimental_random_StatefulPRNG(random_key(0), new_ref(0L)), "jax_stateful_prng")
  expect_error(stateful_rng(1)$nothing, class = "jax_attribute_error")
  # A generator holds a reference, so a compiled function cannot return it.
  expect_error(jit(function() stateful_rng(0))(),
               "returned a mutable array reference of type Ref\\{int32\\[\\]\\}, but mutable array references cannot be returned",
               class = "jax_value_error")

  # Without a seed the generator is seeded from R's generator.
  set.seed(11)
  a <- as.array(stateful_rng()$uniform())
  set.seed(11)
  expect_identical(as.array(stateful_rng()$uniform()), a)
  expect_match(format(stateful_rng(42)), "^StatefulPRNG\\(base_key=Array\\(\\(\\), dtype=key<fry>\\), counter=Ref\\(")
})

test_that("vmap maps Ref arguments through their values", {
  ref <- srng_ref()
  fr <- function(r, x) {
    r$set(r$get() + x)
    jnp_sum(r$get())
  }
  r <- new_ref(jnp_zeros(c(2L, 3L)))
  srng_expect(ref, "ref_vmap_ax1", vmap(fr, in_axes = list(1L, 0L))(r, jnp_arange(3, dtype = "float32")))
  srng_expect(ref, "ref_vmap_ax1_ref", r$get())
  expect_identical(r$shape, c(2L, 3L))

  r <- new_ref(jnp_zeros(3L))
  gr <- function(r, x) {
    r$set(5)
    x
  }
  srng_expect(ref, "ref_vmap_unbatched", vmap(gr)(r, jnp_arange(3, dtype = "float32")))
  srng_expect(ref, "ref_vmap_unbatched_ref", r$get())

  rr <- new_ref(jnp_ones(c(3L, 2L)))
  h <- jit(function(x) vmap(fr)(rr, x))
  srng_expect(ref, "ref_jit_vmap", h(jnp_arange(3, dtype = "float32")))
  srng_expect(ref, "ref_jit_vmap_ref", rr$get())

  e <- expect_error(vmap(function(a, b) a$get() + b$get())(rr, rr), class = "jax_value_error")
  expect_true(startsWith(conditionMessage(e), ref$errors$ref_twice[[2L]]))

  # The shape seen inside is the shape of one element, and an error restores the ref.
  r <- new_ref(jnp_arange(3, dtype = "float32"))
  expect_identical(as.numeric(as.array(vmap(function(r, x) x + length(r$shape))(r, jnp_zeros(3L)))),
                   c(0, 0, 0))
  expect_error(vmap(function(r, x) stop("boom"))(r, jnp_zeros(3L)), "boom")
  expect_identical(r$shape, 3L)
  expect_identical(as.numeric(as.array(r$get())), c(0, 1, 2))
  # An unmapped ref argument is read like a closed over value.
  r0 <- new_ref(2)
  expect_identical(as.numeric(as.array(vmap(function(r, x) x + r$get(), in_axes = list(NULL, 0L))(r0, jnp_arange(3, dtype = "float32")))),
                   c(2, 3, 4))
})

test_that("refs closed over by control flow bodies are state of the operation", {
  ref <- srng_ref()
  f32 <- function(x) as_jax(x, "float32")
  r <- new_ref(f32(0))
  wbody <- function(i) {
    r$set(r$get() + i)
    i + 1
  }
  srng_expect(ref, "ref_while", lax_while_loop(function(i) i < 4, wbody, f32(0)))
  srng_expect(ref, "ref_while_ref", r$get())

  r <- new_ref(f32(1))
  tb <- function(x) {
    r$set(r$get() + x)
    x * 2
  }
  fb <- function(x) x * 3
  srng_expect(ref, "ref_cond_t", lax_cond(TRUE, tb, fb, f32(5)))
  srng_expect(ref, "ref_cond_t_ref", r$get())
  srng_expect(ref, "ref_cond_f", lax_cond(FALSE, tb, fb, f32(5)))
  srng_expect(ref, "ref_cond_f_ref", r$get())
  # A traced predicate takes the same path.
  r <- new_ref(f32(1))
  expect_identical(as.numeric(as.array(jit(function(p) lax_cond(p, tb, fb, f32(5)))(TRUE))), 10)
  expect_identical(as.numeric(as.array(r$get())), 6)

  r <- new_ref(f32(0))
  fbody <- function(i, v) {
    r$set(r$get() + v)
    v * 2
  }
  srng_expect(ref, "ref_fori", lax_fori_loop(0L, 4L, fbody, f32(1)))
  srng_expect(ref, "ref_fori_ref", r$get())

  acc <- function(x) {
    a <- new_ref(f32(0))
    body <- function(c, xi) {
      a$set(a$get() + xi * x)
      list(c + a$get(), a$get())
    }
    out <- lax_scan(body, f32(0), jnp_arange(4, dtype = "float32"))
    out[[1L]] + a$get()^2
  }
  srng_expect(ref, "ref_scan_local", acc(f32(2)))
  srng_expect(ref, "ref_scan_local_grad", grad(acc)(f32(2)))
  srng_expect(ref, "ref_scan_local_jit", jit(acc)(f32(2)))

  r <- new_ref(jnp_zeros(3L))
  sb <- function(c, x) {
    r$set(r$get() + x)
    list(c, jnp_sum(r$get()))
  }
  srng_expect(ref, "ref_scan_jit", jit(function(xs) lax_scan(sb, 0, xs)[[2L]])(jnp_arange(4, dtype = "float32")))
  srng_expect(ref, "ref_scan_jit_ref", r$get())

  r2 <- new_ref(f32(0))
  nested <- function(c, x) {
    inner <- function(c2, y) {
      r2$set(r2$get() + y)
      list(c2, y)
    }
    lax_scan(inner, 0, jnp_arange(3, dtype = "float32"))
    list(c, r2$get())
  }
  srng_expect(ref, "ref_scan_nested", lax_scan(nested, 0, jnp_arange(2, dtype = "float32"))[[2L]])
  srng_expect(ref, "ref_scan_nested_ref", r2$get())

  rg <- stateful_rng(11)
  out <- lax_while_loop(function(s) s[[1L]] < 3L,
                        function(s) list(s[[1L]] + 1L, s[[2L]] + rg$uniform()), list(0L, f32(0)))
  srng_expect(ref, "rng_while", out[[2L]])
  srng_expect(ref, "rng_while_counter", rg$counter$get())
  srng_expect(ref, "rng_cond", lax_cond(TRUE, function() rg$normal(), function() f32(0)))
  srng_expect(ref, "rng_cond_counter", rg$counter$get())
  srng_expect(ref, "rng_vmap_scan",
              vmap(function(r_) lax_scan(function(c, x) list(c, r_$uniform()), 0,
                                         jnp_arange(2, dtype = "float32"))[[2L]])(stateful_rng(12)$split(3L)))
})
