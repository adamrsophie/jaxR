# Ports of the CPU parts of tests/layout_test.py.

L <- experimental_layout_Layout

test_that("Layout and Format objects", {
  expect_output(print(L(c(1L, 0L))),
                "Layout(major_to_minor=(1, 0), tiling=None, sub_byte_element_size_in_bits=0)", fixed = TRUE)
  expect_true(L(c(1L, 0L)) == L(c(1L, 0L)))
  expect_false(L(c(1L, 0L)) == L(c(0L, 1L)))
  expect_equal(experimental_layout_Layout_update(L(c(1L, 0L)), major_to_minor = c(0L, 1L))$major_to_minor,
               c(0L, 1L))
  skip_if_no_backend()
  la <- experimental_layout_Layout_for_array(jnp_ones(c(2L, 3L, 4L)))
  expect_equal(la$major_to_minor, 0:2)
  expect_equal(la$tiling, list())
  expect_s3_class(experimental_layout_Format(L(c(1L, 0L)), devices()[1]), "jax_Format")
  expect_s3_class(experimental_layout_Format(experimental_layout_Layout_AUTO), "jax_Format")
  expect_error(experimental_layout_Format(L(c(1L, 0L))), "Sharding has to be concrete",
               class = "jax_value_error")
  expect_error(experimental_layout_Format("x"), class = "jax_type_error")
})

test_that("with_layout_constraint keeps values under jit, grad and vmap", {
  skip_if_no_backend()
  x <- jnp_reshape(jnp_arange(6, dtype = "float32"), c(2L, 3L))
  wl <- experimental_layout_with_layout_constraint
  f <- jit(function(x) wl(x * 2, L(c(1L, 0L))) + 1)
  expect_equal(as.array(f(x)), matrix(c(1, 7, 3, 9, 5, 11), 2))
  expect_match(jit(function(x) wl(x, L(c(1L, 0L))))$lower(x)$as_text(),
               'call_target_name = "LayoutConstraint"', fixed = TRUE)
  expect_equal(as.array(wl(x, L(c(1L, 0L)))), as.array(x))
  expect_equal(as.array(grad(function(x) jnp_sum(wl(x, L(c(1L, 0L)))^2))(x)), 2 * as.array(x))
  o <- jit(vmap(function(r) wl(jnp_outer(r, r), L(c(1L, 0L)))))(x)
  expect_equal(as.array(o)[2, , ], outer(3:5, 3:5))
  # A tree of layouts.
  r <- wl(list(a = x, b = jnp_ones(3L)), list(a = L(c(0L, 1L)), b = L(0L)))
  expect_equal(as.numeric(r$b), rep(1, 3))
  expect_error(wl(x, L(0L)), "Length of major_to_minor and the rank", class = "jax_value_error")
  expect_error(wl(x, "x"), "must be of type `Layout`", class = "jax_value_error")
  expect_error(wl(x, L(c(1L, 0L), tiling = list(c(8L, 128L)))), class = "jax_not_implemented_error")
})

test_that("explicit_layout checks the layouts and runs the function", {
  skip_if_no_backend()
  x <- jnp_ones(c(2L, 3L))
  g <- experimental_layout_explicit_layout(function(a) a + 1, in_layouts = L(c(0L, 1L)))
  expect_equal(as.array(g(x)), matrix(2, 2, 3))
  h <- experimental_layout_explicit_layout(function(a) a * 2)
  expect_equal(as.array(h(x, in_layouts = L(c(1L, 0L)))), matrix(2, 2, 3))
  expect_error(h(x), "Missing required keyword argument: 'in_layouts'", class = "jax_type_error")
  expect_error(experimental_layout_explicit_layout(function(a) a, in_layouts = L(0L))(x),
               class = "jax_value_error")
})

test_that("layouts follow mapped axes and broadcast over subtrees (JAX values)", {
  skip_if_no_backend()
  wl <- experimental_layout_with_layout_constraint
  expect_equal(as.array(jit(vmap(function(x) wl(x, L(0L)), in_axes = 1L, out_axes = 1L))(jnp_ones(c(2L, 3L)))),
               matrix(1, 2, 3))
  x <- jnp_reshape(jnp_arange(24, dtype = "float32"), c(2L, 3L, 4L))
  f <- jit(vmap(function(z) wl(z, L(c(1L, 0L))) * 2, in_axes = 1L, out_axes = 1L))
  expect_equal(as.numeric(jnp_sum(f(x))), 552)
  x4 <- list(a = list(jnp_ones(c(2L, 3L)), jnp_ones(c(2L, 3L))), b = jnp_ones(4L))
  y <- jit(function(x) wl(x, list(a = L(c(1L, 0L)), b = L(0L))))(x4)
  expect_equal(as.numeric(jnp_sum(y$a[[2]])), 6)
})
