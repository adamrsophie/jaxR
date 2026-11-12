# Pytree utilities: flatten, unflatten, map, structure, custom nodes, ravel_pytree.

test_that("tree_flatten treats lists as containers, NULL as empty and the rest as leaves", {
  tree <- list(a = 1, b = list(2, list(c = 3)), d = NULL, e = "x")
  fl <- tree_flatten(tree)
  expect_identical(fl$leaves, list(1, 2, 3, "x"))
  expect_identical(treedef_num_leaves(fl$treedef), 4L)
  expect_identical(format(fl$treedef), "list(a = *, b = list(*, list(c = *)), d = NULL, e = *)")
  expect_output(print(fl$treedef), "^PyTreeDef\\(list\\(a = \\*")
  expect_identical(tree_unflatten(fl$treedef, fl$leaves), tree)
  expect_identical(tree_leaves(list(1, NULL, list())), list(1))
  expect_identical(tree_flatten(5)$leaves, list(5))
  expect_identical(format(tree_structure(5)), "*")
  expect_identical(format(tree_structure(NULL)), "NULL")
  expect_identical(tree_leaves(data.frame(a = 1)), list(data.frame(a = 1)))
})

test_that("tree_unflatten checks the number of leaves", {
  td <- tree_structure(list(1, 2))
  expect_error(tree_unflatten(td, list(1)), "expected 2 leaves, got 1")
})

test_that("tree_flatten honors is_leaf", {
  tree <- list(a = list(1, 2), b = list(3))
  fl <- tree_flatten(tree, is_leaf = function(x) is.list(x) && is.null(names(x)) && length(x) == 2L)
  expect_identical(fl$leaves, list(list(1, 2), 3))
  expect_identical(tree_unflatten(fl$treedef, list("A", "B")), list(a = "A", b = list("B")))
})

test_that("tree_map maps over one or several trees with the same structure", {
  expect_identical(tree_map(function(x) x * 10, list(a = 1, b = list(2, 3))),
                   list(a = 10, b = list(20, 30)))
  expect_identical(tree_map(`+`, list(1, list(2)), list(10, list(20))), list(11, list(22)))
  expect_error(tree_map(`+`, list(1, 2), list(1)), "pytree structure mismatch")
  expect_identical(tree_map(function(x, y) y, list(a = 1, b = 2), list(a = list(9), b = 8)),
                   list(a = list(9), b = 8))
  expect_identical(tree_map(length, list(1:3, list(1:2)), is_leaf = is.integer), list(3L, list(2L)))
})

test_that("tree_all, tree_reduce and tree_transpose", {
  skip_if_no_backend()
  expect_true(tree_all(list(TRUE, list(as_jax(TRUE)))))
  expect_false(tree_all(list(TRUE, FALSE)))
  expect_identical(tree_reduce(`+`, list(1, list(2, 3))), 6)
  expect_identical(tree_reduce(`+`, list(1, list(2, 3)), 10), 16)
  outer <- tree_structure(list(1, 2))
  inner <- tree_structure(list(x = 1, y = 2))
  tt <- tree_transpose(outer, inner, list(list(x = 1, y = 2), list(x = 3, y = 4)))
  expect_identical(tt, list(x = list(1, 3), y = list(2, 4)))
})

test_that("registered pytree nodes flatten, unflatten and pass through transformations", {
  skip_if_no_backend()
  register_pytree_node("fx_point",
    flatten = function(p) list(children = list(p$x, p$y), aux = p$label),
    unflatten = function(aux, children) structure(list(x = children[[1L]], y = children[[2L]], label = aux),
                                                  class = "fx_point"))
  p <- structure(list(x = as_jax(2), y = as_jax(c(1, 3)), label = "pt"), class = "fx_point")
  fl <- tree_flatten(p)
  expect_length(fl$leaves, 2L)
  expect_identical(format(fl$treedef), "fx_point(*, *)")
  q <- tree_unflatten(fl$treedef, list(1, 2))
  expect_s3_class(q, "fx_point")
  expect_identical(q$label, "pt")
  f <- function(p) p$x * jnp_sum(p$y^2)
  g <- grad(f)(p)
  expect_s3_class(g, "fx_point")
  expect_jax_equal(g$x, 10)
  expect_jax_equal(g$y, c(4, 12))
  expect_jax_equal(jit(f)(p), 20)
  m <- tree_map(function(l) l * 2, p)
  expect_jax_equal(m$y, c(2, 6))
})

test_that("pytree structure and names survive jit, grad and vmap", {
  skip_if_no_backend()
  p <- list(w = as_jax(c(1, 2)), extra = list(b = as_jax(3)))
  f <- function(p) jnp_sum(p$w) * p$extra$b
  g <- grad(f)(p)
  expect_identical(format(tree_structure(g)), format(tree_structure(p)))
  expect_jax_equal(g$w, c(3, 3))
  expect_jax_equal(g$extra$b, 3)
  out <- jit(function(p) list(z = p$w * 2, NULL))(p)
  expect_identical(names(out), c("z", ""))
  expect_null(out[[2L]])
  v <- vmap(function(p) p$w * p$extra$b, in_axes = list(list(w = 0L, extra = NULL)))(p)
  expect_jax_equal(v, c(3, 6))
})

test_that("ravel_pytree round trips shapes and dtypes", {
  skip_if_no_backend()
  tree <- list(a = as_jax(matrix(1:6, 2), "float32"), b = list(as_jax(c(7, 8))), c = as_jax(9L))
  r <- ravel_pytree(tree)
  expect_jax_equal(r$flat, c(1, 3, 5, 2, 4, 6, 7, 8, 9), dtype = "float32")
  back <- r$unravel(r$flat)
  expect_identical(format(tree_structure(back)), format(tree_structure(tree)))
  expect_identical(dtype(back$c), "int32")
  expect_identical(dim(back$a), c(2L, 3L))
  expect_jax_equal(back$a, matrix(1:6, 2))
  e <- ravel_pytree(list())
  expect_identical(dim(e$flat), 0L)
  expect_identical(e$unravel(e$flat), list())
  expect_jax_equal(grad(function(v) jnp_sum(r$unravel(v)$a^2))(r$flat), c(2, 6, 10, 4, 8, 12, 0, 0, 0))
})
