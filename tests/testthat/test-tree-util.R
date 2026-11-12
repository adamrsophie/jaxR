# jax.tree_util / jax.tree: key paths, treedef helpers, broadcast, registration helpers,
# Partial. Expected values from jax 0.12.0.1 (lists stand for tuples, named lists for
# dicts, flattened in sorted key order).

test_that("key entries format as in JAX and keystr joins them", {
  expect_identical(format(SequenceKey(0)), "[0]")
  expect_identical(format(DictKey("a")), "['a']")
  expect_identical(format(DictKey(1)), "[1]")
  expect_identical(format(GetAttrKey("x")), ".x")
  expect_identical(format(FlattenedIndexKey(2)), "[<flat index 2>]")
  expect_output(print(DictKey("a")), "DictKey(key='a')", fixed = TRUE)
  expect_output(print(SequenceKey(0)), "SequenceKey(idx=0)", fixed = TRUE)
  keys <- list(DictKey("a"), SequenceKey(0), GetAttrKey("w"), FlattenedIndexKey(1))
  expect_identical(keystr(keys), "['a'][0].w[<flat index 1>]")
  expect_identical(keystr(keys, simple = TRUE, separator = "/"), "a/0/w/1")
  expect_identical(keystr(list("x", 1)), "x1")
})

test_that("tree_leaves_with_path and tree_flatten_with_path follow the flatten order", {
  tree <- list(b = list(1, list(2, NULL)), a = 3)
  kl <- tree_leaves_with_path(tree)
  expect_length(kl, 3L)
  expect_identical(vapply(kl, function(p) keystr(p[[1L]]), ""), c("['a']", "['b'][0]", "['b'][1][0]"))
  expect_identical(lapply(kl, `[[`, 2L), list(3, 1, 2))
  fl <- tree_flatten_with_path(tree)
  expect_true(jaxR:::treedef_equal(fl$treedef, tree_structure(tree)))
  expect_identical(tree_unflatten(fl$treedef, list("x", "y", "z")),
                   list(b = list("y", list("z", NULL)), a = "x"))
  # unnamed lists give SequenceKey, mixed names give both kinds
  expect_identical(keystr(tree_leaves_with_path(list(1, list(z = 2)))[[2L]][[1L]]), "[1]['z']")
  expect_identical(tree_leaves_with_path(5), list(list(list(), 5)))
  expect_identical(tree_leaves_with_path(NULL), list())
})

test_that("is_leaf may take the path", {
  tree <- list(a = list(1, 2), b = list(3))
  fl <- tree_flatten_with_path(tree, is_leaf = function(p, x) length(p) == 1L && keystr(p) == "['a']",
                               is_leaf_takes_path = TRUE)
  expect_identical(lapply(fl$leaves, `[[`, 2L), list(list(1, 2), 3))
  expect_true(jaxR:::treedef_equal(
    fl$treedef, tree_structure(tree, is_leaf = function(x) identical(x, list(1, 2)))))
  fl2 <- tree_flatten_with_path(tree, is_leaf = function(x) identical(x, list(3)))
  expect_identical(keystr(fl2$leaves[[3L]][[1L]]), "['b']")
})

test_that("tree_map_with_path maps with paths over one or several trees", {
  out <- tree_map_with_path(function(p, x) paste0(keystr(p), x), list(a = 1, b = list(2, 3)))
  expect_identical(out, list(a = "['a']1", b = list("['b'][0]2", "['b'][1]3")))
  out2 <- tree_map_with_path(function(p, x, y) x + y, list(1, list(2)), list(10, list(20)))
  expect_identical(out2, list(11, list(22)))
  expect_error(tree_map_with_path(function(p, x, y) x, list(1, 2), list(1)), "structure mismatch")
})

test_that("treedef_children, treedef_is_leaf and treedef_tuple", {
  td <- tree_structure(list(list(1, 2), 3, list(a = 4)))
  ch <- treedef_children(td)
  expect_length(ch, 3L)
  expect_identical(vapply(ch, format, ""), c("list(*, *)", "*", "list(a = *)"))
  expect_true(treedef_is_leaf(tree_structure(1)))
  expect_true(treedef_is_leaf(tree_structure(NULL)))
  expect_true(treedef_is_leaf(tree_structure(list())))
  expect_false(treedef_is_leaf(tree_structure(list(1))))
  xy <- treedef_tuple(list(tree_structure(list(1, 2, 3)), tree_structure(list(a = 4, b = 5))))
  expect_true(jaxR:::treedef_equal(xy, tree_structure(list(list(1, 2, 3), list(a = 4, b = 5)))))
  expect_identical(treedef_num_leaves(xy), 5L)
  expect_identical(treedef_children(tree_structure(1)), list())
})

test_that("all_leaves, is_tree_node and flatten_one_level", {
  expect_true(all_leaves(tree_leaves(list(a = list(1, 2, 3)))))
  expect_false(all_leaves(list(list(a = list(1, 2, 3)))))
  expect_false(all_leaves(list(1, NULL)))
  expect_true(all_leaves(list(1, 2)))
  # jax: all_leaves([1, 2], is_leaf=is_list) is False (the outer list is a leaf)
  expect_false(all_leaves(list(1, 2), is_leaf = is.list))
  expect_false(all_leaves(list(list(1), 2), is_leaf = is.list))
  expect_true(is_tree_node("list"))
  expect_true(is_tree_node(NULL))
  expect_false(is_tree_node("data.frame"))
  expect_false(is_tree_node(1))
  expect_identical(flatten_one_level(list(a = list(1, 2), b = list(c = 3))),
                   list(children = list(list(1, 2), list(c = 3)), aux = c("a", "b")))
  expect_identical(flatten_one_level(list(1, list(2))), list(children = list(1, list(2)), aux = NULL))
  expect_identical(flatten_one_level(NULL), list(children = list(), aux = NULL))
  expect_error(flatten_one_level(1), "can't tree-flatten type: numeric")
  fk <- flatten_one_level_with_keys(list(a = 1, 2))
  expect_identical(lapply(fk$children, function(kc) format(kc[[1L]])), list("['a']", "[1]"))
  expect_identical(lapply(fk$children, `[[`, 2L), list(1, 2))
})

test_that("tree_broadcast and tree_reduce_associative", {
  expect_identical(tree_broadcast(1, list(2, list(3, 4))), list(1, list(1, 1)))
  expect_identical(tree_broadcast(list(1, 2), list(list(3, 4), list(a = 5, b = 6))),
                   list(list(1, 1), list(a = 2, b = 2)))
  expect_error(tree_broadcast(list(1, 2), list(2)), "structure mismatch")
  expect_identical(tree_reduce_associative(`+`, list(1, 2, list(3, 4))), 10)
  expect_identical(tree_reduce_associative(paste0, list("a", "b", "c", "d", "e")), "abcde")
  expect_identical(tree_reduce_associative(`+`, list(), identity = 0), 0)
  expect_error(tree_reduce_associative(`+`, list()),
               "Must specify identity for parallel reduction of empty sequence.", fixed = TRUE)
})

test_that("register_pytree_with_keys reports custom keys and round trips", {
  register_pytree_with_keys("tu_box",
    function(b) list(children = list(list(GetAttrKey("x"), b$x), list(GetAttrKey("y"), b$y)),
                     aux = b$tag),
    function(aux, children) structure(list(x = children[[1L]], y = children[[2L]], tag = aux),
                                      class = "tu_box"))
  b <- structure(list(x = 1, y = list(2, 3), tag = "t"), class = "tu_box")
  kl <- tree_leaves_with_path(list(b))
  expect_identical(vapply(kl, function(p) keystr(p[[1L]]), ""), c("[0].x", "[0].y[0]", "[0].y[1]"))
  expect_identical(tree_map(function(v) v * 2, b),
                   structure(list(x = 2, y = list(4, 6), tag = "t"), class = "tu_box"))
  # nodes registered without keys get FlattenedIndexKey paths
  register_pytree_node("tu_pair", function(p) list(children = unclass(p), aux = NULL),
                       function(aux, ch) structure(ch, class = "tu_pair"))
  kl2 <- tree_leaves_with_path(structure(list(5, 6), class = "tu_pair"))
  expect_identical(vapply(kl2, function(p) keystr(p[[1L]]), ""), c("[<flat index 0>]", "[<flat index 1>]"))
  expect_true(is_tree_node("tu_box"))
})

tree_flatten.tu_cls <- function(x) list(children = list(x$a), aux = x$k)
tree_unflatten.tu_cls <- function(aux, children) structure(list(a = children[[1L]], k = aux), class = "tu_cls")
tree_flatten_with_keys.tu_kcls <- function(x) list(children = list(list(GetAttrKey("a"), x$a)), aux = NULL)
tree_unflatten.tu_kcls <- function(aux, children) structure(list(a = children[[1L]]), class = "tu_kcls")

test_that("register_pytree_node_class and register_pytree_with_keys_class find methods by name", {
  register_pytree_node_class("tu_cls")
  obj <- structure(list(a = 3, k = "meta"), class = "tu_cls")
  expect_identical(tree_leaves(obj), list(3))
  expect_identical(tree_map(function(v) v + 1, obj), structure(list(a = 4, k = "meta"), class = "tu_cls"))
  register_pytree_with_keys_class("tu_kcls")
  kl <- tree_leaves_with_path(structure(list(a = 7), class = "tu_kcls"))
  expect_identical(keystr(kl[[1L]][[1L]]), ".a")
  expect_error(register_pytree_node_class("tu_none"), "has no method tree_flatten.tu_none")
})

test_that("register_dataclass registers S3 lists with data and meta fields", {
  register_dataclass("tu_struct", data_fields = c("x", "y"), meta_fields = "op")
  m <- structure(list(x = c(1, 1, 1), op = "add", y = 0:2), class = "tu_struct")
  fl <- tree_flatten(m)
  expect_identical(fl$leaves, list(c(1, 1, 1), 0:2))
  expect_identical(tree_unflatten(fl$treedef, fl$leaves), m)
  kl <- tree_leaves_with_path(m)
  expect_identical(vapply(kl, function(p) keystr(p[[1L]]), ""), c(".x", ".y"))
  expect_error(register_dataclass("tu_bad", data_fields = "x"),
               "data_fields and meta_fields must both be specified")
  expect_error(register_dataclass("tu_bad"), "required when nodetype is not a dataclass")
  expect_error(register_dataclass("tu_bad", "x", c("x", "z")), "must not overlap. Overlapping fields: 'x'")
})

test_that("register_dataclass nodes pass through transformations", {
  skip_if_no_backend()
  register_dataclass("tu_struct2", data_fields = c("x", "y"), meta_fields = "op")
  m <- structure(list(x = as_jax(c(1, 2, 3)), y = as_jax(c(0, 1, 2)), op = "add"), class = "tu_struct2")
  f <- function(m) if (m$op == "add") m$x + m$y else m$x - m$y
  expect_jax_equal(jit(f)(m), c(1, 3, 5))
  g <- grad(function(m) jnp_sum(m$x * m$y))(m)
  expect_s3_class(g, "tu_struct2")
  expect_jax_equal(g$x, c(0, 1, 2))
  expect_jax_equal(g$y, c(1, 2, 3))
  expect_identical(g$op, "add")
  out <- vmap(function(m) jnp_sum(m$x * m$y))(
    structure(list(x = jnp_ones(c(2L, 3L)), y = jnp_ones(c(2L, 3L)), op = "add"), class = "tu_struct2"))
  expect_jax_equal(out, c(3, 3))
})

test_that("register_static makes leafless nodes", {
  register_static("tu_static")
  s <- structure(list(mode = "sub"), class = "tu_static")
  expect_identical(tree_leaves(list(1, s)), list(1))
  expect_identical(tree_unflatten(tree_structure(list(1, s)), list(2)), list(2, s))
  expect_identical(tree_leaves_with_path(s), list())
})

test_that("Partial binds arguments and is a pytree", {
  skip_if_no_backend()
  add <- function(a, b) a + b
  p <- Partial(add, as_jax(1))
  expect_jax_equal(p(as_jax(2)), 3)
  expect_length(tree_leaves(p), 1L)
  expect_output(print(p), "Partial(<function>, 1 bound argument(s))", fixed = TRUE)
  call_func <- jit(function(f, x) f(x))
  expect_jax_equal(call_func(p, as_jax(2)), 3)
  expect_jax_equal(jit(function(f, x, y) f(x, y))(Partial(add), as_jax(1), as_jax(2)), 3)
  q <- Partial(function(x, scale) x * scale, scale = as_jax(10))
  expect_jax_equal(q(as_jax(2)), 20)
  expect_jax_equal(grad(function(s) jnp_sum(q(as_jax(c(1, 2)))) * s)(as_jax(1)), 30)
  pm <- tree_map(function(v) v * 2, q)
  expect_jax_equal(pm(as_jax(1)), 20)
  expect_error(Partial(1), "func must be a function")
})

test_that("jit keys its cache on the static data of custom nodes", {
  skip_if_no_backend()
  skip_if_not(exists("treedef_key", envir = asNamespace("jaxR")),
              "needs treedef_key() (R/tree.R) in the jit cache key (R/dispatch.R)")
  call_func <- jit(function(f, x) f(x))
  expect_jax_equal(call_func(Partial(function(a, b) a + b, as_jax(1)), as_jax(2)), 3)
  expect_jax_equal(call_func(Partial(function(a, b) a * b, as_jax(1)), as_jax(2)), 2)
  register_dataclass("tu_struct3", data_fields = "x", meta_fields = "op")
  f <- jit(function(m) if (m$op == "add") m$x + 1 else m$x - 1)
  expect_jax_equal(f(structure(list(x = as_jax(1), op = "add"), class = "tu_struct3")), 2)
  expect_jax_equal(f(structure(list(x = as_jax(1), op = "sub"), class = "tu_struct3")), 0)
})
