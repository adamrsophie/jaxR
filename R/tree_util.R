# Pytree utilities beyond R/tree.R (port of jax/_src/tree_util.py and the jax.tree
# aliases): key paths, treedef helpers, broadcasting, Partial, and the registration
# helpers (with keys, dataclass-like S3 lists, static classes).
#
# Key paths mirror JAX: elements of unnamed lists get SequenceKey(i) (0-based), named
# list elements DictKey(name), children of nodes registered without keys
# FlattenedIndexKey(i), and fields of register_dataclass() nodes GetAttrKey(name).

.jx$tu_keyed <- new.env(parent = emptyenv())

# ---- key entries ---------------------------------------------------------------------------

tu_key <- function(cls, ...) structure(list(...), class = c(cls, "jax_key_entry"))

#' Key path entries
#'
#' Ports of `jax.tree_util.SequenceKey`, `DictKey`, `GetAttrKey` and
#' `FlattenedIndexKey`. A key path (as returned by [tree_flatten_with_path()]) is a list
#' of these entries; [keystr()] formats it. `format()` gives JAX's `str()` of an entry
#' (`"[0]"`, `"['a']"`, `".x"`, `"[<flat index 0>]"`).
#' @param idx 0-based position in an unnamed list.
#' @param key Name (or value) of a named list element; for `FlattenedIndexKey`, the
#'   0-based child position.
#' @param name Field name.
#' @return A key entry.
#' @export
SequenceKey <- function(idx) tu_key("jax_SequenceKey", idx = as.integer(idx))

#' @rdname SequenceKey
#' @export
DictKey <- function(key) tu_key("jax_DictKey", key = key)

#' @rdname SequenceKey
#' @export
GetAttrKey <- function(name) tu_key("jax_GetAttrKey", name = as.character(name))

#' @rdname SequenceKey
#' @export
FlattenedIndexKey <- function(key) tu_key("jax_FlattenedIndexKey", key = as.integer(key))

tu_py_repr <- function(v) {
  if (is.character(v)) paste0("'", v, "'") else format(v)
}

#' @export
format.jax_key_entry <- function(x, ...) {
  switch(class(x)[1L],
    jax_SequenceKey = paste0("[", x$idx, "]"),
    jax_DictKey = paste0("[", tu_py_repr(x$key), "]"),
    jax_GetAttrKey = paste0(".", x$name),
    jax_FlattenedIndexKey = paste0("[<flat index ", x$key, ">]"))
}

#' @export
print.jax_key_entry <- function(x, ...) {
  field <- names(unclass(x))[1L]
  cat(sub("^jax_", "", class(x)[1L]), "(", field, "=", tu_py_repr(x[[1L]]), ")\n", sep = "")
  invisible(x)
}

tu_simple_entry <- function(k) {
  if (!inherits(k, "jax_key_entry")) return(as.character(k))
  as.character(k[[1L]])
}

#' Format a key path
#'
#' Port of `jax.tree_util.keystr`.
#' @param keys A key path (list of key entries, or values convertible to strings).
#' @param simple Use the compact representation (`"a/0"` style with `separator`).
#' @param separator String placed between entries.
#' @return A character scalar.
#' @examples
#' params <- list(foo = list(bar = list(baz = 1, bat = list(2, 3))))
#' for (kl in tree_leaves_with_path(params)) print(keystr(kl[[1]]))
#' @export
keystr <- function(keys, simple = FALSE, separator = "") {
  f <- if (simple) tu_simple_entry else function(k) {
    if (inherits(k, "jax_key_entry")) format(k) else as.character(k)
  }
  paste(vapply(keys, f, ""), collapse = separator)
}

# ---- flattening with keys --------------------------------------------------------------------

tu_null_def <- function() null_def

# Class under which a classed object is registered (as pytree_handler() resolves it).
tu_registered_class <- function(x) {
  for (c in oldClass(x)) if (!is.null(.pytree_registry[[c]])) return(c)
  NULL
}

# Keys of the children of a node, in visiting order (see dict_order() in R/tree.R).
tu_child_keys <- function(x, children) {
  if (is.list(x) && is.null(oldClass(x))) {
    nms <- names(x) %||% rep("", length(x))
    ord <- dict_order(names(x))
    if (!is.null(ord)) return(lapply(nms[ord], DictKey))
    return(lapply(seq_along(x), function(i) {
      if (!is.na(nms[i]) && nzchar(nms[i])) DictKey(nms[i]) else SequenceKey(i - 1L)
    }))
  }
  fk <- .jx$tu_keyed[[tu_registered_class(x)]]
  if (!is.null(fk)) return(lapply(fk(x)$children, `[[`, 1L))
  lapply(seq_along(children), function(i) FlattenedIndexKey(i - 1L))
}

# Same traversal and treedef as tree_flatten_impl() (R/tree.R), also recording paths.
tu_flatten_path <- function(x, path, is_leaf, acc) {
  if (!is.null(is_leaf) && isTRUE(is_leaf(path, x))) {
    acc$leaves[[length(acc$leaves) + 1L]] <- list(path, x)
    return(leaf_def)
  }
  if (is.null(x)) return(null_def)
  plain <- is.list(x) && is.null(oldClass(x))
  h <- if (plain) NULL else pytree_handler(x)
  if (plain || !is.null(h)) {
    ord <- if (plain) dict_order(names(x)) else NULL
    children <- if (!plain) h$flatten(x) else list(children = if (is.null(ord)) unname(x) else unname(x[ord]))
    keys <- tu_child_keys(x, children$children)
    kids <- lapply(seq_along(children$children), function(i) {
      tu_flatten_path(children$children[[i]], c(path, list(keys[[i]])), is_leaf, acc)
    })
    if (plain) return(list_def(names(x), ord, kids))
    n <- sum(vapply(kids, function(k) k$n, 1L))
    return(structure(list(type = "custom", class = oldClass(x)[1L], aux = children$aux,
                          children = kids, n = n), class = "jax_treedef"))
  }
  acc$leaves[[length(acc$leaves) + 1L]] <- list(path, x)
  leaf_def
}

#' Flatten a pytree with key paths
#'
#' Ports of `jax.tree.flatten_with_path`, `jax.tree.leaves_with_path` and
#' `jax.tree.map_with_path` (and their `jax.tree_util` aliases).
#' @param tree A pytree.
#' @param is_leaf Optional predicate marking subtrees to treat as leaves.
#' @param is_leaf_takes_path Whether `is_leaf` takes `(path, x)` instead of `x`.
#' @return `tree_flatten_with_path()`: a list with `leaves` (a list of
#'   `list(path, leaf)` pairs, paths being lists of key entries) and `treedef`.
#'   `tree_leaves_with_path()`: the pairs. `tree_map_with_path()`: the mapped tree.
#' @examples
#' tree_leaves_with_path(list(a = 1, b = list(2, 3)))
#' tree_map_with_path(function(p, x) paste(keystr(p), x), list(a = 1, b = list(2, 3)))
#' @export
tree_flatten_with_path <- function(tree, is_leaf = NULL, is_leaf_takes_path = FALSE) {
  il <- is_leaf
  if (!is.null(is_leaf) && !is_leaf_takes_path) il <- function(p, x) is_leaf(x)
  acc <- new.env(parent = emptyenv())
  acc$leaves <- list()
  td <- tu_flatten_path(tree, list(), il, acc)
  list(leaves = acc$leaves, treedef = td)
}

#' @rdname tree_flatten_with_path
#' @export
tree_leaves_with_path <- function(tree, is_leaf = NULL, is_leaf_takes_path = FALSE) {
  tree_flatten_with_path(tree, is_leaf, is_leaf_takes_path)$leaves
}

#' @rdname tree_flatten_with_path
#' @param f Function of `(path, leaf, ...)` (one leaf per tree).
#' @param ... Further pytrees with the structure of `tree` (`tree_map_with_path()`).
#' @export
tree_map_with_path <- function(f, tree, ..., is_leaf = NULL, is_leaf_takes_path = FALSE) {
  fl <- tree_flatten_with_path(tree, is_leaf, is_leaf_takes_path)
  rest <- lapply(list(...), function(r) flatten_up_to(fl$treedef, r))
  leaves <- lapply(seq_along(fl$leaves), function(i) {
    kl <- fl$leaves[[i]]
    do.call(f, c(list(kl[[1L]], kl[[2L]]), lapply(rest, `[[`, i)))
  })
  tree_unflatten(fl$treedef, leaves)
}

# ---- treedef helpers -------------------------------------------------------------------------

#' Treedef helpers
#'
#' Ports of `jax.tree_util.treedef_children`, `treedef_is_leaf` and `treedef_tuple`.
#' `treedef_is_leaf()` is `TRUE` for a treedef with one node: a leaf, `NULL`, or a
#' container without children (as in JAX). `treedef_num_leaves()` is the port of
#' `PyTreeDef.num_leaves`.
#' @param treedef A treedef (from [tree_structure()]).
#' @param treedefs A list of treedefs.
#' @return `treedef_children()`: a list of treedefs; `treedef_is_leaf()`: a logical;
#'   `treedef_tuple()`: the treedef of an unnamed list of the given structures;
#'   `treedef_num_leaves()`: the number of leaves.
#' @export
treedef_children <- function(treedef) {
  if (treedef$type %in% c("list", "custom")) treedef$children else list()
}

#' @rdname treedef_children
#' @export
treedef_is_leaf <- function(treedef) {
  treedef$type %in% c("leaf", "null") || length(treedef$children) == 0L
}

#' @rdname treedef_children
#' @export
treedef_tuple <- function(treedefs) {
  treedefs <- as.list(treedefs)
  n <- sum(vapply(treedefs, function(t) t$n, 1L))
  structure(list(type = "list", names = NULL, children = treedefs, n = as.integer(n)),
            class = "jax_treedef")
}

#' Test whether all elements are leaves
#'
#' Port of `jax.tree_util.all_leaves`.
#' @param iterable A list.
#' @param is_leaf Optional leaf predicate.
#' @return A logical.
#' @export
all_leaves <- function(iterable, is_leaf = NULL) {
  items <- as.list(iterable)
  if (is.null(is_leaf)) return(all(vapply(items, function(x) !is_container(x), NA)))
  leaves <- tree_leaves(items, is_leaf)
  length(leaves) == length(items) && all(mapply(identical, items, leaves))
}

#' Test whether a class is a pytree node type
#'
#' Port of `jax.tree_util.is_tree_node`. R has no type objects: pass a class name
#' (`"list"` and `"NULL"` are the built-in containers) or an object (its class is
#' used).
#' @param typ A class name or an object.
#' @return A logical.
#' @export
is_tree_node <- function(typ) {
  if (!is.character(typ)) {
    if (is.null(typ) || (is.list(typ) && is.null(oldClass(typ)))) return(TRUE)
    return(!is.null(tu_registered_class(typ)))
  }
  typ %in% c("list", "NULL") || any(vapply(typ, function(c) !is.null(.pytree_registry[[c]]), NA))
}

#' Flatten one level of a pytree
#'
#' Ports of `jax.tree_util.flatten_one_level` and `flatten_one_level_with_keys`.
#' @param tree A container (list, `NULL` or a registered node).
#' @return A list with `children` (for the `_with_keys` variant, `list(key, child)`
#'   pairs) and `aux` (the names of a list, or the aux data of a registered node).
#' @export
flatten_one_level <- function(tree) {
  if (is.null(tree)) return(list(children = list(), aux = NULL))
  if (is.list(tree) && is.null(oldClass(tree))) {
    ord <- dict_order(names(tree))
    if (is.null(ord)) return(list(children = unname(tree), aux = names(tree)))
    return(list(children = unname(tree[ord]), aux = names(tree)[ord]))
  }
  h <- pytree_handler(tree)
  if (is.null(h))
    jax_error("can't tree-flatten type: ", class(tree)[1L], class = "jax_value_error")
  fl <- h$flatten(tree)
  list(children = fl$children, aux = fl$aux)
}

#' @rdname flatten_one_level
#' @export
flatten_one_level_with_keys <- function(tree) {
  fl <- flatten_one_level(tree)
  keys <- if (is.null(tree)) list() else tu_child_keys(tree, fl$children)
  list(children = Map(function(k, c) list(k, c), keys, fl$children), aux = fl$aux)
}

#' Broadcast a pytree prefix over a full pytree
#'
#' Port of `jax.tree.broadcast` (`jax.tree_util.tree_broadcast`): each leaf of
#' `prefix_tree` is repeated over the leaves of the corresponding subtree of
#' `full_tree`.
#' @param prefix_tree A pytree that is a prefix of `full_tree`.
#' @param full_tree A pytree.
#' @param is_leaf Optional leaf predicate for `prefix_tree`.
#' @return A pytree with the structure of `full_tree`.
#' @export
tree_broadcast <- function(prefix_tree, full_tree, is_leaf = NULL) {
  pfl <- tree_flatten(prefix_tree, is_leaf)
  subs <- flatten_up_to(pfl$treedef, full_tree)
  out <- unlist(Map(function(x, s) rep(list(x), tree_structure(s)$n), pfl$leaves, subs),
                recursive = FALSE)
  tree_unflatten(tree_structure(full_tree), out %||% list())
}

#' Reduce the leaves of a pytree with an associative operation
#'
#' Port of `jax.tree.reduce_associative`: a balanced (parallel) reduction.
#' @param operation Associative function of two values.
#' @param tree A pytree.
#' @param identity Value returned for a tree without leaves (required then).
#' @param is_leaf Optional leaf predicate.
#' @return The reduced value.
#' @export
tree_reduce_associative <- function(operation, tree, identity, is_leaf = NULL) {
  seq <- tree_leaves(tree, is_leaf)
  has_identity <- !missing(identity)
  red <- function(s) {
    n <- length(s)
    if (n == 0L) {
      if (!has_identity)
        jax_error("Must specify identity for parallel reduction of empty sequence.",
                  class = "jax_type_error")
      return(identity)
    }
    if (n == 1L) return(s[[1L]])
    k <- n %/% 2L
    operation(red(s[seq_len(k)]), red(s[(k + 1L):n]))
  }
  red(seq)
}

# ---- registration -----------------------------------------------------------------------------

#' Register a pytree node class with key paths
#'
#' Port of `jax.tree_util.register_pytree_with_keys`: like [register_pytree_node()],
#' with a flatten function that also names each child, so that
#' [tree_flatten_with_path()] reports these keys.
#' @param nodetype Class name.
#' @param flatten_with_keys Function of a node returning `list(children = <list of
#'   list(key, child) pairs>, aux = <any>)`.
#' @param unflatten_func Function of `(aux, children)` returning a node.
#' @param flatten_func Optional faster flatten function returning
#'   `list(children, aux)` with the same children and aux.
#' @return `nodetype`, invisibly.
#' @export
register_pytree_with_keys <- function(nodetype, flatten_with_keys, unflatten_func,
                                      flatten_func = NULL) {
  force(flatten_with_keys)
  if (is.null(flatten_func)) {
    flatten_func <- function(x) {
      kc <- flatten_with_keys(x)
      list(children = lapply(kc$children, `[[`, 2L), aux = kc$aux)
    }
  }
  register_pytree_node(nodetype, flatten_func, unflatten_func)
  assign(nodetype, flatten_with_keys, envir = .jx$tu_keyed)
  invisible(nodetype)
}

tu_class_method <- function(name, cls, envir, required = TRUE) {
  f <- get0(paste0(name, ".", cls), envir = envir, mode = "function")
  if (is.null(f) && required)
    jax_error("class '", cls, "' has no method ", name, ".", cls, class = "jax_type_error")
  f
}

#' Register a class through its flatten methods
#'
#' Ports of `jax.tree_util.register_pytree_node_class` and
#' `register_pytree_with_keys_class`. R classes have no methods of their own, so the
#' functions are found by name: `tree_flatten.<class>(x)` (returning
#' `list(children, aux)`), `tree_unflatten.<class>(aux, children)` and, for the keyed
#' variant, `tree_flatten_with_keys.<class>(x)`.
#' @param cls Class name.
#' @param envir Environment where the methods are looked up.
#' @return `cls`, invisibly.
#' @export
register_pytree_node_class <- function(cls, envir = parent.frame()) {
  register_pytree_node(cls, tu_class_method("tree_flatten", cls, envir),
                       tu_class_method("tree_unflatten", cls, envir))
  invisible(cls)
}

#' @rdname register_pytree_node_class
#' @export
register_pytree_with_keys_class <- function(cls, envir = parent.frame()) {
  register_pytree_with_keys(cls, tu_class_method("tree_flatten_with_keys", cls, envir),
                            tu_class_method("tree_unflatten", cls, envir),
                            tu_class_method("tree_flatten", cls, envir, required = FALSE))
}

#' Register an S3 list class with fixed fields as a pytree node
#'
#' Port of `jax.tree_util.register_dataclass` for S3 classes built on named lists:
#' `data_fields` become children (with [GetAttrKey()] paths), `meta_fields` are
#' static data kept in the treedef. R has no dataclass declarations, so both field
#' lists are required, as JAX requires them for classes that are not dataclasses.
#' Unflattening rebuilds the list with the data and meta fields (in their original
#' order) and the original class attribute.
#' @param nodetype Class name.
#' @param data_fields,meta_fields Character vectors of field names.
#' @param drop_fields Accepted for compatibility (fields outside `data_fields` and
#'   `meta_fields` are always dropped).
#' @return `nodetype`, invisibly.
#' @export
register_dataclass <- function(nodetype, data_fields = NULL, meta_fields = NULL,
                               drop_fields = character(0)) {
  if (is.null(data_fields) || is.null(meta_fields)) {
    if (is.null(data_fields) != is.null(meta_fields))
      jax_error("register_dataclass: data_fields and meta_fields must both be specified when ",
                "either is specified. Got data_fields=", deparse(data_fields), " meta_fields=",
                deparse(meta_fields), ".", class = "jax_type_error")
    jax_error("register_dataclass: data_fields and meta_fields are required when nodetype is ",
              "not a dataclass. Got nodetype=", nodetype, ".", class = "jax_type_error")
  }
  data_fields <- as.character(data_fields)
  meta_fields <- as.character(meta_fields)
  overlap <- intersect(data_fields, meta_fields)
  if (length(overlap))
    jax_error("data_fields and meta_fields must not overlap. Overlapping fields: ",
              paste0("'", overlap, "'", collapse = ", "), ".", class = "jax_value_error")
  flatten <- function(x) {
    v <- unclass(x)
    meta <- lapply(meta_fields, function(f) v[[f]])
    names(meta) <- meta_fields
    keep <- intersect(names(v), c(data_fields, meta_fields))
    list(children = unname(lapply(data_fields, function(f) v[[f]])),
         aux = list(class = oldClass(x), fields = keep, meta = meta))
  }
  unflatten <- function(aux, children) {
    names(children) <- data_fields
    vals <- c(children, aux$meta)
    fields <- c(aux$fields, setdiff(c(data_fields, meta_fields), aux$fields))
    structure(vals[fields], class = aux$class)
  }
  register_pytree_with_keys(nodetype, function(x) {
    fl <- flatten(x)
    list(children = Map(function(f, c) list(GetAttrKey(f), c), data_fields, fl$children),
         aux = fl$aux)
  }, unflatten, flatten)
}

#' Register a class as a static pytree node
#'
#' Port of `jax.tree_util.register_static`: instances have no leaves; the object itself
#' is kept in the treedef (so [jit()] treats it as static).
#' @param cls Class name.
#' @return `cls`, invisibly.
#' @export
register_static <- function(cls) {
  register_pytree_with_keys(cls, function(x) list(children = list(), aux = x),
                            function(aux, children) aux,
                            function(x) list(children = list(), aux = x))
}

# ---- Partial ------------------------------------------------------------------------------------

tu_make_partial <- function(func, args) {
  force(func)
  force(args)
  f <- function(...) do.call(func, c(args, list(...)))
  structure(f, class = c("jax_Partial", "function"), func = func, args = args)
}

#' A partially applied function that is a pytree
#'
#' Port of `jax.tree_util.Partial`: `Partial(func, ...)` calls
#' `func(<bound args>, <new args>)`. Its bound arguments are the leaves of a pytree and
#' `func` is static data, so it can be passed to transformed functions such as
#' [jit()].
#' @param func A function.
#' @param ... Arguments to bind (named arguments are bound by name).
#' @return A function of class `jax_Partial`.
#' @examples
#' \dontrun{
#' add_one <- Partial(jnp_add, 1)
#' jit(function(f, x) f(x))(add_one, 2)
#' }
#' @export
Partial <- function(func, ...) {
  if (!is.function(func)) jax_error("Partial: func must be a function", class = "jax_type_error")
  tu_make_partial(func, list(...))
}

#' @export
print.jax_Partial <- function(x, ...) {
  cat("Partial(<function>, ", length(attr(x, "args")), " bound argument(s))\n", sep = "")
  invisible(x)
}

register_pytree_node("jax_Partial",
                     function(p) list(children = list(attr(p, "args")), aux = attr(p, "func")),
                     function(aux, children) tu_make_partial(aux, children[[1L]]))
