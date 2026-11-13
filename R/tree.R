# Pytrees. Containers are plain (unclassed) lists, named or not, and NULL (an empty
# container). Classes registered with register_pytree_node() are containers too.
# Everything else is a leaf.

.pytree_registry <- new.env(parent = emptyenv())

#' Register a class as a pytree node
#'
#' @param class Class name (matched with `inherits()`).
#' @param flatten Function of a node returning `list(children = <list>, aux = <any>)`.
#' @param unflatten Function of `(aux, children)` returning a node.
#' @return Invisibly `NULL`.
#' @export
register_pytree_node <- function(class, flatten, unflatten) {
  assign(class, list(flatten = flatten, unflatten = unflatten), envir = .pytree_registry)
  invisible(NULL)
}

pytree_handler <- function(x) {
  cl <- oldClass(x)
  if (is.null(cl)) return(NULL)
  for (c in cl) {
    h <- .pytree_registry[[c]]
    if (!is.null(h)) return(h)
  }
  NULL
}

is_container <- function(x) {
  is.null(x) || (is.list(x) && is.null(oldClass(x))) || !is.null(pytree_handler(x))
}

leaf_def <- structure(list(type = "leaf", n = 1L), class = "jax_treedef")
null_def <- structure(list(type = "null", n = 0L), class = "jax_treedef")

# A list whose elements all have distinct names is the analog of a Python dict: JAX
# visits dict children in sorted key order and matches two dicts by key. Returns the
# positions of the elements in visiting order, or NULL for a positional list.
dict_order <- function(nms) {
  if (is.null(nms) || anyNA(nms) || !all(nzchar(nms)) || anyDuplicated(nms)) return(NULL)
  order(nms, method = "radix")
}

# Treedef of a plain list: `children` and `names` are in visiting order; `ord[i]` is
# the position child i has in the list (NULL for a positional list).
list_def <- function(nms, ord, kids) {
  structure(list(type = "list", names = if (is.null(ord)) nms else nms[ord], ord = ord,
                 children = kids, n = sum(vapply(kids, function(k) k$n, 1L))),
            class = "jax_treedef")
}

tree_flatten_impl <- function(x, is_leaf, acc) {
  if (!is.null(is_leaf) && isTRUE(is_leaf(x))) {
    acc$leaves[length(acc$leaves) + 1L] <- list(x)
    return(leaf_def)
  }
  if (is.null(x)) return(null_def)
  if (is.list(x) && is.null(oldClass(x))) {
    nms <- names(x)
    ord <- dict_order(nms)
    kids <- lapply(if (is.null(ord)) seq_along(x) else ord,
                   function(i) tree_flatten_impl(x[[i]], is_leaf, acc))
    return(list_def(nms, ord, kids))
  }
  h <- pytree_handler(x)
  if (!is.null(h)) {
    fl <- h$flatten(x)
    kids <- lapply(fl$children, function(el) tree_flatten_impl(el, is_leaf, acc))
    n <- sum(vapply(kids, function(k) k$n, 1L))
    return(structure(list(type = "custom", class = oldClass(x)[1L], aux = fl$aux,
                          children = kids, n = n), class = "jax_treedef"))
  }
  acc$leaves[length(acc$leaves) + 1L] <- list(x)
  leaf_def
}

#' Flatten a pytree
#'
#' Containers are plain lists, `NULL` and classes registered with
#' [register_pytree_node()]; everything else is a leaf. A list whose elements all have
#' distinct names plays the role of a Python dict: as in JAX its leaves are visited in
#' sorted key order, and two such lists are matched by name, not by position (in
#' [tree_map()], cotangents, `in_axes` and so on). [tree_unflatten()] restores the
#' original order of the elements.
#' @param tree A pytree (nested lists of arrays).
#' @param is_leaf Optional predicate marking subtrees to treat as leaves.
#' @return `tree_flatten()`: a list with `leaves` and `treedef`. `tree_leaves()`: the
#'   list of leaves. `tree_structure()`: the treedef.
#' @export
tree_flatten <- function(tree, is_leaf = NULL) {
  acc <- new.env(parent = emptyenv())
  acc$leaves <- list()
  td <- tree_flatten_impl(tree, is_leaf, acc)
  list(leaves = acc$leaves, treedef = td)
}

#' Rebuild a pytree from a treedef and leaves
#' @param treedef A treedef from [tree_flatten()].
#' @param leaves A list of leaves.
#' @export
tree_unflatten <- function(treedef, leaves) {
  if (length(leaves) != treedef$n)
    jax_error("tree_unflatten: expected ", treedef$n, " leaves, got ", length(leaves),
              class = "jax_value_error")
  pos <- 0L
  build <- function(td) {
    switch(td$type,
      leaf = { pos <<- pos + 1L; leaves[[pos]] },
      null = NULL,
      list = {
        out <- lapply(td$children, build)
        nms <- td$names
        if (!is.null(td$ord)) {
          back <- order(td$ord)
          out <- out[back]
          nms <- nms[back]
        }
        names(out) <- nms
        out
      },
      custom = {
        kids <- lapply(td$children, build)
        .pytree_registry[[td$class]]$unflatten(td$aux, kids)
      })
  }
  build(treedef)
}

#' @rdname tree_flatten
#' @export
tree_leaves <- function(tree, is_leaf = NULL) tree_flatten(tree, is_leaf)$leaves

#' @rdname tree_flatten
#' @export
tree_structure <- function(tree, is_leaf = NULL) tree_flatten(tree, is_leaf)$treedef

#' @rdname treedef_children
#' @export
treedef_num_leaves <- function(treedef) treedef$n

#' Map and reduce over the leaves of pytrees
#'
#' Ports of `jax.tree.map`, `jax.tree.all` and `jax.tree.reduce`. `tree_map()` applies
#' `f` to the leaves of one or more pytrees of the same structure. `tree_all()` is `TRUE`
#' when every leaf is true. `tree_reduce()` folds the leaves from the left with `f`.
#' @param f Function applied to leaves (`tree_map()`), or a function of two arguments
#'   that combines the accumulated value with the next leaf (`tree_reduce()`).
#' @param tree Pytree.
#' @param ... Further pytrees with the same structure as `tree`.
#' @param is_leaf Optional leaf predicate.
#' @param initializer Optional start value of `tree_reduce()`. Without it a tree with no
#'   leaves is an error.
#' @return `tree_map()`: a pytree with the structure of `tree`. `tree_all()`: a logical.
#'   `tree_reduce()`: the accumulated value.
#' @export
tree_map <- function(f, tree, ..., is_leaf = NULL) {
  fl <- tree_flatten(tree, is_leaf)
  rest <- lapply(list(...), function(r) flatten_up_to(fl$treedef, r))
  leaves <- if (length(rest)) {
    do.call(Map, c(list(f, fl$leaves), rest))
  } else lapply(fl$leaves, f)
  tree_unflatten(fl$treedef, unname(leaves))
}

tree_mismatch <- function(...) jax_error("pytree structure mismatch: ", ..., class = "jax_value_error")

# Flatten `tree` up to the structure of `treedef` (treedef is a prefix). A list whose
# elements all have distinct names is a dict and is matched by name; any other list is
# matched by position, and the two kinds do not match each other. With `null_ok` a NULL
# in place of a subtree stands for NULL at every leaf of it (the zero cotangents a
# custom VJP rule may return).
flatten_up_to <- function(treedef, tree, null_ok = FALSE) {
  out <- list()
  go <- function(td, x) {
    if (null_ok && is.null(x) && !(td$type %in% c("null", "leaf"))) {
      out[length(out) + seq_len(td$n)] <<- vector("list", td$n)
      return(invisible())
    }
    switch(td$type,
      leaf = { out[length(out) + 1L] <<- list(x) },
      null = if (!is.null(x)) tree_mismatch("expected NULL"),
      list = {
        if (!(is.list(x) && is.null(oldClass(x))) || length(x) != length(td$children))
          tree_mismatch("expected a list of length ", length(td$children))
        nx <- names(x)
        x_dict <- !is.null(dict_order(nx))
        if (!is.null(td$ord)) {
          if (!x_dict || !setequal(nx, td$names))
            tree_mismatch("expected a list with names (", paste(td$names, collapse = ", "),
                          "), got ", if (x_dict) paste0("names (", paste(nx, collapse = ", "), ")")
                                     else "a list without names")
          for (i in seq_along(td$children)) go(td$children[[i]], x[[td$names[i]]])
        } else {
          if (x_dict && length(x))
            tree_mismatch("expected a list without names, got names (",
                          paste(nx, collapse = ", "), ")")
          for (i in seq_along(td$children)) go(td$children[[i]], x[[i]])
        }
      },
      custom = {
        h <- pytree_handler(x)
        if (is.null(h) || !inherits(x, td$class))
          tree_mismatch("expected a node of class ", td$class, ", got ", class(x)[1L])
        fl <- h$flatten(x)
        if (length(fl$children) != length(td$children))
          tree_mismatch("expected a ", td$class, " node with ", length(td$children),
                        " children, got ", length(fl$children))
        if (!identical(td$aux, fl$aux))
          tree_mismatch("the static data of two ", td$class, " nodes differ")
        for (i in seq_along(td$children)) go(td$children[[i]], fl$children[[i]])
      })
  }
  go(treedef, tree)
  out
}

#' @rdname tree_map
#' @export
tree_all <- function(tree, is_leaf = NULL) {
  all(vapply(tree_leaves(tree, is_leaf), function(l) isTRUE(as.logical(l)), NA))
}

#' @rdname tree_map
#' @export
tree_reduce <- function(f, tree, initializer, is_leaf = NULL) {
  leaves <- tree_leaves(tree, is_leaf)
  if (!missing(initializer)) return(Reduce(f, leaves, initializer))
  if (!length(leaves))
    jax_error("reduce() of empty iterable with no initial value", class = "jax_type_error")
  Reduce(f, leaves)
}

#' Transpose a pytree of pytrees
#' @param outer_treedef,inner_treedef Treedefs.
#' @param tree Pytree with structure outer of inner.
#' @export
tree_transpose <- function(outer_treedef, inner_treedef, tree) {
  outer_leaves <- flatten_up_to(outer_treedef, tree)
  inner_flat <- lapply(outer_leaves, function(l) flatten_up_to(inner_treedef, l))
  n_in <- inner_treedef$n
  cols <- lapply(seq_len(n_in), function(j) {
    tree_unflatten(outer_treedef, lapply(inner_flat, `[[`, j))
  })
  tree_unflatten(inner_treedef, cols)
}

#' @export
print.jax_treedef <- function(x, ...) {
  cat("PyTreeDef(", format(x), ")\n", sep = "")
  invisible(x)
}

#' @export
format.jax_treedef <- function(x, ...) {
  switch(x$type,
    leaf = "*",
    null = "NULL",
    list = {
      kids <- vapply(x$children, format, "")
      if (!is.null(x$names)) kids <- ifelse(nzchar(x$names), paste0(x$names, " = ", kids), kids)
      # Shown in the order of the list itself (children are stored in visiting order).
      if (!is.null(x$ord)) kids <- kids[order(x$ord)]
      paste0("list(", paste(kids, collapse = ", "), ")")
    },
    custom = paste0(x$class, "(", paste(vapply(x$children, format, ""), collapse = ", "), ")"))
}

# Structural equality as in JAX: lists with the same distinct names are equal whatever
# the order of their elements.
treedef_equal <- function(a, b) {
  if (!identical(a$type, b$type) || a$n != b$n || length(a$children) != length(b$children))
    return(FALSE)
  if (identical(a$type, "list") &&
      (!identical(a$names, b$names) || is.null(a$ord) != is.null(b$ord))) return(FALSE)
  if (identical(a$type, "custom") && (!identical(a$class, b$class) || !identical(a$aux, b$aux)))
    return(FALSE)
  for (i in seq_along(a$children)) if (!treedef_equal(a$children[[i]], b$children[[i]])) return(FALSE)
  TRUE
}

# Cache key of a treedef for jit: the structure itself (node kinds, names and their
# order, classes and the aux data of custom nodes, such as register_dataclass() meta
# fields or the function of a Partial). The key is compared with identical(), so
# functions in the aux data are told apart by their environments.
treedef_key <- function(treedef) {
  list(treedef$type, treedef$names, treedef$ord, treedef$class,
       if (identical(treedef$type, "custom")) treedef$aux, lapply(treedef$children, treedef_key))
}

#' Flatten a pytree of arrays into one vector
#'
#' Port of `jax.flatten_util.ravel_pytree`.
#' @param pytree A pytree of arrays.
#' @return A list with `flat` (1-D array) and `unravel` (function).
#' @export
ravel_pytree <- function(pytree) {
  fl <- tree_flatten(pytree)
  leaves <- lapply(fl$leaves, as_jax)
  # Port of _ravel_list: no leaves give an empty float32 vector.
  if (!length(leaves)) {
    return(list(flat = lax_full(0L, 0, "float32"), unravel = function(v) tree_unflatten(fl$treedef, list())))
  }
  shapes <- lapply(leaves, function(l) aval_of(l)$shape)
  dts <- vapply(leaves, function(l) aval_of(l)$dtype, "")
  sizes <- vapply(shapes, prod_int, 1L)
  to_dtype <- Reduce(promote_types_dtype, dts)
  flat <- jnp_concatenate(lapply(leaves, function(l) jnp_ravel(lax_convert_element_type(l, to_dtype))))
  total <- sum(sizes)
  single <- all(dts == to_dtype)
  unravel <- function(v) {
    v <- as_jax(v)
    a <- aval_of(v)
    if (!identical(a$shape, total))
      jax_error("Sum of sizes ", total, " must be equal to dimension 0 of the operand shape [",
                paste(a$shape, collapse = ", "), "]", class = "jax_value_error")
    # With one common dtype JAX keeps the dtype of `v`; with mixed dtypes `v` must have
    # the promoted dtype and each piece is cast back to its leaf dtype.
    if (!single && a$dtype != to_dtype)
      jax_error("unravel function given array of dtype ", a$dtype, ", but expected dtype ", to_dtype,
                class = "jax_type_error")
    out <- vector("list", length(sizes))
    off <- 0L
    for (i in seq_along(sizes)) {
      piece <- lax_slice(v, off, off + sizes[i])
      if (!single) piece <- lax_convert_element_type(piece, dts[i])
      out[[i]] <- lax_reshape(piece, shapes[[i]])
      off <- off + sizes[i]
    }
    tree_unflatten(fl$treedef, out)
  }
  list(flat = flat, unravel = unravel)
}
