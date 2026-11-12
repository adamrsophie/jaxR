# jax.experimental.sparse: the generic API, autodiff wrappers, random BCOO arrays and
# the sparsify transformation (ports of jax/experimental/sparse/api.py, ad.py, random.py
# and transform.py).
#
# sparsify uses the jaxpr interpreter of JAX (`use_tracer = FALSE`); `use_tracer = TRUE`
# is accepted and uses the same interpreter. Inside sparsify, custom_jvp functions are
# evaluated through their primal jaxpr (their custom derivative rule is not kept).

sp_is_sparse <- function(x) inherits(x, "jax_sparse")

# ---- todense, empty, eye ------------------------------------------------------------------

todense_p <- new_primitive("todense")

sp_todense_impl <- function(args, params) {
  arr <- tree_unflatten(params$tree, args)
  if (sp_is_sparse(arr)) arr$todense() else arr
}
todense_p$impl <- sp_todense_impl
todense_p$abstract_eval <- function(avals, params) {
  arr <- tree_unflatten(params$tree, avals)
  if (inherits(arr, "jax_aval")) return(arr)
  data <- .subset2(arr, "data")
  new_aval(.subset2(arr, "shape"), data$dtype, data$weak)
}
todense_p$lowering <- lower_fun(sp_todense_impl)
todense_p$jvp <- function(primals, tangents, params) {
  out <- bind(todense_p, primals, params)
  t <- tangents[[1L]]
  if (is_zero(t)) return(list(out, sp_zero_tangent(out)))
  list(out, bind(todense_p, c(list(t), primals[-1L]), params))
}
todense_p$transpose <- function(ct, args, params) {
  obj <- tree_unflatten(params$tree, args)
  ct <- instantiate_zero(ct)
  if (!sp_is_sparse(obj)) return(list(ct))
  if (inherits(obj, "jax_bcoo")) return(list(bc_extract(args[[2L]], ct), NULL))
  if (inherits(obj, "jax_bcsr")) return(list(bs_extract(args[[2L]], args[[3L]], ct), NULL, NULL))
  if (inherits(obj, "jax_coo")) return(list(sp_coo_extract(args[[2L]], args[[3L]], ct), NULL, NULL))
  jax_error("todense_transpose for ", class(obj)[1L], class = "jax_not_implemented_error")
}
todense_p$batch <- function(vals, dims, params, axis) {
  ax <- lapply(dims, function(d) if (is.na(d)) NULL else d)
  f <- function(...) sp_todense_impl(list(...), params)
  list(do.call(vmap(f, in_axes = ax), vals), 0L)
}

#' Convert a sparse array to a dense array
#'
#' Port of `jax.experimental.sparse.todense` (dense arrays are returned unchanged).
#' @param arr A sparse or dense array.
#' @return A dense array.
#' @export
experimental_sparse_todense <- function(arr) {
  fl <- tree_flatten(arr)
  bind(todense_p, fl$leaves, list(tree = fl$treedef))
}

sp_formats <- c(bcoo = "BCOO", bcsr = "BCSR", coo = "COO", csr = "CSR", csc = "CSC")

sp_format_class <- function(fmt) {
  switch(fmt, bcoo = experimental_sparse_BCOO, bcsr = experimental_sparse_BCSR, coo = experimental_sparse_COO,
         csr = experimental_sparse_CSR, csc = experimental_sparse_CSC)
}

#' Empty and identity sparse arrays
#'
#' Ports of `jax.experimental.sparse.empty` and `eye`.
#' @param shape Shape.
#' @param N,M Number of rows and columns.
#' @param k Diagonal offset.
#' @param dtype Data dtype.
#' @param index_dtype Index dtype.
#' @param sparse_format One of `"bcoo"`, `"bcsr"`, `"coo"`, `"csr"`, `"csc"` (`eye` has no
#'   `"bcsr"`).
#' @param ... Extra format arguments (`n_batch`, `n_dense`, `nse`).
#' @return A sparse array.
#' @export
experimental_sparse_empty <- function(shape, dtype = NULL, index_dtype = "int32", sparse_format = "bcoo", ...) {
  if (!sparse_format %in% names(sp_formats))
    jax_error("sparse_format='", sparse_format, "' not recognized; must be one of ['bcsr', 'bcoo', 'coo', 'csr', 'csc']",
              class = "jax_value_error")
  sp_format_class(sparse_format)$`_empty`(as.integer(shape), dtype = dtype, index_dtype = index_dtype, ...)
}

#' @rdname experimental_sparse_empty
#' @export
experimental_sparse_eye <- function(N, M = NULL, k = 0L, dtype = NULL, index_dtype = "int32", sparse_format = "bcoo", ...) {
  M <- M %||% N
  N <- sp_static_int(N, "N"); M <- sp_static_int(M, "M"); k <- sp_static_int(k, "k")
  if (!sparse_format %in% c("bcoo", "coo", "csr", "csc"))
    jax_error("sparse_format='", sparse_format, "' not recognized for eye", class = "jax_value_error")
  sp_format_class(sparse_format)$`_eye`(N, M, k, dtype = dtype, index_dtype = index_dtype, ...)
}

#' Abstract sparse classes
#'
#' `experimental_sparse_JAXSparse` is the port of the abstract base class
#' `jax.experimental.sparse.JAXSparse`: every jaxR sparse array inherits from class
#' `jax_sparse` (test with `inherits(x, "jax_sparse")`). `experimental_sparse_SparseTracer`
#' names the tracer class of `sparsify(use_tracer = TRUE)` in JAX; jaxR's sparsify uses
#' the jaxpr interpreter in both modes, so neither object can be instantiated.
#' @param ... Ignored: neither class can be instantiated.
#' @format Class objects.
#' @name experimental_sparse_abstract
NULL

#' @rdname experimental_sparse_abstract
#' @export
experimental_sparse_JAXSparse <- sp_class(function(...) {
  jax_error("Can't instantiate abstract class JAXSparse", class = "jax_type_error")
}, list(), "JAXSparse")

#' @rdname experimental_sparse_abstract
#' @export
experimental_sparse_SparseTracer <- sp_class(function(...) {
  jax_error("SparseTracer objects are created by sparsify, not by users", class = "jax_type_error")
}, list(), "SparseTracer")

# ---- random -------------------------------------------------------------------------------

#' Random BCOO arrays
#'
#' Port of `jax.experimental.sparse.random_bcoo`.
#' @param key A PRNG key.
#' @param shape Shape.
#' @param dtype Data dtype.
#' @param indices_dtype Index dtype (default integer type).
#' @param nse Number of stored elements, or a fraction in `(0, 1)` of the sparse size.
#' @param n_batch,n_dense Numbers of batch and dense dimensions.
#' @param unique_indices Draw indices without replacement.
#' @param sorted_indices Sort the indices.
#' @param generator Function `(key, shape, dtype, ...)` drawing the data.
#' @param ... Passed to `generator`.
#' @return A BCOO array.
#' @export
experimental_sparse_random_bcoo <- function(key, shape, dtype = "float", indices_dtype = NULL, nse = 0.2, n_batch = 0L,
                                            n_dense = 0L, unique_indices = TRUE, sorted_indices = FALSE,
                                            generator = random_uniform, ...) {
  shape <- as.integer(shape); n_batch <- as.integer(n_batch); n_dense <- as.integer(n_dense)
  if (n_batch < 0L || n_dense < 0L || n_batch + n_dense > length(shape))
    jax_error("Invalid n_batch=", n_batch, ", n_dense=", n_dense, " for shape=", sp_tuple(shape), class = "jax_value_error")
  n_sparse <- length(shape) - n_batch - n_dense
  parts <- sp_split(shape, c(n_batch, n_sparse))
  batch_size <- prod_int(parts[[1L]]); sparse_size <- prod_int(parts[[2L]])
  if (!(nse >= 0 && nse < sparse_size))
    jax_error("got nse=", nse, ", expected to be between 0 and ", sparse_size, class = "jax_value_error")
  if (nse > 0 && nse < 1) nse <- ceiling(nse * sparse_size)
  nse <- as.integer(nse)
  indices_dtype <- if (is.null(indices_dtype)) default_int() else dtype(indices_dtype)
  if (sparse_size > iinfo(indices_dtype)$max)
    jax_error("indices_dtype=", indices_dtype, " does not have enough range to generate sparse indices of size ",
              sparse_size, ".", class = "jax_value_error")
  ind_fun <- function(k) {
    if (!length(parts[[2L]])) return(lax_full(c(nse, n_sparse), 0, indices_dtype))
    flat <- jnp_astype(random_choice(k, sparse_size, shape = nse, replace = !unique_indices), indices_dtype)
    jnp_column_stack(jnp_unravel_index(flat, parts[[2L]]))
  }
  keys <- random_split(key, batch_size + 1L)
  data <- generator(keys[0L], shape = c(parts[[1L]], nse, parts[[3L]]), dtype = dtype, ...)
  indices <- jnp_reshape(vmap(ind_fun)(keys[sl(1L, NULL)]), c(parts[[1L]], nse, n_sparse))
  mat <- experimental_sparse_BCOO(list(data, indices), shape = shape)
  if (sorted_indices) mat$sort_indices() else mat
}

# ---- autodiff -----------------------------------------------------------------------------

# Port of ad.flatten_fun_for_sparse_ad: sparse arguments are differentiated with
# respect to their data buffer only.
sp_flatten_for_ad <- function(fun, argnums, args) {
  argnums <- as.integer(argnums)
  fl1 <- tree_flatten(args, is_leaf = sp_is_sparse)
  sizes1 <- vapply(args, function(a) length(tree_flatten(a, is_leaf = sp_is_sparse)$leaves), 1L)
  leaf_of_arg <- rep(seq_along(args) - 1L, sizes1)
  argnums_flat1 <- which(leaf_of_arg %in% argnums)
  fl2 <- lapply(fl1$leaves, tree_flatten)
  sizes2 <- vapply(fl2, function(f) length(f$leaves), 1L)
  starts <- cumsum(c(0L, sizes2))
  args_flat <- do.call(c, lapply(fl2, `[[`, "leaves"))
  argnums_flat <- unlist(lapply(argnums_flat1, function(i) {
    nums <- starts[i] + seq_len(sizes2[i]) - 1L
    if (sp_is_sparse(fl1$leaves[[i]])) nums[1L] else nums
  }))
  fun_flat <- function(...) {
    flat <- list(...)
    leaves1 <- lapply(seq_along(fl2), function(i) tree_unflatten(fl2[[i]]$treedef, flat[starts[i] + seq_len(sizes2[i])]))
    do.call(fun, tree_unflatten(fl1$treedef, leaves1))
  }
  reconstruct <- function(i, g) {
    leaf <- fl1$leaves[[i]]
    if (!sp_is_sparse(leaf)) return(g)
    bufs <- fl2[[i]]$leaves
    extra <- sp_ndim(g) - sp_ndim(bufs[[1L]])
    out <- tree_unflatten(fl2[[i]]$treedef, c(list(g), bufs[-1L]))
    if (extra > 0L) {
      # Jacobians: the leading dims of g become new batch dims of the sparse result.
      lead <- sp_shape(g)[seq_len(extra)]
      rest <- lapply(bufs[-1L], function(b) jnp_broadcast_to(b, c(lead, sp_shape(b))))
      out <- tree_unflatten(fl2[[i]]$treedef, c(list(g), rest))
      # Only the batched formats are vmappable: the others keep their shape upstream.
      if (inherits(leaf, c("jax_bcoo", "jax_bcsr"))) out <- sp_set_shape(out, c(lead, .subset2(leaf, "shape")))
    }
    out
  }
  post <- function(grads_out) {
    if (length(argnums_flat) == 1L && !is.list(grads_out)) grads_out <- list(grads_out)
    leaf_grads <- rep(list(NULL), length(fl1$leaves))
    k <- 0L
    for (i in argnums_flat1) {
      k <- k + 1L
      leaf_grads[[i]] <- reconstruct(i, grads_out[[k]])
    }
    per_arg <- split(leaf_grads, leaf_of_arg)
    keep <- lapply(argnums + 1L, function(a) {
      tree_unflatten(tree_flatten(args[[a]], is_leaf = sp_is_sparse)$treedef,
                     leaf_grads[leaf_of_arg == a - 1L])
    })
    if (length(keep) == 1L) keep[[1L]] else keep
  }
  list(fun = fun_flat, argnums = as.integer(argnums_flat), args = args_flat, post = post)
}

sp_ad_wrapper <- function(transform, fun, argnums, has_aux, returns_value = FALSE) {
  force(transform); force(fun); force(argnums); force(has_aux)
  function(...) {
    s <- sp_flatten_for_ad(fun, argnums, list(...))
    out <- do.call(transform(s$fun, argnums = s$argnums, has_aux = has_aux), s$args)
    if (returns_value) {
      if (has_aux) return(list(out[[1L]], s$post(out[[2L]])))
      return(list(out[[1L]], s$post(out[[2L]])))
    }
    if (has_aux) return(list(s$post(out[[1L]]), out[[2L]]))
    s$post(out)
  }
}

#' Differentiation with sparse arguments
#'
#' Ports of `jax.experimental.sparse.grad`, `value_and_grad`, `jacfwd`, `jacrev` and
#' `jacobian`: like [grad()] and friends, but sparse arguments are differentiated with
#' respect to their stored values, and their gradients are sparse arrays with the same
#' indices.
#' @param fun Function to differentiate.
#' @param argnums 0-based argument numbers.
#' @param has_aux Whether `fun` returns `list(value, aux)`.
#' @return A function.
#' @name experimental_sparse_ad
NULL

#' @rdname experimental_sparse_ad
#' @export
experimental_sparse_grad <- function(fun, argnums = 0L, has_aux = FALSE) {
  sp_ad_wrapper(function(f, argnums, has_aux) grad(f, argnums = argnums, has_aux = has_aux), fun, argnums, has_aux)
}

#' @rdname experimental_sparse_ad
#' @export
experimental_sparse_value_and_grad <- function(fun, argnums = 0L, has_aux = FALSE) {
  sp_ad_wrapper(function(f, argnums, has_aux) value_and_grad(f, argnums = argnums, has_aux = has_aux), fun, argnums,
                has_aux, returns_value = TRUE)
}

#' @rdname experimental_sparse_ad
#' @export
experimental_sparse_jacfwd <- function(fun, argnums = 0L, has_aux = FALSE) {
  sp_ad_wrapper(function(f, argnums, has_aux) jacfwd(f, argnums = argnums, has_aux = has_aux), fun, argnums, has_aux)
}

#' @rdname experimental_sparse_ad
#' @export
experimental_sparse_jacrev <- function(fun, argnums = 0L, has_aux = FALSE) {
  sp_ad_wrapper(function(f, argnums, has_aux) jacrev(f, argnums = argnums, has_aux = has_aux), fun, argnums, has_aux)
}

#' @rdname experimental_sparse_ad
#' @export
experimental_sparse_jacobian <- experimental_sparse_jacrev

# ---- sparsify -----------------------------------------------------------------------------

sp_env <- function(bufs = list()) {
  e <- new.env(parent = emptyenv())
  e$buffers <- bufs
  e
}
sp_push <- function(spenv, arr) {
  spenv$buffers[[length(spenv$buffers) + 1L]] <- as_jax(arr)
  length(spenv$buffers)
}
sp_value <- function(shape, data_ref, indices_ref = NULL, indptr_ref = NULL, indices_sorted = FALSE,
                     unique_indices = FALSE) {
  structure(list(shape = as.integer(shape), data_ref = data_ref, indices_ref = indices_ref, indptr_ref = indptr_ref,
                 indices_sorted = isTRUE(indices_sorted), unique_indices = isTRUE(unique_indices)),
            class = "jax_spvalue")
}
sv_is_sparse <- function(v) !is.null(v$indices_ref)
sv_is_bcoo <- function(v) sv_is_sparse(v) && is.null(v$indptr_ref)
sv_is_bcsr <- function(v) sv_is_sparse(v) && !is.null(v$indptr_ref)
sv_data <- function(spenv, v) spenv$buffers[[v$data_ref]]
sv_indices <- function(spenv, v) spenv$buffers[[v$indices_ref]]
sv_indptr <- function(spenv, v) spenv$buffers[[v$indptr_ref]]
sv_dense <- function(spenv, data) {
  data <- as_jax(data)
  sp_value(sp_shape(data), sp_push(spenv, data))
}
sv_sparse <- function(spenv, shape, data = NULL, indices = NULL, indptr = NULL, data_ref = NULL, indices_ref = NULL,
                      indptr_ref = NULL, indices_sorted = FALSE, unique_indices = FALSE) {
  if (!is.null(data)) data_ref <- sp_push(spenv, data)
  if (!is.null(indices)) indices_ref <- sp_push(spenv, indices)
  if (!is.null(indptr)) indptr_ref <- sp_push(spenv, indptr)
  sp_value(shape, data_ref, indices_ref, indptr_ref, indices_sorted, unique_indices)
}

sp_arrays_to_spvalues <- function(spenv, args) {
  tree_map(function(a) {
    if (inherits(a, "jax_bcoo")) return(sv_sparse(spenv, a$shape, a$data, a$indices, indices_sorted = a$indices_sorted,
                                                  unique_indices = a$unique_indices))
    if (inherits(a, "jax_bcsr")) return(sv_sparse(spenv, a$shape, a$data, a$indices, a$indptr,
                                                  indices_sorted = a$indices_sorted, unique_indices = a$unique_indices))
    if (sp_is_sparse(a)) jax_error("sparsify supports BCOO and BCSR arrays, got ", class(a)[1L],
                                   class = "jax_not_implemented_error")
    sv_dense(spenv, a)
  }, args, is_leaf = sp_is_sparse)
}

sp_spvalues_to_arrays <- function(spenv, spvalues) {
  tree_map(function(v) {
    if (sv_is_bcoo(v)) return(bc_new(sv_data(spenv, v), sv_indices(spenv, v), v$shape, v$indices_sorted, v$unique_indices))
    if (sv_is_bcsr(v)) return(bs_new(sv_data(spenv, v), sv_indices(spenv, v), sv_indptr(spenv, v), v$shape,
                                     v$indices_sorted, v$unique_indices))
    sv_data(spenv, v)
  }, spvalues, is_leaf = function(x) inherits(x, "jax_spvalue"))
}

sp_spvalue_aval <- function(spenv, v) {
  d <- aval_of(sv_data(spenv, v))
  new_aval(v$shape, d$dtype, d$weak)
}

sp_unimplemented <- function(prim) {
  densifying <- c("acos", "acosh", "bessel_i0e", "cos", "cosh", "eq", "exp", "exp2", "ge", "gt", "le", "lt", "log",
                  "log2", "ne", "xor")
  if (prim$name %in% densifying)
    jax_error("sparse rule for ", prim$name, " is not implemented because it would result in dense output. If this ",
              "is your intent, use sparse.todense() to convert your arguments to dense matrices.",
              class = "jax_not_implemented_error")
  jax_error("sparse rule for ", prim$name, " is not implemented.", class = "jax_not_implemented_error")
}

sp_rules_bcoo <- new.env(parent = emptyenv())
sp_rules_bcsr <- new.env(parent = emptyenv())

sp_eval_sparse <- function(jaxpr, consts, spvalues, spenv) {
  env <- list()
  key <- function(v) as.character(v$id)
  for (i in seq_along(jaxpr$constvars)) env[[key(jaxpr$constvars[[i]])]] <- sv_dense(spenv, consts[[i]])
  for (i in seq_along(jaxpr$invars)) env[[key(jaxpr$invars[[i]])]] <- spvalues[[i]]
  rd <- function(a) if (is_lit(a)) sv_dense(spenv, a$val) else env[[key(a)]]
  for (e in jaxpr$eqns) {
    invals <- lapply(e$invars, rd)
    nm <- e$prim$name
    if (any(vapply(invals, sv_is_bcsr, NA))) {
      rule <- sp_rules_bcsr[[nm]]
      if (is.null(rule)) sp_unimplemented(e$prim)
      out <- rule(spenv, invals, e$params)
    } else if (any(vapply(invals, sv_is_bcoo, NA))) {
      rule <- sp_rules_bcoo[[nm]]
      if (is.null(rule)) sp_unimplemented(e$prim)
      out <- rule(spenv, invals, e$params)
    } else {
      res <- bind(e$prim, lapply(invals, function(v) sv_data(spenv, v)), e$params)
      if (!e$prim$multiple_results) res <- list(res)
      out <- lapply(res, function(b) sv_dense(spenv, b))
    }
    for (j in seq_along(e$outvars)) env[[key(e$outvars[[j]])]] <- out[[j]]
  }
  lapply(jaxpr$outvars, rd)
}

sp_sparsify_raw <- function(f) {
  function(spenv, spvalues) {
    fl <- tree_flatten(spvalues, is_leaf = function(x) inherits(x, "jax_spvalue"))
    avals <- lapply(fl$leaves, function(v) sp_spvalue_aval(spenv, v))
    tr <- trace_flat_fun(f, fl$treedef, avals)
    result <- sp_eval_sparse(tr$cj$jaxpr, tr$cj$consts, fl$leaves, spenv)
    list(result = result, out_tree = tr$out_tree)
  }
}

#' Sparsify a function
#'
#' Port of `jax.experimental.sparse.sparsify`: the returned function accepts BCOO and
#' BCSR arrays wherever `f` accepts dense arrays, as long as every primitive applied to a
#' sparse value has a sparse rule (elementwise functions that keep zeros, `+`, `-`, `*`,
#' `/` by a dense value, `%*%`, sums, reshapes, transposes, slicing, indexing, broadcast,
#' concatenation, `lax_while_loop`, `lax_cond` and `lax_scan` with sparse carries, and
#' [experimental_sparse_todense()]).
#' @param f A function.
#' @param use_tracer Accepted for compatibility; both modes use the jaxpr interpreter.
#' @return A function.
#' @export
#' @examples
#' \dontrun{
#' m <- experimental_sparse_BCOO$fromdense(matrix(c(0, 1, 0, 2), 2))
#' experimental_sparse_sparsify(function(m, v) jnp_sin(m) %*% v)(m, c(1, 1))
#' }
experimental_sparse_sparsify <- function(f, use_tracer = FALSE) {
  force(f)
  function(...) {
    spenv <- sp_env()
    spvalues <- sp_arrays_to_spvalues(spenv, list(...))
    r <- sp_sparsify_raw(f)(spenv, spvalues)
    tree_unflatten(r$out_tree, sp_spvalues_to_arrays(spenv, r$result))
  }
}

# Port of transform._sparsify_jaxpr: a closed jaxpr (with consts) over the flat buffers
# of the sparse values, plus its output tree.
sp_sparsify_jaxpr <- function(spenv, cj, spvalues) {
  args <- sp_spvalues_to_arrays(spenv, spvalues)
  fl <- tree_flatten(args)
  store <- new.env(parent = emptyenv())
  wrapped <- function(leaves) {
    a <- tree_unflatten(fl$treedef, leaves)
    svs <- sp_arrays_to_spvalues(spenv, a)
    res <- sp_eval_sparse(cj$jaxpr, cj$consts, svs, spenv)
    ofl <- tree_flatten(sp_spvalues_to_arrays(spenv, res))
    store$out_tree <- ofl$treedef
    ofl$leaves
  }
  list(cj = trace_to_jaxpr(wrapped, lapply(fl$leaves, aval_of)), out_tree = store$out_tree)
}

sp_hoist <- function(cj) list(jaxpr = closed_jaxpr(convert_constvars(cj$jaxpr)), consts = cj$consts)

sp_rule_unary <- function(name, linear) {
  force(name); force(linear)
  function(spenv, invals, params) {
    v <- invals[[1L]]
    if (!linear) v <- sp_ensure_unique(spenv, v)
    out <- bind(.prims[[name]], list(sv_data(spenv, v)), params)
    list(sv_sparse(spenv, v$shape, out, indices_ref = v$indices_ref, indptr_ref = v$indptr_ref,
                   indices_sorted = v$indices_sorted, unique_indices = v$unique_indices))
  }
}

sp_ensure_unique <- function(spenv, v) {
  if (!sv_is_sparse(v) || v$unique_indices) return(v)
  arr <- sp_spvalues_to_arrays(spenv, v)
  arr <- arr$sum_duplicates(nse = arr$nse, remove_zeros = FALSE)
  sp_arrays_to_spvalues(spenv, arr)
}

for (.n in c("abs", "asin", "asinh", "atan", "atanh", "bessel_i1e", "expm1", "log1p", "sign", "sin", "sinh", "sqrt",
             "square", "tan", "tanh", "convert_element_type")) {
  assign(.n, sp_rule_unary(.n, FALSE), envir = sp_rules_bcoo)
  assign(.n, sp_rule_unary(.n, FALSE), envir = sp_rules_bcsr)
}
for (.n in c("conj", "copy", "imag", "neg", "real", "stage")) {
  assign(.n, sp_rule_unary(.n, TRUE), envir = sp_rules_bcoo)
  assign(.n, sp_rule_unary(.n, TRUE), envir = sp_rules_bcsr)
}
rm(.n)

sp_standard_rule <- function(fun, multiple = FALSE) {
  force(fun)
  function(spenv, invals, params) {
    res <- fun(sp_spvalues_to_arrays(spenv, invals), params)
    sp_arrays_to_spvalues(spenv, if (multiple) res else list(res))
  }
}

sp_dg_dn <- function(dn) list(list(dn$lhs_contracting, dn$rhs_contracting), list(dn$lhs_batch, dn$rhs_batch))

sp_rules_bcoo$broadcast_in_dim <- sp_standard_rule(function(a, p)
  experimental_sparse_bcoo_broadcast_in_dim(a[[1L]], p$shape, p$broadcast_dimensions))
sp_rules_bcoo$concatenate <- sp_standard_rule(function(a, p) experimental_sparse_bcoo_concatenate(a, p$dimension))
sp_rules_bcoo$conv_general_dilated <- sp_standard_rule(function(a, p) {
  experimental_sparse_bcoo_conv_general_dilated(a[[1L]], a[[2L]], p$window_strides, p$padding, p$lhs_dilation,
                                                 p$rhs_dilation, NULL, p$feature_group_count, p$batch_group_count)
})
sp_rules_bcoo$dot_general <- sp_standard_rule(function(a, p)
  experimental_sparse_bcoo_dot_general(a[[1L]], a[[2L]], sp_dg_dn(p$dimension_numbers),
                                       preferred_element_type = p$preferred_element_type))
sp_rules_bcoo$dynamic_slice <- sp_standard_rule(function(a, p)
  experimental_sparse_bcoo_dynamic_slice(a[[1L]], a[-1L], p$slice_sizes))
sp_rules_bcoo$reshape <- sp_standard_rule(function(a, p) experimental_sparse_bcoo_reshape(a[[1L]], p$new_sizes))
sp_rules_bcoo$rev <- sp_standard_rule(function(a, p) experimental_sparse_bcoo_rev(a[[1L]], p$dimensions))
sp_rules_bcoo$slice <- sp_standard_rule(function(a, p)
  experimental_sparse_bcoo_slice(a[[1L]], p$start_indices, p$limit_indices, p$strides))
sp_rules_bcoo$squeeze <- sp_standard_rule(function(a, p) experimental_sparse_bcoo_squeeze(a[[1L]], p$dimensions))
sp_rules_bcsr$dot_general <- sp_standard_rule(function(a, p)
  experimental_sparse_bcsr_dot_general(a[[1L]], a[[2L]], sp_dg_dn(p$dimension_numbers),
                                       preferred_element_type = p$preferred_element_type))
sp_rules_bcsr$broadcast_in_dim <- sp_standard_rule(function(a, p)
  experimental_sparse_bcsr_broadcast_in_dim(a[[1L]], p$shape, p$broadcast_dimensions))
sp_rules_bcsr$concatenate <- sp_standard_rule(function(a, p) experimental_sparse_bcsr_concatenate(a, p$dimension))

sp_rules_bcoo$integer_pow <- function(spenv, invals, params) {
  if (params$y <= 0L)
    jax_error("sparse rule for integer_pow with non-positive exponent ", params$y, " is not implemented because it ",
              "would result in dense output. If this is your intent, use sparse.todense() to convert your argument to ",
              "a dense array.", class = "jax_not_implemented_error")
  sp_rule_unary("integer_pow", FALSE)(spenv, invals, params)
}
sp_rules_bcsr$integer_pow <- sp_rules_bcoo$integer_pow

sp_rules_bcoo$transpose <- function(spenv, invals, params) {
  perm <- as.integer(params$permutation)
  v <- invals[[1L]]
  arr <- sp_spvalues_to_arrays(spenv, v)
  mt <- experimental_sparse_bcoo_transpose(arr, perm)
  nb <- arr$n_batch; ns <- arr$n_sparse
  n <- length(arr$shape)
  b_same <- identical(perm[seq_len(nb)], sp_pos(nb))
  d_same <- identical(perm[seq_along(perm) > nb + ns], if (n > nb + ns) seq.int(nb + ns, n - 1L) else integer(0))
  s_same <- identical(perm[nb + seq_len(ns)], nb + sp_pos(ns))
  args <- list(spenv = spenv, shape = arr$shape[perm + 1L], indices_sorted = mt$indices_sorted,
               unique_indices = mt$unique_indices)
  if (b_same && d_same) args$data_ref <- v$data_ref else args$data <- mt$data
  if (b_same && s_same) args$indices_ref <- v$indices_ref else args$indices <- mt$indices
  list(do.call(sv_sparse, args))
}

sp_stack_rule <- function(bid, cat) {
  function(spenv, invals, params) {
    arrays <- sp_spvalues_to_arrays(spenv, invals)
    axis <- params$axis
    base <- arrays[[1L]]$shape
    new_shape <- append(base, 1L, after = axis)
    bdims <- vapply(sp_pos(length(base)), function(d) if (d < axis) d else d + 1L, 1L)
    expanded <- lapply(arrays, function(x) bid(x, new_shape, bdims))
    sp_arrays_to_spvalues(spenv, list(cat(expanded, axis)))
  }
}
sp_rules_bcoo$stack <- sp_stack_rule(experimental_sparse_bcoo_broadcast_in_dim, experimental_sparse_bcoo_concatenate)
sp_rules_bcsr$stack <- sp_stack_rule(experimental_sparse_bcsr_broadcast_in_dim, experimental_sparse_bcsr_concatenate)

sp_rules_bcoo$add <- function(spenv, invals, params) {
  X <- invals[[1L]]; Y <- invals[[2L]]
  out_shape <- broadcast_shapes_impl(list(X$shape, Y$shape))
  if (sv_is_sparse(X) && sv_is_sparse(Y)) {
    if (!identical(X$shape, Y$shape))
      jax_error("Addition between sparse matrices of different shapes.", class = "jax_not_implemented_error")
    if (identical(X$indices_ref, Y$indices_ref)) {
      out <- lax_add(sv_data(spenv, X), sv_data(spenv, Y))
      return(list(sv_sparse(spenv, X$shape, out, indices_ref = X$indices_ref, indices_sorted = X$indices_sorted,
                            unique_indices = X$unique_indices)))
    }
    xi <- sv_indices(spenv, X); yi <- sv_indices(spenv, Y)
    if (sp_ndim(xi) != sp_ndim(yi) || sp_ndim(sv_data(spenv, X)) != sp_ndim(sv_data(spenv, Y)))
      jax_error("Addition between sparse matrices with different batch/dense dimensions.",
                class = "jax_not_implemented_error")
    k <- sp_ndim(xi) - 2L
    return(list(sv_sparse(spenv, X$shape, lax_concatenate(list(sv_data(spenv, X), sv_data(spenv, Y)), k),
                          lax_concatenate(list(xi, yi), k))))
  }
  if (sv_is_sparse(Y)) { tmp <- X; X <- Y; Y <- tmp }
  if (!identical(Y$shape, out_shape))
    jax_error("Addition between a sparse array X and a dense array Y is not implemented when the output shape is ",
              "larger than Y.shape. This is to prevent silent densification of a large sparse array. If this is your ",
              "intent, you can explicitly cast the sparse array to a dense matrix.", class = "jax_not_implemented_error")
  xy <- sp_spvalues_to_arrays(spenv, list(X, Y))
  list(sv_dense(spenv, jnp_add(xy[[1L]]$todense(), xy[[2L]])))
}

sp_rules_bcoo$sub <- function(spenv, invals, params) {
  if (sv_is_sparse(invals[[1L]]) && sv_is_sparse(invals[[2L]]))
    return(sp_rules_bcoo$add(spenv, list(invals[[1L]], sp_rules_bcoo$neg(spenv, invals[2L], list())[[1L]]), list()))
  jax_error("Subtraction between sparse and dense array.", class = "jax_not_implemented_error")
}

sp_rules_bcoo$mul <- function(spenv, invals, params) {
  X <- invals[[1L]]; Y <- invals[[2L]]
  if (sv_is_sparse(X) && sv_is_sparse(Y)) {
    if (identical(X$indices_ref, Y$indices_ref) && X$unique_indices) {
      out <- lax_mul(sv_data(spenv, X), sv_data(spenv, Y))
      return(list(sv_sparse(spenv, X$shape, out, indices_ref = X$indices_ref, indices_sorted = X$indices_sorted,
                            unique_indices = TRUE)))
    }
    xy <- sp_spvalues_to_arrays(spenv, invals)
    mat <- experimental_sparse_bcoo_multiply_sparse(xy[[1L]], xy[[2L]])
    return(list(sv_sparse(spenv, mat$shape, mat$data, mat$indices)))
  }
  if (sv_is_sparse(Y)) { tmp <- X; X <- Y; Y <- tmp }
  xp <- sp_spvalues_to_arrays(spenv, X)
  out <- experimental_sparse_bcoo_multiply_dense(xp, sv_data(spenv, Y))
  list(sv_sparse(spenv, X$shape, out, indices_ref = X$indices_ref, indices_sorted = X$indices_sorted,
                 unique_indices = X$unique_indices))
}

sp_rules_bcoo$div <- function(spenv, invals, params) {
  X <- invals[[1L]]; Y <- invals[[2L]]
  if (sv_is_sparse(Y))
    jax_error("Division by a sparse array is not implemented because it would result in dense output. If this is ",
              "your intent, use sparse.todense() to convert your arguments to a dense array.",
              class = "jax_not_implemented_error")
  xp <- sp_spvalues_to_arrays(spenv, X)
  yd <- sv_data(spenv, Y)
  out <- experimental_sparse_bcoo_multiply_dense(xp, lax_div(lax_full_like(yd, 1), yd))
  list(sv_sparse(spenv, X$shape, out, indices_ref = X$indices_ref, indices_sorted = X$indices_sorted,
                 unique_indices = X$unique_indices))
}

sp_rules_bcoo$reduce_sum <- function(spenv, invals, params) {
  xp <- sp_spvalues_to_arrays(spenv, invals[[1L]])
  mat <- experimental_sparse_bcoo_reduce_sum(xp, params$axes)
  if (!length(mat$shape)) return(list(sv_dense(spenv, jnp_sum(mat$data))))
  list(sv_sparse(spenv, mat$shape, mat$data, mat$indices))
}

sp_rules_bcoo$gather <- function(spenv, invals, params) {
  a <- sp_spvalues_to_arrays(spenv, invals)
  r <- experimental_sparse_bcoo_gather(a[[1L]], a[[2L]], params$dimension_numbers, params$slice_sizes,
                                       unique_indices = params$unique_indices,
                                       indices_are_sorted = params$indices_are_sorted, mode = params$mode,
                                       fill_value = params$fill_value)
  sp_arrays_to_spvalues(spenv, list(r))
}

sp_rules_bcoo$`while` <- function(spenv, invals, params) {
  nc <- params$cond_nconsts; nb <- params$body_nconsts
  cconst <- invals[seq_len(nc)]
  bconst <- invals[nc + seq_len(nb)]
  init <- invals[seq_along(invals) > nc + nb]
  csp <- sp_sparsify_jaxpr(spenv, params$cond_jaxpr, c(cconst, init))
  bsp <- sp_sparsify_jaxpr(spenv, params$body_jaxpr, c(bconst, init))
  ch <- sp_hoist(csp$cj); bh <- sp_hoist(bsp$cj)
  flat <- function(vs) tree_flatten(sp_spvalues_to_arrays(spenv, vs))$leaves
  cc <- flat(cconst); bc <- flat(bconst); iv <- flat(init)
  out <- bind(while_p, c(ch$consts, cc, bh$consts, bc, iv),
              list(cond_jaxpr = ch$jaxpr, body_jaxpr = bh$jaxpr, cond_nconsts = length(ch$consts) + length(cc),
                   body_nconsts = length(bh$consts) + length(bc)))
  sp_arrays_to_spvalues(spenv, tree_unflatten(bsp$out_tree, out))
}

sp_rules_bcoo$scan <- function(spenv, invals, params) {
  ncon <- params$num_consts; ncar <- params$num_carry
  consts <- invals[seq_len(ncon)]
  carry <- invals[ncon + seq_len(ncar)]
  xs <- invals[seq_along(invals) > ncon + ncar]
  if (length(xs)) jax_error("sparse rule for scan with x values.", class = "jax_not_implemented_error")
  ssp <- sp_sparsify_jaxpr(spenv, params$jaxpr, c(consts, carry))
  sh <- sp_hoist(ssp$cj)
  cf <- tree_flatten(sp_spvalues_to_arrays(spenv, consts))$leaves
  cr <- tree_flatten(sp_spvalues_to_arrays(spenv, carry))
  p <- update_params(params, list(jaxpr = sh$jaxpr, num_consts = length(sh$consts) + length(cf),
                                  num_carry = length(cr$leaves)))
  out <- bind(scan_p, c(sh$consts, cf, cr$leaves), p)
  carry_out <- tree_unflatten(cr$treedef, out[seq_along(cr$leaves)])
  sp_arrays_to_spvalues(spenv, c(carry_out, out[seq_along(out) > length(cr$leaves)]))
}

sp_rules_bcoo$cond <- function(spenv, invals, params) {
  ops <- invals[-1L]
  sps <- lapply(params$branches, function(b) sp_sparsify_jaxpr(spenv, b, ops))
  for (s in sps[-1L]) if (!treedef_equal(sps[[1L]]$out_tree, s$out_tree))
    jax_error("sparsified true_fun output and sparsified false_fun output must have identical types",
              class = "jax_type_error")
  jc <- join_consts(lapply(sps, `[[`, "cj"))
  args <- tree_flatten(sp_spvalues_to_arrays(spenv, ops))$leaves
  out <- bind(cond_p, c(list(sv_data(spenv, invals[[1L]])), jc$consts, args), list(branches = jc$jaxprs))
  sp_arrays_to_spvalues(spenv, tree_unflatten(sps[[1L]]$out_tree, out))
}

sp_rules_bcoo$todense <- function(spenv, invals, params) {
  list(sv_dense(spenv, sp_spvalues_to_arrays(spenv, invals[[1L]])$todense()))
}
sp_rules_bcsr$todense <- sp_rules_bcoo$todense

sp_custom_jvp_rule <- function(spenv, invals, params) {
  cj <- params$call_jaxpr
  sp_eval_sparse(cj$jaxpr, cj$consts, invals, spenv)
}
sp_rules_bcoo$custom_jvp_call <- sp_custom_jvp_rule
sp_rules_bcsr$custom_jvp_call <- sp_custom_jvp_rule

# ---- operators ----------------------------------------------------------------------------

sp_swap <- function(f) function(a, b) f(b, a)

sp_binary <- function(op, e1, e2) {
  fn <- switch(op,
    "+" = jnp_add, "-" = jnp_subtract, "*" = jnp_multiply, "/" = jnp_true_divide,
    ">" = jnp_greater, ">=" = jnp_greater_equal, "<" = jnp_less, "<=" = jnp_less_equal,
    "==" = jnp_equal, "!=" = jnp_not_equal,
    "^" = NULL,
    jax_error("unsupported operator ", op, " for sparse arrays", class = "jax_not_implemented_error"))
  if (op == "^") {
    if (sp_is_sparse(e1)) return(experimental_sparse_sparsify(function(x) jnp_power(x, e2))(e1))
    return(experimental_sparse_sparsify(sp_swap(jnp_power))(e2, e1))
  }
  if (!inherits(e1, c("jax_bcoo", "jax_bcsr")) && !inherits(e2, c("jax_bcoo", "jax_bcsr")))
    jax_error(op, " is not implemented for ", class(if (sp_is_sparse(e1)) e1 else e2)[1L], " arrays",
              class = "jax_not_implemented_error")
  experimental_sparse_sparsify(fn)(e1, e2)
}

#' Operators for sparse arrays
#'
#' Arithmetic, comparison and `%*%` operators for BCOO arrays (and `%*%` for BCSR, COO,
#' CSR and CSC), ports of the sparse array dunder methods of JAX (which wrap the jnp
#' functions with [experimental_sparse_sparsify()]).
#' @param e1,e2,x,y Sparse or dense operands.
#' @param ... Indices for `[`.
#' @param mx,my,cl,reverse See [base::chooseOpsMethod()].
#' @name jax_sparse_operators
NULL

#' @rdname jax_sparse_operators
#' @export
Ops.jax_sparse <- function(e1, e2) sp_ops(.Generic, e1, if (missing(e2)) NULL else e2, missing(e2))

sp_ops <- function(op, e1, e2, unary = FALSE) {
  if (unary) {
    return(switch(op,
      "-" = experimental_sparse_sparsify(jnp_negative)(e1),
      "+" = experimental_sparse_sparsify(jnp_positive)(e1),
      jax_error("unsupported operator ", op, " for sparse arrays", class = "jax_not_implemented_error")))
  }
  sp_binary(op, e1, e2)
}

#' @rdname jax_sparse_operators
#' @export
matrixOps.jax_sparse <- function(x, y) sp_matmul(x, y)

sp_matmul <- function(x, y) {
  if (inherits(x, "jax_coo")) return(co_matmul(x, y))
  if (inherits(x, c("jax_csr", "jax_csc"))) return(cs_matmul(x, y))
  if (inherits(y, c("jax_coo", "jax_csr", "jax_csc")))
    jax_error(class(y)[1L], ".__rmatmul__", class = "jax_not_implemented_error")
  experimental_sparse_sparsify(jnp_matmul)(x, y)
}

#' @rdname jax_sparse_operators
#' @export
chooseOpsMethod.jax_sparse <- function(x, y, mx, my, cl, reverse) TRUE

#' @rdname jax_sparse_operators
#' @export
Math.jax_sparse <- function(x, ...) {
  if (.Generic == "abs") return(experimental_sparse_sparsify(jnp_abs)(x))
  jax_error(.Generic, "() is not defined for sparse arrays; use experimental_sparse_sparsify(jnp_", .Generic, ")",
            class = "jax_not_implemented_error")
}

#' @rdname jax_sparse_operators
#' @export
`[.jax_sparse` <- function(x, ...) {
  # Re-evaluates the caller's own `x[...]` expression on the dense stand in of `x`, in
  # the caller's frame: the same code R would have run, so no new code is executed.
  cl <- sys.call()
  cl[[1L]] <- quote(`[`)
  cl[[2L]] <- quote(.sparse_operand)
  env <- parent.frame()
  f <- function(m) eval(cl, list(.sparse_operand = m), env)
  result <- experimental_sparse_sparsify(f)(x)
  if (inherits(x, "jax_bcoo") && !sp_is_sparse(result) && prod_int(sp_shape(result)) == 0L)
    result <- experimental_sparse_BCOO$fromdense(result)
  result
}
