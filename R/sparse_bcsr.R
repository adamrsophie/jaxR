# jax.experimental.sparse: the BCSR, COO, CSR and CSC formats (ports of
# jax/experimental/sparse/bcsr.py, coo.py and csr.py). The cuSPARSE and CPU kernel
# lowerings are not ported; every primitive lowers through its reference implementation.

# ---- shared CSR helpers ---------------------------------------------------------------

# Port of util._csr_to_coo.
sp_csr_to_coo <- function(indices, indptr) {
  marks <- at_add(jnp_zeros_like(indices), indptr, value = 1L, mode = "drop")
  list(jnp_subtract(jnp_cumsum(marks), 1L), indices)
}

sp_csr_extract <- function(indices, indptr, mat) {
  rc <- sp_csr_to_coo(indices, indptr)
  sp_coo_extract(rc[[1L]], rc[[2L]], mat)
}

sp_coo_extract <- function(row, col, mat) at_get(as_jax(mat), row, col)

# ---- BCSR ---------------------------------------------------------------------------------

bs_validate_indices <- function(indices, indptr, shape) {
  shape <- as.integer(shape)
  is <- sp_shape(indices); ps <- sp_shape(indptr)
  nse <- is[length(is)]
  n_batch <- length(is) - 1L
  n_dense <- length(shape) - n_batch - 2L
  if (n_dense < 0L) jax_error("BCSR shape has too few dimensions")
  compatible <- function(a, b) all(a == 1L | a == b)
  if (!compatible(is[seq_len(n_batch)], shape[seq_len(n_batch)]))
    jax_error("indices batch dimensions not compatible for indices.shape=", sp_tuple(is), ", shape=",
              sp_tuple(shape), class = "jax_value_error")
  if (!compatible(ps[seq_len(n_batch)], shape[seq_len(n_batch)]))
    jax_error("indptr batch dimensions not compatible for indptr.shape=", sp_tuple(ps), ", shape=",
              sp_tuple(shape), class = "jax_value_error")
  if (!identical(as.integer(ps[seq_along(ps) > n_batch]), shape[n_batch + 1L] + 1L))
    jax_error("indptr shape must match the matrix shape plus 1.", class = "jax_value_error")
  list(n_batch = n_batch, n_dense = n_dense, nse = nse)
}

bs_validate <- function(data, indices, indptr, shape) {
  p <- bs_validate_indices(indices, indptr, shape)
  shape <- as.integer(shape)
  nb <- p$n_batch; nd <- p$n_dense
  n_sparse <- length(shape) - nb - nd
  if (n_sparse != 2L) jax_error("BCSR array must have 2 sparse dimensions; ", n_sparse, " is given.",
                                class = "jax_value_error")
  ds <- sp_shape(data)
  if (!all(ds[seq_len(nb)] == 1L | ds[seq_len(nb)] == shape[seq_len(nb)]))
    jax_error("data batch dimensions not compatible for data.shape=", sp_tuple(ds), ", shape=",
              sp_tuple(shape), class = "jax_value_error")
  tail <- ds[seq_along(ds) > length(ds) - (nd + 1L)]
  if (!identical(as.integer(tail), as.integer(c(p$nse, shape[seq_along(shape) > nb + 2L]))))
    jax_error("Invalid data.shape=", sp_tuple(ds), " for nse=", p$nse, ", n_batch=", nb, ", n_dense=", nd,
              class = "jax_value_error")
  p
}

bs_batch_dims_to_front <- function(vals, dims, spinfo, batch_size = NULL) {
  n_batch <- sp_ndim(vals[[2L]]) - 1L + as.integer(is.na(dims[2L]))
  if (!all(is.na(dims) | (dims >= 0L & dims < n_batch)))
    jax_error("batch_dims must be None or satisfy 0 < dim < n_batch. Got batch_dims=", sp_tuple(dims),
              " for n_batch=", n_batch, class = "jax_not_implemented_error")
  mv <- function(a, d) if (is.na(d)) lax_expand_dims(a, 0L) else moveaxis(a, d, 0L)
  if (is.null(batch_size))
    batch_size <- max(unlist(Map(function(a, d) if (is.na(d)) NULL else sp_shape(a)[d + 1L], vals, dims)))
  list(data = mv(vals[[1L]], dims[1L]), indices = mv(vals[[2L]], dims[2L]), indptr = mv(vals[[3L]], dims[3L]),
       spinfo = sp_spinfo(c(batch_size, spinfo$shape), spinfo$indices_sorted, spinfo$unique_indices))
}

bs_to_bcoo <- function(indices, indptr, shape) {
  p <- bs_validate_indices(indices, indptr, shape)
  f <- sp_nfold_vmap(function(i, p) sp_csr_to_coo(i, p), p$n_batch)
  rc <- f(indices, indptr)
  jnp_stack(rc, axis = sp_ndim(indices))
}

bs_from_bcoo_indices <- function(indices, shape, index_dtype) {
  p <- bc_validate_indices(indices, shape)
  if (p$n_sparse != 2L) jax_error("Must have 2 sparse dimensions to be converted to BCSR.", class = "jax_value_error")
  n_rows <- shape[p$n_batch + 1L]
  get_ptr <- sp_nfold_vmap(function(i) {
    counts <- jnp_cumsum(jnp_astype(jnp_bincount(i, length = n_rows), index_dtype))
    lax_concatenate(list(lax_full(1L, 0, index_dtype), counts), 0L)
  }, p$n_batch, broadcasted = FALSE)
  list(sp_index_last(indices, 1L), get_ptr(sp_index_last(indices, 0L)))
}

bcsr_fromdense_p <- new_primitive("bcsr_fromdense", multiple_results = TRUE)
bcsr_fromdense_p$abstract_eval <- function(avals, params) {
  ms <- avals[[1L]]$shape
  nb <- params$n_batch; nd <- params$n_dense
  if (length(ms) - nb - nd != 2L) jax_error("bcsr_fromdense: must have 2 sparse dimensions.", class = "jax_value_error")
  list(new_aval(c(ms[seq_len(nb)], params$nse, ms[seq_along(ms) > nb + 2L]), sp_zero_where_dtype(avals[[1L]]$dtype)),
       new_aval(c(ms[seq_len(nb)], params$nse), params$index_dtype),
       new_aval(c(ms[seq_len(nb)], ms[nb + 1L] + 1L), params$index_dtype))
}
bcsr_fromdense_p$lowering <- lower_fun(function(args, params) {
  mat <- args[[1L]]
  if (sp_ndim(mat) - params$n_dense - params$n_batch != 2L)
    jax_error("bcsr_fromdense: must have 2 sparse dimensions.", class = "jax_value_error")
  m <- experimental_sparse_bcoo_fromdense(mat, nse = params$nse, index_dtype = params$index_dtype,
                                          n_dense = params$n_dense, n_batch = params$n_batch)
  ip <- bs_from_bcoo_indices(m$indices, sp_shape(mat), params$index_dtype)
  list(m$data, ip[[1L]], ip[[2L]])
}, multiple_results = TRUE)
bcsr_fromdense_p$batch <- function(vals, dims, params, axis) {
  bdim <- dims[1L]
  if (!(bdim >= 0L && bdim <= params$n_batch))
    jax_error("Expected 0 < bdim <= n_batch; got bdim=", bdim, ", n_batch=", params$n_batch, class = "jax_value_error")
  params$n_batch <- params$n_batch + 1L
  list(bind(bcsr_fromdense_p, vals, params), c(bdim, bdim, bdim))
}
bcsr_fromdense_p$jvp <- function(primals, tangents, params) {
  out <- bind(bcsr_fromdense_p, primals, params)
  t <- tangents[[1L]]
  dd <- if (is_zero(t)) sp_zero_tangent(out[[1L]]) else bs_extract(out[[2L]], out[[3L]], t)
  list(out, list(dd, sp_zero_tangent(out[[2L]]), sp_zero_tangent(out[[3L]])))
}
bcsr_fromdense_p$transpose <- function(cts, args, params) {
  if (is_zero(cts[[2L]]) || is_zero(cts[[3L]]))
    jax_error("Cannot transpose with respect to sparse indices", class = "jax_value_error")
  list(bs_todense(instantiate_zero(cts[[1L]]), cts[[2L]], cts[[3L]], sp_spinfo(args[[1L]]$aval$shape)))
}

bcsr_todense_p <- new_primitive("bcsr_todense")
bcsr_todense_p$abstract_eval <- function(avals, params) {
  bs_validate(avals[[1L]], avals[[2L]], avals[[3L]], params$spinfo$shape)
  new_aval(params$spinfo$shape, avals[[1L]]$dtype)
}
bcsr_todense_p$lowering <- lower_fun(function(args, params) {
  shape <- params$spinfo$shape
  bc_todense(args[[1L]], bs_to_bcoo(args[[2L]], args[[3L]], shape), sp_spinfo(shape))
})
defjvp(bcsr_todense_p, function(t, primals, params) bs_todense(t, primals[[2L]], primals[[3L]], params$spinfo),
       NULL, NULL)
bcsr_todense_p$transpose <- function(ct, args, params) {
  if (is_undefined(args[[2L]]) || is_undefined(args[[3L]]))
    jax_error("Cannot transpose with respect to sparse indices", class = "jax_value_error")
  list(bs_extract(args[[2L]], args[[3L]], ct), NULL, NULL)
}
bcsr_todense_p$batch <- function(vals, dims, params, axis) {
  b <- bs_batch_dims_to_front(vals, dims, params$spinfo)
  list(bs_todense(b$data, b$indices, b$indptr, b$spinfo), 0L)
}
bs_todense <- function(data, indices, indptr, spinfo) {
  bind(bcsr_todense_p, lapply(list(data, indices, indptr), as_jax), list(spinfo = spinfo))
}

bcsr_extract_p <- new_primitive("bcsr_extract")
bcsr_extract_p$abstract_eval <- function(avals, params) {
  ms <- avals[[3L]]$shape
  p <- bs_validate_indices(avals[[1L]], avals[[2L]], ms)
  new_aval(c(ms[seq_len(p$n_batch)], p$nse, ms[seq_along(ms) > length(ms) - p$n_dense]), avals[[3L]]$dtype)
}
bcsr_extract_p$lowering <- lower_fun(function(args, params) {
  bc_extract(bs_to_bcoo(args[[1L]], args[[2L]], sp_shape(args[[3L]])), args[[3L]])
})
defjvp(bcsr_extract_p, NULL, NULL, function(t, primals, params) bs_extract(primals[[1L]], primals[[2L]], t))
bcsr_extract_p$transpose <- function(ct, args, params) {
  if (is_undefined(args[[1L]]) || is_undefined(args[[2L]]))
    jax_error("Cannot transpose with respect to sparse indices", class = "jax_value_error")
  list(NULL, NULL, bs_todense(ct, args[[1L]], args[[2L]], sp_spinfo(args[[3L]]$aval$shape)))
}
bcsr_extract_p$batch <- function(vals, dims, params, axis) {
  bset <- unique(dims[!is.na(dims)])
  if (length(bset) != 1L) jax_error("bcoo_extract with unequal batch dimensions.", class = "jax_not_implemented_error")
  bdim <- bset
  indices <- vals[[1L]]; indptr <- vals[[2L]]; arr <- vals[[3L]]
  if (is.na(dims[1L])) indices <- lax_expand_dims(indices, bdim)
  if (is.na(dims[2L])) indptr <- lax_expand_dims(indptr, bdim)
  if (is.na(dims[3L])) {
    rs <- append(sp_shape(arr), sp_shape(indices)[bdim + 1L], after = bdim)
    arr <- lax_broadcast_in_dim(arr, rs, sp_pos(length(rs))[-(bdim + 1L)])
  }
  if (bdim >= sp_ndim(indices) - 1L)
    jax_error("batch_dims out of range for indices with n_batch=", sp_ndim(indices) - 1L, class = "jax_value_error")
  list(bs_extract(indices, indptr, arr), bdim)
}
bs_extract <- function(indices, indptr, mat) bind(bcsr_extract_p, lapply(list(indices, indptr, mat), as_jax))

#' Extract values of a dense array at the indices of a BCSR array
#'
#' Port of `jax.experimental.sparse.bcsr_extract`.
#' @param indices,indptr BCSR index buffers.
#' @param mat A dense array.
#' @return The data buffer of a BCSR array.
#' @export
experimental_sparse_bcsr_extract <- function(indices, indptr, mat) bs_extract(indices, indptr, mat)

bcsr_dot_general_p <- new_primitive("bcsr_dot_general")
bcsr_dot_general_p$abstract_eval <- function(avals, params) {
  dn <- params$dimension_numbers
  p <- bs_validate_indices(avals[[2L]], avals[[3L]], params$lhs_spinfo$shape)
  out <- bc_dot_shape(new_aval(params$lhs_spinfo$shape, avals[[1L]]$dtype), avals[[4L]], dn, params$preferred_element_type)
  lb <- dn[[2L]][[1L]]; lc <- dn[[1L]][[1L]]
  if (length(lb) && max(lb) >= p$n_batch)
    jax_error("bcsr_dot_general batch dimensions must be among the batch dimensions in the sparse representtaion.\n",
              "got lhs_batch=", sp_tuple(lb), ", props.n_batch=", p$n_batch, class = "jax_not_implemented_error")
  if (any(lc >= p$n_batch + 2L)) jax_error("bcsr_dot_general: contracting over dense dimensions.",
                                           class = "jax_not_implemented_error")
  out
}
bcsr_dot_general_p$lowering <- lower_fun(function(args, params) {
  idx <- bs_to_bcoo(args[[2L]], args[[3L]], params$lhs_spinfo$shape)
  bc_dot_general_impl(list(args[[1L]], idx, args[[4L]]),
                      list(dimension_numbers = params$dimension_numbers,
                           preferred_element_type = params$preferred_element_type, lhs_spinfo = params$lhs_spinfo))
})
defjvp(bcsr_dot_general_p,
       function(t, primals, params) bs_dot_general(t, primals[[2L]], primals[[3L]], primals[[4L]], params),
       NULL, NULL,
       function(t, primals, params) bs_dot_general(primals[[1L]], primals[[2L]], primals[[3L]], t, params))
bcsr_dot_general_p$transpose <- function(ct, args, params) {
  idx <- bs_to_bcoo(args[[2L]], args[[3L]], params$lhs_spinfo$shape)
  r <- bcoo_dot_general_p$transpose(ct, list(args[[1L]], idx, args[[4L]]), params)
  list(r[[1L]], NULL, NULL, r[[3L]])
}
bcsr_dot_general_p$batch <- function(vals, dims, params, axis) {
  rhs <- vals[[4L]]; rbd <- dims[4L]
  b <- bs_batch_dims_to_front(vals[1:3], dims[1:3], params$lhs_spinfo,
                              batch_size = if (is.na(rbd)) NULL else sp_shape(rhs)[rbd + 1L])
  nd <- bc_dot_general_batch_dim_nums(c(length(params$lhs_spinfo$shape), sp_ndim(rhs)), c(0L, rbd),
                                      params$dimension_numbers)
  out <- bs_dot_general(b$data, b$indices, b$indptr, rhs,
                        list(dimension_numbers = nd$dn, preferred_element_type = params$preferred_element_type,
                             lhs_spinfo = b$spinfo))
  list(out, nd$bdim)
}
bs_dot_general <- function(data, indices, indptr, rhs, params) {
  params$dimension_numbers <- bc_dn(params$dimension_numbers)
  if (!is.null(params$preferred_element_type)) params$preferred_element_type <- dtype(params$preferred_element_type)
  bind(bcsr_dot_general_p, lapply(list(data, indices, indptr, rhs), as_jax), params)
}

#' Functions on BCSR arrays
#'
#' Ports of `jax.experimental.sparse.bcsr_fromdense`, `bcsr_todense`,
#' `bcsr_dot_general` (BCSR lhs, dense rhs), `bcsr_broadcast_in_dim`,
#' `bcsr_concatenate` and `bcsr_sum_duplicates`.
#' @param mat A dense array (`bcsr_fromdense`) or a BCSR array.
#' @param nse,n_batch,n_dense,index_dtype As for [experimental_sparse_bcoo_fromdense()].
#' @param lhs,rhs Operands of `bcsr_dot_general`.
#' @param dimension_numbers,precision,preferred_element_type As for [lax_dot_general()].
#' @param shape,broadcast_dimensions,sharding As for [lax_broadcast_in_dim()].
#' @param operands A list of BCSR arrays.
#' @param dimension 0-based dimension.
#' @return A BCSR or dense array.
#' @name experimental_sparse_bcsr_fromdense
NULL

#' @rdname experimental_sparse_bcsr_fromdense
#' @export
experimental_sparse_bcsr_fromdense <- function(mat, nse = NULL, n_batch = 0L, n_dense = 0L, index_dtype = "int32") {
  mat <- as_jax(mat)
  if (is.null(nse)) nse <- bc_count_stored(mat, n_batch, n_dense)
  nse <- sp_static_int(nse, sub("bcoo_fromdense", "bcsr_fromdense", bc_traced_nse))
  bufs <- bind(bcsr_fromdense_p, list(mat), list(nse = nse, n_batch = as.integer(n_batch), n_dense = as.integer(n_dense),
                                                 index_dtype = dtype(index_dtype)))
  experimental_sparse_BCSR(bufs, shape = sp_shape(mat))
}

#' @rdname experimental_sparse_bcsr_fromdense
#' @export
experimental_sparse_bcsr_todense <- function(mat) bs_todense(mat$data, mat$indices, mat$indptr, bc_info(mat))

#' @rdname experimental_sparse_bcsr_fromdense
#' @export
experimental_sparse_bcsr_dot_general <- function(lhs, rhs, dimension_numbers, precision = NULL,
                                                 preferred_element_type = NULL) {
  if (!inherits(rhs, "jax_sparse")) {
    if (!inherits(lhs, "jax_sparse"))
      return(lax_dot_general(lhs, rhs, dimension_numbers, preferred_element_type = preferred_element_type))
    if (inherits(lhs, "jax_bcsr"))
      return(bs_dot_general(lhs$data, lhs$indices, lhs$indptr, rhs,
                            list(dimension_numbers = dimension_numbers, preferred_element_type = preferred_element_type,
                                 lhs_spinfo = bc_info(lhs))))
  }
  jax_error("bcsr_dot_general currently implemented for BCSR lhs and ndarray rhs.", class = "jax_not_implemented_error")
}

#' @rdname experimental_sparse_bcsr_fromdense
#' @export
experimental_sparse_bcsr_broadcast_in_dim <- function(mat, shape, broadcast_dimensions, sharding = NULL) {
  bs_from_bcoo(experimental_sparse_bcoo_broadcast_in_dim(bs_to_bcoo_mat(mat), shape, broadcast_dimensions))
}

#' @rdname experimental_sparse_bcsr_fromdense
#' @export
experimental_sparse_bcsr_concatenate <- function(operands, dimension) {
  bs_from_bcoo(experimental_sparse_bcoo_concatenate(lapply(operands, bs_to_bcoo_mat), dimension))
}

#' @rdname experimental_sparse_bcsr_fromdense
#' @export
experimental_sparse_bcsr_sum_duplicates <- function(mat, nse = NULL) {
  bs_from_bcoo(experimental_sparse_bcoo_sum_duplicates(bs_to_bcoo_mat(mat), nse = nse))
}

bs_to_bcoo_mat <- function(mat) {
  experimental_sparse_BCOO(list(mat$data, bs_to_bcoo(mat$indices, mat$indptr, mat$shape)), shape = mat$shape)
}

bs_from_bcoo <- function(arr) {
  if (arr$n_sparse != 2L)
    jax_error("BSCR.from_bcoo requires n_sparse=2; got arr.n_sparse=", arr$n_sparse, class = "jax_not_implemented_error")
  if (!arr$indices_sorted) arr <- arr$sort_indices()
  ip <- bs_from_bcoo_indices(arr$indices, arr$shape, sp_dtype(arr$indices))
  experimental_sparse_BCSR(list(arr$data, ip[[1L]], ip[[2L]]), shape = arr$shape)
}

bs_new <- function(data, indices, indptr, shape, indices_sorted = FALSE, unique_indices = FALSE) {
  structure(list(data = data, indices = indices, indptr = indptr, shape = as.integer(shape),
                 indices_sorted = isTRUE(indices_sorted), unique_indices = isTRUE(unique_indices)),
            class = c("jax_bcsr", "jax_sparse"))
}

bs_empty <- function(shape, dtype = NULL, index_dtype = "int32", n_dense = 0L, n_batch = 0L, nse = 0L) {
  shape <- as.integer(shape)
  if (n_dense < 0L || n_batch < 0L || nse < 0L)
    jax_error("Invalid inputs: shape=", sp_tuple(shape), ", n_dense=", n_dense, ", n_batch=", n_batch, ", nse=", nse,
              class = "jax_value_error")
  if (length(shape) - n_dense - n_batch != 2L) jax_error("BCSR sparse.empty: must have 2 sparse dimensions.",
                                                        class = "jax_value_error")
  parts <- sp_split(shape, c(n_batch, 2L))
  idt <- dtype(index_dtype)
  bs_new(jnp_zeros(c(parts[[1L]], nse, parts[[3L]]), dtype), lax_full(c(parts[[1L]], nse), parts[[2L]][2L], idt),
         lax_full(c(parts[[1L]], parts[[2L]][1L] + 1L), 0, idt), shape)
}

#' Batched compressed sparse row (BCSR) arrays
#'
#' Port of `jax.experimental.sparse.BCSR`: `list(data, indices, indptr)` plus a shape
#' with exactly two sparse dimensions after the batch dimensions. Class methods:
#' `experimental_sparse_BCSR$fromdense()`, `$from_bcoo()`. Properties and methods:
#' `$data`, `$indices`, `$indptr`, `$shape`, `$dtype`, `$nse`, `$n_batch`, `$n_sparse`,
#' `$n_dense`, `$todense()`, `$to_bcoo()`, `$sum_duplicates()`. `%*%` and `[` go
#' through [experimental_sparse_sparsify()].
#' @param args `list(data, indices, indptr)`.
#' @param shape Shape.
#' @param indices_sorted,unique_indices Flags.
#' @return A BCSR array (class `jax_bcsr`).
#' @export
experimental_sparse_BCSR <- sp_class(function(args, shape, indices_sorted = FALSE, unique_indices = FALSE) {
  data <- as_jax(args[[1L]]); indices <- as_jax(args[[2L]]); indptr <- as_jax(args[[3L]])
  bs_validate(data, indices, indptr, shape)
  bs_new(data, indices, indptr, shape, indices_sorted, unique_indices)
}, list(
  fromdense = function(mat, nse = NULL, index_dtype = "int32", n_dense = 0L, n_batch = 0L)
    experimental_sparse_bcsr_fromdense(mat, nse = nse, n_batch = n_batch, n_dense = n_dense, index_dtype = index_dtype),
  from_bcoo = bs_from_bcoo,
  from_scipy_sparse = function(mat, index_dtype = NULL, n_dense = 0L, n_batch = 0L) {
    m <- sp_from_matrix(mat, "RsparseMatrix", "BCSR", n_dense, n_batch)
    idt <- index_dtype %||% default_int()
    experimental_sparse_BCSR(list(jnp_asarray(m$x), as_jax(array(m$j, length(m$j)), idt),
                                  as_jax(array(m$p, length(m$p)), idt)), shape = m$shape)
  },
  `_empty` = bs_empty
), "BCSR")

#' @export
`$.jax_bcsr` <- function(x, name) {
  f <- function(n) .subset2(x, n)
  switch(name,
    data = , indices = , indptr = , shape = , indices_sorted = , unique_indices = f(name),
    dtype = sp_dtype(f("data")),
    nse = { s <- sp_shape(f("indices")); s[length(s)] },
    n_batch = sp_ndim(f("indices")) - 1L,
    n_sparse = 2L,
    n_dense = sp_ndim(f("data")) - sp_ndim(f("indices")),
    ndim = length(f("shape")),
    size = prod_int(f("shape")),
    T = jax_error("Transpose is not implemented.", class = "jax_not_implemented_error"),
    transpose = function(...) jax_error("Transpose is not implemented.", class = "jax_not_implemented_error"),
    `_info` = bc_info(x),
    `_bufs` = list(f("data"), f("indices"), f("indptr")),
    todense = function() experimental_sparse_bcsr_todense(x),
    to_bcoo = function() bs_to_bcoo_mat(x),
    sum_duplicates = function(nse = NULL, remove_zeros = TRUE) {
      b <- bs_to_bcoo_mat(x)
      bs_from_bcoo(if (remove_zeros) bc_eliminate_zeros(b, nse) else experimental_sparse_bcoo_sum_duplicates(b, nse))
    },
    block_until_ready = function() { block_until_ready(list(f("data"), f("indices"), f("indptr"))); x },
    NULL)
}

register_pytree_node("jax_bcsr",
  function(x) list(children = list(.subset2(x, "data"), .subset2(x, "indices"), .subset2(x, "indptr")),
                   aux = list(shape = .subset2(x, "shape"), indices_sorted = .subset2(x, "indices_sorted"),
                              unique_indices = .subset2(x, "unique_indices"))),
  function(aux, children) bs_new(children[[1L]], children[[2L]], children[[3L]], aux$shape, aux$indices_sorted,
                                 aux$unique_indices))

#' @export
print.jax_bcsr <- function(x, ...) {
  extra <- paste0(", nse=", x$nse, if (x$n_batch) paste0(", n_batch=", x$n_batch),
                  if (x$n_dense) paste0(", n_dense=", x$n_dense))
  cat("BCSR(", x$dtype, "[", paste(x$shape, collapse = ", "), "]", extra, ")\n", sep = "")
  invisible(x)
}

# ---- COO ----------------------------------------------------------------------------------

coo_todense_p <- new_primitive("coo_todense")
coo_todense_p$abstract_eval <- function(avals, params) new_aval(params$spinfo$shape, avals[[1L]]$dtype)
coo_todense_p$lowering <- lower_fun(function(args, params) {
  at_add(lax_full(params$spinfo$shape, 0, sp_dtype(args[[1L]])), args[[2L]], args[[3L]], value = args[[1L]])
})
defjvp(coo_todense_p, function(t, primals, params) co_todense(t, primals[[2L]], primals[[3L]], params$spinfo), NULL, NULL)
coo_todense_p$transpose <- function(ct, args, params) {
  if (is_undefined(args[[2L]]) || is_undefined(args[[3L]]))
    jax_error("Cannot transpose with respect to sparse indices", class = "jax_value_error")
  list(sp_coo_extract(args[[2L]], args[[3L]], ct), NULL, NULL)
}
co_todense <- function(data, row, col, spinfo) bind(coo_todense_p, lapply(list(data, row, col), as_jax), list(spinfo = spinfo))
co_info <- function(shape, rows_sorted = FALSE, cols_sorted = FALSE) {
  list(shape = as.integer(shape), rows_sorted = isTRUE(rows_sorted), cols_sorted = isTRUE(cols_sorted))
}

coo_fromdense_p <- new_primitive("coo_fromdense", multiple_results = TRUE)
coo_fromdense_p$abstract_eval <- function(avals, params) {
  list(new_aval(params$nse, sp_zero_where_dtype(avals[[1L]]$dtype)), new_aval(params$nse, params$index_dtype),
       new_aval(params$nse, params$index_dtype))
}
coo_fromdense_p$lowering <- lower_fun(function(args, params) {
  mat <- args[[1L]]; nse <- params$nse
  rc <- jnp_nonzero(mat, size = nse)
  data <- at_get(mat, rc[[1L]], rc[[2L]])
  true_nz <- jnp_less(jnp_arange(nse), jnp_sum(lax_ne(mat, lax_full_like(mat, 0))))
  list(sp_zero_where(true_nz, data), jnp_astype(rc[[1L]], params$index_dtype), jnp_astype(rc[[2L]], params$index_dtype))
}, multiple_results = TRUE)
coo_fromdense_p$jvp <- function(primals, tangents, params) {
  out <- bind(coo_fromdense_p, primals, params)
  t <- tangents[[1L]]
  dd <- if (is_zero(t)) sp_zero_tangent(out[[1L]]) else sp_coo_extract(out[[2L]], out[[3L]], t)
  list(out, list(dd, sp_zero_tangent(out[[2L]]), sp_zero_tangent(out[[3L]])))
}
coo_fromdense_p$transpose <- function(cts, args, params) {
  if (is_zero(cts[[2L]]) || is_zero(cts[[3L]]))
    jax_error("Cannot transpose with respect to sparse indices", class = "jax_value_error")
  list(co_todense(instantiate_zero(cts[[1L]]), cts[[2L]], cts[[3L]], co_info(args[[1L]]$aval$shape)))
}

# The assertions of the COO and CSR product abstract evaluation rules (coo.py and
# csr.py): operand ranks, buffer shapes, dtypes and the contracted size.
sp_matmul_check <- function(name, avals, shape, transpose, other_ndim, csr = FALSE) {
  d <- avals[[1L]]; i <- avals[[2L]]; p <- avals[[3L]]; o <- avals[[4L]]
  ok <- length(shape) == 2L && length(o$shape) == other_ndim && length(d$shape) == 1L &&
    identical(d$shape, i$shape) && d$dtype == o$dtype && i$dtype == p$dtype &&
    o$shape[1L] == (if (transpose) shape[1L] else shape[2L]) &&
    (if (csr) length(p$shape) == 1L && p$shape == shape[1L] + 1L else identical(d$shape, p$shape))
  if (!isTRUE(ok))
    jax_error(name, ": incompatible operands for a sparse matrix of shape ", py_tuple(shape),
              if (transpose) " (transposed)", ": buffers ", py_tuple(d$shape), " ", d$dtype, ", ",
              py_tuple(i$shape), ", ", py_tuple(p$shape), " and operand ", py_tuple(o$shape), " ",
              o$dtype, class = "jax_assertion_error")
}

co_matvec_impl <- function(args, params) {
  data <- args[[1L]]; row <- args[[2L]]; col <- args[[3L]]; v <- args[[4L]]
  if (params$transpose) { tmp <- row; row <- col; col <- tmp }
  out_n <- if (params$transpose) params$spinfo$shape[2L] else params$spinfo$shape[1L]
  dv <- jnp_multiply(data, at_get(v, col))
  at_add(lax_full(out_n, 0, sp_dtype(dv)), row, value = dv)
}
coo_matvec_p <- new_primitive("coo_matvec")
coo_matvec_p$abstract_eval <- function(avals, params) {
  sp_matmul_check("coo_matvec", avals, params$spinfo$shape, params$transpose, 1L)
  out_n <- if (params$transpose) params$spinfo$shape[2L] else params$spinfo$shape[1L]
  new_aval(out_n, avals[[1L]]$dtype)
}
coo_matvec_p$lowering <- lower_fun(co_matvec_impl)
defjvp(coo_matvec_p,
       function(t, primals, params) bind(coo_matvec_p, list(t, primals[[2L]], primals[[3L]], primals[[4L]]), params),
       NULL, NULL,
       function(t, primals, params) bind(coo_matvec_p, list(primals[[1L]], primals[[2L]], primals[[3L]], t), params))
coo_matvec_p$transpose <- function(ct, args, params) {
  data <- args[[1L]]; row <- args[[2L]]; col <- args[[3L]]; v <- args[[4L]]
  ct <- instantiate_zero(ct)
  if (is_undefined(v)) {
    p2 <- params; p2$transpose <- !params$transpose
    return(list(NULL, NULL, NULL, bind(coo_matvec_p, list(data, row, col, ct), p2)))
  }
  d <- if (params$transpose) jnp_multiply(at_get(ct, col), at_get(v, row)) else jnp_multiply(at_get(ct, row), at_get(v, col))
  list(d, NULL, NULL, NULL)
}

co_matmat_impl <- function(args, params) {
  data <- args[[1L]]; row <- args[[2L]]; col <- args[[3L]]; B <- args[[4L]]
  if (params$transpose) { tmp <- row; row <- col; col <- tmp }
  out_n <- if (params$transpose) params$spinfo$shape[2L] else params$spinfo$shape[1L]
  dB <- jnp_multiply(lax_expand_dims(data, 1L), at_get(B, col))
  at_add(lax_full(c(out_n, sp_shape(B)[2L]), 0, sp_dtype(dB)), row, value = dB)
}
coo_matmat_p <- new_primitive("coo_matmat")
coo_matmat_p$abstract_eval <- function(avals, params) {
  sp_matmul_check("coo_matmat", avals, params$spinfo$shape, params$transpose, 2L)
  out_n <- if (params$transpose) params$spinfo$shape[2L] else params$spinfo$shape[1L]
  new_aval(c(out_n, avals[[4L]]$shape[2L]), avals[[1L]]$dtype)
}
coo_matmat_p$lowering <- lower_fun(co_matmat_impl)
defjvp(coo_matmat_p,
       function(t, primals, params) bind(coo_matmat_p, list(t, primals[[2L]], primals[[3L]], primals[[4L]]), params),
       NULL, NULL,
       function(t, primals, params) bind(coo_matmat_p, list(primals[[1L]], primals[[2L]], primals[[3L]], t), params))
coo_matmat_p$transpose <- function(ct, args, params) {
  data <- args[[1L]]; row <- args[[2L]]; col <- args[[3L]]; B <- args[[4L]]
  ct <- instantiate_zero(ct)
  if (is_undefined(B)) {
    p2 <- params; p2$transpose <- !params$transpose
    return(list(NULL, NULL, NULL, bind(coo_matmat_p, list(data, row, col, ct), p2)))
  }
  prod <- if (params$transpose) jnp_multiply(at_get(ct, col), at_get(B, row)) else jnp_multiply(at_get(ct, row), at_get(B, col))
  list(jnp_sum(prod, axis = 1L), NULL, NULL, NULL)
}

co_promote <- function(mat, other) {
  if (inherits(other, "jax_sparse")) jax_error("matmul between two sparse objects.", class = "jax_not_implemented_error")
  promote_dtypes(list(mat$data, other))
}

#' Functions on COO and CSR matrices
#'
#' Ports of `jax.experimental.sparse.coo_fromdense`, `coo_todense`, `coo_matvec`,
#' `coo_matmat`, `csr_fromdense`, `csr_todense`, `csr_matvec` and `csr_matmat` (two
#' dimensional matrices only).
#' @param mat A dense matrix (`*_fromdense`) or a COO/CSR matrix.
#' @param nse Number of stored elements (default: the number of nonzeros).
#' @param index_dtype Index dtype.
#' @param v A vector.
#' @param B A matrix.
#' @param transpose Multiply by the transpose of `mat`.
#' @return A COO/CSR matrix or a dense array.
#' @name experimental_sparse_coo_csr
NULL

#' @rdname experimental_sparse_coo_csr
#' @export
experimental_sparse_coo_fromdense <- function(mat, nse = NULL, index_dtype = "int32") {
  mat <- as_jax(mat)
  if (is.null(nse)) nse <- jnp_sum(lax_ne(mat, lax_full_like(mat, 0)))
  nse <- sp_static_int(nse, "coo_fromdense nse argument")
  bufs <- bind(coo_fromdense_p, list(mat), list(nse = nse, index_dtype = dtype(index_dtype)))
  experimental_sparse_COO(bufs, shape = sp_shape(mat), rows_sorted = TRUE)
}
#' @rdname experimental_sparse_coo_csr
#' @export
experimental_sparse_coo_todense <- function(mat) co_todense(mat$data, mat$row, mat$col, mat$`_info`)
#' @rdname experimental_sparse_coo_csr
#' @export
experimental_sparse_coo_matvec <- function(mat, v, transpose = FALSE) {
  bind(coo_matvec_p, lapply(list(mat$data, mat$row, mat$col, v), as_jax),
       list(spinfo = mat$`_info`, transpose = isTRUE(transpose)))
}
#' @rdname experimental_sparse_coo_csr
#' @export
experimental_sparse_coo_matmat <- function(mat, B, transpose = FALSE) {
  bind(coo_matmat_p, lapply(list(mat$data, mat$row, mat$col, B), as_jax),
       list(spinfo = mat$`_info`, transpose = isTRUE(transpose)))
}

co_new <- function(data, row, col, shape, rows_sorted = FALSE, cols_sorted = FALSE) {
  structure(list(data = data, row = row, col = col, shape = as.integer(shape), rows_sorted = isTRUE(rows_sorted),
                 cols_sorted = isTRUE(cols_sorted)), class = c("jax_coo", "jax_sparse"))
}

co_eye <- function(N, M, k, dtype = NULL, index_dtype = "int32") {
  diag_size <- if (k > 0L) min(N, M - k) else min(N + k, M)
  if (diag_size <= 0L) return(co_empty(c(N, M), dtype, index_dtype))
  idx <- jnp_arange(diag_size, dtype = dtype(index_dtype))
  co_new(jnp_ones(diag_size, dtype), jnp_subtract(idx, if (k >= 0L) 0L else k), jnp_add(idx, if (k <= 0L) 0L else k),
         c(N, M), TRUE, TRUE)
}
co_empty <- function(shape, dtype = NULL, index_dtype = "int32") {
  if (length(shape) != 2L) jax_error("COO must have ndim=2; got shape=", sp_tuple(shape), class = "jax_value_error")
  idt <- dtype(index_dtype)
  co_new(jnp_zeros(0L, dtype), lax_full(0L, 0, idt), lax_full(0L, 0, idt), shape, TRUE, TRUE)
}

#' Coordinate (COO) sparse matrices
#'
#' Port of `jax.experimental.sparse.COO`: `list(data, row, col)` and a two dimensional
#' shape. `experimental_sparse_COO$fromdense(mat, nse, index_dtype)` builds one from a
#' dense matrix; `$todense()`, `$transpose()` (or `$T`) and `%*%` work as in JAX.
#' @param args `list(data, row, col)`.
#' @param shape Shape.
#' @param rows_sorted,cols_sorted Sortedness flags.
#' @return A COO matrix (class `jax_coo`).
#' @export
experimental_sparse_COO <- sp_class(function(args, shape, rows_sorted = FALSE, cols_sorted = FALSE) {
  co_new(as_jax(args[[1L]]), as_jax(args[[2L]]), as_jax(args[[3L]]), shape, rows_sorted, cols_sorted)
}, list(
  fromdense = function(mat, nse = NULL, index_dtype = "int32") experimental_sparse_coo_fromdense(mat, nse, index_dtype),
  `_empty` = co_empty,
  `_eye` = co_eye
), "COO")

#' @export
`$.jax_coo` <- function(x, name) {
  f <- function(n) .subset2(x, n)
  switch(name,
    data = , row = , col = , shape = f(name),
    `_rows_sorted` = f("rows_sorted"), `_cols_sorted` = f("cols_sorted"),
    dtype = sp_dtype(f("data")),
    nse = prod_int(sp_shape(f("data"))),
    ndim = length(f("shape")), size = prod_int(f("shape")),
    `_info` = co_info(f("shape"), f("rows_sorted"), f("cols_sorted")),
    `_bufs` = list(f("data"), f("row"), f("col")),
    T = co_new(f("data"), f("col"), f("row"), rev(f("shape")), f("cols_sorted"), f("rows_sorted")),
    transpose = function(axes = NULL) {
      if (!is.null(axes)) jax_error("axes argument to transpose()", class = "jax_not_implemented_error")
      x$T
    },
    todense = function() experimental_sparse_coo_todense(x),
    `_sort_indices` = function() {
      if (f("rows_sorted")) return(x)
      s <- lax_sort(list(f("row"), f("col"), f("data")), num_keys = 2L)
      co_new(s[[3L]], s[[1L]], s[[2L]], f("shape"), TRUE)
    },
    block_until_ready = function() { block_until_ready(list(f("data"), f("row"), f("col"))); x },
    NULL)
}

register_pytree_node("jax_coo",
  function(x) list(children = list(.subset2(x, "data"), .subset2(x, "row"), .subset2(x, "col")),
                   aux = list(shape = .subset2(x, "shape"), rows_sorted = .subset2(x, "rows_sorted"),
                              cols_sorted = .subset2(x, "cols_sorted"))),
  function(aux, children) co_new(children[[1L]], children[[2L]], children[[3L]], aux$shape, aux$rows_sorted,
                                 aux$cols_sorted))

#' @export
print.jax_coo <- function(x, ...) {
  cat("COO(", x$dtype, "[", paste(x$shape, collapse = ", "), "], nse=", x$nse, ")\n", sep = "")
  invisible(x)
}

co_matmul <- function(mat, other) {
  dv <- co_promote(mat, other)
  m <- co_new(dv[[1L]], mat$row, mat$col, mat$shape, mat$`_rows_sorted`, mat$`_cols_sorted`)
  nd <- sp_ndim(dv[[2L]])
  if (nd == 1L) return(experimental_sparse_coo_matvec(m, dv[[2L]]))
  if (nd == 2L) return(experimental_sparse_coo_matmat(m, dv[[2L]]))
  jax_error("matmul with object of shape ", sp_tuple(sp_shape(dv[[2L]])), class = "jax_not_implemented_error")
}

# ---- CSR and CSC --------------------------------------------------------------------------

csr_todense_p <- new_primitive("csr_todense")
csr_todense_p$abstract_eval <- function(avals, params) new_aval(params$shape, avals[[1L]]$dtype)
csr_todense_p$lowering <- lower_fun(function(args, params) {
  rc <- sp_csr_to_coo(args[[2L]], args[[3L]])
  co_todense(args[[1L]], rc[[1L]], rc[[2L]], co_info(params$shape))
})
defjvp(csr_todense_p, function(t, primals, params) cs_todense(t, primals[[2L]], primals[[3L]], params$shape), NULL, NULL)
csr_todense_p$transpose <- function(ct, args, params) {
  if (is_undefined(args[[2L]]) || is_undefined(args[[3L]]))
    jax_error("Cannot transpose with respect to sparse indices", class = "jax_value_error")
  list(sp_csr_extract(args[[2L]], args[[3L]], ct), NULL, NULL)
}
cs_todense <- function(data, indices, indptr, shape) {
  bind(csr_todense_p, lapply(list(data, indices, indptr), as_jax), list(shape = as.integer(shape)))
}

csr_fromdense_p <- new_primitive("csr_fromdense", multiple_results = TRUE)
csr_fromdense_p$abstract_eval <- function(avals, params) {
  list(new_aval(params$nse, sp_zero_where_dtype(avals[[1L]]$dtype)), new_aval(params$nse, params$index_dtype),
       new_aval(avals[[1L]]$shape[1L] + 1L, params$index_dtype))
}
csr_fromdense_p$lowering <- lower_fun(function(args, params) {
  mat <- args[[1L]]; nse <- params$nse; idt <- params$index_dtype
  m <- sp_shape(mat)[1L]
  rc <- jnp_nonzero(mat, size = nse)
  data <- at_get(mat, rc[[1L]], rc[[2L]])
  true_nz <- jnp_less(jnp_arange(nse), jnp_sum(lax_ne(mat, lax_full_like(mat, 0))))
  data <- sp_zero_where(true_nz, data)
  row <- jnp_where(true_nz, rc[[1L]], m)
  counts <- jnp_cumsum(jnp_astype(jnp_bincount(row, length = m), idt))
  list(data, jnp_astype(rc[[2L]], idt), lax_concatenate(list(lax_full(1L, 0, idt), counts), 0L))
}, multiple_results = TRUE)
csr_fromdense_p$jvp <- function(primals, tangents, params) {
  out <- bind(csr_fromdense_p, primals, params)
  t <- tangents[[1L]]
  dd <- if (is_zero(t)) sp_zero_tangent(out[[1L]]) else sp_csr_extract(out[[2L]], out[[3L]], t)
  list(out, list(dd, sp_zero_tangent(out[[2L]]), sp_zero_tangent(out[[3L]])))
}
csr_fromdense_p$transpose <- function(cts, args, params) {
  if (is_zero(cts[[2L]]) || is_zero(cts[[3L]]))
    jax_error("Cannot transpose with respect to sparse indices", class = "jax_value_error")
  list(cs_todense(instantiate_zero(cts[[1L]]), cts[[2L]], cts[[3L]], args[[1L]]$aval$shape))
}

cs_via_coo <- function(prim_impl) {
  function(args, params) {
    rc <- sp_csr_to_coo(args[[2L]], args[[3L]])
    prim_impl(list(args[[1L]], rc[[1L]], rc[[2L]], args[[4L]]),
              list(spinfo = co_info(params$shape), transpose = params$transpose))
  }
}

csr_matvec_p <- new_primitive("csr_matvec")
csr_matvec_p$abstract_eval <- function(avals, params) {
  sp_matmul_check("csr_matvec", avals, params$shape, params$transpose, 1L, csr = TRUE)
  new_aval(if (params$transpose) params$shape[2L] else params$shape[1L], avals[[1L]]$dtype)
}
csr_matvec_p$lowering <- lower_fun(cs_via_coo(co_matvec_impl))
defjvp(csr_matvec_p,
       function(t, primals, params) bind(csr_matvec_p, list(t, primals[[2L]], primals[[3L]], primals[[4L]]), params),
       NULL, NULL,
       function(t, primals, params) bind(csr_matvec_p, list(primals[[1L]], primals[[2L]], primals[[3L]], t), params))
csr_matvec_p$transpose <- function(ct, args, params) {
  data <- args[[1L]]; indices <- args[[2L]]; indptr <- args[[3L]]; v <- args[[4L]]
  ct <- instantiate_zero(ct)
  if (is_undefined(v)) {
    p2 <- params; p2$transpose <- !params$transpose
    return(list(NULL, NULL, NULL, bind(csr_matvec_p, list(data, indices, indptr, ct), p2)))
  }
  rc <- sp_csr_to_coo(indices, indptr)
  d <- if (params$transpose) jnp_multiply(at_get(ct, rc[[2L]]), at_get(v, rc[[1L]]))
       else jnp_multiply(at_get(ct, rc[[1L]]), at_get(v, rc[[2L]]))
  list(d, NULL, NULL, NULL)
}

csr_matmat_p <- new_primitive("csr_matmat")
csr_matmat_p$abstract_eval <- function(avals, params) {
  sp_matmul_check("csr_matmat", avals, params$shape, params$transpose, 2L, csr = TRUE)
  new_aval(c(if (params$transpose) params$shape[2L] else params$shape[1L], avals[[4L]]$shape[2L]), avals[[1L]]$dtype)
}
csr_matmat_p$lowering <- lower_fun(cs_via_coo(co_matmat_impl))
defjvp(csr_matmat_p,
       function(t, primals, params) bind(csr_matmat_p, list(t, primals[[2L]], primals[[3L]], primals[[4L]]), params),
       NULL, NULL,
       function(t, primals, params) bind(csr_matmat_p, list(primals[[1L]], primals[[2L]], primals[[3L]], t), params))
csr_matmat_p$transpose <- function(ct, args, params) {
  data <- args[[1L]]; indices <- args[[2L]]; indptr <- args[[3L]]; B <- args[[4L]]
  ct <- instantiate_zero(ct)
  if (is_undefined(B)) {
    p2 <- params; p2$transpose <- !params$transpose
    return(list(NULL, NULL, NULL, bind(csr_matmat_p, list(data, indices, indptr, ct), p2)))
  }
  rc <- sp_csr_to_coo(indices, indptr)
  prod <- if (params$transpose) jnp_multiply(at_get(ct, rc[[2L]]), at_get(B, rc[[1L]]))
          else jnp_multiply(at_get(ct, rc[[1L]]), at_get(B, rc[[2L]]))
  list(jnp_sum(prod, axis = 1L), NULL, NULL, NULL)
}

cs_matvec <- function(data, indices, indptr, v, shape, transpose = FALSE) {
  bind(csr_matvec_p, lapply(list(data, indices, indptr, v), as_jax),
       list(shape = as.integer(shape), transpose = isTRUE(transpose)))
}
cs_matmat <- function(data, indices, indptr, B, shape, transpose = FALSE) {
  bind(csr_matmat_p, lapply(list(data, indices, indptr, B), as_jax),
       list(shape = as.integer(shape), transpose = isTRUE(transpose)))
}

#' @rdname experimental_sparse_coo_csr
#' @export
experimental_sparse_csr_fromdense <- function(mat, nse = NULL, index_dtype = "int32") {
  mat <- as_jax(mat)
  if (is.null(nse)) nse <- jnp_sum(lax_ne(mat, lax_full_like(mat, 0)))
  nse <- sp_static_int(nse, "coo_fromdense nse argument")
  bufs <- bind(csr_fromdense_p, list(mat), list(nse = nse, index_dtype = dtype(index_dtype)))
  experimental_sparse_CSR(bufs, shape = sp_shape(mat))
}
#' @rdname experimental_sparse_coo_csr
#' @export
experimental_sparse_csr_todense <- function(mat) cs_todense(mat$data, mat$indices, mat$indptr, mat$shape)
#' @rdname experimental_sparse_coo_csr
#' @export
experimental_sparse_csr_matvec <- function(mat, v, transpose = FALSE) {
  cs_matvec(mat$data, mat$indices, mat$indptr, v, mat$shape, transpose)
}
#' @rdname experimental_sparse_coo_csr
#' @export
experimental_sparse_csr_matmat <- function(mat, B, transpose = FALSE) {
  cs_matmat(mat$data, mat$indices, mat$indptr, B, mat$shape, transpose)
}

cs_new <- function(data, indices, indptr, shape, cls) {
  structure(list(data = data, indices = indices, indptr = indptr, shape = as.integer(shape)),
            class = c(cls, "jax_sparse"))
}

cs_eye <- function(N, M, k, dtype = NULL, index_dtype = "int32") {
  diag_size <- if (k > 0L) min(N, M - k) else min(N + k, M)
  idt <- dtype(index_dtype)
  if (diag_size <= 0L) return(cs_empty(c(N, M), dtype, index_dtype))
  idx <- jnp_arange(diag_size, dtype = idt)
  col <- jnp_add(idx, if (k <= 0L) 0L else k)
  row <- jnp_subtract(idx, if (k >= 0L) 0L else k)
  counts <- jnp_cumsum(jnp_astype(jnp_bincount(row, length = N), idt))
  cs_new(jnp_ones(diag_size, dtype), jnp_astype(col, idt), lax_concatenate(list(lax_full(1L, 0, idt), counts), 0L),
         c(N, M), "jax_csr")
}
cs_empty <- function(shape, dtype = NULL, index_dtype = "int32", which = 1L, cls = "jax_csr") {
  if (length(shape) != 2L) jax_error(if (cls == "jax_csr") "CSR" else "CSC", " must have ndim=2; got shape=",
                                     sp_tuple(shape), class = "jax_value_error")
  idt <- dtype(index_dtype)
  cs_new(jnp_zeros(0L, dtype), lax_full(0L, 0, idt), lax_full(shape[which] + 1L, 0, idt), shape, cls)
}

#' Compressed sparse row and column matrices
#'
#' Ports of `jax.experimental.sparse.CSR` and `CSC`: `list(data, indices, indptr)` and a
#' two dimensional shape. Class methods `$fromdense()`; properties and methods `$data`,
#' `$indices`, `$indptr`, `$shape`, `$nse`, `$dtype`, `$todense()`, `$transpose()` (or
#' `$T`, which turns a CSR into a CSC and back) and `%*%` with dense operands.
#' @param args `list(data, indices, indptr)`.
#' @param shape Shape.
#' @return A CSR (class `jax_csr`) or CSC (class `jax_csc`) matrix.
#' @export
experimental_sparse_CSR <- sp_class(function(args, shape) {
  cs_new(as_jax(args[[1L]]), as_jax(args[[2L]]), as_jax(args[[3L]]), shape, "jax_csr")
}, list(
  fromdense = function(mat, nse = NULL, index_dtype = "int32") experimental_sparse_csr_fromdense(mat, nse, index_dtype),
  `_empty` = function(shape, dtype = NULL, index_dtype = "int32") cs_empty(shape, dtype, index_dtype, 1L, "jax_csr"),
  `_eye` = cs_eye
), "CSR")

#' @rdname experimental_sparse_CSR
#' @export
experimental_sparse_CSC <- sp_class(function(args, shape) {
  cs_new(as_jax(args[[1L]]), as_jax(args[[2L]]), as_jax(args[[3L]]), shape, "jax_csc")
}, list(
  fromdense = function(mat, nse = NULL, index_dtype = "int32") {
    experimental_sparse_csr_fromdense(jnp_transpose(as_jax(mat)), nse, index_dtype)$T
  },
  `_empty` = function(shape, dtype = NULL, index_dtype = "int32") cs_empty(shape, dtype, index_dtype, 2L, "jax_csc"),
  `_eye` = function(N, M, k, dtype = NULL, index_dtype = "int32") cs_eye(M, N, -k, dtype, index_dtype)$T
), "CSC")

#' @export
`$.jax_csr` <- function(x, name) cs_dollar(x, name, "jax_csc")
#' @export
`$.jax_csc` <- function(x, name) cs_dollar(x, name, "jax_csr")

cs_dollar <- function(x, name, other) {
  f <- function(n) .subset2(x, n)
  is_csr <- inherits(x, "jax_csr")
  switch(name,
    data = , indices = , indptr = , shape = f(name),
    dtype = sp_dtype(f("data")),
    nse = prod_int(sp_shape(f("data"))),
    ndim = length(f("shape")), size = prod_int(f("shape")),
    `_bufs` = list(f("data"), f("indices"), f("indptr")),
    T = cs_new(f("data"), f("indices"), f("indptr"), rev(f("shape")), other),
    transpose = function(axes = NULL) {
      if (!is.null(axes)) jax_error("axes argument to transpose()", class = "jax_not_implemented_error")
      x$T
    },
    todense = function() if (is_csr) experimental_sparse_csr_todense(x) else jnp_transpose(experimental_sparse_csr_todense(x$T)),
    block_until_ready = function() { block_until_ready(list(f("data"), f("indices"), f("indptr"))); x },
    NULL)
}

for (.cls in c("jax_csr", "jax_csc")) local({
  cls <- .cls
  register_pytree_node(cls,
    function(x) list(children = list(.subset2(x, "data"), .subset2(x, "indices"), .subset2(x, "indptr")),
                     aux = list(shape = .subset2(x, "shape"))),
    function(aux, children) cs_new(children[[1L]], children[[2L]], children[[3L]], aux$shape, cls))
})
rm(.cls)

#' @export
print.jax_csr <- function(x, ...) {
  cat("CSR(", x$dtype, "[", paste(x$shape, collapse = ", "), "], nse=", x$nse, ")\n", sep = "")
  invisible(x)
}
#' @export
print.jax_csc <- function(x, ...) {
  cat("CSC(", x$dtype, "[", paste(x$shape, collapse = ", "), "], nse=", x$nse, ")\n", sep = "")
  invisible(x)
}

cs_matmul <- function(mat, other) {
  dv <- co_promote(mat, other)
  nd <- sp_ndim(dv[[2L]])
  tr <- inherits(mat, "jax_csc")
  shape <- if (tr) rev(mat$shape) else mat$shape
  if (nd == 1L) return(cs_matvec(dv[[1L]], mat$indices, mat$indptr, dv[[2L]], shape, tr))
  if (nd == 2L) return(cs_matmat(dv[[1L]], mat$indices, mat$indptr, dv[[2L]], shape, tr))
  jax_error("matmul with object of shape ", sp_tuple(sp_shape(dv[[2L]])), class = "jax_not_implemented_error")
}
