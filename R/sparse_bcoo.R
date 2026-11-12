# jax.experimental.sparse: the BCOO format (port of jax/experimental/sparse/bcoo.py,
# _base.py and util.py).
#
# A BCOO array of shape `(*batch, *sparse, *dense)` stores `data` of shape
# `(*batch, nse, *dense)` and `indices` of shape `(*batch, nse, n_sparse)`. Indices are
# 0-based; out of bound indices (equal to the dimension size) mark padding. Not ported:
# the cuSPARSE GPU lowerings (jaxR runs on CPU), and the specialized sampled dot product
# kernels (bcoo_dot_general_sampled always extracts from the dense product).

# ---- small helpers --------------------------------------------------------------------

# sp_shape() and sp_dtype() are defined in R/lax_special.R.
sp_ndim <- function(x) length(sp_shape(x))
sp_pos <- function(x) seq_len(x) - 1L
sp_remaining <- function(original, ...) {
  drop <- unlist(list(...))
  original[!original %in% drop]
}
sp_spinfo <- function(shape, indices_sorted = FALSE, unique_indices = FALSE) {
  list(shape = as.integer(shape), indices_sorted = isTRUE(indices_sorted),
       unique_indices = isTRUE(unique_indices))
}
sp_split <- function(x, sizes) {
  out <- list()
  start <- 0L
  for (s in sizes) {
    out[[length(out) + 1L]] <- x[start + seq_len(s)]
    start <- start + s
  }
  out[[length(out) + 1L]] <- x[seq_along(x) > start]
  out
}
sp_tuple <- function(x) py_tuple(as.integer(x))
sp_static_int <- function(x, what = "this value") as.integer(as.vector(concrete_or_error(x, what)))
# jnp.where(mask, data, 0): the zero is a weak integer as in Python (an R `0` is a weak
# float and would promote integer data), so boolean data become integers.
sp_zero_where <- function(mask, data) jnp_where(mask, data, 0L)
sp_zero_where_dtype <- function(dt) if (dt == "bool") default_int() else dt
sp_zero_tangent <- function(x) zero_of(tangent_aval(aval_of(x)))
sp_index_last <- function(x, i) lax_index_in_dim(x, i, axis = sp_ndim(x) - 1L, keepdims = FALSE)
sp_take_last <- function(x, idx) {
  if (!length(idx)) return(lax_slice_in_dim(x, 0L, 0L, axis = sp_ndim(x) - 1L))
  lax_concatenate(lapply(idx, function(i) lax_index_in_dim(x, i, axis = sp_ndim(x) - 1L)),
                  sp_ndim(x) - 1L)
}
sp_dims_to_shape <- function(shape, n, pos) {
  s <- rep(1L, n)
  s[pos + 1L] <- shape
  s
}

#' Sparse efficiency conditions
#'
#' Ports of the warning and error classes of `jax.experimental.sparse`. They are the
#' condition classes jaxR signals: `experimental_sparse_SparseEfficiencyError` is the
#' class of errors raised by [experimental_sparse_bcoo_update_layout()] and the warnings
#' carry `experimental_sparse_SparseEfficiencyWarning`. `CuSparseEfficiencyWarning` is
#' never signaled (there is no cuSPARSE path) but kept for compatibility.
#' @format Character scalars naming condition classes.
#' @name experimental_sparse_conditions
NULL

#' @rdname experimental_sparse_conditions
#' @export
experimental_sparse_SparseEfficiencyError <- "jax_sparse_efficiency_error"
#' @rdname experimental_sparse_conditions
#' @export
experimental_sparse_SparseEfficiencyWarning <- "jax_sparse_efficiency_warning"
#' @rdname experimental_sparse_conditions
#' @export
experimental_sparse_CuSparseEfficiencyWarning <- "jax_cusparse_efficiency_warning"

# Port of util.nfold_vmap / broadcasting_vmap. Arguments must be arrays (not trees).
sp_nfold_vmap <- function(fun, N, broadcasted = TRUE, in_axes = 0L) {
  for (i in seq_len(N)) {
    fun <- if (broadcasted) sp_broadcasting_vmap(fun, in_axes) else vmap(fun, in_axes = in_axes)
  }
  fun
}

sp_broadcasting_vmap <- function(fun, in_axes = 0L) {
  force(fun); force(in_axes)
  function(...) {
    args <- lapply(list(...), as_jax)
    axes <- if (is.list(in_axes)) lapply(in_axes, function(a) a) else rep(list(in_axes), length(args))
    sizes <- unlist(Map(function(a, i) if (is.null(i)) NULL else sp_shape(a)[i + 1L], args, axes))
    size <- max(sizes)
    if (size > 1L) {
      for (k in seq_along(args)) {
        i <- axes[[k]]
        if (is.null(i)) next
        s <- sp_shape(args[[k]])[i + 1L]
        if (!s %in% c(1L, size)) jax_error("broadcasting_vmap: mismatched input shapes", class = "jax_value_error")
        if (s == 1L) {
          args[[k]] <- lax_squeeze(args[[k]], i)
          axes[k] <- list(NULL)
        }
      }
    }
    do.call(vmap(fun, in_axes = axes), args)
  }
}

# ---- validation -------------------------------------------------------------------------

bc_validate_indices <- function(indices, shape) {
  is <- sp_shape(indices)
  if (!is_int_dtype(sp_dtype(indices))) jax_error("BCOO indices must have an integer dtype")
  shape <- as.integer(shape)
  r <- length(is)
  nse <- is[r - 1L]; n_sparse <- is[r]
  n_batch <- r - 2L
  n_dense <- length(shape) - n_batch - n_sparse
  if (n_dense < 0L) jax_error("BCOO indices have more sparse dimensions than the shape allows")
  bi <- is[seq_len(n_batch)]
  if (any(!(bi == 1L | bi == shape[seq_len(n_batch)])))
    jax_error("indices batch dimensions not compatible for indices.shape=", sp_tuple(is),
              ", shape=", sp_tuple(shape), class = "jax_value_error")
  list(n_batch = n_batch, n_sparse = n_sparse, n_dense = n_dense, nse = nse)
}

bc_validate <- function(data, indices, shape) {
  p <- bc_validate_indices(indices, shape)
  ds <- sp_shape(data)
  shape <- as.integer(shape)
  nb <- p$n_batch
  db <- ds[seq_len(nb)]
  if (length(ds) < nb || any(!(db == 1L | db == shape[seq_len(nb)])))
    jax_error("data batch dimensions not compatible for data.shape=", sp_tuple(ds), ", shape=",
              sp_tuple(shape), class = "jax_value_error")
  want <- c(p$nse, shape[seq_along(shape) > nb + p$n_sparse])
  if (!identical(as.integer(ds[seq_along(ds) > nb]), as.integer(want)))
    jax_error("Invalid data.shape=", sp_tuple(ds), " for nse=", p$nse, ", n_batch=", nb,
              ", n_dense=", p$n_dense, class = "jax_value_error")
  p
}

# Port of _bcoo_batch_dims_to_front (dims are 0-based, NA when unmapped).
bc_batch_dims_to_front <- function(vals, dims, spinfo, batch_size = NULL) {
  data <- vals[[1L]]; indices <- vals[[2L]]
  n_batch <- sp_ndim(indices) - 2L + as.integer(is.na(dims[2L]))
  if (!all(is.na(dims) | (dims >= 0L & dims < n_batch)))
    jax_error("batch_dims must be None or satisfy 0 < dim < n_batch. Got batch_dims=",
              sp_tuple(dims), " for n_batch=", n_batch, class = "jax_not_implemented_error")
  mv <- function(a, d) if (is.na(d)) lax_expand_dims(a, 0L) else moveaxis(a, d, 0L)
  if (is.null(batch_size)) {
    batch_size <- max(unlist(Map(function(a, d) if (is.na(d)) NULL else sp_shape(a)[d + 1L],
                                 vals[1:2], dims)))
  }
  list(data = mv(data, dims[1L]), indices = mv(indices, dims[2L]),
       spinfo = sp_spinfo(c(batch_size, spinfo$shape), spinfo$indices_sorted, spinfo$unique_indices))
}

# Index arrays for every batch element: `(*batch, nse)` shaped sparse index columns
# and broadcastable batch iotas (the meshgrid of _bcoo_todense_impl).
bc_index_grid <- function(indices, shape, n_batch, n_sparse) {
  is <- sp_shape(indices)
  full <- if (n_batch) jnp_broadcast_to(indices, c(shape[seq_len(n_batch)], is[n_batch + 1:2]))
          else indices
  sparse_ind <- lapply(sp_pos(n_sparse), function(i) sp_index_last(full, i))
  batch_ind <- lapply(sp_pos(n_batch), function(b) {
    lax_broadcasted_iota(sp_dtype(indices), sp_dims_to_shape(shape[b + 1L], n_batch + 1L, b), b)
  })
  list(batch = batch_ind, sparse = sparse_ind)
}

# ---- bcoo_todense -------------------------------------------------------------------------

bcoo_todense_p <- new_primitive("bcoo_todense")

bc_todense_impl <- function(args, params) {
  data <- args[[1L]]; indices <- args[[2L]]
  shape <- params$spinfo$shape
  p <- bc_validate(data, indices, shape)
  g <- bc_index_grid(indices, shape, p$n_batch, p$n_sparse)
  if (!length(g$sparse)) {
    data <- jnp_sum(data, axis = p$n_batch, keepdims = length(g$batch) > 0L, dtype = sp_dtype(data))
  }
  zeros <- lax_full(shape, 0, sp_dtype(data))
  idx <- c(g$batch, g$sparse)
  if (!length(idx)) return(lax_add(zeros, lax_broadcast_like(data, zeros)))
  do.call(at_add, c(list(zeros), idx, list(value = data)))
}

bcoo_todense_p$abstract_eval <- function(avals, params) {
  bc_validate(avals[[1L]], avals[[2L]], params$spinfo$shape)
  new_aval(params$spinfo$shape, avals[[1L]]$dtype)
}
bcoo_todense_p$lowering <- lower_fun(bc_todense_impl)
defjvp(bcoo_todense_p, function(t, primals, params) bc_todense(t, primals[[2L]], params$spinfo), NULL)
bcoo_todense_p$transpose <- function(ct, args, params) {
  if (is_undefined(args[[2L]])) jax_error("Cannot transpose with respect to sparse indices",
                                          class = "jax_value_error")
  list(bc_extract(args[[2L]], ct), NULL)
}
bcoo_todense_p$batch <- function(vals, dims, params, axis) {
  b <- bc_batch_dims_to_front(vals, dims, params$spinfo)
  list(bc_todense(b$data, b$indices, b$spinfo), 0L)
}

bc_todense <- function(data, indices, spinfo) {
  bind(bcoo_todense_p, list(as_jax(data), as_jax(indices)), list(spinfo = spinfo))
}

# ---- bcoo_fromdense -----------------------------------------------------------------------

bcoo_fromdense_p <- new_primitive("bcoo_fromdense", multiple_results = TRUE)

bc_fromdense_impl <- function(args, params) {
  mat <- args[[1L]]
  nse <- params$nse; nb <- params$n_batch; nd <- params$n_dense
  idt <- params$index_dtype
  ms <- sp_shape(mat)
  ns <- length(ms) - nd - nb
  mask <- lax_ne(mat, lax_full_like(mat, 0))
  if (nd > 0L) mask <- jnp_any(mask, axis = -seq_len(nd))
  if (ns == 0L) {
    indices <- lax_full(c(ms[seq_len(nb)], nse, 0L), 0, idt)
  } else {
    nonzero <- sp_nfold_vmap(function(a) {
      jnp_stack(lapply(jnp_nonzero(a, size = nse, fill_value = as.list(sp_shape(a)[seq_len(ns)])),
                       as_jax, dtype = idt))
    }, nb, broadcasted = FALSE)
    indices <- jnp_moveaxis(nonzero(mask), nb, nb + 1L)
  }
  data <- bc_extract(indices, mat)
  mr <- sp_ndim(mask)
  true_nse <- lax_expand_dims(jnp_sum(mask, axis = if (mr > nb) seq.int(nb, mr - 1L) else integer(0)), nb)
  tnz <- lax_lt(lax_broadcasted_iota(sp_dtype(true_nse), c(rep(1L, nb), nse), nb), true_nse)
  if (nd > 0L) tnz <- lax_expand_dims(tnz, nb + seq_len(nd))
  list(sp_zero_where(tnz, data), indices)
}

bcoo_fromdense_p$abstract_eval <- function(avals, params) {
  ms <- avals[[1L]]$shape
  nb <- params$n_batch; nd <- params$n_dense
  ns <- length(ms) - nb - nd
  list(new_aval(c(ms[seq_len(nb)], params$nse, ms[seq_along(ms) > nb + ns]), sp_zero_where_dtype(avals[[1L]]$dtype)),
       new_aval(c(ms[seq_len(nb)], params$nse, ns), params$index_dtype))
}
bcoo_fromdense_p$lowering <- lower_fun(bc_fromdense_impl, multiple_results = TRUE)
bcoo_fromdense_p$jvp <- function(primals, tangents, params) {
  out <- bind(bcoo_fromdense_p, primals, params)
  t <- tangents[[1L]]
  data_dot <- if (is_zero(t)) sp_zero_tangent(out[[1L]]) else bc_extract(out[[2L]], t)
  list(out, list(data_dot, sp_zero_tangent(out[[2L]])))
}
bcoo_fromdense_p$transpose <- function(cts, args, params) {
  if (is_zero(cts[[2L]])) jax_error("Cannot transpose with respect to sparse indices",
                                    class = "jax_value_error")
  list(bc_todense(instantiate_zero(cts[[1L]]), cts[[2L]], sp_spinfo(args[[1L]]$aval$shape)))
}
bcoo_fromdense_p$batch <- function(vals, dims, params, axis) {
  bdim <- dims[1L]
  if (!(bdim >= 0L && bdim <= params$n_batch))
    jax_error("Expected 0 < bdim <= n_batch; got bdim=", bdim, ", n_batch=", params$n_batch,
              class = "jax_value_error")
  params$n_batch <- params$n_batch + 1L
  list(bind(bcoo_fromdense_p, vals, params), c(bdim, bdim))
}

bc_traced_nse <- paste0("\nThe error arose for the nse argument of bcoo_fromdense. In order for\n",
                        "BCOO.fromdense() to be used in traced/compiled code, you must pass a concrete\n",
                        "value to the nse (number of stored elements) argument.\n")

bc_count_stored <- function(mat, n_batch = 0L, n_dense = 0L) {
  mask <- lax_ne(mat, lax_full_like(mat, 0))
  if (n_dense > 0L) mask <- jnp_any(mask, axis = -seq_len(n_dense))
  r <- sp_ndim(mask)
  per <- jnp_sum(mask, axis = if (r > n_batch) seq.int(n_batch, r - 1L) else integer(0))
  if (!length(sp_shape(per))) return(per)
  if (!prod_int(sp_shape(per))) return(0L)
  jnp_max(per)
}

bc_fromdense <- function(mat, nse, n_batch = 0L, n_dense = 0L, index_dtype = "int32") {
  nse <- sp_static_int(nse, bc_traced_nse)
  bind(bcoo_fromdense_p, list(as_jax(mat)),
       list(nse = nse, n_batch = as.integer(n_batch), n_dense = as.integer(n_dense),
            index_dtype = dtype(index_dtype)))
}

#' Convert between dense arrays and BCOO
#'
#' Ports of `jax.experimental.sparse.bcoo_fromdense` and `bcoo_todense`. `nse` (the
#' number of stored elements) must be concrete under transformations; by default it is
#' the largest number of nonzeros per batch element.
#' @param mat A dense array, or for `bcoo_todense` a BCOO array.
#' @param nse Number of stored elements.
#' @param n_batch,n_dense Number of batch and dense dimensions.
#' @param index_dtype Index dtype.
#' @return A BCOO array or a dense array.
#' @export
#' @examples
#' \dontrun{
#' m <- experimental_sparse_bcoo_fromdense(matrix(c(0, 1, 2, 0), 2))
#' experimental_sparse_bcoo_todense(m)
#' }
experimental_sparse_bcoo_fromdense <- function(mat, nse = NULL, n_batch = 0L, n_dense = 0L,
                                               index_dtype = "int32") {
  mat <- as_jax(mat)
  if (is.null(nse)) nse <- bc_count_stored(mat, n_batch, n_dense)
  nse <- sp_static_int(nse, bc_traced_nse)
  bufs <- bc_fromdense(mat, nse, n_batch, n_dense, index_dtype)
  experimental_sparse_BCOO(bufs, shape = sp_shape(mat), indices_sorted = TRUE, unique_indices = TRUE)
}

#' @rdname experimental_sparse_bcoo_fromdense
#' @export
experimental_sparse_bcoo_todense <- function(mat) bc_todense(mat$data, mat$indices, bc_info(mat))

# ---- bcoo_extract -------------------------------------------------------------------------

bcoo_extract_p <- new_primitive("bcoo_extract")

bc_extract_impl <- function(args, params) {
  indices <- args[[1L]]; arr <- args[[2L]]
  as <- sp_shape(arr)
  props <- bc_validate_indices(indices, as)
  orig <- props
  sort_ind <- NULL
  if (!params$assume_unique) {
    u <- bc_unique_indices(indices, as, return_index = TRUE)
    indices <- u[[1L]]; sort_ind <- u[[2L]]
    props <- bc_validate_indices(indices, as)
  }
  nb <- props$n_batch
  g <- bc_index_grid(indices, as, nb, props$n_sparse)
  idx <- c(g$batch, g$sparse)
  result <- if (!length(idx)) lax_expand_dims(arr, 0L)
            else do.call(at_get, c(list(arr), idx, list(mode = "fill", fill_value = 0)))
  rs <- sp_shape(result)
  if (props$n_sparse == 0L && props$nse != 1L) {
    if (params$assume_unique) {
      rs2 <- rs; rs2[nb + 1L] <- props$nse
      result <- lax_broadcast_in_dim(result, rs2, sp_pos(length(rs)))
    } else {
      rs2 <- rs; rs2[nb + 1L] <- orig$nse
      result <- do.call(at_set, c(list(lax_full(rs2, 0, sp_dtype(result))),
                                  rep(list(sl()), nb), list(sl(NULL, 1L), value = result)))
    }
  }
  if (!params$assume_unique) {
    rs <- sp_shape(result)
    unbatched <- c(orig$nse, rs[seq_along(rs) > nb + 1L])
    f <- function(r, i) at_add(lax_full(unbatched, 0, sp_dtype(r)), i, value = r)
    for (k in seq_len(nb)) f <- vmap(f)
    result <- f(result, sort_ind)
  }
  result
}

bcoo_extract_p$abstract_eval <- function(avals, params) {
  as <- avals[[2L]]$shape
  p <- bc_validate_indices(avals[[1L]], as)
  new_aval(c(as[seq_len(p$n_batch)], p$nse, as[seq_along(as) > length(as) - p$n_dense]),
           avals[[2L]]$dtype)
}
bcoo_extract_p$lowering <- lower_fun(bc_extract_impl)
defjvp(bcoo_extract_p, NULL,
       function(t, primals, params) bc_extract(primals[[1L]], t, params$assume_unique))
bcoo_extract_p$transpose <- function(ct, args, params) {
  if (!params$assume_unique)
    jax_error("transpose of bcoo_extract with assume_unique=False", class = "jax_not_implemented_error")
  if (is_undefined(args[[1L]])) jax_error("Cannot transpose with respect to sparse indices",
                                          class = "jax_value_error")
  list(NULL, bc_todense(ct, args[[1L]], sp_spinfo(args[[2L]]$aval$shape)))
}
bcoo_extract_p$batch <- function(vals, dims, params, axis) {
  indices <- vals[[1L]]; arr <- vals[[2L]]
  if (is.na(dims[1L])) {
    bdim <- dims[2L]
    indices <- lax_expand_dims(indices, bdim)
  } else if (is.na(dims[2L])) {
    bdim <- dims[1L]
    s <- sp_shape(arr)
    rs <- append(s, sp_shape(indices)[bdim + 1L], after = bdim)
    arr <- lax_broadcast_in_dim(arr, rs, sp_pos(length(rs))[-(bdim + 1L)])
  } else {
    if (dims[1L] != dims[2L])
      jax_error("bcoo_extract with unequal batch dimensions.", class = "jax_not_implemented_error")
    bdim <- dims[1L]
  }
  if (bdim >= sp_ndim(indices) - 2L)
    jax_error("batch_dims out of range for indices with n_batch=", sp_ndim(indices) - 2L,
              class = "jax_value_error")
  list(bc_extract(indices, arr, params$assume_unique), bdim)
}

bc_extract <- function(indices, arr, assume_unique = TRUE) {
  bind(bcoo_extract_p, list(as_jax(indices), as_jax(arr)), list(assume_unique = isTRUE(assume_unique)))
}

#' Extract values of a dense array at the indices of a BCOO array
#'
#' Port of `jax.experimental.sparse.bcoo_extract`.
#' @param sparr A BCOO array.
#' @param arr A dense array of the same shape.
#' @param assume_unique Whether the indices are unique (default: `sparr$unique_indices`).
#' @return A BCOO array with the indices of `sparr`.
#' @export
experimental_sparse_bcoo_extract <- function(sparr, arr, assume_unique = NULL) {
  if (!inherits(sparr, "jax_bcoo"))
    jax_error("First argument to bcoo_extract should be a BCOO array.", class = "jax_type_error")
  a <- as_jax(arr)
  if (!identical(sp_shape(a), sparr$shape))
    jax_error("shape mismatch: sparr.shape=", sp_tuple(sparr$shape), " a.shape=", sp_tuple(sp_shape(a)),
              class = "jax_value_error")
  assume_unique <- assume_unique %||% sparr$unique_indices
  data <- bc_extract(sparr$indices, a, assume_unique)
  experimental_sparse_BCOO(list(data, sparr$indices), shape = sparr$shape,
                           indices_sorted = sparr$indices_sorted, unique_indices = sparr$unique_indices)
}

# ---- bcoo_transpose -----------------------------------------------------------------------

bcoo_transpose_p <- new_primitive("bcoo_transpose", multiple_results = TRUE)

bc_validate_permutation <- function(data, indices, permutation, shape) {
  permutation <- as.integer(permutation)
  if (!identical(sort(permutation), sp_pos(length(shape))))
    jax_error("transpose permutation isn't a permutation of operand dimensions, got permutation ",
              sp_tuple(permutation), " for shape ", sp_tuple(shape), class = "jax_type_error")
  p <- bc_validate(data, indices, shape)
  nb <- p$n_batch; ns <- p$n_sparse; nd <- p$n_dense
  batch_perm <- permutation[seq_len(nb)]
  sparse_perm <- permutation[nb + seq_len(ns)] - nb
  dense_perm <- permutation[nb + ns + seq_len(nd)] - ns - nb
  if (nb && !identical(sort(batch_perm), sp_pos(nb)))
    jax_error("transpose permutation cannot permute batch axes with non-batch axes; got permutation ",
              sp_tuple(permutation), ", with n_batch=", nb, ".", class = "jax_not_implemented_error")
  if (nd && !identical(sort(dense_perm), sp_pos(nd)))
    jax_error("transpose permutation cannot permute dense axes with non-dense axes; got permutation ",
              sp_tuple(permutation), ", with n_dense=", nd, ".", class = "jax_not_implemented_error")
  list(batch = batch_perm, sparse = sparse_perm, dense = dense_perm)
}

bc_transpose_impl <- function(args, params) {
  data <- args[[1L]]; indices <- args[[2L]]
  pr <- bc_validate_permutation(data, indices, params$permutation, params$spinfo$shape)
  nb <- length(pr$batch)
  indices <- lax_transpose(sp_take_last(indices, pr$sparse), c(pr$batch, nb, nb + 1L))
  data <- lax_transpose(data, c(pr$batch, nb, pr$dense + nb + 1L))
  list(data, indices)
}

bcoo_transpose_p$abstract_eval <- function(avals, params) {
  pr <- bc_validate_permutation(avals[[1L]], avals[[2L]], params$permutation, params$spinfo$shape)
  nb <- length(pr$batch)
  is <- avals[[2L]]$shape; ds <- avals[[1L]]$shape
  list(new_aval(ds[c(pr$batch, nb, pr$dense + nb + 1L) + 1L], avals[[1L]]$dtype),
       new_aval(is[c(pr$batch, nb, nb + 1L) + 1L], avals[[2L]]$dtype))
}
bcoo_transpose_p$lowering <- lower_fun(bc_transpose_impl, multiple_results = TRUE)
bcoo_transpose_p$jvp <- function(primals, tangents, params) {
  out <- bind(bcoo_transpose_p, primals, params)
  t <- tangents[[1L]]
  dd <- if (is_zero(t)) sp_zero_tangent(out[[1L]])
        else bind(bcoo_transpose_p, list(t, primals[[2L]]), params)[[1L]]
  list(out, list(dd, sp_zero_tangent(out[[2L]])))
}
bcoo_transpose_p$transpose <- function(cts, args, params) {
  if (is_undefined(args[[2L]])) jax_error("Cannot transpose with respect to sparse indices",
                                          class = "jax_value_error")
  perm <- params$permutation
  ct_spinfo <- sp_spinfo(params$spinfo$shape[perm + 1L])
  rev <- order(perm) - 1L
  is <- sp_shape(args[[2L]])
  dummy <- lax_full(c(rep(1L, length(is) - 2L), is[length(is) - 1:0]), 0, "int32")
  data_trans <- bc_transpose(instantiate_zero(cts[[1L]]), dummy, rev, ct_spinfo)[[1L]]
  list(data_trans, NULL)
}
bcoo_transpose_p$batch <- function(vals, dims, params, axis) {
  b <- bc_batch_dims_to_front(vals, dims, params$spinfo)
  out <- bc_transpose(b$data, b$indices, c(0L, params$permutation + 1L), b$spinfo)
  od <- ifelse(is.na(dims), NA_integer_, 0L)
  out <- Map(function(a, d) if (is.na(d)) lax_squeeze(a, 0L) else a, out, od)
  list(out, od)
}

bc_transpose <- function(data, indices, permutation, spinfo) {
  permutation <- as.integer(permutation)
  if (identical(permutation, sp_pos(length(spinfo$shape)))) return(list(as_jax(data), as_jax(indices)))
  bind(bcoo_transpose_p, list(as_jax(data), as_jax(indices)),
       list(permutation = permutation, spinfo = spinfo))
}

#' Transpose a BCOO array
#'
#' Port of `jax.experimental.sparse.bcoo_transpose`: batch, sparse and dense dimensions
#' may only be permuted among themselves.
#' @param mat A BCOO array.
#' @param permutation 0-based permutation.
#' @return A BCOO array.
#' @export
experimental_sparse_bcoo_transpose <- function(mat, permutation) {
  permutation <- as.integer(permutation)
  bufs <- bc_transpose(mat$data, mat$indices, permutation, bc_info(mat))
  experimental_sparse_BCOO(bufs, shape = mat$shape[permutation + 1L], unique_indices = mat$unique_indices)
}

# ---- bcoo_dot_general ---------------------------------------------------------------------

bcoo_dot_general_p <- new_primitive("bcoo_dot_general")

bc_dn <- function(dn) {
  list(list(as.integer(dn[[1L]][[1L]]), as.integer(dn[[1L]][[2L]])),
       list(as.integer(dn[[2L]][[1L]] %||% integer(0)), as.integer(dn[[2L]][[2L]] %||% integer(0))))
}

bc_dot_shape <- function(lhs_aval, rhs_aval, dn, pet = NULL) {
  out <- eval_shape(function(a, b) lax_dot_general(a, b, dn, preferred_element_type = pet),
                    new_aval(lhs_aval$shape, lhs_aval$dtype), new_aval(rhs_aval$shape, rhs_aval$dtype))
  new_aval(out$shape, out$dtype)
}

bc_dot_general_impl <- function(args, params) {
  lhs_data <- args[[1L]]; lhs_indices <- args[[2L]]; rhs <- args[[3L]]
  dn <- params$dimension_numbers
  pet <- params$preferred_element_type
  out_aval <- bcoo_dot_general_p$abstract_eval(lapply(args, aval_of), params)
  is <- sp_shape(lhs_indices)
  n_sparse <- is[length(is)]
  n_batch <- length(is) - 2L
  lc <- dn[[1L]][[1L]]; rc <- dn[[1L]][[2L]]; lb <- dn[[2L]][[1L]]; rb <- dn[[2L]][[2L]]
  inb <- lc < n_batch
  lc_b <- lc[inb]; rc_b <- rc[inb]
  lc_s <- lc[!inb]; rc_s <- rc[!inb]
  if (length(lb) || length(lc_b)) {
    batch_perm <- c(lb, sp_remaining(sp_pos(n_batch), lb, lc_b), lc_b)
    lhs_data <- lax_transpose(lhs_data, c(batch_perm, seq.int(n_batch, sp_ndim(lhs_data) - 1L)))
    lhs_indices <- lax_transpose(lhs_indices, c(batch_perm, seq.int(n_batch, sp_ndim(lhs_indices) - 1L)))
  }
  if (length(lc_s)) {
    lc_s <- lc_s - n_batch
    lhs_indices <- sp_take_last(lhs_indices, c(lc_s, sp_remaining(sp_pos(n_sparse), lc_s)))
  }
  rhs_perm <- c(rb, rc_b, rc_s, sp_remaining(sp_pos(sp_ndim(rhs)), rb, rc))
  rhs <- lax_transpose(rhs, rhs_perm)
  n_cs <- length(lc_s)
  result <- function(out_array, lhs_data, lhs_indices, rhs) {
    idx <- lapply(sp_pos(n_sparse), function(i) sp_index_last(lhs_indices, i))
    idx_right <- idx[seq_len(n_cs)]
    idx_out <- idx[seq_along(idx) > n_cs]
    li <- sp_shape(lhs_indices)
    if (length(idx_right) && length(li) > 2L) {
      nbd <- length(li) - 2L
      idx_batch <- lapply(sp_pos(nbd), function(b) {
        lax_broadcasted_iota(sp_dtype(lhs_indices), li[-length(li)], b)
      })
      idx_right <- c(idx_batch, idx_right)
    }
    bd <- sp_pos(length(lc_b) + as.integer(n_cs > 0L))
    gathered <- if (length(idx_right)) do.call(at_get, c(list(rhs), idx_right, list(mode = "fill", fill_value = 0)))
                else rhs
    prod <- lax_dot_general(lhs_data, gathered, list(list(integer(0), integer(0)), list(bd, bd)),
                            preferred_element_type = pet)
    if (length(idx_out)) {
      do.call(at_add, c(list(out_array), idx_out, list(value = prod)))
    } else {
      extra <- sp_ndim(prod) - sp_ndim(out_array)
      jnp_sum(prod, axis = sp_pos(extra), dtype = sp_dtype(out_array))
    }
  }
  nv <- n_batch - length(lc_b)
  result <- sp_nfold_vmap(result, nv)
  if (nv > length(rb)) rhs <- lax_expand_dims(rhs, seq.int(length(rb), nv - 1L))
  out_array <- lax_full(out_aval$shape, 0, out_aval$dtype)
  result(out_array, lhs_data, lhs_indices, rhs)
}

bcoo_dot_general_p$abstract_eval <- function(avals, params) {
  dn <- params$dimension_numbers
  spinfo <- params$lhs_spinfo
  out <- bc_dot_shape(new_aval(spinfo$shape, avals[[1L]]$dtype), avals[[3L]], dn,
                      params$preferred_element_type)
  p <- bc_validate(avals[[1L]], avals[[2L]], spinfo$shape)
  lb <- dn[[2L]][[1L]]; lc <- dn[[1L]][[1L]]
  if (length(lb) && max(lb) >= p$n_batch)
    jax_error("bcoo_dot_general batch dimensions must be among the batch dimensions in the sparse ",
              "representation.\ngot lhs_batch=", sp_tuple(lb), ", n_batch=", p$n_batch,
              class = "jax_not_implemented_error")
  if (any(lc >= p$n_batch + p$n_sparse))
    jax_error("bcoo_dot_general: contracting over dense dimensions.", class = "jax_not_implemented_error")
  out
}
bcoo_dot_general_p$lowering <- lower_fun(bc_dot_general_impl)
defjvp(bcoo_dot_general_p,
       function(t, primals, params) bc_dot_general(t, primals[[2L]], primals[[3L]], params),
       NULL,
       function(t, primals, params) bc_dot_general(primals[[1L]], primals[[2L]], t, params))

bc_ranges_like <- function(...) {
  xs <- list(...)
  start <- 0L
  lapply(xs, function(x) {
    r <- start + sp_pos(length(x))
    start <<- start + length(x)
    r
  })
}

bcoo_dot_general_p$transpose <- function(ct, args, params) {
  lhs_data <- args[[1L]]; lhs_indices <- args[[2L]]; rhs <- args[[3L]]
  dn <- params$dimension_numbers
  lc <- dn[[1L]][[1L]]; rc <- dn[[1L]][[2L]]; lb <- dn[[2L]][[1L]]; rb <- dn[[2L]][[2L]]
  spinfo <- params$lhs_spinfo
  lhs_ndim <- length(spinfo$shape)
  rhs_ndim <- if (is_undefined(rhs)) length(rhs$aval$shape) else sp_ndim(rhs)
  lhs_kept <- sp_remaining(sp_pos(lhs_ndim), lc, lb)
  rhs_kept <- sp_remaining(sp_pos(rhs_ndim), rc, rb)
  rl <- bc_ranges_like(lb, lhs_kept, rhs_kept)
  ans_batch <- rl[[1L]]; ans_lhs <- rl[[2L]]; ans_rhs <- rl[[3L]]
  ct <- instantiate_zero(ct)
  if (is_undefined(lhs_data)) {
    dims <- list(list(ans_rhs, rhs_kept), list(ans_batch, rb))
    lc_sorted <- lc[order(rc)]
    permutation <- c(lb, lhs_kept, lc_sorted)
    out_axes <- order(permutation) - 1L
    is <- sp_shape(lhs_indices)
    nbi <- length(is) - 2L
    placeholder_data <- lax_full(c(rep(1L, nbi), is[nbi + 1L]), 0, "float32")
    placeholder_shape <- c(is[seq_len(nbi)], rep(1L, is[length(is)]))
    ok <- tryCatch({
      bc_validate_permutation(placeholder_data, lhs_indices, permutation, placeholder_shape)
      TRUE
    }, jax_not_implemented_error = function(e) FALSE)
    if (ok) {
      lhs_indices_T <- bc_transpose(placeholder_data, lhs_indices, permutation,
                                    sp_spinfo(placeholder_shape))[[2L]]
      result_T_shape <- placeholder_shape[permutation + 1L]
      result_T <- experimental_sparse_bcoo_dot_general_sampled(ct, rhs, lhs_indices_T, dims)
      result <- bc_transpose(result_T, lhs_indices_T, out_axes, sp_spinfo(result_T_shape))[[1L]]
    } else {
      out_dense <- lax_transpose(lax_dot_general(ct, rhs, dims), out_axes)
      result <- bc_extract(lhs_indices, out_dense)
    }
    result <- unbroadcast(lhs_data$aval, result)
    list(result, NULL, NULL)
  } else {
    dims <- list(list(lhs_kept, ans_lhs), list(lb, ans_batch))
    rc_sorted <- rc[order(lc)]
    out_axes <- order(c(rb, rc_sorted, rhs_kept)) - 1L
    result <- bc_dot_general(lhs_data, lhs_indices, ct,
                             list(dimension_numbers = dims, preferred_element_type = params$preferred_element_type,
                                  lhs_spinfo = spinfo))
    list(NULL, NULL, lax_transpose(result, out_axes))
  }
}

bc_dot_general_batch_dim_nums <- function(ndims, bdims, dn) {
  lhs_ndim <- ndims[1L]; rhs_ndim <- ndims[2L]
  lbd <- bdims[1L]; rbd <- bdims[2L]
  lc <- dn[[1L]][[1L]]; rc <- dn[[1L]][[2L]]; lb <- dn[[2L]][[1L]]; rb <- dn[[2L]][[2L]]
  bump <- function(d, b) d + as.integer(d >= b)
  if (!is.na(lbd) && !is.na(rbd)) {
    lb <- c(lbd, bump(lb, lbd)); rb <- c(rbd, bump(rb, rbd))
    lc <- bump(lc, lbd); rc <- bump(rc, rbd)
    rbdim <- 0L
  } else if (!is.na(lbd)) {
    lhs_tensor <- sp_remaining(sp_pos(lhs_ndim), lb, lc)
    rbdim <- length(lb) + sum(lhs_tensor < lbd)
    lb <- bump(lb, lbd); lc <- bump(lc, lbd)
  } else {
    rhs_tensor <- sp_remaining(sp_pos(rhs_ndim), rb, rc)
    rbdim <- lhs_ndim - length(lc) + sum(rhs_tensor < rbd)
    rb <- bump(rb, rbd); rc <- bump(rc, rbd)
  }
  list(dn = list(list(as.integer(lc), as.integer(rc)), list(as.integer(lb), as.integer(rb))),
       bdim = as.integer(rbdim))
}

bcoo_dot_general_p$batch <- function(vals, dims, params, axis) {
  rhs <- vals[[3L]]
  rbd <- dims[3L]
  b <- bc_batch_dims_to_front(vals[1:2], dims[1:2], params$lhs_spinfo,
                              batch_size = if (is.na(rbd)) NULL else sp_shape(rhs)[rbd + 1L])
  nd <- bc_dot_general_batch_dim_nums(c(length(params$lhs_spinfo$shape), sp_ndim(rhs)),
                                      c(0L, rbd), params$dimension_numbers)
  out <- bc_dot_general(b$data, b$indices, rhs,
                        list(dimension_numbers = nd$dn, preferred_element_type = params$preferred_element_type,
                             lhs_spinfo = b$spinfo))
  list(out, nd$bdim)
}

bc_dot_general <- function(lhs_data, lhs_indices, rhs, params) {
  params$dimension_numbers <- bc_dn(params$dimension_numbers)
  if (!is.null(params$preferred_element_type)) params$preferred_element_type <- dtype(params$preferred_element_type)
  bind(bcoo_dot_general_p, list(as_jax(lhs_data), as_jax(lhs_indices), as_jax(rhs)), params)
}

bc_rdot_general <- function(lhs, rhs_data, rhs_indices, dn, pet, rhs_spinfo) {
  dn <- bc_dn(dn)
  dn_rev <- list(rev(dn[[1L]]), rev(dn[[2L]]))
  result <- bc_dot_general(rhs_data, rhs_indices, lhs,
                           list(dimension_numbers = dn_rev, preferred_element_type = pet, lhs_spinfo = rhs_spinfo))
  n_contract <- length(dn[[1L]][[1L]]); n_batch <- length(dn[[2L]][[1L]])
  n_swap <- length(rhs_spinfo$shape) - n_contract
  r <- sp_ndim(result)
  perm <- c(sp_pos(n_batch), if (r > n_swap) seq.int(n_swap, r - 1L), if (n_swap > n_batch) seq.int(n_batch, n_swap - 1L))
  lax_transpose(result, perm)
}

#' General dot product with BCOO operands
#'
#' Port of `jax.experimental.sparse.bcoo_dot_general`: either operand may be a BCOO
#' array (both sparse gives a BCOO result, otherwise a dense array).
#' @param lhs,rhs BCOO or dense arrays.
#' @param dimension_numbers As for [lax_dot_general()] (0-based).
#' @param precision Ignored, as in JAX.
#' @param preferred_element_type Optional result dtype.
#' @return A BCOO or dense array.
#' @export
experimental_sparse_bcoo_dot_general <- function(lhs, rhs, dimension_numbers, precision = NULL,
                                                 preferred_element_type = NULL) {
  dn <- bc_dn(dimension_numbers)
  lsp <- inherits(lhs, "jax_bcoo"); rsp <- inherits(rhs, "jax_bcoo")
  if (lsp && rsp) {
    shape <- bc_dot_shape(new_aval(lhs$shape, lhs$dtype), new_aval(rhs$shape, rhs$dtype), dn)$shape
    bufs <- bc_spdot_general(lhs$data, lhs$indices, rhs$data, rhs$indices, bc_info(lhs), bc_info(rhs),
                             dn, preferred_element_type)
    return(experimental_sparse_BCOO(bufs, shape = shape))
  }
  if (lsp) {
    return(bc_dot_general(lhs$data, lhs$indices, rhs,
                          list(dimension_numbers = dn, preferred_element_type = preferred_element_type,
                               lhs_spinfo = bc_info(lhs))))
  }
  if (rsp) return(bc_rdot_general(lhs, rhs$data, rhs$indices, dn, preferred_element_type, bc_info(rhs)))
  lax_dot_general(lhs, rhs, dn, preferred_element_type = preferred_element_type)
}

# ---- bcoo_dot_general_sampled -------------------------------------------------------------

bcoo_dot_general_sampled_p <- new_primitive("bcoo_dot_general_sampled")

# ponytail: always the dense-then-extract path of _bcoo_dot_general_sampled_slow (values
# equal JAX; O(dense) memory). Port the simple/simple2 kernels if memory matters.
bc_sampled_impl <- function(args, params) {
  dense <- lax_dot_general(args[[1L]], args[[2L]], params$dimension_numbers, precision = "highest")
  bc_extract(args[[3L]], dense)
}
bcoo_dot_general_sampled_p$abstract_eval <- function(avals, params) {
  dense <- eval_shape(function(a, b) lax_dot_general(a, b, params$dimension_numbers), avals[[1L]], avals[[2L]])
  bcoo_extract_p$abstract_eval(list(avals[[3L]], new_aval(dense$shape, dense$dtype)), list(assume_unique = TRUE))
}
bcoo_dot_general_sampled_p$lowering <- lower_fun(bc_sampled_impl)
defjvp(bcoo_dot_general_sampled_p,
       function(t, primals, params) bind(bcoo_dot_general_sampled_p, list(t, primals[[2L]], primals[[3L]]), params),
       function(t, primals, params) bind(bcoo_dot_general_sampled_p, list(primals[[1L]], t, primals[[3L]]), params),
       NULL)
bcoo_dot_general_sampled_p$transpose <- function(ct, args, params) {
  A <- args[[1L]]; B <- args[[2L]]; indices <- args[[3L]]
  sa <- if (is_undefined(A)) A$aval else aval_of(A)
  sb <- if (is_undefined(B)) B$aval else aval_of(B)
  mat_shape <- bc_dot_shape(sa, sb, params$dimension_numbers)$shape
  ctmat <- bc_todense(instantiate_zero(ct), indices, sp_spinfo(mat_shape))
  dn <- params$dimension_numbers
  dg <- list(dimension_numbers = list(lhs_contracting = dn[[1L]][[1L]], rhs_contracting = dn[[1L]][[2L]],
                                      lhs_batch = dn[[2L]][[1L]], rhs_batch = dn[[2L]][[2L]]),
             precision = NULL, preferred_element_type = NULL)
  ab <- dot_general_p$transpose(ctmat, list(A, B), dg)
  list(ab[[1L]], ab[[2L]], NULL)
}
bcoo_dot_general_sampled_p$batch <- function(vals, dims, params, axis) {
  f <- function(A, B, indices) bc_sampled_impl(list(A, B, indices), params)
  ax <- lapply(dims, function(d) if (is.na(d)) NULL else d)
  list(do.call(vmap(f, in_axes = ax), vals), 0L)
}

#' Sampled dense dot product
#'
#' Port of `jax.experimental.sparse.bcoo_dot_general_sampled`: the values of
#' `lax_dot_general(A, B, dimension_numbers)` at `indices`.
#' @param A,B Dense arrays.
#' @param indices BCOO indices.
#' @param dimension_numbers As for [lax_dot_general()].
#' @return The data of a BCOO array with the given indices.
#' @export
experimental_sparse_bcoo_dot_general_sampled <- function(A, B, indices, dimension_numbers) {
  bind(bcoo_dot_general_sampled_p, list(as_jax(A), as_jax(B), as_jax(indices)),
       list(dimension_numbers = bc_dn(dimension_numbers)))
}

# ---- bcoo_spdot_general -------------------------------------------------------------------

bcoo_spdot_general_p <- new_primitive("bcoo_spdot_general", multiple_results = TRUE)

bc_spdot_unbatched <- function(lhs_data, lhs_indices, rhs_data, rhs_indices, lhs_shape, rhs_shape,
                               lhs_contracting, rhs_contracting, out_nse) {
  lhs <- bc_validate(lhs_data, lhs_indices, lhs_shape)
  rhs <- bc_validate(rhs_data, rhs_indices, rhs_shape)
  out_shape <- c(lhs_shape[!sp_pos(length(lhs_shape)) %in% lhs_contracting],
                 rhs_shape[!sp_pos(length(rhs_shape)) %in% rhs_contracting])
  lhs_i <- sp_take_last(lhs_indices, lhs_contracting)
  rhs_i <- sp_take_last(rhs_indices, rhs_contracting)
  lhs_j <- sp_take_last(lhs_indices, sp_remaining(sp_pos(lhs$n_sparse), lhs_contracting))
  rhs_j <- sp_take_last(rhs_indices, sp_remaining(sp_pos(rhs$n_sparse), rhs_contracting))
  overlap <- jnp_all(jnp_equal(lax_expand_dims(lhs_i, 1L), lax_expand_dims(rhs_i, 0L)), axis = -1L)
  idt <- sp_dtype(lhs_i)
  lhs_fill <- as_jax(array(lhs_shape[lhs_contracting + 1L], c(1L, length(lhs_contracting))), idt)
  rhs_fill <- as_jax(array(rhs_shape[rhs_contracting + 1L], c(1L, length(rhs_contracting))), idt)
  lhs_valid <- jnp_all(jnp_less(lhs_i, lhs_fill), axis = -1L)
  rhs_valid <- jnp_all(jnp_less(rhs_i, rhs_fill), axis = -1L)
  keep <- overlap & lax_expand_dims(lhs_valid, 1L) & lax_expand_dims(rhs_valid, 0L)
  out_data <- jnp_ravel(jnp_where(keep, jnp_multiply(lax_expand_dims(lhs_data, 1L),
                                                     lax_expand_dims(rhs_data, 0L)), 0))
  nl <- sp_shape(lhs_j)[2L]; nr <- sp_shape(rhs_j)[2L]
  oit <- result_type(lhs_indices, rhs_indices)
  out_indices <- jnp_concatenate(list(
    jnp_broadcast_to(lax_expand_dims(lax_convert_element_type(lhs_j, oit), 1L), c(lhs$nse, rhs$nse, nl)),
    jnp_broadcast_to(lax_expand_dims(lax_convert_element_type(rhs_j, oit), 0L), c(lhs$nse, rhs$nse, nr))),
    axis = 2L)
  out_indices <- lax_reshape(out_indices, c(prod_int(sp_shape(out_data)), nl + nr))
  bc_sum_duplicates(out_data, out_indices, sp_spinfo(out_shape), out_nse)
}

bc_spdot_impl <- function(args, params) {
  lhs_data <- args[[1L]]; lhs_indices <- args[[2L]]; rhs_data <- args[[3L]]; rhs_indices <- args[[4L]]
  lhs_shape <- params$lhs_spinfo$shape; rhs_shape <- params$rhs_spinfo$shape
  lhs <- bc_validate(lhs_data, lhs_indices, lhs_shape)
  rhs <- bc_validate(rhs_data, rhs_indices, rhs_shape)
  out_nse <- bcoo_spdot_general_p$abstract_eval(lapply(args, aval_of), params)[[1L]]$shape
  out_nse <- out_nse[length(out_nse)]
  dn <- params$dimension_numbers
  lc <- dn[[1L]][[1L]]; rc <- dn[[1L]][[2L]]; lb <- dn[[2L]][[1L]]; rb <- dn[[2L]][[2L]]
  lperm <- c(lb, sp_remaining(sp_pos(lhs$n_batch), lb))
  rperm <- c(rb, sp_remaining(sp_pos(rhs$n_batch), rb))
  tr <- function(x, perm, nb) lax_transpose(x, c(perm, if (sp_ndim(x) > nb) seq.int(nb, sp_ndim(x) - 1L)))
  lhs_data <- tr(lhs_data, lperm, lhs$n_batch); lhs_indices <- tr(lhs_indices, lperm, lhs$n_batch)
  rhs_data <- tr(rhs_data, rperm, rhs$n_batch); rhs_indices <- tr(rhs_indices, rperm, rhs$n_batch)
  lsub <- lhs_shape[seq_along(lhs_shape) > lhs$n_batch]
  rsub <- rhs_shape[seq_along(rhs_shape) > rhs$n_batch]
  func <- function(ld, li, rd, ri) bc_spdot_unbatched(ld, li, rd, ri, lsub, rsub, lc - lhs$n_batch,
                                                      rc - rhs$n_batch, out_nse)
  func <- sp_nfold_vmap(func, rhs$n_batch - length(rb), in_axes = list(NULL, NULL, 0L, 0L))
  func <- sp_nfold_vmap(func, lhs$n_batch - length(lb), in_axes = list(0L, 0L, NULL, NULL))
  func <- sp_nfold_vmap(func, length(lb))
  func(lhs_data, lhs_indices, rhs_data, rhs_indices)
}

bcoo_spdot_general_p$abstract_eval <- function(avals, params) {
  lhs_shape <- params$lhs_spinfo$shape; rhs_shape <- params$rhs_spinfo$shape
  out <- bc_dot_shape(new_aval(lhs_shape, avals[[1L]]$dtype), new_aval(rhs_shape, avals[[3L]]$dtype),
                      params$dimension_numbers, params$preferred_element_type)
  lhs <- bc_validate(avals[[1L]], avals[[2L]], lhs_shape)
  rhs <- bc_validate(avals[[3L]], avals[[4L]], rhs_shape)
  dn <- params$dimension_numbers
  lc <- dn[[1L]][[1L]]; rc <- dn[[1L]][[2L]]; lb <- dn[[2L]][[1L]]; rb <- dn[[2L]][[2L]]
  if (lhs$n_dense || rhs$n_dense)
    jax_error("bcoo_spdot_general with dense dimensions.", class = "jax_not_implemented_error")
  if ((length(lb) && max(lb) >= lhs$n_batch) || (length(rb) && max(rb) >= rhs$n_batch))
    jax_error("bcoo_spdot_general: batch_dims must correspond to batch dimensions of the sparse ",
              "representation.", class = "jax_not_implemented_error")
  if (length(lc) && (min(lc) < lhs$n_batch || max(lc) >= lhs$n_batch + lhs$n_sparse))
    jax_error("bcoo_spdot_general only supports contraction of sparse indices.", class = "jax_not_implemented_error")
  if (length(rc) && (min(rc) < rhs$n_batch || max(rc) >= rhs$n_batch + rhs$n_sparse))
    jax_error("bcoo_spdot_general only supports contraction of sparse indices.", class = "jax_not_implemented_error")
  if (rhs$n_batch > length(rb) && lhs$n_sparse > length(lc))
    jax_error("bcoo_spdot_general: cannot have unused batch dims on rhs with unused sparse dims on lhs.",
              class = "jax_value_error")
  out_nse <- (if (lhs$n_sparse > length(lc)) lhs$nse else 1L) * (if (rhs$n_sparse > length(rc)) rhs$nse else 1L)
  out_n_batch <- lhs$n_batch + rhs$n_batch - length(lb)
  out_nse <- min(out_nse, prod_int(out$shape[seq_along(out$shape) > out_n_batch]))
  bshape <- function(d, i, nb, bat) {
    keep <- !sp_pos(nb) %in% bat
    broadcast_shapes_impl(list(d$shape[seq_len(nb)][keep], i$shape[seq_len(nb)][keep]))
  }
  lbs <- bshape(avals[[1L]], avals[[2L]], lhs$n_batch, lb)
  rbs <- bshape(avals[[3L]], avals[[4L]], rhs$n_batch, rb)
  data_shape <- c(lhs_shape[lb + 1L], lbs, rbs, out_nse)
  indices_shape <- c(data_shape, lhs$n_sparse + rhs$n_sparse - 2L * length(lc))
  list(new_aval(data_shape, out$dtype), new_aval(indices_shape, avals[[2L]]$dtype))
}
bcoo_spdot_general_p$lowering <- lower_fun(bc_spdot_impl, multiple_results = TRUE)
bcoo_spdot_general_p$batch <- function(vals, dims, params, axis) {
  bs <- max(unlist(Map(function(a, d) if (is.na(d)) NULL else sp_shape(a)[d + 1L], vals, dims)))
  l <- bc_batch_dims_to_front(vals[1:2], dims[1:2], params$lhs_spinfo, batch_size = bs)
  r <- bc_batch_dims_to_front(vals[3:4], dims[3:4], params$rhs_spinfo, batch_size = bs)
  nd <- bc_dot_general_batch_dim_nums(c(length(params$lhs_spinfo$shape), length(params$rhs_spinfo$shape)),
                                      c(0L, 0L), params$dimension_numbers)
  out <- bc_spdot_general(l$data, l$indices, r$data, r$indices, l$spinfo, r$spinfo, nd$dn,
                          params$preferred_element_type)
  list(out, c(nd$bdim, nd$bdim))
}
bcoo_spdot_general_p$jvp <- function(primals, tangents, params) {
  out <- bind(bcoo_spdot_general_p, primals, params)
  dd <- NULL
  if (!is_zero(tangents[[1L]]))
    dd <- bind(bcoo_spdot_general_p, list(tangents[[1L]], primals[[2L]], primals[[3L]], primals[[4L]]), params)[[1L]]
  if (!is_zero(tangents[[3L]])) {
    t2 <- bind(bcoo_spdot_general_p, list(primals[[1L]], primals[[2L]], tangents[[3L]], primals[[4L]]), params)[[1L]]
    dd <- if (is.null(dd)) t2 else lax_add(dd, t2)
  }
  if (is.null(dd)) dd <- sp_zero_tangent(out[[1L]])
  list(out, list(dd, sp_zero_tangent(out[[2L]])))
}

bc_spdot_general <- function(lhs_data, lhs_indices, rhs_data, rhs_indices, lhs_spinfo, rhs_spinfo, dn, pet) {
  bind(bcoo_spdot_general_p, lapply(list(lhs_data, lhs_indices, rhs_data, rhs_indices), as_jax),
       list(lhs_spinfo = lhs_spinfo, rhs_spinfo = rhs_spinfo, dimension_numbers = bc_dn(dn),
            preferred_element_type = if (is.null(pet)) NULL else dtype(pet)))
}

# ---- bcoo_sort_indices --------------------------------------------------------------------

bcoo_sort_indices_p <- new_primitive("bcoo_sort_indices", multiple_results = TRUE)

bc_sort_unbatched <- function(indices) {
  s <- sp_shape(indices)
  nse <- s[1L]; N <- s[2L]
  cols <- lapply(sp_pos(N), function(i) sp_index_last(indices, i))
  out <- lax_sort(c(cols, list(lax_iota(sp_dtype(indices), nse))), num_keys = N)
  list(jnp_column_stack(out[seq_len(N)]), out[[N + 1L]])
}

bc_permute <- function(nb) sp_nfold_vmap(function(d, p) jnp_take(d, p, axis = 0L), nb)

bc_sort_impl <- function(args, params) {
  data <- args[[1L]]; indices <- args[[2L]]
  p <- bc_validate(data, indices, params$spinfo$shape)
  if (p$n_sparse == 0L) return(list(data, indices))
  ip <- sp_nfold_vmap(bc_sort_unbatched, p$n_batch, broadcasted = FALSE)(indices)
  list(bc_permute(p$n_batch)(data, ip[[2L]]), ip[[1L]])
}

bcoo_sort_indices_p$abstract_eval <- function(avals, params) {
  p <- bc_validate(avals[[1L]], avals[[2L]], params$spinfo$shape)
  if (p$n_sparse == 0L) return(avals)
  d <- avals[[1L]]; i <- avals[[2L]]
  nb <- p$n_batch
  ds <- c(pmax(i$shape[seq_len(nb)], d$shape[seq_len(nb)]), p$nse, d$shape[seq_along(d$shape) > nb + 1L])
  list(new_aval(ds, d$dtype, d$weak), i)
}
bcoo_sort_indices_p$lowering <- lower_fun(bc_sort_impl, multiple_results = TRUE)
bcoo_sort_indices_p$batch <- function(vals, dims, params, axis) {
  b <- bc_batch_dims_to_front(vals, dims, params$spinfo)
  out <- bind(bcoo_sort_indices_p, list(b$data, b$indices), list(spinfo = b$spinfo))
  if (is.na(dims[2L])) return(list(list(out[[1L]], lax_index_in_dim(out[[2L]], 0L, keepdims = FALSE)), c(0L, NA)))
  list(out, c(0L, 0L))
}
bcoo_sort_indices_p$jvp <- function(primals, tangents, params) {
  data <- primals[[1L]]; indices <- primals[[2L]]
  p <- bc_validate(data, indices, params$spinfo$shape)
  if (p$n_sparse == 0L) return(list(primals, tangents))
  ip <- sp_nfold_vmap(bc_sort_unbatched, p$n_batch)(indices)
  perm <- bc_permute(p$n_batch)
  data_out <- perm(data, ip[[2L]])
  dd <- if (is_zero(tangents[[1L]])) sp_zero_tangent(data_out) else perm(tangents[[1L]], ip[[2L]])
  list(list(data_out, ip[[1L]]), list(dd, sp_zero_tangent(indices)))
}

#' Sort the indices of a BCOO array
#'
#' Port of `jax.experimental.sparse.bcoo_sort_indices` (row major order).
#' @param mat A BCOO array.
#' @return A BCOO array with sorted indices.
#' @export
experimental_sparse_bcoo_sort_indices <- function(mat) {
  out <- bind(bcoo_sort_indices_p, list(mat$data, mat$indices), list(spinfo = bc_info(mat)))
  experimental_sparse_BCOO(out, shape = mat$shape, indices_sorted = TRUE, unique_indices = mat$unique_indices)
}

# ---- bcoo_sum_duplicates ------------------------------------------------------------------

bcoo_sum_duplicates_p <- new_primitive("bcoo_sum_duplicates", multiple_results = TRUE)

bc_unique_unbatched <- function(indices, shape, return_inverse = FALSE, return_index = FALSE,
                                return_true_size = FALSE) {
  props <- bc_validate_indices(indices, shape)
  if (props$n_sparse == 0L) {
    out <- list(lax_full(c(1L, 0L), 0, sp_dtype(indices)))
    if (return_index) out <- c(out, list(lax_full(1L, 0, "int32")))
    if (return_inverse) out <- c(out, list(lax_full(1L, 0, "int32")))
    if (return_true_size) out <- c(out, list(as_jax(1L, "int32")))
    return(if (length(out) == 1L) out[[1L]] else out)
  }
  fill_value <- as_jax(array(shape[seq_len(props$n_sparse)], c(1L, props$n_sparse)), sp_dtype(indices))
  oob <- jnp_any(jnp_greater_equal(indices, fill_value), axis = -1L, keepdims = TRUE)
  indices <- jnp_where(oob, fill_value, indices)
  out <- set_unique(indices, 0L, return_index = return_index, return_inverse = return_inverse,
                    return_true_size = return_true_size, size = props$nse, fill_value = fill_value)
  if (!is.list(out)) out <- list(out)
  if (return_inverse) {
    k <- if (return_index) 3L else 2L
    out[[k]] <- jnp_ravel(out[[k]])
  }
  if (return_true_size) {
    n <- length(out)
    nse <- out[[n]]
    out[[n]] <- jnp_subtract(nse, jnp_astype(jnp_any(jnp_all(jnp_equal(indices, fill_value), axis = -1L)),
                                             sp_dtype(nse)))
  }
  if (length(out) == 1L) out[[1L]] else out
}

bc_unique_indices <- function(indices, shape, return_inverse = FALSE, return_index = FALSE,
                              return_true_size = FALSE) {
  props <- bc_validate_indices(indices, shape)
  sub <- shape[seq_along(shape) > props$n_batch]
  f <- function(i) bc_unique_unbatched(i, sub, return_inverse, return_index, return_true_size)
  sp_nfold_vmap(f, props$n_batch, broadcasted = FALSE)(indices)
}

bc_adjust_indices_nse <- function(indices, nse, shape) {
  props <- bc_validate_indices(indices, shape)
  is <- sp_shape(indices)
  r <- length(is)
  if (nse <= props$nse) return(lax_slice_in_dim(indices, 0L, nse, axis = r - 2L))
  fill <- lax_broadcast_in_dim(as_jax(array(shape[props$n_batch + seq_len(props$n_sparse)], props$n_sparse),
                                      sp_dtype(indices)),
                               c(is[seq_len(r - 2L)], nse - props$nse, is[r]), r - 1L)
  lax_concatenate(list(indices, fill), r - 2L)
}

bc_sum_duplicates_compute <- function(data, indices, spinfo, nse) {
  props <- bc_validate(data, indices, spinfo$shape)
  u <- bc_unique_indices(indices, spinfo$shape, return_inverse = TRUE, return_true_size = TRUE)
  indices_out <- u[[1L]]; mapping <- u[[2L]]; nse_batched <- u[[3L]]
  if (is.null(nse)) nse <- if (props$n_sparse == 0L) 1L else sp_static_int(jnp_max(nse_batched), "bcoo_sum_duplicates nse")
  indices_out <- bc_adjust_indices_nse(indices_out, nse, spinfo$shape)
  nb <- props$n_batch
  if (props$n_sparse == 0L) data <- jnp_sum(data, axis = nb, keepdims = TRUE, dtype = sp_dtype(data))
  ds <- sp_shape(data); is <- sp_shape(indices)
  data_out <- lax_full(c(pmax(is[seq_len(nb)], ds[seq_len(nb)]), nse, ds[seq_along(ds) > nb + 1L]), 0, sp_dtype(data))
  permute <- sp_nfold_vmap(function(d_out, m, d) at_add(d_out, m, value = d, mode = "drop"), nb)
  list(permute(data_out, mapping, data), indices_out)
}

bcoo_sum_duplicates_p$impl <- function(args, params) {
  bc_sum_duplicates_compute(args[[1L]], args[[2L]], params$spinfo, params$nse)
}
bcoo_sum_duplicates_p$abstract_eval <- function(avals, params) {
  if (is.null(params$nse))
    jax_error("bcoo_sum_duplicates: nse must be specified when using the function within jit, vmap, ",
              "and other transformations requiring abstract evaluation.", class = "jax_value_error")
  p <- bc_validate(avals[[1L]], avals[[2L]], params$spinfo$shape)
  d <- avals[[1L]]; i <- avals[[2L]]
  nb <- p$n_batch
  list(new_aval(c(pmax(i$shape[seq_len(nb)], d$shape[seq_len(nb)]), params$nse,
                  d$shape[seq_along(d$shape) > nb + 1L]), d$dtype, d$weak),
       new_aval(c(i$shape[seq_len(nb)], params$nse, p$n_sparse), i$dtype, i$weak))
}
bcoo_sum_duplicates_p$lowering <- lower_fun(function(args, params) {
  bc_sum_duplicates_compute(args[[1L]], args[[2L]], params$spinfo, params$nse)
}, multiple_results = TRUE)
bcoo_sum_duplicates_p$batch <- function(vals, dims, params, axis) {
  b <- bc_batch_dims_to_front(vals, dims, params$spinfo)
  out <- bind(bcoo_sum_duplicates_p, list(b$data, b$indices), list(spinfo = b$spinfo, nse = params$nse))
  if (is.na(dims[2L])) return(list(list(out[[1L]], lax_squeeze(out[[2L]], 0L)), c(0L, NA)))
  list(out, c(0L, 0L))
}
bcoo_sum_duplicates_p$jvp <- function(primals, tangents, params) {
  data <- primals[[1L]]; indices <- primals[[2L]]
  spinfo <- params$spinfo
  props <- bc_validate(data, indices, spinfo$shape)
  u <- bc_unique_indices(indices, spinfo$shape, return_inverse = TRUE, return_true_size = TRUE)
  nse <- params$nse
  if (is.null(nse)) {
    nse <- tryCatch(sp_static_int(jnp_sum(u[[3L]])), jax_concretization_error = function(e)
      jax_error("bcoo_sum_duplicates: nse must be specified when using the function within jit, vmap, ",
                "and other transformations requiring abstract evaluation.", class = "jax_value_error"))
  }
  indices_out <- bc_adjust_indices_nse(u[[1L]], nse, spinfo$shape)
  nb <- props$n_batch
  data_dot <- tangents[[1L]]
  if (props$n_sparse == 0L) {
    data <- jnp_sum(data, axis = nb, keepdims = TRUE, dtype = sp_dtype(data))
    if (!is_zero(data_dot)) data_dot <- jnp_sum(data_dot, axis = nb, keepdims = TRUE, dtype = sp_dtype(data_dot))
  }
  ds <- sp_shape(data); is <- sp_shape(indices)
  zeros <- lax_full(c(pmax(is[seq_len(nb)], ds[seq_len(nb)]), nse, ds[seq_along(ds) > nb + 1L]), 0, sp_dtype(data))
  permute <- if (prod_int(sp_shape(zeros))) function(x, i, y) at_add(x, i, value = y, mode = "drop")
             else function(x, i, y) x
  permute <- sp_nfold_vmap(permute, nb)
  data_out <- permute(zeros, u[[2L]], data)
  dd <- if (is_zero(data_dot)) sp_zero_tangent(data_out) else permute(zeros, u[[2L]], data_dot)
  list(list(data_out, indices_out), list(dd, sp_zero_tangent(indices_out)))
}

bc_sum_duplicates <- function(data, indices, spinfo, nse) {
  if (!is.null(nse)) nse <- sp_static_int(nse, "nse argument of bcoo_sum_duplicates.")
  bind(bcoo_sum_duplicates_p, list(as_jax(data), as_jax(indices)), list(spinfo = spinfo, nse = nse))
}

#' Sum duplicate entries of a BCOO array
#'
#' Ports of `jax.experimental.sparse.bcoo_sum_duplicates` and `bcoo_eliminate_zeros`
#' (the latter is reached through `mat$sum_duplicates(remove_zeros = TRUE)`). `nse` must
#' be given under transformations.
#' @param mat A BCOO array.
#' @param nse Number of stored elements of the result.
#' @return A BCOO array with sorted, unique indices.
#' @export
experimental_sparse_bcoo_sum_duplicates <- function(mat, nse = NULL) {
  out <- bc_sum_duplicates(mat$data, mat$indices, bc_info(mat), nse)
  experimental_sparse_BCOO(out, shape = mat$shape, indices_sorted = TRUE, unique_indices = TRUE)
}

bc_eliminate_zeros <- function(mat, nse = NULL) {
  data <- mat$data; indices <- mat$indices; shape <- mat$shape
  props <- bc_validate(data, indices, shape)
  nb <- props$n_batch
  r <- sp_ndim(data)
  mask <- jnp_all(jnp_equal(data, 0), axis = if (r > nb + 1L) seq.int(nb + 1L, r - 1L) else integer(0))
  ib <- sp_shape(indices)[seq_len(nb)]
  contract <- which(ib == 1L) - 1L
  if (length(contract)) mask <- jnp_all(mask, axis = contract, keepdims = TRUE)
  fill <- as_jax(array(shape[nb + seq_len(props$n_sparse)], c(1L, props$n_sparse)), sp_dtype(indices))
  f <- function(i, m) jnp_where(lax_expand_dims(m, 1L), fill, i)
  indices <- sp_nfold_vmap(f, nb)(indices, mask)
  experimental_sparse_bcoo_sum_duplicates(experimental_sparse_BCOO(list(data, indices), shape = shape), nse = nse)
}

bc_set_nse <- function(mat, nse) {
  nse <- as.integer(nse)
  if (mat$nse == nse) return(mat)
  nb <- mat$n_batch
  if (nse <= mat$nse) {
    data <- lax_slice_in_dim(mat$data, 0L, nse, axis = nb)
    indices <- lax_slice_in_dim(mat$indices, 0L, nse, axis = nb)
  } else {
    ds <- sp_shape(mat$data); is <- sp_shape(mat$indices)
    pad_d <- lapply(seq_along(ds), function(k) c(0L, if (k == nb + 1L) nse - mat$nse else 0L, 0L))
    data <- lax_pad(mat$data, lax_full(integer(0), 0, mat$dtype), pad_d)
    fill <- lax_broadcast_in_dim(as_jax(mat$shape[nb + seq_len(mat$n_sparse)], sp_dtype(mat$indices)),
                                 c(is[seq_len(nb)], nse - mat$nse, is[nb + 2L]), nb + 1L)
    indices <- lax_concatenate(list(mat$indices, fill), nb)
  }
  experimental_sparse_BCOO(list(data, indices), shape = mat$shape, indices_sorted = mat$indices_sorted,
                           unique_indices = mat$unique_indices)
}

# ---- layout and structural operations -----------------------------------------------------

sp_efficiency <- function(msg, on_inefficient) {
  if (identical(on_inefficient, "error")) {
    jax_error(msg, " To disable this error, set the on_inefficient argument of bcoo_update_layout to ",
              "'warn' or None.", class = c(experimental_sparse_SparseEfficiencyError, "jax_value_error"))
  } else if (identical(on_inefficient, "warn")) {
    w <- simpleWarning(paste0(msg, " To disable this warning, set the on_inefficient argument of ",
                              "bcoo_update_layout to None."))
    class(w) <- c(experimental_sparse_SparseEfficiencyWarning, class(w))
    warning(w)
  }
}

#' Change the batch and dense dimensions of a BCOO array
#'
#' Port of `jax.experimental.sparse.bcoo_update_layout`.
#' @param mat A BCOO array.
#' @param n_batch,n_dense New numbers of batch and dense dimensions.
#' @param on_inefficient `"error"`, `"warn"` or `NULL`: what to do when the new layout
#'   stores many explicit zeros.
#' @return A BCOO array.
#' @export
experimental_sparse_bcoo_update_layout <- function(mat, n_batch = NULL, n_dense = NULL, on_inefficient = "error") {
  n_batch <- if (is.null(n_batch)) mat$n_batch else as.integer(n_batch)
  n_dense <- if (is.null(n_dense)) mat$n_dense else as.integer(n_dense)
  if (n_batch == mat$n_batch && n_dense == mat$n_dense) return(mat)
  ndim <- mat$ndim
  n_sparse <- ndim - n_batch - n_dense
  if (!is.null(on_inefficient) && !on_inefficient %in% c("error", "warn"))
    jax_error("on_inefficent=", on_inefficient, "; expected one of ['error', 'warn', None].", class = "jax_value_error")
  if (n_batch < 0L) jax_error("n_batch must be non-negative; got ", n_batch, class = "jax_value_error")
  if (n_dense < 0L) jax_error("n_dense must be non-negative; got ", n_dense, class = "jax_value_error")
  if (n_sparse < 0L) jax_error("sum of n_batch=", n_batch, " and n_dense=", n_dense,
                               " cannot be larger than mat.ndim=", ndim, ".", class = "jax_value_error")
  shape <- mat$shape
  if (n_dense > mat$n_dense && any(shape[seq.int(ndim - n_dense + 1L, ndim - mat$n_dense)] > 1L))
    sp_efficiency(paste0("For matrix of shape ", sp_tuple(shape), ", increasing n_dense from ", mat$n_dense,
                         " to ", n_dense, " results in inefficient storage."), on_inefficient)
  if (n_batch > mat$n_batch && any(shape[seq.int(mat$n_batch + 1L, n_batch)] > 1L))
    sp_efficiency(paste0("For matrix of shape ", sp_tuple(shape), ", increasing n_batch from ", mat$n_batch,
                         " to ", n_batch, " results in inefficient storage."), on_inefficient)
  new_data <- mat$data; new_indices <- mat$indices
  cur_nb <- mat$n_batch; cur_nd <- mat$n_dense
  idt <- sp_dtype(new_indices)
  if (n_dense < cur_nd) {
    n <- cur_nd - n_dense
    upd <- function(d, i) {
      dsh <- sp_shape(d)
      new_d <- lax_reshape(d, c(prod_int(dsh[seq_len(n)]), dsh[seq_along(dsh) > n]))
      meshes <- do.call(jnp_meshgrid, c(lapply(dsh[seq_len(n)], function(s) jnp_arange(s, dtype = idt)),
                                        list(indexing = "ij")))
      m <- sp_shape(new_d)[1L]
      new_i <- jnp_column_stack(c(list(jnp_broadcast_to(i, c(m, prod_int(sp_shape(i))))), lapply(meshes, jnp_ravel)))
      list(new_d, new_i)
    }
    r <- sp_nfold_vmap(upd, cur_nb + 1L)(new_data, new_indices)
    ds <- sp_shape(r[[1L]]); is <- sp_shape(r[[2L]])
    new_data <- lax_reshape(r[[1L]], c(ds[seq_len(cur_nb)], prod_int(ds[cur_nb + 1:2]), ds[seq_along(ds) > cur_nb + 2L]))
    new_indices <- lax_reshape(r[[2L]], c(is[seq_len(cur_nb)], prod_int(is[cur_nb + 1:2]), is[seq_along(is) > cur_nb + 2L]))
    cur_nd <- n_dense
  }
  if (n_batch < cur_nb) {
    n <- cur_nb - n_batch
    upd <- function(d, i) {
      is <- sp_shape(i); dsh <- sp_shape(d)
      nse <- is[length(is) - 1L]
      new_d <- lax_reshape(d, c(prod_int(dsh[seq_len(n + 1L)]), dsh[seq_along(dsh) > n + 1L]))
      meshes <- do.call(jnp_meshgrid, c(lapply(c(is[seq_len(n)], nse), function(s) jnp_arange(s, dtype = idt)),
                                        list(indexing = "ij")))
      new_i <- lax_reshape(i, c(prod_int(is[seq_len(n + 1L)]), is[seq_along(is) > n + 1L]))
      new_i <- jnp_column_stack(c(lapply(meshes[seq_len(n)], jnp_ravel), list(new_i)))
      list(new_d, new_i)
    }
    r <- sp_nfold_vmap(upd, n_batch)(new_data, new_indices)
    new_data <- r[[1L]]; new_indices <- r[[2L]]
    cur_nb <- n_batch
  }
  if (n_dense > cur_nd) {
    n <- n_dense - cur_nd
    upd <- function(d, i) {
      ni <- sp_shape(i)[1L]
      idx <- lapply(seq.int(ni - n, ni - 1L), function(k) lax_index_in_dim(i, k, keepdims = FALSE))
      new_d <- do.call(at_set, c(list(lax_full(shape[seq.int(ndim - n_dense + 1L, ndim)], 0, sp_dtype(d))), idx,
                                 list(value = d)))
      list(new_d, lax_slice_in_dim(i, 0L, ni - n))
    }
    r <- sp_nfold_vmap(upd, cur_nb + 1L)(new_data, new_indices)
    new_data <- r[[1L]]; new_indices <- r[[2L]]
    cur_nd <- n_dense
  }
  if (n_batch > cur_nb) {
    n <- n_batch - cur_nb
    upd <- function(d, i) {
      is <- sp_shape(i)
      nse <- is[1L]
      idx <- c(lapply(sp_pos(n), function(j) sp_index_last(i, j)), list(jnp_arange(nse)))
      new_i <- jnp_broadcast_to(lax_slice_in_dim(i, n, is[2L], axis = 1L), c(shape[seq.int(cur_nb + 1L, n_batch)], nse, is[2L] - n))
      dsh <- sp_shape(d)
      new_d <- do.call(at_set, c(list(lax_full(c(shape[seq.int(cur_nb + 1L, n_batch)], nse,
                                                 dsh[seq_along(dsh) > length(dsh) - n_dense]), 0, sp_dtype(d))),
                                 idx, list(value = d)))
      list(new_d, new_i)
    }
    r <- sp_nfold_vmap(upd, cur_nb)(new_data, new_indices)
    new_data <- r[[1L]]; new_indices <- r[[2L]]
    cur_nb <- n_batch
  }
  experimental_sparse_BCOO(list(new_data, new_indices), shape = shape)
}

#' Broadcast a BCOO array
#'
#' Port of `jax.experimental.sparse.bcoo_broadcast_in_dim`.
#' @param mat A BCOO array.
#' @param shape Output shape.
#' @param broadcast_dimensions 0-based output dimension of each input dimension.
#' @param sharding Ignored.
#' @return A BCOO array.
#' @export
experimental_sparse_bcoo_broadcast_in_dim <- function(mat, shape, broadcast_dimensions, sharding = NULL) {
  bufs <- bc_broadcast_in_dim(mat$data, mat$indices, bc_info(mat), as.integer(shape), as.integer(broadcast_dimensions))
  experimental_sparse_BCOO(bufs, shape = shape)
}

bc_broadcast_in_dim <- function(data, indices, spinfo, shape, broadcast_dimensions) {
  if (length(spinfo$shape) != length(broadcast_dimensions))
    jax_error("spinfo.shape=", sp_tuple(spinfo$shape), " and broadcast_dimensions=",
              sp_tuple(broadcast_dimensions), " must have the same length", class = "jax_value_error")
  props <- bc_validate(data, indices, spinfo$shape)
  parts <- sp_split(broadcast_dimensions, c(props$n_batch, props$n_sparse))
  batch_dims <- parts[[1L]]; sparse_dims <- parts[[2L]]; dense_dims <- parts[[3L]]
  mx <- function(x, d) if (length(x)) max(x) else d
  mn <- function(x, d) if (length(x)) min(x) else d
  if (mx(batch_dims, 0L) > mn(sparse_dims, length(shape)))
    jax_error("Cannot mix batch and sparse dimensions during broadcast_in_dim", class = "jax_value_error")
  if (mx(sparse_dims, 0L) > mn(dense_dims, length(shape)))
    jax_error("Cannot mix sparse and dense dimensions during broadcast_in_dim", class = "jax_value_error")
  new_nb <- mn(broadcast_dimensions[seq_along(broadcast_dimensions) > props$n_batch], length(shape))
  new_nd <- if (props$n_dense) length(shape) - min(broadcast_dimensions[seq_along(broadcast_dimensions) >
                                                                           length(broadcast_dimensions) - props$n_dense]) else 0L
  new_ns <- length(shape) - new_nb - new_nd
  if (prod(spinfo$shape[props$n_batch + seq_len(props$n_sparse)]) != prod(shape[new_nb + seq_len(new_ns)]))
    jax_error("Adding sparse dimensions with lengths != 1", class = "jax_not_implemented_error")
  nd <- data; ni <- indices
  if (new_nb != props$n_batch || new_nd != props$n_dense) {
    nd <- lax_broadcast_in_dim(nd, c(shape[seq_len(new_nb)], props$nse, shape[seq_along(shape) > new_nb + new_ns]),
                               c(batch_dims, new_nb, dense_dims + 1L - new_ns))
    ni <- lax_broadcast_in_dim(ni, c(shape[seq_len(new_nb)], props$nse, props$n_sparse),
                               c(batch_dims, new_nb, new_nb + 1L))
  }
  if (new_ns != props$n_sparse) {
    zs <- lax_full(c(shape[seq_len(new_nb)], props$nse, new_ns), 0, sp_dtype(ni))
    cols <- sparse_dims - new_nb
    pieces <- lapply(sp_pos(new_ns), function(j) {
      k <- match(j, cols)
      if (is.na(k)) lax_index_in_dim(zs, j, axis = sp_ndim(zs) - 1L) else lax_index_in_dim(ni, k - 1L, axis = sp_ndim(ni) - 1L)
    })
    ni <- lax_concatenate(pieces, sp_ndim(zs) - 1L)
  }
  list(nd, ni)
}

#' Concatenate BCOO arrays
#'
#' Port of `jax.experimental.sparse.bcoo_concatenate` (along batch or sparse
#' dimensions).
#' @param operands A list of BCOO arrays.
#' @param dimension 0-based dimension.
#' @return A BCOO array.
#' @export
experimental_sparse_bcoo_concatenate <- function(operands, dimension) {
  dimension <- as.integer(dimension)
  if (!all(vapply(operands, inherits, NA, "jax_bcoo")))
    jax_error("bcoo_concatenate: expected operands to be a sequence of BCOO arrays.", class = "jax_value_error")
  out <- do.call(eval_shape, c(list(function(...) lax_concatenate(list(...), dimension)),
                               lapply(operands, function(o) new_aval(o$shape, o$dtype))))
  if (length(unique(vapply(operands, function(o) o$n_dense, 1L))) > 1L)
    jax_error("bcoo_concatenate requires inputs to have matching nse dimensions.", class = "jax_value_error")
  nbs <- unique(vapply(operands, function(o) o$n_batch, 1L))
  if (length(nbs) != 1L && max(nbs) == 1L) {
    if (all(vapply(Filter(function(o) o$n_batch == 0L, operands), function(o) o$shape[1L] == 1L, NA))) {
      operands <- lapply(operands, function(o) if (o$n_batch == 0L) experimental_sparse_bcoo_update_layout(o, n_batch = 1L) else o)
    } else if (all(vapply(Filter(function(o) o$n_batch == 1L, operands), function(o) o$shape[1L] == 1L, NA))) {
      operands <- lapply(operands, function(o) if (o$n_batch == 1L) experimental_sparse_bcoo_update_layout(o, n_batch = 0L) else o)
    }
    nbs <- unique(vapply(operands, function(o) o$n_batch, 1L))
  }
  if (length(nbs) != 1L)
    jax_error("bcoo_concatenate requires inputs to have matching batch dimensions.", class = "jax_value_error")
  nb <- operands[[1L]]$n_batch; ns <- operands[[1L]]$n_sparse
  ib <- lapply(operands, function(o) sp_shape(o$indices)[seq_len(nb)])
  db <- lapply(operands, function(o) sp_shape(o$data)[seq_len(nb)])
  if (dimension < nb) {
    ib <- lapply(ib, function(s) s[-(dimension + 1L)])
    db <- lapply(db, function(s) s[-(dimension + 1L)])
  }
  if (length(unique(ib)) != 1L || length(unique(db)) != 1L)
    jax_error("concatenation of arrays with broadcasted batch indices", class = "jax_not_implemented_error")
  if (dimension < nb) {
    nses <- unique(vapply(operands, function(o) o$nse, 1L))
    if (length(nses) != 1L) operands <- lapply(operands, bc_set_nse, nse = max(nses))
    new_indices <- lax_concatenate(lapply(operands, function(o) o$indices), dimension)
    new_data <- lax_concatenate(lapply(operands, function(o) o$data), dimension)
  } else if (dimension < nb + ns) {
    offsets <- cumsum(c(0L, vapply(operands[-length(operands)], function(o) o$shape[dimension + 1L], 1L)))
    new_data <- lax_concatenate(lapply(operands, function(o) o$data), nb)
    new_indices <- lax_concatenate(Map(function(o, off) {
      k <- dimension - nb
      ind <- o$indices
      r <- sp_ndim(ind)
      col <- lax_add(lax_index_in_dim(ind, k, axis = r - 1L), lax_full(integer(0), off, sp_dtype(ind)))
      parts <- lapply(sp_pos(sp_shape(ind)[r]), function(j) if (j == k) col else lax_index_in_dim(ind, j, axis = r - 1L))
      lax_concatenate(parts, r - 1L)
    }, operands, offsets), nb)
  } else {
    jax_error("Concatenation along dense dimensions.", class = "jax_not_implemented_error")
  }
  experimental_sparse_BCOO(list(new_data, new_indices), shape = out$shape)
}

#' Reshape a BCOO array
#'
#' Port of `jax.experimental.sparse.bcoo_reshape`: the new shape may not mix batch,
#' sparse and dense dimensions.
#' @param mat A BCOO array.
#' @param new_sizes New shape.
#' @param dimensions Optional 0-based permutation applied first.
#' @param sharding Ignored.
#' @return A BCOO array.
#' @export
experimental_sparse_bcoo_reshape <- function(mat, new_sizes, dimensions = NULL, sharding = NULL) {
  new_sizes <- as.integer(new_sizes)
  nb <- mat$n_batch
  if (!identical(sp_shape(mat$indices)[seq_len(nb)], mat$shape[seq_len(nb)]) ||
      !identical(sp_shape(mat$data)[seq_len(nb)], mat$shape[seq_len(nb)]))
    jax_error("reshape of arrays with broadcasted batch dimensions.", class = "jax_not_implemented_error")
  parts <- sp_split(mat$shape, c(nb, mat$n_sparse))
  perms <- bc_validate_permutation(mat$data, mat$indices, dimensions %||% sp_pos(mat$ndim), mat$shape)
  batch_size <- prod(parts[[1L]]); sparse_size <- prod(parts[[2L]])
  cuml <- cumprod(new_sizes)
  if (batch_size != 1 && !batch_size %in% cuml)
    jax_error("bcoo_reshape: new shape cannot mix batch and sparse dimensions; got shape=", sp_tuple(mat$shape),
              " new_shape=", sp_tuple(new_sizes), " with n_batch=", nb, class = "jax_value_error")
  if (sparse_size != 1 && !(batch_size * sparse_size) %in% cuml)
    jax_error("bcoo_reshape: new shape cannot mix sparse and dense dimensions; got shape=", sp_tuple(mat$shape),
              " new_shape=", sp_tuple(new_sizes), " with n_dense=", mat$n_dense, class = "jax_value_error")
  i1 <- sum(cuml <= batch_size)
  i2 <- sum(cuml <= batch_size * sparse_size)
  np <- sp_split(new_sizes, c(i1, i2 - i1))
  nbs <- np[[1L]]; nss <- np[[2L]]; nds <- np[[3L]]
  data <- lax_reshape(mat$data, c(nbs, mat$nse, nds), c(perms$batch, nb, perms$dense + nb + 1L))
  indices <- lax_reshape(mat$indices, c(nbs, mat$nse, mat$n_sparse), c(perms$batch, nb, nb + 1L))
  if (!length(nss)) {
    indices <- lax_full(c(nbs, mat$nse, 0L), 0, sp_dtype(indices))
  } else if (length(parts[[2L]])) {
    cols <- lapply(perms$sparse, function(i) sp_index_last(indices, i))
    sshape <- mat$shape[nb + perms$sparse + 1L]
    flat <- jnp_ravel_multi_index(cols, sshape, mode = "clip")
    ub <- as_jax(mat$shape[nb + seq_len(mat$n_sparse)], sp_dtype(indices))
    oob <- jnp_any(jnp_greater_equal(indices, ub), axis = -1L, keepdims = TRUE)
    new_cols <- jnp_unravel_index(flat, nss)
    indices <- jnp_stack(lapply(new_cols, as_jax, dtype = sp_dtype(indices)), axis = -1L)
    indices <- jnp_where(oob, as_jax(nss, sp_dtype(indices)), indices)
  }
  experimental_sparse_BCOO(list(data, indices), shape = new_sizes)
}

#' Reverse dimensions of a BCOO array
#'
#' Port of `jax.experimental.sparse.bcoo_rev`.
#' @param operand A BCOO array.
#' @param dimensions 0-based dimensions to reverse.
#' @return A BCOO array.
#' @export
experimental_sparse_bcoo_rev <- function(operand, dimensions) {
  dimensions <- as.integer(dimensions)
  eval_shape(function(x) lax_rev(x, dimensions), new_aval(operand$shape, operand$dtype))
  nb <- operand$n_batch; ns <- operand$n_sparse
  bd <- dimensions[dimensions < nb]
  sd <- dimensions[dimensions >= nb & dimensions < nb + ns]
  dd <- dimensions[dimensions >= nb + ns]
  data <- operand$data; indices <- operand$indices
  if (length(bd)) indices <- lax_rev(indices, bd)
  if (length(bd) || length(dd)) data <- lax_rev(data, c(bd, dd + 1L - ns))
  if (length(sd)) {
    r <- sp_ndim(indices)
    cols <- lapply(sp_pos(ns), function(j) {
      col <- lax_index_in_dim(indices, j, axis = r - 1L)
      if (!(j + nb) %in% sd) return(col)
      size <- operand$shape[nb + j + 1L]
      flipped <- lax_sub(lax_full_like(col, size - 1L), col)
      jnp_where(jnp_less(flipped, 0L), lax_full_like(col, size), flipped)
    })
    indices <- lax_concatenate(cols, r - 1L)
  }
  experimental_sparse_BCOO(list(data, indices), shape = operand$shape)
}

#' Remove size one dimensions of a BCOO array
#'
#' Port of `jax.experimental.sparse.bcoo_squeeze`.
#' @param arr A BCOO array.
#' @param dimensions 0-based dimensions of size one.
#' @return A BCOO array.
#' @export
experimental_sparse_bcoo_squeeze <- function(arr, dimensions) {
  dimensions <- norm_axis(dimensions, arr$ndim)
  if (any(arr$shape[dimensions + 1L] != 1L))
    jax_error("cannot select an axis to squeeze out which has size not equal to one, got shape=",
              sp_tuple(arr$shape), " and dimensions=", sp_tuple(dimensions), class = "jax_value_error")
  nb <- arr$n_batch; ns <- arr$n_sparse
  bd <- dimensions[dimensions < nb]
  keep_sparse <- sp_pos(ns)[!(sp_pos(ns) + nb) %in% dimensions]
  dd <- dimensions[dimensions >= nb + ns] - ns + 1L
  data_out <- lax_squeeze(arr$data, c(bd, dd))
  indices_out <- lax_squeeze(sp_take_last(arr$indices, keep_sparse), bd)
  out_shape <- arr$shape[!sp_pos(arr$ndim) %in% dimensions]
  experimental_sparse_BCOO(list(data_out, indices_out), shape = out_shape, indices_sorted = arr$indices_sorted,
                           unique_indices = arr$unique_indices)
}

bc_keep_window <- function(new_indices, new_data, keep, nb, nd) {
  keep_data <- lax_index_in_dim(keep, 0L, axis = sp_ndim(keep) - 1L, keepdims = FALSE)
  if (nd) keep_data <- lax_expand_dims(keep_data, nb + seq_len(nd))
  sp_zero_where(keep_data, new_data)
}

#' Slice a BCOO array
#'
#' Ports of `jax.experimental.sparse.bcoo_slice` (static bounds) and
#' `bcoo_dynamic_slice` (traced start indices, static sizes).
#' @param mat A BCOO array.
#' @param start_indices,limit_indices,strides Static 0-based bounds and strides.
#' @param slice_sizes Static slice sizes.
#' @return A BCOO array.
#' @export
experimental_sparse_bcoo_slice <- function(mat, start_indices, limit_indices, strides = NULL) {
  if (!inherits(mat, "jax_bcoo")) jax_error("bcoo_slice: input should be BCOO array", class = "jax_type_error")
  start <- as.integer(start_indices); limit <- as.integer(limit_indices)
  strides <- if (is.null(strides)) rep(1L, mat$ndim) else as.integer(strides)
  if (length(start) != mat$ndim || length(limit) != mat$ndim)
    jax_error("bcoo_slice: indices must have size mat.ndim=", mat$ndim, class = "jax_value_error")
  if (length(strides) != mat$ndim) jax_error("len(strides) = ", length(strides), "; expected ", mat$ndim,
                                             class = "jax_value_error")
  if (any(strides <= 0L)) jax_error("strides must be a sequence of positive integers; got ", sp_tuple(strides),
                                    class = "jax_value_error")
  if (!all(0L <= start & start <= limit & limit <= mat$shape))
    jax_error("bcoo_slice: invalid indices. Got start_indices=", sp_tuple(start), ", limit_indices=",
              sp_tuple(limit), " and shape=", sp_tuple(mat$shape), class = "jax_value_error")
  nb <- mat$n_batch; ns <- mat$n_sparse; nd <- mat$n_dense
  sp <- function(x) sp_split(x, c(nb, ns))
  st <- sp(start); en <- sp(limit); sr <- sp(strides)
  ds <- sp_shape(mat$data); is <- sp_shape(mat$indices)
  dstart <- integer(0); dlim <- integer(0); dstr <- integer(0)
  istart <- integer(0); ilim <- integer(0); istr <- integer(0)
  for (i in seq_len(nb)) {
    if (ds[i] != mat$shape[i]) { dstart <- c(dstart, 0L); dlim <- c(dlim, ds[i]); dstr <- c(dstr, 1L) }
    else { dstart <- c(dstart, st[[1L]][i]); dlim <- c(dlim, en[[1L]][i]); dstr <- c(dstr, sr[[1L]][i]) }
    if (is[i] != mat$shape[i]) { istart <- c(istart, 0L); ilim <- c(ilim, is[i]); istr <- c(istr, 1L) }
    else { istart <- c(istart, st[[1L]][i]); ilim <- c(ilim, en[[1L]][i]); istr <- c(istr, sr[[1L]][i]) }
  }
  dstart <- c(dstart, 0L, st[[3L]]); dlim <- c(dlim, mat$nse, en[[3L]]); dstr <- c(dstr, 1L, sr[[3L]])
  istart <- c(istart, 0L, 0L); ilim <- c(ilim, mat$nse, ns); istr <- c(istr, 1L, 1L)
  new_data <- lax_slice(mat$data, dstart, dlim, dstr)
  new_indices <- lax_slice(mat$indices, istart, ilim, istr)
  new_shape <- (limit - start + strides - 1L) %/% strides
  if (ns) {
    idt <- sp_dtype(new_indices)
    shp <- c(rep(1L, nb + 1L), ns)
    starts <- as_jax(array(st[[2L]], shp), idt); ends <- as_jax(array(en[[2L]], shp), idt)
    strd <- as_jax(array(sr[[2L]], shp), idt)
    keep <- jnp_all((new_indices >= starts) & (new_indices < ends) &
                      (jnp_remainder(jnp_subtract(new_indices, starts), strd) == 0L), axis = -1L, keepdims = TRUE)
    new_indices <- jnp_where(keep, jnp_floor_divide(jnp_add(jnp_subtract(new_indices, starts), jnp_subtract(strd, 1L)), strd),
                             jnp_floor_divide(jnp_add(jnp_subtract(ends, starts), jnp_subtract(strd, 1L)), strd))
    new_data <- bc_keep_window(new_indices, new_data, keep, nb, nd)
    new_nse <- prod(new_shape[nb + seq_len(ns)])
    if (mat$nse > new_nse) {
      r <- bc_sum_duplicates(new_data, new_indices, sp_spinfo(new_shape), new_nse)
      new_data <- r[[1L]]; new_indices <- r[[2L]]
    }
  }
  experimental_sparse_BCOO(list(new_data, new_indices), shape = new_shape)
}

#' @rdname experimental_sparse_bcoo_slice
#' @export
experimental_sparse_bcoo_dynamic_slice <- function(mat, start_indices, slice_sizes) {
  slice_sizes <- as.integer(slice_sizes)
  if (!inherits(mat, "jax_bcoo")) jax_error("bcoo_slice: input should be BCOO array", class = "jax_type_error")
  start_indices <- lapply(as.list(start_indices), as_jax)
  if (length(start_indices) != mat$ndim || length(slice_sizes) != mat$ndim)
    jax_error("bcoo_dynamic_slice: indices must have size mat.ndim=", mat$ndim, class = "jax_value_error")
  if (!all(0L <= slice_sizes & slice_sizes <= mat$shape))
    jax_error("slice_sizes must be less than or equal to operand shape, got slice_sizes ", sp_tuple(slice_sizes),
              " for operand shape ", sp_tuple(mat$shape), class = "jax_type_error")
  nb <- mat$n_batch; ns <- mat$n_sparse; nd <- mat$n_dense
  sst <- sp_split(start_indices, c(nb, ns)); ssz <- sp_split(slice_sizes, c(nb, ns))
  zero <- as_jax(0L, if (length(start_indices)) sp_dtype(start_indices[[1L]]) else "int32")
  ds <- sp_shape(mat$data); is <- sp_shape(mat$indices)
  dstart <- list(); dsize <- integer(0); istart <- list(); isize <- integer(0)
  for (i in seq_len(nb)) {
    dbc <- ds[i] != mat$shape[i]; ibc <- is[i] != mat$shape[i]
    dstart <- c(dstart, list(if (dbc) zero else sst[[1L]][[i]])); dsize <- c(dsize, if (dbc) 1L else ssz[[1L]][i])
    istart <- c(istart, list(if (ibc) zero else sst[[1L]][[i]])); isize <- c(isize, if (ibc) 1L else ssz[[1L]][i])
  }
  dstart <- c(dstart, list(zero), sst[[3L]]); dsize <- c(dsize, mat$nse, ssz[[3L]])
  istart <- c(istart, list(zero, zero)); isize <- c(isize, mat$nse, ns)
  new_data <- lax_dynamic_slice(mat$data, dstart, dsize)
  new_indices <- lax_dynamic_slice(mat$indices, istart, isize)
  new_shape <- slice_sizes
  if (ns) {
    idt <- sp_dtype(new_indices)
    shp <- c(rep(1L, nb + 1L), ns)
    starts <- jnp_stack(lapply(sst[[2L]], as_jax, dtype = idt))
    sizes <- as_jax(ssz[[2L]], idt)
    sshape <- as_jax(mat$shape[nb + seq_len(ns)], idt)
    starts <- jnp_where(starts < 0L, jnp_add(starts, sshape), starts)
    starts <- jnp_clip(starts, 0L, jnp_subtract(sshape, sizes))
    starts <- lax_reshape(starts, shp)
    sizes <- lax_reshape(sizes, shp)
    keep <- jnp_all((new_indices >= starts) & (new_indices < jnp_add(starts, sizes)), axis = -1L, keepdims = TRUE)
    new_indices <- jnp_where(keep, jnp_subtract(new_indices, starts), sizes)
    new_data <- bc_keep_window(new_indices, new_data, keep, nb, nd)
    if (mat$nse > prod(ssz[[2L]])) {
      r <- bc_sum_duplicates(new_data, new_indices, sp_spinfo(new_shape), prod_int(ssz[[2L]]))
      new_data <- r[[1L]]; new_indices <- r[[2L]]
    }
  }
  experimental_sparse_BCOO(list(new_data, new_indices), shape = new_shape)
}

#' Sum a BCOO array over axes
#'
#' Port of `jax.experimental.sparse.bcoo_reduce_sum`.
#' @param mat A BCOO array.
#' @param axes 0-based axes.
#' @return A BCOO array.
#' @export
experimental_sparse_bcoo_reduce_sum <- function(mat, axes) {
  r <- bc_reduce_sum(mat$data, mat$indices, bc_info(mat), as.integer(axes))
  experimental_sparse_BCOO(r[1:2], shape = r[[3L]])
}

bc_reduce_sum <- function(data, indices, spinfo, axes) {
  shape <- spinfo$shape
  p <- bc_validate(data, indices, shape)
  nb <- p$n_batch; ns <- p$n_sparse
  axes <- sort(unique(axes))
  dense_axes <- axes[axes >= nb + ns] - ns + 1L
  if (length(dense_axes)) data <- jnp_sum(data, axis = dense_axes)
  if (ns) {
    fill <- as_jax(array(shape[nb + seq_len(ns)], c(rep(1L, sp_ndim(indices) - 1L), ns)), sp_dtype(indices))
    mask <- jnp_all(jnp_less(indices, fill), axis = -1L)
    if (sp_ndim(data) > sp_ndim(mask)) mask <- lax_expand_dims(mask, seq.int(sp_ndim(mask), sp_ndim(data) - 1L))
    data <- sp_zero_where(mask, data)
  }
  sparse_idx <- sp_pos(ns)[!(sp_pos(ns) + nb) %in% axes]
  if (!length(sparse_idx)) {
    is <- sp_shape(indices); is[nb + 2L] <- 0L
    indices <- lax_full(is, 0, sp_dtype(indices))
  } else {
    indices <- sp_take_last(indices, sparse_idx)
  }
  batch_axes <- axes[axes < nb]
  for (ax in batch_axes) {
    ds <- sp_shape(data); is <- sp_shape(indices)
    if (ds[ax + 1L] == 1L) {
      if (is[ax + 1L] == 1L) data <- jnp_multiply(data, shape[ax + 1L])
      else { ds[ax + 1L] <- shape[ax + 1L]; data <- lax_broadcast_in_dim(data, ds, sp_pos(length(ds))) }
    } else if (is[ax + 1L] == 1L) {
      data <- jnp_sum(data, axis = ax, keepdims = TRUE)
    }
  }
  ds <- sp_shape(data); is <- sp_shape(indices)
  new_batch_dims <- setdiff(sp_pos(nb), batch_axes)
  new_batch_shape <- ds[new_batch_dims + 1L]
  new_nse <- p$nse * prod_int(ds[batch_axes + 1L])
  data <- lax_reshape(data, c(new_batch_shape, new_nse, ds[seq_along(ds) > nb + 1L]),
                      c(new_batch_dims, batch_axes, seq.int(nb, length(ds) - 1L)))
  indices <- lax_reshape(indices, c(new_batch_shape, new_nse, is[seq_along(is) > nb + 1L]),
                         c(new_batch_dims, batch_axes, seq.int(nb, length(is) - 1L)))
  list(data, indices, shape[!sp_pos(length(shape)) %in% axes])
}

#' Elementwise products with BCOO arrays
#'
#' Ports of `jax.experimental.sparse.bcoo_multiply_sparse` (two BCOO arrays, the result
#' stores the intersection of their indices) and `bcoo_multiply_dense` (the result is the
#' data of a BCOO array with the indices of `sp_mat`).
#' @param lhs,rhs,sp_mat BCOO arrays.
#' @param v A dense array that broadcasts to `sp_mat`'s shape.
#' @return A BCOO array, or the data array for `bcoo_multiply_dense`.
#' @export
experimental_sparse_bcoo_multiply_sparse <- function(lhs, rhs) {
  r <- bc_multiply_sparse(lhs$data, lhs$indices, rhs$data, rhs$indices, bc_info(lhs), bc_info(rhs))
  experimental_sparse_BCOO(r[1:2], shape = r[[3L]])
}

bc_multiply_sparse <- function(lhs_data, lhs_indices, rhs_data, rhs_indices, lhs_spinfo, rhs_spinfo) {
  lhs_shape <- lhs_spinfo$shape; rhs_shape <- rhs_spinfo$shape
  lhs <- bc_validate(lhs_data, lhs_indices, lhs_shape)
  rhs <- bc_validate(rhs_data, rhs_indices, rhs_shape)
  if (length(lhs_shape) != length(rhs_shape))
    jax_error("bcoo_multiply_sparse: arrays must have same number of dimensions, got ", sp_tuple(lhs_shape),
              ", ", sp_tuple(rhs_shape), class = "jax_type_error")
  if (lhs$n_dense != rhs$n_dense)
    jax_error("bcoo_multiply_sparse: arrays with differing numbers of dense dimensions", class = "jax_not_implemented_error")
  nb <- min(lhs$n_batch, rhs$n_batch)
  ls <- lhs_shape[seq_along(lhs_shape) > nb]; rs <- rhs_shape[seq_along(rhs_shape) > nb]
  mul <- sp_nfold_vmap(function(a, b, c, d) bc_multiply_sparse_unbatched(a, b, c, d, ls, rs), nb)
  out <- mul(lhs_data, lhs_indices, rhs_data, rhs_indices)
  list(out[[1L]], out[[2L]], broadcast_shapes_impl(list(lhs_shape, rhs_shape)))
}

bc_multiply_sparse_unbatched <- function(lhs_data, lhs_indices, rhs_data, rhs_indices, lhs_shape, rhs_shape) {
  lhs <- bc_validate(lhs_data, lhs_indices, lhs_shape)
  rhs <- bc_validate(rhs_data, rhs_indices, rhs_shape)
  if (lhs$n_batch) {
    m <- experimental_sparse_bcoo_update_layout(experimental_sparse_BCOO(list(lhs_data, lhs_indices), shape = lhs_shape), n_batch = 0L)
    lhs_data <- m$data; lhs_indices <- m$indices
    lhs <- bc_validate(lhs_data, lhs_indices, lhs_shape)
  } else if (rhs$n_batch) {
    m <- experimental_sparse_bcoo_update_layout(experimental_sparse_BCOO(list(rhs_data, rhs_indices), shape = rhs_shape), n_batch = 0L)
    rhs_data <- m$data; rhs_indices <- m$indices
    rhs <- bc_validate(rhs_data, rhs_indices, rhs_shape)
  }
  k <- min(lhs$n_sparse, rhs$n_sparse)
  dims <- sp_pos(k)[lhs_shape[seq_len(k)] != 1L & rhs_shape[seq_len(k)] != 1L]
  nse <- lhs$nse * rhs$nse
  li <- sp_take_last(lhs_indices, dims); ri <- sp_take_last(rhs_indices, dims)
  mask <- jnp_all(jnp_equal(lax_expand_dims(li, 1L), lax_expand_dims(ri, 0L)), axis = -1L)
  nz <- jnp_nonzero(mask, size = nse, fill_value = list(lhs$nse, rhs$nse))
  i_lhs <- nz[[1L]]; i_rhs <- nz[[2L]]
  data <- jnp_multiply(at_get(lhs_data, i_lhs, mode = "fill", fill_value = 0),
                       at_get(rhs_data, i_rhs, mode = "fill", fill_value = 0))
  mxs <- function(s) if (length(s)) max(s) else 0L
  indices <- jnp_maximum(at_get(lhs_indices, i_lhs, mode = "fill", fill_value = mxs(lhs_shape)),
                         at_get(rhs_indices, i_rhs, mode = "fill", fill_value = mxs(rhs_shape)))
  list(data, indices)
}

#' @rdname experimental_sparse_bcoo_multiply_sparse
#' @export
experimental_sparse_bcoo_multiply_dense <- function(sp_mat, v) {
  bc_multiply_dense(sp_mat$data, sp_mat$indices, as_jax(v), bc_info(sp_mat))
}

bc_multiply_dense <- function(data, indices, v, spinfo) {
  shape <- spinfo$shape
  vs <- sp_shape(v)
  if (!length(vs)) return(jnp_multiply(data, v))
  if (identical(vs, shape)) return(jnp_multiply(data, bc_extract(indices, v)))
  if (!identical(broadcast_shapes_impl(list(vs, shape)), shape))
    jax_error("multiplication between sparse and dense is only implemented for cases where the output shape ",
              "matches the sparse matrix shape. Got shape=", sp_tuple(shape), ", v.shape=", sp_tuple(vs),
              class = "jax_not_implemented_error")
  if (length(shape) > length(vs)) v <- lax_expand_dims(v, sp_pos(length(shape) - length(vs)))
  props <- bc_validate(data, indices, shape)
  mul <- function(data, indices, v) {
    vsh <- sp_shape(v)
    ind <- lapply(sp_pos(sp_shape(indices)[2L]), function(i) {
      col <- lax_index_in_dim(indices, i, axis = 1L, keepdims = FALSE)
      if (vsh[i + 1L] == 1L) jnp_zeros_like(col) else col
    })
    jnp_multiply(data, if (length(ind)) do.call(at_get, c(list(v), ind)) else v)
  }
  sp_nfold_vmap(mul, props$n_batch)(data, indices, v)
}

#' Gather from a BCOO array
#'
#' Port of `jax.experimental.sparse.bcoo_gather` (only `mode = "promise_in_bounds"`).
#' @param operand A BCOO array.
#' @param start_indices,dimension_numbers,slice_sizes As for [lax_gather()].
#' @param unique_indices,indices_are_sorted,mode,fill_value As for [lax_gather()].
#' @return A BCOO array.
#' @export
experimental_sparse_bcoo_gather <- function(operand, start_indices, dimension_numbers, slice_sizes,
                                            unique_indices = FALSE, indices_are_sorted = FALSE,
                                            mode = NULL, fill_value = NULL) {
  bc_validate(operand$data, operand$indices, operand$shape)
  mode <- mode %||% "promise_in_bounds"
  if (lax_GatherScatterMode$from_any(mode) != "promise_in_bounds")
    jax_error("bcoo_gather: mode=", mode, " not yet supported.", class = "jax_not_implemented_error")
  start_indices <- as_jax(start_indices)
  out <- eval_shape(function(x, i) lax_gather(x, i, dimension_numbers, slice_sizes, unique_indices = unique_indices,
                                              indices_are_sorted = indices_are_sorted, mode = mode, fill_value = fill_value),
                    new_aval(operand$shape, operand$dtype), aval_of(start_indices))
  offset_dims <- dimension_numbers$offset_dims
  collapsed <- dimension_numbers$collapsed_slice_dims
  sim <- dimension_numbers$start_index_map
  zero <- as_jax(0L, sp_dtype(start_indices))
  full_start <- rep(list(zero), operand$ndim)
  in_axes <- rep(list(NULL), operand$ndim)
  full_sizes <- operand$shape
  sis <- sp_shape(start_indices)
  for (i in seq_along(sim)) {
    j <- sim[i] + 1L
    full_start[[j]] <- jnp_ravel(lax_index_in_dim(start_indices, i - 1L, axis = length(sis) - 1L, keepdims = FALSE))
    full_sizes[j] <- slice_sizes[j]
    in_axes[j] <- list(0L)
  }
  slice_func <- function(...) {
    slc <- experimental_sparse_bcoo_dynamic_slice(operand, list(...), full_sizes)
    experimental_sparse_bcoo_squeeze(slc, collapsed)
  }
  result <- do.call(vmap(slice_func, in_axes), full_start)
  result <- experimental_sparse_bcoo_reshape(result, c(sis[-length(sis)], result$shape[-1L]),
                                             dimensions = sp_pos(result$ndim))
  if (length(result$shape)) {
    batch_dims <- setdiff(sp_pos(length(out$shape)), offset_dims)
    permutation <- integer(result$ndim)
    permutation[c(batch_dims, offset_dims) + 1L] <- sp_pos(result$ndim)
    if (!setequal(permutation[seq_along(batch_dims)], sp_pos(length(batch_dims))))
      result <- experimental_sparse_bcoo_update_layout(result, n_batch = 0L)
    result <- experimental_sparse_bcoo_transpose(result, permutation)
  }
  result <- experimental_sparse_bcoo_reshape(result, out$shape)
  if (result$dtype != out$dtype) result <- experimental_sparse_BCOO(list(lax_convert_element_type(result$data, out$dtype),
                                                                         result$indices), shape = result$shape)
  result
}

#' Convolution with BCOO operands
#'
#' Port of `jax.experimental.sparse.bcoo_conv_general_dilated`: one dimensional
#' convolutions of operands with leading dimensions `(1, 1)`, unit strides, dilations and
#' group counts.
#' @param lhs,rhs BCOO or dense arrays.
#' @param window_strides,padding,lhs_dilation,rhs_dilation,dimension_numbers,feature_group_count,batch_group_count,precision,preferred_element_type
#'   As for [lax_conv_general_dilated()].
#' @return A BCOO array.
#' @export
experimental_sparse_bcoo_conv_general_dilated <- function(lhs, rhs, window_strides, padding, lhs_dilation = NULL,
                                                          rhs_dilation = NULL, dimension_numbers = NULL,
                                                          feature_group_count = 1L, batch_group_count = 1L,
                                                          precision = NULL, preferred_element_type = NULL) {
  shp <- function(m) if (inherits(m, "jax_bcoo")) m$shape else sp_shape(m)
  dt <- function(m) if (inherits(m, "jax_bcoo")) m$dtype else sp_dtype(m)
  jx <- make_jaxpr(function(a, b) lax_conv_general_dilated(a, b, window_strides, padding, lhs_dilation, rhs_dilation,
                                                           dimension_numbers, feature_group_count, batch_group_count,
                                                           precision, preferred_element_type))(
    ShapeDtypeStruct(shp(lhs), dt(lhs)), ShapeDtypeStruct(shp(rhs), dt(rhs)))
  eqns <- jx$jaxpr$eqns
  if (length(eqns) != 1L) jax_error("internal: expected one conv equation")
  params <- eqns[[1L]]$params
  nl <- length(shp(lhs)) - 2L
  if (!all(unlist(params$lhs_dilation) == 1L)) jax_error("bcoo convolution with lhs_dilation.", class = "jax_not_implemented_error")
  if (!all(unlist(params$rhs_dilation) == 1L)) jax_error("bcoo convolution with lhs_dilation.", class = "jax_not_implemented_error")
  if (!all(unlist(params$window_strides) == 1L)) jax_error("bcoo convolution with non-unit window_strides.",
                                                           class = "jax_not_implemented_error")
  if (params$batch_group_count != 1L || params$feature_group_count != 1L)
    jax_error("bcoo convolution with non-unit group counts.", class = "jax_not_implemented_error")
  if (!identical(shp(lhs)[1:2], c(1L, 1L)) || !identical(shp(rhs)[1:2], c(1L, 1L)))
    jax_error("bcoo convolution with leading dimensions other than (1, 1)", class = "jax_not_implemented_error")
  if (nl != 1L) jax_error("bcoo convolution is only implemented for one spatial dimension",
                          class = "jax_not_implemented_error")
  idt <- if (inherits(lhs, "jax_bcoo")) sp_dtype(lhs$indices) else if (inherits(rhs, "jax_bcoo")) sp_dtype(rhs$indices) else "int32"
  pad <- unlist(params$padding[[1L]])
  bc_conv_1d(bc_to_1d(lhs, idt), bc_to_1d(rhs, idt), pad)
}

bc_to_1d <- function(mat, idt) {
  if (inherits(mat, "jax_bcoo")) {
    m <- experimental_sparse_bcoo_update_layout(mat, n_batch = 2L, n_dense = 0L)
    data <- lax_squeeze(m$data, c(0L, 1L))
    indices <- lax_squeeze(m$indices, c(0L, 1L))
    data <- sp_zero_where(jnp_less(lax_squeeze(indices, 1L), m$shape[length(m$shape)]), data)
    return(experimental_sparse_BCOO(list(data, indices), shape = m$shape[-(1:2)]))
  }
  mat <- as_jax(mat)
  data <- lax_squeeze(mat, c(0L, 1L))
  n <- sp_shape(data)[1L]
  experimental_sparse_BCOO(list(data, lax_broadcasted_iota(idt, c(n, 1L), 0L)), shape = sp_shape(mat)[-(1:2)])
}

bc_conv_1d <- function(lhs, rhs, padding) {
  new_data <- jnp_ravel(jnp_multiply(lax_expand_dims(lhs$data, 1L), lax_expand_dims(rhs$data, 0L)))
  offset <- jnp_subtract(padding[1L], rhs$indices)
  new_indices <- jnp_ravel(jnp_add(lax_expand_dims(lhs$indices, 1L), lax_expand_dims(offset, 0L)))
  mask <- jnp_less(new_indices, 0L)
  new_indices <- jnp_where(mask, 0L, new_indices)
  new_data <- sp_zero_where(lax_bitwise_not(mask), new_data)
  dimsize <- max(0L, lhs$shape[1L] + padding[1L] + padding[2L] - rhs$shape[1L] + 1L)
  experimental_sparse_BCOO(list(lax_expand_dims(new_data, c(0L, 1L)), lax_expand_dims(new_indices, c(0L, 1L, 3L))),
                           shape = c(1L, 1L, dimsize))
}

# ---- the BCOO class -----------------------------------------------------------------------

bc_info <- function(mat) sp_spinfo(mat$shape, mat$indices_sorted, mat$unique_indices)

sp_class <- function(fun, methods, name) {
  structure(fun, class = c("jax_sparse_class", "function"), methods = methods, class_name = name)
}

#' @export
`$.jax_sparse_class` <- function(x, name) {
  m <- attr(x, "methods")[[name]]
  if (is.null(m)) jax_error("type object '", attr(x, "class_name"), "' has no attribute '", name, "'",
                            class = "jax_attribute_error")
  m
}

#' @export
print.jax_sparse_class <- function(x, ...) {
  cat("<class '", attr(x, "class_name"), "'>\n", sep = "")
  invisible(x)
}

bc_new <- function(data, indices, shape, indices_sorted, unique_indices) {
  structure(list(data = data, indices = indices, shape = as.integer(shape),
                 indices_sorted = isTRUE(indices_sorted), unique_indices = isTRUE(unique_indices)),
            class = c("jax_bcoo", "jax_sparse"))
}

bc_empty <- function(shape, dtype = NULL, index_dtype = "int32", n_dense = 0L, n_batch = 0L, nse = 0L) {
  shape <- as.integer(shape)
  n_sparse <- length(shape) - n_dense - n_batch
  if (n_sparse < 0L || n_dense < 0L || n_batch < 0L || nse < 0L)
    jax_error("Invalid inputs: shape=", sp_tuple(shape), ", n_dense=", n_dense, ", n_batch=", n_batch, ", nse=", nse,
              class = "jax_value_error")
  parts <- sp_split(shape, c(n_batch, n_sparse))
  data <- jnp_zeros(c(parts[[1L]], nse, parts[[3L]]), dtype)
  idt <- dtype(index_dtype)
  indices <- jnp_broadcast_to(as_jax(parts[[2L]], idt), c(parts[[1L]], nse, n_sparse))
  bc_new(data, indices, shape, TRUE, TRUE)
}

bc_eye <- function(N, M, k, dtype = NULL, index_dtype = "int32", n_batch = 0L, n_dense = 0L) {
  n_sparse <- 2L - n_batch - n_dense
  if (n_sparse < 0L || n_dense < 0L || n_batch < 0L)
    jax_error("Invalid inputs: shape=", sp_tuple(c(N, M)), ", n_dense=", n_dense, ", n_batch=", n_batch,
              class = "jax_value_error")
  diag_size <- if (k > 0L) min(N, M - k) else min(N + k, M)
  if (diag_size <= 0L) return(bc_empty(c(N, M), dtype, index_dtype, n_batch = n_batch, n_dense = n_dense))
  if (n_dense > 0L || n_batch > 1L)
    return(experimental_sparse_bcoo_fromdense(jnp_eye(N, M, k, dtype = dtype), n_batch = n_batch, n_dense = n_dense,
                                              index_dtype = index_dtype))
  idt <- dtype(index_dtype)
  if (n_batch == 0L) {
    data <- jnp_ones(diag_size, dtype)
    idx <- jnp_arange(diag_size, dtype = idt)
    indices <- jnp_column_stack(list(jnp_subtract(idx, if (k >= 0L) 0L else k),
                                     jnp_add(idx, if (k <= 0L) 0L else k)))
  } else {
    data <- jnp_ones(N, dtype)
    indices <- jnp_add(jnp_arange(N, dtype = idt), k)
    if (k < 0L) {
      data <- at_set(data, sl(NULL, abs(k)), value = 0)
      indices <- at_set(indices, sl(NULL, abs(k)), value = M)
    } else if (k > 0L) {
      data <- at_set(data, sl(M - abs(k), NULL), value = 0)
      indices <- at_set(indices, M - abs(k), value = M)
    }
    data <- lax_expand_dims(data, 1L)
    indices <- lax_expand_dims(indices, c(1L, 2L))
  }
  bc_new(data, indices, c(N, M), TRUE, TRUE)
}

#' Batched coordinate (BCOO) sparse arrays
#'
#' Port of `jax.experimental.sparse.BCOO`. Construct from `list(data, indices)` and a
#' shape, or with `experimental_sparse_BCOO$fromdense(mat, nse, index_dtype, n_dense,
#' n_batch)`. A BCOO array is a pytree (leaves `data` and `indices`) with the properties
#' `$data`, `$indices`, `$shape`, `$dtype`, `$nse`, `$n_batch`, `$n_sparse`, `$n_dense`,
#' `$ndim`, `$size`, `$indices_sorted`, `$unique_indices` and `$T`, and the methods
#' `$todense()`, `$transpose(axes)`, `$sum_duplicates(nse, remove_zeros)`,
#' `$sort_indices()`, `$update_layout(n_batch, n_dense, on_inefficient)`,
#' `$reshape(...)`, `$astype(dtype)`, `$sum(...)` and `$block_until_ready()`. The
#' arithmetic operators, `%*%`, comparisons and `[` work through
#' [experimental_sparse_sparsify()], as the BCOO dunder methods do in JAX.
#' @param args `list(data, indices)`.
#' @param shape Shape of the array.
#' @param indices_sorted,unique_indices Flags describing the indices.
#' @return A BCOO array (class `jax_bcoo`).
#' @export
#' @examples
#' \dontrun{
#' m <- experimental_sparse_BCOO(list(c(1, 2), matrix(c(0L, 1L, 1L, 0L), 2)), shape = c(2, 2))
#' m$todense()
#' m %*% c(1, 1)
#' }
experimental_sparse_BCOO <- sp_class(function(args, shape, indices_sorted = FALSE, unique_indices = FALSE) {
  data <- as_jax(args[[1L]]); indices <- as_jax(args[[2L]])
  shape <- as.integer(shape)
  bc_validate(data, indices, shape)
  bc_new(data, indices, shape, indices_sorted, unique_indices)
}, list(
  fromdense = function(mat, nse = NULL, index_dtype = "int32", n_dense = 0L, n_batch = 0L) {
    experimental_sparse_bcoo_fromdense(mat, nse = nse, n_batch = n_batch, n_dense = n_dense, index_dtype = index_dtype)
  },
  from_scipy_sparse = function(mat, index_dtype = NULL, n_dense = 0L, n_batch = 0L) {
    m <- sp_from_matrix(mat, "TsparseMatrix", "BCOO", n_dense, n_batch)
    idt <- index_dtype %||% default_int()
    experimental_sparse_BCOO(list(jnp_asarray(m$x), as_jax(cbind(m$i, m$j), idt)), shape = m$shape)
  },
  `_empty` = bc_empty,
  `_eye` = function(N, M, k, ...) bc_eye(N, M, k, ...)
), "BCOO")

#' @export
`$.jax_bcoo` <- function(x, name) {
  f <- function(n) .subset2(x, n)
  switch(name,
    data = , indices = , shape = , indices_sorted = , unique_indices = f(name),
    dtype = sp_dtype(f("data")),
    nse = { s <- sp_shape(f("indices")); s[length(s) - 1L] },
    n_batch = sp_ndim(f("indices")) - 2L,
    n_sparse = { s <- sp_shape(f("indices")); s[length(s)] },
    n_dense = sp_ndim(f("data")) - 1L - (sp_ndim(f("indices")) - 2L),
    ndim = length(f("shape")),
    size = prod_int(f("shape")),
    T = sp_bcoo_transpose(x),
    `_info` = bc_info(x),
    `_bufs` = list(f("data"), f("indices")),
    todense = function() experimental_sparse_bcoo_todense(x),
    transpose = function(axes = NULL) sp_bcoo_transpose(x, axes),
    sum_duplicates = function(nse = NULL, remove_zeros = TRUE) {
      if (remove_zeros) bc_eliminate_zeros(x, nse) else experimental_sparse_bcoo_sum_duplicates(x, nse)
    },
    sort_indices = function() experimental_sparse_bcoo_sort_indices(x),
    update_layout = function(n_batch = NULL, n_dense = NULL, on_inefficient = "error")
      experimental_sparse_bcoo_update_layout(x, n_batch, n_dense, on_inefficient),
    reshape = function(...) experimental_sparse_sparsify(function(m) jnp_reshape(m, ...))(x),
    astype = function(dtype) experimental_sparse_sparsify(function(m) jnp_astype(m, dtype))(x),
    sum = function(...) experimental_sparse_sparsify(function(m) jnp_sum(m, ...))(x),
    block_until_ready = function() { block_until_ready(list(f("data"), f("indices"))); x },
    NULL)
}

sp_bcoo_transpose <- function(x, axes = NULL) {
  perm <- if (is.null(axes)) rev(sp_pos(x$ndim)) else as.integer(axes)
  mt <- experimental_sparse_bcoo_transpose(x, perm)
  sparse_perm <- perm[x$n_batch + seq_len(x$n_sparse)] - x$n_batch
  sorted <- if (identical(sparse_perm, sp_pos(x$n_sparse))) x$indices_sorted else FALSE
  bc_new(mt$data, mt$indices, x$shape[perm + 1L], sorted, x$unique_indices)
}

register_pytree_node("jax_bcoo",
  function(x) list(children = list(.subset2(x, "data"), .subset2(x, "indices")),
                   aux = list(shape = .subset2(x, "shape"), indices_sorted = .subset2(x, "indices_sorted"),
                              unique_indices = .subset2(x, "unique_indices"))),
  function(aux, children) bc_new(children[[1L]], children[[2L]], aux$shape, aux$indices_sorted, aux$unique_indices))

#' @export
print.jax_bcoo <- function(x, ...) {
  extra <- paste0(", nse=", x$nse, if (x$n_batch) paste0(", n_batch=", x$n_batch),
                  if (x$n_dense) paste0(", n_dense=", x$n_dense))
  pre <- if (is_tracer(x$data)) "Tracer[" else ""
  cat(pre, "BCOO(", x$dtype, "[", paste(x$shape, collapse = ", "), "]", extra, ")", if (nzchar(pre)) "]", "\n", sep = "")
  invisible(x)
}

#' @export
format.jax_bcoo <- function(x, ...) paste0("BCOO(", x$dtype, "[", paste(x$shape, collapse = ", "), "], nse=", x$nse, ")")

#' @export
dim.jax_sparse <- function(x) x$shape

#' @export
as.array.jax_sparse <- function(x, ...) as.array(x$todense())

#' @export
t.jax_sparse <- function(x) x$T

# from_scipy_sparse: the R counterpart of a scipy.sparse matrix is a sparse matrix of the
# Matrix package. Returns the 0-based slots of its general (not symmetric, triangular or
# diagonal) form in triplet ("TsparseMatrix") or compressed row ("RsparseMatrix") storage.
sp_from_matrix <- function(mat, storage, name, n_dense, n_batch) {
  if (n_dense != 0L || n_batch != 0L)
    jax_error(name, ".from_scipy_sparse with nonzero n_dense/n_batch", class = "jax_not_implemented_error")
  if (!requireNamespace("Matrix", quietly = TRUE) || !methods::is(mat, "sparseMatrix"))
    jax_error(name, ".from_scipy_sparse expects a sparse matrix of the Matrix package, got an ",
              "object of class ", class(mat)[1L], class = "jax_type_error")
  m <- methods::as(methods::as(mat, "generalMatrix"), storage)
  slots <- intersect(c("i", "j", "p", "x"), methods::slotNames(m))
  out <- stats::setNames(lapply(slots, function(s) methods::slot(m, s)), slots)
  # Pattern matrices store no values.
  if (is.null(out$x)) out$x <- rep(TRUE, length(out$j))
  c(out, list(shape = dim(m)))
}

# Ports of _bcoo_to_elt / _bcoo_from_elt and _bcsr_to_elt / _bcsr_from_elt: under vmap
# the mapped axis leaves the shape of a sparse input and enters the shape of a sparse
# output (whose buffers already carry it).
sp_set_shape <- function(x, shape) {
  cl <- oldClass(x)
  x <- unclass(x)
  x$shape <- as.integer(shape)
  structure(x, class = cl)
}
sp_vmappable <- function(name, batched) list(
  to_elt = function(x, axis) {
    if (axis < 0L || axis >= x$n_batch)
      jax_error("Cannot map in_axis=", axis, " for ", name, " array with n_batch=", x$n_batch,
                ". in_axes for batched ", name, " operations must correspond to a ", batched,
                " dimension.", class = "jax_value_error")
    sp_set_shape(x, x$shape[-(axis + 1L)])
  },
  from_elt = function(x, axis_size, axis) {
    n_batch <- x$n_batch - 1L
    if (axis < 0L || axis > n_batch)
      jax_error(name, ": cannot add out_axis=", axis, " for ", name, " array with n_batch=",
                n_batch, ". ", name, " batch axes must be a contiguous block of leading ",
                "dimensions.", class = "jax_value_error")
    sp_set_shape(x, append(x$shape, axis_size, after = axis))
  })
.jx$vmappable$jax_bcoo <- sp_vmappable("BCOO", "batch")
.jx$vmappable$jax_bcsr <- sp_vmappable("BCSR", "batched")
