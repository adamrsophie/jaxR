# jax.numpy set operations (port of jax/_src/numpy/setops.py): unique and its array API
# variants, union1d, intersect1d, setdiff1d, setxor1d and isin. Without `size` the output
# shape depends on the values, so the inputs must be concrete (not traced); with `size`
# (and optionally `fill_value`) every function works under jit and vmap.

# ---- private helpers (`set_` prefix) -----------------------------------------------------------

set_hint <- paste0("To make jnp.unique() compatible with JIT and other transforms, you can ",
                   "specify a concrete value for the size argument, which will determine the ",
                   "output size.")

set_size <- function(size, what) {
  if (is.null(size)) return(NULL)
  if (is_jax_value(size)) size <- as.vector(concrete_or_error(size, what))
  if (!is.numeric(size) || length(size) != 1L || size != round(size))
    jax_error("size must be an integer, got ", format(size), class = "jax_type_error")
  as.integer(size)
}

set_n <- function(x) prod_int(aval_of(x)$shape)

set_arange <- function(n) lax_iota(default_int(), n)

# x[i:j] along axis 0 with Python slice semantics for the bounds used here.
set_slice <- function(x, start, stop) {
  n <- aval_of(x)$shape[1L]
  lax_slice_in_dim(x, max(0L, min(start, n)), max(0L, min(stop, n)), axis = 0L)
}

# Port of jax/_src/numpy/hijax.py:_nonzero_impl for a 1-D input, as used by
# jnp.nonzero(a, size = size, fill_value = fill_value)[0].
set_nonzero <- function(a, size, fill_value = NULL) {
  a <- as_jax(a)
  dt <- if (is.null(fill_value)) default_int() else result_type(default_int(), fill_value)
  n <- set_n(a)
  if (n == 0L || size == 0L) return(lax_full(size, 0, dt))
  mask <- if (aval_of(a)$dtype == "bool") a else lax_ne(a, lax_full_like(a, 0))
  cs_mask <- jnp_cumsum(mask)
  bincount <- at_add(lax_full(size, 0, aval_of(cs_mask)$dtype), cs_mask, value = 1L, mode = "drop")
  flat_indices <- jnp_cumsum(bincount)
  out <- lax_convert_element_type(jnp_remainder(flat_indices, n), dt)
  counts <- red_sum(mask)
  fill_mask <- lax_ge(set_arange(size), lax_convert_element_type(counts, default_int()))
  jnp_where(fill_mask, if (is.null(fill_value)) 0L else jnp_asarray(fill_value, dtype = dt), out)
}

# Indices of the true entries of a concrete boolean vector.
set_concrete_nonzero <- function(mask, what) {
  m <- as.vector(as.array(concrete_or_error(mask, what)))
  # ndmin: one surviving index is still a vector, not a scalar.
  jnp_array(as.integer(which(m) - 1L), dtype = default_int(), ndmin = 1L)
}

set_moveaxis <- function(x, source, destination) {
  r <- length(aval_of(x)$shape)
  if (source == destination || r < 2L) return(x)
  perm <- setdiff(seq_len(r) - 1L, source)
  lax_transpose(x, append(perm, source, after = destination))
}

# Port of setops.py:_unique_sorted_mask.
set_unique_sorted_mask <- function(ar, axis, equal_nan) {
  aux <- set_moveaxis(ar, axis, 0L)
  dt <- aval_of(aux)$dtype
  if (is_complex_dtype(dt)) aux <- jnp_where(red_isnan(aux), lax_full(integer(0), NaN, dt), aux)
  shape <- aval_of(aux)$shape
  size <- shape[1L]
  m <- prod_int(shape[-1L])
  if (m == 0L) {
    size <- min(size, 1L)
    perm <- lax_full(size, 0, default_int())
  } else {
    keys <- lax_rev(lax_transpose(lax_reshape(aux, c(size, m)), c(1L, 0L)), 0L)
    perm <- jnp_lexsort(keys)
  }
  aux <- jnp_take(aux, perm, axis = 0L)
  if (set_n(aux)) {
    a1 <- set_slice(aux, 1L, size)
    a0 <- set_slice(aux, 0L, size - 1L)
    neq <- lax_ne(a1, a0)
    if (is_inexact_dtype(dt) && equal_nan)
      neq <- lax_bitwise_and(neq, lax_bitwise_not(lax_bitwise_and(red_isnan(a1), red_isnan(a0))))
    rest <- red_any(neq, axis = seq_len(length(shape) - 1L))
    mask <- lax_concatenate(list(lax_full(1L, TRUE, "bool"), rest), 0L)
  } else {
    mask <- lax_full(size, FALSE, "bool")
  }
  list(aux = aux, mask = mask, perm = perm)
}

# Port of setops.py:_unique.
set_unique <- function(ar, axis, return_index = FALSE, return_inverse = FALSE,
                       return_counts = FALSE, equal_nan = TRUE, size = NULL, fill_value = NULL,
                       return_true_size = FALSE) {
  shape <- aval_of(ar)$shape
  axis <- norm_axis(axis, length(shape))
  if (shape[axis + 1L] == 0L && isTRUE(size > 0L) && is.null(fill_value))
    jax_error("jnp.unique: for zero-sized input with nonzero size argument, fill_value must be ",
              "specified", class = "jax_value_error")
  sm <- set_unique_sorted_mask(ar, axis, equal_nan)
  aux <- sm$aux
  mask <- sm$mask
  perm <- sm$perm
  int_t <- default_int()
  ind <- if (is.null(size)) set_concrete_nonzero(mask, paste("jnp.unique().", set_hint))
         else set_nonzero(mask, size)
  result <- if (set_n(aux)) jnp_take(aux, ind, axis = 0L) else aux
  if (!is.null(size) && !is.null(fill_value)) {
    rdt <- aval_of(result)$dtype
    fill_value <- lax_convert_element_type(fill_value, rdt)
    rs <- aval_of(result)$shape
    if (rs[1L]) {
      valid <- lax_lt(set_arange(size), lax_convert_element_type(red_sum(mask), int_t))
      valid <- lax_expand_dims(valid, seq_len(length(rs) - 1L))
      result <- jnp_where(valid, result, fill_value)
    } else {
      result <- jnp_broadcast_to(fill_value, c(size, rs[-1L]))
    }
  }
  result <- set_moveaxis(result, 0L, axis)
  ret <- list(result)
  if (return_index)
    ret <- c(ret, list(if (set_n(aux)) jnp_take(perm, ind) else perm))
  if (return_inverse) {
    inv_idx <- if (set_n(aux)) {
      imask <- lax_sub(jnp_cumsum(mask), 1L)
      at_set(lax_full(aval_of(mask)$shape, 0, int_t), perm, value = imask)
    } else lax_full(shape[axis + 1L], 0, int_t)
    ret <- c(ret, list(inv_idx))
  }
  if (return_counts) {
    counts <- if (set_n(aux)) {
      msize <- set_n(mask)
      if (is.null(size)) {
        idx <- jnp_concatenate(list(set_concrete_nonzero(mask, "jnp.unique()"),
                                    as_jax(msize, int_t)), axis = NULL)
      } else {
        idx <- set_nonzero(mask, size + 1L)
        tail <- set_slice(idx, 1L, size + 1L)
        idx <- lax_concatenate(list(set_slice(idx, 0L, 1L),
                                    jnp_where(tail, tail, lax_full_like(tail, msize))), 0L)
      }
      n <- aval_of(idx)$shape[1L]
      lax_sub(set_slice(idx, 1L, n), set_slice(idx, 0L, n - 1L))
    } else if (shape[axis + 1L]) {
      lax_full(1L, shape[axis + 1L], int_t)
    } else {
      lax_full(0L, 0, int_t)
    }
    ret <- c(ret, list(counts))
  }
  if (return_true_size) ret <- c(ret, list(red_sum(mask)))
  if (length(ret) == 1L) ret[[1L]] else ret
}

# Port of setops.py:_in1d.
set_in1d <- function(ar1, ar2, invert, method = "auto", assume_unique = FALSE) {
  arrs <- promote_dtypes(list(ar1, ar2))
  arr1 <- jnp_ravel(arrs[[1L]])
  arr2 <- jnp_ravel(arrs[[2L]])
  n1 <- set_n(arr1)
  if (n1 == 0L || set_n(arr2) == 0L) return(lax_full(n1, invert, "bool"))
  if (identical(method, "auto") || identical(method, "compare_all")) {
    a <- lax_expand_dims(arr1, 1L)
    b <- lax_expand_dims(arr2, 0L)
    if (invert) return(red_all(jnp_not_equal(a, b), axis = -1L))
    return(red_any(jnp_equal(a, b), axis = -1L))
  }
  if (identical(method, "binary_search")) {
    arr2 <- lax_sort(arr2)
    found <- jnp_take(arr2, jnp_searchsorted(arr2, arr1), mode = "clip")
    return(if (invert) lax_ne(arr1, found) else lax_eq(arr1, found))
  }
  if (identical(method, "sort")) {
    ind_out <- NULL
    if (!assume_unique) {
      u <- jnp_unique(arr1, size = n1, return_inverse = TRUE, fill_value = red_max(arr2))
      arr1 <- u[[1L]]
      ind_out <- u[[2L]]
    }
    n <- n1 + set_n(arr2)
    kv <- lax_sort_key_val(jnp_concatenate(list(arr1, arr2)), set_arange(n))
    aux <- kv[[1L]]
    ind <- set_slice(kv[[2L]], 0L, n - 1L)
    a1 <- set_slice(aux, 1L, n)
    a0 <- set_slice(aux, 0L, n - 1L)
    out <- if (invert) at_set(lax_full(n1, TRUE, "bool"), ind, value = lax_ne(a1, a0), mode = "drop")
           else at_set(lax_full(n1, FALSE, "bool"), ind, value = lax_eq(a1, a0), mode = "drop")
    return(if (is.null(ind_out)) out else jnp_take(out, ind_out))
  }
  jax_error("method='", format(method), "' is not implemented; options are 'compare_all', ",
            "'binary_search', 'sort', and 'auto'", class = "jax_value_error")
}

# Port of setops.py:_concat_unique.
set_concat_unique <- function(arr1, arr2) {
  arr1 <- jnp_ravel(arr1)
  arr2 <- jnp_ravel(arr2)
  u1 <- set_unique(arr1, 0L, size = set_n(arr1), return_true_size = TRUE)
  u2 <- set_unique(arr2, 0L, size = set_n(arr2), return_true_size = TRUE)
  arr <- lax_full(set_n(u1[[1L]]) + set_n(u2[[1L]]), 0, result_type(u1[[1L]], u2[[1L]]))
  arr <- lax_dynamic_update_slice(arr, u1[[1L]], list(0L))
  arr <- lax_dynamic_update_slice(arr, u2[[1L]], list(u1[[2L]]))
  list(arr, lax_add(u1[[2L]], u2[[2L]]))
}

# The padding of the size = ... variants (setops.py:_setxor1d_size, _intersect1d_size).
set_pad_results <- function(vals, num_results, fill_value) {
  if (set_n(vals) == 0L) return(vals)
  valid <- lax_lt(set_arange(set_n(vals)), lax_convert_element_type(num_results, default_int()))
  if (!is.null(fill_value)) return(jnp_where(valid, vals, fill_value))
  vals <- jnp_where(valid, vals, red_max(vals))
  jnp_where(valid, vals, red_min(vals))
}

# Port of setops.py:_setxor1d_size.
set_setxor1d_size <- function(arr1, arr2, fill_value, assume_unique, size) {
  if (assume_unique) {
    aux <- lax_sort(jnp_concatenate(list(arr1, arr2)))
    n <- set_n(aux)
    flag <- jnp_concatenate(list(n > 0L, lax_ne(set_slice(aux, 1L, n), set_slice(aux, 0L, n - 1L)),
                                 TRUE), axis = NULL)
  } else {
    cu <- set_concat_unique(arr1, arr2)
    arr <- cu[[1L]]
    num_unique <- cu[[2L]]
    n <- set_n(arr)
    mask <- lax_lt(set_arange(n + 1L), lax_add(lax_convert_element_type(num_unique, default_int()), 1L))
    aux <- lax_sort(list(lax_bitwise_not(set_slice(mask, 1L, n + 1L)), arr), num_keys = 2L)[[2L]]
    inner <- jnp_concatenate(list(n > 0L, lax_ne(set_slice(aux, 1L, n), set_slice(aux, 0L, n - 1L)),
                                  FALSE), axis = NULL)
    flag <- lax_bitwise_and(mask, at_set(inner, num_unique, value = TRUE))
  }
  m <- set_n(flag)
  aux_mask <- lax_bitwise_and(set_slice(flag, 1L, m), set_slice(flag, 0L, m - 1L))
  num_results <- red_sum(aux_mask)
  vals <- if (set_n(aux)) {
    indices <- set_nonzero(aux_mask, size, fill_value = set_n(aux))
    at_get(aux, indices, mode = "fill", fill_value = 0)
  } else lax_full(size, 0, aval_of(aux)$dtype)
  set_pad_results(vals, num_results, fill_value)
}

# Port of setops.py:_intersect1d_sorted_mask.
set_intersect1d_sorted_mask <- function(arr1, arr2, return_indices) {
  arr <- jnp_concatenate(list(arr1, arr2))
  n <- set_n(arr)
  indices <- NULL
  if (return_indices) {
    idx_dtype <- if (n > 2147483647) "int64" else "int32"
    kv <- lax_sort_key_val(arr, lax_iota(idx_dtype, n))
    aux <- kv[[1L]]
    indices <- kv[[2L]]
  } else {
    aux <- lax_sort(arr)
  }
  list(aux, lax_eq(set_slice(aux, 1L, n), set_slice(aux, 0L, n - 1L)), indices)
}

# Port of setops.py:_intersect1d_size.
set_intersect1d_size <- function(arr1, arr2, fill_value, assume_unique, size, return_indices) {
  int_t <- default_int()
  n1 <- set_n(arr1)
  n2 <- set_n(arr2)
  if (assume_unique) {
    ind1 <- set_arange(n1)
    num_unique1 <- lax_full(integer(0), n1, int_t)
    ind2 <- set_arange(n2)
    arr <- jnp_concatenate(list(arr1, arr2))
    s <- lax_sort(list(arr, set_arange(set_n(arr))), num_keys = 1L)
    aux <- s[[1L]]
    aux_sort_indices <- s[[2L]]
    mask <- lax_full(set_n(arr), TRUE, "bool")
  } else {
    u1 <- set_unique(arr1, 0L, size = n1, return_index = TRUE, return_true_size = TRUE, fill_value = 0)
    u2 <- set_unique(arr2, 0L, size = n2, return_index = TRUE, return_true_size = TRUE, fill_value = 0)
    ind1 <- u1[[2L]]
    num_unique1 <- u1[[3L]]
    ind2 <- u2[[2L]]
    arr <- lax_full(n1 + n2, 0, result_type(u1[[1L]], u2[[1L]]))
    arr <- lax_dynamic_update_slice(arr, u1[[1L]], list(0L))
    arr <- lax_dynamic_update_slice(arr, u2[[1L]], list(num_unique1))
    mask <- lax_lt(set_arange(n1 + n2),
                   lax_convert_element_type(lax_add(num_unique1, u2[[3L]]), int_t))
    s <- lax_sort(list(lax_bitwise_not(mask), arr, set_arange(n1 + n2)), num_keys = 2L)
    aux <- s[[2L]]
    aux_sort_indices <- s[[3L]]
  }
  n <- set_n(aux)
  aux_mask <- lax_bitwise_and(lax_eq(set_slice(aux, 1L, n), set_slice(aux, 0L, n - 1L)),
                              set_slice(mask, 1L, n))
  num_results <- red_sum(aux_mask)
  if (n) {
    val_indices <- set_nonzero(aux_mask, size, fill_value = n)
    vals <- at_get(aux, val_indices, mode = "fill", fill_value = 0)
  } else {
    val_indices <- set_arange(0L)
    vals <- lax_full(size, 0, aval_of(aux)$dtype)
  }
  vals <- set_pad_results(vals, num_results, fill_value)
  if (!return_indices) return(vals)
  valid <- function(v) lax_lt(set_arange(set_n(v)), lax_convert_element_type(num_results, default_int()))
  arr1_indices <- at_get(aux_sort_indices, val_indices, mode = "fill", fill_value = n1)
  arr1_indices <- jnp_where(valid(arr1_indices), arr1_indices, n1)
  arr2_indices <- jnp_subtract(at_get(aux_sort_indices, jnp_add(val_indices, 1L), mode = "fill",
                                      fill_value = n2), num_unique1)
  arr2_indices <- jnp_where(valid(arr2_indices), arr2_indices, n2)
  if (!assume_unique) {
    arr1_indices <- at_get(ind1, arr1_indices, mode = "fill", fill_value = set_n(ind1))
    arr2_indices <- at_get(ind2, arr2_indices, mode = "fill", fill_value = set_n(ind2))
  }
  list(vals, arr1_indices, arr2_indices)
}

# ---- public API -------------------------------------------------------------------------------

#' Unique elements of an array
#'
#' Port of `jnp.unique` and of the array API functions `jnp.unique_all`,
#' `jnp.unique_counts`, `jnp.unique_inverse` and `jnp.unique_values`. Without `size`
#' the input must be concrete (outside of `jit`); with a static `size` the result has
#' exactly `size` entries along the unique axis, padded with `fill_value` (by default the
#' smallest unique value) or truncated.
#' @param ar,x Array.
#' @param return_index Also return the 0-based indices of the first occurrences.
#' @param return_inverse Also return the indices that rebuild `ar` from the result.
#' @param return_counts Also return the number of occurrences of each unique value.
#' @param axis 0-based axis along which to find unique slices; `NULL` flattens.
#' @param equal_nan Treat all NaN values as equal.
#' @param size Optional static number of unique values to return.
#' @param fill_value Padding value when `size` exceeds the number of unique values.
#' @param sorted Ignored (results are always sorted).
#' @return An array, or a list of arrays when any `return_*` flag is set.
#'   `unique_all`, `unique_counts` and `unique_inverse` return named lists with the
#'   fields `values`, `indices`, `inverse_indices` and `counts`.
#' @export
jnp_unique <- function(ar, return_index = FALSE, return_inverse = FALSE, return_counts = FALSE,
                       axis = NULL, equal_nan = TRUE, size = NULL, fill_value = NULL,
                       sorted = TRUE) {
  arr <- as_jax(ar)
  size <- set_size(size, paste("the size argument of jnp.unique().", set_hint))
  if (is.null(size))
    arr <- concrete_or_error(arr, paste("the first argument of jnp.unique().", set_hint))
  arr_shape <- aval_of(arr)$shape
  if (is.null(axis)) {
    axis_int <- 0L
    arr <- jnp_ravel(arr)
  } else {
    axis_int <- norm_axis(axis, length(arr_shape))
  }
  result <- set_unique(arr, axis_int, return_index, return_inverse, return_counts,
                       equal_nan = equal_nan, size = size, fill_value = fill_value)
  if (return_inverse && is.null(axis)) {
    idx <- if (return_index) 3L else 2L
    result[[idx]] <- lax_reshape(result[[idx]], arr_shape)
  }
  result
}

#' @rdname jnp_unique
#' @export
jnp_unique_all <- function(x, size = NULL, fill_value = NULL) {
  r <- jnp_unique(x, return_index = TRUE, return_inverse = TRUE, return_counts = TRUE,
                  equal_nan = FALSE, size = size, fill_value = fill_value)
  stats::setNames(r, c("values", "indices", "inverse_indices", "counts"))
}

#' @rdname jnp_unique
#' @export
jnp_unique_counts <- function(x, size = NULL, fill_value = NULL) {
  r <- jnp_unique(x, return_counts = TRUE, equal_nan = FALSE, size = size,
                  fill_value = fill_value)
  stats::setNames(r, c("values", "counts"))
}

#' @rdname jnp_unique
#' @export
jnp_unique_inverse <- function(x, size = NULL, fill_value = NULL) {
  r <- jnp_unique(x, return_inverse = TRUE, equal_nan = FALSE, size = size,
                  fill_value = fill_value)
  stats::setNames(r, c("values", "inverse_indices"))
}

#' @rdname jnp_unique
#' @export
jnp_unique_values <- function(x, size = NULL, fill_value = NULL) {
  jnp_unique(x, equal_nan = FALSE, size = size, fill_value = fill_value)
}

#' Set operations on 1-D arrays
#'
#' Ports of `jnp.union1d`, `jnp.intersect1d`, `jnp.setdiff1d`, `jnp.setxor1d` and
#' `jnp.isin`. Inputs are flattened. Without `size` the inputs must be concrete; with a
#' static `size` the result has exactly `size` entries, padded with `fill_value`.
#' @param ar1,ar2 Arrays.
#' @param assume_unique Whether the inputs are known to have no repeated entries.
#' @param return_indices `intersect1d`: also return the 0-based indices of the common
#'   values in `ar1` and `ar2`.
#' @param size Optional static output size.
#' @param fill_value Padding value when `size` exceeds the number of results.
#' @param element,test_elements `isin`: test each entry of `element` for membership in
#'   `test_elements`.
#' @param invert `isin`: return the negated result.
#' @param method `isin`: `"auto"`, `"compare_all"`, `"binary_search"` or `"sort"`.
#' @return An array (`intersect1d` with `return_indices = TRUE`: a list of three arrays;
#'   `isin`: a boolean array of the shape of `element`).
#' @name jnp_setops
NULL

#' @rdname jnp_setops
#' @export
jnp_union1d <- function(ar1, ar2, size = NULL, fill_value = NULL) {
  ar1 <- as_jax(ar1)
  ar2 <- as_jax(ar2)
  size <- set_size(size, "union1d()")
  if (is.null(size)) {
    ar1 <- concrete_or_error(ar1, "union1d()")
    ar2 <- concrete_or_error(ar2, "union1d()")
  }
  jnp_unique(jnp_concatenate(list(ar1, ar2), axis = NULL), size = size, fill_value = fill_value)
}

#' @rdname jnp_setops
#' @export
jnp_setdiff1d <- function(ar1, ar2, assume_unique = FALSE, size = NULL, fill_value = NULL) {
  arr1 <- jnp_ravel(as_jax(ar1))
  arr2 <- jnp_ravel(as_jax(ar2))
  size <- set_size(size, "setdiff1d()")
  if (is.null(size)) arr1 <- concrete_or_error(arr1, "setdiff1d()")
  fill_value <- lax_full(integer(0), if (is.null(fill_value)) 0 else fill_value,
                         aval_of(arr1)$dtype)
  if (set_n(arr1) == 0L || identical(size, 0L))
    return(jnp_broadcast_to(fill_value, if (is.null(size)) 0L else size))
  if (!assume_unique) arr1 <- jnp_unique(arr1, size = if (is.null(size)) NULL else set_n(arr1))
  mask <- set_in1d(arr1, arr2, invert = TRUE, assume_unique = assume_unique)
  if (is.null(size)) return(arr1[mask])
  n1 <- set_n(arr1)
  if (!assume_unique) {
    n_unique <- lax_sub(lax_full(integer(0), n1 + 1L, default_int()),
                        red_sum(lax_eq(arr1, lax_index_in_dim(arr1, 0L, keepdims = FALSE))))
    mask <- jnp_where(lax_lt(set_arange(n1), n_unique), mask, FALSE)
  }
  vals <- jnp_take(arr1, set_nonzero(mask, size))
  valid <- lax_lt(set_arange(size), lax_convert_element_type(red_sum(mask), default_int()))
  jnp_where(valid, vals, fill_value)
}

#' @rdname jnp_setops
#' @export
jnp_setxor1d <- function(ar1, ar2, assume_unique = FALSE, size = NULL, fill_value = NULL) {
  arrs <- promote_dtypes(list(jnp_ravel(as_jax(ar1)), jnp_ravel(as_jax(ar2))))
  arr1 <- arrs[[1L]]
  arr2 <- arrs[[2L]]
  size <- set_size(size, "setxor1d()")
  if (!is.null(size)) return(set_setxor1d_size(arr1, arr2, fill_value, assume_unique, size))
  if (!assume_unique) {
    arr1 <- jnp_unique(arr1)
    arr2 <- jnp_unique(arr2)
  }
  aux <- jnp_concatenate(list(arr1, arr2))
  n <- set_n(aux)
  if (n == 0L) return(aux)
  aux <- lax_sort(aux)
  flag <- jnp_concatenate(list(TRUE, lax_ne(set_slice(aux, 1L, n), set_slice(aux, 0L, n - 1L)),
                               TRUE), axis = NULL)
  aux[lax_bitwise_and(set_slice(flag, 1L, n + 1L), set_slice(flag, 0L, n))]
}

#' @rdname jnp_setops
#' @export
jnp_intersect1d <- function(ar1, ar2, assume_unique = FALSE, return_indices = FALSE, size = NULL,
                            fill_value = NULL) {
  arrs <- promote_dtypes(list(ar1, ar2))
  arr1 <- jnp_ravel(arrs[[1L]])
  arr2 <- jnp_ravel(arrs[[2L]])
  size <- set_size(size, "intersect1d()")
  if (!is.null(size))
    return(set_intersect1d_size(arr1, arr2, fill_value, assume_unique, size, return_indices))
  ind1 <- ind2 <- NULL
  if (!assume_unique) {
    if (return_indices) {
      u1 <- jnp_unique(arr1, return_index = TRUE)
      u2 <- jnp_unique(arr2, return_index = TRUE)
      arr1 <- u1[[1L]]
      ind1 <- u1[[2L]]
      arr2 <- u2[[1L]]
      ind2 <- u2[[2L]]
    } else {
      arr1 <- jnp_unique(arr1)
      arr2 <- jnp_unique(arr2)
    }
  }
  sm <- set_intersect1d_sorted_mask(arr1, arr2, return_indices)
  aux <- sm[[1L]]
  mask <- sm[[2L]]
  n <- set_n(aux)
  int1d <- set_slice(aux, 0L, n - 1L)[mask]
  if (!return_indices) return(int1d)
  arr1_indices <- set_slice(sm[[3L]], 0L, n - 1L)[mask]
  arr2_indices <- lax_sub(set_slice(sm[[3L]], 1L, n)[mask], set_n(arr1))
  if (!assume_unique) {
    arr1_indices <- jnp_take(ind1, arr1_indices)
    arr2_indices <- jnp_take(ind2, arr2_indices)
  }
  list(int1d, arr1_indices, arr2_indices)
}

#' @rdname jnp_setops
#' @export
jnp_isin <- function(element, test_elements, assume_unique = FALSE, invert = FALSE,
                     method = "auto") {
  element <- as_jax(element)
  result <- set_in1d(element, as_jax(test_elements), invert = invert, method = method,
                     assume_unique = assume_unique)
  lax_reshape(result, aval_of(element)$shape)
}
