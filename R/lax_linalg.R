# lax linear algebra (port of jax/_src/lax/linalg.py).
#
# The CPU lowerings call jaxlib's LAPACK custom call targets (lapack_<x><routine>_ffi)
# as typed FFI calls. The targets are implemented in src/lapack_ffi.cpp on R's LAPACK
# and registered with the XLA runtime right after the plugin loads.

.jx$ffi_hooks$lapack <- function() {
  .jx$lapack_ffi_error <- .Call("jaxr_lapack_register", PACKAGE = "jaxR")
}

# ---- utilities (jax/_src/lax/linalg.py, "Utilities") -------------------------------------

linalg_ndim <- function(x) length(aval_of(x)$shape)

py_shape <- function(s) {
  if (length(s) == 1L) paste0("(", s, ",)") else paste0("(", paste(s, collapse = ", "), ")")
}

# Port of jax/_src/lax/lax.py:_eye.
linalg_eye <- function(dtype, shape, offset = 0L) {
  shape <- as.integer(shape)
  b <- lax_eq(lax_add(lax_broadcasted_iota("int32", shape, 0L),
                      as_jax(as.integer(offset), "int32")),
              lax_broadcasted_iota("int32", shape, 1L))
  lax_convert_element_type(b, dtype)
}

# Port of jax/_src/lax/lax.py:_tri (ones on and below the diagonal `offset`).
linalg_tri <- function(dtype, shape, offset) {
  shape <- as.integer(shape)
  dt <- default_int()
  b <- lax_ge(lax_add(lax_broadcasted_iota(dt, shape, 0L), as_jax(as.integer(offset), dt)),
              lax_broadcasted_iota(dt, shape, 1L))
  lax_convert_element_type(b, dtype)
}

# Port of jax/_src/lax/linalg.py:_T.
linalg_T <- function(x) {
  r <- linalg_ndim(x)
  lax_transpose(x, c(seq_len(r - 2L) - 1L, r - 1L, r - 2L))
}

linalg_conj <- function(x) if (is_complex_dtype(aval_of(x)$dtype)) lax_conj(x) else x

# Port of jax/_src/lax/linalg.py:_H.
linalg_H <- function(x) linalg_conj(linalg_T(x))

# Port of jax/_src/lax/linalg.py:symmetrize.
linalg_symmetrize <- function(x) lax_div(lax_add(x, linalg_H(x)), lax_full_like(x, 2))

# Port of jax/_src/lax/linalg.py:_tril.
linalg_tril <- function(m, k = 0L) {
  s <- aval_of(m)$shape
  mask <- linalg_tri("bool", utils::tail(s, 2L), k)
  lax_select(lax_broadcast(mask, utils::head(s, -2L)), m, lax_full_like(m, 0))
}

# Port of jax/_src/lax/linalg.py:_triu.
linalg_triu <- function(m, k = 0L) {
  s <- aval_of(m)$shape
  mask <- linalg_tri("bool", utils::tail(s, 2L), k - 1L)
  lax_select(lax_broadcast(mask, utils::head(s, -2L)), lax_full_like(m, 0), m)
}

# Port of jax/_src/lax/linalg.py:_construct_diagonal. JAX scatters into a zero matrix;
# here the diagonal is selected with an identity mask (no scatter primitive needed).
construct_diagonal <- function(s) {
  sh <- aval_of(s)$shape
  r <- length(sh)
  n <- sh[r]
  eye <- lax_broadcast(linalg_eye("bool", c(n, n)), sh[-r])
  lax_select(eye, lax_broadcast_in_dim(s, c(sh, n), seq_len(r) - 1L),
             lax_full(c(sh, n), 0, aval_of(s)$dtype))
}

# Port of jax/_src/lax/linalg.py:_extract_diagonal. JAX gathers s[..., i, i]; here the
# diagonal is the row sum of the matrix masked by the identity (no gather needed).
extract_diagonal <- function(s) {
  sh <- aval_of(s)$shape
  r <- length(sh)
  k <- min(sh[r - 1L], sh[r])
  batch <- sh[seq_len(r - 2L)]
  if (sh[r - 1L] != sh[r]) s <- lax_slice(s, rep(0L, r), c(batch, k, k))
  eye <- lax_broadcast(linalg_eye("bool", c(k, k)), batch)
  lax_reduce_sum(lax_select(eye, s, lax_full_like(s, 0)), r - 1L)
}

# Port of jax/_src/lax/linalg.py:_broadcast_to.
linalg_broadcast_to <- function(x, shape) {
  shape <- as.integer(shape)
  r <- linalg_ndim(x)
  lax_broadcast_in_dim(x, shape, length(shape) - r + seq_len(r) - 1L)
}

# Port of jax/_src/lax/lax.py:batch_matmul.
linalg_batch_matmul <- function(lhs, rhs, precision = NULL) {
  ln <- linalg_ndim(lhs)
  rn <- linalg_ndim(rhs)
  if (min(ln, rn) < 2L)
    jax_error("Arguments to batch_matmul must be at least 2D, got ", ln, ", ", rn,
              class = "jax_value_error")
  if (ln != rn)
    jax_error("Arguments to batch_matmul must have same ndim, got ", ln, ", ", rn,
              class = "jax_value_error")
  batch <- seq_len(ln - 2L) - 1L
  lax_dot_general(lhs, rhs, list(list(ln - 1L, rn - 2L), list(batch, batch)), precision)
}

# The `dot` of the JVP rules: lax.dot for matrices, lax.batch_matmul for batches, both
# at the highest precision.
linalg_dot <- function(a, b) linalg_batch_matmul(a, b, precision = "highest")

linalg_astype <- function(x, dtype) lax_convert_element_type(x, dtype)

# Rows of x (shape [..., m, n]) picked by an integer index vector (shape [..., m]):
# x[..., perm, :]; with inverse = TRUE the inverse permutation is applied instead.
# JAX indexes with a gather (and inverts with sort_key_val); the gather and sort
# primitives are not in this file's dependencies, so the permutation is applied as a
# one hot matrix product.
# ponytail: O(m^2 n) instead of O(m n), and 0 * Inf turns non finite entries into NaN
# for the whole column; switch to the gather primitive once lax_slicing provides it.
permute_rows <- function(x, perm, inverse = FALSE) {
  xs <- aval_of(x)$shape
  r <- length(xs)
  m <- xs[r - 1L]
  batch <- xs[seq_len(r - 2L)]
  onehot <- lax_eq(lax_broadcast_in_dim(perm, c(batch, m, m), seq_len(r - 1L) - 1L),
                   lax_broadcasted_iota(aval_of(perm)$dtype, c(batch, m, m), r - 1L))
  p <- lax_convert_element_type(onehot, aval_of(x)$dtype)
  if (inverse) p <- linalg_T(p)
  linalg_batch_matmul(p, x, precision = "highest")
}

# ---- primitive registration helpers ----------------------------------------------------

# Accepted dtype sets of jax/_src/lax/linalg.py (_float, _complex, _any_float, ...).
accept_float_complex <- list(
  ok = function(dt) dt %in% c("float32", "float64", "complex64", "complex128"),
  names = "float32, float64, complex64, complex128")
accept_float <- list(ok = function(dt) dt %in% c("float32", "float64"), names = "float32, float64")
accept_any_inexact <- list(ok = function(dt) dtype_kind(dt) %in% c("f", "c"),
                           names = "floating, complexfloating")
accept_int <- list(ok = function(dt) dtype_kind(dt) %in% c("i", "u"), names = "integer")
accept_int32 <- list(ok = function(dt) dt == "int32", names = "int32")

# Port of jax/_src/lax/lax.py:naryop_dtype_rule.
linalg_dtype_rule <- function(result_dtype, accepted, name, avals, params, require_same) {
  dts <- vapply(avals, `[[`, "", "dtype")
  for (i in seq_along(dts)) {
    if (accepted[[i]]$ok(dts[i])) next
    if (dts[i] == "float0")
      jax_error("Called ", name, " with a float0 at position ", i - 1L, ". ",
                "float0s do not support any operations by design, because they ",
                "are not compatible with non-trivial vector spaces. No implicit dtype ",
                "conversion is done. You can use np.zeros_like(arr, dtype=np.float) ",
                "to cast a float0 array to a regular zeros array. \n",
                "If you didn't expect to get a float0 you might have accidentally ",
                "taken a gradient with respect to an integer argument.",
                class = "jax_type_error")
    jax_error(name, " does not accept dtype ", dts[i], " at position ", i - 1L,
              ". Accepted dtypes at position ", i - 1L, " are subtypes of ",
              accepted[[i]]$names, ".", class = "jax_type_error")
  }
  if (require_same && length(unique(dts)) > 1L)
    jax_error("lax.", name, " requires arguments to have the same dtypes, got ",
              paste(dts, collapse = ", "), ".", class = "jax_type_error")
  do.call(result_dtype, c(as.list(dts), list(params)))
}

# Port of jax/_src/lax/linalg.py:linalg_shape_rule. `result_shape` receives the core
# (non batch) shape of every operand followed by the primitive parameters.
linalg_shape_rule <- function(multiple_results, supports_batching, ranks, result_shape, name,
                              avals, params) {
  batch_dims <- vector("list", length(avals))
  dims <- vector("list", length(avals))
  for (i in seq_along(avals)) {
    rank <- ranks[i]
    shape <- avals[[i]]$shape
    if (length(shape) < rank)
      jax_error("Input ", i - 1L, " to ", name, " must have rank at least ", rank,
                ", but got shape=", py_shape(shape), class = "jax_type_error")
    if (!supports_batching && length(shape) != rank)
      jax_error("Input ", i - 1L, " to ", name, " must have a rank of exactly ", rank,
                ", but got shape=", py_shape(shape), class = "jax_type_error")
    nb <- length(shape) - rank
    batch_dims[[i]] <- shape[seq_len(nb)]
    dims[[i]] <- shape[nb + seq_len(rank)]
  }
  nbs <- vapply(batch_dims, length, 1L)
  if (any(nbs != nbs[1L]))
    jax_error("All inputs to ", name, " must have the same number of batch dimensions, ",
              "but got [", paste(nbs, collapse = ", "), "] batch dimensions for the inputs.",
              class = "jax_type_error")
  batch <- batch_dims[[1L]]
  out <- do.call(result_shape, c(dims, list(params)))
  if (multiple_results) lapply(out, function(d) c(batch, as.integer(d))) else c(batch, as.integer(out))
}

# Port of jax/_src/interpreters/batching.py:expand_dims_batcher: a batching rule for
# primitives that support matching leading batch dimensions in all arguments.
linalg_expand_dims_batcher <- function(prim) {
  function(vals, dims, params, axis) {
    vals <- Map(function(v, d) bdim_at_front(v, d, axis$size), vals, dims)
    out <- bind(prim, vals, params)
    if (prim$multiple_results) list(out, rep(0L, length(out))) else list(out, 0L)
  }
}

linalg_input_dtype <- function(dtype, ...) dtype

# Port of jax/_src/lax/linalg.py:linalg_primitive.
linalg_primitive <- function(result_dtype, accepted_dtypes, ranks, result_shape, name,
                             multiple_results = FALSE, supports_batching = TRUE,
                             require_same = TRUE) {
  prim <- new_primitive(name, multiple_results = multiple_results)
  prim$abstract_eval <- function(avals, params) {
    dts <- linalg_dtype_rule(result_dtype, accepted_dtypes, name, avals, params, require_same)
    shapes <- linalg_shape_rule(multiple_results, supports_batching, ranks, result_shape, name,
                                avals, params)
    weak <- all(vapply(avals, `[[`, NA, "weak"))
    if (!multiple_results) return(new_aval(shapes, dts, weak))
    Map(function(s, d) new_aval(s, d, weak), shapes, as.list(dts))
  }
  if (supports_batching) prim$batch <- linalg_expand_dims_batcher(prim)
  prim
}

standard_linalg_primitive <- function(...) linalg_primitive(linalg_input_dtype, ...)

# ---- FFI lowering helpers ------------------------------------------------------------

# Port of jaxlib/lapack.py:prepare_lapack_call.
prepare_lapack_call <- function(fn_base, dtype) {
  ensure_backend()
  if (nzchar(.jx$lapack_ffi_error %||% ""))
    jax_error("The LAPACK kernels of jaxR could not be registered with the XLA runtime: ",
              .jx$lapack_ffi_error, class = "jax_not_implemented_error")
  prefix <- switch(dtype, float32 = "s", float64 = "d", complex64 = "c", complex128 = "z", NULL)
  if (is.null(prefix))
    jax_error("Unsupported dtype ", dtype, " for LAPACK operations. Supported dtypes are ",
              "float32, float64, complex64, complex128. Please cast the input to a supported type.",
              class = "jax_not_implemented_error")
  paste0("lapack_", prefix, fn_base)
}

# Attribute helpers (jax/_src/lax/linalg.py:_char_attr, _enum_attr and friends).
char_attr <- function(ch) paste0(utf8ToInt(ch), " : ui8")
matrix_side_attr <- function(left_side) char_attr(if (left_side) "L" else "R")
matrix_uplo_attr <- function(lower) char_attr(if (lower) "L" else "U")
matrix_transpose_attr <- function(transpose, conjugate) {
  char_attr(if (transpose) (if (conjugate) "C" else "T") else "N")
}
matrix_diagonal_attr <- function(unit_diag) char_attr(if (unit_diag) "U" else "N")

# Port of jax/_src/lax/linalg.py:_column_major_matrix_layout: the minor to major
# layout of a batch of matrices in Fortran order.
column_major_matrix_layout <- function(dim) c(dim - 2L, dim - 1L, rev(seq_len(dim - 2L)) - 1L)

layout_attr <- function(l) {
  if (!length(l)) return("dense<> : tensor<0xindex>")
  if (length(l) == 1L) return(paste0("dense<", l, "> : tensor<1xindex>"))
  paste0("dense<[", paste(l, collapse = ", "), "]> : tensor<", length(l), "xindex>")
}

# Typed FFI custom call. Port of jax/_src/lax/linalg.py:_linalg_ffi_lowering,
# jax/_src/ffi.py:build_ffi_lowering_function and jax/_src/interpreters/mlir.py:
# custom_call: the attribute dictionary travels in mhlo.backend_config with
# api_version 1, which XLA turns into a typed FFI (api_version 4) call.
#   attrs: named list of MLIR attribute values (the kernel attributes).
#   operand_output_aliases: list of c(operand index, output index), 0-based.
# The sdy.sharding_rule attribute of JAX is not emitted (single device only).
linalg_ffi_call <- function(ctx, target_name, ins, avals_in, avals_out, attrs = list(),
                            operand_output_aliases = NULL, column_major = TRUE,
                            num_non_batch_dims = 2L) {
  ranks_in <- vapply(avals_in, function(a) length(a$shape), 1L)
  max_num_dims <- max(ranks_in)
  layout <- function(aval) {
    r <- length(aval$shape)
    if (column_major && r == max_num_dims) column_major_matrix_layout(r) else rev(seq_len(r)) - 1L
  }
  nres <- length(avals_out)
  props <- c(
    api_version = "1 : i32", backend_config = "\"\"",
    call_target_name = attr_str(target_name), called_computations = "[]",
    has_side_effect = "false",
    operand_layouts = paste0("[", paste(vapply(avals_in, function(a) layout_attr(layout(a)), ""),
                                        collapse = ", "), "]"))
  if (!is.null(operand_output_aliases)) {
    props[["output_operand_aliases"]] <- paste0("[", paste(vapply(operand_output_aliases, function(al) {
      paste0("#stablehlo.output_operand_alias<output_tuple_indices = [",
             if (nres > 1L) al[2L] else "", "], operand_index = ", al[1L],
             ", operand_tuple_indices = []>")
    }, ""), collapse = ", "), "]")
  }
  props[["result_layouts"]] <- paste0(
    "[", paste(vapply(avals_out, function(a) layout_attr(layout(a)), ""), collapse = ", "), "]")
  if (length(attrs)) attrs <- attrs[order(names(attrs))]
  config <- if (length(attrs)) {
    paste0("{", paste(names(attrs), unlist(attrs), sep = " = ", collapse = ", "), "}")
  } else "{}"
  extra <- paste0("{mhlo.backend_config = ", config,
                  ", mhlo.frontend_attributes = {num_batch_dims = \"",
                  max_num_dims - num_non_batch_dims, "\"}}")
  v <- ssa_name(ctx)
  outs <- if (nres == 1L) v else paste0(v, "#", seq_len(nres) - 1L)
  emit_line(ctx, paste0(
    v, if (nres > 1L) paste0(":", nres), " = \"stablehlo.custom_call\"(", paste(ins, collapse = ", "),
    ") <{", paste(names(props), props, sep = " = ", collapse = ", "), "}> ", extra, " : (",
    paste(types_of(avals_in), collapse = ", "), ") -> ", fmt_types(types_of(avals_out))))
  outs
}

# Note for all emitters below: operands are computed into variables before emit_op is
# called. emit_op evaluates its arguments lazily, after it has reserved the result
# name and the output line, so an operand expression that emits ops itself would
# corrupt the module text.
linalg_compare_hlo <- function(ctx, x, y, aval, direction, compare_type = "SIGNED") {
  t <- tensor_type(aval)
  emit_op(ctx, "stablehlo.compare", c(x, y), c(t, t), tensor_type(new_aval(aval$shape, "bool")),
          attrs = list(comparison_direction = paste0("#stablehlo<comparison_direction ", direction, ">"),
                       compare_type = paste0("#stablehlo<comparison_type ", compare_type, ">")))
}

# info == 0 for every matrix of the batch (the usual success test of the lowerings).
info_ok <- function(ctx, info, batch_dims, direction = "EQ") {
  info_aval <- new_aval(batch_dims, "int32")
  zeros <- full_constant(ctx, 0, info_aval)
  linalg_compare_hlo(ctx, info, zeros, info_aval, direction)
}

# Port of jax/_src/lax/linalg.py:_nan_like_hlo.
nan_like_hlo <- function(ctx, aval) {
  if (!is_complex_dtype(aval$dtype)) return(full_constant(ctx, NaN, aval))
  ra <- new_aval(aval$shape, complex_component(aval$dtype))
  re <- full_constant(ctx, NaN, ra)
  emit_op(ctx, "stablehlo.complex", c(re, re), rep(tensor_type(ra), 2L), tensor_type(aval))
}

# Port of jax/_src/lax/linalg.py:_replace_not_ok_with_nan (with
# _broadcasting_select_hlo inlined).
replace_not_ok_with_nan <- function(ctx, batch_dims, ok, x, x_aval) {
  nb <- length(batch_dims)
  select_aval <- new_aval(c(batch_dims, rep(1L, length(x_aval$shape) - nb)), "bool")
  which_aval <- new_aval(x_aval$shape, "bool")
  w <- hlo_broadcast(ctx, ok, new_aval(batch_dims, "bool"), select_aval, seq_len(nb) - 1L)
  w <- hlo_broadcast(ctx, w, select_aval, which_aval, seq_along(x_aval$shape) - 1L)
  t <- tensor_type(x_aval)
  nan <- nan_like_hlo(ctx, x_aval)
  emit_op(ctx, "stablehlo.select", c(w, x, nan), c(tensor_type(which_aval), t, t), t)
}

hlo_conj <- function(ctx, x, aval) {
  if (!is_complex_dtype(aval$dtype)) return(x)
  emit_op(ctx, "chlo.conj", x, tensor_type(aval), tensor_type(aval))
}

# ---- Cholesky decomposition --------------------------------------------------------------

cholesky_shape_rule <- function(shape, params) {
  if (shape[1L] != shape[2L])
    jax_error("The input to cholesky must be a square matrix. Got shape ", py_shape(shape), ".",
              class = "jax_value_error")
  shape
}

cholesky_p <- standard_linalg_primitive(list(accept_float_complex), 2L, cholesky_shape_rule,
                                        "cholesky")

# Port of jax/_src/lax/linalg.py:_cholesky_jvp_rule.
cholesky_p$jvp <- function(primals, tangents, params) {
  x <- primals[[1L]]
  sigma_dot <- tangents[[1L]]
  if (is_complex_dtype(aval_of(sigma_dot)$dtype)) {
    r <- lax_real(sigma_dot)
    i <- lax_imag(sigma_dot)
    sigma_dot <- lax_complex(lax_add(linalg_tril(r), linalg_T(linalg_tril(r, -1L))),
                             lax_sub(linalg_tril(i, -1L), linalg_T(linalg_tril(i, -1L))))
  } else {
    sigma_dot <- lax_add(linalg_tril(sigma_dot), linalg_T(linalg_tril(sigma_dot, -1L)))
  }
  L <- linalg_tril(bind(cholesky_p, list(x)))
  # Forward-mode rule from https://arxiv.org/pdf/1602.07527.pdf
  phi <- function(X) {
    l <- linalg_tril(X)
    a <- aval_of(X)
    n <- utils::tail(a$shape, 1L)
    lax_div(l, lax_expand_dims(lax_add(as_jax(1, a$dtype), linalg_eye(a$dtype, c(n, n))),
                               seq_len(length(a$shape) - 2L) - 1L))
  }
  tmp <- lax_linalg_triangular_solve(L, sigma_dot, left_side = FALSE, transpose_a = TRUE,
                                     conjugate_a = TRUE, lower = TRUE)
  L_dot <- linalg_batch_matmul(L, phi(lax_linalg_triangular_solve(L, tmp, left_side = TRUE,
                                                           transpose_a = FALSE, lower = TRUE)),
                        precision = "highest")
  list(L, L_dot)
}

# Port of jax/_src/lax/linalg.py:_cholesky_cpu_lowering.
cholesky_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  operand_aval <- avals_in[[1L]]
  batch_dims <- utils::head(operand_aval$shape, -2L)
  target_name <- prepare_lapack_call("potrf_ffi", operand_aval$dtype)
  info_aval <- new_aval(batch_dims, "int32")
  res <- linalg_ffi_call(ctx, target_name, ins, avals_in, list(operand_aval, info_aval),
                         attrs = list(uplo = matrix_uplo_attr(TRUE)),
                         operand_output_aliases = list(c(0L, 0L)))
  ok <- info_ok(ctx, res[2L], batch_dims)
  replace_not_ok_with_nan(ctx, batch_dims, ok, res[1L], avals_out[[1L]])
}

# ---- Cholesky update ---------------------------------------------------------------------

cholesky_update_shape_rule <- function(r_shape, w_shape, params) {
  if (r_shape[1L] != r_shape[2L] || w_shape[1L] != r_shape[2L])
    jax_error("Rank-1 update to Cholesky decomposition takes a square matrix ",
              "and a vector of the same size as input. Got shapes ", py_shape(r_shape), " and ",
              py_shape(w_shape), " instead", class = "jax_value_error")
  r_shape
}

# Port of jax/_src/lax/linalg.py:_cholesky_update_jax_fn.
cholesky_update_jax_fn <- function(R, z) {
  dt <- aval_of(R)$dtype
  one_and_zero <- list(as_jax(1, dt), as_jax(0, dt))
  # Coefficients of a Givens rotation, computed in a numerically stable way.
  drotg_nonzero <- function(x, y) {
    abs_x <- lax_abs(x)
    abs_y <- lax_abs(y)
    denominator <- lax_select(lax_gt(abs_x, abs_y), abs_x, abs_y)
    x <- lax_div(x, denominator)
    y <- lax_div(y, denominator)
    rh <- lax_div(one_and_zero[[1L]],
                  lax_sqrt(lax_add(lax_integer_pow(x, 2L), lax_integer_pow(y, 2L))))
    list(lax_mul(x, rh), lax_mul(lax_neg(y), rh))
  }
  drotg <- function(x, y) {
    lax_cond(lax_eq(y, one_and_zero[[2L]]), function(x, y) one_and_zero, drotg_nonzero, x, y)
  }
  n <- aval_of(z)$shape[1L]
  for (k in seq_len(n) - 1L) {
    cs <- drotg(lax_squeeze(lax_slice(R, c(k, k), c(k + 1L, k + 1L)), c(0L, 1L)),
                lax_squeeze(lax_slice(z, k, k + 1L), 0L))
    row <- lax_squeeze(lax_slice(R, c(k, 0L), c(k + 1L, n)), 0L)
    row_k <- lax_sub(lax_mul(cs[[1L]], row), lax_mul(cs[[2L]], z))
    z <- lax_add(lax_mul(cs[[1L]], z), lax_mul(cs[[2L]], row))
    R <- lax_dynamic_update_slice(R, lax_expand_dims(row_k, 0L), c(k, 0L))
  }
  R
}

cholesky_update_p <- standard_linalg_primitive(
  list(accept_float, accept_float), c(2L, 1L), cholesky_update_shape_rule, "cholesky_update",
  supports_batching = FALSE)
cholesky_update_p$lowering <- lower_fun(function(args, params) {
  cholesky_update_jax_fn(args[[1L]], args[[2L]])
})

# ---- General eigendecomposition ------------------------------------------------------------

eig_count <- function(params) {
  isTRUE(params$compute_left_eigenvectors) + isTRUE(params$compute_right_eigenvectors)
}

eig_dtype_rule <- function(a_dtype, params) rep(complex_of(a_dtype), 1L + eig_count(params))

eig_shape_rule <- function(shape, params) {
  if (shape[1L] != shape[2L])
    jax_error("The input to eig must be a square matrix. Got shape ", py_shape(shape), ".",
              class = "jax_value_error")
  c(list(shape[-length(shape)]), rep(list(shape), eig_count(params)))
}

# Port of jax/_src/lax/linalg.py:_eig_compute_attr.
eig_compute_attr <- function(compute) char_attr(if (compute) "V" else "N")

eig_p <- linalg_primitive(eig_dtype_rule, list(accept_float_complex), 2L, eig_shape_rule, "eig",
                          multiple_results = TRUE)

# Port of jax/_src/lax/linalg.py:_eig_cpu_lowering.
eig_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  if (!is.null(params$implementation) && !identical(params$implementation, "lapack"))
    jax_error("Only the lapack implementation is supported on CPU.", class = "jax_value_error")
  operand_aval <- avals_in[[1L]]
  out_aval <- avals_out[[1L]]
  batch_dims <- utils::head(operand_aval$shape, -2L)
  real <- operand_aval$dtype %in% c("float32", "float64")
  eigvals_aval <- new_aval(utils::head(operand_aval$shape, -1L), operand_aval$dtype)
  eigvecs_aval <- new_aval(operand_aval$shape, complex_of(operand_aval$dtype))
  info_aval <- new_aval(batch_dims, "int32")
  ffi_out <- list(eigvals_aval, eigvecs_aval, eigvecs_aval, info_aval)
  if (real) ffi_out <- c(list(eigvals_aval), ffi_out)
  target_name <- prepare_lapack_call("geev_ffi", operand_aval$dtype)
  res <- linalg_ffi_call(
    ctx, target_name, ins, avals_in, ffi_out,
    attrs = list(compute_left = eig_compute_attr(params$compute_left_eigenvectors),
                 compute_right = eig_compute_attr(params$compute_right_eigenvectors)))
  nres <- length(res)
  w <- if (real) {
    emit_op(ctx, "stablehlo.complex", res[1:2], rep(tensor_type(eigvals_aval), 2L),
            tensor_type(out_aval))
  } else res[1L]
  ok <- info_ok(ctx, res[nres], batch_dims)
  output <- replace_not_ok_with_nan(ctx, batch_dims, ok, w, out_aval)
  if (params$compute_left_eigenvectors)
    output <- c(output, replace_not_ok_with_nan(ctx, batch_dims, ok, res[nres - 2L],
                                                avals_out[[length(output) + 1L]]))
  if (params$compute_right_eigenvectors)
    output <- c(output, replace_not_ok_with_nan(ctx, batch_dims, ok, res[nres - 1L],
                                                avals_out[[length(output) + 1L]]))
  output
}

# Port of jax/_src/lax/linalg.py:_eig_vec_jvp. Nondegenerate perturbation theory, see
# Sec 3.1 of https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf. LAPACK's geev
# normalizes each eigenvector to unit 2-norm with a real largest-magnitude component,
# and the rule differentiates through that choice.
eig_vec_jvp <- function(w, v, da) {
  n <- utils::tail(aval_of(w)$shape, 1L)
  nd <- linalg_ndim(v)
  eye_n <- linalg_eye(aval_of(w)$dtype, c(n, n))
  Fmat <- lax_integer_pow(eye_n + lax_expand_dims(w, -2L) - lax_expand_dims(w, -1L), -1L) - eye_n
  P <- linalg_dot(linalg_solve(v, da), v)
  dw <- extract_diagonal(P)
  U <- linalg_dot(v, Fmat * P)
  # dv_j = u_j + c_j v_j with
  #   c_j = -Re(v_j* . u_j) - i Im(u_{k_j j}) / v_{k_j j},  k_j = argmax_i |v_ij|.
  k <- lax_argmax(lax_abs(v), axis = nd - 2L, index_dtype = "int32")
  mask <- linalg_astype(lax_eq(lax_broadcasted_iota("int32", aval_of(v)$shape, nd - 2L),
                               linalg_broadcast_axis(k, aval_of(v)$shape, nd - 2L)),
                        aval_of(v)$dtype)
  cc <- lax_complex(lax_neg(lax_real(jnp_sum(lax_conj(v) * U, axis = -2L))),
                    lax_neg(lax_imag(jnp_sum(mask * U, axis = -2L))) /
                      lax_real(jnp_sum(mask * v, axis = -2L)))
  list(dw, U + v * lax_expand_dims(cc, nd - 2L))
}

# x with a new axis inserted at `axis`, broadcast to `shape`.
linalg_broadcast_axis <- function(x, shape, axis) {
  lax_broadcast_in_dim(x, shape, setdiff(seq_along(shape) - 1L, axis))
}

# Port of jax/_src/lax/linalg.py:eig_jvp_rule.
eig_p$jvp <- function(primals, tangents, params) {
  a <- primals[[1L]]
  da <- tangents[[1L]]
  left <- params$compute_left_eigenvectors
  right <- params$compute_right_eigenvectors
  if ((left || right) && !params$enable_eigvec_derivs)
    jax_error("Derivatives of non-symmetric eigenvectors are only valid under ",
              "assumptions on the input that JAX cannot check (see the ",
              "enable_eigvec_derivs argument to jax.lax.linalg.eig). Pass ",
              "enable_eigvec_derivs=True to jax.lax.linalg.eig to opt in. See ",
              "https://github.com/jax-ml/jax/issues/2748 for discussion.",
              class = "jax_not_implemented_error")
  outs <- lax_linalg_eig(a, compute_left_eigenvectors = left, compute_right_eigenvectors = TRUE,
                         enable_eigvec_derivs = params$enable_eigvec_derivs,
                         implementation = params$implementation)
  w <- outs[[1L]]
  vr <- outs[[length(outs)]]
  da <- linalg_astype(da, aval_of(vr)$dtype)
  if (!(left || right))
    return(list(list(w), list(jnp_sum(linalg_solve(vr, da) * linalg_T(vr), axis = -1L))))
  r <- eig_vec_jvp(w, vr, da)
  primal_out <- list(w)
  tangent_out <- list(r[[1L]])
  if (left) {
    vl <- outs[[2L]]
    primal_out <- c(primal_out, list(vl))
    tangent_out <- c(tangent_out, list(eig_vec_jvp(lax_conj(w), vl, linalg_H(da))[[2L]]))
  }
  if (right) {
    primal_out <- c(primal_out, list(vr))
    tangent_out <- c(tangent_out, list(r[[2L]]))
  }
  list(primal_out, tangent_out)
}

# ---- Symmetric/Hermitian eigendecomposition ---------------------------------------------

full_subset <- function(subset_by_index, n) {
  is.null(subset_by_index) || (subset_by_index[1L] == 0L && subset_by_index[2L] == n)
}

eigh_shape_rule <- function(shape, params) {
  if (shape[1L] != shape[2L])
    jax_error("Argument to symmetric eigendecomposition must have shape [..., n, n], ",
              "got shape ", py_shape(shape), class = "jax_value_error")
  n <- shape[1L]
  sub <- params$subset_by_index
  d <- if (is.null(sub)) n else sub[2L] - sub[1L]
  list(c(n, d), d)
}

eigh_dtype_rule <- function(dtype, params) c(dtype, complex_component(dtype))

eigh_p <- linalg_primitive(eigh_dtype_rule, list(accept_float_complex), 2L, eigh_shape_rule,
                           "eigh", multiple_results = TRUE)

# Port of jax/_src/lax/linalg.py:_eigh_cpu_gpu_lowering (CPU branch). The LAPACK
# implementation always sorts the eigenvalues.
eigh_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  operand_aval <- avals_in[[1L]]
  n <- utils::tail(operand_aval$shape, 1L)
  if (!full_subset(params$subset_by_index, n))
    jax_error("subset_by_index not supported on CPU and GPU", class = "jax_not_implemented_error")
  batch_dims <- utils::head(operand_aval$shape, -2L)
  if (identical(params$algorithm, "qdwh"))
    jax_error("QDWH implementation is only supported on TPU", class = "jax_not_implemented_error")
  if (identical(params$algorithm, "jacobi"))
    jax_error("Jacobi implementation is not supported on CPU", class = "jax_not_implemented_error")
  prefix <- if (is_complex_dtype(operand_aval$dtype)) "he" else "sy"
  target_name <- prepare_lapack_call(paste0(prefix, "evd_ffi"), operand_aval$dtype)
  info_aval <- new_aval(batch_dims, "int32")
  res <- linalg_ffi_call(ctx, target_name, ins, avals_in, c(avals_out, list(info_aval)),
                         attrs = list(mode = char_attr("V"),
                                      uplo = matrix_uplo_attr(params$lower)),
                         operand_output_aliases = list(c(0L, 0L)))
  ok <- info_ok(ctx, res[3L], batch_dims)
  c(replace_not_ok_with_nan(ctx, batch_dims, ok, res[1L], avals_out[[1L]]),
    replace_not_ok_with_nan(ctx, batch_dims, ok, res[2L], avals_out[[2L]]))
}

# Port of jax/_src/lax/linalg.py:_eigh_jvp_rule: the derivative in the simplest case of
# distinct eigenvalues (classic nondegenerate perturbation theory, see
# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf).
eigh_p$jvp <- function(primals, tangents, params) {
  a <- primals[[1L]]
  a_dot <- tangents[[1L]]
  dt <- aval_of(a)$dtype
  n <- utils::tail(aval_of(a)$shape, 1L)
  if (!full_subset(params$subset_by_index, n))
    jax_error("Derivatives not defined for partial eigen decomposition.",
              class = "jax_not_implemented_error")
  vw <- bind(eigh_p, list(linalg_symmetrize(a)), params)
  v <- vw[[1L]]
  w_real <- vw[[2L]]
  # for complex numbers the eigenvalues need the full dtype of v and a
  w <- linalg_astype(w_real, dt)
  eye_n <- linalg_eye(dt, c(n, n))
  # carefully build the reciprocal delta-eigenvalue matrix, avoiding NaNs
  delta_w <- lax_expand_dims(w, -2L) - lax_expand_dims(w, -1L)
  Fmat <- lax_integer_pow(delta_w + eye_n, -1L) - eye_n
  vdag_adot_v <- linalg_dot(linalg_dot(linalg_H(v), a_dot), v)
  dv <- linalg_dot(v, Fmat * vdag_adot_v)
  dw <- extract_diagonal(jnp_real(vdag_adot_v))
  list(list(v, w_real), list(dv, dw))
}

# ---- Hessenberg reduction -----------------------------------------------------------------

hessenberg_shape_rule <- function(shape, params) {
  if (shape[1L] != shape[2L])
    jax_error("Argument to Hessenberg reduction must have shape [..., n, n], ",
              "got shape ", py_shape(shape), class = "jax_value_error")
  list(shape, shape[2L] - 1L)
}

hessenberg_p <- linalg_primitive(function(dtype, params) c(dtype, dtype),
                                 list(accept_float_complex), 2L, hessenberg_shape_rule,
                                 "hessenberg", multiple_results = TRUE)

# Port of jax/_src/lax/linalg.py:_hessenberg_cpu_lowering.
hessenberg_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  a_aval <- avals_in[[1L]]
  batch_dims <- utils::head(a_aval$shape, -2L)
  n <- utils::tail(a_aval$shape, 1L)
  target_name <- prepare_lapack_call("gehrd_ffi", a_aval$dtype)
  res <- linalg_ffi_call(ctx, target_name, ins, avals_in,
                         c(avals_out, list(new_aval(batch_dims, "int32"))),
                         attrs = list(low = attr_i32(1L), high = attr_i32(n)),
                         operand_output_aliases = list(c(0L, 0L)))
  ok <- info_ok(ctx, res[3L], batch_dims)
  c(replace_not_ok_with_nan(ctx, batch_dims, ok, res[1L], avals_out[[1L]]),
    replace_not_ok_with_nan(ctx, batch_dims, ok, res[2L], avals_out[[2L]]))
}

# ---- Householder product -------------------------------------------------------------------

householder_product_shape_rule <- function(a_shape, taus_shape, params) {
  m <- a_shape[1L]
  n <- a_shape[2L]
  if (m < n)
    jax_error("The first argument to householder_product must have at least as many ",
              "rows as columns, got shape ", py_shape(a_shape), class = "jax_value_error")
  if (taus_shape[1L] > min(m, n))
    jax_error("The second argument to householder_product must not have more rows ",
              "than the minimum of the first argument's rows and columns.",
              class = "jax_value_error")
  a_shape
}

householder_product_p <- standard_linalg_primitive(
  list(accept_float_complex, accept_float_complex), c(2L, 1L), householder_product_shape_rule,
  "householder_product")

# Port of jax/_src/lax/linalg.py:_householder_product_cpu_gpu_lowering (CPU branch).
householder_product_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  dt <- avals_in[[1L]]$dtype
  prefix <- if (is_complex_dtype(dt)) "un" else "or"
  target_name <- prepare_lapack_call(paste0(prefix, "gqr_ffi"), dt)
  linalg_ffi_call(ctx, target_name, ins, avals_in, avals_out,
                  operand_output_aliases = list(c(0L, 0L)))
}

# ---- Orthogonal QR multiply ----------------------------------------------------------------

ormqr_shape_rule <- function(a_shape, taus_shape, c_shape, params) {
  m <- a_shape[1L]
  if (params$left && c_shape[1L] != m)
    jax_error("ormqr with left=True expects c to have the same number of rows as ",
              "the Householder matrix a. Got a shape ", py_shape(a_shape), " and c shape ",
              py_shape(c_shape), ".", class = "jax_value_error")
  if (!params$left && c_shape[2L] != m)
    jax_error("ormqr with left=False expects c to have the same number of columns as ",
              "the Householder matrix a has rows. Got a shape ", py_shape(a_shape),
              " and c shape ", py_shape(c_shape), ".", class = "jax_value_error")
  c_shape
}

ormqr_p <- standard_linalg_primitive(
  list(accept_float_complex, accept_float_complex, accept_float_complex), c(2L, 1L, 2L),
  ormqr_shape_rule, "ormqr")

# Port of jax/_src/lax/linalg.py:_ormqr_cpu_gpu_lowering (CPU branch).
ormqr_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  dt <- avals_in[[1L]]$dtype
  prefix <- if (is_complex_dtype(dt)) "un" else "or"
  target_name <- prepare_lapack_call(paste0(prefix, "mqr_ffi"), dt)
  linalg_ffi_call(ctx, target_name, ins, avals_in, avals_out,
                  attrs = list(left = attr_bool(params$left),
                               transpose = attr_bool(params$transpose)),
                  operand_output_aliases = list(c(2L, 0L)))
}

# ---- LU decomposition ----------------------------------------------------------------------

# Computes a pivoted LU decomposition such that PA = LU. In the style of LAPACK, L and
# U are stored in the same matrix.

lu_shape_rule <- function(shape, params) list(shape, min(shape), shape[1L])

lu_p <- linalg_primitive(function(dtype, params) c(dtype, "int32", "int32"),
                         list(accept_float_complex), 2L, lu_shape_rule, "lu",
                         multiple_results = TRUE)

# Port of jax/_src/lax/linalg.py:_lu_jvp_inner, written for batches of matrices (JAX
# maps the matrix rule over the batch dimensions with vmap).
#
# Differentiation of Matrix Functionals Using Triangular Factorization
# F. R. De Hoog, R. S. Anderssen, and M. A. Lukas
#
#     LU = A
# ==> L'U + LU' = A'
# ==> inv(L) . L' + U' . inv(U) = inv(L) A' inv(U)
# ==> L' = L . tril(inv(L) . A' . inv(U), -1)
#     U' = triu(inv(L) . A' . inv(U)) . U
lu_jvp_inner <- function(lu, a_dot, permutation) {
  a <- aval_of(a_dot)
  r <- length(a$shape)
  m <- a$shape[r - 1L]
  n <- a$shape[r]
  k <- min(m, n)
  no_pad <- rep(list(c(0L, 0L, 0L)), r - 2L)
  zero <- as_jax(0, a$dtype)
  l <- lax_pad(linalg_tril(lax_slice_in_dim(lu, 0L, k, axis = -1L), -1L), zero,
               c(no_pad, list(c(0L, 0L, 0L), c(0L, m - k, 0L))))
  l <- l + linalg_eye(a$dtype, c(m, m))
  u_eye <- lax_pad(linalg_eye(a$dtype, c(n - k, n - k)), zero, list(c(k, 0L, 0L), c(k, 0L, 0L)))
  u <- lax_pad(linalg_triu(lax_slice_in_dim(lu, 0L, k, axis = -2L)), zero,
               c(no_pad, list(c(0L, n - k, 0L), c(0L, 0L, 0L)))) + u_eye
  la <- lax_linalg_triangular_solve(l, permute_rows(a_dot, permutation), left_side = TRUE,
                                    transpose_a = FALSE, lower = TRUE, unit_diagonal = TRUE)
  lau <- lax_linalg_triangular_solve(u, la, left_side = FALSE, transpose_a = FALSE, lower = FALSE)
  l_dot <- jnp_matmul(l, linalg_tril(lau, -1L), precision = "highest")
  u_dot <- jnp_matmul(linalg_triu(lau), u, precision = "highest")
  l_dot + u_dot
}

# Port of jax/_src/lax/linalg.py:_lu_jvp_rule.
lu_p$jvp <- function(primals, tangents, params) {
  res <- bind(lu_p, primals)
  lu_dot <- lu_jvp_inner(res[[1L]], tangents[[1L]], res[[3L]])
  list(res, list(lu_dot, zero_of(tangent_aval(aval_of(res[[2L]]))),
                 zero_of(tangent_aval(aval_of(res[[3L]])))))
}

# Port of jax/_src/lax/linalg.py:_lu_cpu_gpu_lowering (CPU branch).
lu_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  operand_aval <- avals_in[[1L]]
  out_aval <- avals_out[[1L]]
  pivot_aval <- avals_out[[2L]]
  perm_aval <- avals_out[[3L]]
  batch_dims <- utils::head(operand_aval$shape, -2L)
  info_aval <- new_aval(batch_dims, "int32")
  m <- operand_aval$shape[length(operand_aval$shape) - 1L]
  target_name <- prepare_lapack_call("getrf_ffi", operand_aval$dtype)
  res <- linalg_ffi_call(ctx, target_name, ins, avals_in, list(out_aval, pivot_aval, info_aval),
                         operand_output_aliases = list(c(0L, 0L)))
  # Subtract 1 from the pivot to get 0-based indices.
  tp <- tensor_type(pivot_aval)
  ones <- full_constant(ctx, 1, pivot_aval)
  pivot <- emit_op(ctx, "stablehlo.subtract", c(res[2L], ones), c(tp, tp), tp)
  ok <- info_ok(ctx, res[3L], batch_dims, "GE")
  lu <- replace_not_ok_with_nan(ctx, batch_dims, ok, res[1L], out_aval)
  perm <- lu_pivots_to_permutation_p$lowering(ctx, pivot, list(pivot_aval), list(perm_aval),
                                              list(permutation_size = m))
  c(lu, pivot, perm)
}

# Port of jax/_src/lax/linalg.py:_lu_solve_core, written for batches (lu [..., m, m],
# permutation [..., m], b [..., m, k]); JAX maps the matrix version with vmap.
lu_solve_core <- function(lu, permutation, b, trans) {
  if (trans == 0L) {
    x <- permute_rows(b, permutation)
    x <- lax_linalg_triangular_solve(lu, x, left_side = TRUE, lower = TRUE, unit_diagonal = TRUE)
    x <- lax_linalg_triangular_solve(lu, x, left_side = TRUE, lower = FALSE)
  } else if (trans == 1L || trans == 2L) {
    conj <- trans == 2L
    x <- lax_linalg_triangular_solve(lu, b, left_side = TRUE, lower = FALSE, transpose_a = TRUE,
                                     conjugate_a = conj)
    x <- lax_linalg_triangular_solve(lu, x, left_side = TRUE, lower = TRUE, unit_diagonal = TRUE,
                                     transpose_a = TRUE, conjugate_a = conj)
    x <- permute_rows(x, permutation, inverse = TRUE)
  } else {
    jax_error("'trans' value must be 0, 1, or 2, got ", trans, class = "jax_value_error")
  }
  x
}

# LU solve with broadcasting. Port of jax/_src/lax/linalg.py:lu_solve and _lu_solve.
lu_solve <- function(lu, permutation, b, trans = 0L) {
  lu <- as_jax(lu)
  permutation <- as_jax(permutation)
  b <- as_jax(b)
  ls <- aval_of(lu)$shape
  bs <- aval_of(b)$shape
  nl <- length(ls)
  if (nl < 2L || ls[nl] != ls[nl - 1L])
    jax_error("last two dimensions of LU decomposition must be equal, got shape ", py_shape(ls),
              class = "jax_value_error")
  if (length(bs) < 1L)
    jax_error("b matrix must have rank >= 1, got shape ", py_shape(bs), class = "jax_value_error")
  # Broadcasting follows NumPy's convention for linalg.solve: the RHS is treated as a
  # (batched) vector if the number of dimensions differ by 1. Otherwise, broadcasting
  # rules apply.
  rhs_vector <- nl == length(bs) + 1L
  if (rhs_vector) {
    if (utils::tail(bs, 1L) != ls[nl])
      jax_error("When LU decomposition matrix and b have the same number of dimensions, ",
                "last axis of LU decomposition matrix (shape ", py_shape(ls),
                ") and b array (shape ", py_shape(bs), ") must match", class = "jax_value_error")
    b <- lax_expand_dims(b, -1L)
    bs <- aval_of(b)$shape
  } else if (length(bs) < 2L || bs[length(bs) - 1L] != ls[nl]) {
    jax_error("When LU decomposition matrix and b different numbers of dimensions, ",
              "last axis of LU decomposition matrix (shape ", py_shape(ls),
              ") and second to last axis of b array (shape ", py_shape(bs), ") must match",
              class = "jax_value_error")
  }
  ps <- aval_of(permutation)$shape
  batch_shape <- broadcast_shapes_impl(list(utils::head(ls, -2L), utils::head(ps, -1L),
                                            utils::head(bs, -2L)))
  lu <- linalg_broadcast_to(lu, c(batch_shape, utils::tail(ls, 2L)))
  permutation <- linalg_broadcast_to(permutation, c(batch_shape, utils::tail(ps, 1L)))
  b <- linalg_broadcast_to(b, c(batch_shape, utils::tail(bs, 2L)))
  x <- lu_solve_core(lu, permutation, b, as.integer(trans))
  if (rhs_vector) lax_squeeze(x, -1L) else x
}

# Support operation for LU decomposition: transformation of the pivots returned by LU
# decomposition into permutations.

lu_pivots_to_permutation_shape_rule <- function(shape, params) {
  if (!(params$permutation_size >= shape[1L]))
    jax_error("Output permutation size ", params$permutation_size, " has to exceed the ",
              "trailing dimension of the pivots. Got pivots size ", shape[1L],
              class = "jax_value_error")
  params$permutation_size
}

lu_pivots_to_permutation_p <- standard_linalg_primitive(
  list(accept_int32), 1L, lu_pivots_to_permutation_shape_rule, "lu_pivots_to_permutation")

# Port of jax/_src/lax/linalg.py:_lu_pivots_body_fn: swap entries i and swaps[i] of the
# permutation. JAX swaps with gather and scatter on one vector and maps the batch
# dimensions with vmap; here the two entries are swapped with masks, which handles
# the batch dimensions directly and needs no gather or scatter primitive.
lu_pivots_body_fn <- function(i, permutation_and_swaps) {
  perm <- permutation_and_swaps[[1L]]
  swaps <- permutation_and_swaps[[2L]]
  full <- aval_of(perm)$shape
  nb <- length(full) - 1L
  batch <- full[seq_len(nb)]
  start <- c(as.list(rep(0L, nb)), list(i))
  j <- lax_dynamic_slice(swaps, start, c(batch, 1L))
  x <- lax_dynamic_slice(perm, start, c(batch, 1L))
  iota <- lax_broadcasted_iota("int32", full, nb)
  at_j <- lax_eq(iota, broadcast_to_shape(j, full))
  at_i <- lax_eq(iota, lax_convert_element_type(i, "int32"))
  y <- lax_expand_dims(lax_reduce_sum(lax_select(at_j, perm, lax_full_like(perm, 0)), nb), nb)
  perm <- lax_select(at_i, broadcast_to_shape(y, full), perm)
  perm <- lax_select(at_j, broadcast_to_shape(x, full), perm)
  list(perm, swaps)
}

# Port of jax/_src/lax/linalg.py:_generic_lu_pivots_to_permutation: converts the pivots
# (row swaps) returned by LU to a permutation. A permutation is built rather than
# applying the swaps to the rows of a matrix because lax loops are not differentiable.
generic_lu_pivots_to_permutation <- function(swaps, permutation_size) {
  s <- aval_of(swaps)$shape
  batch_dims <- utils::head(s, -1L)
  k <- utils::tail(s, 1L)
  m <- permutation_size
  permutation <- lax_broadcasted_iota("int32", c(batch_dims, m), length(batch_dims))
  if (m == 0L || k == 0L) return(permutation)
  lax_fori_loop(as_jax(0L, "int32"), as_jax(k, "int32"), lu_pivots_body_fn,
                list(permutation, swaps))[[1L]]
}

lu_pivots_to_permutation_p$lowering <- lower_fun(function(args, params) {
  generic_lu_pivots_to_permutation(args[[1L]], params$permutation_size)
})

# ---- QR decomposition ----------------------------------------------------------------------

# QR decomposition is implemented as a composition of two lower-level primitives geqrf
# and orgqr. The names are LAPACK's names for the routines.

# Computes the QR decomposition of a matrix: returns (a, taus) where r is in the upper
# triangle of a and q is represented in the lower triangle of a and in taus as
# elementary Householder reflectors. Port of jax/_src/lax/linalg.py:geqrf.
geqrf <- function(a) bind(geqrf_p, list(as_jax(a)))

geqrf_p <- linalg_primitive(function(dtype, params) c(dtype, dtype), list(accept_float_complex),
                            2L, function(shape, params) list(shape, min(shape)), "geqrf",
                            multiple_results = TRUE)

# Port of jax/_src/lax/linalg.py:_geqrf_cpu_gpu_lowering (CPU branch).
geqrf_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  target_name <- prepare_lapack_call("geqrf_ffi", avals_in[[1L]]$dtype)
  linalg_ffi_call(ctx, target_name, ins, avals_in, avals_out,
                  operand_output_aliases = list(c(0L, 0L)))
}

# Computes the column-pivoted QR decomposition of a matrix: returns (a, jpvt, taus)
# with a[:, jpvt] = q @ r. Port of jax/_src/lax/linalg.py:geqp3.
geqp3 <- function(a, jpvt, use_magma = NULL) {
  bind(geqp3_p, list(as_jax(a), as_jax(jpvt)), list(use_magma = use_magma))
}

geqp3_p <- linalg_primitive(
  function(dtype, jpvt_dtype, params) c(dtype, jpvt_dtype, dtype),
  list(accept_float_complex, accept_int), c(2L, 1L),
  function(a_shape, jpvt_shape, params) list(a_shape, jpvt_shape, min(a_shape)), "geqp3",
  multiple_results = TRUE, require_same = FALSE)

# Port of jax/_src/lax/linalg.py:_geqp3_cpu_gpu_lowering (CPU branch).
geqp3_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  target_name <- prepare_lapack_call("geqp3_ffi", avals_in[[1L]]$dtype)
  linalg_ffi_call(ctx, target_name, ins, avals_in, avals_out,
                  operand_output_aliases = list(c(0L, 0L), c(1L, 1L)))
}

qr_shape_rule <- function(shape, params) {
  m <- shape[1L]
  n <- shape[2L]
  k <- if (params$full_matrices) m else min(m, n)
  if (params$pivoting) list(c(m, k), c(k, n), n) else list(c(m, k), c(k, n))
}

qr_dtype_rule <- function(dtype, params) {
  if (params$pivoting) c(dtype, dtype, "int32") else c(dtype, dtype)
}

qr_p <- linalg_primitive(qr_dtype_rule, list(accept_float_complex), 2L, qr_shape_rule, "qr",
                         multiple_results = TRUE)

# x[..., p]: columns picked by the index vectors p (shape [..., n]).
permute_cols <- function(x, p) linalg_T(permute_rows(linalg_T(x), p))

# JVP for QR decompositions of [..., m, n] matrices with m >= n. See
# j-towns.github.io/papers/qr-derivative.pdf for a terse derivation.
# Port of jax/_src/lax/linalg.py:_thin_qr_jvp.
thin_qr_jvp <- function(q, r, dx) {
  dx_rinv <- lax_linalg_triangular_solve(r, dx)  # Right side solve by default
  qt_dx_rinv <- linalg_H(q) %*% dx_rinv
  qt_dx_rinv_lower <- linalg_tril(qt_dx_rinv, -1L)
  do <- qt_dx_rinv_lower - linalg_H(qt_dx_rinv_lower)  # This is skew-symmetric
  # The following correction is necessary for complex inputs
  n <- utils::tail(aval_of(r)$shape, 1L)
  dt <- aval_of(do)$dtype
  I <- lax_expand_dims(linalg_eye(dt, c(n, n)), seq_len(linalg_ndim(qt_dx_rinv) - 2L) - 1L)
  do <- do + I * (qt_dx_rinv - linalg_astype(jnp_real(qt_dx_rinv), dt))
  dq <- q %*% (do - qt_dx_rinv) + dx_rinv
  dr <- (qt_dx_rinv - do) %*% r
  list(dq, dr)
}

# Port of jax/_src/lax/linalg.py:qr_jvp_rule.
qr_p$jvp <- function(primals, tangents, params) {
  x <- primals[[1L]]
  dx <- tangents[[1L]]
  res <- bind(qr_p, list(x), params)
  q <- res[[1L]]
  r <- res[[2L]]
  s <- aval_of(x)$shape
  nd <- length(s)
  m <- s[nd - 1L]
  n <- s[nd]
  pivoting <- params$pivoting
  if (pivoting) dx <- permute_cols(dx, res[[3L]])
  finish <- function(dq, dr) {
    if (pivoting) return(list(res, list(dq, dr, zero_of(tangent_aval(aval_of(res[[3L]]))))))
    list(res, list(dq, dr))
  }
  cols <- function(z, from, to) lax_slice_in_dim(z, from, to, axis = -1L)
  rows <- function(z, from, to) lax_slice_in_dim(z, from, to, axis = -2L)
  if (m < n) {
    # Wide matrix case: partition X = [X1, X2] and R = [R1, R2] with square X1, R1.
    # X1 = Q R1 is the QR decomposition of X1, so dQ and dR1 follow from the square
    # rule; R2 = Q^H X2 gives dR2 = (dQ)^H X2 + Q^H dX2.
    x2 <- cols(x, m, n)
    dx1 <- cols(dx, 0L, m)
    dx2 <- cols(dx, m, n)
    r1 <- cols(r, 0L, m)
    d1 <- thin_qr_jvp(q, r1, dx1)
    dq <- d1[[1L]]
    dr2 <- linalg_H(dq) %*% x2 + linalg_H(q) %*% dx2
    return(finish(dq, lax_concatenate(list(d1[[2L]], dr2), nd - 1L)))
  }
  full_matrices <- params$full_matrices
  q_mn <- if (full_matrices) cols(q, 0L, n) else q
  r_nn <- if (full_matrices) cols(rows(r, 0L, n), 0L, n) else r
  d <- thin_qr_jvp(q_mn, r_nn, dx)
  dq_mn <- d[[1L]]
  dr_nn <- d[[2L]]
  if (!full_matrices || m == n) return(finish(dq_mn, dr_nn))
  # See https://arxiv.org/pdf/2409.13374 for the full matrices derivation.
  q_nn <- rows(q_mn, 0L, n)
  q_pn <- rows(q_mn, n, m)
  q_mp <- cols(q, n, m)
  dq_nn <- rows(dq_mn, 0L, n)
  dq_pn <- rows(dq_mn, n, m)
  # Eq 33: Z_pn = Q_pn @ (Q_nn - I)^{-1}, computed with a linear solve:
  # (Q_nn - I)^H @ Z_pn^H = Q_pn^H  =>  Z_pn^H = solve((Q_nn - I)^H, Q_pn^H)
  dt <- aval_of(q_nn)$dtype
  I_n <- lax_expand_dims(linalg_eye(dt, c(n, n)), seq_len(nd - 2L) - 1L)
  f <- lax_linalg_lu(q_nn - I_n)
  z_pn_H <- lu_solve(f[[1L]], f[[3L]], linalg_H(q_pn), trans = 2L)  # solves A^H X = B
  z_pn <- linalg_H(z_pn_H)
  # Eq 32: dq_mp = dq_mn @ Z_pn^H - (Q_mn + Q_mp @ Z_pn) @ (dq_pn - Z_pn @ dq_nn)^H
  dq_mp <- dq_mn %*% z_pn_H - (q_mn + q_mp %*% z_pn) %*% linalg_H(dq_pn - z_pn %*% dq_nn)
  dq <- lax_concatenate(list(dq_mn, dq_mp), nd - 1L)
  zeros <- lax_full(c(utils::head(aval_of(dr_nn)$shape, -2L), m - n, n), 0, aval_of(dr_nn)$dtype)
  finish(dq, lax_concatenate(list(dr_nn, zeros), nd - 2L))
}

# Port of jax/_src/lax/linalg.py:_qr_lowering.
qr_lowering <- function(a, params) {
  s <- aval_of(a)$shape
  dt <- aval_of(a)$dtype
  nd <- length(s)
  batch_dims <- utils::head(s, -2L)
  nb <- length(batch_dims)
  m <- s[nd - 1L]
  n <- s[nd]
  if (m == 0L || n == 0L) {
    k <- if (params$full_matrices) m else min(m, n)
    q <- lax_broadcast_in_dim(linalg_eye(dt, c(m, k)), c(batch_dims, m, k), c(nb, nb + 1L))
    r <- lax_full(c(batch_dims, k, n), 0, dt)
    if (params$pivoting) return(list(q, r, lax_full(c(batch_dims, n), 0, "int32")))
    return(list(q, r))
  }
  p <- NULL
  if (params$pivoting) {
    jpvt <- lax_full(c(batch_dims, n), 0, "int32")
    f <- geqp3(a, jpvt, use_magma = params$use_magma)
    r <- f[[1L]]
    # Convert geqp3's 1-based indices to 0-based indices by subtracting 1.
    p <- lax_sub(f[[2L]], lax_full_like(f[[2L]], 1))
    taus <- f[[3L]]
  } else {
    f <- geqrf(a)
    r <- f[[1L]]
    taus <- f[[2L]]
  }
  if (m < n) {
    q <- lax_linalg_householder_product(lax_slice_in_dim(r, 0L, m, axis = -1L), taus)
  } else if (params$full_matrices) {
    pads <- c(rep(list(c(0L, 0L, 0L)), nb + 1L), list(c(0L, m - n, 0L)))
    q <- lax_linalg_householder_product(lax_pad(r, as_jax(0, dt), pads), taus)
  } else {
    q <- lax_linalg_householder_product(r, taus)
    r <- lax_slice_in_dim(r, 0L, n, axis = -2L)
  }
  r <- linalg_triu(r)
  if (params$pivoting) list(q, r, p) else list(q, r)
}

qr_p$lowering <- lower_fun(function(args, params) qr_lowering(args[[1L]], params),
                           multiple_results = TRUE)

# ---- Schur decomposition -------------------------------------------------------------------

schur_shape_rule <- function(shape, params) {
  if (shape[1L] != shape[2L])
    jax_error("The input to schur must be a square matrix. Got shape ", py_shape(shape), ".",
              class = "jax_value_error")
  if (params$compute_schur_vectors) list(shape, shape) else list(shape)
}

schur_p <- linalg_primitive(
  function(dtype, params) if (params$compute_schur_vectors) c(dtype, dtype) else dtype,
  list(accept_float_complex), 2L, schur_shape_rule, "schur", multiple_results = TRUE)

# Port of jax/_src/lax/linalg.py:_schur_cpu_lowering.
schur_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  if (isTRUE(params$sort_eig_vals))
    jax_error("The sort feature of LAPACK's gees routine is not implemented.",
              class = "jax_not_implemented_error")
  operand_aval <- avals_in[[1L]]
  batch_dims <- utils::head(operand_aval$shape, -2L)
  real <- operand_aval$dtype %in% c("float32", "float64")
  target_name <- prepare_lapack_call("gees_ffi", operand_aval$dtype)
  info_aval <- new_aval(batch_dims, "int32")
  eigvals_aval <- new_aval(utils::head(operand_aval$shape, -1L), operand_aval$dtype)
  ffi_out <- c(list(operand_aval, operand_aval, eigvals_aval), if (real) list(eigvals_aval),
               list(info_aval, info_aval))
  res <- linalg_ffi_call(
    ctx, target_name, ins, avals_in, ffi_out,
    attrs = list(mode = char_attr(if (params$compute_schur_vectors) "V" else "N"),
                 sort = char_attr("N")),
    operand_output_aliases = list(c(0L, 0L)))
  ok <- info_ok(ctx, res[length(res)], batch_dims)
  output <- replace_not_ok_with_nan(ctx, batch_dims, ok, res[1L], avals_out[[1L]])
  if (params$compute_schur_vectors)
    output <- c(output, replace_not_ok_with_nan(ctx, batch_dims, ok, res[2L], avals_out[[2L]]))
  output
}

# ---- Singular value decomposition ---------------------------------------------------------

svd_shape_rule <- function(shape, params) {
  m <- shape[1L]
  n <- shape[2L]
  rank <- min(m, n)
  sub <- params$subset_by_index
  if (!is.null(sub)) {
    if (params$full_matrices && !(sub[1L] == 0L && sub[2L] == rank))
      jax_error("full_matrices and subset_by_index cannot both be set", class = "jax_value_error")
    rank <- min(rank, sub[2L] - sub[1L])
  }
  if (!params$compute_uv) return(list(rank))
  list(rank, c(m, if (params$full_matrices) m else rank),
       c(if (params$full_matrices) n else rank, n))
}

svd_dtype_rule <- function(dtype, params) {
  real_dtype <- complex_component(dtype)
  if (params$compute_uv) c(real_dtype, dtype, dtype) else real_dtype
}

svd_p <- linalg_primitive(svd_dtype_rule, list(accept_float_complex), 2L, svd_shape_rule, "svd",
                          multiple_results = TRUE)

# Port of jax/_src/lax/linalg.py:_svd_jvp_rule.
svd_p$jvp <- function(primals, tangents, params) {
  A <- primals[[1L]]
  dA <- tangents[[1L]]
  dt <- aval_of(A)$dtype
  shape <- aval_of(A)$shape
  m <- shape[length(shape) - 1L]
  n <- shape[length(shape)]
  res <- bind(svd_p, list(A), list(full_matrices = FALSE, compute_uv = TRUE,
                                   subset_by_index = params$subset_by_index,
                                   algorithm = params$algorithm))
  s <- res[[1L]]
  U <- res[[2L]]
  Vt <- res[[3L]]
  if (params$compute_uv && params$full_matrices && m != n)
    jax_error("Singular value decomposition JVP not implemented for full matrices",
              class = "jax_not_implemented_error")
  if (params$compute_uv && !full_subset(params$subset_by_index, min(m, n)))
    jax_error("Singular value decomposition JVP not implemented for ",
              "subset_by_index with compute_uv=True", class = "jax_not_implemented_error")
  Ut <- linalg_H(U)
  V <- linalg_H(Vt)
  s_dim <- lax_expand_dims(s, -2L)
  dS <- Ut %*% dA %*% V
  ds <- extract_diagonal(jnp_real(dS))
  if (!params$compute_uv) return(list(list(s), list(ds)))
  k <- utils::tail(aval_of(s)$shape, 1L)
  s_diffs <- (s_dim + linalg_T(s_dim)) * (s_dim - linalg_T(s_dim))
  # is 1. where s_diffs is 0. and is 0. everywhere else
  s_diffs_zeros <- lax_expand_dims(linalg_eye(aval_of(s)$dtype, c(k, k)),
                                   seq_len(linalg_ndim(s_diffs) - 2L) - 1L)
  Fm <- linalg_astype(1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros, dt)
  s_dim_a <- linalg_astype(s_dim, dt)
  dSS <- s_dim_a * dS  # dS.dot(jnp.diag(s))
  SdS <- linalg_T(s_dim_a) * dS  # jnp.diag(s).dot(dS)
  s_zeros <- linalg_astype(s == 0, aval_of(s)$dtype)
  s_inv <- 1 / (s + s_zeros) - s_zeros
  s_inv_mat <- construct_diagonal(s_inv)
  dUdV_diag <- .5 * (dS - linalg_H(dS)) * linalg_astype(s_inv_mat, dt)
  dU <- U %*% (Fm * (dSS + linalg_H(dSS)) + dUdV_diag)
  dV <- V %*% (Fm * (SdS + linalg_H(SdS)))
  if (m > n) {
    dAV <- dA %*% V
    dU <- dU + (dAV - U %*% (Ut %*% dAV)) / s_dim_a
  }
  if (n > m) {
    dAHU <- linalg_H(dA) %*% U
    dV <- dV + (dAHU - V %*% (Vt %*% dAHU)) / s_dim_a
  }
  list(list(s, U, Vt), list(ds, dU, linalg_H(dV)))
}

# Port of jax/_src/lax/linalg.py:_empty_svd.
empty_svd <- function(a, full_matrices, compute_uv) {
  shape <- aval_of(a)$shape
  dt <- aval_of(a)$dtype
  batch_shape <- utils::head(shape, -2L)
  nb <- length(batch_shape)
  m <- shape[nb + 1L]
  n <- shape[nb + 2L]
  s <- lax_full(c(batch_shape, 0L), 0, complex_component(dt))
  if (!compute_uv) return(list(s))
  if (full_matrices) {
    size <- max(m, n)
    u <- lax_broadcast_in_dim(linalg_eye(dt, c(size, size)), c(batch_shape, size, size),
                              c(nb, nb + 1L))
  } else {
    u <- lax_full(c(batch_shape, m, n), 0, dt)
  }
  v <- lax_full(c(batch_shape, 0L, 0L), 0, dt)
  if (m < n) list(s, v, u) else list(s, u, v)
}

# Port of jax/_src/lax/linalg.py:_svd_cpu_gpu_lowering (CPU branch).
svd_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  operand_aval <- avals_in[[1L]]
  shape <- operand_aval$shape
  batch_dims <- utils::head(shape, -2L)
  m <- shape[length(shape) - 1L]
  n <- shape[length(shape)]
  full_matrices <- params$full_matrices
  compute_uv <- params$compute_uv
  if (!full_subset(params$subset_by_index, min(m, n)))
    jax_error("subset_by_index not implemented for CPU and GPU", class = "jax_not_implemented_error")
  if (m == 0L || n == 0L) {
    rule <- lower_fun(function(args, params) empty_svd(args[[1L]], full_matrices, compute_uv),
                      multiple_results = TRUE)
    return(rule(ctx, ins, avals_in, avals_out, params))
  }
  algorithm <- params$algorithm
  routine <- if (is.null(algorithm) || algorithm %in% c("default", "divide_and_conquer")) {
    "gesdd_ffi"
  } else if (identical(algorithm, "QR")) {
    "gesvd_ffi"
  } else {
    jax_error("The SVD Jacobi and Polar algorithms are not implemented on CPU.",
              class = "jax_not_implemented_error")
  }
  target_name <- prepare_lapack_call(routine, operand_aval$dtype)
  # Port of jax/_src/lax/linalg.py:_svd_computation_attr.
  mode <- char_attr(if (!compute_uv) "N" else if (!full_matrices) "S" else "A")
  info_aval <- new_aval(batch_dims, "int32")
  s_aval <- avals_out[[1L]]
  if (compute_uv) {
    u_aval <- avals_out[[2L]]
    vt_aval <- avals_out[[3L]]
  } else {
    # JAX note: it should be possible to skip instantiating these arrays when they
    # are not used.
    k <- min(m, n)
    u_aval <- new_aval(c(batch_dims, m, if (full_matrices) m else k), operand_aval$dtype)
    vt_aval <- new_aval(c(batch_dims, if (full_matrices) n else k, n), operand_aval$dtype)
  }
  res <- linalg_ffi_call(ctx, target_name, ins, avals_in,
                         list(operand_aval, s_aval, u_aval, vt_aval, info_aval),
                         attrs = list(mode = mode), operand_output_aliases = list(c(0L, 0L)))
  ok <- info_ok(ctx, res[5L], batch_dims)
  result <- replace_not_ok_with_nan(ctx, batch_dims, ok, res[2L], s_aval)
  if (compute_uv)
    result <- c(result, replace_not_ok_with_nan(ctx, batch_dims, ok, res[3L], u_aval),
                replace_not_ok_with_nan(ctx, batch_dims, ok, res[4L], vt_aval))
  result
}

# ---- Symmetric product ---------------------------------------------------------------------

symmetric_product_shape_rule <- function(a_shape, c_shape, params) {
  if (a_shape[1L] != c_shape[2L] || c_shape[1L] != c_shape[2L])
    jax_error("symmetric_update expects a rectangular matrix of shape (m, n) and a ",
              "square matrix of shape (n, n). Got shapes ", py_shape(a_shape), " and ",
              py_shape(c_shape), ".", class = "jax_value_error")
  c_shape
}

symmetric_product_p <- standard_linalg_primitive(
  list(accept_float, accept_float), c(2L, 2L), symmetric_product_shape_rule, "symmetric_product")

# Port of jax/_src/lax/linalg.py:_symmetric_product_jax_fn.
symmetric_product_p$lowering <- lower_fun(function(args, params) {
  a <- args[[1L]]
  params$alpha * linalg_batch_matmul(a, linalg_T(a), precision = "highest") + params$beta * args[[2L]]
})

# ---- Triangular solve ----------------------------------------------------------------------

triangular_solve_shape_rule <- function(a_shape, b_shape, params) {
  if (a_shape[1L] != a_shape[2L])
    jax_error("The first input to triangular_solve must be a square matrix. Got ",
              "shape ", py_shape(a_shape), ".", class = "jax_value_error")
  common <- if (isTRUE(params$left_side)) b_shape[1L] else b_shape[2L]
  if (a_shape[2L] != common)
    jax_error("Incompatible shapes for arguments to triangular_solve: ", py_shape(a_shape),
              " and ", py_shape(b_shape), ".", class = "jax_value_error")
  b_shape
}

triangular_solve_p <- linalg_primitive(
  linalg_input_dtype, list(accept_any_inexact, accept_any_inexact), c(2L, 2L),
  triangular_solve_shape_rule, "triangular_solve")

triangular_solve_bind <- function(a, b, params) {
  lax_linalg_triangular_solve(a, b, left_side = params$left_side, lower = params$lower,
                              transpose_a = params$transpose_a, conjugate_a = params$conjugate_a,
                              unit_diagonal = params$unit_diagonal)
}

# Port of jax/_src/lax/linalg.py:_triangular_solve_jvp_rule_a.
triangular_solve_jvp_rule_a <- function(g_a, ans, primals, params) {
  a <- primals[[1L]]
  bs <- aval_of(primals[[2L]])$shape
  m <- bs[length(bs) - 1L]
  n <- bs[length(bs)]
  k <- if (params$unit_diagonal) 1L else 0L
  g_a <- if (params$lower) linalg_tril(g_a, k = -k) else linalg_triu(g_a, k = k)
  g_a <- lax_neg(g_a)
  if (params$transpose_a) g_a <- linalg_T(g_a)
  if (params$conjugate_a) g_a <- linalg_conj(g_a)
  a_inverse <- function(rhs) triangular_solve_bind(a, rhs, params)
  # triangular_solve is about the same cost as matrix multiplication (~n^2 FLOPs for
  # matrix/vector inputs). Order these operations in whichever order is cheaper.
  if (params$left_side) {
    if (m > n) a_inverse(linalg_dot(g_a, ans))  # A^{-1} (dA X)
    else linalg_dot(a_inverse(g_a), ans)  # (A^{-1} dA) X
  } else {
    if (m < n) a_inverse(linalg_dot(ans, g_a))  # (X dA) A^{-1}
    else linalg_dot(ans, a_inverse(g_a))  # X (dA A^{-1})
  }
}

defjvp2(triangular_solve_p, triangular_solve_jvp_rule_a,
        function(g_b, ans, primals, params) triangular_solve_bind(primals[[1L]], g_b, params))

# Port of jax/_src/lax/linalg.py:_triangular_solve_transpose_rule. Triangular solve is
# nonlinear in its first argument and linear in its second argument, analogous to
# `div` but swapped.
triangular_solve_p$transpose <- function(ct, args, params) {
  a <- args[[1L]]
  b <- args[[2L]]
  if (is_undefined(a) || !is_undefined(b))
    jax_error("internal: triangular_solve is linear in its second argument only")
  if (is_zero(ct)) return(list(NULL, zero_of(b$aval)))
  params$transpose_a <- !params$transpose_a
  list(NULL, triangular_solve_bind(a, ct, params))
}

# Port of jax/_src/lax/linalg.py:_triangular_solve_batching_rule.
triangular_solve_p$batch <- function(vals, dims, params, axis) {
  x <- vals[[1L]]
  y <- vals[[2L]]
  if (is.na(dims[1L])) {
    y <- moveaxis(y, dims[2L], if (params$left_side) -1L else -2L)
    ys <- aval_of(y)$shape
    nd <- length(ys)
    if (params$left_side) {
      y_flat <- lax_reshape(y, c(ys[seq_len(nd - 2L)], ys[nd - 1L] * ys[nd]))
      bdim_out <- nd - 1L
    } else {
      y_flat <- lax_reshape(y, c(ys[seq_len(nd - 3L)], ys[nd - 2L] * ys[nd - 1L], ys[nd]))
      bdim_out <- nd - 2L
    }
    return(list(lax_reshape(triangular_solve_bind(x, y_flat, params), ys), bdim_out))
  }
  x <- bdim_at_front(x, dims[1L], axis$size)
  y <- bdim_at_front(y, dims[2L], axis$size)
  list(triangular_solve_bind(x, y, params), 0L)
}

# Port of jax/_src/lax/linalg.py:_triangular_solve_cpu_lower.
triangular_solve_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  a_aval <- avals_in[[1L]]
  a <- ins[[1L]]
  conjugate_a <- params$conjugate_a
  if (conjugate_a && !params$transpose_a) {
    a <- hlo_conj(ctx, a, a_aval)
    conjugate_a <- FALSE
  }
  if (a_aval$dtype %in% c("float32", "float64", "complex64", "complex128")) {
    target_name <- prepare_lapack_call("trsm_ffi", a_aval$dtype)
    return(linalg_ffi_call(
      ctx, target_name, c(a, ins[[2L]]), avals_in, avals_out,
      attrs = list(side = matrix_side_attr(params$left_side),
                   uplo = matrix_uplo_attr(params$lower),
                   trans_x = matrix_transpose_attr(params$transpose_a, conjugate_a),
                   diag = matrix_diagonal_attr(params$unit_diagonal)),
      operand_output_aliases = list(c(1L, 0L))))
  }
  # Fall back to the HLO implementation for unsupported types.
  transpose <- if (params$transpose_a) (if (conjugate_a) "ADJOINT" else "TRANSPOSE") else "NO_TRANSPOSE"
  emit_op(ctx, "stablehlo.triangular_solve", c(a, ins[[2L]]), types_of(avals_in),
          tensor_type(avals_out[[1L]]),
          attrs = list(left_side = attr_bool(params$left_side), lower = attr_bool(params$lower),
                       unit_diagonal = attr_bool(params$unit_diagonal),
                       transpose_a = paste0("#stablehlo<transpose ", transpose, ">")))
}

# ---- tridiagonal: reduction of a symmetric/Hermitian matrix --------------------------------

tridiagonal_shape_rule <- function(shape, params) {
  if (shape[1L] != shape[2L] || shape[2L] == 0L)
    jax_error("The input to tridiagonal must be a square matrix. Got shape ", py_shape(shape), ".",
              class = "jax_value_error")
  n <- shape[1L]
  list(shape, n, n - 1L, n - 1L)
}

tridiagonal_dtype_rule <- function(dtype, params) {
  real_dtype <- complex_component(dtype)
  c(dtype, real_dtype, real_dtype, dtype)
}

tridiagonal_p <- linalg_primitive(tridiagonal_dtype_rule, list(accept_float_complex), 2L,
                                  tridiagonal_shape_rule, "tridiagonal", multiple_results = TRUE)

# Port of jax/_src/lax/linalg.py:_tridiagonal_cpu_gpu_lowering (CPU branch).
tridiagonal_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  a_aval <- avals_in[[1L]]
  batch_dims <- utils::head(a_aval$shape, -2L)
  prefix <- if (a_aval$dtype %in% c("float32", "float64")) "sy" else "he"
  target_name <- prepare_lapack_call(paste0(prefix, "trd_ffi"), a_aval$dtype)
  res <- linalg_ffi_call(ctx, target_name, ins, avals_in,
                         c(avals_out, list(new_aval(batch_dims, "int32"))),
                         attrs = list(uplo = matrix_uplo_attr(params$lower)),
                         operand_output_aliases = list(c(0L, 0L)))
  ok <- info_ok(ctx, res[5L], batch_dims)
  vapply(1:4, function(i) replace_not_ok_with_nan(ctx, batch_dims, ok, res[i], avals_out[[i]]), "")
}

# ---- Tridiagonal solve ---------------------------------------------------------------------

tridiagonal_solve_shape_rule <- function(dl_shape, d_shape, du_shape, b_shape, params) {
  if (!identical(dl_shape, d_shape) || !identical(dl_shape, du_shape))
    jax_error("tridiagonal_solve requires that all diagonal arguments have the same ",
              "shape.", class = "jax_type_error")
  if (!identical(dl_shape, b_shape[-length(b_shape)]))
    jax_error("tridiagonal_solve requires that the leading ndim-1 dimensions of b ",
              "equal the dimensions of the diagonal arguments.", class = "jax_type_error")
  b_shape
}

tridiagonal_solve_p <- standard_linalg_primitive(
  rep(list(accept_float_complex), 4L), c(1L, 1L, 1L, 2L), tridiagonal_solve_shape_rule,
  "tridiagonal_solve")

# Port of jax/_src/lax/linalg.py:_tridiagonal_solve_cpu_lowering.
tridiagonal_solve_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  b_aval <- avals_in[[4L]]
  batch_dims <- utils::head(b_aval$shape, -2L)
  if (isTRUE(params$perturb_singular)) {
    prepare_lapack_call("gtsv_ffi", b_aval$dtype)  # dtype and registration check only
    return(linalg_ffi_call(ctx, "tridiagonal_solve_perturbed_ffi", ins, avals_in, list(b_aval)))
  }
  target_name <- prepare_lapack_call("gtsv_ffi", b_aval$dtype)
  res <- linalg_ffi_call(ctx, target_name, ins, avals_in,
                         c(avals_in, list(new_aval(batch_dims, "int32"))),
                         operand_output_aliases = list(c(0L, 0L), c(1L, 1L), c(2L, 2L), c(3L, 3L)))
  ok <- info_ok(ctx, res[5L], batch_dims)
  replace_not_ok_with_nan(ctx, batch_dims, ok, res[4L], b_aval)
}

# Port of jax/_src/lax/linalg.py:_tridiagonal_product: the product of the tridiagonal
# matrix with b. JAX adds the off diagonal terms with scatter-add; here they are
# shifted into place with pad.
tridiagonal_product <- function(dl, d, du, b) {
  bs <- aval_of(b)$shape
  r <- length(bs)
  m <- bs[r - 1L]
  y <- lax_expand_dims(d, -1L) * b
  if (m < 2L) return(y)
  zero <- as_jax(0, aval_of(b)$dtype)
  shift <- function(lo, hi) {
    cfg <- rep(list(c(0L, 0L, 0L)), r)
    cfg[[r - 1L]] <- c(lo, hi, 0L)
    cfg
  }
  below <- lax_expand_dims(lax_slice_in_dim(dl, 1L, m, axis = -1L), -1L) *
    lax_slice_in_dim(b, 0L, m - 1L, axis = -2L)
  above <- lax_expand_dims(lax_slice_in_dim(du, 0L, m - 1L, axis = -1L), -1L) *
    lax_slice_in_dim(b, 1L, m, axis = -2L)
  y + lax_pad(below, zero, shift(1L, 0L)) + lax_pad(above, zero, shift(0L, 1L))
}

# Port of jax/_src/lax/linalg.py:_tridiagonal_solve_jvp_rule.
tridiagonal_solve_p$jvp <- function(primals, tangents, params) {
  diags_dot <- tangents[1:3]
  b_dot <- tangents[[4L]]
  ans <- bind(tridiagonal_solve_p, primals, params)
  if (all(vapply(diags_dot, is_zero, NA))) {
    rhs <- b_dot
  } else {
    dd <- lapply(diags_dot, instantiate_zero)
    matvec_dot <- tridiagonal_product(dd[[1L]], dd[[2L]], dd[[3L]], ans)
    rhs <- add_tangents(b_dot, lax_neg(matvec_dot))
  }
  list(ans, bind(tridiagonal_solve_p, c(primals[1:3], list(rhs)), params))
}

# Port of jax/_src/lax/linalg.py:_tridiagonal_solve_transpose_rule. Tridiagonal solve is
# nonlinear in the tridiagonal arguments and linear otherwise.
tridiagonal_solve_p$transpose <- function(ct, args, params) {
  dl <- args[[1L]]
  d <- args[[2L]]
  du <- args[[3L]]
  b <- args[[4L]]
  if (is_undefined(dl) || is_undefined(d) || is_undefined(du) || !is_undefined(b))
    jax_error("internal: tridiagonal_solve is linear in its last argument only")
  if (is_zero(ct)) return(list(NULL, NULL, NULL, zero_of(b$aval)))
  r <- linalg_ndim(du)
  m <- utils::tail(aval_of(du)$shape, 1L)
  # An empty system has nothing to shift.
  if (m == 0L) return(list(NULL, NULL, NULL, ct))
  last <- function(z, from, to) lax_slice_in_dim(z, from, to, axis = -1L)
  dl_trans <- lax_concatenate(list(lax_full_like(last(du, m - 1L, m), 0), last(du, 0L, m - 1L)),
                              r - 1L)
  du_trans <- lax_concatenate(list(last(dl, 1L, m), lax_full_like(last(dl, 0L, 1L), 0)), r - 1L)
  list(NULL, NULL, NULL,
       lax_linalg_tridiagonal_solve(dl_trans, d, du_trans, ct,
                                    perturb_singular = params$perturb_singular))
}

# Port of jax/_src/lax/linalg.py:_tridiagonal_solve_batching_rule.
tridiagonal_solve_p$batch <- function(vals, dims, params, axis) {
  solve <- function(dl, d, du, b) {
    lax_linalg_tridiagonal_solve(dl, d, du, b, perturb_singular = params$perturb_singular)
  }
  if (all(is.na(dims[1:3]))) {
    b <- moveaxis(vals[[4L]], dims[4L], -2L)
    bs <- aval_of(b)$shape
    nd <- length(bs)
    b_flat <- lax_reshape(b, c(bs[seq_len(nd - 3L)], bs[nd - 2L], bs[nd - 1L] * bs[nd]))
    out_flat <- solve(vals[[1L]], vals[[2L]], vals[[3L]], b_flat)
    return(list(lax_reshape(out_flat, bs), nd - 2L))
  }
  v <- Map(function(x, d) bdim_at_front(x, d, axis$size), vals, dims)
  list(solve(v[[1L]], v[[2L]], v[[3L]], v[[4L]]), 0L)
}

# ---- linear solve used by the JVP rules ----------------------------------------------------

# Port of jax/_src/lax/linalg.py:_check_solve_shapes.
check_solve_shapes <- function(a, b) {
  as_ <- aval_of(a)$shape
  bs <- aval_of(b)$shape
  na <- length(as_)
  ok <- na >= 2L && length(bs) %in% c(na, na - 1L) && as_[na] == as_[na - 1L] &&
    as_[na] == bs[na - 1L]
  if (!isTRUE(ok))
    jax_error("The arguments to solve must have shapes a=[..., m, m] and ",
              "b=[..., m, k] or b=[..., m]; got a=", py_shape(as_), " and b=", py_shape(bs),
              class = "jax_value_error")
}

# Port of jax/_src/lax/linalg.py:_broadcasted_matvec: a broadcasted dot_general with
# signature (...,n,m),(...,m)->(...,n).
broadcasted_matvec <- function(a, b) {
  as_ <- aval_of(a)$shape
  bs <- aval_of(b)$shape
  batch_shape <- broadcast_shapes_impl(list(utils::head(as_, -2L), utils::head(bs, -1L)))
  n_batch <- length(batch_shape)
  a <- linalg_broadcast_to(a, c(batch_shape, utils::tail(as_, 2L)))
  b <- linalg_broadcast_to(b, c(batch_shape, utils::tail(bs, 1L)))
  batch <- seq_len(n_batch) - 1L
  lax_dot_general(a, b, list(list(n_batch + 1L, n_batch), list(batch, batch)),
                  precision = "highest")
}

# Port of jax/_src/lax/linalg.py:_solve.
linalg_solve <- function(a, b) {
  check_solve_shapes(a, b)
  as_ <- aval_of(a)$shape
  bs <- aval_of(b)$shape
  # Broadcast leading dimensions of b to the shape of a, as is required by
  # custom_linear_solve.
  ref <- c(utils::head(as_, -1L), 1L)[seq_along(bs)]
  b <- lax_broadcast_in_dim(b, ifelse(bs == 1L, ref, bs), seq_along(bs) - 1L)
  # With custom_linear_solve, the same factorization is reused when computing
  # sensitivities. This is considerably faster.
  f <- lax_linalg_lu(lax_stop_gradient(a))
  lu_ <- f[[1L]]
  permutation <- f[[3L]]
  custom_solve <- function(b) {
    lax_custom_linear_solve(
      function(x) broadcasted_matvec(a, x), b,
      solve = function(matvec, x) lu_solve(lu_, permutation, x, trans = 0L),
      transpose_solve = function(matvec, x) lu_solve(lu_, permutation, x, trans = 1L))
  }
  # b.shape == [..., m]
  if (length(as_) == length(bs) + 1L) return(custom_solve(b))
  # b.shape == [..., m, k]
  vmap(custom_solve, in_axes = length(bs) - 1L, out_axes = max(length(as_), length(bs)) - 1L)(b)
}

# ---- QDWH polar decomposition (port of jax/_src/tpu/linalg/qdwh.py) ----------------------
#
# Reference: Nakatsukasa, Yuji, Zhaojun Bai, and Francois Gygi. "Optimizing Halley's
# iteration for computing the matrix polar decomposition." SIAM Journal on Matrix
# Analysis and Applications 31, no. 5 (2010): 2700-2720.

# Port of qdwh.py:_mask: masks `x` up to the dynamic shape `dims`; values outside are
# replaced with `alternative`.
qdwh_mask <- function(x, dims, alternative = 0) {
  mask <- NULL
  for (i in seq_along(dims)) {
    if (is.null(dims[[i]])) next
    mask_dim_i <- lax_broadcasted_iota("int32", aval_of(x)$shape, i - 1L) < dims[[i]]
    mask <- if (is.null(mask)) mask_dim_i else mask & mask_dim_i
  }
  if (is.null(mask)) x else jnp_where(mask, x, alternative)
}

# Port of qdwh.py:_dynamic_concat: concatenates the padded arrays `a` and `b` along
# axis 0, where the true size of `a` is `m`.
qdwh_dynamic_concat <- function(a, b, m) {
  if (is.null(m)) return(jnp_concatenate(list(a, b), axis = 0L))
  padded <- lax_pad(a, as_jax(0, aval_of(a)$dtype),
                    c(list(c(0L, aval_of(b)$shape[1L], 0L)),
                      rep(list(c(0L, 0L, 0L)), linalg_ndim(a) - 1L)))
  lax_dynamic_update_slice(padded, b, c(list(m), as.list(rep(0L, linalg_ndim(a) - 1L))))
}

# The three QDWH parameters, given as R numbers or as a vector of length 3.
qdwh_params <- function(params) {
  if (is.list(params)) return(params)
  lapply(0:2, function(i) lax_index_in_dim(params, i, axis = 0L, keepdims = FALSE))
}

# Port of qdwh.py:_use_qr: QDWH iteration using the QR decomposition. `u` has the static
# (padded) shape M x N and the dynamic shape m x n.
qdwh_use_qr <- function(u, m, n, params) {
  p <- qdwh_params(params)  # a_minus_e_by_sqrt_c, sqrt_c, e
  s <- aval_of(u)$shape
  M <- s[1L]
  N <- s[2L]
  y <- qdwh_dynamic_concat(p[[2L]] * u, jnp_eye(N, dtype = aval_of(u)$dtype), m)
  q <- lax_linalg_qr(y, full_matrices = FALSE)[[1L]]
  # q1 = q[:m, :]
  q1 <- qdwh_mask(lax_slice(q, c(0L, 0L), c(M, N)), list(m, n))
  # q2 = (q[m:, :]).T.conj()
  q2 <- lax_dynamic_slice(q, list(m, 0L), c(N, N))
  q2 <- linalg_H(qdwh_mask(q2, list(n, n)))
  p[[3L]] * u + p[[1L]] * (q1 %*% q2)
}

# Port of qdwh.py:_use_cholesky: QDWH iteration using the Cholesky decomposition.
qdwh_use_cholesky <- function(u, m, n, params) {
  p <- qdwh_params(params)  # a_minus_e, c, e
  N <- aval_of(u)$shape[2L]
  dt <- aval_of(u)$dtype
  x <- p[[2L]] * (linalg_H(u) %*% u) + jnp_eye(N, dtype = dt)
  # Pads the lower-right corner with the identity matrix to prevent the Cholesky
  # decomposition from failing due to the matrix not being PSD if padded with zeros.
  x <- qdwh_mask(x, list(n, n), jnp_eye(N, dtype = dt))
  # `y` is lower triangular.
  y <- lax_linalg_cholesky(x, symmetrize_input = FALSE)
  z <- linalg_conj(lax_linalg_triangular_solve(y, linalg_T(u), left_side = TRUE, lower = TRUE,
                                               conjugate_a = TRUE))
  z <- linalg_H(lax_linalg_triangular_solve(y, z, left_side = TRUE, lower = TRUE,
                                            transpose_a = TRUE, conjugate_a = TRUE))
  p[[3L]] * u + p[[1L]] * z
}

# Port of qdwh.py:_qdwh: QR-based dynamically weighted Halley iteration for the polar
# decomposition.
qdwh_impl <- function(x, m, n, max_iterations, eps) {
  dt <- aval_of(x)$dtype
  # Estimates `alpha` and `beta = alpha * l`, where `alpha` is an estimate of
  # norm(x, 2) such that `alpha >= norm(x, 2)` and `beta` is a lower bound for the
  # smallest singular value of x.
  if (is.null(eps)) eps <- finfo(dt)$eps
  one_norm <- jnp_linalg_norm(x, ord = 1)
  inf_norm <- jnp_linalg_norm(x, ord = Inf)
  alpha_inverse <- lax_rsqrt(one_norm) * lax_rsqrt(inf_norm)
  alpha_inverse <- jnp_where(one_norm == 0, 1, alpha_inverse)
  u <- x * jnp_astype(alpha_inverse, dt)
  l <- eps
  # Iteration tolerances.
  tol_l <- 10.0 * eps / 2.0
  tol_norm <- jnp_cbrt(tol_l)
  cholesky_cutoff <- 100
  qr_coefs <- list()
  chol_coefs <- list()
  k <- 0L
  while (l + tol_l < 1 && k < max_iterations) {
    k <- k + 1L
    l2 <- l * l
    dd <- (4 * (1 / l2 - 1) / l2)^(1 / 3)
    sqd <- (1.0 + dd)^(1 / 2)
    a <- sqd + (2 - dd + 2 * (2 - l2) / (l2 * sqd))^(1 / 2)
    b <- (a - 1)^2 / 4
    c_ <- a + b - 1
    l <- l * (a + b * l2) / (1 + c_ * l2)
    e <- b / c_
    if (c_ > cholesky_cutoff) {
      qr_coefs[[length(qr_coefs) + 1L]] <- c((a - e) / c_^(1 / 2), c_^(1 / 2), e)
    } else {
      chol_coefs[[length(chol_coefs) + 1L]] <- c(a - e, c_, e)
    }
  }
  iteration <- function(k, state, update_fn, coefs, test_convergence) {
    u <- state[[1L]]
    params <- if (is.null(coefs)) {
      # As l -> 1, the coefficients a, b, c -> 3, 1, 3, which is Halley's method.
      list(3 - 1 / 3, 3, 1 / 3)
    } else {
      lax_squeeze(lax_dynamic_slice(coefs, list(k, 0L), c(1L, 3L)), 0L)
    }
    u_prev <- u
    u <- update_fn(u, m, n, params)
    is_not_converged <- if (test_convergence) jnp_linalg_norm(u - u_prev) > tol_norm else as_jax(TRUE)
    list(u, is_not_converged)
  }
  iterate <- function(u, coefs, update_fn, test_convergence) {
    if (!length(coefs)) return(list(u, as_jax(TRUE)))
    coefs_arr <- jnp_astype(as_jax(do.call(rbind, coefs)), dt)
    lax_fori_loop(0L, length(coefs), function(k, state) {
      iteration(k, state, update_fn, coefs_arr, test_convergence)
    }, list(u, as_jax(TRUE)))
  }
  u <- iterate(u, qr_coefs, qdwh_use_qr, FALSE)[[1L]]
  st <- iterate(u, chol_coefs, qdwh_use_cholesky, TRUE)
  # If l has converged but u still has not, continue with Halley's method (no
  # coefficients) until convergence.
  cond_fun <- function(state) jnp_logical_and(state[[3L]], state[[1L]] < max_iterations)
  body_fun <- function(state) {
    r <- iteration(state[[1L]], list(state[[2L]], state[[3L]]), qdwh_use_cholesky, NULL, TRUE)
    list(state[[1L]] + 1L, r[[1L]], r[[2L]])
  }
  res <- lax_while_loop(cond_fun, body_fun,
                        list(as_jax(length(qr_coefs) + length(chol_coefs)), st[[1L]], st[[2L]]))
  u <- res[[2L]]
  # Applies Newton-Schulz refinement for better accuracy.
  u <- 1.5 * u - 0.5 * u %*% (linalg_H(u) %*% u)
  h <- linalg_H(u) %*% x
  h <- (h + linalg_H(h)) / 2
  # Converged within the maximum number of iterations.
  list(u, h, res[[1L]], jnp_logical_not(res[[3L]]))
}

# ---- public API (jax.lax.linalg) -----------------------------------------------------------

#' Linear algebra primitives
#'
#' Ports of the functions of `jax.lax.linalg`. Every function works on batches of
#' matrices (leading batch dimensions) unless noted, and none of them promotes dtypes.
#' On CPU the decompositions run LAPACK through XLA custom calls; a failed
#' decomposition returns arrays full of NaN for that batch element.
#'
#' * `lax_linalg_cholesky()`: `A = L L^H` with `L` lower triangular.
#' * `lax_linalg_cholesky_update()`: Cholesky factor of `A + w w^T` from the upper
#'   triangular factor `R` of `A = R^T R` (no batching).
#' * `lax_linalg_eig()`: eigendecomposition of a general matrix; returns
#'   `list(w, vl, vr)` (eigenvectors only when requested).
#' * `lax_linalg_eigh()`: eigendecomposition of a Hermitian matrix; returns
#'   `list(v, w)`.
#' * `lax_linalg_hessenberg()`: reduction to upper Hessenberg form; returns
#'   `list(a, taus)`.
#' * `lax_linalg_householder_product()`: product of elementary Householder reflectors.
#' * `lax_linalg_lu()`: LU decomposition with partial pivoting, `P A = L U`; returns
#'   `list(lu, pivots, permutation)` with 0-based pivots and permutation.
#' * `lax_linalg_lu_pivots_to_permutation()`: converts LU row swaps to a permutation.
#' * `lax_linalg_ormqr()`: multiplies a matrix by the `Q` of a QR factorization without
#'   forming `Q`.
#' * `lax_linalg_qr()`: QR decomposition; returns `list(q, r)` or `list(q, r, p)` with
#'   `pivoting`.
#' * `lax_linalg_schur()`: Schur decomposition `A = Q U Q^H`; returns `list(U, Q)` or
#'   `list(U)`.
#' * `lax_linalg_svd()`: singular value decomposition; returns `list(u, s, vh)` or `s`.
#' * `lax_linalg_symmetric_product()`: `alpha A A^T + beta C`.
#' * `lax_linalg_triangular_solve()`: solves `op(A) X = B` (`left_side`) or
#'   `X op(A) = B` for triangular `A`.
#' * `lax_linalg_tridiagonal()`: reduction of a Hermitian matrix to tridiagonal form;
#'   returns `list(a, d, e, taus)`.
#' * `lax_linalg_tridiagonal_solve()`: solves a tridiagonal system given the lower,
#'   middle and upper diagonals (`dl[0]` and `du[m - 1]` are ignored).
#'
#' @param x,a Array with shape `[..., m, n]` (square where the operation requires it).
#' @param symmetrize_input If `TRUE`, the matrix is symmetrized first by computing
#'   `(x + x^H) / 2`; otherwise only one triangle is used.
#' @param r_matrix Upper triangular matrix `R` with `A = R^T R`.
#' @param w_vector Vector of the rank one update.
#' @param compute_left_eigenvectors,compute_right_eigenvectors Whether to compute the
#'   left and right eigenvectors.
#' @param enable_eigvec_derivs Enable differentiation of the eigenvectors (valid only
#'   for distinct eigenvalues, under the LAPACK `geev` normalization).
#' @param implementation Algorithm selection (see `lax_linalg_EigImplementation` and
#'   `lax_linalg_EighImplementation`); only LAPACK is available on CPU.
#' @param use_magma Accepted for compatibility with JAX; only used on GPU.
#' @param lower Which triangle of the matrix is used.
#' @param sort_eigenvalues Sort the eigenvalues in ascending order (always the case on
#'   CPU).
#' @param subset_by_index Optional pair `c(start, end)` of 0-based indices selecting a
#'   range of eigenvalues or singular values (not supported on CPU).
#' @param taus Scalar factors of the elementary Householder reflectors, shape `[..., k]`.
#' @param pivots Integer (int32) array of row swaps with shape `[..., k]`.
#' @param permutation_size Size of the permutation (at least `k`).
#' @param c Matrix to multiply by `Q`.
#' @param left If `TRUE` compute `Q C`, else `C Q`.
#' @param transpose If `TRUE` use `Q^T` (`Q^H` for complex types).
#' @param pivoting Column pivoted (rank revealing) QR.
#' @param full_matrices Whether full or reduced matrices are returned.
#' @param compute_schur_vectors Whether to compute the Schur vectors `Q`.
#' @param sort_eig_vals,select_callable Unused (as in JAX).
#' @param compute_uv Whether to return the singular vectors.
#' @param algorithm SVD algorithm (see `lax_linalg_SvdAlgorithm`); `"default"`,
#'   `"divide_and_conquer"` (gesdd) and `"QR"` (gesvd) are available on CPU.
#' @param a_matrix,c_matrix Matrices with shapes `[..., m, n]` and `[..., m, m]`.
#' @param alpha,beta Scalars.
#' @param symmetrize_output If `TRUE` the upper triangle of the result is replaced by
#'   the transpose of the lower triangle.
#' @param b Right hand side.
#' @param left_side Solve `op(A) X = B` if `TRUE`, else `X op(A) = B`.
#' @param transpose_a,conjugate_a Whether `op` transposes and conjugates `a`.
#' @param unit_diagonal Assume a unit diagonal (the diagonal is not accessed).
#' @param dl,d,du Lower, middle and upper diagonals, each with shape `[..., m]`.
#' @param perturb_singular Perturb near zero pivots so that singular systems return a
#'   finite result.
#' @return An array, or a list of arrays as described above.
#' @name lax_linalg
NULL

#' @rdname lax_linalg
#' @export
lax_linalg_cholesky <- function(x, symmetrize_input = TRUE) {
  x <- as_jax(x)
  s <- aval_of(x)$shape
  r <- length(s)
  if (r < 2L || s[r] != s[r - 1L])
    jax_error("The input to linalg.cholesky must have shape [..., n, n], got shape ", py_shape(s),
              class = "jax_value_error")
  if (symmetrize_input) x <- linalg_symmetrize(x)
  linalg_tril(bind(cholesky_p, list(x)))
}

#' @rdname lax_linalg
#' @export
lax_linalg_cholesky_update <- function(r_matrix, w_vector) {
  bind(cholesky_update_p, list(as_jax(r_matrix), as_jax(w_vector)))
}

#' Algorithm selectors of jax.lax.linalg
#'
#' Values of the enumerations `jax.lax.linalg.EigImplementation`,
#' `EighImplementation` and `SvdAlgorithm`, as named lists of strings.
#' @format Named lists of character scalars.
#' @name lax_linalg_enums
#' @export
lax_linalg_EigImplementation <- list(CUSOLVER = "cusolver", MAGMA = "magma", LAPACK = "lapack")

#' @rdname lax_linalg_enums
#' @export
lax_linalg_EighImplementation <- list(QR = "qr", JACOBI = "jacobi", QDWH = "qdwh")

#' @rdname lax_linalg_enums
#' @export
lax_linalg_SvdAlgorithm <- list(DEFAULT = "default", QR = "QR", JACOBI = "Jacobi", POLAR = "polar",
                                DIVIDE_AND_CONQUER = "divide_and_conquer")

check_enum <- function(value, enum, what) {
  if (is.null(value)) return(NULL)
  if (!is.character(value) || length(value) != 1L || !(value %in% unlist(enum)))
    jax_error(what, " must be NULL or one of ", paste0("\"", unlist(enum), "\"", collapse = ", "),
              class = "jax_value_error")
  value
}

#' @rdname lax_linalg
#' @export
lax_linalg_eig <- function(x, compute_left_eigenvectors = TRUE, compute_right_eigenvectors = TRUE,
                           enable_eigvec_derivs = FALSE, implementation = NULL, use_magma = NULL) {
  if (!is.null(use_magma)) {
    warning("use_magma is deprecated, please use implementation = \"magma\" instead.")
    implementation <- if (isTRUE(use_magma)) "magma" else "lapack"
  }
  bind(eig_p, list(as_jax(x)),
       list(compute_left_eigenvectors = isTRUE(compute_left_eigenvectors),
            compute_right_eigenvectors = isTRUE(compute_right_eigenvectors),
            enable_eigvec_derivs = isTRUE(enable_eigvec_derivs),
            implementation = check_enum(implementation, lax_linalg_EigImplementation,
                                        "implementation")))
}

subset_param <- function(subset_by_index) {
  if (is.null(subset_by_index)) NULL else as.integer(subset_by_index)
}

#' @rdname lax_linalg
#' @export
lax_linalg_eigh <- function(x, lower = TRUE, symmetrize_input = TRUE, sort_eigenvalues = TRUE,
                            subset_by_index = NULL, implementation = NULL) {
  x <- as_jax(x)
  s <- aval_of(x)$shape
  r <- length(s)
  if (r < 2L || s[r] != s[r - 1L])
    jax_error("The input to linalg.eigh must have shape [..., n, n], got shape ", py_shape(s),
              class = "jax_value_error")
  if (symmetrize_input) x <- linalg_symmetrize(x)
  bind(eigh_p, list(x),
       list(lower = isTRUE(lower), sort_eigenvalues = isTRUE(sort_eigenvalues),
            subset_by_index = subset_param(subset_by_index),
            algorithm = check_enum(implementation, lax_linalg_EighImplementation,
                                   "implementation")))
}

#' @rdname lax_linalg
#' @export
lax_linalg_hessenberg <- function(a) bind(hessenberg_p, list(as_jax(a)))

#' @rdname lax_linalg
#' @export
lax_linalg_householder_product <- function(a, taus) {
  bind(householder_product_p, list(as_jax(a), as_jax(taus)))
}

#' @rdname lax_linalg
#' @export
lax_linalg_lu <- function(x) bind(lu_p, list(as_jax(x)))

#' @rdname lax_linalg
#' @export
lax_linalg_lu_pivots_to_permutation <- function(pivots, permutation_size) {
  bind(lu_pivots_to_permutation_p, list(as_jax(pivots)),
       list(permutation_size = as.integer(permutation_size)))
}

#' @rdname lax_linalg
#' @export
lax_linalg_ormqr <- function(a, taus, c, left = TRUE, transpose = FALSE) {
  bind(ormqr_p, list(as_jax(a), as_jax(taus), as_jax(c)),
       list(left = isTRUE(left), transpose = isTRUE(transpose)))
}

#' @rdname lax_linalg
#' @export
lax_linalg_qr <- function(x, pivoting = FALSE, full_matrices = TRUE, use_magma = NULL) {
  bind(qr_p, list(as_jax(x)), list(pivoting = isTRUE(pivoting),
                                   full_matrices = isTRUE(full_matrices), use_magma = use_magma))
}

#' @rdname lax_linalg
#' @export
lax_linalg_schur <- function(x, compute_schur_vectors = TRUE, sort_eig_vals = FALSE,
                             select_callable = NULL) {
  bind(schur_p, list(as_jax(x)),
       list(compute_schur_vectors = isTRUE(compute_schur_vectors),
            sort_eig_vals = isTRUE(sort_eig_vals), select_callable = select_callable))
}

#' @rdname lax_linalg
#' @export
lax_linalg_svd <- function(x, full_matrices = TRUE, compute_uv = TRUE, subset_by_index = NULL,
                           algorithm = NULL) {
  result <- bind(svd_p, list(as_jax(x)),
                 list(full_matrices = isTRUE(full_matrices), compute_uv = isTRUE(compute_uv),
                      subset_by_index = subset_param(subset_by_index),
                      algorithm = check_enum(algorithm, lax_linalg_SvdAlgorithm, "algorithm")))
  if (isTRUE(compute_uv)) list(result[[2L]], result[[1L]], result[[3L]]) else result[[1L]]
}

#' @rdname lax_linalg
#' @export
lax_linalg_symmetric_product <- function(a_matrix, c_matrix, alpha = 1, beta = 0,
                                         symmetrize_output = FALSE) {
  result <- bind(symmetric_product_p, list(as_jax(a_matrix), as_jax(c_matrix)),
                 list(alpha = alpha, beta = beta))
  if (symmetrize_output) {
    upper_half <- linalg_T(linalg_tril(result, k = -1L))
    result <- linalg_tril(result, k = 0L) + upper_half
  }
  result
}

#' @rdname lax_linalg
#' @export
lax_linalg_triangular_solve <- function(a, b, left_side = FALSE, lower = FALSE,
                                        transpose_a = FALSE, conjugate_a = FALSE,
                                        unit_diagonal = FALSE) {
  a <- as_jax(a)
  b <- as_jax(b)
  conjugate_a <- isTRUE(conjugate_a) && is_complex_dtype(aval_of(a)$dtype)
  singleton <- linalg_ndim(b) == linalg_ndim(a) - 1L
  if (singleton) b <- lax_expand_dims(b, if (left_side) -1L else -2L)
  out <- bind(triangular_solve_p, list(a, b),
              list(left_side = isTRUE(left_side), lower = isTRUE(lower),
                   transpose_a = isTRUE(transpose_a), conjugate_a = conjugate_a,
                   unit_diagonal = isTRUE(unit_diagonal)))
  if (singleton) out <- lax_squeeze(out, if (left_side) -1L else -2L)
  out
}

#' @rdname lax_linalg
#' @export
lax_linalg_tridiagonal <- function(a, lower = TRUE) {
  bind(tridiagonal_p, list(as_jax(a)), list(lower = isTRUE(lower)))
}

#' @rdname lax_linalg
#' @export
lax_linalg_tridiagonal_solve <- function(dl, d, du, b, perturb_singular = FALSE) {
  bind(tridiagonal_solve_p, list(as_jax(dl), as_jax(d), as_jax(du), as_jax(b)),
       list(perturb_singular = isTRUE(perturb_singular)))
}

#' QR-based dynamically weighted Halley iteration for the polar decomposition
#'
#' Port of `jax.lax.linalg.qdwh`: computes the polar decomposition `x = u h`.
#' @param x A full-rank matrix with shape `M x N`, `M >= N`. The matrix may be padded up
#'   to that size from a smaller true shape (`dynamic_shape`).
#' @param is_hermitian Unused; exists for compatibility with JAX.
#' @param max_iterations Maximum number of iterations (default 10).
#' @param eps The final result satisfies `|x_k - x_k-1| < |x_k| * (4 eps)^(1/3)`.
#' @param dynamic_shape Optional unpadded shape as `list(m, n)`.
#' @return `list(u, h, num_iters, is_converged)`.
#' @export
lax_linalg_qdwh <- function(x, is_hermitian = FALSE, max_iterations = NULL, eps = NULL,
                            dynamic_shape = NULL) {
  x <- as_jax(x)
  if (!is.logical(is_hermitian) || length(is_hermitian) != 1L)
    jax_error("The `is_hermitian` argument must be statically specified to use `qdwh` within ",
              "JAX transformations.", class = "jax_concretization_error")
  if (is.null(max_iterations)) {
    max_iterations <- 10L
  } else if (!is.numeric(max_iterations) || length(max_iterations) != 1L) {
    jax_error("The `max_iterations` argument must be statically specified to use `qdwh` within ",
              "JAX transformations.", class = "jax_concretization_error")
  }
  s <- aval_of(x)$shape
  if (length(s) != 2L)
    jax_error("qdwh requires a matrix, got shape ", py_shape(s), class = "jax_value_error")
  if (s[1L] < s[2L])
    jax_error("The input matrix of shape M x N must have M >= N.", class = "jax_value_error")
  if (!is.null(dynamic_shape)) {
    m <- dynamic_shape[[1L]]
    n <- dynamic_shape[[2L]]
    x <- qdwh_mask(x, list(m, n))
  } else {
    m <- s[1L]
    n <- s[2L]
  }
  qdwh_impl(x, m, n, as.integer(max_iterations), eps)
}
