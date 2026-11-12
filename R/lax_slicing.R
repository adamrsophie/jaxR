# lax gather and scatter (port of jax/_src/lax/slicing.py).
#
# Not ported: sharding, vma and unreduced rules, the GPU complex128 scatter lowering,
# extended dtype lowerings, DynamicGatherOp (symbolic shapes) and the RefAccum branches
# of the transpose rules.

# ---- dimension numbers and modes ----------------------------------------------------------

#' Dimension numbers for gather and scatter
#'
#' Ports of `jax.lax.GatherDimensionNumbers` and `jax.lax.ScatterDimensionNumbers`.
#' All dimensions are 0-based. The index vector dimension is implicit: it is always
#' the last dimension of the indices.
#' @param offset_dims Output dimensions that offset into the slice taken from the
#'   operand (ascending).
#' @param collapsed_slice_dims Operand dimensions of slice size 1 that are removed
#'   from the output (ascending).
#' @param start_index_map For each index component, the operand dimension it indexes.
#' @param operand_batching_dims Operand batching dimensions (ascending).
#' @param start_indices_batching_dims Batching dimensions of the indices, matched
#'   with `operand_batching_dims`.
#' @return A named list.
#' @export
lax_GatherDimensionNumbers <- function(offset_dims, collapsed_slice_dims, start_index_map,
                                   operand_batching_dims = integer(0),
                                   start_indices_batching_dims = integer(0)) {
  list(offset_dims = as.integer(offset_dims),
       collapsed_slice_dims = as.integer(collapsed_slice_dims),
       start_index_map = as.integer(start_index_map),
       operand_batching_dims = as.integer(operand_batching_dims),
       start_indices_batching_dims = as.integer(start_indices_batching_dims))
}

#' @rdname lax_GatherDimensionNumbers
#' @param update_window_dims Dimensions of the updates that are window dimensions
#'   (ascending).
#' @param inserted_window_dims Size 1 window dimensions inserted into the updates
#'   (operand dimensions, ascending).
#' @param scatter_dims_to_operand_dims For each index component, the operand
#'   dimension it indexes.
#' @param scatter_indices_batching_dims Batching dimensions of the indices, matched
#'   with `operand_batching_dims`.
#' @export
lax_ScatterDimensionNumbers <- function(update_window_dims, inserted_window_dims,
                                    scatter_dims_to_operand_dims,
                                    operand_batching_dims = integer(0),
                                    scatter_indices_batching_dims = integer(0)) {
  list(update_window_dims = as.integer(update_window_dims),
       inserted_window_dims = as.integer(inserted_window_dims),
       scatter_dims_to_operand_dims = as.integer(scatter_dims_to_operand_dims),
       operand_batching_dims = as.integer(operand_batching_dims),
       scatter_indices_batching_dims = as.integer(scatter_indices_batching_dims))
}

# Port of GatherScatterMode.from_any. Modes are canonical strings; parsing is idempotent.
gs_mode_from_any <- function(s) {
  if (is.null(s)) return("fill_or_drop")
  if (!is.character(s) || length(s) != 1L)
    jax_error("Unknown gather mode \"", format(s), "\"", class = "jax_value_error")
  switch(s,
    clip = "clip",
    fill = , drop = , fill_or_drop = "fill_or_drop",
    promise_in_bounds = "promise_in_bounds",
    one_hot = "one_hot",
    jax_error("Unknown gather mode \"", s, "\"", class = "jax_value_error"))
}

#' Out of bounds modes for gather and scatter
#'
#' Port of `jax.lax.GatherScatterMode`. `CLIP` clamps indices so that the whole
#' window is in range. `FILL_OR_DROP` fills gathered windows that are out of bounds
#' with a constant and drops scattered windows that are out of bounds.
#' `PROMISE_IN_BOUNDS` performs no checking (with XLA, gathers clamp and scatters
#' drop). `from_any()` parses `"clip"`, `"fill"`, `"drop"`, `"promise_in_bounds"`,
#' `"one_hot"` or `NULL` (fill or drop).
#' @export
lax_GatherScatterMode <- list(CLIP = "clip", FILL_OR_DROP = "fill_or_drop",
                          PROMISE_IN_BOUNDS = "promise_in_bounds", ONE_HOT = "one_hot",
                          from_any = gs_mode_from_any)

# Python tuple formatting for error messages.
gs_tuple <- function(x) paste0("(", paste(x, collapse = ", "), if (length(x) == 1L) ",", ")")

# R data used as indices: integer valued numerics become integer arrays.
gs_as_indices <- function(x) {
  if (is_jax_value(x)) return(x)
  if (is.double(x)) {
    if (length(x) && any(is.na(x) | x != round(x)))
      jax_error("indices must have an integer type", class = "jax_value_error")
    if (all(abs(x) <= .Machine$integer.max)) storage.mode(x) <- "integer"
    else return(as_jax(x, default_int()))
  }
  as_jax(x)
}

# Integer extreme of a dtype as an array like x, exact for 64-bit types (whose
# extremes are not representable as R doubles).
gs_int_extreme_like <- function(x, which = c("max", "min")) {
  which <- match.arg(which)
  dt <- aval_of(x)$dtype
  if (is_unsigned_dtype(dt)) {
    z <- lax_full_like(x, 0)
    return(if (which == "min") z else lax_bitwise_not(z))
  }
  mn <- lax_full_like(x, iinfo(dt)$min)
  if (which == "min") mn else lax_bitwise_not(mn)
}

# ---- validation helpers (ports of _is_sorted, _dims_in_range, ...) --------------------------

gs_is_sorted <- function(dims, op_name, name) {
  if (is.unsorted(dims))
    jax_error(name, " in ", op_name, " op must be sorted; got ", gs_tuple(dims),
              class = "jax_type_error")
}

gs_dims_in_range <- function(dims, rank, op_name, name) {
  for (d in dims) if (d < 0L || d >= rank)
    jax_error("Invalid ", name, " set in ", op_name, " op; valid range is [0, ", rank,
              "); got: ", d, ".", class = "jax_type_error")
}

gs_sorted_dims_in_range <- function(dims, rank, op_name, name) {
  if (!length(dims)) return(invisible())
  invalid <- NULL
  if (dims[1L] < 0L) invalid <- dims[1L]
  else if (dims[length(dims)] >= rank) invalid <- dims[length(dims)]
  # As upstream, an invalid dimension of 0 is not reported.
  if (!is.null(invalid) && invalid != 0L)
    jax_error("Invalid ", name, " set in ", op_name, " op; valid range is [0, ", rank,
              "); got: ", invalid, ".", class = "jax_type_error")
}

gs_no_duplicate_dims <- function(dims, op_name, name) {
  if (anyDuplicated(dims))
    jax_error(name, " in ", op_name, " op must not repeat; got: ", gs_tuple(dims), ".",
              class = "jax_type_error")
}

gs_disjoint_dims <- function(dims1, dims2, op_name, name1, name2) {
  if (length(intersect(dims1, dims2)))
    jax_error(name1, " and ", name2, " in ", op_name, " op must be disjoint; got: ",
              gs_tuple(dims1), " and ", gs_tuple(dims2), ".", class = "jax_type_error")
}

# ---- gather: shape rule -------------------------------------------------------------------

# Port of jax/_src/lax/slicing.py:_gather_shape_rule.
gs_gather_shape_rule <- function(operand, indices, dnums, slice_sizes) {
  terr <- function(...) jax_error(..., class = "jax_type_error")
  offset_dims <- dnums$offset_dims
  collapsed_slice_dims <- dnums$collapsed_slice_dims
  operand_batching_dims <- dnums$operand_batching_dims
  start_indices_batching_dims <- dnums$start_indices_batching_dims
  start_index_map <- dnums$start_index_map
  orank <- length(operand$shape)
  irank <- length(indices$shape)
  index_vector_dim <- irank - 1L
  if (index_vector_dim < 0L)
    terr("Gather index leaf dimension must be within [0, rank(indices) + 1). rank(indices) is ",
         irank, " and gather index leaf dimension is ", index_vector_dim, ".")

  gs_is_sorted(offset_dims, "gather", "offset_dims")
  gs_no_duplicate_dims(offset_dims, "gather", "offset_dims")
  output_shape_rank <- length(offset_dims) + irank - 1L
  for (i in seq_along(offset_dims)) {
    od <- offset_dims[i]
    if (od < 0L || od >= output_shape_rank)
      terr("Offset dimension ", i - 1L, " in gather op is out of bounds; got ", od,
           ", but should have been in [0, ", output_shape_rank, ")")
  }
  if (length(start_index_map) != indices$shape[index_vector_dim + 1L])
    terr("Gather op has ", length(start_index_map), " elements in start_index_map and the ",
         "bound of dimension index_vector_dim=", index_vector_dim, " of indices is ",
         indices$shape[index_vector_dim + 1L], ". These two numbers must be equal.")
  for (i in seq_along(start_index_map)) {
    d <- start_index_map[i]
    if (d < 0L || d >= orank)
      terr("Invalid start_index_map; domain is [0, ", orank, "), got: ", i - 1L, "->", d, ".")
  }
  gs_no_duplicate_dims(start_index_map, "gather", "start_index_map")

  gs_is_sorted(collapsed_slice_dims, "gather", "collapsed_slice_dims")
  gs_sorted_dims_in_range(collapsed_slice_dims, orank, "gather", "collapsed_slice_dims")
  gs_no_duplicate_dims(collapsed_slice_dims, "gather", "collapsed_slice_dims")

  gs_no_duplicate_dims(operand_batching_dims, "gather", "operand_batching_dims")
  gs_is_sorted(operand_batching_dims, "gather", "operand_batching_dims")
  gs_sorted_dims_in_range(operand_batching_dims, orank, "gather", "operand_batching_dims")

  gs_disjoint_dims(collapsed_slice_dims, operand_batching_dims, "gather",
                   "collapsed_slice_dims", "operand_batching_dims")
  gs_disjoint_dims(start_index_map, operand_batching_dims, "gather",
                   "start_index_map", "operand_batching_dims")

  gs_no_duplicate_dims(start_indices_batching_dims, "gather", "start_indices_batching_dims")
  gs_dims_in_range(start_indices_batching_dims, irank, "gather", "start_indices_batching_dims")
  if (index_vector_dim %in% start_indices_batching_dims)
    terr("Gather op cannot have the index vector dimension as a batching dimension; got ",
         gs_tuple(start_indices_batching_dims), ".")
  if (length(operand_batching_dims) != length(start_indices_batching_dims))
    terr("Gather op requires equal numbers of operand_batching_dims and ",
         "start_indices_batching_dims, got ", gs_tuple(operand_batching_dims), " and",
         gs_tuple(start_indices_batching_dims), ".")
  operand_batch_shape <- operand$shape[operand_batching_dims + 1L]
  indices_batch_shape <- indices$shape[start_indices_batching_dims + 1L]
  if (!identical(operand_batch_shape, indices_batch_shape))
    terr("Gather op requires operand batching dimensions and indices batching dimensions ",
         "to have the same shape, got ", gs_tuple(operand_batch_shape), " and ",
         gs_tuple(indices_batch_shape), ".")

  if (orank != length(slice_sizes))
    terr("Gather op must have one slice size for every input dimension; got: ",
         "len(slice_sizes)=", length(slice_sizes), ", input_shape.rank=", orank)
  if (length(slice_sizes) != length(offset_dims) + length(collapsed_slice_dims) +
      length(operand_batching_dims))
    terr("All components of the offset index in a gather op must either be a offset ",
         "dimension or explicitly collapsed/batching; got len(slice_sizes)=",
         length(slice_sizes), ", output_slice_sizes=", gs_tuple(offset_dims),
         ", collapsed_slice_dims=", gs_tuple(collapsed_slice_dims),
         ", operand_batching_dims=", gs_tuple(operand_batching_dims), ".")
  indices_empty <- any(indices$shape == 0L)
  for (i in seq_along(slice_sizes)) {
    ss <- slice_sizes[i]
    if (!indices_empty && !(ss >= 0L && operand$shape[i] >= ss))
      terr("Slice size at index ", i - 1L, " in gather op is out of range, must be within [0, ",
           operand$shape[i], " + 1), got ", ss, ".")
  }
  for (i in seq_along(collapsed_slice_dims)) {
    bound <- slice_sizes[collapsed_slice_dims[i] + 1L]
    if (bound != 1L)
      terr("Gather op can only collapse slice dims with bound 1, but bound is ", bound,
           " for index ", collapsed_slice_dims[i], " at position ", i - 1L, ".")
  }
  for (i in seq_along(operand_batching_dims)) {
    bound <- slice_sizes[operand_batching_dims[i] + 1L]
    if (bound > 1L)
      terr("Gather op can only have operand batching dims with bound 0/1, but bound is ",
           bound, " for index ", operand_batching_dims[i], " at position ", i - 1L, ".")
  }
  gs_gather_shape_computation(indices, dnums, slice_sizes)
}

# Port of jax/_src/lax/slicing.py:_gather_shape_computation.
gs_gather_shape_computation <- function(indices, dnums, slice_sizes) {
  offset_dims <- dnums$offset_dims
  output_shape_rank <- length(offset_dims) + length(indices$shape) - 1L
  indices_shape <- indices$shape[-length(indices$shape)]
  keep <- setdiff(seq_along(slice_sizes) - 1L,
                  c(dnums$collapsed_slice_dims, dnums$operand_batching_dims))
  window <- slice_sizes[keep + 1L]
  out <- integer(output_shape_rank)
  is_offset <- (seq_len(output_shape_rank) - 1L) %in% offset_dims
  out[is_offset] <- window[seq_len(sum(is_offset))]
  out[!is_offset] <- indices_shape[seq_len(sum(!is_offset))]
  as.integer(out)
}

# ---- gather: primitive ---------------------------------------------------------------------

gather_p <- new_primitive("gather")

# Port of _gather_dtype_rule and _gather_shape_rule; weak type follows the operand.
gather_p$abstract_eval <- function(avals, params) {
  operand <- avals[[1L]]
  indices <- avals[[2L]]
  if (!is_int_dtype(indices$dtype))
    jax_error("indices must have an integer type", class = "jax_value_error")
  shape <- gs_gather_shape_rule(operand, indices, params$dimension_numbers, params$slice_sizes)
  new_aval(shape, operand$dtype, operand$weak)
}

# Text of a #stablehlo.gather<...> or #stablehlo.scatter<...> attribute. As the MLIR
# printer does, empty lists and a zero index_vector_dim are omitted.
gs_dnums_attr <- function(kind, fields, index_vector_dim) {
  fields <- fields[vapply(fields, length, 1L) > 0L]
  parts <- if (length(fields))
    paste0(names(fields), " = ", vapply(fields, attr_i64_list, "")) else character(0)
  if (index_vector_dim != 0L) parts <- c(parts, paste0("index_vector_dim = ", index_vector_dim))
  paste0("#stablehlo.", kind, "<", paste(parts, collapse = ", "), ">")
}

# Port of jax/_src/lax/slicing.py:_gather_fill: a FILL_OR_DROP gather is a
# PROMISE_IN_BOUNDS gather followed by masking.
gs_gather_fill <- function(args, params) {
  operand <- args[[1L]]
  indices <- args[[2L]]
  dnums <- params$dimension_numbers
  slice_sizes <- params$slice_sizes
  oshape <- aval_of(operand)$shape
  index_dtype <- "int32"
  indices <- lax_convert_element_type(indices, index_dtype)
  num_batch_dims <- length(aval_of(indices)$shape) - 1L
  sim <- dnums$start_index_map
  upper_bound <- as_jax(array(as.integer(oshape[sim + 1L] - slice_sizes[sim + 1L]),
                              length(sim)), index_dtype)
  mask <- lax_bitwise_and(
    lax_ge(indices, 0L),
    lax_le(indices, lax_expand_dims(upper_bound, seq_len(num_batch_dims) - 1L)))
  mask <- lax_reduce_and(mask, num_batch_dims)
  output_ndims <- num_batch_dims + length(dnums$offset_dims)
  batch_dims_in_output <- setdiff(seq_len(output_ndims) - 1L, dnums$offset_dims)
  gather_out <- lax_gather(operand, indices, dnums, slice_sizes,
                           indices_are_sorted = params$indices_are_sorted,
                           mode = "promise_in_bounds")
  out_aval <- aval_of(gather_out)
  fv <- params$fill_value
  fill <- if (is_unsigned_dtype(out_aval$dtype) && isTRUE(fv == iinfo(out_aval$dtype)$max))
    gs_int_extreme_like(gather_out, "max") else lax_full_like(gather_out, fv)
  lax_select(lax_broadcast_in_dim(mask, out_aval$shape, batch_dims_in_output), gather_out, fill)
}

# Port of jax/_src/lax/slicing.py:_gather_lower.
gather_p$lowering <- function(ctx, ins, ai, ao, params) {
  if (params$mode == "fill_or_drop")
    return(lower_fun(gs_gather_fill)(ctx, ins, ai, ao, params))
  if (!(params$mode %in% c("promise_in_bounds", "clip")))
    jax_error("gather cannot be lowered with mode '", params$mode, "'")
  o <- ao[[1L]]
  if (any(o$shape == 0L)) return(full_constant(ctx, 0, o))
  d <- params$dimension_numbers
  emit_op(ctx, "stablehlo.gather", ins, types_of(ai), tensor_type(o), attrs = list(
    dimension_numbers = gs_dnums_attr("gather", list(
      offset_dims = d$offset_dims, collapsed_slice_dims = d$collapsed_slice_dims,
      operand_batching_dims = d$operand_batching_dims,
      start_indices_batching_dims = d$start_indices_batching_dims,
      start_index_map = d$start_index_map), length(ai[[2L]]$shape) - 1L),
    indices_are_sorted = attr_bool(params$indices_are_sorted),
    slice_sizes = attr_i64_array(params$slice_sizes)))
}

# Port of _gather_jvp_rule (tangents of out of bounds windows are zero, not NaN).
defjvp(gather_p, function(g, primals, params) {
  lax_gather(g, primals[[2L]], params$dimension_numbers, params$slice_sizes,
             unique_indices = params$unique_indices,
             indices_are_sorted = params$indices_are_sorted, mode = params$mode,
             fill_value = 0)
}, NULL)

gs_gather_to_scatter_dnums <- function(d) {
  lax_ScatterDimensionNumbers(
    update_window_dims = d$offset_dims, inserted_window_dims = d$collapsed_slice_dims,
    scatter_dims_to_operand_dims = d$start_index_map,
    operand_batching_dims = d$operand_batching_dims,
    scatter_indices_batching_dims = d$start_indices_batching_dims)
}

gs_scatter_to_gather_dnums <- function(d) {
  lax_GatherDimensionNumbers(
    offset_dims = d$update_window_dims, collapsed_slice_dims = d$inserted_window_dims,
    start_index_map = d$scatter_dims_to_operand_dims,
    operand_batching_dims = d$operand_batching_dims,
    start_indices_batching_dims = d$scatter_indices_batching_dims)
}

# Slice sizes of the gather that inverts a scatter with dimension numbers d.
gs_scatter_slice_sizes <- function(operand_shape, updates_shape, d) {
  out <- integer(length(operand_shape))
  pos <- 0L
  for (i in seq_along(operand_shape)) {
    if ((i - 1L) %in% d$inserted_window_dims || (i - 1L) %in% d$operand_batching_dims) {
      out[i] <- 1L
    } else {
      pos <- pos + 1L
      out[i] <- updates_shape[d$update_window_dims[pos] + 1L]
    }
  }
  out
}

# Port of _gather_transpose_fancy (dense accumulation branch): the transpose of a
# gather is a scatter-add into zeros.
gather_p$transpose <- function(ct, args, params) {
  if (is_zero(ct)) return(list(NULL, NULL))
  oa <- args[[1L]]$aval
  zeros <- lax_full(oa$shape, 0, aval_of(ct)$dtype)
  list(lax_scatter_add(zeros, args[[2L]], ct,
                       gs_gather_to_scatter_dnums(params$dimension_numbers),
                       unique_indices = params$unique_indices,
                       indices_are_sorted = params$indices_are_sorted, mode = params$mode),
       NULL)
}

# Port of jax/_src/lax/slicing.py:_gather_batching_rule.
gs_gather_batch <- function(vals, dims, params, axis) {
  operand <- vals[[1L]]
  indices <- vals[[2L]]
  d <- params$dimension_numbers
  slice_sizes <- params$slice_sizes
  if (!is.na(dims[1L]) && is.na(dims[2L])) {
    operand <- moveaxis(operand, dims[1L], 0L)
    dnums <- lax_GatherDimensionNumbers(
      offset_dims = c(0L, d$offset_dims + 1L),
      collapsed_slice_dims = d$collapsed_slice_dims + 1L,
      start_index_map = d$start_index_map + 1L,
      operand_batching_dims = d$operand_batching_dims + 1L,
      start_indices_batching_dims = d$start_indices_batching_dims)
    out <- lax_gather(operand, indices, dnums, c(aval_of(operand)$shape[1L], slice_sizes),
                      unique_indices = params$unique_indices,
                      indices_are_sorted = params$indices_are_sorted, mode = params$mode,
                      fill_value = params$fill_value)
    return(list(out, 0L))
  }
  if (is.na(dims[1L]) && !is.na(dims[2L])) {
    indices <- moveaxis(indices, dims[2L], 0L)
    dnums <- lax_GatherDimensionNumbers(
      offset_dims = d$offset_dims + 1L,
      collapsed_slice_dims = d$collapsed_slice_dims,
      start_index_map = d$start_index_map,
      operand_batching_dims = d$operand_batching_dims,
      start_indices_batching_dims = d$start_indices_batching_dims + 1L)
    # Batched accesses into the same array may no longer be sorted or unique.
    out <- lax_gather(operand, indices, dnums, slice_sizes, unique_indices = FALSE,
                      indices_are_sorted = FALSE, mode = params$mode,
                      fill_value = params$fill_value)
    return(list(out, 0L))
  }
  operand <- moveaxis(operand, dims[1L], 0L)
  indices <- moveaxis(indices, dims[2L], 0L)
  lead <- if (aval_of(operand)$shape[1L] == 0L) 0L else 1L
  dnums <- lax_GatherDimensionNumbers(
    offset_dims = d$offset_dims + 1L,
    collapsed_slice_dims = d$collapsed_slice_dims + 1L,
    start_index_map = d$start_index_map + 1L,
    operand_batching_dims = c(0L, d$operand_batching_dims + 1L),
    start_indices_batching_dims = c(0L, d$start_indices_batching_dims + 1L))
  out <- lax_gather(operand, indices, dnums, c(lead, slice_sizes),
                    unique_indices = params$unique_indices,
                    indices_are_sorted = params$indices_are_sorted, mode = params$mode,
                    fill_value = params$fill_value)
  list(out, 0L)
}
gather_p$batch <- gs_gather_batch

#' Gather
#'
#' Port of `jax.lax.gather` (XLA Gather). Most code should use NumPy style indexing
#' (`x[i]`, [at_get()]) instead.
#' @param operand Array from which slices are taken.
#' @param start_indices Integer array of start indices; its last dimension is the
#'   index vector dimension.
#' @param dimension_numbers A [lax_GatherDimensionNumbers()] value.
#' @param slice_sizes Size of each slice, one entry per operand dimension.
#' @param unique_indices Whether the gathered elements are guaranteed not to overlap.
#' @param indices_are_sorted Whether the indices are known to be sorted.
#' @param mode Out of bounds mode: `"clip"`, `"fill"` (or `"drop"`),
#'   `"promise_in_bounds"`; `NULL` means `"promise_in_bounds"`. See
#'   [lax_GatherScatterMode].
#' @param fill_value Value of out of bounds slices in fill mode. Defaults to NaN for
#'   inexact types, the minimum for signed integers, the maximum for unsigned
#'   integers and `TRUE` for booleans.
#' @return The gathered array.
#' @export
lax_gather <- function(operand, start_indices, dimension_numbers, slice_sizes,
                       unique_indices = FALSE, indices_are_sorted = FALSE, mode = NULL,
                       fill_value = NULL) {
  operand <- as_jax(operand)
  parsed_mode <- if (is.null(mode)) "promise_in_bounds" else gs_mode_from_any(mode)
  if (parsed_mode == "fill_or_drop") {
    if (is.null(fill_value)) {
      dt <- aval_of(operand)$dtype
      fill_value <- if (is_inexact_dtype(dt)) NaN
                    else if (is_signed_dtype(dt)) iinfo(dt)$min
                    else if (is_unsigned_dtype(dt)) iinfo(dt)$max
                    else if (dt == "bool") TRUE
                    else jax_error("Unsupported dtype for gather fill_value ", dt,
                                   class = "jax_value_error")
    }
  } else {
    fill_value <- NULL
  }
  bind(gather_p, list(operand, gs_as_indices(start_indices)),
       list(dimension_numbers = dimension_numbers, slice_sizes = as.integer(slice_sizes),
            unique_indices = isTRUE(unique_indices),
            indices_are_sorted = isTRUE(indices_are_sorted), mode = parsed_mode,
            fill_value = fill_value))
}

# ---- scatter: shape rule ------------------------------------------------------------------

# Port of jax/_src/lax/slicing.py:_get_updates_batching_dims.
gs_updates_batching_dims <- function(indices_batching_dims, update_window_dims,
                                     index_vector_dim) {
  v <- seq_len(index_vector_dim) - 1L
  for (i in update_window_dims) v <- append(v, NA_integer_, after = i)
  vapply(indices_batching_dims, function(i) which(v == i)[1L] - 1L, 1L)
}

# Port of jax/_src/lax/slicing.py:_scatter_shape_rule.
gs_scatter_shape_rule <- function(operand, indices, updates, dnums) {
  terr <- function(...) jax_error(..., class = "jax_type_error")
  update_window_dims <- dnums$update_window_dims
  inserted_window_dims <- dnums$inserted_window_dims
  operand_batching_dims <- dnums$operand_batching_dims
  scatter_indices_batching_dims <- dnums$scatter_indices_batching_dims
  scatter_dims_to_operand_dims <- dnums$scatter_dims_to_operand_dims
  orank <- length(operand$shape)
  irank <- length(indices$shape)
  urank <- length(updates$shape)
  index_vector_dim <- irank - 1L
  if (index_vector_dim < 0L)
    terr("Scatter index leaf dimension must be within [0, rank(indices) + 1). rank(indices) is ",
         irank, " and scatter index leaf dimension is ", index_vector_dim, ".")
  expected_updates_rank <- irank - 1L + length(update_window_dims)
  if (urank != expected_updates_rank)
    terr("Updates tensor must be of rank ", expected_updates_rank, "; got ", urank, ".")

  gs_is_sorted(update_window_dims, "scatter", "update_window_dims")
  gs_no_duplicate_dims(update_window_dims, "scatter", "update_window_dims")
  gs_sorted_dims_in_range(update_window_dims, urank, "scatter", "update_window_dims")

  gs_is_sorted(inserted_window_dims, "scatter", "inserted_window_dims")
  gs_no_duplicate_dims(inserted_window_dims, "scatter", "inserted_window_dims")
  gs_sorted_dims_in_range(inserted_window_dims, orank, "scatter", "inserted_window_dims")

  gs_is_sorted(operand_batching_dims, "scatter", "operand_batching_dims")
  gs_no_duplicate_dims(operand_batching_dims, "scatter", "operand_batching_dims")
  gs_sorted_dims_in_range(operand_batching_dims, orank, "scatter", "operand_batching_dims")
  gs_disjoint_dims(inserted_window_dims, operand_batching_dims, "scatter",
                   "inserted_window_dims", "operand_batching_dims")
  gs_disjoint_dims(scatter_dims_to_operand_dims, operand_batching_dims, "scatter",
                   "scatter_dims_to_operand_dims", "operand_batching_dims")

  gs_no_duplicate_dims(scatter_indices_batching_dims, "scatter",
                       "scatter_indices_batching_dims")
  gs_dims_in_range(scatter_indices_batching_dims, irank, "scatter",
                   "scatter_indices_batching_dims")
  if (index_vector_dim %in% scatter_indices_batching_dims)
    terr("Scatter op cannot have the index vector dimension as a batching dimension; got ",
         gs_tuple(scatter_indices_batching_dims), ".")
  if (length(operand_batching_dims) != length(scatter_indices_batching_dims))
    terr("Scatter op requires equal numbers of operand_batching_dims and ",
         "scatter_indices_batching_dims, got ", gs_tuple(operand_batching_dims), " and ",
         gs_tuple(scatter_indices_batching_dims), ".")
  operand_batch_shape <- operand$shape[operand_batching_dims + 1L]
  indices_batch_shape <- indices$shape[scatter_indices_batching_dims + 1L]
  if (!identical(operand_batch_shape, indices_batch_shape))
    terr("Scatter op requires operand batching dimensions and indices batching dimensions ",
         "to have the same shape, got ", gs_tuple(operand_batch_shape), " and ",
         gs_tuple(indices_batch_shape), ".")
  updates_batching_dims <- gs_updates_batching_dims(
    scatter_indices_batching_dims, update_window_dims, index_vector_dim)
  updates_batch_shape <- updates$shape[updates_batching_dims + 1L]
  if (!identical(operand_batch_shape, updates_batch_shape))
    terr("Scatter op requires operand batching dimensions and updates batching dimensions ",
         "to have the same shape, got ", gs_tuple(operand_batch_shape), " and ",
         gs_tuple(updates_batch_shape), ".")

  window_size <- length(update_window_dims) + length(inserted_window_dims) +
    length(operand_batching_dims)
  if (orank != window_size)
    terr("Scatter op has window of size ", window_size, "; doesn't match operand of rank ",
         orank, ".")
  if (length(scatter_dims_to_operand_dims) != indices$shape[index_vector_dim + 1L])
    terr("Scatter op has ", length(scatter_dims_to_operand_dims), " elements in ",
         "scatter_dims_to_operand_dims and the bound of dimension index_vector_dim=",
         index_vector_dim, " of indices is ", indices$shape[index_vector_dim + 1L],
         ". These two numbers must be equal")
  for (i in seq_along(scatter_dims_to_operand_dims)) {
    d <- scatter_dims_to_operand_dims[i]
    if (d < 0L || d >= orank)
      terr("Invalid scatter_dims_to_operand_dims mapping; domain is [0, ", orank, "), got: ",
           i - 1L, "->", d, ".")
  }
  gs_no_duplicate_dims(scatter_dims_to_operand_dims, "scatter", "scatter_dims_to_operand_dims")

  keep <- setdiff(seq_len(orank) - 1L, c(inserted_window_dims, operand_batching_dims))
  max_update_slice_sizes <- operand$shape[keep + 1L]
  for (i in seq_along(update_window_dims)) {
    uwd <- update_window_dims[i]
    if (max_update_slice_sizes[i] < updates$shape[uwd + 1L])
      terr("Bounds of the window dimensions of updates must not exceed the bounds of the ",
           "corresponding dimensions of operand. For dimension ", uwd, ", updates bound is ",
           updates$shape[uwd + 1L], ", operand bound is ", max_update_slice_sizes[i], ".")
  }
  update_scatter_dims <- setdiff(seq_len(urank) - 1L, update_window_dims)
  scatter_dims_seen <- 0L
  for (i in update_scatter_dims) {
    if (scatter_dims_seen == index_vector_dim) scatter_dims_seen <- scatter_dims_seen + 1L
    if (updates$shape[i + 1L] != indices$shape[scatter_dims_seen + 1L])
      terr("Bounds of the scatter dimensions of updates must be the same as the bounds of ",
           "the corresponding dimensions of scatter indices. For scatter dimension ", i,
           ", updates bound is ", updates$shape[i + 1L], ", indices bound is ",
           indices$shape[scatter_dims_seen + 1L], ".")
    scatter_dims_seen <- scatter_dims_seen + 1L
  }
  operand$shape
}

# ---- scatter: primitives -------------------------------------------------------------------

# Port of jax/_src/lax/slicing.py:_clamp_scatter_indices.
gs_clamp_scatter_indices <- function(operand, indices, updates, dnums) {
  oshape <- aval_of(operand)$shape
  ia <- aval_of(indices)
  slice_sizes <- gs_scatter_slice_sizes(oshape, aval_of(updates)$shape, dnums)
  sdod <- dnums$scatter_dims_to_operand_dims
  upper <- oshape[sdod + 1L] - slice_sizes[sdod + 1L]
  upper <- pmin(upper, min(iinfo("int32")$max, iinfo(ia$dtype)$max))
  upper_bound <- lax_convert_element_type(as_jax(array(as.integer(upper), length(upper))),
                                          ia$dtype)
  upper_bound <- lax_broadcast_in_dim(upper_bound, ia$shape, length(ia$shape) - 1L)
  lax_clamp(0L, indices, upper_bound)
}

# Port of _scatter_dtype_rule, _scatter_shape_rule and _scatter_lower. The update
# computation of the named scatter variants is the fixed binary op `hlo` (upstream
# stores the equivalent jaxpr in the equation); only scatter_apply carries a jaxpr.
gs_def_scatter <- function(name, hlo = NULL, kinds = "any", reducer = NULL) {
  p <- new_primitive(name)
  p$abstract_eval <- function(avals, params) {
    operand <- avals[[1L]]
    indices <- avals[[2L]]
    updates <- avals[[3L]]
    if (!is_int_dtype(indices$dtype))
      jax_error("indices must have an integer type", class = "jax_value_error")
    if (operand$dtype != updates$dtype)
      jax_error("lax.scatter requires arguments to have the same dtypes, got ",
                operand$dtype, ", ", updates$dtype, ".", class = "jax_type_error")
    if (!kind_ok(operand$dtype, kinds))
      jax_error(reducer, " does not accept dtype ", operand$dtype, " at position 0. ",
                "Accepted dtypes at position 0 are subtypes of integer, floating, ",
                "complexfloating.", class = "jax_type_error")
    gs_scatter_shape_rule(operand, indices, updates, params$dimension_numbers)
    new_aval(operand$shape, operand$dtype, operand$weak)
  }
  p$lowering <- function(ctx, ins, ai, ao, params) {
    o <- ao[[1L]]
    indices <- ins[[2L]]
    if (params$mode == "clip") {
      clip_fn <- lower_fun(function(args, params) {
        gs_clamp_scatter_indices(args[[1L]], args[[2L]], args[[3L]], params$dimension_numbers)
      })
      indices <- clip_fn(ctx, ins, ai, NULL, params)[[1L]]
    }
    d <- params$dimension_numbers
    st <- tensor_type(new_aval(integer(0), o$dtype))
    jaxpr <- params$update_jaxpr
    region <- build_region(ctx, c(st, st), function(sub, args) {
      v <- if (!is.null(jaxpr)) {
        consts <- lapply(params$update_consts, function(c) {
          ir_constant(sub, array_host(c), aval_of(c))
        })
        jaxpr_subcomp(sub, jaxpr$jaxpr, list(), c(consts, as.list(args)))
      } else if (is.null(hlo)) {
        args[[2L]]
      } else {
        emit_op(sub, hlo, args, c(st, st), st)
      }
      list(values = v, types = st)
    })
    emit_op(ctx, "stablehlo.scatter", c(ins[[1L]], indices, ins[[3L]]), types_of(ai),
            tensor_type(o), attrs = list(
              indices_are_sorted = attr_bool(params$indices_are_sorted),
              scatter_dimension_numbers = gs_dnums_attr("scatter", list(
                update_window_dims = d$update_window_dims,
                inserted_window_dims = d$inserted_window_dims,
                input_batching_dims = d$operand_batching_dims,
                scatter_indices_batching_dims = d$scatter_indices_batching_dims,
                scatter_dims_to_operand_dims = d$scatter_dims_to_operand_dims),
                length(ai[[2L]]$shape) - 1L),
              unique_indices = attr_bool(params$unique_indices)),
            regions = region)
  }
  p$batch <- gs_scatter_batch(p)
  p
}

# Port of jax/_src/lax/slicing.py:_scatter_batching_rule.
gs_scatter_batch <- function(prim) {
  function(vals, dims, params, axis) {
    size <- axis$size
    operand <- bdim_at_front(vals[[1L]], dims[1L], size)
    indices <- vals[[2L]]
    updates <- bdim_at_front(vals[[3L]], dims[3L], size)
    d <- params$dimension_numbers
    if (is.na(dims[2L])) {
      dnums <- lax_ScatterDimensionNumbers(
        update_window_dims = c(0L, d$update_window_dims + 1L),
        inserted_window_dims = d$inserted_window_dims + 1L,
        scatter_dims_to_operand_dims = d$scatter_dims_to_operand_dims + 1L,
        operand_batching_dims = d$operand_batching_dims + 1L,
        scatter_indices_batching_dims = d$scatter_indices_batching_dims)
    } else {
      # See the third case of the gather batching rule.
      indices <- bdim_at_front(indices, dims[2L], size)
      dnums <- lax_ScatterDimensionNumbers(
        update_window_dims = d$update_window_dims + 1L,
        inserted_window_dims = d$inserted_window_dims + 1L,
        scatter_dims_to_operand_dims = d$scatter_dims_to_operand_dims + 1L,
        operand_batching_dims = c(0L, d$operand_batching_dims + 1L),
        scatter_indices_batching_dims = c(0L, d$scatter_indices_batching_dims + 1L))
    }
    params$dimension_numbers <- dnums
    list(bind(prim, list(operand, indices, updates), params), 0L)
  }
}

scatter_add_p <- gs_def_scatter("scatter-add", "stablehlo.add", "number", "add")
scatter_sub_p <- gs_def_scatter("scatter-sub", "stablehlo.subtract", "number", "sub")
scatter_mul_p <- gs_def_scatter("scatter-mul", "stablehlo.multiply", "number", "mul")
scatter_min_p <- gs_def_scatter("scatter-min", "stablehlo.minimum")
scatter_max_p <- gs_def_scatter("scatter-max", "stablehlo.maximum")
scatter_p <- gs_def_scatter("scatter")

gs_bind_scatter <- function(prim, operand, scatter_indices, updates, dimension_numbers,
                            indices_are_sorted, unique_indices, mode, update_jaxpr = NULL,
                            update_consts = list()) {
  args <- lax_args(operand, updates)
  bind(prim, list(args[[1L]], gs_as_indices(scatter_indices), args[[2L]]),
       list(update_jaxpr = update_jaxpr, update_consts = update_consts,
            dimension_numbers = dimension_numbers,
            indices_are_sorted = isTRUE(indices_are_sorted),
            unique_indices = isTRUE(unique_indices), mode = gs_mode_from_any(mode)))
}

#' Scatter
#'
#' Ports of `jax.lax.scatter` (updates replace values), `scatter_add`,
#' `scatter_sub`, `scatter_mul`, `scatter_min`, `scatter_max` and `scatter_apply`
#' (XLA Scatter). Most code should use [at_set()], [at_add()] and friends instead.
#' With repeated indices, `lax_scatter` applies the updates in an unspecified order.
#' `lax_scatter_apply` is not compatible with automatic differentiation.
#' @param operand Array to which the scatter is applied.
#' @param scatter_indices Integer array of indices; its last dimension is the index
#'   vector dimension.
#' @param updates Updates scattered onto `operand`.
#' @param dimension_numbers A [lax_ScatterDimensionNumbers()] value.
#' @param indices_are_sorted Whether the indices are known to be sorted.
#' @param unique_indices Whether the updated elements are guaranteed not to overlap.
#' @param mode Out of bounds mode: `"clip"` clamps indices, `"fill"` or `"drop"`
#'   (the default, also for `NULL`) and `"promise_in_bounds"` drop out of bounds
#'   updates. See [lax_GatherScatterMode].
#' @return The updated array.
#' @export
lax_scatter <- function(operand, scatter_indices, updates, dimension_numbers,
                        indices_are_sorted = FALSE, unique_indices = FALSE, mode = NULL) {
  gs_bind_scatter(scatter_p, operand, scatter_indices, updates, dimension_numbers,
                  indices_are_sorted, unique_indices, mode)
}

#' @rdname lax_scatter
#' @export
lax_scatter_add <- function(operand, scatter_indices, updates, dimension_numbers,
                            indices_are_sorted = FALSE, unique_indices = FALSE, mode = NULL) {
  gs_bind_scatter(scatter_add_p, operand, scatter_indices, updates, dimension_numbers,
                  indices_are_sorted, unique_indices, mode)
}

#' @rdname lax_scatter
#' @export
lax_scatter_sub <- function(operand, scatter_indices, updates, dimension_numbers,
                            indices_are_sorted = FALSE, unique_indices = FALSE, mode = NULL) {
  gs_bind_scatter(scatter_sub_p, operand, scatter_indices, updates, dimension_numbers,
                  indices_are_sorted, unique_indices, mode)
}

#' @rdname lax_scatter
#' @export
lax_scatter_mul <- function(operand, scatter_indices, updates, dimension_numbers,
                            indices_are_sorted = FALSE, unique_indices = FALSE, mode = NULL) {
  gs_bind_scatter(scatter_mul_p, operand, scatter_indices, updates, dimension_numbers,
                  indices_are_sorted, unique_indices, mode)
}

#' @rdname lax_scatter
#' @export
lax_scatter_min <- function(operand, scatter_indices, updates, dimension_numbers,
                            indices_are_sorted = FALSE, unique_indices = FALSE, mode = NULL) {
  gs_bind_scatter(scatter_min_p, operand, scatter_indices, updates, dimension_numbers,
                  indices_are_sorted, unique_indices, mode)
}

#' @rdname lax_scatter
#' @export
lax_scatter_max <- function(operand, scatter_indices, updates, dimension_numbers,
                            indices_are_sorted = FALSE, unique_indices = FALSE, mode = NULL) {
  gs_bind_scatter(scatter_max_p, operand, scatter_indices, updates, dimension_numbers,
                  indices_are_sorted, unique_indices, mode)
}

#' @rdname lax_scatter
#' @param func Unary function of a scalar applied at each index; repeated indices
#'   apply it several times.
#' @param update_shape Shape of the updates at the given indices.
#' @export
lax_scatter_apply <- function(operand, scatter_indices, func, dimension_numbers,
                              update_shape = integer(0), indices_are_sorted = FALSE,
                              unique_indices = FALSE, mode = NULL) {
  operand <- as_jax(operand)
  dt <- aval_of(operand)$dtype
  unused <- lax_full(as.integer(update_shape), 0, dt)
  # Port of lax._reduction_jaxpr for the computation function(x, y) func(x).
  saval <- new_aval(integer(0), dt)
  cj <- trace_hoisted(function(args) list(as_jax(func(args[[1L]]))), list(saval, saval))
  if (any(vapply(cj$consts, is_tracer, NA)))
    jax_error("Reduction computations can't close over Tracers.",
              class = "jax_not_implemented_error")
  out <- jaxpr_out_avals(cj$jaxpr$jaxpr)[[1L]]
  if (length(out$shape) || out$dtype != dt)
    jax_error("scatter_apply func must map a scalar of dtype ", dt, " to a scalar of the ",
              "same dtype, got ", aval_str(out), class = "jax_type_error")
  gs_bind_scatter(scatter_p, operand, scatter_indices, unused, dimension_numbers,
                  indices_are_sorted, unique_indices, mode, update_jaxpr = cj$jaxpr,
                  update_consts = cj$consts)
}

# ---- scatter: differentiation rules ----------------------------------------------------------

# Inverse gather of a scatter (used by the transpose and JVP rules).
gs_inverse_gather <- function(x, indices, updates_shape, params, ...) {
  d <- params$dimension_numbers
  lax_gather(x, indices, gs_scatter_to_gather_dnums(d),
             gs_scatter_slice_sizes(aval_of(x)$shape, updates_shape, d), ...)
}

# Port of _scatter_addsub_jvp (linear in operand and updates) and
# _scatter_addsub_transpose_rule.
gs_scatter_addsub_transpose <- function(prim) {
  function(ct, args, params) {
    operand <- args[[1L]]
    updates <- args[[3L]]
    if (is_zero(ct)) return(list(NULL, NULL, NULL))
    operand_t <- if (is_undefined(operand)) ct else NULL
    update_t <- NULL
    if (is_undefined(updates)) {
      update_t <- gs_inverse_gather(ct, args[[2L]], updates$aval$shape, params,
                                    mode = params$mode, fill_value = 0)
      if (identical(prim, scatter_sub_p)) update_t <- lax_neg(update_t)
    }
    list(operand_t, NULL, update_t)
  }
}
deflinear(scatter_add_p, gs_scatter_addsub_transpose(scatter_add_p), linear = c(1L, 3L))
deflinear(scatter_sub_p, gs_scatter_addsub_transpose(scatter_sub_p), linear = c(1L, 3L))

# Port of the scatter_mul JVP rules (_scatter_mul_jvp_rhs) and
# _scatter_mul_transpose_rule.
defjvp(scatter_mul_p,
  function(g, primals, params) bind(scatter_mul_p, list(g, primals[[2L]], primals[[3L]]), params),
  NULL,
  function(g, primals, params) {
    if (!params$unique_indices)
      jax_error("scatter_mul gradients are only implemented if `unique_indices=True`",
                class = "jax_not_implemented_error")
    x <- primals[[1L]]
    lax_mul(x, lax_scatter_add(lax_full_like(x, 0), primals[[2L]], g,
                               params$dimension_numbers,
                               indices_are_sorted = params$indices_are_sorted,
                               unique_indices = params$unique_indices, mode = params$mode))
  })
scatter_mul_p$transpose <- function(ct, args, params) {
  operand <- args[[1L]]
  indices <- args[[2L]]
  updates <- args[[3L]]
  if (is_zero(ct)) return(list(NULL, NULL, NULL))
  operand_t <- update_t <- NULL
  if (is_undefined(operand))
    operand_t <- lax_scatter_mul(ct, indices, updates, params$dimension_numbers,
                                 indices_are_sorted = params$indices_are_sorted,
                                 unique_indices = params$unique_indices, mode = params$mode)
  if (is_undefined(updates)) {
    if (!params$unique_indices)
      jax_error("scatter_mul gradients are only implemented if `unique_indices=True`",
                class = "jax_not_implemented_error")
    update_t <- gs_inverse_gather(lax_mul(ct, operand), indices, updates$aval$shape, params,
                                  mode = params$mode, fill_value = 0)
  }
  list(operand_t, NULL, update_t)
}

# Port of jax/_src/lax/slicing.py:_scatter_extremal_jvp.
gs_scatter_extremal_jvp <- function(prim) {
  function(primals, tangents, params) {
    operand <- primals[[1L]]
    indices <- primals[[2L]]
    updates <- primals[[3L]]
    val_out <- bind(prim, primals, params)
    if (is_zero(tangents[[1L]]) && is_zero(tangents[[3L]]))
      return(list(val_out, zero_of(tangent_aval(aval_of(val_out)))))
    g_operand <- instantiate_zero(tangents[[1L]])
    g_updates <- instantiate_zero(tangents[[3L]])
    d <- params$dimension_numbers
    ias <- params$indices_are_sorted
    ui <- params$unique_indices
    mode <- params$mode
    updates_shape <- aval_of(updates)$shape
    gather <- function(x) gs_inverse_gather(x, indices, updates_shape, params,
                                            indices_are_sorted = ias, unique_indices = ui,
                                            mode = mode)
    scatter_add <- function(x, u) lax_scatter_add(x, indices, u, d, indices_are_sorted = ias,
                                                  unique_indices = ui, mode = mode)
    # With several updates contending for the same location, the tangent is the
    # average of the tangents of the winning values.
    initial_vals <- gather(operand)
    target_vals <- gather(val_out)
    successful_updates <- lax_eq(updates, target_vals)
    retained_values <- lax_eq(initial_vals, target_vals)
    zeros_op <- lax_full_like(operand, 0)
    ones_up <- lax_full_like(updates, 1)
    zeros_up <- lax_full_like(updates, 0)
    num_updates <- gather(scatter_add(zeros_op, lax_select(successful_updates, ones_up, zeros_up)))
    num_refs <- gather(scatter_add(zeros_op, ones_up))
    inv_plus_one <- jnp_true_divide(1, jnp_add(num_updates, 1))
    updates_normalizer <- lax_select(retained_values, inv_plus_one,
                                     jnp_true_divide(1, num_updates))
    updates_coef <- lax_select(successful_updates, updates_normalizer, zeros_up)
    operand_normalizer <- lax_select(retained_values, inv_plus_one,
                                     lax_full_like(num_updates, 0))
    operand_coef <- jnp_true_divide(jnp_add(-1, operand_normalizer), num_refs)
    target_tangents <- gather(g_operand)
    tangent_updates <- jnp_add(jnp_multiply(target_tangents, operand_coef),
                               jnp_multiply(g_updates, updates_coef))
    list(val_out, scatter_add(g_operand, tangent_updates))
  }
}
scatter_min_p$jvp <- gs_scatter_extremal_jvp(scatter_min_p)
scatter_max_p$jvp <- gs_scatter_extremal_jvp(scatter_max_p)

# Port of jax/_src/lax/slicing.py:_scatter_jvp.
scatter_p$jvp <- function(primals, tangents, params) {
  if (!is.null(params$update_jaxpr))
    jax_error("scatter_apply JVP not implemented", class = "jax_not_implemented_error")
  operand <- primals[[1L]]
  indices <- primals[[2L]]
  updates <- primals[[3L]]
  dnums <- params$dimension_numbers
  ias <- params$indices_are_sorted
  mode <- params$mode
  if (is_zero(tangents[[1L]]) && is_zero(tangents[[3L]])) {
    val_out <- bind(scatter_p, primals, params)
    return(list(val_out, zero_of(tangent_aval(aval_of(val_out)))))
  }
  g_operand <- instantiate_zero(tangents[[1L]])
  g_updates <- instantiate_zero(tangents[[3L]])
  if (params$unique_indices) {
    # Without overlapping updates the scatter is linear in (operand, updates).
    return(list(bind(scatter_p, primals, params),
                bind(scatter_p, list(g_operand, indices, g_updates), params)))
  }
  # With overlapping indices it is unspecified which update wins, so:
  # a) attach a positive id to each update and scatter the ids,
  # b) gather the scattered ids back,
  # c) mask primal and tangent values that do not correspond to a primal output,
  # d) scatter-add the masked values (so no scatter transpose rule is needed).
  ua <- aval_of(updates)
  ids_shape <- ua$shape
  ids_shape[dnums$update_window_dims + 1L] <- 1L
  num_ids <- prod_int(ids_shape)
  id_dtype <- "uint32"
  update_ids <- lax_add(lax_reshape(lax_iota(id_dtype, num_ids), ids_shape),
                        lax_full(ua$shape, 1, id_dtype))
  scattered_ids <- lax_scatter(lax_full(aval_of(operand)$shape, 0, id_dtype), indices,
                               update_ids, dnums, indices_are_sorted = ias,
                               unique_indices = FALSE, mode = mode)
  gathered_update_ids <- gs_inverse_gather(scattered_ids, indices, ua$shape, params)
  keep_operand <- lax_eq(scattered_ids, lax_full_like(scattered_ids, 0))
  keep_updates <- lax_eq(update_ids, gathered_update_ids)
  masked_operand <- lax_select(keep_operand, operand, lax_full_like(operand, 0))
  masked_updates <- lax_select(keep_updates, updates, lax_full_like(updates, 0))
  masked_g_operand <- lax_select(keep_operand, g_operand, lax_full_like(g_operand, 0))
  masked_g_updates <- lax_select(keep_updates, g_updates, lax_full_like(g_updates, 0))
  val_out <- lax_scatter_add(masked_operand, indices, masked_updates, dnums,
                             indices_are_sorted = ias, unique_indices = FALSE, mode = mode)
  tangent_out <- lax_scatter_add(masked_g_operand, indices, masked_g_updates, dnums,
                                 indices_are_sorted = ias, unique_indices = FALSE, mode = mode)
  list(val_out, tangent_out)
}

# Port of jax/_src/lax/slicing.py:_scatter_transpose_rule.
scatter_p$transpose <- function(ct, args, params) {
  if (!params$unique_indices)
    jax_error("scatter transpose is only implemented where unique_indices=True",
              class = "jax_not_implemented_error")
  operand <- args[[1L]]
  indices <- args[[2L]]
  updates <- args[[3L]]
  if (is_zero(ct)) return(list(NULL, NULL, NULL))
  updates_shape <- arg_aval(updates)$shape
  operand_t <- update_t <- NULL
  if (is_undefined(operand))
    # Zero out cotangent entries that correspond to updated indices.
    operand_t <- lax_scatter(ct, indices, lax_full(updates_shape, 0, aval_of(ct)$dtype),
                             params$dimension_numbers,
                             indices_are_sorted = params$indices_are_sorted,
                             unique_indices = TRUE, mode = params$mode)
  if (is_undefined(updates))
    update_t <- gs_inverse_gather(ct, indices, updates_shape, params, mode = params$mode,
                                  fill_value = 0)
  list(operand_t, NULL, update_t)
}

# ---- index_take and the *_in_dim wrappers ---------------------------------------------------

#' Take elements along several axes
#'
#' Port of `jax.lax.index_take`: `src[idxs[[1]][k], idxs[[2]][k], ...]` along `axes`
#' for every `k`. Indices wrap around.
#' @param src Array.
#' @param idxs List of 1-D integer arrays of equal length, one per entry of `axes`.
#' @param axes 0-based axes of `src` (ascending).
#' @return An array whose leading dimension indexes the taken elements.
#' @export
lax_index_take <- function(src, idxs, axes) {
  src <- as_jax(src)
  axes <- as.integer(axes)
  s <- aval_of(src)$shape
  indices <- lax_concatenate(lapply(idxs, function(i) lax_expand_dims(gs_as_indices(i), 1L)), 1L)
  max_idx <- as_jax(array(s[axes + 1L], c(1L, length(axes))))
  indices <- jnp_remainder(indices, max_idx)
  slice_sizes <- s
  slice_sizes[axes + 1L] <- 1L
  ncols <- aval_of(indices)$shape[2L]
  dnums <- lax_GatherDimensionNumbers(offset_dims = seq_len(length(s) - ncols),
                                  collapsed_slice_dims = axes, start_index_map = axes)
  lax_gather(src, indices, dnums, slice_sizes)
}

# Port of jax/_src/lax/slicing.py:_dynamic_slice_indices. R numeric scalars are static
# indices; jax values are dynamic.
gs_dynamic_slice_indices <- function(shape, start_indices, allow_negative_indices) {
  if (length(start_indices) != length(shape))
    jax_error("Length of slice indices must match number of operand dimensions (",
              length(start_indices), " vs ", gs_tuple(shape), ")", class = "jax_value_error")
  if (is_jax_value(start_indices)) {
    if (length(aval_of(start_indices)$shape) != 1L)
      jax_error("Slice indices must be a 1D sequence, got ",
                gs_tuple(aval_of(start_indices)$shape), class = "jax_value_error")
    v <- start_indices
    start_indices <- lapply(seq_along(shape) - 1L, function(k) {
      lax_index_in_dim(v, k, 0L, keepdims = FALSE)
    })
  } else if (!is.list(start_indices)) {
    start_indices <- as.list(start_indices)
  }
  allow <- rep_len(allow_negative_indices, length(start_indices))
  # Static indices are weakly typed, like Python ints; lax_args() then gives them the
  # dtype of the traced indices, because the primitive wants one index dtype.
  idx <- Map(function(i, d, allow_negative) {
    if (!is_jax_value(i)) {
      if (!is.numeric(i) || length(i) != 1L || i != round(i))
        jax_error("index arguments to dynamic_slice must be integers of the same type",
                  class = "jax_type_error")
      if (i < 0 && !allow_negative)
        jax_error("Index ", i, " is out of bounds for dimension ", d,
                  " if allow_negative_indices=False", class = "jax_value_error")
      return(new_array(new_aval(integer(0), default_int(), TRUE),
                       host = as.double(if (i < 0) i + d else i)))
    }
    if (is_unsigned_dtype(aval_of(i)$dtype) || !allow_negative) return(i)
    lax_select(lax_lt(i, lax_full_like(i, 0)), lax_add(i, lax_full_like(i, d)), i)
  }, start_indices, as.list(shape), as.list(allow))
  do.call(lax_args, unname(idx))
}

# A zero with the type of the index `i` (lax._const(i, 0)).
gs_index_zero <- function(i) if (is_jax_value(i)) lax_full_like(i, 0) else 0L

#' Dynamic slices and updates along one axis
#'
#' Ports of `jax.lax.dynamic_slice_in_dim`, `dynamic_index_in_dim`,
#' `dynamic_update_slice_in_dim` and `dynamic_update_index_in_dim`. The start index
#' may be a traced value; slices that overrun the array are shifted to fit.
#' @param operand Array.
#' @param start_index,index 0-based (possibly traced) scalar index.
#' @param slice_size Static size of the slice.
#' @param axis 0-based axis.
#' @param keepdims Keep the indexed dimension (size 1).
#' @param update Array written into `operand`.
#' @param allow_negative_indices Whether negative indices count from the end. If
#'   `FALSE` negative indices are out of bounds (clamped to the first index).
#' @return An array.
#' @export
lax_dynamic_slice_in_dim <- function(operand, start_index, slice_size, axis = 0L,
                                     allow_negative_indices = TRUE) {
  x <- as_jax(operand)
  s <- aval_of(x)$shape
  ax <- norm_axis(axis, length(s))
  start_index <- if (is_jax_value(start_index) || is.numeric(start_index)) start_index
                 else as_jax(start_index)
  start_indices <- rep(list(gs_index_zero(start_index)), length(s))
  start_indices[[ax + 1L]] <- start_index
  slice_sizes <- s
  slice_sizes[ax + 1L] <- as.integer(slice_size)
  allow <- rep(FALSE, length(s))
  allow[ax + 1L] <- allow_negative_indices
  idx <- gs_dynamic_slice_indices(s, start_indices, allow)
  bind(dynamic_slice_p, c(list(x), idx), list(slice_sizes = slice_sizes))
}

#' @rdname lax_dynamic_slice_in_dim
#' @export
lax_dynamic_index_in_dim <- function(operand, index, axis = 0L, keepdims = TRUE,
                                     allow_negative_indices = TRUE) {
  result <- lax_dynamic_slice_in_dim(operand, index, 1L, axis,
                                     allow_negative_indices = allow_negative_indices)
  if (keepdims) result else lax_squeeze(result, axis)
}

#' @rdname lax_dynamic_slice_in_dim
#' @export
lax_dynamic_update_slice_in_dim <- function(operand, update, start_index, axis,
                                            allow_negative_indices = TRUE) {
  args <- lax_args(operand, update)
  s <- aval_of(args[[1L]])$shape
  ax <- norm_axis(axis, length(s))
  start_indices <- rep(list(gs_index_zero(start_index)), length(s))
  start_indices[[ax + 1L]] <- start_index
  allow <- rep(FALSE, length(s))
  allow[ax + 1L] <- allow_negative_indices
  idx <- gs_dynamic_slice_indices(s, start_indices, allow)
  bind(dynamic_update_slice_p, c(args, idx))
}

#' @rdname lax_dynamic_slice_in_dim
#' @export
lax_dynamic_update_index_in_dim <- function(operand, update, index, axis,
                                            allow_negative_indices = TRUE) {
  operand <- as_jax(operand)
  update <- as_jax(update)
  nd <- length(aval_of(operand)$shape)
  ax <- norm_axis(axis, nd)
  if (length(aval_of(update)$shape) != nd) {
    if (length(aval_of(update)$shape) + 1L != nd)
      jax_error("dynamic_update_index_in_dim: update must have the rank of operand or one ",
                "less", class = "jax_type_error")
    update <- lax_expand_dims(update, ax)
  }
  lax_dynamic_update_slice_in_dim(operand, update, index, ax,
                                  allow_negative_indices = allow_negative_indices)
}

# ---- batching of dynamic_slice / dynamic_update_slice through gather / scatter ----------------

# Port of jax/_src/lax/slicing.py:_batch_dynamic_slice_indices.
gs_batch_dynamic_slice_indices <- function(indices, bdims, size) {
  if (!length(indices)) return(list(index = as_jax(integer(0)), bdim = NA_integer_))
  if (all(is.na(bdims)))
    return(list(index = lax_concatenate(lapply(indices, lax_broadcast, sizes = 1L), 0L),
                bdim = NA_integer_))
  cols <- Map(function(x, d) {
    lax_broadcast_in_dim(x, c(size, 1L), if (is.na(d)) integer(0) else 0L)
  }, indices, bdims)
  list(index = lax_concatenate(cols, 1L), bdim = 0L)
}

# Port of _dynamic_slice_batching_rule: a dynamic slice is a special case of gather.
dynamic_slice_p$batch <- function(vals, dims, params, axis) {
  operand <- vals[[1L]]
  ndims <- length(aval_of(operand)$shape) - (if (is.na(dims[1L])) 0L else 1L)
  dd <- seq_len(ndims) - 1L
  dnums <- lax_GatherDimensionNumbers(offset_dims = dd, collapsed_slice_dims = integer(0),
                                  start_index_map = dd)
  ib <- gs_batch_dynamic_slice_indices(vals[-1L], dims[-1L], axis$size)
  gs_gather_batch(list(operand, ib$index), c(dims[1L], ib$bdim),
                  list(dimension_numbers = dnums, slice_sizes = params$slice_sizes,
                       unique_indices = TRUE, indices_are_sorted = TRUE, mode = "clip",
                       fill_value = NULL), axis)
}

# Port of _dynamic_update_slice_batching_rule: a special case of scatter. Upstream
# calls vmap(scatter); applying the scatter batching rule directly is equivalent.
dynamic_update_slice_p$batch <- function(vals, dims, params, axis) {
  update_rank <- length(aval_of(vals[[2L]])$shape) - (if (is.na(dims[2L])) 0L else 1L)
  dd <- seq_len(update_rank) - 1L
  dnums <- lax_ScatterDimensionNumbers(update_window_dims = dd, inserted_window_dims = integer(0),
                                   scatter_dims_to_operand_dims = dd)
  ib <- gs_batch_dynamic_slice_indices(vals[-(1:2)], dims[-(1:2)], axis$size)
  gs_scatter_batch(scatter_p)(
    list(vals[[1L]], ib$index, vals[[2L]]), c(dims[1L], ib$bdim, dims[2L]),
    list(update_jaxpr = NULL, update_consts = list(), dimension_numbers = dnums,
         indices_are_sorted = TRUE, unique_indices = TRUE, mode = "clip"), axis)
}
