# Windowed reductions (port of jax/_src/lax/windowed_reductions.py) and the window
# shape helpers they share with convolutions (jax/_src/lax/lax.py, jax/_src/core.py).

# ---- shape helpers ------------------------------------------------------------------

# Python style tuple text for error messages: (2,) and (2, 3).
fmt_tuple <- function(x) {
  if (is.list(x)) return(paste0("(", paste(vapply(x, fmt_tuple, ""), collapse = ", "),
                                if (length(x) == 1L) "," else "", ")"))
  x <- format(x, scientific = FALSE, trim = TRUE)
  if (length(x) == 1L) paste0("(", x, ",)") else paste0("(", paste(x, collapse = ", "), ")")
}

# Port of lax._dilate_shape and core.dilate_dim.
dilate_shape <- function(shape, dilation) {
  shape <- as.integer(shape)
  dilation <- as.integer(dilation)
  if (!all(dilation > 0L))
    jax_error("All dilations must be positive, got ", fmt_tuple(dilation), ".",
              class = "jax_type_error")
  dilation <- c(rep(1L, length(shape) - length(dilation)), dilation)
  as.integer(ifelse(dilation == 1L, shape, pmax(0L, 1L + dilation * (shape - 1L))))
}

# Port of core.stride_dim: max(0, (d - window_size) // window_stride + 1).
stride_dim <- function(d, window_size, window_stride) {
  as.integer(pmax((as.integer(d) - as.integer(window_size)) %/% as.integer(window_stride) + 1L, 0L))
}

# Port of lax._check_shapelike.
check_shapelike <- function(fun_name, arg_name, obj, non_zero_shape = FALSE) {
  if (!length(obj)) return(invisible())
  if (!is.numeric(obj) || any(obj != round(obj)))
    jax_error(fun_name, " ", arg_name, " must have every element be an integer type, got ",
              paste(obj, collapse = ", "), ".", class = "jax_type_error")
  if (!all(obj >= if (non_zero_shape) 1 else 0))
    jax_error(fun_name, " ", arg_name, " must have every element be ",
              if (non_zero_shape) "strictly positive" else "nonnegative", ", got ",
              fmt_tuple(obj), ".", class = "jax_type_error")
  invisible()
}

# Explicit (low, high) padding pairs as a list of integer vectors of length 2. Accepts a
# list of pairs or a two column matrix.
normalize_pairs <- function(padding, what = "padding") {
  if (is.matrix(padding) && ncol(padding) == 2L)
    padding <- lapply(seq_len(nrow(padding)), function(i) padding[i, ])
  if (!is.list(padding) || !all(vapply(padding, function(p) is.numeric(p) && length(p) == 2L, NA)))
    jax_error(what, " argument should be a string or a sequence of (low, high) pairs, got ",
              paste(format(padding), collapse = " "), class = "jax_value_error")
  lapply(padding, function(p) as.integer(p))
}

#' Convert a padding specification to explicit padding pairs
#'
#' Port of `jax.lax.padtype_to_pads`.
#' @param in_shape Integer vector, the input spatial shape.
#' @param window_shape Integer vector, the window (kernel) spatial shape.
#' @param window_strides Integer vector, the spatial strides.
#' @param padding One of `"VALID"`, `"SAME"` or `"SAME_LOWER"` (case insensitive).
#' @return A list with one `c(low, high)` integer pair per spatial dimension.
#' @export
lax_padtype_to_pads <- function(in_shape, window_shape, window_strides, padding) {
  if (!is.character(padding) || length(padding) != 1L)
    jax_error("Unknown padding type: ", paste(format(padding), collapse = " "), ".",
              class = "jax_type_error")
  p <- toupper(padding)
  if (!p %in% c("VALID", "SAME", "SAME_LOWER"))
    jax_error("Unrecognized padding type: expected 'VALID', 'SAME', or 'SAME_LOWER', got ",
              padding, ".")
  in_shape <- as.integer(in_shape)
  if (p == "VALID") return(rep(list(c(0L, 0L)), length(in_shape)))
  window_shape <- as.integer(window_shape)
  window_strides <- as.integer(window_strides)
  if (length(window_shape) != length(in_shape) || length(window_strides) != length(in_shape))
    jax_error("padtype_to_pads: in_shape, window_shape and window_strides must have the ",
              "same length, got ", fmt_tuple(in_shape), ", ", fmt_tuple(window_shape), " and ",
              fmt_tuple(window_strides), ".", class = "jax_value_error")
  out_shape <- -((-in_shape) %/% window_strides)
  pad_sizes <- pmax((out_shape - 1L) * window_strides + window_shape - in_shape, 0L)
  lo <- if (p == "SAME") pad_sizes %/% 2L else pad_sizes - pad_sizes %/% 2L
  lapply(seq_along(pad_sizes), function(i) as.integer(c(lo[i], pad_sizes[i] - lo[i])))
}

#' Shape of a windowed reduction
#'
#' Port of `jax.lax.reduce_window_shape_tuple`.
#' @param operand_shape Integer vector.
#' @param window_dimensions,window_strides Integer vectors.
#' @param padding List of `c(low, high)` pairs.
#' @param base_dilation,window_dilation Optional integer vectors.
#' @return An integer vector.
#' @export
lax_reduce_window_shape_tuple <- function(operand_shape, window_dimensions, window_strides,
                                          padding, base_dilation = NULL,
                                          window_dilation = NULL) {
  operand_shape <- as.integer(operand_shape)
  window_dimensions <- as.integer(window_dimensions)
  if (!is.null(base_dilation)) operand_shape <- dilate_shape(operand_shape, base_dilation)
  if (!is.null(window_dilation)) window_dimensions <- dilate_shape(window_dimensions, window_dilation)
  pads <- normalize_pairs(padding)
  if (length(pads) != length(operand_shape) || length(window_strides) != length(operand_shape))
    jax_error("reduce_window got padding ", fmt_tuple(pads), " and window_strides ",
              fmt_tuple(window_strides), " for operand shape ", fmt_tuple(operand_shape),
              "; all must have one entry per dimension.", class = "jax_type_error")
  padded <- operand_shape + vapply(pads, function(p) sum(p), 1L)
  stride_dim(padded, window_dimensions, window_strides)
}

# Port of windowed_reductions._common_reduce_window_shape_rule.
common_reduce_window_shape_rule <- function(a, params) {
  wd <- params$window_dimensions
  ws <- params$window_strides
  bd <- params$base_dilation
  wdl <- params$window_dilation
  check_shapelike("reduce_window", "window_dimensions", wd, non_zero_shape = TRUE)
  check_shapelike("reduce_window", "window_strides", ws, non_zero_shape = TRUE)
  check_shapelike("reduce_window", "base_dilation", bd)
  check_shapelike("reduce_window", "window_dilation", wdl)
  if (length(a$shape) != length(wd))
    jax_error("reduce_window got the wrong number of window_dimensions for operand: got ",
              "operand shape ", fmt_tuple(a$shape), " with window_dimensions ", fmt_tuple(wd),
              ".", class = "jax_type_error")
  if (length(ws) != length(wd))
    jax_error("reduce_window got inconsistent window_strides and window_dimensions: got ",
              "window_strides ", fmt_tuple(ws), " and window_dimensions ", fmt_tuple(wd), ".",
              class = "jax_type_error")
  if (length(bd) != length(wd))
    jax_error("reduce_window got inconsistent base_dilation and window_dimensions: got ",
              "base_dilation ", fmt_tuple(bd), " and window_dimensions ", fmt_tuple(wd), ".",
              class = "jax_type_error")
  if (length(wdl) != length(wd))
    jax_error("reduce_window got inconsistent window_dilation and window_dimensions: got ",
              "window_dilation ", fmt_tuple(wdl), " and window_dimensions ", fmt_tuple(wd), ".",
              class = "jax_type_error")
  lax_reduce_window_shape_tuple(a$shape, wd, ws, params$padding, bd, wdl)
}

# Port of convolution._conv_general_vjp_lhs_padding (shared with reduce_window_sum).
conv_general_vjp_lhs_padding <- function(in_shape, window_dimensions, window_strides,
                                         out_shape, padding, lhs_dilation, rhs_dilation) {
  lhs_dilated_shape <- dilate_shape(in_shape, lhs_dilation)
  rhs_dilated_shape <- dilate_shape(window_dimensions, rhs_dilation)
  out_dilated_shape <- dilate_shape(out_shape, window_strides)
  lo <- vapply(padding, `[`, 1L, 1L)
  pad_before <- rhs_dilated_shape - lo - 1L
  pad_after <- lhs_dilated_shape + rhs_dilated_shape - 1L - out_dilated_shape - pad_before
  lapply(seq_along(pad_before), function(i) as.integer(c(pad_before[i], pad_after[i])))
}

# ---- lowering helpers ---------------------------------------------------------------

# Dense i64 attribute of shape (n, 2) (padding of reduce_window, select_and_scatter and
# convolution).
attr_i64_pairs <- function(pairs) {
  n <- length(pairs)
  if (!n) return("dense<> : tensor<0x2xi64>")
  rows <- vapply(pairs, function(p) paste0("[", paste(int_literal(p), collapse = ", "), "]"), "")
  paste0("dense<[", paste(rows, collapse = ", "), "]> : tensor<", n, "x2xi64>")
}

scalar_type <- function(dt) tensor_type(new_aval(integer(0), dt))

# Exact literal of the extreme value of an integer dtype (64 bit extremes are not
# representable as R doubles).
int_extreme_literal <- function(dt, which) {
  if (dtype_bits(dt) == 64L) {
    return(switch(paste0(dt, "_", which), int64_min = "-9223372036854775808",
                  int64_max = "9223372036854775807", uint64_min = "0",
                  uint64_max = "18446744073709551615"))
  }
  int_literal(iinfo(dt)[[which]])
}

# Scalar constant holding lax._get_max_identity ("max"), lax._get_min_identity ("min"),
# lax._get_sum_identity ("sum") or lax._get_prod_identity ("prod") of dtype dt.
identity_constant <- function(ctx, which, dt) {
  sa <- new_aval(integer(0), dt)
  k <- dtype_kind(dt)
  if (which %in% c("sum", "prod")) return(ir_constant(ctx, if (which == "sum") 0 else 1, sa))
  if (k == "b") return(ir_constant(ctx, which == "min", sa))
  inf <- if (which == "max") -Inf else Inf
  if (dt %in% c("float16", "bfloat16")) return(ir_constant(ctx, inf, sa))
  lit <- switch(k,
    f = float_hex(inf, dt),
    c = paste0("(", float_hex(inf, complex_component(dt)), ", ",
               float_hex(0, complex_component(dt)), ")"),
    int_extreme_literal(dt, if (which == "max") "min" else "max"))
  ty <- scalar_type(dt)
  emit_op(ctx, "stablehlo.constant", character(0), character(0), ty,
          attrs = list(value = paste0("dense<", lit, "> : ", ty)))
}

# The same identities as jax scalars (for rules written with lax functions).
identity_value <- function(which, dt) {
  k <- dtype_kind(dt)
  if (k %in% c("i", "u") && dtype_bits(dt) == 64L) {
    lo <- lax_full(integer(0), if (k == "i") -2^63 else 0, dt)
    return(if (which == "max") lo else lax_bitwise_not(lo))
  }
  v <- if (which == "max") identity_max(dt) else identity_min(dt)
  lax_full(integer(0), if (k == "c") complex(real = v, imaginary = 0) else v, dt)
}

# Port of mlir._minmax_hlo (min_hlo / max_hlo): complex values compare
# lexicographically as (real, imag) pairs.
minmax_hlo <- function(ctx, op, cmp, x, y, dt) {
  t <- scalar_type(dt)
  if (!is_complex_dtype(dt)) return(emit_op(ctx, op, c(x, y), c(t, t), t))
  rt <- scalar_type(complex_component(dt))
  bt <- scalar_type("bool")
  cmp_op <- function(a, b, dir) {
    emit_op(ctx, "stablehlo.compare", c(a, b), c(rt, rt), bt,
            attrs = list(comparison_direction = paste0("#stablehlo<comparison_direction ", dir, ">"),
                         compare_type = "#stablehlo<comparison_type FLOAT>"))
  }
  rx <- emit_op(ctx, "stablehlo.real", x, t, rt)
  ry <- emit_op(ctx, "stablehlo.real", y, t, rt)
  real_eq <- cmp_op(rx, ry, "EQ")
  real_cmp <- cmp_op(rx, ry, cmp)
  ix <- emit_op(ctx, "stablehlo.imag", x, t, rt)
  iy <- emit_op(ctx, "stablehlo.imag", y, t, rt)
  imag_cmp <- cmp_op(ix, iy, cmp)
  which <- emit_op(ctx, "stablehlo.select", c(real_eq, imag_cmp, real_cmp), c(bt, bt, bt), bt)
  emit_op(ctx, "stablehlo.select", c(which, x, y), c(bt, t, t), t)
}

# Port of mlir.reduce_window (static shapes): a stablehlo.reduce_window whose region is
# built by body(sub_ctx, region_args) returning the SSA names to return.
hlo_reduce_window <- function(ctx, operands, inits, avals_op, init_dtypes, avals_out,
                              params, body) {
  sts <- vapply(init_dtypes, scalar_type, "")
  region <- build_region(ctx, c(sts, sts), function(sub, args) {
    list(values = body(sub, args), types = sts)
  })
  emit_op(ctx, "stablehlo.reduce_window", c(operands, inits), c(types_of(avals_op), sts),
          vapply(avals_out, tensor_type, ""),
          attrs = list(base_dilations = attr_i64_array(params$base_dilation),
                       padding = attr_i64_pairs(params$padding),
                       window_dilations = attr_i64_array(params$window_dilation),
                       window_dimensions = attr_i64_array(params$window_dimensions),
                       window_strides = attr_i64_array(params$window_strides)),
          regions = region)
}

# Lower a closed reduction jaxpr into the current region (consts become constants).
lower_reduction_jaxpr <- function(sub, cj, args) {
  consts <- lapply(seq_along(cj$consts), function(i) {
    ir_constant(sub, array_host(cj$consts[[i]]), cj$jaxpr$constvars[[i]]$aval)
  })
  jaxpr_subcomp(sub, cj$jaxpr, consts, as.list(args))
}

# Port of lax._reduction_jaxpr / lax._variadic_reduction_jaxpr: trace
# computation(a, b) on scalars with the structure of the init values.
reduction_jaxpr <- function(computation, init_dtypes, init_treedef = NULL, what = "reduce_window") {
  n <- length(init_dtypes)
  savals <- lapply(init_dtypes, function(dt) new_aval(integer(0), dt))
  store <- new.env(parent = emptyenv())
  cj <- trace_to_jaxpr(function(args) {
    a <- args[seq_len(n)]
    b <- args[n + seq_len(n)]
    out <- if (is.null(init_treedef)) computation(a[[1L]], b[[1L]])
           else computation(tree_unflatten(init_treedef, a), tree_unflatten(init_treedef, b))
    fl <- tree_flatten(out)
    store$out_tree <- fl$treedef
    lapply(fl$leaves, as_jax)
  }, c(savals, savals))
  if (any(vapply(cj$consts, is_tracer, NA)))
    jax_error(what, " computation must not close over traced values")
  list(jaxpr = cj, out_tree = store$out_tree)
}

window_params <- function(window_dimensions, window_strides, padding, base_dilation = NULL,
                          window_dilation = NULL) {
  n <- length(window_dimensions)
  list(window_dimensions = as.integer(window_dimensions),
       window_strides = as.integer(window_strides),
       padding = normalize_pairs(padding),
       base_dilation = as.integer(base_dilation %||% rep(1L, n)),
       window_dilation = as.integer(window_dilation %||% rep(1L, n)))
}

# Port of windowed_reductions._reduce_window_batch_rule.
window_batch_rule <- function(prim) {
  function(vals, dims, params, axis) {
    d <- dims[1L]
    ins <- function(v, x) append(v, x, after = d)
    np <- params
    np$window_dimensions <- ins(params$window_dimensions, 1L)
    np$window_strides <- ins(params$window_strides, 1L)
    np$padding <- append(params$padding, list(c(0L, 0L)), after = d)
    np$base_dilation <- ins(params$base_dilation, 1L)
    np$window_dilation <- ins(params$window_dilation, 1L)
    list(bind(prim, vals, np), d)
  }
}

# ---- reduce_window_sum / max / min --------------------------------------------------

def_monoid_window <- function(name, which, hlo, cmp = NULL) {
  p <- new_primitive(name)
  p$abstract_eval <- function(avals, params) {
    a <- avals[[1L]]
    if (which == "sum" && !kind_ok(a$dtype, "number"))
      jax_error("operand to reduce_window_sum must have a number dtype, got ", a$dtype,
                class = "jax_type_error")
    new_aval(common_reduce_window_shape_rule(a, params), a$dtype, a$weak)
  }
  # Port of windowed_reductions._reduce_window_lower.
  p$lowering <- function(ctx, ins, ai, ao, params) {
    dt <- ai[[1L]]$dtype
    init <- identity_constant(ctx, which, dt)
    hlo_reduce_window(ctx, ins, init, ai, dt, ao, params, function(sub, args) {
      if (is.null(cmp)) {
        t <- scalar_type(dt)
        emit_op(sub, hlo, args, c(t, t), t)
      } else {
        minmax_hlo(sub, hlo, cmp, args[[1L]], args[[2L]], dt)
      }
    })
  }
  p$batch <- window_batch_rule(p)
  p
}

reduce_window_sum_p <- def_monoid_window("reduce_window_sum", "sum", "stablehlo.add")
reduce_window_max_p <- def_monoid_window("reduce_window_max", "max", "stablehlo.maximum", "GT")
reduce_window_min_p <- def_monoid_window("reduce_window_min", "min", "stablehlo.minimum", "LT")

# Ports of windowed_reductions._reduce_window_sum / _max / _min.
reduce_window_monoid <- function(prim) {
  function(operand, window_dimensions, window_strides, padding, base_dilation = NULL,
           window_dilation = NULL) {
    bind(prim, list(as_jax(operand)),
         window_params(window_dimensions, window_strides, padding, base_dilation, window_dilation))
  }
}
reduce_window_sum <- reduce_window_monoid(reduce_window_sum_p)
reduce_window_max <- reduce_window_monoid(reduce_window_max_p)
reduce_window_min <- reduce_window_monoid(reduce_window_min_p)

# Port of windowed_reductions._reduce_window_sum_transpose_rule.
deflinear(reduce_window_sum_p, function(ct, args, params) {
  a <- args[[1L]]$aval
  if (is_zero(ct)) return(list(zero_of(a)))
  input_shape <- a$shape
  pads <- conv_general_vjp_lhs_padding(input_shape, params$window_dimensions,
                                       params$window_strides, aval_of(ct)$shape, params$padding,
                                       params$base_dilation, params$window_dilation)
  ones <- rep(1L, length(input_shape))
  padding_config <- Map(function(p, s) c(p[1L], p[2L], s - 1L), pads, params$window_strides)
  pad_ct <- lax_pad(ct, lax_full(integer(0), 0, aval_of(ct)$dtype), padding_config)
  result <- reduce_window_sum(pad_ct, params$window_dimensions, params$base_dilation,
                              rep(list(c(0L, 0L)), length(input_shape)),
                              base_dilation = ones, window_dilation = params$window_dilation)
  list(result)
})

# Port of windowed_reductions._reduce_window_chooser_jvp_rule.
reduce_window_chooser_jvp <- function(select_prim) {
  function(g, primals, params) {
    select_and_gather_add(g, primals[[1L]], select_prim, params$window_dimensions,
                          params$window_strides, params$padding, params$base_dilation,
                          params$window_dilation)
  }
}
defjvp(reduce_window_max_p, reduce_window_chooser_jvp("ge"))
defjvp(reduce_window_min_p, reduce_window_chooser_jvp("le"))

# Ports of windowed_reductions._reduce_window_prod and _reduce_window_logaddexp.
reduce_window_prod <- function(operand, window_dimensions, window_strides, padding,
                               base_dilation = NULL, window_dilation = NULL) {
  x <- as_jax(operand)
  dt <- aval_of(x)$dtype
  init <- lax_full(integer(0), 1, dt)
  rj <- reduction_jaxpr(lax_mul, dt)
  bind(reduce_window_p, list(x, init),
       c(list(jaxpr = rj$jaxpr),
         window_params(window_dimensions, window_strides, padding, base_dilation,
                       window_dilation)))[[1L]]
}

reduce_window_logaddexp <- function(operand, window_dimensions, window_strides, padding,
                                    base_dilation = NULL, window_dilation = NULL) {
  x <- as_jax(operand)
  dt <- aval_of(x)$dtype
  init <- lax_full(integer(0), -Inf, dt)
  rj <- reduction_jaxpr(lax_other_logaddexp, dt)
  bind(reduce_window_p, list(x, init),
       c(list(jaxpr = rj$jaxpr),
         window_params(window_dimensions, window_strides, padding, base_dilation,
                       window_dilation)))[[1L]]
}

# ---- generic reduce_window ------------------------------------------------------------

reduce_window_p <- new_primitive("reduce_window", multiple_results = TRUE)

# Port of windowed_reductions._reduce_window_abstract_eval_rule.
reduce_window_p$abstract_eval <- function(avals, params) {
  n <- length(avals) %/% 2L
  ops <- avals[seq_len(n)]
  inits <- avals[n + seq_len(n)]
  odt <- vapply(ops, `[[`, "", "dtype")
  idt <- vapply(inits, `[[`, "", "dtype")
  if (any(odt != idt))
    jax_error("reduce_window got inconsistent dtypes for operands and init_values: got ",
              "operand dtypes [", paste(odt, collapse = ", "), "] and init_value dtypes [",
              paste(idt, collapse = ", "), "].", class = "jax_type_error")
  if (any(vapply(inits, function(a) length(a$shape) != 0L, NA)))
    jax_error("reduce_window expected init_values to be scalars but init_values have shapes [",
              paste(vapply(inits, function(a) fmt_tuple(a$shape), ""), collapse = ", "), "].",
              class = "jax_type_error")
  out_shape <- common_reduce_window_shape_rule(ops[[1L]], params)
  lapply(ops, function(o) new_aval(out_shape, o$dtype))
}

# Port of windowed_reductions._generic_reduce_window_lower.
reduce_window_p$lowering <- function(ctx, ins, ai, ao, params) {
  n <- length(ins) %/% 2L
  idt <- vapply(ai[n + seq_len(n)], `[[`, "", "dtype")
  hlo_reduce_window(ctx, ins[seq_len(n)], ins[n + seq_len(n)], ai[seq_len(n)], idt, ao, params,
                    function(sub, args) lower_reduction_jaxpr(sub, params$jaxpr, args))
}

# Port of windowed_reductions.reduce_window_jvp: the reduction is replaced by its JVP
# (ad.jvp_jaxpr) and evaluated on the operands paired with their tangents.
reduce_window_p$jvp <- function(primals, tangents, params) {
  n <- length(primals) %/% 2L
  operand <- primals[seq_len(n)]
  init_value <- primals[n + seq_len(n)]
  operand_tangent <- lapply(tangents[seq_len(n)], instantiate_zero)
  if (!all(vapply(tangents[n + seq_len(n)], is_zero, NA)))
    jax_error("reduce_window jvp does not support non-zero init_value_tangent.",
              class = "jax_type_error")
  init_value_tangent <- lapply(init_value, function(i) {
    a <- aval_of(i)
    lax_full(integer(0), 0, tangent_dtype(a$dtype))
  })
  cj <- params$jaxpr
  wrapper <- function(left, right) {
    pl <- left[seq_len(n)]
    tl <- left[n + seq_len(n)]
    pr <- right[seq_len(n)]
    tr <- right[n + seq_len(n)]
    r <- jvp(function(...) eval_closed(cj, list(...)), c(pl, pr), c(tl, tr))
    c(r[[1L]], r[[2L]])
  }
  out <- reduce_window_impl(c(operand, operand_tangent), c(init_value, init_value_tangent),
                            wrapper, params$window_dimensions, params$window_strides,
                            params$padding, params$base_dilation, params$window_dilation)
  list(out[seq_len(n)], out[n + seq_len(n)])
}

# Port of windowed_reductions._generic_reduce_window_batch_rule.
reduce_window_p$batch <- function(vals, dims, params, axis) {
  n <- length(vals) %/% 2L
  if (any(!is.na(dims[n + seq_len(n)])))
    jax_error("reduce_window batching is not implemented for initial values")
  ops <- Map(function(v, d) bdim_at_front(v, d, axis$size), vals[seq_len(n)], dims[seq_len(n)])
  np <- params
  np$window_dimensions <- c(1L, params$window_dimensions)
  np$window_strides <- c(1L, params$window_strides)
  np$padding <- c(list(c(0L, 0L)), params$padding)
  np$base_dilation <- c(1L, params$base_dilation)
  np$window_dilation <- c(1L, params$window_dilation)
  list(bind(reduce_window_p, c(ops, vals[n + seq_len(n)]), np), rep(0L, n))
}

# Port of windowed_reductions._get_monoid_window_reducer.
monoid_window_reducer <- function(monoid_op, init) {
  if (is_tracer(init)) return(NULL)
  x <- as_jax(init)
  a <- aval_of(x)
  if (length(a$shape) != 0L) return(NULL)
  val <- as.vector(array_host(x))
  if (identical(monoid_op, lax_add)) {
    if (isTRUE(val == 0)) return(reduce_window_sum)
  } else if (identical(monoid_op, lax_max)) {
    if (isTRUE(val == identity_max(a$dtype))) return(reduce_window_max)
  } else if (identical(monoid_op, lax_min)) {
    if (isTRUE(val == identity_min(a$dtype))) return(reduce_window_min)
  }
  NULL
}

# Port of windowed_reductions._reduce_window. `padding` is already resolved when it is
# not a string.
reduce_window_impl <- function(operand, init_value, computation, window_dimensions,
                               window_strides, padding, base_dilation = NULL,
                               window_dilation = NULL) {
  ofl <- tree_flatten(operand)
  ifl <- tree_flatten(init_value)
  if (!treedef_equal(ofl$treedef, ifl$treedef))
    jax_error("Operands must have the same tree structure as init_values: ",
              format(ofl$treedef), " vs. ", format(ifl$treedef), class = "jax_value_error")
  if (length(ofl$leaves) != length(ifl$leaves))
    jax_error("Must have same total number of operands as init_values: ",
              length(ofl$leaves), " vs. ", length(ifl$leaves), class = "jax_value_error")
  if (!length(ofl$leaves))
    jax_error("reduce_window must have at least one operand.", class = "jax_value_error")
  flat_operands <- lapply(ofl$leaves, as_jax)
  window_dimensions <- as.integer(window_dimensions)
  if (is.character(padding)) {
    dilated <- if (is.null(window_dilation)) window_dimensions
               else dilate_shape(window_dimensions, window_dilation)
    padding <- lax_padtype_to_pads(aval_of(flat_operands[[1L]])$shape, dilated,
                                   window_strides %||% integer(0), padding)
  } else {
    padding <- normalize_pairs(padding)
  }
  n <- length(window_dimensions)
  window_strides <- window_strides %||% rep(1L, n)
  base_dilation <- base_dilation %||% rep(1L, n)
  window_dilation <- window_dilation %||% rep(1L, n)
  if (length(flat_operands) == 1L) {
    mr <- monoid_window_reducer(computation, ifl$leaves[[1L]])
    if (!is.null(mr))
      return(mr(flat_operands[[1L]], window_dimensions, window_strides, padding,
                base_dilation, window_dilation))
  }
  # R scalars are weakly typed: they take the dtype of their operand (as Python
  # scalars do in JAX).
  flat_inits <- Map(function(i, o) {
    i <- as_jax(i)
    ai <- aval_of(i)
    od <- aval_of(o)$dtype
    if (ai$weak && ai$dtype != od) lax_convert_element_type(i, od) else i
  }, ifl$leaves, flat_operands)
  rj <- reduction_jaxpr(computation, vapply(flat_inits, function(i) aval_of(i)$dtype, ""),
                        ifl$treedef)
  if (!treedef_equal(ofl$treedef, rj$out_tree))
    jax_error("reduce_window output must have the same tree structure as the operands ",
              format(ofl$treedef), " vs. ", format(rj$out_tree), class = "jax_value_error")
  out <- bind(reduce_window_p, c(flat_operands, flat_inits),
              c(list(jaxpr = rj$jaxpr),
                window_params(window_dimensions, window_strides, padding, base_dilation,
                              window_dilation)))
  tree_unflatten(ofl$treedef, out)
}

#' Reduction over padded windows
#'
#' Port of `jax.lax.reduce_window` (XLA ReduceWindowWithGeneralPadding). With `lax_add`,
#' `lax_max` or `lax_min` and their identity as initial value the specialized
#' reductions `reduce_window_sum`, `reduce_window_max` and `reduce_window_min` are used.
#' @param operand Array or pytree (list) of arrays.
#' @param init_value Scalar or pytree of scalars with the structure of `operand`.
#' @param computation Function of two arguments with the structure of `operand`.
#' @param window_dimensions Integer vector of window sizes.
#' @param window_strides Optional integer vector of strides (default 1).
#' @param padding `"VALID"` (default), `"SAME"`, `"SAME_LOWER"` or a list of
#'   `c(low, high)` pairs.
#' @param base_dilation,window_dilation Optional integer vectors of dilations.
#' @return An array or a pytree of arrays with the structure of `operand`.
#' @export
lax_reduce_window <- function(operand, init_value, computation, window_dimensions,
                              window_strides = NULL, padding = "VALID", base_dilation = NULL,
                              window_dilation = NULL) {
  reduce_window_impl(operand, init_value, computation, window_dimensions, window_strides,
                     padding, base_dilation, window_dilation)
}

# ---- select_and_scatter -------------------------------------------------------------

select_and_scatter_p <- new_primitive("select_and_scatter")

# Port of windowed_reductions._select_and_scatter_shape_rule.
select_and_scatter_p$abstract_eval <- function(avals, params) {
  check_shapelike("select_and_scatter", "window_dimensions", params$window_dimensions)
  check_shapelike("select_and_scatter", "window_strides", params$window_strides)
  if (length(params$window_dimensions) != length(params$window_strides))
    jax_error("select_and_scatter got inconsistent window_strides and window_dimensions: got ",
              "window_strides ", fmt_tuple(params$window_strides), " and window_dimensions ",
              fmt_tuple(params$window_dimensions), ".", class = "jax_type_error")
  a <- avals[[1L]]
  new_aval(a$shape, a$dtype, a$weak)
}

# Port of windowed_reductions._select_and_scatter_lower.
select_and_scatter_p$lowering <- function(ctx, ins, ai, ao, params) {
  st <- scalar_type(ai[[1L]]$dtype)
  sel <- build_region(ctx, c(st, st), function(sub, args) {
    list(values = lower_reduction_jaxpr(sub, params$select_jaxpr, args),
         types = scalar_type("bool"))
  })
  sca <- build_region(ctx, c(st, st), function(sub, args) {
    list(values = lower_reduction_jaxpr(sub, params$scatter_jaxpr, args), types = st)
  })
  emit_op(ctx, "stablehlo.select_and_scatter", ins, types_of(ai), tensor_type(ao[[1L]]),
          attrs = list(padding = attr_i64_pairs(params$padding),
                       window_dimensions = attr_i64_array(params$window_dimensions),
                       window_strides = attr_i64_array(params$window_strides)),
          regions = c(sel, sca))
}

# Port of windowed_reductions._select_and_scatter.
select_and_scatter <- function(operand, select, window_dimensions, window_strides, padding,
                               source, init_value, scatter) {
  dt <- aval_of(init_value)$dtype
  sj <- reduction_jaxpr(select, dt, what = "select_and_scatter select")
  cj <- reduction_jaxpr(scatter, dt, what = "select_and_scatter scatter")
  bind(select_and_scatter_p, list(operand, source, init_value),
       list(select_jaxpr = sj$jaxpr, scatter_jaxpr = cj$jaxpr,
            window_dimensions = as.integer(window_dimensions),
            window_strides = as.integer(window_strides), padding = normalize_pairs(padding)))
}

select_prim_of <- function(name) switch(name, ge = ge_p, le = le_p)

# ---- select_and_scatter_add ---------------------------------------------------------

select_and_scatter_add_p <- new_primitive("select_and_scatter_add")
select_and_scatter_add_p$abstract_eval <- function(avals, params) {
  s <- avals[[1L]]
  o <- avals[[2L]]
  new_aval(o$shape, s$dtype, s$weak)
}

# Port of windowed_reductions._select_and_scatter_add (the wrapper).
select_and_scatter_add <- function(source, operand, select_prim, window_dimensions,
                                   window_strides, padding) {
  bind(select_and_scatter_add_p, list(source, operand),
       list(select_prim = select_prim, window_dimensions = as.integer(window_dimensions),
            window_strides = as.integer(window_strides), padding = normalize_pairs(padding)))
}

# Port of windowed_reductions._select_and_scatter_add_impl with expand_padding = True
# (the CPU registration).
select_and_scatter_add_p$lowering <- lower_fun(function(args, params) {
  source <- args[[1L]]
  operand <- args[[2L]]
  dt <- aval_of(source)$dtype
  sp <- select_prim_of(params$select_prim)
  select <- function(x, y) bind(sp, list(x, y))
  scatter <- if (dt == "bool") lax_bitwise_or else lax_add
  padding <- params$padding
  operand_shape <- aval_of(operand)$shape
  ident <- identity_value(if (params$select_prim == "ge") "max" else "min", dt)
  operand <- lax_pad(operand, ident, lapply(padding, function(p) c(p[1L], p[2L], 0L)))
  out <- select_and_scatter(operand, select, params$window_dimensions, params$window_strides,
                            lapply(padding, function(p) c(0L, 0L)), source,
                            lax_full(integer(0), 0, aval_of(operand)$dtype), scatter)
  start <- vapply(padding, `[`, 1L, 1L)
  lax_slice(out, start, start + operand_shape)
})

# Port of windowed_reductions._select_and_scatter_add_jvp.
select_and_scatter_add_p$jvp <- function(primals, tangents, params) {
  source <- primals[[1L]]
  operand <- primals[[2L]]
  val_out <- bind(select_and_scatter_add_p, primals, params)
  g <- tangents[[1L]]
  t_out <- if (is_zero(g)) zero_of(tangent_aval(aval_of(val_out)))
           else bind(select_and_scatter_add_p, list(g, operand), params)
  list(val_out, t_out)
}

# Port of windowed_reductions._select_and_scatter_add_transpose.
select_and_scatter_add_p$transpose <- function(ct, args, params) {
  source <- args[[1L]]
  operand <- args[[2L]]
  if (!is_undefined(source) || is_undefined(operand))
    jax_error("internal: select_and_scatter_add transpose expects a linear source")
  if (is_zero(ct)) return(list(zero_of(source$aval), NULL))
  ones <- rep(1L, length(params$window_dimensions))
  list(select_and_gather_add(ct, operand, params$select_prim, params$window_dimensions,
                             params$window_strides, params$padding, ones, ones), NULL)
}

# Port of windowed_reductions._select_and_scatter_add_batch_rule.
select_and_scatter_add_p$batch <- function(vals, dims, params, axis) {
  s <- bdim_at_front(vals[[1L]], dims[1L], axis$size)
  o <- bdim_at_front(vals[[2L]], dims[2L], axis$size)
  np <- params
  np$window_dimensions <- c(1L, params$window_dimensions)
  np$window_strides <- c(1L, params$window_strides)
  np$padding <- c(list(c(0L, 0L)), params$padding)
  list(bind(select_and_scatter_add_p, list(s, o), np), 0L)
}

# ---- select_and_gather_add ----------------------------------------------------------

select_and_gather_add_p <- new_primitive("select_and_gather_add")

# Port of windowed_reductions._select_and_gather_add_shape_rule.
select_and_gather_add_p$abstract_eval <- function(avals, params) {
  t <- avals[[1L]]
  o <- avals[[2L]]
  if (!identical(t$shape, o$shape))
    jax_error("select_and_gather_add tangents and operand shapes must match, got ",
              fmt_tuple(t$shape), " and ", fmt_tuple(o$shape), ".", class = "jax_type_error")
  new_aval(common_reduce_window_shape_rule(o, params), t$dtype, t$weak)
}

# Port of windowed_reductions._select_and_gather_add (the wrapper).
select_and_gather_add <- function(tangents, operand, select_prim, window_dimensions,
                                  window_strides, padding, base_dilation, window_dilation) {
  bind(select_and_gather_add_p, list(tangents, operand),
       list(select_prim = select_prim, window_dimensions = as.integer(window_dimensions),
            window_strides = as.integer(window_strides), padding = normalize_pairs(padding),
            base_dilation = as.integer(base_dilation),
            window_dilation = as.integer(window_dilation)))
}

# Port of windowed_reductions._select_and_gather_add_lowering.
select_and_gather_add_p$lowering <- lower_fun(function(args, params) {
  tangents <- args[[1L]]
  operand <- args[[2L]]
  sp <- select_prim_of(params$select_prim)
  reducer <- function(x, y) {
    kx <- x[[1L]]; vx <- x[[2L]]
    ky <- y[[1L]]; vy <- y[[2L]]
    which <- bind(sp, list(kx, ky))
    list(lax_select(which, kx, ky), lax_select(which, vx, vy))
  }
  dt <- aval_of(operand)$dtype
  init <- if (params$select_prim == "ge") -Inf else Inf
  out <- reduce_window_impl(list(operand, tangents),
                            list(lax_full(integer(0), init, dt), lax_full(integer(0), 0, dt)),
                            reducer, params$window_dimensions, params$window_strides,
                            params$padding, params$base_dilation, params$window_dilation)
  out[[2L]]
})

# Port of windowed_reductions._select_and_gather_add_jvp.
select_and_gather_add_p$jvp <- function(primals, tangents, params) {
  operand <- primals[[2L]]
  val_out <- bind(select_and_gather_add_p, primals, params)
  g <- tangents[[1L]]
  t_out <- if (is_zero(g)) zero_of(tangent_aval(aval_of(val_out)))
           else bind(select_and_gather_add_p, list(g, operand), params)
  list(val_out, t_out)
}

# Port of windowed_reductions._select_and_gather_add_transpose.
select_and_gather_add_p$transpose <- function(ct, args, params) {
  tangents <- args[[1L]]
  operand <- args[[2L]]
  if (!is_undefined(tangents) || is_undefined(operand))
    jax_error("internal: select_and_gather_add transpose expects linear tangents")
  if (any(params$window_dilation != 1L))
    jax_error("VJP not implemented for select_and_gather (MaxPool) with window dilation, got ",
              "window_dilation=", fmt_tuple(params$window_dilation), ".",
              class = "jax_not_implemented_error")
  if (is_zero(ct)) return(list(zero_of(tangents$aval), NULL))
  bd <- params$base_dilation
  has_base_dilation <- any(bd != 1L)
  if (has_base_dilation) {
    ident <- identity_value(if (params$select_prim == "ge") "max" else "min",
                            aval_of(operand)$dtype)
    operand <- lax_pad(operand, ident, lapply(bd, function(d) c(0L, 0L, d - 1L)))
  }
  result <- select_and_scatter_add(ct, operand, params$select_prim, params$window_dimensions,
                                   params$window_strides, params$padding)
  if (has_base_dilation) {
    rs <- aval_of(result)$shape
    result <- lax_slice(result, rep(0L, length(rs)), rs, bd)
  }
  list(result, NULL)
}

# Port of windowed_reductions._select_and_gather_add_batching_rule.
select_and_gather_add_p$batch <- function(vals, dims, params, axis) {
  t <- bdim_at_front(vals[[1L]], dims[1L], axis$size)
  x <- bdim_at_front(vals[[2L]], dims[2L], axis$size)
  np <- params
  np$window_dimensions <- c(1L, params$window_dimensions)
  np$window_strides <- c(1L, params$window_strides)
  np$padding <- c(list(c(0L, 0L)), params$padding)
  np$base_dilation <- c(1L, params$base_dilation)
  np$window_dilation <- c(1L, params$window_dilation)
  list(bind(select_and_gather_add_p, list(t, x), np), 0L)
}
