# Lowering of jaxprs to StableHLO in MLIR generic op form (port of the parts of
# jax/_src/interpreters/mlir.py that build modules, constants and nested regions).

tensor_type <- function(aval) {
  dims <- if (length(aval$shape)) paste0(paste(aval$shape, collapse = "x"), "x") else ""
  paste0("tensor<", dims, mlir_elem_type(aval$dtype), ">")
}

new_lowering_ctx <- function(parent = NULL, indent = "    ") {
  ctx <- new.env(parent = emptyenv())
  ctx$lines <- vector("list", 64L)
  ctx$n <- 0L
  ctx$indent <- indent
  ctx$counter <- if (is.null(parent)) new.env(parent = emptyenv()) else parent$counter
  if (is.null(parent)) ctx$counter$i <- 0L
  ctx
}

ssa_name <- function(ctx) {
  i <- ctx$counter$i
  ctx$counter$i <- i + 1L
  paste0("%", i)
}

emit_line <- function(ctx, line) {
  line <- paste0(ctx$indent, line)  # force first: building the line may emit operands
  ctx$n <- ctx$n + 1L
  if (ctx$n > length(ctx$lines)) length(ctx$lines) <- 2L * length(ctx$lines)
  ctx$lines[[ctx$n]] <- line
}

ctx_text <- function(ctx) paste(unlist(ctx$lines[seq_len(ctx$n)]), collapse = "\n")

fmt_types <- function(types) {
  if (length(types) == 1L) types else paste0("(", paste(types, collapse = ", "), ")")
}

# Emit one operation in generic form. Returns the SSA names of its results.
#   attrs: named character vector/list of attribute values (already MLIR syntax).
#   regions: character vector of region texts.
emit_op <- function(ctx, name, operands, operand_types, result_types,
                    attrs = NULL, regions = NULL) {
  # Operand expressions may emit ops themselves: evaluate them before naming results.
  force(operands); force(operand_types); force(result_types); force(attrs); force(regions)
  nres <- length(result_types)
  lhs <- ""
  outs <- character(0)
  if (nres == 1L) {
    v <- ssa_name(ctx)
    lhs <- paste0(v, " = ")
    outs <- v
  } else if (nres > 1L) {
    v <- ssa_name(ctx)
    lhs <- paste0(v, ":", nres, " = ")
    outs <- paste0(v, "#", seq_len(nres) - 1L)
  }
  props <- if (length(attrs)) {
    paste0(" <{", paste(names(attrs), unlist(attrs), sep = " = ", collapse = ", "), "}>")
  } else ""
  regs <- if (length(regions)) paste0(" (", paste(regions, collapse = ", "), ")") else ""
  rt <- if (nres == 1L) result_types else paste0("(", paste(result_types, collapse = ", "), ")")
  emit_line(ctx, paste0(lhs, "\"", name, "\"(", paste(operands, collapse = ", "), ")", props,
                        regs, " : (", paste(operand_types, collapse = ", "), ") -> ", rt))
  outs
}

# Build a region with one block. `body(sub_ctx, args)` returns list(values, types)
# that are passed to the terminator `term` (default stablehlo.return).
build_region <- function(ctx, arg_types, body, term = "stablehlo.return") {
  sub <- new_lowering_ctx(ctx, paste0(ctx$indent, "  "))
  args <- vapply(seq_along(arg_types), function(i) ssa_name(ctx), "")
  res <- body(sub, args)
  emit_line(sub, paste0("\"", term, "\"(", paste(res$values, collapse = ", "), ") : (",
                        paste(res$types, collapse = ", "), ") -> ()"))
  head <- if (length(args)) paste0("{\n", ctx$indent, "^bb0(",
                                   paste(paste0(args, ": ", arg_types), collapse = ", "), "):\n")
          else "{\n"
  paste0(head, ctx_text(sub), "\n", ctx$indent, "}")
}

# ---- attributes ---------------------------------------------------------------------

attr_i64_array <- function(x) {
  if (!length(x)) return("array<i64>")
  paste0("array<i64: ", paste(format(as.numeric(x), scientific = FALSE, trim = TRUE),
                              collapse = ", "), ">")
}
attr_i64_list <- function(x) paste0("[", paste(as.integer(x), collapse = ", "), "]")
attr_i64 <- function(x) paste0(format(as.numeric(x), scientific = FALSE), " : i64")
attr_i32 <- function(x) paste0(as.integer(x), " : i32")
attr_bool <- function(x) if (isTRUE(x)) "true" else "false"
attr_str <- function(x) paste0("\"", x, "\"")
attr_bool_array <- function(x) {
  if (!length(x)) return("array<i1>")
  paste0("array<i1: ", paste(ifelse(x, "true", "false"), collapse = ", "), ">")
}

# ---- constants ----------------------------------------------------------------------

hex_bytes <- function(r) paste0(sprintf("%02X", as.integer(r)), collapse = "")

float_hex <- function(v, dt) {
  size <- if (dt == "float64") 8L else 4L
  paste0("0x", hex_bytes(writeBin(as.double(v), raw(), size = size, endian = "big")))
}

int_literal <- function(v) format(as.numeric(v), scientific = FALSE, trim = TRUE)

# Non-finite components have no decimal literal: they are written as bit patterns.
complex_literal <- function(v, dt = "complex128") {
  comp <- if (dt == "complex64") "float32" else "float64"
  part <- function(x) if (is.finite(x)) sprintf("%.17e", x) else float_hex(x, comp)
  sprintf("(%s, %s)", part(Re(v)), part(Im(v)))
}

scalar_literal <- function(v, dt) {
  k <- dtype_kind(dt)
  if (k == "b" || k == "V") return(if (isTRUE(as.logical(v))) "true" else "false")
  if (k == "f") return(float_hex(v, dt))
  if (k == "c") return(complex_literal(v, dt))
  int_literal(v)
}

le_bytes_int <- function(v, bits) {
  v <- as.numeric(v)
  if (bits <= 32L) {
    m <- 2^bits
    u <- v %% m
    s <- ifelse(u >= 2^31, u - 2^32, u)
    if (bits == 32L) return(writeBin(as.integer(ifelse(s == -2^31, NA, s)), raw(), size = 4L, endian = "little"))
    return(writeBin(as.integer(u), raw(), size = bits %/% 8L, endian = "little"))
  }
  u <- v %% 2^64
  lo <- u %% 2^32
  hi <- (u - lo) / 2^32
  enc <- function(z) { s <- ifelse(z >= 2^31, z - 2^32, z); as.integer(ifelse(s == -2^31, NA, s)) }
  r <- rbind(matrix(writeBin(enc(lo), raw(), size = 4L, endian = "little"), 4),
             matrix(writeBin(enc(hi), raw(), size = 4L, endian = "little"), 4))
  as.vector(r)
}

# Dense elements literal for row-major data `v` of dtype `dt` (not bool/f16/bf16).
dense_blob <- function(v, dt) {
  k <- dtype_kind(dt)
  bytes <- if (k == "f") {
    writeBin(as.double(v), raw(), size = if (dt == "float64") 8L else 4L, endian = "little")
  } else if (k == "c") {
    size <- if (dt == "complex128") 8L else 4L
    writeBin(as.vector(rbind(Re(v), Im(v))), raw(), size = size, endian = "little")
  } else {
    le_bytes_int(v, dtype_bits(dt))
  }
  paste0("\"0x", hex_bytes(bytes), "\"")
}

# Emit a constant of the given aval from R data (any layout with matching shape).
ir_constant <- function(ctx, data, aval) {
  dt <- aval$dtype
  ty <- tensor_type(aval)
  flat <- c_ravel(data)
  if (dt %in% c("float16", "bfloat16", ext_dtypes)) {
    a32 <- new_aval(aval$shape, if (dt %in% ext_int_dtypes) "int32" else "float32")
    v <- ir_constant(ctx, data, a32)
    return(emit_op(ctx, "stablehlo.convert", v, tensor_type(a32), ty))
  }
  if (dt %in% c("bool", "float0") && length(flat) > 1L && length(unique(flat)) > 1L) {
    a8 <- new_aval(aval$shape, "uint8")
    v <- ir_constant(ctx, with_shape(as.numeric(as.logical(data)), aval$shape), a8)
    z <- ir_constant(ctx, 0, new_aval(integer(0), "uint8"))
    zb <- emit_op(ctx, "stablehlo.broadcast_in_dim", z, "tensor<ui8>", tensor_type(a8),
                  attrs = list(broadcast_dimensions = "array<i64>"))
    return(emit_op(ctx, "stablehlo.compare", c(v, zb), c(tensor_type(a8), tensor_type(a8)), ty,
                   attrs = list(comparison_direction = "#stablehlo<comparison_direction NE>",
                                compare_type = "#stablehlo<comparison_type UNSIGNED>")))
  }
  if (dt == "float0") flat <- rep(FALSE, length(flat))
  val <- if (length(flat) == 0L) {
    "dense<>"
  } else if (length(unique(flat)) == 1L && !(is.double(flat) && anyNA(flat) && !all(is.nan(flat)))) {
    paste0("dense<", scalar_literal(flat[1L], dt), ">")
  } else {
    paste0("dense<", dense_blob(flat, dt), ">")
  }
  emit_op(ctx, "stablehlo.constant", character(0), character(0), ty,
          attrs = list(value = paste0(val, " : ", ty)))
}

# Constant scalar broadcast to the shape of `aval` (dtype of aval).
full_constant <- function(ctx, value, aval) {
  s <- ir_constant(ctx, value, new_aval(integer(0), aval$dtype))
  if (!length(aval$shape)) return(s)
  hlo_broadcast(ctx, s, new_aval(integer(0), aval$dtype), aval, integer(0))
}

hlo_broadcast <- function(ctx, x, aval_in, aval_out, dims) {
  emit_op(ctx, "stablehlo.broadcast_in_dim", x, tensor_type(aval_in), tensor_type(aval_out),
          attrs = list(broadcast_dimensions = attr_i64_array(dims)))
}

# ---- lowering jaxprs ----------------------------------------------------------------

# Lower the equations of a jaxpr. consts/args are SSA names; returns output SSA names.
jaxpr_subcomp <- function(ctx, jaxpr, consts, args) {
  env <- character(jaxpr$nvars)
  for (i in seq_along(jaxpr$constvars)) env[jaxpr$constvars[[i]]$id] <- consts[[i]]
  for (i in seq_along(jaxpr$invars)) env[jaxpr$invars[[i]]$id] <- args[[i]]
  rd <- function(a) {
    if (is_lit(a)) return(ir_constant(ctx, array_host(a$val), a$aval))
    env[a$id]
  }
  for (e in jaxpr$eqns) {
    rule <- e$prim$lowering
    if (is.null(rule))
      jax_error("MLIR translation rule for primitive '", e$prim$name, "' not found")
    ins <- vapply(e$invars, rd, "")
    avals_in <- lapply(e$invars, `[[`, "aval")
    avals_out <- lapply(e$outvars, `[[`, "aval")
    n0 <- ctx$n
    outs <- rule(ctx, ins, avals_in, avals_out, e$params)
    # Port of wrap_xla_metadata_in_place: the operations producing the outputs of an
    # equation staged under set_xla_metadata carry its metadata.
    if (length(e$xla_metadata))
      for (o in setdiff(unique(outs), ins)) xm_tag_value(ctx, o, e$xla_metadata, n0 + 1L, TRUE)
    for (j in seq_along(e$outvars)) env[e$outvars[[j]]$id] <- outs[[j]]
  }
  vapply(jaxpr$outvars, rd, "")
}

# Lower a jaxpr (constvars become leading arguments) to a module text.
lower_jaxpr_module <- function(jaxpr, name = "jit_fun") {
  ctx <- new_lowering_ctx(indent = "    ")
  in_avals <- c(lapply(jaxpr$constvars, `[[`, "aval"), lapply(jaxpr$invars, `[[`, "aval"))
  arg_names <- sprintf("%%arg%d", seq_along(in_avals) - 1L)
  nc <- length(jaxpr$constvars)
  outs <- jaxpr_subcomp(ctx, jaxpr, as.list(arg_names[seq_len(nc)]),
                        as.list(arg_names[nc + seq_along(jaxpr$invars)]))
  out_types <- vapply(jaxpr$outvars, function(v) tensor_type(v$aval), "")
  in_types <- vapply(in_avals, tensor_type, "")
  emit_line(ctx, paste0("\"func.return\"(", paste(outs, collapse = ", "), ") : (",
                        paste(out_types, collapse = ", "), ") -> ()"))
  ftype <- paste0("(", paste(in_types, collapse = ", "), ") -> ",
                  if (length(out_types) == 1L) out_types else
                    paste0("(", paste(out_types, collapse = ", "), ")"))
  paste0(
    "\"builtin.module\"() <{sym_name = \"", gsub("[^A-Za-z0-9_]", "_", name), "\"}> ({\n",
    "  \"func.func\"() <{function_type = ", ftype,
    ", sym_name = \"main\", sym_visibility = \"public\"}> ({\n",
    if (length(arg_names)) paste0("  ^bb0(", paste(paste0(arg_names, ": ", in_types), collapse = ", "), "):\n"),
    ctx_text(ctx), "\n",
    "  }) : () -> ()\n",
    "}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()\n")
}

# Lowering rule from a function of jax values (port of mlir.lower_fun): the function
# is traced to a jaxpr which is lowered inline.
lower_fun <- function(fun, multiple_results = FALSE) {
  function(ctx, ins, avals_in, avals_out, params) {
    cj <- with_trace(eval_trace(), trace_to_jaxpr(function(args) {
      out <- fun(args, params)
      if (multiple_results) out else list(out)
    }, avals_in, parent = eval_trace()))
    consts <- lapply(seq_along(cj$consts), function(i) {
      ir_constant(ctx, array_host(cj$consts[[i]]), cj$jaxpr$constvars[[i]]$aval)
    })
    jaxpr_subcomp(ctx, cj$jaxpr, consts, as.list(ins))
  }
}

#' Lower a function to StableHLO text
#'
#' Analogue of `jax.jit(f).lower(*args).as_text()` (generic op form).
#' @param fun Function.
#' @param ... Example arguments (arrays or [ShapeDtypeStruct()]).
#' @param static_argnums 0-based static argument positions.
#' @return A character scalar with the module text.
#' @export
lower_text <- function(fun, ..., static_argnums = integer(0)) {
  cj <- make_jaxpr(fun, static_argnums = static_argnums)(...)
  lower_jaxpr_module(cj$jaxpr, "jit_fun")
}
