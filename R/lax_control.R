# Structured control flow: cond/switch, while_loop, fori_loop, scan, map
# (port of jax/_src/lax/control_flow/conditionals.py and loops.py).

# ---- closed jaxpr transformations ---------------------------------------------------------

lower_closed_consts <- function(ctx, cj) {
  lapply(seq_along(cj$consts), function(i) {
    c <- cj$consts[[i]]
    if (is_tracer(c)) jax_error("internal: control flow jaxpr closes over a tracer")
    ir_constant(ctx, array_host(c), cj$jaxpr$constvars[[i]]$aval)
  })
}

# Port of ad.jvp_jaxpr: inputs (primals, nonzero tangents) -> (primal outs, tangent outs
# where nonzero or instantiated).
jvp_jaxpr <- function(cj, nonzeros, instantiate) {
  in_avals <- jaxpr_in_avals(cj$jaxpr)
  n <- length(in_avals)
  nout <- length(cj$jaxpr$outvars)
  instantiate <- rep_len(as.logical(instantiate), nout)
  t_avals <- lapply(in_avals[nonzeros], tangent_aval)
  store <- new.env(parent = emptyenv())
  f <- function(args) {
    primals <- args[seq_len(n)]
    tans <- args[n + seq_along(t_avals)]
    trace <- new_jvp_trace(current_trace())
    on.exit(trace$alive <- FALSE, add = TRUE)
    ins <- primals
    k <- 0L
    for (i in which(nonzeros)) {
      k <- k + 1L
      ins[[i]] <- new_tracer(trace, aval_of(primals[[i]]), primal = primals[[i]], tangent = tans[[k]])
    }
    outs <- with_trace(trace, eval_jaxpr(cj$jaxpr, cj$consts, ins))
    pts <- lapply(outs, primal_tangent, trace = trace)
    po <- lapply(pts, `[[`, 1L)
    to <- lapply(pts, `[[`, 2L)
    nz <- (!vapply(to, is_zero, NA) | instantiate) &
      vapply(po, function(p) is_inexact_dtype(aval_of(p)$dtype), NA)
    store$out_nz <- nz
    c(po, lapply(to[nz], instantiate_zero))
  }
  list(jaxpr = trace_to_jaxpr(f, c(in_avals, t_avals)), out_nz = store$out_nz)
}

# Port of batching.batch_jaxpr with batch dims at 0.
batch_jaxpr <- function(cj, size, name, batched, instantiate) {
  in_avals <- jaxpr_in_avals(cj$jaxpr)
  nout <- length(cj$jaxpr$outvars)
  instantiate <- rep_len(as.logical(instantiate), nout)
  b_avals <- Map(function(a, b) { if (b) a$shape <- c(size, a$shape); a }, in_avals, batched)
  store <- new.env(parent = emptyenv())
  f <- function(args) {
    trace <- new_batch_trace(current_trace(), size, name)
    on.exit(trace$alive <- FALSE, add = TRUE)
    ins <- Map(function(v, b) if (b) batch_tracer(trace, v, 0L) else v, args, batched)
    outs <- with_trace(trace, eval_jaxpr(cj$jaxpr, cj$consts, ins))
    ob <- vapply(outs, is_own, NA, trace = trace) | instantiate
    store$out_batched <- ob
    Map(function(o, b) {
      if (is_own(o, trace)) moveaxis(o$val, o$bdim, 0L)
      else if (b) lax_broadcast(o, size) else o
    }, outs, ob)
  }
  list(jaxpr = trace_to_jaxpr(f, b_avals), out_batched = store$out_batched)
}

# Transpose a closed jaxpr: inputs (non-linear args in order, one cotangent per output)
# -> cotangents of the linear inputs. Port of ad_checkpoint._transpose_jaxpr: the
# non-linear part is first split off by partial evaluation (so equations mixing primal
# and tangent values, such as a JVP scan, are separated) and the linear rest transposed.
transpose_jaxpr <- function(cj, linear) {
  in_avals <- jaxpr_in_avals(cj$jaxpr)
  out_avals <- jaxpr_out_avals(cj$jaxpr)
  nl <- in_avals[!linear]
  f <- function(args) {
    res_in <- args[seq_along(nl)]
    cts <- args[length(nl) + seq_along(out_avals)]
    pe <- partial_eval_jaxpr(cj, linear, TRUE)
    known <- eval_closed(pe$known, res_in)
    res <- drop_first(known, length(known) - pe$num_res)
    lin_avals <- in_avals[linear]
    a <- c(res, lapply(lin_avals, undefined_primal))
    out <- backward_pass(pe$unknown$jaxpr, pe$unknown$consts, a, cts)
    out <- drop_first(out, pe$num_res)
    Map(function(o, av) if (is_zero(o)) zeros_aval(av) else o, out, lin_avals)
  }
  trace_to_jaxpr(f, c(nl, out_avals))
}

# New jaxpr whose input k is old input in_order[k] and output k is old output out_order[k].
rearrange_jaxpr <- function(cj, in_order, out_order = NULL) {
  in_avals <- jaxpr_in_avals(cj$jaxpr)
  f <- function(args) {
    ins <- vector("list", length(in_avals))
    ins[in_order] <- args
    outs <- eval_jaxpr(cj$jaxpr, cj$consts, ins)
    if (is.null(out_order)) outs else outs[out_order]
  }
  trace_to_jaxpr(f, in_avals[in_order])
}

# Trace f (list -> list) and return a closed jaxpr plus the output tree.
trace_flat_fun <- function(fun, in_tree, in_avals) {
  store <- new.env(parent = emptyenv())
  cj <- trace_to_jaxpr(function(leaves) {
    out <- do.call(fun, tree_unflatten(in_tree, leaves))
    ofl <- tree_flatten(out)
    store$out_tree <- ofl$treedef
    lapply(ofl$leaves, as_jax)
  }, in_avals)
  list(cj = cj, out_tree = store$out_tree)
}

# Hoist consts of several closed jaxprs into common leading inputs.
join_consts <- function(cjs) {
  all_consts <- do.call(c, lapply(cjs, `[[`, "consts"))
  counts <- vapply(cjs, function(c) length(c$consts), 1L)
  offs <- cumsum(c(0L, counts))
  avals <- do.call(c, lapply(cjs, function(c) lapply(c$jaxpr$constvars, `[[`, "aval")))
  jaxprs <- lapply(seq_along(cjs), function(i) {
    cj <- cjs[[i]]
    f <- function(args) {
      mine <- args[offs[i] + seq_len(counts[i])]
      eval_jaxpr(cj$jaxpr, mine, drop_first(args, length(avals)))
    }
    trace_to_jaxpr(f, c(avals, jaxpr_in_avals(cj$jaxpr)))
  })
  list(jaxprs = jaxprs, consts = all_consts)
}

strip_weak <- function(a) { a$weak <- FALSE; a }

# Port of loops._promote_weak_typed_input. A weakly typed initial carry whose type
# differs from the carry the body returns (core.typematch ignores weak types) is
# converted to their common dtype, and the body is traced again.
weak_carry_mismatch <- function(in_avals, out_avals) {
  if (length(in_avals) != length(out_avals)) return(logical(length(in_avals)))
  vapply(seq_along(in_avals), function(i) in_avals[[i]]$weak && !aval_eq(in_avals[[i]], out_avals[[i]]), NA)
}
promote_weak_carry <- function(vals, out_avals) {
  Map(function(x, o) {
    a <- aval_of(x)
    lax_convert_element_type(x, promote_types(c(a$dtype, o$dtype), c(a$weak, o$weak))$dtype)
  }, vals, out_avals)
}

# ---- cond ------------------------------------------------------------------------------------

cond_p <- new_primitive("cond", multiple_results = TRUE)
cond_p$abstract_eval <- function(avals, params) {
  outs <- jaxpr_out_avals(params$branches[[1L]]$jaxpr)
  for (b in params$branches[-1L]) {
    oa <- jaxpr_out_avals(b$jaxpr)
    for (i in seq_along(outs)) outs[[i]]$weak <- outs[[i]]$weak && oa[[i]]$weak
  }
  outs
}
cond_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  idx <- ins[[1L]]
  ia <- avals_in[[1L]]
  if (ia$dtype != "int32")
    idx <- emit_op(ctx, "stablehlo.convert", idx, tensor_type(ia), "tensor<i32>")
  ops <- as.list(ins[-1L])
  out_types <- vapply(avals_out, tensor_type, "")
  regions <- vapply(params$branches, function(b) {
    build_region(ctx, character(0), function(sub, args) {
      outs <- jaxpr_subcomp(sub, b$jaxpr, lower_closed_consts(sub, b), ops)
      list(values = outs, types = out_types)
    })
  }, "")
  emit_op(ctx, "stablehlo.case", idx, "tensor<i32>", out_types, regions = regions)
}

# Port of conditionals._cond_jvp.
cond_p$jvp <- function(primals, tangents, params) {
  ops_nz <- !vapply(tangents[-1L], is_zero, NA)
  branches <- params$branches
  out_nz <- Reduce(`|`, lapply(branches, function(b) jvp_jaxpr(b, ops_nz, FALSE)$out_nz))
  bj <- lapply(branches, function(b) jvp_jaxpr(b, ops_nz, out_nz)$jaxpr)
  ops_dot <- lapply(tangents[-1L][ops_nz], instantiate_zero)
  out <- bind(cond_p, c(primals, ops_dot), list(branches = bj))
  n <- length(out_nz)
  po <- out[seq_len(n)]
  to <- drop_first(out, n)
  k <- 0L
  tans <- lapply(seq_len(n), function(i) {
    if (out_nz[i]) { k <<- k + 1L; to[[k]] } else zero_of(tangent_aval(aval_of(po[[i]])))
  })
  list(po, tans)
}

bcast_select <- function(pred, on_true, on_false) {
  s <- aval_of(on_true)$shape
  if (length(aval_of(pred)$shape) < length(s))
    pred <- lax_broadcast_in_dim(pred, s, seq_along(aval_of(pred)$shape) - 1L)
  lax_select(pred, on_true, on_false)
}

# Port of conditionals._cond_batching_rule.
cond_p$batch <- function(vals, dims, params, axis) {
  size <- axis$size
  branches <- params$branches
  index <- vals[[1L]]
  ops <- vals[-1L]
  op_dims <- dims[-1L]
  if (!is.na(dims[1L])) {
    index <- bdim_at_front(index, dims[1L], size)
    ops <- Map(function(x, d) bdim_at_front(x, d, size), ops, op_dims)
    nout <- length(branches[[1L]]$jaxpr$outvars)
    bb <- lapply(branches, function(b) batch_jaxpr(b, size, axis$name, rep(TRUE, length(ops)), TRUE)$jaxpr)
    branch_outs <- lapply(seq_along(bb), function(i) {
      pred <- lax_eq(index, lax_full_like(index, i - 1L))
      ops_ <- lapply(ops, function(x) bcast_select(pred, x, lax_stop_gradient(x)))
      eval_closed(bb[[i]], ops_)
    })
    out <- lapply(seq_len(nout), function(j) {
      cases <- lapply(branch_outs, `[[`, j)
      s <- aval_of(cases[[1L]])$shape
      w <- if (length(s) > 1L) lax_broadcast_in_dim(index, s, 0L) else index
      do.call(lax_select_n, c(list(w), cases))
    })
    return(list(out, rep(0L, nout)))
  }
  ops_bat <- !is.na(op_dims)
  ops <- Map(function(x, d) if (is.na(d)) x else moveaxis(x, d, 0L), ops, op_dims)
  out_bat <- Reduce(`|`, lapply(branches, function(b) batch_jaxpr(b, size, axis$name, ops_bat, FALSE)$out_batched))
  bb <- lapply(branches, function(b) batch_jaxpr(b, size, axis$name, ops_bat, out_bat)$jaxpr)
  out <- bind(cond_p, c(list(index), ops), list(branches = bb))
  list(out, ifelse(out_bat, 0L, NA_integer_))
}

# Port of conditionals._cond_partial_eval.
cond_p$override$pe <- function(trace, args, params) {
  uk <- vapply(args, is_own, NA, trace = trace)
  if (uk[1L]) return(stage_eqn(trace, cond_p, args, params))
  ops_uk <- uk[-1L]
  branches <- params$branches
  out_uks <- Reduce(`|`, lapply(branches, function(b) partial_eval_jaxpr(b, ops_uk, FALSE)$out_unknowns))
  pes <- lapply(branches, function(b) partial_eval_jaxpr(b, ops_uk, out_uks))
  nk <- sum(!out_uks)
  nres <- vapply(pes, `[[`, 1L, "num_res")
  offs <- cumsum(c(0L, nres))
  res_avals <- do.call(c, lapply(pes, function(p) {
    oa <- jaxpr_out_avals(p$known$jaxpr)
    oa[nk + seq_len(p$num_res)]
  }))
  known_in <- jaxpr_in_avals(branches[[1L]]$jaxpr)[!ops_uk]
  known_br <- lapply(seq_along(pes), function(i) {
    p <- pes[[i]]
    trace_to_jaxpr(function(a) {
      o <- eval_closed(p$known, a)
      res <- lapply(res_avals, zeros_aval)
      if (p$num_res) res[offs[i] + seq_len(p$num_res)] <- o[nk + seq_len(p$num_res)]
      c(o[seq_len(nk)], res)
    }, known_in)
  })
  uk_in <- jaxpr_in_avals(branches[[1L]]$jaxpr)[ops_uk]
  unknown_br <- lapply(seq_along(pes), function(i) {
    p <- pes[[i]]
    trace_to_jaxpr(function(a) {
      res <- a[seq_along(res_avals)]
      rest <- drop_first(a, length(res_avals))
      eval_closed(p$unknown, c(res[offs[i] + seq_len(p$num_res)], rest))
    }, c(res_avals, uk_in))
  })
  out <- with_trace(trace$parent, bind(cond_p, args[!uk], list(branches = known_br)))
  known_outs <- out[seq_len(nk)]
  res <- out[nk + seq_along(res_avals)]
  uk_outs <- stage_eqn(trace, cond_p, c(args[1L], res, args[-1L][ops_uk]),
                       list(branches = unknown_br))
  merged <- vector("list", length(out_uks))
  merged[!out_uks] <- known_outs
  merged[out_uks] <- uk_outs
  merged
}

# Port of conditionals._cond_transpose (generalized to any linear mask).
cond_p$transpose <- function(cts, args, params) {
  ops <- args[-1L]
  linear <- vapply(ops, is_undefined, NA)
  bt <- lapply(params$branches, transpose_jaxpr, linear = linear)
  out_avals <- jaxpr_out_avals(params$branches[[1L]]$jaxpr)
  cts <- Map(function(c, a) if (is_zero(c)) zeros_aval(strip_weak(a)) else c, cts, out_avals)
  out <- bind(cond_p, c(args[1L], ops[!linear], cts), list(branches = bt))
  res <- vector("list", length(args))
  res[c(FALSE, linear)] <- out
  res
}

prepare_branches <- function(branches, ops) {
  fl <- tree_flatten(ops)
  leaves <- lapply(fl$leaves, as_jax)
  avals <- lapply(leaves, aval_of)
  traced <- lapply(branches, function(b) {
    t <- trace_flat_fun(b, fl$treedef, avals)
    kr_revert(attr(t$cj, "kr"))
    t
  })
  t1 <- traced[[1L]]
  for (t in traced[-1L]) {
    if (!treedef_equal(t$out_tree, t1$out_tree))
      jax_error("branch outputs must have the same pytree structure, got ", format(t1$out_tree),
                " and ", format(t$out_tree), class = "jax_type_error")
    oa1 <- jaxpr_out_avals(t1$cj$jaxpr)
    oa <- jaxpr_out_avals(t$cj$jaxpr)
    for (i in seq_along(oa)) if (!aval_eq(oa[[i]], oa1[[i]]))
      jax_error("branches must have equal output types; got ", aval_str(oa1[[i]]), " and ",
                aval_str(oa[[i]]), class = "jax_type_error")
  }
  # Branches that disagree only in weak types are made strongly typed (as in JAX).
  weaks <- lapply(traced, function(t) vapply(jaxpr_out_avals(t$cj$jaxpr), `[[`, NA, "weak"))
  mixed <- Reduce(`|`, weaks) & !Reduce(`&`, weaks)
  if (any(mixed)) {
    traced <- lapply(branches, function(b) trace_flat_fun(function(...) {
      out <- b(...)
      ofl <- tree_flatten(out)
      ls <- Map(function(l, m) {
        l <- as_jax(l)
        if (m) lax_convert_element_type(l, aval_of(l)$dtype, weak_type = FALSE) else l
      }, ofl$leaves, mixed)
      tree_unflatten(ofl$treedef, ls)
    }, fl$treedef, avals))
    for (t in traced) kr_revert(attr(t$cj, "kr"))
  }
  jc <- join_consts(lapply(traced, `[[`, "cj"))
  list(jaxprs = jc$jaxprs, consts = jc$consts, leaves = leaves, out_tree = traced[[1L]]$out_tree,
       kr = kr_combine_branches(lapply(traced, function(t) attr(t$cj, "kr"))))
}

#' Conditional execution
#'
#' Port of `jax.lax.cond`: applies `true_fun(...)` or `false_fun(...)` to the operands
#' depending on the scalar `pred`. Both branches are traced; the predicate may be traced.
#' @param pred Scalar predicate.
#' @param true_fun,false_fun Functions of the operands with identical output types.
#' @param ... Operands (pytrees).
#' @export
lax_cond <- function(pred, true_fun, false_fun, ...) {
  p <- as_jax(pred)
  pa <- aval_of(p)
  if (length(pa$shape))
    jax_error("Pred must be a scalar, got shape (", paste(pa$shape, collapse = ", "), ")",
              class = "jax_type_error")
  if (!dtype_kind(pa$dtype) %in% c("b", "i", "u", "f"))
    jax_error("Pred type must be either boolean or number, got ", pa$dtype, ".",
              class = "jax_type_error")
  if (isTRUE(.jx$config$jax_disable_jit) && is_jax_array(p)) {
    return(if (isTRUE(as.logical(p))) true_fun(...) else false_fun(...))
  }
  idx <- if (pa$dtype == "bool") lax_convert_element_type(p, "int32")
         else lax_convert_element_type(lax_ne(p, lax_full_like(p, 0)), "int32")
  switch_impl(idx, list(false_fun, true_fun), list(...))
}

#' Select and apply one of several branches
#'
#' Port of `jax.lax.switch`; `index` is 0-based and clamped to the valid range.
#' @param index Scalar integer.
#' @param branches List of functions.
#' @param ... Operands.
#' @export
lax_switch <- function(index, branches, ...) {
  if (!length(branches)) jax_error("Empty branch sequence")
  idx <- as_jax(index)
  if (length(aval_of(idx)$shape)) jax_error("Branch index must be scalar")
  if (!is_int_dtype(aval_of(idx)$dtype)) jax_error("Index type must be an integer")
  idx <- lax_convert_element_type(idx, "int32")
  if (length(branches) == 1L) return(branches[[1L]](...))
  idx <- lax_clamp(0L, idx, length(branches) - 1L)
  if (isTRUE(.jx$config$jax_disable_jit) && is_jax_array(idx))
    return(branches[[as.integer(as.array(idx)) + 1L]](...))
  switch_impl(idx, branches, list(...))
}

# Refs that the branches close over are passed in and out as operands (ref_threaded).
switch_impl <- function(idx, branches, ops) ref_threaded(function(refs) {
  threaded <- length(refs) > 0L
  bs <- lapply(branches, function(b) {
    force(b)
    if (!threaded) return(function(...) ref_body(refs, list(), function() b(...))$out)
    function(vals, ...) {
      r <- ref_body(refs, vals, function() b(...))
      list(r$out, r$finals)
    }
  })
  pb <- prepare_branches(bs, if (threaded) c(list(lapply(refs, ref_read)), ops) else ops)
  kr_apply_in(pb$kr, pb$leaves, "cond", offset = 1L)
  out <- bind(cond_p, c(list(idx), pb$consts, pb$leaves), list(branches = pb$jaxprs))
  kr_apply_out(pb$kr, pb$leaves, out)
  out <- tree_unflatten(pb$out_tree, out)
  if (!threaded) return(out)
  for (i in seq_along(refs)) ref_write(refs[[i]], out[[2L]][[i]])
  out[[1L]]
})

# ---- while_loop --------------------------------------------------------------------------------

while_p <- new_primitive("while", multiple_results = TRUE)
# Port of loops._while_loop_abstract_eval: the result has the types of the body output.
while_p$abstract_eval <- function(avals, params) jaxpr_out_avals(params$body_jaxpr$jaxpr)
# Port of loops._while_lowering: consts are threaded through the loop carry.
while_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  nc <- params$cond_nconsts
  nb <- params$body_nconsts
  types <- vapply(avals_in, tensor_type, "")
  cj <- params$cond_jaxpr
  bj <- params$body_jaxpr
  ncarry <- length(ins) - nc - nb
  cond_region <- build_region(ctx, types, function(sub, args) {
    pred <- jaxpr_subcomp(sub, cj$jaxpr, lower_closed_consts(sub, cj),
                          as.list(c(args[seq_len(nc)], args[nc + nb + seq_len(ncarry)])))
    list(values = pred, types = "tensor<i1>")
  })
  body_region <- build_region(ctx, types, function(sub, args) {
    new <- jaxpr_subcomp(sub, bj$jaxpr, lower_closed_consts(sub, bj),
                         as.list(c(args[nc + seq_len(nb)], args[nc + nb + seq_len(ncarry)])))
    list(values = c(args[seq_len(nc + nb)], new), types = types)
  })
  outs <- emit_op(ctx, "stablehlo.while", ins, types, types, regions = c(cond_region, body_region))
  outs[nc + nb + seq_len(ncarry)]
}

# Port of loops._while_loop_jvp.
while_p$jvp <- function(primals, tangents, params) {
  nc <- params$cond_nconsts
  nb <- params$body_nconsts
  nz <- !vapply(tangents, is_zero, NA)
  bconst_nz <- nz[nc + seq_len(nb)]
  carry_nz <- drop_first(nz, nc + nb)
  ncarry <- length(carry_nz)
  for (it in seq_len(1L + ncarry)) {
    r <- jvp_jaxpr(params$body_jaxpr, c(bconst_nz, carry_nz), carry_nz)
    if (identical(r$out_nz, carry_nz)) break
    carry_nz <- carry_nz | r$out_nz
  }
  body_jvp <- r$jaxpr
  nbd <- sum(bconst_nz)
  ncd <- sum(carry_nz)
  # body_jvp inputs: (bconst, carry, bconst_dot, carry_dot) -> (bconst, bconst_dot, carry, carry_dot)
  in_order <- c(seq_len(nb), nb + ncarry + seq_len(nbd), nb + seq_len(ncarry), nb + ncarry + nbd + seq_len(ncd))
  body_r <- rearrange_jaxpr(body_jvp, in_order)
  cond_avals <- jaxpr_in_avals(params$cond_jaxpr$jaxpr)
  carry_avals <- cond_avals[nc + seq_len(ncarry)]
  dot_avals <- lapply(carry_avals[carry_nz], tangent_aval)
  cond_aug <- trace_to_jaxpr(function(a) eval_closed(params$cond_jaxpr, a[seq_len(nc + ncarry)]),
                             c(cond_avals, dot_avals))
  cond_aug$batched_predicate <- params$cond_jaxpr$batched_predicate
  cconst <- primals[seq_len(nc)]
  bconst <- primals[nc + seq_len(nb)]
  init <- drop_first(primals, nc + nb)
  bdot <- lapply(tangents[nc + seq_len(nb)][bconst_nz], instantiate_zero)
  idot <- lapply(drop_first(tangents, nc + nb)[carry_nz], instantiate_zero)
  out <- bind(while_p, c(cconst, bconst, bdot, init, idot),
              list(cond_jaxpr = cond_aug, body_jaxpr = body_r, cond_nconsts = nc,
                   body_nconsts = nb + nbd))
  po <- out[seq_len(ncarry)]
  to <- drop_first(out, ncarry)
  k <- 0L
  tans <- lapply(seq_len(ncarry), function(i) {
    if (carry_nz[i]) { k <<- k + 1L; to[[k]] } else zero_of(tangent_aval(aval_of(po[[i]])))
  })
  list(po, tans)
}

# Port of loops._while_loop_batching_rule.
while_p$batch <- function(vals, dims, params, axis) {
  size <- axis$size
  name <- axis$name
  nc <- params$cond_nconsts
  nb <- params$body_nconsts
  bat <- !is.na(dims)
  cconst_bat <- bat[seq_len(nc)]
  bconst_bat <- bat[nc + seq_len(nb)]
  carry_bat <- drop_first(bat, nc + nb)
  ncarry <- length(carry_bat)
  for (it in seq_len(1L + ncarry)) {
    r <- batch_jaxpr(params$body_jaxpr, size, name, c(bconst_bat, carry_bat), carry_bat)
    if (identical(r$out_batched, carry_bat)) break
    carry_bat <- carry_bat | r$out_batched
  }
  pred_bat <- batch_jaxpr(params$cond_jaxpr, size, name, c(cconst_bat, carry_bat), FALSE)$out_batched
  if (pred_bat) carry_bat <- rep(TRUE, ncarry)
  vals <- Map(function(v, d) if (is.na(d)) v else moveaxis(v, d, 0L), vals, dims)
  carry <- drop_first(vals, nc + nb)
  carry <- Map(function(v, was, now) if (now && !was) lax_broadcast(v, size) else v,
               carry, as.list(drop_first(bat, nc + nb)), as.list(carry_bat))
  cconst <- vals[seq_len(nc)]
  bconst <- vals[nc + seq_len(nb)]
  body_b <- batch_jaxpr(params$body_jaxpr, size, name, c(bconst_bat, carry_bat), carry_bat)$jaxpr
  cond_b <- batch_jaxpr(params$cond_jaxpr, size, name, c(cconst_bat, carry_bat), FALSE)$jaxpr
  if (pred_bat) {
    c_in <- jaxpr_in_avals(cond_b$jaxpr)
    b_in <- jaxpr_in_avals(body_b$jaxpr)
    # carry -> any(pred); body -> select(pred, body(carry), carry)
    cond_new <- trace_to_jaxpr(function(a) {
      p <- eval_closed(cond_b, a)[[1L]]
      list(lax_reduce_or(p, 0L))
    }, c_in)
    # In JAX the test of this loop is batched; checkify refuses such loops.
    cond_new$batched_predicate <- TRUE
    body_new <- trace_to_jaxpr(function(a) {
      cc <- a[seq_len(nc)]
      bc <- a[nc + seq_len(nb)]
      cr <- drop_first(a, nc + nb)
      p <- eval_closed(cond_b, c(cc, cr))[[1L]]
      new <- eval_closed(body_b, c(bc, cr))
      Map(function(n, o) bcast_select(p, n, o), new, cr)
    }, c(c_in[seq_len(nc)], b_in))
    out <- bind(while_p, c(cconst, cconst, bconst, carry),
                list(cond_jaxpr = cond_new, body_jaxpr = body_new, cond_nconsts = nc,
                     body_nconsts = nc + nb))
    return(list(out, rep(0L, ncarry)))
  }
  out <- bind(while_p, c(cconst, bconst, carry),
              list(cond_jaxpr = cond_b, body_jaxpr = body_b, cond_nconsts = nc, body_nconsts = nb))
  list(out, ifelse(carry_bat, 0L, NA_integer_))
}

# Port of loops._while_partial_eval: known carries come from a known loop; the whole loop
# is staged for the unknown carries (reverse mode then fails at transposition, as in JAX).
while_p$override$pe <- function(trace, args, params) {
  uk <- vapply(args, is_own, NA, trace = trace)
  nc <- params$cond_nconsts
  nb <- params$body_nconsts
  cc_uk <- uk[seq_len(nc)]
  bc_uk <- uk[nc + seq_len(nb)]
  carry_uk <- drop_first(uk, nc + nb)
  for (it in seq_len(1L + length(carry_uk))) {
    pe <- partial_eval_jaxpr(params$body_jaxpr, c(bc_uk, carry_uk), carry_uk)
    if (identical(pe$out_unknowns, carry_uk)) break
    carry_uk <- carry_uk | pe$out_unknowns
  }
  cpe <- partial_eval_jaxpr(params$cond_jaxpr, c(cc_uk, carry_uk), FALSE)
  if (any(cc_uk) || any(cpe$out_unknowns) || all(carry_uk))
    return(stage_eqn(trace, while_p, args, params))
  nk <- sum(!carry_uk)
  body_known <- trace_to_jaxpr(function(a) eval_closed(pe$known, a)[seq_len(nk)],
                               jaxpr_in_avals(pe$known$jaxpr))
  cond_known <- trace_to_jaxpr(function(a) eval_closed(cpe$known, a)[1L],
                               jaxpr_in_avals(cpe$known$jaxpr))
  cond_known$batched_predicate <- params$cond_jaxpr$batched_predicate
  carry <- drop_first(args, nc + nb)
  known_out <- with_trace(trace$parent, bind(while_p,
    c(args[seq_len(nc)][!cc_uk], args[nc + seq_len(nb)][!bc_uk], carry[!carry_uk]),
    list(cond_jaxpr = cond_known, body_jaxpr = body_known, cond_nconsts = sum(!cc_uk),
         body_nconsts = sum(!bc_uk))))
  staged <- stage_eqn(trace, while_p, args, params)
  out <- staged
  out[!carry_uk] <- known_out
  out
}

while_p$transpose <- function(cts, args, params) {
  jax_error("Reverse-mode differentiation does not work for lax.while_loop or lax.fori_loop ",
            "with dynamic start/stop values. Try using lax.scan, or using fori_loop with ",
            "static start/stop.", class = "jax_value_error")
}

#' While loop
#'
#' Port of `jax.lax.while_loop`. Reverse mode differentiation is not supported (as in
#' JAX); forward mode and vmap are.
#' @param cond_fun Function of the carry returning a scalar boolean.
#' @param body_fun Function of the carry returning a new carry of the same type.
#' @param init_val Initial carry (a pytree).
#' @export
lax_while_loop <- function(cond_fun, body_fun, init_val) {
  if (isTRUE(.jx$config$jax_disable_jit)) {
    val <- init_val
    while (isTRUE(as.logical(cond_fun(val)))) val <- body_fun(val)
    return(val)
  }
  # Refs that the functions close over become part of the loop state (ref_threaded).
  ref_threaded(function(refs) {
    if (!length(refs))
      return(while_main(function(c) ref_body(refs, list(), function() cond_fun(c))$out,
                        function(c) ref_body(refs, list(), function() body_fun(c))$out, init_val))
    out <- while_main(function(s) ref_body(refs, s[[2L]], function() cond_fun(s[[1L]]))$out,
                      function(s) {
                        r <- ref_body(refs, s[[2L]], function() body_fun(s[[1L]]))
                        list(r$out, r$finals)
                      }, list(init_val, lapply(refs, ref_read)))
    for (i in seq_along(refs)) ref_write(refs[[i]], out[[2L]][[i]])
    out[[1L]]
  })
}

while_main <- function(cond_fun, body_fun, init_val) {
  fl <- tree_flatten(init_val)
  init <- lapply(fl$leaves, as_jax)
  for (attempt in 1:2) {
    avals <- lapply(init, aval_of)
    wrapped <- tree_structure(list(init_val))
    ct <- trace_flat_fun(function(c) cond_fun(c), wrapped, avals)
    kr_revert(attr(ct$cj, "kr"))
    bt <- trace_flat_fun(function(c) body_fun(c), wrapped, avals)
    kr_revert(attr(bt$cj, "kr"))
    pa <- jaxpr_out_avals(ct$cj$jaxpr)
    if (length(pa) != 1L || pa[[1L]]$dtype != "bool" || length(pa[[1L]]$shape))
      jax_error("cond_fun must return a boolean scalar, but got output type(s) [",
                paste(vapply(pa, aval_str, ""), collapse = ", "), "].", class = "jax_type_error")
    if (!treedef_equal(bt$out_tree, fl$treedef))
      jax_error("body_fun output and input must have identical types (pytree structure ",
                format(bt$out_tree), " vs ", format(fl$treedef), ")", class = "jax_type_error")
    oa <- jaxpr_out_avals(bt$cj$jaxpr)
    fix <- weak_carry_mismatch(avals, oa)
    if (attempt == 2L || !any(fix)) break
    init[fix] <- promote_weak_carry(init[fix], oa[fix])
  }
  for (i in seq_along(oa)) if (!aval_eq(oa[[i]], avals[[i]]))
    jax_error("body_fun output and input must have identical types, got ", aval_str(oa[[i]]),
              " and ", aval_str(avals[[i]]), class = "jax_type_error")
  cc <- ct$cj
  bc <- bt$cj
  cj <- closed_jaxpr(convert_constvars(cc$jaxpr))
  bj <- closed_jaxpr(convert_constvars(bc$jaxpr))
  kr_check_while(attr(cc, "kr"), attr(bc, "kr"))
  kr_apply_in(attr(bc, "kr"), init, "while")
  out <- bind(while_p, c(cc$consts, bc$consts, init),
              list(cond_jaxpr = cj, body_jaxpr = bj, cond_nconsts = length(cc$consts),
                   body_nconsts = length(bc$consts)))
  kr_apply_out(attr(bc, "kr"), init, out)
  tree_unflatten(fl$treedef, out)
}

#' Loop from lower to upper
#'
#' Port of `jax.lax.fori_loop`: `body_fun(i, val)` for `i` in `lower..upper-1`. With
#' concrete bounds the loop is a scan (reverse mode differentiable); otherwise a while
#' loop.
#' @param lower,upper Loop bounds (integers).
#' @param body_fun Function `(i, val)`.
#' @param init_val Initial value.
#' @param unroll Ignored.
#' @export
lax_fori_loop <- function(lower, upper, body_fun, init_val, unroll = NULL) {
  concrete <- function(v) !is_tracer(v) || !is.null(tracer_concrete_value(v))
  lo <- as_jax(lower)
  hi <- as_jax(upper)
  dt <- promote_types(c(aval_of(lo)$dtype, aval_of(hi)$dtype), c(aval_of(lo)$weak, aval_of(hi)$weak))$dtype
  if (!is_int_dtype(dt)) jax_error("lower and upper arguments to fori_loop must have integer dtypes")
  if (aval_of(lo)$weak && aval_of(hi)$weak) dt <- default_int()
  lo <- lax_convert_element_type(lo, dt)
  hi <- lax_convert_element_type(hi, dt)
  if (concrete(lower) && concrete(upper)) {
    l <- as.integer(as.numeric(as.array(concrete_or_error(as_jax(lower)))))
    u <- as.integer(as.numeric(as.array(concrete_or_error(as_jax(upper)))))
    res <- lax_scan(function(carry, x) {
      i <- carry[[1L]]
      list(list(lax_add(i, lax_full_like(i, 1)), body_fun(i, carry[[2L]])), NULL)
    }, list(lax_convert_element_type(as_jax(l), dt), init_val), NULL, length = max(u - l, 0L),
    unroll = unroll %||% FALSE)
    return(res[[1L]][[2L]])
  }
  if (!is.null(unroll) && !isFALSE(unroll) && !identical(as.numeric(unroll), 1))
    jax_error("Can only use `unroll` in `fori_loop` if the loop bounds are statically known.",
              class = "jax_value_error")
  out <- lax_while_loop(function(c) lax_lt(c[[1L]], c[[2L]]),
                        function(c) list(lax_add(c[[1L]], lax_full_like(c[[1L]], 1)), c[[2L]],
                                         body_fun(c[[1L]], c[[3L]])),
                        list(lo, hi, init_val))
  out[[3L]]
}

# ---- scan ----------------------------------------------------------------------------------------

scan_p <- new_primitive("scan", multiple_results = TRUE)
scan_p$abstract_eval <- function(avals, params) {
  nk <- params$num_carry
  oa <- jaxpr_out_avals(params$jaxpr$jaxpr)
  nc <- params$num_consts
  carry <- oa[seq_len(nk)]
  ys <- lapply(drop_first(oa, nk), function(a) new_aval(c(params$length, a$shape), a$dtype, a$weak))
  c(carry, ys)
}

dyn_index0 <- function(x, i) {
  a <- aval_of(x)
  z <- lax_full_like(i, 0)
  idx <- c(list(i), rep(list(z), length(a$shape) - 1L))
  s <- bind(dynamic_slice_p, c(list(x), idx), list(slice_sizes = c(1L, a$shape[-1L])))
  lax_reshape(s, a$shape[-1L])
}

dyn_update0 <- function(buf, v, i) {
  a <- aval_of(buf)
  z <- lax_full_like(i, 0)
  idx <- c(list(i), rep(list(z), length(a$shape) - 1L))
  bind(dynamic_update_slice_p, c(list(buf, lax_reshape(v, c(1L, a$shape[-1L]))), idx))
}

# Port of loops._scan_impl: a while loop with dynamic indexing.
scan_impl <- function(args, params) {
  nc <- params$num_consts
  nk <- params$num_carry
  len <- params$length
  rev <- isTRUE(params$reverse)
  cj <- params$jaxpr
  consts <- args[seq_len(nc)]
  init <- args[nc + seq_len(nk)]
  xs <- drop_first(args, nc + nk)
  ya <- drop_first(jaxpr_out_avals(cj$jaxpr), nk)
  ys0 <- lapply(ya, function(a) lax_full(c(len, a$shape), 0, a$dtype))
  if (len == 0L) return(c(init, ys0))  # loops._scan_impl skips the loop for length 0
  i0 <- lax_convert_element_type(as_jax(0L), "int32")
  ny <- length(ys0)
  out <- lax_while_loop(
    function(c) lax_lt(c[[1L]], lax_full_like(c[[1L]], len)),
    function(c) {
      i <- c[[1L]]
      carry <- c[1L + seq_len(nk)]
      ys <- c[1L + nk + seq_len(ny)]
      idx <- if (rev) lax_sub(lax_full_like(i, len - 1L), i) else i
      x <- lapply(xs, dyn_index0, i = idx)
      o <- eval_jaxpr(cj$jaxpr, cj$consts, c(consts, carry, x))
      ys <- Map(function(b, v) dyn_update0(b, v, idx), ys, drop_first(o, nk))
      c(list(lax_add(i, lax_full_like(i, 1))), o[seq_len(nk)], ys)
    },
    c(list(i0), init, ys0))
  c(out[1L + seq_len(nk)], out[1L + nk + seq_len(ny)])
}
scan_p$lowering <- lower_fun(scan_impl, multiple_results = TRUE)

# Port of loops._scan_jvp.
scan_p$jvp <- function(primals, tangents, params) {
  nc <- params$num_consts
  nk <- params$num_carry
  cj <- params$jaxpr
  nx <- length(primals) - nc - nk
  ny <- length(cj$jaxpr$outvars) - nk
  nz <- !vapply(tangents, is_zero, NA)
  const_nz <- nz[seq_len(nc)]
  carry_nz <- nz[nc + seq_len(nk)]
  xs_nz <- nz[nc + nk + seq_len(nx)]
  for (it in seq_len(1L + nk)) {
    r <- jvp_jaxpr(cj, c(const_nz, carry_nz, xs_nz), c(carry_nz, rep(FALSE, ny)))
    cno <- r$out_nz[seq_len(nk)]
    if (identical(cno, carry_nz)) break
    carry_nz <- carry_nz | cno
  }
  out_nz <- r$out_nz
  ncd <- sum(const_nz); nkd <- sum(carry_nz); nxd <- sum(xs_nz)
  base <- nc + nk + nx
  in_order <- c(seq_len(nc), base + seq_len(ncd), nc + seq_len(nk), base + ncd + seq_len(nkd),
                nc + nk + seq_len(nx), base + ncd + nkd + seq_len(nxd))
  nyd <- sum(out_nz[nk + seq_len(ny)])
  ob <- nk + ny
  out_order <- c(seq_len(nk), ob + seq_len(nkd), nk + seq_len(ny), ob + nkd + seq_len(nyd))
  jr <- rearrange_jaxpr(r$jaxpr, in_order, out_order)
  consts <- primals[seq_len(nc)]
  init <- primals[nc + seq_len(nk)]
  xs <- primals[nc + nk + seq_len(nx)]
  cdot <- lapply(tangents[seq_len(nc)][const_nz], instantiate_zero)
  idot <- lapply(tangents[nc + seq_len(nk)][carry_nz], instantiate_zero)
  xdot <- lapply(tangents[nc + nk + seq_len(nx)][xs_nz], instantiate_zero)
  out <- bind(scan_p, c(consts, cdot, init, idot, xs, xdot),
              update_params(params, list(jaxpr = jr, num_consts = nc + ncd, num_carry = nk + nkd)))
  carry <- out[seq_len(nk)]
  carry_dot <- out[nk + seq_len(nkd)]
  ys <- out[nk + nkd + seq_len(ny)]
  ys_dot <- out[nk + nkd + ny + seq_len(nyd)]
  po <- c(carry, ys)
  td <- c(carry_dot, ys_dot)
  k <- 0L
  tans <- lapply(seq_along(po), function(i) {
    if (out_nz[i]) { k <<- k + 1L; td[[k]] } else zero_of(tangent_aval(aval_of(po[[i]])))
  })
  list(po, tans)
}

# Port of loops._scan_batching_rule.
scan_p$batch <- function(vals, dims, params, axis) {
  size <- axis$size
  name <- axis$name
  nc <- params$num_consts
  nk <- params$num_carry
  nx <- length(vals) - nc - nk
  ny <- length(params$jaxpr$jaxpr$outvars) - nk
  bat <- !is.na(dims)
  const_bat <- bat[seq_len(nc)]
  carry_bat <- bat[nc + seq_len(nk)]
  xs_bat <- bat[nc + nk + seq_len(nx)]
  for (it in seq_len(1L + nk)) {
    r <- batch_jaxpr(params$jaxpr, size, name, c(const_bat, carry_bat, xs_bat), c(carry_bat, rep(FALSE, ny)))
    cbo <- r$out_batched[seq_len(nk)]
    if (identical(cbo, carry_bat)) break
    carry_bat <- carry_bat | cbo
  }
  ys_bat <- r$out_batched[nk + seq_len(ny)]
  consts <- Map(function(v, d) if (is.na(d)) v else moveaxis(v, d, 0L), vals[seq_len(nc)], dims[seq_len(nc)])
  init <- Map(function(v, d, now) {
    if (now && is.na(d)) lax_broadcast(v, size) else if (now) moveaxis(v, d, 0L) else v
  }, vals[nc + seq_len(nk)], dims[nc + seq_len(nk)], as.list(carry_bat))
  xs <- Map(function(v, d) if (is.na(d) || d == 1L) v else moveaxis(v, d, 1L),
            vals[nc + nk + seq_len(nx)], dims[nc + nk + seq_len(nx)])
  out <- bind(scan_p, c(consts, init, xs), update_params(params, list(jaxpr = r$jaxpr)))
  list(out, c(ifelse(carry_bat, 0L, NA_integer_), ifelse(ys_bat, 1L, NA_integer_)))
}

# Port of loops._scan_partial_eval (residuals are stacked as extensive outputs).
scan_p$override$pe <- function(trace, args, params) {
  uk <- vapply(args, is_own, NA, trace = trace)
  nc <- params$num_consts
  nk <- params$num_carry
  nx <- length(args) - nc - nk
  cj <- params$jaxpr
  ny <- length(cj$jaxpr$outvars) - nk
  const_uk <- uk[seq_len(nc)]
  carry_uk <- uk[nc + seq_len(nk)]
  xs_uk <- uk[nc + nk + seq_len(nx)]
  for (it in seq_len(1L + nk)) {
    pe <- partial_eval_jaxpr(cj, c(const_uk, carry_uk, xs_uk), c(carry_uk, rep(FALSE, ny)))
    cuo <- pe$out_unknowns[seq_len(nk)]
    if (identical(cuo, carry_uk)) break
    carry_uk <- carry_uk | cuo
  }
  if (all(c(const_uk, carry_uk, xs_uk))) return(stage_eqn(trace, scan_p, args, params))
  ys_uk <- pe$out_unknowns[nk + seq_len(ny)]
  nres <- pe$num_res
  consts <- args[seq_len(nc)]
  init <- args[nc + seq_len(nk)]
  xs <- args[nc + nk + seq_len(nx)]
  nkk <- sum(!carry_uk)
  known_out <- with_trace(trace$parent, bind(scan_p,
    c(consts[!const_uk], init[!carry_uk], xs[!xs_uk]),
    update_params(params, list(jaxpr = pe$known, num_consts = sum(!const_uk), num_carry = nkk))))
  nyk <- sum(!ys_uk)
  carry_k <- known_out[seq_len(nkk)]
  ys_k <- known_out[nkk + seq_len(nyk)]
  res <- known_out[nkk + nyk + seq_len(nres)]
  # unknown jaxpr inputs: (res, uk consts, uk carry, uk xs) -> (uk consts, uk carry, uk xs, res)
  ncu <- sum(const_uk); nku <- sum(carry_uk); nxu <- sum(xs_uk)
  in_order <- c(nres + seq_len(ncu), nres + ncu + seq_len(nku), nres + ncu + nku + seq_len(nxu), seq_len(nres))
  ju <- rearrange_jaxpr(pe$unknown, in_order)
  staged <- stage_eqn(trace, scan_p, c(consts[const_uk], init[carry_uk], xs[xs_uk], res),
                      update_params(params, list(jaxpr = ju, num_consts = ncu, num_carry = nku)))
  carry_out <- vector("list", nk)
  carry_out[!carry_uk] <- carry_k
  carry_out[carry_uk] <- staged[seq_len(nku)]
  ys_out <- vector("list", ny)
  ys_out[!ys_uk] <- ys_k
  ys_out[ys_uk] <- staged[nku + seq_len(sum(ys_uk))]
  c(carry_out, ys_out)
}

# Port of loops._scan_transpose: consts = [non-linear residuals, linear consts],
# carry linear, xs = [linear xs, non-linear residuals].
scan_p$transpose <- function(cts, args, params) {
  nc <- params$num_consts
  nk <- params$num_carry
  cj <- params$jaxpr
  nx <- length(args) - nc - nk
  ny <- length(cj$jaxpr$outvars) - nk
  lin <- vapply(args, is_undefined, NA)
  consts_lin <- lin[seq_len(nc)]
  xs_lin <- lin[nc + nk + seq_len(nx)]
  nires <- sum(!consts_lin)
  neres <- sum(!xs_lin)
  if (!identical(consts_lin, c(rep(FALSE, nires), rep(TRUE, nc - nires))) ||
      !identical(xs_lin, c(rep(TRUE, nx - neres), rep(FALSE, neres)))) {
    # Reorder to consts = [non-linear, linear], xs = [linear, non-linear] and retry.
    cperm <- c(which(!consts_lin), which(consts_lin))
    xperm <- c(which(xs_lin), which(!xs_lin))
    perm <- c(cperm, nc + seq_len(nk), nc + nk + xperm)
    jr <- rearrange_jaxpr(cj, perm)
    r <- scan_p$transpose(cts, args[perm], update_params(params, list(jaxpr = jr)))
    back <- vector("list", length(args))
    back[perm] <- r
    return(back)
  }
  in_avals <- jaxpr_in_avals(cj$jaxpr)
  out_avals <- jaxpr_out_avals(cj$jaxpr)
  ires <- args[seq_len(nires)]
  eres <- args[nc + nk + nx - neres + seq_len(neres)]
  ncl <- nc - nires
  nxl <- nx - neres
  ct_carry <- Map(function(c, a) if (is_zero(c)) zeros_aval(strip_weak(a)) else c, cts[seq_len(nk)], out_avals[seq_len(nk)])
  ct_ys <- Map(function(c, a) if (is_zero(c)) lax_full(c(params$length, a$shape), 0, a$dtype) else c,
               cts[nk + seq_len(ny)], out_avals[nk + seq_len(ny)])
  ct_consts <- lapply(in_avals[nires + seq_len(ncl)], function(a) zeros_aval(strip_weak(a)))
  # transposed body: (ires) ; carry (ct_consts, ct_carry) ; xs (ct_y, eres) ->
  #                  carry (ct_consts', ct_carry') ; ys (ct_x)
  c_avals <- lapply(in_avals[nires + seq_len(ncl)], strip_weak)
  # As in JAX, the carried cotangents have the types of the incoming cotangents.
  k_avals <- lapply(ct_carry, function(c) if (is_symbolic_zero(c)) c$aval else aval_of(c))
  x_avals <- in_avals[nc + nk + seq_len(nxl)]
  e_avals <- in_avals[nc + nk + nxl + seq_len(neres)]
  y_avals <- lapply(out_avals[nk + seq_len(ny)], strip_weak)
  # The carried cotangents are strongly typed when the incoming one or the contribution
  # of the body is (the fixed point of the types, as in JAX): a weak one that the body
  # makes strong is made strong and the body traced again.
  trace_body <- function(k_avals) trace_to_jaxpr(function(a) {
    r1 <- a[seq_len(nires)]
    cbar <- a[nires + seq_len(ncl)]
    kbar <- a[nires + ncl + seq_len(nk)]
    ybar <- a[nires + ncl + nk + seq_len(ny)]
    r2 <- a[nires + ncl + nk + ny + seq_len(neres)]
    primals <- c(r1, lapply(in_avals[nires + seq_len(ncl)], undefined_primal),
                 lapply(in_avals[nc + seq_len(nk)], undefined_primal),
                 lapply(x_avals, undefined_primal), r2)
    bars <- backward_pass(cj$jaxpr, cj$consts, primals, c(kbar, ybar))
    new_cbar <- bars[nires + seq_len(ncl)]
    new_kbar <- bars[nc + seq_len(nk)]
    xbar <- bars[nc + nk + seq_len(nxl)]
    cbar2 <- Map(function(a, b, av) instantiate_zero(add_tangents(a, b)), cbar, new_cbar, c_avals)
    kbar2 <- Map(function(b, av) {
      if (is_zero(b)) return(zeros_aval(av))
      if (aval_of(b)$weak && !av$weak) b <- lax_convert_element_type(b, av$dtype, weak_type = FALSE)
      b
    }, new_kbar, k_avals)
    xbar2 <- Map(function(b, av) if (is_zero(b)) zeros_aval(strip_weak(av)) else b, xbar, x_avals)
    c(cbar2, kbar2, xbar2)
  }, c(in_avals[seq_len(nires)], c_avals, k_avals, y_avals, e_avals))
  # Each pass makes at least one more carry strong, so there are at most nk + 1.
  repeat {
    body_t <- trace_body(k_avals)
    kw <- vapply(jaxpr_out_avals(body_t$jaxpr)[ncl + seq_len(nk)], `[[`, NA, "weak")
    strengthen <- vapply(k_avals, `[[`, NA, "weak") & !kw
    if (!any(strengthen)) break
    k_avals[strengthen] <- lapply(k_avals[strengthen], strip_weak)
    ct_carry[strengthen] <- Map(function(c, a) {
      if (is_zero(c) || is_symbolic_zero(c)) zeros_aval(a)
      else lax_convert_element_type(c, a$dtype, weak_type = FALSE)
    }, ct_carry[strengthen], k_avals[strengthen])
  }
  out <- bind(scan_p, c(ires, ct_consts, ct_carry, ct_ys, eres),
              update_params(params, list(jaxpr = body_t, num_consts = nires, num_carry = ncl + nk,
                                      reverse = !isTRUE(params$reverse))))
  ct_c <- out[seq_len(ncl)]
  ct_init <- out[ncl + seq_len(nk)]
  ct_x <- out[ncl + nk + seq_len(nxl)]
  c(rep(list(NULL), nires), ct_c, ct_init, ct_x, rep(list(NULL), neres))
}

#' Scan a function over leading array axes while carrying state
#'
#' Port of `jax.lax.scan`: `f(carry, x)` returns `list(new_carry, y)`; the result is
#' `list(final_carry, stacked_ys)`.
#' @param f Function `(carry, x)`.
#' @param init Initial carry (pytree).
#' @param xs Pytree of arrays scanned along their leading axis, or `NULL`.
#' @param length Number of iterations (required when `xs` is `NULL`).
#' @param reverse Scan from the end.
#' @param unroll Number of iterations per loop step, or a logical. Validated as in JAX;
#'   the loop is always left rolled.
#' @export
lax_scan <- function(f, init, xs = NULL, length = NULL, reverse = FALSE, unroll = 1L) {
  # Refs that the body closes over become part of the carry (ref_threaded).
  ref_threaded(function(refs) {
    if (!base::length(refs))
      return(scan_main(function(carry, x) ref_body(refs, list(), function() f(carry, x))$out,
                       init, xs, length, reverse, unroll))
    out <- scan_main(function(carry, x) {
      r <- ref_body(refs, carry[[2L]], function() f(carry[[1L]], x))
      if (!is.list(r$out) || base::length(r$out) != 2L)
        jax_error("scan body function must return a list(carry, y)")
      list(list(r$out[[1L]], r$finals), r$out[[2L]])
    }, list(init, lapply(refs, ref_read)), xs, length, reverse, unroll)
    for (i in seq_along(refs)) ref_write(refs[[i]], out[[1L]][[2L]][[i]])
    list(out[[1L]][[1L]], out[[2L]])
  })
}

scan_main <- function(f, init, xs, length, reverse, unroll) {
  ifl <- tree_flatten(init)
  xfl <- tree_flatten(xs)
  init_l <- lapply(ifl$leaves, as_jax)
  xs_l <- lapply(xfl$leaves, as_jax)
  lens <- unique(vapply(xs_l, function(x) {
    s <- aval_of(x)$shape
    if (!base::length(s)) jax_error("scan got value with no leading axis to scan over")
    s[1L]
  }, 1L))
  if (base::length(lens) > 1L)
    jax_error("scan got values with different leading axis sizes: ", paste(lens, collapse = ", "))
  if (is.null(length)) {
    if (!base::length(lens)) jax_error("scan got no values to scan over and `length` not provided.")
    len <- lens
  } else {
    len <- as.integer(length)
    if (base::length(lens) && lens != len)
      jax_error("scan got `length` argument of ", len, " which disagrees with leading axis sizes ", lens)
  }
  # The loop is never unrolled here, but the argument is validated as in JAX.
  if (is.logical(unroll) && base::length(unroll) == 1L && !is.na(unroll))
    unroll <- if (unroll) max(len, 1L) else 1L
  if (!is.numeric(unroll) || base::length(unroll) != 1L || is.na(unroll) || unroll < 0 ||
      unroll != round(unroll))
    jax_error("`unroll` must be a `bool` or a non-negative `int`.", class = "jax_value_error")
  x_avals <- lapply(xs_l, function(x) { a <- aval_of(x); new_aval(a$shape[-1L], a$dtype, a$weak) })
  nk <- base::length(init_l)
  store <- new.env(parent = emptyenv())
  fun <- function(leaves) {
    carry <- tree_unflatten(ifl$treedef, leaves[seq_len(nk)])
    x <- tree_unflatten(xfl$treedef, drop_first(leaves, nk))
    r <- f(carry, x)
    if (!is.list(r) || base::length(r) != 2L)
      jax_error("scan body function must return a list(carry, y)")
    cfl <- tree_flatten(r[[1L]])
    if (!treedef_equal(cfl$treedef, ifl$treedef))
      jax_error("scan body function carry input and carry output must have the same pytree ",
                "structure, but they differ: ", format(ifl$treedef), " vs ", format(cfl$treedef),
                class = "jax_type_error")
    yfl <- tree_flatten(r[[2L]])
    store$y_tree <- yfl$treedef
    c(lapply(cfl$leaves, as_jax), lapply(yfl$leaves, as_jax))
  }
  for (attempt in 1:2) {
    c_avals <- lapply(init_l, aval_of)
    cj <- trace_to_jaxpr(fun, c(c_avals, x_avals))
    kr_revert(attr(cj, "kr"))
    oa <- jaxpr_out_avals(cj$jaxpr)[seq_len(nk)]
    fix <- weak_carry_mismatch(c_avals, oa)
    if (attempt == 2L || !any(fix)) break
    init_l[fix] <- promote_weak_carry(init_l[fix], oa[fix])
  }
  for (i in seq_len(nk)) if (!aval_eq(oa[[i]], c_avals[[i]]))
    jax_error("scan body function carry input and carry output must have equal types, got ",
              aval_str(c_avals[[i]]), " and ", aval_str(oa[[i]]), class = "jax_type_error")
  consts <- cj$consts
  jaxpr <- closed_jaxpr(convert_constvars(cj$jaxpr))
  kr_check_scan(attr(cj, "kr"), nk)
  kr_apply_in(attr(cj, "kr"), c(init_l, xs_l), "scan")
  out <- bind(scan_p, c(consts, init_l, xs_l),
              list(jaxpr = jaxpr, num_consts = base::length(consts), num_carry = nk,
                   length = len, reverse = isTRUE(reverse), unroll = as.integer(unroll)))
  kr_apply_out(attr(cj, "kr"), c(init_l, xs_l), out)
  list(tree_unflatten(ifl$treedef, out[seq_len(nk)]),
       tree_unflatten(store$y_tree, drop_first(out, nk)))
}

#' Map a function over leading array axes
#'
#' Port of `jax.lax.map` (sequential, via scan).
#' @param f Function of one pytree slice.
#' @param xs Pytree of arrays.
#' @param batch_size Optional batch size: slices are processed in batches with vmap.
#' @export
lax_map <- function(f, xs, batch_size = NULL) {
  if (!is.null(batch_size)) {
    fl <- tree_flatten(xs)
    n <- aval_of(as_jax(fl$leaves[[1L]]))$shape[1L]
    batch_size <- canon_shape(batch_size)
    # A batch size of 0 (or one above the length) leaves everything to one vmap.
    nb <- if (batch_size == 0L) 0L else n %/% batch_size
    if (nb == 0L) return(vmap(f)(xs))
    rem <- n - nb * batch_size
    main <- tree_map(function(x) {
      x <- as_jax(x)
      s <- aval_of(x)$shape
      lax_reshape(lax_slice_in_dim(x, 0L, nb * batch_size), c(nb, batch_size, s[-1L]))
    }, xs)
    out <- lax_scan(function(c, x) list(c, vmap(f)(x)), NULL, main)[[2L]]
    out <- tree_map(function(y) {
      s <- aval_of(y)$shape
      lax_reshape(y, c(s[1L] * s[2L], s[-(1:2)]))
    }, out)
    if (rem > 0L) {
      tail <- tree_map(function(x) lax_slice_in_dim(as_jax(x), nb * batch_size, n), xs)
      rest <- vmap(f)(tail)
      out <- tree_map(function(a, b) lax_concatenate(list(a, b), 0L), out, rest)
    }
    return(out)
  }
  lax_scan(function(c, x) list(c, f(x)), NULL, xs)[[2L]]
}

# ---- custom_linear_solve (port of jax/_src/lax/control_flow/solves.py) -------------------

linear_solve_p <- new_primitive("custom_linear_solve", multiple_results = TRUE)

ls_split <- function(args, cl) {
  off <- 0L
  parts <- lapply(cl, function(n) { r <- args[off + seq_len(n)]; off <<- off + n; r })
  list(params = parts, b = drop_first(args, off))
}

ls_transpose_tuple <- function(x) list(matvec = x$vecmat, vecmat = x$matvec,
                                       solve = x$transpose_solve, transpose_solve = x$solve)

linear_solve_p$abstract_eval <- function(avals, params) {
  jaxpr_out_avals(params$jaxprs$solve$jaxpr)
}

linear_solve_impl <- function(args, params) {
  s <- ls_split(args, params$const_lengths)
  eval_closed(params$jaxprs$solve, c(s$params$solve, s$b))
}
linear_solve_p$lowering <- lower_fun(linear_solve_impl, multiple_results = TRUE)

linear_solve_p$jvp <- function(primals, tangents, params) {
  x <- bind(linear_solve_p, primals, params)
  s <- ls_split(primals, params$const_lengths)
  st <- ls_split(tangents, params$const_lengths)
  nx <- length(st$b)
  x_leaves <- x[seq_len(nx)]
  b_dot <- st$b
  if (all(vapply(st$params$matvec, is_zero, NA))) {
    rhs <- b_dot
  } else {
    mv <- params$jaxprs$matvec
    np <- length(s$params$matvec)
    f <- function(...) eval_closed(mv, list(...))
    pt <- c(lapply(st$params$matvec, instantiate_zero),
            lapply(x_leaves, function(v) lax_full_like(v, 0)))
    mt <- jvp(f, c(s$params$matvec, x_leaves), pt)[[2L]]
    rhs <- Map(function(b, m) add_tangents(b, lax_neg(m)), b_dot, mt)
  }
  rhs <- Map(function(r, v) if (is_zero(r)) lax_full_like(v, 0) else r, rhs, x_leaves)
  x_dot <- bind(linear_solve_p, c(do.call(c, unname(s$params)), rhs), params)
  tans <- lapply(seq_along(x), function(i) {
    if (i <= nx) x_dot[[i]] else zero_of(tangent_aval(aval_of(x[[i]])))
  })
  list(x, tans)
}

linear_solve_p$transpose <- function(cts, args, params) {
  if (is.null(params$jaxprs$transpose_solve))
    jax_error("transpose_solve required for backwards mode automatic differentiation of ",
              "custom_linear_solve", class = "jax_type_error")
  s <- ls_split(args, params$const_lengths)
  nb <- length(s$b)
  xct <- Map(function(c, v) if (is_zero(c)) zeros_aval(strip_weak(v$aval)) else c, cts[seq_len(nb)], s$b)
  tp <- ls_transpose_tuple(s$params)
  out <- bind(linear_solve_p, c(do.call(c, unname(tp)), xct),
              list(const_lengths = ls_transpose_tuple(params$const_lengths),
                   jaxprs = ls_transpose_tuple(params$jaxprs)))
  c(rep(list(NULL), sum(unlist(params$const_lengths))), out[seq_len(nb)])
}

linear_solve_p$batch <- function(vals, dims, params, axis) {
  size <- axis$size
  name <- axis$name
  cl <- params$const_lengths
  j <- params$jaxprs
  bat <- !is.na(dims)
  s_bat <- ls_split(bat, cl)
  pb <- s_bat$params
  orig_b_bat <- s_bat$b
  n_op <- length(j$matvec$jaxpr$outvars)
  n_aux <- length(j$solve$jaxpr$outvars) - n_op
  b_bat <- orig_b_bat
  x_bat <- rep(FALSE, length(j$solve$jaxpr$outvars))
  bj <- function(cj, b, inst) if (is.null(cj)) NULL else batch_jaxpr(cj, size, name, b, inst)
  for (it in seq_len(1L + length(b_bat) + length(x_bat))) {
    sr <- bj(j$solve, c(pb$solve, b_bat), x_bat)
    if (is.null(j$vecmat)) {
      vr <- NULL
      x_bat_out <- sr$out_batched
    } else {
      vr <- bj(j$vecmat, c(pb$vecmat, b_bat), b_bat)
      x_bat_out <- c(vr$out_batched, rep(TRUE, n_aux)) | sr$out_batched
    }
    x_noaux <- x_bat_out[seq_len(n_op)]
    mr <- bj(j$matvec, c(pb$matvec, x_noaux), b_bat)
    if (is.null(j$transpose_solve)) {
      tr <- NULL
      b_bat_out <- mr$out_batched | orig_b_bat
    } else {
      tr <- bj(j$transpose_solve, c(pb$transpose_solve, x_noaux), x_bat_out)
      b_bat_out <- mr$out_batched | tr$out_batched[seq_along(orig_b_bat)] | orig_b_bat
    }
    if (identical(x_bat_out, x_bat) && identical(b_bat_out, b_bat)) break
    x_bat <- x_bat_out
    b_bat <- b_bat_out
  }
  new_j <- list(matvec = mr$jaxpr, vecmat = vr$jaxpr, solve = sr$jaxpr, transpose_solve = tr$jaxpr)
  s <- ls_split(vals, cl)
  sd <- ls_split(dims, cl)
  new_params <- Map(function(x, d) if (is.na(d)) x else moveaxis(x, d, 0L),
                    do.call(c, unname(s$params)), do.call(c, unname(sd$params)))
  new_b <- Map(function(x, d, was, now) {
    if (now && !was) lax_broadcast(x, size) else if (now) moveaxis(x, d, 0L) else x
  }, s$b, sd$b, as.list(orig_b_bat), as.list(b_bat))
  out <- bind(linear_solve_p, c(new_params, new_b), list(const_lengths = cl, jaxprs = new_j))
  list(out, ifelse(sr$out_batched, 0L, NA_integer_))
}

#' Matrix free linear solve with implicit differentiation
#'
#' Port of `jax.lax.custom_linear_solve`. Gradients are defined by implicit
#' differentiation at the solution instead of differentiating through `solve`.
#' @param matvec Linear function computing `A x`.
#' @param b Right hand side (pytree).
#' @param solve Function `(matvec, b)` returning `x` with `matvec(x) == b`.
#' @param transpose_solve Function `(vecmat, b)` solving the transposed system; needed
#'   for reverse mode unless `symmetric`.
#' @param symmetric Whether the linear map is symmetric.
#' @param has_aux Whether `solve` returns `list(x, aux)`.
#' @export
lax_custom_linear_solve <- function(matvec, b, solve, transpose_solve = NULL,
                                    symmetric = FALSE, has_aux = FALSE) {
  if (is.null(transpose_solve) && symmetric) transpose_solve <- solve
  fl <- tree_flatten(b)
  b_l <- lapply(fl$leaves, as_jax)
  b_avals <- lapply(b_l, aval_of)
  tree <- tree_structure(list(b))
  tr <- function(f, nm, aux = FALSE) {
    t <- trace_flat_fun(function(x) {
      y <- f(x)
      yy <- if (aux) y[[1L]] else y
      ys <- tree_leaves(yy)
      if (length(ys) != length(b_l) || !all(mapply(function(u, v) identical(aval_of(u)$shape, v$shape), ys, b_avals)))
        jax_error(nm, "() output shapes must match b", class = "jax_type_error")
      y
    }, tree, b_avals)
    list(jaxpr = closed_jaxpr(convert_constvars(t$cj$jaxpr)), consts = t$cj$consts,
         out_tree = t$out_tree)
  }
  mv <- tr(matvec, "matvec")
  sv <- tr(function(x) solve(matvec, x), "solve", has_aux)
  vm <- NULL
  ts <- NULL
  if (!is.null(transpose_solve)) {
    vecmat <- if (symmetric) matvec else {
      lt <- NULL
      function(x) {
        if (is.null(lt)) lt <<- linear_transpose(matvec, b)
        lt(x)[[1L]]
      }
    }
    vm <- if (symmetric) mv else tr(vecmat, "vecmat")
    ts <- tr(function(x) transpose_solve(vecmat, x), "transpose_solve", has_aux)
  }
  cl <- list(matvec = length(mv$consts), vecmat = length(vm$consts),
             solve = length(sv$consts), transpose_solve = length(ts$consts))
  jaxprs <- list(matvec = mv$jaxpr, vecmat = vm$jaxpr, solve = sv$jaxpr, transpose_solve = ts$jaxpr)
  out <- bind(linear_solve_p, c(mv$consts, vm$consts, sv$consts, ts$consts, b_l),
              list(const_lengths = cl, jaxprs = jaxprs))
  tree_unflatten(sv$out_tree, out)
}
