# Gradient checkpointing (port of jax/_src/ad_checkpoint.py).

# ---- policies ------------------------------------------------------------------------------

#' Checkpoint policies
#'
#' Port of `jax.checkpoint_policies`. A policy is a function `(prim_name, params)`
#' returning `TRUE` when the output of that primitive may be saved as a residual
#' instead of being recomputed in the backward pass. Offloading policies behave like
#' the corresponding saving policies (there is one memory space).
#' @format A list of policies and policy constructors.
#' @export
checkpoint_policies <- list(
  everything_saveable = function(prim, params) TRUE,
  nothing_saveable = function(prim, params) FALSE,
  dots_saveable = function(prim, params) prim == "dot_general",
  checkpoint_dots = function(prim, params) prim == "dot_general",
  dots_with_no_batch_dims_saveable = function(prim, params) {
    prim == "dot_general" && !length(params$dimension_numbers$lhs_batch)
  },
  checkpoint_dots_with_no_batch_dims = function(prim, params) {
    prim == "dot_general" && !length(params$dimension_numbers$lhs_batch)
  },
  save_anything_except_these_names = function(...) {
    nms <- c(...)
    function(prim, params) !(prim == "name" && params$name %in% nms)
  },
  save_any_names_but_these = function(...) {
    nms <- c(...)
    function(prim, params) prim == "name" && !(params$name %in% nms)
  },
  save_only_these_names = function(...) {
    nms <- c(...)
    function(prim, params) prim == "name" && params$name %in% nms
  },
  save_and_offload_only_these_names = function(names_which_can_be_saved = character(0),
                                               names_which_can_be_offloaded = character(0),
                                               offload_src = NULL, offload_dst = NULL) {
    nms <- c(names_which_can_be_saved, names_which_can_be_offloaded)
    function(prim, params) prim == "name" && params$name %in% nms
  },
  offload_dot_with_no_batch_dims = function(offload_src = NULL, offload_dst = NULL) {
    function(prim, params) prim == "dot_general" && !length(params$dimension_numbers$lhs_batch)
  },
  save_from_both_policies = function(policy_1, policy_2) {
    function(prim, params) policy_1(prim, params) || policy_2(prim, params)
  }
)

# ---- name primitive (checkpoint_name) --------------------------------------------------------

name_p <- new_primitive("name")
name_p$abstract_eval <- function(avals, params) avals[[1L]]
name_p$lowering <- function(ctx, ins, ai, ao, params) ins[[1L]]
deflinear(name_p, function(ct, args, params) list(ct))
name_p$batch <- function(vals, dims, params, axis) list(bind(name_p, vals, params), dims[1L])

#' Name an intermediate value for checkpoint policies
#'
#' Port of `jax.ad_checkpoint.checkpoint_name`.
#' @param x A pytree of arrays.
#' @param name Name used by [checkpoint_policies] such as `save_only_these_names`.
#' @export
checkpoint_name <- function(x, name) {
  tree_map(function(l) bind(name_p, list(as_jax(l)), list(name = name)), x)
}

# ---- optimization barrier -------------------------------------------------------------------

optimization_barrier_p <- new_primitive("optimization_barrier", multiple_results = TRUE)
optimization_barrier_p$abstract_eval <- function(avals, params) avals
optimization_barrier_p$lowering <- function(ctx, ins, ai, ao, params) {
  if (!length(ins)) return(character(0))
  ts <- vapply(ai, tensor_type, "")
  emit_op(ctx, "stablehlo.optimization_barrier", ins, ts, ts)
}
optimization_barrier_p$jvp <- function(primals, tangents, params) {
  nz <- !vapply(tangents, is_zero, NA)
  out <- bind(optimization_barrier_p, c(primals, lapply(tangents[nz], instantiate_zero)))
  n <- length(primals)
  po <- out[seq_len(n)]
  td <- drop_first(out, n)
  k <- 0L
  list(po, lapply(seq_len(n), function(i) {
    if (nz[i]) { k <<- k + 1L; td[[k]] } else zero_of(tangent_aval(aval_of(po[[i]])))
  }))
}
optimization_barrier_p$transpose <- function(cts, args, params) {
  lapply(seq_along(args), function(i) if (is_undefined(args[[i]])) cts[[i]] else NULL)
}
optimization_barrier_p$batch <- function(vals, dims, params, axis) {
  list(bind(optimization_barrier_p, vals), dims)
}

#' Optimization barrier
#'
#' Port of `jax.lax.optimization_barrier`: prevents XLA from moving computations across.
#' @param operand A pytree of arrays.
#' @export
lax_optimization_barrier <- function(operand) {
  fl <- tree_flatten(operand)
  out <- bind(optimization_barrier_p, lapply(fl$leaves, as_jax))
  tree_unflatten(fl$treedef, out)
}

# ---- remat primitive ------------------------------------------------------------------------

remat_p <- new_primitive("checkpoint", multiple_results = TRUE)
remat_p$abstract_eval <- function(avals, params) jaxpr_out_avals(params$jaxpr$jaxpr)
remat_p$override$eval <- function(trace, args, params) eval_closed(params$jaxpr, args)

# Port of ad_checkpoint.remat_lowering: inputs pass through an optimization barrier when
# the computation was differentiated, so XLA does not undo the rematerialization.
remat_p$lowering <- function(ctx, ins, ai, ao, params) {
  args <- as.list(ins)
  if (isTRUE(params$differentiated) && isTRUE(params$prevent_cse) && length(ins)) {
    ts <- vapply(ai, tensor_type, "")
    args <- as.list(emit_op(ctx, "stablehlo.optimization_barrier", ins, ts, ts))
  }
  cj <- params$jaxpr
  jaxpr_subcomp(ctx, cj$jaxpr, lower_closed_consts(ctx, cj), args)
}

# Port of ad_checkpoint.remat_jvp.
remat_p$jvp <- function(primals, tangents, params) {
  nz <- !vapply(tangents, is_zero, NA)
  r <- jvp_jaxpr(params$jaxpr, nz, FALSE)
  out <- bind(remat_p, c(primals, lapply(tangents[nz], instantiate_zero)),
              update_params(params, list(jaxpr = r$jaxpr)))
  n <- length(params$jaxpr$jaxpr$outvars)
  po <- out[seq_len(n)]
  td <- drop_first(out, n)
  k <- 0L
  list(po, lapply(seq_len(n), function(i) {
    if (r$out_nz[i]) { k <<- k + 1L; td[[k]] } else zero_of(tangent_aval(aval_of(po[[i]])))
  }))
}

# Port of ad_checkpoint.remat_vmap.
remat_p$batch <- function(vals, dims, params, axis) {
  vals <- Map(function(v, d) if (is.na(d)) v else moveaxis(v, d, 0L), vals, dims)
  r <- batch_jaxpr(params$jaxpr, axis$size, axis$name, !is.na(dims), FALSE)
  out <- bind(remat_p, vals, update_params(params, list(jaxpr = r$jaxpr)))
  list(out, ifelse(r$out_batched, 0L, NA_integer_))
}

# Port of ad_checkpoint.remat_partial_eval: known outputs are computed now (by partial
# evaluation, so control flow with mixed known and unknown inputs is split correctly);
# unknown outputs are staged as a remat over all inputs plus the residuals the policy
# saves, and everything else is recomputed when the staged computation runs.
remat_p$override$pe <- function(trace, args, params) {
  uk <- vapply(args, is_own, NA, trace = trace)
  cj <- params$jaxpr
  jaxpr <- cj$jaxpr
  policy <- params$policy %||% checkpoint_policies$nothing_saveable
  pe <- partial_eval_jaxpr(cj, uk, FALSE)
  out_uk <- pe$out_unknowns
  nko <- sum(!out_uk)
  known_res <- with_trace(trace$parent, eval_closed(pe$known, args[!uk]))
  known_outs <- known_res[seq_len(nko)]
  if (identical(policy, checkpoint_policies$everything_saveable)) {
    res <- drop_first(known_res, nko)
    staged <- stage_eqn(trace, remat_p, c(res, args[uk]),
                        update_params(params, list(jaxpr = pe$unknown, differentiated = TRUE)))
  } else {
    n <- jaxpr$nvars
    val <- vector("list", n)
    kn <- logical(n)
    saved <- logical(n)
    for (i in seq_along(jaxpr$constvars)) {
      id <- jaxpr$constvars[[i]]$id
      val[[id]] <- cj$consts[[i]]
      kn[id] <- TRUE
    }
    for (i in seq_along(jaxpr$invars)) {
      id <- jaxpr$invars[[i]]$id
      if (!uk[i]) { val[[id]] <- args[[i]]; kn[id] <- TRUE }
    }
    known_atom <- function(a) is_lit(a) || kn[a$id]
    rd <- function(a) if (is_lit(a)) a$val else val[[a$id]]
    # Residuals saved by the policy: outputs of saveable equations with known inputs.
    for (e in jaxpr$eqns) {
      if (!all(vapply(e$invars, known_atom, NA))) next
      out <- with_trace(trace$parent, bind(e$prim, lapply(e$invars, rd), e$params))
      if (!e$prim$multiple_results) out <- list(out)
      sv <- isTRUE(policy(e$prim$name, e$params))
      for (k in seq_along(e$outvars)) {
        id <- e$outvars[[k]]$id
        val[[id]] <- out[[k]]
        kn[id] <- TRUE
        saved[id] <- sv
      }
    }
    res_ids <- which(saved)
    res_vals <- val[res_ids]
    in_avals <- jaxpr_in_avals(jaxpr)
    res_avals <- lapply(res_vals, function(v) strip_weak(aval_of(v)))
    nin <- length(in_avals)
    staged_fun <- function(a) {
      env <- vector("list", n)
      for (i in seq_along(jaxpr$constvars)) env[[jaxpr$constvars[[i]]$id]] <- cj$consts[[i]]
      for (i in seq_len(nin)) env[[jaxpr$invars[[i]]$id]] <- a[[i]]
      for (k in seq_along(res_ids)) env[[res_ids[k]]] <- a[[nin + k]]
      r2 <- function(x) if (is_lit(x)) x$val else env[[x$id]]
      for (e in jaxpr$eqns) {
        ids <- vapply(e$outvars, `[[`, 1L, "id")
        if (all(saved[ids])) next
        out <- bind(e$prim, lapply(e$invars, r2), e$params)
        if (!e$prim$multiple_results) out <- list(out)
        for (k in seq_along(ids)) env[[ids[k]]] <- out[[k]]
      }
      lapply(jaxpr$outvars[out_uk], r2)
    }
    traced <- trace_to_jaxpr(staged_fun, c(in_avals, res_avals))
    staged_cj <- closed_jaxpr(convert_constvars(traced$jaxpr), list())
    staged <- stage_eqn(trace, remat_p, c(traced$consts, args, res_vals),
                        update_params(params, list(jaxpr = staged_cj, differentiated = TRUE)))
  }
  out <- vector("list", length(out_uk))
  out[!out_uk] <- known_outs
  out[out_uk] <- staged
  out
}

# Port of ad_checkpoint.remat_transpose: the transposed computation recomputes the
# non-linear parts (backward_pass forward-evaluates them) inside another remat.
remat_p$transpose <- function(cts, args, params) {
  linear <- vapply(args, is_undefined, NA)
  out_avals <- jaxpr_out_avals(params$jaxpr$jaxpr)
  cts <- Map(function(c, a) if (is_zero(c)) zeros_aval(a) else c, cts, out_avals)
  tj <- transpose_jaxpr(params$jaxpr, linear)
  out <- bind(remat_p, c(args[!linear], cts), update_params(params, list(jaxpr = tj)))
  res <- vector("list", length(args))
  res[linear] <- out
  res
}

#' Gradient checkpointing
#'
#' Port of `jax.checkpoint` (alias `remat`): the function's intermediate values are not
#' saved for the backward pass but recomputed, except those allowed by `policy`.
#' @param fun Function.
#' @param prevent_cse Keep XLA from merging the recomputation with the forward pass.
#' @param policy A policy from [checkpoint_policies] or `NULL` (save nothing).
#' @param static_argnums 0-based positions of static arguments.
#' @return The wrapped function.
#' @export
checkpoint <- function(fun, prevent_cse = TRUE, policy = NULL, static_argnums = integer(0)) {
  force(fun)
  function(...) {
    args <- list(...)
    # Port of ad_checkpoint._remat_static_argnums.
    nums <- as.integer(static_argnums)
    if (any(nums < -length(args) | nums >= length(args)))
      jax_error("the `static_argnums` argument to `checkpoint` / `remat` can only take ",
                "integer values greater than or equal to `-length(args)` and less than ",
                "`length(args)`, but got (", paste(nums, collapse = ", "), "), while ",
                "`length(args)` = ", length(args), class = "jax_value_error")
    dyn <- setdiff(seq_along(args), nums %% length(args) + 1L)
    fl <- tree_flatten(args[dyn])
    leaves <- lapply(fl$leaves, as_jax)
    store <- new.env(parent = emptyenv())
    cj <- trace_to_jaxpr(function(ls) {
      a <- args
      a[dyn] <- tree_unflatten(fl$treedef, ls)
      ofl <- tree_flatten(do.call(fun, a))
      store$out_tree <- ofl$treedef
      lapply(ofl$leaves, as_jax)
    }, lapply(leaves, aval_of))
    jaxpr <- closed_jaxpr(convert_constvars(cj$jaxpr))
    kr_apply_in(attr(cj, "kr"), leaves, "remat")
    out <- bind(remat_p, c(cj$consts, leaves),
                list(jaxpr = jaxpr, prevent_cse = isTRUE(prevent_cse), differentiated = FALSE,
                     policy = policy))
    kr_apply_out(attr(cj, "kr"), leaves, out)
    tree_unflatten(store$out_tree, out)
  }
}

#' @rdname checkpoint
#' @export
remat <- checkpoint
