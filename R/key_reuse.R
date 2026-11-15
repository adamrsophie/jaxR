# Key reuse checking (port of jax.experimental.key_reuse._core).
#
# With the option jax_debug_key_reuse, a typed key array that is used by a random
# function (a "sink": random_bits, split, gamma, slicing) is marked consumed, and a key
# that is consumed already raises a KeyReuseError. JAX finds the uses inside a function
# by reading its jaxpr, where keys have a key dtype. In jaxR keys are uint32 arrays in a
# jaxpr, so the uses are recorded while the function is traced: the state lives on the
# environment of the key data (an array or a tracer). At the boundaries of a trace (jit,
# cond, scan, while_loop, vmap) the uses of the inputs make up the signature of the
# function (KeyReuseSignature: sinks, sources and forwards), which is applied to the
# arguments of each call as in JAX.

kr_on <- function() isTRUE(.jx$config$jax_debug_key_reuse)

# The fields are read with get0(), since some classed environments define `$`.
kr_get <- function(d, what = "kr_consumed") {
  if (is.environment(d)) get0(what, envir = d, inherits = FALSE)
}
kr_set <- function(d, value, what = "kr_consumed") assign(what, value, envir = d)

# The internal functions of random (prng_jit) are traced quietly: as in JAX, where the
# signature of their jaxpr does not see the reuse of a key inside them (the
# implementation of poisson passes its key to two loops), only the use of their key
# argument counts.
kr_quiet <- function() isTRUE(.jx$kr_quiet)

# The state: NULL or FALSE (not consumed), TRUE, or a logical array of the key shape.
kr_state <- function(d) {
  if (!is.environment(d) || !inherits(d, "jax_value")) return(FALSE)
  kr_get(d) %||% FALSE
}

kr_norm <- function(m) {
  if (!length(m) || !any(m)) return(FALSE)
  if (all(m)) return(TRUE)
  m
}

kr_overlap <- function(a, b) {
  if (length(a) == length(b) || length(a) == 1L || length(b) == 1L) any(a & b) else any(a) && any(b)
}

kr_or <- function(a, b) {
  if (length(a) == length(b) || length(a) == 1L || length(b) == 1L) kr_norm(a | b)
  else kr_norm(any(a) | any(b))
}

# Called when a key array is made: a new key is not consumed (a Source).
kr_init <- function(data) {
  if (kr_on() && is.environment(data) && inherits(data, "jax_value") && is.null(kr_get(data)))
    kr_set(data, FALSE)
  invisible(data)
}

# Inside a trace the message is the one of JAX's jaxpr check, outside the one of a call.
kr_reuse_error <- function(name, idx, d = NULL) {
  if (kr_quiet()) return(invisible())
  if (is_tracer(d))
    jax_error("In ", name, ", argument ", idx, " is already consumed.", class = "jax_key_reuse_error")
  jax_error("Previously-consumed key passed to ", name, " at index ", idx,
            class = "jax_key_reuse_error")
}

# Marks `d` consumed where `mask` is TRUE; an overlap with what was consumed before is an
# error. The trace being built (if any) records the use, so that a key it closes over is
# part of its signature.
kr_consume <- function(d, mask, name, idx) {
  if (!is.environment(d) || !inherits(d, "jax_value")) return(invisible())
  cur <- kr_state(d)
  if (kr_overlap(cur, mask)) kr_reuse_error(name, idx, d)
  kr_set(d, name, "kr_by")
  log <- .jx$kr_log
  if (!is.null(log)) log$items[[length(log$items) + 1L]] <- list(d = d, prev = kr_get(d))
  kr_set(d, kr_or(cur, mask))
  # A traced key that forwards another (Forward in JAX) consumes that one too.
  a <- kr_get(d, "kr_alias")
  if (!is.null(a)) kr_consume(a, TRUE, name, idx)
  invisible()
}

# A random function consumes its key (Sink(0)).
kr_sink <- function(key, name) {
  if (kr_on() && is_prng_key(key)) kr_consume(key$data, TRUE, name, 0L)
  invisible(key)
}

# ---- signatures of traced functions --------------------------------------------------

kr_begin <- function() {
  if (!kr_on()) return(NULL)
  log <- new.env(parent = emptyenv())
  log$items <- list()
  log$old <- .jx$kr_log
  .jx$kr_log <- log
  log
}

# The signature of a trace from its inputs and outputs: `sinks` (per input, the mask
# consumed, or NULL), `out` (per output, NULL for a value that is not a key,
# list(fwd = i) for an input returned as it is, or list(state = ) for a new key), and
# `consts`, the keys from outside the trace that it consumed (with their states before).
# The uses of keys from outside stay recorded (also in the trace around this one): a
# boundary that runs the traced function undoes them and applies the signature at each
# call (kr_revert, kr_apply), and one that only traces undoes them (kr_trace_only).
kr_end <- function(log, trace, ins, outs) {
  .jx$kr_log <- log$old
  consts <- list()
  prevs <- list()
  for (it in log$items) {
    d <- it$d
    if (is_tracer(d) && identical(d$trace, trace)) next
    if (any(vapply(consts, identical, NA, d))) next
    consts[[length(consts) + 1L]] <- d
    prevs[length(prevs) + 1L] <- list(it$prev)
  }
  if (!is.null(log$old))
    for (i in seq_along(consts))
      log$old$items[[length(log$old$items) + 1L]] <- list(d = consts[[i]], prev = prevs[[i]])
  sinks <- lapply(ins, function(t) {
    m <- kr_get(t)
    if (!is.null(m) && any(m)) m
  })
  names <- vapply(ins, function(t) kr_get(t, "kr_by") %||% "", "")
  out <- lapply(outs, function(o) {
    if (is.null(kr_get(o))) return(NULL)
    i <- which(vapply(ins, identical, NA, o))
    if (length(i)) return(list(fwd = i[1L]))
    # A key from outside the trace that is returned forwards its state at each call.
    if (!is_tracer(o) || !identical(o$trace, trace)) return(list(const = o))
    list(state = kr_get(o))
  })
  list(sinks = sinks, out = out, consts = consts, prevs = prevs, names = names)
}

# As in JAX, a function that is only traced (make_jaxpr, eval_shape, lower) consumes no
# key: the uses are undone, here and in the trace around this one.
kr_trace_only <- function(cj) {
  sig <- attr(cj, "kr")
  if (is.null(sig) || !length(sig$consts)) return(invisible(cj))
  kr_revert(sig)
  log <- .jx$kr_log
  if (!is.null(log))
    log$items <- Filter(function(it) !any(vapply(sig$consts, identical, NA, it$d)), log$items)
  invisible(cj)
}

# Applies the signature `sig` to `ins` and runs `expr`, which evaluates the traced
# function again (the R rules of custom derivative functions run again under it). The
# keys the signature consumes get their states from before while `expr` runs, so that
# the uses the rules make again are checked as usual and the uses they add (a forward
# rule using a key the function does not) count; afterwards each key holds both.
kr_replay <- function(sig, ins, name, expr, offset = 0L) {
  if (is.null(sig) || !kr_on()) {
    kr_apply_in(sig, ins, name, offset)
    return(expr)
  }
  # The keys an input forwards (kr_alias) are consumed with it.
  targets <- list()
  for (t in c(ins, sig$consts))
    while (is.environment(t) && inherits(t, "jax_value") &&
           !any(vapply(targets, identical, NA, t))) {
      targets[[length(targets) + 1L]] <- t
      t <- kr_get(t, "kr_alias")
    }
  pre <- lapply(targets, kr_get)
  kr_apply_in(sig, ins, name, offset)
  post <- lapply(targets, kr_get)
  changed <- which(!mapply(identical, pre, post))
  for (k in changed) kr_set(targets[[k]], pre[[k]])
  # A batched rule may use the value inside a batch tracer directly; that use belongs
  # to the tracer, which the vmap around then counts once. With nested vmaps the use
  # moves outward one batch tracer at a time, from the innermost value.
  bt <- list()
  for (t in c(ins, sig$consts)) {
    chain <- list()
    while (is_tracer(t) && identical(t$trace$kind, "batch") &&
           is.environment(t$val) && inherits(t$val, "jax_value") &&
           !any(vapply(bt, identical, NA, t))) {
      chain <- c(list(t), chain)
      t <- t$val
    }
    bt <- c(bt, chain)
  }
  vpre <- lapply(bt, function(t) kr_get(t$val))
  on.exit({
    for (k in changed)
      kr_set(targets[[k]], kr_or(post[[k]] %||% FALSE, kr_get(targets[[k]]) %||% FALSE))
    for (k in seq_along(bt)) {
      after <- kr_get(bt[[k]]$val)
      if (!identical(kr_norm(vpre[[k]] %||% FALSE), FALSE) || is.null(after) || !any(after)) next
      kr_set(bt[[k]]$val, vpre[[k]])
      kr_set(bt[[k]], kr_or(kr_get(bt[[k]]) %||% FALSE, kr_unlift(after)))
    }
  }, add = TRUE)
  expr
}

# Runs `expr` without counting the uses of keys it makes: the recomputation of the
# backward pass of checkpoint, which JAX gives an empty signature (the forward pass has
# checked and counted those uses).
kr_uncounted <- function(expr) {
  if (!kr_on()) return(expr)
  log <- kr_begin()
  old_quiet <- .jx$kr_quiet
  .jx$kr_quiet <- TRUE
  on.exit({
    .jx$kr_log <- log$old
    .jx$kr_quiet <- old_quiet
    for (it in rev(log$items)) kr_set(it$d, it$prev)
  }, add = TRUE)
  expr
}

# Undoes the uses of keys from outside that happened while tracing: a boundary applies
# the signature itself, once per call.
kr_revert <- function(sig) {
  if (is.null(sig)) return(invisible())
  for (i in rev(seq_along(sig$consts))) {
    kr_set(sig$consts[[i]], sig$prevs[[i]])
  }
  invisible()
}

# Port of KeyReuseSignature.check_signature and update_consumption.
# The internal jit functions of random (prng_jit) report the random function by its name.
# `offset` is the number of operands before `ins` (the predicate of cond).
kr_apply_in <- function(sig, ins, name, offset = 0L) {
  if (is.null(sig) || !kr_on()) return(invisible())
  n <- length(ins) + offset
  nm <- function(i) if (isTRUE(.jx$kr_prim) && nzchar(sig$names[i] %||% "")) sig$names[i] else name
  for (i in seq_along(sig$sinks))
    if (!is.null(sig$sinks[[i]]) && kr_overlap(kr_state(ins[[i]]), sig$sinks[[i]]))
      kr_reuse_error(nm(i), i - 1L + offset, ins[[i]])
  for (k in seq_along(sig$consts))
    if (kr_overlap(kr_state(sig$consts[[k]]), TRUE)) kr_reuse_error(name, n + k - 1L, sig$consts[[k]])
  for (i in seq_along(sig$sinks))
    if (!is.null(sig$sinks[[i]])) kr_consume(ins[[i]], sig$sinks[[i]], nm(i), i - 1L + offset)
  for (k in seq_along(sig$consts)) kr_consume(sig$consts[[k]], TRUE, name, n + k - 1L)
  invisible()
}

kr_apply_out <- function(sig, ins, outs) {
  if (is.null(sig) || !kr_on()) return(invisible())
  for (j in seq_along(sig$out)) {
    o <- sig$out[[j]]
    v <- outs[[j]]
    if (is.null(o) || !is.environment(v) || !inherits(v, "jax_value")) next
    # The uses a replay recorded on the output (a forward rule consuming a key it
    # returns) are kept.
    kr_set(v, kr_or(kr_state(v), if (!is.null(o$fwd)) kr_state(ins[[o$fwd]])
                                 else if (!is.null(o$const)) kr_state(o$const) else o$state %||% FALSE))
  }
  invisible()
}

kr_apply <- function(sig, ins, outs, name) {
  kr_apply_in(sig, ins, name)
  kr_apply_out(sig, ins, outs)
}

# Port of _cond_key_type_signature: a key is consumed if a branch consumes it, an output
# is forwarded if every branch forwards it, and otherwise new where every branch that
# makes it new does.
kr_combine_branches <- function(sigs) {
  sigs <- Filter(Negate(is.null), sigs)
  if (!length(sigs)) return(NULL)
  n_in <- length(sigs[[1L]]$sinks)
  sinks <- lapply(seq_len(n_in), function(i) {
    ms <- Filter(Negate(is.null), lapply(sigs, function(s) s$sinks[[i]]))
    if (length(ms)) Reduce(kr_or, ms)
  })
  out <- lapply(seq_along(sigs[[1L]]$out), function(j) {
    os <- lapply(sigs, function(s) s$out[[j]])
    if (all(vapply(os, is.null, NA))) return(NULL)
    fwd <- vapply(os, function(o) if (is.null(o$fwd)) NA_integer_ else o$fwd, 1L)
    if (!anyNA(fwd) && length(unique(fwd)) == 1L) return(list(fwd = fwd[1L]))
    cs <- lapply(os, `[[`, "const")
    if (!any(vapply(cs, is.null, NA)) && all(vapply(cs, identical, NA, cs[[1L]])))
      return(list(const = cs[[1L]]))
    # A closed over key that a branch returns counts with its state at the trace.
    os <- lapply(os, function(o) if (!is.null(o$const)) list(state = kr_state(o$const)) else o)
    # Otherwise the output is a Source: consumed where a branch that makes it consumed it
    # (the sources of the branches are combined, the forwards dropped).
    src <- os[is.na(fwd) & !vapply(os, is.null, NA)]
    if (!length(src)) return(list(state = TRUE))
    list(state = Reduce(kr_or, lapply(src, function(o) o$state %||% FALSE)))
  })
  consts <- list()
  prevs <- list()
  for (s in sigs) for (k in seq_along(s$consts)) {
    if (any(vapply(consts, identical, NA, s$consts[[k]]))) next
    consts[[length(consts) + 1L]] <- s$consts[[k]]
    prevs[length(prevs) + 1L] <- list(s$prevs[[k]])
  }
  list(sinks = sinks, out = out, consts = consts, prevs = prevs,
       names = vapply(seq_len(n_in), function(i) {
         nms <- vapply(sigs, function(s) s$names[i] %||% "", "")
         c(nms[nzchar(nms)], "")[1L]
       }, ""))
}

# Whether output j is a new key that is not entirely consumed (a Source).
kr_is_source <- function(sig, j) {
  o <- sig$out[[j]]
  !is.null(o) && is.null(o$fwd) && !all(o$state)
}

kr_loop_error <- function(...) if (!kr_quiet()) jax_error(..., class = "jax_key_reuse_error")

# Port of _scan_key_type_signature.
kr_check_scan <- function(sig, n_carry, n_consts = 0L) {
  if (is.null(sig)) return(invisible())
  if (length(sig$consts) || any(!vapply(sig$sinks[seq_len(n_consts)], is.null, NA)))
    kr_loop_error("scan body function leads to key reuse when repeatedly executed, because ",
                  "key constants are repeatedly consumed")
  for (i in n_consts + seq_len(n_carry))
    if (!is.null(sig$sinks[[i]]) && !kr_is_source(sig, i))
      kr_loop_error("scan body function leads to key reuse when repeatedly executed, because ",
                    "consumed inputs don't match sourced outputs")
  invisible()
}

# Port of _while_key_type_signature.
kr_check_while <- function(cond_sig, body_sig) {
  if (is.null(cond_sig) || is.null(body_sig)) return(invisible())
  if (length(cond_sig$consts))
    kr_loop_error("while_loop cond function leads to key reuse when repeatedly executed")
  if (length(body_sig$consts))
    kr_loop_error("while_loop body function leads to key reuse when repeatedly executed")
  n <- length(body_sig$sinks)
  cs <- which(!vapply(cond_sig$sinks, is.null, NA))
  bs <- which(!vapply(body_sig$sinks, is.null, NA))
  src <- which(vapply(seq_len(n), function(j) kr_is_source(body_sig, j), NA))
  if (!all(cs %in% src))
    kr_loop_error("while_loop cond function leads to key reuse when repeatedly executed")
  if (!all(bs %in% src))
    kr_loop_error("while_loop body function leads to key reuse when repeatedly executed")
  if (length(intersect(cs, bs)))
    kr_loop_error("while_loop cond and body functions both use the same key")
  invisible()
}

# The mask of one element of a batch, lifted across the mapped axis `ax` (0 based) of a
# batch of `size`. Only a leading mapped axis keeps the mask; JAX consumes the whole
# array otherwise (the batched function moves the axis first).
# The signature of a scan body for the stacked operands and outputs of the scan: the
# masks of the scanned inputs (from position `first_x`) and of the stacked outputs (from
# position `first_y`) are lifted across the `len` iterations.
kr_scan_sig <- function(sig, first_x, first_y, len) {
  if (is.null(sig)) return(sig)
  lift <- function(m) if (is.logical(m) && length(m) == 1L && is.null(dim(m))) m else kr_lift(m, len, 0L)
  for (i in seq_along(sig$sinks))
    if (i >= first_x && !is.null(sig$sinks[[i]])) sig$sinks[[i]] <- lift(sig$sinks[[i]])
  for (j in seq_along(sig$out))
    if (j >= first_y && !is.null(sig$out[[j]]) && !is.null(sig$out[[j]]$state))
      sig$out[[j]]$state <- lift(sig$out[[j]]$state)
  sig
}

kr_lift <- function(m, size, ax) {
  if (isTRUE(m) || !identical(as.integer(ax), 0L)) return(TRUE)
  inner <- dim(m) %||% length(m)
  arr <- array(rep(as.vector(m), times = size), c(inner, size))
  aperm(arr, append(seq_along(inner), length(inner) + 1L, after = ax))
}

# The mask of one element from the mask of a batch whose leading axis is the batch.
kr_unlift <- function(m) {
  if (isTRUE(m) || is.null(dim(m)) || length(dim(m)) < 2L) return(kr_norm(any(m)))
  kr_norm(apply(m, seq_along(dim(m))[-1L], any))
}

# vmap: the mask an input key is consumed with is lifted across its mapped axis (`axes`,
# 0 based, NA for an unmapped input). An output gets the lifted mask of its tracer
# (`out_axes`, the axes the outputs were moved to), together with the state of the
# value inside the tracer when the output is that value.
kr_vmap <- function(ins, leaves, outs, res, axes, out_axes = NULL) {
  if (!kr_on()) return(invisible())
  for (i in seq_along(ins)) {
    t <- ins[[i]]
    if (identical(t, leaves[[i]]) || !is.environment(t)) next
    m <- kr_get(t)
    if (is.null(m) || !any(m)) next
    ax <- axes[[i]]
    full <- if (isTRUE(m) || is.na(ax)) TRUE
            else kr_lift(m, aval_of(leaves[[i]])$shape[ax + 1L], ax)
    kr_consume(leaves[[i]], full, "vmap", i - 1L)
  }
  for (j in seq_along(outs)) {
    o <- outs[[j]]
    r <- res[[j]]
    if (identical(o, r) || !is.environment(r) || !inherits(r, "jax_value")) next
    m <- kr_get(o)
    batch <- is_tracer(o) && identical(o$trace$kind, "batch")
    v <- if (batch) kr_get(o$val)
    if (is.null(m) && is.null(v)) next
    i <- which(vapply(ins, identical, NA, o))
    if (length(i)) {
      kr_set(r, kr_state(leaves[[i[1L]]]))
      next
    }
    ax <- if (is.null(out_axes)) 0L else out_axes[[j]]
    lifted <- if (is.null(m) || !any(m)) FALSE
              else if (batch && !isTRUE(m) && length(aval_of(r)$shape))
                kr_lift(m, aval_of(r)$shape[ax + 1L], ax)
              else TRUE
    inner <- if (is.null(v)) FALSE else if (identical(r, o$val)) v else kr_norm(any(v))
    kr_set(r, kr_or(inner, lifted))
  }
  invisible()
}

# The batching rule of a custom derivative function runs the function (or a rule) on
# the values inside the batch tracers `args`; the keys it consumes there are moved from
# those values onto the tracers, so that the vmap around counts them once, also when a
# jit replay has applied its signature to the tracers. `before` are the states of the
# values before the run.
kr_to_tracers <- function(args, vals, before) {
  if (!kr_on()) return(invisible())
  for (i in seq_along(args)) {
    a <- args[[i]]
    v <- vals[[i]]
    if (identical(a, v) || !is_tracer(a) || !identical(a$trace$kind, "batch")) next
    if (!is.environment(v) || !inherits(v, "jax_value")) next
    if (!identical(kr_norm(before[[i]] %||% FALSE), FALSE)) next
    after <- kr_get(v)
    if (is.null(after) || !any(after)) next
    kr_set(v, before[[i]])
    kr_consume(a, kr_unlift(after), "vmap", i - 1L)
  }
  invisible()
}

# A copy of key data in a new environment, so that a new key has its own state (also
# with the check off, since it can be turned on later).
kr_copy <- function(data) {
  if (is_tracer(data)) return(bind(copy_p, list(data), list()))
  new_array(data$aval, buf = data$buf, host = data$host)
}

# Static slicing of a key array consumes the selected keys and returns new ones
# (_slice_signature); indexing with traced indices forwards the state, and so does an
# index that only adds axes (broadcast_in_dim forwards in JAX). The positions are found
# outside any trace, so static indices inside jit count as static.
kr_slice <- function(x, out, ...) {
  if (!kr_on() || identical(out$data, x$data)) return(out)
  s <- prng_keys_shape(x)
  n <- max(prod(s), 1L)
  pos <- tryCatch(with_trace(eval_trace(), {
    ids <- jnp_reshape(jnp_arange(n, dtype = "int32"), if (length(s)) s else integer(0))
    as.integer(c_ravel(as.array(ids[...]))) + 1L
  }), error = function(e) NULL)
  fwd <- function(state) {
    kr_set(out$data, state)
    if (is_tracer(out$data)) kr_set(out$data, x$data, "kr_alias")
    out
  }
  if (is.null(pos)) return(fwd(any(kr_state(x$data))))
  if (length(pos) == n && identical(pos, seq_len(n)) && length(prng_keys_shape(out)) > length(s))
    return(fwd(kr_state(x$data)))
  # Indexing with integer or boolean arrays is a gather, which forwards in JAX.
  idx <- tryCatch(list(...), error = function(e) list())
  if (any(vapply(idx, function(i) is_jax_value(i) || is.logical(i) || (is.numeric(i) && length(i) > 1L), NA)))
    return(fwd(any(kr_state(x$data))))
  # The state is an R array of the key shape; pos are row major positions.
  mv <- logical(n)
  mv[pos] <- TRUE
  mask <- if (length(s) > 1L) aperm(array(mv, rev(s))) else array(mv, max(n, 1L))
  kr_consume(x$data, kr_norm(mask), "slice", 0L)
  kr_set(out$data, FALSE)
  out
}

# Internal ports of consume, assert_consumed and assert_unconsumed.
kr_consume_key <- function(key) {
  kr_sink(key, "consume")
  key
}

kr_assert_consumed <- function(key, value = TRUE) {
  st <- kr_state(key$data)
  ok <- if (length(st) == 1L && length(value) == 1L) identical(as.logical(st), as.logical(value))
        else all(as.vector(st) == as.vector(value))
  if (!ok) {
    if (all(value)) jax_error("Expected key to be consumed", class = "jax_assertion_error")
    if (!any(value)) jax_error("Expected key to not be consumed", class = "jax_assertion_error")
    jax_error("Expected ", paste(value, collapse = " "), ", got ", paste(st, collapse = " "),
              class = "jax_assertion_error")
  }
  invisible(key)
}

kr_assert_unconsumed <- function(key) kr_assert_consumed(key, FALSE)
