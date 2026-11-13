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
    if (length(i)) list(fwd = i[1L]) else list(state = kr_get(o))
  })
  list(sinks = sinks, out = out, consts = consts, prevs = prevs, names = names)
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
    kr_set(v, if (!is.null(o$fwd)) kr_state(ins[[o$fwd]]) else o$state)
  }
  invisible()
}

kr_apply <- function(sig, ins, outs, name) {
  kr_apply_in(sig, ins, name)
  kr_apply_out(sig, ins, outs)
}

# Port of _cond_key_type_signature: a key is consumed if a branch consumes it, an output
# is new if it is new in every branch, and forwarded if every branch forwards it.
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
    if (all(is.na(fwd))) return(list(state = Reduce(kr_or, lapply(os, function(o) o$state %||% FALSE))))
    list(state = TRUE)
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
kr_check_scan <- function(sig, n_carry) {
  if (is.null(sig)) return(invisible())
  if (length(sig$consts))
    kr_loop_error("scan body function leads to key reuse when repeatedly executed, because ",
                  "key constants are repeatedly consumed")
  for (i in seq_len(n_carry))
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

# vmap: an input key consumed by the function is consumed (all of it); an output is
# consumed if any of its elements was.
kr_vmap <- function(ins, leaves, outs, res) {
  if (!kr_on()) return(invisible())
  for (i in seq_along(ins)) {
    t <- ins[[i]]
    if (identical(t, leaves[[i]]) || !is.environment(t)) next
    m <- kr_get(t)
    if (!is.null(m) && any(m)) kr_consume(leaves[[i]], TRUE, "vmap", i - 1L)
  }
  for (j in seq_along(outs)) {
    o <- outs[[j]]
    r <- res[[j]]
    if (is.null(kr_get(o)) || identical(o, r)) next
    if (!is.environment(r) || !inherits(r, "jax_value")) next
    i <- which(vapply(ins, identical, NA, o))
    kr_set(r, if (length(i)) kr_state(leaves[[i[1L]]]) else any(kr_get(o)))
  }
  invisible()
}

# A copy of key data in a new environment, so that a new key has its own state.
kr_copy <- function(data) {
  if (!kr_on()) return(data)
  if (is_tracer(data)) return(bind(copy_p, list(data), list()))
  new_array(data$aval, buf = data$buf, host = data$host)
}

# Static slicing of a key array consumes the selected keys and returns new ones
# (_slice_signature); indexing with traced indices forwards the state.
kr_slice <- function(x, out, ...) {
  if (!kr_on() || identical(out$data, x$data)) return(out)
  s <- prng_keys_shape(x)
  pos <- tryCatch({
    ids <- jnp_reshape(jnp_arange(max(prod(s), 1L), dtype = "int32"), if (length(s)) s else integer(0))
    sel <- ids[...]
    if (is_tracer(sel)) NULL else as.integer(as.array(sel)) + 1L
  }, error = function(e) NULL)
  if (is.null(pos)) {
    kr_set(out$data, any(kr_state(x$data)))
    return(out)
  }
  mask <- array(FALSE, if (length(s)) s else 1L)
  mask[pos] <- TRUE
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
