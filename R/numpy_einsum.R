# jnp.einsum and jnp.einsum_path (port of jax/_src/numpy/einsum.py, with the parts of
# opt_einsum's contract_path, parser and path optimizers that JAX relies on) and the
# tensor contractions of jax/_src/numpy/tensor_contractions.py that are not in
# numpy_core.R (tensordot, matvec, vecmat, vecdot). Private helpers use the prefix `es_`.
#
# Labels are handled as character vectors of single symbols. Path optimizers: "optimal"
# (opt_einsum.paths.optimal) and "greedy" (opt_einsum.paths.ssa_greedy_optimize) are
# ported; "auto" uses optimal up to four operands like opt_einsum and greedy beyond.
# ponytail: the branch and bound and dynamic programming optimizers ("branch-all",
# "branch-2", "branch-1", "dp", and "auto" for 5 to 14 operands) fall back to greedy;
# values are the same, only the contraction order (and cost) can differ.

es_symbol <- function(i) {
  # Port of opt_einsum.parser.get_symbol (0-based i).
  if (i < 52L) c(letters, LETTERS)[i + 1L]
  else if (i >= 55296L) intToUtf8(i + 2048L)
  else intToUtf8(i + 140L)
}

es_chars <- function(s) if (!nzchar(s)) character(0) else strsplit(s, "")[[1L]]
es_sort <- function(x) sort(unique(x), method = "radix")
es_str <- function(x) paste(x, collapse = "")
es_size <- function(idx, size_dict) prod(as.numeric(size_dict[idx]))

es_flop_count <- function(idx, inner, num_terms, size_dict) {
  es_size(idx, size_dict) * (max(1L, num_terms - 1L) + as.integer(inner))
}

# Port of opt_einsum.parser.convert_interleaved_input.
es_convert_interleaved <- function(ops) {
  n <- length(ops) %/% 2L
  operands <- ops[2L * seq_len(n) - 1L]
  subs <- ops[2L * seq_len(n)]
  out_sub <- if (length(ops) %% 2L) ops[[length(ops)]] else NULL
  items <- function(s) if (is.list(s)) s else as.list(s)
  syms <- unlist(lapply(c(subs, if (!is.null(out_sub)) list(out_sub)), function(s)
    Filter(function(e) !inherits(e, "jax_ellipsis"), items(s))))
  if (any(vapply(syms, function(e) !(is.numeric(e) || is.character(e)) || length(e) != 1L, NA)))
    jax_error("For this input type lists must contain either Ellipsis or hashable and comparable ",
              "object (e.g. int, str).", class = "jax_type_error")
  keys <- unique(unlist(syms))
  keys <- if (is.numeric(keys)) sort(keys) else sort(keys, method = "radix")
  map <- vapply(seq_along(keys) - 1L, es_symbol, "")
  conv <- function(s) paste(vapply(items(s), function(e) {
    if (inherits(e, "jax_ellipsis")) "..." else map[match(e, keys)]
  }, ""), collapse = "")
  subscripts <- paste(vapply(subs, conv, ""), collapse = ",")
  if (!is.null(out_sub)) subscripts <- paste0(subscripts, "->", conv(out_sub))
  list(subscripts = subscripts, operands = operands)
}

es_find_output_str <- function(subscripts) {
  ch <- es_chars(gsub(",", "", subscripts, fixed = TRUE))
  u <- es_sort(ch)
  u[vapply(u, function(s) sum(ch == s) == 1L, NA)]
}

es_count <- function(s, pat) {
  m <- gregexpr(pat, s, fixed = TRUE)[[1L]]
  if (m[1L] == -1L) 0L else length(m)
}

# Comma separated subscripts; a trailing empty one (a scalar operand) is kept, which
# strsplit alone drops.
es_split <- function(x) strsplit(paste0(x, ","), ",", fixed = TRUE)[[1L]]

# Port of opt_einsum.parser.parse_einsum_input: returns list(inputs, output, operands),
# with inputs a list of symbol vectors and output a symbol vector.
es_parse <- function(ops) {
  if (!length(ops)) jax_error("No input operands", class = "jax_value_error")
  if (is.character(ops[[1L]])) {
    subscripts <- gsub(" ", "", ops[[1L]], fixed = TRUE)
    operands <- ops[-1L]
  } else {
    ci <- es_convert_interleaved(ops)
    subscripts <- ci$subscripts
    operands <- ci$operands
  }
  operands <- lapply(operands, as_jax)
  shapes <- lapply(operands, function(o) aval_of(o)$shape)
  if (grepl("-", subscripts, fixed = TRUE) || grepl(">", subscripts, fixed = TRUE)) {
    if (es_count(subscripts, "-") > 1L || es_count(subscripts, ">") > 1L ||
        es_count(subscripts, "->") != 1L)
      jax_error("Subscripts can only contain one '->'.", class = "jax_value_error")
  }
  if (grepl(".", subscripts, fixed = TRUE)) {
    used <- es_chars(gsub("->", "", gsub("[.,]", "", subscripts), fixed = TRUE))
    nmax <- max(c(0L, vapply(shapes, length, 1L)))
    ell <- character(0)
    i <- 0L
    while (length(ell) < nmax) {
      s <- es_symbol(i); i <- i + 1L
      if (!s %in% used) ell <- c(ell, s)
    }
    longest <- 0L
    has_out <- grepl("->", subscripts, fixed = TRUE)
    if (has_out) {
      parts <- strsplit(subscripts, "->", fixed = TRUE)[[1L]]
      split_subs <- es_split(parts[1L])
      out_sub <- if (length(parts) > 1L) parts[2L] else ""
    } else {
      split_subs <- es_split(subscripts)
    }
    for (num in seq_along(split_subs)) {
      sub <- split_subs[num]
      if (!grepl(".", sub, fixed = TRUE)) next
      if (es_count(sub, ".") != 3L || es_count(sub, "...") != 1L)
        jax_error("Invalid Ellipses.", class = "jax_value_error")
      sh <- shapes[[num]]
      cnt <- if (!length(sh)) 0L else max(length(sh), 1L) - (length(es_chars(sub)) - 3L)
      if (cnt > longest) longest <- cnt
      if (cnt < 0L) jax_error("Ellipses lengths do not match.", class = "jax_value_error")
      rep_s <- if (cnt == 0L) "" else es_str(utils::tail(ell, cnt))
      split_subs[num] <- sub("...", rep_s, sub, fixed = TRUE)
    }
    subscripts <- paste(split_subs, collapse = ",")
    out_ell <- if (longest == 0L) "" else es_str(utils::tail(ell, longest))
    if (has_out) {
      subscripts <- paste0(subscripts, "->", sub("...", out_ell, out_sub, fixed = TRUE))
    } else {
      normal <- setdiff(es_find_output_str(subscripts), es_chars(out_ell))
      subscripts <- paste0(subscripts, "->", out_ell, es_str(es_sort(normal)))
    }
  }
  if (grepl("->", subscripts, fixed = TRUE)) {
    parts <- strsplit(subscripts, "->", fixed = TRUE)[[1L]]
    input_subs <- parts[1L]
    output <- if (length(parts) > 1L) es_chars(parts[2L]) else character(0)
  } else {
    input_subs <- subscripts
    output <- es_find_output_str(subscripts)
  }
  in_chars <- es_chars(input_subs)
  for (ch in output) {
    if (sum(output == ch) != 1L)
      jax_error("Output character '", ch, "' appeared more than once in the output.",
                class = "jax_value_error")
    if (!ch %in% in_chars)
      jax_error("Output character '", ch, "' did not appear in the input", class = "jax_value_error")
  }
  terms <- strsplit(paste0(input_subs, " "), ",", fixed = TRUE)[[1L]]
  terms[length(terms)] <- sub(" $", "", terms[length(terms)])
  if (length(terms) != length(operands))
    jax_error("Number of einsum subscripts, ", length(terms), ", must be equal to the number of ",
              "operands, ", length(operands), ".", class = "jax_value_error")
  list(inputs = lapply(terms, es_chars), output = output, operands = operands, shapes = shapes)
}

# ---- path optimizers (ports of opt_einsum.paths) -----------------------------------------------

es_ssa_to_linear <- function(ssa_path) {
  n <- sum(lengths(ssa_path)) - length(ssa_path) + 1L
  ids <- seq_len(n) - 1L
  ssa <- n
  path <- vector("list", length(ssa_path))
  for (k in seq_along(ssa_path)) {
    con <- sort(match(ssa_path[[k]], ids) - 1L)
    ids <- c(ids[-(con + 1L)], ssa)
    path[[k]] <- con
    ssa <- ssa + 1L
  }
  path
}

es_optimal <- function(inputs, output, size_dict) {
  # Port of opt_einsum.paths.optimal (no memory limit; JAX never sets one).
  inputs <- lapply(inputs, es_sort)
  best <- new.env(parent = emptyenv())
  best$flops <- Inf
  best$path <- list(seq_along(inputs) - 1L)
  cache <- new.env(parent = emptyenv())
  iterate <- function(path, inputs, remaining, flops) {
    if (length(remaining) == 1L) {
      best$flops <- flops
      best$path <- path
      return(invisible())
    }
    rem <- sort(remaining)
    for (a in seq_len(length(rem) - 1L)) for (b in seq(a + 1L, length(rem))) {
      i <- rem[a]; j <- rem[b]
      key <- paste(es_str(inputs[[i + 1L]]), es_str(inputs[[j + 1L]]), sep = "\001")
      hit <- cache[[key]]
      if (is.null(hit)) {
        k1 <- inputs[[i + 1L]]; k2 <- inputs[[j + 1L]]
        either <- es_sort(c(k1, k2))
        shared <- intersect(k1, k2)
        keep <- unique(c(output, unlist(inputs[setdiff(remaining, c(i, j)) + 1L])))
        hit <- list(k12 = intersect(either, keep),
                    flops = es_flop_count(either, length(setdiff(shared, keep)) > 0L, 2L, size_dict))
        cache[[key]] <- hit
      }
      new_flops <- flops + hit$flops
      if (new_flops >= best$flops) next
      iterate(c(path, list(c(i, j))), c(inputs, list(hit$k12)),
              c(setdiff(remaining, c(i, j)), length(inputs)), new_flops)
    }
  }
  iterate(list(), inputs, seq_along(inputs) - 1L, 0)
  es_ssa_to_linear(best$path)
}

es_greedy <- function(inputs, output, sizes) {
  # Port of opt_einsum.paths.ssa_greedy_optimize with the default "memory-removed" cost.
  if (length(inputs) == 1L) return(list(0L))
  inputs <- lapply(inputs, es_sort)
  output <- es_sort(c(output, Reduce(intersect, inputs)))
  # Keys are prefixed so that the empty index set of a scalar operand is a valid name.
  ks <- function(k) paste0("k", paste(k, collapse = "\001"))
  keyset <- list()
  remaining <- list()
  next_id <- length(inputs)
  new_id <- function() { id <- next_id; next_id <<- next_id + 1L; id }
  ssa_path <- list()
  for (ssa_id in seq_along(inputs) - 1L) {
    key <- inputs[[ssa_id + 1L]]
    s <- ks(key)
    keyset[[s]] <- key
    if (!is.null(remaining[[s]])) {
      ssa_path <- c(ssa_path, list(c(remaining[[s]], ssa_id)))
      remaining[[s]] <- new_id()
    } else {
      remaining[[s]] <- ssa_id
    }
  }
  dim_to_keys <- list()
  for (s in names(remaining)) for (d in setdiff(keyset[[s]], output))
    dim_to_keys[[d]] <- union(dim_to_keys[[d]], s)
  refc <- list(c2 = character(0), c3 = character(0))
  for (d in names(dim_to_keys)) {
    n <- length(dim_to_keys[[d]])
    if (n >= 2L && !d %in% output) refc$c2 <- c(refc$c2, d)
    if (n >= 3L && !d %in% output) refc$c3 <- c(refc$c3, d)
  }
  footprints <- list()
  for (s in names(remaining)) footprints[[s]] <- es_size(keyset[[s]], sizes)
  candidate <- function(k1, k2) {
    a <- keyset[[k1]]; b <- keyset[[k2]]
    either <- union(a, b); two <- intersect(a, b); one <- setdiff(either, two)
    k12 <- es_sort(c(intersect(either, output), intersect(two, refc$c3), intersect(one, refc$c2)))
    cost <- es_size(k12, sizes) - footprints[[k1]] - footprints[[k2]]
    id1 <- remaining[[k1]]; id2 <- remaining[[k2]]
    if (id1 > id2) { t <- k1; k1 <- k2; k2 <- t; t <- id1; id1 <- id2; id2 <- t }
    list(cost = c(cost, id2, id1), k1 = k1, k2 = k2, k12 = k12)
  }
  better <- function(x, y) {
    for (q in 1:3) if (x$cost[q] != y$cost[q]) return(x$cost[q] < y$cost[q])
    FALSE
  }
  queue <- list()
  push_min <- function(k1, k2s) {
    cands <- lapply(k2s, function(k2) candidate(k1, k2))
    m <- cands[[1L]]
    for (cd in cands[-1L]) if (better(cd, m)) m <- cd
    queue[[length(queue) + 1L]] <<- m
  }
  for (d in names(dim_to_keys)) {
    dk <- dim_to_keys[[d]]
    dk <- dk[order(vapply(dk, function(s) remaining[[s]], 1))]
    for (i in seq_len(length(dk) - 1L)) push_min(dk[i], dk[(i + 1L):length(dk)])
  }
  while (length(queue)) {
    m <- 1L
    for (q in seq_along(queue)[-1L]) if (better(queue[[q]], queue[[m]])) m <- q
    con <- queue[[m]]
    queue[[m]] <- NULL
    if (is.null(remaining[[con$k1]]) || is.null(remaining[[con$k2]])) next
    k1 <- con$k1; k2 <- con$k2
    k12 <- con$k12
    s12 <- ks(k12)
    keyset[[s12]] <- k12
    ssa_id1 <- remaining[[k1]]; remaining[[k1]] <- NULL
    ssa_id2 <- remaining[[k2]]; remaining[[k2]] <- NULL
    for (d in setdiff(keyset[[k1]], output)) dim_to_keys[[d]] <- setdiff(dim_to_keys[[d]], k1)
    for (d in setdiff(keyset[[k2]], output)) dim_to_keys[[d]] <- setdiff(dim_to_keys[[d]], k2)
    ssa_path <- c(ssa_path, list(c(ssa_id1, ssa_id2)))
    if (!is.null(remaining[[s12]])) {
      ssa_path <- c(ssa_path, list(c(remaining[[s12]], new_id())))
    } else {
      for (d in setdiff(k12, output)) dim_to_keys[[d]] <- union(dim_to_keys[[d]], s12)
    }
    remaining[[s12]] <- new_id()
    for (d in union(keyset[[k1]], setdiff(keyset[[k2]], output))) {
      n <- length(dim_to_keys[[d]])
      if (n <= 1L) {
        refc$c2 <- setdiff(refc$c2, d); refc$c3 <- setdiff(refc$c3, d)
      } else if (n == 2L) {
        refc$c2 <- union(refc$c2, d); refc$c3 <- setdiff(refc$c3, d)
      } else {
        refc$c2 <- union(refc$c2, d); refc$c3 <- union(refc$c3, d)
      }
    }
    footprints[[s12]] <- es_size(k12, sizes)
    k2s <- setdiff(unique(unlist(lapply(k12, function(d) dim_to_keys[[d]]))), s12)
    if (length(k2s)) push_min(s12, k2s)
  }
  fq <- lapply(names(remaining), function(s)
    list(size = es_size(intersect(keyset[[s]], output), sizes), id = remaining[[s]], key = keyset[[s]]))
  pop <- function() {
    o <- order(vapply(fq, `[[`, 1, "size"), vapply(fq, `[[`, 1, "id"))[1L]
    it <- fq[[o]]
    fq[[o]] <<- NULL
    it
  }
  cur <- pop()
  while (length(fq)) {
    nx <- pop()
    ssa_path <- c(ssa_path, list(c(min(cur$id, nx$id), max(cur$id, nx$id))))
    k12 <- intersect(union(cur$key, nx$key), output)
    fq[[length(fq) + 1L]] <- list(size = es_size(k12, sizes), id = new_id(), key = k12)
    cur <- pop()
  }
  es_ssa_to_linear(ssa_path)
}

es_path_fn <- function(name, n) {
  opts <- c("auto", "auto-hq", "optimal", "branch-all", "branch-2", "branch-1", "greedy", "eager",
            "opportunistic", "dp", "dynamic-programming")
  name <- tolower(name)
  if (!name %in% opts)
    jax_error("Path optimizer '", name, "' not found, valid options are {'",
              paste(opts, collapse = "', '"), "'}.", class = "jax_key_error")
  if (name == "optimal" || (name == "auto" && n <= 4L) || (name == "auto-hq" && n <= 5L))
    return(es_optimal)
  es_greedy
}

# ---- blas classification (port of opt_einsum.blas.can_blas, for einsum_path) --------------------

es_can_blas <- function(inputs, result, removed, shapes) {
  if (length(inputs) != 2L) return(FALSE)
  l <- inputs[[1L]]; r <- inputs[[2L]]
  for (ch in unique(c(l, r))) {
    nl <- sum(l == ch); nr <- sum(r == ch)
    if (nl > 1L || nr > 1L || nl + nr > 2L) return(FALSE)
    if (nl + nr - 1L == as.integer(ch %in% result)) return(FALSE)
  }
  for (ch in removed)
    if (shapes[[1L]][match(ch, l)] != shapes[[2L]][match(ch, r)]) return(FALSE)
  if (!length(removed)) return("OUTER/EINSUM")
  keep_l <- setdiff(l, removed); keep_r <- setdiff(r, removed)
  rs <- length(removed)
  same <- function(a, b) identical(a, b)
  if (same(l, r)) return("DOT")
  if (setequal(l, r)) return("DOT/EINSUM")
  tl <- utils::tail(l, rs); hl <- utils::head(l, rs)
  tr <- utils::tail(r, rs); hr <- utils::head(r, rs)
  if (same(tl, hr) || same(hl, tr) || same(tl, tr) || same(hl, hr)) return("GEMM")
  if (!length(keep_l) || !length(keep_r)) return("GEMV/EINSUM")
  "TDOT"
}

# ---- contract_path ---------------------------------------------------------------------------

# Port of opt_einsum.contract.contract_path (with einsum_call and use_blas).
es_contract_path <- function(ops, optimize) {
  p <- es_parse(ops)
  input_list <- p$inputs
  output <- p$output
  input_shapes <- p$shapes
  size_dict <- integer(0)
  for (t in seq_along(input_list)) {
    term <- input_list[[t]]
    sh <- input_shapes[[t]]
    if (length(sh) != length(term))
      jax_error("Einstein sum subscript '", es_str(term), "' does not contain the correct number ",
                "of indices for operand ", t - 1L, ".", class = "jax_value_error")
    for (k in seq_along(term)) {
      ch <- term[k]
      d <- sh[k]
      if (!is.na(size_dict[ch])) {
        if (size_dict[ch] == 1L) size_dict[ch] <- d
        else if (!d %in% c(1L, size_dict[ch]))
          jax_error("Size of label '", ch, "' for operand ", t - 1L, " (", size_dict[ch],
                    ") does not match previous terms (", d, ").", class = "jax_value_error")
      } else {
        size_dict[ch] <- d
      }
    }
  }
  indices <- es_sort(unlist(input_list))
  num_ops <- length(input_list)
  input_sets <- lapply(input_list, es_sort)
  inner <- (sum(lengths(input_sets)) - length(indices)) > 0L
  naive_cost <- es_flop_count(indices, inner, num_ops, size_dict)
  if (is.null(optimize)) optimize <- "auto"
  path <- if (isFALSE(optimize)) {
    if (num_ops <= 2L) list(seq_len(num_ops) - 1L) else rep(list(c(0L, 1L)), num_ops - 1L)
  } else if (isTRUE(optimize)) {
    if (num_ops <= 2L) list(seq_len(num_ops) - 1L) else es_optimal(input_sets, output, size_dict)
  } else if (is.list(optimize)) {
    lapply(optimize, as.integer)
  } else if (is.character(optimize) && length(optimize) == 1L) {
    if (num_ops <= 2L) list(seq_len(num_ops) - 1L)
    else es_path_fn(optimize, num_ops)(input_sets, output, size_dict)
  } else {
    jax_error("optimize must be TRUE, FALSE, a path optimizer name or a list path",
              class = "jax_type_error")
  }
  contractions <- vector("list", length(path))
  scale_list <- numeric(0); size_list <- numeric(0); cost_list <- numeric(0)
  sets <- input_sets
  for (cnum in seq_along(path)) {
    inds <- sort(as.integer(path[[cnum]]), decreasing = TRUE)
    if (any(inds < 0L | inds >= length(input_list)))
      jax_error("einsum path step ", a2_fmt(path[[cnum]]), " is out of range", class = "jax_value_error")
    contract <- unique(unlist(sets[inds + 1L]))
    rest <- sets[-(inds + 1L)]
    idx_remain <- unique(c(output, unlist(rest)))
    out_inds <- intersect(contract, idx_remain)
    removed <- es_sort(setdiff(contract, out_inds))
    sets <- c(rest, list(es_sort(out_inds)))
    cost_list <- c(cost_list, es_flop_count(contract, length(removed) > 0L, length(inds), size_dict))
    scale_list <- c(scale_list, length(contract))
    size_list <- c(size_list, es_size(out_inds, size_dict))
    tmp_inputs <- input_list[inds + 1L]
    tmp_shapes <- input_shapes[inds + 1L]
    input_list <- input_list[-(inds + 1L)]
    input_shapes <- input_shapes[-(inds + 1L)]
    blas <- es_can_blas(tmp_inputs, out_inds, removed, tmp_shapes)
    if (cnum == length(path)) {
      idx_result <- output
    } else {
      all_in <- unlist(tmp_inputs)
      idx_result <- out_inds[order(match(out_inds, all_in))]
    }
    shp <- vapply(idx_result, function(ch) {
      max(unlist(Map(function(term, s) s[term == ch], tmp_inputs, tmp_shapes)))
    }, 1L)
    input_list <- c(input_list, list(idx_result))
    input_shapes <- c(input_shapes, list(as.integer(shp)))
    contractions[[cnum]] <- list(inds = inds, removed = removed, inputs = tmp_inputs,
                                 result = idx_result, remaining = input_list, blas = blas)
  }
  list(operands = p$operands, contractions = contractions, path = path, inputs = p$inputs,
       output = output, indices = indices, naive_cost = naive_cost, cost = sum(cost_list),
       scale_list = scale_list, size_list = size_list, size_dict = size_dict)
}

es_fmt_e <- function(x) {
  if (x == 0) return("0.000e+0")
  e <- floor(log10(abs(x)))
  m <- round(x / 10^e, 3L)
  if (abs(m) >= 10) { e <- e + 1; m <- round(x / 10^e, 3L) }
  sprintf("%.3fe%s%d", m, if (e >= 0) "+" else "-", as.integer(abs(e)))
}

es_einstr <- function(c) paste0(paste(vapply(c$inputs, es_str, ""), collapse = ","), "->", es_str(c$result))

# Port of opt_einsum.contract.PathInfo.__repr__.
es_path_info <- function(cp) {
  eq <- paste0(paste(vapply(cp$inputs, es_str, ""), collapse = ","), "->", es_str(cp$output))
  out <- c(sprintf("  Complete contraction:  %s\n", eq),
           sprintf("         Naive scaling:  %d\n", length(cp$indices)),
           sprintf("     Optimized scaling:  %d\n", as.integer(max(c(0, cp$scale_list)))),
           sprintf("      Naive FLOP count:  %s\n", es_fmt_e(cp$naive_cost)),
           sprintf("  Optimized FLOP count:  %s\n", es_fmt_e(cp$cost)),
           sprintf("   Theoretical speedup:  %s\n", es_fmt_e(cp$naive_cost / max(cp$cost, 1))),
           sprintf("  Largest intermediate:  %s elements\n", es_fmt_e(max(c(1, cp$size_list)))),
           paste0(strrep("-", 80L), "\n"),
           sprintf("%6s %11s %22s %37s\n", "scaling", "BLAS", "current", "remaining"),
           strrep("-", 80L))
  for (n in seq_along(cp$contractions)) {
    c <- cp$contractions[[n]]
    es <- es_einstr(c)
    rem <- paste0(paste(vapply(c$remaining, es_str, ""), collapse = ","), "->", es_str(cp$output))
    width <- max(0L, 56L - max(22L, nchar(es)))
    blas <- if (isFALSE(c$blas)) "0" else c$blas
    out <- c(out, sprintf("\n%4d %14s %22s    %*s", as.integer(cp$scale_list[n]), blas, es, width, rem))
  }
  paste(out, collapse = "")
}

# ---- einsum ----------------------------------------------------------------------------------

es_dot_general <- function(lhs, rhs, dn, precision, pet) {
  # JAX's CPU lowering converts both operands to the output dtype when their dtypes differ.
  if (aval_of(lhs)$dtype != aval_of(rhs)$dtype) {
    lhs <- lax_convert_element_type(lhs, pet)
    rhs <- lax_convert_element_type(rhs, pet)
  }
  lax_dot_general(lhs, rhs, dn, precision, pet)
}

# Port of einsum.py:_einsum.
es_einsum_impl <- function(operands, contractions, precision, preferred_element_type) {
  if (is.null(preferred_element_type)) {
    r <- promote_types(vapply(operands, function(o) aval_of(o)$dtype, ""),
                       vapply(operands, function(o) aval_of(o)$weak, NA))
    pet <- r$dtype
    out_weak <- r$weak
  } else {
    pet <- dtype(preferred_element_type)
    out_weak <- FALSE
  }
  red <- function(x, axes) {
    if (result_type(x, pet) != aval_of(x)$dtype) x <- lax_convert_element_type(x, pet)
    if (aval_of(x)$dtype == "bool") lax_reduce_or(x, axes) else lax_reduce_sum(x, axes)
  }
  sum_uniques <- function(operand, names, uniques) {
    if (length(uniques)) {
      operand <- red(operand, match(uniques, names) - 1L)
      names <- names[!names %in% uniques]
    }
    list(operand, names)
  }
  sum_repeats <- function(operand, names, counts, keep_names) {
    for (name in names(counts)) {
      cnt <- counts[[name]]
      if (cnt > 1L) {
        axes <- which(names == name) - 1L
        eye <- a2_delta(aval_of(operand)$shape, axes)
        operand <- lax_select(eye, operand, lax_full_like(operand, 0))
        if (!name %in% keep_names) {
          operand <- red(operand, axes)
          names <- names[names != name]
        } else {
          operand <- red(operand, axes[-length(axes)])
          names <- names[-which(names == name)[seq_len(cnt - 1L)]]
        }
      }
    }
    list(operand, names)
  }
  counter <- function(names) {
    u <- unique(names)
    stats::setNames(vapply(u, function(n) sum(names == n), 1L), u)
  }
  cnt_of <- function(counts, n) if (is.na(counts[n])) 0L else counts[[n]]
  filter_singleton <- function(operand, names, other_shape, other_names) {
    s <- aval_of(operand)$shape
    j <- match(names, other_names)
    keep <- s != 1L | is.na(j) | (!is.na(j) & other_shape[ifelse(is.na(j), 1L, j)] == 1L)
    sq <- which(!keep) - 1L
    list(if (length(sq)) lax_squeeze(operand, sq) else operand, names[keep])
  }
  for (c in contractions) {
    contracted <- c$removed
    result_names <- c$result
    if (length(c$inds) == 1L) {
      operand <- operands[[c$inds + 1L]]
      operands <- operands[-(c$inds + 1L)]
      names <- c$inputs[[1L]]
      counts <- counter(names)
      uniques <- contracted[vapply(contracted, function(n) cnt_of(counts, n) == 1L, NA)]
      r <- sum_uniques(operand, names, uniques)
      r <- sum_repeats(r[[1L]], r[[2L]], counts, result_names)
      operand <- r[[1L]]; names <- r[[2L]]
    } else if (length(c$inds) == 2L) {
      lhs <- operands[[c$inds[1L] + 1L]]
      rhs <- operands[[c$inds[2L] + 1L]]
      operands <- operands[-(c$inds + 1L)]
      lhs_names <- c$inputs[[1L]]; rhs_names <- c$inputs[[2L]]
      r <- filter_singleton(lhs, lhs_names, aval_of(rhs)$shape, rhs_names)
      lhs <- r[[1L]]; lhs_names <- r[[2L]]
      r <- filter_singleton(rhs, rhs_names, aval_of(lhs)$shape, lhs_names)
      rhs <- r[[1L]]; rhs_names <- r[[2L]]
      lc <- counter(lhs_names); rc <- counter(rhs_names)
      lu <- contracted[vapply(contracted, function(n) cnt_of(lc, n) == 1L && cnt_of(rc, n) == 0L, NA)]
      r <- sum_uniques(lhs, lhs_names, lu); lhs <- r[[1L]]; lhs_names <- r[[2L]]
      ru <- contracted[vapply(contracted, function(n) cnt_of(rc, n) == 1L && cnt_of(lc, n) == 0L, NA)]
      r <- sum_uniques(rhs, rhs_names, ru); rhs <- r[[1L]]; rhs_names <- r[[2L]]
      r <- sum_repeats(lhs, lhs_names, lc, c(result_names, rhs_names)); lhs <- r[[1L]]; lhs_names <- r[[2L]]
      r <- sum_repeats(rhs, rhs_names, rc, c(result_names, lhs_names)); rhs <- r[[1L]]; rhs_names <- r[[2L]]
      cn <- contracted[contracted %in% union(lhs_names, rhs_names)]
      batch_names <- result_names[result_names %in% intersect(lhs_names, rhs_names)]
      lb <- match(batch_names, lhs_names) - 1L
      rb <- match(batch_names, rhs_names) - 1L
      ls <- aval_of(lhs)$shape; rs <- aval_of(rhs)$shape
      if (!all(cn %in% lhs_names & cn %in% rhs_names) ||
          any(ls[match(cn, lhs_names)] != rs[match(cn, rhs_names)]))
        jax_error("Incompatible reduction dimensions: lhs.shape=", a2_fmt_shape(ls), " lhs_names=",
                  es_str(lhs_names), " rhs.shape=", a2_fmt_shape(rs), " rhs_names=", es_str(rhs_names),
                  class = "jax_assertion_error")
      lcont <- match(cn, lhs_names) - 1L
      rcont <- match(cn, rhs_names) - 1L
      deleted <- c(batch_names, cn)
      rem_l <- lhs_names[!lhs_names %in% deleted]
      rem_r <- rhs_names[!rhs_names %in% deleted]
      names <- c(batch_names, rem_r, rem_l)
      if (identical(names, result_names)) {
        operand <- es_dot_general(rhs, lhs, list(list(rcont, lcont), list(rb, lb)), precision, pet)
      } else {
        names <- c(batch_names, rem_l, rem_r)
        operand <- es_dot_general(lhs, rhs, list(list(lcont, rcont), list(lb, rb)), precision, pet)
      }
    } else {
      jax_error("jax.numpy.einsum does not support simultaneous contraction of 3 or more operands. ",
                "Typically this means you've passed an unsupported path to the einsum optimize ",
                "parameter.", class = "jax_not_implemented_error")
    }
    if (!identical(names, result_names)) operand <- lax_transpose(operand, match(result_names, names) - 1L)
    operands <- c(operands, list(operand))
  }
  lax_convert_element_type(operands[[1L]], pet, weak_type = out_weak)
}

#' Einstein summation
#'
#' Ports of `jnp.einsum` and `jnp.einsum_path`. Subscripts follow NumPy (`"ij,jk->ik"`,
#' implicit output, `...` broadcasting); the interleaved form
#' `jnp_einsum(x, c(0, 1), y, c(1, 2), c(0, 2))` is also accepted (sublists may be lists
#' containing [ellipsis]). Contractions are planned pairwise as in `opt_einsum`:
#' `optimize = "auto"` (optimal for up to four operands, greedy beyond), `TRUE`
#' (optimal), `"optimal"`, `"greedy"`, `FALSE` (left to right) or an explicit path
#' (a list of 0-based index pairs).
#' @param subscripts Subscript string (or the first operand of the interleaved form).
#' @param ... Operands (and sublists in the interleaved form).
#' @param out Not supported.
#' @param optimize Path optimization (see Details).
#' @param precision Optional precision for the dot products.
#' @param preferred_element_type Optional accumulation and result dtype.
#' @return `jnp_einsum` returns an array. `jnp_einsum_path` returns a list of two
#'   elements: the path (a list of 0-based integer vectors) and a printable summary
#'   string.
#' @export
jnp_einsum <- function(subscripts, ..., out = NULL, optimize = "auto", precision = NULL,
                       preferred_element_type = NULL) {
  # Port of einsum.py:einsum
  if (!is.null(out))
    jax_error("The 'out' argument to jnp.einsum is not supported.", class = "jax_not_implemented_error")
  cp <- es_contract_path(c(list(subscripts), list(...)), optimize)
  es_einsum_impl(cp$operands, cp$contractions, precision, preferred_element_type)
}

#' @rdname jnp_einsum
#' @export
jnp_einsum_path <- function(subscripts, ..., optimize = "auto") {
  # Port of einsum.py:einsum_path
  cp <- es_contract_path(c(list(subscripts), list(...)), optimize)
  list(cp$path, es_path_info(cp))
}

# ---- tensor contractions ---------------------------------------------------------------------

#' Tensor contractions
#'
#' Ports of `jnp.tensordot`, `jnp.matvec`, `jnp.vecmat` and `jnp.vecdot`
#' (`vecmat` and `vecdot` conjugate their first argument). Axes are 0-based.
#' @param a,b,x1,x2 Arrays.
#' @param axes For `jnp_tensordot`: a number of trailing axes of `a` to contract with
#'   leading axes of `b`, or a list of two integer vectors (axes of `a`, axes of `b`).
#' @param axis Axis of `x1` and `x2` to contract.
#' @param precision Optional precision.
#' @param preferred_element_type Optional accumulation and result dtype.
#' @return An array.
#' @name jnp_tensordot
NULL

#' @rdname jnp_tensordot
#' @export
jnp_tensordot <- function(a, b, axes = 2L, precision = NULL, preferred_element_type = NULL) {
  # Port of tensor_contractions.py:tensordot
  a <- as_jax(a); b <- as_jax(b)
  an <- length(aval_of(a)$shape); bn <- length(aval_of(b)$shape)
  if (is.null(preferred_element_type)) {
    r <- promote_types(c(aval_of(a)$dtype, aval_of(b)$dtype), c(aval_of(a)$weak, aval_of(b)$weak))
    pet <- r$dtype; out_weak <- r$weak
  } else {
    pet <- dtype(preferred_element_type); out_weak <- FALSE
  }
  if (is.numeric(axes) && length(axes) == 1L && is.null(dim(axes))) {
    axes <- as.integer(axes)
    if (axes > min(an, bn))
      jax_error("Number of tensordot axes (axes ", axes, ") exceeds input ranks (",
                a2_fmt_shape(aval_of(a)$shape), " and ", a2_fmt_shape(aval_of(b)$shape), ")",
                class = "jax_type_error")
    cd <- list(an - axes + seq_len(axes) - 1L, seq_len(axes) - 1L)
  } else if ((is.list(axes) || is.numeric(axes)) && length(axes) == 2L) {
    ax1 <- as.integer(unlist(axes[[1L]])); ax2 <- as.integer(unlist(axes[[2L]]))
    if (length(ax1) != length(ax2))
      jax_error("tensordot requires axes lists to have equal length, got ", a2_fmt(axes[[1L]]),
                " and ", a2_fmt(axes[[2L]]), ".", class = "jax_type_error")
    cd <- list(norm_axis(ax1, an), norm_axis(ax2, bn))
  } else {
    jax_error("tensordot axes argument must be an int, a pair of ints, or a pair of lists/tuples of ints.",
              class = "jax_type_error")
  }
  out <- es_dot_general(a, b, list(cd, list(integer(0), integer(0))), precision, pet)
  lax_convert_element_type(out, pet, weak_type = out_weak)
}

#' @rdname jnp_tensordot
#' @export
jnp_matvec <- function(x1, x2) {
  # Port of tensor_contractions.py:matvec
  jnp_vectorize(jnp_matmul, signature = "(n,m),(m)->(n)")(as_jax(x1), as_jax(x2))
}

#' @rdname jnp_tensordot
#' @export
jnp_vecmat <- function(x1, x2) {
  # Port of tensor_contractions.py:vecmat
  jnp_vectorize(jnp_matmul, signature = "(n),(n,m)->(m)")(jnp_conjugate(x1), as_jax(x2))
}

#' @rdname jnp_tensordot
#' @export
jnp_vecdot <- function(x1, x2, axis = -1L, precision = NULL, preferred_element_type = NULL) {
  # Port of tensor_contractions.py:vecdot
  x1 <- as_jax(x1); x2 <- as_jax(x2)
  s1 <- aval_of(x1)$shape; s2 <- aval_of(x2)$shape
  ax <- as.integer(axis)
  d1 <- s1[norm_axis(ax, length(s1)) + 1L]; d2 <- s2[norm_axis(ax, length(s2)) + 1L]
  if (d1 != d2)
    jax_error("axes must match; got shapes ", a2_fmt_shape(s1), " and ", a2_fmt_shape(s2),
              " with axis=", ax, class = "jax_value_error")
  x1 <- jnp_moveaxis(x1, ax, -1L)
  x2 <- jnp_moveaxis(x2, ax, -1L)
  vdot <- function(a, b) {
    if (is_complex_dtype(aval_of(a)$dtype)) a <- lax_conj(a)
    jnp_dot(jnp_ravel(a), jnp_ravel(b), precision = precision,
            preferred_element_type = preferred_element_type)
  }
  jnp_vectorize(vdot, signature = "(n),(n)->()")(x1, x2)
}
