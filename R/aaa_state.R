# Global mutable state of the package. The namespace is locked after loading, so all
# mutable state lives inside this environment.
.jx <- new.env(parent = emptyenv())
.jx$id <- 0L
.jx$trace <- NULL
.jx$fallback_hits <- new.env(parent = emptyenv())
.jx$user_contexts <- list()
.jx$live <- new.env(parent = emptyenv())
.jx$live_n <- 0L
.jx$track_live <- FALSE

next_id <- function() {
  .jx$id <- .jx$id + 1L
  .jx$id
}

`%||%` <- function(a, b) if (is.null(a)) b else a

jax_error <- function(..., class = NULL) {
  stop(structure(class = c(class, "jax_error", "error", "condition"),
                 list(message = paste0(...), call = NULL)))
}

prod_int <- function(x) if (length(x)) as.integer(prod(x)) else 1L

# x without its first n elements (x[-seq_len(n)] is empty when n is 0).
drop_first <- function(x, n) if (n <= 0L) x else x[-seq_len(n)]

# Replace named entries of a parameter list (modifyList would merge nested lists).
update_params <- function(params, new) {
  for (n in names(new)) params[n] <- list(new[[n]])
  params
}
