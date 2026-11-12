# lax primitives of R/lax.R, R/lax_shape.R and R/lax_reduce.R against the wheel
# (fixtures/lax_*.json): values, dtypes, weak types and errors, eagerly and under jit.
# A skipped case is a documented deviation (see vignette("differences")).

L_NEGAXIS <- "known deviation: R lax functions accept negative axes; jax.lax raises"

lax_known <- c(
  "lax.rev/5" = L_NEGAXIS, "lax.broadcasted_iota/4" = L_NEGAXIS, "lax.argmax/8" = L_NEGAXIS,
  "lax.argmin/8" = L_NEGAXIS, "lax.concatenate/3" = L_NEGAXIS,
  "lax.reshape/6" = "known deviation: lax_reshape checks the total size at trace time (jax_type_error); JAX fails later in the MLIR verifier (ValueError)",
  "lax.convert_element_type/21" = "known deviation: R accepts 'float0' as a dtype name; NumPy (and so JAX) does not")

# Documented deviation (R/lax.R, lax_args): an R scalar adopts the dtype of the other
# operand, while jax.lax raises when a Python scalar's default dtype differs (for example
# lax.add(float32 array, 2.0) with x64 enabled). Those wheel errors are skipped.
lax_scalar_deviation <- function(x64) {
  fx <- fx_load("lax", x64)
  ids <- character()
  for (case in fx$cases) {
    has_scalar <- any(vapply(case$args, function(a) identical(a$t, "scalar"), NA))
    if (has_scalar && !is.null(case$error) &&
        grepl("same dtype|dtypes should match|argument to pow must have", case$error$message))
      ids <- c(ids, paste0(case$id, if (x64) "@x64" else "@x32"))
  }
  stats::setNames(rep("known deviation: R scalars adopt the other operand's dtype in lax functions (jax.lax raises for mismatched Python scalar dtypes)",
                      length(ids)), ids)
}

lax_opts <- list("lax.cos/7" = list(zero_sign = FALSE))

fx_run_cases("lax", known = c(lax_known, lax_scalar_deviation(FALSE), lax_scalar_deviation(TRUE)), opts = lax_opts)
