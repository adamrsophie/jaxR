# jax.numpy functions of R/numpy_core.R against the wheel (fixtures/jnp_*.json):
# values, dtypes, weak types and errors, eagerly and under jit, x64 off and on.
# A skipped case is a documented deviation (see vignette("differences")).

R_INTVEC <- "known deviation: R integer vectors are int32 (a Python int list is int64 under x64)"

jnp_known <- c("jnp.asarray/1@x64" = R_INTVEC, "jnp.array/1@x64" = R_INTVEC)

jnp_opts <- list("jnp.cos/10" = list(zero_sign = FALSE))

fx_run_cases("jnp", known = jnp_known, opts = jnp_opts)
