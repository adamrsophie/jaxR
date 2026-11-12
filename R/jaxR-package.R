#' jaxR: composable transformations of array programs in R
#'
#' An R mirror of the JAX Python package. Programs written with `jnp_*`, `lax_*` and
#' the usual R operators are traced to jaxprs, transformed ([grad()], [vmap()],
#' [jvp()], [vjp()]) and compiled by XLA through the PJRT C API ([jit()]).
#' Run [install_pjrt()] once to install the XLA runtime.
#'
#' @keywords internal
#' @useDynLib jaxR, .registration = TRUE
"_PACKAGE"
