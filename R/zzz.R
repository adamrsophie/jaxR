.onLoad <- function(libname, pkgname) {
  read_env_config()
  .jx$track_live <- TRUE
  invisible(NULL)
}
