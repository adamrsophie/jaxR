# Precompute the vignettes. The code needs the XLA runtime (jaxR::install_pjrt()), which
# CRAN machines do not have, so each vignette is knitted here from its .Rmd.orig source
# into a static .Rmd. Run from the package root: Rscript vignettes/precompute.R [name ...]
# Each vignette is knitted in a fresh R process, so that the packages one vignette
# attaches cannot mask functions in the next.
args <- commandArgs(TRUE)
src <- list.files("vignettes", pattern = "\\.Rmd\\.orig$")
if (length(args)) src <- intersect(src, paste0(args, ".Rmd.orig"))
for (f in src) {
  message("knitting ", f)
  code <- sprintf(
    "setwd('vignettes'); knitr::knit('%s', output = '%s', quiet = TRUE, envir = globalenv())",
    f, sub("\\.orig$", "", f))
  status <- system2(file.path(R.home("bin"), "Rscript"), c("-e", shQuote(code)))
  if (status != 0L) stop("knitting ", f, " failed")
}
