# No exported jaxR function may mask a function of base R or the default packages.
test_that("exports do not mask base R functions", {
  pkgs <- c("base", "stats", "utils", "methods", "graphics", "grDevices", "datasets", "tools")
  base_names <- unique(unlist(lapply(pkgs, function(p) getNamespaceExports(p))))
  exports <- getNamespaceExports("jaxR")
  clash <- sort(intersect(exports, base_names))
  expect_identical(clash, character(0))
})
