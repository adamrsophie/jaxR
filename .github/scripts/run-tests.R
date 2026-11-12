# Run one shard of the test suite against the installed package, with the XLA runtime,
# and report every failure, error and skip.
# Usage: Rscript .github/scripts/run-tests.R <shard> <shards>
args <- as.integer(commandArgs(TRUE))
shard <- args[[1L]]
shards <- args[[2L]]

library(testthat)
library(jaxR)
install_pjrt()
print(jax_backend_init())

dir <- file.path("tests", "testthat")
files <- sub("^test-(.*)\\.R$", "\\1", list.files(dir, pattern = "^test-.*\\.R$"))
# The fixture files are short but run most of the expectations: each gets its own shard.
# The other files are dealt out by decreasing size.
heavy <- intersect(c("core-jnp", "core-lax", "core-transforms"), files)
stopifnot(shards > length(heavy))
rest <- setdiff(files, heavy)
rest <- rest[order(-file.size(file.path(dir, paste0("test-", rest, ".R"))), rest)]
mine <- if (shard <= length(heavy)) heavy[[shard]] else
  rest[(seq_along(rest) - 1L) %% (shards - length(heavy)) == shard - length(heavy) - 1L]
cat("shard", shard, "of", shards, ":", paste(mine, collapse = " "), "\n")

options(testthat.summary.max_reports = Inf)
res <- test_dir(dir, filter = paste0("^(", paste(mine, collapse = "|"), ")$"), reporter = "summary",
                stop_on_failure = FALSE, load_package = "none")
df <- as.data.frame(res)

kinds <- c(expectation_failure = "FAILURE", expectation_error = "ERROR", expectation_skip = "SKIP")
found <- unlist(lapply(res, function(t) unlist(lapply(t$results, function(r) {
  k <- kinds[intersect(names(kinds), class(r))]
  if (length(k)) paste0(k[[1L]], " | ", t$file, " | ", t$test, "\n  ",
                        gsub("\n", "\n  ", conditionMessage(r)))
}))))
found <- as.character(found)
dir.create("test-results", showWarnings = FALSE)
writeLines(found, file.path("test-results", sprintf("shard-%d.txt", shard)))
saveRDS(df[c("file", "test", "nb", "failed", "skipped", "error", "real")],
        file.path("test-results", sprintf("shard-%d.rds", shard)))

cat("\n==== skips ====\n")
writeLines(grep("^SKIP", found, value = TRUE))
cat("\n==== failures and errors ====\n")
writeLines(grep("^SKIP", found, value = TRUE, invert = TRUE))
cat(sprintf("\nfiles=%d tests=%d failed=%d skipped=%d errors=%d seconds=%.0f\n", length(unique(df$file)),
            sum(df$nb), sum(df$failed), sum(df$skipped), sum(df$error), sum(df$real)))
if (!length(unique(df$file)) || sum(df$failed) + sum(df$error) > 0) quit(status = 1L)
