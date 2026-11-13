# Persistent compilation cache

Ports of
`jax.experimental.compilation_cache.compilation_cache.set_cache_dir` and
`reset_cache`. With a cache directory set (here, or with the option
`jax_compilation_cache_dir` of
[`jax_config_update()`](https://adamrsophie.github.io/jaxR/reference/jax_config_update.md)),
compiled programs are written to the directory and read back instead of
being compiled again, also by later R sessions. As in JAX, only programs
whose compilation took at least
`jax_persistent_cache_min_compile_time_secs` seconds (default 1) are
written, and `jax_enable_compilation_cache = FALSE` turns the cache off.
Programs with host callbacks are not cached. Problems with the cache are
warnings unless `jax_raise_persistent_cache_errors` is `TRUE`.

## Usage

``` r
experimental_compilation_cache_set_cache_dir(path)

experimental_compilation_cache_reset_cache()
```

## Arguments

- path:

  Directory of the cache.

## Value

`NULL`, invisibly.

## Details

The cache is trusted: a cached file is native code that is run. Do not
use a directory that others can write to.

## Examples

``` r
if (FALSE) { # \dontrun{
experimental_compilation_cache_set_cache_dir(file.path(tempdir(), "jax_cache"))
} # }
```
