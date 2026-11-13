# Install the XLA runtime plugin

Downloads a prebuilt PJRT CPU plugin (built from OpenXLA by the zml
project) into the user data directory of jaxR. Set the environment
variable `JAXR_PJRT_PLUGIN` to use a different plugin (for example a
CUDA plugin) instead. Prebuilt plugins exist for Linux and macOS on
x86-64 and arm64; on other systems, Windows included, the function stops
with an error and `JAXR_PJRT_PLUGIN` is the only way to supply a
runtime.

## Usage

``` r
install_pjrt(release = pjrt_release, force = FALSE)
```

## Arguments

- release:

  Release tag of `zml/pjrt-artifacts`.

- force:

  Reinstall even if present.

## Value

The path of the installed plugin, invisibly.
