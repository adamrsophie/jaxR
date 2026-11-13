# Enter the R browser at a point in a (possibly compiled) program

Port of `jax.debug.breakpoint`. The variables of the calling frames are
captured; traced values among them are passed through a debug callback,
so inside [`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md)
the breakpoint triggers when the compiled program runs, with their
values (as R arrays). The default debugger evaluates
[`browser()`](https://rdrr.io/r/base/browser.html) in an environment
holding the variables of the innermost frame (with the other frames as
`.frames`).

## Usage

``` r
debug_breakpoint(
  backend = NULL,
  filter_frames = TRUE,
  num_frames = NULL,
  ordered = FALSE,
  token = NULL,
  ...
)
```

## Arguments

- backend:

  `NULL` (the R browser) or a function called with the list of frame
  environments (innermost first), a jaxR extension standing for JAX's
  debugger registry.

- filter_frames:

  Drop frames of jaxR itself.

- num_frames:

  Number of frames to capture (all if `NULL`).

- ordered:

  Whether ordered breakpoints keep their program order.

- token:

  Optional pytree of arrays: the breakpoint runs once its value is
  computed and it is returned unchanged (thread it into later
  computation).

- ...:

  Ignored.

## Value

`token` (invisibly `NULL` when no token is given).
