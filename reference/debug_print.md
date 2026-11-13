# Print values, also inside compiled code

Port of `jax.debug.print`. `fmt` is a Python format string (`"x = {x}"`,
`"{} and {}"`, `"{0}"`, with optional specs such as `"{x:.3f}"`); its
fields are filled from the unnamed (positional) and named arguments.
Arrays are formatted with R's formatting (as
[`print()`](https://rdrr.io/r/base/print.html) shows them); scalars as
[`format()`](https://rdrr.io/r/base/format.html) shows them. Inside
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) the
values are printed when the compiled program runs; under
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) once
per batch element. Without placeholders the arguments are appended to
`fmt`, separated by spaces. `debug_log()` writes the message with
[`message()`](https://rdrr.io/r/base/message.html) instead. Called
without `fmt`, they return a function taking `(fmt, ...)` (two-call
form).

## Usage

``` r
debug_print(
  fmt = NULL,
  ...,
  ordered = FALSE,
  partitioned = FALSE,
  skip_format_check = FALSE
)

debug_log(
  fmt = NULL,
  ...,
  ordered = FALSE,
  partitioned = FALSE,
  skip_format_check = FALSE
)
```

## Arguments

- fmt:

  Format string.

- ...:

  Values to format (unnamed: positional fields; named: named fields).

- ordered:

  Whether ordered debug prints keep their program order.

- partitioned:

  Ignored (single device).

- skip_format_check:

  Skip the check that every argument is used by `fmt`.

## Value

`NULL`, invisibly (or a function in the two-call form).

## Examples

``` r
if (FALSE) { # \dontrun{
f <- jit(function(x) { debug_print("x = {x}, sum = {s:.2f}", x = x, s = jnp_sum(x)); x * 2 })
f(as_jax(c(1, 2, 3)))
} # }
```
