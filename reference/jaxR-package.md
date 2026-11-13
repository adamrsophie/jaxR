# jaxR: composable transformations of array programs in R

An R mirror of the JAX Python package. Programs written with `jnp_*`,
`lax_*` and the usual R operators are traced to jaxprs, transformed
([`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md),
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md),
[`jvp()`](https://adamrsophie.github.io/jaxR/reference/jvp.md),
[`vjp()`](https://adamrsophie.github.io/jaxR/reference/vjp.md)) and
compiled by XLA through the PJRT C API
([`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md)). Run
[`install_pjrt()`](https://adamrsophie.github.io/jaxR/reference/install_pjrt.md)
once to install the XLA runtime.

## See also

Useful links:

- <https://adamrsophie.github.io/jaxR/>

- <https://github.com/adamrsophie/jaxR>

- Report bugs at <https://github.com/adamrsophie/jaxR/issues>

## Author

**Maintainer**: Adam R Sophie <adamrsophie@gmail.com> \[copyright
holder\]

Authors:

- Adam R Sophie <adamrsophie@gmail.com> \[copyright holder\]
