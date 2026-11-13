# Spherical linear interpolation of rotations

Port of `jax.scipy.spatial.transform.Slerp`.
`scipy_spatial_transform_Slerp_init()` (the `Slerp.init` class method)
builds the interpolator from increasing `times` and a rotation holding
one rotation per time; `scipy_spatial_transform_Slerp()` is the raw
constructor of the named tuple. The result is callable: `slerp(times)`
returns the interpolated rotations (a single rotation for a scalar
time). Its fields `times`, `timedelta`, `rotations` and `rotvecs` are
read with `$`, and it is a pytree.

## Usage

``` r
scipy_spatial_transform_Slerp(times, timedelta, rotations, rotvecs)

# S3 method for class 'scipy_spatial_transform_Slerp'
x$name

scipy_spatial_transform_Slerp_init(times, rotations)
```

## Arguments

- times:

  Times of the key rotations (1-D, increasing); for the raw constructor
  the stored times.

- timedelta, rotvecs:

  Stored fields of the raw constructor.

- rotations:

  A
  [`scipy_spatial_transform_Rotation()`](https://adamrsophie.github.io/jaxR/reference/scipy_spatial_transform_Rotation.md)
  with at least two rotations.

- x:

  Slerp object.

- name:

  Field name.

## Value

A callable Slerp object; for `$`, the field.

## Examples

``` r
if (FALSE) { # \dontrun{
rots <- scipy_spatial_transform_Rotation_from_euler(
  "z", matrix(c(0, 90), 2), degrees = TRUE)
slerp <- scipy_spatial_transform_Slerp_init(c(0, 1), rots)
scipy_spatial_transform_Rotation_as_euler(slerp(0.5), "zyx", degrees = TRUE)
} # }
```
