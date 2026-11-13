# Rotations in three dimensions

Port of `jax.scipy.spatial.transform.Rotation`, a pytree holding unit
quaternions in scalar last `(x, y, z, w)` order: `quat` has shape `(4,)`
for a single rotation or `(N, 4)` for `N` rotations. The class methods
of JAX are functions: `scipy_spatial_transform_Rotation_from_quat(quat)`
and the other `_from_*` constructors, `_identity`, `_concatenate`;
instance methods take the rotation as first argument
(`scipy_spatial_transform_Rotation_apply(r, v)` is `r.apply(v)`). `r[i]`
indexes the rotations (0-based) and `r1 * r2` composes them (`r2` is
applied first). `scipy_spatial_transform_Rotation()` itself stores
`quat` as given, like the JAX named tuple; `_from_quat` normalizes.

## Usage

``` r
scipy_spatial_transform_Rotation(quat)

scipy_spatial_transform_Rotation_concatenate(rotations)

scipy_spatial_transform_Rotation_from_euler(seq, angles, degrees = FALSE)

scipy_spatial_transform_Rotation_from_matrix(matrix)

scipy_spatial_transform_Rotation_from_mrp(mrp)

scipy_spatial_transform_Rotation_from_quat(quat)

scipy_spatial_transform_Rotation_from_rotvec(rotvec, degrees = FALSE)

scipy_spatial_transform_Rotation_identity(num = NULL, dtype = "float")

scipy_spatial_transform_Rotation_random(random_key, num = NULL)

# S3 method for class 'scipy_spatial_transform_Rotation'
x[...]

scipy_spatial_transform_Rotation_len(self)

# S3 method for class 'scipy_spatial_transform_Rotation'
e1 * e2

scipy_spatial_transform_Rotation_apply(self, vectors, inverse = FALSE)

scipy_spatial_transform_Rotation_as_euler(self, seq, degrees = FALSE)

scipy_spatial_transform_Rotation_as_matrix(self)

scipy_spatial_transform_Rotation_as_mrp(self)

scipy_spatial_transform_Rotation_as_rotvec(self, degrees = FALSE)

scipy_spatial_transform_Rotation_as_quat(
  self,
  canonical = FALSE,
  scalar_first = FALSE
)

scipy_spatial_transform_Rotation_inv(self)

scipy_spatial_transform_Rotation_magnitude(self)

scipy_spatial_transform_Rotation_mean(self, weights = NULL)

scipy_spatial_transform_Rotation_single(self)
```

## Arguments

- quat:

  Quaternion array (scalar last).

- rotations:

  List of rotations.

- seq:

  Axis sequence for Euler angles: up to three characters from `"xyz"`
  (extrinsic) or `"XYZ"` (intrinsic).

- angles:

  Euler angles, shape `(..., length(seq))`.

- degrees:

  Whether angles are in degrees.

- matrix:

  Rotation matrices, shape `(..., 3, 3)`.

- mrp:

  Modified Rodrigues parameters, shape `(..., 3)`.

- rotvec:

  Rotation vectors (axis times angle), shape `(..., 3)`.

- num:

  Must be `NULL` (one identity rotation).

- dtype:

  Dtype of the identity quaternion.

- random_key:

  Unused (`random` is not implemented upstream).

- ...:

  Index arguments (see
  [jax_indexing](https://adamrsophie.github.io/jaxR/reference/jax_indexing.md));
  0-based.

- self, x, e1, e2:

  Rotations.

- vectors:

  Vectors to rotate, shape `(..., 3)`.

- inverse:

  Apply the inverse rotation.

- canonical:

  Return the quaternion with non-negative `w` (or the first nonzero
  component positive).

- scalar_first:

  Return quaternions in `(w, x, y, z)` order.

- weights:

  Optional weights of the rotations for the mean.

## Value

A rotation, or an array for the `as_*`, `apply` and `magnitude` methods.

## Examples

``` r
if (FALSE) { # \dontrun{
r <- scipy_spatial_transform_Rotation_from_euler("xyz", c(90, 0, 0), degrees = TRUE)
scipy_spatial_transform_Rotation_apply(r, c(0, 1, 0))
scipy_spatial_transform_Rotation_as_rotvec(r)
} # }
```
