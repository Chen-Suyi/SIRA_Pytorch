import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import transforms3d.quaternions as t3d
from scipy.spatial.transform import Rotation
from typing import Optional, List, Union


def apply_transform(points: torch.Tensor,
                    transform: torch.Tensor,
                    normals: Optional[torch.Tensor] = None):
    r"""Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and transform {}.'.format(
                tuple(points.shape), tuple(transform.shape)))
    if normals is not None:
        return points, normals
    else:
        return points


def apply_rotation(points: torch.Tensor,
                   rotation: torch.Tensor,
                   normals: Optional[torch.Tensor] = None):
    r"""Rotate points and normals (optional) along the origin.

    Given a point cloud P(3, N), normals V(3, N) and a rotation matrix R, the output point cloud Q = RP, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), rotation is (3, 3), the output points are (*, 3).
       In this case, the rotation is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 3, 3), the output points are (B, N, 3).
       In this case, the rotation is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        rotation (Tensor): (3, 3) or (B, 3, 3)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if rotation.ndim == 2:
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2))
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif rotation.ndim == 3 and points.ndim == 3:
        points = torch.matmul(points, rotation.transpose(-1, -2))
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and rotation{}.'.format(
                tuple(points.shape), tuple(rotation.shape)))
    if normals is not None:
        return points, normals
    else:
        return points


def get_rotation_translation_from_transform(transform):
    r"""Decompose transformation matrix into rotation matrix and translation vector.

    Args:
        transform (Tensor): (*, 4, 4) or (*, 3, 4)

    Returns:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)
    """
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return rotation, translation


def get_transform_from_rotation_translation(rotation, translation):
    r"""Compose transformation matrix from rotation matrix and translation vector.

    Args:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)

    Returns:
        transform (Tensor): (*, 4, 4)
    """
    input_shape = rotation.shape
    rotation = rotation.view(-1, 3, 3)
    translation = translation.view(-1, 3)
    transform = torch.eye(4).to(rotation).unsqueeze(0).repeat(
        rotation.shape[0], 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation
    output_shape = input_shape[:-2] + (4, 4)
    transform = transform.view(*output_shape)
    return transform


def np_get_transform_from_rotation_translation(
        rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def complete_transform(transform):
    r"""Complete rigid transform.

    Args:
        transform (Tensor): (*, 3, 4)

    Return:
        completed_transform (Tensor): (*, 4, 4)
    """
    rotation, translation = get_rotation_translation_from_transform(
        transform)  # (*, 3, 3), (*, 3)
    completed_transform = get_transform_from_rotation_translation(
        rotation, translation)
    return completed_transform


def incomplete_transform(transform):
    r"""Change rigid transform into incomplete form.

    Args:
        transform (Tensor): (*, 4, 4)

    Return:
        incomplete_transform (Tensor): (*, 3, 4)
    """
    return transform[:, :3, :]


def inverse_transform(transform):
    r"""Inverse rigid transform.

    Args:
        transform (Tensor): (*, 4, 4) or (*, 3, 4)

    Return:
        inv_transform (Tensor): (*, 4, 4)
    """
    rotation, translation = get_rotation_translation_from_transform(
        transform)  # (*, 3, 3), (*, 3)
    inv_rotation = rotation.transpose(-1, -2)  # (*, 3, 3)
    inv_translation = -torch.matmul(
        inv_rotation, translation.unsqueeze(-1)).squeeze(-1)  # (*, 3)
    inv_transform = get_transform_from_rotation_translation(
        inv_rotation, inv_translation)  # (*, 4, 4)
    return inv_transform


def skew_symmetric_matrix(inputs):
    r"""Compute Skew-symmetric Matrix.

    [v]_{\times} =  0 -z  y
                    z  0 -x
                   -y  x  0

    Args:
        inputs (Tensor): input vectors (*, c)

    Returns:
        skews (Tensor): output skew-symmetric matrix (*, 3, 3)
    """
    input_shape = inputs.shape
    output_shape = input_shape[:-1] + (3, 3)
    skews = torch.zeros(size=output_shape).cuda()
    skews[..., 0, 1] = -inputs[..., 2]
    skews[..., 0, 2] = inputs[..., 1]
    skews[..., 1, 0] = inputs[..., 2]
    skews[..., 1, 2] = -inputs[..., 0]
    skews[..., 2, 0] = -inputs[..., 1]
    skews[..., 2, 1] = inputs[..., 0]
    return skews


def rodrigues_rotation_matrix(axes, angles):
    r"""Compute Rodrigues Rotation Matrix.

    R = I + \sin{\theta} K + (1 - \cos{\theta}) K^2,
    where K is the skew-symmetric matrix of the axis vector.

    Args:
        axes (Tensor): axis vectors (*, 3)
        angles (Tensor): rotation angles in right-hand direction in rad. (*)

    Returns:
        rotations (Tensor): Rodrigues rotation matrix (*, 3, 3)
    """
    input_shape = axes.shape
    axes = axes.view(-1, 3)
    angles = angles.view(-1)
    axes = F.normalize(axes, p=2, dim=1)
    skews = skew_symmetric_matrix(axes)  # (B, 3, 3)
    sin_values = torch.sin(angles).view(-1, 1, 1)  # (B,)
    cos_values = torch.cos(angles).view(-1, 1, 1)  # (B,)
    eyes = torch.eye(3).cuda().unsqueeze(0).expand_as(skews)  # (B, 3, 3)
    rotations = eyes + sin_values * skews + (1.0 - cos_values) * torch.matmul(
        skews, skews)
    output_shape = input_shape[:-1] + (3, 3)
    rotations = rotations.view(*output_shape)
    return rotations


def rodrigues_alignment_matrix(src_vectors, tgt_vectors):
    r"""Compute the Rodrigues rotation matrix aligning source vectors to target vectors.

    Args:
        src_vectors (Tensor): source vectors (*, 3)
        tgt_vectors (Tensor): target vectors (*, 3)

    Returns:
        rotations (Tensor): rotation matrix (*, 3, 3)
    """
    input_shape = src_vectors.shape
    src_vectors = src_vectors.view(-1, 3)  # (B, 3)
    tgt_vectors = tgt_vectors.view(-1, 3)  # (B, 3)

    # compute axes
    src_vectors = F.normalize(src_vectors, dim=-1, p=2)  # (B, 3)
    tgt_vectors = F.normalize(tgt_vectors, dim=-1, p=2)  # (B, 3)
    src_skews = skew_symmetric_matrix(src_vectors)  # (B, 3, 3)
    axes = torch.matmul(src_skews,
                        tgt_vectors.unsqueeze(-1)).squeeze(-1)  # (B, 3)

    # compute rodrigues rotation matrix
    sin_values = torch.linalg.norm(axes, dim=-1)  # (B,)
    cos_values = (src_vectors * tgt_vectors).sum(dim=-1)  # (B,)
    axes = F.normalize(axes, dim=-1, p=2)  # (B, 3)
    skews = skew_symmetric_matrix(axes)  # (B, 3, 3)
    eyes = torch.eye(3).cuda().unsqueeze(0).expand_as(skews)  # (B, 3, 3)
    sin_values = sin_values.view(-1, 1, 1)
    cos_values = cos_values.view(-1, 1, 1)
    rotations = eyes + sin_values * skews + (1.0 - cos_values) * torch.matmul(
        skews, skews)

    # handle opposite direction
    sin_values = sin_values.view(-1)
    cos_values = cos_values.view(-1)
    masks = torch.logical_and(torch.eq(sin_values, 0.0),
                              torch.lt(cos_values, 0.0))
    rotations[masks] *= -1

    output_shape = input_shape[:-1] + (3, 3)
    rotations = rotations.view(*output_shape)

    return rotations


def relative_rotation_error(gt_rotations, rotations):
    r"""Isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotations (Tensor): ground truth rotation matrix (*, 3, 3)
        rotations (Tensor): estimated rotation matrix (*, 3, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    mat = torch.matmul(rotations.transpose(-1, -2), gt_rotations)
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    x = 0.5 * (trace - 1.0)
    x = x.clamp(min=-1.0, max=1.0)
    x = torch.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def relative_translation_error(gt_translations, translations):
    r"""Isotropic Relative Rotation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translations (Tensor): ground truth translation vector (*, 3)
        translations (Tensor): estimated translation vector (*, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    rte = torch.linalg.norm(gt_translations - translations, dim=-1)
    return rte


def torch_identity(batch_size):
    return torch.eye(3, 4)[None, ...].repeat(batch_size, 1, 1)


def torch_inverse(g):
    """ Returns the inverse of the SE3 transform

    Args:
        g: (B, 3/4, 4) transform

    Returns:
        (B, 3, 4) matrix containing the inverse

    """
    # Compute inverse
    rot = g[..., 0:3, 0:3]
    trans = g[..., 0:3, 3]
    inverse_transform = torch.cat(
        [rot.transpose(-1, -2),
         rot.transpose(-1, -2) @ -trans[..., None]],
        dim=-1)

    return inverse_transform


def torch_concatenate(a, b):
    """Concatenate two SE3 transforms,
    i.e. return a@b (but note that our SE3 is represented as a 3x4 matrix)

    Args:
        a: (B, 3/4, 4)
        b: (B, 3/4, 4)

    Returns:
        (B, 3/4, 4)
    """

    rot1 = a[..., :3, :3]
    trans1 = a[..., :3, 3]
    rot2 = b[..., :3, :3]
    trans2 = b[..., :3, 3]

    rot_cat = rot1 @ rot2
    trans_cat = rot1 @ trans2[..., None] + trans1[..., None]
    concatenated = torch.cat([rot_cat, trans_cat], dim=-1)

    return concatenated


def torch_transform(g, a, normals=None):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformation matrix of size ([1,] 3/4, 4) or (B, 3/4, 4)
        a: Points to be transformed (N, 3) or (B, N, 3)
        normals: (Optional). If provided, normals will be transformed

    Returns:
        transformed points of size (N, 3) or (B, N, 3)

    """
    R = g[..., :3, :3]  # (B, 3, 3)
    p = g[..., :3, 3]  # (B, 3)

    if len(g.size()) == len(a.size()):
        b = torch.matmul(a, R.transpose(-1, -2)) + p[..., None, :]
    else:
        raise NotImplementedError
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p  # No batch. Not checked

    if normals is not None:
        rotated_normals = normals @ R.transpose(-1, -2)
        return b, rotated_normals

    else:
        return b


def torch_mat2quat(M):
    all_pose = []
    for i in range(M.size()[0]):
        rotate = M[i, :3, :3]
        translate = M[i, :3, 3]

        # Qyx refers to the contribution of the y input vector component to
        # the x output vector component.  Qyx is therefore the same as
        # M[0,1].  The notation is from the Wikipedia article.
        Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = rotate.flatten()
        #     print(Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz)
        # Fill only lower half of symmetric matrix
        K = torch.tensor(
            [[Qxx - Qyy - Qzz, 0, 0, 0], [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
             [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
             [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]]) / 3.0
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = torch.symeig(K, True, False)
        # Select largest eigenvector, reorder to w,x,y,z quaternion

        q = vecs[[3, 0, 1, 2], torch.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[0] < 0:
            q *= -1

        pose = torch.cat((q, translate), dim=0)
        all_pose.append(pose)
    all_pose = torch.stack(all_pose, dim=0)
    return all_pose  # (B, 7)


def np_identity():
    return np.eye(3, 4)


def np_transform(g: np.ndarray, pts: np.ndarray):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformation matrix of size ([B,] 3/4, 4)
        pts: Points to be transformed ([B,] N, 3)

    Returns:
        transformed points of size (N, 3)
    """
    rot = g[..., :3, :3]  # (3, 3)
    trans = g[..., :3, 3]  # (3)

    transformed = pts[..., :3] @ np.swapaxes(rot, -1, -2) + trans[..., None, :]
    return transformed


def np_inverse(g: np.ndarray):
    """Returns the inverse of the SE3 transform

    Args:
        g: ([B,] 3/4, 4) transform

    Returns:
        ([B,] 3/4, 4) matrix containing the inverse

    """
    rot = g[..., :3, :3]  # (3, 3)
    trans = g[..., :3, 3]  # (3)

    inv_rot = np.swapaxes(rot, -1, -2)
    inverse_transform = np.concatenate([inv_rot, inv_rot @ -trans[..., None]],
                                       axis=-1)
    if g.shape[-2] == 4:
        inverse_transform = np.concatenate(
            [inverse_transform, [[0.0, 0.0, 0.0, 1.0]]], axis=-2)

    return inverse_transform


def np_concatenate(a: np.ndarray, b: np.ndarray):
    """ Concatenate two SE3 transforms

    Args:
        a: First transform ([B,] 3/4, 4)
        b: Second transform ([B,] 3/4, 4)

    Returns:
        a*b ([B, ] 3/4, 4)

    """

    r_a, t_a = a[..., :3, :3], a[..., :3, 3]
    r_b, t_b = b[..., :3, :3], b[..., :3, 3]

    r_ab = r_a @ r_b
    t_ab = r_a @ t_b[..., None] + t_a[..., None]

    concatenated = np.concatenate([r_ab, t_ab], axis=-1)

    if a.shape[-2] == 4:
        concatenated = np.concatenate([concatenated, [[0.0, 0.0, 0.0, 1.0]]],
                                      axis=-2)

    return concatenated


def np_from_xyzquat(xyzquat):
    """Constructs SE3 matrix from x, y, z, qx, qy, qz, qw

    Args:
        xyzquat: np.array (7,) containing translation and quaterion

    Returns:
        SE3 matrix (4, 4)
    """
    rot = Rotation.from_quat(xyzquat[3:])
    trans = rot.apply(-xyzquat[:3])
    transform = np.concatenate([rot.as_dcm(), trans[:, None]], axis=1)
    transform = np.concatenate([transform, [[0.0, 0.0, 0.0, 1.0]]], axis=0)

    return transform


def np_mat2quat(transform):
    rotate = transform[:3, :3]
    translate = transform[:3, 3]
    quat = t3d.mat2quat(rotate)
    pose = np.concatenate([quat, translate], axis=0)
    return pose  # (7, )


def np_quat2mat(pose):
    # Separate each quaternion value.
    q0, q1, q2, q3 = pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3]
    # Convert quaternion to rotation matrix.
    # Ref: 	http://www-evasion.inrialpes.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf
    # A method for Registration of 3D shapes paper by Paul J. Besl and Neil D McKay.
    R11 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    R12 = 2 * (q1 * q2 - q0 * q3)
    R13 = 2 * (q1 * q3 + q0 * q2)
    R21 = 2 * (q1 * q2 + q0 * q3)
    R22 = q0 * q0 + q2 * q2 - q1 * q1 - q3 * q3
    R23 = 2 * (q2 * q3 - q0 * q1)
    R31 = 2 * (q1 * q3 - q0 * q2)
    R32 = 2 * (q2 * q3 + q0 * q1)
    R33 = q0 * q0 + q3 * q3 - q1 * q1 - q2 * q2
    R = np.stack((np.stack(
        (R11, R12, R13), axis=0), np.stack(
            (R21, R22, R23), axis=0), np.stack((R31, R32, R33), axis=0)),
                 axis=0)

    rot_mat = R.transpose((2, 0, 1))  # (B, 3, 3)
    translation = pose[:, 4:][:, :, None]  # (B, 3, 1)
    transform = np.concatenate((rot_mat, translation), axis=2)
    return transform  # (B, 3, 4)


def list_apply_transform(points: List[Tensor], transform: Union[List[Tensor],
                                                                Tensor]):
    """Similar to apply_transform, but processes lists of tensors instead

    Args:
        points: List(B) of (N, 3)
        transform: List(B) of (3 or 4,4) or Tensor(B, 3 or 4, 4) 

    Returns:
        List of transformed points, shape likes points
    """
    B = len(points)  # batch_size
    if isinstance(transform, torch.Tensor):
        if len(transform.shape) == 2:
            transform = transform[None, :, :]
    assert all([
        points[b].shape[-1] == 3
        and transform[b].shape[:-2] == points[b].shape[:-2] for b in range(B)
    ])

    transformed_all = []
    for b in range(B):
        transformed = apply_transform(points[b], transform[b])
        transformed_all.append(transformed)

    return transformed_all
