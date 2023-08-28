import torch
import torch.nn as nn
import importlib
import numpy as np
import open3d as o3d
from typing import Optional
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from common.utils.se3 import apply_transform, get_rotation_translation_from_transform, relative_rotation_error, relative_translation_error, np_transform

ext_module = importlib.import_module("geotransformer.ext")


def grid_subsample(points, lengths, voxel_size):
    """Grid subsampling in stack mode.

    This function is implemented on CPU.

    Args:
        points (Tensor): stacked points. (N, 3)
        lengths (Tensor): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
    """
    s_points, s_lengths = ext_module.grid_subsampling(points, lengths,
                                                      voxel_size)
    return s_points, s_lengths


def pairwise_distance(x: torch.Tensor,
                      y: torch.Tensor,
                      normalized: bool = False,
                      channel_first: bool = False) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2),
                          y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(
            -1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x**2, dim=channel_dim).unsqueeze(
            -1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y**2, dim=channel_dim).unsqueeze(
            -2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances


@torch.no_grad()
def get_node_correspondences(
    ref_nodes: torch.Tensor,
    src_nodes: torch.Tensor,
    ref_knn_points: torch.Tensor,
    src_knn_points: torch.Tensor,
    transform: torch.Tensor,
    pos_radius: float,
    ref_masks: Optional[torch.Tensor] = None,
    src_masks: Optional[torch.Tensor] = None,
    ref_knn_masks: Optional[torch.Tensor] = None,
    src_knn_masks: Optional[torch.Tensor] = None,
):
    r"""Generate ground-truth superpoint/patch correspondences.

    Each patch is composed of at most k nearest points of the corresponding superpoint.
    A pair of points match if the distance between them is smaller than `self.pos_radius`.

    Args:
        ref_nodes: torch.Tensor (M, 3)
        src_nodes: torch.Tensor (N, 3)
        ref_knn_points: torch.Tensor (M, K, 3)
        src_knn_points: torch.Tensor (N, K, 3)
        transform: torch.Tensor (4, 4)
        pos_radius: float
        ref_masks (optional): torch.BoolTensor (M,) (default: None)
        src_masks (optional): torch.BoolTensor (N,) (default: None)
        ref_knn_masks (optional): torch.BoolTensor (M, K) (default: None)
        src_knn_masks (optional): torch.BoolTensor (N, K) (default: None)

    Returns:
        corr_indices: torch.LongTensor (C, 2)
        corr_overlaps: torch.Tensor (C,)
    """
    src_nodes = apply_transform(src_nodes, transform)
    src_knn_points = apply_transform(src_knn_points, transform)

    # generate masks
    if ref_masks is None:
        ref_masks = torch.ones(size=(ref_nodes.shape[0], ),
                               dtype=torch.bool).cuda()
    if src_masks is None:
        src_masks = torch.ones(size=(src_nodes.shape[0], ),
                               dtype=torch.bool).cuda()
    if ref_knn_masks is None:
        ref_knn_masks = torch.ones(size=(ref_knn_points.shape[0],
                                         ref_knn_points.shape[1]),
                                   dtype=torch.bool).cuda()
    if src_knn_masks is None:
        src_knn_masks = torch.ones(size=(src_knn_points.shape[0],
                                         src_knn_points.shape[1]),
                                   dtype=torch.bool).cuda()

    node_mask_mat = torch.logical_and(ref_masks.unsqueeze(1),
                                      src_masks.unsqueeze(0))  # (M, N)

    # filter out non-overlapping patches using enclosing sphere
    ref_knn_dists = torch.linalg.norm(ref_knn_points - ref_nodes.unsqueeze(1),
                                      dim=-1)  # (M, K)
    ref_knn_dists.masked_fill_(~ref_knn_masks, 0.0)
    ref_max_dists = ref_knn_dists.max(1)[0]  # (M,)
    src_knn_dists = torch.linalg.norm(src_knn_points - src_nodes.unsqueeze(1),
                                      dim=-1)  # (N, K)
    src_knn_dists.masked_fill_(~src_knn_masks, 0.0)
    src_max_dists = src_knn_dists.max(1)[0]  # (N,)
    dist_mat = torch.sqrt(pairwise_distance(ref_nodes, src_nodes))  # (M, N)
    intersect_mat = torch.gt(
        ref_max_dists.unsqueeze(1) + src_max_dists.unsqueeze(0) + pos_radius -
        dist_mat, 0)
    intersect_mat = torch.logical_and(intersect_mat, node_mask_mat)
    sel_ref_indices, sel_src_indices = torch.nonzero(intersect_mat,
                                                     as_tuple=True)

    # select potential patch pairs
    ref_knn_masks = ref_knn_masks[sel_ref_indices]  # (B, K)
    src_knn_masks = src_knn_masks[sel_src_indices]  # (B, K)
    ref_knn_points = ref_knn_points[sel_ref_indices]  # (B, K, 3)
    src_knn_points = src_knn_points[sel_src_indices]  # (B, K, 3)

    point_mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2),
                                       src_knn_masks.unsqueeze(1))  # (B, K, K)

    # compute overlaps
    dist_mat = pairwise_distance(ref_knn_points, src_knn_points)  # (B, K, K)
    dist_mat.masked_fill_(~point_mask_mat, 1e12)
    point_overlap_mat = torch.lt(dist_mat, pos_radius**2)  # (B, K, K)
    ref_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-1),
                                             dim=-1).float()  # (B,)
    src_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-2),
                                             dim=-1).float()  # (B,)
    ref_overlaps = ref_overlap_counts / ref_knn_masks.sum(-1).float()  # (B,)
    src_overlaps = src_overlap_counts / src_knn_masks.sum(-1).float()  # (B,)
    overlaps = (ref_overlaps + src_overlaps) / 2  # (B,)

    overlap_masks = torch.gt(overlaps, 0)
    ref_corr_indices = sel_ref_indices[overlap_masks]
    src_corr_indices = sel_src_indices[overlap_masks]
    corr_indices = torch.stack([ref_corr_indices, src_corr_indices], dim=1)
    corr_overlaps = overlaps[overlap_masks]

    return corr_indices, corr_overlaps


@torch.no_grad()
def point_to_node_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    point_limit: int,
    return_count: bool = False,
):
    r"""Point-to-Node partition to the point cloud.

    Fixed knn bug.

    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): max number of points to each node
        return_count (bool=False): whether to return `node_sizes`

    Returns:
        point_to_node (Tensor): (N,)
        node_sizes (LongTensor): (M,)
        node_masks (BoolTensor): (M,)
        node_knn_indices (LongTensor): (M, K)
        node_knn_masks (BoolTensor) (M, K)
    """
    sq_dist_mat = pairwise_distance(nodes, points)  # (M, N)

    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool).cuda()  # (M,)
    node_masks.index_fill_(0, point_to_node, True)

    matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)  # (M, N)
    point_indices = torch.arange(points.shape[0]).cuda()  # (N,)
    matching_masks[point_to_node, point_indices] = True  # (M, N)
    sq_dist_mat.masked_fill_(~matching_masks, 1e12)  # (M, N)

    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1,
                                        largest=False)[1]  # (M, K)
    node_knn_node_indices = index_select(point_to_node,
                                         node_knn_indices,
                                         dim=0)  # (M, K)
    node_indices = torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(
        -1, point_limit)  # (M, K)
    node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
    node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node,
                                                     return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0],
                                 dtype=torch.long).cuda()  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks


def index_select(data: torch.Tensor, index: torch.LongTensor,
                 dim: int) -> torch.Tensor:
    r"""Advanced index select.

    Returns a tensor `output` which indexes the `data` tensor along dimension `dim`
    using the entries in `index` which is a `LongTensor`.

    Different from `torch.index_select`, `index` does not has to be 1-D. The `dim`-th
    dimension of `data` will be expanded to the number of dimensions in `index`.

    For example, suppose the shape `data` is $(a_0, a_1, ..., a_{n-1})$, the shape of `index` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.

    Args:
        data (Tensor): (a_0, a_1, ..., a_{n-1})
        index (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim: int

    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    output = data.index_select(dim, index.view(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)

    return output


def isotropic_transform_error(gt_transforms, transforms, reduction="mean"):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transforms (Tensor): ground truth transformation matrix (*, 4, 4)
        transforms (Tensor): estimated transformation matrix (*, 4, 4)
        reduction (str="mean"): reduction method, "mean", "sum" or "none"

    Returns:
        rre (Tensor): relative rotation error.
        rte (Tensor): relative translation error.
    """
    assert reduction in ["mean", "sum", "none"]

    gt_rotations, gt_translations = get_rotation_translation_from_transform(
        gt_transforms)
    rotations, translations = get_rotation_translation_from_transform(
        transforms)

    rre = relative_rotation_error(gt_rotations, rotations)  # (*)
    rte = relative_translation_error(gt_translations, translations)  # (*)

    if reduction == "mean":
        rre = rre.mean()
        rte = rte.mean()
    elif reduction == "sum":
        rre = rre.sum()
        rte = rte.sum()

    return rre, rte


def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:
    """Random rotation matrix generated from uniformly distributed eular angles

    """
    # uniform angle_z, angle_y, angle_x
    euler = np.random.rand(
        3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation


def uniform_sample_angle() -> np.ndarray:
    """Random rotation matrix generated from uniformly distributed axis and angle

    You can get uniformly distributed RRE with this function

    NOTE: Rotation generated by this function is NOT uniformly distributed on SO(3)

    Axis is uniformly distributed on unit shpere

    Angle is uniformly distributed over the interval [0, 2*pi)
    
    """
    # axis-angle
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    theta = 2 * np.pi * np.random.rand()
    rotvec = axis * theta
    rotation = Rotation.from_rotvec(rotvec).as_matrix()

    return rotation


def uniform_sample_rotation() -> np.ndarray:
    """Random rotation matrix generated from QR decomposition

    Rotation generated by this function is uniformly distributed on SO(3) w.r.t Haar measure

    NOTE: RRE of the rotation generated by this function is NOT uniformly distributed
    
    """
    # QR decomposition
    z = np.random.randn(3, 3)
    while np.linalg.matrix_rank(z) != z.shape[0]:
        z = np.random.randn(3, 3)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = np.diag(d / np.absolute(d))
    q = np.matmul(q, ph)
    # # if det(rotation) == -1, project q to so(3)
    # #rotation = np.linalg.det(q) * q # det(q) 有误差，用乘法可能放大误差 而用除法则可以一定程度上修正误差
    rotation = q / np.linalg.det(
        q)  # 虽然根据行列式数乘的性质，除以det(q)的3次方根更合适，但是感觉从精度上讲没有必要

    return rotation


def get_augmentation_rotation(rotation_factor, aug_rot_type) -> np.ndarray:
    if aug_rot_type == "eular":
        aug_rotation = random_sample_rotation(rotation_factor)
    elif aug_rot_type == "angle":
        aug_rotation = uniform_sample_angle()
    elif aug_rot_type == "haar":
        aug_rotation = uniform_sample_rotation()
    else:
        raise NotImplementedError(
            "Augmentation type of rotation is NOT Implemented!")

    return aug_rotation


def random_sample_translation(translation_factor: float = 5.0) -> np.ndarray:
    """Random translation with uniformly distributed xyz

    NOTE: RTE of translation generated by this function is NOT uniformly distributed

    """
    # uniform x, y, z
    translation = 2 * (np.random.rand(3) - 0.5) * translation_factor
    return translation


def uniform_sample_translation(translation_factor: float = 5.0) -> np.ndarray:
    """Random translation uniformly distributed in sphere

    You can get uniformly distributed RTE with this function

    """
    # uniform in sphere
    unit = np.random.randn(3)
    unit = unit / np.linalg.norm(unit)
    translation = unit * np.random.rand() * translation_factor
    return translation


def gaussian_sample_translation(translation_factor: float = 1.0) -> np.ndarray:
    """Random translation uniformly distributed in sphere
    """
    # gaussian
    translation = np.random.randn(3) * translation_factor
    return translation


def exponential_sample_translation(
        translation_factor: float = 1.0) -> np.ndarray:
    """Random translation uniformly distributed in sphere
    """
    # exponential
    unit = np.random.randn(3)
    unit = unit / np.linalg.norm(unit)
    translation = unit * np.random.exponential(translation_factor)
    return translation


def threedmatch_sample_translation() -> np.ndarray:
    """Random translation simulating 3DMatch's distribution
    """
    if np.random.rand() < 0.225:
        translation = gaussian_sample_translation(0.345)
    elif np.random.rand() < 0.625:
        translation = uniform_sample_translation(3)
    else:
        translation = exponential_sample_translation(2 / 3)

    return translation


def get_augmentation_translation(translation_factor,
                                 aug_trans_type) -> np.ndarray:
    if aug_trans_type == "cube":
        aug_translation = random_sample_translation(translation_factor)
    elif aug_trans_type == "sphere":
        aug_translation = uniform_sample_translation(translation_factor)
    elif aug_trans_type == "gauss":
        aug_translation = gaussian_sample_translation(translation_factor)
    elif aug_trans_type == "exp":
        aug_translation = exponential_sample_translation(translation_factor)
    elif aug_trans_type == "threedmatch":
        aug_translation = threedmatch_sample_translation()
    else:
        raise NotImplementedError(
            "Augmentation type of translation is NOT Implemented!")

    return aug_translation


def rand_noise(num, noise_factor: float = 0.005) -> np.ndarray:
    mu = 0.5
    sigma = np.sqrt(1 / 12)
    noise = (np.random.rand(num, 3) - mu) / sigma * noise_factor

    return noise


def randn_noise(num, noise_factor: float = 0.005) -> np.ndarray:
    mu = 0
    sigma = 1
    noise = (np.random.randn(num, 3) - mu) / sigma * noise_factor

    return noise


def get_augmentation_noise(num, noise_factor, aug_noise_type) -> np.ndarray:
    if aug_noise_type == "rand":
        noise = rand_noise(num, noise_factor)
    elif aug_noise_type == "randn":
        noise = randn_noise(num, noise_factor)
    elif aug_noise_type == "origin":
        noise = (np.random.rand(num, 3) - 0.5) * noise_factor
    else:
        raise NotImplementedError(
            "Augmentation type of translation is NOT Implemented!")

    return noise


# def get_correspondences(ref_points, src_points, transform, matching_radius):
#     r"""Find the ground truth correspondences within the matching radius between two point clouds.

#     Return correspondence indices [indices in ref_points, indices in src_points]
#     """
#     src_points = apply_transform(src_points, transform)
#     src_tree = cKDTree(src_points)
#     indices_list = src_tree.query_ball_point(ref_points, matching_radius)
#     corr_indices = np.array(
#         [(i, j) for i, indices in enumerate(indices_list) for j in indices],
#         dtype=np.long,
#     )

#     return corr_indices


def get_correspondences(ref_points, src_points, transform, matching_radius):
    r"""Compute mutual correspondences
    Adapted from compute_overlap in RegTR's source codes

    Returns:
        corr_indices: [indices in ref_points, indices in src_points]
        has_corr_ref: Whether each reference point is in the overlap region
        has_corr_src: Whether each source point is in the overlap region
        mutual_corr_indices: Indices of mutual correspondences [indices in ref_points, indices in src_points]
    """
    src_points = np_transform(transform, src_points)
    src_tree = cKDTree(src_points)
    indices_list = src_tree.query_ball_point(ref_points, matching_radius)

    # correspondence indices
    corr_indices = []

    # Indices for reference's correspondences
    ref_corr = np.full(ref_points.shape[0], -1)
    for i, indices in enumerate(indices_list):
        if len(indices) > 0:
            ref_corr[i] = indices[0]  # src's idx
        for j in indices:
            corr_indices.append((i, j))  # (ref's idx, src's idx)

    ref_tree = cKDTree(ref_points)
    indices_list = ref_tree.query_ball_point(src_points, matching_radius)

    # Indices for source's correspondences
    src_corr = np.full(src_points.shape[0], -1)
    for i, indices in enumerate(indices_list):
        if len(indices) > 0:
            src_corr[i] = indices[0]  # ref's idx

    # Compute mutual correspondences
    src_corr_is_mutual = np.logical_and(ref_corr[src_corr] == np.arange(
        len(src_corr)), src_corr > 0)  # True or False
    mutual_corr_indices = np.stack(
        [src_corr[src_corr_is_mutual],
         np.nonzero(src_corr_is_mutual)[0]])  # (ref's idx, src's idx)

    has_corr_ref = ref_corr >= 0
    has_corr_src = src_corr >= 0

    corr = {}
    corr["corr_indices"] = corr_indices
    corr["has_corr_ref"] = has_corr_ref
    corr["has_corr_src"] = has_corr_src
    corr["mutual_corr_indices"] = mutual_corr_indices

    return corr


def get_overlap_mask(ref_points, src_points, transform, matching_radius):
    r"""Compute overlap region mask

    Returns:
        ref_overlap: Whether each reference point is in the overlap region
        src_overlap: Whether each source point is in the overlap region
    """
    src_points = np_transform(transform, src_points)

    # Mask for reference's overlap region
    src_tree = cKDTree(src_points)
    _, indices_list = src_tree.query(
        ref_points,
        distance_upper_bound=matching_radius,
        workers=-1,
    )
    invalid_index = src_points.shape[0]
    ref_overlap = indices_list < invalid_index

    # Mask for source's overlap region
    ref_tree = cKDTree(ref_points)
    _, indices_list = ref_tree.query(
        src_points,
        distance_upper_bound=matching_radius,
        workers=-1,
    )
    invalid_index = ref_points.shape[0]
    src_overlap = indices_list < invalid_index

    return ref_overlap, src_overlap


def radius_search(q_points, s_points, q_lengths, s_lengths, radius,
                  neighbor_limit):
    r"""Computes neighbors for a batch of q_points and s_points, apply radius search (in stack mode).

    This function is implemented on CPU.

    Args:
        q_points (Tensor): the query points (N, 3)
        s_points (Tensor): the support points (M, 3)
        q_lengths (Tensor): the list of lengths of batch elements in q_points
        s_lengths (Tensor): the list of lengths of batch elements in s_points
        radius (float): maximum distance of neighbors
        neighbor_limit (int): maximum number of neighbors

    Returns:
        neighbors (Tensor): the k nearest neighbors of q_points in s_points (N, k).
            Filled with M if there are less than k neighbors.
    """
    neighbor_indices = ext_module.radius_neighbors(q_points, s_points,
                                                   q_lengths, s_lengths,
                                                   radius)
    if neighbor_limit > 0:
        neighbor_indices = neighbor_indices[:, :neighbor_limit]
    return neighbor_indices
