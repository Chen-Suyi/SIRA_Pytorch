import logging
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional
from common.utils.func import pairwise_distance
from common.utils.se3 import apply_transform
from model.kpconv import ConvBlock, ResidualBlock, UnaryBlock, LastUnaryBlock, nearest_upsample
from model.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer


# Define some sub-modules of model
# #####################################################################################
class KPConvFPN(nn.Module):

    def __init__(self, input_dim, output_dim, init_dim, kernel_size,
                 init_radius, init_sigma, group_norm):
        super(KPConvFPN, self).__init__()

        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size,
                                    init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim * 2, kernel_size,
                                        init_radius, init_sigma, group_norm)

        self.encoder2_1 = ResidualBlock(init_dim * 2,
                                        init_dim * 2,
                                        kernel_size,
                                        init_radius,
                                        init_sigma,
                                        group_norm,
                                        strided=True)
        self.encoder2_2 = ResidualBlock(init_dim * 2, init_dim * 4,
                                        kernel_size, init_radius * 2,
                                        init_sigma * 2, group_norm)
        self.encoder2_3 = ResidualBlock(init_dim * 4, init_dim * 4,
                                        kernel_size, init_radius * 2,
                                        init_sigma * 2, group_norm)

        self.encoder3_1 = ResidualBlock(init_dim * 4,
                                        init_dim * 4,
                                        kernel_size,
                                        init_radius * 2,
                                        init_sigma * 2,
                                        group_norm,
                                        strided=True)
        self.encoder3_2 = ResidualBlock(init_dim * 4, init_dim * 8,
                                        kernel_size, init_radius * 4,
                                        init_sigma * 4, group_norm)
        self.encoder3_3 = ResidualBlock(init_dim * 8, init_dim * 8,
                                        kernel_size, init_radius * 4,
                                        init_sigma * 4, group_norm)

        self.encoder4_1 = ResidualBlock(init_dim * 8,
                                        init_dim * 8,
                                        kernel_size,
                                        init_radius * 4,
                                        init_sigma * 4,
                                        group_norm,
                                        strided=True)
        self.encoder4_2 = ResidualBlock(init_dim * 8, init_dim * 16,
                                        kernel_size, init_radius * 8,
                                        init_sigma * 8, group_norm)
        self.encoder4_3 = ResidualBlock(init_dim * 16, init_dim * 16,
                                        kernel_size, init_radius * 8,
                                        init_sigma * 8, group_norm)

        self.decoder3 = UnaryBlock(init_dim * 24, init_dim * 8, group_norm)
        self.decoder2 = LastUnaryBlock(init_dim * 12, output_dim)

    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']

        # print("index:\n", data_dict['index'])
        # print("points_list:\n", data_dict['points'][:3])
        # print("neighbors:\n", data_dict['neighbors'][:3])

        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0],
                                   neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0],
                                   neighbors_list[0])

        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0],
                                   subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1],
                                   neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1],
                                   neighbors_list[1])

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1],
                                   subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2],
                                   neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2],
                                   neighbors_list[2])

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2],
                                   subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3],
                                   neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3],
                                   neighbors_list[3])

        latent_s4 = feats_s4
        feats_list.append(feats_s4)

        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)
        feats_list.append(latent_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)
        feats_list.append(latent_s2)

        feats_list.reverse()

        return feats_list


class GeometricStructureEmbedding(nn.Module):

    def __init__(self,
                 hidden_dim,
                 sigma_d,
                 sigma_a,
                 angle_k,
                 reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(
                f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2,
                                    largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k,
                                                      3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point,
                                                     num_point,
                                                     3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2,
                                  index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point,
                                                      num_point, k,
                                                      3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point,
                                                      num_point, k,
                                                      3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors,
                                                   anc_vectors,
                                                   dim=-1),
                                       dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors,
                               dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        embeddings = d_embeddings + a_embeddings

        return embeddings


class GeometricTransformer(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a,
        angle_k,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(GeometricTransformer, self).__init__()

        self.embedding = GeometricStructureEmbedding(hidden_dim,
                                                     sigma_d,
                                                     sigma_a,
                                                     angle_k,
                                                     reduction_a=reduction_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = RPEConditionalTransformer(
            blocks,
            hidden_dim,
            num_heads,
            dropout=dropout,
            activation_fn=activation_fn)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_masks=None,
        src_masks=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_embeddings = self.embedding(ref_points)
        src_embeddings = self.embedding(src_points)

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats


class SuperPointTargetGenerator(nn.Module):

    def __init__(self, num_targets, overlap_threshold):
        super(SuperPointTargetGenerator, self).__init__()
        self.num_targets = num_targets
        self.overlap_threshold = overlap_threshold

    @torch.no_grad()
    def forward(self, gt_corr_indices, gt_corr_overlaps):
        r"""Generate ground truth superpoint (patch) correspondences.

        Randomly select "num_targets" correspondences whose overlap is above "overlap_threshold".

        Args:
            gt_corr_indices (LongTensor): ground truth superpoint correspondences (N, 2)
            gt_corr_overlaps (Tensor): ground truth superpoint correspondences overlap (N,)

        Returns:
            gt_ref_corr_indices (LongTensor): selected superpoints in reference point cloud.
            gt_src_corr_indices (LongTensor): selected superpoints in source point cloud.
            gt_corr_overlaps (LongTensor): overlaps of the selected superpoint correspondences.
        """
        gt_corr_masks = torch.gt(gt_corr_overlaps, self.overlap_threshold)
        gt_corr_overlaps = gt_corr_overlaps[gt_corr_masks]
        gt_corr_indices = gt_corr_indices[gt_corr_masks]

        if gt_corr_indices.shape[0] > self.num_targets:
            indices = np.arange(gt_corr_indices.shape[0])
            sel_indices = np.random.choice(indices,
                                           self.num_targets,
                                           replace=False)
            sel_indices = torch.from_numpy(sel_indices).cuda()
            gt_corr_indices = gt_corr_indices[sel_indices]
            gt_corr_overlaps = gt_corr_overlaps[sel_indices]

        gt_ref_corr_indices = gt_corr_indices[:, 0]
        gt_src_corr_indices = gt_corr_indices[:, 1]

        return gt_ref_corr_indices, gt_src_corr_indices, gt_corr_overlaps


class SuperPointMatching(nn.Module):

    def __init__(self, num_correspondences, dual_normalization=True):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0], ),
                                   dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0], ),
                                   dtype=torch.bool).cuda()
        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        # select top-k proposals
        matching_scores = torch.exp(
            -pairwise_distance(ref_feats, src_feats, normalized=True))
        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(
                dim=1, keepdim=True)
            src_matching_scores = matching_scores / matching_scores.sum(
                dim=0, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores

        num_correspondences = min(self.num_correspondences,
                                  matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(
            k=num_correspondences, largest=True)
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        # recover original indices
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]

        return ref_corr_indices, src_corr_indices, corr_scores


def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-5,
    return_transform=False,
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh),
                          torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    src_centroid = torch.sum(src_points * weights, dim=1,
                             keepdim=True)  # (B, 1, 3)
    ref_centroid = torch.sum(ref_points * weights, dim=1,
                             keepdim=True)  # (B, 1, 3)
    src_points_centered = src_points - src_centroid  # (B, N, 3)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    U, _, V = torch.svd(H.cpu())  # H = USV^T
    Ut, V = U.transpose(1, 2).cuda(), V.cuda()
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t


class WeightedProcrustes(nn.Module):

    def __init__(self, weight_thresh=0.0, eps=1e-5, return_transform=False):
        super(WeightedProcrustes, self).__init__()
        self.weight_thresh = weight_thresh
        self.eps = eps
        self.return_transform = return_transform

    def forward(self, src_points, ref_points, weights=None):
        return weighted_procrustes(
            src_points,
            ref_points,
            weights=weights,
            weight_thresh=self.weight_thresh,
            eps=self.eps,
            return_transform=self.return_transform,
        )


class LocalGlobalRegistration(nn.Module):

    def __init__(
        self,
        k: int,
        acceptance_radius: float,
        mutual: bool = True,
        confidence_threshold: float = 0.05,
        use_dustbin: bool = False,
        use_global_score: bool = False,
        correspondence_threshold: int = 3,
        correspondence_limit: Optional[int] = None,
        num_refinement_steps: int = 5,
    ):
        r"""Point Matching with Local-to-Global Registration.

        Args:
            k (int): top-k selection for matching.
            acceptance_radius (float): acceptance radius for LGR.
            mutual (bool=True): mutual or non-mutual matching.
            confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
            use_dustbin (bool=False): whether dustbin row/column is used in the score matrix.
            use_global_score (bool=False): whether use patch correspondence scores.
            correspondence_threshold (int=3): minimal number of correspondences for each patch correspondence.
            correspondence_limit (optional[int]=None): maximal number of verification correspondences.
            num_refinement_steps (int=5): number of refinement steps.
        """
        super(LocalGlobalRegistration, self).__init__()
        self.k = k
        self.acceptance_radius = acceptance_radius
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score
        self.correspondence_threshold = correspondence_threshold
        self.correspondence_limit = correspondence_limit
        self.num_refinement_steps = num_refinement_steps
        self.procrustes = WeightedProcrustes(return_transform=True)

    def compute_correspondence_matrix(self, score_mat, ref_knn_masks,
                                      src_knn_masks):
        r"""Compute matching matrix and score matrix for each patch correspondence."""
        mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2),
                                     src_knn_masks.unsqueeze(1))

        batch_size, ref_length, src_length = score_mat.shape
        batch_indices = torch.arange(batch_size).cuda()

        # correspondences from reference side
        ref_topk_scores, ref_topk_indices = score_mat.topk(k=self.k,
                                                           dim=2)  # (B, N, K)
        ref_batch_indices = batch_indices.view(batch_size, 1, 1).expand(
            -1, ref_length, self.k)  # (B, N, K)
        ref_indices = torch.arange(ref_length).cuda().view(
            1, ref_length, 1).expand(batch_size, -1, self.k)  # (B, N, K)
        ref_score_mat = torch.zeros_like(score_mat)
        ref_score_mat[ref_batch_indices, ref_indices,
                      ref_topk_indices] = ref_topk_scores
        ref_corr_mat = torch.gt(ref_score_mat, self.confidence_threshold)

        # correspondences from source side
        src_topk_scores, src_topk_indices = score_mat.topk(k=self.k,
                                                           dim=1)  # (B, K, N)
        src_batch_indices = batch_indices.view(batch_size, 1, 1).expand(
            -1, self.k, src_length)  # (B, K, N)
        src_indices = torch.arange(src_length).cuda().view(
            1, 1, src_length).expand(batch_size, self.k, -1)  # (B, K, N)
        src_score_mat = torch.zeros_like(score_mat)
        src_score_mat[src_batch_indices, src_topk_indices,
                      src_indices] = src_topk_scores
        src_corr_mat = torch.gt(src_score_mat, self.confidence_threshold)

        # merge results from two sides
        if self.mutual:
            corr_mat = torch.logical_and(ref_corr_mat, src_corr_mat)
        else:
            corr_mat = torch.logical_or(ref_corr_mat, src_corr_mat)

        if self.use_dustbin:
            corr_mat = corr_mat[:, -1:, -1]

        corr_mat = torch.logical_and(corr_mat, mask_mat)

        return corr_mat

    @staticmethod
    def convert_to_batch(ref_corr_points, src_corr_points, corr_scores,
                         chunks):
        r"""Convert stacked correspondences to batched points.

        The extracted dense correspondences from all patch correspondences are stacked. However, to compute the
        transformations from all patch correspondences in parallel, the dense correspondences need to be reorganized
        into a batch.

        Args:
            ref_corr_points (Tensor): (C, 3)
            src_corr_points (Tensor): (C, 3)
            corr_scores (Tensor): (C,)
            chunks (List[Tuple[int, int]]): the starting index and ending index of each patch correspondences.

        Returns:
            batch_ref_corr_points (Tensor): (B, K, 3), padded with zeros.
            batch_src_corr_points (Tensor): (B, K, 3), padded with zeros.
            batch_corr_scores (Tensor): (B, K), padded with zeros.
        """
        batch_size = len(chunks)
        indices = torch.cat([torch.arange(x, y) for x, y in chunks],
                            dim=0).cuda()
        ref_corr_points = ref_corr_points[indices]  # (total, 3)
        src_corr_points = src_corr_points[indices]  # (total, 3)
        corr_scores = corr_scores[indices]  # (total,)

        max_corr = np.max([y - x for x, y in chunks])
        target_chunks = [(i * max_corr, i * max_corr + y - x)
                         for i, (x, y) in enumerate(chunks)]
        indices = torch.cat([torch.arange(x, y) for x, y in target_chunks],
                            dim=0).cuda()
        indices0 = indices.unsqueeze(1).expand(indices.shape[0],
                                               3)  # (total,) -> (total, 3)
        indices1 = torch.arange(3).unsqueeze(0).expand(
            indices.shape[0], 3).cuda()  # (3,) -> (total, 3)

        batch_ref_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
        batch_ref_corr_points.index_put_([indices0, indices1], ref_corr_points)
        batch_ref_corr_points = batch_ref_corr_points.view(
            batch_size, max_corr, 3)

        batch_src_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
        batch_src_corr_points.index_put_([indices0, indices1], src_corr_points)
        batch_src_corr_points = batch_src_corr_points.view(
            batch_size, max_corr, 3)

        batch_corr_scores = torch.zeros(batch_size * max_corr).cuda()
        batch_corr_scores.index_put_([indices], corr_scores)
        batch_corr_scores = batch_corr_scores.view(batch_size, max_corr)

        return batch_ref_corr_points, batch_src_corr_points, batch_corr_scores

    def recompute_correspondence_scores(self, ref_corr_points, src_corr_points,
                                        corr_scores, estimated_transform):
        aligned_src_corr_points = apply_transform(src_corr_points,
                                                  estimated_transform)
        corr_residuals = torch.linalg.norm(ref_corr_points -
                                           aligned_src_corr_points,
                                           dim=1)
        inlier_masks = torch.lt(corr_residuals, self.acceptance_radius)
        new_corr_scores = corr_scores * inlier_masks.float()
        return new_corr_scores

    def local_to_global_registration(self, ref_knn_points, src_knn_points,
                                     score_mat, corr_mat):
        # extract dense correspondences
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat,
                                                                as_tuple=True)
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]

        # build verification set
        if self.correspondence_limit is not None and global_corr_scores.shape[
                0] > self.correspondence_limit:
            corr_scores, sel_indices = global_corr_scores.topk(
                k=self.correspondence_limit, largest=True)
            ref_corr_points = global_ref_corr_points[sel_indices]
            src_corr_points = global_src_corr_points[sel_indices]
        else:
            ref_corr_points = global_ref_corr_points
            src_corr_points = global_src_corr_points
            corr_scores = global_corr_scores

        # compute starting and ending index of each patch correspondence.
        # torch.nonzero is row-major, so the correspondences from the same patch correspondence are consecutive.
        # find the first occurrence of each batch index, then the chunk of this batch can be obtained.
        unique_masks = torch.ne(batch_indices[1:], batch_indices[:-1])
        unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
        unique_indices = unique_indices.detach().cpu().numpy().tolist()
        unique_indices = [0] + unique_indices + [batch_indices.shape[0]]
        chunks = [(x, y)
                  for x, y in zip(unique_indices[:-1], unique_indices[1:])
                  if y - x >= self.correspondence_threshold]

        batch_size = len(chunks)
        if batch_size > 0:
            # local registration
            batch_ref_corr_points, batch_src_corr_points, batch_corr_scores = self.convert_to_batch(
                global_ref_corr_points, global_src_corr_points,
                global_corr_scores, chunks)
            batch_transforms = self.procrustes(batch_src_corr_points,
                                               batch_ref_corr_points,
                                               batch_corr_scores)
            batch_aligned_src_corr_points = apply_transform(
                src_corr_points.unsqueeze(0), batch_transforms)
            batch_corr_residuals = torch.linalg.norm(
                ref_corr_points.unsqueeze(0) - batch_aligned_src_corr_points,
                dim=2)
            batch_inlier_masks = torch.lt(batch_corr_residuals,
                                          self.acceptance_radius)  # (P, N)
            best_index = batch_inlier_masks.sum(dim=1).argmax()
            cur_corr_scores = corr_scores * batch_inlier_masks[
                best_index].float()
        else:
            # degenerate: initialize transformation with all correspondences
            estimated_transform = self.procrustes(src_corr_points,
                                                  ref_corr_points, corr_scores)
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores,
                estimated_transform)

        # global refinement
        estimated_transform = self.procrustes(src_corr_points, ref_corr_points,
                                              cur_corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores,
                estimated_transform)
            estimated_transform = self.procrustes(src_corr_points,
                                                  ref_corr_points,
                                                  cur_corr_scores)

        return global_ref_corr_points, global_src_corr_points, global_corr_scores, estimated_transform

    def forward(
        self,
        ref_knn_points,
        src_knn_points,
        ref_knn_masks,
        src_knn_masks,
        score_mat,
        global_scores,
    ):
        r"""Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            ref_knn_points (Tensor): (B, K, 3)
            src_knn_points (Tensor): (B, K, 3)
            ref_knn_masks (BoolTensor): (B, K)
            src_knn_masks (BoolTensor): (B, K)
            score_mat (Tensor): (B, K, K) or (B, K + 1, K + 1), log likelihood
            global_scores (Tensor): (B,)

        Returns:
            ref_corr_points: torch.LongTensor (C, 3)
            src_corr_points: torch.LongTensor (C, 3)
            corr_scores: torch.Tensor (C,)
            estimated_transform: torch.Tensor (4, 4)
        """
        score_mat = torch.exp(score_mat)

        corr_mat = self.compute_correspondence_matrix(
            score_mat, ref_knn_masks, src_knn_masks)  # (B, K, K)

        if self.use_dustbin:
            score_mat = score_mat[:, :-1, :-1]
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.float()

        ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.local_to_global_registration(
            ref_knn_points, src_knn_points, score_mat, corr_mat)

        return ref_corr_points, src_corr_points, corr_scores, estimated_transform


class LearnableLogOptimalTransport(nn.Module):

    def __init__(self, num_iterations, inf=1e12):
        r"""Sinkhorn Optimal transport with dustbin parameter (SuperGlue style)."""
        super(LearnableLogOptimalTransport, self).__init__()
        self.num_iterations = num_iterations
        self.register_parameter('alpha', torch.nn.Parameter(torch.tensor(1.0)))
        self.inf = inf

    def log_sinkhorn_normalization(self, scores, log_mu, log_nu):
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.num_iterations):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
        return scores + u.unsqueeze(2) + v.unsqueeze(1)

    def forward(self, scores, row_masks=None, col_masks=None):
        r"""Sinkhorn Optimal Transport (SuperGlue style) forward.

        Args:
            scores: torch.Tensor (B, M, N)
            row_masks: torch.Tensor (B, M)
            col_masks: torch.Tensor (B, N)

        Returns:
            matching_scores: torch.Tensor (B, M+1, N+1)
        """
        batch_size, num_row, num_col = scores.shape

        if row_masks is None:
            row_masks = torch.ones(size=(batch_size, num_row),
                                   dtype=torch.bool).cuda()
        if col_masks is None:
            col_masks = torch.ones(size=(batch_size, num_col),
                                   dtype=torch.bool).cuda()

        padded_row_masks = torch.zeros(size=(batch_size, num_row + 1),
                                       dtype=torch.bool).cuda()
        padded_row_masks[:, :num_row] = ~row_masks
        padded_col_masks = torch.zeros(size=(batch_size, num_col + 1),
                                       dtype=torch.bool).cuda()
        padded_col_masks[:, :num_col] = ~col_masks
        padded_score_masks = torch.logical_or(padded_row_masks.unsqueeze(2),
                                              padded_col_masks.unsqueeze(1))

        padded_col = self.alpha.expand(batch_size, num_row, 1)
        padded_row = self.alpha.expand(batch_size, 1, num_col + 1)
        padded_scores = torch.cat(
            [torch.cat([scores, padded_col], dim=-1), padded_row], dim=1)
        padded_scores.masked_fill_(padded_score_masks, -self.inf)

        num_valid_row = row_masks.float().sum(1)
        num_valid_col = col_masks.float().sum(1)
        norm = -torch.log(num_valid_row + num_valid_col)  # (B,)

        log_mu = torch.empty(size=(batch_size, num_row + 1)).cuda()
        log_mu[:, :num_row] = norm.unsqueeze(1)
        log_mu[:, num_row] = torch.log(num_valid_col) + norm
        log_mu[padded_row_masks] = -self.inf

        log_nu = torch.empty(size=(batch_size, num_col + 1)).cuda()
        log_nu[:, :num_col] = norm.unsqueeze(1)
        log_nu[:, num_col] = torch.log(num_valid_row) + norm
        log_nu[padded_col_masks] = -self.inf

        outputs = self.log_sinkhorn_normalization(padded_scores, log_mu,
                                                  log_nu)
        outputs = outputs - norm.unsqueeze(1).unsqueeze(2)

        return outputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '(num_iterations={})'.format(
            self.num_iterations)
        return format_string
