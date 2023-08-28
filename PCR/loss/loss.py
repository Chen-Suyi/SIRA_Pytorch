import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import open3d as o3d
from torch.nn import functional as F
from common.tool import tensor_gpu
from common.utils.func import pairwise_distance, isotropic_transform_error
from common.utils.se3 import apply_transform, list_apply_transform, inverse_transform, complete_transform, incomplete_transform


def circle_loss(
    pos_masks,
    neg_masks,
    feat_dists,
    pos_margin,
    neg_margin,
    pos_optimal,
    neg_optimal,
    log_scale,
):
    # get anchors that have both positive and negative pairs
    row_masks = (torch.gt(pos_masks.sum(-1), 0)
                 & torch.gt(neg_masks.sum(-1), 0)).detach()
    col_masks = (torch.gt(pos_masks.sum(-2), 0)
                 & torch.gt(neg_masks.sum(-2), 0)).detach()

    # get alpha for both positive and negative pairs
    pos_weights = feat_dists - 1e5 * (
        ~pos_masks).float()  # mask the non-positive
    pos_weights = pos_weights - pos_optimal  # mask the uninformative positive
    pos_weights = torch.maximum(torch.zeros_like(pos_weights),
                                pos_weights).detach()

    neg_weights = feat_dists + 1e5 * (
        ~neg_masks).float()  # mask the non-negative
    neg_weights = neg_optimal - neg_weights  # mask the uninformative negative
    neg_weights = torch.maximum(torch.zeros_like(neg_weights),
                                neg_weights).detach()

    loss_pos_row = torch.logsumexp(log_scale * (feat_dists - pos_margin) *
                                   pos_weights,
                                   dim=-1)
    loss_pos_col = torch.logsumexp(log_scale * (feat_dists - pos_margin) *
                                   pos_weights,
                                   dim=-2)

    loss_neg_row = torch.logsumexp(log_scale * (neg_margin - feat_dists) *
                                   neg_weights,
                                   dim=-1)
    loss_neg_col = torch.logsumexp(log_scale * (neg_margin - feat_dists) *
                                   neg_weights,
                                   dim=-2)

    loss_row = F.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = F.softplus(loss_pos_col + loss_neg_col) / log_scale

    loss = (loss_row[row_masks].mean() + loss_col[col_masks].mean()) / 2

    return loss


def weighted_circle_loss(
    pos_masks,
    neg_masks,
    feat_dists,
    pos_margin,
    neg_margin,
    pos_optimal,
    neg_optimal,
    log_scale,
    pos_scales=None,
    neg_scales=None,
):
    # get anchors that have both positive and negative pairs
    row_masks = (torch.gt(pos_masks.sum(-1), 0)
                 & torch.gt(neg_masks.sum(-1), 0)).detach()
    col_masks = (torch.gt(pos_masks.sum(-2), 0)
                 & torch.gt(neg_masks.sum(-2), 0)).detach()

    # get alpha for both positive and negative pairs
    pos_weights = feat_dists - 1e5 * (
        ~pos_masks).float()  # mask the non-positive
    pos_weights = pos_weights - pos_optimal  # mask the uninformative positive
    pos_weights = torch.maximum(torch.zeros_like(pos_weights), pos_weights)
    if pos_scales is not None:
        pos_weights = pos_weights * pos_scales
    pos_weights = pos_weights.detach()

    neg_weights = feat_dists + 1e5 * (
        ~neg_masks).float()  # mask the non-negative
    neg_weights = neg_optimal - neg_weights  # mask the uninformative negative
    neg_weights = torch.maximum(torch.zeros_like(neg_weights), neg_weights)
    if neg_scales is not None:
        neg_weights = neg_weights * neg_scales
    neg_weights = neg_weights.detach()

    loss_pos_row = torch.logsumexp(log_scale * (feat_dists - pos_margin) *
                                   pos_weights,
                                   dim=-1)
    loss_pos_col = torch.logsumexp(log_scale * (feat_dists - pos_margin) *
                                   pos_weights,
                                   dim=-2)

    loss_neg_row = torch.logsumexp(log_scale * (neg_margin - feat_dists) *
                                   neg_weights,
                                   dim=-1)
    loss_neg_col = torch.logsumexp(log_scale * (neg_margin - feat_dists) *
                                   neg_weights,
                                   dim=-2)

    loss_row = F.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = F.softplus(loss_pos_col + loss_neg_col) / log_scale

    loss = (loss_row[row_masks].mean() + loss_col[col_masks].mean()) / 2

    return loss


class CircleLoss(nn.Module):

    def __init__(self, pos_margin, neg_margin, pos_optimal, neg_optimal,
                 log_scale):
        super(CircleLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def forward(self, pos_masks, neg_masks, feat_dists):
        return circle_loss(
            pos_masks,
            neg_masks,
            feat_dists,
            self.pos_margin,
            self.neg_margin,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
        )


class WeightedCircleLoss(nn.Module):

    def __init__(self, pos_margin, neg_margin, pos_optimal, neg_optimal,
                 log_scale):
        super(WeightedCircleLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def forward(self,
                pos_masks,
                neg_masks,
                feat_dists,
                pos_scales=None,
                neg_scales=None):
        return weighted_circle_loss(
            pos_masks,
            neg_masks,
            feat_dists,
            self.pos_margin,
            self.neg_margin,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
            pos_scales=pos_scales,
            neg_scales=neg_scales,
        )


class CoarseMatchingLoss(nn.Module):

    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.cfg = cfg
        self.weighted_circle_loss = WeightedCircleLoss(
            self.cfg.coarse_loss.positive_margin,
            self.cfg.coarse_loss.negative_margin,
            self.cfg.coarse_loss.positive_optimal,
            self.cfg.coarse_loss.negative_optimal,
            self.cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = self.cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict["ref_feats_c"]
        src_feats = output_dict["src_feats_c"]
        gt_node_corr_indices = output_dict["gt_node_corr_indices"]
        gt_node_corr_overlaps = output_dict["gt_node_corr_overlaps"]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(
            pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices,
                 gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists,
                                         pos_scales)

        return loss


class FineMatchingLoss(nn.Module):

    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.cfg = cfg
        self.positive_radius = self.cfg.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict["ref_node_corr_knn_points"]
        src_node_corr_knn_points = output_dict["src_node_corr_knn_points"]
        ref_node_corr_knn_masks = output_dict["ref_node_corr_knn_masks"]
        src_node_corr_knn_masks = output_dict["src_node_corr_knn_masks"]
        matching_scores = output_dict["matching_scores"]
        transform = data_dict["transform"]

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points,
                                                   transform)
        dists = pairwise_distance(ref_node_corr_knn_points,
                                  src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2),
                                     src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius**2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0),
                                             ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0),
                                             src_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss


def cdist(a, b, metric="euclidean"):
    """Similar to scipy.spatial"s cdist, but symbolic.
    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - "euclidean", although with a fudge-factor epsilon.
        - "sqeuclidean", the squared euclidean.
        - "cityblock", the manhattan or L1 distance.
    Args:
        a: The left-hand side, shaped ([*,] F, B1).  <- Not that dimension ordering is different from torch.cdist
        b: The right-hand side, shaped ([*,], F, B2).
        metric (string): Which distance metric to use, see notes.
    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).
    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.

    Taken from Predator source code, which was modified from D3Feat.
    """
    if metric == "sqeuclidean":
        diffs = a[..., :, None] - b[..., None, :]
        return torch.sum(diffs**2, dim=-3)
    elif metric == "euclidean":
        diffs = a[..., :, None] - b[..., None, :]
        return torch.sqrt(torch.sum(diffs**2, dim=-3) + 1e-12)
    elif metric == "cityblock":
        diffs = a[..., :, None] - b[..., None, :]
        return torch.sum(torch.abs(diffs), dim=-3)
    elif metric == "cosine":
        numer = a.transpose(-1, -2) @ b
        denom = torch.clamp_min(
            torch.norm(a, dim=-2)[..., :, None] *
            torch.norm(b, dim=-2)[..., None, :], 1e-8)
        dist = 1 - numer / denom
        return dist
    else:
        raise NotImplementedError(
            "The following metric is not implemented by `cdist` yet: {}".
            format(metric))


def make_open3d_point_cloud(points, colors=None, normals=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().detach().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def registration_with_ransac_from_correspondences(
    src_points,
    ref_points,
    correspondences=None,
    distance_threshold=0.05,
    ransac_n=3,
    num_iterations=10000,
):
    r"""
    Compute the transformation matrix from src_points to ref_points
    """
    src_pcd = make_open3d_point_cloud(src_points)
    ref_pcd = make_open3d_point_cloud(ref_points)

    if correspondences is None:
        indices = np.arange(src_points.shape[0])
        correspondences = np.stack([indices, indices], axis=1)
    correspondences = o3d.utility.Vector2iVector(correspondences)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_pcd,
        ref_pcd,
        correspondences,
        distance_threshold,
        estimation_method=o3d.pipelines.registration.
        TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            num_iterations, num_iterations),
    )

    return result.transformation


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def compute_loss(cfg, input, output):

    loss = {}
    if cfg.model_name == "geotransformer":

        coarse_loss = CoarseMatchingLoss(cfg)
        fine_loss = FineMatchingLoss(cfg)
        weight_coarse_loss = cfg.loss.weight_coarse_loss
        weight_fine_loss = cfg.loss.weight_fine_loss

        loss["coarse"] = coarse_loss(output)
        loss["fine"] = fine_loss(output, input)
        loss["total"] = loss["coarse"] * weight_coarse_loss + loss[
            "fine"] * weight_fine_loss
        loss["total"] = reduce_mean(loss["total"], cfg.world_size)

    else:
        raise NotImplementedError

    return loss


def compute_metric(cfg, input, output):
    metric = {}
    if cfg.model_name == "geotransformer":
        ref_length_c = output["ref_points_c"].shape[0]
        src_length_c = output["src_points_c"].shape[0]
        gt_node_corr_overlaps = output["gt_node_corr_overlaps"]
        gt_node_corr_indices = output["gt_node_corr_indices"]
        masks = torch.gt(gt_node_corr_overlaps, cfg.eval.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices,
                         gt_src_node_corr_indices] = 1.0
        ref_node_corr_indices = output["ref_node_corr_indices"]
        src_node_corr_indices = output["src_node_corr_indices"]
        c_precision = gt_node_corr_map[ref_node_corr_indices,
                                       src_node_corr_indices].mean()

        transform = input["transform"]
        ref_corr_points = output["ref_corr_points"]
        src_corr_points = output["src_corr_points"]
        # est_transform = registration_with_ransac_from_correspondences(
        #     src_corr_points.cpu().detach().numpy(),
        #     ref_corr_points.cpu().detach().numpy(),
        #     distance_threshold=cfg.ransac.distance_threshold,
        #     ransac_n=cfg.ransac.num_points,
        #     num_iterations=cfg.ransac.num_iterations,
        # )
        # est_transform = torch.Tensor(est_transform).cuda()
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points,
                                           dim=1)
        f_precision = torch.lt(corr_distances,
                               cfg.eval.acceptance_radius).float().mean()

        est_transform = output["estimated_transform"]
        src_points = output["src_points"]
        rre, rte = isotropic_transform_error(transform, est_transform)
        realignment_transform = torch.matmul(torch.inverse(transform),
                                             est_transform)
        realigned_src_points_f = apply_transform(src_points,
                                                 realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points_f - src_points,
                                 dim=1).mean()
        recall = torch.lt(rmse, cfg.eval.acceptance_rmse).float()

        ori_rre, ori_rte = isotropic_transform_error(
            transform, torch.eye(4, device=transform.device))

        metric["PIR"] = c_precision
        metric["IR"] = f_precision
        metric["RRE"] = rre
        metric["RTE"] = rte
        metric["RMSE"] = rmse
        metric["RR"] = recall
        metric["score"] = rre + rte
        metric["ori_RRE"] = ori_rre
        metric["ori_RTE"] = ori_rte

    else:
        raise NotImplementedError

    if cfg.world_size > 1:
        for k, v in metric.items():
            metric[k] = reduce_mean(v, cfg.world_size)

    metric = tensor_gpu(metric, local_rank=cfg.local_rank, check_on=False)

    return metric
