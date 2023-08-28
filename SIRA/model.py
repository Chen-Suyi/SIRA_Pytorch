from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d
import numpy as np
import itertools
import time

from kpconv.modules import *


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


class PatchResampleBlock(nn.Module):

    def __init__(self, feat_channels) -> None:
        r"""Initialize a patch resample block.

        Args:
            feat_channels: dimension of input features
        """
        super(PatchResampleBlock, self).__init__()
        self.feat_channels = feat_channels
        self.feat_proj = nn.Linear(self.feat_channels, self.feat_channels)

    def forward(self, points, feats, neighbor_indices):
        point_num, neighbor_limit = neighbor_indices.shape

        # adjust neighbor_indices (stand still when the patch has few neighbors)
        point_indices = torch.arange(point_num,
                                     device=neighbor_indices.device).reshape(
                                         (point_num, 1)).repeat(
                                             (1, neighbor_limit))  # (N ,K)
        neighbor_indices = torch.where(neighbor_indices < point_num,
                                       neighbor_indices, point_indices)

        # feature projection
        feats = self.feat_proj(feats)
        neighbor_feats = feats[neighbor_indices]  # (N, K, d)
        neighbor_points = points[neighbor_indices]  # (N, K, 3)

        neighbor_weights = torch.einsum(
            "nd,nkd->nk", feats,
            neighbor_feats)  # (N, d) x (N, K, d) -> (N, K)
        neighbor_weights = nn.functional.softmax(neighbor_weights /
                                                 self.feat_channels**0.5,
                                                 dim=-1)  # (N, K) -> (N, K)

        output_points = torch.einsum(
            "nk,nkp->np", neighbor_weights,
            neighbor_points)  # (N, K) x (N, K, 3) -> (N, 3)

        return output_points


class ResampleKPConvEncoder(nn.Module):

    def __init__(self, input_dim, output_dim, init_dim, kernel_size,
                 init_radius, init_sigma, group_norm):
        super(ResampleKPConvEncoder, self).__init__()

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

        self.resample4 = PatchResampleBlock(feat_channels=init_dim * 16)
        self.decoder4 = UnaryBlock(init_dim * 16 + 3, init_dim * 16,
                                   group_norm)

        self.resample3 = PatchResampleBlock(feat_channels=init_dim * 8)
        self.decoder3 = UnaryBlock(init_dim * 24 + 3, init_dim * 8, group_norm)

        self.resample2 = PatchResampleBlock(feat_channels=init_dim * 4)
        self.decoder2 = UnaryBlock(init_dim * 12 + 3, init_dim * 4, group_norm)

        self.resample1 = PatchResampleBlock(feat_channels=init_dim * 2)
        self.decoder1 = UnaryBlock(init_dim * 6 + 3, output_dim, group_norm)

        self.outputlayer = LastUnaryBlock(output_dim, output_dim)

    def forward(self, data_dict):
        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsamples']
        upsampling_list = data_dict['upsamples']

        feats_s1 = torch.ones((points_list[0].shape[0], 1),
                              dtype=torch.float32).to(points_list[0])
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0],
                                   neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0],
                                   neighbors_list[0])

        feats_s2 = feats_s1
        feats_s2 = self.encoder2_1(feats_s2, points_list[1], points_list[0],
                                   subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1],
                                   neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1],
                                   neighbors_list[1])

        feats_s3 = feats_s2
        feats_s3 = self.encoder3_1(feats_s3, points_list[2], points_list[1],
                                   subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2],
                                   neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2],
                                   neighbors_list[2])

        feats_s4 = feats_s3
        feats_s4 = self.encoder4_1(feats_s4, points_list[3], points_list[2],
                                   subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3],
                                   neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3],
                                   neighbors_list[3])

        resampled_points4 = self.resample4(points_list[3], feats_s4,
                                           neighbors_list[3])
        latent_s4 = torch.cat([feats_s4, resampled_points4],
                              dim=1)  # (N4, 64*16+3)
        latent_s4 = self.decoder4(latent_s4)

        resampled_points3 = self.resample3(points_list[2], feats_s3,
                                           neighbors_list[2])
        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3, resampled_points3],
                              dim=1)  # (N3, 64*16+64*8+3)
        latent_s3 = self.decoder3(latent_s3)

        resampled_points2 = self.resample2(points_list[1], feats_s2,
                                           neighbors_list[1])
        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2, resampled_points2],
                              dim=1)  # (N2, 64*8+64*4+3)
        latent_s2 = self.decoder2(latent_s2)  # (N1, 256)

        resampled_points1 = self.resample1(points_list[0], feats_s1,
                                           neighbors_list[0])
        latent_s1 = nearest_upsample(latent_s2, upsampling_list[0])
        latent_s1 = torch.cat([latent_s1, feats_s1, resampled_points1],
                              dim=1)  # (N1, 64*4+64*2+3)
        latent_s1 = self.decoder1(latent_s1)  # (N1, 256)

        output = self.outputlayer(latent_s1)  # (N1, 256)

        return output


class MLPDecoder(nn.Module):

    def __init__(self, dimoffeat=256):
        super(MLPDecoder, self).__init__()
        self.sharedmlp = nn.Sequential(
            nn.Conv1d(dimoffeat, int(dimoffeat / 2), 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(int(dimoffeat / 2), int(dimoffeat / 8), 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(int(dimoffeat / 8), 3, 1))

    def forward(self, feat):
        feat = feat.transpose(-1, -2)  # (N, d) -> (d, N)
        feat = feat.unsqueeze(0)  # (d, N) -> (batch, d, N)
        output = self.sharedmlp(feat)  # (batch, d, N) -> (batch, 3, N)
        output = output.squeeze(0)  # (batch, 3, N) -> (3, N)
        output = output.transpose(-1, -2)  # (3, N) -> (N, 3)

        return output


class ResampleGAN(nn.Module):

    def __init__(self, input_dim, dimofbottelneck, init_dim, kernel_size,
                 init_radius, init_sigma, group_norm):
        super(ResampleGAN, self).__init__()
        self.encoder = ResampleKPConvEncoder(input_dim, dimofbottelneck,
                                             init_dim, kernel_size,
                                             init_radius, init_sigma,
                                             group_norm)
        self.decoder = MLPDecoder(dimoffeat=dimofbottelneck)

    def forward(self, data_dict):
        feat = self.encoder(data_dict)
        points_recovered = self.decoder(feat)

        return points_recovered


# class LowLevelPointNet(nn.Module):

#     def __init__(self, dimoffeat=256) -> None:
#         super(LowLevelPointNet, self).__init__()
#         self.patchpointnet = nn.Sequential(
#             nn.Conv1d(3, dimoffeat // 4, kernel_size=1, stride=1),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv1d(dimoffeat // 4, dimoffeat // 2, kernel_size=1, stride=1),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv1d(dimoffeat // 2, dimoffeat, kernel_size=1, stride=1))
#         self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
#         self.sharedmlp = nn.Sequential(
#             nn.Conv1d(dimoffeat, dimoffeat // 2, kernel_size=1, stride=1),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv1d(dimoffeat // 2, dimoffeat // 4, kernel_size=1, stride=1),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv1d(dimoffeat // 4, dimoffeat // 8, kernel_size=1, stride=1),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv1d(dimoffeat // 8, 1, kernel_size=1, stride=1))

#     def forward(self, points, neighbor_indices):
#         neighbors = points[neighbor_indices]  # shape (N, K, 3)
#         neighbors = neighbors - points[:, None, :]
#         neighbors = neighbors.transpose(1, 2)  # (N, K, 3) -> (N, 3, K)

#         feats = self.patchpointnet(neighbors)  # (N, 3, K) -> (N, d, K)
#         patchfeats = self.maxpool(feats)  # (N, d, K) -> (N, d, 1)
#         patchfeats = patchfeats.squeeze(-1)  # (N, d, 1) -> (N, d)
#         patchfeats = patchfeats.transpose(0, 1)  # (N, d) -> (d, N)

#         out = self.sharedmlp(patchfeats)  # (d, N) -> (1, N)
#         out = out.squeeze(0)

#         return out


class MultiScalePointNet(nn.Module):

    def __init__(self, dimoffeat=256, multiscale=[5, 10, 20]) -> None:
        super(MultiScalePointNet, self).__init__()
        self.multiscale = multiscale
        self.patchpointnets = nn.ModuleList()
        for scale in multiscale:
            self.patchpointnets.append(
                nn.Sequential(
                    nn.Conv1d(3, dimoffeat // 4, kernel_size=1, stride=1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Conv1d(dimoffeat // 4,
                              dimoffeat // 2,
                              kernel_size=1,
                              stride=1), nn.LeakyReLU(negative_slope=0.2),
                    nn.Conv1d(dimoffeat // 2,
                              dimoffeat,
                              kernel_size=1,
                              stride=1)))
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        self.sharedmlp = nn.Sequential(
            nn.Conv1d(len(multiscale) * dimoffeat,
                      dimoffeat,
                      kernel_size=1,
                      stride=1), nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(dimoffeat, dimoffeat // 2, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(dimoffeat // 2, dimoffeat // 4, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(dimoffeat // 4, dimoffeat // 8, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(dimoffeat // 8, 1, kernel_size=1, stride=1))

    def forward(self, points, neighbor_indices):
        patchfeats_list = []
        for idx in range(len(self.multiscale)):
            neighbors = points[
                neighbor_indices[:, :self.multiscale[idx]]]  # shape (N, K, 3)
            neighbors = neighbors - points[:, None, :]
            neighbors = neighbors.transpose(1, 2)  # (N, K, 3) -> (N, 3, K)

            feats = self.patchpointnets[idx](
                neighbors)  # (N, 3, K) -> (N, d, K)
            patchfeats = self.maxpool(feats)  # (N, d, K) -> (N, d, 1)
            patchfeats = patchfeats.squeeze(-1)  # (N, d, 1) -> (N, d)
            patchfeats = patchfeats.transpose(0, 1)  # (N, d) -> (d, N)
            patchfeats_list.append(patchfeats)

        multiscalefeats = torch.cat(patchfeats_list,
                                    dim=0)  # m x (d, N) -> (md, N)

        multiscalefeats = multiscalefeats.unsqueeze(
            0)  # (md, N) -> (batch, md, N)
        out = self.sharedmlp(
            multiscalefeats)  # (batch, md, N) -> (batch, 1, N)
        out = out.squeeze(0)  # (batch, 1, N) -> (1, N)
        out = out.squeeze(0)

        return out
