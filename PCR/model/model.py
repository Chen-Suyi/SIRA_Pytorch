import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from model.module import *
from common.utils.func import point_to_node_partition, index_select, get_node_correspondences


# GeoTransformer
#####################################################################################
class GeoTransformer(nn.Module):

    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.cfg = cfg
        self.num_points_in_patch = self.cfg.model.num_points_in_patch
        self.matching_radius = self.cfg.model.ground_truth_matching_radius
        self.cfg.backbone.init_radius = self.cfg.backbone.init_voxel_size * self.cfg.backbone.base_radius
        self.cfg.backbone.init_sigma = self.cfg.backbone.init_voxel_size * self.cfg.backbone.base_sigma

        self.backbone = KPConvFPN(
            self.cfg.backbone.input_dim,
            self.cfg.backbone.output_dim,
            self.cfg.backbone.init_dim,
            self.cfg.backbone.kernel_size,
            self.cfg.backbone.init_radius,
            self.cfg.backbone.init_sigma,
            self.cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            self.cfg.geotransformer.input_dim,
            self.cfg.geotransformer.output_dim,
            self.cfg.geotransformer.hidden_dim,
            self.cfg.geotransformer.num_heads,
            self.cfg.geotransformer.blocks,
            self.cfg.geotransformer.sigma_d,
            self.cfg.geotransformer.sigma_a,
            self.cfg.geotransformer.angle_k,
            reduction_a=self.cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            self.cfg.coarse_matching.num_targets,
            self.cfg.coarse_matching.overlap_threshold,
        )

        self.coarse_matching = SuperPointMatching(
            self.cfg.coarse_matching.num_correspondences,
            self.cfg.coarse_matching.dual_normalization,
        )

        self.fine_matching = LocalGlobalRegistration(
            self.cfg.fine_matching.topk,
            self.cfg.fine_matching.acceptance_radius,
            mutual=self.cfg.fine_matching.mutual,
            confidence_threshold=self.cfg.fine_matching.confidence_threshold,
            use_dustbin=self.cfg.fine_matching.use_dustbin,
            use_global_score=self.cfg.fine_matching.use_global_score,
            correspondence_threshold=self.cfg.fine_matching.
            correspondence_threshold,
            correspondence_limit=self.cfg.fine_matching.correspondence_limit,
            num_refinement_steps=self.cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(
            self.cfg.model.num_sinkhorn_iterations)

    def forward(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch)
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch)

        ref_padded_points_f = torch.cat(
            [ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat(
            [src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f,
                                           ref_node_knn_indices,
                                           dim=0)
        src_node_knn_points = index_select(src_padded_points_f,
                                           src_node_knn_indices,
                                           dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        torch.cuda.empty_cache()

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks,
                src_node_masks)

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps)

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[
            ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[
            src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[
            ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[
            src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[
            ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[
            src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat(
            [ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat(
            [src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f,
                                               ref_node_corr_knn_indices,
                                               dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f,
                                               src_node_corr_knn_indices,
                                               dim=0)  # (P, K, C)

        output_dict[
            'ref_node_corr_knn_points'] = ref_node_corr_knn_points  # local patch points
        output_dict[
            'src_node_corr_knn_points'] = src_node_corr_knn_points  # local patch points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats,
                                       src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1]**0.5
        matching_scores = self.optimal_transport(matching_scores,
                                                 ref_node_corr_knn_masks,
                                                 src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        return output_dict


def fetch_model(cfg):
    if cfg.model_name == "geotransformer":
        model = GeoTransformer(cfg)
        model = model.cuda(cfg.local_rank)
        model = DDP(model,
                    device_ids=[cfg.local_rank],
                    output_device=cfg.local_rank)
    else:
        raise NotImplementedError

    return model
