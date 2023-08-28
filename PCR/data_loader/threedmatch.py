import os.path as osp
import pickle
import random
import numpy as np
from pyrsistent import b
import torch
import open3d as o3d
from typing import Dict
from torch.utils.data import Dataset
from common.utils.func import random_sample_rotation, uniform_sample_rotation, get_correspondences, get_overlap_mask, grid_subsample, radius_search
from common.utils.se3 import np_get_transform_from_rotation_translation


class ThreeDMatch(Dataset):

    def __init__(self,
                 dataset_root,
                 subset,
                 point_limit=None,
                 use_augmentation=False,
                 augmentation_noise=0.005,
                 augmentation_rotation=1,
                 overlap_threshold=None,
                 return_corr_indices=False,
                 matching_radius=None,
                 rotated=False):
        super(ThreeDMatch, self).__init__()

        self.dataset_root = dataset_root
        self.metadata_root = osp.join(self.dataset_root, "3DMatch", "metadata")
        self.data_root = osp.join(self.dataset_root, "3DMatch", "data")

        self.subset = subset
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError(
                "'matching_radius' is None but 'return_corr_indices' is set.")

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation

        with open(osp.join(self.metadata_root, f"{subset}.pkl"), "rb") as f:
            self.metadata_list = pickle.load(f)
            if self.overlap_threshold is not None:
                self.metadata_list = [
                    x for x in self.metadata_list
                    if x["overlap"] > self.overlap_threshold
                ]

    def __len__(self):
        return len(self.metadata_list)

    def _load_point_cloud(self, file_name):
        file_name = osp.join(self.data_root, file_name)
        if file_name.endswith('.pth'):
            points = torch.load(file_name)
        elif file_name.endswith('.ply'):
            pcd = o3d.io.read_point_cloud(file_name)
            points = np.asarray(pcd.points)
        elif file_name.endswith('.bin'):
            points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
        else:
            raise AssertionError('Cannot recognize point cloud format')
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        # if self.point_limit is not None and points.shape[0] > self.point_limit:
        #     indices = np.random.permutation(points.shape[0])[:self.point_limit]
        #     points = points[indices]
        return points

    def _voxel_down_sample(self, points):
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(points)
        o3d_down = o3d_pc.voxel_down_sample(voxel_size=0.025)
        points_down = np.asarray(o3d_down.points)

        return points_down

    def _augment_point_cloud(self, ref_points, src_points, rotation,
                             translation):
        r"""Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)

        ref_points += (np.random.rand(ref_points.shape[0], 3) -
                       0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) -
                       0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["ref_frame"] = metadata["frag_id0"]
        data_dict["src_frame"] = metadata["frag_id1"]
        data_dict["overlap"] = metadata["overlap"]

        # get transformation
        rotation = metadata["rotation"]
        translation = metadata["translation"]

        # get point cloud
        ref_path = metadata["pcd0"].replace("pth", "ply")
        src_path = metadata["pcd1"].replace("pth", "ply")
        ref_points = self._load_point_cloud(metadata["pcd0"].replace(
            "pth", "ply"))
        src_points = self._load_point_cloud(metadata["pcd1"].replace(
            "pth", "ply"))

        # # down sample
        # if self.subset == "train":
        #     ref_points = self._voxel_down_sample(ref_points)
        #     src_points = self._voxel_down_sample(src_points)

        # point limit
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None:
            if self.subset in ["val", "3DMatch", "3DLoMatch"]:
                np.random.seed(0)
            if ref_points.shape[0] > self.point_limit:
                ref_indices = np.random.permutation(
                    ref_points.shape[0])[:self.point_limit]
                ref_points = ref_points[ref_indices]
            if src_points.shape[0] > self.point_limit:
                src_indices = np.random.permutation(
                    src_points.shape[0])[:self.point_limit]
                src_points = src_points[src_indices]


        # augmentation
        if self.use_augmentation:
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation)

        if self.rotated:
            ref_rotation = uniform_sample_rotation()
            ref_points = np.matmul(ref_points, ref_rotation.T)
            rotation = np.matmul(ref_rotation, rotation)
            translation = np.matmul(ref_rotation, translation)

            src_rotation = uniform_sample_rotation()
            src_points = np.matmul(src_points, src_rotation.T)
            rotation = np.matmul(rotation, src_rotation.T)

        transform = np_get_transform_from_rotation_translation(
            rotation, translation)

        if self.return_corr_indices:
            corr = get_correspondences(ref_points, src_points, transform,
                                       self.matching_radius)
            data_dict["corr_indices"] = corr["corr_indices"]
            data_dict["ref_overlap"] = corr["has_corr_ref"]
            data_dict["src_overlap"] = corr["has_corr_src"]
            data_dict["mutual_corr_indices"] = corr["mutual_corr_indices"]
        else:
            if self.matching_radius is not None:
                ref_overlap, src_overlap = get_overlap_mask(
                    ref_points, src_points, transform, self.matching_radius)
                data_dict["ref_overlap"] = ref_overlap
                data_dict["src_overlap"] = src_overlap

        data_dict["ref_points"] = ref_points.astype(np.float32)
        data_dict["src_points"] = src_points.astype(np.float32)
        data_dict["ref_feats"] = np.ones((ref_points.shape[0], 1),
                                         dtype=np.float32)
        data_dict["src_feats"] = np.ones((src_points.shape[0], 1),
                                         dtype=np.float32)
        data_dict["transform"] = transform.astype(np.float32)
        data_dict["index"] = index
        data_dict["ref_path"] = ref_path
        data_dict["src_path"] = src_path

        return data_dict
