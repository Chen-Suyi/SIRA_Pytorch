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


class ETH(Dataset):

    def __init__(self,
                 dataset_root,
                 point_limit=None,
                 use_augmentation=False,
                 augmentation_noise=0.005,
                 augmentation_rotation=1,
                 overlap_threshold=None,
                 return_corr_indices=False,
                 matching_radius=None,
                 rotated=False):
        super(ETH, self).__init__()

        self.dataset_root = dataset_root
        self.metadata_root = osp.join(self.dataset_root, "ETH")
        self.data_root = osp.join(self.dataset_root, "ETH")

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

        self.metadata_list = []

        self.scene_names = [
            'gazebo_summer', 'gazebo_winter', 'wood_autumn', 'wood_summer'
        ]

        for scene_name in self.scene_names:
            PointCloud_path = osp.join(self.metadata_root, scene_name,
                                       "PointCloud")
            with open(osp.join(PointCloud_path, "gt.log"), "r") as f:
                lines = f.readlines()
                pair_num = len(lines) // 5
                pair_id2transform = {}
                for k in range(pair_num):
                    id0, id1 = np.fromstring(lines[k * 5],
                                             dtype=np.float32,
                                             sep='\t')[0:2]
                    id0 = int(id0)
                    id1 = int(id1)
                    row0 = np.fromstring(lines[k * 5 + 1],
                                         dtype=np.float32,
                                         sep=' ')
                    row1 = np.fromstring(lines[k * 5 + 2],
                                         dtype=np.float32,
                                         sep=' ')
                    row2 = np.fromstring(lines[k * 5 + 3],
                                         dtype=np.float32,
                                         sep=' ')
                    row3 = np.fromstring(lines[k * 5 + 4],
                                         dtype=np.float32,
                                         sep=' ')
                    transform = np.stack([row0, row1, row2, row3], 0)

                    ref_path = osp.join(PointCloud_path,
                                        "cloud_bin_{}.ply".format(id0))
                    src_path = osp.join(PointCloud_path,
                                        "cloud_bin_{}.ply".format(id1))

                    metadata = {}
                    metadata["scene_name"] = scene_name
                    metadata["frag_id0"] = id0
                    metadata["frag_id1"] = id1
                    metadata["ref_path"] = ref_path
                    metadata["src_path"] = src_path
                    metadata["transform"] = transform
                    metadata["rotation"] = transform[:3, :3]
                    metadata["translation"] = transform[:3, 3]

                    self.metadata_list.append(metadata)

    def __len__(self):
        return len(self.metadata_list)

    def _load_point_cloud(self, file_name):
        if file_name.endswith('.pth'):
            points = torch.load(file_name)
        elif file_name.endswith('.ply'):
            pcd = o3d.io.read_point_cloud(file_name)
            points = np.asarray(pcd.points)
        elif file_name.endswith('.bin'):
            points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
        else:
            raise AssertionError('Cannot recognize point cloud format')

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

    def _scale(self,
               ref_points: np.ndarray,
               src_points: np.ndarray,
               rotation: np.ndarray,
               translation: np.ndarray,
               scale=1.0):
        ref_points = ref_points * scale
        # src_points = ((src_points @ rotation.T + translation) * scale -
        #               translation) @ rotation
        src_points = src_points * scale

        return ref_points, src_points

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["ref_frame"] = metadata["frag_id0"]
        data_dict["src_frame"] = metadata["frag_id1"]

        # get transformation
        rotation = metadata["rotation"]
        translation = metadata["translation"]

        # get point cloud
        ref_path = metadata["ref_path"]
        src_path = metadata["src_path"]
        ref_points = self._load_point_cloud(ref_path)
        src_points = self._load_point_cloud(src_path)

        # # down sample
        # ref_points = self._voxel_down_sample(ref_points)
        # src_points = self._voxel_down_sample(src_points)

        # point limit
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None:
            np.random.seed(index)
            if ref_points.shape[0] > self.point_limit:
                ref_indices = np.random.permutation(
                    ref_points.shape[0])[:self.point_limit]
                ref_points = ref_points[ref_indices]
                del ref_indices
            if src_points.shape[0] > self.point_limit:
                src_indices = np.random.permutation(
                    src_points.shape[0])[:self.point_limit]
                src_points = src_points[src_indices]
                del src_indices

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
        data_dict["overlap"] = ref_overlap.astype(np.float32).mean()

        # re-scale
        scale_factor = 0.1
        ref_points, src_points = self._scale(ref_points,
                                             src_points,
                                             rotation,
                                             translation,
                                             scale=scale_factor)
        data_dict["scale"] = scale_factor
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
