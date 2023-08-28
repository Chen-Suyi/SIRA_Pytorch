import os.path as osp
import pickle
import random
import numpy as np
from pyrsistent import b
import torch
import open3d as o3d
from typing import Dict
from torch.utils.data import Dataset
from common.utils.func import get_augmentation_rotation, get_augmentation_translation, get_augmentation_noise, get_correspondences, get_overlap_mask, grid_subsample, radius_search
from common.utils.se3 import np_get_transform_from_rotation_translation
import gc


class SIRA(Dataset):

    def __init__(
        self,
        dataset_root,
        gt_log,
        subset,
        voxel_size=None,
        sphere_limit=None,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        augmentation_translation=5,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=None,
        scene_num=20,
        train_scene_num=18,
        test_scene_num=2,
        train_scene_indices=None,
        test_scene_indices=None,
        augmentation_rotation_type="eular",
        augmentation_translation_type="cube",
        augmentation_noise_type="rand",
    ):
        super(SIRA, self).__init__()

        self.dataset_root = dataset_root

        self.subset = subset
        self.voxel_size = voxel_size
        self.sphere_limit = sphere_limit
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError(
                "'matching_radius' is None but 'return_corr_indices' is set.")

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        self.aug_translation = augmentation_translation

        self.aug_rot_type = augmentation_rotation_type
        self.aug_trans_type = augmentation_translation_type
        self.aug_noise_type = augmentation_noise_type

        if self.subset == "train":
            self.metadata_root = osp.join(self.dataset_root, "SIRA")
            self.data_root = osp.join(self.dataset_root, "SIRA")

            self.train_index_list = []
            scene_index_list = list(range(scene_num))
            if test_scene_indices is not None:
                assert isinstance(test_scene_indices, list)
                assert len(test_scene_indices) <= test_scene_num
                for idx in test_scene_indices:
                    scene_index_list.remove(idx)
            if train_scene_indices is not None:
                assert isinstance(train_scene_indices, list)
                assert len(train_scene_indices) <= train_scene_num
                for idx in train_scene_indices:
                    scene_index_list.remove(idx)
                self.train_index_list += train_scene_indices

            if train_scene_num > len(self.train_index_list):
                self.train_index_list += scene_index_list[-(
                    train_scene_num - len(self.train_index_list)):]
            self.train_index_list.sort()
            assert len(self.train_index_list) == train_scene_num

            scene_indices_training = []
            self.data_list = []
            with open(osp.join(self.metadata_root, gt_log), "r") as f:
                for line in f.readlines():
                    data = {}
                    scene_index, src_path, ref_path, _ = line.split("\t")
                    scene_index = int(scene_index)
                    if scene_index not in self.train_index_list:
                        continue
                    else:
                        data["scene_name"] = "scene_index_" + str(scene_index)
                        data["ref_path"] = osp.join(self.data_root, ref_path)
                        data["src_path"] = osp.join(self.data_root, src_path)
                        self.data_list.append(data)
                        if scene_index not in scene_indices_training:
                            scene_indices_training.append(scene_index)

            # print("gt loaded from: ", gt_log)
            # print("scene indices for training: ", scene_indices_training)
            random.seed(0)
            random.shuffle(self.data_list)
            self.metadata_list = self.data_list[:]

        elif self.subset == "test":
            self.metadata_root = osp.join(self.dataset_root, "SIRA")
            self.data_root = osp.join(self.dataset_root, "SIRA")

            self.test_index_list = []
            scene_index_list = list(range(scene_num))
            if train_scene_indices is not None:
                assert isinstance(train_scene_indices, list)
                assert len(train_scene_indices) <= train_scene_num
                for idx in train_scene_indices:
                    scene_index_list.remove(idx)
            if test_scene_indices is not None:
                assert isinstance(test_scene_indices, list)
                assert len(test_scene_indices) <= test_scene_num
                for idx in test_scene_indices:
                    scene_index_list.remove(idx)
                self.test_index_list += test_scene_indices

            if test_scene_num > len(self.test_index_list):
                self.test_index_list += scene_index_list[:test_scene_num -
                                                         len(self.
                                                             test_index_list)]
            self.test_index_list.sort()
            assert len(self.test_index_list) == test_scene_num

            scene_indices_testing = []
            self.data_list = []
            with open(osp.join(self.metadata_root, gt_log), "r") as f:
                for line in f.readlines():
                    data = {}
                    scene_index, src_path, ref_path, _ = line.split("\t")
                    scene_index = int(scene_index)
                    if scene_index not in self.test_index_list:
                        continue
                    else:
                        data["scene_name"] = "scene_index_" + str(scene_index)
                        data["ref_path"] = osp.join(self.data_root, ref_path)
                        data["src_path"] = osp.join(self.data_root, src_path)
                        self.data_list.append(data)
                        if scene_index not in scene_indices_testing:
                            scene_indices_testing.append(scene_index)

            # print("gt loaded from: ", gt_log)
            # print("scene indices for testing: ", scene_indices_testing)
            # random.seed(0)
            # random.shuffle(self.data_list)
            # self.metadata_list = self.data_list[:1600]

        elif self.subset in ["val", "3DMatch", "3DLoMatch"]:
            self.metadata_root = osp.join(self.dataset_root, "3DMatch",
                                          "metadata")
            self.data_root = osp.join(self.dataset_root, "3DMatch", "data")

            with open(osp.join(self.metadata_root, f"{self.subset}.pkl"),
                      "rb") as f:
                self.metadata_list = pickle.load(f)
                if self.overlap_threshold is not None:
                    self.metadata_list = [
                        x for x in self.metadata_list
                        if x["overlap"] > self.overlap_threshold
                    ]

        else:
            raise NotImplementedError("Subset Value Error!")

    def __len__(self):
        return len(self.metadata_list)

    def _load_point_cloud(self, file_name):
        if self.subset in ["val", "3DMatch", "3DLoMatch"]:
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

        return points

    def _voxel_down_sample(self, points):
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(points)
        o3d_down = o3d_pc.voxel_down_sample(voxel_size=self.voxel_size)
        points_down = np.asarray(o3d_down.points)

        # release memory
        del points, o3d_pc
        gc.collect()

        return points_down

    def _augment_point_cloud(self,
                             ref_points,
                             src_points,
                             rotation,
                             translation,
                             aug_rot_type="eular",
                             aug_trans_type="cube",
                             aug_noise_type="rand"):
        r"""Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation and translation to one point cloud.
        2. Random noise.
        """
        aug_rotation = get_augmentation_rotation(self.aug_rotation,
                                                 aug_rot_type)
        aug_translation = get_augmentation_translation(self.aug_translation,
                                                       aug_trans_type)

        if np.random.rand() > 0.5:
            ref_points_ = ref_points
            ref_points = np.matmul(ref_points,
                                   aug_rotation.T) + aug_translation
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation,
                                    translation) + aug_translation
            if not np.allclose(
                    np.matmul(ref_points_, rotation.T) + translation,
                    ref_points,
                    atol=1e-6):
                print("False transform!!!")
        else:
            src_points_ = src_points
            src_points = np.matmul(src_points,
                                   aug_rotation.T) + aug_translation
            rotation = np.matmul(rotation, aug_rotation.T)
            translation = translation - np.matmul(rotation, aug_translation)
            if not np.allclose(np.matmul(src_points, rotation.T) + translation,
                               src_points_,
                               atol=1e-6):
                print("False transform!!!")

        ref_noise = get_augmentation_noise(ref_points.shape[0], self.aug_noise,
                                           aug_noise_type)
        src_noise = get_augmentation_noise(src_points.shape[0], self.aug_noise,
                                           aug_noise_type)
        ref_points += ref_noise
        src_points += src_noise

        return ref_points, src_points, rotation, translation

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]

        # get point cloud
        if self.subset in ["train", "test"]:
            data_dict["scene_name"] = metadata["scene_name"]

            # get transformation
            rotation = np.identity(n=3)
            translation = np.zeros(shape=(3, ))

            # get point cloud
            ref_path = metadata["ref_path"]
            src_path = metadata["src_path"]

        elif self.subset in ["val", "3DMatch", "3DLoMatch"]:
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

        else:
            raise NotImplementedError("Dataset Not Implemented")

        ref_points = self._load_point_cloud(ref_path)
        src_points = self._load_point_cloud(src_path)

        # down sample
        if self.voxel_size is not None:
            ref_points = self._voxel_down_sample(ref_points)
            src_points = self._voxel_down_sample(src_points)

        # sphere limit
        if self.sphere_limit is not None:
            src_points_aligned = np.matmul(src_points,
                                           rotation.T) + translation

            transform = np_get_transform_from_rotation_translation(
                rotation, translation)

            ref_overlap, src_overlap = get_overlap_mask(
                ref_points, src_points, transform, 0.05)

            ref_points_overlap = ref_points[ref_overlap]
            src_points_overlap = src_points_aligned[src_overlap]

            centroid = (np.sum(ref_points_overlap, axis=0) + np.sum(
                src_points_overlap, axis=0)) / (ref_points_overlap.shape[0] +
                                                src_points_overlap.shape[0])

            ref_points = ref_points[np.sum(np.square(ref_points - centroid),
                                           axis=1) <= self.sphere_limit**2]
            src_points = src_points[
                np.sum(np.square(src_points_aligned -
                                 centroid), axis=1) <= self.sphere_limit**2]

            # release memory
            del src_points_aligned, ref_points_overlap, src_points_overlap
            gc.collect()

        # point limit
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None:
            if self.subset in ["test", "val", "3DMatch", "3DLoMatch"]:
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

            # release memory
            gc.collect()

        # augmentation
        if self.subset in ["train", "test"]:
            self.use_augmentation = True
        elif self.subset in ["val", "3DMatch", "3DLoMatch"]:
            self.use_augmentation = False
        if self.use_augmentation:  # train or test: set True; 3DMatch or 3DLoMatch: set False
            if self.subset in ["test", "val", "3DMatch", "3DLoMatch"]:
                np.random.seed(index)
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation,
                self.aug_rot_type, self.aug_trans_type, self.aug_noise_type)

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
