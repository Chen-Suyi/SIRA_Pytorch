import logging
import math
import random
from common.utils import se3

import cv2
import numpy as np
import torch
import torchvision
import open3d
from pytorch3d import ops as p3d_ops
from common.utils import so3
from scipy.spatial.distance import minkowski
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
from sklearn.neighbors import NearestNeighbors
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from scipy.spatial import cKDTree
from other_methods.rgm import rgm_build_graphs

_logger = logging.getLogger(__name__)


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        num: Number of vectors to sample (or None if single)

    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


class SplitSourceRef:
    """Clones the point cloud into separate source and reference point clouds"""

    def __init__(self, mode="hdf"):
        self.mode = mode

    def __call__(self, sample):
        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        if self.mode == "hdf":
            sample["points_raw"] = sample.pop("points").astype(
                np.float32)[:, :3]
            sample["points_src"] = sample["points_raw"].copy()
            sample["points_ref"] = sample["points_raw"].copy()
            sample["points_src_raw"] = sample["points_src"].copy().astype(
                np.float32)
            sample["points_ref_raw"] = sample["points_ref"].copy().astype(
                np.float32)

        elif self.mode == "resample":
            points_raw = sample.pop("points").astype(np.float32)
            sample["points_src"] = points_raw[
                np.random.choice(points_raw.shape[0], 2048, replace=False), :]
            sample["points_ref"] = points_raw[
                np.random.choice(points_raw.shape[0], 2048, replace=False), :]
            sample["points_src_raw"] = sample["points_src"].copy().astype(
                np.float32)
            sample["points_ref_raw"] = sample["points_ref"].copy().astype(
                np.float32)

        elif self.mode == "donothing":
            points_raw = sample.pop("points").astype(np.float32)
            points_raw = points_raw[
                np.random.choice(points_raw.shape[0], 2, replace=False), :, :]
            sample["points_src"] = points_raw[0, :, :].astype(np.float32)
            sample["points_ref"] = points_raw[1, :, :].astype(np.float32)
            sample["points_src_raw"] = sample["points_src"].copy()
            sample["points_ref_raw"] = sample["points_ref"].copy()

        elif self.mode == "7scenes":
            points_src = sample["points_src"].astype(np.float32)
            points_ref = sample["points_ref"].astype(np.float32)
            sample["points_src_raw"] = points_src.astype(np.float32)
            sample["points_ref_raw"] = points_ref.astype(np.float32)
            if sample["num_points"] == -1:
                sample["points_src"] = points_src
                sample["points_ref"] = points_ref
            else:
                if points_src.shape[0] > sample["num_points"]:
                    sample["points_src"] = points_src[
                        np.random.choice(points_src.shape[0],
                                         sample["num_points"],
                                         replace=False), :]
                else:
                    rand_idxs = np.concatenate([
                        np.random.choice(points_src.shape[0],
                                         points_src.shape[0],
                                         replace=False),
                        np.random.choice(points_src.shape[0],
                                         sample["num_points"] -
                                         points_src.shape[0],
                                         replace=True)
                    ])
                    sample["points_src"] = points_src[rand_idxs, :]

                if points_ref.shape[0] > sample["num_points"]:
                    sample["points_ref"] = points_ref[
                        np.random.choice(points_ref.shape[0],
                                         sample["num_points"],
                                         replace=False), :]
                else:
                    rand_idxs = np.concatenate([
                        np.random.choice(points_ref.shape[0],
                                         points_ref.shape[0],
                                         replace=False),
                        np.random.choice(points_ref.shape[0],
                                         sample["num_points"] -
                                         points_ref.shape[0],
                                         replace=True)
                    ])
                    sample["points_ref"] = points_ref[rand_idxs, :]

        elif self.mode == "kitti_real":
            points_src = sample["points_src"].astype(np.float32)
            points_ref = sample["points_ref"].astype(np.float32)
            points_src = torch.from_numpy(points_src).unsqueeze(0).cuda()
            points_ref = torch.from_numpy(points_ref).unsqueeze(0).cuda()
            points_src_flipped = points_src.transpose(1,
                                                      2).contiguous().detach()
            points_src_fps = gather_operation(
                points_src_flipped, furthest_point_sample(
                    points_src, 4096)).transpose(1, 2).contiguous().detach()
            points_ref_flipped = points_ref.transpose(1,
                                                      2).contiguous().detach()
            points_ref_fps = gather_operation(
                points_ref_flipped, furthest_point_sample(
                    points_ref, 4096)).transpose(1, 2).contiguous().detach()
            sample["points_src"] = points_src_fps.squeeze(0).cpu().numpy()
            sample["points_ref"] = points_ref_fps.squeeze(0).cpu().numpy()
            sample["points_src_raw"] = sample["points_src"].copy().astype(
                np.float32)
            sample["points_ref_raw"] = sample["points_ref"].copy().astype(
                np.float32)

        elif self.mode == "kitti_sync":
            points_raw = sample.pop("points").astype(np.float32)
            sample["points_src"] = points_raw[
                np.random.choice(points_raw.shape[0], 4096, replace=False), :]
            sample["points_ref"] = points_raw[
                np.random.choice(points_raw.shape[0], 4096, replace=False), :]
            sample["points_src_raw"] = sample["points_src"].copy().astype(
                np.float32)
            sample["points_ref_raw"] = sample["points_ref"].copy().astype(
                np.float32)

        else:
            raise NotImplementedError

        return sample


class Resampler:

    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """
        # print("===", points.shape[0], k)
        if k < points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([
                np.random.choice(points.shape[0],
                                 points.shape[0],
                                 replace=False),
                np.random.choice(points.shape[0],
                                 k - points.shape[0],
                                 replace=True)
            ])
            return points[rand_idxs, :]

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        if "points" in sample:
            sample["points"] = self._resample(sample["points"], self.num)
        else:
            if "crop_proportion" not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample["crop_proportion"]) == 1:
                src_size = math.ceil(sample["crop_proportion"][0] * self.num)
                ref_size = self.num
            elif len(sample["crop_proportion"]) == 2:
                src_size = math.ceil(sample["crop_proportion"][0] * self.num)
                ref_size = math.ceil(sample["crop_proportion"][1] * self.num)
            else:
                raise ValueError("Crop proportion must have 1 or 2 elements")

            sample["points_src"] = self._resample(sample["points_src"],
                                                  src_size)
            sample["points_ref"] = self._resample(sample["points_ref"],
                                                  ref_size)

            # sample for the raw point clouds
            sample["points_src_raw"] = sample["points_src_raw"][:self.num, :]
            sample["points_ref_raw"] = sample["points_ref_raw"][:self.num, :]

        return sample


class RandomJitter:
    """ generate perturbations """

    def __init__(self, noise_std=0.01, clip=0.05):
        self.noise_std = noise_std
        self.clip = clip

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0,
                                         scale=self.noise_std,
                                         size=(pts.shape[0], 3)),
                        a_min=-self.clip,
                        a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def __call__(self, sample):

        if "points" in sample:
            sample["points"] = self.jitter(sample["points"])
        else:
            sample["points_src"] = self.jitter(sample["points_src"])
            sample["points_ref"] = self.jitter(sample["points_ref"])

        return sample


class RandomCrop:
    """Randomly crops the *source* point cloud, approximately retaining half the points

    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """

    def __init__(self, p_keep=None):
        if p_keep is None:
            p_keep = [0.7, 0.7]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)

    @staticmethod
    def crop(points, p_keep):
        if p_keep == 1.0:
            mask = np.ones(shape=(points.shape[0], )) > 0

        else:
            rand_xyz = uniform_2_sphere()
            centroid = np.mean(points[:, :3], axis=0)
            points_centered = points[:, :3] - centroid
            dist_from_plane = np.dot(points_centered, rand_xyz)

            if p_keep == 0.5:
                mask = dist_from_plane > 0
            else:
                mask = dist_from_plane > np.percentile(dist_from_plane,
                                                       (1.0 - p_keep) * 100)

        return points[mask, :]

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        sample["crop_proportion"] = self.p_keep

        if len(sample["crop_proportion"]) == 1:
            sample["points_src"] = self.crop(sample["points_src"],
                                             self.p_keep[0])
            sample["points_ref"] = self.crop(sample["points_ref"], 1.0)
        else:
            sample["points_src"] = self.crop(sample["points_src"],
                                             self.p_keep[0])
            sample["points_ref"] = self.crop(sample["points_ref"],
                                             self.p_keep[1])

        return sample


class RandomScale:

    def __init__(self, min_factor=0.5, max_factor=2):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        src_scale_factor = np.random.uniform(self.min_factor, self.max_factor)
        ref_scale_factor = np.random.uniform(self.min_factor, self.max_factor)

        sample["points_src"] = sample["points_src"] * src_scale_factor
        sample["points_ref"] = sample["points_ref"] * ref_scale_factor
        sample[
            "scale_gt"] = 1 / src_scale_factor * ref_scale_factor  # scale src to ref

        return sample


class Normalization():

    def __call__(self, sample):
        src = sample["points_src"]
        ref = sample["points_ref"]
        src_raw = sample["points_src_raw"]
        ref_raw = sample["points_ref_raw"]
        # Normalization
        src_mean = np.mean(src, axis=0)
        ref_mean = np.mean(ref, axis=0)
        src = src - src_mean
        src_raw = src_raw - src_mean
        ref = ref - ref_mean
        ref_raw = ref_raw - ref_mean
        src_ref = np.concatenate((src, ref), axis=0)
        norm = np.max(np.sqrt(np.sum(src_ref**2, axis=1)))
        # new_src = (old_src - mean) / norm
        src = src / norm
        src_raw = src_raw / norm
        ref = ref / norm
        ref_raw = ref_raw / norm
        sample["points_src_raw"] = src_raw
        sample["points_ref_raw"] = ref_raw
        return sample


class RandomTransformSE3:

    def __init__(self,
                 rot_mag: float = 180.0,
                 trans_mag: float = 1.0,
                 random_mag: bool = False):
        """Applies a random rigid transformation to the source point cloud

        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                               perturbations
        """
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag
        self._random_mag = random_mag

    def generate_transform(self):
        """Generate a random SE3 transformation (3, 4) """

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        rand_rot = special_ortho_group.rvs(3)
        axis_angle = Rotation.as_rotvec(Rotation.from_dcm(rand_rot))
        axis_angle *= rot_mag / 180.0
        rand_rot = Rotation.from_rotvec(axis_angle).as_dcm()

        # Generate translation
        rand_trans = np.random.uniform(-trans_mag, trans_mag, 3)
        rand_SE3 = np.concatenate((rand_rot, rand_trans[:, None]),
                                  axis=1).astype(np.float32)

        return rand_SE3

    def apply_transform(self, p0, transform_mat):
        p1 = se3.np_transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        igt = transform_mat
        gt = se3.np_inverse(igt)

        return p1, gt, igt

    def transform(self, tensor):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, transform_mat)

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        if "points" in sample:
            sample["points"], _, _ = self.transform(sample["points"])
        else:
            src_transformed, transform_r_s, transform_s_r = self.transform(
                sample["points_src"])
            # Apply to source to get reference
            sample["transform_gt"] = transform_r_s
            sample["pose_gt"] = se3.np_mat2quat(transform_r_s)
            sample["transform_igt"] = transform_s_r
            sample["points_src"] = src_transformed
            # transnform the raw source point cloud
            sample["points_src_raw"] = se3.np_transform(
                transform_s_r, sample["points_src_raw"][:, :3])

        return sample


# noinspection PyPep8Naming
class RandomTransformSE3_euler(RandomTransformSE3):
    """Same as RandomTransformSE3, but rotates using euler angle rotations

    This transformation is consistent to Deep Closest Point but does not
    generate uniform rotations

    """

    def generate_transform(self):

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        anglex = np.random.uniform() * np.pi * rot_mag / 180.0
        angley = np.random.uniform() * np.pi * rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-trans_mag, trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]),
                                  axis=1).astype(np.float32)
        return rand_SE3


class ShufflePoints:
    """Shuffles the order of the points"""

    def __call__(self, sample):
        if "points" in sample:
            sample["points"] = np.random.permutation(sample["points"])
        else:
            sample["points_ref"] = np.random.permutation(sample["points_ref"])
            sample["points_src"] = np.random.permutation(sample["points_src"])

        return sample


class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""

    def __call__(self, sample):
        sample["deterministic"] = True
        return sample


class PRNetTorch:

    def __init__(self,
                 num_points,
                 rot_mag,
                 trans_mag,
                 noise_std=0.01,
                 clip=0.05,
                 add_noise=True,
                 only_z=False,
                 partial=True):
        self.num_points = num_points
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.noise_std = noise_std
        self.clip = clip
        self.add_noise = add_noise
        self.only_z = only_z
        self.partial = partial

    def apply_transform(self, p0, transform_mat):
        p1 = se3.np_transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        gt = transform_mat

        return p1, gt

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0,
                                         scale=self.noise_std,
                                         size=(pts.shape[0], 3)),
                        a_min=-self.clip,
                        a_max=self.clip)
        noise = torch.from_numpy(noise).to(pts.device)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def knn(self, pts, random_pt, k):
        random_pt = torch.from_numpy(random_pt).to(pts.device)
        distance = torch.sum((pts - random_pt)**2, dim=1)
        idx = distance.topk(k=k, dim=0,
                            largest=False)[1]  # (batch_size, num_points, k)
        return idx

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        src = sample["points_src"]
        ref = sample["points_ref"]
        # Generate rigid transform
        anglex = np.random.uniform() * np.pi * self.rot_mag / 180.0
        angley = np.random.uniform() * np.pi * self.rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * self.rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

        if not self.only_z:
            R_ab = Rx @ Ry @ Rz
        else:
            R_ab = Rz
        t_ab = np.random.uniform(-self.trans_mag, self.trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]),
                                  axis=1).astype(np.float32)
        ref, transform_s_r = self.apply_transform(ref, rand_SE3)
        # Apply to source to get reference
        sample["transform_gt"] = transform_s_r
        sample["pose_gt"] = se3.np_mat2quat(transform_s_r)

        # Crop and sample
        if self.partial:
            src = torch.from_numpy(src)
            ref = torch.from_numpy(ref)
            random_p1 = np.random.random(size=(1, 3)) + np.array(
                [[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
            idx1 = self.knn(src, random_p1, k=768)
            # np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
            random_p2 = random_p1
            idx2 = self.knn(ref, random_p2, k=768)
        else:
            idx1 = np.random.choice(src.shape[0], 1024, replace=False),
            idx2 = np.random.choice(ref.shape[0], 1024, replace=False),
            # src = np.squeeze(src, axis=-1)
            # ref = np.squeeze(ref, axis=-1)
            src = torch.from_numpy(src)
            ref = torch.from_numpy(ref)

        # add noise
        if self.add_noise:
            sample["points_src"] = self.jitter(src[idx1, :])
            sample["points_ref"] = self.jitter(ref[idx2, :])
        else:
            sample["points_src"] = src[idx1, :]
            sample["points_ref"] = ref[idx2, :]
        if sample["points_src"].size()[0] == 1:
            sample["points_src"] = sample["points_src"].squeeze(0)
            sample["points_ref"] = sample["points_ref"].squeeze(0)

        # # for inference time
        # if sample["points_src"].shape[0] < self.num_points:
        #     rand_idxs = np.concatenate(
        #         [np.random.choice(sample["points_src"].shape[0], sample["points_src"].shape[0], replace=False),
        #          np.random.choice(sample["points_src"].shape[0], self.num_points - sample["points_src"].shape[0],
        #                           replace=True)])
        #     sample["points_src"] = sample["points_src"][rand_idxs, :]
        #     rand_idxs = np.concatenate(
        #         [np.random.choice(sample["points_ref"].shape[0], sample["points_ref"].shape[0], replace=False),
        #          np.random.choice(sample["points_ref"].shape[0], self.num_points - sample["points_ref"].shape[0],
        #                           replace=True)])
        #     sample["points_ref"] = sample["points_ref"][rand_idxs, :]
        # else:
        #     rand_idxs = np.random.choice(sample["points_src"].shape[0], self.num_points, replace=False),
        #     sample["points_src"] = sample["points_src"][rand_idxs, :]
        #     rand_idxs = np.random.choice(sample["points_ref"].shape[0], self.num_points, replace=False),
        #     sample["points_ref"] = sample["points_ref"][rand_idxs, :]
        # if sample["points_src"].shape[0] == 1:
        #     sample["points_src"] = sample["points_src"].squeeze(0)
        #     sample["points_ref"] = sample["points_ref"].squeeze(0)

        return sample


class PRNet:

    def __init__(self,
                 num_points,
                 rot_mag,
                 trans_mag,
                 noise_std=0.01,
                 clip=0.05,
                 add_noise=True,
                 only_z=False,
                 partial=True):
        self.num_points = num_points
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.noise_std = noise_std
        self.clip = clip
        self.add_noise = add_noise
        self.only_z = only_z
        self.partial = partial

    def apply_transform(self, p0, transform_mat):
        p1 = se3.np_transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        gt = transform_mat

        return p1, gt

    def jitter(self, pts):
        noise = np.clip(np.random.normal(0.0,
                                         scale=self.noise_std,
                                         size=(pts.shape[0], 3)),
                        a_min=-self.clip,
                        a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def knn(self, pts, random_pt, k):
        distance = np.sum((pts - random_pt)**2, axis=1)
        idx = np.argsort(distance)[:k]  # (k,)
        return idx

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        src = sample["points_src"]
        ref = sample["points_ref"]
        # Generate rigid transform
        anglex = np.random.uniform() * np.pi * self.rot_mag / 180.0
        angley = np.random.uniform() * np.pi * self.rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * self.rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

        if not self.only_z:
            R_ab = Rx @ Ry @ Rz
        else:
            R_ab = Rz
        t_ab = np.random.uniform(-self.trans_mag, self.trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]),
                                  axis=1).astype(np.float32)
        ref, transform_s_r = self.apply_transform(ref, rand_SE3)
        # Apply to source to get reference
        sample["transform_gt"] = transform_s_r
        sample["pose_gt"] = se3.np_mat2quat(transform_s_r)

        # Crop and sample
        if self.partial:
            random_p1 = np.random.random(size=(1, 3)) + np.array(
                [[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
            idx1 = self.knn(src, random_p1, k=768)
            random_p2 = random_p1
            idx2 = self.knn(ref, random_p2, k=768)
        else:
            idx1 = np.random.choice(src.shape[0], 1024, replace=False),
            idx2 = np.random.choice(ref.shape[0], 1024, replace=False),

        # add noise
        if self.add_noise:
            sample["points_src"] = self.jitter(src[idx1, :])
            sample["points_ref"] = self.jitter(ref[idx2, :])
        else:
            sample["points_src"] = src[idx1, :]
            sample["points_ref"] = ref[idx2, :]

        return sample


class PRNetRotationTorch:

    def __init__(self,
                 num_points,
                 rot_mag,
                 trans_mag,
                 noise_std=0.01,
                 clip=0.05,
                 add_noise=True,
                 only_z=False,
                 partial=True):
        self.num_points = num_points
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.noise_std = noise_std
        self.clip = clip
        self.add_noise = add_noise
        self.only_z = only_z
        self.partial = partial

    def apply_transform(self, p0, transform_mat):
        p1 = se3.np_transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        gt = transform_mat

        return p1, gt

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0,
                                         scale=self.noise_std,
                                         size=(pts.shape[0], 3)),
                        a_min=-self.clip,
                        a_max=self.clip)
        noise = torch.from_numpy(noise).to(pts.device)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def knn(self, pts, random_pt, k):
        random_pt = torch.from_numpy(random_pt).to(pts.device)
        distance = torch.sum((pts - random_pt)**2, dim=1)
        idx = distance.topk(k=k, dim=0,
                            largest=False)[1]  # (batch_size, num_points, k)
        return idx

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        src = sample["points_src"]
        ref = sample["points_ref"]

        # random rotate src and ref first
        anglex = np.random.uniform() * np.pi * 2
        angley = np.random.uniform() * np.pi * 2
        anglez = np.random.uniform() * np.pi * 2
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
        R_ab = Rx @ Ry @ Rz
        t_ab = np.array([0, 0, 0])
        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]),
                                  axis=1).astype(np.float32)
        src, _ = self.apply_transform(src, rand_SE3)
        ref, _ = self.apply_transform(ref, rand_SE3)

        # Generate rigid transform
        anglex = np.random.uniform() * np.pi * self.rot_mag / 180.0
        angley = np.random.uniform() * np.pi * self.rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * self.rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

        if not self.only_z:
            R_ab = Rx @ Ry @ Rz
        else:
            R_ab = Rz
        t_ab = np.random.uniform(-self.trans_mag, self.trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]),
                                  axis=1).astype(np.float32)
        ref, transform_s_r = self.apply_transform(ref, rand_SE3)
        # Apply to source to get reference
        sample["transform_gt"] = transform_s_r
        sample["pose_gt"] = se3.np_mat2quat(transform_s_r)

        # Crop and sample
        if self.partial:
            src = torch.from_numpy(src)
            ref = torch.from_numpy(ref)
            random_p1 = np.random.random(size=(1, 3)) + np.array(
                [[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
            idx1 = self.knn(src, random_p1, k=768)
            # np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
            random_p2 = random_p1
            idx2 = self.knn(ref, random_p2, k=768)
        else:
            idx1 = np.random.choice(src.shape[0], 1024, replace=False),
            idx2 = np.random.choice(ref.shape[0], 1024, replace=False),
            # src = np.squeeze(src, axis=-1)
            # ref = np.squeeze(ref, axis=-1)
            src = torch.from_numpy(src)
            ref = torch.from_numpy(ref)

        # add noise
        if self.add_noise:
            sample["points_src"] = self.jitter(src[idx1, :])
            sample["points_ref"] = self.jitter(ref[idx2, :])
        else:
            sample["points_src"] = src[idx1, :]
            sample["points_ref"] = ref[idx2, :]
        if sample["points_src"].size()[0] == 1:
            sample["points_src"] = sample["points_src"].squeeze(0)
            sample["points_ref"] = sample["points_ref"].squeeze(0)

        # # for inference time
        # if sample["points_src"].shape[0] < self.num_points:
        #     rand_idxs = np.concatenate(
        #         [np.random.choice(sample["points_src"].shape[0], sample["points_src"].shape[0], replace=False),
        #          np.random.choice(sample["points_src"].shape[0], self.num_points - sample["points_src"].shape[0],
        #                           replace=True)])
        #     sample["points_src"] = sample["points_src"][rand_idxs, :]
        #     rand_idxs = np.concatenate(
        #         [np.random.choice(sample["points_ref"].shape[0], sample["points_ref"].shape[0], replace=False),
        #          np.random.choice(sample["points_ref"].shape[0], self.num_points - sample["points_ref"].shape[0],
        #                           replace=True)])
        #     sample["points_ref"] = sample["points_ref"][rand_idxs, :]
        # else:
        #     rand_idxs = np.random.choice(sample["points_src"].shape[0], self.num_points, replace=False),
        #     sample["points_src"] = sample["points_src"][rand_idxs, :]
        #     rand_idxs = np.random.choice(sample["points_ref"].shape[0], self.num_points, replace=False),
        #     sample["points_ref"] = sample["points_ref"][rand_idxs, :]
        # if sample["points_src"].shape[0] == 1:
        #     sample["points_src"] = sample["points_src"].squeeze(0)
        #     sample["points_ref"] = sample["points_ref"].squeeze(0)

        return sample


class PRNetUSIP:

    def __init__(self,
                 num_points,
                 rot_mag,
                 trans_mag,
                 noise_std=0.01,
                 clip=0.05,
                 add_noise=True,
                 partial=True):
        self.num_points = num_points
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.noise_std = noise_std
        self.clip = clip
        self.add_noise = add_noise
        self.partial = partial

    def apply_transform(self, p0, transform_mat):
        p1 = se3.np_transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        gt = transform_mat

        return p1, gt

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0,
                                         scale=self.noise_std,
                                         size=(pts.shape[0], 3)),
                        a_min=-self.clip,
                        a_max=self.clip)
        noise = torch.from_numpy(noise).to(pts.device)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def knn(self, pts, random_pt, k):
        random_pt = torch.from_numpy(random_pt).to(pts.device)
        distance = torch.sum((pts - random_pt)**2, dim=1)
        idx = distance.topk(k=k, dim=0,
                            largest=False)[1]  # (batch_size, num_points, k)
        return idx

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        src = sample["points_src"]
        ref = sample["points_ref"]

        # # random rotate src and ref first
        # anglex = np.random.uniform() * np.pi * 2
        # angley = np.random.uniform() * np.pi * 2
        # anglez = np.random.uniform() * np.pi * 2
        # cosx = np.cos(anglex)
        # cosy = np.cos(angley)
        # cosz = np.cos(anglez)
        # sinx = np.sin(anglex)
        # siny = np.sin(angley)
        # sinz = np.sin(anglez)
        # Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        # Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        # Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
        # R_ab = Rx @ Ry @ Rz
        # t_ab = np.array([0, 0, 0])
        # rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        # src, _ = self.apply_transform(src, rand_SE3)
        # ref, _ = self.apply_transform(ref, rand_SE3)

        # Generate rigid transform
        anglex = np.random.uniform() * np.pi * self.rot_mag / 180.0
        angley = np.random.uniform() * np.pi * self.rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * self.rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-self.trans_mag, self.trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]),
                                  axis=1).astype(np.float32)
        ref, transform_s_r = self.apply_transform(ref, rand_SE3)
        src_aug, _ = self.apply_transform(src, rand_SE3)
        # Apply to source to get reference
        sample["transform_gt"] = transform_s_r
        sample["pose_gt"] = se3.np_mat2quat(transform_s_r)

        # Crop and sample
        if self.partial:
            src = torch.from_numpy(src)
            ref = torch.from_numpy(ref)
            src_aug = torch.from_numpy(src_aug)
            random_p1 = np.random.random(size=(1, 3)) + np.array(
                [[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
            idx1 = self.knn(src, random_p1, k=768)
            # np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
            random_p2 = random_p1
            idx2 = self.knn(ref, random_p2, k=768)
        else:
            idx1 = np.random.choice(src.shape[0], 1024, replace=False),
            idx2 = np.random.choice(ref.shape[0], 1024, replace=False),
            # src = np.squeeze(src, axis=-1)
            # ref = np.squeeze(ref, axis=-1)
            src = torch.from_numpy(src)
            ref = torch.from_numpy(ref)
            src_aug = torch.from_numpy(src_aug)

        # add noise
        if self.add_noise:
            sample["points_src"] = self.jitter(src[idx1, :])
            sample["points_ref"] = self.jitter(ref[idx2, :])
            sample["points_src_aug"] = self.jitter(src_aug[idx1, :])
        else:
            sample["points_src"] = src[idx1, :]
            sample["points_ref"] = ref[idx2, :]
            sample["points_src_aug"] = src_aug[idx1, :]

        pc_src, pc_src_aug, pc_ref = open3d.geometry.PointCloud(
        ), open3d.geometry.PointCloud(), open3d.geometry.PointCloud()
        pc_src.points = open3d.utility.Vector3dVector(
            sample["points_src"].numpy())
        pc_src_aug.points = open3d.utility.Vector3dVector(
            sample["points_src_aug"].numpy())
        pc_ref.points = open3d.utility.Vector3dVector(
            sample["points_ref"].numpy())
        pc_src.estimate_normals(
            open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pc_src_aug.estimate_normals(
            open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pc_ref.estimate_normals(
            open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        sample["points_src_norm"] = np.asarray(pc_src.normals)
        sample["points_src_aug_norm"] = np.asarray(pc_src_aug.normals)
        sample["points_ref_norm"] = np.asarray(pc_ref.normals)

        # # for batch size = 1
        # if sample["points_src"].size()[0] == 1:
        #     sample["points_src"] = sample["points_src"].squeeze(0)
        #     sample["points_ref"] = sample["points_ref"].squeeze(0)
        #     sample["points_src_aug"] = sample["points_src_aug"].squeeze(0)

        # # for inference time
        # if sample["points_src"].shape[0] < self.num_points:
        #     rand_idxs = np.concatenate(
        #         [np.random.choice(sample["points_src"].shape[0], sample["points_src"].shape[0], replace=False),
        #          np.random.choice(sample["points_src"].shape[0], self.num_points - sample["points_src"].shape[0],
        #                           replace=True)])
        #     sample["points_src"] = sample["points_src"][rand_idxs, :]
        #     rand_idxs = np.concatenate(
        #         [np.random.choice(sample["points_ref"].shape[0], sample["points_ref"].shape[0], replace=False),
        #          np.random.choice(sample["points_ref"].shape[0], self.num_points - sample["points_ref"].shape[0],
        #                           replace=True)])
        #     sample["points_ref"] = sample["points_ref"][rand_idxs, :]
        # else:
        #     rand_idxs = np.random.choice(sample["points_src"].shape[0], self.num_points, replace=False),
        #     sample["points_src"] = sample["points_src"][rand_idxs, :]
        #     rand_idxs = np.random.choice(sample["points_ref"].shape[0], self.num_points, replace=False),
        #     sample["points_ref"] = sample["points_ref"][rand_idxs, :]
        # if sample["points_src"].shape[0] == 1:
        #     sample["points_src"] = sample["points_src"].squeeze(0)
        #     sample["points_ref"] = sample["points_ref"].squeeze(0)

        return sample


class StanfordPRNetTorch:

    def __init__(self,
                 num_points,
                 rot_mag,
                 trans_mag,
                 noise_std=0.01,
                 clip=0.05,
                 add_noise=True,
                 only_z=False,
                 partial=True):
        self.num_points = num_points
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.noise_std = noise_std
        self.clip = clip
        self.add_noise = add_noise
        self.only_z = only_z
        self.partial = partial

    def apply_transform(self, p0, transform_mat):
        p1 = se3.np_transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        gt = transform_mat

        return p1, gt

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0,
                                         scale=self.noise_std,
                                         size=(pts.shape[0], 3)),
                        a_min=-self.clip,
                        a_max=self.clip)
        noise = torch.from_numpy(noise).to(pts.device)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def knn(self, pts, random_pt, k):
        random_pt = torch.from_numpy(random_pt).to(pts.device)
        distance = torch.sum((pts - random_pt)**2, dim=1)
        idx = distance.topk(k=k, dim=0,
                            largest=False)[1]  # (batch_size, num_points, k)
        return idx

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        src = sample["points_src"]
        ref = sample["points_ref"]
        # Generate rigid transform
        anglex = np.random.uniform() * np.pi * self.rot_mag / 180.0
        angley = np.random.uniform() * np.pi * self.rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * self.rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

        if not self.only_z:
            R_ab = Rx @ Ry @ Rz
        else:
            R_ab = Rz
        t_ab = np.random.uniform(-self.trans_mag, self.trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]),
                                  axis=1).astype(np.float32)
        ref, transform_s_r = self.apply_transform(ref, rand_SE3)
        # Apply to source to get reference
        sample["transform_gt"] = transform_s_r
        sample["pose_gt"] = se3.np_mat2quat(transform_s_r)

        # Crop and sample
        if self.partial:
            src = torch.from_numpy(src)
            ref = torch.from_numpy(ref)
            random_p1 = np.random.random(size=(1, 3)) + np.array(
                [[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
            idx1 = self.knn(src, random_p1, k=768)
            # np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
            random_p2 = random_p1
            idx2 = self.knn(ref, random_p2, k=768)
        else:
            idx1 = np.random.choice(src.shape[0], 1024, replace=False),
            idx2 = np.random.choice(ref.shape[0], 1024, replace=False),
            # src = np.squeeze(src, axis=-1)
            # ref = np.squeeze(ref, axis=-1)
            src = torch.from_numpy(src)
            ref = torch.from_numpy(ref)

        # add noise
        if self.add_noise:
            sample["points_src"] = self.jitter(src[idx1, :])
            sample["points_ref"] = self.jitter(ref[idx2, :])
        else:
            sample["points_src"] = src[idx1, :]
            sample["points_ref"] = ref[idx2, :]
        if sample["points_src"].size()[0] == 1:
            sample["points_src"] = sample["points_src"].squeeze(0)
            sample["points_ref"] = sample["points_ref"].squeeze(0)

        # # for inference time
        # if sample["points_src"].shape[0] < self.num_points:
        #     rand_idxs = np.concatenate(
        #         [np.random.choice(sample["points_src"].shape[0], sample["points_src"].shape[0], replace=False),
        #          np.random.choice(sample["points_src"].shape[0], self.num_points - sample["points_src"].shape[0],
        #                           replace=True)])
        #     sample["points_src"] = sample["points_src"][rand_idxs, :]
        #     rand_idxs = np.concatenate(
        #         [np.random.choice(sample["points_ref"].shape[0], sample["points_ref"].shape[0], replace=False),
        #          np.random.choice(sample["points_ref"].shape[0], self.num_points - sample["points_ref"].shape[0],
        #                           replace=True)])
        #     sample["points_ref"] = sample["points_ref"][rand_idxs, :]
        # else:
        #     rand_idxs = np.random.choice(sample["points_src"].shape[0], self.num_points, replace=False),
        #     sample["points_src"] = sample["points_src"][rand_idxs, :]
        #     rand_idxs = np.random.choice(sample["points_ref"].shape[0], self.num_points, replace=False),
        #     sample["points_ref"] = sample["points_ref"][rand_idxs, :]
        # if sample["points_src"].shape[0] == 1:
        #     sample["points_src"] = sample["points_src"].squeeze(0)
        #     sample["points_ref"] = sample["points_ref"].squeeze(0)

        return sample


class Scenes7Real:

    def __init__(self,
                 rot_mag,
                 trans_mag,
                 noise_std=0.01,
                 clip=0.05,
                 transform=False,
                 normalization=True):
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.noise_std = noise_std
        self.clip = clip
        self.transform = transform
        self.normalization = normalization

    def apply_transform(self, p0, transform_mat):
        p1 = se3.np_transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        gt = transform_mat

        return p1, gt

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0,
                                         scale=self.noise_std,
                                         size=(pts.shape[0], 3)),
                        a_min=-self.clip,
                        a_max=self.clip)
        # noise = torch.from_numpy(noise).to(pts.device)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def knn(self, pts, random_pt, k):
        random_pt = torch.from_numpy(random_pt).to(pts.device)
        distance = torch.sum((pts - random_pt)**2, dim=1)
        idx = distance.topk(k=k, dim=0,
                            largest=False)[1]  # (batch_size, num_points, k)
        return idx

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        src = sample["points_src"]
        ref = sample["points_ref"]

        # Normalization
        if self.normalization:
            src_ref = np.concatenate((src, ref), axis=0)
            mean = np.mean(src_ref, axis=0)
            src = src - mean
            ref = ref - mean
            norm = np.max(np.sqrt(np.sum(src_ref**2, axis=1)))
            src = src / norm
            ref = ref / norm
            R_ab = sample["transform_gt"][:3, :3]
            t_ab = sample["transform_gt"][:3, 3]
            t_ab = (np.matmul(R_ab, mean) + t_ab - mean) / norm
            transform_gt = np.concatenate((R_ab, t_ab[:, None]),
                                          axis=1).astype(np.float32)
            # Apply to source to get reference
            sample["transform_gt"] = transform_gt
            sample["pose_gt"] = se3.np_mat2quat(transform_gt)
            sample["points_src_raw"] = (sample["points_src_raw"] - mean) / norm
            sample["points_ref_raw"] = (sample["points_ref_raw"] - mean) / norm
        else:
            sample["transform_gt"] = sample["transform_gt"]
            sample["pose_gt"] = se3.np_mat2quat(sample["transform_gt"])

        # Generate rigid transform
        if self.transform:
            anglex = np.random.uniform() * np.pi / 4
            angley = np.random.uniform() * np.pi / 4
            anglez = np.random.uniform() * np.pi / 4

            cosx = np.cos(anglex)
            cosy = np.cos(angley)
            cosz = np.cos(anglez)
            sinx = np.sin(anglex)
            siny = np.sin(angley)
            sinz = np.sin(anglez)
            Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
            Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
            Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

            rand_R_ab = Rx @ Ry @ Rz
            rand_t_ab = np.random.uniform(-self.trans_mag, self.trans_mag, 3)

            R_ab = sample["transform_gt"][:3, :3]
            t_ab = sample["transform_gt"][:3, 3]

            if (np.random.rand(1)[0] > 0.5):
                src = np.matmul(rand_R_ab, src.T).T + rand_t_ab
                R_ab = np.matmul(R_ab, rand_R_ab.T)
                t_ab = t_ab - np.matmul(R_ab, rand_t_ab)
            else:
                ref = np.matmul(rand_R_ab, ref.T).T + rand_t_ab
                R_ab = np.matmul(rand_R_ab, R_ab)
                t_ab = np.matmul(rand_R_ab, t_ab) + rand_t_ab

            src = src.astype(np.float32)
            ref = ref.astype(np.float32)
            transform_gt = np.concatenate((R_ab, t_ab[:, None]),
                                          axis=1).astype(np.float32)
            # Apply to source to get reference
            sample["transform_gt"] = transform_gt
            sample["pose_gt"] = se3.np_mat2quat(transform_gt)

            # add noise
            src = self.jitter(src)
            ref = self.jitter(ref)
        else:
            sample["transform_gt"] = sample["transform_gt"]
            sample["pose_gt"] = se3.np_mat2quat(sample["transform_gt"])

        sample["points_src"] = src
        sample["points_ref"] = ref
        if sample["points_src"].shape[0] == 1:
            sample["points_src"] = sample["points_src"].squeeze(0)
            sample["points_ref"] = sample["points_ref"].squeeze(0)

        # del sample["points_src_raw"]
        # del sample["points_ref_raw"]

        return sample


class Scenes7Sync:

    def __init__(self,
                 num_points,
                 rot_mag,
                 trans_mag,
                 partial_ratio=[0.75, 0.75],
                 noise_std=0.01,
                 clip=0.05,
                 add_noise=True,
                 normalization=True):
        self.num_points = num_points
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.partial_ratio = partial_ratio
        self.noise_std = noise_std
        self.clip = clip
        self.add_noise = add_noise
        self.normalization = normalization

    def apply_transform(self, p0, transform_mat):
        p1 = se3.np_transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        gt = transform_mat

        return p1, gt

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0,
                                         scale=self.noise_std,
                                         size=(pts.shape[0], 3)),
                        a_min=-self.clip,
                        a_max=self.clip)
        noise = torch.from_numpy(noise).to(pts.device)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def knn(self, pts, random_pt, k):
        random_pt = torch.from_numpy(random_pt).to(pts.device)
        distance = torch.sum((pts - random_pt)**2, dim=1)
        idx = distance.topk(k=k, dim=0,
                            largest=False)[1]  # (batch_size, num_points, k)
        return idx

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        src = sample["points_src"]
        ref = sample["points_ref"]
        src_raw = sample["points_src_raw"]
        ref_raw = sample["points_ref_raw"]

        # Normalization
        if self.normalization:
            src_mean = np.mean(src, axis=0)
            ref_mean = np.mean(ref, axis=0)
            src = src - src_mean
            src_raw = src_raw - src_mean
            ref = ref - ref_mean
            ref_raw = ref_raw - ref_mean
            src_ref = np.concatenate((src, ref), axis=0)
            norm = np.max(np.sqrt(np.sum(src_ref**2, axis=1)))
            # new_src = (old_src - mean) / norm
            src = src / norm
            src_raw = src_raw / norm
            ref = ref / norm
            ref_raw = ref_raw / norm
            sample["points_src_raw"] = src_raw
            sample["points_ref_raw"] = ref_raw

        # Crop and sample
        src = torch.from_numpy(src)
        ref = torch.from_numpy(ref)
        random_pt = np.random.random(size=(1, 3))
        random_p1 = random_pt + np.array([[500, 500, 500]])
        idx1 = self.knn(src,
                        random_p1,
                        k=int(src.size()[0] * self.partial_ratio[0]))
        random_p2 = random_pt + np.array([[500, -500, 500]])
        idx2 = self.knn(ref,
                        random_p2,
                        k=int(ref.size()[0] * self.partial_ratio[1]))
        sample["random_pt"] = np.concatenate((random_p1, random_p2), axis=0)

        # Generate rigid transform
        anglex = np.random.uniform() * np.pi * self.rot_mag / 180.0
        angley = np.random.uniform() * np.pi * self.rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * self.rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-self.trans_mag, self.trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]),
                                  axis=1).astype(np.float32)
        ref, transform_s_r = self.apply_transform(ref, rand_SE3)
        # Apply to source to get reference
        sample["transform_gt"] = transform_s_r
        sample["pose_gt"] = se3.np_mat2quat(transform_s_r)

        # add noise
        if self.add_noise:
            sample["points_src"] = self.jitter(src[idx1, :])
            sample["points_ref"] = self.jitter(ref[idx2, :])
        else:
            sample["points_src"] = src[idx1, :]
            sample["points_ref"] = ref[idx2, :]
        if sample["points_src"].size()[0] == 1:
            sample["points_src"] = sample["points_src"].squeeze(0)
            sample["points_ref"] = sample["points_ref"].squeeze(0)

        return sample


class KITTIReal:

    def __init__(self,
                 num_points,
                 rot_mag,
                 trans_mag,
                 noise_std=0.01,
                 clip=0.05,
                 add_noise=True,
                 transform=False,
                 normalization=True):
        self.num_points = num_points
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.noise_std = noise_std
        self.clip = clip
        self.add_noise = add_noise
        self.transform = transform
        self.normalization = normalization

    def resample(self, points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """
        # print("===", points.shape[0], k)
        if k < points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([
                np.random.choice(points.shape[0],
                                 points.shape[0],
                                 replace=False),
                np.random.choice(points.shape[0],
                                 k - points.shape[0],
                                 replace=True)
            ])
            return points[rand_idxs, :]

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0,
                                         scale=self.noise_std,
                                         size=(pts.shape[0], 3)),
                        a_min=-self.clip,
                        a_max=self.clip)
        # noise = torch.from_numpy(noise).to(pts.device)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        src = sample["points_src"]
        ref = sample["points_ref"]
        sample["transform_gt"] = se3.np_quat2mat(sample["pose_gt"]).squeeze(0)

        src = self.resample(src, k=4096)
        ref = self.resample(ref, k=4096)
        # sample["points_src_raw"] = self.resample(src, k=4000)
        # sample["points_ref_raw"] = self.resample(ref, k=4000)

        src = np.random.permutation(src)
        ref = np.random.permutation(ref)

        if self.transform:

            # Generate rigid transform
            anglex = np.random.uniform() * 0.25 * np.pi
            angley = np.random.uniform() * 0.25 * np.pi
            anglez = 0
            cosx = np.cos(anglex)
            cosy = np.cos(angley)
            cosz = np.cos(anglez)
            sinx = np.sin(anglex)
            siny = np.sin(angley)
            sinz = np.sin(anglez)
            Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
            Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
            Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

            rand_R_ab = Rx @ Ry @ Rz

            R_ab = sample["transform_gt"][:3, :3]
            t_ab = sample["transform_gt"][:3, 3]
            # import pdb
            # from data_loader.npz import NpzMaker
            # pdb.set_trace()

            if (np.random.rand(1)[0] > 0.5):
                src = np.matmul(rand_R_ab, src.T).T
                R_ab = np.matmul(R_ab, rand_R_ab.T)
            else:
                ref = np.matmul(rand_R_ab, ref.T).T
                R_ab = np.matmul(rand_R_ab, R_ab)
                t_ab = np.matmul(rand_R_ab, t_ab)

            # src = src.astype(np.float32)
            # ref = ref.astype(np.float32)
            transform_gt = np.concatenate((R_ab, t_ab[:, None]), axis=1)
            sample["transform_gt"] = transform_gt.astype(
                np.float32)  # Apply to source to get reference
            sample["pose_gt"] = se3.np_mat2quat(transform_gt).astype(
                np.float32)

        else:
            sample["transform_gt"] = sample["transform_gt"].astype(np.float32)
            sample["pose_gt"] = sample["pose_gt"].astype(np.float32)

        # Normalization
        if self.normalization:
            src_mean = np.mean(src, axis=0)
            ref_mean = np.mean(ref, axis=0)
            src = src - src_mean
            ref = ref - ref_mean
            src_ref = np.concatenate((src, ref), axis=0)
            norm = np.max(np.sqrt(np.sum(src_ref**2, axis=1)))
            # new_src = (old_src - mean) / norm
            src = src / norm
            ref = ref / norm
            R_ab = sample["transform_gt"][:3, :3]
            t_ab = sample["transform_gt"][:3, 3]
            t_ab = (np.matmul(R_ab, src_mean) + t_ab - ref_mean) / norm
            transform_gt = np.concatenate((R_ab, t_ab[:, None]),
                                          axis=1).astype(np.float32)
            sample["transform_gt"] = transform_gt.astype(
                np.float32)  # Apply to source to get reference
            sample["pose_gt"] = se3.np_mat2quat(transform_gt).astype(
                np.float32)
            # src_mean = torch.mean(src, dim=0)
            # ref_mean = torch.mean(ref, dim=0)
            # src = src - src_mean
            # ref = ref - ref_mean
            # src_ref = torch.cat((src, ref), dim=0)
            # norm = torch.max(torch.sqrt(torch.sum(src_ref ** 2, dim=1)))
            # # new_src = (old_src - mean) / norm
            # src = src / norm
            # ref = ref / norm
            # R_ab = sample["transform_gt"][:3, :3]
            # t_ab = sample["transform_gt"][:3, 3]
            # t_ab = (torch.matmul(R_ab, src_mean) + t_ab - ref_mean) / norm
            # transform_gt = torch.cat((R_ab, t_ab[:, None]), dim=1)
            # sample["transform_gt"] = transform_gt  # Apply to source to get reference
            # sample["pose_gt"] = se3_torch.mat2quat(transform_gt)

        # import pdb
        # pdb.set_trace()
        # import open3d
        # src_pc, ref_pc, src_trans_pc = open3d.PointCloud(), open3d.PointCloud(), open3d.PointCloud()
        # src_pc.points = open3d.utility.Vector3dVector(src)
        # ref_pc.points = open3d.utility.Vector3dVector(ref)
        # src_trans_pc.points = open3d.utility.Vector3dVector(se3.transform(sample["transform_gt"], src))
        # open3d.write_point_cloud("./src.ply", src_pc)
        # open3d.write_point_cloud("./ref.ply", ref_pc)
        # open3d.write_point_cloud("./src_trans.ply", src_trans_pc)

        # add noise
        if self.add_noise:
            sample["points_src"] = self.jitter(src).astype(np.float32)
            sample["points_ref"] = self.jitter(ref).astype(np.float32)
        else:
            sample["points_src"] = src.astype(np.float32)
            sample["points_ref"] = ref.astype(np.float32)
        if sample["points_src"].shape[0] == 1:
            sample["points_src"] = sample["points_src"].squeeze(0)
            sample["points_ref"] = sample["points_ref"].squeeze(0)

        return sample


class KITTISync:

    def __init__(self,
                 num_points,
                 rot_mag,
                 trans_mag,
                 partial_ratio=[0.75, 0.75],
                 noise_std=0.01,
                 clip=0.05,
                 add_noise=True,
                 normalization=True):
        self.num_points = num_points
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.partial_ratio = partial_ratio
        self.noise_std = noise_std
        self.clip = clip
        self.add_noise = add_noise
        self.normalization = normalization

    def apply_transform(self, p0, transform_mat):
        p1 = se3.np_transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        gt = transform_mat

        return p1, gt

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0,
                                         scale=self.noise_std,
                                         size=(pts.shape[0], 3)),
                        a_min=-self.clip,
                        a_max=self.clip)
        noise = torch.from_numpy(noise).to(pts.device)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def knn(self, pts, random_pt, k):
        random_pt = torch.from_numpy(random_pt).to(pts.device)
        distance = torch.sum((pts - random_pt)**2, dim=1)
        idx = distance.topk(k=k, dim=0,
                            largest=False)[1]  # (batch_size, num_points, k)
        return idx

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        src = sample["points_src"]
        ref = sample["points_ref"]
        src_raw = sample["points_src_raw"]
        ref_raw = sample["points_ref_raw"]

        # Normalization
        if self.normalization:
            src_mean = np.mean(src, axis=0)
            ref_mean = np.mean(ref, axis=0)
            src = src - src_mean
            src_raw = src_raw - src_mean
            ref = ref - ref_mean
            ref_raw = ref_raw - ref_mean
            src_ref = np.concatenate((src, ref), axis=0)
            norm = np.max(np.sqrt(np.sum(src_ref**2, axis=1)))
            # new_src = (old_src - mean) / norm
            src = src / norm
            src_raw = src_raw / norm
            ref = ref / norm
            ref_raw = ref_raw / norm
            sample["points_src_raw"] = src_raw
            sample["points_ref_raw"] = ref_raw

        # Crop and sample
        src = torch.from_numpy(src)
        ref = torch.from_numpy(ref)
        random_pt = np.random.random(size=(1, 3))
        random_p1 = random_pt + np.array([[500, 500, 500]])
        idx1 = self.knn(src,
                        random_p1,
                        k=int(src.size()[0] * self.partial_ratio[0]))
        random_p2 = random_pt + np.array([[500, -500, 500]])
        idx2 = self.knn(ref,
                        random_p2,
                        k=int(ref.size()[0] * self.partial_ratio[1]))
        sample["random_pt"] = np.concatenate((random_p1, random_p2), axis=0)

        # Generate rigid transform
        anglex = np.random.uniform() * np.pi * self.rot_mag / 180.0
        angley = np.random.uniform() * np.pi * self.rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * self.rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-self.trans_mag, self.trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]),
                                  axis=1).astype(np.float32)
        ref, transform_s_r = self.apply_transform(ref, rand_SE3)
        # Apply to source to get reference
        sample["transform_gt"] = transform_s_r
        sample["pose_gt"] = se3.np_mat2quat(transform_s_r)

        # add noise
        if self.add_noise:
            sample["points_src"] = self.jitter(src[idx1, :])
            sample["points_ref"] = self.jitter(ref[idx2, :])
        else:
            sample["points_src"] = src[idx1, :]
            sample["points_ref"] = ref[idx2, :]
        if sample["points_src"].size()[0] == 1:
            sample["points_src"] = sample["points_src"].squeeze(0)
            sample["points_ref"] = sample["points_ref"].squeeze(0)

        return sample


class PRNetTorchOverlapRatio:

    def __init__(self,
                 num_points,
                 rot_mag,
                 trans_mag,
                 noise_std=0.01,
                 clip=0.05,
                 add_noise=True,
                 overlap_ratio=0.8):
        self.num_points = num_points
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.noise_std = noise_std
        self.clip = clip
        self.add_noise = add_noise
        self.overlap_ratio = overlap_ratio

    def apply_transform(self, p0, transform_mat):
        p1 = se3.np_transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        gt = transform_mat

        return p1, gt

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0,
                                         scale=self.noise_std,
                                         size=(pts.shape[0], 3)),
                        a_min=-self.clip,
                        a_max=self.clip)
        noise = torch.from_numpy(noise).to(pts.device)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def knn(self, pts, random_pt, k1, k2):
        random_pt = torch.from_numpy(random_pt).to(pts.device)
        distance = torch.sum((pts - random_pt)**2, dim=1)
        idx1 = distance.topk(k=k1, dim=0,
                             largest=False)[1]  # (batch_size, num_points, k)
        idx2 = distance.topk(k=k2, dim=0, largest=True)[1]
        return idx1, idx2

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        src = sample["points_src"]
        ref = sample["points_ref"]

        # Crop and sample
        src = torch.from_numpy(src)
        ref = torch.from_numpy(ref)
        random_p1 = np.random.random(size=(1, 3)) + np.array(
            [[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
        # pdb.set_trace()
        src_idx1, src_idx2 = self.knn(src, random_p1, k1=768, k2=2048 - 768)
        ref_idx1, ref_idx2 = self.knn(ref, random_p1, k1=768, k2=2048 - 768)

        k1_idx = ref_idx1[np.random.randint(ref_idx1.size()[0])]
        k1 = ref[k1_idx, :]
        k2_idx = ref_idx1[0]
        k2 = ref[k2_idx, :]

        distance = torch.sum((ref[ref_idx1, :] - k1)**2, dim=1)
        overlap_idx = distance.topk(k=int(768 * self.overlap_ratio),
                                    dim=0,
                                    largest=False)[1]
        k1_points = ref[ref_idx1, :][overlap_idx, :]
        distance = torch.sum((ref[ref_idx2, :] - k2)**2, dim=1)
        nonoverlap_idx = distance.topk(k=768 - int(768 * self.overlap_ratio),
                                       dim=0,
                                       largest=False)[1]
        k2_points = ref[ref_idx2, :][nonoverlap_idx, :]
        ref = torch.cat((k1_points, k2_points), dim=0)
        src = src[src_idx1, :]
        # pdb.set_trace()

        # Generate rigid transform
        anglex = np.random.uniform() * np.pi * self.rot_mag / 180.0
        angley = np.random.uniform() * np.pi * self.rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * self.rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-self.trans_mag, self.trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]),
                                  axis=1).astype(np.float32)
        ref, transform_s_r = self.apply_transform(ref, rand_SE3)
        # Apply to source to get reference
        sample["transform_gt"] = transform_s_r
        sample["pose_gt"] = se3.np_mat2quat(transform_s_r)

        # add noise
        if self.add_noise:
            sample["points_src"] = self.jitter(src)
            sample["points_ref"] = self.jitter(ref)
        else:
            sample["points_src"] = src
            sample["points_ref"] = ref

        if sample["points_src"].size()[0] == 1:
            sample["points_src"] = sample["points_src"].squeeze(0)
            sample["points_ref"] = sample["points_ref"].squeeze(0)

        return sample


class Timming:

    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        sample["transform_gt"] = np.eye(4)
        sample["pose_gt"] = se3.np_mat2quat(np.eye(4))

        # for inference time
        if sample["points_src"].shape[0] < self.num_points:
            rand_idxs = np.concatenate([
                np.random.choice(sample["points_src"].shape[0],
                                 sample["points_src"].shape[0],
                                 replace=False),
                np.random.choice(sample["points_src"].shape[0],
                                 self.num_points -
                                 sample["points_src"].shape[0],
                                 replace=True)
            ])
            sample["points_src"] = sample["points_src"][rand_idxs, :]
            rand_idxs = np.concatenate([
                np.random.choice(sample["points_ref"].shape[0],
                                 sample["points_ref"].shape[0],
                                 replace=False),
                np.random.choice(sample["points_ref"].shape[0],
                                 self.num_points -
                                 sample["points_ref"].shape[0],
                                 replace=True)
            ])
            sample["points_ref"] = sample["points_ref"][rand_idxs, :]
        else:
            rand_idxs = np.random.choice(sample["points_src"].shape[0],
                                         self.num_points,
                                         replace=False),
            sample["points_src"] = sample["points_src"][rand_idxs, :]
            rand_idxs = np.random.choice(sample["points_ref"].shape[0],
                                         self.num_points,
                                         replace=False),
            sample["points_ref"] = sample["points_ref"][rand_idxs, :]

        if sample["points_src"].shape[0] == 1:
            sample["points_src"] = sample["points_src"].squeeze(0)
            sample["points_ref"] = sample["points_ref"].squeeze(0)

        return sample


class DeepGMR:

    def __init__(self):
        self.k = 20

    def knn_idx(self, pts):
        kdt = cKDTree(pts)
        _, idx = kdt.query(pts, k=self.k + 1)
        return idx[:, 1:]

    def get_rri(self, pts, k):
        # pts: N x 3, original points
        # q: N x K x 3, nearest neighbors
        q = pts[self.knn_idx(pts)]
        p = np.repeat(pts[:, None], self.k, axis=1)
        # rp, rq: N x K x 1, norms
        rp = np.linalg.norm(p, axis=-1, keepdims=True)
        rq = np.linalg.norm(q, axis=-1, keepdims=True)
        pn = p / rp
        qn = q / rq
        dot = np.sum(pn * qn, -1, keepdims=True)
        # theta: N x K x 1, angles
        theta = np.arccos(np.clip(dot, -1, 1))
        T_q = q - dot * p
        sin_psi = np.sum(
            np.cross(T_q[:, None], T_q[:, :, None]) * pn[:, None], -1)
        cos_psi = np.sum(T_q[:, None] * T_q[:, :, None], -1)
        psi = np.arctan2(sin_psi, cos_psi) % (2 * np.pi)
        idx = np.argpartition(psi, 1)[:, :, 1:2]
        # phi: N x K x 1, projection angles
        phi = np.take_along_axis(psi, idx, axis=-1)
        feat = np.concatenate([rp, rq, theta, phi], axis=-1)
        return feat.reshape(-1, self.k * 4)

    def get_rri_cuda(self, pts, npts_per_block=1):
        import pycuda.autoinit
        mod_rri = SourceModule(
            open("other_methods/deepgmr/rri.cu").read() %
            (self.k, npts_per_block))
        rri_cuda = mod_rri.get_function("get_rri_feature")

        N = len(pts)
        pts_gpu = gpuarray.to_gpu(pts.astype(np.float32).ravel())
        k_idx = self.knn_idx(pts)
        k_idx_gpu = gpuarray.to_gpu(k_idx.astype(np.int32).ravel())
        feat_gpu = gpuarray.GPUArray((N * self.k * 4, ), np.float32)

        rri_cuda(pts_gpu,
                 np.int32(N),
                 k_idx_gpu,
                 feat_gpu,
                 grid=(((N - 1) // npts_per_block) + 1, 1),
                 block=(npts_per_block, self.k, 1))

        feat = feat_gpu.get().reshape(N, self.k * 4).astype(np.float32)
        return feat

    def __call__(self, sample):
        if isinstance(sample["points_src"], torch.Tensor):
            pcd1 = sample["points_src"].numpy()
            pcd2 = sample["points_ref"].numpy()
        else:
            pcd1 = sample["points_src"]
            pcd2 = sample["points_ref"]
        if torch.cuda.is_available():
            pcd1 = np.concatenate([
                pcd1[:, :3],
                self.get_rri_cuda(pcd1[:, :3] - pcd1[:, :3].mean(axis=0))
            ],
                                  axis=1)
            pcd2 = np.concatenate([
                pcd2[:, :3],
                self.get_rri_cuda(pcd2[:, :3] - pcd2[:, :3].mean(axis=0))
            ],
                                  axis=1)
        else:
            pcd1 = np.concatenate([
                pcd1[:, :3],
                self.get_rri(pcd1[:, :3] - pcd1[:, :3].mean(axis=0))
            ],
                                  axis=1)
            pcd2 = np.concatenate([
                pcd2[:, :3],
                self.get_rri(pcd2[:, :3] - pcd2[:, :3].mean(axis=0))
            ],
                                  axis=1)
        sample["points_src"] = pcd1
        sample["points_ref"] = pcd2

        return sample


class ComputeNormals:

    def __init__(self):
        self.k = 20
        self.radius = 0.1

    def __call__(self, sample):
        src = sample["points_src"]
        ref = sample["points_ref"]
        if isinstance(src, torch.Tensor):
            src = src.numpy()
            ref = ref.numpy()
        src_pcd, ref_pcd = open3d.geometry.PointCloud(
        ), open3d.geometry.PointCloud()

        src_pcd.points = open3d.utility.Vector3dVector(src)
        ref_pcd.points = open3d.utility.Vector3dVector(ref)

        src_pcd.estimate_normals(
            open3d.geometry.KDTreeSearchParamHybrid(radius=self.radius,
                                                    max_nn=self.k))
        ref_pcd.estimate_normals(
            open3d.geometry.KDTreeSearchParamHybrid(radius=self.radius,
                                                    max_nn=self.k))

        sample["norm_src"] = np.asarray(src_pcd.normals)
        sample["norm_ref"] = np.asarray(ref_pcd.normals)
        return sample


class RGM:

    def __init__(self):
        pass

    def nearest_neighbor(self, src, dst):
        """
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        """

        assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()

    def __call__(self, sample):
        if isinstance(sample["points_src"], torch.Tensor):
            sample["points_src"] = sample["points_src"].numpy()
            sample["points_ref"] = sample["points_ref"].numpy()
        if isinstance(sample["transform_gt"], torch.Tensor):
            sample["transform_gt"] = sample["transform_gt"].numpy()
        perm_mat = np.zeros(
            (sample["points_src"].shape[0], sample["points_ref"].shape[0]))
        inlier_src = np.zeros((sample["points_src"].shape[0], 1))
        inlier_ref = np.zeros((sample["points_ref"].shape[0], 1))
        points_src_transform = se3.np_transform(sample["transform_gt"],
                                                sample["points_src"][:, :3])
        points_ref = sample["points_ref"][:, :3]
        dist_s2et, indx_s2et = self.nearest_neighbor(points_src_transform,
                                                     points_ref)
        dist_t2es, indx_t2es = self.nearest_neighbor(points_ref,
                                                     points_src_transform)
        padtype = 3  #双边对应填充， 完全填充，双边对应填充+部分对应填充，
        padth = 0.05
        if padtype == 1:
            for row_i in range(sample["points_src"].shape[0]):
                if indx_t2es[indx_s2et[
                        row_i]] == row_i and dist_s2et[row_i] < padth:
                    perm_mat[row_i, indx_s2et[row_i]] = 1
        elif padtype == 2:
            for row_i in range(sample["points_src"].shape[0]):
                if dist_s2et[row_i] < padth:
                    perm_mat[row_i, indx_s2et[row_i]] = 1
            for col_i in range(sample["points_ref"].shape[0]):
                if dist_t2es[col_i] < padth:
                    perm_mat[indx_t2es[col_i], col_i] = 1
        elif padtype == 3:
            for row_i in range(sample["points_src"].shape[0]):
                if indx_t2es[indx_s2et[
                        row_i]] == row_i and dist_s2et[row_i] < padth:
                    perm_mat[row_i, indx_s2et[row_i]] = 1
            for row_i in range(sample["points_src"].shape[0]):
                if np.sum(perm_mat[row_i, :])==0 \
                        and np.sum(perm_mat[:, indx_s2et[row_i]])==0 \
                        and dist_s2et[row_i]<padth:
                    perm_mat[row_i, indx_s2et[row_i]] = 1
            for col_i in range(sample["points_ref"].shape[0]):
                if np.sum(perm_mat[:, col_i])==0 \
                        and np.sum(perm_mat[indx_t2es[col_i], :])==0 \
                        and dist_t2es[col_i]<padth:
                    perm_mat[indx_t2es[col_i], col_i] = 1
            outlier_src_ind = np.where(np.sum(perm_mat, axis=1) == 0)[0]
            outlier_ref_ind = np.where(np.sum(perm_mat, axis=0) == 0)[0]
            points_src_transform_rest = points_src_transform[outlier_src_ind]
            points_ref_rest = points_ref[outlier_ref_ind]
            if points_src_transform_rest.shape[
                    0] > 0 and points_ref_rest.shape[0] > 0:
                dist_s2et, indx_s2et = self.nearest_neighbor(
                    points_src_transform_rest, points_ref_rest)
                dist_t2es, indx_t2es = self.nearest_neighbor(
                    points_ref_rest, points_src_transform_rest)
                for row_i in range(points_src_transform_rest.shape[0]):
                    if indx_t2es[indx_s2et[
                            row_i]] == row_i and dist_s2et[row_i] < padth * 2:
                        perm_mat[outlier_src_ind[row_i],
                                 outlier_ref_ind[indx_s2et[row_i]]] = 1
        inlier_src_ind = np.where(np.sum(perm_mat, axis=1))[0]
        inlier_ref_ind = np.where(np.sum(perm_mat, axis=0))[0]
        inlier_src[inlier_src_ind] = 1
        inlier_ref[inlier_ref_ind] = 1

        T_ab = sample["transform_gt"]
        T_ba = np.concatenate(
            (T_ab[:, :3].T,
             np.expand_dims(-(T_ab[:, :3].T).dot(T_ab[:, 3]), axis=1)),
            axis=-1)

        n1_gt, n2_gt = perm_mat.shape
        A1_gt, e1_gt = rgm_build_graphs.build_graphs(sample["points_src"],
                                                     inlier_src,
                                                     n1_gt,
                                                     stg="fc")
        A2_gt, e2_gt = rgm_build_graphs.build_graphs(sample["points_ref"],
                                                     inlier_ref,
                                                     n2_gt,
                                                     stg="fc")

        src_o3 = open3d.geometry.PointCloud()
        ref_o3 = open3d.geometry.PointCloud()
        sample["points_src"] = np.concatenate(
            (sample["points_src"], np.zeros_like(sample["points_src"])),
            axis=1)
        sample["points_ref"] = np.concatenate(
            (sample["points_ref"], np.zeros_like(sample["points_ref"])),
            axis=1)
        src_o3.points = open3d.utility.Vector3dVector(
            sample["points_src"][:, :3])
        ref_o3.points = open3d.utility.Vector3dVector(
            sample["points_ref"][:, :3])
        src_o3.estimate_normals(
            search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                                 max_nn=30))
        ref_o3.estimate_normals(
            search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                                 max_nn=30))
        sample["points_src"][:, 3:6] = src_o3.normals
        sample["points_ref"][:, 3:6] = ref_o3.normals

        sample = {
            "points_src": sample["points_src"],
            "points_ref": sample["points_ref"],
            "n1_gt": n1_gt,
            "n2_gt": n2_gt,
            "e1_gt": e1_gt,
            "e2_gt": e2_gt,
            "gt_perm_mat": perm_mat.astype("float32"),
            "A1_gt": A1_gt,
            "A2_gt": A2_gt,
            "T_ab": T_ab.astype("float32"),
            "T_ba": T_ba.astype("float32"),
            "src_inlier": inlier_src,
            "ref_inlier": inlier_ref,
            "transform_gt": T_ab,
            "label": torch.tensor(sample["label"]),
        }

        return sample


def fetch_transform(params):

    if params.transform_type == "modelnet_os_rpmnet_noise":
        train_transforms = [
            SplitSourceRef(mode="hdf"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag,
                                     trans_mag=params.trans_mag),
            Resampler(params.num_points),
            RandomJitter(),
            ShufflePoints()
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="hdf"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag,
                                     trans_mag=params.trans_mag),
            Resampler(params.num_points),
            RandomJitter(),
            ShufflePoints()
        ]

    elif params.transform_type == "modelnet_os_rpmnet_clean":
        train_transforms = [
            SplitSourceRef(mode="hdf"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag,
                                     trans_mag=params.trans_mag),
            Resampler(params.num_points),
            # RandomJitter(),
            ShufflePoints()
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="hdf"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag,
                                     trans_mag=params.trans_mag),
            Resampler(params.num_points),
            # RandomJitter(),
            ShufflePoints()
        ]

    elif params.transform_type == "modelnet_ts_rpmnet_noise":
        train_transforms = [
            SplitSourceRef(mode="donothing"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag,
                                     trans_mag=params.trans_mag),
            Resampler(params.num_points),
            RandomJitter(noise_std=params.noise_std),
            ShufflePoints()
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="donothing"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag,
                                     trans_mag=params.trans_mag),
            Resampler(params.num_points),
            RandomJitter(noise_std=params.noise_std),
            ShufflePoints()
        ]

    elif params.transform_type == "modelnet_ts_rpmnet_clean":
        train_transforms = [
            SplitSourceRef(mode="donothing"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag,
                                     trans_mag=params.trans_mag),
            Resampler(params.num_points),
            # RandomJitter(),
            ShufflePoints()
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="donothing"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag,
                                     trans_mag=params.trans_mag),
            Resampler(params.num_points),
            # RandomJitter(),
            ShufflePoints()
        ]

    elif params.transform_type == "modelnet_ts_prnet_noise":
        train_transforms = [
            SplitSourceRef(mode="donothing"),
            ShufflePoints(),
            PRNetTorch(num_points=params.num_points,
                       rot_mag=params.rot_mag,
                       trans_mag=params.trans_mag,
                       noise_std=params.noise_std,
                       add_noise=True)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="donothing"),
            ShufflePoints(),
            PRNetTorch(num_points=params.num_points,
                       rot_mag=params.rot_mag,
                       trans_mag=params.trans_mag,
                       noise_std=params.noise_std,
                       add_noise=True)
        ]

    elif params.transform_type == "modelnet_ts_prnet_clean":
        train_transforms = [
            SplitSourceRef(mode="donothing"),
            ShufflePoints(),
            PRNetTorch(num_points=params.num_points,
                       rot_mag=params.rot_mag,
                       trans_mag=params.trans_mag,
                       add_noise=False)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="donothing"),
            ShufflePoints(),
            PRNetTorch(num_points=params.num_points,
                       rot_mag=params.rot_mag,
                       trans_mag=params.trans_mag,
                       add_noise=False)
        ]

    elif params.transform_type == "modelnet_os_prnet_noise":
        train_transforms = [
            SplitSourceRef(mode="hdf"),
            ShufflePoints(),
            PRNetTorch(num_points=params.num_points,
                       rot_mag=params.rot_mag,
                       trans_mag=params.trans_mag,
                       add_noise=True)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="hdf"),
            ShufflePoints(),
            PRNetTorch(num_points=params.num_points,
                       rot_mag=params.rot_mag,
                       trans_mag=params.trans_mag,
                       add_noise=True)
        ]

    elif params.transform_type == "modelnet_os_prnet_clean":
        train_transforms = [
            SplitSourceRef(mode="hdf"),
            ShufflePoints(),
            PRNetTorch(num_points=params.num_points,
                       rot_mag=params.rot_mag,
                       trans_mag=params.trans_mag,
                       add_noise=False)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="hdf"),
            ShufflePoints(),
            PRNetTorch(num_points=params.num_points,
                       rot_mag=params.rot_mag,
                       trans_mag=params.trans_mag,
                       add_noise=False)
        ]

    elif params.transform_type == "modelnet_ts_prnet_rotation_noise":
        train_transforms = [
            SplitSourceRef(mode="donothing"),
            ShufflePoints(),
            PRNetRotationTorch(num_points=params.num_points,
                               rot_mag=params.rot_mag,
                               trans_mag=params.trans_mag,
                               noise_std=params.noise_std,
                               add_noise=True)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="donothing"),
            ShufflePoints(),
            PRNetRotationTorch(num_points=params.num_points,
                               rot_mag=params.rot_mag,
                               trans_mag=params.trans_mag,
                               noise_std=params.noise_std,
                               add_noise=True)
        ]

    elif params.transform_type == "modelnet_ts_prnet_usip_noise":
        train_transforms = [
            SplitSourceRef(mode="donothing"),
            ShufflePoints(),
            PRNetUSIP(num_points=params.num_points,
                      rot_mag=params.rot_mag,
                      trans_mag=params.trans_mag,
                      noise_std=params.noise_std,
                      add_noise=True)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="donothing"),
            ShufflePoints(),
            PRNetUSIP(num_points=params.num_points,
                      rot_mag=params.rot_mag,
                      trans_mag=params.trans_mag,
                      noise_std=params.noise_std,
                      add_noise=True)
        ]

    elif params.transform_type == "modelnet_os_prnet_usip_clean":
        train_transforms = [
            SplitSourceRef(mode="hdf"),
            ShufflePoints(),
            PRNetUSIP(num_points=params.num_points,
                      rot_mag=params.rot_mag,
                      trans_mag=params.trans_mag,
                      add_noise=False)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="hdf"),
            ShufflePoints(),
            PRNetUSIP(num_points=params.num_points,
                      rot_mag=params.rot_mag,
                      trans_mag=params.trans_mag,
                      add_noise=False)
        ]

    elif params.transform_type == "modelnet_os_prnet_clean_onlyz":
        train_transforms = [
            SplitSourceRef(mode="hdf"),
            ShufflePoints(),
            PRNetTorch(num_points=params.num_points,
                       rot_mag=params.rot_mag,
                       trans_mag=params.trans_mag,
                       add_noise=False,
                       only_z=True,
                       partial=False)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="hdf"),
            ShufflePoints(),
            PRNetTorch(num_points=params.num_points,
                       rot_mag=params.rot_mag,
                       trans_mag=params.trans_mag,
                       add_noise=False,
                       only_z=True,
                       partial=False)
        ]

    elif params.transform_type == "modelnet_ts_prnet_clean_onlyz":
        train_transforms = [
            SplitSourceRef(mode="donothing"),
            ShufflePoints(),
            PRNetTorch(num_points=params.num_points,
                       rot_mag=params.rot_mag,
                       trans_mag=params.trans_mag,
                       add_noise=False,
                       only_z=True,
                       partial=False)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="donothing"),
            ShufflePoints(),
            PRNetTorch(num_points=params.num_points,
                       rot_mag=params.rot_mag,
                       trans_mag=params.trans_mag,
                       add_noise=False,
                       only_z=True,
                       partial=False)
        ]

    elif params.transform_type == "modelnet_ts_prnet_noise_overlap_ratio":
        train_transforms = [
            SplitSourceRef(mode="donothing"),
            PRNetTorchOverlapRatio(num_points=params.num_points,
                                   rot_mag=params.rot_mag,
                                   trans_mag=params.trans_mag,
                                   add_noise=True,
                                   overlap_ratio=params.overlap_ratio),
            ShufflePoints()
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="donothing"),
            PRNetTorchOverlapRatio(num_points=params.num_points,
                                   rot_mag=params.rot_mag,
                                   trans_mag=params.trans_mag,
                                   add_noise=True,
                                   overlap_ratio=params.overlap_ratio),
            ShufflePoints()
        ]

    elif params.transform_type == "modelnet_ts_timming":
        train_transforms = [
            SplitSourceRef(mode="donothing"),
            ShufflePoints(),
            Timming(num_points=params.num_points)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="donothing"),
            ShufflePoints(),
            Timming(num_points=params.num_points)
        ]

    elif params.transform_type == "stanford_prnet_clean":
        train_transforms = [
            SplitSourceRef(mode="resample"),
            ShufflePoints(),
            StanfordPRNetTorch(num_points=params.num_points,
                               rot_mag=params.rot_mag,
                               trans_mag=params.trans_mag,
                               add_noise=False)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="resample"),
            ShufflePoints(),
            StanfordPRNetTorch(num_points=params.num_points,
                               rot_mag=params.rot_mag,
                               trans_mag=params.trans_mag,
                               add_noise=False)
        ]

    elif params.transform_type == "stanford_rpmnet_clean":
        train_transforms = [
            SplitSourceRef(mode="resample"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag,
                                     trans_mag=params.trans_mag),
            Resampler(params.num_points),
            # RandomJitter(),
            ShufflePoints()
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="resample"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag,
                                     trans_mag=params.trans_mag),
            Resampler(params.num_points),
            # RandomJitter(),
            ShufflePoints()
        ]

    elif params.transform_type == "stanford_rpmnet_noise":
        train_transforms = [
            SplitSourceRef(mode="resample"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag,
                                     trans_mag=params.trans_mag),
            Resampler(params.num_points),
            RandomJitter(noise_std=params.noise_std),
            ShufflePoints()
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="resample"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag,
                                     trans_mag=params.trans_mag),
            Resampler(params.num_points),
            RandomJitter(noise_std=params.noise_std),
            ShufflePoints()
        ]

    elif params.transform_type == "7scenes_real":
        train_transforms = [
            SplitSourceRef(mode="7scenes"),
            ShufflePoints(),
            Scenes7Real(rot_mag=params.rot_mag,
                        trans_mag=params.trans_mag,
                        transform=True,
                        normalization=True)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="7scenes"),
            ShufflePoints(),
            Scenes7Real(rot_mag=params.rot_mag,
                        trans_mag=params.trans_mag,
                        transform=False,
                        normalization=True)
        ]

    elif params.transform_type == "7scenes_traditional_methods":
        train_transforms = [
            SplitSourceRef(mode="7scenes"),
            ShufflePoints(),
            Scenes7Real(rot_mag=params.rot_mag,
                        trans_mag=params.trans_mag,
                        transform=True,
                        normalization=False)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="7scenes"),
            ShufflePoints(),
            Scenes7Real(rot_mag=params.rot_mag,
                        trans_mag=params.trans_mag,
                        transform=False,
                        normalization=False)
        ]

    elif params.transform_type == "7scenes_sync_prnet":
        train_transforms = [
            SplitSourceRef(mode="resample"),
            ShufflePoints(),
            Scenes7Sync(num_points=params.num_points,
                        rot_mag=params.rot_mag,
                        trans_mag=params.trans_mag,
                        partial_ratio=params.partial_ratio,
                        add_noise=True,
                        normalization=True)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="resample"),
            ShufflePoints(),
            Scenes7Sync(num_points=params.num_points,
                        rot_mag=params.rot_mag,
                        trans_mag=params.trans_mag,
                        partial_ratio=params.partial_ratio,
                        add_noise=True,
                        normalization=True)
        ]

    elif params.transform_type == "kitti_real":
        train_transforms = [
            KITTIReal(num_points=params.num_points,
                      rot_mag=params.rot_mag,
                      trans_mag=params.trans_mag,
                      add_noise=True,
                      transform=True,
                      normalization=True)
        ]

        test_transforms = [
            SetDeterministic(),
            KITTIReal(num_points=params.num_points,
                      rot_mag=params.rot_mag,
                      trans_mag=params.trans_mag,
                      add_noise=True,
                      transform=True,
                      normalization=True)
        ]

    elif params.transform_type == "modelnet_ts_rpmnet_scale_noise":
        train_transforms = [
            SplitSourceRef(mode="donothing"),
            RandomCrop(params.partial_ratio),
            RandomScale(),
            RandomTransformSE3_euler(rot_mag=params.rot_mag,
                                     trans_mag=params.trans_mag),
            Resampler(params.num_points),
            RandomJitter(noise_std=params.noise_std),
            ShufflePoints()
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="donothing"),
            RandomCrop(params.partial_ratio),
            RandomScale(),
            RandomTransformSE3_euler(rot_mag=params.rot_mag,
                                     trans_mag=params.trans_mag),
            Resampler(params.num_points),
            RandomJitter(noise_std=params.noise_std),
            ShufflePoints()
        ]

    else:
        raise NotImplementedError

    # add post process
    if params.net_type == "deepgmr":
        train_transforms.append(DeepGMR())
        test_transforms.append(DeepGMR())

    if params.net_type == "rpmnet" or params.net_type == "omrpmnet":
        train_transforms.append(ComputeNormals())
        test_transforms.append(ComputeNormals())

    if params.net_type == "rgm" or params.net_type == "omrgm":
        train_transforms.append(RGM())
        test_transforms.append(RGM())

    _logger.info("Train transforms: {}".format(", ".join(
        [type(t).__name__ for t in train_transforms])))
    _logger.info("Val and Test transforms: {}".format(", ".join(
        [type(t).__name__ for t in test_transforms])))
    train_transforms = torchvision.transforms.Compose(train_transforms)
    test_transforms = torchvision.transforms.Compose(test_transforms)
    return train_transforms, test_transforms


if __name__ == "__main__":
    print("hello world")
