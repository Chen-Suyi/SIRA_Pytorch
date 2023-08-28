import torch
import numpy as np
import importlib

ext_module = importlib.import_module("geotransformer.ext")


def voxel_down_sample(points: torch.Tensor, voxel_size: float):
    lengths = torch.LongTensor([points.shape[0]])
    points, lengths = ext_module.grid_subsampling(points, lengths, voxel_size)

    return points


def radius_search(querys: torch.Tensor,
                  supports: torch.Tensor,
                  neighbor_limit=30,
                  radius=0.0625) -> torch.LongTensor:

    q_lengths = torch.LongTensor([querys.shape[0]])
    s_lengths = torch.LongTensor([supports.shape[0]])
    neighbor_indices = ext_module.radius_neighbors(querys, supports, q_lengths,
                                                   s_lengths, radius)
    if neighbor_limit > 0:
        neighbor_indices = neighbor_indices[:, :neighbor_limit]
    return neighbor_indices


def precompute(points: torch.Tensor,
               voxel_size=0.025,
               neighbor_limits=[100, 30, 30, 30],
               radius=0.0625,
               num_stages=4):

    points_list = []
    for stage in range(num_stages):
        if stage != 0:
            points = voxel_down_sample(points, voxel_size=voxel_size)
        points_list.append(points)

        voxel_size *= 2

    neighbors_list = []
    subsamples_list = []
    upsamples_list = []
    for stage in range(num_stages):
        neighbors = radius_search(points_list[stage],
                                  points_list[stage],
                                  neighbor_limit=neighbor_limits[stage],
                                  radius=radius)
        neighbors_list.append(neighbors)

        if stage < num_stages - 1:
            subsamples = radius_search(points_list[stage + 1],
                                       points_list[stage],
                                       neighbor_limit=neighbor_limits[stage],
                                       radius=radius)
            subsamples_list.append(subsamples)

            upsamples = radius_search(points_list[stage],
                                      points_list[stage + 1],
                                      neighbor_limit=neighbor_limits[stage +
                                                                     1],
                                      radius=radius * 2)
            upsamples_list.append(upsamples)

        radius *= 2

    return {
        "points": points_list,
        "neighbors": neighbors_list,
        "subsamples": subsamples_list,
        "upsamples": upsamples_list
    }


def numpy2tensor_dict(data_dict, device):
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value).to(device)
        elif isinstance(value, list):
            for idx, v in enumerate(value):
                if isinstance(v, np.ndarray):
                    value[idx] = torch.from_numpy(v).to(device)
        data_dict[key] = value


def collate_func(data_dicts):
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                if key in ["synth", "real", "fake"]:
                    value = precompute(torch.from_numpy(value))
                else:
                    value = torch.Tensor(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    return collated_dict
