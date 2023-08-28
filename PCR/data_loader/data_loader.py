import random
import logging
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from data_loader.threedmatch import ThreeDMatch
from data_loader.flyingshapes import FlyingShapes
from data_loader.structuredthreed import StructuredThreeD
from data_loader.mixture import Mixture
from data_loader.sira import SIRA
from data_loader.eth import ETH
from data_loader.collate_func import calibrate_neighbors_stack_mode, registration_collate_fn_stack_mode

_logger = logging.getLogger(__name__)


def reset_seed_worker_init_fn(worker_id):
    r"""Reset seed for data loader worker."""
    seed = torch.initial_seed() % (2**32)
    # print(worker_id, seed)
    np.random.seed(seed)
    random.seed(seed)


def fetch_dataloader(cfg):
    _logger.info("Dataset type: {}".format(cfg.dataset_name))
    if cfg.dataset_name == "3dmatch":
        if cfg.model_name == "geotransformer":
            train_ds = ThreeDMatch(
                dataset_root=cfg.data.dataset_root,
                subset=cfg.train.subset,
                point_limit=cfg.train.point_limit,
                use_augmentation=cfg.train.use_augmentation,
                augmentation_noise=cfg.train.augmentation_noise,
                augmentation_rotation=cfg.train.augmentation_rotation,
            )
            # neighbor_limits = calibrate_neighbors_stack_mode(
            #     train_ds,
            #     registration_collate_fn_stack_mode,
            #     cfg.backbone.num_stages,
            #     cfg.backbone.init_voxel_size,
            #     cfg.backbone.init_radius,
            # )
            neighbor_limits = [100, 30, 30, 30]
            train_s = DistributedSampler(train_ds, shuffle=True)
            train_dl = DataLoader(
                train_ds,
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.num_workers,
                sampler=train_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                worker_init_fn=reset_seed_worker_init_fn,
                pin_memory=False,
                drop_last=False,
            )

            valid_ds = ThreeDMatch(
                dataset_root=cfg.data.dataset_root,
                subset=cfg.val.subset,
                point_limit=cfg.test.point_limit,
                use_augmentation=False,
            )
            valid_s = DistributedSampler(valid_ds, shuffle=False)
            valid_dl = DataLoader(
                valid_ds,
                batch_size=cfg.test.batch_size,
                num_workers=cfg.test.num_workers,
                sampler=valid_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                pin_memory=False,
                drop_last=False,
            )

            test_ds = ThreeDMatch(
                dataset_root=cfg.data.dataset_root,
                subset=cfg.test.subset,
                point_limit=cfg.test.point_limit,
                use_augmentation=False,
            )
            test_s = DistributedSampler(test_ds, shuffle=False)
            test_dl = DataLoader(
                test_ds,
                batch_size=cfg.test.batch_size,
                num_workers=cfg.test.num_workers,
                sampler=test_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                pin_memory=False,
                drop_last=False,
            )
            dl = {"train": train_dl, "val": valid_dl, "test": test_dl}

        else:
            raise NotImplementedError(
                "Dataset For Specific Model Not Implemented")

    elif cfg.dataset_name == "flyingshapes":
        if cfg.model_name == "geotransformer":
            cfg.backbone.init_radius = cfg.backbone.init_voxel_size * cfg.backbone.base_radius
            cfg.backbone.init_sigma = cfg.backbone.init_voxel_size * cfg.backbone.base_sigma
            train_ds = FlyingShapes(
                dataset_root=cfg.data.dataset_root,
                gt_log=cfg.data.gt_log,
                subset=cfg.train.subset,
                voxel_size=cfg.train.voxel_size,
                sphere_limit=cfg.train.sphere_limit,
                point_limit=cfg.train.point_limit,
                use_augmentation=cfg.train.use_augmentation,
                augmentation_noise=cfg.train.augmentation_noise,
                augmentation_rotation=cfg.train.augmentation_rotation,
                augmentation_translation=cfg.train.augmentation_translation,
                scene_num=cfg.data.scene_num,
                train_scene_num=cfg.data.train_scene_num,
                test_scene_num=cfg.data.test_scene_num,
                train_scene_indices=cfg.data.train_scene_indices,
                test_scene_indices=cfg.data.test_scene_indices,
                augmentation_rotation_type=cfg.data.augmentation_rotation_type,
                augmentation_translation_type=cfg.data.
                augmentation_translation_type,
                augmentation_noise_type=cfg.data.augmentation_noise_type,
            )
            # neighbor_limits = calibrate_neighbors_stack_mode(
            #     train_ds,
            #     registration_collate_fn_stack_mode,
            #     cfg.backbone.num_stages,
            #     cfg.backbone.init_voxel_size,
            #     cfg.backbone.init_radius,
            # )
            neighbor_limits = [100, 30, 30, 30]
            train_s = DistributedSampler(train_ds, shuffle=True)
            train_dl = DataLoader(
                train_ds,
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.num_workers,
                sampler=train_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                worker_init_fn=reset_seed_worker_init_fn,
                pin_memory=False,
                drop_last=False,
            )

            valid_ds = FlyingShapes(
                dataset_root=cfg.data.dataset_root,
                gt_log=cfg.data.gt_log,
                subset=cfg.val.subset,
                voxel_size=cfg.val.voxel_size,
                sphere_limit=cfg.val.sphere_limit,
                point_limit=cfg.val.point_limit,
                use_augmentation=cfg.val.use_augmentation,
                augmentation_noise=cfg.val.augmentation_noise,
                augmentation_rotation=cfg.val.augmentation_rotation,
                augmentation_translation=cfg.val.augmentation_translation,
                scene_num=cfg.data.scene_num,
                train_scene_num=cfg.data.train_scene_num,
                test_scene_num=cfg.data.test_scene_num,
                train_scene_indices=cfg.data.train_scene_indices,
                test_scene_indices=cfg.data.test_scene_indices,
                augmentation_rotation_type=cfg.data.augmentation_rotation_type,
                augmentation_translation_type=cfg.data.
                augmentation_translation_type,
                augmentation_noise_type=cfg.data.augmentation_noise_type,
            )
            valid_s = DistributedSampler(valid_ds, shuffle=False)
            valid_dl = DataLoader(
                valid_ds,
                batch_size=cfg.val.batch_size,
                num_workers=cfg.val.num_workers,
                sampler=valid_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                pin_memory=False,
                drop_last=False,
            )

            test_ds = FlyingShapes(
                dataset_root=cfg.data.dataset_root,
                gt_log=cfg.data.gt_log,
                subset=cfg.test.subset,
                voxel_size=cfg.test.voxel_size,
                sphere_limit=cfg.test.sphere_limit,
                point_limit=cfg.test.point_limit,
                use_augmentation=cfg.test.use_augmentation,
                augmentation_noise=cfg.test.augmentation_noise,
                augmentation_rotation=cfg.test.augmentation_rotation,
                augmentation_translation=cfg.test.augmentation_translation,
                scene_num=cfg.data.scene_num,
                train_scene_num=cfg.data.train_scene_num,
                test_scene_num=cfg.data.test_scene_num,
                train_scene_indices=cfg.data.train_scene_indices,
                test_scene_indices=cfg.data.test_scene_indices,
                augmentation_rotation_type=cfg.data.augmentation_rotation_type,
                augmentation_translation_type=cfg.data.
                augmentation_translation_type,
                augmentation_noise_type=cfg.data.augmentation_noise_type,
            )
            test_s = DistributedSampler(test_ds, shuffle=False)
            test_dl = DataLoader(
                test_ds,
                batch_size=cfg.test.batch_size,
                num_workers=cfg.test.num_workers,
                sampler=test_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                pin_memory=False,
                drop_last=False,
            )
            dl = {"train": train_dl, "val": valid_dl, "test": test_dl}

        else:
            raise NotImplementedError(
                "Dataset For Specific Model Not Implemented")

    elif cfg.dataset_name == "structured3d":
        if cfg.model_name == "geotransformer":
            cfg.backbone.init_radius = cfg.backbone.init_voxel_size * cfg.backbone.base_radius
            cfg.backbone.init_sigma = cfg.backbone.init_voxel_size * cfg.backbone.base_sigma
            train_ds = StructuredThreeD(
                dataset_root=cfg.data.dataset_root,
                gt_log=cfg.data.gt_log,
                subset=cfg.train.subset,
                voxel_size=cfg.train.voxel_size,
                sphere_limit=cfg.train.sphere_limit,
                point_limit=cfg.train.point_limit,
                use_augmentation=cfg.train.use_augmentation,
                augmentation_noise=cfg.train.augmentation_noise,
                augmentation_rotation=cfg.train.augmentation_rotation,
                augmentation_translation=cfg.train.augmentation_translation,
                scene_num=cfg.data.scene_num,
                train_scene_num=cfg.data.train_scene_num,
                test_scene_num=cfg.data.test_scene_num,
                train_scene_indices=cfg.data.train_scene_indices,
                test_scene_indices=cfg.data.test_scene_indices,
                augmentation_rotation_type=cfg.data.augmentation_rotation_type,
                augmentation_translation_type=cfg.data.
                augmentation_translation_type,
                augmentation_noise_type=cfg.data.augmentation_noise_type,
            )
            # neighbor_limits = calibrate_neighbors_stack_mode(
            #     train_ds,
            #     registration_collate_fn_stack_mode,
            #     cfg.backbone.num_stages,
            #     cfg.backbone.init_voxel_size,
            #     cfg.backbone.init_radius,
            # )
            neighbor_limits = [100, 30, 30, 30]
            train_s = DistributedSampler(train_ds, shuffle=True)
            train_dl = DataLoader(
                train_ds,
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.num_workers,
                sampler=train_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                worker_init_fn=reset_seed_worker_init_fn,
                pin_memory=False,
                drop_last=False,
            )

            # fix neighbor_limits
            # neighbor_limits = [30 for i in range(cfg.backbone.num_stages)]
            valid_ds = StructuredThreeD(
                dataset_root=cfg.data.dataset_root,
                gt_log=cfg.data.gt_log,
                subset=cfg.val.subset,
                voxel_size=cfg.val.voxel_size,
                sphere_limit=cfg.val.sphere_limit,
                point_limit=cfg.val.point_limit,
                use_augmentation=cfg.val.use_augmentation,
                augmentation_noise=cfg.val.augmentation_noise,
                augmentation_rotation=cfg.val.augmentation_rotation,
                augmentation_translation=cfg.val.augmentation_translation,
                scene_num=cfg.data.scene_num,
                train_scene_num=cfg.data.train_scene_num,
                test_scene_num=cfg.data.test_scene_num,
                train_scene_indices=cfg.data.train_scene_indices,
                test_scene_indices=cfg.data.test_scene_indices,
                augmentation_rotation_type=cfg.data.augmentation_rotation_type,
                augmentation_translation_type=cfg.data.
                augmentation_translation_type,
                augmentation_noise_type=cfg.data.augmentation_noise_type,
            )
            valid_s = DistributedSampler(valid_ds, shuffle=False)
            valid_dl = DataLoader(
                valid_ds,
                batch_size=cfg.val.batch_size,
                num_workers=cfg.val.num_workers,
                sampler=valid_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                pin_memory=False,
                drop_last=False,
            )

            test_ds = StructuredThreeD(
                dataset_root=cfg.data.dataset_root,
                gt_log=cfg.data.gt_log,
                subset=cfg.test.subset,
                voxel_size=cfg.test.voxel_size,
                sphere_limit=cfg.test.sphere_limit,
                point_limit=cfg.test.point_limit,
                use_augmentation=cfg.test.use_augmentation,
                augmentation_noise=cfg.test.augmentation_noise,
                augmentation_rotation=cfg.test.augmentation_rotation,
                augmentation_translation=cfg.test.augmentation_translation,
                scene_num=cfg.data.scene_num,
                train_scene_num=cfg.data.train_scene_num,
                test_scene_num=cfg.data.test_scene_num,
                train_scene_indices=cfg.data.train_scene_indices,
                test_scene_indices=cfg.data.test_scene_indices,
                augmentation_rotation_type=cfg.data.augmentation_rotation_type,
                augmentation_translation_type=cfg.data.
                augmentation_translation_type,
                augmentation_noise_type=cfg.data.augmentation_noise_type,
            )
            test_s = DistributedSampler(test_ds, shuffle=False)
            test_dl = DataLoader(
                test_ds,
                batch_size=cfg.test.batch_size,
                num_workers=cfg.test.num_workers,
                sampler=test_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                pin_memory=False,
                drop_last=False,
            )
            dl = {"train": train_dl, "val": valid_dl, "test": test_dl}

        else:
            raise NotImplementedError(
                "Dataset For Specific Model Not Implemented")

    elif cfg.dataset_name == "mixture":
        if cfg.model_name == "geotransformer":
            cfg.backbone.init_radius = cfg.backbone.init_voxel_size * cfg.backbone.base_radius
            cfg.backbone.init_sigma = cfg.backbone.init_voxel_size * cfg.backbone.base_sigma
            train_ds = Mixture(
                dataset_root=cfg.data.dataset_root,
                gt_log=cfg.data.gt_log,
                subset=cfg.train.subset,
                voxel_size=cfg.train.voxel_size,
                sphere_limit=cfg.train.sphere_limit,
                point_limit=cfg.train.point_limit,
                use_augmentation=cfg.train.use_augmentation,
                augmentation_noise=cfg.train.augmentation_noise,
                augmentation_rotation=cfg.train.augmentation_rotation,
                augmentation_translation=cfg.train.augmentation_translation,
                scene_num=cfg.data.scene_num,
                train_scene_num=cfg.data.train_scene_num,
                test_scene_num=cfg.data.test_scene_num,
                train_scene_indices=cfg.data.train_scene_indices,
                test_scene_indices=cfg.data.test_scene_indices,
                augmentation_rotation_type=cfg.data.augmentation_rotation_type,
                augmentation_translation_type=cfg.data.
                augmentation_translation_type,
                augmentation_noise_type=cfg.data.augmentation_noise_type,
            )
            # neighbor_limits = calibrate_neighbors_stack_mode(
            #     train_ds,
            #     registration_collate_fn_stack_mode,
            #     cfg.backbone.num_stages,
            #     cfg.backbone.init_voxel_size,
            #     cfg.backbone.init_radius,
            # )
            neighbor_limits = [100, 30, 30, 30]
            train_s = DistributedSampler(train_ds, shuffle=True)
            train_dl = DataLoader(
                train_ds,
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.num_workers,
                sampler=train_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                worker_init_fn=reset_seed_worker_init_fn,
                pin_memory=False,
                drop_last=False,
            )

            # fix neighbor_limits
            # neighbor_limits = [30 for i in range(cfg.backbone.num_stages)]
            valid_ds = Mixture(
                dataset_root=cfg.data.dataset_root,
                gt_log=cfg.data.gt_log,
                subset=cfg.val.subset,
                voxel_size=cfg.val.voxel_size,
                sphere_limit=cfg.val.sphere_limit,
                point_limit=cfg.val.point_limit,
                use_augmentation=cfg.val.use_augmentation,
                augmentation_noise=cfg.val.augmentation_noise,
                augmentation_rotation=cfg.val.augmentation_rotation,
                augmentation_translation=cfg.val.augmentation_translation,
                scene_num=cfg.data.scene_num,
                train_scene_num=cfg.data.train_scene_num,
                test_scene_num=cfg.data.test_scene_num,
                train_scene_indices=cfg.data.train_scene_indices,
                test_scene_indices=cfg.data.test_scene_indices,
                augmentation_rotation_type=cfg.data.augmentation_rotation_type,
                augmentation_translation_type=cfg.data.
                augmentation_translation_type,
                augmentation_noise_type=cfg.data.augmentation_noise_type,
            )
            valid_s = DistributedSampler(valid_ds, shuffle=False)
            valid_dl = DataLoader(
                valid_ds,
                batch_size=cfg.val.batch_size,
                num_workers=cfg.val.num_workers,
                sampler=valid_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                pin_memory=False,
                drop_last=False,
            )

            test_ds = Mixture(
                dataset_root=cfg.data.dataset_root,
                gt_log=cfg.data.gt_log,
                subset=cfg.test.subset,
                voxel_size=cfg.test.voxel_size,
                sphere_limit=cfg.test.sphere_limit,
                point_limit=cfg.test.point_limit,
                use_augmentation=cfg.test.use_augmentation,
                augmentation_noise=cfg.test.augmentation_noise,
                augmentation_rotation=cfg.test.augmentation_rotation,
                augmentation_translation=cfg.test.augmentation_translation,
                scene_num=cfg.data.scene_num,
                train_scene_num=cfg.data.train_scene_num,
                test_scene_num=cfg.data.test_scene_num,
                train_scene_indices=cfg.data.train_scene_indices,
                test_scene_indices=cfg.data.test_scene_indices,
                augmentation_rotation_type=cfg.data.augmentation_rotation_type,
                augmentation_translation_type=cfg.data.
                augmentation_translation_type,
                augmentation_noise_type=cfg.data.augmentation_noise_type,
            )
            test_s = DistributedSampler(test_ds, shuffle=False)
            test_dl = DataLoader(
                test_ds,
                batch_size=cfg.test.batch_size,
                num_workers=cfg.test.num_workers,
                sampler=test_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                pin_memory=False,
                drop_last=False,
            )
            dl = {"train": train_dl, "val": valid_dl, "test": test_dl}

        else:
            raise NotImplementedError(
                "Dataset For Specific Model Not Implemented")

    elif cfg.dataset_name == "sira":
        if cfg.model_name == "geotransformer":
            cfg.backbone.init_radius = cfg.backbone.init_voxel_size * cfg.backbone.base_radius
            cfg.backbone.init_sigma = cfg.backbone.init_voxel_size * cfg.backbone.base_sigma
            train_ds = SIRA(
                dataset_root=cfg.data.dataset_root,
                gt_log=cfg.data.gt_log,
                subset=cfg.train.subset,
                voxel_size=cfg.train.voxel_size,
                sphere_limit=cfg.train.sphere_limit,
                point_limit=cfg.train.point_limit,
                use_augmentation=cfg.train.use_augmentation,
                augmentation_noise=cfg.train.augmentation_noise,
                augmentation_rotation=cfg.train.augmentation_rotation,
                augmentation_translation=cfg.train.augmentation_translation,
                scene_num=cfg.data.scene_num,
                train_scene_num=cfg.data.train_scene_num,
                test_scene_num=cfg.data.test_scene_num,
                train_scene_indices=cfg.data.train_scene_indices,
                test_scene_indices=cfg.data.test_scene_indices,
                augmentation_rotation_type=cfg.data.augmentation_rotation_type,
                augmentation_translation_type=cfg.data.
                augmentation_translation_type,
                augmentation_noise_type=cfg.data.augmentation_noise_type,
            )
            # neighbor_limits = calibrate_neighbors_stack_mode(
            #     train_ds,
            #     registration_collate_fn_stack_mode,
            #     cfg.backbone.num_stages,
            #     cfg.backbone.init_voxel_size,
            #     cfg.backbone.init_radius,
            # )
            neighbor_limits = [100, 30, 30, 30]
            train_s = DistributedSampler(train_ds, shuffle=True)
            train_dl = DataLoader(
                train_ds,
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.num_workers,
                sampler=train_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                worker_init_fn=reset_seed_worker_init_fn,
                pin_memory=False,
                drop_last=False,
            )

            valid_ds = SIRA(
                dataset_root=cfg.data.dataset_root,
                gt_log=cfg.data.gt_log,
                subset=cfg.val.subset,
                voxel_size=cfg.val.voxel_size,
                sphere_limit=cfg.val.sphere_limit,
                point_limit=cfg.val.point_limit,
                use_augmentation=cfg.val.use_augmentation,
                augmentation_noise=cfg.val.augmentation_noise,
                augmentation_rotation=cfg.val.augmentation_rotation,
                augmentation_translation=cfg.val.augmentation_translation,
                scene_num=cfg.data.scene_num,
                train_scene_num=cfg.data.train_scene_num,
                test_scene_num=cfg.data.test_scene_num,
                train_scene_indices=cfg.data.train_scene_indices,
                test_scene_indices=cfg.data.test_scene_indices,
                augmentation_rotation_type=cfg.data.augmentation_rotation_type,
                augmentation_translation_type=cfg.data.
                augmentation_translation_type,
                augmentation_noise_type=cfg.data.augmentation_noise_type,
            )
            valid_s = DistributedSampler(valid_ds, shuffle=False)
            valid_dl = DataLoader(
                valid_ds,
                batch_size=cfg.val.batch_size,
                num_workers=cfg.val.num_workers,
                sampler=valid_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                pin_memory=False,
                drop_last=False,
            )

            test_ds = SIRA(
                dataset_root=cfg.data.dataset_root,
                gt_log=cfg.data.gt_log,
                subset=cfg.test.subset,
                voxel_size=cfg.test.voxel_size,
                sphere_limit=cfg.test.sphere_limit,
                point_limit=cfg.test.point_limit,
                use_augmentation=cfg.test.use_augmentation,
                augmentation_noise=cfg.test.augmentation_noise,
                augmentation_rotation=cfg.test.augmentation_rotation,
                augmentation_translation=cfg.test.augmentation_translation,
                scene_num=cfg.data.scene_num,
                train_scene_num=cfg.data.train_scene_num,
                test_scene_num=cfg.data.test_scene_num,
                train_scene_indices=cfg.data.train_scene_indices,
                test_scene_indices=cfg.data.test_scene_indices,
                augmentation_rotation_type=cfg.data.augmentation_rotation_type,
                augmentation_translation_type=cfg.data.
                augmentation_translation_type,
                augmentation_noise_type=cfg.data.augmentation_noise_type,
            )
            test_s = DistributedSampler(test_ds, shuffle=False)
            test_dl = DataLoader(
                test_ds,
                batch_size=cfg.test.batch_size,
                num_workers=cfg.test.num_workers,
                sampler=test_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                pin_memory=False,
                drop_last=False,
            )
            dl = {"train": train_dl, "val": valid_dl, "test": test_dl}

        else:
            raise NotImplementedError(
                "Dataset For Specific Model Not Implemented")

    elif cfg.dataset_name == "eth":
        if cfg.model_name == "geotransformer":
            neighbor_limits = [100, 30, 30, 30]
            test_ds = ETH(
                dataset_root=cfg.data.dataset_root,
                point_limit=cfg.test.point_limit,
                use_augmentation=False,
                matching_radius=cfg.eval.acceptance_radius,
            )
            test_s = DistributedSampler(test_ds, shuffle=False)
            test_dl = DataLoader(
                test_ds,
                batch_size=cfg.test.batch_size,
                num_workers=cfg.test.num_workers,
                sampler=test_s,
                collate_fn=partial(
                    registration_collate_fn_stack_mode,
                    num_stages=cfg.backbone.num_stages,
                    voxel_size=cfg.backbone.init_voxel_size,
                    search_radius=cfg.backbone.init_radius,
                    neighbor_limits=neighbor_limits,
                    precompute_data=True,
                ),
                pin_memory=False,
                drop_last=False,
            )
            dl = {"test": test_dl}
        else:
            raise NotImplementedError(
                "Dataset For Specific Model Not Implemented")

    else:
        raise NotImplementedError("Dataset Not Implemented")

    return dl
