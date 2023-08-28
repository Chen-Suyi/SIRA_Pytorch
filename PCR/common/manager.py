import os
import os.path as osp
import pickle
from collections import defaultdict
import boto3
import numpy as np
import open3d as o3d
import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from common import tool


class Manager():

    def __init__(self, model, optimizer, scheduler, cfg, dataloader, logger):
        # Config status
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.logger = logger

        # Define client
        self.s3_client = boto3.client("s3",
                                      endpoint_url="http://oss.i.brainpp.cn")
        self.bucket_name = self.cfg.bucket_name

        # Init some recorders
        self.init_status()
        self.init_tb()

    def init_status(self):
        self.epoch = 0
        self.step = 0
        # Loss status
        self.loss_status = defaultdict(tool.AverageMeter)
        # Metric status: val, test
        self.metric_status = defaultdict(
            lambda: defaultdict(tool.AverageMeter))
        # Metric w.r.t scene: val, test
        self.metric_wrt_scene = defaultdict(
            lambda: defaultdict(lambda: defaultdict(tool.AverageMeter)))
        # Score status: val, test
        self.score_status = {}
        for split in ["val", "test"]:
            self.score_status[split] = {"cur": np.inf, "best": np.inf}

    def init_tb(self):
        # Tensorboard
        loss_tb_dir = osp.join(self.cfg.model_dir, "summary/loss")
        os.makedirs(loss_tb_dir, exist_ok=True)
        self.loss_writter = SummaryWriter(log_dir=loss_tb_dir)
        metric_tb_dir = osp.join(self.cfg.model_dir, "summary/metric")
        os.makedirs(metric_tb_dir, exist_ok=True)
        self.metric_writter = SummaryWriter(log_dir=metric_tb_dir)
        metric_scene_tb_dir = osp.join(self.cfg.model_dir,
                                       "summary/metric_wrt_scene")
        os.makedirs(metric_scene_tb_dir, exist_ok=True)
        self.metric_scene_writter = SummaryWriter(log_dir=metric_scene_tb_dir)

    def update_step(self):
        self.step += 1

    def update_epoch(self):
        self.epoch += 1

    def update_loss_status(self, loss, batch_size):
        for k, v in loss.items():
            self.loss_status[k].update(val=v.item(), num=batch_size)

    def update_metric_status(self, metric, split, batch_size):
        for k, v in metric.items():
            self.metric_status[split][k].update(val=v.item(), num=batch_size)
            self.score_status[split]["cur"] = self.metric_status[split][
                self.cfg.major_metric].avg

    def reset_loss_status(self):
        for k, v in self.loss_status.items():
            self.loss_status[k].reset()

    def reset_metric_status(self, split):
        for k, v in self.metric_status[split].items():
            self.metric_status[split][k].reset()

    def tqdm_info(self, split):
        if split == "train":
            exp_name = self.cfg.model_dir.split("/")[-1]
            print_str = "{} Epoch: {:4d}, lr={:.8f} ".format(
                exp_name, self.epoch,
                self.scheduler.get_last_lr()[0])
            print_str += "total loss: {:.4f}/{:.4f}".format(
                self.loss_status["total"].val, self.loss_status["total"].avg)
        else:
            print_str = ""
            for k, v in self.metric_status[split].items():
                print_str += "{}:{:.3f} --".format(k, v.avg)
        return print_str

    def print_metric(self, split, only_best=False, wrt_scene=False):
        is_best = self.score_status[split]["cur"] < self.score_status[split][
            "best"]
        color = "green" if split == "val" else "red"
        print_str = " | ".join("{}: {:4g}".format(k, v.avg)
                               for k, v in self.metric_status[split].items())
        if only_best:
            if is_best:
                self.logger.info(
                    colored("Best Epoch: {}, {} Results: {}".format(
                        self.epoch, split, print_str),
                            color,
                            attrs=["bold"]))
                if wrt_scene:
                    self.print_metric_wrt_scene(split)
        else:
            if is_best:
                self.logger.info(
                    colored("Best Epoch: {}, {} Results: {}".format(
                        self.epoch, split, print_str),
                            color,
                            attrs=["bold"]))
                if wrt_scene:
                    self.print_metric_wrt_scene(split)
            else:
                self.logger.info(
                    colored("Epoch: {}, {} Results: {}".format(
                        self.epoch, split, print_str),
                            "white",
                            attrs=["bold"]))
                if wrt_scene:
                    self.print_metric_wrt_scene(split)

    def write_loss_to_tb(self, split):
        if self.step % self.cfg.save_summary_steps:
            for k, v in self.loss_status.items():
                self.loss_writter.add_scalar("{}_loss/{}".format(split, k),
                                             v.val, self.step)

    def write_metric_to_tb(self, split):
        for k, v in self.metric_status[split].items():
            self.metric_writter.add_scalar("{}_metric/{}".format(split, k),
                                           v.avg, self.epoch)

    def save_ckpt(self):
        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "score_status": self.score_status
        }

        # Save latest metrics and best checkpoints
        for split in ["val", "test"]:
            if split not in self.dataloader:
                continue
            latest_metric_path = osp.join(
                self.cfg.model_dir, "{}_metric_latest.json".format(split))
            tool.save_dict_to_json(self.metric_status[split],
                                   latest_metric_path)
            is_best = self.score_status[split]["cur"] < self.score_status[
                split]["best"]
            if is_best:
                self.score_status[split]["best"] = self.score_status[split][
                    "cur"]
                # update score_status in state
                state["score_status"] = self.score_status
                best_metric_path = osp.join(
                    self.cfg.model_dir, "{}_metric_latest.json".format(split))
                tool.save_dict_to_json(self.metric_status[split],
                                       best_metric_path)
                self.logger.info("Current is {} best, score={:.7f}".format(
                    split, self.score_status[split]["best"]))
                # save checkpoint
                if self.epoch > self.cfg.save_best_after:
                    best_ckpt_path = osp.join(
                        self.cfg.model_dir, "{}_model_best.pth".format(split))
                    if self.cfg.save_mode == "local":
                        torch.save(state, best_ckpt_path)
                    elif self.cfg.save_mode == "oss":
                        save_dict = pickle.dumps(state)
                        resp = self.s3_client.put_object(
                            Bucket=self.bucket_name,
                            Key=best_ckpt_path,
                            Body=save_dict[0:])
                    else:
                        raise NotImplementedError
                    self.logger.info("Saved {} best checkpoint to: {}".format(
                        split, best_ckpt_path))

        # Save latest checkpoint
        if self.epoch % self.cfg.save_latest_freq == 0:
            latest_ckpt_path = osp.join(self.cfg.model_dir, "model_latest.pth")
            if self.cfg.save_mode == "local":
                torch.save(state, latest_ckpt_path)
            elif self.cfg.save_mode == "oss":
                save_dict = pickle.dumps(state)
                resp = self.s3_client.put_object(Bucket=self.bucket_name,
                                                 Key=latest_ckpt_path,
                                                 Body=save_dict[0:])
            else:
                raise NotImplementedError
            self.logger.info(
                "Saved latest checkpoint to: {}".format(latest_ckpt_path))

    def load_ckpt(self):
        if self.cfg.save_mode == "local":
            state = torch.load(self.cfg.resume)
        elif self.cfg.save_mode == "oss":
            resp = self.s3_client.get_object(Bucket=self.bucket_name,
                                             Key=self.cfg.resume)
            state = resp["Body"].read()
            state = pickle.loads(state)
        else:
            raise NotImplementedError

        if self.cfg.resume.endswith(".tar"):  # for loading 3dmatch pretrain
            state["state_dict"] = state["model"]
        ckpt_component = []
        if "state_dict" in state and self.model is not None:
            try:
                self.model.load_state_dict(state["state_dict"])
            except RuntimeError:
                self.logger.info("Using custom loading net")
                net_dict = self.model.state_dict()
                if "module" not in list(state["state_dict"].keys())[0]:
                    state_dict = {
                        "module." + k: v
                        for k, v in state["state_dict"].items()
                        if "module." + k in net_dict.keys()
                    }
                else:
                    state_dict = {
                        k: v
                        for k, v in state["state_dict"].items()
                        if k in net_dict.keys()
                    }
                net_dict.update(state_dict)
                self.model.load_state_dict(net_dict, strict=False)
            ckpt_component.append("net")

        if not self.cfg.only_weights:
            if "optimizer" in state and self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(state["optimizer"])
                except RuntimeError:
                    self.logger.info("Using custom loading optimizer")
                    optimizer_dict = self.optimizer.state_dict()
                    state_dict = {
                        k: v
                        for k, v in state["optimizer"].items()
                        if k in optimizer_dict.keys()
                    }
                    optimizer_dict.update(state_dict)
                    self.optimizer.load_state_dict(optimizer_dict)
                ckpt_component.append("opt")

            if "scheduler" in state and self.scheduler is not None:
                try:
                    self.scheduler.load_state_dict(state["scheduler"])
                except RuntimeError:
                    self.logger.info("Using custom loading scheduler")
                    scheduler_dict = self.scheduler.state_dict()
                    state_dict = {
                        k: v
                        for k, v in state["scheduler"].items()
                        if k in scheduler_dict.keys()
                    }
                    scheduler_dict.update(state_dict)
                    self.scheduler.load_state_dict(scheduler_dict)
                ckpt_component.append("sch")

            if "step" in state:
                self.step = state["step"] + 1
                ckpt_component.append("step")

            if "epoch" in state:
                self.epoch = state["epoch"] + 1
                ckpt_component.append("epoch")

            if "score_status" in state:
                self.score_status = state["score_status"]
                ckpt_component.append("score status: {}".format(
                    self.score_status))

        ckpt_component = ", ".join(i for i in ckpt_component)
        self.logger.info("Loaded models from: {}".format(self.cfg.resume))
        self.logger.info("Ckpt load: {}".format(ckpt_component))

    def after_test_step(self, split, data_dict, output_dict):
        scene_name = data_dict['scene_name']
        ref_id = data_dict['ref_frame']
        src_id = data_dict['src_frame']

        os.makedirs(osp.join(self.cfg.model_dir, "features", split,
                             scene_name),
                    exist_ok=True)
        file_name = osp.join(self.cfg.model_dir, "features", split, scene_name,
                             f'{ref_id}_{src_id}.npz')
        np.savez_compressed(
            file_name,
            ref_points=tool.release_cuda(output_dict['ref_points']),
            src_points=tool.release_cuda(output_dict['src_points']),
            ref_points_f=tool.release_cuda(output_dict['ref_points_f']),
            src_points_f=tool.release_cuda(output_dict['src_points_f']),
            ref_points_c=tool.release_cuda(output_dict['ref_points_c']),
            src_points_c=tool.release_cuda(output_dict['src_points_c']),
            ref_feats_c=tool.release_cuda(output_dict['ref_feats_c']),
            src_feats_c=tool.release_cuda(output_dict['src_feats_c']),
            ref_node_corr_indices=tool.release_cuda(
                output_dict['ref_node_corr_indices']),
            src_node_corr_indices=tool.release_cuda(
                output_dict['src_node_corr_indices']),
            ref_corr_points=tool.release_cuda(output_dict['ref_corr_points']),
            src_corr_points=tool.release_cuda(output_dict['src_corr_points']),
            corr_scores=tool.release_cuda(output_dict['corr_scores']),
            gt_node_corr_indices=tool.release_cuda(
                output_dict['gt_node_corr_indices']),
            gt_node_corr_overlaps=tool.release_cuda(
                output_dict['gt_node_corr_overlaps']),
            estimated_transform=tool.release_cuda(
                output_dict['estimated_transform']),
            transform=tool.release_cuda(data_dict['transform']),
            overlap=data_dict['overlap'],
        )

    def update_metric_wrt_scene(self, split, input, metric, batch_size):
        """
        metrics w.r.t. scenes
        """
        for k, v in metric.items():
            self.metric_wrt_scene[split][input["scene_name"]][k].update(
                val=v.item(), num=batch_size)

    def print_metric_wrt_scene(self, split):
        for scene_name, scene_metric in self.metric_wrt_scene[split].items():
            print_str = " | ".join("{}: {:4g}".format(k, v.avg)
                                   for k, v in scene_metric.items())
            self.logger.info("Scene name: {}\nResults: {}".format(
                scene_name, print_str))

    def reset_metric_wrt_scene(self, split):
        for scene_name, scene_metric in self.metric_wrt_scene[split].items():
            for k, v in scene_metric.items():
                self.metric_wrt_scene[split][scene_name][k].reset()

    def write_metric_scene_to_tb(self, split):
        for scene_name, scene_metric in self.metric_wrt_scene[split].items():
            for k, v in scene_metric.items():
                self.metric_scene_writter.add_scalar(
                    "{}_metric/{}".format(scene_name, k), v.avg, self.epoch)
