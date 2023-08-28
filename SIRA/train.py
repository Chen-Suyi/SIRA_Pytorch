# key point variance loss
import os
import torch
from torch.functional import norm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points

from model import ResampleGAN, MultiScalePointNet

from dataset import UnalignedDataset
from shutil import copyfile

from arguments import Arguments

import time
# import visdom
import numpy as np
import open3d as o3d

from functools import partial
from collate_func import collate_func
import tool


class SIRATrain():

    def __init__(self, args):
        self.args = args
        # ------------------------------------------------Dataset---------------------------------------------- #

        self.data = UnalignedDataset(root=args.dataset_path)
        self.dataLoader = torch.utils.data.DataLoader(
            self.data,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=partial(collate_func))
        print("Training Dataset : {} prepared.".format(len(self.data)))
        # ----------------------------------------------------------------------------------------------------- #

        # -------------------------------------------------Module---------------------------------------------- #
        self.G = ResampleGAN(input_dim=1,
                             dimofbottelneck=256,
                             init_dim=64,
                             kernel_size=15,
                             init_radius=args.radius,
                             init_sigma=args.sigma,
                             group_norm=32).to(args.device)

        self.D = MultiScalePointNet().to(args.device)

        # -------------------------------------------------adam---------------------------------------------- #
        self.optimizerG = optim.Adam(self.G.parameters(),
                                     lr=args.lr,
                                     betas=(0.5, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(),
                                     lr=args.lr,
                                     betas=(0, 0.99))

        self.criterionGAN = nn.MSELoss()

        print("Network prepared.")
        # ----------------------------------------------------------------------------------------------------- #

        # ---------------------------------------------Visualization------------------------------------------- #
        loss_writer_path = os.path.join(args.summary_path, "loss")
        os.makedirs(loss_writer_path, exist_ok=True)
        self.loss_writer = SummaryWriter(log_dir=loss_writer_path)
        pointcloud_writer_path = os.path.join(args.summary_path, "pointcloud")
        os.makedirs(pointcloud_writer_path, exist_ok=True)
        self.pointcloud_writer = SummaryWriter(log_dir=pointcloud_writer_path)
        # ----------------------------------------------------------------------------------------------------- #

    def run(self, save_ckpt=None, load_ckpt=None, result_path=None):
        epoch_log = 0

        loss_log = {'Chamfer_loss': [], 'G_loss': [], 'D_loss': []}
        loss_legend = list(loss_log.keys())

        if load_ckpt is not None:
            checkpoint = torch.load(load_ckpt, map_location=self.args.device)
            self.G.load_state_dict(checkpoint['G_state_dict'])
            self.D.load_state_dict(checkpoint['D_state_dict'])

            epoch_log = checkpoint['epoch'] + 1

            loss_log['Chamfer_loss'] = checkpoint['Chamfer_loss']
            loss_log['G_loss'] = checkpoint['G_loss']
            loss_log['D_loss'] = checkpoint['D_loss']

            loss_legend = list(loss_log.keys())

            print("Checkpoint loaded.")

        for epoch in range(epoch_log, self.args.epochs):
            self.G.train()
            self.D.train()

            for _iter, data in enumerate(self.dataLoader):
                # Start Time
                start_time = time.time()
                # To GPU
                data = tool.tensor_gpu(data, self.args.device)

                # ----------------- Generator forward ------------------- #
                fake_points = self.G(data["synth"])

                # ---------------------- Preprocess ----------------------#
                real_points = data["real"]["points"][0]
                real_KNN = knn_points(real_points[None, ...],
                                      real_points[None, ...],
                                      K=20)
                real_neighbors = real_KNN.idx.squeeze()

                fake_KNN = knn_points(fake_points[None, ...],
                                      fake_points[None, ...],
                                      K=20)
                fake_neighbors = fake_KNN.idx.squeeze()

                # -------------------- Discriminator -------------------- #
                for d_iter in range(self.args.D_iter):

                    D_real = self.D(real_points, real_neighbors)
                    d_real_loss = self.criterionGAN(
                        D_real,
                        torch.ones(D_real.shape,
                                   device=self.args.device)).mean()

                    D_fake = self.D(fake_points.detach(),
                                    fake_neighbors.detach())
                    d_fake_loss = self.criterionGAN(
                        D_fake,
                        torch.zeros(D_fake.shape,
                                    device=self.args.device)).mean()

                    d_loss = (d_real_loss + d_fake_loss) * 0.5

                    self.D.zero_grad()
                    d_loss.backward()
                    self.optimizerD.step()

                loss_log['D_loss'].append(d_loss.item())

                torch.cuda.empty_cache()

                # ------------------ Generator backward ------------------ #
                G_fake = self.D(fake_points, fake_neighbors)

                g_loss = self.criterionGAN(
                    G_fake, torch.ones(G_fake.shape,
                                       device=self.args.device)).mean()

                synth_points = data["synth"]["points"][0]
                chamfer_loss, _ = chamfer_distance(fake_points[None, ...],
                                                   synth_points[None, ...],
                                                   point_reduction="sum")

                weight = torch.exp(-10 * g_loss.detach())  # default: 10
                g_loss_chamfer = g_loss + weight * chamfer_loss

                self.G.zero_grad()
                g_loss_chamfer.backward()
                self.optimizerG.step()

                loss_log['Chamfer_loss'].append(chamfer_loss.item())
                loss_log['G_loss'].append(g_loss.item())

                torch.cuda.empty_cache()

                # # -------------------------Save points------------------------#
                # fake_points = self.G(data["synth"])
                # synth_rot = data["synth_rot"]
                # synth_centroid = data["synth_centroid"]
                # fake_points = fake_points @ synth_rot + synth_centroid
                # o3d_pc = o3d.geometry.PointCloud()
                # o3d_pc.points = o3d.utility.Vector3dVector(
                #     fake_points.detach().cpu().numpy())

                # ply_path = data["fake_path"]
                # os.makedirs(ply_path.rstrip("pc.ply"), exist_ok=True)
                # o3d.io.write_point_cloud(ply_path, o3d_pc)
                # # print("Successfully saved in {}".format(ply_path))

                # --------------------- Visualization -------------------- #

                print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                      "[ Chamfer_loss ] ", "{: 7.6f}".format(chamfer_loss),
                      "[ D_Loss ] ", "{: 7.6f}".format(d_loss), "[ G_Loss ] ",
                      "{: 7.6f}".format(g_loss), "[ Time ] ",
                      "{:4.2f}s".format(time.time() - start_time))

                if _iter % 1 == 0:
                    self.loss_writer.add_scalar("Loss/Discriminator Loss",
                                                d_loss.item(),
                                                epoch * len(self.data) + _iter)
                    self.loss_writer.add_scalar("Loss/Generator Loss",
                                                g_loss.item(),
                                                epoch * len(self.data) + _iter)
                    self.loss_writer.add_scalar("Loss/Chamfer_loss Loss Fake",
                                                chamfer_loss.item(),
                                                epoch * len(self.data) + _iter)

                if _iter % 10 == 0:
                    self.pointcloud_writer.add_mesh("Synth Pointcloud",
                                                    synth_points[None, ...])
                    self.pointcloud_writer.add_mesh("Real Pointcloud",
                                                    real_points[None, ...])
                    self.pointcloud_writer.add_mesh("Fake Pointcloud",
                                                    fake_points[None, ...])

                    print('Pointclouds are saved.')

            # ---------------------- Save checkpoint --------------------- #
            if not save_ckpt == None:
                torch.save(
                    {
                        'epoch': epoch,
                        'D_state_dict': self.D.state_dict(),
                        'G_state_dict': self.G.state_dict(),
                        'Chamfer_loss': loss_log['Chamfer_loss'],
                        'D_loss': loss_log['D_loss'],
                        'G_loss': loss_log['G_loss'],
                    }, save_ckpt + '{:04d}_latest.pt'.format(epoch))

                print('Checkpoint is saved.')


if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.experiment_name = 'synth2real'
    args.dataset_path = 'dataset/synth2real'
    args.ckpt_path = 'experiment/{}/ckpt/'.format(args.experiment_name)
    args.result_path = 'experiment/{}/generated/'.format(args.experiment_name)
    args.summary_path = 'experiment/{}/summary'.format(args.experiment_name)

    args.device = torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)
    print("Run on {}".format(args.device))

    SAVE_CHECKPOINT = args.ckpt_path + args.ckpt_save if args.ckpt_save is not None else None
    LOAD_CHECKPOINT = args.ckpt_path + args.ckpt_load if args.ckpt_load is not None else None
    RESULT_PATH = args.result_path  # + args.result_save
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    model = SIRATrain(args)
    model.run(save_ckpt=SAVE_CHECKPOINT,
              load_ckpt=LOAD_CHECKPOINT,
              result_path=RESULT_PATH)
