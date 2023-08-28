# key point variance loss
import os
import shutil
import torch

from model import ResampleGAN

from dataset import TestDataset

from arguments import Arguments

import time
import open3d as o3d

from functools import partial
from collate_func import collate_func
import tool
from tqdm import tqdm


class SIRATest():

    def __init__(self, args):
        self.args = args
        # ------------------------------------------------Dataset---------------------------------------------- #

        self.data = TestDataset(root=args.dataset_path)
        self.dataLoader = torch.utils.data.DataLoader(
            self.data,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=partial(collate_func))
        print("Testing Dataset : {} prepared.".format(len(self.data)))
        # ----------------------------------------------------------------------------------------------------- #

        # -------------------------------------------------Module---------------------------------------------- #
        self.G = ResampleGAN(input_dim=1,
                             dimofbottelneck=256,
                             init_dim=64,
                             kernel_size=15,
                             init_radius=args.radius,
                             init_sigma=args.sigma,
                             group_norm=32).to(args.device)

        print("Network prepared.")

        # ----------------------------------------------------------------------------------------------------- #

    def run(self, save_ckpt=None, load_ckpt=None, result_path=None):
        epoch_log = 0

        loss_log = {'Chamfer_loss': [], 'G_loss': [], 'D_loss': []}
        loss_legend = list(loss_log.keys())

        if load_ckpt is not None:
            checkpoint = torch.load(load_ckpt, map_location=self.args.device)
            self.G.load_state_dict(checkpoint['G_state_dict'])

            epoch_log = checkpoint['epoch']

            loss_log['Chamfer_loss'] = checkpoint['Chamfer_loss']
            loss_log['G_loss'] = checkpoint['G_loss']
            loss_log['D_loss'] = checkpoint['D_loss']

            loss_legend = list(loss_log.keys())

            print("Checkpoint loaded.")

        pbar = tqdm(self.dataLoader)

        self.G.eval()
        for _iter, data in enumerate(pbar):
            # Start Time
            start_time = time.time()
            # To GPU
            data = tool.tensor_gpu(data, self.args.device)

            # -------------------------Save points------------------------#
            with torch.no_grad():
                fake_points = self.G(data["synth"])
            synth_rot = data["synth_rot"]
            synth_centroid = data["synth_centroid"]
            fake_points = fake_points @ synth_rot + synth_centroid
            o3d_pc = o3d.geometry.PointCloud()
            o3d_pc.points = o3d.utility.Vector3dVector(
                fake_points.detach().cpu().numpy())

            ply_path = data["synth_path"].replace("FlyingShapes",
                                                  args.experiment_name)
            ply_path = ply_path.replace("pc_tsdf.ply", "pc.ply")
            os.makedirs(ply_path.rstrip("pc.ply"), exist_ok=True)
            o3d.io.write_point_cloud(ply_path, o3d_pc)
            pbar.set_description("Successfully saved in {}".format(ply_path))

        # send gt.log
        shutil.copy("dataset/generated_pairs/gt.log",
                    "../PCR/dataset/{}/gt.log".format(args.experiment_name))


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

    model = SIRATest(args)
    model.run(save_ckpt=SAVE_CHECKPOINT,
              load_ckpt=LOAD_CHECKPOINT,
              result_path=RESULT_PATH)
