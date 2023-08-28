import argparse
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from data_loader.data_loader import fetch_dataloader
from model.model import fetch_model
from loss.loss import compute_loss, compute_metric
from common import tool
from common.manager import Manager
from common.config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",
                    default="",
                    type=str,
                    help="Directory containing params.json")
parser.add_argument("--resume",
                    default="",
                    type=str,
                    help="Path of model weights")
parser.add_argument('--benchmark',
                    choices=['3DMatch', '3DLoMatch', 'ETH'],
                    required=True,
                    help='test benchmark')


def recover_scale(input, output):
    if "scale" in input.keys():
        scale_factor = input["scale"]
        output["ref_points"] = output["ref_points"] / scale_factor
        output["src_points"] = output["src_points"] / scale_factor
        output["ref_points_f"] = output["ref_points_f"] / scale_factor
        output["src_points_f"] = output["src_points_f"] / scale_factor
        output["ref_points_c"] = output["ref_points_c"] / scale_factor
        output["src_points_c"] = output["src_points_c"] / scale_factor
        output["ref_corr_points"] = output["ref_corr_points"] / scale_factor
        output["src_corr_points"] = output["src_corr_points"] / scale_factor
        output["estimated_transform"][:3, 3] = output[
            "estimated_transform"][:3, 3] / scale_factor

    return input, output


def test(model, mng: Manager):
    # Set model to evaluation mode
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        # Compute metrics over the dataset
        for split in ["test"]:
            if split not in mng.dataloader:
                continue
            # Initialize loss and metric statuses
            mng.reset_loss_status()
            mng.reset_metric_status(split)
            mng.reset_metric_wrt_scene(split)
            # Use tqdm for progress bar
            if mng.cfg.is_master:
                t = tqdm(total=len(mng.dataloader[split]))

            for batch_idx, batch_input in enumerate(mng.dataloader[split]):
                # Move data to GPU if available
                batch_input = tool.tensor_gpu(batch_input, mng.cfg.local_rank)
                # Compute model output
                batch_output = model(batch_input)
                # Recover scale for ETH dataset
                batch_input, batch_output = recover_scale(
                    batch_input, batch_output)
                # # Compute all loss on this batch
                # loss = compute_loss(mng.cfg, batch_input, batch_output)
                # mng.update_loss_status(loss, batch_size)
                # Compute all metrics on this batch
                metric = compute_metric(mng.cfg, batch_input, batch_output)
                mng.after_test_step(mng.cfg.test.subset, batch_input,
                                    batch_output)
                mng.update_metric_status(metric, split,
                                         mng.cfg.test.batch_size)
                mng.update_metric_wrt_scene(split, batch_input, metric,
                                            mng.cfg.test.batch_size)

                if mng.cfg.is_master:
                    # Test info print
                    print_str = mng.tqdm_info(split)
                    # Tqdm settings
                    t.set_description(desc=print_str)
                    t.update()

            if mng.cfg.is_master:
                # Print the metric
                mng.print_metric(split, only_best=True, wrt_scene=True)
                t.close()


def main(local_rank, cfg):
    # Set rank and is_master flag
    cfg.local_rank = local_rank
    cfg.is_master = cfg.local_rank == 0
    cfg.only_weights = True
    # Set the logger
    logger = tool.set_logger(os.path.join(cfg.model_dir, "test.log"))
    # Print GPU ids
    gpu_ids = ", ".join(str(i) for i in [j for j in range(cfg.num_gpu)])
    logger.info("Using GPU ids: [{}]".format(gpu_ids))
    logger.info("Running DDP experiments on rank {}.".format(cfg.local_rank))
    # DDP setup
    tool.setup(cfg)
    # Fetch dataloader
    dl = fetch_dataloader(cfg)
    # Fetch model
    model = fetch_model(cfg)
    # Initialize manager
    mng = Manager(model=model,
                  optimizer=None,
                  scheduler=None,
                  cfg=cfg,
                  dataloader=dl,
                  logger=logger)
    # Test the model
    mng.logger.info("Starting test.")
    # Load weights from restore_file if specified
    if mng.cfg.resume is not None:
        mng.load_ckpt()
    test(model, mng)
    # DDP cleanup
    tool.cleanup()


if __name__ == "__main__":
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "cfg.json")
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    cfg = Config(json_path)
    # Update args into cfg
    cfg = cfg.update(vars(args))
    # Use GPU if available
    cfg.cuda = torch.cuda.is_available()
    if cfg.cuda:
        cfg.num_gpu = torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True
    # Set world size
    cfg.world_size = cfg.num_gpu
    # Set benchmark
    cfg.test.subset = args.benchmark
    # Main function
    mp.spawn(main, nprocs=cfg.world_size, args=(cfg, ))
