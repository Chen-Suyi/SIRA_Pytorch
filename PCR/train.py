import argparse
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from easydict import EasyDict as edict
from data_loader.data_loader import fetch_dataloader
from model.model import fetch_model
from optimizer.optimizer import fetch_optimizer
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
                    default=None,
                    type=str,
                    help="Path of model weights")
parser.add_argument("-ow",
                    "--only_weights",
                    action="store_true",
                    help="Only load model weights or load all train status")


def train(model, mng: Manager):
    # Reset loss status
    mng.reset_loss_status()
    # Set model to training mode
    torch.cuda.empty_cache()
    model.train()
    # Use tqdm for progress bar
    if mng.cfg.is_master:
        t = tqdm(total=len(mng.dataloader["train"]))
    # Train loop
    for batch_idx, batch_input in enumerate(mng.dataloader["train"]):
        # Move input to GPU if available
        batch_input = tool.tensor_gpu(batch_input, mng.cfg.local_rank)
        # Compute model output and loss
        batch_output = model(batch_input)
        loss = compute_loss(mng.cfg, batch_input, batch_output)
        # Update loss status and print current loss and average loss
        mng.update_loss_status(loss=loss, batch_size=mng.cfg.train.batch_size)
        # Clean previous gradients, compute gradients of all variables wrt loss
        mng.optimizer.zero_grad()
        loss["total"].backward()
        # Perform updates using calculated gradients
        mng.optimizer.step()
        # Update step: step += 1
        mng.update_step()
        # Record some information
        if mng.cfg.is_master:
            # Write loss to tensorboard
            mng.write_loss_to_tb(split="train")
            # Training info print
            print_str = mng.tqdm_info(split="train")
            # Tqdm settings
            t.set_description(desc=print_str)
            t.update()
        torch.cuda.empty_cache()
    # Close tqdm
    if mng.cfg.is_master:
        t.close()
    # Update scheduler
    mng.scheduler.step()


def evaluate(model, mng: Manager):
    # Set model to evaluation mode
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        # Compute metrics over the dataset
        for split in ["val", "test"]:
            if split not in mng.dataloader:
                continue

            # Initialize loss and metric statuses
            mng.reset_loss_status()
            mng.reset_metric_status(split)
            mng.reset_metric_wrt_scene(split)
            if mng.cfg.is_master:
                # Use tqdm for progress bar
                t = tqdm(total=len(mng.dataloader[split]))

            for batch_idx, batch_input in enumerate(mng.dataloader[split]):
                # Move data to GPU if available
                batch_input = tool.tensor_gpu(batch_input, mng.cfg.local_rank)
                # Compute model output
                batch_output = model(batch_input)
                # # Compute all loss on this batch
                # loss = compute_loss(mng.cfg, batch_input, batch_output)
                # mng.update_loss_status(loss, batch_size)
                # Compute all metrics on this batch
                metric = compute_metric(mng.cfg, batch_input, batch_output)
                mng.update_metric_status(metric, split,
                                         mng.cfg.test.batch_size)
                mng.update_metric_wrt_scene(split, batch_input, metric,
                                            mng.cfg.test.batch_size)
                torch.cuda.synchronize()

                if mng.cfg.is_master:
                    # Training info print
                    print_str = mng.tqdm_info(split=split)
                    # Tqdm settings
                    t.set_description(desc=print_str)
                    t.update()
                torch.cuda.empty_cache()

            if mng.cfg.is_master:
                # Update data to tensorboard
                mng.write_metric_to_tb(split)
                mng.write_metric_scene_to_tb(split)
                # For each epoch, update and print the metric
                mng.print_metric(split, only_best=False, wrt_scene=True)


def train_and_evaluate(model, mng: Manager):
    mng.logger.info("Starting training for {} epoch(s)".format(
        mng.cfg.num_epochs))
    # Load weights from restore_file if specified
    if mng.cfg.resume is not None:
        mng.load_ckpt()

    for epoch in range(mng.epoch, mng.cfg.num_epochs):
        # Train one epoch
        train(model, mng)
        # Evaluate one epoch
        evaluate(model, mng)
        # Check if current is best, save best and latest checkpoints
        if mng.cfg.is_master:
            mng.save_ckpt()
        # Update epoch: epoch += 1
        mng.update_epoch()


def main(local_rank, cfg):
    # Set rank and is_master flag
    cfg.local_rank = local_rank
    cfg.is_master = cfg.local_rank == 0
    # Set the logger
    logger = tool.set_logger(os.path.join(cfg.model_dir, "train.log"))
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
    # Define optimizer and scheduler
    optimizer, scheduler = fetch_optimizer(cfg, model)
    # Initialize manager
    mng = Manager(model=model,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  cfg=cfg,
                  dataloader=dl,
                  logger=logger)
    # Train the model
    train_and_evaluate(model, mng)
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
    # Main function
    mp.spawn(main, nprocs=cfg.world_size, args=(cfg, ))
