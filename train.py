from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime
import logging
import os
from dotenv import load_dotenv
from utils import load_config, create_output_dir_and_save_config, upload_to_gcs_bucket
from dataset import SatMapDataset, graph_collate_fn
from model import SAMRoad

import wandb

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor


load_dotenv()

parser = ArgumentParser()
parser.add_argument(
    "--config",
    default='config/toponet_vitb_256_os.yaml',
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the nnU-Net config. See /config for examples.",
)
parser.add_argument(
    "--resume", default=None, help="checkpoint of the last epoch of the model"
)
parser.add_argument(
    "--precision", default=16, help="32 or 16"
)
parser.add_argument(
    "--fast_dev_run", default=False, action='store_true'
)
parser.add_argument(
    "--dev_run", default=False, action='store_true'
)
parser.add_argument(
    "--upload_to_gcs", default=True, action='store_true',
    help="Whether to upload the output directory to GCS bucket after training"
)
parser.add_argument(
    "--gcs_bucket", default="extract-general",
    help="Name of the GCS bucket to upload to"
)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    args = parser.parse_args()
    config = load_config(args.config)
    dev_run = args.dev_run or args.fast_dev_run

    # Create output directory and save config
    custom_tags = 'new_dataset'
    name = f'{config.DATASET_ID}_{custom_tags}_{datetime.datetime.now().strftime("%d_%H%M")}'
    
    # Create shared directory for checkpoints, wandb logs, and config files
    output_dir_prefix = 'output/'
    shared_dir = create_output_dir_and_save_config(
        output_dir_prefix=f"{output_dir_prefix}/{name}", 
        config=config
    )
    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="sam_road",
        # track hyperparameters and run metadata
        config=config,
        # disable wandb if debugging
        mode='disabled' if dev_run else None,
        name=name,
        dir=shared_dir  # Using the shared directory created above
    )


    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    

    net = SAMRoad(config)

    train_ds, val_ds = SatMapDataset(config, is_train=True, dev_run=dev_run), SatMapDataset(config, is_train=False, dev_run=dev_run)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )

    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1, dirpath=shared_dir)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize WandbLogger with the same directory
    wandb_logger = WandbLogger(save_dir=shared_dir)

    # from lightning.pytorch.profilers import AdvancedProfiler
    # profiler = AdvancedProfiler(dirpath='profile', filename='result_fast_matcher')

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        fast_dev_run=args.fast_dev_run,
        # strategy='ddp_find_unused_parameters_true',
        precision=args.precision,
        default_root_dir=shared_dir,  # Using the shared directory
        # profiler=profiler
        )

    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # After training is complete, upload to GCS if specified
    if args.upload_to_gcs and not dev_run:
        logging.info(f"Training completed. Uploading output directory {shared_dir} to GCS bucket {args.gcs_bucket}")
        
        # Use the basename of the directory as the destination prefix 
        # to maintain the directory structure in the bucket
        destination_prefix = os.path.join('sam_road/outputs/', os.path.basename(shared_dir))
        print(destination_prefix)
        # Upload the directory to GCS
        success = upload_to_gcs_bucket(
            source_directory=shared_dir,
            bucket_name=args.gcs_bucket,
            destination_prefix=destination_prefix
        )
        
        if success:
            logging.info(f"Successfully uploaded output directory to gs://{args.gcs_bucket}/{destination_prefix}")
        else:
            logging.error(f"Failed to upload output directory to GCS bucket")