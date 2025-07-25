# trainVAE_vit.py
import os
import sys
import torch
import pickle
import numpy as np
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy

run_dir = '/n/holylabs/LABS/iaifi_lab/Users/agagliano/galaxyAutoencoder'
sys.path.insert(0, run_dir + "/ssl-legacysurvey/")

# Your data loaders & utilities
from ssl_legacysurvey.data_loaders import datamodules
from ssl_legacysurvey.utils import format_logger

# Your custom loss / metrics (optional)
from focal_frequency_loss import FocalFrequencyLoss as FFL
from daep.LitWrapper import Litdaep

def main():
    os.chdir(run_dir)
    torch.set_float32_matmul_precision("highest")

    params = {
        "img_size": 72,
        "bottleneck_len": 5,#previously 5
        "bottleneck_dim": 1,#previously 2
        "patch_size": 3,
        "model_dim": 256,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "dropout": 0.1,
        "beta": 0.000,
        "diffusion_steps": 1000,
        "fixed_positional_encoding": False,
        "learning_rate": 5.e-4,      # default fallback
        "batch_size": 128,
        "max_epochs": 1000,
        "gpus": 4,                  # will be overridden if you pass --gpus on CLI
        "strategy": "ddp",           # e.g. "ddp" or DDPStrategy(...), if desired
        "output_dir": "./logs",
        "prefix": "daep_nommd_noaug",
        "checkpoint_every_n_epochs": 5,
        "ckpt_path": None,          # if you already have a checkpoint
    }

    params["learning_rate"] = float(sys.argv[1])
    params['run_dir'] = run_dir

    params["data_path"] = run_dir + "/split_files/Train/"
    params["val_data_path"] = run_dir + "/split_files/Val/"
    params["augmentations"] = "rgcc" #grgnrr
    params["val_augmentations"] = "rgcc" #grgnrr

    prefix = params['prefix']
    dir_path = run_dir + f"/models/fullTrain_{prefix}/"
    os.makedirs(dir_path, exist_ok=True)

    params["max_num_samples"] = None
    params["max_num_samples_val"] = None
    params["pin_memory"] = True
    params["num_workers"] = 64
    params["output_dir"] = run_dir + f"/models/fullTrain_{prefix}"
    #params["ckpt_path"] = ''
    params["training_precision"] = 32

    if "--test_cpu" in sys.argv:
        params["gpus"] = 0
        params["strategy"] = None
        params["max_epochs"] = 1
        params['max_num_samples'] = 32
        params['max_num_samples_val'] = 32 
        params['batch_size'] = 64
        params['num_workers'] = 1
        params["diffusion_steps"] = 10

    # If you have normalization info:
    params["mean_dict_path"] = run_dir + "/data/norm_values_alldata.pkl"
    with open(params["mean_dict_path"], "rb") as input_file:
        params["mean_dict"] = pickle.load(input_file)

    params["segment"] = False
    params["lambda_roi"] = 0.9

    # Logging
    params["logfile_name"] = f"daep_2M_train_lr{params['learning_rate']:.2e}_{prefix}.log"

    if params["segment"]:
        print(f"Initializing segmentation module...")

    print(f"learning rate: {params['learning_rate']:.1e}")
    print(f"number of epochs: {params['max_epochs']}")

    datamodule = datamodules.DecalsDataModule(params)
    datamodule.train_transforms = None
    datamodule.val_transforms = None

    if params["ckpt_path"] is not None:
        daep = Litdaep(params=params)
        checkpoint_data = torch.load(params["ckpt_path"])
        daep.load_state_dict(checkpoint_data["state_dict"])
        print(f"Loading checkpoint {params['ckpt_path']}.")
    else:
        daep = Litdaep(params=params)
        print("Training new model")

    if params["gpus"] > 0:
        daep = daep.cuda()

    # Construct the file prefix for logging/checkpoint naming
    file_output_head = (
        f"{prefix}"
        + f"_bs{params['batch_size']}"
    )

    os.makedirs(params["output_dir"], exist_ok=True)

    logger = format_logger.create_logger(
        filename=os.path.join(params["output_dir"], params["logfile_name"])
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=params["output_dir"], name=file_output_head
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=params["output_dir"],
        filename=file_output_head + "_{epoch:03d}",
        every_n_epochs=params["checkpoint_every_n_epochs"],
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.1, patience=50, mode="min"
    )

    trainer_kwargs = {
        "max_epochs": params["max_epochs"],
        "precision": params['training_precision'],
        "devices": params["gpus"],
        "accelerator": "gpu" if params["gpus"] > 0 else "cpu",
        "callbacks": [checkpoint_callback, lr_monitor, early_stop_callback],
        "logger": [tb_logger],
    }

    if "--test_cpu" in sys.argv:
        trainer_kwargs["fast_dev_run"] = True

    if params["gpus"] == 0:
        trainer_kwargs["accelerator"] = "cpu"
        trainer_kwargs["devices"] = 1
    else:
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = params["gpus"]

    trainer = pl.Trainer(**trainer_kwargs)

    # If user explicitly set a strategy (e.g. "ddp" or DDPStrategy), pass it:
    if params["strategy"] is not None:
        trainer_kwargs["strategy"] = params["strategy"]
    else:
        # Example: if you want DDP by default on >1 GPUs:
        if params["gpus"] > 1:
            trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(**trainer_kwargs)

    logger.info("Training Model")
    print("All is initialized, time to train!")

    trainer.fit(model=daep, datamodule=datamodule)

if __name__ == "__main__":
    main()

