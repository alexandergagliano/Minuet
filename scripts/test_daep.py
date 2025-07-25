#!/usr/bin/env python
import os, glob, time, argparse, h5py
import numpy as np
import sys
import torch, torchvision
import pickle

# wipe anything that begins with "SLURM_"
#for k in list(os.environ):
#    if k.startswith("SLURM_"):
#        os.environ.pop(k)

# optional: be extra sure no distributed world is set
#os.environ.pop("WORLD_SIZE",  None)
#os.environ.pop("LOCAL_RANK",  None)
#os.environ.pop("RANK",        None)

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import loggers as pl_loggers
runDir  = '/n/holylabs/LABS/iaifi_lab/Users/agagliano/galaxyAutoencoder/'
sys.path.insert(0, runDir + '/ssl-legacysurvey/')
from ssl_legacysurvey.data_loaders import datamodules
from daep.LitWrapper import LitMaskeddaep

class Collectmaskeddaep(pl.Callback):
    """Collect per-object outputs during `trainer.test` and dump to HDF5."""
    BRIGHT_LIMIT      = 18.0        # r-mag cut
    N_SAMPLES_TO_SAVE = 100

    def __init__(self, out_path: str):
        super().__init__()
        self.out_path = out_path
        self._buf     = []

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self._buf.append({
            "img":   outputs["img"].detach().cpu(), 
            "mask":  outputs["mask"].detach().cpu(),
            "rec":   outputs["rec"].detach().cpu(),
            "lat":   outputs["lat"].detach().cpu(),
            "y":     outputs["y"].detach().cpu(),
            "rmag":  outputs["rmag"].detach().cpu(),
            "mmd":   outputs["mmd"].detach().cpu().unsqueeze(0),
            "score": outputs["score"].detach().cpu().unsqueeze(0),
        })

    def on_test_epoch_end(self, trainer, pl_module):
        def cat(key):
            t = torch.cat([d[key] for d in self._buf], dim=0)  # real Tensor
            return t.cpu()

        imgs   = cat("img").numpy()
        masks  = cat("mask").numpy()
        recs   = cat("rec").numpy()
        lats   = cat("lat").numpy()
        ys     = cat("y").numpy()
        rmags  = cat("rmag").numpy()
        mmds   = cat("mmd").numpy()
        scores = cat("score").numpy()

        bright = rmags.ravel() < self.BRIGHT_LIMIT

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with h5py.File(self.out_path, "w") as hf:
            hf.create_dataset("images", data=imgs)
            hf.create_dataset("seg_mask", data=masks)
            #hf.create_dataset("recon",  data=recs)
            hf.create_dataset("recon",      data=recs[:, 0]) # save one sample if using ddpm
            hf.create_dataset("latent", data=lats)
            hf.create_dataset("y_true", data=ys)
            hf.create_dataset("mmd",    data=mmds)
            hf.create_dataset("score",  data=scores)

            # bright subset
            hf.create_dataset("bright/images",   data=imgs[bright])
            hf.create_dataset("bright/seg_mask", data=masks[bright])
            hf.create_dataset("bright/recon",    data=recs[bright])
            hf.create_dataset("bright/latent",   data=lats[bright])
            hf.create_dataset("bright/y_true",   data=ys[bright])

            # quick-look samples
            keep = slice(0, self.N_SAMPLES_TO_SAVE)
            hf.create_dataset("samples/images",   data=imgs[keep])
            hf.create_dataset("samples/seg_mask", data=masks[keep])
            #hf.create_dataset("samples/recon",    data=recs[keep])
            hf.create_dataset("samples/recon",    data=recs[keep, 0])

            # metadata
            hf.attrs.update(
                avg_mmd   = float(mmds.mean()),
                avg_score = float(scores.mean()),
                n_total   = int(imgs.shape[0]),
                n_bright  = int(bright.sum()),
                saved_at  = int(time.time()),
            )

            pl_module.log("test/avg_mmd",   hf.attrs["avg_mmd"])
            pl_module.log("test/avg_score", hf.attrs["avg_score"])
        print(f"✓ Wrote HDF5 to {self.out_path}")

# ------------------------------------------------------------------
def run_single_checkpoint(ckpt_path: str, dm_params: dict):
    prefix  = os.path.basename(os.path.dirname(ckpt_path)).replace("fullTrain_", "")
    runRoot = os.path.dirname(os.path.dirname(os.path.dirname(ckpt_path)))

    # ---------------- DataModule -----------------------------------
    dm = datamodules.DecalsDataModule(dm_params)
    dm.test_transforms = None
    dm.prepare_data(); dm.setup(stage="test")

    # ---------------- model ----------------------------------------
    model = LitMaskeddaep.load_from_checkpoint(ckpt_path, map_location="cuda", weights_only=False)
    model.eval()

    # ---------------- logging + callback ---------------------------
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.dirname(ckpt_path),
        name=f"{prefix}_test",
    )

    out_h5 = os.path.join(
        runRoot, "test_results",
        f"maskeddaep5_results_sample_{int(time.time())}_{prefix}.h5"
    )
    collector = Collectmaskeddaep(out_h5)

    trainer = pl.Trainer(
        accelerator="gpu", devices=1,
        strategy='auto', #DDPStrategy(),
        logger=tb_logger,
        callbacks=[collector],
        max_epochs=1,
        limit_test_batches=0.25,
        #limit_test_batches=0.1,
        num_sanity_val_steps=0,
    )

    trainer.test(model, dataloaders=dm.test_dataloader())

# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="/n/holylabs/LABS/iaifi_lab/Users/agagliano/galaxyAutoencoder/models/fullTrain_maskdaep5/last.ckpt",
        help="Glob pattern to maskeddaep checkpoint(s) to evaluate."
    )
    args = parser.parse_args()

    # Shared datamodule params
    dm_kwargs = dict(
        data_path         = "/n/holylabs/LABS/iaifi_lab/Users/agagliano/galaxyAutoencoder/split_files/Test/",
        val_data_path     = "/n/holylabs/LABS/iaifi_lab/Users/agagliano/galaxyAutoencoder/split_files/Test/",
        test_data_path    = "/n/holylabs/LABS/iaifi_lab/Users/agagliano/galaxyAutoencoder/split_files/Test/",
        augmentations     = "grccrggn",
        val_augmentations = "grccrg",
        test_augmentations = 'grccrg',
        max_num_samples_test = None,
        batch_size        = 512,
        num_workers       = 64,
        pin_memory        = True,
        segment           = True,
    )

    checkpoints = sorted(glob.glob(args.glob))
    if not checkpoints:
        raise RuntimeError(f"No checkpoints matched {args.glob}")

    for ckpt in checkpoints:
        print(f"▶ Testing {ckpt}")
        run_single_checkpoint(ckpt, dm_kwargs)

