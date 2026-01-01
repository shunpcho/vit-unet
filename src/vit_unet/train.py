from __future__ import annotations

from pathlib import Path
from typing import Literal

import albumentations
import cv2
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold

import vit_unet.models.functions as fn
import vit_unet.models.model as models
import wandb
from vit_unet import dataset


def train(
    input_folder: str = "/home/s.chochi/denoiser/data/CC15",
    n_epochs: int = 5,
    folds: int = 5,
    model_string: Literal["lite", "base", "large"] = "lite",
    lr: float = 0.0001,
    batch_size: int = 4,
    im_size: int | tuple[int, int] = 224,
):
    torch.random.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    WB_ENTITY = "kshuchi0203-hitachi"
    wandb.login(key="556896069873c52312295f88d3f78b1d6e67e0a9")  # WANDB KEY
    with wandb.init(project="ViT_UNet", entity=WB_ENTITY) as run:
        wandb.config.update({"n_epochs": n_epochs, "fold": folds, "model": model_string, "lr": lr})

        # prepare Data
        clean = np.array(sorted(Path(input_folder).glob("*mean*")))
        noisy = np.array(sorted(Path(input_folder).glob("*real*")))

        assert len(clean) == len(noisy), f"Clean length {len(clean)} is not equal to Noisy length {len(noisy)}"

        cv = KFold(folds, shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(noisy)):
            train = noisy[train_idx]
            test = noisy[test_idx]

            print(f"FOLD {fold}: Training on {len(train)} samples and testing on {len(test)} samples")
            # Create dataset and dataloader
            train_transform = albumentations.Compose(
                [
                    albumentations.ShiftScaleRotate(
                        shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0
                    ),
                    # albumentations.Normalize(mean=(0.456), std=(0.224), max_pixel_value=255.0, p=1.0)
                ]
            )

            test_transform = albumentations.Compose(
                [albumentations.Normalize(mean=(0.456), std=(0.224), max_pixel_value=255.0, p=1.0)]
            )
            train_dataloader = torch.utils.data.DataLoader(
                dataset.DenoisingDataset(
                    train,
                    clean_folder=input_folder,
                    noisy_folder=input_folder,
                    augments=train_transform,
                    im_size=im_size,
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
            )
            test_dataloader = torch.utils.data.DataLoader(
                dataset.DenoisingDataset(
                    test,
                    clean_folder=input_folder,
                    noisy_folder=input_folder,
                    # augments=test_transform,
                    im_size=im_size,
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
            )

            # Create model
            model = models.get_vit_unet(model_string)
            model.to(device)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

            # Create fitter
            fitter = dataset.ImageFitter(model, loss=criterion, optimizer=optimizer, device=device, folder="models")

            def wandb_update(x):
                data_log = x.copy()
                del data_log["epoch"]
                wandb.log({"training_" + str(fold): data_log})

            history = fitter.fit(train_dataloader, test_dataloader, n_epochs=n_epochs, callbacks=[wandb_update])

            fitter.load("models/best-checkpoint.bin")

            # Calculate PSNR
            model = fitter.model
            model.eval()
            score = fn.psnr(model, test_dataloader)
            print(f"FOLD {fold}: Mean PSNR {np.mean(score)}")
            results.append(score)

        print(f"Average Mean PSNR{np.mean(results)}. STD Mean PSNR {np.std(results)}")

        run.log({"psnr_mean": np.mean(results), "psnr_std": np.std(results)})

        run.finish()


def eval(
    model_string: Literal["lite", "base", "large"] = "lite",
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.get_vit_unet(model_string)
    model.load("models/best-checkpoint.bin")
    model.to(device)

    test_dataloader = torch.utils.data.DataLoader(
        dataset.DenoisingDataset(
            test,
            clean_folder=input_folder,
            noisy_folder=input_folder,
            # augments=test_transform,
            im_size=im_size,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    model.eval()
    results = []
    with torch.no_grad():
        for batch in test_dataloader:
            x = batch["x"].to(device).float()
            output = model(x)
            y = batch["y"]
            results.append((x[0].cpu().numpy(), y[0].numpy(), output[0].cpu().numpy()))

    original = x[0].cpu().numpy().transpose(1, 2, 0)
    clean = y[0].numpy().transpose(1, 2, 0)
    reconstructed = output[0].cpu().numpy().transpose(1, 2, 0)

    for a, b, c in results:
        original = a.transpose(1, 2, 0)
        clean = b.transpose(1, 2, 0)
        reconstructed = c.transpose(1, 2, 0)
        fig, ax = plt.subplots(1, 3, figsize=(10, 10))

        ax[0].imshow(original)
        ax[1].imshow(clean)
        ax[2].imshow(reconstructed)
        plt.show()


if __name__ == "__main__":
    fire.Fire(train)
