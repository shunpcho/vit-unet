from __future__ import annotations

from pathlib import Path
from typing import Literal

import albumentations
import cv2
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from sklearn.model_selection import KFold

import vit_unet.models.functions as fn
import wandb
from vit_unet import dataset
from vit_unet.models.build_model import get_vit_unet


def train(
    input_folder: str = "/home/s.chochi/ai-works/denoiser/data/CC15",
    n_epochs: int = 80,
    folds: int = 3,
    model_string: Literal["lite", "base", "large"] = "lite",
    lr: float = 1e-5,
    batch_size: int = 8,
    im_size: int | tuple[int, int] = 256,
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
            model = get_vit_unet(model_string)
            model.to(device)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

            # Add learning rate scheduler for stability
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr / 10)

            # Create fitter
            fitter = dataset.ImageFitter(
                model, loss=criterion, optimizer=optimizer, device=device, folder="results/models"
            )

            def wandb_update(x):
                data_log = x.copy()
                del data_log["epoch"]
                wandb.log({"training_" + str(fold): data_log})
                # Step learning rate scheduler after each epoch
                scheduler.step()

            history = fitter.fit(train_dataloader, test_dataloader, n_epochs=n_epochs, callbacks=[wandb_update])

            fitter.load("results/models/best-checkpoint.bin")

            # Calculate PSNR
            model = fitter.model
            model.eval()
            score = fn.psnr(model, test_dataloader, device)
            print(f"FOLD {fold}: Mean PSNR {np.mean(score)}")
            results.append(score)

        print(f"Average Mean PSNR{np.mean(results)}. STD Mean PSNR {np.std(results)}")

        run.log({"psnr_mean": np.mean(results), "psnr_std": np.std(results)})

        run.finish()


def eval(
    input_folder: str = "/home/s.chochi/ai-works/denoiser/data/CC15",
    model_string: Literal["lite", "base", "large"] = "lite",
    model_path: str = "results/models/best-checkpoint.bin",
    wandb_run_path: str | None = None,
    batch_size: int = 4,
    im_size: int | tuple[int, int] = 256,
    output_folder: str = "results/inference_results",
) -> None:
    """Run inference with a model trained on wandb.

    Args:
        input_folder: Path to the input images folder
        model_string: Model type (lite, base, large)
        model_path: Path to the local model file
        wandb_run_path: Run path to download model from wandb (e.g., "entity/project/run_id")
        batch_size: Batch size for inference
        im_size: Image size
        output_folder: Folder to save inference results
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_vit_unet(model_string)

    # Download model from wandb
    if wandb_run_path:
        print(f"Downloading model from wandb: {wandb_run_path}")
        api = wandb.Api()
        run = api.run(wandb_run_path)
        # Download model file
        model_file = run.file("results/models/best-checkpoint.bin")
        model_file.download(replace=True)
        model_path = "results/models/best-checkpoint.bin"

    # Load model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Prepare test data
    test_files = np.array(sorted(Path(input_folder).glob("*real*")))
    print(f"Found {len(test_files)} test images")

    test_dataloader = torch.utils.data.DataLoader(
        dataset.DenoisingDataset(
            test_files,
            clean_folder=input_folder,
            noisy_folder=input_folder,
            im_size=im_size,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Run inference
    results = []
    psnr_scores = []
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            x = batch["x"].to(device).float()
            y = batch["y"]
            output = model(x)

            # Process each image in the batch
            for i in range(x.shape[0]):
                noisy_img = x[i].cpu().numpy().transpose(1, 2, 0)
                clean_img = y[i].numpy().transpose(1, 2, 0)
                denoised_img = output[i].cpu().numpy().transpose(1, 2, 0)

                # Calculate PSNR
                psnr = peak_signal_noise_ratio(clean_img, denoised_img)
                psnr_scores.append(psnr)

                results.append((noisy_img, clean_img, denoised_img, psnr))

                # Save images
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(np.clip(noisy_img, 0, 1))
                ax[0].set_title("Noisy Input")
                ax[0].axis("off")

                ax[1].imshow(np.clip(clean_img, 0, 1))
                ax[1].set_title("Ground Truth")
                ax[1].axis("off")

                ax[2].imshow(np.clip(denoised_img, 0, 1))
                ax[2].set_title(f"Denoised (PSNR: {psnr:.2f})")
                ax[2].axis("off")

                plt.tight_layout()
                plt.savefig(output_path / f"result_{batch_idx * batch_size + i:04d}.png")
                plt.close()

    # Display statistics
    mean_psnr = np.mean(psnr_scores)
    std_psnr = np.std(psnr_scores)
    print("\nInference Results:")
    print(f"Number of images processed: {len(results)}")
    print(f"Mean PSNR: {mean_psnr:.2f} dB")
    print(f"PSNR Std Dev: {std_psnr:.2f} dB")
    print(f"Results saved to: {output_folder}")


if __name__ == "__main__":
    fire.Fire()
    # fire.Fire(eval)
