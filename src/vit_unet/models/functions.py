import itertools
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import torch
from skimage.metrics import peak_signal_noise_ratio  # pyright: ignore[reportUnknownVariableType]


class Batch(TypedDict):
    x: torch.Tensor
    y: torch.Tensor


def psnr(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader[Batch]) -> npt.NDArray[np.float64]:
    # Calculate PSNR
    score: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to("cuda").float()
            output = model(x).cpu().numpy()
            y = batch["y"].numpy()

            score.extend(peak_signal_noise_ratio(y[i], output[i]) for i in range(len(output)))
    score_np = np.array(score)
    return score_np


def softmax_top(x: torch.Tensor, top: int) -> torch.Tensor:
    batch_size, channels, shape, _ = x.size()
    x = x.clone()
    values, idx = torch.topk(x, top, dim=-1)
    values = torch.nn.functional.softmax(values, dim=-1)
    idx = torch.unsqueeze(idx.flatten(), 0)
    dct_axis = torch.as_tensor(
        list(itertools.product(range(batch_size), range(channels), range(shape))),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    axis = torch.stack([elem.T for elem in dct_axis for _ in range(top)], dim=-1)
    idx = torch.cat([axis, idx], dim=0)
    y = torch.sparse_coo_tensor(idx, values.flatten(), x.size(), device=x.device, dtype=x.dtype).to_dense()  # pyright: ignore[reportUnknownMemberType]
    return y
