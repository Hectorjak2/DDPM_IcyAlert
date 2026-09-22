import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.dataloader import to_data_range


def _fmt(d):
    return np.datetime_as_string(d, unit="D")


def _prep_field(field: torch.Tensor, finite_mask: torch.Tensor = None, model_range: bool = True):
    """Squeeze to [H, W], map to data range [0, 1] if needed, and blank land."""
    field = field.detach().cpu().squeeze()
    if model_range:
        field = to_data_range(field)
    if finite_mask is not None:
        mask = finite_mask.cpu().squeeze()
        field = field.masked_fill(mask == 0, float("nan"))
    return field


def visualize_conditional_sample_row(
    axes_row,
    context: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    dates: tuple,
    finite_mask: torch.Tensor = None,
):
    """Plot one (k-1, k, k+1 ground truth, prediction) row into ``axes_row``.

    Args:
        axes_row: Sequence of 4 matplotlib Axes to draw into.
        context: (k, k-1) stack, shape [2, H, W], in model range [-1, 1] (as
            stored by the dataset/dumped in samples.pkl).
        prediction: Sampled k+1 field, shape [1, H, W] or [H, W], already in
            data range [0, 1] with land as NaN (as returned by ddpm.sample).
        target: Ground-truth k+1 field, shape [1, H, W] or [H, W], in model
            range [-1, 1].
        dates: (target_date, k_date, km1_date) as returned by
            CARRA2Forecast.get_dates.
        finite_mask: Optional land/water mask used to blank out land pixels.
    """
    target_date, k_date, km1_date = dates
    k, km1 = context[0], context[1]

    cmap = plt.cm.Blues_r
    cmap.set_bad("gray")

    fields = (
        (_prep_field(km1, finite_mask), f"k-1\n{_fmt(km1_date)}"),
        (_prep_field(k, finite_mask), f"k\n{_fmt(k_date)}"),
        (_prep_field(target, finite_mask), f"k+1 (ground truth)\n{_fmt(target_date)}"),
        (_prep_field(prediction, finite_mask, model_range=False), f"k+1 (prediction)\n{_fmt(target_date)}"),
    )

    img = None
    for ax, (field, subtitle) in zip(axes_row, fields):
        img = ax.imshow(field, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto", origin="lower")
        ax.set_title(subtitle, fontsize=9)
        ax.axis("off")

    return img


def visualize_conditional_samples(
    pickle_path: str,
    finite_mask: torch.Tensor = None,
    indices: list[int] = None,
    samples_per_figure: int = 2,
):
    """Load (samples, contexts, targets, times) from a conditional-sampling
    pickle and plot k-1, k, k+1 ground truth, prediction as one row per
    sample, ``samples_per_figure`` rows per figure.

    Args:
        pickle_path: Path to the samples.pkl written by
            helper_functions.download_conditional_samples.
        finite_mask: Optional land/water mask.
        indices: Which samples to plot (default: all of them).
        samples_per_figure: Rows per figure (default 2).
    """
    with open(pickle_path, "rb") as f:
        samples, contexts, targets, times = pickle.load(f)

    if indices is None:
        indices = range(len(samples))
    indices = list(indices)

    for chunk_start in range(0, len(indices), samples_per_figure):
        chunk = indices[chunk_start:chunk_start + samples_per_figure]

        fig, axes = plt.subplots(
            len(chunk), 4, figsize=(16, 4 * len(chunk)), squeeze=False
        )

        img = None
        for row, i in enumerate(chunk):
            img = visualize_conditional_sample_row(
                axes[row], contexts[i], samples[i], targets[i], times[i], finite_mask=finite_mask
            )

        fig.colorbar(img, ax=axes, label="sea ice concentration", shrink=0.7)
        plt.show()


if __name__ == "__main__":
    result_set = "full_conditional_10epochsv2"
    finite_mask = torch.load("utils/finite_WEST.pt")
    visualize_conditional_samples(f"results/{result_set}/samples.pkl", finite_mask=finite_mask)
