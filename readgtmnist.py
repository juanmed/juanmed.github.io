
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_gtmnist(path: Path, max_images: int = 9):
    """Return random (image, label, key) tuples from the JSON dataset."""
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    entries = list(data.items())
    if not entries:
        return []

    count = min(max_images, len(entries))
    selected = random.sample(entries, count)
    return [(np.array(entry["image"]), entry["label"], key) for key, entry in selected]


def plot_grid(samples):
    """Display samples in a 3x3 matplotlib grid using the viridis colormap."""
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for ax, sample in zip(axes.flat, samples):
        img, label, key = sample
        ax.imshow(img, cmap="viridis")
        ax.set_title(f"{label} ({key})", fontsize=8)
        ax.axis("off")

    # Hide any unused axes if dataset has fewer than 9 samples.
    for ax in axes.flat[len(samples) :]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


def main():
    dataset_path = Path("gtmnist.json")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Could not find {dataset_path.resolve()}")

    samples = load_gtmnist(dataset_path, max_images=9)
    if not samples:
        raise ValueError("No samples found in the dataset.")

    plot_grid(samples)


if __name__ == "__main__":
    main()
