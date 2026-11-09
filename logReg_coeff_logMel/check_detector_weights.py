from pathlib import Path
import matplotlib.pyplot as plt
import torch

fakes_kind = "mel"  # change to "mel" or "mel_hann" if needed
checkpoint_path = Path("detectors") / f"mel_logreg_{fakes_kind}.pt"

checkpoint = torch.load(checkpoint_path, map_location="cpu")
weights = checkpoint["weight"].numpy()  # shape: (80,)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

mel_bands = range(len(weights))
axes[0].bar(mel_bands, weights, color="steelblue", alpha=0.8)
axes[0].axshaded_region = None  
if fakes_kind == "rect":
    axes[0].axvspan(43, 52, color="orange", alpha=0.3, label="Swapped Band (43-52)")
    axes[0].legend()
axes[0].set_title(f"Logistic Regression Weights per Mel Band ({fakes_kind})")
axes[0].set_xlabel("Mel Band Index (0 - 79)")
axes[0].set_ylabel("Coefficient Value")
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].hist(weights, bins=15, color="forestgreen", edgecolor="black", alpha=0.8)
axes[1].set_title("Distribution of Coefficient Weights")
axes[1].set_xlabel("Coefficient Value")
axes[1].set_ylabel("Count")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
output_path = f"histograms/weights_plot_{fakes_kind}.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

