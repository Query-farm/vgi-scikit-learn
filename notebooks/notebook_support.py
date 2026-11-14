"""Chart presentation only; model fitting and evaluation remain visible SQL."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import seaborn as sns


def theme():
    """Apply readable plotting defaults."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 150,
            "text.parse_math": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def finish(fig, name, title, subtitle):
    """Add synthetic provenance, save a review image and display."""
    fig.suptitle(title, x=0.1, y=0.98, ha="left", fontsize=17, fontweight="bold", color="#243e32")
    fig.text(0.1, 0.92, subtitle, ha="left", fontsize=11, color="#526b5b")
    fig.text(
        0.1,
        0.025,
        "Source: seeded synthetic regression · 80 training / 120 held-out samples · no observed subjects\n"
        "vgi-scikit-learn / scikit-learn 1.9.0 through Haybarn / VGI · dimensionless data",
        fontsize=9,
        color="#526b5b",
    )
    fig.tight_layout(rect=(0.02, 0.12, 0.98, 0.82))
    folder = Path("validation/charts")
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", bbox_inches="tight")
    plt.show()


class NotebookRegistry:
    """Keep the worker's named model writes inside a disposable directory."""

    def __init__(self):
        """Override only the supported model-registry environment setting."""
        self.previous = os.environ.get("SKLEARN_MODELS_DIR")
        self.directory = TemporaryDirectory(prefix="vgi-sklearn-notebook-")
        os.environ["SKLEARN_MODELS_DIR"] = self.directory.name

    def close(self):
        """Restore the environment and remove artifacts after the connection closes."""
        if self.previous is None:
            os.environ.pop("SKLEARN_MODELS_DIR", None)
        else:
            os.environ["SKLEARN_MODELS_DIR"] = self.previous
        self.directory.cleanup()
