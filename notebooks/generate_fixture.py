"""Regenerate explicitly synthetic train/test inputs with the locked NumPy RNG."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    """Write independent noisy samples from one stated, dimensionless response curve."""
    root = Path(__file__).resolve().parent / "fixtures"
    root.mkdir(exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(20261114))
    rows = []
    sample_id = 0
    for split, count in [("train", 80), ("test", 120)]:
        values = rng.uniform(-3, 3, count)
        noise = rng.normal(0, 0.35, count)
        for x, error in zip(values, noise, strict=True):
            signal = np.sin(1.5 * x) + 0.2 * x
            rows.append([sample_id, split, format(x, ".17g"), format(signal + error, ".17g")])
            sample_id += 1
    file = root / "synthetic_regression.csv"
    with file.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["sample_id", "split", "x", "target"])
        writer.writerows(rows)
    manifest = {
        "source": "Entirely invented regression samples; no observed subjects, transactions or timestamps.",
        "file": file.name,
        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "generator": "NumPy Generator(PCG64(20261114)); uniform(-3,3) x and independent Normal(0,0.35) noise.",
        "numpy_version": np.__version__,
        "conditional_mean": "sin(1.5*x) + 0.2*x",
        "units": "x and target are dimensionless synthetic quantities.",
        "train_rows": 80,
        "test_rows": 120,
        "split": "Separate draws from the same stated distribution; labels are fixed before fitting.",
    }
    (root / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
