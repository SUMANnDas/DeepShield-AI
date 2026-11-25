"""Binary classification metrics and non-interactive report plots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_binary_metrics(
    y_true: Iterable[int],
    fake_probabilities: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    labels = np.asarray(list(y_true), dtype=np.int64)
    probabilities = np.asarray(list(fake_probabilities), dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.size != probabilities.size or labels.size == 0:
        raise ValueError("y_true and fake_probabilities must be non-empty one-dimensional arrays of equal size")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("CIFAKE labels must be 0 (REAL) or 1 (FAKE)")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be finite values between 0 and 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    predicted = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in matrix.ravel())
    specificity = tn / (tn + fp) if tn + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    auc = float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else float("nan")
    return {
        "threshold": float(threshold),
        "positive_class": "FAKE",
        "class_order": ["REAL", "FAKE"],
        "support": {"REAL": int(np.count_nonzero(labels == 0)), "FAKE": int(np.count_nonzero(labels == 1))},
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision_fake": float(precision_score(labels, predicted, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(labels, predicted, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(labels, predicted, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predicted, average="weighted", zero_division=0)),
        "roc_auc": auc,
        "specificity_real": float(specificity),
        "sensitivity_fake": float(sensitivity),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_layout": "rows=true [REAL,FAKE], columns=predicted [REAL,FAKE]",
    }


def _require_new(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite plot: {path}")
    return path


def save_confusion_matrix(metrics: dict[str, object], output_path: Path, *, title: str) -> Path:
    path = _require_new(output_path)
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"],
             xlabel="Predicted label", ylabel="True label", title=title)
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black")
    figure.colorbar(image, ax=axis)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def save_roc_curve(y_true: Iterable[int], fake_probabilities: Iterable[float], output_path: Path, *, title: str) -> Path:
    labels = np.asarray(list(y_true), dtype=np.int64)
    probabilities = np.asarray(list(fake_probabilities), dtype=np.float64)
    if np.unique(labels).size != 2:
        raise ValueError("ROC curve requires both REAL and FAKE examples")
    fpr, tpr, _ = roc_curve(labels, probabilities, pos_label=1)
    path = _require_new(output_path)
    figure, axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    axis.plot(fpr, tpr, label=f"AUC={roc_auc_score(labels, probabilities):.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=title, xlim=(0, 1), ylim=(0, 1.02))
    axis.legend(loc="lower right")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def save_training_curves(history: list[dict[str, object]], output_dir: Path) -> list[Path]:
    if not history:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    paths: list[Path] = []
    for key, title, ylabel in (
        ("train_loss", "Training loss", "Loss"),
        ("validation_loss", "Validation loss", "Loss"),
        ("validation_accuracy", "Validation accuracy", "Accuracy"),
        ("validation_f1_fake", "Validation F1 (FAKE)", "F1"),
        ("validation_roc_auc", "Validation ROC-AUC", "ROC-AUC"),
    ):
        path = _require_new(output_dir / f"{key}.png")
        figure, axis = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        axis.plot(epochs, [float(row[key]) for row in history], marker="o", markersize=2)
        axis.set(xlabel="Epoch", ylabel=ylabel, title=title)
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)
    return paths
