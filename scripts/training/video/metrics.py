from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from .storage_budget import VideoStorageBudget


def binary_metrics(y_true, y_prob, *, threshold: float = 0.5):
    labels = np.asarray(y_true)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be equal-length one-dimensional arrays")
    if not len(labels):
        raise ValueError("at least one evaluated video is required")
    if not set(labels.tolist()).issubset({0, 1}):
        raise ValueError("video labels must use 0 for REAL and 1 for MANIPULATED")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in matrix.ravel())
    has_both = len(set(labels.tolist())) == 2
    return {
        "threshold": float(threshold),
        "positive_class": "MANIPULATED",
        "class_order": ["REAL", "MANIPULATED"],
        "support": {"REAL": int(np.count_nonzero(labels == 0)), "MANIPULATED": int(np.count_nonzero(labels == 1))},
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if has_both else None,
        "pr_auc": float(average_precision_score(labels, probabilities)) if has_both else None,
        "specificity_real": float(tn / (tn + fp)) if tn + fp else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_layout": "rows=true [REAL,MANIPULATED], columns=predicted [REAL,MANIPULATED]",
    }


def compute_manipulation_strata(rows: list[dict[str, Any]], *, threshold: float = 0.5) -> dict[str, Any]:
    """Compute per-manipulation-method detection statistics alongside all REAL samples."""
    real_rows = [row for row in rows if int(row["label"]) == 0]
    methods = sorted({
        str(row.get("manipulation_method") or "unknown")
        for row in rows
        if int(row["label"]) == 1
    })
    strata: dict[str, Any] = {}
    for method in methods:
        method_rows = [
            row for row in rows
            if int(row["label"]) == 1 and str(row.get("manipulation_method") or "unknown") == method
        ]
        if not method_rows:
            continue
        probs = [float(row["probability_manipulated"]) for row in method_rows]
        detected = sum(1 for value in probs if value >= threshold)
        entry: dict[str, Any] = {
            "manipulated_count": len(method_rows),
            "detected_manipulated_count": detected,
            "detection_rate": float(detected / len(method_rows)),
            "mean_probability_manipulated": float(np.mean(probs)),
        }
        if real_rows:
            combined = real_rows + method_rows
            entry["vs_all_real"] = binary_metrics(
                [int(item["label"]) for item in combined],
                [float(item["probability_manipulated"]) for item in combined],
                threshold=threshold,
            )
        strata[method] = entry
    return strata


def _budgeted_plot_write(path: Path, figure: plt.Figure, budget: VideoStorageBudget) -> Path:
    import io

    target = budget.validate_destination(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150)
    plt.close(figure)
    payload = buffer.getvalue()
    budget.require_capacity(len(payload), purpose=f"writing video plot {target.name}")
    with target.open("xb") as handle:
        handle.write(payload)
    return target


def save_video_confusion_matrix(
    metrics: dict[str, Any],
    output_path: Path,
    *,
    title: str,
    budget: VideoStorageBudget,
) -> Path:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["REAL", "MANIPULATED"],
        yticklabels=["REAL", "MANIPULATED"],
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black")
    figure.colorbar(image, ax=axis)
    return _budgeted_plot_write(output_path, figure, budget)


def save_video_roc_curve(
    y_true: list[int],
    y_prob: list[float],
    output_path: Path,
    *,
    title: str,
    budget: VideoStorageBudget,
) -> Path | None:
    labels = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if np.unique(labels).size != 2:
        return None
    fpr, tpr, _ = roc_curve(labels, probabilities, pos_label=1)
    figure, axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    axis.plot(fpr, tpr, label=f"AUC={roc_auc_score(labels, probabilities):.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=title, xlim=(0, 1), ylim=(0, 1.02))
    axis.legend(loc="lower right")
    return _budgeted_plot_write(output_path, figure, budget)


def save_video_training_curves(
    history: list[dict[str, Any]],
    output_dir: Path,
    *,
    budget: VideoStorageBudget,
) -> list[Path]:
    if not history:
        return []
    epochs = [int(row["epoch"]) for row in history]
    paths: list[Path] = []
    for key, title, ylabel in (
        ("train_loss", "Video training loss", "Loss"),
        ("validation_loss", "Video validation loss", "Loss"),
        ("validation_accuracy", "Video validation accuracy", "Accuracy"),
        ("validation_f1", "Video validation F1", "F1"),
        ("validation_roc_auc", "Video validation ROC-AUC", "ROC-AUC"),
    ):
        values = [row.get(key) for row in history]
        if any(value is None for value in values):
            continue
        figure, axis = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        axis.plot(epochs, [float(value) for value in values], marker="o", markersize=2)
        axis.set(xlabel="Epoch", ylabel=ylabel, title=title)
        paths.append(_budgeted_plot_write(output_dir / f"{key}.png", figure, budget))
    return paths
