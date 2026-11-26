import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

def binary_metrics(y_true,y_prob):
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
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc_score(labels, probabilities)) if len(set(labels.tolist())) == 2 else None,
        'confusion_matrix': confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
