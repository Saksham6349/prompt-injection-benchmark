"""Empirical Threshold Calibration & Protocol Locking Script.

Implements Phase 1 Amendment 6:
1. Operates EXCLUSIVELY on data/splits/calibration.csv.
2. Selects:
   - Optimal threshold tau* maximizing F1 on calibration
   - Threshold at <= 1.0% FPR (tau_1pct) and recall
   - Threshold at <= 0.1% FPR (tau_0_1pct) and recall
3. Computes SHA-256 of calibration dataset for protocol locking.
4. Saves locked protocol to results/frozen_threshold.json.
5. Does NOT touch or evaluate the test set.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAL_PATH = PROJECT_ROOT / "data" / "splits" / "calibration.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
FROZEN_THRESHOLD_PATH = PROJECT_ROOT / "results" / "frozen_threshold.json"
CAL_SCORES_PATH = PROJECT_ROOT / "results" / "scores_calibration.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_metrics_at_threshold(y_true: np.ndarray, y_probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (y_probs >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "accuracy": round(float(acc), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def calibrate_protocol() -> dict[str, Any]:
    log.info("Starting threshold calibration protocol on calibration split ONLY...")

    if not CAL_PATH.exists():
        raise FileNotFoundError(f"Calibration split not found at {CAL_PATH}. Run split_dataset.py first.")

    cal_sha256 = compute_file_sha256(CAL_PATH)
    cal_df = pd.read_csv(CAL_PATH)
    y_cal = cal_df["label"].values

    log.info("Loaded %d calibration samples (Benign: %d, Attack: %d)",
             len(cal_df), (y_cal == 0).sum(), (y_cal == 1).sum())

    y_probs = None
    model_source = "tfidf_lr_baseline"

    if CAL_SCORES_PATH.exists():
        try:
            with open(CAL_SCORES_PATH, "r", encoding="utf-8") as f:
                score_data = json.load(f)
                y_probs = np.array(score_data["probabilities"], dtype=float)
                model_source = score_data.get("model_name", "distilbert_retrained")
                log.info("Loaded external calibration scores from %s (%s)", CAL_SCORES_PATH, model_source)
        except Exception as exc:
            log.warning("Could not load calibration scores (%s), falling back to baseline", exc)

    if y_probs is None:
        log.info("Computing calibration scores via TF-IDF + Logistic Regression baseline trained on train.csv...")
        train_df = pd.read_csv(TRAIN_PATH)
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        X_train = tfidf.fit_transform(train_df["text"].tolist())
        clf = LogisticRegression(random_state=42, max_iter=500)
        clf.fit(X_train, train_df["label"].values)

        X_cal = tfidf.transform(cal_df["text"].tolist())
        y_probs = clf.predict_proba(X_cal)[:, 1]

    cal_auroc = roc_auc_score(y_cal, y_probs)
    log.info("Calibration Set AUROC (%s): %.4f", model_source, cal_auroc)

    # Threshold sweep: tau in [0.01, 0.99]
    sweep = []
    best_f1_res = None
    best_f1 = -1.0

    target_fpr_1pct_res = None
    target_fpr_0_1pct_res = None

    threshold_grid = np.linspace(0.01, 0.99, 99)
    for tau in threshold_grid:
        res = find_metrics_at_threshold(y_cal, y_probs, tau)
        sweep.append(res)

        if res["f1"] > best_f1:
            best_f1 = res["f1"]
            best_f1_res = res

        # Target FPR <= 1.0% (0.01)
        if res["fpr"] <= 0.01 and (target_fpr_1pct_res is None or res["recall"] > target_fpr_1pct_res["recall"]):
            target_fpr_1pct_res = res

        # Target FPR <= 0.1% (0.001)
        if res["fpr"] <= 0.001 and (target_fpr_0_1pct_res is None or res["recall"] > target_fpr_0_1pct_res["recall"]):
            target_fpr_0_1pct_res = res

    if target_fpr_1pct_res is None:
        target_fpr_1pct_res = best_f1_res
    if target_fpr_0_1pct_res is None:
        target_fpr_0_1pct_res = target_fpr_1pct_res

    locked_record = {
        "locked": True,
        "protocol_version": "1.0",
        "calibration_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_source": model_source,
        "calibration_file": str(CAL_PATH),
        "calibration_file_sha256": cal_sha256,
        "calibration_sample_count": len(cal_df),
        "calibration_auroc": round(float(cal_auroc), 4),
        "selection_rules": {
            "primary_f1_max": {
                "threshold": best_f1_res["threshold"],
                "f1": best_f1_res["f1"],
                "precision": best_f1_res["precision"],
                "recall": best_f1_res["recall"],
                "fpr": best_f1_res["fpr"],
                "accuracy": best_f1_res["accuracy"],
            },
            "security_target_fpr_1pct": {
                "target_fpr": 0.01,
                "threshold": target_fpr_1pct_res["threshold"],
                "achieved_fpr": target_fpr_1pct_res["fpr"],
                "recall": target_fpr_1pct_res["recall"],
                "f1": target_fpr_1pct_res["f1"],
            },
            "security_target_fpr_0_1pct": {
                "target_fpr": 0.001,
                "threshold": target_fpr_0_1pct_res["threshold"],
                "achieved_fpr": target_fpr_0_1pct_res["fpr"],
                "recall": target_fpr_0_1pct_res["recall"],
                "f1": target_fpr_0_1pct_res["f1"],
            },
        },
        "sweep_summary": sweep,
    }

    FROZEN_THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FROZEN_THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(locked_record, f, indent=2)

    log.info("PROTOCOL LOCKED: Saved frozen threshold to %s", FROZEN_THRESHOLD_PATH)
    log.info("Optimal threshold tau* (F1-max): %.4f (F1=%.4f, Recall=%.4f, FPR=%.4f)",
             best_f1_res["threshold"], best_f1_res["f1"], best_f1_res["recall"], best_f1_res["fpr"])
    log.info("Security threshold at FPR <= 1.0%%: tau=%.4f (Recall=%.4f, FPR=%.4f)",
             target_fpr_1pct_res["threshold"], target_fpr_1pct_res["recall"], target_fpr_1pct_res["fpr"])
    log.info("Security threshold at FPR <= 0.1%%: tau=%.4f (Recall=%.4f, FPR=%.4f)",
             target_fpr_0_1pct_res["threshold"], target_fpr_0_1pct_res["recall"], target_fpr_0_1pct_res["fpr"])

    return locked_record


if __name__ == "__main__":
    calibrate_protocol()
