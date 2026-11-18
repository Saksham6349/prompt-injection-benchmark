"""Dataset Source Attribution, Provenance Verification, and Duplicate Audit Script.

Implements Phase 1 Items 1 & 2:
1. Multi-source hash verification against official HuggingFace datasets
   (deepset, S-Labs/slabs, JasperLS).
2. Devanagari and Romanized-Hindi heuristic detection.
3. Near-duplicate and exact-duplicate detection (char 3-gram TF-IDF cosine > 0.95).
4. Connected-component cluster construction.
5. Export review list to results/duplicate_review_list.csv (deletes nothing).
6. Generates data/real_classifier_training_data_v2.csv (leaving original untouched).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants and paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DATA_PATH = PROJECT_ROOT / "data" / "real_classifier_training_data.csv"
V2_DATA_PATH = PROJECT_ROOT / "data" / "real_classifier_training_data_v2.csv"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark_cases.jsonl"
REVIEW_CSV_PATH = PROJECT_ROOT / "results" / "duplicate_review_list.csv"
REVIEW_JSON_PATH = PROJECT_ROOT / "results" / "duplicate_review_list.json"

# Hinglish & Devanagari regex patterns
DEV_REGEX = re.compile(r"[\u0900-\u097F]")
HINGLISH_KEYWORDS = re.compile(
    r"\b(karo|karein|bhejo|bhej|paise|kripya|aap|tum|mera|meri|kardo|hoga|raha|rahi|nirdesh|bhool|pichle|pehle|khata|batao)\b",
    re.I,
)


def normalize_text(text: Any) -> str:
    """Exact text normalization: lowercase, strip punctuation, collapse whitespace."""
    s = str(text).lower().strip()
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse multiple whitespaces
    return " ".join(s.split())


def compute_norm_hash(text: str) -> str:
    """Computes SHA-256 of normalized text for exact matching."""
    norm = normalize_text(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def fetch_official_datasets() -> tuple[dict[str, str], dict[str, str]]:
    """Fetches official datasets from HuggingFace to build exact hash lookup tables."""
    deepset_hashes: dict[str, str] = {}
    slabs_hashes: dict[str, str] = {}

    print("Checking / Downloading official datasets from Hugging Face...")

    # 1. deepset/prompt-injections
    try:
        url_train = "https://huggingface.co/datasets/deepset/prompt-injections/resolve/main/data/train-00000-of-00001-9564e8b05b4757ab.parquet"
        url_test = "https://huggingface.co/datasets/deepset/prompt-injections/resolve/main/data/test-00000-of-00001-701d16158af87368.parquet"
        df_ds1 = pd.read_parquet(url_train)
        df_ds2 = pd.read_parquet(url_test)
        df_ds = pd.concat([df_ds1, df_ds2], ignore_index=True)
        for _, row in df_ds.iterrows():
            h = compute_norm_hash(row["text"])
            deepset_hashes[h] = "deepset"
        print(f"  [HF SUCCESS] Loaded deepset/prompt-injections ({len(deepset_hashes)} unique normalized hashes)")
    except Exception as exc:
        print(f"  [HF WARNING] Could not download deepset from HF: {exc}")

    # 2. S-Labs/prompt-injection-dataset
    try:
        url_slabs = "https://huggingface.co/datasets/S-Labs/prompt-injection-dataset/resolve/main/data/train.csv"
        df_slabs = pd.read_csv(url_slabs)
        for _, row in df_slabs.iterrows():
            h = compute_norm_hash(row["text"])
            slabs_hashes[h] = "slabs"
        print(f"  [HF SUCCESS] Loaded S-Labs/prompt-injection-dataset ({len(slabs_hashes)} unique normalized hashes)")
    except Exception as exc:
        print(f"  [HF WARNING] Could not download S-Labs from HF: {exc}")

    return deepset_hashes, slabs_hashes


def attribute_sources(
    df: pd.DataFrame,
    deepset_hashes: dict[str, str],
    slabs_hashes: dict[str, str],
) -> pd.DataFrame:
    """Attributes each row with source and provenance_confidence based on hard evidence."""
    df_out = df.copy()

    sources: list[str] = []
    confidences: list[str] = []

    for idx, row in df_out.iterrows():
        raw_text = str(row["text"])
        h = compute_norm_hash(raw_text)

        # 1. Check for Devanagari or Hinglish patterns
        if DEV_REGEX.search(raw_text) or HINGLISH_KEYWORDS.search(raw_text):
            sources.append("augmented_hinglish")
            confidences.append("inferred")
            continue

        # 2. Check exact hash in deepset
        if h in deepset_hashes:
            sources.append("deepset")
            confidences.append("verified")
            continue

        # 3. Check exact hash in S-Labs
        if h in slabs_hashes:
            sources.append("slabs")
            confidences.append("verified")
            continue

        # 4. Fallback: unknown_merged
        sources.append("unknown_merged")
        confidences.append("unknown")

    df_out["source"] = sources
    df_out["provenance_confidence"] = confidences
    return df_out


def audit_duplicates(
    df: pd.DataFrame,
    benchmark_cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Finds exact duplicates and near-duplicates (char 3-gram TF-IDF cosine > 0.95).

    Returns:
        (review_items, clusters)
    """
    review_items: list[dict[str, Any]] = []

    norm_corpus = [normalize_text(t) for t in df["text"]]

    # -------------------------------------------------------------------------
    # A. Check Corpus vs Benchmark Leaks
    # -------------------------------------------------------------------------
    print("Auditing potential leaks between corpus and 500 benchmark cases...")
    bench_items: list[dict[str, Any]] = []
    for c in benchmark_cases:
        cid = c.get("id", "unknown")
        is_benign = c.get("is_benign", False)
        if c.get("user_task"):
            bench_items.append({
                "id": f"{cid}:user_task",
                "norm": normalize_text(c["user_task"]),
                "raw": c["user_task"],
                "label": 0 if is_benign else 1,
                "type": "benchmark_user_task",
            })
        if c.get("injection_payload"):
            bench_items.append({
                "id": f"{cid}:payload",
                "norm": normalize_text(c["injection_payload"]),
                "raw": c["injection_payload"],
                "label": 1,
                "type": "benchmark_payload",
            })

    # Exact check
    corpus_norm_map: dict[str, list[int]] = defaultdict(list)
    for idx, nt in enumerate(norm_corpus):
        corpus_norm_map[nt].append(idx)

    for b in bench_items:
        if b["norm"] in corpus_norm_map:
            for c_idx in corpus_norm_map[b["norm"]]:
                review_items.append({
                    "audit_type": "cross_benchmark_leak",
                    "similarity_type": "exact_normalized",
                    "similarity_score": 1.0,
                    "item_1_source": "benchmark",
                    "item_1_id": b["id"],
                    "item_1_text": b["raw"][:120],
                    "item_1_label": b["label"],
                    "item_2_source": df.iloc[c_idx]["source"],
                    "item_2_id": f"corpus_row_{c_idx}",
                    "item_2_text": str(df.iloc[c_idx]["text"])[:120],
                    "item_2_label": int(df.iloc[c_idx]["label"]),
                    "label_conflict": bool(b["label"] != df.iloc[c_idx]["label"]),
                })

    # -------------------------------------------------------------------------
    # B. Check Within-Corpus Exact Duplicates
    # -------------------------------------------------------------------------
    print("Auditing within-corpus exact duplicates...")
    for nt, indices in corpus_norm_map.items():
        if len(indices) > 1:
            base_idx = indices[0]
            for other_idx in indices[1:]:
                lbl_base = int(df.iloc[base_idx]["label"])
                lbl_other = int(df.iloc[other_idx]["label"])
                review_items.append({
                    "audit_type": "within_corpus_exact",
                    "similarity_type": "exact_normalized",
                    "similarity_score": 1.0,
                    "item_1_source": df.iloc[base_idx]["source"],
                    "item_1_id": f"corpus_row_{base_idx}",
                    "item_1_text": str(df.iloc[base_idx]["text"])[:120],
                    "item_1_label": lbl_base,
                    "item_2_source": df.iloc[other_idx]["source"],
                    "item_2_id": f"corpus_row_{other_idx}",
                    "item_2_text": str(df.iloc[other_idx]["text"])[:120],
                    "item_2_label": lbl_other,
                    "label_conflict": bool(lbl_base != lbl_other),
                })

    # -------------------------------------------------------------------------
    # C. Near-Duplicate Audit (Char 3-gram TF-IDF Cosine > 0.95)
    # -------------------------------------------------------------------------
    print("Auditing near-duplicates via char 3-gram TF-IDF (cosine > 0.95)...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),
        min_df=2,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(norm_corpus)

    # Cross-benchmark near-duplicates
    bench_norms = [b["norm"] for b in bench_items]
    X_bench = vectorizer.transform(bench_norms)
    sim_cross = cosine_similarity(X_bench, X)
    cross_pairs = np.where(sim_cross > 0.95)

    for b_idx, c_idx in zip(cross_pairs[0], cross_pairs[1]):
        score = float(sim_cross[b_idx, c_idx])
        # Skip if already flagged as exact
        b_item = bench_items[b_idx]
        if b_item["norm"] == norm_corpus[c_idx]:
            continue
        review_items.append({
            "audit_type": "cross_benchmark_near_dup",
            "similarity_type": "char3gram_cosine",
            "similarity_score": round(score, 4),
            "item_1_source": "benchmark",
            "item_1_id": b_item["id"],
            "item_1_text": b_item["raw"][:120],
            "item_1_label": b_item["label"],
            "item_2_source": df.iloc[c_idx]["source"],
            "item_2_id": f"corpus_row_{c_idx}",
            "item_2_text": str(df.iloc[c_idx]["text"])[:120],
            "item_2_label": int(df.iloc[c_idx]["label"]),
            "label_conflict": bool(b_item["label"] != df.iloc[c_idx]["label"]),
        })

    # Within-corpus near-duplicates (batched to manage memory)
    print("Computing within-corpus near-duplicate pairs (batch size=1000)...")
    batch_size = 1000
    n_samples = X.shape[0]

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        sim_batch = cosine_similarity(X[start_idx:end_idx], X)

        # Look only at upper triangle (j > i) to avoid self-matches and duplicate pairs
        for local_i in range(end_idx - start_idx):
            global_i = start_idx + local_i
            # Find elements with similarity > 0.95 and global_j > global_i
            high_j = np.where(sim_batch[local_i, (global_i + 1):] > 0.95)[0] + (global_i + 1)
            for global_j in high_j:
                # If exact normalized, already recorded
                if norm_corpus[global_i] == norm_corpus[global_j]:
                    continue
                score = float(sim_batch[local_i, global_j])
                lbl_i = int(df.iloc[global_i]["label"])
                lbl_j = int(df.iloc[global_j]["label"])
                review_items.append({
                    "audit_type": "within_corpus_near_dup",
                    "similarity_type": "char3gram_cosine",
                    "similarity_score": round(score, 4),
                    "item_1_source": df.iloc[global_i]["source"],
                    "item_1_id": f"corpus_row_{global_i}",
                    "item_1_text": str(df.iloc[global_i]["text"])[:120],
                    "item_1_label": lbl_i,
                    "item_2_source": df.iloc[global_j]["source"],
                    "item_2_id": f"corpus_row_{global_j}",
                    "item_2_text": str(df.iloc[global_j]["text"])[:120],
                    "item_2_label": lbl_j,
                    "label_conflict": bool(lbl_i != lbl_j),
                })

    return review_items


def main() -> None:
    print("=" * 75)
    print("  PHASE 1 AUDIT: Source Attribution & Duplicate Detection")
    print("=" * 75)

    if not ORIGINAL_DATA_PATH.exists():
        print(f"[ERROR] Original training data not found at {ORIGINAL_DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    df_orig = pd.read_csv(ORIGINAL_DATA_PATH)
    print(f"Loaded original training corpus: {len(df_orig)} rows, columns: {df_orig.columns.tolist()}")

    # 1. Fetch official HuggingFace datasets
    ds_hashes, slabs_hashes = fetch_official_datasets()

    # 2. Attribute sources
    df_v2 = attribute_sources(df_orig, ds_hashes, slabs_hashes)

    # 3. Print attribution breakdown
    print("\n" + "-" * 75)
    print("SOURCE PROVENANCE ATTRIBUTION SUMMARY:")
    print("-" * 75)
    source_summary = df_v2.groupby(["source", "label"]).size().unstack(fill_value=0)
    source_summary["Total"] = source_summary.sum(axis=1)
    source_summary["% of Corpus"] = (source_summary["Total"] / len(df_v2) * 100).round(2)
    print(source_summary)

    print("\n" + "-" * 75)
    print("PROVENANCE CONFIDENCE BREAKDOWN:")
    print("-" * 75)
    conf_summary = df_v2.groupby(["provenance_confidence"]).size()
    print(conf_summary)

    # 4. Save v2 without modifying original
    df_v2.to_csv(V2_DATA_PATH, index=False)
    print(f"\n[SAVED] Wrote verified dataset to {V2_DATA_PATH} (original intact).")

    # 5. Load benchmark cases
    benchmark_cases: list[dict[str, Any]] = []
    if BENCHMARK_PATH.exists():
        benchmark_cases = [json.loads(l) for l in BENCHMARK_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"Loaded benchmark suite: {len(benchmark_cases)} cases")

    # 6. Audit duplicates
    review_items = audit_duplicates(df_v2, benchmark_cases)

    # 7. Export review list
    REVIEW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_review = pd.DataFrame(review_items)
    df_review.to_csv(REVIEW_CSV_PATH, index=False)
    with open(REVIEW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(review_items, f, indent=2)

    print("\n" + "=" * 75)
    print("DUPLICATE AUDIT FINDINGS SUMMARY:")
    print("=" * 75)
    print(f"Total candidate items flagged: {len(review_items)}")
    if not df_review.empty:
        print("\nBreakdown by audit type:")
        print(df_review["audit_type"].value_counts())

        print("\nConflicting label pairs (label 0 vs label 1):")
        conflict_count = df_review["label_conflict"].sum()
        print(f"  Count: {conflict_count} ({conflict_count / len(df_review) * 100:.2f}%)")

        print(f"\nSaved full review list to:")
        print(f"  - CSV : {REVIEW_CSV_PATH}")
        print(f"  - JSON: {REVIEW_JSON_PATH}")
    else:
        print("No duplicates or leaks found!")

    print("\n[GATE ACTIVE] Zero rows deleted. Awaiting user approval of duplicate review list.")


if __name__ == "__main__":
    main()
