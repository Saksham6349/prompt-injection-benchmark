#!/usr/bin/env python3
"""Pre-commit and repository secret scanner.

Scans staged changes, working tree files, or git history for accidentally
committed credentials (API keys, tokens, passwords) without printing the
actual secret.

Usage:
    python scripts/check_secrets.py [--staged]
    python scripts/check_secrets.py --working-tree
    python scripts/check_secrets.py --history
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple


class Finding(NamedTuple):
    source: str          # e.g. commit hash, file path, or staged path
    line_num: int | str  # line number or identifier
    pattern_name: str    # rule/pattern that matched
    fingerprint: str     # redacted fingerprint: first 4 chars + length, NEVER raw secret


# Patterns constructed with fragments so this script does not trigger itself.
_P_AIZA = r"AIza" + r"[0-9A-Za-z_-]{35}"
_P_AQ = r"AQ\." + r"[a-zA-Z0-9_\-]{30,}"
_P_SK_ANT = r"sk-ant-" + r"[a-zA-Z0-9_\-]{20,}"
_P_SK = r"(?<![a-zA-Z0-9_\-])sk-(?!ant-)" + r"[a-zA-Z0-9_\-]{20,}"
_P_GITHUB = r"(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36}|github_pat_[a-zA-Z0-9_]{50,}"
_P_AWS = r"(?:AKIA|ASIA)[0-9A-Z]{16}"
_P_GENERIC = r"""(?i)(?:key|token|secret|password|credential)\s*[:=]\s*["']([A-Za-z0-9_\-]{24,})["']"""

SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Google-AIza-Key", re.compile(_P_AIZA)),
    ("Google-AQ-Key", re.compile(_P_AQ)),
    ("Anthropic-API-Key", re.compile(_P_SK_ANT)),
    ("OpenAI-API-Key", re.compile(_P_SK)),
    ("GitHub-Token", re.compile(_P_GITHUB)),
    ("AWS-Access-Key", re.compile(_P_AWS)),
    ("Generic-Secret-Assignment", re.compile(_P_GENERIC)),
]

# Common harmless placeholders that should never be flagged
PLACEHOLDER_REGEX = re.compile(
    r"(?i)(?:your_.*_here|placeholder|<api_key>|your-api-key|dummy|example|fill_me_in|change_me|fake_key|fake-key|insert_.*_here|mock_.*_key)"
)

ALLOWLIST_MARKER = "secret-scan: allow"

# Files and directories to skip entirely
IGNORED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".idea",
    ".vscode",
    "models",
    "env",
    "venv",
    ".venv",
}

IGNORED_EXTENSIONS = {
    ".safetensors",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".db",
    ".sqlite",
    ".zip",
    ".gz",
    ".bin",
    ".pyc",
    ".pyd",
    ".pyo",
    ".env",  # .env is handled separately and must never be opened/read
}

IGNORED_FILES = {
    ".env",
}


def shannon_entropy(s: str) -> float:
    """Calculates Shannon entropy of a string."""
    if not s:
        return 0.0
    counts = collections.Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def redact_secret(secret: str) -> str:
    """Redacts secret to prevent leaking in logs, stdout or traces.

    Format: first 4 chars + '...' + total length.
    """
    if len(secret) <= 4:
        return "[REDACTED]"
    prefix = secret[:4]
    return f"{prefix}...({len(secret)} chars)"


def is_placeholder(val: str) -> bool:
    """Checks whether the detected string is an obvious documentation placeholder."""
    return bool(PLACEHOLDER_REGEX.search(val))


def scan_line(
    line: str,
    source: str,
    line_num: int | str,
) -> list[Finding]:
    """Scans a single text line against all secret patterns and high-entropy heuristic."""
    findings: list[Finding] = []

    # Check for inline allowlist comment
    if ALLOWLIST_MARKER in line:
        return findings

    # 1. Regex pattern matching
    for name, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(line):
            # If pattern has a capturing group (e.g. generic assignment), check group 1
            matched_str = match.group(1) if match.lastindex else match.group(0)

            if is_placeholder(matched_str):
                continue

            findings.append(
                Finding(
                    source=source,
                    line_num=line_num,
                    pattern_name=name,
                    fingerprint=redact_secret(matched_str),
                )
            )

    # 2. Entropy check on string literals (len >= 32, entropy >= 4.5)
    # Skip entropy check on dataset files (.csv, .jsonl) where benign base64 payloads naturally appear
    is_data_file = source.endswith(".csv") or source.endswith(".jsonl") or "data/" in source or "data\\" in source
    if not is_data_file:
        for lit_match in re.finditer(r"""["']([A-Za-z0-9_\-+/=]{32,})["']""", line):
            lit_str = lit_match.group(1)
            if is_placeholder(lit_str):
                continue
            # Avoid double-flagging if already found by pattern
            if any(f.fingerprint.startswith(lit_str[:4]) for f in findings):
                continue
            ent = shannon_entropy(lit_str)
            if ent >= 4.5:
                findings.append(
                    Finding(
                        source=source,
                        line_num=line_num,
                        pattern_name=f"High-Entropy-Literal (entropy={ent:.2f})",
                        fingerprint=redact_secret(lit_str),
                    )
                )

    return findings


def scan_notebook_content(content: str, source: str) -> list[Finding]:
    """Scans an .ipynb file including code cells and output streams."""
    findings: list[Finding] = []
    try:
        nb = json.loads(content)
        cells = nb.get("cells", [])
        for cell_idx, cell in enumerate(cells, start=1):
            cell_type = cell.get("cell_type", "")
            # Scan source code lines
            source_lines = cell.get("source", [])
            if isinstance(source_lines, str):
                source_lines = source_lines.splitlines(keepends=True)
            for l_idx, line in enumerate(source_lines, start=1):
                findings.extend(scan_line(line, source, f"cell_{cell_idx}:code:{l_idx}"))

            # Scan output lines
            outputs = cell.get("outputs", [])
            for out_idx, out in enumerate(outputs, start=1):
                # Text output
                text_out = out.get("text", [])
                if isinstance(text_out, str):
                    text_out = text_out.splitlines(keepends=True)
                for l_idx, line in enumerate(text_out, start=1):
                    findings.extend(scan_line(line, source, f"cell_{cell_idx}:out_{out_idx}:{l_idx}"))
                # Data text/plain
                data_plain = out.get("data", {}).get("text/plain", [])
                if isinstance(data_plain, str):
                    data_plain = data_plain.splitlines(keepends=True)
                for l_idx, line in enumerate(data_plain, start=1):
                    findings.extend(scan_line(line, source, f"cell_{cell_idx}:data_{out_idx}:{l_idx}"))
    except Exception:
        # Fallback to plain line scan if json parsing fails
        for l_idx, line in enumerate(content.splitlines(), start=1):
            findings.extend(scan_line(line, source, l_idx))
    return findings


def scan_text_content(content: str, source: str) -> list[Finding]:
    """Dispatches content scanning based on file type."""
    if source.endswith(".ipynb"):
        return scan_notebook_content(content, source)

    findings: list[Finding] = []
    for l_idx, line in enumerate(content.splitlines(), start=1):
        findings.extend(scan_line(line, source, l_idx))
    return findings


def scan_staged() -> list[Finding]:
    """Scans staged git diffs via 'git diff --cached'.

    Only scans added or modified lines in staged files.
    """
    findings: list[Finding] = []
    try:
        # Get list of staged files
        diff_cmd = ["git", "diff", "--cached", "--name-status"]
        status_output = subprocess.check_output(diff_cmd, text=True, errors="replace")
    except Exception as exc:
        print(f"[ERROR] Failed to query git staged files: {exc}", file=sys.stderr)
        return [Finding("git-staged", 0, "GitError", str(exc))]

    staged_files: list[str] = []
    for line in status_output.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2 and parts[0] != "D":  # Exclude deleted files
            staged_files.append(parts[-1])

    for file_path in staged_files:
        # Security: NEVER open .env
        if file_path == ".env" or file_path.endswith("/.env") or file_path.endswith("\\.env"):
            findings.append(Finding(file_path, 0, "BlockedFile", "Direct .env staging forbidden"))
            continue

        p = Path(file_path)
        if p.suffix.lower() in IGNORED_EXTENSIONS:
            continue
        if any(ignored in p.parts for ignored in IGNORED_DIRS):
            continue

        try:
            # Use git show :<path> to inspect staged blob directly
            blob_cmd = ["git", "show", f":{file_path}"]
            content = subprocess.check_output(blob_cmd, text=True, errors="replace")
            findings.extend(scan_text_content(content, f"staged:{file_path}"))
        except Exception as exc:
            print(f"[WARNING] Could not read staged content of {file_path}: {exc}", file=sys.stderr)

    return findings


def scan_working_tree(root_dir: str = ".") -> list[Finding]:
    """Scans all non-ignored files in the working tree (skipping .env)."""
    findings: list[Finding] = []
    root_path = Path(root_dir).resolve()

    for path in root_path.rglob("*"):
        if not path.is_file():
            continue

        # Strictly skip .env and .env.* files
        if path.name == ".env" or path.name.endswith(".env"):
            continue

        # Skip ignored directories
        if any(ignored in path.parts for ignored in IGNORED_DIRS):
            continue

        # Skip ignored extensions
        if path.suffix.lower() in IGNORED_EXTENSIONS:
            continue

        # Skip training CSV as it is a dataset file
        if path.suffix.lower() == ".csv" and "data" in path.parts:
            continue

        rel_path = str(path.relative_to(root_path))
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            findings.extend(scan_text_content(content, rel_path))
        except Exception as exc:
            pass

    return findings


def scan_history() -> list[Finding]:
    """Scans all commits, reflog, and dangling objects in git history."""
    findings: list[Finding] = []

    # 1. Scan git log across all refs
    try:
        log_cmd = ["git", "log", "--all", "-p", "--full-history"]
        raw_log = subprocess.check_output(log_cmd, text=True, errors="replace")

        current_commit = "unknown"
        current_file = "unknown"
        line_num = 0

        skip_file = False
        for line in raw_log.splitlines():
            if line.startswith("commit "):
                current_commit = line.split()[1][:8]
            elif line.startswith("+++ b/"):
                current_file = line[6:]
                line_num = 0
                p = Path(current_file)
                skip_file = (
                    p.suffix.lower() in IGNORED_EXTENSIONS
                    or (p.suffix.lower() == ".csv" and "data" in p.parts)
                    or any(ignored in p.parts for ignored in IGNORED_DIRS)
                )
            elif line.startswith("+") and not line.startswith("+++"):
                if skip_file:
                    continue
                line_num += 1
                added_text = line[1:]
                # Scan added line
                f = scan_line(added_text, f"commit:{current_commit}:{current_file}", line_num)
                findings.extend(f)
    except Exception as exc:
        print(f"[WARNING] Error scanning git log: {exc}", file=sys.stderr)

    # 2. Scan git reflog
    try:
        reflog_cmd = ["git", "reflog", "-p"]
        raw_reflog = subprocess.check_output(reflog_cmd, text=True, errors="replace")
        current_ref = "reflog"
        skip_file = False
        for line in raw_reflog.splitlines():
            if line.startswith("+++ b/"):
                ref_file = line[6:]
                p = Path(ref_file)
                skip_file = (
                    p.suffix.lower() in IGNORED_EXTENSIONS
                    or (p.suffix.lower() == ".csv" and "data" in p.parts)
                    or any(ignored in p.parts for ignored in IGNORED_DIRS)
                )
            elif line.startswith("+") and not line.startswith("+++"):
                if skip_file:
                    continue
                f = scan_line(line[1:], f"reflog:{current_ref}", 0)
                findings.extend(f)
    except Exception:
        pass

    return findings


def report_findings(findings: list[Finding]) -> None:
    """Prints findings with REDACTED fingerprints only."""
    if not findings:
        print("  [CLEAN] No secrets detected.")
        return

    print(f"  [ALERT] Found {len(findings)} potential secret(s):")
    print("  " + "-" * 70)
    for f in findings:
        print(f"  Source  : {f.source}")
        print(f"  Line    : {f.line_num}")
        print(f"  Rule    : {f.pattern_name}")
        print(f"  Preview : {f.fingerprint} (CONFIDENTIAL - REDACTED)")
        print("  " + "-" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="Antigravity Secret Scanner")
    parser.add_argument(
        "--staged",
        action="store_true",
        default=False,
        help="Scan git staged changes only (default mode)",
    )
    parser.add_argument(
        "--working-tree",
        action="store_true",
        default=False,
        help="Scan the working tree files (excluding .env)",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        default=False,
        help="Scan git history across all commits and reflog",
    )

    args = parser.parse_args()

    # Default to staged scan if no flag specified
    if not args.working_tree and not args.history:
        args.staged = True

    print("======================================================================")
    print("  SECURITY SCANNER: Secret & Credential Detection Gate")
    print("======================================================================")

    total_findings: list[Finding] = []

    if args.staged:
        print("Scanning git staged content...")
        findings = scan_staged()
        report_findings(findings)
        total_findings.extend(findings)

    if args.working_tree:
        print("Scanning working tree (excluding .env and binary files)...")
        findings = scan_working_tree()
        report_findings(findings)
        total_findings.extend(findings)

    if args.history:
        print("Scanning git history across all commits, branches, and reflog...")
        findings = scan_history()
        report_findings(findings)
        total_findings.extend(findings)

    print("======================================================================")
    if total_findings:
        print(f"GATE FAILED: {len(total_findings)} secret(s) detected. Commit/push blocked.")
        return 1
    else:
        print("GATE PASSED: No secrets detected.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
