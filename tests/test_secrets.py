"""Tests for the secret scanner (scripts/check_secrets.py).

Fake credentials are constructed at runtime dynamically so that test files
never contain static key literals.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_secrets import (
    ALLOWLIST_MARKER,
    Finding,
    is_placeholder,
    redact_secret,
    scan_line,
    scan_notebook_content,
    scan_text_content,
    shannon_entropy,
)


def _make_key(prefix: str, pad_char: str, total_len: int) -> str:
    """Dynamically builds a fake key at runtime to avoid static string matches."""
    needed = total_len - len(prefix)
    return prefix + (pad_char * needed)


def test_google_aiza_key_detected() -> None:
    fake_key = _make_key("AIza", "X", 39)
    line = f"export GOOGLE_KEY={fake_key}"
    findings = scan_line(line, "test_file.py", 10)
    assert len(findings) == 1
    assert findings[0].pattern_name == "Google-AIza-Key"
    assert fake_key not in findings[0].fingerprint
    assert findings[0].fingerprint.startswith("AIza")


def test_google_aq_key_detected() -> None:
    fake_key = _make_key("AQ.", "Z", 48)
    line = f"GEMINI_KEY = '{fake_key}'"
    findings = scan_line(line, "test_file.py", 15)
    assert len(findings) >= 1
    assert any(f.pattern_name == "Google-AQ-Key" for f in findings)
    for f in findings:
        assert fake_key not in f.fingerprint


def test_anthropic_key_detected() -> None:
    fake_key = _make_key("sk-ant-", "k", 40)
    line = f"ANTHROPIC_API_KEY={fake_key}"
    findings = scan_line(line, "config.py", 22)
    assert len(findings) >= 1
    assert any(f.pattern_name == "Anthropic-API-Key" for f in findings)
    assert fake_key not in findings[0].fingerprint


def test_openai_key_detected() -> None:
    fake_key = _make_key("sk-", "o", 51)
    line = f"OPENAI_KEY = '{fake_key}'"
    findings = scan_line(line, "agent.py", 5)
    assert len(findings) >= 1
    assert any(f.pattern_name == "OpenAI-API-Key" for f in findings)
    for f in findings:
        assert fake_key not in f.fingerprint


def test_github_token_detected() -> None:
    fake_ghp = _make_key("ghp_", "1", 40)
    line = f"GH_TOKEN={fake_ghp}"
    findings = scan_line(line, "deploy.sh", 1)
    assert len(findings) == 1
    assert findings[0].pattern_name == "GitHub-Token"
    assert fake_ghp not in findings[0].fingerprint


def test_aws_key_detected() -> None:
    fake_aws = _make_key("AKIA", "9", 20)
    line = f"AWS_KEY = '{fake_aws}'"
    findings = scan_line(line, "aws.py", 3)
    assert len(findings) == 1
    assert findings[0].pattern_name == "AWS-Access-Key"
    assert fake_aws not in findings[0].fingerprint


def test_generic_assignment_detected() -> None:
    fake_val = "v" * 36
    line = f'my_secret_token = "{fake_val}"'
    findings = scan_line(line, "settings.py", 8)
    assert len(findings) >= 1
    assert any("Generic-Secret" in f.pattern_name for f in findings)
    for f in findings:
        assert fake_val not in f.fingerprint


def test_placeholders_ignored() -> None:
    placeholders = [
        "your_api_key_here",
        "your_google_api_key_here",
        "placeholder",
        "<API_KEY>",
        "dummy",
        "example",
        "insert_token_here",
    ]
    for ph in placeholders:
        line = f"GOOGLE_API_KEY={ph}"
        findings = scan_line(line, "test.env", 1)
        assert len(findings) == 0, f"Failed for placeholder: {ph}"


def test_allowlist_comment_respected() -> None:
    fake_key = _make_key("AIza", "Y", 39)
    line = f"KEY = '{fake_key}'  # {ALLOWLIST_MARKER}"
    findings = scan_line(line, "test.py", 1)
    assert len(findings) == 0


def test_redaction_guarantee() -> None:
    raw_secret = _make_key("AQ.", "A", 55)
    redacted = redact_secret(raw_secret)
    assert raw_secret not in redacted
    assert redacted.startswith("AQ.")
    assert f"({len(raw_secret)} chars)" in redacted

    finding = Finding("test.py", 1, "Rule", redacted)
    assert raw_secret not in str(finding)
    assert raw_secret not in finding.fingerprint


def test_notebook_cell_scanning() -> None:
    fake_key = _make_key("sk-ant-", "m", 35)
    nb_dict = {
        "cells": [
            {
                "cell_type": "code",
                "source": [f"import os\n", f"key = '{fake_key}'\n"],
                "outputs": [],
            }
        ]
    }
    content = json.dumps(nb_dict)
    findings = scan_notebook_content(content, "test_nb.ipynb")
    assert len(findings) >= 1
    assert any(f.pattern_name == "Anthropic-API-Key" for f in findings)
    for f in findings:
        assert fake_key not in f.fingerprint


def test_clean_content_returns_empty() -> None:
    clean_lines = [
        "def hello_world():",
        "    return 'Welcome to Antigravity Capstone'",
        "# Just a comment with no tokens",
        "model_name = 'distilbert-base-uncased'",
    ]
    findings = scan_text_content("\n".join(clean_lines), "clean.py")
    assert len(findings) == 0
