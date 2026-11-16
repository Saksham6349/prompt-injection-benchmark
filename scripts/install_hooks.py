#!/usr/bin/env python3
"""Installs git pre-commit hook for secret scanning.

Configures git core.hooksPath to point to .githooks and copies the hook
to .git/hooks/pre-commit as a fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    source_hook = repo_root / ".githooks" / "pre-commit"
    legacy_hook_dir = repo_root / ".git" / "hooks"
    legacy_hook_dest = legacy_hook_dir / "pre-commit"

    if not source_hook.exists():
        print(f"[ERROR] Source hook not found at {source_hook}", file=sys.stderr)
        return 1

    print("Configuring git hooks...")

    # 1. Set core.hooksPath to .githooks (version-controlled hooks directory)
    try:
        subprocess.check_call(
            ["git", "config", "core.hooksPath", ".githooks"],
            cwd=str(repo_root),
        )
        print("  [SUCCESS] Configured 'git config core.hooksPath .githooks'")
    except Exception as exc:
        print(f"  [WARNING] Could not set core.hooksPath: {exc}")

    # 2. Copy hook directly to .git/hooks/pre-commit as a secondary fallback
    try:
        legacy_hook_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_hook, legacy_hook_dest)
        # Ensure executable permissions on Unix systems
        try:
            current_mode = os.stat(legacy_hook_dest).st_mode
            os.chmod(legacy_hook_dest, current_mode | 0o755)
        except Exception:
            pass
        print(f"  [SUCCESS] Copied fallback hook to {legacy_hook_dest}")
    except Exception as exc:
        print(f"  [WARNING] Could not copy fallback to .git/hooks: {exc}")

    print("\nPre-commit secret scanner successfully activated!")
    print("Every 'git commit' will now automatically verify staged files for secrets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
