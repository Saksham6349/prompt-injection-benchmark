# Security Cleanup & Credential Incident Response Guide

> **STATUS**: Reviewed, not executed.  
> **PURPOSE**: Operational recovery playbook in case API keys, secrets, or personal access tokens are committed or exposed in git history.  
> **RULE**: Ground rules strictly prohibit running history-rewriting commands without explicit owner authorization.

---

## 1. Golden Rule: Revocation & Rotation First

> [!CAUTION]
> **Purging git history does NOT protect an active secret.**  
> If an API key was committed, anyone who cloned, forked, or scraped GitHub caches (including search engines or git mirrors) may already possess it.  
> **Step 0 must always be invalidating the credential at the provider.**

### Immediate Rotation Checklist

1. **Google AI Studio / Google Cloud Console**:
   - Go to [Google AI Studio API Keys](https://aistudio.google.com/app/apikey) or [GCP Credentials](https://console.cloud.google.com/apis/credentials).
   - Locate the leaked key and click **Delete / Revoke**.
   - Generate a brand-new API key.
   - Restrict the new key:
     - Set API restrictions to only allow **Generative Language API**.
     - Set application restrictions (e.g. IP address or HTTP referrer if applicable).
   - Review the Cloud Console **Audit Logs / Metrics** for unexpected billing spikes or unauthorized requests.
2. **OpenAI Platform**:
   - Navigate to [OpenAI API Keys Dashboard](https://platform.openai.com/api-keys).
   - Revoke compromised keys and generate a replacement with minimal scope.
3. **Anthropic Console**:
   - Go to [Anthropic Console Keys](https://console.anthropic.com/settings/keys).
   - Delete the exposed key and generate a fresh key.
4. **Local Artifacts Verification**:
   - Verify that Jupyter notebook outputs (`notebooks/*.ipynb`) do not contain printouts of credentials.
   - Verify presentations (`*.pptx`) or generated documentation do not embed API keys in screenshots or code snippets.

---

## 2. Git History Purge Playbook (git-filter-repo)

> [!WARNING]
> Rewriting git history modifies commit SHAs. Any collaborator with existing clones will have diverging trees and must re-clone after the rewrite.  
> **These commands are reviewed, not executed.**

### Step-by-Step Procedure

#### Step A: Install git-filter-repo
Ensure Python and git are available, then install `git-filter-repo`:
```powershell
pip install git-filter-repo
```

#### Step B: Create a Full Mirror Backup
Before altering git history, create a standalone mirror backup outside the working tree:
```powershell
# In an external directory (e.g. C:\Users\Saksham\Desktop\backup_repo)
git clone --mirror C:\Users\Saksham\Downloads\Capstone C:\Users\Saksham\Desktop\Capstone_Mirror_Backup.git
```

#### Step C: Prepare a Fresh Working Clone
Always run filter-repo on a dedicated fresh clone to avoid dirty working trees or uncommitted stashes:
```powershell
git clone C:\Users\Saksham\Downloads\Capstone C:\Users\Saksham\Downloads\Capstone_Clean_Clone
cd C:\Users\Saksham\Downloads\Capstone_Clean_Clone
```

#### Step D: Create Expressions Replacement File OUTSIDE the Repo
> [!IMPORTANT]
> **Never write the real secret into a file inside the repository.**  
> Store the replacement expressions file in a temporary folder outside the repo root:
```powershell
# Create replacement file in user temp directory
Set-Content -Path "$env:TEMP\git_secrets_replace.txt" -Value "YOUR_OLD_LEAKED_KEY_STRING==>[REVOKED_SECRET_REDACTED]"
```

#### Step E: Execute git-filter-repo
Run the text replacement across all commits and trees:
```powershell
git-filter-repo --replace-text "$env:TEMP\git_secrets_replace.txt" --force
# Clean up temporary replacement file immediately
Remove-Item "$env:TEMP\git_secrets_replace.txt" -Force
```

#### Step F: Re-Add Remote and Force-Push
Because `git-filter-repo` automatically drops the `origin` remote as a safety precaution against accidental pushes, re-add the remote explicitly:
```powershell
git remote add origin https://github.com/Saksham6349/prompt-injection-benchmark.git

# Force-push all rewritten branches and tags
git push origin --force --all
git push origin --force --tags
```

---

## 3. GitHub Platform Cleanup Steps

Once the force-push is complete:

1. **GitHub Cached Views**:
   - Commits reachable via pull requests or cached commit SHAs may persist in GitHub's internal commit cache.
   - Contact [GitHub Support](https://support.github.com/contact) to request a garbage collection purge (`git gc`) of cached views for the repository.
2. **Enable GitHub Secret Scanning & Push Protection**:
   - In repository settings: **Settings → Code security and analysis → Secret scanning**.
   - Enable **Secret scanning** and check **Push protection** to reject any future push containing known token formats.
3. **Notify Collaborators**:
   - Team members must delete local clones and perform a fresh clone:
     ```powershell
     rm -r -fo C:\path\to\old_repo
     git clone https://github.com/Saksham6349/prompt-injection-benchmark.git
     ```

---

## 4. Local Guardrails & Pre-Commit Hook

Our repository includes a built-in pre-commit secret scanner:
- Located at `.githooks/pre-commit` and mirrored to `.git/hooks/pre-commit`.
- Activated via:
  ```powershell
  python scripts/install_hooks.py
  ```
- **How it works**: Before every `git commit`, `scripts/check_secrets.py --staged` inspects added lines for Google (`AQ.*`, `AIza*`), Anthropic (`sk-ant-*`), OpenAI (`sk-*`), GitHub, and AWS keys, plus high-entropy tokens.
- **Bypassing caveat**: Using `git commit --no-verify` will bypass pre-commit hooks. Never use `--no-verify` when staging config or environment files.
- **Optional cross-platform framework**: You may alternatively install the standard [`pre-commit`](https://pre-commit.com/) Python package (`pip install pre-commit && pre-commit install`) if collaborating across diverse OS platforms.
