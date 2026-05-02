"""Consolidated .env loader for DCQG.

Searches for .env in this priority:
  1. Explicit path argument
  2. DCQG_ROOT/.env (environment variable)
  3. Project root .env (canonical)
  4. Current working directory
"""
import os
from pathlib import Path


def load_env(env_path=None):
    """Load .env file into os.environ. Idempotent (uses setdefault)."""
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    if os.environ.get("DCQG_ROOT"):
        candidates.append(Path(os.environ["DCQG_ROOT"]) / ".env")
    # Project root: dcqg/utils/config.py -> DCQG/. Root `.env` is canonical.
    project_root = Path(__file__).resolve().parent.parent.parent
    candidates.append(project_root / ".env")
    candidates.append(Path.cwd() / ".env")

    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


# Load .env at import time (same behavior as old code)
load_env()


def get_api_config():
    """Return a dict with all API configuration keys."""
    return {
        "SILICONFLOW_API_URL": os.environ.get("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1/chat/completions"),
        "SILICONFLOW_API_KEY": os.environ.get("SILICONFLOW_API_KEY", ""),
        "MODEL": os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "JUDGE_MODEL": os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
        "AIHUBMIX_API_URL": os.environ.get("AIHUBMIX_API_URL", "https://aihubmix.com/v1/chat/completions"),
        "AIHUBMIX_API_KEY": os.environ.get("AIHUBMIX_API_KEY", ""),
        "AIHUBMIX_MODEL": os.environ.get("AIHUBMIX_MODEL", "gpt-4o-mini"),
    }
