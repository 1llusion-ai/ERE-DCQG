"""Consolidated API client for OpenAI-compatible endpoints.

Two functions:
  - call_api: simple call using SiliconFlow config from .env
  - call_openai_compatible: explicit config variant for any endpoint
"""
import json
import urllib.request

from dcqg.utils.config import get_api_config


def call_api(prompt, system="", temperature=0.1, max_tokens=150, model=None, timeout=90):
    """Simple API call using SiliconFlow config. Returns text or None on failure."""
    cfg = get_api_config()
    api_key = cfg["SILICONFLOW_API_KEY"]
    api_url = cfg["SILICONFLOW_API_URL"]

    if not api_key:
        return None

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model or cfg["MODEL"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["\n\n"],
    }

    try:
        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def call_openai_compatible(prompt, api_url, api_key, model, max_tokens=300,
                           temperature=0.0, timeout=90, json_mode=True,
                           system=None):
    """API call with explicit endpoint config. Raises on failure."""
    if not api_key:
        raise RuntimeError("API key is empty. Check your .env configuration.")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    else:
        messages.append({
            "role": "system",
            "content": "You are a strict JSON-only evaluator for event-path question generation.",
        })
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]
