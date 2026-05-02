"""LLM response parsing and single-call generation.

- generate_one: single API call with retry, returns (dict|None, raw_text)
- parse_json_response: extract JSON from LLM text response
"""
import json
import time
import urllib.request

from dcqg.utils.config import get_api_config


def parse_json_response(text):
    """Parse JSON from LLM response, with fallback substring extraction."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError, TypeError):
            return None


def generate_one(prompt, temperature=0.1, max_retries=2, model=None):
    """Single API call to SiliconFlow. Returns (json_dict_or_None, raw_text).
    Retries on empty/timeout responses.
    """
    cfg = get_api_config()
    api_url = cfg["SILICONFLOW_API_URL"]
    api_key = cfg["SILICONFLOW_API_KEY"]
    model = model or cfg["MODEL"]

    if not api_key:
        return None, "ERROR: no API key"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Output ONLY a valid JSON object. Follow the question requirements EXACTLY."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 300,
        "stop": ["\n\n"],
    }

    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.0)
                continue
            return None, f"ERROR: {e}"

        if not text:
            if attempt < max_retries:
                time.sleep(0.5)
                continue
            return None, "EMPTY_RESPONSE"

        gen = parse_json_response(text)

        if gen and isinstance(gen, dict) and gen.get("question", "").strip():
            return gen, text

        if attempt < max_retries:
            time.sleep(0.5)
            continue

    return gen, text
