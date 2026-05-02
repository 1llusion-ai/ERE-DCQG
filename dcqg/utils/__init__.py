"""DCQG utility modules."""
from dcqg.utils.config import load_env, get_api_config
from dcqg.utils.api_client import call_api, call_openai_compatible
from dcqg.utils.text import simple_stem, normalize, fuzzy_match, text_similarity, detect_loop
from dcqg.utils.jsonl import read_jsonl, write_jsonl
