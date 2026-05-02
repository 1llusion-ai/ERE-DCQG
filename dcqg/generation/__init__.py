"""Generation module: prompts, parsing, repair, generator, baselines, faithfulness."""
from dcqg.generation.prompts import (
    FEW_SHOT_EASY,
    FEW_SHOT_MEDIUM,
    FEW_SHOT_HARD,
    DIFFICULTY_DEFINITIONS_HA,
    BANNED_PATTERNS_HARD,
    fmt_ctx,
    prompt_pathqg_easy,
    prompt_pathqg_medium,
    prompt_pathqg_hard,
)
from dcqg.generation.parser import generate_one, parse_json_response
from dcqg.generation.repair import build_repair_prompt, REPAIRABLE_REASONS
from dcqg.generation.generator import generate_with_retry_hardaware
from dcqg.generation.faithfulness import (
    path_faithfulness_judge,
    evaluate_item_with_faithfulness,
    evaluate_file_with_faithfulness,
)
from dcqg.generation.baselines import (
    load_or_create_sample,
    DIFFICULTY_DEFINITIONS,
    ICL_EXAMPLES,
    build_zero_shot_targetqg_prompt,
    build_icl_targetqg_prompt,
    build_self_refine_v2_prompt,
    self_refine_critique_v2_prompt,
    self_refine_revise_v2_prompt,
    build_direct_llm_prompt,
    build_path_only_prompt,
    build_relation_type_prompt,
    generate_baseline,
    generate_self_refine_v2,
    evaluate_method,
)

__all__ = [
    # prompts
    "FEW_SHOT_EASY", "FEW_SHOT_MEDIUM", "FEW_SHOT_HARD",
    "DIFFICULTY_DEFINITIONS_HA", "BANNED_PATTERNS_HARD",
    "fmt_ctx", "prompt_pathqg_easy", "prompt_pathqg_medium", "prompt_pathqg_hard",
    # parser
    "generate_one", "parse_json_response",
    # repair
    "build_repair_prompt", "REPAIRABLE_REASONS",
    # generator
    "generate_with_retry_hardaware",
    # faithfulness
    "path_faithfulness_judge", "evaluate_item_with_faithfulness", "evaluate_file_with_faithfulness",
    # baselines
    "load_or_create_sample", "DIFFICULTY_DEFINITIONS", "ICL_EXAMPLES",
    "build_zero_shot_targetqg_prompt", "build_icl_targetqg_prompt",
    "build_self_refine_v2_prompt", "self_refine_critique_v2_prompt", "self_refine_revise_v2_prompt",
    "build_direct_llm_prompt", "build_path_only_prompt", "build_relation_type_prompt",
    "generate_baseline", "generate_self_refine_v2", "evaluate_method",
]
