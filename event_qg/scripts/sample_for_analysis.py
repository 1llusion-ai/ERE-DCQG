"""Sample 20 Easy/Medium/Hard for difficulty failure analysis."""
import json
import random
random.seed(42)

path = 'F:/Projects/DCQG/event_qg/outputs/compare_PathQG_evaluated_retry_v2.jsonl'
with open(path, encoding='utf-8') as f:
    items = [json.loads(l) for l in f]

by_diff = {'Easy': [], 'Medium': [], 'Hard': []}
for r in items:
    by_diff[r['difficulty']].append(r)

# Sample 20 per level
sampled = {}
for d in ['Easy', 'Medium', 'Hard']:
    sampled[d] = random.sample(by_diff[d], min(20, len(by_diff[d])))

# Print for annotation
sep = '=' * 80
for d in ['Easy', 'Medium', 'Hard']:
    print(f'\n{sep}')
    print(f'{d} (n={len(sampled[d])})')
    print(sep)
    for i, r in enumerate(sampled[d]):
        q = r['generated_question']
        gold = r['gold_answer_trigger']
        events = [e['trigger'] for e in r.get('events', [])]
        path_str = ' -> '.join(events)
        sents = r.get('supporting_sentences', [])
        target_sents = []
        if sents and isinstance(sents[0], list):
            target_sents = [(s[0], s[1]) for s in sents if gold.lower() in s[1].lower()]

        print(f'\n--- Item {i+1} (id={r["item_id"]}) ---')
        print(f'Question: {q}')
        print(f'Gold trigger: "{gold}"')
        print(f'Path: {path_str}')
        print(f'Solver answer: {r.get("solver_answer", "?")}')
        print(f'solver_correct={r["judge_solver_correct"]} target_hit={r["target_event_hit"]}')
        print(f'#sentences: {len(sents)}')
        print(f'Last sentence (idx={sents[-1][0] if sents and isinstance(sents[-1], list) else "?"}):')
        if sents and isinstance(sents[-1], list):
            print(f'  {sents[-1][1][:300]}')
        # Print all sentences briefly
        print(f'All sentences:')
        for s in sents:
            if isinstance(s, list):
                print(f'  [S{s[0]}] {s[1][:200]}')
            else:
                print(f'  {s[:200]}')
        print()
