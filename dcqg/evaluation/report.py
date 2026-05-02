"""Reporting functions (from baselines.py).

print_comparison_table: comparison table of all methods.
print_fair_metrics_table: difficulty control and fair metrics tables.
These are pure print functions.
"""


def print_comparison_table(all_scored):
    """Print a comparison table of all methods."""
    print("\n" + "=" * 80)
    print("METHOD COMPARISON (Unified LLM Judge)")
    print("=" * 80)

    header = f"{'Method':<25} {'N':>4} {'Pass':>4} {'Pass%':>6} {'Ansble':>7} {'SolCor':>7} {'Comp':>7}"
    print(header)
    print("-" * len(header))

    for name, (results, scored) in all_scored.items():
        n_gen = len(results)
        n_pass = sum(1 for r in results if r["grammar_pass"])
        n_scored = len(scored)
        pass_rate = n_pass / n_gen * 100 if n_gen > 0 else 0
        avg_ans = sum(s["judge_answerable"] for s in scored) / n_scored if scored else 0
        avg_cor = sum(s["judge_solver_correct"] for s in scored) / n_scored if scored else 0
        avg_com = sum(s["composite"] for s in scored) / n_scored if scored else 0
        print(f"{name:<25} {n_gen:>4} {n_pass:>4} {pass_rate:>5.1f}% {avg_ans:>7.3f} {avg_cor:>7.3f} {avg_com:>7.3f}")

    # By difficulty
    for d in ["Easy", "Medium", "Hard"]:
        print(f"\n--- {d} ---")
        print(header)
        print("-" * len(header))
        for name, (results, scored) in all_scored.items():
            d_results = [r for r in results if r["difficulty"] == d]
            d_scored = [s for s in scored if s["difficulty"] == d]
            n_gen = len(d_results)
            n_pass = sum(1 for r in d_results if r["grammar_pass"])
            pass_rate = n_pass / n_gen * 100 if n_gen > 0 else 0
            n_scored = len(d_scored)
            avg_ans = sum(s["judge_answerable"] for s in d_scored) / n_scored if d_scored else 0
            avg_cor = sum(s["judge_solver_correct"] for s in d_scored) / n_scored if d_scored else 0
            avg_com = sum(s["composite"] for s in d_scored) / n_scored if d_scored else 0
            print(f"{name:<25} {n_gen:>4} {n_pass:>4} {pass_rate:>5.1f}% {avg_ans:>7.3f} {avg_cor:>7.3f} {avg_com:>7.3f}")


def print_fair_metrics_table(all_fair):
    """Print difficulty control and fair metrics tables."""
    print("\n" + "=" * 90)
    print("DIFFICULTY CONTROL (Primary Metrics)")
    print("=" * 90)
    header = f"{'Method':<25} {'Easy SolCor':>11} {'Med SolCor':>10} {'Hard SolCor':>11} {'E-H gap':>7} {'DC Score':>8} {'Violations':>10}"
    print(header)
    print("-" * len(header))
    for name, fair in all_fair.items():
        print(f"{name:<25} {fair['easy_sol']:>11.3f} {fair['med_sol']:>10.3f} {fair['hard_sol']:>11.3f} {fair['e_h_gap']:>7.3f} {fair['dc_score']:>8.3f} {fair['violations']:>10}")

    print("\n" + "=" * 90)
    print("FAIR METRICS (Secondary)")
    print("=" * 90)
    header2 = f"{'Method':<25} {'Pass%':>6} {'Cond SolCor':>11} {'Macro SolCor':>12} {'E2E SolCor':>10} {'Cond Comp':>9} {'Macro Comp':>10} {'E2E Comp':>8}"
    print(header2)
    print("-" * len(header2))
    for name, fair in all_fair.items():
        pass_pct = fair['n_scored'] / fair['n_total'] * 100 if fair['n_total'] > 0 else 0
        print(f"{name:<25} {pass_pct:>5.1f}% {fair['macro_sol']:>11.3f} {fair['macro_sol']:>12.3f} {fair['e2e_sol']:>10.3f} {fair['macro_comp']:>9.3f} {fair['macro_comp']:>10.3f} {fair['e2e_comp']:>8.3f}")
