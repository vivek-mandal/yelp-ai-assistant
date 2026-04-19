"""
run_all.py — Run all 4 tasks in sequence and print a final summary
Usage: uv run python src/run_all.py
"""
import json
import os
import sys

os.makedirs("results", exist_ok=True)
sys.path.insert(0, os.path.dirname(__file__))

from evaluate import print_results_table

def load_metrics(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    print("\n" + "█"*60)
    print("  RUNNING ALL TASKS — Yelp AI Assignment")
    print("█"*60)

    import task1_zero_few_shot
    task1_zero_few_shot.main()

    import task2_cot_vs_direct
    task2_cot_vs_direct.main()

    import task3_multi_objective
    task3_multi_objective.main()

    import task4_domain_shift
    task4_domain_shift.main()

    # ── Final consolidated summary ────────────────────────────
    print("\n" + "█"*60)
    print("  FINAL SUMMARY")
    print("█"*60)

    t1 = load_metrics("results/task1_metrics.json")
    t2 = load_metrics("results/task2_metrics.json")
    t4 = load_metrics("results/task4_metrics.json")

    all_results = {}
    if t1:
        all_results["T1: Zero-Shot"] = t1["zero_shot"]
        all_results["T1: Few-Shot"]  = t1["few_shot"]
    if t2:
        all_results["T2: Direct"]   = t2["direct"]
        all_results["T2: CoT"]      = t2["cot"]
    if t4:
        all_results["T4: Yelp"]     = t4["yelp"]
        all_results["T4: Amazon"]   = t4["amazon"]
        all_results["T4: IMDB"]     = t4["imdb"]

    if all_results:
        print_results_table(all_results)

    t3 = load_metrics("results/task3_metrics.json")
    if t3:
        print(f"\n  Task 3 (Multi-Obj) — Parse rate: {t3['parse_rate']:.1%} | "
              f"Avg Faithfulness: {t3['avg_faithfulness']:.2f}/5 | "
              f"Avg Actionability: {t3['avg_actionability']:.2f}/5")

    print("\n✓ All done. Results saved in results/\n")


if __name__ == "__main__":
    main()
