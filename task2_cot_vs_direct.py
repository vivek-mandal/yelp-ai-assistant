"""
task2_cot_vs_direct.py — Chain-of-Thought vs Direct prompting comparison
Run: uv run python src/task2_cot_vs_direct.py
"""
import json
import time
import pandas as pd
from tqdm import tqdm

from config import get_client, DEPLOYMENT, SAMPLE_SIZE
from data_loader import load_yelp
from prompts import DIRECT_SYSTEM, DIRECT_USER, COT_SYSTEM, COT_USER
from evaluate import (
    parse_star_direct, parse_cot_response,
    compute_metrics, detect_cot_mismatch, print_results_table
)


def call_direct(client, review: str) -> str:
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user",   "content": DIRECT_USER.format(review=review)},
        ],
        temperature=0,
        max_tokens=10,
    )
    return resp.choices[0].message.content.strip()


def call_cot(client, review: str) -> str:
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user",   "content": COT_USER.format(review=review)},
        ],
        temperature=0,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def run_direct(client, df: pd.DataFrame) -> pd.DataFrame:
    stars_pred, raw_outputs = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Task2 Direct"):
        try:
            raw = call_direct(client, row["review"][:1500])
        except Exception as e:
            print(f"  [API error] {e}")
            raw = ""
        raw_outputs.append(raw)
        stars_pred.append(parse_star_direct(raw))
        time.sleep(0.3)

    result = df.copy()
    result["raw_output"]  = raw_outputs
    result["stars_pred"]  = stars_pred
    return result


def run_cot(client, df: pd.DataFrame) -> pd.DataFrame:
    stars_pred, reasonings, mismatches, raw_outputs = [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Task2 CoT"):
        try:
            raw = call_cot(client, row["review"][:1500])
        except Exception as e:
            print(f"  [API error] {e}")
            raw = ""
        raw_outputs.append(raw)
        reasoning, stars = parse_cot_response(raw)
        stars_pred.append(stars)
        reasonings.append(reasoning)
        mismatches.append(detect_cot_mismatch(reasoning, stars))
        time.sleep(0.3)

    result = df.copy()
    result["raw_output"]  = raw_outputs
    result["stars_pred"]  = stars_pred
    result["reasoning"]   = reasonings
    result["mismatch"]    = mismatches
    return result


def analyse_error_types(direct_df: pd.DataFrame, cot_df: pd.DataFrame):
    """Print a breakdown of error types for both strategies."""
    print("\n--- Error Type Analysis ---")

    for name, df in [("Direct", direct_df), ("CoT", cot_df)]:
        parsed = df[df["stars_pred"].notna()].copy()
        parsed["error"] = (parsed["stars"] - parsed["stars_pred"]).abs()

        total   = len(df)
        failed  = df["stars_pred"].isna().sum()
        exact   = (parsed["error"] == 0).sum()
        off1    = (parsed["error"] == 1).sum()
        off2p   = (parsed["error"] >= 2).sum()

        print(f"\n  [{name}]  N={total} | Parsed={total-failed} | ParseFail={failed}")
        print(f"    Exact match : {exact}  ({exact/total:.1%})")
        print(f"    Off by 1    : {off1}  ({off1/total:.1%})")
        print(f"    Off by 2+   : {off2p}  ({off2p/total:.1%})")

    # CoT-specific: reasoning mismatch
    if "mismatch" in cot_df.columns:
        n_mismatch = cot_df["mismatch"].sum()
        print(f"\n  [CoT] Reasoning↔Stars mismatches : {n_mismatch} / {len(cot_df)}"
              f"  ({n_mismatch/len(cot_df):.1%})")
        print("\n  Example mismatches:")
        examples = cot_df[cot_df["mismatch"]].head(3)
        for _, r in examples.iterrows():
            print(f"    ★ True:{r['stars']} Pred:{r['stars_pred']}")
            print(f"      Reasoning: {r['reasoning'][:120]}...")


def main():
    client = get_client()
    df = load_yelp(split="test", n=SAMPLE_SIZE)

    print("\n" + "="*60)
    print("TASK 2 — Direct Prompting")
    print("="*60)
    direct_df = run_direct(client, df)
    direct_df.to_csv("results/task2_direct.csv", index=False)

    print("\n" + "="*60)
    print("TASK 2 — Chain-of-Thought Prompting")
    print("="*60)
    cot_df = run_cot(client, df)
    cot_df.to_csv("results/task2_cot.csv", index=False)

    # ── Metrics ──────────────────────────────────────────────
    direct_metrics = compute_metrics(direct_df["stars"].tolist(), direct_df["stars_pred"].tolist())
    cot_metrics    = compute_metrics(cot_df["stars"].tolist(),    cot_df["stars_pred"].tolist())

    print("\n" + "="*60)
    print("TASK 2 — Results")
    print("="*60)
    print_results_table({"Direct": direct_metrics, "Chain-of-Thought": cot_metrics})

    analyse_error_types(direct_df, cot_df)

    with open("results/task2_metrics.json", "w") as f:
        json.dump({"direct": direct_metrics, "cot": cot_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
