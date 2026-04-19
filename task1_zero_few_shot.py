"""
task1_zero_few_shot.py — Zero-Shot & Few-Shot classification with JSON output
Run: uv run python src/task1_zero_few_shot.py
"""
import json
import time
import pandas as pd
from tqdm import tqdm

from config import get_client, DEPLOYMENT, SAMPLE_SIZE
from data_loader import load_yelp
from prompts import ZERO_SHOT_SYSTEM, ZERO_SHOT_USER, build_few_shot_messages
from evaluate import (
    parse_star_from_json, parse_json_response,
    json_compliance_rate, compute_metrics, print_results_table
)


def call_zero_shot(client, review: str) -> str:
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": ZERO_SHOT_SYSTEM},
            {"role": "user",   "content": ZERO_SHOT_USER.format(review=review)},
        ],
        temperature=0,
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


def call_few_shot(client, review: str) -> str:
    messages = build_few_shot_messages(review)
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        temperature=0,
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


def run_experiment(client, df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Run zero-shot or few-shot on the dataframe. mode='zero' or 'few'."""
    raw_outputs, stars_pred, explanations = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Task1 {mode}-shot"):
        review = row["review"][:1500]   # cap length to control tokens
        try:
            if mode == "zero":
                raw = call_zero_shot(client, review)
            else:
                raw = call_few_shot(client, review)
        except Exception as e:
            print(f"  [API error] {e}")
            raw = ""

        raw_outputs.append(raw)
        stars_pred.append(parse_star_from_json(raw))

        parsed = parse_json_response(raw)
        explanations.append(parsed.get("explanation", "") if parsed else "")

        time.sleep(0.3)   # gentle rate limit

    result_df = df.copy()
    result_df["raw_output"]   = raw_outputs
    result_df["stars_pred"]   = stars_pred
    result_df["explanation"]  = explanations
    return result_df


def main():
    import os
    os.makedirs("results", exist_ok=True)

    client = get_client()
    df = load_yelp(split="test", n=SAMPLE_SIZE)

    print("\n" + "="*60)
    print("TASK 1 — Zero-Shot Prompting")
    print("="*60)
    zero_df = run_experiment(client, df, mode="zero")
    zero_df.to_csv("results/task1_zero_shot.csv", index=False)

    print("\n" + "="*60)
    print("TASK 1 — Few-Shot Prompting")
    print("="*60)
    few_df = run_experiment(client, df, mode="few")
    few_df.to_csv("results/task1_few_shot.csv", index=False)

    # ── Metrics ──────────────────────────────────────────────
    zero_metrics = compute_metrics(zero_df["stars"].tolist(), zero_df["stars_pred"].tolist())
    zero_metrics["json_compliance"] = f"{json_compliance_rate(zero_df['raw_output'].tolist()):.2%}"

    few_metrics = compute_metrics(few_df["stars"].tolist(), few_df["stars_pred"].tolist())
    few_metrics["json_compliance"] = f"{json_compliance_rate(few_df['raw_output'].tolist()):.2%}"

    print("\n" + "="*60)
    print("TASK 1 — Results")
    print("="*60)
    print_results_table({"Zero-Shot": zero_metrics, "Few-Shot": few_metrics})

    # Save metrics
    with open("results/task1_metrics.json", "w") as f:
        json.dump({"zero_shot": zero_metrics, "few_shot": few_metrics}, f, indent=2)

    # ── Sample outputs ────────────────────────────────────────
    print("\n--- Sample Zero-Shot Output ---")
    sample = zero_df[zero_df["stars_pred"].notna()].head(3)
    for _, r in sample.iterrows():
        print(f"  True: {r['stars']} | Pred: {r['stars_pred']} | Expl: {r['explanation'][:80]}")

    print("\n--- Sample Few-Shot Output ---")
    sample = few_df[few_df["stars_pred"].notna()].head(3)
    for _, r in sample.iterrows():
        print(f"  True: {r['stars']} | Pred: {r['stars_pred']} | Expl: {r['explanation'][:80]}")


if __name__ == "__main__":
    main()
