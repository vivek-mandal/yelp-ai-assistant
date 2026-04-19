"""
task3_multi_objective.py — Multi-Objective AI Assistant + LLM-as-Judge evaluation
Run: uv run python src/task3_multi_objective.py
"""
import json
import time
import pandas as pd
from tqdm import tqdm

from config import get_client, DEPLOYMENT, MULTI_OBJ_SAMPLE
from data_loader import load_yelp
from prompts import MULTI_OBJ_SYSTEM, MULTI_OBJ_USER, LLM_JUDGE_SYSTEM, LLM_JUDGE_USER
from evaluate import parse_json_response


# ─────────────────────────────────────────────
# Step 1 — Generate multi-objective output
# ─────────────────────────────────────────────

def call_multi_obj(client, review: str) -> str:
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": MULTI_OBJ_SYSTEM},
            {"role": "user",   "content": MULTI_OBJ_USER.format(review=review)},
        ],
        temperature=0.3,   # slight creativity for business response
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def parse_multi_obj(raw: str) -> dict:
    parsed = parse_json_response(raw)
    if parsed:
        return {
            "stars":             parsed.get("stars"),
            "key_point":         parsed.get("key_point", ""),
            "business_response": parsed.get("business_response", ""),
            "parse_ok":          True,
        }
    return {"stars": None, "key_point": "", "business_response": "", "parse_ok": False}


def run_multi_obj(client, df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Task3 MultiObj"):
        try:
            raw = call_multi_obj(client, row["review"][:1500])
        except Exception as e:
            print(f"  [API error] {e}")
            raw = ""
        parsed = parse_multi_obj(raw)
        records.append({
            "review":            row["review"],
            "true_stars":        row["stars"],
            "raw_output":        raw,
            **parsed,
        })
        time.sleep(0.4)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Step 2 — LLM-as-Judge
# ─────────────────────────────────────────────

def call_judge(client, review: str, stars, key_point: str, business_response: str) -> str:
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": LLM_JUDGE_SYSTEM},
            {"role": "user",   "content": LLM_JUDGE_USER.format(
                review=review,
                stars=stars,
                key_point=key_point,
                business_response=business_response,
            )},
        ],
        temperature=0,
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


def run_judge(client, df: pd.DataFrame) -> pd.DataFrame:
    """Only judge rows where parsing succeeded."""
    faithfulness_scores, actionability_scores, judge_reasons = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Task3 Judge"):
        if not row["parse_ok"]:
            faithfulness_scores.append(None)
            actionability_scores.append(None)
            judge_reasons.append("parse failed")
            continue
        try:
            raw = call_judge(
                client,
                row["review"][:1000],
                row["stars"],
                row["key_point"],
                row["business_response"],
            )
            parsed = parse_json_response(raw)
            if parsed:
                faithfulness_scores.append(parsed.get("faithfulness"))
                actionability_scores.append(parsed.get("actionability"))
                judge_reasons.append(parsed.get("reason", ""))
            else:
                faithfulness_scores.append(None)
                actionability_scores.append(None)
                judge_reasons.append(raw[:100])
        except Exception as e:
            faithfulness_scores.append(None)
            actionability_scores.append(None)
            judge_reasons.append(str(e))
        time.sleep(0.4)

    result = df.copy()
    result["faithfulness"]  = faithfulness_scores
    result["actionability"] = actionability_scores
    result["judge_reason"]  = judge_reasons
    return result


def print_summary(df: pd.DataFrame):
    valid = df[df["parse_ok"]]
    print(f"\n  Parse success rate : {len(valid)}/{len(df)}  ({len(valid)/len(df):.1%})")

    if "faithfulness" in df.columns:
        f_scores = df["faithfulness"].dropna()
        a_scores = df["actionability"].dropna()
        print(f"  Avg Faithfulness   : {f_scores.mean():.2f} / 5  (n={len(f_scores)})")
        print(f"  Avg Actionability  : {a_scores.mean():.2f} / 5  (n={len(a_scores)})")

    print("\n  --- Success Examples (faithfulness ≥ 4 & actionability ≥ 4) ---")
    if "faithfulness" in df.columns:
        good = df[(df["faithfulness"] >= 4) & (df["actionability"] >= 4)].head(2)
        for _, r in good.iterrows():
            print(f"\n  Review   : {r['review'][:120]}...")
            print(f"  Stars    : {r['stars']} (true: {r['true_stars']})")
            print(f"  KeyPoint : {r['key_point']}")
            print(f"  Response : {r['business_response']}")
            print(f"  Judge    : F={r['faithfulness']} A={r['actionability']}")

    print("\n  --- Failure Examples (faithfulness ≤ 2 or actionability ≤ 2) ---")
    if "faithfulness" in df.columns:
        bad = df[(df["faithfulness"] <= 2) | (df["actionability"] <= 2)].head(2)
        for _, r in bad.iterrows():
            print(f"\n  Review   : {r['review'][:120]}...")
            print(f"  Stars    : {r['stars']} (true: {r['true_stars']})")
            print(f"  KeyPoint : {r['key_point']}")
            print(f"  Response : {r['business_response']}")
            print(f"  Judge    : F={r['faithfulness']} A={r['actionability']} — {r['judge_reason']}")


def main():
    client = get_client()
    df = load_yelp(split="test", n=MULTI_OBJ_SAMPLE)

    print("\n" + "="*60)
    print("TASK 3 — Multi-Objective Assistant")
    print("="*60)
    multi_df = run_multi_obj(client, df)

    print("\n" + "="*60)
    print("TASK 3 — LLM-as-Judge Evaluation")
    print("="*60)
    judged_df = run_judge(client, multi_df)
    judged_df.to_csv("results/task3_multi_obj.csv", index=False)

    print("\n" + "="*60)
    print("TASK 3 — Summary")
    print("="*60)
    print_summary(judged_df)

    # Save metrics
    metrics = {
        "parse_rate":         round(judged_df["parse_ok"].mean(), 4),
        "avg_faithfulness":   round(judged_df["faithfulness"].dropna().mean(), 3),
        "avg_actionability":  round(judged_df["actionability"].dropna().mean(), 3),
    }
    with open("results/task3_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → results/task3_metrics.json")


if __name__ == "__main__":
    main()
