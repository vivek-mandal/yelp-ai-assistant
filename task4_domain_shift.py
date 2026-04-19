"""
task4_domain_shift.py — Domain Shift & Adversarial Robustness
Run: uv run python src/task4_domain_shift.py
"""
import json
import time
import pandas as pd
from tqdm import tqdm

from config import get_client, DEPLOYMENT, DOMAIN_SHIFT_SIZE
from data_loader import load_yelp, load_amazon, load_imdb
from prompts import ZERO_SHOT_SYSTEM, ZERO_SHOT_USER, build_few_shot_messages
from evaluate import parse_star_from_json, compute_metrics, print_results_table


# ─────────────────────────────────────────────
# Adversarial examples (binary: 1 or 5 stars)
# ─────────────────────────────────────────────
ADVERSARIAL_REVIEWS = [
    # Sarcasm — true label 1
    {"review": "Oh wow, absolutely PHENOMENAL service. Waited 45 minutes to be ignored completely. 10/10 would not recommend.", "stars": 1},
    # Sarcasm — true label 1
    {"review": "The food was so good I immediately got food poisoning. Definitely coming back! 🤢", "stars": 1},
    # Mixed signals — true label 2
    {"review": "The ambiance was nice but everything else was a disaster. Pretty decor doesn't make up for cold food.", "stars": 2},
    # Typos and slang — true label 5
    {"review": "omg dis place is litty!! fod was amazin nd staff were so frendly highly reccomend!!", "stars": 5},
    # Negative framing but positive — true label 4
    {"review": "Not the worst meal I've had this year. Actually surprisingly decent. Would cautiously return.", "stars": 4},
    # Very short — true label 1
    {"review": "Trash.", "stars": 1},
    # Emoji heavy — true label 5
    {"review": "🔥🔥🔥 Best tacos EVER!!!! 😍😍 Will 100% be back every single week!", "stars": 5},
    # Contrast/qualifier — true label 3
    {"review": "I mean, it wasn't bad per se, but it definitely wasn't great. Just kind of... there.", "stars": 3},
]


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


def run_on_domain(client, df: pd.DataFrame, domain_name: str) -> dict:
    """Classify a dataset and return metrics dict."""
    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Domain: {domain_name}"):
        try:
            raw = call_zero_shot(client, row["review"][:1500])
            preds.append(parse_star_from_json(raw))
        except Exception as e:
            print(f"  [API error] {e}")
            preds.append(None)
        time.sleep(0.3)

    metrics = compute_metrics(df["stars"].tolist(), preds)
    return metrics


def run_adversarial(client) -> dict:
    """Run classification on hand-crafted adversarial examples."""
    adv_df = pd.DataFrame(ADVERSARIAL_REVIEWS)
    preds, raw_outputs = [], []

    for _, row in adv_df.iterrows():
        try:
            raw = call_zero_shot(client, row["review"])
            pred = parse_star_from_json(raw)
        except Exception as e:
            raw, pred = str(e), None
        preds.append(pred)
        raw_outputs.append(raw)
        time.sleep(0.3)

    adv_df["stars_pred"] = preds
    adv_df["raw_output"] = raw_outputs

    print("\n  --- Adversarial Examples ---")
    for _, r in adv_df.iterrows():
        correct = "✓" if r["stars"] == r["stars_pred"] else "✗"
        print(f"  {correct} True:{r['stars']} Pred:{r['stars_pred']}  | {r['review'][:80]}")

    metrics = compute_metrics(adv_df["stars"].tolist(), adv_df["stars_pred"].tolist())
    adv_df.to_csv("results/task4_adversarial.csv", index=False)
    return metrics


def compute_domain_drop(yelp_metrics: dict, other_metrics: dict, domain: str):
    acc_drop = yelp_metrics["accuracy"] - other_metrics["accuracy"]
    f1_drop  = yelp_metrics["macro_f1"] - other_metrics["macro_f1"]
    print(f"\n  [{domain}] Accuracy drop: {acc_drop:+.2%}  |  Macro-F1 drop: {f1_drop:+.4f}")


def main():
    client = get_client()

    # ── Yelp (in-domain) ─────────────────────────────────────
    print("\n" + "="*60)
    print("TASK 4 — Domain Shift")
    print("="*60)
    yelp_df   = load_yelp(split="test", n=DOMAIN_SHIFT_SIZE)
    amazon_df = load_amazon(n=DOMAIN_SHIFT_SIZE)
    imdb_df   = load_imdb(n=DOMAIN_SHIFT_SIZE)

    yelp_metrics   = run_on_domain(client, yelp_df,   "Yelp (in-domain)")
    amazon_metrics = run_on_domain(client, amazon_df, "Amazon")
    imdb_metrics   = run_on_domain(client, imdb_df,   "IMDB")

    print("\n" + "="*60)
    print("TASK 4 — Domain Comparison Table")
    print("="*60)
    print_results_table({
        "Yelp (in-domain)": yelp_metrics,
        "Amazon":           amazon_metrics,
        "IMDB":             imdb_metrics,
    })

    compute_domain_drop(yelp_metrics, amazon_metrics, "Amazon")
    compute_domain_drop(yelp_metrics, imdb_metrics,   "IMDB")

    # ── Adversarial ───────────────────────────────────────────
    print("\n" + "="*60)
    print("TASK 4 — Adversarial Robustness")
    print("="*60)
    adv_metrics = run_adversarial(client)
    print(f"\n  Adversarial accuracy: {adv_metrics['accuracy']:.2%}  |  Macro-F1: {adv_metrics['macro_f1']:.4f}")

    # ── Mitigation note ───────────────────────────────────────
    print("\n" + "="*60)
    print("TASK 4 — Suggested Mitigation")
    print("="*60)
    print("""
  Mitigation: Domain-Adaptive Few-Shot Prompting
  ─────────────────────────────────────────────
  Instead of fixed Yelp examples in few-shot prompts, dynamically
  select examples from the TARGET domain (Amazon/IMDB) at inference time.

  Implementation:
    1. Maintain a small labelled pool (~20 examples) per domain
    2. For each input, embed it + retrieve the 3 nearest neighbours
       from the target domain's pool (semantic similarity)
    3. Use those as few-shot examples

  Expected benefit: closes ~30-50% of the accuracy gap in 2-class domains
  (per literature on domain-adaptive prompting, e.g. Su et al. 2022).
""")

    # ── Save all metrics ──────────────────────────────────────
    with open("results/task4_metrics.json", "w") as f:
        json.dump({
            "yelp":        yelp_metrics,
            "amazon":      amazon_metrics,
            "imdb":        imdb_metrics,
            "adversarial": adv_metrics,
        }, f, indent=2)
    print("  Metrics saved → results/task4_metrics.json")


if __name__ == "__main__":
    main()
