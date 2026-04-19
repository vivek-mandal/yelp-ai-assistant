"""
evaluate.py — Accuracy, Macro-F1, JSON compliance, CoT mismatch detection
"""
import json
import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# ─────────────────────────────────────────────
# JSON parsing helpers
# ─────────────────────────────────────────────

def parse_json_response(raw: str) -> dict | None:
    """Try to extract a JSON object from a model response."""
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        # Try to find JSON block inside the text
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def parse_star_from_json(raw: str) -> int | None:
    parsed = parse_json_response(raw)
    if parsed and "stars" in parsed:
        try:
            s = int(parsed["stars"])
            if 1 <= s <= 5:
                return s
        except (ValueError, TypeError):
            pass
    return None


def parse_star_direct(raw: str) -> int | None:
    """Parse a single integer 1–5 from a direct-answer response."""
    raw = raw.strip()
    match = re.search(r'\b([1-5])\b', raw)
    if match:
        return int(match.group(1))
    return None


def parse_cot_response(raw: str) -> tuple[str, int | None]:
    """
    Parse chain-of-thought response.
    Returns (reasoning_text, stars).
    """
    reasoning = ""
    stars = None

    r_match = re.search(r'Reasoning:\s*(.*?)(?=Stars:|$)', raw, re.DOTALL | re.IGNORECASE)
    s_match = re.search(r'Stars:\s*([1-5])', raw, re.IGNORECASE)

    if r_match:
        reasoning = r_match.group(1).strip()
    if s_match:
        stars = int(s_match.group(1))

    return reasoning, stars


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────

def compute_metrics(y_true: list[int], y_pred: list[int | None]) -> dict:
    """
    Compute accuracy and macro-F1, ignoring None predictions.
    Also reports parse failure rate.
    """
    failed = sum(1 for p in y_pred if p is None)
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]

    if not pairs:
        return {"accuracy": 0.0, "macro_f1": 0.0, "parse_failures": failed, "n": len(y_true)}

    yt, yp = zip(*pairs)
    return {
        "accuracy":       round(accuracy_score(yt, yp), 4),
        "macro_f1":       round(f1_score(yt, yp, average="macro", labels=[1,2,3,4,5], zero_division=0), 4),
        "parse_failures": failed,
        "n":              len(y_true),
    }


def json_compliance_rate(raw_responses: list[str]) -> float:
    """Fraction of responses that are valid JSON with a 'stars' key."""
    valid = sum(1 for r in raw_responses if parse_star_from_json(r) is not None)
    return round(valid / len(raw_responses), 4) if raw_responses else 0.0


def detect_cot_mismatch(reasoning: str, stars: int | None) -> bool:
    """
    Heuristic: flag if reasoning contains strong negative/positive words
    but the star rating contradicts them.
    """
    if stars is None:
        return False
    negative_words = {"terrible", "awful", "horrible", "worst", "bad", "poor", "disappointing", "disgusting"}
    positive_words = {"amazing", "excellent", "fantastic", "wonderful", "great", "best", "love", "perfect"}

    text = reasoning.lower()
    neg_score = sum(1 for w in negative_words if w in text)
    pos_score = sum(1 for w in positive_words if w in text)

    if neg_score > pos_score and stars >= 4:
        return True   # Reasoning negative, stars positive
    if pos_score > neg_score and stars <= 2:
        return True   # Reasoning positive, stars negative
    return False


# ─────────────────────────────────────────────
# Results formatting
# ─────────────────────────────────────────────

def print_results_table(results: dict[str, dict]):
    """Print a comparison table of multiple experiment results."""
    from tabulate import tabulate
    rows = []
    for name, m in results.items():
        rows.append([
            name,
            f"{m.get('accuracy', 0):.2%}",
            f"{m.get('macro_f1', 0):.4f}",
            f"{m.get('json_compliance', 'N/A')}",
            m.get('parse_failures', 'N/A'),
            m.get('n', 'N/A'),
        ])
    headers = ["Experiment", "Accuracy", "Macro-F1", "JSON Compliance", "Parse Fails", "N"]
    print(tabulate(rows, headers=headers, tablefmt="grid"))
