"""
prompts.py — All prompt templates for every task
"""

# ─────────────────────────────────────────────
# TASK 1 — Zero-Shot
# ─────────────────────────────────────────────

ZERO_SHOT_SYSTEM = """\
You are a review classification assistant.
Given a Yelp review, classify it as 1–5 stars.
Respond ONLY with valid JSON in this exact format (no extra text, no markdown):
{"stars": <integer 1-5>, "explanation": "<one sentence reason>"}
"""

ZERO_SHOT_USER = """\
Review: {review}
"""

# ─────────────────────────────────────────────
# TASK 1 — Few-Shot (3 examples hardcoded)
# ─────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "review": "Absolutely terrible. Cold food, rude staff, and they got my order wrong twice. Never coming back.",
        "stars": 1,
        "explanation": "Multiple serious complaints about food quality, service, and accuracy."
    },
    {
        "review": "Decent place, nothing special. Food was okay, service was average. Probably wouldn't go out of my way to return.",
        "stars": 3,
        "explanation": "Neutral tone with no strong positives or negatives."
    },
    {
        "review": "One of the best meals I've ever had! The staff was incredibly attentive and the pasta was to die for. Highly recommend!",
        "stars": 5,
        "explanation": "Enthusiastic praise for food and service with a strong recommendation."
    },
]

def build_few_shot_messages(review: str) -> list[dict]:
    messages = [{"role": "system", "content": ZERO_SHOT_SYSTEM}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"Review: {ex['review']}"})
        messages.append({
            "role": "assistant",
            "content": f'{{"stars": {ex["stars"]}, "explanation": "{ex["explanation"]}"}}'
        })
    messages.append({"role": "user", "content": f"Review: {review}"})
    return messages

# ─────────────────────────────────────────────
# TASK 2 — Direct (no reasoning)
# ─────────────────────────────────────────────

DIRECT_SYSTEM = """\
You are a review classification assistant.
Given a Yelp review, output ONLY the star rating as a single integer between 1 and 5.
Do not explain. Do not add any other text. Just the number.
"""

DIRECT_USER = "Review: {review}"

# ─────────────────────────────────────────────
# TASK 2 — Chain-of-Thought
# ─────────────────────────────────────────────

COT_SYSTEM = """\
You are a review classification assistant.
Given a Yelp review, reason step by step about the sentiment, then output the star rating.

Format your response EXACTLY as:
Reasoning: <your step-by-step analysis>
Stars: <integer 1-5>
"""

COT_USER = "Review: {review}"

# ─────────────────────────────────────────────
# TASK 3 — Multi-Objective Assistant
# ─────────────────────────────────────────────

MULTI_OBJ_SYSTEM = """\
You are a helpful assistant for restaurant business owners.
Given a customer review, provide:
1. A star rating (1–5)
2. The key complaint or compliment (one sentence)
3. A short, polite business response the owner could post publicly (2–3 sentences)

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{
  "stars": <integer 1-5>,
  "key_point": "<main complaint or compliment>",
  "business_response": "<polite public reply>"
}
"""

MULTI_OBJ_USER = "Review: {review}"

# ─────────────────────────────────────────────
# TASK 3 — LLM-as-Judge
# ─────────────────────────────────────────────

LLM_JUDGE_SYSTEM = """\
You are an objective evaluator. Given a customer review and an AI-generated assistant output,
score the output on two dimensions using integers 1–5:

- faithfulness: Does the key_point accurately reflect the review content? (1=completely wrong, 5=perfectly accurate)
- actionability: Is the business_response polite, relevant, and professionally useful? (1=useless/inappropriate, 5=excellent)

Respond ONLY with valid JSON:
{"faithfulness": <1-5>, "actionability": <1-5>, "reason": "<one sentence>"}
"""

LLM_JUDGE_USER = """\
Review: {review}

AI Output:
stars: {stars}
key_point: {key_point}
business_response: {business_response}
"""
