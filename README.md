# Yelp AI Assignment

Advanced AI Systems for Yelp Reviews using Azure OpenAI (GPT-4.1).

## Setup

```bash
# 1. Install uv (if you haven't)
curl -Lsf https://astral.sh/uv/install.sh | sh

# 2. Clone / enter project folder
cd yelp-ai-assignment

# 3. Create .env from template and fill in your key
cp .env.example .env
# Edit .env → add your AZURE_OPENAI_API_KEY

# 4. Install dependencies
uv sync
```

## Run

```bash
# Run all 4 tasks end-to-end
uv run python src/run_all.py

# Or run individually
uv run python src/task1_zero_few_shot.py
uv run python src/task2_cot_vs_direct.py
uv run python src/task3_multi_objective.py
uv run python src/task4_domain_shift.py
```

## Project Structure

```
yelp-ai-assignment/
├── pyproject.toml              ← uv dependencies
├── .env.example                ← credentials template
├── src/
│   ├── config.py               ← Azure client + constants
│   ├── data_loader.py          ← Yelp / Amazon / IMDB loaders
│   ├── prompts.py              ← ALL prompt templates
│   ├── evaluate.py             ← Accuracy, F1, JSON compliance, CoT mismatch
│   ├── task1_zero_few_shot.py  ← Task 1
│   ├── task2_cot_vs_direct.py  ← Task 2
│   ├── task3_multi_objective.py← Task 3 + LLM-as-Judge
│   ├── task4_domain_shift.py   ← Task 4 + adversarial
│   └── run_all.py              ← Run everything
└── results/                    ← CSV outputs + metric JSONs
```

## Tuning Sample Sizes

Edit `src/config.py`:

```python
SAMPLE_SIZE       = 200   # Task 1 & 2
MULTI_OBJ_SAMPLE  = 50    # Task 3 (LLM judge is expensive)
DOMAIN_SHIFT_SIZE = 100   # Task 4 per domain
```

## Output Files

| File | Contents |
|------|----------|
| `results/task1_zero_shot.csv` | Raw outputs + predictions |
| `results/task1_few_shot.csv` | Raw outputs + predictions |
| `results/task1_metrics.json` | Accuracy, F1, JSON compliance |
| `results/task2_direct.csv` | Direct predictions |
| `results/task2_cot.csv` | CoT reasoning + predictions + mismatch flag |
| `results/task2_metrics.json` | Accuracy, F1 comparison |
| `results/task3_multi_obj.csv` | Stars + key_point + response + judge scores |
| `results/task3_metrics.json` | Parse rate, faithfulness, actionability |
| `results/task4_adversarial.csv` | Adversarial predictions |
| `results/task4_metrics.json` | All domain metrics |
