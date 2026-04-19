"""
data_loader.py — Load & sample Yelp / Amazon / IMDB datasets
"""
import pandas as pd
from datasets import load_dataset
from config import SAMPLE_SIZE, DOMAIN_SHIFT_SIZE, RANDOM_SEED


def load_yelp(split="test", n=SAMPLE_SIZE) -> pd.DataFrame:
    """
    Yelp Review Full — labels are 0-indexed (0–4), we convert to 1–5 stars.
    """
    print(f"[data] Loading Yelp ({split}, n={n})...")
    ds = load_dataset("yelp_review_full", split=split)
    df = ds.to_pandas().sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    df["stars"] = df["label"] + 1          # 0-indexed → 1-5
    df = df.rename(columns={"text": "review"})
    df = df[["review", "stars"]]
    print(f"[data] Yelp loaded: {len(df)} reviews. Star dist:\n{df['stars'].value_counts().sort_index().to_dict()}")
    return df


def load_amazon(n=DOMAIN_SHIFT_SIZE) -> pd.DataFrame:
    """
    Amazon Polarity (2-class) mapped to rough 1/5 stars for domain-shift test.
    """
    print(f"[data] Loading Amazon (n={n})...")
    ds = load_dataset("amazon_polarity", split="test")
    df = ds.to_pandas().sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    df["stars"] = df["label"].map({0: 1, 1: 5})   # negative→1, positive→5
    df = df.rename(columns={"content": "review"})
    df = df[["review", "stars"]]
    print(f"[data] Amazon loaded: {len(df)} reviews.")
    return df


def load_imdb(n=DOMAIN_SHIFT_SIZE) -> pd.DataFrame:
    """
    IMDB sentiment (2-class) mapped to rough 1/5 stars.
    """
    print(f"[data] Loading IMDB (n={n})...")
    ds = load_dataset("imdb", split="test")
    df = ds.to_pandas().sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    df["stars"] = df["label"].map({0: 1, 1: 5})
    df = df.rename(columns={"text": "review"})
    df = df[["review", "stars"]]
    print(f"[data] IMDB loaded: {len(df)} reviews.")
    return df


if __name__ == "__main__":
    yelp = load_yelp()
    print(yelp.head(2))
