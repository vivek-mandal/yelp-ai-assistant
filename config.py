"""
config.py — Shared Azure OpenAI client + constants
"""
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4.1")

# Subset sizes — increase for real runs, keep small for fast iteration
SAMPLE_SIZE       = 200   # reviews for Tasks 1 & 2
MULTI_OBJ_SAMPLE  = 50    # reviews for Task 3 (LLM judge costs more)
DOMAIN_SHIFT_SIZE = 100   # per domain for Task 4
RANDOM_SEED       = 42
