"""Prefetch only model assets, without starting an experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from provguard.core import environment
environment()
from huggingface_hub import snapshot_download
snapshot_download("distilbert/distilbert-base-uncased", allow_patterns=["*.json", "vocab.txt", "model.safetensors"])
