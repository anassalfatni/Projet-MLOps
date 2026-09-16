"""
preprocess.py
-------------
Charge le dataset brut sauvegardé par get_data.py, applique la tokenisation
avec le tokenizer du checkpoint choisi, puis sauvegarde :
  - le dataset tokenizé (Arrow) dans preprocessing.tokenized_data_dir
  - le tokenizer dans preprocessing.tokenizer_dir

Usage:
    python src/preprocess.py
    python src/preprocess.py --config config.yaml
"""

import argparse
import logging
from pathlib import Path

import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess(config: dict) -> None:
    raw_data_dir = config["data"]["raw_data_dir"]
    checkpoint = config["model"]["checkpoint"]
    max_length = config["model"]["max_length"]

    tokenized_data_dir = Path(config["preprocessing"]["tokenized_data_dir"])
    tokenizer_dir = Path(config["preprocessing"]["tokenizer_dir"])

    logger.info("Chargement du dataset brut depuis '%s'...", raw_data_dir)
    dataset = load_from_disk(raw_data_dir)

    logger.info("Chargement du tokenizer '%s'...", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    logger.info("Tokenisation en cours...")
    tokenized = dataset.map(tokenize_fn, batched=True)

    tokenized_data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    tokenized.save_to_disk(str(tokenized_data_dir))
    tokenizer.save_pretrained(str(tokenizer_dir))

    logger.info("Dataset tokenizé sauvegardé dans '%s'", tokenized_data_dir)
    logger.info("Tokenizer sauvegardé dans '%s'", tokenizer_dir)


def main():
    parser = argparse.ArgumentParser(description="Tokenize le dataset brut.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Chemin vers config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    preprocess(config)


if __name__ == "__main__":
    main()
