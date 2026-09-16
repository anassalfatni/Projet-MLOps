"""
evaluate.py
-----------
Charge le modèle fine-tuné (training.output_dir) et le dataset tokenizé,
évalue sur le split "test", puis sauvegarde les métriques en JSON.

Usage:
    python src/evaluate.py
    python src/evaluate.py --config config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent

sys.path = [p for p in sys.path if Path(p).resolve() != SRC_DIR]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml
import evaluate as hf_evaluate
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_compute_metrics():
    accuracy = hf_evaluate.load("accuracy")
    f1 = hf_evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy.compute(predictions=predictions, references=labels)
        f1_score = f1.compute(predictions=predictions, references=labels)
        return {**acc, **f1_score}

    return compute_metrics


def evaluate_model(config: dict) -> dict:
    tokenized_data_dir = config["preprocessing"]["tokenized_data_dir"]
    model_dir = config["training"]["output_dir"]
    results_path = Path(config["evaluation"]["results_path"])

    logger.info("Chargement du dataset tokenizé depuis '%s'...", tokenized_data_dir)
    tokenized = load_from_disk(tokenized_data_dir)

    logger.info("Chargement du modèle entraîné depuis '%s'...", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TrainingArguments minimal, uniquement pour piloter l'évaluation
    eval_args = TrainingArguments(
        output_dir="tmp_eval",
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
    )

    logger.info("Évaluation sur le split 'test'...")
    metrics = trainer.evaluate(tokenized["test"])
    logger.info("Résultats : %s", metrics)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Métriques sauvegardées dans '%s'", results_path)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Évalue le modèle entraîné sur le split test.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Chemin vers config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_model(config)


if __name__ == "__main__":
    main()
