"""
train.py
--------
Charge le dataset tokenizé et le tokenizer produits par preprocess.py,
fine-tune un AutoModelForSequenceClassification (DistilBERT par défaut),
puis sauvegarde le modèle + le tokenizer dans training.output_dir.

Usage:
    python src/train.py
    python src/train.py --config config.yaml
"""

import argparse
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
import evaluate
from datasets import load_from_disk

try:
    import mlflow
    import mlflow.transformers
except ImportError:
    mlflow = None

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
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    return compute_metrics


def train(config: dict) -> Trainer:
    tokenized_data_dir = config["preprocessing"]["tokenized_data_dir"]
    tokenizer_dir = config["preprocessing"]["tokenizer_dir"]
    checkpoint = config["model"]["checkpoint"]
    num_labels = config["model"]["num_labels"]
    id2label = {int(k): v for k, v in config["model"]["id2label"].items()}
    label2id = config["model"]["label2id"]

    train_cfg = config["training"]

    if mlflow is not None:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

    logger.info("Chargement du dataset tokenizé depuis '%s'...", tokenized_data_dir)
    tokenized = load_from_disk(tokenized_data_dir)

    logger.info("Chargement du tokenizer depuis '%s'...", tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    logger.info("Chargement du modèle de base '%s'...", checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        num_train_epochs=train_cfg["num_train_epochs"],
        weight_decay=train_cfg["weight_decay"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        report_to=train_cfg["report_to"],
        push_to_hub=train_cfg["push_to_hub_during_training"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
    )

    if mlflow is not None:
        with mlflow.start_run(run_name=f"{checkpoint}-training"):
            mlflow.log_params({
                "learning_rate": float(train_cfg["learning_rate"]),
                "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
                "per_device_eval_batch_size": train_cfg["per_device_eval_batch_size"],
                "num_train_epochs": train_cfg["num_train_epochs"],
                "weight_decay": train_cfg["weight_decay"],
                "checkpoint": checkpoint,
                "num_labels": num_labels,
            })

            logger.info("Début de l'entraînement...")
            trainer.train()

            eval_metrics = trainer.evaluate(tokenized["validation"])
            mlflow.log_metrics({
                key: float(value) if isinstance(value, (int, float)) else float(str(value))
                for key, value in eval_metrics.items()
                if isinstance(value, (int, float, str))
            })

            logger.info("Sauvegarde du modèle final dans '%s'...", train_cfg["output_dir"])
            trainer.save_model(train_cfg["output_dir"])
            tokenizer.save_pretrained(train_cfg["output_dir"])
            mlflow.transformers.log_model(model=model, artifact_path="model", tokenizer=tokenizer)
            mlflow.log_artifact(train_cfg["output_dir"])
            return trainer

    logger.info("Début de l'entraînement...")
    trainer.train()

    logger.info("Sauvegarde du modèle final dans '%s'...", train_cfg["output_dir"])
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune le modèle de classification.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Chemin vers config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
