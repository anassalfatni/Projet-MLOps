"""
get_data.py
-----------
Télécharge le dataset (rotten_tomatoes par défaut) depuis le Hugging Face Hub
et le sauvegarde sur disque au format Arrow, pour que les étapes suivantes
du pipeline n'aient pas à retélécharger les données à chaque exécution.

Usage:
    python src/get_data.py
    python src/get_data.py --config config.yaml
"""

import argparse
import logging
from pathlib import Path

import yaml
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_data(config: dict) -> None:
    dataset_name = config["data"]["dataset_name"]
    raw_data_dir = Path(config["data"]["raw_data_dir"])

    logger.info("Téléchargement du dataset '%s'...", dataset_name)
    dataset = load_dataset(dataset_name)
    logger.info("Dataset chargé : %s", dataset)

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(raw_data_dir))
    logger.info("Dataset brut sauvegardé dans '%s'", raw_data_dir)


def main():
    parser = argparse.ArgumentParser(description="Télécharge et sauvegarde le dataset brut.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Chemin vers config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    get_data(config)


if __name__ == "__main__":
    main()
