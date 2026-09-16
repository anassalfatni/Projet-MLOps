"""
deploy.py
---------
Publie le modèle fine-tuné (training.output_dir) sur le Hugging Face Hub,
sous hub.repo_id.

Prérequis :
    - Être authentifié : soit via `huggingface-cli login`, soit en exportant
      la variable d'environnement HF_TOKEN avant d'exécuter ce script.

Usage:
    python src/deploy.py
    python src/deploy.py --config config.yaml
"""

import argparse
import logging
import os
from pathlib import Path

import yaml
from huggingface_hub import HfApi, login, whoami
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env_file(env_path: str | None = None) -> None:
    env_file = Path(env_path) if env_path else Path(__file__).resolve().parents[1] / ".env"
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def deploy(config: dict) -> None:
    load_env_file()

    model_dir = config["training"]["output_dir"]
    repo_id = config["hub"]["repo_id"]

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        logger.info("Authentification via la variable d'environnement HF_TOKEN.")
        login(token=hf_token)
    else:
        logger.info(
            "Aucune variable d'environnement HF_TOKEN détectée. "
            "Vérification d'une session Hugging Face déjà connectée..."
        )

    try:
        user = whoami()
        logger.info("Authentification Hugging Face OK pour l'utilisateur '%s'.", user["name"])
    except Exception as exc:
        raise RuntimeError(
            "Authentification Hugging Face absente. "
            "Exécutez `huggingface-cli login` ou définissez la variable d'environnement HF_TOKEN avant de lancer deploy.py."
        ) from exc

    logger.info("Chargement du modèle et du tokenizer depuis '%s'...", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    logger.info("Publication sur le Hub : '%s'...", repo_id)
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)

    model.push_to_hub(repo_id, token=hf_token)
    tokenizer.push_to_hub(repo_id, token=hf_token)

    logger.info("Modèle publié avec succès sur https://huggingface.co/%s", repo_id)


def main():
    parser = argparse.ArgumentParser(description="Publie le modèle sur le Hugging Face Hub.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Chemin vers config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    deploy(config)


if __name__ == "__main__":
    main()
