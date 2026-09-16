# Analyse de sentiment — DistilBERT (MLOps)

Pipeline MLOps pour le fine-tuning, l'évaluation et le déploiement d'un modèle
DistilBERT pour l'analyse de sentiment, entraîné sur le dataset
[`rotten_tomatoes`](https://huggingface.co/datasets/rotten_tomatoes).

Portage du notebook Colab d'origine vers un projet VSCode structuré, avec une
interface **Streamlit** (remplace le Gradio du prototype Colab).

## Structure du projet

```
.
├── config.yaml            # Configuration centrale du pipeline
├── Dockerfile              # Image pour l'app Streamlit
├── requirements.txt
├── README.md
└── src/
    ├── get_data.py         # Étape 1 : téléchargement du dataset
    ├── preprocess.py       # Étape 2 : tokenisation
    ├── train.py            # Étape 3 : fine-tuning
    ├── evaluate.py         # Étape 4 : évaluation sur le test set
    ├── deploy.py           # Étape 5 : publication sur le Hugging Face Hub
    └── app.py              # Étape 6 : interface Streamlit d'inférence
```

Chaque étape lit ses paramètres depuis `config.yaml`, pour ne jamais avoir de
valeurs en dur dans le code (dataset, checkpoint, hyperparamètres, chemins...).

## Installation

```bash
python -m venv venv
source venv/bin/activate        # sur Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Crée un fichier `.env` à la racine du projet à partir de `.env.example` :

```bash
copy .env.example .env
```

Puis mets ton token Hugging Face dans `.env` :

```env
HF_TOKEN=ton_token_hf
```

## Utilisation du pipeline

Exécuter les étapes dans l'ordre, depuis la racine du projet :

```bash
python src/get_data.py       # télécharge le dataset -> data/raw
python src/preprocess.py     # tokenize -> data/processed + models/tokenizer
python src/train.py          # fine-tune -> models/distilbert-sentiment
python src/evaluate.py       # évalue sur le test -> reports/eval_results.json
python src/deploy.py         # publie sur le Hub (nécessite HF_TOKEN)
```

Chaque script accepte un chemin de config personnalisé :

```bash
python src/train.py --config config.yaml
```

### Authentification au Hugging Face Hub (pour `deploy.py`)

Pour l’entraînement / deployment, tu peux utiliser l’une des méthodes suivantes :

```bash
export HF_TOKEN="votre_token_write"
```

ou bien :

```bash
huggingface-cli login
```

Pensez aussi à mettre à jour `hub.repo_id` dans `config.yaml` avec votre
propre namespace Hugging Face.

## Lancer l'application Streamlit

```bash
streamlit run src/app.py
```

Par défaut, le projet est configuré pour charger le modèle depuis le Hub
(`app.model_source: "hub"` dans `config.yaml`). Cela est préférable pour les
conteneurs et pour un partage plus propre. Si tu veux charger un modèle local,
remplace la valeur par `"local"`.

## Docker

Construis l'image :

```bash
docker build -t sentiment-app .
```

Puis lance le conteneur avec le token Hugging Face :

```bash
docker run -p 8501:8501 -e HF_TOKEN=ton_token_hf sentiment-app
```

L'app sera accessible sur http://localhost:8501

> Le projet est désormais configuré pour charger le modèle depuis le Hub dans
> le conteneur, ce qui évite d’intégrer les poids du modèle dans l’image.

## Différences par rapport au notebook Colab d'origine

- Le notebook monolithique est découpé en étapes indépendantes et réutilisables.
- Tous les paramètres (dataset, checkpoint, hyperparamètres, chemins, repo Hub)
  sont centralisés dans `config.yaml`.
- Chaque étape sauvegarde ses artefacts sur disque, pour permettre de relancer
  uniquement l'étape nécessaire (pas besoin de retélécharger ou retokenizer
  à chaque entraînement).
- L'interface de démonstration passe de **Gradio** à **Streamlit**.
