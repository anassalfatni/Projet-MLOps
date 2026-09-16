"""
app.py
------
Interface Streamlit pour tester le modèle d'analyse de sentiment.

Le modèle est chargé soit :
  - depuis le disque (config['app']['model_source'] == "local"),
    à partir de config['training']['output_dir']
  - depuis le Hugging Face Hub (config['app']['model_source'] == "hub"),
    à partir de config['hub']['repo_id']

Usage:
    streamlit run src/app.py
"""

import yaml
import streamlit as st
from transformers import pipeline

try:
    import mlflow
except ImportError:
    mlflow = None


@st.cache_resource(show_spinner="Chargement du modèle...")
def load_classifier(model_source: str, local_dir: str, hub_repo_id: str):
    model_path = local_dir if model_source == "local" else hub_repo_id
    return pipeline("sentiment-analysis", model=model_path)


def log_inference(model_source: str, text: str, label: str, score: float):
    if mlflow is None:
        return

    try:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("sentiment-analysis-inference")
        with mlflow.start_run(run_name="inference"):
            mlflow.log_param("model_source", model_source)
            mlflow.log_param("text_length", len(text))
            mlflow.log_metric("prediction_score", float(score))
            mlflow.log_dict({"label": label, "score": float(score), "text": text}, "prediction.json")
    except Exception:
        pass


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    st.set_page_config(page_title="Analyse de sentiment", page_icon="🎬", layout="centered")

    config = load_config()
    model_source = config["app"]["model_source"]
    local_dir = config["training"]["output_dir"]
    hub_repo_id = config["hub"]["repo_id"]

    st.title("🎬 Analyse de sentiment")
    st.write("Analysez le sentiment d'un texte avec le modèle DistilBERT fine-tuné.")

    with st.sidebar:
        st.header("Configuration")
        st.write(f"**Source du modèle :** `{model_source}`")
        st.write(f"**Chemin/Repo :** `{local_dir if model_source == 'local' else hub_repo_id}`")

    classifier = load_classifier(model_source, local_dir, hub_repo_id)

    text = st.text_area(
        "Texte à analyser",
        placeholder="Écrivez une phrase, par exemple : 'This movie was absolutely fantastic!'",
        height=120,
    )

    if st.button("Analyser", type="primary"):
        if not text.strip():
            st.warning("Veuillez entrer un texte.")
        else:
            with st.spinner("Analyse en cours..."):
                result = classifier(text)[0]

            label = result["label"]
            score = result["score"]

            if label.upper() == "POSITIVE":
                st.success(f"**Sentiment : {label}** (confiance : {score:.2%})")
            else:
                st.error(f"**Sentiment : {label}** (confiance : {score:.2%})")

            log_inference(model_source, text, label, score)

            st.progress(score)
            st.json({"label": label, "score": round(score, 4)})


if __name__ == "__main__":
    main()
