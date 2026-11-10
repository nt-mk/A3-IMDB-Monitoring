
import re
import string
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import plotly.express as px
import streamlit as st
from prometheus_client import Gauge, start_http_server

st.set_page_config(
    page_title="IMDb Sentiment Monitoring",
    page_icon="🎬",
    layout="wide",
)

mlflow.set_tracking_uri("http://127.0.0.1:5001")

MODEL_NAME = "IMDb_Sentiment_Classifier"
EXPERIMENT_NAME = "A3_IMDB_Sentiment_Monitoring"


# ----------------------------------------
# Prometheus Monitoring
# ----------------------------------------

@st.cache_resource
def initialize_prometheus():
    review_count = Gauge(
        "imdb_review_count",
        "Number of IMDb reviews",
        ["dataset_version"],
    )

    positive_ratio = Gauge(
        "imdb_positive_ratio",
        "Ratio of positive IMDb reviews",
        ["dataset_version"],
    )

    model_accuracy = Gauge(
        "imdb_model_accuracy",
        "Test accuracy of IMDb sentiment models",
        ["model_version"],
    )

    start_http_server(8001)

    return review_count, positive_ratio, model_accuracy


IMDB_REVIEW_COUNT, IMDB_POSITIVE_RATIO, IMDB_MODEL_ACCURACY = initialize_prometheus()


# ----------------------------------------
# Streamlit Interface
# ----------------------------------------

st.title("IMDb Sentiment Analysis — A3")

st.caption(
    "Dataset Versioning • Model Registry • Predictions • Monitoring"
)

st.sidebar.header("Version Selection")

dataset_version = st.sidebar.selectbox(
    "Dataset Version",
    ["v1", "v2"]
)


# ----------------------------------------
# MLflow Model Registry
# ----------------------------------------

client = mlflow.MlflowClient()

try:
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    available_versions = sorted(
        {int(v.version) for v in versions}
    )

    if not available_versions:
        st.error("No registered model versions found.")
        st.stop()

    model_version = st.sidebar.selectbox(
        "Model Version",
        available_versions,
        index=0
    )

except Exception as e:
    st.error(
        f"Cannot connect to MLflow Model Registry: {e}"
    )
    st.stop()


# ----------------------------------------
# Dataset Overview
# ----------------------------------------

st.subheader("Dataset Overview")

dataset_path = Path("datasets") / f"imdb_{dataset_version}.csv"

if dataset_path.exists():

    df = pd.read_csv(dataset_path)

    col1, col2 = st.columns(2)

    col1.metric("Total Reviews", len(df))

    IMDB_REVIEW_COUNT.labels(
        dataset_version=dataset_version
    ).set(len(df))

    if "label" in df.columns:

        positive_ratio = (df["label"] == 1).mean()

        IMDB_POSITIVE_RATIO.labels(
            dataset_version=dataset_version
        ).set(float(positive_ratio))

        col2.metric(
            "Positive Reviews",
            f"{positive_ratio:.1%}"
        )

        counts = (
            df["label"]
            .map({0: "Negative", 1: "Positive"})
            .value_counts()
            .reset_index()
        )

        counts.columns = ["Sentiment", "Count"]

        fig = px.bar(
            counts,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            title=f"Dataset {dataset_version.upper()} Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.warning(
            "The selected dataset has no 'label' column."
        )

else:
    st.warning(
        f"Dataset not found: {dataset_path}"
    )


# ----------------------------------------
# Sentiment Prediction
# ----------------------------------------

st.subheader("Sentiment Prediction")

review = st.text_area(
    "Enter an IMDb movie review",
    placeholder="This movie was amazing!"
)

if st.button("Predict Sentiment"):

    if not review.strip():
        st.warning("Please enter a movie review.")

    else:
        try:
            import nltk

            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            from nltk.tokenize import word_tokenize

            model = mlflow.sklearn.load_model(
                f"models:/{MODEL_NAME}/{model_version}"
            )

            vectorizer = joblib.load("tfidf.pkl")

            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)

            stop_words = set(
                stopwords.words("english")
            )

            lemmatizer = WordNetLemmatizer()

            def preprocess_text(text):

                text = text.lower()

                text = re.sub(
                    r"<.*?>",
                    " ",
                    text
                )

                text = text.translate(
                    str.maketrans(
                        "",
                        "",
                        string.punctuation
                    )
                )

                text = re.sub(
                    r"\s+",
                    " ",
                    text
                ).strip()

                tokens = word_tokenize(text)

                tokens = [
                    lemmatizer.lemmatize(word)
                    for word in tokens
                    if word not in stop_words
                    and word.isalpha()
                ]

                return " ".join(tokens)

            processed = preprocess_text(review)

            features = vectorizer.transform(
                [processed]
            )

            prediction = int(
                model.predict(features)[0]
            )

            if prediction == 1:
                st.success("Positive Review")
            else:
                st.error("Negative Review")

            st.caption(
                f"Dataset: {dataset_version} | "
                f"Model Registry Version: {model_version}"
            )

        except Exception as e:
            st.error(
                f"Prediction error: {e}"
            )


# ----------------------------------------
# Model Performance Comparison
# ----------------------------------------

st.subheader("Model Performance Comparison")

try:

    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:

        st.warning(
            f"MLflow experiment not found: {EXPERIMENT_NAME}"
        )

    else:

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id]
        )

        if "tags.mlflow.runName" not in runs.columns:

            st.warning(
                "MLflow run names are unavailable."
            )

        else:

            model_runs = runs[
                runs["tags.mlflow.runName"].isin(
                    ["IMDb_Model_V1", "IMDb_Model_V2"]
                )
            ].copy()

            if model_runs.empty:

                st.warning(
                    "No V1 or V2 model runs found in MLflow."
                )

            else:

                model_runs["Model"] = model_runs[
                    "tags.mlflow.runName"
                ]

                # --------------------------------
                # Model Accuracy + Prometheus
                # --------------------------------

                if "metrics.test_accuracy" in model_runs.columns:

                    accuracy_runs = model_runs.dropna(
                        subset=["metrics.test_accuracy"]
                    ).copy()

                    accuracy_runs["metrics.test_accuracy"] = pd.to_numeric(
                        accuracy_runs["metrics.test_accuracy"],
                        errors="coerce"
                    )

                    accuracy_runs = accuracy_runs.dropna(
                        subset=["metrics.test_accuracy"]
                    )

                    if not accuracy_runs.empty:

                        if "start_time" in accuracy_runs.columns:

                            accuracy_runs = accuracy_runs.sort_values(
                                "start_time",
                                ascending=False
                            )

                        accuracy_runs = accuracy_runs.drop_duplicates(
                            subset=["Model"],
                            keep="first"
                        )

                        for _, row in accuracy_runs.iterrows():

                            IMDB_MODEL_ACCURACY.labels(
                                model_version=row["Model"]
                            ).set(
                                float(row["metrics.test_accuracy"])
                            )

                        fig = px.bar(
                            accuracy_runs,
                            x="Model",
                            y="metrics.test_accuracy",
                            color="Model",
                            title="Model Accuracy Comparison",
                            range_y=[0, 1]
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "No model accuracy values were logged."
                        )

                else:

                    st.warning(
                        "The test_accuracy metric is missing in MLflow."
                    )

                # --------------------------------
                # F1 Score Comparison
                # --------------------------------

                if "metrics.test_f1" in model_runs.columns:

                    f1_runs = model_runs.dropna(
                        subset=["metrics.test_f1"]
                    ).copy()

                    f1_runs["metrics.test_f1"] = pd.to_numeric(
                        f1_runs["metrics.test_f1"],
                        errors="coerce"
                    )

                    f1_runs = f1_runs.dropna(
                        subset=["metrics.test_f1"]
                    )

                    if not f1_runs.empty:

                        if "start_time" in f1_runs.columns:

                            f1_runs = f1_runs.sort_values(
                                "start_time",
                                ascending=False
                            )

                        f1_runs = f1_runs.drop_duplicates(
                            subset=["Model"],
                            keep="first"
                        )

                        fig_f1 = px.bar(
                            f1_runs,
                            x="Model",
                            y="metrics.test_f1",
                            color="Model",
                            title="Model F1 Score",
                            range_y=[0, 1]
                        )

                        st.plotly_chart(
                            fig_f1,
                            use_container_width=True
                        )

except Exception as e:

    st.warning(
        
        f"Metrics unavailable: {e}"
    )


st.caption(
    "A3 Data Mining Project | MLflow + Streamlit"
)
