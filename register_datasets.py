
import mlflow
import pandas as pd
from pathlib import Path

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("A3_IMDB_Sentiment_Monitoring")

for version in ["v1", "v2"]:
    path = Path("datasets") / f"imdb_{version}.csv"
    df = pd.read_csv(path)

    dataset = mlflow.data.from_pandas(
        df,
        source=str(path),
        name=f"imdb_{version}",
        targets="label"
    )

    with mlflow.start_run(run_name=f"Dataset_{version.upper()}"):
        mlflow.log_input(dataset, context="training")
        mlflow.log_param("dataset_version", version)
        mlflow.log_metric("review_count", len(df))
        mlflow.log_metric(
            "positive_ratio",
            float((df["label"] == 1).mean())
        )

    print(f"Dataset {version.upper()} registered: {len(df)} reviews")

print("Both datasets registered successfully!")
