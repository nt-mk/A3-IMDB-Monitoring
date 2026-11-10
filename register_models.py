
import mlflow
import joblib

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("A3_IMDB_Sentiment_Monitoring")

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

print("A1 model and TF-IDF loaded successfully!")
with mlflow.start_run(run_name="IMDb_Model_V1"):
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_metric("test_accuracy", 0.8815)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="imdb_model",
        registered_model_name="IMDb_Sentiment_Classifier"
    )

print("Model Version 1 registered successfully!")
