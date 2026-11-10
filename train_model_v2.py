

import re
import string
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Download NLTK resources if missing
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(resource, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Same text cleaning as A1
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Same preprocessing as A1
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word.isalpha()
    ]
    return " ".join(tokens)

# MLflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("A3_IMDB_Sentiment_Monitoring")

# Load Dataset V2 and original A1 TF-IDF
df = pd.read_csv("datasets/imdb_v2.csv")
tfidf = joblib.load("tfidf.pkl")

# Apply A1 preprocessing before TF-IDF
texts = (
    df["text"]
    .fillna("")
    .astype(str)
    .apply(clean_text)
    .apply(preprocess_text)
)

X = tfidf.transform(texts)
y = df["label"]

# Split Dataset V2
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model Version 2
model_v2 = LogisticRegression(C=1, max_iter=1000)
model_v2.fit(X_train, y_train)

# Evaluate Model Version 2
predictions = model_v2.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Register Model Version 2 in MLflow
with mlflow.start_run(run_name="IMDb_Model_V2"):
    mlflow.log_param("dataset_version", "v2")
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("vectorizer", "A1 TF-IDF")
    mlflow.log_param("training_samples", len(y_train))
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1", f1)

    mlflow.sklearn.log_model(
        sk_model=model_v2,
        name="imdb_model",
        registered_model_name="IMDb_Sentiment_Classifier"
    )

# Save Model V2 separately
joblib.dump(model_v2, "model_v2.pkl")

print(f"Model V2 Accuracy: {accuracy:.4f}")
print(f"Model V2 F1 Score: {f1:.4f}")
print("Model Version 2 registered successfully!")
