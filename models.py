# 📦 Import necessary libraries
import os
import joblib
import pandas as pd
from typing import Dict, Any
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import find_entities, prepare_output


# 🧹 Preprocess and mask text data
def preprocess_texts(texts):
    """
    Apply entity masking on a list of email texts.
    """
    return [find_entities(str(text))[1] for text in texts]


# 🧠 Train the SVM model and save it along with the vectorizer
def train_svm_model(X_train, y_train, save_dir='model'):
    """
    Train an SVM classifier with TF-IDF vectorization and save both to disk.
    """
    os.makedirs(save_dir, exist_ok=True)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        max_df=0.95,
        min_df=2
    )

    classifier = SVC(kernel='linear')

    X_train_vectorized = vectorizer.fit_transform(X_train)
    classifier.fit(X_train_vectorized, y_train)

    # Save model artifacts
    joblib.dump(vectorizer, os.path.join(save_dir, 'vectorizer.pkl'))
    joblib.dump(classifier, os.path.join(save_dir, 'classifier.pkl'))

    return vectorizer, classifier


# 📥 Load trained models from disk
def load_models(model_dir='model'):
    """
    Load vectorizer and classifier from a specified directory.
    """
    vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
    classifier = joblib.load(os.path.join(model_dir, 'classifier.pkl'))
    return vectorizer, classifier


# 📊 Train using a CSV file
def train_from_csv(csv_path, test_size=0.2, random_state=44):
    """
    Train the model from a CSV file containing 'email' and 'type' columns.
    """
    df = pd.read_csv(csv_path)

    if 'email' not in df.columns or 'type' not in df.columns:
        raise ValueError("CSV must contain 'email' and 'type' columns.")

    print("🔍 Class distribution:")
    print(df['type'].value_counts(normalize=True))

    X_raw = df['email'].astype(str).tolist()
    y = df['type'].values

    X = preprocess_texts(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    vectorizer, classifier = train_svm_model(X_train, y_train)

    # Evaluate the accuracy on test set
    X_test_vectorized = vectorizer.transform(X_test)
    accuracy = classifier.score(X_test_vectorized, y_test)
    print(f"✅ Model accuracy (SVM): {accuracy:.2f}")

    return vectorizer, classifier


# 📬 Predict category and format masked output
def predict_and_format(email_text: str, vectorizer, classifier) -> Dict[str, Any]:
    """
    Predicts the category of an email and returns the output in the required JSON format.
    """
    _, masked_email = find_entities(email_text)

    X_vectorized = vectorizer.transform([masked_email])
    predicted_category = classifier.predict(X_vectorized)[0]

    return prepare_output(email_text, predicted_category)


# 🚀 Entry point for testing prediction
if __name__ == '__main__':
    try:
        # Load trained models
        vectorizer, classifier = load_models('model')

        # Sample test input
        email_text = (
            "Hi, my name is John Doe. My card number is 1234-5678-9876-5432. "
            "Please process my payment."
        )

        result = predict_and_format(email_text, vectorizer, classifier)

        # Show result
        print(result)

    except Exception as e:
        print(f"❌ Error: {e}")