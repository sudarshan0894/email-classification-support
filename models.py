import os
import joblib
import pandas as pd
from typing import Dict, Any
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from utils import find_entities, prepare_output


def preprocess_texts(texts):
    """
    Apply entity masking on a list of email texts.
    
    Args:
        texts (list): List of email texts to preprocess.

    Returns:
        list: Preprocessed text with masked entities.
    """
    return [find_entities(str(text))[1] for text in texts]


def train_svm_model(X_train, y_train, save_dir='model'):
    """
    Train an SVM classifier with TF-IDF vectorization and save both to disk.
    
    Args:
        X_train (list): Training data (email texts).
        y_train (list): Labels for training data.
        save_dir (str): Directory to save the model and vectorizer.

    Returns:
        tuple: The vectorizer and classifier.
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


def load_models(model_dir='model'):
    """
    Load vectorizer and classifier from a specified directory.
    
    Args:
        model_dir (str): Directory to load the model and vectorizer.

    Returns:
        tuple: Loaded vectorizer and classifier.
    """
    vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
    classifier = joblib.load(os.path.join(model_dir, 'classifier.pkl'))
    return vectorizer, classifier


def train_from_csv(csv_path, test_size=0.2, random_state=44):
    """
    Train the model from a CSV file containing 'email' and 'type' columns.
    
    Args:
        csv_path (str): Path to the CSV file.
        test_size (float): Proportion of the data to be used for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: The trained vectorizer and classifier.
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

    # Evaluate the accuracy on the test set
    X_test_vectorized = vectorizer.transform(X_test)
    accuracy = classifier.score(X_test_vectorized, y_test)
    print(f"✅ Model accuracy (SVM): {accuracy:.2f}")

    # Evaluate additional metrics
    y_pred = classifier.predict(X_test_vectorized)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return vectorizer, classifier


def predict_and_format(email_text: str, vectorizer, classifier) -> Dict[str, Any]:
    """
    Predict the category of an email and returns the output in the required JSON format.
    
    Args:
        email_text (str): The email text to be classified.
        vectorizer: The trained vectorizer.
        classifier: The trained classifier.

    Returns:
        dict: Formatted prediction output.
    """
    _, masked_email = find_entities(email_text)

    X_vectorized = vectorizer.transform([masked_email])
    predicted_category = classifier.predict(X_vectorized)[0]

    return prepare_output(email_text, predicted_category)


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
