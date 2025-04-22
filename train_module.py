import pandas as pd
from models import train_from_csv


def main():
    """
    Main function to initiate the SVM training pipeline.
    Reads data from a CSV file and trains a model using the imported train_from_csv function.
    """

    # Define the CSV file path
    csv_path = "combined_emails_with_natural_pii.csv"

    print("🔁 Training with SVM...")
    
    # Train the model from CSV data
    train_from_csv(csv_path)

    print("\n✅ Training complete with SVM!")


# Entry point
if __name__ == "__main__":
    main()
