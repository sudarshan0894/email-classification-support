import pandas as pd
from models import train_from_csv

def main():
    # Define the CSV file path
    csv_path = "combined_emails_with_natural_pii.csv"
    
    print("🔁 Training with SVM...")
    train_from_csv(csv_path)

    print("\n✅ Training complete with SVM!")

if __name__ == "__main__":
    main()


