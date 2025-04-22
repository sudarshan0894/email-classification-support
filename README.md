# email-classification-support
Email Classifier API


# 📧 Email Classification and PII Detection API

A FastAPI and Gradio-powered application that classifies emails into predefined categories and masks Personally Identifiable Information (PII) such as names, phone numbers, credit card details, and more.

---

## 🚀 Features

- Detects and masks PII (email, phone, card numbers, name, DOB, etc.)
- Classifies emails into categories (like billing, technical, etc.)
- Offers both API endpoints (FastAPI) and a web UI (Gradio)
- Automatically trains an SVM model if no saved model is found

---

## 🧠 Tech Stack

- **Backend**: FastAPI
- **Machine Learning**: Scikit-learn (SVM), TfidfVectorizer
- **Frontend**: Gradio (for UI)
- **Others**: Regex (for PII masking), Joblib (for model saving/loading)

---

## 📁 Project Structure



Markdown

# Email Classification System for Support Team

This repository contains the code for an email classification system designed for a support team. The system categorizes incoming support emails and masks personally identifiable information (PII) before processing.

## Objective

The goal of this project is to build an email classification system that can:

1.  Classify support emails into predefined categories (e.g., Billing Issues, Technical Support, Account Management).
2.  Identify and mask various types of PII within the emails without using Large Language Models (LLMs).
3.  Expose this functionality as an API.

## Problem Statement

The system takes an email as input, masks sensitive PII, classifies the masked email, and returns the classification result along with the details of the masked entities.

## Repository Structure

├── api.py             # FastAPI application for the API endpoint
├── app.py             # Gradio application for a simple UI (optional for deployment)
├── combined_emails_with_natural_pii.csv # Dataset for training the classification model
├── model/
│   ├── classifier.pkl
│   └── vectorizer.pkl
├── models.py          # Functions for loading and training the classification model
├── README.md          # This file
├── requirements.txt   # List of Python dependencies
├── train_module.py    # Script to train the classification model
└── utils.py           # Utility functions for PII masking and output formatting

