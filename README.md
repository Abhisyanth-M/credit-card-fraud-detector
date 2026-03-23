# Credit Card Fraud Detector

A Machine Learning web app that analyses 284,807 real credit card transactions to detect fraud patterns using Random Forest classifier with 99.9% accuracy.

## Live Demo
https://huggingface.co/spaces/Abhisyanth-M/credit-card-fraud-detector

## Problem Statement
Credit card fraud costs banks and individuals billions of rupees annually. In India alone, banks lose over Rs 1,000 crores every year to fraudulent transactions. Most victims only discover fraud after it has already happened.

## Solution
An ML-powered fraud detection dashboard that analyses real transaction patterns, identifies which features indicate fraud, and evaluates model performance using industry-standard metrics.

## Features
- Transaction distribution visualisation — fraud vs legitimate breakdown
- Top 10 most important features for fraud detection
- Confusion matrix showing model performance
- Precision, Recall and Accuracy metrics
- Trained on 284,807 real anonymized credit card transactions

## Tech Stack
- Python
- Scikit-learn
- Random Forest Classifier
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit

## Dataset
- Source: Kaggle — Credit Card Fraud Detection by ULB Machine Learning Group
- Size: 284,807 transactions
- Fraud cases: 492 (0.17% of total)
- Features: 28 PCA-transformed features (V1-V28) + Amount + Time

## ML Model
- Algorithm: Random Forest Classifier
- Trees: 100
- Class weight: Balanced (to handle imbalanced dataset)
- Accuracy: 99.9%
- Fraud Precision: 100%
- Fraud Recall: 33.3%

## Key Concepts Demonstrated
- Imbalanced dataset handling using class_weight balanced
- Difference between Accuracy, Precision and Recall
- Feature importance analysis
- Confusion matrix interpretation

## How to Run Locally
```bash
git clone https://github.com/Abhisyanth-M/credit-card-fraud-detector
cd credit-card-fraud-detector
pip install -r requirements.txt
# Download creditcard.csv from Kaggle and place in project folder
streamlit run streamlit_app.py
```

## Limitations
- Dataset features V1-V28 are PCA transformed for confidentiality — real feature names are not available
- Recall of 33.3% means the model misses some fraud cases due to extreme class imbalance
- Model is trained on European cardholder data from 2013 — patterns may differ for Indian transactions

## GitHub
https://github.com/Abhisyanth-M/credit-card-fraud-detector
