import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳")
st.title("Credit Card Fraud Detector")
st.markdown("Analysing 284,807 real credit card transactions to detect fraud patterns using Machine Learning.")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("creditcard.csv")
    sample = df.sample(n=10000, random_state=42)
    X = sample.drop("Class", axis=1)
    y = sample["Class"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, scaler, accuracy, cm, report, feature_importance, df

model, scaler, accuracy, cm, report, feature_importance, df = load_and_train()

# Sidebar
st.sidebar.markdown("## Model Info")
st.sidebar.markdown("**Algorithm:** Random Forest")
st.sidebar.markdown(f"**Accuracy:** {accuracy * 100:.1f}%")
st.sidebar.markdown(f"**Total transactions:** {len(df):,}")
st.sidebar.markdown(f"**Fraud transactions:** {df['Class'].sum():,}")
st.sidebar.markdown(f"**Fraud rate:** {df['Class'].mean()*100:.2f}%")

# Section 1 - Fraud Distribution
st.header("Transaction Distribution")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    labels = ["Legitimate", "Fraud"]
    sizes = [df["Class"].value_counts()[0], df["Class"].value_counts()[1]]
    colors = ["#2ecc71", "#e74c3c"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.2f%%", startangle=90)
    ax.set_title("Fraud vs Legitimate Transactions")
    st.pyplot(fig)

with col2:
    st.metric("Total Transactions", f"{len(df):,}")
    st.metric("Legitimate", f"{df['Class'].value_counts()[0]:,}")
    st.metric("Fraud", f"{df['Class'].value_counts()[1]:,}")
    st.metric("Fraud Rate", f"{df['Class'].mean()*100:.2f}%")

# Section 2 - Feature Importance
st.header("Top 10 Most Important Features")
fig2, ax2 = plt.subplots(figsize=(10, 5))
feature_importance.head(10).plot(kind="bar", color="#3498db", ax=ax2)
ax2.set_title("Feature Importance — Which patterns matter most for fraud detection")
ax2.set_xlabel("Feature")
ax2.set_ylabel("Importance Score")
plt.tight_layout()
st.pyplot(fig2)

# Section 3 - Confusion Matrix
st.header("Model Performance")
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"], ax=ax3)
ax3.set_title("Confusion Matrix")
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
plt.tight_layout()
st.pyplot(fig3)

# Section 4 - Model Metrics
st.header("Detection Metrics")
col3, col4, col5 = st.columns(3)
col3.metric("Accuracy", f"{accuracy*100:.1f}%")
col4.metric("Fraud Precision", f"{report['1']['precision']*100:.1f}%")
col5.metric("Fraud Recall", f"{report['1']['recall']*100:.1f}%")