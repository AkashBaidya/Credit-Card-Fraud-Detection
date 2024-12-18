import asyncio

# Ensure that an event loop is available
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Function to evaluate models
def evaluate_model(y_true, y_pred, y_prob):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)
    
    # AUROC Score
    auroc = roc_auc_score(y_true, y_prob)
    st.write(f"AUROC Score: {auroc:.4f}")
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, marker='.')
    ax.set_title("ROC Curve")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    st.pyplot(fig)

# Function for Model Training and Evaluation
def train_and_evaluate_model(X, y, model):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Fit model
    model.fit(X_train_resampled, y_train_resampled)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    evaluate_model(y_test, y_pred, y_prob)

# Models to use
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(),
}

# Start Streamlit App
st.title("Credit Card Fraud Detection")

# Create tabs
tab1, tab2 = st.tabs(["Dataset Analysis and Prediction", "Upload Dataset and Apply Models"])

# Tab 1: Dataset Analysis and Prediction
with tab1:
    st.header("Dataset Part 1 Analysis and Prediction")
    
    # Load dataset part1 (assuming you've already split and saved it)
    data_part1 = pd.read_csv('credit_card_transactions_part1.csv')
    st.write("Dataset Part 1 Overview:")
    st.write(data_part1.head())
    
    # Exploratory Data Analysis using Plotly
    st.write("Exploratory Data Analysis:")
    fig_city = px.histogram(data_part1, x='city', color='Class', barmode='group',
                            title="Fraudulent vs Non-Fraudulent Transactions by City")
    st.plotly_chart(fig_city)
    
    fig_time = px.histogram(data_part1, x='Time', color='Class', barmode='group',
                            title="Fraudulent Transactions by Time")
    st.plotly_chart(fig_time)
    
    fig_state = px.histogram(data_part1, x='State', color='Class', barmode='group',
                             title="Fraudulent Transactions by State")
    st.plotly_chart(fig_state)
    
    # Select a model to apply
    st.write("Select a model to predict fraud cases:")
    model_choice = st.selectbox("Choose Model", list(models.keys()))
    
    # Perform prediction
    if st.button("Train and Predict on Dataset Part 1"):
        X = data_part1.drop(columns=['Class'])
        y = data_part1['Class']
        selected_model = models[model_choice]
        train_and_evaluate_model(X, y, selected_model)

# Tab 2: Upload Dataset and Apply Models
with tab2:
    st.header("Upload a Dataset and Apply Models")
    
    # File uploader for new dataset
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load uploaded data
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Overview:")
        st.write(data.head())
        
        # Select a model to apply
        st.write("Select a model to predict fraud cases:")
        model_choice = st.selectbox("Choose Model (Upload Tab)", list(models.keys()))
        
        # Perform prediction on uploaded data
        if st.button("Train and Predict on Uploaded Dataset"):
            X = data.drop(columns=['Class'])
            y = data['Class']
            selected_model = models[model_choice]
            train_and_evaluate_model(X, y, selected_model)