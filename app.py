! pip install streamlit imbalanced-learn scikit-learn pandas matplotlib seaborn qrcode pillow opendatasets joblib

# app.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import qrcode
from PIL import Image
import joblib
import opendatasets as od
import os

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------

def load_data():
    
    data = pd.read_csv('data/german_credit_data_rev.csv', index_col=0)
    return data

def preprocess_data(data):
    # Handle missing values
    data['Saving accounts'] = data['Saving accounts'].fillna('none')
    data['Checking account'] = data['Checking account'].fillna('none')
    
    # Feature engineering
    bins = [0, 25, 45, 60, 120]
    labels = ['0-25', '26-45', '46-60', '60+']
    data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels)
    
    # Convert risk to binary
    data['Risk'] = data['Risk'].map({'good': 0, 'bad': 1})
    
    return data

# ---------------------------
# Model Training
# ---------------------------

def train_model(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, 
                             scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

# ---------------------------
# Streamlit App
# ---------------------------

def main():
    st.set_page_config(page_title="Advanced Credit Risk Dashboard", layout="wide")
    
    # Load and preprocess data
    data = load_data()
    processed_data = preprocess_data(data)
    
    # Sidebar with QR Code
    st.sidebar.title("Dashboard Access")
    qr_img = generate_qr('http://localhost:8501')
    st.sidebar.image(qr_img, caption="Scan QR to Access")
    
    # Main content
    st.title("🚀 Advanced Credit Risk Analysis Dashboard")
    
    # Data Preprocessing Section
    with st.expander("Data Preprocessing Details"):
        st.subheader("Data Cleaning Steps")
        st.write("""
        - Handled missing values in savings/checking accounts
        - Created age groups for better risk segmentation
        - Encoded categorical variables using label encoding
        - Applied SMOTE to handle class imbalance
        """)
        st.write("Processed Data Preview:", processed_data.head())
    
    # Model Training Section
    if st.sidebar.button("Train Model"):
        with st.spinner("Training Advanced Risk Model..."):
            # Split data
            X = processed_data.drop(['Risk', 'Age'], axis=1)
            y = processed_data['Risk']
            
            # Label encoding
            label_encoders = {}
            categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 
                              'Checking account', 'Purpose', 'AgeGroup']
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42)
            
            # Handle class imbalance
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            
            # Train model
            model = train_model(X_res, y_res)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Save artifacts
            joblib.dump(model, 'model.pkl')
            joblib.dump(label_encoders, 'label_encoders.pkl')
            joblib.dump({'bins': [0, 25, 45, 60, 120], 
                       'labels': ['0-25', '26-45', '46-60', '60+']}, 
                      'age_params.pkl')
            
            st.success("Model trained successfully!")
            
            # Show metrics
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**ROC Curve**")
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2)
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                st.pyplot(plt)
                
            with col2:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                st.pyplot(plt)
                
            st.write(f"**ROC AUC Score:** {roc_auc_score(y_test, y_proba):.2f}")
            st.write("**Classification Report:**")
            st.code(classification_report(y_test, y_pred))
    
    # Prediction Interface
    st.sidebar.header("Risk Prediction")
    if os.path.exists('model.pkl'):
        model = joblib.load('model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        age_params = joblib.load('age_params.pkl')
        
        # Input form
        with st.sidebar.form("prediction_form"):
            st.subheader("Loan Application Details")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            credit_amount = st.number_input("Credit Amount (€)", min_value=0, value=5000)
            duration = st.number_input("Duration (months)", min_value=1, value=12)
            purpose = st.selectbox("Purpose", data['Purpose'].unique())
            savings = st.selectbox("Savings Account", data['Saving accounts'].unique())
            
            submitted = st.form_submit_button("Assess Risk")
            
        if submitted:
            # Create age group
            age_group = pd.cut([age], bins=age_params['bins'], 
                             labels=age_params['labels'])[0]
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'Sex': [data['Sex'].mode()[0]],
                'Job': [data['Job'].mode()[0]],
                'Housing': [data['Housing'].mode()[0]],
                'Saving accounts': [savings],
                'Checking account': [data['Checking account'].mode()[0]],
                'Purpose': [purpose],
                'Credit amount': [credit_amount],
                'Duration': [duration],
                'AgeGroup': [age_group]
            })
            
        
            # Encode categorical variables
            for col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
            
            # Add these lines to ensure the column order matches training:
            expected_columns = processed_data.drop(['Risk', 'Age'], axis=1).columns
            input_data = input_data[expected_columns]
        

                
            # Predict
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]
            
            # Display result
            if prediction[0] == 1:
                st.sidebar.error(f"High Risk Alert! (Probability: {probability:.2%})")
            else:
                st.sidebar.success(f"Low Risk (Probability: {probability:.2%})")
                
            st.sidebar.write("Key Factors Contributing to Risk:")
            feature_importance = pd.Series(model.named_steps['classifier'].feature_importances_, 
                                index=expected_columns)
            top_features = feature_importance.nlargest(3)
            for feat, imp in top_features.items():
                st.sidebar.write(f"- {feat}: {imp:.2f}")

def generate_qr(url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save("qr_code.png")
    return Image.open("qr_code.png")

if __name__ == "__main__":
    main()
