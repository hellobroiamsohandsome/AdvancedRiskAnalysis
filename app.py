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
import plotly.express as px


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

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

# ---------------------------
# Page Definitions
# ---------------------------
def show_home(data, processed_data):
    st.title("🚀 Advanced Credit Risk Analysis Dashboard")
    st.write("Welcome to the credit risk dashboard. Use the tabs above to navigate between pages.")
    st.subheader("Data Preprocessing Details")
    st.write("""
        - Handled missing values in savings/checking accounts  
        - Created age groups for better risk segmentation  
        - Encoded categorical variables with label encoding  
        - Applied SMOTE to handle class imbalance  
    """)
    st.write("Processed Data Preview:")
    st.dataframe(processed_data.head())
    
    # Display a small QR code for access
    qr_img = generate_qr('https://advancedriskanalysis-frp3xdyvnbex8a4rdhqk8j.streamlit.app/')
    st.image(qr_img, caption="Scan QR to Access", width=150)

def show_train_model(data, processed_data):
    st.title("Train Model")
    with st.spinner("Training Advanced Risk Model..."):
        X = processed_data.drop(['Risk', 'Age'], axis=1)
        y = processed_data['Risk']

        # Label encoding for categorical columns
        label_encoders = {}
        categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 
                              'Checking account', 'Purpose', 'AgeGroup']
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        # Split data with reproducibility
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)

        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        # Train model using grid search
        model = train_model(X_res, y_res)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Save artifacts for later prediction
        joblib.dump(model, 'model.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        joblib.dump({'bins': [0, 25, 45, 60, 120],
                     'labels': ['0-25', '26-45', '46-60', '60+']},
                    'age_params.pkl')

    st.success("Model trained successfully!")
    st.subheader("Model Performance on Test Data")
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

def show_risk_prediction(data, processed_data):
    st.title("Risk Prediction")
    st.write("""
        This section predicts the probability that a loan application is high risk.
        The **Probability** is the model’s confidence that the application is high risk.
        For example, a probability of 14.33% means the model believes there is a 14.33% chance 
        the application is high risk.
        
        Adjust the threshold sliders below to categorize the risk into:
        - Low Risk  
        - Medium Risk  
        - High Risk  
        - Severe High Risk
    """)
    # Set three threshold values
    medium_threshold = st.slider("Medium Risk Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    high_threshold = st.slider("High Risk Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    severe_threshold = st.slider("Severe High Risk Threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
    
    with st.form("prediction_form"):
        st.subheader("Loan Application Details")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        credit_amount = st.number_input("Credit Amount (€)", min_value=0, value=5000)
        duration = st.number_input("Duration (months)", min_value=1, value=12)
        purpose = st.selectbox("Purpose", data['Purpose'].unique())
        savings = st.selectbox("Savings Account", data['Saving accounts'].unique())
        submitted = st.form_submit_button("Assess Risk")
    
    if submitted:
        if os.path.exists('model.pkl'):
            model = joblib.load('model.pkl')
            label_encoders = joblib.load('label_encoders.pkl')
            age_params = joblib.load('age_params.pkl')
            
            age_group = pd.cut([age], bins=age_params['bins'], labels=age_params['labels'])[0]
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
            
            for col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
            
            expected_columns = processed_data.drop(['Risk', 'Age'], axis=1).columns
            input_data = input_data[expected_columns]
            
            probability = model.predict_proba(input_data)[0][1]
            
            # Split prediction into four distinct risk categories
            if probability < medium_threshold:
                risk_level = "Low Risk"
            elif probability < high_threshold:
                risk_level = "Medium Risk"
            elif probability < severe_threshold:
                risk_level = "High Risk"
            else:
                risk_level = "Severe High Risk"
            
            if risk_level == "Low Risk":
                st.success(f"{risk_level} (Probability: {probability:.2%})")
            elif risk_level == "Medium Risk":
                st.warning(f"{risk_level} (Probability: {probability:.2%})")
            else:
                st.error(f"{risk_level} (Probability: {probability:.2%})")
            
            st.write("Key Factors Contributing to Risk:")
            feature_importance = pd.Series(model.named_steps['classifier'].feature_importances_, index=expected_columns)
            top_features = feature_importance.nlargest(3)
            for feat, imp in top_features.items():
                st.write(f"- {feat}: {imp:.2f}")
        else:
            st.error("Model not found! Please train the model first.")

# ---------------------------
# Auxiliary Functions
# ---------------------------
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

# ---------------------------
# Shiny Dashboard
# ---------------------------

def show_bi_dashboard(data, processed_data):
    # Sidebar filters for dynamic dashboard
    st.sidebar.header("Dashboard Filters")
    selected_purpose = st.sidebar.multiselect(
        "Select Loan Purpose",
        options=data["Purpose"].unique(),
        default=list(data["Purpose"].unique())
    )
    
    # Map numeric risk column to descriptive labels
    risk_map = {0: "Good", 1: "Bad"}
    data["RiskLabel"] = data["Risk"].map(risk_map)
    selected_risk = st.sidebar.multiselect(
        "Select Risk",
        options=["Good", "Bad"],
        default=["Good", "Bad"]
    )
    
    # Filter data based on sidebar selections
    filtered_data = data[
        (data["Purpose"].isin(selected_purpose)) &
        (data["RiskLabel"].isin(selected_risk))
    ]
    
    st.title("🔍 BI Dashboard")
    st.write("Explore key trends and insights from the credit data. Use the filters on the sidebar to refine your view.")
    
    # Example 1: Distribution of Credit Amount
    fig1 = px.histogram(filtered_data, x="Credit amount", nbins=30, 
                        title="Distribution of Credit Amount")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Example 2: Age vs. Credit Amount Scatter Plot colored by Risk
    fig2 = px.scatter(filtered_data, x="Age", y="Credit amount", 
                      color=filtered_data["RiskLabel"],
                      title="Age vs. Credit Amount", labels={"color": "Risk"})
    st.plotly_chart(fig2, use_container_width=True)
    
    # Example 3: Bar Chart of Purpose Count
    df_purpose = filtered_data['Purpose'].value_counts().reset_index()
    df_purpose.columns = ['Purpose', 'Count']  # Rename columns for clarity
    fig3 = px.bar(df_purpose, x='Purpose', y='Count',
                  title="Loan Purpose Frequency", 
                  labels={"Purpose": "Purpose", "Count": "Count"})
    st.plotly_chart(fig3, use_container_width=True)
    
    # Example 4: Pie Chart of Savings Account Distribution
    fig4 = px.pie(filtered_data, names="Saving accounts", 
                  title="Saving Accounts Distribution")
    st.plotly_chart(fig4, use_container_width=True)
    
    st.write("These interactive charts provide a dynamic view of your credit data. Adjust the filters in the sidebar to explore different segments.")


# ---------------------------
# Main App Function with Top Navigation Tabs
# ---------------------------
def main():
    st.set_page_config(page_title="Advanced Credit Risk Dashboard", layout="wide")
    
    data = load_data()
    processed_data = preprocess_data(data)
    
    # Use top horizontal tabs for navigation, user-friendly for mobile devices.
    tabs = st.tabs(["Home", "Train Model", "Risk Prediction", "BI Dashboard"])
    with tabs[0]:
        show_home(data, processed_data)
    with tabs[1]:
        show_train_model(data, processed_data)
    with tabs[2]:
        show_risk_prediction(data, processed_data)
    with tabs[3]:
        show_bi_dashboard(data, processed_data)

if __name__ == "__main__":
    main()