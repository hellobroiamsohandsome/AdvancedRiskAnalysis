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
def show_train_model(X_train, y_train):
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

def show_bi_dashboard(data, processed_data):
    st.title("🔍 BI Dashboard")
    st.write("Explore key trends and insights from the credit data. Use the filters below to refine your view.")
    
    # Filters in an expander (only on BI Dashboard tab)
    with st.expander("Filters", expanded=True):
        selected_purpose = st.multiselect(
            "Select Loan Purpose",
            options=data["Purpose"].unique(),
            default=list(data["Purpose"].unique())
        )
        # Map risk values to descriptive labels
        risk_map = {0: "Good", 1: "Bad"}
        data["RiskLabel"] = data["Risk"].map(risk_map)
        selected_risk = st.multiselect(
            "Select Risk",
            options=["Good", "Bad"],
            default=["Good", "Bad"]
        )
    
    # Filter data according to the selections
    filtered_data = data[
        (data["Purpose"].isin(selected_purpose)) &
        (data["RiskLabel"].isin(selected_risk))
    ]
    
    # Chart 1: Distribution of Credit Amount
    fig1 = px.histogram(
        filtered_data, x="Credit amount", nbins=30, 
        title="Distribution of Credit Amount"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2: Age vs. Credit Amount Scatter Plot colored by Risk
    fig2 = px.scatter(
        filtered_data, x="Age", y="Credit amount", 
        color=filtered_data["RiskLabel"],
        title="Age vs. Credit Amount", 
        labels={"color": "Risk"}
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Chart 3: Bar Chart of Loan Purpose Frequency
    df_purpose = filtered_data['Purpose'].value_counts().reset_index()
    df_purpose.columns = ['Purpose', 'Count']  # Rename columns for clarity
    fig3 = px.bar(
        df_purpose, x='Purpose', y='Count',
        title="Loan Purpose Frequency", 
        labels={"Purpose": "Purpose", "Count": "Count"}
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Chart 4: Pie Chart of Savings Account Distribution
    fig4 = px.pie(
        filtered_data, names="Saving accounts", 
        title="Saving Accounts Distribution"
    )
    st.plotly_chart(fig4, use_container_width=True)
    
    # Additional Chart 5: Distribution of Age
    fig5 = px.histogram(
        filtered_data, x="Age", nbins=20, 
        title="Distribution of Age"
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    # Additional Chart 6: Box Plot of Credit Amount by Risk
    fig6 = px.box(
        filtered_data, x="RiskLabel", y="Credit amount",
        title="Credit Amount by Risk Level", 
        labels={"RiskLabel": "Risk Level", "Credit amount": "Credit Amount (€)"}
    )
    st.plotly_chart(fig6, use_container_width=True)
    
    # Additional Chart 7: Box Plot of Credit Amount by Loan Purpose
    fig7 = px.box(
        filtered_data, x="Purpose", y="Credit amount",
        title="Credit Amount by Loan Purpose", 
        labels={"Purpose": "Loan Purpose", "Credit amount": "Credit Amount (€)"}
    )
    st.plotly_chart(fig7, use_container_width=True)
    
    # Additional Chart 8: Treemap of Loan Purposes by Total Credit Amount
    fig8 = px.treemap(
        filtered_data, path=['Purpose'], values='Credit amount',
        title="Treemap of Loan Purposes by Total Credit Amount"
    )
    st.plotly_chart(fig8, use_container_width=True)
    
    # Additional Chart 9: Correlation Heatmap (using seaborn)
    # Select only the numeric columns
    corr = filtered_data.select_dtypes(include=[np.number]).corr()
    fig9, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    plt.title("Correlation Heatmap")
    st.pyplot(fig9)
    
    st.write("These interactive charts provide a dynamic view of your credit data. Adjust the filters above to explore different segments.")

def show_risk_prediction(data, processed_data):
    st.title("Risk Prediction")
    st.write("""
        This section predicts the probability that a loan application is high risk.
        The **Probability** is the model’s confidence that the application is high risk.
        For example, a probability of 14.33% means the model believes there is a 14.33% chance 
        the application is likely to default on his/her loan.
        
        Adjust the threshold sliders below to categorize the risk into:
        - Low Risk  
        - Medium Risk  
        - High Risk  
        - Severe High Risk
    """)
    # Set three threshold values with dependencies:
    medium_threshold = st.slider("Medium Risk Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    high_threshold = st.slider("High Risk Threshold", min_value=medium_threshold, max_value=1.0, value=0.6, step=0.05)
    severe_threshold = st.slider("Severe High Risk Threshold", min_value=high_threshold, max_value=1.0, value=0.8, step=0.05)
    
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
            
            # Split prediction into four distinct risk categories based on thresholds
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
    st.title("🔍 BI Dashboard")
    st.write("Explore key trends and insights from the credit data. Use the filters below to refine your view.")
    
    # Filters in an expander (only on BI Dashboard tab)
    with st.expander("Filters", expanded=True):
        selected_purpose = st.multiselect(
            "Select Loan Purpose",
            options=data["Purpose"].unique(),
            default=list(data["Purpose"].unique())
        )
        # Map risk values to descriptive labels
        risk_map = {0: "Good", 1: "Bad"}
        data["RiskLabel"] = data["Risk"].map(risk_map)
        selected_risk = st.multiselect(
            "Select Risk",
            options=["Good", "Bad"],
            default=["Good", "Bad"]
        )
    
    # Filter data according to the selections
    filtered_data = data[
        (data["Purpose"].isin(selected_purpose)) &
        (data["RiskLabel"].isin(selected_risk))
    ]
    
    # Chart 1: Distribution of Credit Amount
    fig1 = px.histogram(
        filtered_data, x="Credit amount", nbins=30, 
        title="Distribution of Credit Amount"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2: Age vs. Credit Amount Scatter Plot colored by Risk
    fig2 = px.scatter(
        filtered_data, x="Age", y="Credit amount", 
        color=filtered_data["RiskLabel"],
        title="Age vs. Credit Amount", 
        labels={"color": "Risk"}
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Chart 3: Bar Chart of Loan Purpose Frequency
    df_purpose = filtered_data['Purpose'].value_counts().reset_index()
    df_purpose.columns = ['Purpose', 'Count']  # Rename columns for clarity
    fig3 = px.bar(
        df_purpose, x='Purpose', y='Count',
        title="Loan Purpose Frequency", 
        labels={"Purpose": "Purpose", "Count": "Count"}
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Chart 4: Pie Chart of Savings Account Distribution
    fig4 = px.pie(
        filtered_data, names="Saving accounts", 
        title="Saving Accounts Distribution"
    )
    st.plotly_chart(fig4, use_container_width=True)
    
    # Additional Chart 5: Distribution of Age
    fig5 = px.histogram(
        filtered_data, x="Age", nbins=20, 
        title="Distribution of Age"
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    # Additional Chart 6: Box Plot of Credit Amount by Risk
    fig6 = px.box(
        filtered_data, x="RiskLabel", y="Credit amount",
        title="Credit Amount by Risk Level", 
        labels={"RiskLabel": "Risk Level", "Credit amount": "Credit Amount (€)"}
    )
    st.plotly_chart(fig6, use_container_width=True)
    
    # Additional Chart 7: Box Plot of Credit Amount by Loan Purpose
    fig7 = px.box(
        filtered_data, x="Purpose", y="Credit amount",
        title="Credit Amount by Loan Purpose", 
        labels={"Purpose": "Loan Purpose", "Credit amount": "Credit Amount (€)"}
    )
    st.plotly_chart(fig7, use_container_width=True)
    
    # Additional Chart 8: Treemap of Loan Purposes by Total Credit Amount
    fig8 = px.treemap(
        filtered_data, path=['Purpose'], values='Credit amount',
        title="Treemap of Loan Purposes by Total Credit Amount"
    )
    st.plotly_chart(fig8, use_container_width=True)
    
    # Additional Chart 9: Correlation Heatmap (using seaborn)
    # Select only the numeric columns
    corr = filtered_data.select_dtypes(include=[np.number]).corr()
    fig9, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    plt.title("Correlation Heatmap")
    st.pyplot(fig9)

# ---------------------------
# Toggle View Mode
# ---------------------------

def toggle_view_mode():
    # Let the user select view mode.
    mode = st.radio("Select View Mode", options=["Desktop Mode", "Mobile Mode"], index=0, key="view_mode")
    if mode == "Mobile Mode":
        # Mobile-friendly CSS: narrow container and smaller paddings.
        mobile_css = """
        <style>
        /* Adjust the main container to be full width and reduce padding */
        .main .block-container {
            max-width: 100% !important;
            padding: 1rem !important;
        }
        /* Optionally, reduce font-size for a mobile feel */
        body {
            font-size: 14px;
        }
        </style>
        """
        st.markdown(mobile_css, unsafe_allow_html=True)
    else:
        # Desktop-up CSS: wider container and larger paddings.
        desktop_css = """
        <style>
        .main .block-container {
            max-width: 1200px !important;
            padding: 2rem !important;
        }
        body {
            font-size: 16px;
        }
        </style>
        """
        st.markdown(desktop_css, unsafe_allow_html=True)


# ---------------------------
# Main App Function with Top Navigation Tabs
# ---------------------------
def main():
    st.set_page_config(page_title="Advanced Credit Risk Dashboard", layout="wide")
    
    # Allow user to toggle view mode.
    toggle_view_mode()
    
    data = load_data()
    processed_data = preprocess_data(data)
    
    # Use top horizontal tabs for navigation.
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