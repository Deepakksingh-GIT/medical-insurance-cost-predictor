import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

# Page configuration
st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="wide")

# Title and description
st.title("🏥 Medical Insurance Cost Predictor")
st.write("Predict your medical insurance charges based on personal health and demographic information.")

# Load data and train models
@st.cache_resource
def load_and_train_models():
    # Load data
    df = pd.read_csv(r'C:\Users\user\Desktop\medical insurance cost predict- project\data\medical_insurance.csv')
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
    
    # Separate features and target
    x = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']
    
    # Scale features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
    
    # Train Linear Regression
    linreg = LinearRegression()
    linreg.fit(x_train, y_train)
    
    # Train Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train, y_train)
    
    # Train Lasso Regression
    lasso = LassoCV(cv=5)
    lasso.fit(x_train, y_train)
    
    # Store model info
    feature_names = x.columns.tolist()
    
    return {
        'linreg': linreg,
        'ridge': ridge,
        'lasso': lasso,
        'scaler': scaler,
        'feature_names': feature_names,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'original_data': df_encoded
    }

# Load models
models_data = load_and_train_models()
linreg = models_data['linreg']
ridge = models_data['ridge']
lasso = models_data['lasso']
scaler = models_data['scaler']
feature_names = models_data['feature_names']
x_test = models_data['x_test']
y_test = models_data['y_test']
y_train = models_data['y_train']
x_train = models_data['x_train']

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a section:", ["Prediction", "Model Performance"])

if page == "Prediction":
    st.header("Make a Prediction")
    
    # Create columns for input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", min_value=18, max_value=100, value=30)
        bmi = st.slider("BMI (Body Mass Index)", min_value=10.0, max_value=55.0, value=25.0, step=0.1)
        children = st.selectbox("Number of Children", options=[0, 1, 2, 3, 4, 5])
    
    with col2:
        sex = st.selectbox("Sex", options=["Female", "Male"])
        smoker = st.selectbox("Smoker", options=["No", "Yes"])
    
    with col3:
        region = st.selectbox("Region", options=["Northeast", "Northwest", "Southeast", "Southwest"])
    
    # Prepare input data
    if st.button("Predict Cost", use_container_width=True):
        # Create a dataframe with user input
        user_data = {
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex': [sex.lower()],
            'smoker': [smoker.lower()],
            'region': [region.lower()]
        }
        
        # Encode categorical variables to match training data
        user_df = pd.DataFrame(user_data)
        user_df_encoded = pd.get_dummies(user_df, columns=['sex', 'smoker', 'region'], drop_first=True)
        
        # Ensure all columns are present (add missing ones with 0)
        for col in feature_names:
            if col not in user_df_encoded.columns:
                user_df_encoded[col] = 0
        
        # Reorder columns to match training data
        user_df_encoded = user_df_encoded[feature_names]
        
        # Scale the input
        user_scaled = scaler.transform(user_df_encoded)
        
        # Make predictions
        pred_linreg = linreg.predict(user_scaled)[0]
        pred_ridge = ridge.predict(user_scaled)[0]
        pred_lasso = lasso.predict(user_scaled)[0]
        
        # Average prediction
        avg_pred = (pred_linreg + pred_ridge + pred_lasso) / 3
        
        # Display results
        st.success("Prediction Complete!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Linear Regression", f"${pred_linreg:,.2f}")
        with col2:
            st.metric("Ridge Regression", f"${pred_ridge:,.2f}")
        with col3:
            st.metric("Lasso Regression", f"${pred_lasso:,.2f}")
        with col4:
            st.metric("Average Prediction", f"${avg_pred:,.2f}", delta=None)
        
        # Input summary
        st.subheader("Input Summary")
        summary_data = {
            'Parameter': ['Age', 'BMI', 'Children', 'Sex', 'Smoker', 'Region'],
            'Value': [age, f"{bmi:.1f}", children, sex, smoker, region]
        }
        st.table(pd.DataFrame(summary_data))

elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    # Calculate metrics for all models
    def calculate_metrics(model, x_train, x_test, y_train, y_test):
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        return {
            'RMSE (Train)': rmse_train,
            'RMSE (Test)': rmse_test,
            'MAE (Train)': mae_train,
            'MAE (Test)': mae_test,
            'R² (Train)': r2_train,
            'R² (Test)': r2_test
        }
    
    metrics_lr = calculate_metrics(linreg, x_train, x_test, y_train, y_test)
    metrics_ridge = calculate_metrics(ridge, x_train, x_test, y_train, y_test)
    metrics_lasso = calculate_metrics(lasso, x_train, x_test, y_train, y_test)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Linear Regression': metrics_lr,
        'Ridge Regression': metrics_ridge,
        'Lasso Regression': metrics_lasso
    }).round(4)
    
    st.subheader("Performance Comparison")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Highlight best model for each metric
    st.subheader("Best Models by Metric")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_rmse = comparison_df.loc['RMSE (Test)'].idxmin()
        st.metric("Best RMSE (Test)", best_rmse)
    
    with col2:
        best_mae = comparison_df.loc['MAE (Test)'].idxmin()
        st.metric("Best MAE (Test)", best_mae)
    
    with col3:
        best_r2 = comparison_df.loc['R² (Test)'].idxmax()
        st.metric("Best R² (Test)", best_r2)
    
    # Model details
    st.subheader("Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Linear Regression**")
        st.info("Baseline linear model that assumes a linear relationship between features and target.")
    
    with col2:
        st.write("**Ridge Regression**")
        st.info("Linear regression with L2 regularization to reduce overfitting and multicollinearity.")
    
    with col3:
        st.write("**Lasso Regression**")
        st.info("Linear regression with L1 regularization for feature selection and sparsity.")
    
    # Feature importance (for Linear Regression)
    st.subheader("Feature Importance (Linear Regression Coefficients)")
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': linreg.coef_
    }).sort_values(by='Coefficient', key=lambda x: x.abs(), ascending=False)
    
    st.bar_chart(coef_df.set_index('Feature')['Coefficient'])
    
    with st.expander("View Coefficient Details"):
        st.dataframe(coef_df.sort_values(by='Coefficient', key=lambda x: x.abs(), ascending=False).reset_index(drop=True))

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
Medical Insurance Cost Prediction Model | Built with Streamlit and Scikit-learn
</div>
""", unsafe_allow_html=True)
