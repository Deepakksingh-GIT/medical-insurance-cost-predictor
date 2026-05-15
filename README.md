# 🏥 Medical Insurance Cost Predictor

An end-to-end Machine Learning project that predicts medical insurance charges based on personal health and demographic information.

Built using Python, Scikit-learn, and Streamlit.

---

## 📌 Project Overview

This project helps estimate medical insurance costs using machine learning regression models.

The application takes user inputs such as:

- Age
- BMI
- Number of children
- Smoking status
- Gender
- Region

and predicts estimated medical insurance charges.

---

## 🚀 Features

✅ Interactive Streamlit Web App  
✅ Multiple Regression Models  
✅ Real-time Insurance Cost Prediction  
✅ Feature Scaling and Encoding  
✅ Model Performance Comparison  
✅ Feature Importance Visualization  
✅ Clean User Interface  

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

---

## 📊 Machine Learning Models Used

### 1. Linear Regression
Baseline regression model for predicting insurance charges.

### 2. Ridge Regression
Uses L2 regularization to reduce overfitting.

### 3. Lasso Regression
Uses L1 regularization for feature selection.

---

## 📂 Dataset Information

Dataset contains:

- Age
- Sex
- BMI
- Children
- Smoker
- Region
- Charges (Target Variable)

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

- Removed duplicate records
- One-hot encoding for categorical variables
- Feature scaling using StandardScaler
- Train-test split for evaluation

---

## 📈 Evaluation Metrics

The models were evaluated using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

---

## 🖥️ Application Screenshots

### Home Page
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/2dd8b2a2-0f3e-4d0c-abf9-ae683b834e65" />

### Prediction Page
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/0be2c791-1b8a-4c92-9eeb-25c6ebfe6436" />

### Model Performance
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/88ae6f1f-a3eb-4a98-9064-098adcc4a66e" />
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/36b3c082-7644-4858-803c-1e64e8358ebb" />
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/cf3196e9-744a-4cfb-b429-1a2159a56e5c" />
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/62032787-8b68-43af-9267-bc4f81e50732" />



---

## ▶️ How to Run Locally

### Clone the repository

```bash
git clone https://github.com/your-username/medical-insurance-cost-predictor.git
```

### Navigate to project folder

```bash
cd medical-insurance-cost-predictor
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🌐 Future Improvements

- Deploy using Streamlit Cloud
- Add XGBoost and Random Forest models
- Add Docker support
- Improve UI design
- Add model saving/loading

---

## 👨‍💻 Author

Deepak Kumar Singh

Aspiring Data Analyst / Machine Learning Enthusiast
