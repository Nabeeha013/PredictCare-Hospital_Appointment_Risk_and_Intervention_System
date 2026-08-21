# PredictCare - Hospital Appointment Risk and Intervention System

**PredictCare** is a machine learning-based hospital appointment no-show prediction system designed to help healthcare staff identify patients who may be at higher risk of missing their appointments.

## 🚀 Live Application

👉 **[Open PredictCare](https://predictcare-ai.streamlit.app/)**

The application is deployed with Streamlit and can be accessed directly.

## 🎯 Project Objective

Hospital appointment no-shows can lead to wasted resources, empty appointment slots, and difficulties in managing healthcare services.

PredictCare aims to support hospitals by:

* Predicting whether a patient is likely to miss an appointment
* Identifying patients with higher no-show risk
* Providing easy-to-understand risk levels
* Supporting individual and batch predictions
* Providing model performance and insights

## ✨ Features

### 📊 Dashboard

Provides an overview of appointment data and prediction insights.

### 👤 Single Patient Prediction

Enter information for an individual patient and appointment to receive a prediction, no-show probability, and risk level.

### 📁 Batch Prediction

Upload multiple patient records through an Excel/CSV file and generate predictions for multiple appointments.

### ⚡ Quick Prediction

Provides a simplified interface for quickly generating an appointment prediction.

### 📈 Model Insights

Displays the selected machine learning model, evaluation metrics, and decision threshold used for prediction.

## 🤖 Machine Learning

Several classification models were evaluated during development:

* Logistic Regression
* Decision Tree
* Random Forest
* Hist Gradient Boosting

The final system uses **HistGradientBoostingClassifier**.

A preprocessing pipeline is used to handle categorical and numerical features before prediction.

Because the dataset is imbalanced, threshold tuning was performed rather than relying solely on the default 0.50 classification threshold.

The final decision threshold used by PredictCare is **0.25**.

## 📊 Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

The final model achieved approximately **80% accuracy** and a **ROC-AUC of around 0.75** during evaluation.

## 🛠️ Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Plotly
* Altair
* OpenPyXL
* Jupyter Notebook

## 📂 Repository Structure

```text
PredictCare-AI-No-Show-Prediction/
│
├── Final_Project.ipynb
├── App.py
├── prediction_functions.py
├── charts.py
├── theme.py
├── predictcare_model.pkl
├── requirements.txt
├── README.md
│
└── images/
    ├── logo.png
    └── header.png
```

## 🧠 Features Used

The model uses patient and appointment-related information such as:

* Age
* Gender
* Neighbourhood
* Scholarship
* Hypertension
* Diabetes
* Alcoholism
* Handicap
* SMS Received
* Waiting Days
* Appointment Day
* Scheduled Day
* Scheduled Hour
* Previous Appointments
* Previous No-Shows
* Previous Shows
* Previous No-Show Rate
* Same-Day Appointment

Additional date and time features were created during feature engineering.

## 📓 Project Notebook

The `Final_Project.ipynb` notebook contains the main machine learning workflow, including:

* Data loading
* Data cleaning
* Exploratory data analysis
* Feature engineering
* Data preprocessing
* Model training
* Model comparison
* Model evaluation
* Threshold tuning
* Final model selection

## ⚠️ Disclaimer

PredictCare is an **educational machine learning project** and is not intended to replace medical professionals or make clinical decisions.

The predictions demonstrate how machine learning can be applied to hospital appointment management and no-show risk analysis.

## 👩‍💻 Project

**PredictCare – AI No-Show Prediction**

An educational project demonstrating the application of machine learning, data preprocessing, feature engineering, model evaluation, threshold tuning, and Streamlit deployment to a real-world healthcare-related problem.

