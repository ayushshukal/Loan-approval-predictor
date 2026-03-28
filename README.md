# 🏦 Loan Approval Prediction

A machine learning web app that predicts whether a loan application will be **approved or rejected**, built with Python, scikit-learn, and Streamlit.

## 📸 Features

- Predicts loan approval based on applicant details
- Interactive web UI built with **Streamlit**
- Trained using **Random Forest** and **Decision Tree** classifiers
- Shows prediction confidence percentage

## 📁 Project Structure

```
loan_approval ML/
├── model.py          # Train & save the ML model
├── app.py            # Streamlit web application
├── loan_data.csv     # Dataset
├── requirements.txt  # Python dependencies
└── .gitignore
```

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python model.py
```

This generates `model.pkl`.

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## 🧠 Model Details

| Model | Notes |
|-------|-------|
| Decision Tree | `max_depth=6` |
| Random Forest | `100 estimators, max_depth=6` |

**Key features used:** Credit History, Total Income, Loan Amount, Property Area, and more.

## 📦 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, streamlit
