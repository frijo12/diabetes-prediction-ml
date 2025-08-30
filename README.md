# 🩺 Diabetes Prediction Using Machine Learning

A machine learning–powered web application that predicts whether a person is **diabetic or not** using their **medical attributes**.
The app provides real-time predictions through a simple web interface built with **Streamlit**.

---

## 📌 Problem Statement

Diabetes is a chronic health condition affecting how the body turns food into energy.
Early prediction can help in better management, lifestyle changes, and reducing complications.

This project aims to:

> 🧪 Predict an individual's **diabetes outcome (0 = Non-Diabetic, 1 = Diabetic)** using medical indicators like glucose level, BMI, and blood pressure.

---

## 🎯 Objectives

* Analyze health parameters that influence diabetes.
* Build and evaluate multiple ML models.
* Deploy a **user-friendly Streamlit app** for real-time predictions.

---

## 🗃️ Dataset Details

* **Name**: PIMA Indians Diabetes Dataset
* **Format**: CSV (`pima_diabetes_synthetic_5000.csv`)
* **Rows**: \~5000 entries
* **Label**: `Outcome` (0 = Non-Diabetic, 1 = Diabetic)

### 🔑 Features Used:

| Feature                  | Description                                 |
| ------------------------ | ------------------------------------------- |
| Pregnancies              | Number of times pregnant                    |
| Glucose                  | Plasma glucose concentration                |
| BloodPressure            | Diastolic blood pressure (mm Hg)            |
| SkinThickness            | Triceps skinfold thickness (mm)             |
| Insulin                  | 2-Hour serum insulin (mu U/ml)              |
| BMI                      | Body Mass Index                             |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age                      | Age in years                                |

---

## 🧠 Machine Learning Approach

### ✅ Data Preprocessing

* Replaced invalid **0 values** in medical features (like Insulin).
* Handled outliers using **clipping** based on medical ranges.
* Applied **StandardScaler** for normalization.
* Split dataset → **80% train / 20% test**.

### 🤖 Algorithms Tested

| Model                   | Accuracy | Precision | Recall | F1 Score |
| ----------------------- | -------- | --------- | ------ | -------- |
| **SVM**                 | 0.897    | 0.878     | 0.916  | 0.896    |
| **Random Forest**       | 0.893    | 0.863     | 0.928  | 0.894    |
| **K-Nearest Neighbors** | 0.893    | 0.864     | 0.926  | 0.894    |
| **Gradient Boosting**   | 0.876    | 0.846     | 0.912  | 0.877    |
| **Decision Tree**       | 0.853    | 0.833     | 0.873  | 0.853    |
| **Logistic Regression** | 0.802    | 0.781     | 0.825  | 0.802    |
| **Naive Bayes**         | 0.766    | 0.748     | 0.784  | 0.766    |

* **Best model**: `SVM` (highest accuracy and balanced metrics)
* Evaluation metrics include **Accuracy**, **Precision**, **Recall**, and **F1 Score**.

---

## 💡 Project Workflow

```
1. Load and preprocess dataset → handle missing values/outliers
2. Train multiple ML models → SVM, Random Forest, KNN, etc.
3. Evaluate models → Accuracy, Precision, Recall, F1
4. Save best model → joblib
5. Build Streamlit app → interactive input form
6. Make predictions in real-time
```

---

## 💻 Web App Features

* Interactive **Streamlit interface**
* Input fields for all medical parameters:

  * Pregnancies, Glucose, Blood Pressure
  * Skin Thickness, Insulin, BMI
  * Diabetes Pedigree Function, Age
* Model prediction displayed with clear result:

  * ✅ Not Diabetic
  * ⚠️ Diabetic

---

## 📸 App Screenshot

![Diabetes Prediction App](assets/app_screenshot.png)

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/frijo12/diabetes-prediction-ml.git
cd diabetes-prediction-ml
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit App

```bash
streamlit run app.py
```

### 4. Open in Browser

👉 [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure

```
diabetes-prediction-ml/
├── app.py                  # Streamlit app
├── model_training.py       # ML training code
├── data/                   
│   └── pima_diabetes_synthetic_5000.csv
├── models/                 
│   └── diabetes_model.pkl  # Saved best model
├── assets/                 # Screenshots or images
│   └── app_screenshot.png
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```txt
streamlit==1.32.0
pandas==2.2.0
numpy==1.26.0
scikit-learn==1.4.2
joblib==1.3.2
matplotlib==3.8.3
seaborn==0.13.2
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📌 Future Improvements

* Add **visual analytics** for predictions.
* Deploy on **Streamlit Cloud** / **Heroku**.
* Train on larger real-world diabetes datasets.
* Enhance UI with charts and comparisons.

---

## 🧾 License

This project is open-source and available under the **MIT License**.

---

## ✨ Author

**Frijo Antony CF**
🎓 Final Year B.Tech CSE Student
💡 Passionate about AI, ML & Web Apps
📫 Contact: [LinkedIn](https://www.linkedin.com/in/frijoantonycf)

---

## 🙌 Acknowledgments

* [Kaggle Datasets](https://www.kaggle.com/)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [Streamlit Documentation](https://docs.streamlit.io/)
