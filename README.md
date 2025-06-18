# 🩺 Symptom-based Medical Diagnosis System

An intelligent disease prediction system based on user-reported symptoms using **Machine Learning**. The model predicts likely diseases using **Random Forest** and **Support Vector Machine (SVM)** classifiers, optimized with advanced preprocessing and feature selection.

---

## 📌 Project Objectives

- Predict diseases based on a list of symptoms provided by the user.
- Improve diagnostic accuracy using:
  - Feature selection techniques
  - Data normalization
  - Hyperparameter tuning

---

## 🧠 Models Used

- 🌳 **Random Forest Classifier**
- ➗ **Support Vector Machine (SVM)**

---

## 📊 Dataset

- Symptom-based disease dataset with rows representing various medical conditions.
- Columns: Binary/One-hot encoded symptoms, Diagnosis label.

📁 Path: `data/symptoms_dataset.csv`

---

## ⚙️ Workflow

1. **Data Preprocessing**
   - Null handling
   - Symptom encoding
   - Standardization/Normalization

2. **Feature Engineering**
   - Recursive Feature Elimination (RFE)
   - Principal Component Analysis (PCA)

3. **Model Training**
   - 10-fold Cross Validation
   - GridSearchCV for hyperparameter tuning

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrix visualization

---

## 📂 Project Structure

| File / Folder        | Description                                 |
|----------------------|---------------------------------------------|
| `data/`              | Contains the dataset used                   |
| `src/preprocess.py`  | Data cleaning and feature engineering       |
| `src/train_model.py` | Contains model training code                |
| `notebooks/`         | Jupyter notebooks for EDA and experiments   |
| `results/`           | Confusion matrices and performance reports  |
| `run_diagnosis.py`   | CLI tool for predicting diseases            |

---

## 🚀 Getting Started

### 🛠️ Installation

```bash
git clone https://github.com/yourusername/Symptom-Based-Medical-Diagnosis.git
cd Symptom-Based-Medical-Diagnosis
pip install -r requirements.txt
