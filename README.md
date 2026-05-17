# 🏥 Hospital No-Show Prediction System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-blue.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/Cloud-AWS_Elastic_Beanstalk-orange.svg)](https://aws.amazon.com/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-green.svg)](https://mlflow.org/)

---

## 🎯 Project Overview

This project is an end-to-end Machine Learning system designed to predict the likelihood of a patient missing a medical appointment (**No-Show**).

### 💡 Innovation
Unlike traditional healthcare prediction systems that only use patient history, this project integrates **real-world historical weather data** (Temperature, Precipitation, Wind Speed) using the **Open-Meteo API**.

The research demonstrates that environmental conditions can act as important secondary behavioral factors influencing patient attendance.

---

## 🚀 Key Features

- **Multi-Source Data Integration**
  - Combined 110k healthcare appointment records with ERA5 Reanalysis weather data.

- **Modular ML Pipeline**
  - Complete production-style workflow:
  
  ```text
  Clean ➜ Merge ➜ Feature Engineering ➜ Train ➜ Tune ➜ Evaluate
  ```

- **Class Imbalance Handling**
  - Applied **Balanced Class Weights** to address the 80/20 dataset imbalance.
  - Achieved **73% Recall** for detecting no-show patients.

- **MLOps Workflow**
  - Experiment tracking using **MLflow**
  - Model serialization using **Joblib**
  - Containerization with **Docker**
  - Cloud deployment on **AWS Elastic Beanstalk**

---

## 🏗️ System Architecture

1. **Data Layer**
   - Healthcare CSV records
   - Open-Meteo API (ERA5 Reanalysis Weather Data)

2. **Processing Layer**
   - Data cleaning
   - Typo correction
   - Date-based merging scripts

3. **Feature Engineering Layer**
   - `waiting_days` (most influential feature)
   - `is_rainy`
   - `age_groups`

4. **Modeling Layer**
   - Tuned Random Forest Classifier
   - Hyperparameter optimization using `RandomizedSearchCV`

5. **Interface Layer**
   - Flask web application for real-time prediction

6. **Cloud Layer**
   - Dockerized deployment on AWS Elastic Beanstalk using AWS ECR

---

## 📊 Experimental Results

Two experiments were conducted:

- **Experiment A:** Medical dataset only
- **Experiment B:** Medical + Weather dataset

The **Random Forest Classifier** delivered the best overall performance.

| Metric | Random Forest (Baseline) | Random Forest (+ Weather) |
|--------|--------------------------|----------------------------|
| **Accuracy** | 58.42% | **58.52%** |
| **F1-Score** | 0.4442 | **0.4453** |
| **Recall (No-Shows)** | 72.80% | **73.24%** |

### 📌 Observation
Adding weather features produced a small but consistent improvement across all major evaluation metrics, especially in identifying no-show patients.

---

## 📁 Project Structure

```text
LAB_PROJECT/
│
├── artifacts/
│   └── models/              # Serialized .joblib models and feature lists
│
├── data/
│   ├── raw/                 # Original datasets (Healthcare + Weather)
│   └── processed/           # Cleaned and engineered datasets
│
├── src/
│   ├── data_processing/     # Cleaning and merging scripts
│   ├── features/            # Feature engineering logic
│   ├── models/              # Training, tuning, and evaluation
│   ├── visualization/       # EDA and performance visualizations
│   └── deployment/          # Flask application and templates
│
├── reports/
│   └── figures/             # Confusion matrices and ranking charts
│
├── mlruns/                  # MLflow experiment tracking
│
├── main.py                  # Main pipeline controller
├── Dockerfile               # Docker container configuration
├── Dockerrun.aws.json       # AWS deployment configuration
└── requirements.txt         # Python dependencies
```

---

# 💻 Installation & Usage

## 1️⃣ Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/hospital-noshow-prediction.git

# Move into the project directory
cd hospital-noshow-prediction

# Install dependencies
pip install -r requirements.txt
```

---

## 2️⃣ Run the Complete ML Pipeline

This command executes the full workflow:

```text
Clean ➜ Merge ➜ Feature Engineering ➜ Training ➜ Hyperparameter Tuning ➜ Evaluation ➜ Serialization
```

```bash
python main.py
```

---

## 3️⃣ Launch the Flask Web Application

```bash
python src/deployment/app.py
```

Visit:

```text
http://localhost:5000
```

to access the prediction dashboard.

---

# 🐳 Docker & Cloud Deployment

## 🔹 Local Docker Testing

```bash
# Build Docker image
docker build -t noshow-app .

# Run container
docker run -p 5000:5000 noshow-app
```

---

## ☁️ AWS Deployment Workflow

### Step 1: Initialize Elastic Beanstalk

```bash
eb init -p docker noshow-prediction-app
```

### Step 2: Create Environment

```bash
eb create noshow-env --service-role LabRole --instance-profile LabRole
```

### Step 3: Deploy Application

```bash
eb deploy
```

---

## 🛠️ Technologies Used

| Category | Tools & Technologies |
|----------|----------------------|
| Programming | Python 3.11 |
| ML Libraries | Scikit-learn, Pandas, NumPy |
| Experiment Tracking | MLflow |
| Visualization | Matplotlib, Seaborn |
| Backend | Flask |
| Containerization | Docker |
| Cloud Deployment | AWS Elastic Beanstalk, AWS ECR |
| Model Serialization | Joblib |

---

## 👨‍💻 Contributors

- **Muhammad Adil Shahzad** *(BSSE-23010)*
- **Zainab Faisal**

---

## 🎓 Acknowledgments

- **Dataset:** Kaggle Medical Appointment No Shows Dataset
- **Weather API:** Open-Meteo Historical Weather Data API
- **Guidance:** Machine Learning Lab — 6th Semester

---

## 📌 Future Improvements

- Deep Learning integration
- Real-time weather forecasting support
- Appointment reminder notification system
- Mobile application interface
- Explainable AI (XAI) integration using SHAP/LIME

---

## ⭐ Repository Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.
