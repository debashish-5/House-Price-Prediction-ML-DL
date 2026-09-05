# Elevate Lab - AIML Task 6: House Price Prediction (ML & DL)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Framework](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-green.svg)](https://scikit-learn.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%2FKeras-red.svg)](https://tensorflow.org/)

An end-to-end Machine Learning and Deep Learning pipeline for predicting residential property prices based on real estate feature metrics. Developed as part of the **Elevate Lab AIML Internship Program (Task 6)**.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Dataset Summary](#-dataset-summary)
- [Workflow & Methodology](#-workflow--methodology)
- [Technologies & Dependencies](#-technologies--dependencies)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Execution Steps](#-execution-steps)
- [Model Evaluation Metrics](#-model-evaluation-metrics)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 📊 Project Overview

Accurate housing price estimation is vital for buyers, real estate investors, and financial institutions. This project focuses on building predictive models leveraging both traditional **Machine Learning algorithms** and **Deep Learning neural networks** to evaluate property valuations based on numerical and categorical features (e.g., square footage, location attributes, room counts, and structural characteristics).

### Key Objectives:
* Perform thorough **Exploratory Data Analysis (EDA)** to uncover feature relationships and spatial price trends.
* Apply robust **Data Preprocessing**, including handling missing values, outlier detection, feature encoding, and normalization.
* Train and evaluate multiple **Classical ML Models** (Linear Regression, Decision Trees, Random Forest, Gradient Boosting).
* Implement a **Deep Learning Neural Network** architecture tailored for tabular regression problems.
* Benchmark and compare model performances to identify the optimal price estimation model.

---

## 📁 Repository Structure


```

Elevate-Lab-Jan-task-6/
│
├── .ipynb_checkpoints/           # Jupyter Notebook execution checkpoints
├── .virtual_documents/           # Virtual environment document configurations
├── anaconda_projects/db          # Local Anaconda workspace configuration database
│
├── House_Price-ML-DL.ipynb       # Main Jupyter Notebook containing EDA, ML, & DL pipelines
├── Housing Price data set.csv    # Raw real estate housing dataset
├── task 6.pdf                    # Task specification & problem statement guidelines
└── README.md                     # Comprehensive project documentation

```

---

## 💾 Dataset Summary

The primary dataset used in this repository is `Housing Price data set.csv`. It contains historical records of property attributes along with their corresponding sale prices.

* **Target Variable:** `Price` / `SalePrice` (Continuous numerical variable for real estate valuation)
* **Feature Categories:**
  * **Property Dimensions:** Lot area, total living area (sq. ft.), basement area, garage space.
  * **Building Specifications:** Number of bedrooms, bathrooms, floors, construction year, and renovation status.
  * **Location & Quality:** Neighborhood rating, structural quality metrics, and feature amenities.

---

## ⚙️ Workflow & Methodology


```

┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Raw CSV Data    │ ──> │ Data Preprocess  │ ──> │ Feature Scale &  │
│ Import & Audit  │     │ & Missing Values │     │ Categorical Encd │
└─────────────────┘     └──────────────────┘     └──────────────────┘
│
┌─────────────────┐     ┌──────────────────┐              ▼
│ Performance     │ <── │ ML & Deep        │ <── ┌──────────────────┐
│ Benchmark & Vis │     │ Learning Training│     │ Train / Test     │
└─────────────────┘     └──────────────────┘     │ Data Split       │
└──────────────────┘

```

1. **Exploratory Data Analysis (EDA):**
   * Univariate analysis of price distribution (skewness & kurtosis).
   * Bivariate correlation heatmaps to extract high-impact predictors.
   * Scatter plots and box plots for outlier identification.

2. **Data Preprocessing & Feature Engineering:**
   * Imputation of missing values using median/mode strategies.
   * Categorical feature transformation via One-Hot Encoding and Label Encoding.
   * Min-Max Feature Scaling and Standard Scaling ($Z$-score normalization) for numerical attributes.

3. **Model Development:**
   * **Base Regression:** Ordinary Least Squares (OLS) Linear Regression.
   * **Ensemble Learning:** Random Forest Regressor, XGBoost / Gradient Boosting Regressor.
   * **Deep Neural Network (DNN):** Multi-Layer Perceptron (MLP) built with Dense layers, ReLU activations, Dropout regularizers, and Adam optimizer.

---

## 🛠️ Technologies & Dependencies

* **Core Language:** Python 3.8+
* **Data Processing & Analysis:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Deep Learning:** `tensorflow` / `keras`
* **Interactive Environment:** `jupyter` / `anaconda`

---

## 💻 Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/debashish-5/Elevate-Lab-Jan-task-6.git](https://github.com/debashish-5/Elevate-Lab-Jan-task-6.git)
cd Elevate-Lab-Jan-task-6

```

### 2. Create and Activate a Virtual Environment

```bash
# On Linux/macOS
python3 -m venv env
source env/bin/activate

# On Windows
python -m venv env
.\env\Scripts\activate

```

### 3. Install Required Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter

```

---

## 🚀 Execution Steps

1. Launch Jupyter Notebook server:
```bash
jupyter notebook

```


2. Navigate to `House_Price-ML-DL.ipynb` in your browser interface.
3. Run all cells sequentially (`Menu -> Cell -> Run All`) to re-generate visualizations, model training logs, and performance metrics.

---

## 📐 Model Evaluation Metrics

Model performance across both Machine Learning and Deep Learning algorithms is evaluated using standard regression criteria:

* **Mean Absolute Error (MAE):** Measures average absolute deviation from actual prices.
* **Mean Squared Error (MSE):** Quantifies variance and penalizes large prediction errors.
* **Root Mean Squared Error (RMSE):** Provides error evaluation in the original currency unit.
* **$R^2$ Score (Coefficient of Determination):** Evaluates overall target variable variance explained by the model.

---

## 👤 Author

* **Debashish** - [@debashish-5](https://github.com/debashish-5)
* **Repository:** [Elevate-Lab-Jan-task-6](https://github.com/debashish-5/Elevate-Lab-Jan-task-6)

```

```
