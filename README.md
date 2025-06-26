# Thyroid Cancer Recurrence Prediction

This project aims to develop a machine learning model to predict the recurrence of well-differentiated thyroid cancer after initial treatment. The model is built using the Thyroid Disease dataset from the UCI Machine Learning Repository. The project covers the entire machine learning pipeline, including data preprocessing, feature engineering, handling common pitfalls like data leakage, model training, and evaluation.

## About the Dataset

The dataset contains 13 clinicopathologic features collected over a 15-year period, with each patient being followed for at least 10 years.

### Key Features
*   **Age:** Patient's age at the time of diagnosis.
*   **Gender:** Patient's gender.
*   **Smoking:** Current smoking status.
*   **Hx Smoking:** History of smoking.
*   **Hx Radiotherapy:** History of radiotherapy treatment.
*   **Thyroid Function:** Status of thyroid function.
*   **Physical Examination:** Findings from a physical examination.
*   **Adenopathy:** Presence of enlarged lymph nodes.
*   **Pathology:** Specific types of thyroid cancer.
*   **Focality:** Unifocal or multifocal cancer.
*   **Risk:** Cancer risk category.
*   **T, N, M:** TNM classification (Tumor, Node, Metastasis).
*   **Stage:** The overall cancer stage.
*   **Response:** Response to treatment.
*   **Recurred:** Whether the cancer has recurred (Target Variable).

---

## Project Workflow & Methodology

The project follows a standard machine learning pipeline to ensure a robust and reliable model.

### 1. Data Preprocessing

*   **Handling Missing Values:** Missing values in the dataset were imputed. The **median** was used for numerical columns, and the **mode** (most frequent value) was used for categorical columns.
*   **Encoding Categorical Data:**
    *   **Binary Encoding:** Binary features like `Gender` and `Smoking` were converted to `0` and `1`.
    *   **Ordinal Encoding:** Ordinal features such as `Risk` were mapped to numerical values that preserve their inherent order (e.g., `Low`: 0, `Intermediate`: 1, `High`: 2).
    *   **One-Hot Encoding:** Nominal features with no intrinsic order, like `Pathology` and `Adenopathy`, were converted into new binary columns using `pd.get_dummies()`.

### 2. Tackling the Data Leakage Problem

An initial model training attempt yielded a perfect **100% accuracy score** on the test set. This is a classic symptom of a common machine learning pitfall: **data leakage**.

*   **The Problem:** The feature `Response` (Response to Treatment) is a post-treatment outcome. It is highly correlated with the target variable `Recurred`, as the response to treatment directly informs whether the cancer is still present or has been eradicated. Using it as a predictor means the model is not learning to predict the future but is instead "cheating" by looking at information that would not be available at the time of prediction.

*   **The Solution:** To build a fair and realistic model, the `Response` feature and all columns derived from it were **removed** from the feature set (`X`). This ensures the model learns from pre-treatment and diagnostic data only.

### 3. Model Building Pipeline

After addressing data leakage, the following structured pipeline was implemented:

1.  **Train-Test Split:** The dataset was split into **80% training** and **20% testing** sets using `train_test_split`. The `stratify=y` parameter was used to maintain the same proportion of target classes in both sets, which is crucial for imbalanced datasets.

2.  **Handling Imbalanced Data:** The target variable `Recurred` was imbalanced. To address this, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied.
    *   **Crucial Note:** SMOTE was applied **only to the training set** to prevent data leakage from the test set into the training process.

3.  **Feature Scaling:** `StandardScaler` was used to scale the numerical features. The scaler was fitted on the training data (`fit_transform`) and then used to transform the test data (`transform`), ensuring that no information from the test set influenced the scaling process.

### 4. Modeling and Evaluation

*   **Model Selection:** A **Random Forest Classifier** was chosen for this classification task due to its high performance and robustness.
*   **Evaluation:** The model's performance was evaluated using a `classification_report`, focusing on **Precision**, **Recall**, and **F1-Score**, which are more informative metrics than accuracy for imbalanced datasets. After fixing the data leakage issue, the model produced more realistic and meaningful results.

---

## How to Run

To replicate this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn
    ```

3.  **Run the script:**
    Execute the Python script or Jupyter Notebook containing the project code.
    ```bash
    python your_script_name.py
    ```

## Conclusion

This project highlights the critical importance of meticulous data preprocessing and methodological rigor in developing machine learning models. Identifying and mitigating common issues like data leakage is essential for accurately assessing a model's real-world performance. The final model serves as a valuable tool with the potential to aid in predicting the risk of thyroid cancer recurrence based on clinicopathologic features.

## Dataset Link

https://www.kaggle.com/datasets/jainaru/thyroid-disease-data
