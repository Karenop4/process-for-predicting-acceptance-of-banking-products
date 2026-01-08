# 🏦 Predicting Acceptance of Banking Products with Deep Learning and XAI

> **A systematic process using Neural Networks, Ensemble Models, and Explainability Techniques.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-Machine%20Learning-yellow)
![SHAP](https://img.shields.io/badge/XAI-SHAP-green)

## 📋 Project Description

This project addresses the challenge of predicting whether a customer will subscribe to a fixed-term deposit based on data from direct marketing campaigns by a Portuguese banking institution.

The **CRISP-DM** methodology is followed, implementing a robust workflow that includes data cleaning, feature engineering, dimensionality reduction (PCA), class balancing (SMOTE), and the comparison of classic machine learning architectures vs. deep learning and stacking.

**Authors:**
* Andrés Encalada
* Karen Ortiz

## 📊 Dataset

The dataset comes from the UCI Machine Learning repository: **Bank Marketing Dataset**.
* **Instances:** 41,188
* **Variables:** 20 (Demographics, Socioeconomic Context, and Contact Details).
* **Target:** `y` (Deposit subscription: ‘yes’/'no').

## 🛠️ Technologies and Libraries

* **Python 3**
* **Data Manipulation:** Pandas, NumPy.
* **Machine Learning:** Scikit-learn (Pipeline, ColumnTransformer, PCA).
* **Data Balancing:** Imbalanced-learn (SMOTE).
* **Deep Learning:** TensorFlow / Keras.
* **Explainability (XAI):** SHAP.
* **Visualization:** Matplotlib, Seaborn.

## ⚙️ Pipeline Methodology

### 1. Preprocessing and Feature Engineering
* **Cleaning:** Handling ‘unknown’ values and removing duplicates.
* **Transformation:**
* *Numeric:* Imputation (median) and Standardization (StandardScaler).
* *Categorical:* One-Hot and Ordinal encoding.
* **Dimensionality Reduction:** Application of **PCA** preserving 95% of the variance (reduction from 47 to 22 features).
* **Balancing:** Application of **SMOTE** in the training set to mitigate class imbalance.
### 2. Modeling (Evaluated Architectures)
Three strategies were designed and compared:
1.  **Baseline:** Comparison between **KNN** and **Random Forest**. (Winner: Random Forest).
2.  **Deep Learning (Proposal):** Artificial Neural Network (ANN) with dense layers, ReLU activation, and Dropout for regularization.
3.  **Hybrid (Stacking):** An ensemble that combines the predictions of RF, KNN, and ANN using Logistic Regression as a meta-model.

## 🏆 Results and Evaluation

The **Neural Network** model was selected as the best model for production due to its superior generalization ability (AUC) and its balance in the confusion matrix (fewer lost leads).

| Model | AUC Score | F1-Score | Observation |
| :--- | :---: | :---: | :--- |
| **Artificial Neural Network (ANN)** | **0.9406** | **0.6285** | **Best overall and operational performance.** |
| Random Forest | 0.9359 | 0.6137 | Solid baseline. |
| Stacking Ensemble | 0.9346 | 0.5799 | Too conservative (many false negatives). |

### Business Impact (Confusion Matrix)
The Neural Network significantly reduced **False Negatives** (interested customers that the model ignored), capturing **41% more sales opportunities** compared to the Stacking model.


## 🔍 Explainability (XAI) with SHAP

**SHAP (SHapley Additive exPlanations)** was used to interpret the model's decisions.
* **Most influential variable:** `duration` (call duration). The longer the duration, the higher the probability of success.
* **Macroeconomic factors:** `euribor3m` and `nr.employed` play a crucial role, indicating that the economic context affects the customer's decision.

## 🚀 Installation and Use

1.  Clone the repository:
```bash
    git clone [https://github.com/Karenop4/process-for-predicting-acceptance-of-banking-products.git](https://github.com/Karenop4/process-for-predicting-acceptance-of-banking-products.git)
    ```
2.  Install the dependencies:
```bash
pip install pandas numpy scikit-learn tensorflow shap imbalanced-learn matplotlib seaborn
```
3.  Run the `predicting_acceptance_of_banking_products_.ipynb` notebook in Jupyter or Google Colab.

## 📄 License
This project is for educational and academic purposes.

