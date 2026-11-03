# 🌱 Crop Recommendation System

A machine learning classification project that recommends a suitable crop based on soil nutrient levels and environmental conditions. The notebook explores the dataset, prepares the target labels, trains and evaluates multiple classification algorithms, and compares their performance using validation and test data.

## 📌 Project Overview

The goal of this project is to predict which crop is most suitable for a given set of agricultural conditions.

The model uses **7 input features**:

| Feature | Description |
|---|---|
| `N` | Nitrogen level |
| `P` | Phosphorus level |
| `K` | Potassium level |
| `temperature` | Temperature |
| `humidity` | Relative humidity |
| `ph` | Soil pH |
| `rainfall` | Rainfall |

The target column is `label`, which represents the recommended crop.

## 📊 Dataset

The notebook loads the dataset from `Agriculture.csv`.

- **Total records:** 2,200
- **Total columns:** 8 (7 features + 1 target)
- **Crop classes:** 22
- **Records per crop:** 100
- **Missing values:** 0
- **Duplicate rows:** 0

The 22 crop labels in the notebook are:

`rice`, `maize`, `chickpea`, `kidneybeans`, `pigeonpeas`, `mothbeans`, `mungbean`, `blackgram`, `lentil`, `pomegranate`, `banana`, `mango`, `grapes`, `watermelon`, `muskmelon`, `apple`, `orange`, `papaya`, `coconut`, `cotton`, `jute`, and `coffee`.


## 🔍 Project Workflow

1. **Load the data** using Pandas.
2. **Inspect the dataset** with `head()`, `shape`, `info()`, and `describe()`.
3. **Check data quality** by checking missing values and duplicate rows.
4. **Explore feature distributions** using histograms with KDE curves.
5. **Encode the target labels** using `LabelEncoder`.
6. **Separate features and target** into `X` and `y`.
7. **Split the data** into training, validation, and test sets using stratified sampling.
8. **Scale the features** using `StandardScaler`, fitting the scaler only on the training set.
9. **Train and compare classifiers** using accuracy and weighted F1-score.
10. **Review model fit** by comparing training and validation accuracy.

### Data Split

The notebook uses a stratified split with `random_state=42`:

| Dataset | Records | Share |
|---|---:|---:|
| Training | 1,540 | 70% |
| Validation | 330 | 15% |
| Test | 330 | 15% |
| **Total** | **2,200** | **100%** |

## 🤖 Machine Learning Models

The notebook trains and evaluates these five classifiers:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)

## 📈 Model Evaluation

The models are compared using:

- **Accuracy:** The proportion of correctly classified samples.
- **Weighted F1-score:** The weighted average of class-wise F1-scores.

### Test Results

Results below are taken from the notebook output and sorted by test accuracy.

| Model | Validation Accuracy | Test Accuracy | Validation F1 | Test F1 |
|---|---:|---:|---:|---:|
| **Gaussian Naive Bayes** | **99.39%** | **99.09%** | **0.9939** | **0.9909** |
| SVM | 99.09% | 98.79% | 0.9909 | 0.9878 |
| Decision Tree | 97.88% | 98.48% | 0.9788 | 0.9847 |
| Logistic Regression | 96.67% | 97.88% | 0.9665 | 0.9787 |
| KNN | 98.48% | 97.27% | 0.9848 | 0.9724 |

<details>
<summary>Training, validation, and test accuracy comparison</summary>

| Model | Train Accuracy | Validation Accuracy | Test Accuracy | Train–Validation Gap |
|---|---:|---:|---:|---:|
| Gaussian Naive Bayes | 99.55% | 99.39% | 99.09% | 0.0015 |
| SVM | 98.18% | 99.09% | 98.79% | -0.0091 |
| Decision Tree | 100.00% | 97.27% | 98.48% | 0.0273 |
| Logistic Regression | 97.47% | 96.67% | 97.88% | 0.0080 |
| KNN | 98.18% | 98.48% | 97.27% | -0.0030 |

</details>

### Best Performing Model

**Gaussian Naive Bayes** achieved the highest test accuracy in this experiment:

- **Test accuracy:** 99.09%
- **Test weighted F1-score:** 0.9909
- **Validation accuracy:** 99.39%
