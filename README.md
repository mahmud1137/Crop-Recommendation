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

> **Dataset note:** The notebook expects `Agriculture.csv` to be available in the working directory. Add the dataset to your project repository (if you have permission to redistribute it), or update the notebook's file path. The notebook does not specify the dataset's original source.

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

Based on the notebook's comparison, Gaussian Naive Bayes is the best-performing model among the five classifiers tested.

## 🛠️ Technologies & Libraries

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/crop-recommendation.git
cd crop-recommendation
```

### 2. Create and activate a virtual environment (recommended)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Add the dataset

Place `Agriculture.csv` in the project directory, alongside the notebook, or update the CSV path in the notebook.

### 5. Run the notebook

```bash
jupyter notebook
```

Open `Crop_Recommendation(1).ipynb` and run the cells in order.

## 📁 Suggested Repository Structure

```text
crop-recommendation/
│
├── Crop_Recommendation.ipynb
├── Agriculture.csv
├── README.md
└── requirements.txt
```

A `requirements.txt` can be created with:

```bash
pip freeze > requirements.txt
```

## 💾 Model Export

The notebook contains commented-out `joblib` commands intended to save the Naive Bayes model, scaler, feature-column list, and label encoder:

```python
import joblib

joblib.dump(models['Naive Bayes'], 'Naive_Bayes_agrimind.pkl')
joblib.dump(scaler, 'scaler_agrimind.pkl')
joblib.dump(X.columns.tolist(), 'columns_agrimind.pkl')
joblib.dump(encoder, 'label_encoder_agrimind.pkl')
```

These commands are **not executed in the provided notebook**. Uncomment and run them after training if you want to create the files. For a production prediction pipeline, keep the fitted preprocessing objects and model together and ensure inference inputs use the same feature order and preprocessing.

## ⚠️ Limitations & Future Improvements

- Evaluate the model on additional data collected from different regions and seasons to assess generalization.
- Add confusion matrices and per-class precision, recall, and F1-scores for a more detailed error analysis.
- Use cross-validation and hyperparameter tuning to assess model stability and potentially improve performance.
- Validate input ranges and units before making predictions.
- Test the complete saved-model inference pipeline before deployment.
- Treat predictions as decision support, not a replacement for local agronomic expertise or soil testing.

## 👤 Author

**Your Name**

- GitHub: [your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://www.linkedin.com/)

---

If you find this project useful, consider giving the repository a ⭐.
