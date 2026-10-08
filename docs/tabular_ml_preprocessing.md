# Tabular ML Preprocessing and Testing

## 1. Dataset

The crop recommendation dataset contains 69,718 records.

The original dataset contained 14 columns, including an unnecessary index column (`Unnamed: 0`).

The index column was removed, resulting in 13 useful columns:

- 12 input features
- 1 target column (`label`)

### Input Features

- N
- Fe
- Mn
- S
- Mg
- C
- P
- K
- temperature
- humidity
- ph
- rainfall

### Target

- `label` — recommended crop

---

## 2. Data Preprocessing

The following preprocessing steps were performed:

1. Removed the unnecessary `Unnamed: 0` index column.
2. Checked for missing values.
3. Checked for duplicate rows.
4. Separated input features (`X`) and target (`y`).
5. Split the dataset into training and testing sets.
6. Applied feature scaling using `StandardScaler`.

### Data Quality Checks

- Missing values: 0
- Duplicate rows: 0

### Train-Test Split

The dataset was divided using an 80:20 split.

- Training data: 55,774 rows
- Testing data: 13,944 rows

---

## 3. Models Tested

Three classification models were compared:

1. Random Forest Classifier
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

---

## 4. Model Results

| Model         | Accuracy | Precision |  Recall | F1-score |
| ------------- | -------: | --------: | ------: | -------: |
| Random Forest |   0.9999 |   ~0.9999 | ~0.9999 |  ~0.9999 |
| Decision Tree |   0.9997 |   ~0.9997 | ~0.9997 |  ~0.9997 |
| KNN           |   0.9689 |    0.9714 |  0.9689 |   0.9690 |

Random Forest achieved the highest test-set performance among the three models.

Therefore, Random Forest was selected as the baseline crop recommendation model.

---

## 5. Saved Model Files

The trained Random Forest model and preprocessing scaler were saved for later use:

- `ml/crop_recommendation_model.pkl`
- `ml/crop_recommendation_scaler.pkl`

These files can be used by the application to preprocess farmer input and generate crop recommendations.

---

## 6. Testing

A separate prediction test was performed using the saved crop recommendation model and scaler.

The purpose of the test was to verify that the trained model could be loaded successfully and used for prediction.

The Random Forest model also produced a confusion matrix during evaluation.

---

## 7. ML Pipeline

The overall tabular ML pipeline is:

Dataset  
↓  
Data Cleaning  
↓  
Missing Value Check  
↓  
Duplicate Check  
↓  
Feature/Target Separation  
↓  
Train-Test Split  
↓  
Feature Scaling  
↓  
Model Training  
↓  
Model Comparison  
↓  
Model Selection  
↓  
Model Saving  
↓  
Prediction Testing

---

## 8. Note

The reported accuracy values are test-set results obtained from this dataset. They should not be interpreted as guaranteed real-world agricultural accuracy.
