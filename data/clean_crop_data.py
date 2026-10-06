import pandas as pd

df = pd.read_csv("data/crop_recommendation.csv")

print("Original shape:", df.shape)

df = df.drop(columns=["Unnamed: 0"])

print("After removing index column:", df.shape)
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe())

print("\nMinimum values:")
print(df.drop(columns=["label"]).min())

print("\nMaximum values:")
print(df.drop(columns=["label"]).max())

df.to_csv("data/crop_recommendation_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")

X = df.drop(columns=["label"])
y = df["label"]

print("\nFeatures (X):")
print(X.columns.tolist())

print("\nTarget (y):")
print(y.name)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled training data shape:", X_train_scaled.shape)
print("Scaled testing data shape:", X_test_scaled.shape)

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model_rf.fit(X_train_scaled, y_train)

rf_accuracy = model_rf.score(X_test_scaled, y_test)

print("\nRandom Forest Accuracy:", rf_accuracy)

from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(
    random_state=42
)

model_dt.fit(X_train_scaled, y_train)

dt_accuracy = model_dt.score(X_test_scaled, y_test)

print("\nDecision Tree Accuracy:", dt_accuracy)

from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors=5)

model_knn.fit(X_train_scaled, y_train)

knn_accuracy = model_knn.score(X_test_scaled, y_test)

print("\nKNN Accuracy:", knn_accuracy)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predictions
rf_pred = model_rf.predict(X_test_scaled)
dt_pred = model_dt.predict(X_test_scaled)
knn_pred = model_knn.predict(X_test_scaled)

# Model comparison
models = {
    "Random Forest": rf_pred,
    "Decision Tree": dt_pred,
    "KNN": knn_pred
}

print("\nModel Comparison:")

for name, predictions in models.items():
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    print(f"\n{name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, rf_pred)

print("\nRandom Forest Confusion Matrix:")
print(cm)

import joblib

joblib.dump(model_rf, "ml/crop_recommendation_model.pkl")
joblib.dump(scaler, "ml/crop_recommendation_scaler.pkl")

print("\nModel and scaler saved successfully!")