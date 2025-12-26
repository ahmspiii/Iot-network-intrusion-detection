import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings, time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# ====== 1. Load Data ======
csv_file = r"D:\data grad\wataiData\csv\CICIoT2023\output\balanced_dataset.csv"
df = pd.read_csv(csv_file)
df.dropna(inplace=True)

# ====== 2. Split Features/Labels ======
label_col = "label" if "label" in df.columns else df.columns[-1]
X = df.drop(columns=[label_col])
y = df[label_col]

# ====== 3. Encode Labels ======
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ====== 4. Train/Test Split ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ====== 5. Scale ======
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ====== 6. Model & Cross-Validation ======
model = RandomForestClassifier(
    n_estimators=500, max_depth=150,
    n_jobs=-1, class_weight="balanced", random_state=42
)
print("🚀 5-Fold CV (macro F1)...")
cv_scores = cross_val_score(model, X_train_scaled, y_train,
                            cv=5, scoring="f1_macro", n_jobs=-1)
print(f"CV mean F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# ====== 7. Train Final Model ======
print("🚀 Training final model...")
start = time.time()
model.fit(X_train_scaled, y_train)
print(f"Training time: {time.time()-start:.2f}s\n")

# ====== 8. Predictions ======
y_pred = model.predict(X_test_scaled)

# ====== 9. Overall Metrics ======
print("📊 Overall Metrics:")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro Precision : {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Macro Recall    : {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"Macro F1        : {f1_score(y_test, y_pred, average='macro'):.4f}")

# ====== 10. Per-class Metrics ======
report_dict = classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)

# Build DataFrame for class-wise precision/recall/f1
per_class_df = (
    pd.DataFrame(report_dict)
    .transpose()
    .drop(index=["accuracy","macro avg","weighted avg"])
    .reset_index()
    .rename(columns={"index":"Class"})
)
print("\n=== Precision / Recall / F1 per Class ===")
print(per_class_df)

# Save to CSV
per_class_df.to_csv("per_class_metrics.csv", index=False)
print("\n✅ Per-class metrics saved to per_class_metrics.csv")

# ====== 11. Confusion Matrix ======
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=label_encoder.classes_).plot(
                           xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ====== 12. Feature Importance ======
importances = model.feature_importances_
pd.DataFrame({"Feature": X.columns,
              "Importance": importances})\
  .sort_values("Importance", ascending=True)\
  .tail(20)\
  .plot(kind="barh", x="Feature", y="Importance", figsize=(8,6))
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

# ====== 13. ROC-AUC (macro) ======
y_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
y_score = model.predict_proba(X_test_scaled)
print(f"Macro ROC-AUC: {roc_auc_score(y_bin, y_score, average='macro', multi_class='ovr'):.4f}")

# ====== 14. Save Model & Encoders ======
joblib.dump(model, "rf_model_final2.pkl")
joblib.dump(scaler, "scaler3.pkl")
joblib.dump(label_encoder, "label_encoder3.pkl")
print("\n✅ Model, scaler, and label encoder saved.")
