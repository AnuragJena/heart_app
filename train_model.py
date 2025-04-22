import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

# === Load dataset ===
dataset_path = "synthetic_novel_heart_data_1.csv"

if not os.path.exists(dataset_path):
    print("Dataset not found. Generating synthetic data...")
    np.random.seed(42)
    num_samples = 1000
    num_features = 16
    X_synthetic = np.random.randn(num_samples, num_features)
    y_synthetic = (np.sum(X_synthetic[:, :5], axis=1) + np.random.randn(num_samples)) > 0
    df = pd.DataFrame(X_synthetic, columns=[f"feature_{i+1}" for i in range(num_features)])
    df['target'] = y_synthetic.astype(int)
    df.to_csv(dataset_path, index=False)
else:
    df = pd.read_csv(dataset_path)

# === Feature and Target Extraction ===
if "target" not in df.columns:
    raise ValueError("Expected target column 'target' not found in dataset.")

X = df.drop(columns=["target"])
y = df["target"]

# === Standardization ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Custom Lightweight Attention-Like Mechanism ===
class AdaptiveFeatureAttention:
    def __init__(self):
        self.feature_scores = None

    def fit(self, X, y):
        from sklearn.feature_selection import mutual_info_classif
        self.feature_scores = mutual_info_classif(X, y)
        self.feature_scores /= np.max(self.feature_scores)

    def transform(self, X):
        if self.feature_scores is None:
            raise RuntimeError("Feature scores not computed. Call 'fit' before 'transform'.")
        return X * self.feature_scores

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

# Apply Adaptive Attention Layer
attention = AdaptiveFeatureAttention()
X_train_attn = attention.fit_transform(X_train, y_train)
X_test_attn = attention.transform(X_test)

# === Model Training ===
model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.07, max_depth=5, random_state=42)
model.fit(X_train_attn, y_train)

# === Evaluation ===
y_pred = model.predict(X_test_attn)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# === Save trained model and components ===
joblib.dump(model, 'attention_model.pkl')
joblib.dump(attention, 'attention_layer.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model, attention layer, and scaler saved successfully.")

# === Visualizations ===
# 1. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.tight_layout()
plt.show()

# 2. Pairwise Plot for Selected Features
subset_features = df.columns[:5].tolist() + ["target"]
sns.pairplot(df[subset_features], hue="target", diag_kind="kde", palette="husl")
plt.suptitle("Pairwise Feature Distributions by Disease Presence", y=1.02)
plt.show()

# 3. Attention Weights Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=attention.feature_scores, y=X.columns, palette="viridis")
plt.title("Feature Importance via Adaptive Attention Scores")
plt.xlabel("Normalized Attention Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()