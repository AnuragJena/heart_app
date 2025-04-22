import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Sample data — you can replace with real heart dataset
data = pd.DataFrame({
    'age': [63, 37, 41, 56, 57],
    'sex': [1, 1, 0, 1, 0],
    'cp': [3, 2, 1, 1, 2],
    'chol': [233, 250, 204, 236, 354],
    'trestbps': [145, 130, 130, 120, 140],
    'thalach': [150, 187, 172, 178, 163],
    'target': [1, 1, 1, 0, 0]
})

X = data[['age', 'sex', 'cp', 'chol', 'trestbps', 'thalach']]
y = data['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl saved")
