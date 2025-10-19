import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
import json
import base64

# --- 1. Load Data ---
# Note: The data is loaded from a local file in the environment.
# For demonstration purposes, we assume 'heart.csv' is available.
try:
    df = pd.read_csv("heart.csv")
except FileNotFoundError:
    print("Error: heart.csv not found. Please ensure the file is in the same directory.")
    exit()

# --- 2. Separate Features and Target ---
X = df.drop('target', axis=1)
y = df['target']

# --- 3. Identify Continuous Features for Scaling ---
# These are the features where Standard Scaling is most beneficial.
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# --- 4. Preprocessing: Scaling and Normalization ---
scaler = StandardScaler()

# Fit scaler on the entire feature set for simplicity in this small dataset, 
# but only transform the continuous features.
# Note: In a robust pipeline, scaling is typically fit only on the training set.

# Extract the scaling parameters (mean and standard deviation) for deployment
scaler.fit(X[continuous_features])
scaling_params = {
    'mean': scaler.mean_.tolist(),
    'std': scaler.scale_.tolist(),
    'features': continuous_features
}

X[continuous_features] = scaler.transform(X[continuous_features])

# --- 5. Train/Test Split (Optional but good practice) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 6. Model Definition and Training (MLP Classifier) ---
# Architecture: 13 (input) -> 10 (hidden, ReLU) -> 1 (output, Sigmoid/Logistic)
mlp = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation='relu', # ReLU for hidden layer
    solver='adam',
    max_iter=500,
    random_state=42,
    verbose=False
)

mlp.fit(X_train, y_train)

# --- 7. Evaluation and Justification ---
accuracy = mlp.score(X_test, y_test)
print(f"Model Training Complete. Test Accuracy: {accuracy:.4f}")
print("\n--- Model Justification: MLP selected for non-linearity and ease of deployment. ---")
print("Architecture: 13 Features -> 10 Hidden Nodes (ReLU) -> 1 Output Node (Logistic)")

# --- 8. Extract Weights and Biases for Frontend Deployment ---

# Weights are stored as a list of NumPy arrays (one array per layer connection)
# 0: Input (13) -> Hidden (10)
# 1: Hidden (10) -> Output (1)
weights = [w.T.tolist() for w in mlp.coefs_]
biases = [b.tolist() for b in mlp.intercepts_]

# Combine all necessary data into a single JSON object
model_config = {
    'accuracy': round(accuracy, 4),
    'input_features': X.columns.tolist(),
    'scaling_params': scaling_params,
    'weights': weights,
    'biases': biases
}

# Save the configuration to a JSON file (optional, but good practice)
with open('model_config.json', 'w') as f:
    json.dump(model_config, f, indent=4)

print("\nSuccessfully extracted model configuration to 'model_config.json'.")
print("Weights and biases are ready to be embedded into index.html.")
print(f"\nScaling Means (for age, trestbps, chol, thalach, oldpeak): {scaling_params['mean']}")
print(f"Scaling Stds (for age, trestbps, chol, thalach, oldpeak): {scaling_params['std']}")

# --- Print embedded JavaScript model data ---
js_model_data = f"""
const MODEL_CONFIG = {json.dumps(model_config, separators=(',', ':'))};
"""
print("\n--- Copy-Paste the following block into index.html's <script> tag ---")
print(js_model_data)

# NOTE: The weights and biases will be manually copied into the index.html file 
# for a simple, single-file deployment.
