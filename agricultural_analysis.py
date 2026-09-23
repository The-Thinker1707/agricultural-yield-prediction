# Agricultural Data Analysis & Yield Prediction Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set visual styling
sns.set_theme(style="whitegrid")
print("Libraries imported successfully!")

# Simulate Regional Agricultural Dataset
np.random.seed(42)
n_samples = 500

data = {
    'Rainfall_mm': np.random.normal(700, 150, n_samples),
    'Avg_Temperature_C': np.random.normal(26, 3, n_samples),
    'Soil_pH': np.random.normal(6.5, 0.6, n_samples),
    'Pest_Incident_Rate': np.random.randint(0, 5, n_samples),
    'Fertilizer_Usage_kg_ha': np.random.normal(120, 30, n_samples)
}

df = pd.DataFrame(data)
df['Crop_Yield_tons_ha'] = (
    0.004 * df['Rainfall_mm'] +
    -0.15 * df['Avg_Temperature_C'] +
    0.5 * df['Soil_pH'] +
    -0.4 * df['Pest_Incident_Rate'] +
    0.02 * df['Fertilizer_Usage_kg_ha'] +
    np.random.normal(0, 0.5, n_samples)
)

print("\nDataset Preview:")
display(df.head())

# Exploratory Data Analysis & Summary Statistics
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Train Machine Learning Model (Yield Prediction)
X = df[['Rainfall_mm', 'Avg_Temperature_C', 'Soil_pH', 'Pest_Incident_Rate', 'Fertilizer_Usage_kg_ha']]
y = df['Crop_Yield_tons_ha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model Evaluation Results ---")
print("Model Coefficients:", model.coef_)
print(f"R-squared Score: {r2_score(y_test, y_pred):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
