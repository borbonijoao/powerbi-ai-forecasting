# forecast_model.py
# Author: João Borboni
# Description: Trains a regression model to predict revenue based on sales data

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv('data/raw/sales_forecasting_dataset.csv')

# Features & Target
X = df.drop(columns=['revenue', 'date'])
y = df['revenue']

# Categorical and numerical columns
categorical_cols = ['region', 'segment', 'product_category']
numerical_cols = ['units_sold', 'unit_price', 'discount_percent']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'  # keep numerical columns
)

# Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained")
print(f"📊 MAE: {mae:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Save predictions to CSV
X_test_copy = X_test.copy()
X_test_copy['actual_revenue'] = y_test.values
X_test_copy['predicted_revenue'] = y_pred

output_path = 'data/raw/revenue_predictions.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
X_test_copy.to_csv(output_path, index=False)

print(f"💾 Predictions saved to '{output_path}'")
