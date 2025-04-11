# forecast_model.py
# Author: João Borboni
# Description: Trains a regression model to predict revenue and outputs results including date for Power BI use

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Load original dataset (includes Date column)
df = pd.read_csv('data/raw/sales_forecasting_dataset.csv')

# Save Date separately before transforming the dataset
dates = df['date']

# Define features and target
X = df.drop(columns=['revenue', 'date'])
y = df['revenue']

# Define categorical and numerical columns
categorical_cols = ['region', 'segment', 'product_category']
numerical_cols = ['units_sold', 'unit_price', 'discount_percent']

# Preprocessing pipeline for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'  # keep numerical columns
)

# Combine preprocessing and model into a pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Perform train/test split (and keep index for later merge with date)
X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(
    X, y, dates, test_size=0.2, random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict revenue on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained")
print(f"📊 MAE: {mae:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Combine test features, predictions, and original date for export
X_test_copy = X_test.copy()
X_test_copy['date'] = date_test.values
X_test_copy['actual_revenue'] = y_test.values
X_test_copy['predicted_revenue'] = y_pred

# Reorder columns for better readability
cols = ['date'] + [col for col in X_test_copy.columns if col != 'date']
X_test_copy = X_test_copy[cols]

# Save to CSV
output_path = '../data/raw/revenue_predictions.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
X_test_copy.to_csv(output_path, index=False)

print(f"💾 Predictions saved to '{output_path}'")


# Save model coefficients to CSV
coefs = model.named_steps['regressor'].coef_
features = model.named_steps['preprocessor'].get_feature_names_out()
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefs
})
coef_df.to_csv('data/raw/feature_coefficients.csv', index=False)
