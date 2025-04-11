# forecast_model.py
# Author: João Borboni
# Description: Trains a regression model to predict revenue and saves results including full date info

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

# Ensure consistent lowercase column names
df.columns = [col.lower() for col in df.columns]

# Make sure 'date' is datetime type
df['date'] = pd.to_datetime(df['date'])

# Separate features and target
X = df.drop(columns=['revenue'])
y = df['revenue']

# Split keeping track of original indices
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Store original date from df using the index of X_test
dates_test = df.loc[X_test.index, 'date']

# Prepare inputs for model (drop 'date' only at this step)
X_train_model = X_train.drop(columns=['date'])
X_test_model = X_test.drop(columns=['date'])

# Define preprocessing pipeline
categorical_cols = ['region', 'segment', 'product_category']
numerical_cols = ['units_sold', 'unit_price', 'discount_percent']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)

# Pipeline with model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model.fit(X_train_model, y_train)

# Predict
y_pred = model.predict(X_test_model)

# Create final result DataFrame
result_df = X_test_model.copy()
result_df['date'] = dates_test.values
result_df['actual_revenue'] = y_test.values
result_df['predicted_revenue'] = y_pred

# Reorder columns
column_order = ['date'] + [col for col in result_df.columns if col != 'date']
result_df = result_df[column_order]

# Save to CSV
output_path = 'data/raw/revenue_predictions.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
result_df.to_csv(output_path, index=False)

print(f"✅ Model trained and predictions saved to '{output_path}'")

# Save model coefficients to CSV
coefs = model.named_steps['regressor'].coef_
features = model.named_steps['preprocessor'].get_feature_names_out()
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefs
})
coef_df.to_csv('data/raw/feature_coefficients.csv', index=False)

# Create summary metrics as a DataFrame

from sklearn.metrics import mean_absolute_error, r2_score

summary_rows = []

# Add global metrics
mae_global = mean_absolute_error(result_df['actual_revenue'], result_df['predicted_revenue'])
r2_global = r2_score(result_df['actual_revenue'], result_df['predicted_revenue'])

summary_rows.append({
    'group_type': 'global',
    'group_value': 'all',
    'mae': round(mae_global, 2),
    'r2_score': round(r2_global, 3)
})

# Add region-level metrics
for region, group_df in result_df.groupby('region'):
    mae = mean_absolute_error(group_df['actual_revenue'], group_df['predicted_revenue'])
    r2 = r2_score(group_df['actual_revenue'], group_df['predicted_revenue'])
    summary_rows.append({
        'group_type': 'region',
        'group_value': region,
        'mae': round(mae, 2),
        'r2_score': round(r2, 3)
    })

# Add segment-level metrics
for segment, group_df in result_df.groupby('segment'):
    mae = mean_absolute_error(group_df['actual_revenue'], group_df['predicted_revenue'])
    r2 = r2_score(group_df['actual_revenue'], group_df['predicted_revenue'])
    summary_rows.append({
        'group_type': 'segment',
        'group_value': segment,
        'mae': round(mae, 2),
        'r2_score': round(r2, 3)
    })

# Create DataFrame
metrics_df = pd.DataFrame(summary_rows)

# Save to CSV
metrics_output_path = 'data/raw/model_performance_summary.csv'
metrics_df.to_csv(metrics_output_path, index=False)

print(f"📁 Grouped performance metrics saved to '{metrics_output_path}'")
