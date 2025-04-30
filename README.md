# Power BI + AI Forecasting Project

This project demonstrates how to apply machine learning for sales forecasting and explainability within a Power BI reporting environment. It is designed to serve as a public portfolio project and a learning resource for data professionals interested in combining advanced analytics with business intelligence.

---

## 🔍 Project Overview

We simulate a real-world sales dataset and apply forecasting techniques using Python (scikit-learn), along with visual explanations of model outputs. The final output is delivered through an interactive Power BI report, with three dedicated pages:

# Python-Powered Forecast
Leverages Python scripts (scikit-learn) to build a linear regression model on historical sales data. Data preparation, model training, and forecast generation occur in Python, then are visualized in Power BI to compare actual vs. predicted revenue.

# DAX-Driven Insights
All metrics and forecasts are calculated using DAX within Power BI. The regression model is recreated with LINESTX and related functions, providing a fully in-model solution that highlights the power of native DAX analytics.

# Interactive What-If Scenarios
Dynamic scenario analysis using a parameter-driven moving average forecast. End users can select the number of months for the rolling window, instantly updating the forecast—ideal for “what-if” planning.

---

## 📁 Project Structure

📁 data/raw/
📁 notebooks/
📁 scripts/
📁 visuals/screenshots/
📁 powerbi/

---

## 💡 Tools & Technologies

- **Power BI Desktop** – for dashboard creation
- **Python** – data generation, modeling, and explainability
- **Scikit-learn** – regression modeling
- **Git + GitHub** – version control and project sharing
- **Jupyter Notebooks / VSCode** – for development
- **Optional: Supabase / BigQuery / PostgreSQL** – for scalable storage

---

## 🚀 Workflow & Key Concepts

- **Data Generation** via Python notebook and CSV export
- **Python Regression:** Linear regression forecasts via scikit-learn
- **Native DAX Regression:** LINESTX-based models inside Power BI
- **Moving Average Forecast:** Parameter-driven DAX measures
- **What-If Analysis:** Dynamic scenario planning with DAX parameters
- **Visualization:** Interactive time series, bar charts, and scenario sliders

---

## 📊 Forecasting Model

A regression model was trained using Python (scikit-learn) to predict Revenue based on historical sales data. The following steps were included:
- Data preprocessing using a Pipeline (OneHotEncoder + passthrough for numeric features)
- Train/Test split for validation
- Evaluation metrics:
-   MAE (Mean Absolute Error)
-   R² Score
- Predictions were saved to a CSV file for Power BI use

📁 Output file: data/raw/revenue_predictions.csv

You can find the modeling code in:
➡️ scripts/forecast_model.py

---

## 🤝 Contribution

This is a solo public project but feel free to fork, reuse, or give feedback. The goal is to contribute to the community and improve learning around modern BI & AI practices.

---

## 📧 Contact

Made with ❤️ by João Borboni  
[LinkedIn](https://www.linkedin.com/in/joao-borboni/) | [Email](mailto:joaoborboni@hotmail.com)
