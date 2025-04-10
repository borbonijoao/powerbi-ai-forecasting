# Power BI + AI Forecasting Project

This project demonstrates how to apply machine learning for sales forecasting and explainability within a Power BI reporting environment. It is designed to serve as a public portfolio project and a learning resource for data professionals interested in combining advanced analytics with business intelligence.

---

## 🔍 Project Overview

We simulate a real-world sales dataset and apply forecasting techniques using Python (scikit-learn), along with visual explanations of model outputs. The final output is delivered through an interactive Power BI report.

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

## 🚀 Key Concepts

- Forecasting sales using regression (e.g., linear, ridge)
- Explainability of models using feature importance
- Integration of Python scripts into Power BI
- Clear data storytelling through KPIs and visuals

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
