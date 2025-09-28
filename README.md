# E-commerce-dashboard
E-commerce Furniture Popularity Predictor
 Project Description
E-commerce Furniture Popularity Predictor is a full-stack machine learning project that analyzes product performance in a furniture marketplace. It uses simplified product categories and shipping tags to predict whether a product will be popular, and presents insights through an interactive Streamlit dashboard.
This project demonstrates end-to-end ML pipeline development, including:
- Feature Engineering: Simplified product types from raw titles, encoded shipping tags
- Class Balancing: SMOTE applied to address popularity imbalance
- Modeling: Random Forest classifier trained and evaluated with precision/recall metrics
- 📊 Dashboard: Streamlit app with top sellers, price distribution, and prediction interface
- Business Impact: Helps sellers identify high-performing product types and optimize listings
Ideal for showcasing skills in ML deployment, BI visualization, and real-world data handling.


# 🪑 E-commerce Furniture Popularity Predictor

This project uses machine learning to predict product popularity based on simplified categories and shipping tags. It includes a Streamlit dashboard for interactive exploration.

## Features
- Data cleaning and feature engineering
- SMOTE-based class balancing
- Random Forest model training
- Streamlit dashboard with charts and prediction interface

## Files
- `E-commerce Furniture.py`: Model training script
- `ecommerce_dashboard.py`: Streamlit dashboard
- `popularity_model.pkl`: Trained model
- `ecommerce_furniture_dataset_2024.csv`: Dataset

## How to Run
```bash
pip install -r requirements.txt
streamlit run ecommerce_dashboard.py
