# Smart Retail Demand Predictor 🛒

A machine learning project that predicts expected product demand for a retail store-product combination based on retail, pricing, seasonal, and environmental factors.

## Project Overview

Retailers need to make informed inventory decisions. Ordering too much can lead to excess stock, while ordering too little can result in missed sales opportunities.

This project explores retail demand prediction using machine learning and provides a Gradio-based interface for entering store and product information and generating a predicted demand value.

## Features Used

The application accepts inputs related to:

- Date
- Store ID
- Product ID
- Category
- Region
- Inventory Level
- Units Ordered
- Price
- Discount
- Weather Condition
- Holiday / Promotion
- Competitor Pricing
- Seasonality

The application also derives date and pricing features before generating a prediction.

## Project Structure

```text
smart-retail-demand-predictor/
├── app.py
├── requirements.txt
├── retail_store_inventory.csv
├── smart_retail_demand_predictor.ipynb
├── retail_demand_model.pkl
├── model_features.pkl
└── README.md
```

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Gradio
- Joblib

## Installation

Clone the repository:

```bash
git clone https://github.com/lebow67/smart-retail-demand-predictor.git
cd smart-retail-demand-predictor
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

## Application Workflow

```text
Retail Inputs → Feature Processing → Trained ML Model → Demand Prediction
```

## Dataset and Notebook

The repository includes the dataset used for the project and the Jupyter Notebook containing the exploratory analysis, preprocessing, model training, and evaluation work.

## Author

**Pranil Kumar Walwandre**

B.Tech Student | Artificial Intelligence & Machine Learning
