import gradio as gr
import pandas as pd
import joblib


# ==========================================
# LOAD MODEL AND FEATURE INFORMATION
# ==========================================

model = joblib.load("retail_demand_model.pkl")
model_features = joblib.load("model_features.pkl")


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_demand(
    date,
    store_id,
    product_id,
    category,
    region,
    inventory_level,
    units_ordered,
    price,
    discount,
    weather_condition,
    holiday_promotion,
    competitor_pricing,
    seasonality
):

    # Convert date to datetime
    date = pd.to_datetime(date)

    # Extract date features
    year = date.year
    month = date.month
    day_of_week = date.dayofweek
    weekend = int(day_of_week >= 5)

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Store ID": [store_id],
        "Product ID": [product_id],
        "Category": [category],
        "Region": [region],
        "Inventory Level": [inventory_level],
        "Units Ordered": [units_ordered],
        "Price": [price],
        "Discount": [discount],
        "Weather Condition": [weather_condition],
        "Holiday/Promotion": [holiday_promotion],
        "Competitor Pricing": [competitor_pricing],
        "Seasonality": [seasonality],

        # Date features
        "Year": [year],
        "Month": [month],
        "DayOfWeek": [day_of_week],
        "Weekend": [weekend],

        # Engineered features
        "Price Difference": [
            price - competitor_pricing
        ],

        "Price Ratio": [
            price / competitor_pricing
            if competitor_pricing != 0
            else 0
        ]
    })

    # One-hot encode categorical variables
    input_encoded = pd.get_dummies(input_data)

    # Make sure input has exactly the same
    # columns as the training data
    input_encoded = input_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    # Convert everything to float
    input_encoded = input_encoded.astype(float)

    # Make prediction
    prediction = model.predict(input_encoded)[0]

    # Prevent negative predictions
    prediction = max(0, prediction)

    return f"Predicted Demand: {prediction:.0f} units"


# ==========================================
# GRADIO INTERFACE
# ==========================================

with gr.Blocks(
    title="Smart Retail Demand Predictor"
) as demo:

    gr.Markdown(
        """
        # 🛒 Smart Retail Demand Predictor

        Predict the expected number of units sold
        for a store-product combination.
        """
    )

    gr.Markdown(
        """
        Enter the retail information below and click
        **Predict Demand**.
        """
    )

    # ======================================
    # INPUT SECTION
    # ======================================

    with gr.Row():

        # ------------------------------
        # LEFT COLUMN
        # ------------------------------

        with gr.Column():

            date = gr.Textbox(
                label="Date",
                value="2022-07-15",
                placeholder="YYYY-MM-DD"
            )

            store_id = gr.Dropdown(
                choices=[
                    "S001",
                    "S002",
                    "S003",
                    "S004",
                    "S005"
                ],
                label="Store ID",
                value="S001"
            )

            product_id = gr.Dropdown(
                choices=[
                    f"P{i:04d}"
                    for i in range(1, 21)
                ],
                label="Product ID",
                value="P0001"
            )

            category = gr.Textbox(
                label="Category",
                value="Electronics"
            )

            region = gr.Dropdown(
                choices=[
                    "North",
                    "South",
                    "East",
                    "West"
                ],
                label="Region",
                value="North"
            )

            inventory_level = gr.Number(
                label="Inventory Level",
                value=150
            )

            units_ordered = gr.Number(
                label="Units Ordered",
                value=100
            )

        # ------------------------------
        # RIGHT COLUMN
        # ------------------------------

        with gr.Column():

            price = gr.Number(
                label="Price",
                value=50
            )

            discount = gr.Number(
                label="Discount (%)",
                value=10
            )

            weather_condition = gr.Dropdown(
                choices=[
                    "Sunny",
                    "Cloudy",
                    "Rainy",
                    "Snowy"
                ],
                label="Weather Condition",
                value="Sunny"
            )

            holiday_promotion = gr.Radio(
                choices=[0, 1],
                label="Holiday / Promotion",
                value=0
            )

            competitor_pricing = gr.Number(
                label="Competitor Pricing",
                value=48
            )

            seasonality = gr.Dropdown(
                choices=[
                    "Spring",
                    "Summer",
                    "Autumn",
                    "Winter"
                ],
                label="Seasonality",
                value="Summer"
            )

    # ======================================
    # PREDICTION BUTTON
    # ======================================

    predict_button = gr.Button(
        "🔮 Predict Demand",
        variant="primary"
    )

    # ======================================
    # OUTPUT
    # ======================================

    output = gr.Textbox(
        label="Prediction",
        interactive=False
    )

    # ======================================
    # BUTTON ACTION
    # ======================================

    predict_button.click(
        fn=predict_demand,
        inputs=[
            date,
            store_id,
            product_id,
            category,
            region,
            inventory_level,
            units_ordered,
            price,
            discount,
            weather_condition,
            holiday_promotion,
            competitor_pricing,
            seasonality
        ],
        outputs=output
    )


# ==========================================
# LAUNCH APP
# ==========================================

if __name__ == "__main__":
    demo.launch()
