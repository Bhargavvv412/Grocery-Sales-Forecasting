import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---- Load Encoders and Model ----
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = joblib.load(file)

with open('onehot_encoder.pkl', 'rb') as file:
    onehot_encoder = joblib.load(file)

model = joblib.load("model1.pkl")

st.set_page_config(page_title="Grocery Sales Forecast", layout="centered")
st.title("\U0001F6D2 Grocery Sales Forecasting")

# ---- Input UI ----
with st.form("input_form"):
    st.subheader("Store & Item Details")
    col1, col2 = st.columns(2)
    with col1:
        store_nbr = st.number_input("Store Number", min_value=1, max_value=54)
        item_nbr = st.number_input("Item Number", min_value=1, max_value=4100)
        class_ = st.number_input("Class", min_value=1, max_value=5500)
        perishable = st.selectbox("Perishable", [0, 1])
    with col2:
        family = st.selectbox("Family", options=label_encoders['family'].classes_)
        city = st.selectbox("City", options=label_encoders['city'].classes_)
        state = st.selectbox("State", options=label_encoders['state'].classes_)
        cluster = st.selectbox("Cluster", list(range(1, 19)))

    st.subheader("Holiday Information")
    col3, col4 = st.columns(2)
    with col3:
        holiday_type = st.selectbox("Holiday Type", ['Holiday', 'None', 'Work Day'])
    with col4:
        holiday_locale = st.selectbox("Holiday Locale", ['National', 'None'])

    st.subheader("Temporal & Sales Features")
    col5, col6 = st.columns(2)
    with col5:
        year = st.selectbox("Year", [2016, 2017])
        month = st.selectbox("Month", list(range(1, 13)))
        day = st.selectbox("Day", list(range(1, 32)))
    with col6:
        dayofweek = st.selectbox("Day of Week", list(range(0, 7)))
        is_weekend = st.selectbox("Is Weekend", [0, 1])

    st.subheader("Sales Stats")
    transactions = st.number_input("Transactions", value=300)
    total_store_sales = st.number_input("Total Store Sales", value=10000.0)
    avg_item_sales = st.number_input("Average Item Sales", value=5.0)
    sales_per_transaction = st.number_input("Sales per Transaction", value=20.0)

    submitted = st.form_submit_button("Predict Unit Sales")

# ---- Encode Inputs ----
if submitted:
    try:
        # Label Encoding
        family_enc = label_encoders['family'].transform([family])[0]
        city_enc = label_encoders['city'].transform([city])[0]
        state_enc = label_encoders['state'].transform([state])[0]

        # One-hot Encoding
        cat_df = pd.DataFrame([[holiday_type, holiday_locale]], columns=['holiday_type', 'holiday_locale'])
        onehot_encoded = onehot_encoder.transform(cat_df).toarray()[0]

        # Combine all features
        features = [
            store_nbr, item_nbr, family_enc, class_, perishable, city_enc, state_enc, cluster,
            transactions, year, month, day, dayofweek, is_weekend, total_store_sales,
            avg_item_sales, sales_per_transaction
        ]

        final_input = np.concatenate([features, onehot_encoded])

        # Prediction
        prediction = model.predict([final_input])[0]
        st.success(f"\U0001F4E6 Predicted Unit Sales: {prediction:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
