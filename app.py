# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor

# Load encoders and model
label_encoders = joblib.load("label_encoders.pkl")
onehot_encoder = joblib.load("onehot_encoder.pkl")
model = joblib.load("model1.pkl")

st.title("Retail Sales Prediction App")

# Input fields
store_nbr = st.number_input("Store Number", min_value=1, step=1)
item_nbr = st.number_input("Item Number", min_value=1, step=1)
family = st.selectbox("Family", label_encoders['family'].classes_)
item_class = st.number_input("Item Class", min_value=1, step=1)
perishable = st.selectbox("Perishable", [0, 1])
city = st.selectbox("City", label_encoders['city'].classes_)
state = st.selectbox("State", label_encoders['state'].classes_)
cluster = st.number_input("Cluster", min_value=1, step=1)
transactions = st.number_input("Transactions")
year = st.number_input("Year", min_value=2000, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)
day = st.number_input("Day", min_value=1, max_value=31, step=1)
dayofweek = st.number_input("Day of Week", min_value=0, max_value=6, step=1)
is_weekend = st.selectbox("Is Weekend", [0, 1])
total_store_sales = st.number_input("Total Store Sales")
avg_item_sales = st.number_input("Average Item Sales")
sales_per_transaction = st.number_input("Sales per Transaction")
holiday_type = st.selectbox("Holiday Type", onehot_encoder.categories_[0])
holiday_locale = st.selectbox("Holiday Locale", onehot_encoder.categories_[1])

if st.button("Predict"):
    # Encode label columns
    input_data = {
        "store_nbr": store_nbr,
        "item_nbr": item_nbr,
        "family": label_encoders['family'].transform([family])[0],
        "class": item_class,
        "perishable": perishable,
        "city": label_encoders['city'].transform([city])[0],
        "state": label_encoders['state'].transform([state])[0],
        "cluster": cluster,
        "transactions": transactions,
        "year": year,
        "month": month,
        "day": day,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend,
        "total_store_sales": total_store_sales,
        "avg_item_sales": avg_item_sales,
        "sales_per_transaction": sales_per_transaction,
    }
    df_input = pd.DataFrame([input_data])

    # One-hot encode categorical columns
    onehot_array = onehot_encoder.transform([[holiday_type, holiday_locale]]).toarray()
    onehot_cols = onehot_encoder.get_feature_names_out()
    onehot_df = pd.DataFrame(onehot_array, columns=onehot_cols)

    df_input.reset_index(drop=True, inplace=True)
    onehot_df.reset_index(drop=True, inplace=True)

    final_input = pd.concat([df_input, onehot_df], axis=1)

    # Predict
    prediction = model.predict(final_input)[0]
    st.success(f"Predicted Unit Sales: {prediction:.4f}")