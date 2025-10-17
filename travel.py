# travel_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack

st.set_page_config(page_title="Travel Package Recommendation", layout="wide")
st.title("🌍 Travel Package Recommendation App")

# ---------------------------------------------------------
# 1️⃣ Load and Prepare Data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("travel_packages_120000.csv")
    df.rename(columns={'Approx_Cost (₹)': 'Approx_Cost'}, inplace=True)
    return df

df = load_data()
st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

cat_cols = ['From_City', 'Destination', 'Destination_Type', 'Budget_Range', 
            'Accommodation_Type', 'Transport_Mode', 'Meal_Plan', 
            'Activity_Types', 'Season', 'Package_Type', 'Recommended_For']

num_cols = ['Trip_Duration_Days', 'Approx_Cost', 'Activity_Count']

# ---------------------------------------------------------
# 2️⃣ Preprocessing (Encoding + Scaling)
# ---------------------------------------------------------
@st.cache_resource
def preprocess_data(df):
    ohe = OneHotEncoder(handle_unknown='ignore')
    scaler = StandardScaler()

    encoded_cats = ohe.fit_transform(df[cat_cols])
    scaled_nums = scaler.fit_transform(df[num_cols])
    cdata = hstack([scaled_nums, encoded_cats])

    cosinemodel = NearestNeighbors(n_neighbors=5, metric='cosine')
    cosinemodel.fit(cdata)

    return ohe, scaler, cosinemodel, cdata

ohe, scaler, cosinemodel, cdata = preprocess_data(df)
st.success("✅ Model trained successfully!")

# ---------------------------------------------------------
# 3️⃣ User Input Section
# ---------------------------------------------------------
st.header("✈️ Enter Your Travel Preferences")

user_data = {}

# Categorical Inputs
for col in cat_cols:
    user_data[col] = st.selectbox(f"Select {col}", df[col].unique())

# Numeric Inputs
for col in num_cols:
    user_data[col] = st.number_input(
        f"Enter {col} (Range: {df[col].min()} - {df[col].max()})",
        min_value=float(df[col].min()),
        max_value=float(df[col].max()),
        value=float(df[col].mean())
    )

user_df = pd.DataFrame([user_data])
st.subheader("Your Input:")
st.dataframe(user_df)

# ---------------------------------------------------------
# 4️⃣ Recommendations
# ---------------------------------------------------------
if st.button("🔍 Recommend Packages"):
    # Transform input
    user_cat = ohe.transform(user_df[cat_cols])
    user_num = scaler.transform(user_df[num_cols])
    user_vector = np.hstack([user_num, user_cat.toarray()])

    # Find nearest neighbors
    distances, indices = cosinemodel.kneighbors(user_vector)
    top_packages = df.iloc[indices[0]].copy()
    top_packages['Similarity_Score'] = 1 - distances.flatten()

    # Display top recommendations
    top_packages_display = top_packages[['From_City', 'Destination', 'Destination_Type',
                                         'Trip_Duration_Days', 'Budget_Range', 'Approx_Cost',
                                         'Accommodation_Type', 'Transport_Mode', 'Activity_Count',
                                         'Package_Type', 'Similarity_Score']]
    st.subheader("🎯 Top Recommended Packages")
    st.dataframe(top_packages_display)
