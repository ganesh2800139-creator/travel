# ================================================
# 🧭 Travel Package Recommendation System (Streamlit)
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# ================================================
# 🪄 Load Dataset
# ================================================
@st.cache_data
def load_data():
    df = pd.read_csv("C://Users//ganes//OneDrive//Documents//travel_packages_120000.csv")   # Ensure this CSV is in the same folder
    return df

df = load_data()

# ================================================
# 🧰 Define Columns
# ================================================
cat_cols = ['From_City', 'Destination', 'Destination_Type', 
            'Budget_Range', 'Accommodation_Type', 'Transport_Mode', 
            'Meal_Plan', 'Activity_Types', 'Season', 
            'Package_Type', 'Recommended_For']

num_cols = ['Trip_Duration_Days', 'Approx_Cost (₹)', 'Activity_Count']

# ================================================
# 🧼 Preprocessing
# ================================================
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
cat_features = ohe.fit_transform(df[cat_cols])

scaler = MinMaxScaler()
num_features = scaler.fit_transform(df[num_cols])

cdata = np.hstack([num_features, cat_features])

# ================================================
# 🧠 Fit Nearest Neighbors Model
# ================================================
cosinemodel = NearestNeighbors(n_neighbors=5, metric='cosine')
cosinemodel.fit(cdata)

# ================================================
# 🧑 Streamlit UI
# ================================================
st.set_page_config(page_title="Travel Package Recommender", page_icon="✈️", layout="wide")
st.title("✈️ Travel Package Recommendation System")
st.write("Get personalized travel package recommendations based on your preferences.")

# --- User Inputs ---
with st.form("user_input_form"):
    col1, col2 = st.columns(2)
    with col1:
        from_city = st.selectbox("From City", sorted(df['From_City'].unique()))
        destination = st.selectbox("Destination", sorted(df['Destination'].unique()))
        destination_type = st.selectbox("Destination Type", sorted(df['Destination_Type'].unique()))
        budget_range = st.selectbox("Budget Range", sorted(df['Budget_Range'].unique()))
        accommodation = st.selectbox("Accommodation Type", sorted(df['Accommodation_Type'].unique()))
        transport = st.selectbox("Transport Mode", sorted(df['Transport_Mode'].unique()))

    with col2:
        meal_plan = st.selectbox("Meal Plan", sorted(df['Meal_Plan'].unique()))
        activity_types = st.selectbox("Activity Types", sorted(df['Activity_Types'].unique()))
        season = st.selectbox("Season", sorted(df['Season'].unique()))
        package_type = st.selectbox("Package Type", sorted(df['Package_Type'].unique()))
        recommended_for = st.selectbox("Recommended For", sorted(df['Recommended_For'].unique()))
        trip_duration = st.number_input("Trip Duration (Days)", min_value=1, value=5)
        approx_cost = st.number_input("Approx Cost (₹)", min_value=0, value=30000, step=1000)
        activity_count = st.number_input("Activity Count", min_value=1, value=3)

    submit_button = st.form_submit_button("🚀 Get Recommendations")

# ================================================
# 🔍 Recommendation Logic
# ================================================
if submit_button:
    # Build user input dictionary
    user_input = {
        'From_City': from_city,
        'Destination': destination,
        'Destination_Type': destination_type,
        'Budget_Range': budget_range,
        'Accommodation_Type': accommodation,
        'Transport_Mode': transport,
        'Meal_Plan': meal_plan,
        'Activity_Types': activity_types,
        'Season': season,
        'Package_Type': package_type,
        'Recommended_For': recommended_for,
        'Trip_Duration_Days': trip_duration,
        'Approx_Cost (₹)': approx_cost,
        'Activity_Count': activity_count
    }

    user_df = pd.DataFrame([user_input])

    # Encode and scale user input
    user_cat = ohe.transform(user_df[cat_cols])
    user_num = scaler.transform(user_df[num_cols])
    user_vector = np.hstack([user_num, user_cat])

    # Get recommendations
    distances, indices = cosinemodel.kneighbors(user_vector)
    top_packages = df.iloc[indices[0]].copy()
    top_packages['Similarity_Score'] = 1 - distances.flatten()

    # Display top recommendations
    st.subheader("🎯 Top Recommended Packages")
    st.dataframe(
        top_packages[['Package_ID', 'Destination', 'Trip_Duration_Days', 
                      'Approx_Cost (₹)', 'Accommodation_Type', 
                      'Package_Type', 'Similarity_Score']].reset_index(drop=True)
    )

    # Optional: download option
    csv = top_packages.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Recommendations as CSV",
        data=csv,
        file_name="recommended_packages.csv",
        mime="text/csv"
    )
