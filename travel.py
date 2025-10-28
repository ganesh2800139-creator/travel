# ============================================
# 🧭 Travel Package Recommendation App
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# -------------------------------
# 1️⃣ Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("realistic_final_travel_packages_dataset_v4.csv")  # update with your file name
    return df

df = load_data()
st.success(f"✅ Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns.")

st.header("🧳 Travel Package Recommendation System")

# -------------------------------
# 2️⃣ User Inputs
# -------------------------------
# From City
from_city = st.selectbox("✈️ Select your Departure City:", sorted(df["From_City"].unique()))

# Destination (filtered by From City if needed)
destinations_for_city = df[df["From_City"] == from_city]["Destination"].unique()
destination = st.selectbox("📍 Select your Destination:", sorted(destinations_for_city))

# Destination Type (filtered by selected Destination)
destination_types_for_destination = df[df["Destination"] == destination]["Destination_Type"].unique()
destination_type = st.selectbox("🏖️ Select Destination Type:", sorted(destination_types_for_destination))

# Trip Duration
trip_duration = st.number_input("🕒 Trip Duration (Days):", min_value=1, max_value=30, value=5)

# Approx Budget
budget = st.number_input("💰 Budget:", min_value=1000, step=500, value=20000)

# -------------------------------
# 3️⃣ Feature Preparation
# -------------------------------
features = ["From_City", "Destination", "Destination_Type"]
num_features = ["Trip_Duration_Days", "Budget"]

# Copy data for processing
df_features = df.copy()

# Encode categorical
ohe = OneHotEncoder(handle_unknown="ignore")
encoded_cats = ohe.fit_transform(df_features[features]).toarray()
encoded_cats_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(features))

# Normalize numerical
scaler = MinMaxScaler()
scaled_nums = scaler.fit_transform(df_features[num_features])
scaled_nums_df = pd.DataFrame(scaled_nums, columns=num_features)

# Combine features
X = np.hstack([encoded_cats_df, scaled_nums_df])

# -------------------------------
# 4️⃣ Prepare User Input Vector
# -------------------------------
user_df = pd.DataFrame({
    "From_City": [from_city],
    "Destination": [destination],
    "Destination_Type": [destination_type],
    "Trip_Duration_Days": [trip_duration],
    "Budget": [budget]
})

user_encoded = ohe.transform(user_df[features]).toarray()
user_scaled = scaler.transform(user_df[num_features])
user_vector = np.hstack([user_encoded, user_scaled])

# -------------------------------
# 5️⃣ Nearest Neighbors Model
# -------------------------------
model = NearestNeighbors(n_neighbors=5, metric='cosine')
model.fit(X)
distances, indices = model.kneighbors(user_vector)

# -------------------------------
# 6️⃣ Get Recommendations
# -------------------------------
recommended_trips = df.iloc[indices[0]].copy()
recommended_trips["Similarity"] = 1 - distances[0]

# 🔧 Remove duplicate columns (fix ValueError)
recommended_trips = recommended_trips.loc[:, ~recommended_trips.columns.duplicated()]

# 🧹 Clean column names (trim extra spaces)
recommended_trips.columns = recommended_trips.columns.str.strip()

# Columns to show in UI
columns_to_show = [
    'Package_Type', 'From_City', 'Destination', 'Destination_Type',
    'Trip_Duration_Days', 'Budget',
    'Accommodation', 'Transport_Mode', 'Activities_Count',
    'Season', 'Similarity'
]

# Show only existing columns (avoid KeyError)
available_columns = [col for col in columns_to_show if col in recommended_trips.columns]

# -------------------------------
# 7️⃣ Display in Streamlit
# -------------------------------
st.subheader("🔹 Recommended Similar Trips:")

if not available_columns:
    st.error("⚠️ No valid columns available to display. Please check your dataset headers.")
else:
    st.dataframe(
        recommended_trips[available_columns].assign(
            Similarity=recommended_trips["Similarity"].round(6)
            if "Similarity" in recommended_trips.columns else None
        )
    )
