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
    df = pd.read_csv("realistic_final_travel_packages_dataset_v4.csv")  # ensure correct file path
    df.columns = df.columns.str.strip()  # clean up extra spaces
    return df

df = load_data()
st.success(f"✅ Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns.")

# -------------------------------
# 2️⃣ App Header
# -------------------------------
st.header("🧳 Travel Package Recommendation System")

# -------------------------------
# 3️⃣ User Inputs
# -------------------------------
# From City
from_city = st.selectbox("✈️ Select your Departure City:", sorted(df["From_City"].dropna().unique()))

# Destination (filtered by From City)
destinations_for_city = df[df["From_City"] == from_city]["Destination"].dropna().unique()
destination = st.selectbox("📍 Select your Destination:", sorted(destinations_for_city))

# Destination Type
destination_types_for_destination = df[df["Destination"] == destination]["Destination_Type"].dropna().unique()
destination_type = st.selectbox("🏖️ Select Destination Type:", sorted(destination_types_for_destination))

# Trip Duration
trip_duration = st.number_input("🕒 Trip Duration (Days):", min_value=1, max_value=30, value=5)

# Budget
budget = st.number_input("💰 Budget (₹):", min_value=1000, step=500, value=20000)

# Activities Count
activities = st.number_input("🎯 Activities Count:", min_value=1, max_value=20, value=5)

# -------------------------------
# 4️⃣ Feature Setup
# -------------------------------
features = ["From_City", "Destination", "Destination_Type", "Package_Type", "Season"]
num_features = ["Trip_Duration_Days", "Budget", "Activities_Count"]

# Ensure all required columns exist
missing = [c for c in features + num_features if c not in df.columns]
if missing:
    st.error(f"❌ Missing columns in dataset: {missing}")
    st.stop()

# -------------------------------
# 5️⃣ Preprocessing
# -------------------------------
df_features = df.copy()

# Encode categorical features
ohe = OneHotEncoder(handle_unknown="ignore")
encoded_cats = ohe.fit_transform(df_features[features]).toarray()
encoded_cats_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(features))

# Normalize numerical features
scaler = MinMaxScaler()
scaled_nums = scaler.fit_transform(df_features[num_features])
scaled_nums_df = pd.DataFrame(scaled_nums, columns=num_features)

# Combine encoded + scaled data
X = np.hstack([encoded_cats_df, scaled_nums_df])

# -------------------------------
# 6️⃣ Encode User Input
# -------------------------------
# Use Package_Type and Season automatically from selected destination
package_type = df[df["Destination"] == destination]["Package_Type"].iloc[0]
season = df[df["Destination"] == destination]["Season"].iloc[0]

user_df = pd.DataFrame({
    "From_City": [from_city],
    "Destination": [destination],
    "Destination_Type": [destination_type],
    "Package_Type": [package_type],
    "Season": [season],
    "Trip_Duration_Days": [trip_duration],
    "Budget": [budget],
    "Activities_Count": [activities],
})

user_encoded = ohe.transform(user_df[features]).toarray()
user_scaled = scaler.transform(user_df[num_features])
user_vector = np.hstack([user_encoded, user_scaled])

# -------------------------------
# 7️⃣ Nearest Neighbors Model
# -------------------------------
model = NearestNeighbors(n_neighbors=5, metric='cosine')
model.fit(X)
distances, indices = model.kneighbors(user_vector)

recommended_trips = df.iloc[indices[0]].copy()
recommended_trips["Similarity"] = (1 - distances[0])

# -------------------------------
# 8️⃣ Display Recommendations
# -------------------------------
st.subheader("🔹 Recommended Similar Trips:")
st.dataframe(
    recommended_trips[
        [
            'Package_ID', 'From_City', 'Destination', 'Destination_Type',
            'Trip_Duration_Days', 'Activities_Count', 'Accommodation',
            'Transport_Mode', 'Package_Type', 'Budget', 'Season'
        ]
    ].assign(Similarity=recommended_trips["Similarity"].round(4))
)
