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
    df = pd.read_csv("realistic_final_travel_packages_dataset_v4.csv")  # update file name if needed
    df.columns = df.columns.str.strip()           # remove extra spaces
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
# Handle missing columns gracefully
def get_unique_safe(col_name):
    if col_name in df.columns:
        return sorted(df[col_name].dropna().unique())
    return []

from_city = st.selectbox("✈️ Select your Departure City:", get_unique_safe("From_City"))

destinations_for_city = []
if "From_City" in df.columns and "Destination" in df.columns:
    destinations_for_city = df[df["From_City"] == from_city]["Destination"].dropna().unique()

destination = st.selectbox("📍 Select your Destination:", sorted(destinations_for_city))

destination_types_for_destination = []
if "Destination" in df.columns and "Destination_Type" in df.columns:
    destination_types_for_destination = df[df["Destination"] == destination]["Destination_Type"].dropna().unique()

destination_type = st.selectbox("🏖️ Select Destination Type:", sorted(destination_types_for_destination))

trip_duration = st.number_input("🕒 Trip Duration (Days):", min_value=1, max_value=30, value=5)
budget = st.number_input("💰 Budget (₹):", min_value=1000, step=500, value=20000)
activities = st.number_input("🎯 Activities Count:", min_value=1, max_value=20, value=5)

# -------------------------------
# 4️⃣ Feature Setup
# -------------------------------
# Use realistic column names; adapt automatically to dataset
possible_features = [
    "From_City", "Destination", "Destination_Type", "Package_Type", "Season"
]
possible_num_features = ["Trip_Duration_Days", "Budget", "Activities_Count"]

# Only include those that actually exist
features = [f for f in possible_features if f in df.columns]
num_features = [f for f in possible_num_features if f in df.columns]

if not features or not num_features:
    st.error("❌ Required feature columns not found in dataset. Please check your CSV headers.")
    st.stop()

# -------------------------------
# 5️⃣ Preprocessing
# -------------------------------
df_features = df.copy()

# Encode categorical
ohe = OneHotEncoder(handle_unknown="ignore")
encoded_cats = ohe.fit_transform(df_features[features]).toarray()
encoded_cats_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(features))

# Normalize numerical
scaler = MinMaxScaler()
scaled_nums = scaler.fit_transform(df_features[num_features])
scaled_nums_df = pd.DataFrame(scaled_nums, columns=num_features)

# Combine encoded + scaled data
X = np.hstack([encoded_cats_df, scaled_nums_df])

# -------------------------------
# 6️⃣ Encode User Input
# -------------------------------
# Auto-fill Package_Type and Season based on selected destination if present
package_type = None
season = None
if "Package_Type" in df.columns and destination in df["Destination"].values:
    package_type = df[df["Destination"] == destination]["Package_Type"].iloc[0]
if "Season" in df.columns and destination in df["Destination"].values:
    season = df[df["Destination"] == destination]["Season"].iloc[0]

user_data = {
    "From_City": from_city,
    "Destination": destination,
    "Destination_Type": destination_type,
    "Trip_Duration_Days": trip_duration,
    "Budget": budget,
    "Activities_Count": activities,
}
if package_type is not None:
    user_data["Package_Type"] = package_type
if season is not None:
    user_data["Season"] = season

user_df = pd.DataFrame([user_data])

# Match columns safely
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

# Dynamically find displayable columns
preferred_cols = [
    'Package_ID', 'From_City', 'Destination', 'Destination_Type',
    'Trip_Duration_Days', 'Activities_Count', 'Accommodation',
    'Transport_Mode', 'Package_Type', 'Budget', 'Season'
]

available_cols = [c for c in preferred_cols if c in recommended_trips.columns]

if not available_cols:
    st.warning("⚠️ No matching columns found to display. Showing all available columns instead.")
    available_cols = recommended_trips.columns.tolist()

st.dataframe(
    recommended_trips[available_cols].assign(
        Similarity=recommended_trips["Similarity"].round(4)
    )
)
