# ============================================
# 🧭 Travel Package Recommendation App
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


@st.cache_data
def load_data():
    df = pd.read_csv("realistic_final_travel_packages_dataset_v4.csv")  # update with your file name
    return df

df = load_data()
st.success(f"✅ Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns.")


st.header("🧳 Travel Package Recommendation System")

# From City
from_city = st.selectbox("✈️ Select your Departure City:", sorted(df["From_City"].unique()))

# Destination (filtered by from city if needed)
destinations_for_city = df[df["From_City"] == from_city]["Destination"].unique()
destination = st.selectbox("📍 Select your Destination:", sorted(destinations_for_city))

# Destination Type (filtered by selected destination)
destination_types_for_destination = df[df["Destination"] == destination]["Destination_Type"].unique()
destination_type = st.selectbox("🏖️ Select Destination Type:", sorted(destination_types_for_destination))

# Trip Duration
trip_duration = st.number_input("🕒 Trip_Duration_Days:", min_value=1, max_value=30, value=5)

# Approx Cost
approx_cost = st.number_input("💰Budget:", min_value=1000, step=500, value=20000)


features = ["From_City", "Destination", "Destination_Type"]
num_features = ["Trip_Duration_Days", "Budget"]

# Create a copy for processing
df_features = df.copy()

# Encode categorical
ohe = OneHotEncoder(handle_unknown="ignore")
encoded_cats = ohe.fit_transform(df_features[features]).toarray()
encoded_cats_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(features))

# Normalize numerical
scaler = MinMaxScaler()
scaled_nums = scaler.fit_transform(df_features[num_features])
scaled_nums_df = pd.DataFrame(scaled_nums, columns=num_features)

# Combine
X = np.hstack([encoded_cats_df, scaled_nums_df])


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


model = NearestNeighbors(n_neighbors=5, metric='cosine')
model.fit(X)
distances, indices = model.kneighbors(user_vector)


recommended_trips = df.iloc[indices[0]].copy()
recommended_trips["Similarity"] = 1 - distances[0]

st.subheader("🔹 Recommended Similar Trips:")
st.dataframe(recommended_trips[['Package_ID', 'From_City', 'Destination', 'Destination_Type',
       'Trip_Duration_Days', 'Budget',
       'Accommodation', 'Transport_Mode',  'Activities_Count',
        'Season', 'Package_Type']].assign(
    Similarity=recommended_trips["Similarity"].round(6)
))







