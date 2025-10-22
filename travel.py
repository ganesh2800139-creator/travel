# ============================================
# 📌 TRAVEL PACKAGE RECOMMENDATION SYSTEM (Streamlit App)
# ============================================

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# STEP 1: Load Dataset
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv("travel_packages_120000.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()
st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================
# STEP 2: Preprocess Data
# ============================================
feature_cols = [
    'Destination_Type',
    'Trip_Duration_Days',
    'Approx_Cost (₹)',
    'Accommodation_Type',
    'Transport_Mode',
    'Season',
    'Package_Type'
]

df[feature_cols] = df[feature_cols].fillna('Unknown')

numeric_features = ['Trip_Duration_Days', 'Approx_Cost (₹)']
categorical_features = [c for c in feature_cols if c not in numeric_features]

# Encoding and Scaling
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = ohe.fit_transform(df[categorical_features])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(categorical_features))

scaler = MinMaxScaler()
scaled_nums = scaler.fit_transform(df[numeric_features])
scaled_num_df = pd.DataFrame(scaled_nums, columns=numeric_features)

processed_features = pd.concat([encoded_cat_df, scaled_num_df], axis=1)

# ============================================
# STEP 3: Feature Weights
# ============================================
weights = {
    'Destination_Type': 0.4,
    'Trip_Duration_Days': 0.2,
    'Approx_Cost (₹)': 0.25,
    'Accommodation_Type': 0.05,
    'Transport_Mode': 0.05,
    'Season': 0.025,
    'Package_Type': 0.025
}

for col in ohe.get_feature_names_out():
    for key in weights:
        if col.startswith(key):
            processed_features[col] *= weights[key]
for num_col in numeric_features:
    processed_features[num_col] *= weights[num_col]

# ============================================
# STEP 4: Helper Functions
# ============================================
def get_destinations_by_city(from_city):
    return sorted(df[df['From_City'] == from_city]['Destination'].unique().tolist())

def get_destination_types(from_city, destination):
    return sorted(df[(df['From_City'] == from_city) & (df['Destination'] == destination)]
                  ['Destination_Type'].unique().tolist())

def get_reference_vector(destination_type, duration, approx_cost):
    temp = pd.DataFrame([['Unknown'] * len(categorical_features) + [0, 0]],
                        columns=categorical_features + numeric_features)
    temp.loc[0, 'Destination_Type'] = destination_type
    temp.loc[0, 'Trip_Duration_Days'] = duration
    temp.loc[0, 'Approx_Cost (₹)'] = approx_cost

    temp_encoded = ohe.transform(temp[categorical_features])
    temp_encoded_df = pd.DataFrame(temp_encoded, columns=ohe.get_feature_names_out(categorical_features))

    temp_scaled = scaler.transform(temp[numeric_features])
    temp_scaled_df = pd.DataFrame(temp_scaled, columns=numeric_features)

    temp_vector = pd.concat([temp_encoded_df, temp_scaled_df], axis=1)

    for col in ohe.get_feature_names_out():
        for key in weights:
            if col.startswith(key):
                temp_vector[col] *= weights[key]
    for num_col in numeric_features:
        temp_vector[num_col] *= weights[num_col]

    return temp_vector.values

def recommend_packages(from_city, destination, destination_type, duration, approx_cost, top_n=5):
    filtered_df = df[(df['From_City'] == from_city) &
                     (df['Destination'] == destination) &
                     (df['Destination_Type'] == destination_type)]
    if filtered_df.empty:
        return pd.DataFrame()

    filtered_features = processed_features.loc[filtered_df.index]
    ref_vector = get_reference_vector(destination_type, duration, approx_cost)

    sim_scores = cosine_similarity(ref_vector, filtered_features)[0]
    top_idx = np.argsort(sim_scores)[::-1][:top_n]

    top_packages = filtered_df.iloc[top_idx].copy()
    top_packages['Similarity_Score'] = sim_scores[top_idx]
    return top_packages[['From_City','Destination','Destination_Type','Package_Type',
                         'Trip_Duration_Days','Approx_Cost (₹)',
                         'Accommodation_Type','Transport_Mode','Season','Similarity_Score']]

# ============================================
# STEP 5: Streamlit UI
# ============================================
st.title("🌍 Travel Package Recommendation System")
st.write("Get best matching travel packages based on your preferences.")

# --- From City ---
from_city = st.selectbox("✈️ Select From City", sorted(df['From_City'].unique()))
if from_city:
    destinations = get_destinations_by_city(from_city)

    if destinations:
        destination = st.selectbox("🏝️ Select Destination", destinations)
        if destination:
            destination_types = get_destination_types(from_city, destination)

            if destination_types:
                destination_type = st.selectbox("🌄 Select Destination Type", destination_types)

                trip_duration = st.number_input("🗓️ Trip Duration (days)", min_value=1, value=5, step=1)
                approx_cost = st.number_input("💰 Approx Cost (₹)", min_value=1000, value=20000, step=500)

                if st.button("🔍 Recommend Packages"):
                    result = recommend_packages(from_city, destination, destination_type, trip_duration, approx_cost)
                    if not result.empty:
                        st.success(f"Top {len(result)} Recommended Packages:")
                        st.dataframe(result.reset_index(drop=True))
                    else:
                        st.warning("No packages found for the selected options.")
            else:
                st.warning("No destination types found for this destination.")
    else:
        st.warning("No destinations found for this city.")

