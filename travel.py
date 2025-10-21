# app.py

import streamlit as st
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors

# Load artifacts
with open("artifacts/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)
with open("artifacts/nn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load data
df = pd.read_csv("travel_packages_120000.csv")

# ID handling
id_col = 'Package_ID'
feature_cols = ['Activity_Count', 'Approx_Cost (₹)', 'Trip_Duration_Days',
                'Accommodation_Type', 'Activity_Types', 'Budget_Range',
                'Destination_Type', 'Meal_Plan', 'Package_Type',
                'Recommended_For', 'Season', 'Transport_Mode']

# Mappings
id_to_index = pd.Series(df.index.values, index=df[id_col]).to_dict()
index_to_id = pd.Series(df[id_col].values, index=df.index).to_dict()

# UI
st.title("🌍 Travel Package Recommendation System")

st.sidebar.header("🔍 Filter Your Preferences")

# Input form
user_input = {}
for col in feature_cols:
    if df[col].dtype == 'object':
        user_input[col] = st.sidebar.selectbox(col, sorted(df[col].dropna().unique()))
    else:
        min_val = int(df[col].min())
        max_val = int(df[col].max())
        user_input[col] = st.sidebar.slider(col, min_val, max_val, int(df[col].median()))

destination_filter = st.sidebar.selectbox("Filter by Destination (Optional)", ["-- None --"] + sorted(df["Destination"].unique()))
dest_type_filter = st.sidebar.selectbox("Filter by Destination_Type (Optional)", ["-- None --"] + sorted(df["Destination_Type"].unique()))

if st.sidebar.button("Get Recommendations"):
    input_df = pd.DataFrame([user_input])
    X_query = preprocessor.transform(input_df)
    distances, indices = model.kneighbors(X_query, n_neighbors=10)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        package = df.iloc[idx].copy()
        package['Similarity_Score'] = round(1 - dist, 6)
        results.append(package)

    rec_df = pd.DataFrame(results)

    # Apply optional filters
    if destination_filter != "-- None --":
        rec_df = rec_df[rec_df["Destination"] == destination_filter]
    if dest_type_filter != "-- None --":
        rec_df = rec_df[rec_df["Destination_Type"] == dest_type_filter]

    if not rec_df.empty:
        st.success(f"✅ Showing top {len(rec_df)} recommendations")
        st.dataframe(rec_df[[id_col, "Destination", "Destination_Type", "Approx_Cost (₹)", "Trip_Duration_Days", "Accommodation_Type", "Similarity_Score"]])
    else:
        st.warning("⚠️ No matching packages found. Try relaxing the filters.")
