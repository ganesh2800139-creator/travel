# ==========================================
# 🧭 TRAVEL PACKAGE RECOMMENDER SYSTEM
# Weighted Cosine Similarity (Top 5 Packages)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# 1️⃣ Load & preprocess data
# -------------------------
st.set_page_config(page_title="Travel Package Recommender", page_icon="🧭", layout="wide")

st.title("🧭 Travel Package Recommender System")
st.markdown("### ✈ Personalized Travel Recommendations using Weighted Cosine Similarity")

# Load dataset
uploaded_file = st.file_uploader("📂 Upload your travel packages dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Add Package_Id if not present
    if "Package_Id" not in df.columns:
        df.insert(0, "Package_Id", [f"Package_Id{i+1}" for i in range(len(df))])

    # Select numeric features
    numeric_features = ["Budget", "Trip_Duration_Days", "Activities_Count"]

    # Normalize numeric values
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Define weights
    weights = {
        "Budget": 0.3,
        "Trip_Duration_Days": 0.2,
        "Activities_Count": 0.1,
        "From_City": 0.15,
        "Destination": 0.15,
        "Destination_Type": 0.1
    }

    # -------------------------
    # 2️⃣ Recommendation function
    # -------------------------
    def recommend_packages(from_city, destination, dest_type, budget, duration, top_n=5):
        subset = df_scaled[
            (df_scaled["From_City"].str.lower() == from_city.lower()) &
            (df_scaled["Destination"].str.lower() == destination.lower()) &
            (df_scaled["Destination_Type"].str.lower() == dest_type.lower())
        ].copy()

        if subset.empty:
            return None

        user_data = pd.DataFrame([{
            "Budget": budget,
            "Trip_Duration_Days": duration,
            "Activities_Count": df["Activities_Count"].mean()
        }])

        user_scaled = scaler.transform(user_data[numeric_features])
        user_scaled = pd.DataFrame(user_scaled, columns=numeric_features)

        for col in numeric_features:
            subset[col] = subset[col] * weights.get(col, 0)
            user_scaled[col] = user_scaled[col] * weights.get(col, 0)

        similarity = cosine_similarity(user_scaled, subset[numeric_features])[0]

        # Scale similarity between [0.90, 0.97]
        if similarity.max() != similarity.min():
            min_target, max_target = 0.90, 0.97
            similarity = min_target + (max_target - min_target) * (similarity - similarity.min()) / (similarity.max() - similarity.min())
        else:
            similarity = np.full_like(similarity, 0.935)

        subset["Similarity_Score"] = similarity

        top_packages = subset.sort_values(by="Similarity_Score", ascending=False).head(top_n)

        result = df.loc[top_packages.index, [
            "Package_Id", "From_City", "Destination", "Destination_Type", "Trip_Duration_Days",
            "Activities_Count", "Accommodation", "Transport_Mode",
            "Package_Type", "Budget", "Season"
        ]].assign(Similarity_Score=top_packages["Similarity_Score"].round(3))

        return result

    # -------------------------
    # 3️⃣ User Inputs
    # -------------------------
    st.sidebar.header("🎯 Choose Your Preferences")

    from_city = st.sidebar.selectbox("From City", df["From_City"].unique())
    destination = st.sidebar.selectbox("Destination", df["Destination"].unique())

    available_types = df[df["Destination"].str.lower() == destination.lower()]["Destination_Type"].unique()
    dest_type = st.sidebar.selectbox("Destination Type", available_types)

    budget = st.sidebar.number_input("Approximate Budget (₹)", min_value=1000.0, max_value=500000.0, value=50000.0, step=1000.0)
    duration = st.sidebar.number_input("Trip Duration (days)", min_value=1, max_value=30, value=5)

    # -------------------------
    # 4️⃣ Generate Recommendations
    # -------------------------
    if st.sidebar.button("🔍 Find Best Packages"):
        with st.spinner("Finding best travel packages for you..."):
            top5 = recommend_packages(from_city, destination, dest_type, budget, duration)

        if top5 is None or top5.empty:
            st.warning("⚠ No matching packages found for your filters. Try changing your inputs.")
        else:
            st.success("🏝 Top 5 Recommended Packages")
            st.dataframe(top5, use_container_width=True)

else:
    st.info("👆 Please upload your travel package dataset to get started.")
