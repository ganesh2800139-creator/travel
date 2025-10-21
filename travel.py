# ---------------------------------------------------------
# 🧭 Travel Recommendation System using Cosine Similarity
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
import os

# -------------------------------
# 1️⃣ Load & Prepare Data Safely
# -------------------------------
file_path = "travel_packages_120000.csv"

if not os.path.exists(file_path):
    print(f"⚠️ Error: File '{file_path}' does not exist. Please upload the CSV.")
    df = pd.DataFrame()
elif os.path.getsize(file_path) == 0:
    print(f"⚠️ Error: File '{file_path}' is empty. Please provide a CSV with data.")
    df = pd.DataFrame()
else:
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns.")
    except pd.errors.EmptyDataError:
        print(f"⚠️ Error: CSV file is empty or invalid.")
        df = pd.DataFrame()

# If df is empty, skip further processing
if df.empty:
    print("❌ No data to process. Exiting script.")
else:
    # -------------------------------
    # 2️⃣ Define Columns
    # -------------------------------
    cat_cols = ['From_City', 'Destination', 'Budget_Range',
                'Accommodation_Type', 'Transport_Mode', 'Meal_Plan',
                'Activity_Types', 'Season', 'Package_Type', 'Recommended_For']

    num_cols = ['Trip_Duration_Days', 'Approx_Cost (₹)', 'Activity_Count']

    # -------------------------------
    # 3️⃣ Ensure columns exist before dropping NaN
    # -------------------------------
    existing_cols = [col for col in cat_cols + num_cols + ['Destination_Type'] if col in df.columns]
    missing_cols = list(set(cat_cols + num_cols + ['Destination_Type']) - set(existing_cols))
    if missing_cols:
        print(f"⚠️ Warning: Missing columns in CSV and will be ignored: {missing_cols}")

    df = df.dropna(subset=existing_cols).reset_index(drop=True)

    # -------------------------------
    # 4️⃣ Encode Features
    # -------------------------------
    encoder = OneHotEncoder(handle_unknown='ignore')
    scaler = StandardScaler()

    X_cat = encoder.fit_transform(df[[col for col in cat_cols if col in df.columns]])
    X_num = scaler.fit_transform(df[[col for col in num_cols if col in df.columns]])

    X_final = hstack([X_cat, X_num])

    # -------------------------------
    # 5️⃣ Fit Nearest Neighbors (cosine)
    # -------------------------------
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(X_final)

    # -------------------------------
    # 6️⃣ Recommendation Function
    # -------------------------------
    def recommend_similar_trips(from_city, destination, budget_range, accommodation_type,
                                transport_mode, meal_plan, activity_types, season,
                                package_type, recommended_for, trip_duration_days,
                                approx_cost, activity_count):

        # Map Python variable approx_cost to CSV column name
        input_df = pd.DataFrame([{
            'From_City': from_city,
            'Destination': destination,
            'Budget_Range': budget_range,
            'Accommodation_Type': accommodation_type,
            'Transport_Mode': transport_mode,
            'Meal_Plan': meal_plan,
            'Activity_Types': activity_types,
            'Season': season,
            'Package_Type': package_type,
            'Recommended_For': recommended_for,
            'Trip_Duration_Days': trip_duration_days,
            'Approx_Cost (₹)': approx_cost,
            'Activity_Count': activity_count
        }])

        input_cat_cols = [col for col in cat_cols if col in df.columns]
        input_num_cols = [col for col in num_cols if col in df.columns]

        input_cat = encoder.transform(input_df[input_cat_cols])
        input_num = scaler.transform(input_df[input_num_cols])
        input_final = hstack([input_cat, input_num])

        distances, indices = knn.kneighbors(input_final)
        similar = df.iloc[indices[0]].copy()
        similar['Similarity'] = 1 - distances[0]

        output_cols = ['From_City', 'Destination', 'Destination_Type', 'Approx_Cost (₹)', 'Similarity']
        output_cols = [col for col in output_cols if col in similar.columns]

        return similar[output_cols]

    # -------------------------------
    # 7️⃣ Example Usage
    # -------------------------------
    sample_result = recommend_similar_trips(
        from_city='Mumbai',
        destination='Goa',
        budget_range='Medium',
        accommodation_type='Resort',
        transport_mode='Flight',
        meal_plan='Breakfast',
        activity_types='Beach, Adventure',
        season='Winter',
        package_type='Leisure',
        recommended_for='Couples',
        trip_duration_days=5,
        approx_cost=40000,   # Python variable
        activity_count=6
    )

    print("🔹 Recommended Similar Trips:")
    print(sample_result)

