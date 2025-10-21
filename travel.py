# ---------------------------------------------------------
# 🧭 Travel Recommendation System using Cosine Similarity
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack

# -------------------------------
# 1️⃣ Load & Prepare Data
# -------------------------------
df = pd.read_csv("travel_packages_120000.csv")

cat_cols = ['From_City', 'Destination', 'Budget_Range',
            'Accommodation_Type', 'Transport_Mode', 'Meal_Plan',
            'Activity_Types', 'Season', 'Package_Type', 'Recommended_For']

num_cols = ['Trip_Duration_Days', 'Approx_Cost', 'Activity_Count']

# Drop missing values
df = df.dropna(subset=cat_cols + num_cols + ['Destination_Type']).reset_index(drop=True)

# -------------------------------
# 2️⃣ Encode Features
# -------------------------------
encoder = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

X_cat = encoder.fit_transform(df[cat_cols])
X_num = scaler.fit_transform(df[num_cols])

# Combine categorical + numerical
X_final = hstack([X_cat, X_num])

# -------------------------------
# 3️⃣ Fit Nearest Neighbors (cosine)
# -------------------------------
knn = NearestNeighbors(n_neighbors=6, metric='cosine')  # cosine similarity
knn.fit(X_final)

# -------------------------------
# 4️⃣ Recommendation Function
# -------------------------------
def recommend_similar_trips(from_city, destination, budget_range, accommodation_type,
                            transport_mode, meal_plan, activity_types, season,
                            package_type, recommended_for, trip_duration_days,
                            approx_cost, activity_count):
    
    # Build user input DataFrame
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
        'Approx_Cost': approx_cost,
        'Activity_Count': activity_count
    }])
    
    # Transform input
    input_cat = encoder.transform(input_df[cat_cols])
    input_num = scaler.transform(input_df[num_cols])
    input_final = hstack([input_cat, input_num])
    
    # Find nearest neighbors
    distances, indices = knn.kneighbors(input_final)
    
    # Retrieve similar trips
    similar = df.iloc[indices[0]].copy()
    similar['Similarity'] = 1 - distances[0]
    
    return similar[['From_City', 'Destination', 'Destination_Type', 'Approx_Cost', 'Similarity']]

# -------------------------------
# 5️⃣ Example Usage
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
    approx_cost=40000,
    activity_count=6
)

print("🔹 Recommended Similar Trips:")
print(sample_result)


