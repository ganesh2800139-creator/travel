# ================================================
# 🧭 Travel Package Recommendation System
# ================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack

# ================================================
# 🪄 Load Dataset
# ================================================
df = pd.read_csv("travel_packages_120000.csv")

# Define Categorical and Numerical Columns
cat_cols = ['From_City', 'Destination', 'Destination_Type', 'Budget_Range',
            'Accommodation_Type', 'Transport_Mode', 'Meal_Plan', 'Activity_Types',
            'Season', 'Package_Type', 'Recommended_For']

num_cols = ['Trip_Duration_Days', 'Approx_Cost', 'Activity_Count']

# ================================================
# ✨ One-Hot Encode Categorical Features
# ================================================
ohe = OneHotEncoder(handle_unknown='ignore')
cat_features = ohe.fit_transform(df[cat_cols])

# ================================================
# 📏 Scale Numerical Features
# ================================================
scaler = MinMaxScaler()
num_features = scaler.fit_transform(df[num_cols])

# ================================================
# 🧮 Combine Encoded + Scaled Features
# ================================================
cdata = hstack([num_features, cat_features]).tocsr()

# ================================================
# 🤝 Fit Nearest Neighbors Model
# ================================================
cosinemodel = NearestNeighbors(n_neighbors=5, metric='cosine')
cosinemodel.fit(cdata)

# ================================================
# 🧑 User Input Function
# ================================================
def user_input():
    print("🧭 --- Travel Package Selection --- 🧭")
    print()

    # From City
    print(df['From_City'].unique())
    From_City = input("Select your From_City: ")
    print()

    # Destination
    print(df['Destination'].unique())
    Destination = input("Select your Destination: ")
    print()

    # Destination Type
    print(df['Destination_Type'].unique())
    Destination_Type = input("Select your Destination_Type: ")
    print()

    # Trip Duration
    Trip_Duration_Days = int(input(f"Enter Trip_Duration_Days (Range: {df['Trip_Duration_Days'].min()} to {df['Trip_Duration_Days'].max()}): "))
    print()

    # Budget Range
    print(df['Budget_Range'].unique())
    Budget_Range = input("Select your Budget_Range: ")
    print()

    # Approx Cost
    Approx_Cost = float(input(f"Enter your Approx_Cost (Range: {df['Approx_Cost'].min()} to {df['Approx_Cost'].max()}): "))
    print()

    # Accommodation Type
    print(df['Accommodation_Type'].unique())
    Accommodation_Type = input("Select your Accommodation_Type: ")
    print()

    # Transport Mode
    print(df['Transport_Mode'].unique())
    Transport_Mode = input("Select your Transport_Mode: ")
    print()

    # Meal Plan
    print(df['Meal_Plan'].unique())
    Meal_Plan = input("Select your Meal_Plan: ")
    print()

    # Activity Count
    Activity_Count = int(input(f"Enter your Activity_Count (Range: {df['Activity_Count'].min()} to {df['Activity_Count'].max()}): "))
    print()

    # Activity Types
    print(df['Activity_Types'].unique())
    Activity_Types = input("Select Activity_Types: ")
    print()

    # Season
    print(df['Season'].unique())
    Season = input("Select Season: ")
    print()

    # Package Type
    print(df['Package_Type'].unique())
    Package_Type = input("Select Package_Type: ")
    print()

    # Recommended For
    print(df['Recommended_For'].unique())
    Recommended_For = input("Select Recommended_For: ")
    print()

    # Build DataFrame for User Input
    row = pd.DataFrame([[From_City, Destination, Destination_Type, Trip_Duration_Days, Budget_Range,
                         Approx_Cost, Accommodation_Type, Transport_Mode, Meal_Plan, Activity_Count,
                         Activity_Types, Season, Package_Type, Recommended_For]],
                       columns=df.columns)

    print("✅ Given Input Data:")
    print(row)
    print()

    return row

# ================================================
# 🧭 Get User Input
# ================================================
user_df = user_input()

# ================================================
# 🧠 Transform User Input (Encoding + Scaling)
# ================================================
user_cat = ohe.transform(user_df[cat_cols])
user_num = scaler.transform(user_df[num_cols])
user_vector = hstack([user_num, user_cat]).tocsr()

# ================================================
# 🕵️‍♂️ Find Nearest Packages
# ================================================
distances, indices = cosinemodel.kneighbors(user_vector)

# ================================================
# 🏆 Top Recommendations
# ================================================
top_packages = df.iloc[indices[0]].copy()
top_packages['Similarity_Score'] = 1 - distances.flatten()

top_packages_display = top_packages[['From_City', 'Destination', 'Destination_Type', 'Trip_Duration_Days',
                                     'Budget_Range', 'Approx_Cost', 'Accommodation_Type', 'Transport_Mode',
                                     'Activity_Count', 'Package_Type', 'Similarity_Score']]

print("🏖️ Top Recommended Travel Packages 🏖️\n")
print(top_packages_display.to_string(index=False))




