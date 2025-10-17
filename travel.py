# travel.py

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack

# ---------------------------------------------------------
# 1️⃣ Load and Prepare Data
# ---------------------------------------------------------
print("Loading travel data...")
df = pd.read_csv("travel_packages_120000.csv")   # ✅ updated filename
print(f"✅ Data loaded successfully! Shape: {df.shape}")
df.rename(columns={'Approx_Cost (₹)': 'Approx_Cost'}, inplace=True)

# Identify categorical and numerical columns
cat_cols = ['From_City', 'Destination', 'Destination_Type', 'Budget_Range', 
            'Accommodation_Type', 'Transport_Mode', 'Meal_Plan', 
            'Activity_Types', 'Season', 'Package_Type', 'Recommended_For']

num_cols = ['Trip_Duration_Days', 'Approx_Cost', 'Activity_Count']

# ---------------------------------------------------------
# 2️⃣ Preprocessing (Encoding + Scaling)
# ---------------------------------------------------------
print("Preprocessing data...")

ohe = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

encoded_cats = ohe.fit_transform(df[cat_cols])
scaled_nums = scaler.fit_transform(df[num_cols])

cdata = hstack([scaled_nums, encoded_cats])

print(f"✅ Preprocessing complete! Final feature shape: {cdata.shape}")

# ---------------------------------------------------------
# 3️⃣ Fit NearestNeighbors Model
# ---------------------------------------------------------
print("Training cosine similarity model...")
cosinemodel = NearestNeighbors(n_neighbors=5, metric='cosine')
cosinemodel.fit(cdata)
print("✅ Model trained successfully!")

# ---------------------------------------------------------
# 4️⃣ Function: User Input
# ---------------------------------------------------------
def user_input():
    """Collect user travel preferences from console input."""
    inpdata = pd.read_csv("travel_packages_120000.csv")  # ✅ updated filename

    print("\nAvailable From Cities:", inpdata['From_City'].unique())
    From_City = input("Select your From_City: ")

    print("\nAvailable Destinations:", inpdata['Destination'].unique())
    Destination = input("Select your Destination: ")

    print("\nDestination Types:", inpdata['Destination_Type'].unique())
    Destination_Type = input("Select your Destination_Type: ")

    Trip_Duration_Days = int(input(f"\nEnter Trip_Duration_Days (Range: {inpdata['Trip_Duration_Days'].min()} - {inpdata['Trip_Duration_Days'].max()}): "))

    print("\nBudget Ranges:", inpdata['Budget_Range'].unique())
    Budget_Range = input("Select your Budget_Range: ")

    Approx_Cost = float(input(f"\nEnter Approx_Cost (Range: {inpdata['Approx_Cost'].min()} - {inpdata['Approx_Cost'].max()}): "))

    print("\nAccommodation Types:", inpdata['Accommodation_Type'].unique())
    Accommodation_Type = input("Select your Accommodation_Type: ")

    print("\nTransport Modes:", inpdata['Transport_Mode'].unique())
    Transport_Mode = input("Select your Transport_Mode: ")

    print("\nMeal Plans:", inpdata['Meal_Plan'].unique())
    Meal_Plan = input("Select your Meal_Plan: ")

    Activity_Count = int(input(f"\nEnter Activity_Count (Range: {inpdata['Activity_Count'].min()} - {inpdata['Activity_Count'].max()}): "))

    print("\nActivity Types:", inpdata['Activity_Types'].unique())
    Activity_Types = input("Select Activity_Types: ")

    print("\nSeasons:", inpdata['Season'].unique())
    Season = input("Select Season: ")

    print("\nPackage Types:", inpdata['Package_Type'].unique())
    Package_Type = input("Select Package_Type: ")

    print("\nRecommended For:", inpdata['Recommended_For'].unique())
    Recommended_For = input("Select Recommended_For: ")

    # Create dataframe for user input
    row = pd.DataFrame([[From_City, Destination, Destination_Type, Trip_Duration_Days, Budget_Range,
                         Approx_Cost, Accommodation_Type, Transport_Mode, Meal_Plan, Activity_Count,
                         Activity_Types, Season, Package_Type, Recommended_For]],
                       columns=inpdata.columns)

    print("\nGiven Item Input Data:")
    print(row.to_string(index=False))
    print()

    return row

# ---------------------------------------------------------
# 5️⃣ Get User Input and Find Recommendations
# ---------------------------------------------------------
if __name__ == "__main__":
    user_df = user_input()

    # Transform categorical input
    user_cat = ohe.transform(user_df[cat_cols])
    user_num = scaler.transform(user_df[num_cols])
    user_vector = np.hstack([user_num, user_cat.toarray()])

    # Find nearest neighbors
    distances, indices = cosinemodel.kneighbors(user_vector)

    # Get top recommended packages
    top_packages = df.iloc[indices[0]].copy()
    top_packages['Similarity_Score'] = 1 - distances.flatten()

    # Display recommendations
    top_packages_display = top_packages[['From_City', 'Destination', 'Destination_Type',
                                         'Trip_Duration_Days', 'Budget_Range', 'Approx_Cost',
                                         'Accommodation_Type', 'Transport_Mode', 'Activity_Count',
                                         'Package_Type', 'Similarity_Score']]

    print("\n🎯 Top Recommended Travel Packages:\n")
    print(top_packages_display.to_string(index=False))
