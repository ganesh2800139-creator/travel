🧭 Travel Package Recommendation System 📌 Overview

The Travel Package Recommendation System is an intelligent web application built using Streamlit and Machine Learning (KNN - Cosine Similarity) to recommend personalized travel packages based on user preferences such as departure city, destination, trip duration, cost, and travel type.

The app helps users explore similar travel packages from a large dataset by finding the most relevant trips that match their interests and budget.

🚀 Features

✅ Interactive UI – Built using Streamlit for a clean and easy-to-use interface. ✅ Smart Recommendations – Uses K-Nearest Neighbors (KNN) with cosine similarity to suggest top matching travel packages. ✅ Data Preprocessing – Encodes categorical data using OneHotEncoder and normalizes numerical values with MinMaxScaler. ✅ Dynamic Filtering – Destination and type options update automatically based on the user’s selected city. ✅ Scalable Design – Works seamlessly with large datasets (e.g., 200k+ records).

🏗️ Tech Stack Component Description Language Python Frontend Streamlit ML Algorithm K-Nearest Neighbors (Cosine Similarity) Libraries Used pandas, numpy, scikit-learn, streamlit Dataset travel_packages_200k.csv (custom dataset with travel packages) 📊 Dataset Structure

Ensure your dataset file is named travel_packages_200k.csv and contains the following columns:

Column Description Package_ID Unique package identifier From_City Departure city Destination Travel destination Destination_Type Type of destination (e.g., Beach, Historical, Adventure) Trip_Duration_Days Number of days for the trip Budget_Range Price category Approx_Cost (₹) Approximate cost in INR Accommodation_Type Hotel / Resort / Homestay, etc. Transport_Mode Bus / Flight / Train, etc. Meal_Plan Included meals Activity_Count Number of activities included Activity_Types Adventure / Cultural / Relaxation, etc. Season Best time to visit Package_Type Family / Solo / Couple / Group Recommended_For Ideal traveler category

🧠 How It Works

User Inputs preferences such as:

Departure City

Destination

Destination Type

Trip Duration

Approx Cost

Feature Engineering:

Categorical data → OneHotEncoded

Numerical data → MinMax Scaled

Model Training:

K-Nearest Neighbors (NearestNeighbors) model fits all travel packages.

Cosine similarity metric identifies top 5 closest matches.

Recommendations Display:

Top 5 most similar packages are shown with similarity scores.

🧩 Example Screenshot

![App Screenshot](https://i.postimg.cc/9XgZFcXK/Screenshot-2025-10-24-152922.png)

🛠️ Future Enhancements

Integrate Google Maps API for visual trip locations.

Add user reviews and ratings.

Implement Deep Learning-based recommendation engine.

Add filter options (season, activities, budget range).

🤝 Contributing

Contributions are welcome! To contribute:

Fork this repository

Create a new branch (feature/your-feature-name)

Commit your changes

Push to your fork

Create a Pull Request

📄 License

This project is licensed under the MIT License – feel free to use and modify it.

👨‍💻 Author

 Ganesh Sura 
 [email:ganesh2800139@gmail.com] 
 [ GitHub:https://github.com/ganesh2800139-creator]

