🧭 Travel Package Recommendation System
📌 Overview

The Travel Package Recommendation System is an intelligent web application built using Streamlit and Machine Learning (KNN with Cosine Similarity).
It recommends personalized travel packages based on user preferences such as departure city, destination, destination type, trip duration, and budget.

The app analyzes a large dataset of real-world travel packages and finds the most relevant and similar trips that match the user’s interests, offering an intuitive way to explore and plan travel experiences.

🚀 Features

✅ Interactive UI – Built with Streamlit for a simple, user-friendly interface.
✅ Smart Recommendations – Uses K-Nearest Neighbors (KNN) with cosine similarity to find the most relevant travel packages.
✅ Data Preprocessing – Utilizes OneHotEncoder for categorical encoding and MinMaxScaler for numerical normalization.
✅ Dynamic Filtering – Destination and type dropdowns update automatically based on the selected departure city.
✅ Similarity Score Display – Shows how close each recommendation is to the user’s preferences.
✅ Scalable Design – Efficiently handles large datasets without performance issues.

🏗️ Tech Stack
Component	Description
Language	Python
Frontend	Streamlit
ML Algorithm	K-Nearest Neighbors (Cosine Similarity)
Libraries Used	pandas, numpy, scikit-learn, streamlit
Dataset	realistic_final_travel_packages_dataset_v4.csv (custom dataset with detailed travel packages)
📊 Dataset Structure

Ensure your dataset (e.g. realistic_final_travel_packages_dataset_v4.csv) contains the following columns:

Column	Description
Package_ID	Unique travel package identifier
From_City	Departure city
Destination	Travel destination
Destination_Type	Type of destination (e.g., Beach, Hill Station, Cultural, etc.)
Trip_Duration_Days	Number of days for the trip
Budget	Approximate total cost in INR
Accommodation	Hotel / Resort / Homestay, etc.
Transport_Mode	Bus / Flight / Train, etc.
Activities_Count	Number of activities included
Season	Ideal time to visit
Package_Type	Family / Solo / Couple / Group
Similarity	Computed similarity score (added by model)

🧠 How It Works
1️⃣ User Inputs Preferences:

Departure City

Destination

Destination Type

Trip Duration

Budget

2️⃣ Feature Engineering:

Categorical data → Encoded using OneHotEncoder

Numerical data → Normalized using MinMaxScaler

3️⃣ Model Training:

K-Nearest Neighbors (KNN) model trained using all travel packages.

Uses Cosine Similarity metric to identify top 5 most similar packages.

4️⃣ Recommendations Display:

Displays the top 5 recommended trips along with similarity scores in a Streamlit DataFrame.

Each recommendation includes details like destination, duration, transport, and package type.

🧩 Example Screenshot

🛠️ Future Enhancements

🚗 Integrate Google Maps API to show trip locations visually.
⭐ Add user reviews and ratings for each package.
🧮 Implement Deep Learning–based recommendations (e.g., neural collaborative filtering).
🎯 Introduce advanced filters — by season, activity type, or budget range.
📊 Add data visualization dashboards for user insights.

🤝 Contributing

Contributions are welcome! 🙌
To contribute:

Fork this repository

Create a new branch — feature/your-feature-name

Commit your changes

Push to your fork

Open a Pull Request

📄 License

This project is licensed under the MIT License —
You are free to use, modify, and distribute it with attribution.

👨‍💻 Author

 Ganesh Sura 
 [email:ganesh2800139@gmail.com] 
 [ GitHub:https://github.com/ganesh2800139-creator]

