#!/usr/bin/env python
# coding: utf-8

# **Title:** Travel Cost & Package Recommendation Systems
# 
# **Goal:** 
# - We are going to recommend the top 5 Packages according to the User input.

# **Table of Contents**<a id = ToC>
# - 0. [Data Collection](#DC)
# - 1. [Data Validation](#DV)
#        
# - 2. [Exploratory Data Analysis](#EDA)
# - 3. [Missing Values & Outliers Handling](#MVOH)
#     
# - 4. [Predictive Modeling](#PM)

# **0. Data Collection**<a id = DC>
#     
# [Back to ToC](#ToC)
#  
# * For this project , We had taken open source Dataset
# 
#     **DatasetRef:** Open Sources

# **Data Info:**
# - We are having 1,20,000 Packages with 15 factors
# 
# Column|Description
# -----------|--------------
# Package_ID|Unique identifier for each travel package.
# From_City|Starting city of the traveler (source location).
# Destination|Final destination or travel location.
# Destination_Type|Type of place (e.g., Beach, Hill Station, Historical, Adventure, Wildlife).
# Trip_Duration_Days|Total duration of the trip in days.
# Budget_Range|Overall budget category — Low, Medium, or High.
# Approx_Cost (₹)|Estimated total cost of the travel package in Indian Rupees.
# Accommodation_Type|Type of stay offered — Hotel, Resort, Hostel, Homestay, Villa, etc.
# Transport_Mode|Mode of travel between source and destination — Flight, Train, Bus, Car.
# Meal_Plan|Type of meal inclusion — Breakfast Only, Half Board, Full Board, All Inclusive.
# Activity_Count|Number of activities or experiences included in the package.
# Activity_Types|Types of activities — Sightseeing, Trekking, Water Sports, Safari, etc.
# Season|Suitable season or time of the year to visit the destination.
# Package_Type|Category of package — Budget, Standard, Premium, Luxury, Deluxe. (Target column)
# Recommended_For|Ideal traveler type — Couples, Family, Friends, Solo, Corporate Group, etc.

# In[1]:


# Basic Libraries
import pandas as pd
import numpy as np

import warnings, time, json
warnings.filterwarnings("ignore")


# - Reading the CSV File

# In[2]:


data = pd.read_csv("travel_packages_120000.csv")


# In[3]:


#Changing column name 
data.rename(columns={'Approx_Cost (₹)': 'Approx_Cost'}, inplace=True)


# **Basic Checks**

# In[4]:


data.head()


# In[5]:


data.info()


# **2. Data Validation**<a id = DV>
#     
# [Back to ToC](#ToC)
# - Verification of each and every column Data and its Data Type.
# - Removing any Abnormalities and Special Characters in the Data for smooth process of Analysis.
# - Removing the duplicated columns.

# In[6]:


#Taking a copy of original data, so that original data will be safe.
df = data.copy()


# In[7]:


df.columns


# In[8]:


#Writing a Python function with df methods to validate columns
def colvalidate(df, col):
    print(f"Column: {col}")
    print()
    print(f"Number of Unique Values in Column: {df[col].nunique()}")
    print()
    print("Unique Values:")
    if df[col].nunique()>=100:
        for indx in range(0, df[col].nunique(), 100):
            print(df[col].unique()[indx:indx+100])
            print()
    else:
        print(df[col].unique())
        print()
    print("Data Type of Column:", df[col].dtype)
    print()


# **Validation Starts now**
# - We will start Data Validation for each column in order.

# **1.Package_ID**

# - Because of all unique values, no need to perform validation.

# **2.From_City**

# In[9]:


colvalidate(df, 'From_City')


# - Data and Data Type of this column is valid.

# **3.Destination**

# In[10]:


colvalidate(df, 'Destination')


# - Data and Data Type of this column is valid.

# **4.Destination_Type**

# In[11]:


colvalidate(df, 'Destination_Type')


# - Data and Data Type of this column is valid.

# **5.Trip_Duration_Days**

# In[12]:


colvalidate(df, 'Trip_Duration_Days')


# - Data and Data Type of this column is valid.

# **6.Budget_Range**

# In[13]:


colvalidate(df, 'Budget_Range')


# - Data and Data Type of this column is valid.

# **7.Approx_Cost (₹)**

# In[14]:


colvalidate(df, 'Approx_Cost')


# - Data and Data Type of this column is valid.

# **8.Accommodation_Type**

# In[15]:


colvalidate(df, 'Accommodation_Type')


# - Data and Data Type of this column is valid.

# **9.Transport_Mode**

# In[16]:


colvalidate(df, 'Transport_Mode')


# - Data and Data Type of this column is valid.

# **10.Meal_Plan**

# In[17]:


colvalidate(df, 'Meal_Plan')


# - Data and Data Type of this column is valid.

# **11.Activity_Count**

# In[18]:


colvalidate(df, 'Activity_Count')


# - Data and Data Type of this column is valid.

# **12.Activity_Types**

# In[19]:


colvalidate(df, 'Activity_Types')


# - Data and Data Type of this column is valid.

# **13.Season**

# In[20]:


colvalidate(df, 'Season')


# - Data and Data Type of this column is valid.

# **14.Package_Type**

# In[21]:


colvalidate(df, 'Package_Type')


# - Data and Data Type of this column is valid.

# **15.Recommended_For**

# In[22]:


colvalidate(df, 'Recommended_For')


# - Data and Data Type of this column is valid.

# **Duplicated Rows checking**
# - To check the duplicated rows, first we need to convert all categorical columns in lower case letters.

# In[23]:


for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.lower()


# In[24]:


#Checking for duplicates
df[df.duplicated()]


# - No duplicates found.

# **3.EDA(Exploratory Data Analysis)**<a id = EDA>
#     
# [Back to ToC](#ToC)

# **To get Insights on this Data**
# - We can use EDA Techniques
#   - Uni-Variate Analysis(Study of Single column data)
#   - Bi/Multi-Variate Analysis(Data Study between Two/More columns)
# - Above techniques will use **Descriptive Statistics** & **Visualizations.**
# - **Descriptive Statistics**
#   - Stats measures are used to understand column data.
#   - To start with Descriptive Stats, first we need to understand the Types of Variables/Columns in given data.
# - **Visualizations**
#   - Pictorial Representation of the Data.

# **Descriptive Statistics/Visualizations**

# **Types of Variables/Columns**

# Variable/Column | Type
# ----------|-----------
# Package_ID | CO
# From_City | CN
# Destination | CN
# Destination_Type | CN
# Trip_Duration_Days | ND
# Budget_Range | CO
# Approx_Cost (₹) | ND
# Accommodation_Type | CO
# Transport_Mode | CO
# Meal_Plan | CO
# Activity_Count | ND 
# Activity_Types | CN
# Season | CN
# Package_Type | CO
# Recommended_For | CN
# 
# **Note:**
# - **CO** - Categorical Ordinal
# - **CN** - Categorical Nominal
# - **ND** - Numerical Discrete
# - **NC** - Numerical Continuous

# **Uni-Variate Analysis** - Study of Single Column Data
# 

# - Descriptive Stats Univariate measures,
#   - Numeric
#     - Discrete
#       - round(Mean), round(Median), Mode, Five Number Summary, Std, Skewness, Kurtosis
#     - Continuous
#       - Mean, Median, Five Number Summary, Std, Skewness, Kurtosis
#   - Categorical & Boolean
#     - nunique
#     - unique
#     - FDT
#     - Mode
#   - Date Time
#     - Start Date, End Date, Diff between Start & End Date
# - Visualizations for Uni-Variate
#   - Categorical: Comparission: Pie/Bar
#   - Numerical: Distribution: Hist/Box/Density

# In[25]:


# Modules for Visualizations
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import pearsonr
from scipy.stats import f_oneway


# **Taking Uni-Variate Descriptive Stats User defined Functions**

# In[26]:


# Writing functions according to col type

# To Supress Scientific Notation of Values
pd.set_option('display.float_format', lambda x: '%.2f' % x) # for 2 float points

from simple_colors import * # for color print -> pip install simple_colors

############################ Numeric Continuous ############################
def ncstudy(df, col):
    print(green("#######################################################",['bold']))
    print(green("Taken Numeric Continuous Column:",['bold']), black(col,['bold']))
    print(green("#######################################################",['bold']))
    print()
    print(cyan("Descriptive Stats:",['bold']))
    print()
    print(blue("******** Measures of Central Tendancy ************", ['bold']))
    print(magenta("Mean:",['bold']), round(df[col].mean(),2))
    print(magenta("Median:",['bold']), df[col].median())
    print(magenta("Mode:",['bold']), df[col].mode()[0]) # Taking first value
    print()
    print(blue("******** Measures of Dispersion ************",['bold']))
    print(magenta("Range:",['bold']), df[col].max()-df[col].min())
    print(magenta("Variance:",['bold']), round(df[col].var(),2))
    print(magenta("Standard Deviation:",['bold']), round(df[col].std(),2))
    print(magenta("Five Number Summary:",['bold']))
    print(round(data[col].describe(),2)[['min','25%','50%','75%','max']])
    print()
    print(blue("******** Measures of Symmetry ************",['bold']))
    print(magenta("Skewness:",['bold']), round(df[col].skew(),2))
    print(magenta("Kurtosis:",['bold']), round(df[col].kurt(),2))
    print()
    print(cyan("Visualization:",['bold']))
    print()
    px.box(df[col], orientation='h', width=650, height=300).show()
    print()

############################## Numeric Discrete #################################
def ndstudy(df, col):
    print(green("#######################################################",['bold']))
    print(green("Taken Numeric Discrete Column:",['bold']), black(col,['bold']))
    print(green("#######################################################",['bold']))
    print()
    print(cyan("Uni-Variate Descriptive Stats:",['bold']))
    print()
    print("******** Measures of Central Tendancy ************")
    print(magenta("Mean:",['bold']), round(df[col].mean()))
    print(magenta("Median:",['bold']), round(df[col].median()))
    print(magenta("Mode:",['bold']), df[col].mode()[0]) # Taking first value
    print()
    print("******** Measures of Dispersion ************")
    print(magenta("Range:",['bold']), df[col].max()-df[col].min())
    print(magenta("Variance:",['bold']), round(df[col].var()))
    print(magenta("Standard Deviation:",['bold']), round(df[col].std()))
    print(magenta("Five Number Summary:",['bold']))
    print(round(data[col].describe())[['min','25%','50%','75%','max']])
    print()
    print("******** Measures of Symmetry ************")
    print(magenta("Skewness:",['bold']), round(df[col].skew(),2))
    print(magenta("Kurtosis:",['bold']), round(df[col].kurt(),2))
    print()
    print(cyan("Visualization:",['bold']))
    print()
    px.box(df[col], orientation='h', width=650, height=300).show()
    print()

############################# Categorical #######################################
def catstudy(df, col):
    print(green("#######################################################",['bold']))
    print(green("Taken Categorical Column:",['bold']), black(col,['bold']))
    print(green("#######################################################",['bold']))
    print()
    print(cyan("Uni-Variate Descriptive Stats:",['bold']))
    print()
    print(magenta("Number of Categories/Classes in column:",['bold']), df[col].nunique())
    print(magenta("Category Names:",['bold']))
    print(df[col].unique())
    print()
    print(magenta("Value Counts (FD) of each Category:",['bold']))
    print(df[col].value_counts())
    print()
    print(magenta("Value Counts of Each Class (FD) as Percentage:",['bold']))
    print(round((df[col].value_counts()/len(df))*100,2))
    print()
    print(magenta("Mode:",['bold']), df[col].mode()[0])
    print()
    print(cyan("Visualization:",['bold']))
    print()        
    print(black("Top Catgeories:", ['bold']))
    # Considering only top 10 categories for pie chart
    index = df[col].value_counts().sort_values(ascending=False)[0:10].index
    vals = df[col].value_counts().sort_values(ascending=False)[0:10].values
    fig = px.pie(names=index, values=vals, width=700, height=400)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False)
    fig.show()
    print()
    
######################################## DateTime ######################################
def datestudy(df, col):
    print(green("#######################################################",['bold']))
    print(green("Taken Date Column:",['bold']), black(col,['bold']))
    print(green("#######################################################",['bold']))
    print()
    print(cyan("Uni-Variate Descriptive Stats:",['bold']))
    print()
    print(magenta("Start Date:",['bold']), df[col].min())
    print(magenta("End Date:",['bold']), df[col].max())
    print(magenta("Total Time Period (in Years):",['bold']), (df[col].max()-df[col].min()))
    print()
    print(cyan("Visualization:",['bold']))
    print()
    index = df[col].value_counts().index
    vals = df[col].value_counts().values
    px.scatter(x = index, y = vals, width=500, height=400).show()
    print()


# In[27]:


for col in df.columns:
    if 'Year' in col:
        datestudy(df, col)
    elif df[col].dtype == object:
        catstudy(df, col)
    elif df[col].dtype == 'float64':
        ncstudy(df, col)
    elif df[col].dtype == 'int64':
        ndstudy(df, col)


# **Insights for Uni_Variate Columns**
# 1. From_City
# 
# Bengaluru, Lucknow, and Varanasi are top travel origins, showing high outbound demand from major tier-1 and tier-2 cities.
# 
# 2. Destination
# 
# Goa and Andaman are the most popular destinations, highlighting strong traveler interest in beach and coastal tourism.
# 
# 3. Destination_Type
# 
# Hill stations and city trips dominate, showing a preference for cool climates and urban leisure.
# 
# 4. Trip_Duration_Days
# 
# Most travelers prefer 4–6 day trips, making short vacations the most popular travel duration.
# 
# 5. Budget_Range
# 
# Medium-budget packages are most preferred, indicating demand for affordable yet comfortable travel options.
# 
# 6. Approx_Cost (₹)
# 
# Average trip cost is around ₹33K, confirming that travelers favor mid-range, value-for-money packages.
# 
# 7. Accommodation_Type
# 
# Hotels and resorts are top choices, emphasizing comfort and convenience in accommodation preferences.
# 
# 8. Transport_Mode
# 
# Train and flight are the main transport modes, showing travelers prioritize speed and cost-efficiency.
# 
# 9. Meal_Plan
# 
# Most travelers prefer full-board or breakfast-only plans for convenience and flexibility in dining.
# 
# 10. Activity_Count
# 
# Trips typically include 3–4 activities, balancing relaxation with engagement.
# 
# 11. Activity_Types
# 
# Sightseeing and cultural experiences lead, reflecting travelers’ interest in exploration over adventure.
# 
# 12. Season
# 
# Winter is the peak travel season, with significantly higher bookings than in summer.
# 
# 13. Package_Type
# 
# Budget and adventure packages dominate, showing a market leaning toward affordable excitement.
# 
# 14. Recommended_For
# 
# Family and friends groups are the primary travelers, highlighting demand for group-friendly packages.

# **Bi/Multi-Variate Analysis** - Study of Data between two/more columns

# Bi/Multi-Variate Combo|Stats Measures
# -----------|-----------
# Numerical-Numerical|Correlation (-1 to +1)
# Numerical-Categorical|Aggregation Functions (count, min, max, avg, sum)
# Categorical-Categorical|FDT
# - Correlation Coeffiecient Relation Categories
#   - 0.75 to 1 - Strong Correlation
#   - 0.50 to 0.75 - Moderate Correlation
#   - <0.50 - Weak Correlation
# - Visualizations
# 
#   - Pure Numerical : Relations: Scatter/Pairplots/Heatmaps
# 
#   - Pure Categorical : Composition: Pie/Stacked BarCharts/Sunburst Charts
# 
#   - Mixed: Composition : Pie/Barcharts/Sunburst
# 
#  - Apart from above combos if we can have Date & Locations , then we can have
# 
#    - Date: Trends Over time: LineCharts/Area Charts
#    - Locations: GeoSpatial: Choropleth maps

# **Selecting specific column combos from above to study the data**
# 
# Pure Numerical Columns|Mixed Columns|Pure Categorical Columns
# ----------------------|-------------|------------------------
# Trip_Duration_Days vs Approx_Cost (₹)|Approx_Cost (₹) vs Package_Type|Package_Type vs Recommended_For
# Activity_Count vs Approx_Cost (₹)|Trip_Duration_Days vs Destination_Type|Destination_Type vs Season
# 
# 
# 

# **Pure Numerical Columns**
# - Descriptive Stats
#    - Correlation Coefficient 
# - Visualizations
#   - Scatter/Heatmaps/Pairplots.

#                1. Trip_Duration_Days vs Approx_Cost

# In[28]:


print("Checking if longer trips cost more...\n")

# Average cost for each trip duration
avg_cost = df.groupby('Trip_Duration_Days')['Approx_Cost'].mean().reset_index()
print(avg_cost)

# Simple visual
plt.figure(figsize=(7,5))
sns.lineplot(x='Trip_Duration_Days', y='Approx_Cost', data=avg_cost, marker='o')
plt.title("Trip Duration vs Average Cost")
plt.xlabel("Trip Duration (Days)")
plt.ylabel("Average Cost")
plt.show()


# **Insight**
# - Trips that last longer usually cost more, so if you plan a short getaway, it will be cheaper.

#                  2. Activity_Count vs Approx_Cost

# In[29]:


print("Checking if more activities increase cost...\n")

avg_cost_act = df.groupby('Activity_Count')['Approx_Cost'].mean().reset_index()
print(avg_cost_act)

plt.figure(figsize=(7,5))
sns.barplot(x='Activity_Count', y='Approx_Cost', data=avg_cost_act, palette='coolwarm')
plt.title("Activity Count vs Average Cost")
plt.xlabel("Number of Activities")
plt.ylabel("Average Cost")
plt.show()


# **Insight**
# - The more activities included in your trip, the higher the price — pick the number of activities based on your budget.

# **Mixed Columns**
# - Descriptive Stats
#    - Aggregations : Count, Min, Max, Avg, Sum. 
# - Visualizations
#   - Pie/Bar Chart

#                     1. Approx_Cost vs Package_Type

# In[30]:


print("Average cost for each package type:\n")

avg_pkg = df.groupby('Package_Type')['Approx_Cost'].mean().sort_values(ascending=False)
print(avg_pkg)

plt.figure(figsize=(8,5))
sns.barplot(x=avg_pkg.index, y=avg_pkg.values, palette='magma')
plt.title("Average Cost by Package Type")
plt.xlabel("Package Type")
plt.ylabel("Average Cost")
plt.show()


# **Insight**
# - Luxury packages are the most expensive, budget packages are the cheapest — choose a package type that fits your wallet.

#                 2. Trip_Duration_Days vs Destination_Type

# In[31]:


from scipy.stats import kruskal

# Kruskal–Wallis test (non-parametric)
groups = [g["Trip_Duration_Days"].values for _, g in df.groupby("Destination_Type")]
hstat, pval = kruskal(*groups)
print(f"Kruskal–Wallis H-statistic: {hstat:.3f}, p-value: {pval:.4f}")

# Visualization
plt.figure(figsize=(10,5))
sns.boxplot(x='Destination_Type', y='Trip_Duration_Days', data=df, palette='coolwarm')
plt.title("Trip Duration by Destination Type")
plt.xlabel("Destination Type")
plt.ylabel("Trip Duration (Days)")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# **Insight**
# - Hill stations and adventure spots need more days to enjoy fully, while city or beach trips can be shorter.

# **Categorical Columns**
# - Descriptive stats
#   - FDT
# - Visualizations
#   - Sunburst

#                         1. Package_Type vs Recommended_For

# In[32]:


print("How different traveler types choose package types:\n")

count_table = pd.crosstab(df['Recommended_For'], df['Package_Type'])
print(count_table)

plt.figure(figsize=(8,5))
sns.heatmap(count_table, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Traveler Type vs Package Type")
plt.xlabel("Package Type")
plt.ylabel("Traveler Type")
plt.show()


# **Insight**
# - Families, friends, and couples prefer different packages, but Budget will be the right type for each group.

#                         2. Destination_Type vs Season

# In[33]:


print("Season-wise preference for each destination type:\n")

count_table2 = pd.crosstab(df['Destination_Type'], df['Season'])
print(count_table2)

plt.figure(figsize=(9,5))
sns.heatmap(count_table2, annot=True, fmt='d', cmap='coolwarm')
plt.title("Destination Type vs Season")
plt.xlabel("Season")
plt.ylabel("Destination Type")
plt.show()


# **Insight**
# - Different destinations shine in different seasons — beaches are popular in winter, hills in summer, and cities all year round.

# **Overall Insights for Bi/Multi-Variate Columns**
# - 1️. Trip Duration vs Cost
#    - Trips that last longer usually cost more, so if you plan a short getaway, it will be cheaper.
# 
# - 2. Activity Count vs Cost
#    - The more activities included in your trip, the higher the price — pick the number of activities based on your budget.
# 
# - 3. Cost vs Package Type
#    - Luxury packages are the most expensive, budget packages are the cheapest — choose a package type that fits your wallet.
# 
# - 4️. Trip Duration vs Destination Type
#    - Hill stations and adventure spots need more days to enjoy fully, while city or beach trips can be shorter.
# 
# - 5️. Package Type vs Recommended For
#    - Families, friends, and couples prefer different packages — travel companies can suggest the right type for each group.
# 
# - 6️. Destination Type vs Season
#    - Different destinations shine in different seasons — beaches are popular in winter, hills in summer, and cities all year round.

# **3. Missing Values & Outliers Handling**<a id='MVOH'>
#     
# [Back to Top](#ToC)

# 
# **3.1 Missing Values Identification & Handling**

# * Empty values or any data point which is not belongs to column.
# * Identify Missing Values
#     - Check for Standard & Non-Standard nan values 
# * Handle the Missing Values
#     - Drop (Row, Column)
#     - Replace (MCT, Imputation, etc...)
# * Some ML Algorithms will not accept missing data.

# **a) Identification**
# - Using pandas dataframe **isnull()** function

# **i) Row Wise Na Count**

# In[34]:


data.isnull().sum(axis=1)


# **ii) Column Wise Na Count**

# In[35]:


data.isnull().sum()


# - We can proceed to the next step, because there are No missing values.

# **3.2 Outliers Handling**
# * For a Numeric Col, if we got extreme values, then these are considered as outliers according to stats.
# * Outlier can be lower or higher values
# * Outliers will effect mean and std stats params
# * Some of ML Algorithms are sensitive to Outliers

# In[36]:


# Taking user defined module

import plotly.express as px

def outlier_detect(df):
    cols = []
    for col in df.describe().columns:
        print("Column:",col)
        print("------------------------------------------------")
        print("Boxplot For Outlier Identification:")
        px.box(df[col], orientation='h', width=600, height=300, ).show()
        print()
        Q1 = df.describe().at['25%',col]
        Q3 = df.describe().at['75%',col]
        IQR = Q3 - Q1
        lowerbound = Q1 - 1.5 * IQR
        upperbound = Q3 + 1.5 * IQR
        
        print("********* Outlier Data Points *******")
        print()
        lowerout = []
        upperout = []

        for val in df[col]:
            if val<lowerbound:
                if val not in lowerout:
                    lowerout.append(val)
            elif val>upperbound:
                if val not in upperout:
                    upperout.append(val)

        lowerout.sort()
        upperout.sort()

        print("Lower Outliers:")
        print(lowerout)
        print()
        print()
        print("Upper Outliers:")
        print(upperout)
        print()
        print("===============================================")
        print()
        
        if lowerout!=[] or upperout!=[]:
            cols.append(col)
      
    return cols
        
def outlier_replacement(df, cols):
    for col in cols:
        print("Column:",col)
        print("------------------------------------------------")
        Q1 = df.describe().at['25%',col]
        Q3 = df.describe().at['75%',col]
        IQR = Q3 - Q1
        LTV = Q1 - 1.5 * IQR
        UTV = Q3 + 1.5 * IQR
        
        # replacement vals (any one of the below)
        
        # 1. Median
        median = df[col].median()
        
        # 2. Ltv, Utv
        low_bound = LTV
        high_bound = UTV
        
        # 3. 5th & 95th (Suggested)
        fifth = df[col].quantile(0.05)
        ninetyfifth = df[col].quantile(0.95)

        print("Replacing Outliers with 5th percentile for lower Outliers, 95th percentile for Upper Outliers....")
        print("Adjust the module code for any other replacements.........")
        print()
        
        # mask method is used to replace the values
        df[col] = df[col].mask(df[col]<LTV, round(fifth)) # replacing the lower outlier with 5th percentile value
        df[col] = df[col].mask(df[col]>UTV, round(ninetyfifth)) # replacing the outlier with 95th percentile value


# In[37]:


outcols = outlier_detect(data)


# - We do have some Uppere Outliers in Approx_Cost (₹) Column

# In[38]:


# Handling Outliers by replacing values with nearest values (5th and 95th percentile)

outlier_replacement(data, outcols)


# In[39]:


#Final check for outliers
outlier_detect(data)


# **4.2.2 Feature Modification(Data Pre-Processing)**
# 
# **Encoding :** Converting all Categorical columns to Numerical columns.

# In[40]:


df = df.drop(['Package_ID'], axis=1)


# In[41]:


# Saving this for future purpose
df.to_csv("finaltraveldata.csv", index = False)


# In[42]:


# Import for OneHotEncoder, Minmax Scalar
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


# In[43]:


# Encode categorical columns
cat_cols = ['From_City', 'Destination', 'Destination_Type', 
            'Budget_Range', 'Accommodation_Type', 'Transport_Mode', 
            'Meal_Plan', 'Activity_Types', 'Season', 
            'Package_Type', 'Recommended_For']

num_cols = ['Trip_Duration_Days', 'Approx_Cost', 'Activity_Count']

# Preprocess Data

# OneHotEncode categorical columns
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_features = ohe.fit_transform(df[cat_cols])

# Scale numeric columns
scaler = MinMaxScaler()
num_features = scaler.fit_transform(df[num_cols])

# Combine numeric + categorical features
cdata = np.hstack([num_features, cat_features])


# In[44]:


# Import for NearestNeighbors
from sklearn.neighbors import NearestNeighbors


# In[48]:


# Fit NearestNeighbors Model
cosinemodel = NearestNeighbors(n_neighbors=10, metric='cosine')
cosinemodel.fit(cdata)

# User Input
def user_input():
    
    inpdata = pd.read_csv("finaltraveldata.csv")
    
    print(inpdata['From_City'].unique())
    From_City = input("Select your From_City:")
    print()
    
    print(inpdata['Destination'].unique())
    Destination = input("Select your Destination:")
    print()
    
    print(inpdata['Destination_Type'].unique())
    Destination_Type = input("Select your Destination_Type:")
    print()
    
    Trip_Duration_Days = int(input(f"Enter Trip_Duration_Days , Range : {inpdata['Trip_Duration_Days'].min()} to {inpdata['Trip_Duration_Days'].max()}:"))
    print()
    
    print(inpdata['Budget_Range'].unique())
    Budget_Range = input("Select your Budget_Range:")
    print()
    
    Approx_Cost = float(input(f"Enter your Approx_Cost, Range : {inpdata['Approx_Cost'].min()} to {inpdata['Approx_Cost'].max()}:"))
    print()
 
    print(inpdata['Accommodation_Type'].unique())
    Accommodation_Type = input("Select your Accommodation_Type:")
    print()
    
    print(inpdata['Transport_Mode'].unique())
    Transport_Mode = input("Select your Transport_Mode :")
    print()
    
    print(inpdata['Meal_Plan'].unique())
    Meal_Plan = input("Select your Meal_Plan :")
    print()
    
    Activity_Count = int(input(f"Enter your Activity_Count , Range : {inpdata['Activity_Count'].min()} to {inpdata['Activity_Count'].max()} :"))
    print()
    
    print(inpdata['Activity_Types'].unique())
    Activity_Types = input("Select Activity_Types :")
    print()
    
    print(inpdata['Season'].unique())
    Season = input("Select Season :")
    print()   
    
    print(inpdata['Package_Type'].unique())
    Package_Type = input("Select Package_Type :")
    print()
    
    print(inpdata['Recommended_For'].unique())
    Recommended_For = input("Select Recommended_For : ")
    print()
    
    row = pd.DataFrame([[From_City, Destination, Destination_Type, Trip_Duration_Days, Budget_Range, Approx_Cost, Accommodation_Type, Transport_Mode, Meal_Plan, Activity_Count, Activity_Types, Season, Package_Type, Recommended_For]], columns=inpdata.columns)              
                     
    print("Given Item Input Data :")
    display(row)            
    print()
    
    return row

user_df = user_input()

# Transform user categorical input
user_cat = ohe.transform(user_df[cat_cols])

# Scale numeric input
user_num = scaler.transform(user_df[num_cols])

# Combine numeric + categorical
user_vector = np.hstack([user_num, user_cat])

# Find Nearest Packages
distances, indices = cosinemodel.kneighbors(user_vector)

# Get top recommended packages
top_packages = df.iloc[indices[0]].copy()
top_packages['Similarity_Score'] = 1 - distances.flatten()  # Convert distance to similarity

# Display top recommendations
top_packages_display = top_packages[['From_City', 'Destination', 'Destination_Type', 'Trip_Duration_Days', 'Budget_Range', 
                                    'Approx_Cost', 'Accommodation_Type', 'Transport_Mode', 'Activity_Count',  
                                    'Package_Type', 'Similarity_Score']]

print("Top Recommended Packages:\n")
print(top_packages_display)

print(user_input)


# In[49]:


top_packages = top_packages_display.to_dict(orient='records')

# Print JSON output
print(json.dumps(top_packages, indent=4))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import pandas as pd
import streamlit as st

# Read your dataset
df = pd.read_csv("travel_packages_with_images.csv")

# Example: take first 3 recommended packages
for i in range(3):
    st.subheader(df.loc[i, 'Destination'])
    st.image(df.loc[i, 'Destination_Image_URL'], caption=df.loc[i, 'Destination_Type'])


# In[ ]:





# In[ ]:





# In[ ]:




