# EDA_NYC_PROJECT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Show all columns
pd.set_option('display.max_columns', None)

# --- Step 1: Load Data ---
folder_path = "./"  # Update if your path is different
all_months = []

for month in range(1, 13):
    file_name = f"yellow_tripdata_2023-{month:02d}.parquet"
    file_path = os.path.join(folder_path, file_name)
    print(f"Loading {file_name}...")
    df_month = pd.read_parquet(file_path, engine='pyarrow')
    all_months.append(df_month)

df = pd.concat(all_months, ignore_index=True)
print("\n✅ Data Loaded. Shape:", df.shape)

# --- Step 2: Clean & Prepare ---
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

df = df.dropna(subset=['fare_amount', 'trip_distance', 'tip_amount'])

df['trip_duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

df = df[(df['trip_duration_min'] > 0) & (df['trip_duration_min'] < 180)]
df = df[(df['trip_distance'] > 0) & (df['fare_amount'] > 0)]

df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['pickup_day'] = df['tpep_pickup_datetime'].dt.day_name()
df['pickup_month'] = df['tpep_pickup_datetime'].dt.month_name()

payment_map = {
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided trip"
}
df['payment_type_str'] = df['payment_type'].map(payment_map)

# --- Step 3: Univariate Analysis ---
plt.figure(figsize=(10, 5))
sns.countplot(x='pickup_hour', data=df)
plt.title('Trips by Pickup Hour')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df['fare_amount'], bins=100, kde=True)
plt.title('Fare Amount Distribution')
plt.xlim(0, df['fare_amount'].quantile(0.99))
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df['tip_amount'], bins=50)
plt.title('Tip Amount Distribution')
plt.xlim(0, df['tip_amount'].quantile(0.99))
plt.tight_layout()
plt.show()

# --- Step 4: Bivariate Analysis ---
plt.figure(figsize=(8, 5))
sns.boxplot(x='payment_type_str', y='tip_amount', data=df)
plt.title('Tip Amount by Payment Type')
plt.ylim(0, df['tip_amount'].quantile(0.99))
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='trip_distance', y='fare_amount', alpha=0.2)
plt.title('Fare vs Trip Distance')
plt.xlim(0, df['trip_distance'].quantile(0.99))
plt.ylim(0, df['fare_amount'].quantile(0.99))
plt.tight_layout()
plt.show()

# --- Step 5: Grouped Insights ---
df.groupby('pickup_day')['fare_amount'].mean().sort_values().plot(kind='bar', figsize=(8, 5), color='teal')
plt.title('Average Fare by Day of Week')
plt.ylabel('Avg Fare')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

if 'VendorID' in df.columns:
    df.groupby('VendorID')['trip_duration_min'].mean().plot(kind='bar', color='orange')
    plt.title('Avg Trip Duration by Vendor')
    plt.ylabel('Minutes')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    print("\n🚖 Total Trips by Vendor:\n", df['VendorID'].value_counts())

# --- Step 6: Recommendations ---
print("\n📝 Recommendations:")
print("1. Boost taxi presence during peak hours (5–8 PM) and weekends.")
print("2. Encourage card payments to increase tips and ease processing.")
print("3. Position cabs proactively near high-demand zones (Midtown, Downtown).")
print("4. Consider short-trip incentives to improve fleet efficiency.")
