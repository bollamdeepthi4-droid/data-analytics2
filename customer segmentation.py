import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ==============================================================
# LOAD DATASET
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath("C:\Users\bolla\OneDrive\Desktop\Deepu\customer_project"))
FILE_NAME = os.path.join(BASE_DIR, "Mall_Customer.csv")
# If your file is named Mall_Customers.csv, change the filename above.

if not os.path.exists(FILE_NAME):
    raise FileNotFoundError(
        f"\nDataset not found:\n{FILE_NAME}\n"
        "Place the CSV file in the same folder as this script."
    )

df = pd.read_csv(FILE_NAME)

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Information")
df.info()                      # Don't use print(df.info())

print("\nDataset Shape:", df.shape)

# ==============================================================
# CHECK REQUIRED COLUMNS
# ==============================================================

required_columns = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

# ==============================================================
# CHECK DATASET SIZE
# ==============================================================

if len(df) < 20:
    raise ValueError(
        f"""
Only {len(df)} rows found.

The original Mall Customers dataset contains about 200 rows.

Please use the complete dataset instead of only the first few rows.
"""
    )

# ==============================================================
# CLEAN DATA
# ==============================================================

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print("\nDataset Shape After Cleaning:", df.shape)

# ==============================================================
# FEATURE SELECTION
# ==============================================================

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]

# ==============================================================
# FEATURE SCALING
# ==============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)