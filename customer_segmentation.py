import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("customer_data.csv")

# Extract relevant features for clustering
X = data[['Feature1', 'Feature2', 'Feature3']]  # Replace 'Feature1', 'Feature2', 'Feature3' with your actual features

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3)  # Adjust the number of clusters as needed
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Add cluster labels to the dataset
data['Cluster'] = labels

# Visualize the clusters
plt.scatter(data['Feature1'], data['Feature2'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Customer Segmentation')
plt.show()
