import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def prepare_data_for_clustering(rfm_data):
    """
    Prepare RFM data for clustering by scaling features.
    
    Parameters:
    rfm_data (DataFrame): DataFrame with RFM metrics
    
    Returns:
    tuple: Scaled data array and feature names
    """
    # Select features for clustering
    features = ['Recency', 'Frequency', 'Monetary']
    X = rfm_data[features].values
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create a DataFrame with scaled values
    scaled_df = pd.DataFrame(X_scaled, columns=['Scaled_' + feature for feature in features])
    
    return X_scaled, scaled_df, features

def find_optimal_clusters(X, max_clusters=10):
    """
    Find the optimal number of clusters using the Elbow method and Silhouette score.
    
    Parameters:
    X (array): Scaled feature array
    max_clusters (int): Maximum number of clusters to try
    
    Returns:
    tuple: Elbow method results and silhouette scores
    """
    wcss = []  # Within-cluster sum of squares
    silhouette_scores = []
    
    # Try different numbers of clusters
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
        # Calculate silhouette score for i > 1
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, labels))
    
    return wcss, silhouette_scores

def perform_kmeans_clustering(X, n_clusters):
    """
    Perform K-means clustering with the specified number of clusters.
    
    Parameters:
    X (array): Scaled feature array
    n_clusters (int): Number of clusters
    
    Returns:
    KMeans: Fitted K-means model
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    return kmeans

def add_cluster_labels(rfm_data, kmeans, scaled_df):
    """
    Add cluster labels to the RFM data.
    
    Parameters:
    rfm_data (DataFrame): Original RFM data
    kmeans (KMeans): Fitted K-means model
    scaled_df (DataFrame): DataFrame with scaled features
    
    Returns:
    DataFrame: RFM data with cluster labels
    """
    # Create a copy of the original data
    clustered_data = rfm_data.copy()
    
    # Add cluster labels
    clustered_data['Cluster'] = kmeans.labels_
    
    # Add scaled features
    for col in scaled_df.columns:
        clustered_data[col] = scaled_df[col].values
    
    return clustered_data
