import os
import pandas as pd
import matplotlib.pyplot as plt
from src.data_processing import load_data, clean_data
from src.rfm_analysis import calculate_rfm, assign_rfm_scores, segment_customers
from src.clustering import (prepare_data_for_clustering, find_optimal_clusters, 
                          perform_kmeans_clustering, add_cluster_labels)
from src.visualization import (plot_rfm_distribution, plot_elbow_method, 
                             plot_3d_clusters, plot_cluster_profiles,
                             plot_segment_distribution, plot_segment_rfm_boxplots)

def main():
    # Create output directories if they don't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Step 1: Load and clean data
    print("Loading and cleaning data...")
    file_path = "data/raw/ecommerce_data.csv"
    df = load_data(file_path)
    df_clean = clean_data(df)
    
    # Save the cleaned data
    df_clean.to_csv('data/processed/cleaned_ecommerce_data.csv', index=False)
    
    # Step 2: Calculate RFM
    print("Calculating RFM metrics...")
    rfm_data = calculate_rfm(df_clean)
    
    # Step 3: Assign RFM scores
    print("Assigning RFM scores...")
    rfm_scores = assign_rfm_scores(rfm_data)
    
    # Step 4: Segment customers
    print("Segmenting customers based on RFM scores...")
    segmented_data = segment_customers(rfm_scores)
    
    # Save the segmented data
    segmented_data.to_csv('data/processed/rfm_segmented_data.csv', index=False)
    
    # Step 5: Prepare data for clustering
    print("Preparing data for clustering...")
    X_scaled, scaled_df, features = prepare_data_for_clustering(rfm_data)
    
    # Step 6: Find optimal number of clusters
    print("Finding optimal number of clusters...")
    wcss, silhouette_scores = find_optimal_clusters(X_scaled)
    
    # Step 7: Perform K-means clustering
    # Choose the optimal number of clusters based on the Elbow method and Silhouette score
    optimal_clusters = 4  # This value can be adjusted based on the plots
    print(f"Performing K-means clustering with {optimal_clusters} clusters...")
    kmeans = perform_kmeans_clustering(X_scaled, optimal_clusters)
    
    # Step 8: Add cluster labels to the data
    clustered_data = add_cluster_labels(rfm_data, kmeans, scaled_df)
    
    # Save the clustered data
    clustered_data.to_csv('data/processed/clustered_data.csv', index=False)
    
    # Step 9: Create visualizations
    print("Creating visualizations...")
    # Plot RFM distributions
    plot_rfm_distribution(rfm_data)
    
    # Plot Elbow method and Silhouette scores
    plot_elbow_method(wcss, silhouette_scores)
    
    # Plot 3D clusters
    plot_3d_clusters(clustered_data, features)
    
    # Plot cluster profiles
    plot_cluster_profiles(clustered_data)
    
    # Plot segment distribution
    plot_segment_distribution(segmented_data)
    
    # Plot segment RFM boxplots
    plot_segment_rfm_boxplots(segmented_data)
    
    print("Analysis complete! Results saved to the 'results' directory.")

if __name__ == "__main__":
    main()
