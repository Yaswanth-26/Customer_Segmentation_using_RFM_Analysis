import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def plot_rfm_distribution(rfm_data):
    """
    Plot the distribution of RFM metrics.
    
    Parameters:
    rfm_data (DataFrame): DataFrame with RFM metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot Recency distribution
    sns.histplot(data=rfm_data, x='Recency', bins=30, ax=axes[0])
    axes[0].set_title('Recency Distribution')
    
    # Plot Frequency distribution
    sns.histplot(data=rfm_data, x='Frequency', bins=30, ax=axes[1])
    axes[1].set_title('Frequency Distribution')
    
    # Plot Monetary distribution
    sns.histplot(data=rfm_data, x='Monetary', bins=30, ax=axes[2])
    axes[2].set_title('Monetary Distribution')
    
    plt.tight_layout()
    plt.savefig('results/rfm_distributions.png')
    return fig

def plot_elbow_method(wcss, silhouette_scores):
    """
    Plot the Elbow method and Silhouette scores.
    
    Parameters:
    wcss (list): Within-cluster sum of squares
    silhouette_scores (list): Silhouette scores
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Elbow method
    axes[0].plot(range(2, len(wcss) + 2), wcss, marker='o', linestyle='--')
    axes[0].set_title('Elbow Method')
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('WCSS')
    
    # Plot Silhouette scores
    axes[1].plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o', linestyle='--')
    axes[1].set_title('Silhouette Score Analysis')
    axes[1].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.savefig('results/clustering_analysis.png')
    return fig

def plot_3d_clusters(clustered_data, features):
    """
    Create a 3D scatter plot of clusters.
    
    Parameters:
    clustered_data (DataFrame): DataFrame with cluster labels
    features (list): List of feature names
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each cluster
    for cluster in sorted(clustered_data['Cluster'].unique()):
        cluster_data = clustered_data[clustered_data['Cluster'] == cluster]
        ax.scatter(
            cluster_data['Scaled_' + features[0]],
            cluster_data['Scaled_' + features[1]],
            cluster_data['Scaled_' + features[2]],
            label=f'Cluster {cluster}'
        )
    
    ax.set_xlabel('Scaled Recency')
    ax.set_ylabel('Scaled Frequency')
    ax.set_zlabel('Scaled Monetary')
    ax.set_title('3D Plot of Customer Segments')
    ax.legend()
    
    plt.savefig('results/3d_clusters.png')
    return fig

def plot_cluster_profiles(clustered_data):
    """
    Plot the profiles of each cluster.
    
    Parameters:
    clustered_data (DataFrame): DataFrame with cluster labels
    """
    # Calculate mean values for each cluster
    cluster_profiles = clustered_data.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    # Melt the DataFrame for easier plotting
    melted_profiles = pd.melt(
        cluster_profiles.reset_index(),
        id_vars=['Cluster'],
        value_vars=['Recency', 'Frequency', 'Monetary'],
        var_name='Metric',
        value_name='Value'
    )
    
    # Plot the profiles
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Cluster', y='Value', hue='Metric', data=melted_profiles)
    plt.title('Cluster Profiles')
    plt.xlabel('Cluster')
    plt.ylabel('Average Value')
    plt.legend(title='RFM Metric')
    
    plt.tight_layout()
    plt.savefig('results/cluster_profiles.png')
    return plt.gcf()

def plot_segment_distribution(rfm_data):
    """
    Plot the distribution of customer segments.
    
    Parameters:
    rfm_data (DataFrame): DataFrame with customer segments
    """
    # Count the number of customers in each segment
    segment_counts = rfm_data['Segment'].value_counts()
    
    # Plot the distribution
    plt.figure(figsize=(12, 8))
    segment_counts.plot(kind='pie', autopct='%.1f%%')
    plt.title('Distribution of Customer Segments')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig('results/segment_distribution.png')
    return plt.gcf()

def plot_segment_rfm_boxplots(rfm_data):
    """
    Create boxplots showing RFM distributions for each segment.
    
    Parameters:
    rfm_data (DataFrame): DataFrame with customer segments
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    # Plot Recency boxplot by segment
    sns.boxplot(x='Segment', y='Recency', data=rfm_data, ax=axes[0])
    axes[0].set_title('Recency Distribution Across Segments')
    axes[0].set_xlabel('Customer Segments')
    axes[0].set_ylabel('Recency (Days)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot Frequency boxplot by segment
    sns.boxplot(x='Segment', y='Frequency', data=rfm_data, ax=axes[1])
    axes[1].set_title('Frequency Distribution Across Segments')
    axes[1].set_xlabel('Customer Segments')
    axes[1].set_ylabel('Frequency')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot Monetary boxplot by segment
    sns.boxplot(x='Segment', y='Monetary', data=rfm_data, ax=axes[2])
    axes[2].set_title('Monetary Distribution Across Segments')
    axes[2].set_xlabel('Customer Segments')
    axes[2].set_ylabel('Monetary')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/segment_rfm_boxplots.png')
    return fig
