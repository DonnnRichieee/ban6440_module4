# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np
import logging

# Set up logging for debugging and error handling
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logging.info(f'Data loaded successfully from {file_path}')
        return data
    except FileNotFoundError:
        logging.error(f'File not found: {file_path}')
        raise
    except pd.errors.EmptyDataError:
        logging.error(f'No data in file: {file_path}')
        raise
    except pd.errors.ParserError:
        logging.error(f'Error parsing file: {file_path}')
        raise
    except Exception as e:
        logging.error(f'An unexpected error occurred: {str(e)}')
        raise

def preprocess_data(data):
    """Standardize the data."""
    try:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        logging.info('Data preprocessing completed successfully')
        return data_scaled
    except Exception as e:
        logging.error(f'Error during data preprocessing: {str(e)}')
        raise

def kmeans_clustering(data, n_clusters=3):
    """Perform K-Means clustering."""
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        logging.info('K-Means clustering completed successfully')
        return kmeans
    except Exception as e:
        logging.error(f'Error during K-Means clustering: {str(e)}')
        raise

def evaluate_clustering(data, labels):
    """Evaluate the clustering performance."""
    try:
        silhouette_avg = silhouette_score(data, labels)
        davies_bouldin_avg = davies_bouldin_score(data, labels)
        logging.info(f'Silhouette Score: {silhouette_avg}')
        logging.info(f'Davies-Bouldin Score: {davies_bouldin_avg}')
        return silhouette_avg, davies_bouldin_avg
    except Exception as e:
        logging.error(f'Error during clustering evaluation: {str(e)}')
        raise

def visualize_clusters(data, labels):
    """Visualize the clusters."""
    try:
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('K-Means Clustering')
        plt.colorbar(label='Cluster')
        plt.show()
        logging.info('Cluster visualization completed successfully')
    except Exception as e:
        logging.error(f'Error during cluster visualization: {str(e)}')
        raise

# Main script execution
if __name__ == "__main__":
    try:
        # Load and preprocess the data
        data = load_data('segmentation data.csv')
        data_scaled = preprocess_data(data)

        # Perform K-Means clustering
        kmeans = kmeans_clustering(data_scaled, n_clusters=3)

        # Add the cluster labels to the original data
        data['Cluster'] = kmeans.labels_

        # Save the clustered data to a new CSV file
        data.to_csv('segmentation_data_with_clusters.csv', index=False)
        logging.info('Clustered data saved to segmentation_data_with_clusters.csv')

        # Evaluate the clustering
        silhouette_avg, davies_bouldin_avg = evaluate_clustering(data_scaled, kmeans.labels_)
        print(f'Silhouette Score: {silhouette_avg}')
        print(f'Davies-Bouldin Score: {davies_bouldin_avg}')

        # Visualize the clusters
        visualize_clusters(data_scaled, kmeans.labels_)

    except Exception as e:
        logging.error(f'An error occurred during the main script execution: {str(e)}')
