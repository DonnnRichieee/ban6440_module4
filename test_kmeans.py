import unittest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

class TestKMeansClustering(unittest.TestCase):

    def setUp(self):
        # Load the dataset
        try:
            self.data = pd.read_csv('segmentation data.csv')
        except FileNotFoundError as e:
            self.fail(f"File not found: {e}")
        except pd.errors.EmptyDataError as e:
            self.fail(f"No data: {e}")
        except Exception as e:
            self.fail(f"Error reading the file: {e}")

        # Standardize the data
        self.scaler = StandardScaler()
        try:
            self.data_scaled = self.scaler.fit_transform(self.data)
        except Exception as e:
            self.fail(f"Error during data scaling: {e}")

        # Initialize KMeans
        self.kmeans = KMeans(n_clusters=3, random_state=42)

    def test_kmeans_clustering(self):
        try:
            self.kmeans.fit(self.data_scaled)
        except Exception as e:
            self.fail(f"Error fitting KMeans: {e}")
        
        labels = self.kmeans.labels_
        
        # Check if the number of labels matches the number of data points
        self.assertEqual(len(labels), len(self.data), "Mismatch in number of labels and data points")
        
        # Check if the number of unique labels (clusters) is as expected
        self.assertEqual(len(set(labels)), 3, "Number of clusters is not as expected")
        
        # Calculate and check the silhouette score
        try:
            silhouette_avg = silhouette_score(self.data_scaled, labels)
            self.assertTrue(0 <= silhouette_avg <= 1, "Silhouette score out of range")
        except Exception as e:
            self.fail(f"Error calculating silhouette score: {e}")
        
        # Calculate and check the Davies-Bouldin score
        try:
            davies_bouldin_avg = davies_bouldin_score(self.data_scaled, labels)
            self.assertTrue(davies_bouldin_avg >= 0, "Davies-Bouldin score is negative")
        except Exception as e:
            self.fail(f"Error calculating Davies-Bouldin score: {e}")

if __name__ == '__main__':
    unittest.main()
