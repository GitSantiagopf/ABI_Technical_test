import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class Clusterer:
    """Class for performing clustering using original variables and exporting plots.

    This class preprocesses the data (scaling the selected features), applies clustering
    algorithms (KMeans and DBSCAN), and generates and saves plots for evaluation including
    metric curves, boxplots, and PCA scatter plots. Finally, it can save the final DataFrame
    with cluster labels to a CSV file.
    """
    def __init__(self, df: pd.DataFrame, features: list):

        self.df = df.copy()
        self.features = features
        
        if not os.path.exists("plots"):
            os.makedirs("plots")
        self.scaler = RobustScaler()
        self.X_scaled = None

    def preprocess(self):
        """Extracts features and scales them.

        Returns:
            np.ndarray: The scaled feature matrix.
        """
        X = self.df[self.features].copy()
        self.X_scaled = self.scaler.fit_transform(X)
        return self.X_scaled

    def run_kmeans(self, n_clusters: int = 3):
        """Runs KMeans clustering and stores labels in DataFrame as 'Cluster'.

        Args:
            n_clusters (int): Number of clusters. Default is 6.

        Returns:
            pd.Series: The cluster labels.
        """
        if self.X_scaled is None:
            self.preprocess()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['Cluster'] = kmeans.fit_predict(self.X_scaled)
        print(f"KMeans ejecutado con {n_clusters} clusters.")
        return self.df['Cluster']

    def run_dbscan(self, eps: float = 0.5, min_samples: int = 10):
        """Runs DBSCAN clustering and stores labels as 'Cluster_DBSCAN'.

        Args:
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

        Returns:
            pd.Series: The DBSCAN cluster labels.
        """
        if self.X_scaled is None:
            self.preprocess()
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        self.df['Cluster_DBSCAN'] = dbscan.fit_predict(self.X_scaled)
        print("DBSCAN executed")
        return self.df['Cluster_DBSCAN']

    def plot_metric_curves(self):
        """Computes and plots clustering metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
        for k from 2 to 10, saving the plot.
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

        k_values = range(2, 11)
        siluetas, calinski, davies = [], [], []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(self.X_scaled)
            sil_score = silhouette_score(self.X_scaled, clusters)
            calinski_score = calinski_harabasz_score(self.X_scaled, clusters)
            davies_score = davies_bouldin_score(self.X_scaled, clusters)
            siluetas.append(sil_score)
            calinski.append(calinski_score)
            davies.append(davies_score)
            print(f"k={k}: Silhouette={sil_score:.4f}, Calinski-Harabasz={calinski_score:.2f}, Davies-Bouldin={davies_score:.4f}")

        plt.figure(figsize=(14,4))
        plt.subplot(1,3,1)
        plt.plot(list(k_values), siluetas, marker='o')
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Índice de Silhouette")
        plt.title("Silhouette Score")
        plt.grid(True)

        plt.subplot(1,3,2)
        plt.plot(list(k_values), calinski, marker='o', color='green')
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Calinski-Harabasz Score")
        plt.title("Calinski-Harabasz Score")
        plt.grid(True)

        plt.subplot(1,3,3)
        plt.plot(list(k_values), davies, marker='o', color='red')
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Davies-Bouldin Score")
        plt.title("Davies-Bouldin Score (Menor es Mejor)")
        plt.grid(True)
        plt.tight_layout()
        filename = "plots/metric_curves.png"
        plt.savefig(filename)
        print(f"metrics curve saved in {filename}")
        plt.close()

    def plot_boxplots(self):
        """Generates and saves boxplots for each feature by KMeans cluster."""
        for col in self.features:
            plt.figure(figsize=(8,5))
            sns.boxplot(x='Cluster', y=col, data=self.df, palette='Set3')
            plt.title(f"Distribución de {col} por Cluster (KMeans)")
            filename = f"plots/boxplot_{col}_by_cluster.png"
            plt.savefig(filename)
            print(f"Boxplot saved en {filename}")
            plt.close()

    def plot_pca(self):
        """Performs PCA (2 components) and saves scatter plots for KMeans and DBSCAN clusters."""
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(self.X_scaled)
        self.df['pca_one'] = pca_result[:, 0]
        self.df['pca_two'] = pca_result[:, 1]

        # PCA for KMeans
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=self.df, x='pca_one', y='pca_two', hue='Cluster',
                        palette='Set1', legend='full', alpha=0.7)
        plt.title("Visualización PCA de Clusters KMeans")
        filename1 = "plots/pca_clusters_kmeans.png"
        plt.savefig(filename1)
        print(f"PCA scatter (KMeans) saved en {filename1}")
        plt.close()

        # PCA for DBSCAN
        plt.figure(figsize=(8,6))
        if 'Cluster_DBSCAN' in self.df.columns:
            sns.scatterplot(data=self.df, x='pca_one', y='pca_two', hue='Cluster_DBSCAN',
                            palette='Set1', legend='full', alpha=0.7)
            plt.title("Visualización PCA de Clusters DBSCAN")
            filename2 = "plots/pca_clusters_dbscan.png"
            plt.savefig(filename2)
            print(f"PCA scatter (DBSCAN) savend in {filename2}")
            plt.close()
        else:
            print("error for dbscan")

    def save_results(self, output_csv: str = "client_segmentation.csv"):
        """_summary_

        Args:
            output_csv (str, optional): _description_. Defaults to "client_segmentation.csv".
        """
        self.df.to_csv(output_csv, index=False)
        print(f"Segmentation saved in '{output_csv}'")
