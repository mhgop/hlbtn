# --- --- --- PROVIDED CODE --- --- ---
from pathlib import Path
from typing import cast
import json
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

def reduce_dimensions(embeddings:np.ndarray, out_dimension:int, n_neighbors:int, metric:str) -> np.ndarray:
    """Reduce the dimensions of the embeddings to out_dimension using UMAP
       metric: 'cosine' or 'euclidean'
    """
    # Import trick: umap is slow to import, so we import it only when needed.
    import umap

    # Explanation of some parameters:
    min_dist = 0 # Ok for clustering, allow points to be close
    repulsion_strength = 1.5 # Default value is 1. We want to push things appart a bit more (improve separation)
    negative_sample_rate = 10 # Default value is 5. We want to sample more 'negative points', pushing things appart more often (improve separation)

    # Perform the reduction
    reducer = umap.UMAP(
        n_components=out_dimension,
        n_neighbors=n_neighbors,
        metric=metric,
        #
        min_dist=min_dist,
        repulsion_strength=repulsion_strength,
        negative_sample_rate=negative_sample_rate
    )
    reduced_embeddings = cast(np.ndarray, reducer.fit_transform(embeddings))
    return reduced_embeddings


def perform_hdbscan_clustering(distance_matrix, min_cluster_size:int, min_samples:int):
    """Clustering of the embeddings using HDBSCAN"""
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)
    return clusterer, cluster_labels


def load_from_json(file:Path)->tuple[list[str], np.ndarray]:
    """Load a json file from a path, returning the list of source and the corresponding source_embedding as a numpy matrix

        If we have:
            names, vectors = load_data(file)
        Then vectors[0] is the embedding of names[0]
    """
    with open(file, 'r') as f:
        d:list[dict] = list(json.load(f))
        sources = [ item["source"] for item in d]
        _vectors = [ np.array(item["source_embedding"]) for item in d]
        vector = np.stack(_vectors, axis=0)
        assert (_vectors[0] == vector[0]).all()
    #
    return sources, vector

    
if __name__ == '__main__':
    # Constants
    # Input
    input_data_path = Path('task_01_out.json')
    expected_shape = (30, 1536)     # 30 samples, 1536 features
    # Reduction
    reduced_data_path = Path('task_03_out_reduced.json')
    dim = 20    # See DIM below -- Must be below the number of sample for some initialisation in UMAP.
    force_reduce = True  # Set to False to avoid recomputing the reduction, and only experimenting with the clustering parameters.

    # --- Reduction ---
    if not reduced_data_path.exists() or force_reduce:
        #
        def reduce_save():
            names, embeddings = load_from_json(input_data_path)
            emb_a = reduce_dimensions(embeddings, dim, 3, 'cosine')         # Embedding cosine -> DIM euclidean
            # 2D
            emb_b = reduce_dimensions(emb_a, 2, 3, 'euclidean')             # DIM euclidean -> 2D euclidean
            emb_c = reduce_dimensions(embeddings, 2, 3, 'cosine')           # Embedding cosine -> 2D euclidean
            # 3D
            emb_d = reduce_dimensions(emb_a, 3, 3, 'euclidean')             # DIM euclidean -> 3D euclidean
            emb_e = reduce_dimensions(embeddings, 3, 3, 'cosine')           # Embedding cosine -> 3D euclidean
            # Check shapes
            assert emb_a.shape == (embeddings.shape[0], dim)
            assert emb_b.shape == (embeddings.shape[0], 2)
            assert emb_c.shape == (embeddings.shape[0], 2)
            assert emb_d.shape == (embeddings.shape[0], 3)
            assert emb_e.shape == (embeddings.shape[0], 3)
            # Save to file
            jsonlist = [
                {
                    "name":n,
                    "a":a.tolist(),
                    "b":b.tolist(),
                    "c":c.tolist(),
                    "d":d.tolist(),
                    "e":e.tolist()
                } for n, a, b, c, d, e in zip(names, emb_a, emb_b, emb_c, emb_d, emb_e)
            ]
            with open(reduced_data_path, 'w') as f:
                json.dump(jsonlist, f, indent=2)
        #
        reduce_save()
    #

    # --- Load data ---
    with open(reduced_data_path, 'r') as f:
        reduced_data = json.load(f)
        names = [ item["name"] for item in reduced_data]
        a = np.array([ item["a"] for item in reduced_data])
        b = np.array([ item["b"] for item in reduced_data])
        c = np.array([ item["c"] for item in reduced_data])
        d = np.array([ item["d"] for item in reduced_data])
        e = np.array([ item["e"] for item in reduced_data])
    #
    
    # --- 20D Clustering ---

    # Cluster the 20D embeddings using HDBSCAN ('a')
    clusterer, cluster_labels = perform_hdbscan_clustering(euclidean_distances(a), min_cluster_size=2, min_samples=1)



    # --- 2D Clustering & Plotting ---

    # Cluster the 2D embeddings using HDBSCAN ('c')
    clusterer_2d, cluster_labels_2d = perform_hdbscan_clustering(euclidean_distances(c), min_cluster_size=2, min_samples=1)
    
    # Create two side-by-side plots for clustering with a clear legend
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Function to plot points with clusters and annotate noise
    def plot_with_clusters(ax, points, clusters, names, title):
        unique_labels = set(clusters)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'k'  # Black for noise
                label_name = 'Noise'
            else:
                label_name = f'Cluster {label}'
            
            # Plot points belonging to this cluster
            label_points = points[clusters == label]
            ax.scatter(label_points[:, 0], label_points[:, 1], c=[color], label=label_name, s=30)
        
        # Annotate points with names
        for i, name in enumerate(names):
            ax.annotate(name, (points[i, 0], points[i, 1]), fontsize=6, alpha=0.6)
        
        # Add title and axis labels
        ax.set_title(title)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')

        # Place legend outside the plot area
        ax.legend( loc='upper center', fontsize='small', markerscale=1.2, frameon=True, bbox_to_anchor=(0.5, -0.1), borderaxespad=0.0, ncol=3)
    #


    # Plot 1: 20D clustered labels ('a') projected to 2D ('b')
    plot_with_clusters( axes[0], b, cluster_labels, names, 'Reduced to 20D then 2D, clustering in 20D')

    # Plot 2: Direct 2D reduction and clustering ('c')
    plot_with_clusters( axes[1], c, cluster_labels_2d, names, 'Direct Reduction to 2D with Clustering')

    plt.tight_layout()
    plt.show()
    plt.close()



    # --- 3D Clustering & Plotting ---

    # Cluster the 3D embeddings using HDBSCAN ('e')
    clusterer_3d, cluster_labels_3d = perform_hdbscan_clustering(euclidean_distances(e), min_cluster_size=2, min_samples=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})

    # Function to plot points with clusters and annotate noise
    def plot_with_clusters(ax, points, clusters, names, title):
        unique_labels = set(clusters)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'k'  # Black for noise
                label_name = 'Noise'
            else:
                label_name = f'Cluster {label}'
            
            # Plot points belonging to this cluster
            label_points = points[clusters == label]
            ax.scatter(label_points[:, 0], label_points[:, 1], c=[color], label=label_name, s=30)
        
        # Annotate points with names
        for i, name in enumerate(names):
            ax.annotate(name, (points[i, 0], points[i, 1]), fontsize=6, alpha=0.6)
        
        # Add title and axis labels
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # Place legend outside the plot area
        ax.legend( loc='upper center', fontsize='small', markerscale=1.2, frameon=True, bbox_to_anchor=(0.5, -0.1), borderaxespad=0.0, ncol=3)
    #


    # Plot 1: 20D clustered labels ('a') projected to 2D ('d')
    plot_with_clusters( axes[0], d, cluster_labels, names, 'Reduced to 20D then 3D, clustering in 20D')

    # Plot 2: Direct 2D reduction and clustering ('c')
    plot_with_clusters( axes[1], e, cluster_labels_3d, names, 'Direct Reduction to 3D with Clustering')

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
    plt.show()
    
    # --- --- ---
    exit(0)
        