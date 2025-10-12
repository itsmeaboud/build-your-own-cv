import numpy as np
import cv2
import matplotlib.pyplot as plt



# Helper functions

# 1. Euclidean distance between 2 vectors (L-2 Norm)
def L2Distance (a, b, axis=1) :
    dist = np.linalg.norm(a - b, axis = axis)
    return dist


# 2. Randomly select K data points from the dataset
def init_centroids(data, K) :

    """ Randomly selects K unique data points from the dataset
         to serve as initial cluster centroids"""
    # len(data) gives the number of data points (rows)
    indices = np.random.choice(len(data), K, replace=False)
    return data[indices]

def kmeans(data, K, max_iter = 100, epislon = 1) :

    """K-Means Clustering Algorithm
    
    Parameters:
    data (ndarray): Input dataset (M x N matrix)
        M = number of data points (samples)
        N = number of features per point (e.g., RGB values, coordinates, etc.)
    K (int): Number of clusters (centroids) to find.
    max_iter (int): Maximum number of iterations to prevent infinite loop (default=10).
    epsilon (float): Minimum movement threshold for convergence. The algorithm stops 
                      if centroids move less than epsilon between iterations (default=1).
    
    Returns:
    tuple: 
        - means (ndarray): Final centroids (K x N)
        - labels (ndarray): Labels for each data point (M,)
            Each label indicates the cluster index the corresponding data point belongs to.
    
    Example:
    --------
    data = np.random.randn(100, 2)  # 100 data points with 2 features
    K = 3
    means, labels = kmean(data, K, max_iter=100, epsilon=0.01)
    """

    M , N = data.shape      # M = # of samples, N = feature dimension

    # Step 1: Initialize centroids by randomly choosing K data points
    means = init_centroids(data, K)

    # Step 2: Initialize the label array (to store cluster index of each point)
    labels= np.zeros(M, dtype=np.int64)

    # While loop varialbes initilization
    iter = 1
    # initiliaze the mean difference to run the while loop at least once
    mean_diff = 2 * epislon
    #Stopping criteria is either we run -- times or the change in all mean is smaller than an epislon
    while iter < max_iter and not np.all(mean_diff < epislon):
        
        # Assignment Step:
        # For each data point, compute distance to all centroids
        # Assign the point to the nearest cluster
        for row in range(0, M) :
            feature_vector = data[row]
            # Get index of closest centroid (cluster label)
            labels[row] = np.argmin(L2Distance(feature_vector, means, axis = 1))

        # Store current centroids before updating
        old_means = np.copy(means)

        # Update Step:
        # For each cluster, recalculate the mean of the points assigned to it
        for i in range(K) :
            # Get indices of all data points assigned to cluster i
            indices = np.where(labels == i)
            if len(indices[0]) > 0:
                # Compute mean of all points in cluster i
                means[i] = np.mean(data[indices], axis=0)
            else:
                # If a cluster gets no points, reinitialize its centroid randomly
                means[i] = data[np.random.choice(M)]

        # Compute how much each centroid moved since the last update
        mean_diff = L2Distance(means, old_means, axis=1)

        # Move to the next iteration
        iter += 1

     # Return the final centroids and the cluster assignments
    return labels , means

