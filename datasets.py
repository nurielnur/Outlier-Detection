import numpy as np
from sklearn.datasets import make_moons, make_blobs


def make_datasets(n_samples, outliers_fraction):
    """
    n_samples : total samples in the datasets including outliers
    outlier_fraction : fraction of the outliers in the dataset
    
    Returns 4 two-dimensional datasets with different cluster centers and standart deviations.  
    """
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers
    
    df1 = make_blobs(centers=[ [5,5] ], cluster_std = 0.5, n_samples=n_inliers, n_features=2)[0]
    df1 = np.concatenate([df1, np.random.uniform(low=1, high=10, size = (n_outliers, 2))], axis=0)

    df2 = make_blobs(centers=[[3, 4], [8, 4]], cluster_std=[1, 0.3], n_samples=n_inliers)[0]
    df2 = np.concatenate([df2, np.random.uniform(low=1, high=8, size = (n_outliers, 2))], axis=0)

    df3 = make_blobs(centers=[[2, 4], [7, 4]], cluster_std=[0.5, 0.5], n_samples=n_inliers)[0]
    df3 = np.concatenate([df3, np.random.uniform(low=1, high=8, size = (n_outliers, 2))], axis=0)

    df4 = make_moons(n_samples=n_inliers, noise=.05, random_state=0)[0] + np.array([2,2])
    df4 = np.concatenate([df4, np.random.uniform(low=1, high=4, size = (n_outliers, 2))], axis=0)
    
    datasets = [df1, df2, df3, df4]
    return datasets