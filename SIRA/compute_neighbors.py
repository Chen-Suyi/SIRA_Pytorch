import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_neighbors(points: np.ndarray,
                      n_neighbors=30,
                      radius=0.0625) -> np.ndarray:
    invalid_index = points.shape[0]

    knn = NearestNeighbors(n_neighbors=n_neighbors,
                           radius=radius,
                           algorithm="auto").fit(points)
    distances, indices = knn.kneighbors(points)

    # set the index to invalid_index if the point out of radius
    indices[distances > radius] = invalid_index

    return indices
