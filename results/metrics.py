import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage.morphology import binary_erosion as imerode

def DSC(im1, im2):
    """
    dice coefficient 2nt/na + nb.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def as_logical(mask):
    return np.array(mask).astype(dtype=np.bool)


def surface_distance(target, estimated, spacing=[1, 1, 3]):
    a = as_logical(target)
    b = as_logical(estimated)
    #a_bound = np.stack(np.where(np.logical_and(a, np.logical_not(imerode(a)))), axis=1) * spacing
    #b_bound = np.stack(np.where(np.logical_and(b, np.logical_not(imerode(b)))), axis=1) * spacing
    a_bound = np.stack(np.where(a), axis=1) * spacing
    b_bound = np.stack(np.where(b), axis=1) * spacing
    nbrs_a = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(a_bound) if a_bound.size > 0 else None
    nbrs_b = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(b_bound) if b_bound.size > 0 else None
    distances_a, _ = nbrs_a.kneighbors(b_bound) if nbrs_a and b_bound.size > 0 else ([np.inf], None)
    distances_b, _ = nbrs_b.kneighbors(a_bound) if nbrs_b and a_bound.size > 0 else ([np.inf], None)
    return [distances_a, distances_b]

def HD(target, estimated, spacing):
    distances = surface_distance(target, estimated, spacing)
    return np.max([np.mean(distances[0]), np.mean(distances[1])])

