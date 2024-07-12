import numpy as np

def euclidean_distance(vec1, vec2):
    if vec1 is None or vec2 is None:
        raise ValueError("Vectors cannot be None.")
    return np.sqrt(np.sum((np.array(vec1) - np.array(vec2)) ** 2))

def manhattan_distance(vec1, vec2):
    if vec1 is None or vec2 is None:
        raise ValueError("Vectors cannot be None.")
    return np.sum(np.abs(np.array(vec1) - np.array(vec2)))

def chebyshev_distance(vec1, vec2):
    if vec1 is None or vec2 is None:
        raise ValueError("Vectors cannot be None.")
    return np.max(np.abs(np.array(vec1) - np.array(vec2)))

def canberra_distance(vec1, vec2):
    if vec1 is None or vec2 is None:
        raise ValueError("Vectors cannot be None.")
    return np.sum(np.abs(vec1 - vec2) / (np.abs(vec1) + np.abs(vec2) + 1e-10))