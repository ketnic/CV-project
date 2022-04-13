from sklearn import cluster
from enum import Enum


class IClusterer:
    def get(self, similarity_threshold):
        pass


class DBSCAN(IClusterer):
    def get(self, similarity_threshold):
        return cluster.DBSCAN(eps=1-similarity_threshold, metric='precomputed', min_samples=1, n_jobs=20)


class AgglomerativeClustering(IClusterer):
    def get(self, similarity_threshold):
        return cluster.AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete', distance_threshold=1-similarity_threshold, compute_full_tree=True)


class Сlusterizers(Enum):
    dbscan = DBSCAN()
    agglomerative_clustering = AgglomerativeClustering()
