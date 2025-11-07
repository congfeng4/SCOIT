import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist, jaccard
from communities.algorithms import louvain_method
from sklearn.cluster import SpectralClustering
from igraph import Graph
import leidenalg
import os
from umap import UMAP
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# construct KNN graph
def knn_adj_matrix(X, k=20):
    samples_size, features_size = X.shape
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    adj_matrix = nbrs.kneighbors_graph(X).toarray()

    return adj_matrix


# construct SNN graph
def snn_adj_matrix(X, k=20):
    samples_size, features_size = X.shape
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    knn_adj_matrix = nbrs.kneighbors_graph(X).toarray()
    adj_matrix = np.dot(knn_adj_matrix, knn_adj_matrix.T) * knn_adj_matrix * knn_adj_matrix.T
    row, col = np.diag_indices_from(adj_matrix)
    adj_matrix[row, col] = 0

    return adj_matrix


# construct jaccard SNN graph
def jsnn_adj_matrix(X, k=20, prune=1 / 15):
    samples_size, features_size = X.shape
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    knn_adj_matrix = nbrs.kneighbors_graph(X).toarray()
    snn_matrix = np.dot(knn_adj_matrix, knn_adj_matrix.T)
    adj_matrix = snn_matrix / (2 * k - snn_matrix)
    row, col = np.diag_indices_from(adj_matrix)
    adj_matrix[row, col] = 0
    adj_matrix[adj_matrix < prune] = 0

    return adj_matrix


# Louvain algorithm
def RunLouvain(adj_matrix, k=None):
    communities, _ = louvain_method(adj_matrix, n=k)
    labels = np.zeros(adj_matrix.shape[0]).astype(int)
    l = -1
    for each in communities:
        l += 1
        for index in each:
            labels[index] = l

    return list(labels)


# Spectral clustering
def RunSpectral(adj_matrix, k=5):
    clustering = SpectralClustering(n_clusters=k, random_state=0).fit(adj_matrix)

    return list(clustering.labels_)


# Leiden algorithm
def RunLeiden(adj_matrix):
    G = Graph.Weighted_Adjacency(adj_matrix)
    part = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
    labels = np.zeros(adj_matrix.shape[0]).astype(int)
    for i in range(len(part)):
        labels[part[i]] = i

    return list(labels)


from sklearn.metrics import normalized_mutual_info_score # NMI
from sklearn.metrics import adjusted_rand_score # ARI
from sklearn.metrics import fowlkes_mallows_score # FMI
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


def align_cluster(cluster_ids, y):
    kc = len(np.unique(cluster_ids))
    ky = len(np.unique(y))
    # ---------- 3. build cost matrix ----------
    contingency = np.zeros((kc, ky), dtype=int)
    for c, gt in zip(cluster_ids, y):
        contingency[c, gt] += 1

    # cost = number of mismatches if we map cluster c -> class j
    cost_matrix = contingency.max() - contingency        # or: -contingency

    # ---------- 4. Hungarian ----------
    row_ind, col_ind = linear_sum_assignment(cost_matrix)   # same API as old linear_assignment

    # ---------- 5. remap ----------
    # build a dictionary  old_label -> new_label
    mapping = dict(zip(row_ind, col_ind))
    y_pred_aligned = np.vectorize(mapping.get)(cluster_ids)
    return y_pred_aligned


def compute_clustering_metrics(y_true, y_pred):
    y_pred_aligned = align_cluster(y_pred, y_true)
    return dict(
        NMI = normalized_mutual_info_score(y_true, y_pred_aligned),
        ARI = adjusted_rand_score(y_true, y_pred_aligned),
        FMI = fowlkes_mallows_score(y_true, y_pred_aligned),
    )


def cluster_evaluate(cell_embeddings, cell_labels, use_umap=False):
    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(cell_labels)
    km = KMeans(n_clusters=len(label_encoder.classes_))
    km.fit(cell_embeddings if not use_umap else UMAP().fit_transform(cell_embeddings))
    y_pred = km.labels_
    return compute_clustering_metrics(y_true, y_pred)


def save_array(name, array):
    try:
        np.savetxt(name + '.csv', array, delimiter=',')
    except:
        np.save(name + ".npz", array)


def save_scoit_embeddings(sc_model, dataname: str, predict_data=None):
    subdir = f'./embeddings/scoit/{dataname}'
    os.makedirs(subdir, exist_ok=True)
    os.chdir(subdir)

    save_array("cell_embeddings", sc_model.C)
    save_array("gene_embeddings", sc_model.G)
    save_array("omics_embeddings", sc_model.O)
    try:
        save_array("local_gene_embeddings", sc_model.OG)
        save_array("local_cell_embeddings", sc_model.OC)
    except:
        pass

    if predict_data is not None:
        save_array("predict_data_expression", predict_data[0])
        save_array("predict_data_protein", predict_data[1])


def umap_visualize(features, cell_stage=None, method='UMAP', palette=None, s=40, eval_use_umap=False):

    # plt.figure(figsize=(6,4))
    vec = eval(method)(n_components=2).fit_transform(features)
    sns.scatterplot(x=vec[:, 0], y=vec[:, 1], hue=cell_stage,
                    hue_order=sorted(set(cell_stage)) if cell_stage is not None else None,
                    palette=palette,
                    s=s, linewidth=0, alpha=0.5)
    plt.legend()
    plt.title(method)

    # ---- 计算每个类的中心并标注 ----
    if cell_stage is not None:
        df = pd.DataFrame({'UMAP_1': vec[:, 0], 'UMAP_2': vec[:, 1], 'label': cell_stage})
        centroids = df.groupby('label')[['UMAP_1', 'UMAP_2']].mean()

        for label, row in centroids.iterrows():
            plt.text(row['UMAP_1'], row['UMAP_2'], str(label),
                     fontsize=9, ha='center', va='center',)

    plt.tight_layout()

    from scoit.cell_analysis import cluster_evaluate

    return cluster_evaluate(features, cell_stage, use_umap=eval_use_umap)
