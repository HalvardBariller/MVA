"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
        
    # Degree matrix
    degree_sequence = [G.degree(node) for node in G.nodes()]
    D_inv = diags([1/d for d in degree_sequence])
    # Adjacency matrix
    A = nx.adjacency_matrix(G)
    # Laplacian matrix
    L = eye(len(degree_sequence)) - D_inv @ A

    # Compute k smallest eigenvectors of L
    eigs_values, eigs_vectors = eigs(L, k=k, which='SR')
    U = np.real(eigs_vectors)
    print("U shape (should be (%i, %i)): %s" % (D_inv.shape[0], k, U.shape))
    
    # K-means
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(U)

    clustering = {}
    for i, node in enumerate(G.nodes()):
        clustering[node] = kmeans.labels_[i]

    return clustering





############## Task 7

G = nx.read_edgelist('/Users/halvardbariller/Desktop/M2 MVA/_SEMESTER 1/ALTEGRAD/TP/TP2 - Graph Mining/code/datasets/CA-HepTh.txt', 
                        comments='#', delimiter='\t')
largest_cc = max(nx.connected_components(G), key=len)
subG = G.subgraph(largest_cc)

clustering_50 = spectral_clustering(subG, 50)
# print(clustering_50)





############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    modularity = 0
    n_c = len(set(clustering.values()))
    m = G.number_of_edges()

    for value in set(clustering.values()):
        keys = [k for k, v in clustering.items() if v == value]
        subG = G.subgraph(keys)
        l_c = subG.number_of_edges()
        d_c = sum([G.degree(node) for node in subG.nodes()])
        modularity += l_c/m - (d_c/(2*m))**2

    return modularity



############## Task 9

# Compute modularity value of the clustering

print("Modularity of the largest CC after Spectral Clustering with k=50: %f" % modularity(subG, clustering_50))


def random_clustering(G, k):
    clustering = {}
    for node in G.nodes():
        clustering[node] = randint(0, k-1)
    return clustering

print("Modularity of the largest CC after Random Clustering with k=50: %f" % modularity(subG, random_clustering(subG, 50)))




