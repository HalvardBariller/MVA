"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

G = nx.read_edgelist('/Users/halvardbariller/Desktop/M2 MVA/_SEMESTER 1/ALTEGRAD/TP/TP2 - Graph Mining/code/datasets/CA-HepTh.txt', 
                        comments='#', delimiter='\t')
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

############## Task 2

print("Number of connected components:", nx.number_connected_components(G))
largest_cc = max(nx.connected_components(G), key=len)
print("Size of largest connected component:", len(largest_cc))

#Largest_cc is only the indices of the nodes, need to extract subgraph to manipulate it
subG = G.subgraph(largest_cc)
print("Number of nodes in largest connected component:", subG.number_of_nodes())
print("Number of edges in largest connected component:", subG.number_of_edges())
print("Largest component contains", round(subG.number_of_nodes()/G.number_of_nodes()*100, 2), "% of nodes")
print("Largest component contains", round(subG.number_of_edges()/G.number_of_edges()*100, 2), "% of edges")


############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

print("Maximum degree in the graph:", max(degree_sequence))
print("Minimum degree in the graph:", min(degree_sequence))
print("Median degree in the graph:", np.median(degree_sequence))
print("Average degree in the graph:", np.round(np.mean(degree_sequence),3))

############## Task 4

degree_freq = nx.degree_histogram(G)
print(degree_freq)

plt.figure(figsize=(10, 6))
plt.plot(degree_freq, 'b-')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree distribution')
plt.show()

plt.figure(figsize=(10, 6))
plt.loglog(degree_freq, 'b-')
plt.xlabel('logDegree')
plt.ylabel('logFrequency')
plt.title('Degree log-distribution')
plt.show()


############## Task 5

glob_cluster_coeff = nx.transitivity(G)
# Indication of density of triangles in the graph

print("Global clustering coefficient:", glob_cluster_coeff)