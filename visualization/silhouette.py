#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_size_inches(14, 7)


X1 = np.load("tsne_features.npy")
y1 = np.load("tsne_y.npy")

n_clusters = 2
    
ax1.set_xlim([-0.2, 0.3])
ax1.set_ylim([0, len(X1) + (n_clusters + 1) * 10])
    
    
ax3.set_xlim([-0.2, 0.3])
ax3.set_ylim([0, len(X1) + (n_clusters + 1) * 10])


cluster_labels = y1
silhouette_avg = silhouette_score(X1, cluster_labels)
print(
    "For n_clusters =",
    n_clusters,
    "The average silhouette_score is :",
    silhouette_avg,
)
# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X1, cluster_labels)


y_lower = 10
for i in range(n_clusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    if i==1:
        # color = cm.nipy_spectral(float(i) / n_clusters)
        color = 'red'
    else:
        color = 'green'
    
    ax3.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.5,
    )
    ax1.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )
    # Label the silhouette plots with their cluster numbers at the middle
    if i==0:
        ll = 'Benign'
    else:
        ll = 'Vulnerable'
    ax1.text(-0.28, y_lower + 0.5 * size_cluster_i, ll)

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples
    
ax1.set_title("DeepVD embeddings")
# ax1.set_xlabel("The silhouette coefficient values")
# ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
# ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])



X2 = np.load("yi_tsne_features.npy")
y2 = np.load("yi_tsne_y.npy")


    
ax2.set_xlim([-0.2, 0.3])
ax2.set_ylim([0, len(X2) + (n_clusters + 1) * 10])

cluster_labels = y2
silhouette_avg = silhouette_score(X2, cluster_labels)
print(
    "For n_clusters =",
    n_clusters,
    "The average silhouette_score is :",
    silhouette_avg,
)
# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X2, cluster_labels)
y_lower = 10
for i in range(n_clusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    if i==1:
        # color = cm.nipy_spectral(float(i) / n_clusters)
        color = 'red'
    else:
        color = 'green'
    
    ax3.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.5,
    )
    ax2.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )
    # Label the silhouette plots with their cluster numbers at the middle
    if i==0:
        ax2.text(-0.28, y_lower + 0.5 * size_cluster_i, 'Benign')
        ax3.text(-0.28, y_lower + 0.5 * size_cluster_i, 'Benign')
    else:
        ax2.text(-0.28, y_lower + 0.5 * size_cluster_i, 'Vulnerable')
        ax3.text(-0.28, y_lower + 0.5 * size_cluster_i, 'Vulnerable')

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples
    
ax2.set_title("IVDetect embeddings")
# ax2.set_xlabel("The silhouette coefficient values")
# ax2.set_ylabel("Cluster label")
# ax2.axvline(x=silhouette_avg, color="red", linestyle="--")
ax2.set_yticks([])  # Clear the yaxis labels / ticks
ax2.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])


ax3.set_title("Both")
# ax3.set_xlabel("The silhouette coefficient values")
# ax3.set_ylabel("Cluster label")
# ax2.axvline(x=silhouette_avg, color="red", linestyle="--")
ax3.set_yticks([])  # Clear the yaxis labels / ticks
ax3.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])

plt.savefig('silhouette.pdf')
plt.show()


# In[ ]:




