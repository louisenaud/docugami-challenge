"""
File:        exp_clusteval.py
Created by:  Louise Naud
On:          6/15/23
At:          4:01 PM
For project: docugami-challenge
Description:
Usage:
"""
# Import library
from clusteval import clusteval


# Initialize
cl = clusteval()

# Generate random data
X, y = cl.import_example(data='blobs')

# Fit data X
results = cl.fit(X)

# Plot
cl.plot()
cl.plot_silhouette()
cl.scatter()
cl.dendrogram()

# Set parameters, as an example dbscan
ce = clusteval(cluster='dbscan')

# Fit to find optimal number of clusters using dbscan
results= ce.fit(X)
print(results)

# Make plot of the cluster evaluation
ce.plot()

# Make scatter plot. Note that the first two coordinates are used for plotting.
ce.scatter(X)

# results is a dict with various output statistics. One of them are the labels.
cluster_labels = results['labx']