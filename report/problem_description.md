We are given an xml file listing papers on Covid-19 Research until March 25th 2020. 
The goal of this assignment is to:
1. create groups of similar papers, 
2. give either a title or group of tags to each group,
3. find the paper in each group that is the most representative.

There is no ground truth available for this dataset; we are hence going to use clustering, 
and measure the quality of a clustering with metrics that don't require ground truth:
1. Silhouette Coefficient; it can range between -1 and 1, 1 being a score for a 
highly dense clustering. A score around 0 indicates clusters are overlapping.
2. Calinski-Harabasz Index; it is the ratio of the sum of between-clusters 
dispersion and of within-cluster dispersion for all clusters. It measures if the 
clusters are dense and well separated.
3. Davies-Bouldin Index; it computes the average ‘similarity’ between clusters, 
where the similarity is a measure that compares the distance between clusters with
the size of the clusters themselves. It takes values in R+, and values closer to 0 
indicate a better partition.

It is worth noting that all of these scores tend to give better results for convex clusters.