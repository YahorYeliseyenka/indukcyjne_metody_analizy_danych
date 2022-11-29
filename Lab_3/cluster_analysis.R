#PURITY
ClusterPurity <- function(classes, clusters) {
  return(sum(apply(table(classes, clusters), 2, max)) / length(clusters))
}

#PURITY
cp <- ClusterPurity(iris[,(ncol(iris))], kc$cluster)
cp

# compute intercluster distances and intracluster diameters
cls.scatt <- cls.scatt.data(iris4cl, kc$cluster)

#DBI
intraclust = c("complete","average","centroid")
interclust = c("single", "complete", "average","centroid", "aveToCent", "hausdorff")

davies <- clv.Davies.Bouldin(cls.scatt, intraclust, interclust)
davies

#DUNN index
Dist <- dist(iris4cl, method="euclidean")
dunn(Dist, kc$cluster)


dunn <- clv.Dunn(cls.scatt, intraclust, interclust)
dunn

#Silhouette plot
ss <- silhouette(kc$cluster, dist(iris4cl))
windows()
plot(ss)

###################################################################################################

#RAND MEASURE
rm <- rand.index(dataset[,(ncol(dataset))], kc$cluster)
rm