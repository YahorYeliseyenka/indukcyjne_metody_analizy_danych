remove (list = objects())

library(fossil)
library(clv)
library(cluster)
library(clValid)
library(clusterSim)

  #SHOW DATAFRAME INFO
  #str(dataset4cl)
  #summary(dataset4cl)
  #length(dataset[,(ncol(dataset))])

kc <- kmeans(iris4cl, 10)

  #aggregate(dataset4cl,by=list(kc$cluster),FUN=mean)
  #length(kc$cluster)
  #apply(table(dataset[,(ncol(dataset))], kc$cluster), 2, max)
  #kc$cluster

#PLOT
plot(dataset4cl, col = kc$cluster)
plot(dataset[,-(ncol(dataset))], col = dataset[,(ncol(dataset))])
clusplot(iris4cl, kc$cluster, color=TRUE, shade=TRUE, labels=iris$class_dataset, lines=0)

#######################################################################################

kl <- (nrow(dataset4cl)-1)*sum(apply(dataset4cl,2,var))
for (i in 2:15) kl[i] <- sum(kmeans(dataset4cl, centers=i)$withinss)
plot(1:15, kl, type="b", xlab="Число кластеров", ylab="Сумма квадратов расстояний внутри кластеров")

#######################################################################################

