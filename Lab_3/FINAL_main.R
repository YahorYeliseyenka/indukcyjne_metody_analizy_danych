remove (list = objects())

library(cluster)
library(fossil)
library(clv)
library(clValid)
library(clusterSim)

#__________________________________________________________________________________FINAL RESULTS
get_final_results <- function(dfNAME, classes=TRUE){
  setwd(file.path("D:/_Studia/L_Indukcyjne_metody_analizy_danych", "Lab_3"))
  
  #________________________________________________________________________________get data
  df = get_df_from_file(dfNAME)
  if (classes){
    #add factor
    df[,ncol(df)] <- as.factor(df[,ncol(df)])
    #summary(df)
    df.dataonly <- df[,-ncol(df)]
  }
  else{
    df.dataonly <- df[1:700,]
    df <- NULL
  }
  
  #________________________________________________________________________________normalize DATA
  if (classes){
    df.norm <- as.data.frame(lapply(df[,-ncol(df)], normalize))
    df.norm$lable <- df[,ncol(df)]
    df.norm.dataonly <- df.norm[,-ncol(df.norm)]
    #summary(df.norm)
    #head(df.norm)
  }
  else{
    df.norm <- as.data.frame(lapply(df.dataonly, normalize))
    df.norm.dataonly <- df.norm[,]
    df.norm <- NULL
  }
  
  #________________________________________________________________________________STANDARDIZE data
  if (classes){
    df.stand <- as.data.frame(scale(df[,-ncol(df)]))
    df.stand$lable <- df[,ncol(df)]
    df.stand.dataonly <- df.stand[,-ncol(df.stand)]
    #summary(df.stand.dataonly)
    #head(df.stand)
  }
  else{
    df.stand <- as.data.frame(scale(df.dataonly))
    df.stand.dataonly <- df.stand[,]
    df.stand <- NULL
  }
  
  size_df <- nrow(df.dataonly) - 10
  step <- floor(size_df / 9)
  
  clrange <- c(2, 5, 10)
  
  for (i in 1:7){
    clrange <- c(clrange, max(clrange)+step) 
  }
  
  result.kmeans <- get_kmeans_results(clrange, df, df.dataonly)
  result.kmeans.norm <- get_kmeans_results(clrange, df.norm, df.norm.dataonly)
  result.kmeans.stand <- get_kmeans_results(clrange, df.stand, df.stand.dataonly)
  
  result.pam <- get_pam_results(clrange, df, df.dataonly)
  result.pam.norm <- get_pam_results(clrange, df.norm, df.norm.dataonly)
  result.pam.stand <- get_pam_results(clrange, df.stand, df.stand.dataonly)
  
  get_plots(result.kmeans, result.pam, clrange, paste(dfNAME, "default", sep = " "), paste("KM vs PAM ", dfNAME, "default", sep = " "))
  get_plots(result.kmeans.norm, result.pam.norm, clrange, paste(dfNAME, "normalized", sep = " "), paste("KM vs PAM ", dfNAME, "normalized", sep = " "))
  get_plots(result.kmeans.stand, result.pam.stand, clrange, paste(dfNAME, "stadardized", sep = " "), paste("KM vs PAM ", dfNAME, "stadardized", sep = " "))
}

#________________________________________________________________________________GET PLOTS
get_plots <- function(result.kmeans, result.pam, clrange, dfNAME, folderNAME){
  dir.create(file.path("D:/_Studia/L_Indukcyjne_metody_analizy_danych/Lab_3/Results", folderNAME))
  setwd(file.path("D:/_Studia/L_Indukcyjne_metody_analizy_danych/Lab_3/Results", folderNAME))
  
  png(filename="PURITY.png")
  
  plot(clrange, result.kmeans@p, type="b", col="blue", lwd=2, pch=19, xlab="Number of clusters", ylab="Purity", ylim=range(0,1))
  lines(clrange, result.pam@p, type="b", col="orange", lwd=2, pch=15)
  legend("bottomright", bg="transparent",c("kmeans","pam"), lwd=c(2,2), col=c("blue","orange"), pch=c(19,15))
  title(paste("K-means VS PAM", "\n", "PURITY", "\n", dfNAME, sep=""))
  dev.off()
  
  png(filename="DUNN-Index.png")
  
  plot(clrange, result.kmeans@d, type="b", col="blue", lwd=2, pch=19, xlab="Number of clusters", ylab="Dunn index", ylim=range(0,1))
  lines(clrange, result.pam@d, type="b", col="orange", lwd=2, pch=15)
  legend("topleft", bg="transparent",c("kmeans","pam"), lwd=c(2,2), col=c("blue","orange"), pch=c(19,15))
  title(paste("K-means VS PAM", "\n", "DUNN INDEX", "\n", dfNAME, sep=""))
  dev.off()
  
  png(filename="DB Index.png")
  
  plot(clrange, result.kmeans@db, type="b", col="blue", lwd=2, pch=19, xlab="Number of clusters", ylab="DB index", ylim=range(0,3))
  lines(clrange, result.pam@db, type="b", col="orange", lwd=2, pch=15)
  legend("bottomleft", bg="transparent",c("kmeans","pam"), lwd=c(2,2), col=c("blue","orange"), pch=c(19,15))
  title(paste("K-means VS PAM", "\n", "DB INDEX", "\n", dfNAME, sep=""))
  dev.off()
  
  png(filename="SILHOUETTE Score.png")
  
  plot(clrange, result.kmeans@sil, type="b", col="blue", lwd=2, pch=19, xlab="Number of clusters", ylab="Average Silhouette Scores", ylim=range(0,1))
  lines(clrange, result.pam@sil, type="b", col="orange", lwd=2, pch=15)
  legend("topleft", bg="transparent",c("kmeans","pam"), lwd=c(2,2), col=c("blue","orange"), pch=c(19,15)) 
  title(paste("K-means VS PAM", "\n", "SILHOUETTE SCORE", "\n", dfNAME, sep=""))
  dev.off()
  
  result.df <-data.frame("NUM_of_clusters"=clrange,
                         "km_PURITY"=result.kmeans@p, 
                         "pam_PURITY"=result.pam@p,
                         "km_DUNN_INDEX"=result.kmeans@d,
                         "pam_DUNN_INDEX"=result.pam@d,
                         "km_DB_INDEX"=result.kmeans@db,
                         "pam_DB_INDEX"=result.pam@db,
                         "km_AVG_SILHOUETTE"=result.kmeans@sil,
                         "pam_AVG_SILHOUETTE"=result.pam@sil)
  
  write.table(result.df, file="result.csv", quote=F,sep=",",row.names=F)
}

#________________________________________________________________________________K-means 4 DF
get_kmeans_results <- function(clrange, dfWithL, dfWithoutL){
  Putrity <- c()
  DunnIndex <- c()
  DBiIndex <- c()
  Silhouette <- c()
  
  for (i in clrange){
    Kmeans <- kmeans(dfWithoutL, i, nstart=100, iter.max = 100)
    Dist <- dist(dfWithoutL, method="euclidean")
    
    if (is.not.null(dfWithL)){
      Putrity <- c(Putrity, ClusterPurity(dfWithL[,(ncol(dfWithL))], Kmeans$cluster))
    }
    else{
      Putrity <- c(Putrity, -1)
    }
    
    DunnIndex <- c(DunnIndex, dunn(Dist, Kmeans$cluster))
    
    DBiIndex <- c(DBiIndex, index.DB(dfWithoutL, Kmeans$cluster, centrotypes="centroids")$DB)
    
    Silhouette <- c(Silhouette, mean(silhouette(Kmeans$cluster, Dist)[,3]))
  }
  
  return(new("Eval", p=Putrity, d=DunnIndex, db=DBiIndex, sil=Silhouette))
}

#________________________________________________________________________________K-means 4 DF
get_pam_results <- function(clrange, dfWithL, dfWithoutL){
  Putrity <- c()
  DunnIndex <- c()
  DBiIndex <- c()
  Silhouette <- c()
  
  for (i in clrange){
    PAM <- pam(dfWithoutL, i, metric = "euclidean", stand = FALSE)
    Dist <- dist(dfWithoutL, method="euclidean")
    
    if (is.not.null(dfWithL)){
      Putrity <- c(Putrity, ClusterPurity(dfWithL[,(ncol(dfWithL))], PAM$cluster))
    }
    else{
      Putrity <- c(Putrity, -1)
    }
    
    DunnIndex <- c(DunnIndex, dunn(Dist, PAM$cluster))
    
    DBiIndex <- c(DBiIndex, index.DB(dfWithoutL, PAM$cluster, d=Dist, centrotypes="medoids")$DB)
    
    Silhouette <- c(Silhouette, mean(silhouette(PAM$cluster, Dist)[,3]))
  }
  
  return(new("Eval", p=Putrity, d=DunnIndex, db=DBiIndex, sil=Silhouette))
}

#________________________________________________________________________________CREATES CLASS EVAL
setClass(Class="Eval",
         representation(
           p="numeric",
           d="numeric",
           db="numeric",
           sil="numeric"
         )
)

#________________________________________________________________________________IS NOT NULL
is.not.null <- function(x) !is.null(x)

#________________________________________________________________________________PURITY
ClusterPurity <- function(classes, clusters) {
  return(sum(apply(table(classes, clusters), 2, max)) / length(clusters))
}

#________________________________________________________________________________MIN-MAX NORMALIZATION
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

#________________________________________________________________________________GET DATA
get_df_from_file <- function(fileName){
  #READ DATA
  data <- read.csv(paste(fileName, ".csv", sep = ""))
  
  #SHUFFLE DATAFRAME
  set.seed(pi)
  data<-data[sample(nrow(data)),]
  
  if(fileName == "facebook"){
    data <- data[3:11]
  }
  return(data)
}

#file_names <- c("iris", "pima", "wine", "glass", "facebook")

#--------------------------------------------------------------------------------
#____________________________________________________M__A__I__N__________________
#--------------------------------------------------------------------------------

#________________________________________________________________________________PIMA NORM vs STAND
get_final_results("pima")

#________________________________________________________________________________WINE NORM vs STAND
get_final_results("wine")

#________________________________________________________________________________GLASS NORM vs STAND
get_final_results("glass")

#________________________________________________________________________________FACEBOOK NORM vs STAND
get_final_results("facebook", FALSE)
