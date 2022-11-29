#________________________________________________________________________________Dte PAM param results
get_pam_results_param <- function(dfWithL, dfWithoutL, Method=c(), Medoids=list()){
  Putrity <- c()
  DunnIndex <- c()
  DBiIndex <- c()
  Silhouette <- c()
  
  if (is.not.null(Method)){
    for (i in Method){
      PAM <- pam(dfWithoutL, 5, metric = i, stand = FALSE)
      Dist <- dist(dfWithoutL, method=i)
      
      Putrity <- c(Putrity, ClusterPurity(dfWithL[,(ncol(dfWithL))], PAM$cluster))
      
      DunnIndex <- c(DunnIndex, dunn(Dist, PAM$cluster))
      
      DBiIndex <- c(DBiIndex, index.DB(dfWithoutL, PAM$cluster, d=Dist, centrotypes="medoids")$DB)
      
      Silhouette <- c(Silhouette, mean(silhouette(PAM$cluster, Dist)[,3]))
    }
  }
  else{
    for (i in Medoids){
      PAM <- pam(dfWithoutL, 5, metric = "euclidean", medoids=i, stand = FALSE)
      Dist <- dist(dfWithoutL, method="euclidean")
      
      Putrity <- c(Putrity, ClusterPurity(dfWithL[,(ncol(dfWithL))], PAM$cluster))
      
      DunnIndex <- c(DunnIndex, dunn(Dist, PAM$cluster))
      
      DBiIndex <- c(DBiIndex, index.DB(dfWithoutL, PAM$cluster, d=Dist, centrotypes="medoids")$DB)
      
      Silhouette <- c(Silhouette, mean(silhouette(PAM$cluster, Dist)[,3]))  
    }
  }
  
  return(new("Eval", p=Putrity, d=DunnIndex, db=DBiIndex, sil=Silhouette))
}

#________________________________________________________________________________Get kmeans param results
get_kmeans_results_param <- function(dfWithL, dfWithoutL, Iter.max=c(), Nstart=c(), Algorithm=c()){
  Putrity <- c()
  DunnIndex <- c()
  DBiIndex <- c()
  Silhouette <- c()
  
  if (is.not.null(Iter.max)){
    for (i in Iter.max){
      
      Kmeans <- kmeans(dfWithoutL, 5, iter.max=i)
      Dist <- dist(dfWithoutL, method="euclidean")
      
      Putrity <- c(Putrity, ClusterPurity(dfWithL[,(ncol(dfWithL))], Kmeans$cluster))
      
      DunnIndex <- c(DunnIndex, dunn(Dist, Kmeans$cluster))
      
      DBiIndex <- c(DBiIndex, index.DB(dfWithoutL, Kmeans$cluster, centrotypes="centroids")$DB)
      
      Silhouette <- c(Silhouette, mean(silhouette(Kmeans$cluster, Dist)[,3]))  
    }
  }
  else if (is.not.null(Nstart)){
    for (i in Nstart){
      
      Kmeans <- kmeans(dfWithoutL, 5, nstart=i)
      Dist <- dist(dfWithoutL, method="euclidean")
      
      Putrity <- c(Putrity, ClusterPurity(dfWithL[,(ncol(dfWithL))], Kmeans$cluster))
      
      DunnIndex <- c(DunnIndex, dunn(Dist, Kmeans$cluster))
      
      DBiIndex <- c(DBiIndex, index.DB(dfWithoutL, Kmeans$cluster, centrotypes="centroids")$DB)
      
      Silhouette <- c(Silhouette, mean(silhouette(Kmeans$cluster, Dist)[,3]))  
    }
  }
  else{
    for (i in Algorithm){
      
      Kmeans <- kmeans(dfWithoutL, 5, algorithm=i)
      Dist <- dist(dfWithoutL, method="euclidean")
      
      Putrity <- c(Putrity, ClusterPurity(dfWithL[,(ncol(dfWithL))], Kmeans$cluster))
      
      DunnIndex <- c(DunnIndex, dunn(Dist, Kmeans$cluster))
      
      DBiIndex <- c(DBiIndex, index.DB(dfWithoutL, Kmeans$cluster, centrotypes="centroids")$DB)
      
      Silhouette <- c(Silhouette, mean(silhouette(Kmeans$cluster, Dist)[,3]))  
    }
  }
  
  return(new("Eval", p=Putrity, d=DunnIndex, db=DBiIndex, sil=Silhouette))
}



#________________________________________________________________________________KMEANS PARAMETRS
KMparams <- function(){
  setwd(file.path("D:/_Studia/L_Indukcyjne_metody_analizy_danych", "Lab_3"))
  
  dfNAME <- "wine"
  
  df = get_df_from_file(dfNAME)
  #add factor
  df[,ncol(df)] <- as.factor(df[,ncol(df)])
  #summary(df)
  df.dataonly <- df[,-ncol(df)]
  
  df.stand <- as.data.frame(scale(df[,-ncol(df)]))
  df.stand$lable <- df[,ncol(df)]
  df.stand.dataonly <- df.stand[,-ncol(df.stand)]
  
  result.kmeans.itermax <- get_kmeans_results_param(df.stand, df.stand.dataonly, Iter.max=c(5,10,15,20,50,100))
  result.kmeans.numseeds <- get_kmeans_results_param(df.stand, df.stand.dataonly, Nstart=c(2,5,10,15,20,50,100))
  result.kmeans.algorythm <- get_kmeans_results_param(df.stand, df.stand.dataonly, Algorithm=c("Hartigan-Wong", "Lloyd", "MacQueen"))
  
  get_plots_param(result.kmeans.itermax, c(5,10,15,20,50,100), "wine", "KM itermax", "iter.max", "K-Means")
  get_plots_param(result.kmeans.numseeds, c(2,5,10,15,20,50,100), "wine", "KM nstart", "nstart", "K-Means")
  get_plots_param(result.kmeans.algorythm, c(1,2,3), "wine", "KM algorithm", "algorithm", "K-Means")
}

#--------------------------------------------------------------------------------
#____________________________________________________M__A__I__N__________________
#--------------------------------------------------------------------------------

KMparams()

#____________________________________________________P__A__M_____________________

qwe <- function(){
  setwd(file.path("D:/_Studia/L_Indukcyjne_metody_analizy_danych", "Lab_3"))
  
  dfNAME <- "wine"
  
  df = get_df_from_file(dfNAME)
  #add factor
  df[,ncol(df)] <- as.factor(df[,ncol(df)])
  #summary(df)
  df.dataonly <- df[,-ncol(df)]
  
  df.stand <- as.data.frame(scale(df[,-ncol(df)]))
  df.stand$lable <- df[,ncol(df)]
  df.stand.dataonly <- df.stand[,-ncol(df.stand)]
  
  result.pam.method <- get_pam_results_param(df.stand, df.stand.dataonly, Method=c("euclidean", "manhattan"))
  result.pam.medoids <- get_pam_results_param(df.stand, df.stand.dataonly, Medoids=list(c(1,5,10,20,40), c(10,40,60,80,100), c(121,13,20,2,73), c(170,150,100,50,10)))  
  
  get_plots_param(result.pam.method, c(1,2), "wine", "PAM method", "method", "PAM")
  get_plots_param(result.pam.medoids, c(1,2,3,4), "wine", "PAM medoids", "medoids", "PAM")
}

qwe()