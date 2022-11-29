#________________________________________________________________________________PAM METRIC
get_pam_medoids_results <- function(clrange, dfWithL, dfWithoutL, Medoids){
  result <- c()
  
  Putrity <- c()
  DunnIndex <- c()
  DBiIndex <- c()
  Silhouette <- c()
  
  for (i in clrange){
    PAM <- pam(dfWithoutL, i, stand = FALSE, do.swap = FALSE)
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
  result <- c(result, new("Eval", p=Putrity, d=DunnIndex, db=DBiIndex, sil=Silhouette))
  
  Putrity <- c()
  DunnIndex <- c()
  DBiIndex <- c()
  Silhouette <- c()
  
  z <- 1
  for (i in clrange){
    PAM <- pam(dfWithoutL, i, stand = FALSE, do.swap = FALSE, medoids=unlist(Medoids[z], use.names=FALSE))
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
    
    z <- z + 1
  }
  result <- c(result, new("Eval", p=Putrity, d=DunnIndex, db=DBiIndex, sil=Silhouette))
  
  return(result)
}

#________________________________________________________________________________plot PAM METRIC
get_plot_pam_medoids <- function(result, PARAMrange, DFname, FOLDERname, medoids){
  dir.create(file.path("D:/_Studia/L_Indukcyjne_metody_analizy_danych/Lab_3/Results", FOLDERname))
  setwd(file.path("D:/_Studia/L_Indukcyjne_metody_analizy_danych/Lab_3/Results", FOLDERname))
  
  png(filename="PURITY.png")
  
  plot(PARAMrange, result[[1]]@p, type="b", col="blue", lwd=2, pch=15, xlab="Number of clusters", ylab="Purity", ylim=range(0,1))
  lines(PARAMrange, result[[2]]@p, type="b", col="orange", lwd=2, pch=15)
  legend("bottomright", bg="transparent",c('medoids default', 'medoids random'), lwd=c(2,2,2,2), col=c("blue","orange"), pch=c(15,15))
  title(paste("PAM", "\n", "PURITY", "\n", DFname, sep=""))
  dev.off()
  
  png(filename="DUNN-Index.png")
  
  plot(PARAMrange, result[[1]]@d, type="b", col="blue", lwd=2, pch=15, xlab="Number of clusters", ylab="Dunn index", ylim=range(0,1))
  lines(PARAMrange, result[[2]]@d, type="b", col="orange", lwd=2, pch=15)
  legend("topleft", bg="transparent",c('medoids default', 'medoids random'), lwd=c(2,2,2,2), col=c("blue","orange"), pch=c(15,15))
  title(paste("PAM", "\n", "DUNN INDEX", "\n", DFname, sep=""))
  dev.off()
  
  png(filename="DB Index.png")
  
  plot(PARAMrange, result[[1]]@db, type="b", col="blue", lwd=2, pch=15, xlab="Number of clusters", ylab="DB index", ylim=range(0,3))
  lines(PARAMrange, result[[2]]@db, type="b", col="orange", lwd=2, pch=15)
  legend("bottomright", bg="transparent",c('medoids default', 'medoids random'), lwd=c(2,2,2,2), col=c("blue","orange"), pch=c(15,15))
  title(paste("PAM", "\n", "DB INDEX", "\n", DFname, sep=""))
  dev.off()
  
  png(filename="SILHOUETTE Score.png")
  
  plot(PARAMrange, result[[1]]@sil, type="b", col="blue", lwd=2, pch=15, xlab="Number of clusters", ylab="Average Silhouette Scores", ylim=range(0,1))
  lines(PARAMrange, result[[2]]@sil, type="b", col="orange", lwd=2, pch=15)
  legend("topleft", bg="transparent",c('medoids default', 'medoids random'), lwd=c(2,2,2,2), col=c("blue","orange"), pch=c(15,15))
  title(paste("PAM", "\n", "SILHOUETTE SCORE", "\n", DFname, sep=""))
  dev.off()
  
  medoids_str <- c()
  
  for ( i in 1:length(medoids)){
    medoids_str <- c(medoids_str, paste(unlist(medoids[[i]]), collapse=' '))
  }
  
  result.df <-data.frame("MEDOIDS"=medoids_str,
                         "NUM_of_clusters"=PARAMrange,
                         "PURITY_medoids_default"=result[[1]]@p,
                         "PURITY_medoids_random"=result[[2]]@p,
                         "DUNN_INDEX_medoids_default"=result[[1]]@d,
                         "DUNN_INDEX_medoids_random"=result[[2]]@d,
                         "DB_INDEX_medoids_default"=result[[1]]@db,
                         "DB_INDEX_medoids_random"=result[[2]]@db,
                         "AVG_SILHOUETTE_medoids_default"=result[[1]]@sil,
                         "AVG_SILHOUETTE_medoids_random"=result[[2]]@sil)
  
  write.table(result.df, file="result.csv", quote=F,sep=",",row.names=F)
}


f_pam_metric <- function(){
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
  
  size_df <- nrow(df.dataonly) - 10
  step <- floor(size_df / 3)
  
  clrange <- c(2, 5, 10)
  
  for (i in 1:2){
    clrange <- c(clrange, max(clrange)+step) 
  }
  
  medoids <- list()
  z <- 1
  for (i in clrange){
    medoids[[z]] <- sample(c(1:177), size=i, replace=F) 
    z <- z + 1
  }
  
  result.pam.method <- get_pam_medoids_results(clrange, df, df.dataonly, medoids)
  result.pam.method.stand <- get_pam_medoids_results(clrange, df.stand, df.stand.dataonly, medoids)
  
  get_plot_pam_medoids(result.pam.method, clrange, "wine default", "PAM WINE medoids", medoids)
  get_plot_pam_medoids(result.pam.method.stand, clrange, "wine standardized", "PAM WINE STAND medoids", medoids)
}

f_pam_metric()
