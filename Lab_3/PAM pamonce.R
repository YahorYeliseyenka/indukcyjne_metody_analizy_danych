#________________________________________________________________________________PAM METRIC
get_pam_pamonce_results <- function(clrange, dfWithL, dfWithoutL){
  result <- c()
  
  for (m in c(0,1,2,3,4,5)){
    
    Putrity <- c()
    DunnIndex <- c()
    DBiIndex <- c()
    Silhouette <- c()
    
    for (i in clrange){
      PAM <- pam(dfWithoutL, i, stand = FALSE, do.swap = FALSE, pamonce = m)
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
  }
  
  return(result)
}

#________________________________________________________________________________plot PAM METRIC
get_plot_pam_pamonce <- function(result, PARAMrange, DFname, FOLDERname){
  dir.create(file.path("D:/_Studia/L_Indukcyjne_metody_analizy_danych/Lab_3/Results", FOLDERname))
  setwd(file.path("D:/_Studia/L_Indukcyjne_metody_analizy_danych/Lab_3/Results", FOLDERname))
  
  png(filename="PURITY.png")
  
  plot(PARAMrange, result[[1]]@p, type="b", col="blue", lwd=2, pch=19, xlab="Number of clusters", ylab="Purity", ylim=range(0,1))
  lines(PARAMrange, result[[2]]@p, type="b", col="orange", lwd=2, pch=15)
  lines(PARAMrange, result[[3]]@p, type="b", col="red", lwd=2, pch=15)
  lines(PARAMrange, result[[4]]@p, type="b", col="green", lwd=2, pch=15)
  lines(PARAMrange, result[[5]]@p, type="b", col="yellow", lwd=2, pch=15)
  lines(PARAMrange, result[[6]]@p, type="b", col="brown", lwd=2, pch=15)
  legend("bottomright", bg="transparent",c('pamonce=0', 'pamonce=1', 'pamonce=2', 'pamonce=3', 'pamonce=4', 'pamonce=5'), lwd=c(2,2,2,2,2,2), col=c("blue","orange","red","green","yellow","brown"), pch=c(19,15,15,15,15,15))
  title(paste("PAM", "\n", "PURITY", "\n", DFname, sep=""))
  dev.off()
  
  png(filename="DUNN-Index.png")
  
  plot(PARAMrange, result[[1]]@d, type="b", col="blue", lwd=2, pch=19, xlab="Number of clusters", ylab="Dunn index", ylim=range(0,1))
  lines(PARAMrange, result[[2]]@d, type="b", col="orange", lwd=2, pch=15)
  lines(PARAMrange, result[[3]]@d, type="b", col="red", lwd=2, pch=15)
  lines(PARAMrange, result[[4]]@d, type="b", col="green", lwd=2, pch=15)
  lines(PARAMrange, result[[5]]@d, type="b", col="yellow", lwd=2, pch=15)
  lines(PARAMrange, result[[6]]@d, type="b", col="brown", lwd=2, pch=15)
  legend("bottomright", bg="transparent",c('pamonce=0', 'pamonce=1', 'pamonce=2', 'pamonce=3', 'pamonce=4', 'pamonce=5'), lwd=c(2,2,2,2,2,2), col=c("blue","orange","red","green","yellow","brown"), pch=c(19,15,15,15,15,15))
  title(paste("PAM", "\n", "DUNN INDEX", "\n", DFname, sep=""))
  dev.off()
  
  png(filename="DB Index.png")
  
  plot(PARAMrange, result[[1]]@db, type="b", col="blue", lwd=2, pch=19, xlab="Number of clusters", ylab="DB index", ylim=range(0,2))
  lines(PARAMrange, result[[2]]@db, type="b", col="orange", lwd=2, pch=15)
  lines(PARAMrange, result[[3]]@db, type="b", col="red", lwd=2, pch=15)
  lines(PARAMrange, result[[4]]@db, type="b", col="green", lwd=2, pch=15)
  lines(PARAMrange, result[[5]]@db, type="b", col="yellow", lwd=2, pch=15)
  lines(PARAMrange, result[[6]]@db, type="b", col="brown", lwd=2, pch=15)
  legend("bottomright", bg="transparent",c('pamonce=0', 'pamonce=1', 'pamonce=2', 'pamonce=3', 'pamonce=4', 'pamonce=5'), lwd=c(2,2,2,2,2,2), col=c("blue","orange","red","green","yellow","brown"), pch=c(19,15,15,15,15,15))
  title(paste("PAM", "\n", "DB INDEX", "\n", DFname, sep=""))
  dev.off()
  
  png(filename="SILHOUETTE Score.png")
  
  plot(PARAMrange, result[[1]]@sil, type="b", col="blue", lwd=2, pch=19, xlab="Number of clusters", ylab="Average Silhouette Scores", ylim=range(0,1))
  lines(PARAMrange, result[[2]]@sil, type="b", col="orange", lwd=2, pch=15)
  lines(PARAMrange, result[[3]]@sil, type="b", col="red", lwd=2, pch=15)
  lines(PARAMrange, result[[4]]@sil, type="b", col="green", lwd=2, pch=15)
  lines(PARAMrange, result[[5]]@sil, type="b", col="yellow", lwd=2, pch=15)
  lines(PARAMrange, result[[6]]@sil, type="b", col="brown", lwd=2, pch=15)
  legend("bottomright", bg="transparent",c('pamonce=0', 'pamonce=1', 'pamonce=2', 'pamonce=3', 'pamonce=4', 'pamonce=5'), lwd=c(2,2,2,2,2,2), col=c("blue","orange","red","green","yellow","brown"), pch=c(19,15,15,15,15,15))
  title(paste("PAM", "\n", "SILHOUETTE SCORE", "\n", DFname, sep=""))
  dev.off()
  
  result.df <-data.frame("NUM_of_clusters"=PARAMrange,
                         "PURITY_pamonce_0"=result[[1]]@p,
                         "PURITY_pamonce_1"=result[[2]]@p,
                         "PURITY_pamonce_2"=result[[3]]@p,
                         "PURITY_pamonce_3"=result[[4]]@p,
                         "PURITY_pamonce_4"=result[[5]]@p,
                         "PURITY_pamonce_5"=result[[6]]@p,
                         "DUNN_INDEX_pamonce_0"=result[[1]]@d,
                         "DUNN_INDEX_pamonce_1"=result[[2]]@d,
                         "DUNN_INDEX_pamonce_2"=result[[3]]@d,
                         "DUNN_INDEX_pamonce_3"=result[[4]]@d,
                         "DUNN_INDEX_pamonce_4"=result[[5]]@d,
                         "DUNN_INDEX_pamonce_5"=result[[6]]@d,
                         "DB_INDEX_pamonce_0"=result[[1]]@db,
                         "DB_INDEX_pamonce_1"=result[[2]]@db,
                         "DB_INDEX_pamonce_2"=result[[3]]@db,
                         "DB_INDEX_pamonce_3"=result[[4]]@db,
                         "DB_INDEX_pamonce_4"=result[[5]]@db,
                         "DB_INDEX_pamonce_5"=result[[6]]@db,
                         "AVG_SILHOUETTE_pamonce_0"=result[[1]]@sil,
                         "AVG_SILHOUETTE_pamonce_1"=result[[2]]@sil,
                         "AVG_SILHOUETTE_pamonce_2"=result[[3]]@sil,
                         "AVG_SILHOUETTE_pamonce_3"=result[[4]]@sil,
                         "AVG_SILHOUETTE_pamonce_4"=result[[5]]@sil,
                         "AVG_SILHOUETTE_pamonce_5"=result[[6]]@sil)
  
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
  step <- floor(size_df / 9)
  
  clrange <- c(2, 5, 10)
  
  for (i in 1:7){
    clrange <- c(clrange, max(clrange)+step) 
  }
  
  result.pam.method <- get_pam_pamonce_results(clrange, df, df.dataonly)
  result.pam.method.stand <- get_pam_pamonce_results(clrange, df.stand, df.stand.dataonly)
  
  get_plot_pam_pamonce(result.pam.method, clrange, "wine default", "PAM WINE pamonce")
  get_plot_pam_pamonce(result.pam.method.stand, clrange, "wine standardized", "PAM WINE STAND pamonce")
}

f_pam_metric()
