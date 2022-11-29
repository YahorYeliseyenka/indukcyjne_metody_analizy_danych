get_df_from_file <- function(fileName){
  #READ DATA
  data <- read.csv(paste(fileName, ".csv", sep = ""))
  
  #SHUFFLE DATAFRAME
  set.seed(pi)
  data<-data[sample(nrow(data)),]
  
  return(data)
}

file_names <- c("iris", "pima", "wine", "glass", "facebook")

iris <- get_df_from_file(file_names[1])
iris4cl <- iris[,-(ncol(iris))]

pima <- get_df_from_file(file_names[2])
pima4cl <- pima[,-(ncol(pima))]

wine <- get_df_from_file(file_names[3])
wine4cl <- wine[,-(ncol(wine))]

glass <- get_df_from_file(file_names[4])
glass4cl <- glass[,-(ncol(glass))]

facebook <- get_df_from_file(file_names[5])
facebook4cl <- facebook[,3:(ncol(facebook))]
