#install.packages("C50")

library(C50)
library(ROCR)
library(caret)
library(xtable)
library(cap)
library(e1071)
library(formattable)

cross_validation <- function(data, stratified, foldNum, noGlobalPruning){
  countTemp<-0
  
  treeSizeM<-0
  accuracyM<-0
  precisionM<-0
  recallM<-0
  fscoreM<-0
  
  folds <- list()
  
  if (stratified){
    folds <- createFolds(data[,ncol(data)], k = foldNum, list = F)
  }
  else{
    folds <- cut(seq(1,nrow(data)),breaks=foldNum,labels=FALSE)
  }
  
  for (i in 1:foldNum){
    for (j in 1:foldNum){
      if (j != i){
        countTemp<-countTemp + 1
        
        trainData <- data[folds == i,]
        ctrl = C5.0Control(noGlobalPruning=noGlobalPruning)
        model <- C5.0(trainData[,-ncol(trainData)], trainData[,ncol(trainData)], control = ctrl)
        #model <- C5.0(trainData[,-ncol(trainData)], trainData[,ncol(trainData)], minCases=minCases)
        
        testData<-data[folds == j,]
        prediction<-predict(model, testData)
        
        #CONFUSION MATRIX
        #xtab = table(prediction, data$class_dataset)
        #cat("\nConfusion matrix:\n")
        #print(xtab)
        
        cm <- confusionMatrix(testData[,ncol(testData)], prediction, mode = "everything")
        byClass <- cm$byClass
        
        if (class(byClass)=="numeric"){
          accuracyM<-accuracyM + byClass[11]
          precisionM<-precisionM + byClass[5]
          recallM<-recallM + byClass[6]
          fscoreM<-fscoreM + byClass[7]
        }
        else{
          accuracyM<-accuracyM + mean(byClass[,11], na.rm=TRUE)
          precisionM<-precisionM + mean(byClass[,5], na.rm=TRUE)
          recallM<-recallM + mean(byClass[,6], na.rm=TRUE)
          fscoreM<-fscoreM + mean(byClass[,7], na.rm=TRUE)
        }
        
        treeSizeM<-treeSizeM + mean(model$size, na.rm=TRUE)
      }
    }
  }
  
  treeSizeM<-signif(treeSizeM / countTemp, digits = 2)
  accuracyM<-signif(accuracyM / countTemp, digits = 2)
  precisionM<-signif(precisionM / countTemp, digits = 2)
  recallM<-signif(recallM / countTemp, digits = 2)
  fscoreM<-signif(fscoreM / countTemp, digits = 2)
  
  return(new("Eval", size=treeSizeM, acc=accuracyM, prec=precisionM, rec=recallM, fsc=fscoreM))
  
#  cat(paste("Tree size:\t", treeSizeM, "\n",sep=" "))
#  cat(paste("Accuracy:\t", accuracyM, "\n",sep=" "))
#  cat(paste("Precision:\t", precisionM, "\n",sep=" "))
#  cat(paste("Recall:\t\t", recallM, "\n",sep=" "))
#  cat(paste("F-measure:\t", fscoreM, "\n",sep=" "))
}

save_results <- function(dataN, stratified){
  tit <- paste(toupper(dataN), "\n", "STRATIFIED=", stratified, sep="")
  img_name <- paste(dataN, "_stratified=", stratified, sep="")
  
  #READ DATA
  dataSet <- read.csv(paste(dataN, ".csv", sep = ""))
  
  #SET FACTOR
  dataSet[,ncol(dataSet)] <- as.factor(dataSet[,ncol(dataSet)])
  
  #SHOW THE DATAFRAME INFO
  #str(dataSet)
  #summary(dataSet)
  
  #SHUFFLE DATAFRAME
  set.seed(pi)
  data<-dataSet[sample(nrow(dataSet)),]
  
  require(C50)
  
  bins <- c(2,50)
  
  tsz <- c()
  acc <- c()
  prec <- c()
  rec <- c()
  fsc <- c()
  
  count = 0
  
  for (i in bins){
    count<-count + 1
    
    #ctrl = C5.0Control(bands = i)
    #c50_model <- C5.0(dataSet[,-ncol(dataSet)], dataSet[,ncol(dataSet)], rules = T, control = ctrl)
    #c50_model <- C5.0(data[,-ncol(data)], data[,ncol(data)])
    
    ctrl = C5.0Control(minCases=i)
    model <- C5.0(data[,-ncol(data)], data[,ncol(data)], control = ctrl)
    
    plot(model)
    dev.copy(jpeg, paste(img_name, "_Tree", i, ".jpg", sep = ""))
    dev.off()
    
    result <- cross_validation(data, stratified, 2, i)
    
    tsz[count] <- result@size
    acc[count] <- result@acc
    prec[count] <- result@prec
    rec[count] <- result@rec
    fsc[count] <- result@fsc
  }
  
  plot(bins, acc, type="b", col="blue", lwd=2, pch=19, xlab="bands", ylab="value", ylim=range(0,1))
  lines(bins, rec, type="b", col="orange", lwd=2, pch=15)
  lines(bins, prec, type="b", col="green", lwd=2, pch=16)
  lines(bins, fsc, type="b", col="red", lwd=2, pch=17)
  legend("bottomleft",c("ACC","REC","PREC","FSC"), lwd=c(2,2,2,2), col=c("blue","orange","green","red"), pch=c(19,15,16,17))
  title(tit)
  
  dev.copy(jpeg, paste(img_name, ".jpg", sep=""))
  dev.off()
  
  plot(bins, tsz, type="b", col="black", lwd=2, pch=19, xlab="bands", ylab="three_depth", ylim=range(0,50))
  legend("topleft",c("Głębokość drzewa"), lwd=c(2), col=c("black"), pch=c(19))
  title(tit)
  
  dev.copy(jpeg, paste(img_name, "_TreeSize", ".jpg", sep = ""))
  dev.off()
  
  write.csv(data.frame(acc,rec,prec,fsc,tsz), file = paste(img_name, ".csv", sep=""))
}

################################################################    M A I N   ###################################################################

names <- c("pima", "wine", "glass")

for (i in names){
  save_results(i, TRUE) 
}

save_results("pima", TRUE) 

#dataSet <- read.csv(paste(names[3], ".csv", sep = ""))
#dataSet[,ncol(dataSet)] <- as.factor(dataSet[,ncol(dataSet)])
#set.seed(pi)
#data<-dataSet[sample(nrow(dataSet)),]
#summary(data)

##################################################################################################################################################

#CREATES CLASS PRECISION
setClass(Class="Eval",
         representation(
           size="numeric",
           acc="numeric",
           prec="numeric",
           rec="numeric",
           fsc="numeric"
         )
)

niceBBox <- function(xydata) {
  mins = -min(xydata$x, xydata$y)
  maxs = max(xydata$x, xydata$y)
  x = ceiling(max(mins, maxs))
  return(c(-x, x))
}

#c50_model <- C5.0(dataSet[,-ncol(dataSet)], dataSet[,ncol(dataSet)], rules = TRUE)

#ctrl = C5.0Control(bands = 10)
#c50_model <- C5.0(dataSet[,-ncol(dataSet)], dataSet[,ncol(dataSet)], rules = T, control = ctrl)

c50_model <- C5.0(dataSet[,-ncol(dataSet)], dataSet[,ncol(dataSet)])
c50_model
summary(c50_model)
plot(c50_model)

c50_predict <- predict(c50_model, dataSet)
c50_predict

#Compare
table(dataSet[,ncol(dataSet)], c50_predict)

#C5.0 Train Improve performace
c50_model <- C5.0(dataSet[,-ncol(dataSet)], dataSet[,ncol(dataSet)], trials = 10)

summary(c50_model)
plot(c50_model, trial = 10)

evaluation <- function(model, data) {
  prediction = predict(model, data)
  
  #  xtab = table(prediction, data$class_dataset)
  #  cat("\nConfusion matrix:\n")
  #  print(xtab)
  
  cm <- confusionMatrix(data[,ncol(data)], prediction, mode = "everything")
  byClass <- cm$byClass
  
  accuracy = signif(mean(byClass[,11]), digits = 2)
  precision = signif(mean(byClass[,5]), digits = 2)
  recall = signif(mean(byClass[,6]), digits = 2)
  fscore = signif(mean(byClass[,7]), digits = 2)
  meanTreeSize = signif((model$size), digits = 2)
  
  return(new("Eval", size=meanTreeSize, acc=accuracy, prec=precision, rec=recall, fsc=fscore))
}

result <- evaluation(c50_model, dataSet, "class")
result@acc

#CROSS VALIDATION STRATIFIED
folds <- list()
folds <- createFolds(dataSetR[,ncol(dataSetR)], k = 10, list = F)
folds
dataSetR[folds == 1,]

#CROSS VALIDATION NONSTRATIFIED
folds <- cut(seq(1,nrow(dataSetR)),breaks=10,labels=FALSE)
folds
dataSetR[folds == 1,]


plot(dataSetR)

dataSet[,ncol(dataSet)]




save_results <- function(dataN, stratified, trials){
  tit <- paste(toupper(dataN), "\n", "STRATIFIED=", stratified, sep="")
  img_name <- paste(dataN, "_stratified=", stratified, sep="")
  
  #READ DATA
  dataSet <- read.csv(paste(dataN, ".csv", sep = ""))
  
  #SET FACTOR
  dataSet[,ncol(dataSet)] <- as.factor(dataSet[,ncol(dataSet)])
  
  #SHOW THE DATAFRAME INFO
  #str(dataSet)
  #summary(dataSet)
  
  #SHUFFLE DATAFRAME
  set.seed(pi)
  data<-dataSet[sample(nrow(dataSet)),]
  
  require(C50)
  
  bins <- c(2,3,5,10)
  
  tsz <- c()
  acc <- c()
  prec <- c()
  rec <- c()
  fsc <- c()
  
  count = 0
  
  for (i in bins){
    count = count + 1
    result <- cross_validation(data, stratified, i, trials)
    
    tsz[count] <- result@size
    acc[count] <- result@acc
    prec[count] <- result@prec
    rec[count] <- result@rec
    fsc[count] <- result@fsc
  }
  
  plot(bins, acc, type="b", col="blue", lwd=2, pch=19, xlab="fold_num", ylab="pred", ylim=range(0,1))
  lines(bins, rec, type="b", col="orange", lwd=2, pch=15)
  lines(bins, prec, type="b", col="green", lwd=2, pch=16)
  lines(bins, fsc, type="b", col="red", lwd=2, pch=17)
  legend("bottomleft",c("ACC","REC","PREC","FSC"), lwd=c(2,2,2,2), col=c("blue","orange","green","red"), pch=c(19,15,16,17))
  title(tit)
  
  dev.copy(jpeg, paste(img_name, ".jpg", sep=""))
  dev.off()
  
  plot(bins, tsz, type="b", col="black", lwd=2, pch=19, xlab="fold_num", ylab="pred", ylim=range(0,50))
  legend("topleft",c("Głębokość drzewa"), lwd=c(2), col=c("black"), pch=c(19))
  title(tit)
  
  dev.copy(jpeg, paste(img_name, "_TreeSize", ".jpg", sep = ""))
  dev.off()
}