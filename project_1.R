#setwd("C:/Users/Debarshi/Desktop/Fall 2017/Intro to Data Mining")
#install.packages('caret', dependencies = TRUE)
#install.packages('dplyr', dependencies = TRUE)
#install.packages('rweka', dependencies = TRUE)
library(caret)
library(RWeka)
require(dplyr)
#----------------------------------------------------------------------

data <- read.csv("project1.csv")

dataset1 <- data.frame(data$Overall.life.expectancy.at.birth,
    data$Male.life.expectancy.at.birth,
    data$Female.life.expectancy.at.birth,
    data$Continent)

# ---------------------------------------------------------------------
for(i in 1:5) {
  if(exists("final_trainset1")) {
    rm(final_trainset1)
  }
  if(exists("final_testset1")) {
    rm(final_testset1)
  }

if(file.exists(paste("Training",i,".csv",sep = ""))) {
  final_trainset1 <- read.csv(paste("Training",i,".csv",sep = ""))
} else {
  final_trainset1 <- dataset1[sample(nrow(dataset1),round(0.8*nrow(dataset1)),replace = TRUE),]
  write.csv(final_trainset1, file = (paste("Training",i,".csv",sep = "")))
}

if(file.exists(paste("Test",i,".csv",sep = ""))) {
  final_testset1 <- read.csv(paste("Test",i,".csv",sep = ""))
} else {
  final_testset1 <- anti_join(dataset1,final_trainset1)
  write.csv(final_testset1, file = (paste("Test",i,".csv",sep = "")))
}

#----------------------------------------------------------------------
trnctrl_trainset1 <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3333)
#----------------------------------------------------------------------
knnfit_trainset1 <- train(data.Continent ~., data = final_trainset1, method = "knn",
                          trControl=trnctrl_trainset1,
                          preProcess = c("center", "scale"),
                          tuneLength = 10)

knn_pred_set1 <- predict(knnfit_trainset1, newdata = final_testset1)
print(plot(knnfit_trainset1))
print(paste("Printing knn confusion matrix for dataset",i))
print(confusionMatrix(knn_pred_set1, final_testset1$data.Continent))
#----------------------------------------------------------------------

svmLinear_trainset1 <- train(data.Continent ~., data = final_trainset1, method = "svmRadial",
                             trControl=trnctrl_trainset1,
                             preProcess = c("center", "scale"),
                             tuneLength = 10)

svmLinear_pred_set1 <- predict(svmLinear_trainset1, newdata = final_testset1)
print(plot(svmLinear_trainset1))
print(paste("Printing SVM confusion matrix for dataset",i))
print(confusionMatrix(svmLinear_pred_set1, final_testset1$data.Continent))
#----------------------------------------------------------------------

c45_trainset1 <- train(data.Continent ~., data = final_trainset1, method = "J48",
                       trControl=trnctrl_trainset1,
                       preProcess = c("center", "scale"),
                       tuneLength = 10)

c45_pred_set1 <- predict(c45_trainset1, newdata = final_testset1)
print(plot(c45_trainset1))
print(paste("Printing C4.5 confusion matrix for dataset",i))
print(confusionMatrix(c45_pred_set1, final_testset1$data.Continent))
#----------------------------------------------------------------------

Ripper_trainset1 <- train(data.Continent ~., data = final_trainset1, method = "JRip",
                          trControl=trnctrl_trainset1,
                          preProcess = c("center", "scale"),
                          tuneLength = 10)

ripper_pred_set1 <- predict(Ripper_trainset1, newdata = final_testset1)
print(plot(Ripper_trainset1))
print(paste("Printing RIPPER confusion matrix for dataset",i))
print(confusionMatrix(ripper_pred_set1, final_testset1$data.Continent))

}