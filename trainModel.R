# practical machine learning -- course project

library(caret)
library(randomForest)

courseDir <- 'C:\\Users\\Scott\\My Documents\\coursera\\data_science_specialization\\practical_machine_learning\\'
projectDir <- paste0(courseDir, 'project')
setwd(projectDir)

# read training and test data

dfTrain <- read.csv('pml-training.csv', na.strings = c("", "NA"))
dfTest  <- read.csv('pml-testing.csv', na.strings = c("", "NA"))

# remove features with bogus data and first seven non-informative features

goodFeatures <- colSums(is.na(dfTrain)) == 0
dfTrain <- dfTrain[, goodFeatures]
dfTrain <- dfTrain[, -(1:7)]

# create cross-validation set for model tuning (70/30 split)

trainIdx <- createDataPartition(y = dfTrain$classe, p = 0.7, list = FALSE)
dfTrainR <- dfTrain[trainIdx, ]  # "reduced" training set
dfValid  <- dfTrain[-trainIdx, ] # cross-validation set

# 1: fit classification tree model on reduced training data

treeFit <- rpart(classe ~ ., data = dfTrainR, method = "class")

# 2: make classification tree predictions on validation data

treePred <- predict(treeFit, dfValid, type = "class")

# 3: get classification tree accuracy from confusion matrix

treeAcc <- confusionMatrix(treePred, dfValid$classe)$overall[1]

# 1: fit random forest model on reduced training data

rfFit <- randomForest(classe ~ ., data = dfTrainR, ntree = 1000, importance = TRUE)

# 2: make random forest predictions on validation data

rfPred <- predict(rfFit, dfValid, type = "class")

# 3: get random forest accuracy from confusion matrix

rfAcc <- confusionMatrix(rfPred, dfValid$classe)$overall[1]

# generate random forest model predictions on test set

rfTestPred <- predict(rfFit, dfTest, type = "class")

# EOF