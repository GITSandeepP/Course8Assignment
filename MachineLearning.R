library(e1071)
library(caret)
library(dplyr)
library(randomForest)

# Set Main directory path
maindir <- "/Users/mycomputer/Documents/RProgramming"
# Set Working directory path
path <- paste(maindir, "/Course8Assignment", sep = "")

# Check if desired working directory path exist or not
# if exist then set working directory else create directory and set it accordingly
if (dir.exists(file.path(path))){
        setwd(file.path(path))
}else {
        dir.create(file.path(maindir, "Course8Assignment"))
        setwd(file.path(path))
}

# Download the dataset to work on
# training.fileurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
# download.file(training.fileurl, destfile = "pml-training.csv")
# 
# testing.fileurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
# download.file(testing.fileurl, destfile = "pml-testing.csv")

setwd(file.path(path))

training.data <- read.csv("pml-training.csv",na.strings=c("NA","#DIV/0!",""))
testing.data <- read.csv("pml-testing.csv",na.strings=c("NA","#DIV/0!",""))

dim(training.data)

dim(testing.data)

# str(training.data)

colswithallmiss <- sapply(training.data, function(x)all(any(is.na(x))))

clean.training.data <- training.data[,!colswithallmiss]

## apart from removing columns with all NULL values, we have a choice to remove non relevant prediction
## columns as well. However, if we see the result of nearZeroVar function, we cannot confirm
## if the columns we throw out are really not a predictor. Hence I choose to keep all of the columns
## accept X & user_name. X because it contains just row number value and user_name wont help much
## in predicting the outcome
nzv.training.data <- nearZeroVar(clean.training.data, saveMetrics = TRUE)

clean.training.data <- clean.training.data %>% select(-X, -user_name)

isTrainData <- createDataPartition(clean.training.data$classe, p = 0.75, list = FALSE)
cv.Train.Data <- clean.training.data[isTrainData, ]
cv.Test.Data <- clean.training.data[-isTrainData, ]

cv.Train.Data <- cv.Train.Data %>% select(-raw_timestamp_part_1, -raw_timestamp_part_2, -cvtd_timestamp, -new_window, -num_window)
cv.Test.Data <- cv.Test.Data %>% select(-raw_timestamp_part_1, -raw_timestamp_part_2, -cvtd_timestamp, -new_window, -num_window)

set.seed(233)

model1 <- randomForest(classe ~ ., data = cv.Train.Data, importance = TRUE)
model1

model2 <- randomForest(classe ~ ., data = cv.Train.Data, importance = TRUE, ntree = 500, mtry = 15)
model2

predictTrain <- predict(model1, cv.Train.Data, type = "class")
table(predictTrain, cv.Train.Data$classe)

predictTest <- predict(model1, cv.Test.Data, type = "class")
table(predictTest, cv.Test.Data$classe)

mean(predictTest == cv.Test.Data$classe)

varImpPlot(model1)

model_rpart <- train(classe ~ ., data = cv.Train.Data, method = "rpart")
pred_rpart <- predict(model_rpart, cv.Train.Data)
table(pred_rpart, cv.Train.Data$classe)
mean(pred_rpart == cv.Train.Data$classe)

pred_rpart_Test <- predict(model_rpart, cv.Test.Data)
table(pred_rpart_Test, cv.Test.Data$classe)
mean(pred_rpart_Test == cv.Test.Data$classe)

pred_rf_test <- predict(model1, testing.data)
pred_rf_test
