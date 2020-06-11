library(tidyverse)
library(caret)
library(data.table)
library(readr)
library(rpart)
library(ggplot2)
library(gam)
library(dslabs)


##download data to be used
dl <- tempfile()
download.file("https://raw.githubusercontent.com/AndyMukherjee/BioMed_feature_of_orthopedic_patients/master/datasets_2374_3987_column_2C_weka.csv",dl)
Bio <- read_csv(dl)

## View the data : data Visualization
head(Bio)
summary(Bio)
nrow(Bio) # 310 rows 
ncol(Bio) # 7 columns

##Class is Abnormal and Normal. Goal of this project is to build a Machine learning alogorithm
##to determine the class based on the data in the other columns

##Check the relationship of Class with pelvic_incidence


Bio %>% ggplot(aes(`pelvic_incidence`, fill = class))+geom_density(alpha = 0.3)

##The overlaping densities tell us that pelvic_incidence donot independently determine the class

##Check the relationship of Class with pelvic_tilt numeric

Bio %>% ggplot(aes(`pelvic_tilt numeric`, fill = class))+geom_density(alpha = 0.3)
##The overlaping densities tell us that pelvic_tilt numeric donot independently determine the class

##Check the relationship of Class with lumbar_lordosis_angle

Bio %>% ggplot(aes(`lumbar_lordosis_angle`, fill = class))+geom_density(alpha = 0.3)

##The overlaping densities tell us that lumbar_lordosis_angle donot independently determine the class



##Check the relationship of Class with sacral_slope

Bio %>% ggplot(aes(`sacral_slope`, fill = class))+geom_density(alpha = 0.3)

##The overlaping densities tell us that sacral_slope donot independently determine the class


##Check the relationship of Class with pelvic_radius


Bio %>% ggplot(aes(`pelvic_radius`, fill = class))+geom_density(alpha = 0.3)

##The overlaping densities tell us that pelvic_radius donot independently determine the class


##Check the relationship of Class with degree_spondylolisthesis

Bio %>% ggplot(aes(`degree_spondylolisthesis`, fill = class))+geom_density(alpha = 0.3)

##The overlaping densities tell us that degree_spondylolisthesis donot independently determine the class

###################################################
#Training the models:

#Dividing the dataset of "Bio" into train and test sets

test_index <- createDataPartition(y = Bio$class, times = 1, p = 0.2, list = FALSE )
train <- Bio[-test_index,] #248 rows
test <- Bio[test_index,] #62 rows

##creating x and y( subsets are x)
train_subset <- subset(train, select = -class)
test_subset <- subset(test, select = -class)
train_y <- train$class
test_y <- test$class


##Using Logistic Regression to fit a model, to find the combined effect of all the columns
##together on the class, and find the accuracy of this.


  fit <- train(train_subset, train_y, method = "glm")
  fit$results
## Accuracy of the logistic regression model is : 0.850605 

##Using the LDA and QDA methods:

##LDA
fit_lda <- train(class ~ ., method = "lda", data = train)
fit_lda$results["Accuracy"] # 0.8408057

##QDA

# fit_qda <- train(class ~ ., method = "qda", data = train)
# fit_qda$results#["Accuracy"] #no results
## need to comment this out, else the code gives error

##Using the KNN method:
fit <- train(train_subset, train_y, method = "knn",tuneGrid = data.frame(k = seq(1,50,1)))
ggplot(fit)

##Looking at the graph since there is a steady decrease in accuracy from 20, 
##I will narrow the k value from 1 to 20, to have a clearer view

fit_knn <- train(train_subset, train_y, method = "knn",tuneGrid = data.frame(k = seq(1,20,1)))
ggplot(fit_knn)
fit_knn$results

## so K value of 5 has the heighest accuracy : 0.8333217 
## but this is still less than the glm method



##Rpart
fit_rpart <- train(train_subset, train_y, method = "rpart",
             tuneGrid = data.frame(cp = seq(0, 10, 1)))

ggplot(fit_rpart)
confusionMatrix(fit_rpart)
## Accuracy is : 0.7997

## Ploting the final Model
plot(fit_rpart$finalModel, margin = 0.1)
text(fit_rpart$finalModel)

##Using random forest method : 
fit_rf <- train(train_subset, train_y, method = "rf",
                nodesize = 1,
                tuneGrid = data.frame(mtry = seq(0, 10, 2)))
ggplot(fit_rf)
confusionMatrix(fit_rf)  #Accuracy=  0.8391
fit_rf$bestTune #mtry = 2

##GamLoess Model
 #fit_gam <- train(class ~ . , method = "gamLoess", data = train)
 #confusionMatrix(fit_gam) ## no results 
## need to comment this out, else the code gives error

##Multinorm Method:
fit_multi <- train(class ~ . , method = "multinom", data = train)
confusionMatrix(fit_multi) ## Accuracy = 0.847

##SVM Linear model:
 fit_svm <- train(class ~ . , method = "svmLinear", data = train)
 confusionMatrix(fit_svm) ## Accuracy = 0.8408


## Joining all the models together to get the accuracy of each model to compare :

model <- c("glm", "lda", "knn", "rf", "svmLinear",  "multinom")

fits <- lapply(model, function(model){
  #print(model)
  train(class ~ . , method = model, data = train)
})


pred <- sapply(fits, function(x){
  predict(x, newdata = test)
})

## get the Accuracy for each method:

colMeans(pred == test$class)

##0.8709677 0.8548387 0.8548387 0.8387097 0.8709677 0.8870968
##According to the above result the best accuracy is from multinom model

##Ensemble Method: 
En <- rowMeans(pred == "Normal")
y_hat <- ifelse(En > 0.5, "Normal", "Abnormal")
mean(y_hat == test$class)
##Accuracy: 0.8709677

## So the glm and SVM linear models perform the same as Ensemble, and the multinom model performs better

