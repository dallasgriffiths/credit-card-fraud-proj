---
  title: "Credit Card Fraud"
author: "Dallas Griffiths"
date: "3/17/2020"
output:
  html_document:
  code_folding: hide
---

library(tensorflow)
library(caret)
library(dplyr)
tf$constant("Hellow Tensorflow")
library(keras)

orig.csv <- read.csv("~/R/creditcard.csv")

orig.df <- data.frame(orig.csv)

scaledvars <- scale(orig.df[, 1:28])

scaled_df <- data.frame(scaledvars)
refined_scale_df1 <- cbind(scaled_df, Amount = orig.df$Amount)
final_df <- cbind(refined_scale_df1, Class = orig.df$Class)

#Training and test
set.seed(67)
training_indices = createDataPartition(final_df$Class, p=0.7, list=FALSE)

trainPC = final_df[training_indices,]
testPC = final_df[-training_indices,]
#trainPC$Class <- as.factor(trainPC$Class)
#testPC$Class <- as.factor(testPC$Class)
sapply(trainPC,class)
sapply(testPC,class)

#To get started, train a neural network on the training ndata with 1 hidden layer with 5 nodes and
#linear activation functions. As this is a classification task, you must use sigmoid activation functions in
#the output layer. As in the class code, use binary crossentropy loss - this attempts to create the best
#predicted probabilities. This shouldn’t take longer than 30 seconds or so.

PCmodel <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(29)) %>% 
  layer_dense(units = 5, activation = "linear") %>% 
  layer_dense(units = 1, activation = "sigmoid") 
summary(PCmodel)


PCmodel %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy")




#trainPC$Class <- as.numeric(as.character(trainPC$Class))
PCmodel %>% 
  fit(
    x = as.matrix(trainPC[,c(1:29)]), y = trainPC[,"Class"],
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

sapply(trainPC,class)

#Eval pred on test set
test_predictions <- PCmodel %>% 
  predict(as.matrix(data.frame(testPC[,c(1:29)])))
test_predictions

#trainPC$Class <- as.numeric(trainPC$Class)
#testPC$Class <- as.numeric(as.character(testPC$Class))

sapply(trainPC,class)

#Evaluate predictions
library(pROC) #For building ROC curve
SuccessROC = roc(testPC$Class,test_predictions)
plot.roc(SuccessROC) #Plots ROC curve
SuccessROC$auc  #Area Under the Curve

#Confusion Matrix
library(e1071)
delta = 0.50
predicted_fraud = ifelse(test_predictions >= delta,1,0) #Class prediction
confusionMatrix(as.factor(predicted_fraud),as.factor(testPC$Class)) #Create confusion matrix


#Question 2

#    Given this model, with one hidden layer, 5 nodes in that layer and that layer is utilizing a linear activation function, given the training data, on the 5th epoch, this neural network gives a val_accuracy of .9991 and an accuracy of .9989.val_accuracy is just how accurate the model is on unseen data and accuracy is how accurate the model is on the training data. But your question is asking for accuracy in general so I will just take the accuracy given by the confusion matrix. This value is .9989 which is obviously the .9989 value given on the fifth epoch. Or it predicts accurately 99.89% of the time. Given a cutoff of .5, this model predicted that there were 112 fraudulent transactions compared to the actual amount of transactions which is 21 is the test set


####################################################################################################################################
#question 3
sapply(trainPC,class)
################
#Down-sampling
table(trainPC$Class)##199019 valid transactions/345 fraudulent transactions

#Perform down-sampling using built-in Caret function
set.seed(36)
down_train = downSample(x = trainPC[,-ncol(trainPC)], #-ncol says not to include response variable
                        y = as.factor(trainPC[,ncol(trainPC)]))
down_train$Class <- as.integer(as.character(down_train$Class))
#Examine prevalence of classes after down-sampling:
summary(down_train$Class)##345 valid and fraudulent transactions
sapply(testPC,class)



#Neural Net on down_train set
#This model will give optimal metrics of neural net on down_train set and the steps i took to get to optimal metrics of neural net for down_train set.
#down_train$Class <- as.integer(down_train$Class)
sapply(down_train,class)
PCmodel <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(15)) %>% 
  layer_dense(units = 11, activation = "linear") %>%
  layer_dense(units = 11, activation = "linear") %>%
  layer_dense(units = 10, activation = "linear") %>%
  layer_dense(units = 1, activation = "sigmoid") 

summary(PCmodel)

PCmodel %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy")
sapply(down_train,class)



sapply(down_train,class)

PCmodel %>% 
  fit(
    x = as.matrix(down_train[,c(18,17,16,14,12,11,10,9,7,6,5,4,3,2,1)]), y = down_train[,"Class"],
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

#testPC$Class = as.factor(as.character(testPC$Class))

#Build confusion matrix to evaluate predictions
#Eval pred on test set
test_predictions <- PCmodel %>% 
  predict(as.matrix(data.frame(testPC[,c(18,17,16,14,12,11,10,9,7,6,5,4,3,2,1)])))


#Evaluate predictions
roc(testPC$Class,test_predictions)$auc  #Area Under the Curve

#Confusion Matrix
delta = 0.50
predicted_success2 = ifelse(test_predictions >= delta,1,0) #Class prediction
confusionMatrix(as.factor(predicted_success2),as.factor(testPC$Class)) #Create confusion matrix

#####################################################################################################################################
################
#Up-sampling

#Perform down-sampling using built-in Caret function
set.seed(3)
up_train = upSample(x = trainPC[,-ncol(trainPC)], y = as.factor(trainPC[,ncol(trainPC)]))

up_train$Class <- as.integer(as.character(up_train$Class))
#Examine prevalence of classes after down-sampling:

#up_train$Class = as.numeric(as.character(up_train$Class))

PCmodel <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(15)) %>% 
  layer_dense(units = 11, activation = "linear") %>%
  layer_dense(units = 11, activation = "linear") %>%
  layer_dense(units = 10, activation = "linear") %>% 
  layer_dense(units = 1, activation = "sigmoid") 

summary(PCmodel)

PCmodel %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy")
```

```{r}
PCmodel %>% 
  fit(
    x = as.matrix(up_train[,c(18,17,16,14,12,11,10,9,7,6,5,4,3,2,1)]), y = up_train[,"Class"],
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

#testPC$Class = as.factor(as.character(testPC$Class))

#Build confusion matrix to evaluate predictions
#Eval pred on test set
test_predictions <- PCmodel %>% 
  predict(as.matrix(data.frame(testPC[,c(18,17,16,14,12,11,10,9,7,6,5,4,3,2,1)])))


#Evaluate predictions
roc(testPC$Class,test_predictions)$auc  #Area Under the Curve

#Confusion Matrix
delta = 0.50
predicted_success2 = ifelse(test_predictions >= delta,1,0) #Class prediction
confusionMatrix(as.factor(predicted_success2),as.factor(testPC$Class)) #Create confusion matrix



#####################################################################################################################################
#####################
#SMOTE sampling
library(DMwR)

trainPC$Class = as.factor(trainPC$Class)

#Perform ROSE using specific library
set.seed(39393)
smote_train = SMOTE(Class ~ ., 
                    data  = trainPC)  

table(smote_train$Class)

smote_train[,30] <- as.numeric(as.character(smote_train[,30]))


PCmodel <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(29)) %>% 
  layer_dense(units = 9, activation = "linear") %>%
  layer_dense(units = 9, activation = "linear") %>%
  layer_dense(units = 9, activation = "linear") %>%
  layer_dense(units = 1, activation = "sigmoid") 

summary(PCmodel)

PCmodel %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy")

PCmodel %>% 
  fit(
    x = as.matrix(smote_train[,c(1:29)]), y = smote_train[,"Class"],
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

#testPC$Class = as.factor(as.character(testPC$Class))

#Build confusion matrix to evaluate predictions
#Eval pred on test set
test_predictions <- PCmodel %>% 
  predict(as.matrix(data.frame(testPC[,c(1:29)])))


#Evaluate predictions
roc(testPC$Class,test_predictions)$auc  #Area Under the Curve

#Confusion Matrix
delta = 0.50
predicted_success2 = ifelse(test_predictions >= delta,1,0) #Class prediction
confusionMatrix(as.factor(predicted_success2),as.factor(testPC$Class)) #Create confusion matrix



######################################################################################################################################

library(ROSE)
set.seed(59595)
rose_train = ROSE(Class ~ ., data = trainPC)$data
table(rose_train$Class)

rose_train[,30] <- as.numeric(as.character(rose_train[,30]))

PCmodel <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(15)) %>% 
  layer_dense(units = 12, activation = "linear") %>% 
  layer_dense(units = 12, activation = "linear") %>%
  layer_dense(units = 12, activation = "linear") %>% 
  layer_dense(units = 1, activation = "sigmoid") 

summary(PCmodel)

PCmodel %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy")

PCmodel %>% 
  fit(
    x = as.matrix(rose_train[,c(18,17,16,14,12,11,10,9,7,6,5,4,3,2,1)]), y = rose_train[,"Class"],
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

#testPC$Class = as.factor(as.character(testPC$Class))

#Build confusion matrix to evaluate predictions
#Eval pred on test set
test_predictions <- PCmodel %>% 
  predict(as.matrix(data.frame(testPC[,c(18,17,16,14,12,11,10,9,7,6,5,4,3,2,1)])))


#Evaluate predictions
roc(testPC$Class,test_predictions)$auc  #Area Under the Curve

#Confusion Matrix
delta = 0.50
predicted_success2 = ifelse(test_predictions >= delta,1,0) #Class prediction
confusionMatrix(as.factor(predicted_success2),as.factor(testPC$Class)) #Create confusion matrix


############################################

V1_plot <- ggplot(trainPC, aes(x = V1, y = Class))+
  geom_boxplot()
V1_plot

V2_plot <- ggplot(trainPC, aes(x = V2, y = Class))+
  geom_boxplot()
V2_plot

V3_plot <- ggplot(trainPC, aes(x = V3, y = Class))+
  geom_boxplot()
V3_plot

V4_plot <- ggplot(trainPC, aes(x = V4, y = Class))+
  geom_boxplot()
V4_plot

V5_plot <- ggplot(trainPC, aes(x = V5, y = Class))+
  geom_boxplot()
V5_plot

V6_plot <- ggplot(trainPC, aes(x = V6, y = Class))+
  geom_boxplot()
V6_plot

V7_plot <- ggplot(trainPC, aes(x = V7, y = Class))+
  geom_boxplot()
V7_plot

V8_plot <- ggplot(trainPC, aes(x = V8, y = Class))+
  geom_boxplot()
V8_plot

V9_plot <- ggplot(trainPC, aes(x = V9, y = Class))+
  geom_boxplot()
V9_plot

V10_plot <- ggplot(trainPC, aes(x = V10, y = Class))+
  geom_boxplot()
V10_plot

V11_plot <- ggplot(trainPC, aes(x = V11, y = Class))+
  geom_boxplot()
V11_plot

V12_plot <- ggplot(trainPC, aes(x = V12, y = Class))+
  geom_boxplot()
V12_plot

V13_plot <- ggplot(trainPC, aes(x = V13, y = Class))+
  geom_boxplot()
V13_plot

V14_plot <- ggplot(trainPC, aes(x = V14, y = Class))+
  geom_boxplot()
V14_plot

V15_plot <- ggplot(trainPC, aes(x = V15, y = Class))+
  geom_boxplot()
V15_plot

V16_plot <- ggplot(trainPC, aes(x = V16, y = Class))+
  geom_boxplot()
V16_plot

V17_plot <- ggplot(trainPC, aes(x = V17, y = Class))+
  geom_boxplot()
V17_plot

V18_plot <- ggplot(trainPC, aes(x = V18, y = Class))+
  geom_boxplot()
V18_plot

V19_plot <- ggplot(trainPC, aes(x = V19, y = Class))+
  geom_boxplot()
V19_plot

V20_plot <- ggplot(trainPC, aes(x = V20, y = Class))+
  geom_boxplot()
V20_plot

V21_plot <- ggplot(trainPC, aes(x = V21, y = Class))+
  geom_boxplot()
V21_plot

V22_plot <- ggplot(trainPC, aes(x = V22, y = Class))+
  geom_boxplot()
V22_plot

V23_plot <- ggplot(trainPC, aes(x = V23, y = Class))+
  geom_boxplot()
V23_plot

V24_plot <- ggplot(trainPC, aes(x = V24, y = Class))+
  geom_boxplot()
V24_plot

V25_plot <- ggplot(trainPC, aes(x = V25, y = Class))+
  geom_boxplot()
V25_plot

V26_plot <- ggplot(trainPC, aes(x = V26, y = Class))+
  geom_boxplot()
V26_plot

V27_plot <- ggplot(trainPC, aes(x = V27, y = Class))+
  geom_boxplot()
V27_plot

V28_plot <- ggplot(trainPC, aes(x = V28, y = Class))+
  geom_boxplot()
V28_plot

Amount_plot <- ggplot(trainPC, aes(x = Amount, y = Class))+
  geom_boxplot()
Amount_plot

#19,18,17,16,14,12,11,10,9,7,5,4,3,2,1


#  For choosing the final model, first i created 4 neural networks based on the four different samples knowing it would be easier that way because they will not be changing in the model and four is faster than one. The makeup will be different for each balanced set. I would then run all 4 of the models, jot down their values. Then I would add a few nodes and jot down these values. Then i would go back to 5 nodes and different act function. jot down those values. then go back to linear act, and add two hidden layers. jot down those answers, then tried to try as many combinations as i could of these setups as efficiently as I could, all while trying to hone in on the best values and parameters for those values. The main values i was looking at was spec, sens, and kappa. in that order of importance. The one that could give me the highest spec was obviously the best one for detecting fraud, but i knew their was cost of low sens. So i tried to pick the highest spec, with a relatively high sens and this obviously brought a high kappa with it! And the model that seemed to give me the best of both worlds, with a focus on spec, was rose_train. Knowing that there was likely a better setup some how, I knew time was of the essense, and there was not that much better I could do. 


#########################################################################################################################################################################################################

#############################################################
#Question 5 
trainPC$Class = as.numeric(as.character(trainPC$Class))

PCmodel <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(15)) %>% 
  layer_dense(units = 11, activation = "linear") %>%
  layer_dense(units = 11, activation = "linear") %>%
  layer_dense(units = 10, activation = "linear") %>% 
  layer_dense(units = 1, activation = "sigmoid") 

summary(PCmodel)

PCmodel %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy")

PCmodel %>% 
  fit(
    x = as.matrix(rose_train[,c(18,17,16,14,12,11,10,9,7,6,5,4,3,2,1)]), y = rose_train[,"Class"],
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

# #testPC$Class = as.factor(as.character(testPC$Class))

#Build confusion matrix to evaluate predictions
#Eval pred on test set
test_predictions <- PCmodel %>%
  predict(as.matrix(data.frame(testPC[,c(18,17,16,14,12,11,10,9,7,6,5,4,3,2,1)])))

#Evaluate predictions
roc(testPC$Class,test_predictions)$auc  #Area Under the Curve

#Confusion Matrix
delta = 0.50
predicted_success2 = ifelse(test_predictions >= delta,1,0) #Class prediction
confusionMatrix(as.factor(predicted_success2),as.factor(testPC$Class)) #Create confusion matrix


#    Well, the accuracy is very good as it was in the origal model. But the specificity is a lot better. It isnt as high as i hoped, but it is 45-50% better than the original model on the balanced set. On consistent trials this model performed at a specificity of 89% as compared to the original mode had around 45%. The kappa on this model is pretty high, which is somewhat good! This means that we could not have gotten close to these outcomes by making mere predictions by guessing. Our sensitiviy could not get much better than it is which is great at 99+%. This is how accurate the data is at predicting on the valid transactions.This makes sense because there is a great amount more points for that valid transactions. Accuracy could not get much better either, although, it is the same as the original model.


rose_train$Class = as.numeric(as.character(rose_train$Class))

PCmodel <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(15)) %>% 
  layer_dense(units = 11, activation = "linear") %>%
  layer_dense(units = 11, activation = "linear") %>%
  layer_dense(units = 10, activation = "linear") %>% 
  layer_dense(units = 1, activation = "sigmoid") 

summary(PCmodel)
```

```{r}
PCmodel %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy")

PCmodel %>% 
  fit(
    x = as.matrix(rose_train[,c(18,17,16,14,12,11,10,9,7,6,5,4,3,2,1)]), y = rose_train[,"Class"],
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

# #testPC$Class = as.factor(as.character(testPC$Class))

#Build confusion matrix to evaluate predictions
#Eval pred on test set
test_predictions <- PCmodel %>%
  predict(as.matrix(data.frame(testPC[,c(18,17,16,14,12,11,10,9,7,6,5,4,3,2,1)])))

#Evaluate predictions
roc(testPC$Class,test_predictions)$auc  #Area Under the Curve

#Confusion Matrix
delta = 0.50
predicted_success2 = ifelse(test_predictions >= delta,1,0) #Class prediction
confusionMatrix(as.factor(predicted_success2),as.factor(testPC$Class)) #Create confusion matrix

#question 6

##Business Intuition
#     To calculate the cost of a false positive, this can be obtained by the average communication fees that the bank should be able to calculate via survey. You might have to see on average, how many emails/calls it takes for the client to answer and how much each of these emails/calls costs you. Let's say we calculate a rate at which the customer will leave the bank if there is a false hold, we can calculate how much losing a customer costs and multiply it by that rate and put a weight into a sort of hierarchical model. The amount of workers needed for these communications should be directly affecting the cost from False-pos. 
#   We can also calclate the average amount of money lost from a client who has been victimized by fraud and this average will serve as the cost of one false-neg transaction. 


####
trainPC$Class = as.numeric(as.character(trainPC$Class))

PCmodel <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(29)) %>% 
  layer_dense(units = 11, activation = "linear") %>%
  layer_dense(units = 11, activation = "linear") %>%
  layer_dense(units = 10, activation = "linear") %>% 
  layer_dense(units = 1, activation = "sigmoid") 

summary(PCmodel)

PCmodel %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy")

PCmodel %>%
  fit(
    x = as.matrix(trainPC[,c(1:29)]), y = trainPC[,"Class"],
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

testPC$Class = as.numeric(as.character(testPC$Class))

test_predictions <- PCmodel %>% 
  predict(as.matrix(data.frame(testPC[,c(1:29)])), type = "prob")

default_probability = test_predictions[,1]

### Loss function
cutoff = seq(min(default_probability),max(default_probability),.001)
performance = setNames(data.frame(matrix(ncol = 8, nrow = length(cutoff))), c("Cutoff","TN", "FN", "TP", "FP", "Sensitivity", "Specificity","Accuracy"))
performance$Cutoff = cutoff


for (i in 1:length(cutoff)){
  temp = table( default_probability > performance$Cutoff[i], testPC$Class)
  TN = temp[1,1]
  FN = temp[1,2]
  FP = temp[2,1]
  TP = temp[2,2]
  performance$TN[i] = TN
  performance$TP[i] = TP
  performance$FN[i] = FN
  performance$FP[i] = FP
  performance$Sensitivity[i] = TP/(FN+TP)
  performance$Specificity[i] = TN/(TN+FP)
  performance$Accuracy[i] = (TP+TN)/(FP+FN+TP+TN)
}

average_fraud <- group_by(trainPC, Class) %>%
  summarise(mean_fraud = mean(Amount))
average_fraud[2,2]
#Define a loss function:
LossFP = 5
LossFN = 126.1187

performance$Loss = performance$FP*LossFP + performance$FN*LossFN

ggplot(performance,aes(Cutoff,Loss))+
  geom_line()

performance[which.min(performance$Loss),] #Best cutoff

#Question 8
#Confusion Matrix
delta = 0.009
predicted_success2 = ifelse(test_predictions >= delta,1,0) #Class prediction
confusionMatrix(as.factor(predicted_success2),as.factor(testPC$Class)) #Create confusion matrix

#a.)So there are 85,442 transactions in this test set. From here, we know that we predict .299618% of the transactions will be considered fraudulent. From these predictions, 35.8% of these transactions are expected to actually be fraudulent. So we detect that .151% of transactions are fraudulent. Finally, we will detect and stop .151% of 1 million transactions that go through, and this equates to 1510 stoppages of transactions. But another interpretation of the question may be that if we predict that it is fraudulent, and do not ask the owner/bankee verifiable questions and just shut the account down automatically, therefore the answer would then be .29618% will be fraudulent and .299618% of one million will be 2996.18 detected and closed transactions.

#b.)For this question, we know that a False-negative transaction equates to 126.1187$ cost and false-positive equates to 5$ in cost. Therefore, we lose ((28/85,442)*(20,000,000/12))*126.1187 = 69,976.52$ per month due to fraudulent transactions. 

#c.) For this question, we have 127/85442 = .00148639 = .148639% of people are put on hold for a confirmation when they don't need to be. Out of 1,000,000,,,,  1,486.39 transactions will be put on hold for a confirmation. There is no telling how many customers will be put on hold though. 

#d.) I am 95% confident that my answers to these questions are accurate enough to be used in practice on this TEST SET. Though, as a data analyst, I must note that because the number of actual fraudulent transactions is quite low, this may result in a lack of trueness to these predictions.
