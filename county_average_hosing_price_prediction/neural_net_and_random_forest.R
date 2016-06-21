#R Code:
library(neuralnet)
#read data that is converted to CSV and delete the area row, forced it to be numeric.
data<- read.csv("cleaneddata.csv",header=TRUE,colClasses="numeric")

#createtestset and trainset
smp_size <- floor(0.75 * nrow(data))
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]
test2 <- test[,c("DV1",'IV5','IV6','IV9','IV10','IV11','IV12','IV13','IV14',
                 'IV15','IV16','IV21','IV22')]

#first run of neuralnetwork = one hidden layer of five node
neuralnet1 <- neuralnet(DV1 ~ IV1+IV2+IV3+IV4+IV5+IV6+IV7+IV8+IV9+IV10+
                          IV11+IV12+IV13+IV14+IV15+IV16+IV17+IV18+IV19+
                          IV20+IV21+IV22,data=train,hidden=5,
                        linear.output = TRUE,lifesign="full",stepmax=1e6, threshold = 0.075)

#second run of neuralnetwork = two hidden layers
neuralnet2 <- neuralnet(DV1 ~ IV1+IV2+IV3+IV4+IV5+IV6+IV7+IV8+IV9+IV10
                        +IV11+IV12+IV13+IV14+IV15+IV16+IV17+IV18+IV19+
                          IV20+IV21+IV22,data=train,hidden=c(5,3),
                        linear.output = TRUE,lifesign="full",stepmax=1e6, threshold = 0.075)

#third run of neuralnetwork = three hidden layers - Does not converge
neuralnet3 <- neuralnet(DV1 ~ IV1+IV2+IV3+IV4+IV5+IV6+IV7+IV8+IV9+IV10+IV11+IV12+IV13+IV14+IV15+IV16+IV17+IV18+IV19+IV20+IV21+IV22,data=train,hidden=c(8,5,5),linear.output = TRUE,lifesign="full",stepmax=1e7, threshold = 0.075)

#fourth run
neuralnet4 <- neuralnet(DV1 ~ IV5+IV6+IV9+IV10+IV11+IV12+IV13+IV14+IV15+IV16+IV21+IV22,data=train,hidden=c(5),linear.output = TRUE,lifesign="full",stepmax=1e6)

#fifth run
neuralnet5 <- neuralnet(DV1 ~ IV5+IV6+IV9+IV10+IV11+IV12+IV13+IV14+IV15+IV16+IV21+IV22,data=train,hidden=c(5,3),linear.output = TRUE,lifesign="full",stepmax=1e6, threshold = 0.075)

#sixth run
neuralnet6 <- neuralnet(DV1 ~ IV5+IV6+IV9+IV10+IV11+IV12+IV13+IV14+IV15+IV16+IV21+IV22,data=train,hidden=c(8,5,5),linear.output = TRUE,lifesign="full",stepmax=1e6, threshold = 0.075)

#plotting neuralnet

plot(neuralnet6, rep = NULL, x.entry = NULL, x.out = NULL, radius = 0.15, 
              arrow.length = 0.2, intercept = TRUE, intercept.factor = 0.4, 
              information = TRUE, information.pos = 0.1, col.entry.synapse = "black", 
              col.entry = "black", col.hidden = "black", col.hidden.synapse = "black", 
              col.out = "black", col.out.synapse = "black", col.intercept = "blue", 
              fontsize = 8, dimension = 4, show.weights = FALSE, file = NULL) 

#compute based on neuralnet4
test.r <- (test$DV1)*(max(data$DV1)-min(data$DV1))+min(data$DV1)

pr.nn1 <- compute(neuralnet1,test[,2:ncol(test)])
pr.nn1_ <- pr.nn1$net.result*(max(test$DV1)-min(test$DV1))+min(test$DV1)
pr.nn2 <- compute(neuralnet2,test[,2:ncol(test)])
pr.nn2_ <- pr.nn2$net.result*(max(test$DV1)-min(test$DV1))+min(test$DV1)
pr.nn3 <- compute(neuralnet3,test[,2:ncol(test)])
pr.nn3_ <- pr.nn3$net.result*(max(test$DV1)-min(test$DV1))+min(test$DV1)
pr.nn4 <- compute(neuralnet4,test2[,2:ncol(test2)])
pr.nn4_ <- pr.nn4$net.result*(max(test2$DV1)-min(test2$DV1))+min(test2$DV1)
pr.nn5 <- compute(neuralnet5,test[,2:ncol(test2)])
pr.nn5_ <- pr.nn5$net.result*(max(test2$DV1)-min(test2$DV1))+min(test2$DV1)
pr.nn6 <- compute(neuralnet6,test[,2:ncol(test2)])
pr.nn6_ <- pr.nn6$net.result*(max(test2$DV1)-min(test2$DV1))+min(test2$DV1)

#mean squared error calculations
MSE.nn1 <- sum((test.r - pr.nn1_)^2)/nrow(test)
MSE.nn2 <- sum((test.r - pr.nn2_)^2)/nrow(test)
MSE.nn3 <- sum((test.r - pr.nn3_)^2)/nrow(test)
MSE.nn4 <- sum((test.r - pr.nn4_)^2)/nrow(test2)
MSE.nn5 <- sum((test.r - pr.nn5_)^2)/nrow(test2)
MSE.nn6 <- sum((test.r - pr.nn6_)^2)/nrow(test2)

print(paste(MSE.nn1,MSE.nn2,MSE.nn3,MSE.nn4,MSE.nn5,MSE.nn6))

#trying linear regression
linear1 <- lm(DV1 ~., data=train)
pr.linear <- predict.lm(linear1,test[,2:ncol(test)])
pri.linear<-data.frame(predict.lm(linear1,test[,2:ncol(test)]))

#compute based on linear1
MSE.lm <- sum((pr.linear - test$DV1)^2)/nrow(test)
print(paste(MSE.lm))

#test randomforest
library(randomForest)
rf1 <- randomForest(DV1 ~.,data=train,ntree=1000)
imp1 <- importance(rf1)
pr.rf <- data.frame(predict(rf1,test))
MSE.rf <- sum((pr.rf - test$DV1)^2)/nrow(test)
pri.rf<-data.frame(predict(rf1,test))
