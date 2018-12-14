rm(list=ls(all=T))
setwd("C:/Users/RAUNAK/Desktop/edwisor/workspace")

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees','usdm')

install.packages(x)
lapply(x, require, character.only = TRUE) 

rm(x)

## Read the data
bike_rental = read.csv("day.csv", header = T, na.strings = c(" ", "", "NA"))


###########################################Explore the data##########################################
str(bike_rental)

#Remove dteday , instant column 
bike_rental= subset(bike_rental, select = -c(dteday,instant, casual,registered))

# Casual and Registered columns are also removed because that is what we are going to predict.


hist(bike_rental$temp)
hist(bike_rental$atemp)
hist(bike_rental$hum)
hist(bike_rental$windspeed)

##################################Missing Values Analysis###############################################
#Check for null fields
sum(is.na(bike_rental))


############################################Outlier Analysis#############################################
# ## BoxPlots - Distribution and Outlier Check

#First we will convert non continuous variables to Factor
bike_rental[1:7] <- sapply(bike_rental[1:7] , as.factor)


#selecting only numeric index
numeric_index = sapply(bike_rental,is.numeric) 

numeric_index 

numeric_data =bike_rental[,numeric_index]

cnames =  colnames(numeric_data)

#Generate Box Plots for Numeric variables. The same code has been used for Univariate Analysis
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i])), data = subset(bike_rental))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i])+
           ggtitle(paste("Box plot of ",cnames[i])))
}



## Plotting plots together

gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=4)



# 
# #Replace all outliers with NA and impute
# #create NA on 'hum' and 'windspeed'


df=bike_rental

#bike_rental=df

cnames= c( "hum"   , "windspeed")

for(i in cnames){
  val = bike_rental[,i][bike_rental[,i] %in% boxplot.stats(bike_rental[,i])$out]
  #print(length(val))
  bike_rental[,i][bike_rental[,i] %in% val] = NA
}
#Check for NA in the dataset. 
sum(is.na(bike_rental))

#We get count of NA=15 , hence there were a total of 15 outliers

# Temporarily change variable types to numeric because KNN Imputation works on
#numeric data only
bike_rental[1:7] <- sapply(bike_rental[1:7] , as.numeric)

#Apply KNN Imputation
bike_rental = knnImputation(bike_rental, k =5)

#Check for NA in the dataset again. 
sum(is.na(bike_rental))

#We get count of NA=0 

#Convert categorical variables back to Factor
bike_rental[1:7] <- sapply(bike_rental[1:7] , as.factor)



##################################Feature Selection################################################
#Display correlation values for the dataset
symnum(cor(bike_rental))


# Use Random Forest to determine prediction power of each variable
variable_power <- randomForest(cnt ~ ., data = bike_rental,
                        ntree = 100, keep.forest = FALSE, importance = TRUE)

importance(variable_power, type = 1)

## Correlation Plot 
corrgram(bike_rental[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

##Correlation Values
sub=data.frame(bike_rental$temp,bike_rental$atemp,bike_rental$hum,bike_rental$windspeed)

cor(sub)


## Dimension Reduction


bike_rental= subset(bike_rental, select = -c(atemp))

                     
##################################Feature Scaling################################################
#Normality check


hist(bike_rental$temp)
hist(bike_rental$atemp)
hist(bike_rental$hum)
hist(bike_rental$windspeed)

#As we have already removed any outliers, the dataset is already normalised




###################################Model Development#######################################
#Clean the environment 
library(DataCombine)
rmExcept("bike_rental")

#Define MAPE function

MAPE = function(y, yhat){ mean(abs((y - yhat)/y))}


#Divide the data into train and test using Random Sampling
set.seed(123)
train_index =sample(1:nrow(bike_rental), 0.80 * nrow(bike_rental))
train = bike_rental[train_index,]
test = bike_rental[-train_index,]


######### MULTIPLE LINEAR REGRESSION ###############

#We convert all factor variables to numeric, as expected by the model.
bike_rental[1:7] <- sapply(bike_rental[1:7] , as.numeric)

#run regression model
lm_model = lm(cnt ~., data = train)

#Summary of the model
summary(lm_model)

#Predict
predictions_LR = predict(lm_model, test[,1:10])

#Calculate MAPE
MAPE(predictions_LR,test[,11])

#Error: 14.4%
#Accuracy: 85.6%

############# DECISION TREE REGRESSION ################
bike_rental[1:7] <- sapply(bike_rental[1:7] , as.factor)

#Generating Decision Tree
rtmodel <- rpart(cnt ~ ., data = train)
plot(rtmodel, uniform = T, branch = 1, margin = 0.1,cex =0.7)
text(rtmodel, cex = 0.7)

#Apply the Decision Tree Regression through rpart
fit = rpart(cnt ~ ., data = train, method = "anova")

#Predict for new test cases
predictions_DT = predict(fit, test[,-11])

summary(predictions_DT)


MAPE( predictions_DT,test[,11])

#Error Rate: 16.9%
#Accuracy:83.1%



#############RANDOM FOREST REGRESSION################



RF_model = randomForest(cnt ~ ., train, importance = TRUE, ntree = 500)


summary(RF_model)

#Predict test data using random forest model

RF_Predictions= predict(RF_model, test[,-11])

MAPE(RF_Predictions,test[,11])

#Error: 12.3%
#Accuracy: 87.7%


#### SAMPLE INPUT AND OUTPUT#######

#Select 1 random row from dataset
sample_index =sample(1:nrow(bike_rental), 0.002 * nrow(bike_rental))

test_sample = bike_rental[sample_index,]

#View Sample
test_sample

#Cnt for above sample is 7415

#Predict Output Count using the model
RF_Predictions_sample= predict(RF_model, test_sample[,-11])

# View Predicted Value of Cnt
RF_Predictions_sample

#Predicted value is 6836.77

#Calculate Mean Percentage Error
MAPE(RF_Predictions_sample,test_sample[,11])

#Error: 8.4%








