# Example code from: http://r-statistics.co/Logistic-Regression-With-R.html

# Install the necessary package.
install.packages('smbinning')
install.packages('InformationValue')
install.packages('car')
# Load the library for converting continuous variables into categorical variables.
library(smbinning)
# Load the library for determining an optimal prediciton cutoff for the model.
library(InformationValue)
# Load the library for estimating the variance inflaction factor.
library(car)

# Define and set the Working Directory
workdir <- "D:\\Logistic_Regression\\"
setwd(workdir)

# Step 1: Import the Data
inputData <- read.csv("http://rstatistics.net/wp-content/uploads/2015/09/adult.csv")
# View the first six observations of data.
head(inputData)

# Step 2: Check for Class Bias
# (1 == TRUE and 0 == FALSE)
table(inputData$ABOVE50K)

# Step 3: Create Training and Test Samples
# All values with condition equal to 1 are input into the input_ones dataframe.
input_ones <- inputData[which(inputData$ABOVE50K == 1), ]
# All values with condition equal to 0 are input into the input_zeros dataframe.
input_zeros <- inputData[which(inputData$ABOVE50K == 0), ]
# Set a seed to make "random" sampling generation repeatable.
set.seed(100)
# This pulls 70% of the values with a condition equal to 1 (5,488 observations).
input_ones_training_rows <- sample(1:nrow(input_ones), 0.7*nrow(input_ones))
# This pulls an equal number of values from the input_zeros dataframe.
input_zeros_training_rows <- sample(1:nrow(input_zeros), 0.7*nrow(input_ones))

# Puts the sampled values for input_ones into a dataframe.
training_ones <- input_ones[input_ones_training_rows, ]  
# Puts the sampled values for input_zeros into a dataframe.
training_zeros <- input_zeros[input_zeros_training_rows, ]
# Merges the ones and zeros into a trainingData dataframe.
trainingData <- rbind(training_ones, training_zeros)

# Take unused values from the input_ones and put them in a testing dataframe.
test_ones <- input_ones[-input_ones_training_rows, ]
# Take unused values from the input_zeros and put them in a testing dataframe.
test_zeros <- input_zeros[-input_zeros_training_rows, ]
# Combine the testing dataframes into a testData dataframe.
testData <- rbind(test_ones, test_zeros)

# Step 4: Compute Information Value to Find Out Important Variables
# Define the variables as Factors or Continuous.
factor_vars <- c ("WORKCLASS", "EDUCATION", "MARITALSTATUS", "OCCUPATION", "RELATIONSHIP", "RACE", "SEX", "NATIVECOUNTRY")
continuous_vars <- c("AGE", "FNLWGT","EDUCATIONNUM", "HOURSPERWEEK", "CAPITALGAIN", "CAPITALLOSS")

# Creates a dataframe to store information values resulting for the categorical variables.
iv_df <- data.frame(VARS=c(factor_vars, continuous_vars), IV=numeric(14))

# A for loop that runs 8 times (once for each factor variable).
for(factor_var in factor_vars)
  {
    # Checks the factor variable against Above50k variable.
    smb <- smbinning.factor(trainingData, y="ABOVE50K", x=factor_var, maxcat=42)
    # Puts the information values into the iv_df dataframe.
    iv_df[iv_df$VARS == factor_var, "IV"] <- smb$iv
  }

# A for loop that runs 6 times (once for each continuous variable).
for(continuous_var in continuous_vars)
  {
    # Checks the continuous variable against Above50k variable.
    smb <- smbinning(trainingData, y="ABOVE50K", x=continuous_var)
    # Some variables within the dataset need to be cleaned, they are disregarded for this example.
    if(class(smb) != "character")
      {
      # Puts the information values into the iv_df dataframe.
      iv_df[iv_df$VARS == continuous_var, "IV"] <- smb$iv
      }
  }

# This sorts the dataframe in descending order from the information value.
iv_df <- iv_df[order(-iv_df$IV), ]
# Let's look at the results.
iv_df

# Step 5: Build the Logit Model and Predict on Test Data
# Using a binomial family along with glm() function produces a regression model on the formula below.
logitMod <- glm(ABOVE50K ~ RELATIONSHIP + AGE + CAPITALGAIN + OCCUPATION + EDUCATIONNUM, data=trainingData, family=binomial(link="logit"))

# The predict function tests the logitMod against the test data, using type response gives probabilities instead of logit scale.
predicted <- predict(logitMod, testData, type="response") 

# Comput an optimal cutoff to improve the priction of 1's and 0's while reducing misclassification error.
optCutOff <- optimalCutoff(testData$ABOVE50K, predicted)[1] 

# Step 6: Model Diagnostics
# Gives the beta coefficients, standard error, z value, and p value.
summary(logitMod)

# Checks for a Variance Inflation Factor (X variables should have a VIF below 4.)
vif(logitMod)

# Check for misclassification error (the lower the better).
misClassError(testData$ABOVE50K, predicted, threshold = optCutOff)

# Receiving Operating Characteristics Curve.  The greater the AU(Area Under)ROC, the better the model.
plotROC(testData$ABOVE50K, predicted)

# The percentage of pairs, whose scores of actual positive's are greater than the scores of actual negatives.
Concordance(testData$ABOVE50K, predicted)

# The percentage of 1's (actuals) correctly predicted by the model.
sensitivity(testData$ABOVE50K, predicted, threshold = optCutOff)

# The percentage of 0's (actuals) correctly predicted by the model.
specificity(testData$ABOVE50K, predicted, threshold = optCutOff)

# The columns are actuals, while rows are predicteds.
confusionMatrix(testData$ABOVE50K, predicted, threshold = optCutOff)
