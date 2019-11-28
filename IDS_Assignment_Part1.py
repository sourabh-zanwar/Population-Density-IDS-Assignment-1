#!/usr/bin/env python
# coding: utf-8

# # The First Part of the Assignment of IDS 2019-2020
# Document your results as well as the way you obtained them in this jupyter notebook. A seperate report (pdf, word etc.) is _not_ required. However, it is necessary that you provide the python code leading to your results as well as textual answers to the assignment questions in this notebook. *DO NOT CLEAR THE OUTPUT of the notebook your are submitting!* In the cases that the result of an algorithm is pdf, jpg, etc, you should attach the result to this notebook file and refer to that in the text.  
# 
# Next to the jupyter notebook, submit one zip-file containing all data sets that you are asked to submit. Make sure they are easily identifiable, i.e. use names as requested in the corresponding question.
# 
# Do not change the general structure of this notebook, but you can add further markdown or code cells to explain your solutions if necessary. In the end, submit this file and your created data sets in moodle.
# 
# Only <font color="red">one </font> group member should upload your group's solution. *Make sure to include group members' names and matriculation numbers*. If your name and student id are not included in the report, you will not receive any points!
# 
# 
# Hint 1: While answering the questions, you will get a better and better impression of the given data. However, feel free to compute additional results and vizualizations to motivate the decisions you need to make, for example with respect to modification, simplification or sampling of the data. <font color="red"><b>Ensure that all claims you make are supported by the presented facts!</b></font>
# 
# Hint 2: <font color="red"><b>Some of the tasks might need some time to run. Take this into account in your planning.</b></font>
# 
# Hint 3: RWTHonline allows for multiple submissions (each submission overwrites the previous ones). <font color="red"><b>Partial submissions are possible and encouraged. </b></font> This helps in case of technical problems of RWTHonline, which do seldomly happen.

# <font color="red"><b>Student Names and IDs:
#     1. Sourabh Zanwar (Matr.Nr.391923)
#     2. Malek Alhemwaly (Matr. Nr. )
#     3.


# ## Preprocessing of the Dataset (5 points)
#  You should carry out some preprocessing steps before starting the analysis:
#  - Select 90% of 'population_density' dataset by random sampling.
#      - Use one of the group member's student number as a seed.
#  - Add another column to the dataset:
#      - Name it 'population_density_categorical'.
#      - The values in this column depend on values in 'population_density' column and will be defined as follows:
#         - 5824=<population density              corresponding value in 'population_density_categorical' column: 'very high'
#         - 4368=<population density<5824         corresponding value in 'population_density_categorical' column: 'high'
#         - 2912=<population density<4368         corresponding value in 'population_density_categorical' column: 'medium'
#         - 1456=<population density<2912         corresponding value in 'population_density_categorical' column: 'low' 
#         - population density<1456               corresponding value in 'population_density_categorical' column: 'very low'
#  - After completing this preprocessing step, export your final dataset as 'population_density_categorical.csv' dataset and use that for next steps of the assignment.
#  - If it is not directly mentioned, you should always use your extracted (above-created) dataset (without any cleaning).
#  - <font color='red'>Important!</font> Make sure that you submit your extracted dataset with your result in moodle.

# In[ ]:


# Your code:


import numpy as np #For Mathematics
import matplotlib.pyplot as plt #For Plotting charts
import pandas as pd #For importing and managing Datasets
import random

df = pd.read_csv('population_density.csv')
skip = sorted(random.sample(range(int(df.shape[0])),int(df.shape[0]*0.1)))
df = pd.read_csv('population_density.csv', skiprows=skip) #Random sampling of the data
df['population_density_categorical'] = pd.cut(df['population_density'], bins=[0,1456,2912,4368,5824,7281],include_lowest=True,labels=['very low','low','medium','high','very high']) #Binning the data and labeling them
df.to_csv('population_density_categorical.csv')  #Exporting the dataframe to csv


# ## Question 1 - Insights into the Data (20 points):

# ### Dealing with Outliers (10 points)
#    (a) Use a boxplot to find and remove the outliers from "temperature". Note that based on the boxplot the values greater than upper-whisker and less than lower-whisker are considered as the outliers. Now you should  have two datasets (cleaned and original).

# In[ ]:


# Your code:
import seaborn as sns
sns.boxplot(x=df['temperature'])
q1 =  df['temperature'].quantile(0.25)
q3 = df['temperature'].quantile(0.75)
IQR = q3 - q1
filter = (df['temperature'] >= q1 - 1.5 * IQR) & (df['temperature'] <= q3 + 1.5 *IQR)
df_cleaned = df.loc[filter]

#    (b) Compare basic statistical features of "temperature" (median, mean, and mode) in the cleaned and original datasets.    Interpret the differences for these statistical values between the cleaned and original datasets. 

# In[ ]:


# Your code:
print("For uncleaned Dataset :")
print(np.mean(df['temperature'])) #mean for the 'temperature' for uncleaned dataset
print(np.std(df['temperature'])) #standard diviation for the 'temperature' for uncleaned dataset
print(np.var(df['temperature'])) #variance for the 'temperature' for uncleaned datasettotal_bill
print("For Cleaned Dataset :")
print(np.mean(df_cleaned['temperature'])) #mean for the 'temperature' for uncleaned dataset
print(np.std(df_cleaned['temperature'])) #standard diviation for the 'temperature' for uncleaned dataset
print(np.var(df_cleaned['temperature'])) #variance for the 'temperature' for uncleaned datasettotal_bill




#    (c) Compare the number of data rows before and after removing the outliers. How many data rows are removed by removing    outliers?

# In[ ]:


# Your code:
print('number of rows before cleaning : ', df.shape[0])
print('number of rows after cleaning : ', df_cleaned.shape[0])
diff = df.shape[0]-df_cleaned.shape[0]
print('Number of datarows removed by removing outliers : ', diff)


# ### Basic Visualization (10 points)
#    (d) Visualize mean and median of "population_density" for non-'None' "holiday" values in the original dataset. Specify    the "holiday" values for which the mean of "population_density" is maximal and for which it is minimal.

# In[ ]:


# Your code:


#    (e) Plot the distribution of "temperature" in the original and cleaned datasets.

# In[ ]:


# Your code:
sns.distplot(df['temperature'])
sns.distplot(df_cleaned['temperature'])

#    (f) Explore the distribution of "population_density" and "temperature" together in the cleaned dataset. Specify the ranges of "temperature" and "population_density" for which the frequency of the data is the highest.

# In[ ]:


# Your code:
sns.jointplot(df_cleaned['population_density'],df_cleaned['temperature'],kind='kde')
#The population density and the temperature for which the frequency is maximum is
# Temperature range 289 to 292
# Population Density range 400 to 700

# ## Question 2 - Decision Trees (15 points):

#    (a) Add a categorical temperature column to the cleaned dataset based on the following rules and name it                "temperature_categorical":
#         - temperature >= upper_quartile (third quartile) corresponding value in 'temperature_categorical' column: 'high'
#         - temperature <= lower_quartile (first quartile) corresponding value in 'temperature_categorical' column: 'low'
#         - lower_quartile < temperature < upper_quartile corresponding value in 'temperature_categorical' column: 'medium'

# In[ ]:

# Your code:
df_cleaned['temperature_categorical'] = pd.qcut(df_cleaned['temperature'], q = 3, labels = ['low','medium','high'])



#    (b) In the new dataset (created in Section 'a'), consider "holiday", "temperature_categorical", and "weather_type" as    the descriptive features and "population_density_categorical" as the target feature. Set the minimum number of samples  for splitting to 5000 and make a decision tree based on entropy.

# In[ ]:


# Your code:
from sklearn.tree import DecisionTreeClassifier
from p_decision_tree.DecisionTree import DecisionTree
descriptive_features = ['holiday', 'temperature_categorical','weather_type']
target_feature = ['population_density_categorical']
X = df_cleaned[descriptive_features].values
Y = df_cleaned[target_feature].values




#
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#labelencoder_X = LabelEncoder()
#integer_encoded_holidays = labelencoder_X.fit_transform(X['holiday'].values)
#integer_encoded_temp_categorical = labelencoder_X.fit_transform(X['temperature_categorical'].values)
#integer_encoded_weather_type = labelencoder_X.fit_transform(X['weather_type'].values)
#onehotencoder = OneHotEncoder(categorical_features = [0])



#from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#labelencoder_X = LabelEncoder()
#X['holiday'] = labelencoder_X.fit_transform(X['holiday'])
#X['temperature_categorical'] = labelencoder_X.fit_transform(X['temperature_categorical'])
#X['weather_type'] = labelencoder_X.fit_transform(X['weather_type'])
#onehotencoder = OneHotEncoder()
#X_encoded = onehotencoder.fit_transform(X).toarray()
#
#labelencoder_Y = LabelEncoder()
#Y = labelencoder_Y.fit_transform(Y)
#
#
#classifier = DecisionTreeClassifier(criterion = 'entropy',min_samples_split=5000)
#classifier.fit(X,Y)

decisionTree = DecisionTree(X.tolist(),descriptive_features,Y.tolist(),"entropy")
decisionTree.id3(0,5000)
decisionTree.print_visualTree(render = True)


#    (c) What is the best attribute (based on entropy) for splitting the tree in the second round of ID3?  

# Explanation: 
# 

# ## Question 3 - Regression (10 points):
# 
# For this question (Q3), restrict your extracted data set to the columns *holiday, temperature, rain_1h, snow_1h, clouds_percentage, date_time* and *population_density*. Drop the information on the date for *date_time* and restrict the data to time values (hour) only.
# 
# We define *population_density* to be our target feature, while all other features are considered descriptive.

# In[ ]:


# Your code:
reg_features = ['holiday', 'temperature', 'rain_1h', 'snow_1h', 'clouds_percentage', 'date_time','population_density']
reg_df = df_cleaned[reg_features]
reg_df['date_time'] = pd.DatetimeIndex(reg_df['date_time']).time

#    (a) Which features are suitable as an input for linear regression and which need to be modified first? Why?

# Explanation: 
#  To be modified : holiday, date_time
#  Suitable features : temperature, rain_1h, snow_1h, cloud_percentage

#    (b) Implement and briefly motivate an adequate modification. Print the resulting data set limited to the first two data rows.

# Explanation:
# 

# In[ ]:


# Your code:


# (c) To get a first idea about our data, plot the behaviour of the target feature (population_density) over time (date_time).

# In[ ]:


# Your code:
sns.barplot(x="date_time", y="population_density", data=df_cleaned)




#    (d) Create two distinct subsets of the data. Use sampling methods as described in the lecture. You should end up with two DIFFERENT sample data sets *RegA, RegB*. Include these data sets in the submitted data set zip file. 
#    
#    Which sampling methods did you choose and why? Which one do you expect to result in a better classifier?

# In[ ]:


# Your code:


# Explanation:
# 

#    (e) Train a linear regression classifier based on each of the two sample data sets *RegA, RegB* created before.

# In[ ]:


# Your code:


# (f) Visualize both computed regression functions in one plot each, combined with the original data set. For example, you can show a plot similar to 3(c) and insert the regression functions result.

# In[ ]:


# Your code:


#    (g) For each of the two resulting models compute and print the mean squared error, mean absolute error and median squared error with respect to the original, non-sampled data set. Also, present plots showing the errors and squared errors.

# In[ ]:


# Your code:


#    (h) Interpret and evaluate the two models and compare them. Why are they similar/different? Which model do you recommend and why? How do you think the applied methods could be improved to get better results?

# Explanation:
# 

# ## Question 4 - Support Vector Machines (10 points):
# 
# For this question (Q4), restrict your extracted data set to the columns *holiday, temperature, rain_1h, snow_1h, clouds_percentage, date_time* and *population_density_categorical*. For *date_time*, drop the information on the date and restrict the data to time values only.
# We define *population_density_categorical* to be our target feature, while all other features are considered descriptive.
# 

# In[ ]:


# Your code:


#    (a) Which features are suitable as an input for SVM and which need to be modified first? Modify the data as needed and provide a brief explanation.

# In[ ]:


# Your code:


# Explanation:
# 

#    (b) Divide the data set into a training set *svmTrain* and a test set *svmTest* and briefly motivate your division      strategy. Include these data sets in the data set zip file you submit. 
#    
#    Hint: Training the SVMs will take longer for a large training set!

# In[ ]:


# Your code:


# Explanation:
# 

#    (c) Use the training set to train 3 different SVMs with different combinations of the parameters. Use at least two distinct values for the parameters *kernel* and *C*.

# In[ ]:


# Your code:


#    (d) Compute and print the mean accurracy and classification report of the trained SVMs with respect to the test set (as shown in instruction).

# In[ ]:


# Your code:


#    (e) Interpret and evaluate the 3 SVMs and compare them. Why are they similar/different? Which SVM do you recommend and why?

# Explanation:
# 

# ## Question 5 - Neural Networks (20 points)
# In this question consider the whole dataset that you have created in the *Preprocessing of Dataset* section. The target feature is *population_density_categorical*. 
# 
# To avoid performance problems, use the first 4000 rows of your dataset (you will need 2000 for training and 2000 for testing).

#    (a) What are the possible inputs of your network?
#      - Show the possible extracted features.
#      - Show the number of possible patterns of inputs for the categorical data.

# In[ ]:


# Your code:


# Explanation:
# 

#    (b) Before using the Neural Network algorithm, do you think it provides an accurate result? Why? 
#      - Your explanation should be supported by data.

# Explanation:
# 

# In[ ]:


# Your code:


#    (c) Which features can be used for designing your network as inputs? Why?
#        - Which features should be changed (example: from categorical to numerical or numerical to categorical)? Why? 
#        - Convert the features. 

# Explanation:
# 

# In[ ]:


# Your code:


#    (d) Train your network with default hyperparameters and return the parameters.

# In[ ]:


# Your code:


#    (e) Try to find an optimized number of hidden layers and nodes. 
#       - Start with default numbers and then at least go with one number above and one number below the default.
#       - Use the 2000 rows of the data for training.
#       - What are the optimized numbers of hidden layers and nodes that you have found?

# In[ ]:


# Your code: (number of hidden layers)


# In[ ]:


# Your code: (number of nodes in the hidden layer)


# In[ ]:


# Your code: (optimized network) 


# Explanation:
# 

#    (f) Try to train your model with one linear activation function and one non-linear activation function. Use the default number of hidden layers and nodes.
#      - Name the functions and explain how the results are different and why. 
#      - You can use evaluation metrics to show which activation function works better for this data set.
#      - Use the 2000 rows of the data for training.

# In[ ]:


# Your code: (linear activation function):


# In[ ]:


# Your code: (non-linear activation function):


# Explanation:
# 

# ## Question 6 - Evaluation (10 Points)

#    (a) Consider one of the models in question 4 (c) of your choice, one neural network with optimized number of hidden layers and nodes from question 5 (e), and the neural network with non-linear activation function from question 5 (f), for a total of *three* models, together with their respective datasets. Compute the following metrics for a 3-fold cross validation performed on each model:

# - Confusion matrices on the training data (sum cell-by-cell the results of each fold)

# In[ ]:


# Your code for the confusion matrices on the training data


# - Confusion matrices on the test data (sum cell-by-cell the results of each fold)

# In[ ]:


# Your code for the confusion matrices on the test data


# - Precision, recall and F1-scores on the test data (give explicitly the result for each fold)

# In[ ]:


# Your code for precision, recall and F1-scores on the test data


# - Accuracy scores on training and test data (give explicitly the result for each fold)

# In[ ]:


# Your code for accuracy scores on training and test data


# In order to turn numbers into insights, please comment on your findings. Motivate the answers to the following questions using the metrics and the findings in the questions 1 through 5 of the assignment.

#    (b) What is, in your opinion, the best model? Motivate your answer with the correct performance metrics.

# Explanation:
# 

#    (c) Does any model suffer from underfitting or overfitting? Motivate your answer with the correct performance metrics.

# Explanation:
# 

#    (d) What is the effect of employing cross validation, instead of simply holding our a certain percentage of examples as test/validation set? What are the advantages and disadvantages of cross validation?

# Explanation:
# 
