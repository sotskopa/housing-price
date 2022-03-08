# run this in the terminal below.
# activate tensor
# pip install sklearn
# pip install pandas
# pip install numpy allows arrays

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle


# add seperator for csv file
data = pd.read_csv(r'C:\Users\arvin\Documents\KTH INNOVATION\hemnet_data\web-scraping-algorithm\test_dataset.csv', sep=';')
# Look at your trimmed dataset
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

# What do we wanna predict? G3. Based on all other attributes, we want to know the prediction of G3.

predict = 'G3'

# Set up 2 arrays, 1 array defines all the attributes and one array defines the label
# X = attributes
X = np.array(data.drop([predict], 1))

# gives us a new data frame that doesnt have G3 in it.
# Y = labels

y = np.array(data[predict])

# Split up into 4 variables, x test, y test, x train, y train
# 90% is train data 10% is test data.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# the actual training model
linear = linear_model.LinearRegression()
# fit the data to find the best fit line, using the x train and y train data, and it is gonna
# store that line in the variable linear.
linear.fit(x_train, y_train)
# # then we can use this to test our test data on
acc = linear.score(x_test, y_test)
# this is gonna return to us the accuracy of our model
print (acc)
# the accuracy is gonna be different each time becuase when we train it, it will score different each time.


# 0.82343...
# we can with 82% accuracy determine the students final grades based on the other attributes.
# k- values
# y = kx + m is the line in 2-dimensional space, a line in 5 dimensional space needs 5 coefficients, 5 k-values.
# typ y = (kx + zy + cw) + m
# the bigger the Coefficient, the bigger the weight the attribut has in defining the prediction.

print ('Coefficient: \n', + linear.coef_)
# m- value
print ('Interceot: \n', + linear.intercept_)
# getting all coefficients of our 5 variables.

# Now, how can we use this model to predict what grade a student will get?
# this will take an array and do predictions based on the training data but on our test data
predictions = linear.predict(x_test)

# we will print out all the predictions and the input data for that prediction

for x in range(len(predictions)):
    print (predictions[x], x_test[x], y_test[x]) # the input data is x_test[x], the actual value of the final grade is
# y_test[x]
# The actual grade is the value outside the array on the right side, and the prediction is the value on the left side







