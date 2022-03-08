import os

import re
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

# get csv using relative path
ddir = os.path.dirname(__file__)
t_data = os.path.join(ddir, 'datasets', 'ml_data.csv')

# add separator for csv file
data = pd.read_csv(t_data, sep=',')

# removing non-useful attributes, våning was almost all null
data = data.drop(['l�nk', 'adress', 'v�ning'], 1)

# data cleaning
# we split sale date into month and year
month, year = [], []
for i in range(len(data.index)):
    date_sale = data._get_value(i, 'f�rs�ljningsdatum')
    date_built = data._get_value(i, 'bygg�r')
    area = data._get_value(i, 'omr�de')

    # regex different formats for date and area
    date_sale_whole = re.search('(\d+)/(\d+)/(\d+)', date_sale)
    date_built_year = re.search('(\d{4})', str(date_built))
    date_built_two_years = re.search('(\d{4}).+(\d{4})', str(date_built))
    date_built_last_two = re.search('(\d{4}).+(\d{2})', str(date_built))
    date_built_year_tal = re.search('(\d{4}).+(tal)', str(date_built))
    date_built_two_tal = re.search('(\d{2}).+(tal)', str(date_built))
    date_built_ca = re.search('(\D+).+(\d{4})', str(date_built))
    area_lgh = re.search('(Lägenhet\s)([A-ZÅÄÖ].+)', str(area))
    area_other_form = re.search('(ägandeform\s)([A-ZÅÄÖ].+)', str(area))
    area_bstr = re.search('(Bostadsrätt\s)([A-ZÅÄÖ].+)', str(area))

    # replacement group is chosen accordingly
    # for area
    if area_lgh:
        data.at[i, 'omr�de'] = area_lgh.group(2)
    elif area_other_form:
        data.at[i, 'omr�de'] = area_other_form.group(2)
    elif area_bstr:
        data.at[i, 'omr�de'] = area_bstr.group(2)

    # for sale date
    if date_sale_whole:
        month.append(date_sale_whole.group(1))
        year.append(date_sale_whole.group(3))

    # for built date
    if date_built_two_years:
        data.at[i, 'bygg�r'] = date_built_two_years.group(1)
    elif date_built_last_two:
        data.at[i, 'bygg�r'] = date_built_last_two.group(1)
    elif date_built_year_tal:
        data.at[i, 'bygg�r'] = date_built_year_tal.group(1)
    elif date_built_two_tal:
        data.at[i, 'bygg�r'] = '19' + date_built_two_tal.group(1)
    elif date_built_ca:
        data.at[i, 'bygg�r'] = date_built_ca.group(2)
    elif date_built_year:
        data.at[i, 'bygg�r'] = date_built_year.group(1)
    else:
        # we drop any rows with non-matching built date, approximately 15 000
        if date_sale_whole:
            # they need to be removed from the temporary sale date lists
            month.pop()
            year.pop()
        data = data.drop(i)

# dropping sale date after having extracted month and year
data['f�rs�ljningsm�nad'] = month
data['f�rs�ljnings�r'] = year
data = data.drop(['f�rs�ljningsdatum'], 1)

# standard scaling attributes except the ones referencing each area
# TODO: encode month using one-hot or sine/cosine
sc = StandardScaler()
attr = ['rum', 'kvm', 'avgift', 'pris_per_kvm', 'prisf�r�ndring_procent', 'f�rs�ljningspris', 'ut�ngspris',
        'f�rs�ljningsm�nad', 'f�rs�ljnings�r', 'bygg�r']
data[attr] = sc.fit_transform(data[attr])

# dropping attributes deemed to not be accessible given the prediction
data = data.drop(['pris_per_kvm', 'prisf�r�ndring_procent', 'ut�ngspris'], 1)

# making a copy to compare with dimensionally reduced dataset
data_copy = data.copy().drop(['omr�de'], 1)

# one-hot encoding area
data_ec = pd.concat([data, pd.get_dummies(data['omr�de'], prefix='omr')], axis=1).drop('omr�de', 1)

# drop remaining null rows
data = data_ec.dropna()
data_copy = data_copy.dropna()

predict = 'f�rs�ljningspris'

# reducing the number of dimensions from ~4000 to 10 using pca
pca = decomposition.PCA(n_components=10)
X_tra = pca.fit_transform(data.drop([predict], 1))

# two sets of inputs and outputs
X = np.array(X_tra)
X1 = np.array(data_copy.drop([predict], 1))

y = np.array(data[predict])
y1 = np.array(data_copy[predict])

linear = linear_model.LinearRegression()

# split data into testing and training data
# test data is 10% of the total data
# TODO: use 'test_dataset.csv' for testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
linear.fit(x_train, y_train)

# compute and print mean squared error for both models
y_pred = linear.predict(x_test)
acc_pca = metrics.mean_squared_error(y_test, y_pred)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X1, y1, test_size=0.1)
linear.fit(x_train, y_train)

y_pred = linear.predict(x_test)
acc_non_pca = metrics.mean_squared_error(y_test, y_pred)

print('PCA Mean Squared Error : ', acc_pca)
print('Non-PCA Mean Squared Error :', acc_non_pca)
print('Non-PCA Coefficients: \n', + linear.coef_)
