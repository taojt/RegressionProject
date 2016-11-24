# -*-coding: utf-8 -*-
# Create by Jiang Tao on 2016/9/30
# print(__doc__)
# code source: Jiangtao

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes datasets
diabetes = datasets.load_diabetes()
# print(diabetes)
diabetes_X = diabetes.data[:, np.newaxis, 2]

# split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()

regr.fit(diabetes_X_train, diabetes_y_train)

# the coeff
print('Coefficients: ', regr.coef_)

print('Mean squared error: %.2f' % np.mean(regr.predict(diabetes_X_test) - diabetes_y_test) ** 2)

print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# plot output
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)
plt.xticks()
plt.yticks()
plt.show()
