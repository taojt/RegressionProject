# -*-coding: utf-8 -*-
# Create by Jiang Tao on 2016/11/24
from sklearn import linear_model
reg = linear_model.Ridge(alpha=.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print('req.coef_:  ',reg.coef_)
print('reg.intercept_:  ', reg.intercept_)

