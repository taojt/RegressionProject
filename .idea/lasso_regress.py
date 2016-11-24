# -*-coding: utf-8 -*-
# Create by Jiang Tao on 2016/11/24
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
reg.predict([[1,1]])
print('reg.predict([[1,1]]): ', reg.predict([[1,1]]))