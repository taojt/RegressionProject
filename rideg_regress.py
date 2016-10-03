# -*-coding: utf-8 -*-
# Create by Jiang Tao on 2016/9/30
from sklearn import linear_model
reg = linear_model.Ridge(alpha=0.5)
reg.fit([[0,0],[0,0],[1,1]],[0,.1,1])
print('reg.coef_: ', reg.coef_)
print('reg.intercept_:', reg.intercept_)