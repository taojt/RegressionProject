# -*-coding: utf-8 -*-
# Create by Jiang Tao on 2016/9/23
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit([[0,0],[1,1],[2,2]],[0,1,2])
reg.coef_