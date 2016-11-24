# -*-coding: utf-8 -*-
# Create by Jiang Tao on 2016/11/24
"""
========================================
Lasso and Elastic Net for Sparse Signals
========================================

Estimates Lasso and Elastic-Net regression models on a manually generated
sparse signal corrupted with an additive noise. Estimated coefficients are
compared with the ground-truth.

"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

np.random.seed(42)
n_samples, n_features = 50, 200
X = np.random.randn(n_samples, n_features)
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]] = 0
y = np.dot(X, coef)

# add noise
y += 0.01 * np.random.normal((n_samples,))

# split data in train set and test set
n_samples = X.shape[0]
X_train, y_train = X[: int(n_samples / 2)], y[:int(n_samples / 2)]
X_test, y_test = X[int(n_samples / 2):], y[int(n_samples / 2):]

alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print('lasso: ', lasso)
print('r^2 on test data: %f ' % r2_score_lasso)

# elastic net
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)

print('enet: ', enet)
print('r^2 on test data : %f ' % r2_score_enet)

plt.plot(enet.coef_, color='lightgreen', linewidth=2, label='Elastic net coefficients')
plt.plot(lasso.coef_, color='gold', linewidth=2, label='Lasso coefficients')
plt.plot(coef, '--', color='navy', label='original coefficients')
plt.legend(loc='best')
plt.title('Lasso R^2: %f , Elastic Net R^2 : %f' % (r2_score_lasso, r2_score_enet))
plt.show()
