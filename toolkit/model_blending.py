#!/usr/bin/python
'''
Desc:   model blending 
Date:   20180718
Author: Yumin Gao
Revision:
	v1.0	20180718
Refer:
https://github.com/emanuele/kaggle_pbr
'''

from __future__ import division
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

class EnsembleModel(object):
	def __init__(self, n_folds=3, models=None, shuffle=False):
		self.n_folds_ = n_folds
		self.shuffle_ = shuffle
		if models:
			self.models_ = models
		else:
			self.models_ = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
				RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
				ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
		        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
		        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

	def fit(self, x, y, test_x):
		if self.shuffle_:
			idx = np.random.permutation(y.size)
		    x = x[idx]
			y = y[idx]

		skf = StratifiedKFold(y, self.n_folds_)
		num_model = len(self.models_)
		print(x.shape[0])
		blend_train = np.zeros((x.shape[0], num_model))
		blend_test = np.zeros((x_test.shape[0], num_model))
		for i,model in enumerate(self.models_):
			blend_test_kfold = np.zeros((test_x.shape[0], len(skf)))
			for j, (train, test) in enumerate(skf):
				x_train = x[train]
				y_train = y[train]
				x_valid = x[test]
				y_valid = y[test]
				model.fit(x_train, y_train)
				blend_train[test, i] = model.predict_proba(x_valid)[:,1]
				blend_test_kfold[:,j] = model.predict_proba(x_test)[:,1]
			# compute the mean for kfold model predict prob
			blend_test[:,i] = blend_test_kfold.mean(axis=1)

		print("begin to blending model ...")
		clf = LogisticRegression()
		clf.fit(blend_train, y)
		y_test = clf.predict_proba(blend_test)[:,1]
		return y_test
if __name__ == "__main__":
	x = np.random.rand(33,5)
	y = np.random.randint(2,size=33)
	print(x.shape)
	print(y.shape)
	x_test = np.random.rand(3,5)

	model = EnsembleModel()
	y_test = model.fit(x, y, x_test)
	print y_test

				
			


