import xgboost as xgb
'''
Refer:https://github.com/dmlc/xgboost/tree/master/demo/rank
	https://xgboost.readthedocs.io/en/latest/python/python_api.html#callback-api
classify:
regression: 
rank:
xgb_model = xgb.XGBRegressor(**param)
clf = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
 xgb.XGBClassifier(**self.params)
xgb.sklearn.XGBRanker(**params)`
clf.fit(X,y)
'''
def train(param, dtrain, dtest):
	watchlist = [(dtest, 'eval'), (dtrain, 'train')]
	model = xgb.train(param, dtrain, num_round=10, watchlist)
	return model

def save_model(model, model_output_path):
	model.save(model_output_path)

def dump_model(model, model_dump_path):
	model.dump_model(model_dump_path)

def dump_importance(model, importance='weight'):
	scores = model.get_score(impotance=importance)
	scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
	for k,v in scores:
		print("{0:20s}:{1}".format(k,v))

def load_model(model_path):
	bst = xgb.Booster({'nthread':4})
	bst.load_model(model_path)
	return bst

def predict(model, x):
	y = model.predict(x)
	return y

def predict_leaf_feature(model, x):
	index = model.predict(x, pred_leaf=True)
	return index

def generate_xdb_data(x,label=y, weight=w):
	return xgb.DMatrix(x, label=y, weight=w)

def rank():
	train_dmatrix = DMatrix(x_train, y_train)
	train_dmatrix.set_group(group_train)
	params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
          'gamma': 1.0, 'min_child_weight': 0.1,
          'max_depth': 6, 'n_estimators': 4}
	model = xgb.train(params, train_dmatrix, num_boost_round=4,
	                           evals=[(valid_dmatrix, 'validation')])
	pred = model.predict(x_test)

	






