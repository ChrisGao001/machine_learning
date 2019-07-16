import pandas as pd
import numpy as np

def label_encoder(df, colum):
	from sklearn.preprocessing import LabelEncoder
	new_colum = "{0}_label".format(colum)
	encoder = LabelEncoder()
	labels = encoder.fit_transform(df[colum])
	id2label = {index: label for index, label in enumerate(encoder.classes_)}
	df[new_colum] = labels
	return id2label

def label_encoderV2(df, colum):
	new_colum = "{0}_label".format(colum)
	keys = np.unique(df[colum]).tolist()
	keys = sorted(keys)
	key2id = dict([(v,k) for (k,v) in enumerate(keys,start=1)])
	df[new_colum] = df[colum].map(key2id)
	return key2id

def onehot_encoder(df, colum):
	from sklearn.preprocessing import OneHotEncoder
	encoder = OneHotEncoder(sparse=False)
	values = encoder.fit_transform(df[colum].reshape(-1,1))
	columns = ["{0}_{1}".format(colum,label) for label in list(encoder.active_features_) ]
	features = pd.DataFrame(values, columns=columns)
	df = pd.concat([df, features], axis=1)
	return df

def onehot_encoderV2(df, colum):
	onehot_feature = pd.get_dummies(df[colum], prefix=colum)
	#onehot_feature.rename(columns=lambda x:"{0}_{1}".format(colum,x), inplace=True)
	df = pd.concat([df, onehot_feature], axis=1)
	return df

def hash_encoder(df, colum, n_features=4):
	from sklearn.feature_extraction import FeatureHasher
	encoder = FeatureHasher(n_features=n_features, input_type='string',dtype=np.int32,non_negative=True)
	feature = encoder.fit_transform(df[colum].map(lambda x:str(x))).toarray()
	features = pd.DataFrame(feature, columns=["{0}_hash_{1}".format(colum, i) for i in range(n_features)])
	df = pd.concat([df, features], axis=1)
	return df
	
	
	
	
	
	
	
if __name__ == "__main__":
	df = pd.DataFrame({"name" : ["a", "b", "c"], "age":[1,3,2]})
	id2label = label_encoderV2(df, "name")
	df = onehot_encoderV2(df, "name")
	df = hash_encoder(df, "age")
	print(id2label)
	print(df.head())
	
	
