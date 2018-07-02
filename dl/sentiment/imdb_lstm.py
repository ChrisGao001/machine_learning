#coding:utf8

from keras.models import Sequential
from keras.layers import Embedding,Dense,LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence

def save_model(model, model_path):
	model.save(model_path)

def load_model(model_path):
	model = keras.models.load_model(model_path)
	return model

# global parameter definition
max_feature = 20000
maxlen = 80
batch_size = 32

# load dataset
print("load data ...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_feature)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print("build model")
model = Sequential()
model.add(Embedding(max_feature, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("train")
model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data= (x_test, y_test))

model.save("./imdb_lstm_model.h5")

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("test score:", score)
print("acc:", acc)
