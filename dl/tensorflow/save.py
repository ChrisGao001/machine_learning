import tensorflow as tf
import os.path

def save(sess, model_path):
	saver = tf.train.Saver()
	saver.save(sess, model_path)

def restore(model_meta_path):
	sess = tf.Session()
	model_path = os.path.dirname(model_meta_path)
	saver = tf.train.import_meta_graph(model_meta_path)
	saver.restore(sess,  tf.train.latest_checkpoint(model_path))
	return sess
	

