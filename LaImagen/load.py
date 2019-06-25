import numpy as np
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def init():
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
	loaded_model.load_weights("model_weights.h5")
	print("Loaded Model from disk")
	opt = Adam(lr=0.01)  #Optimizer
	loaded_model.summary()
	#compile and evaluate loaded model
	loaded_model.compile(opt,loss='categorical_crossentropy',metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)


	return loaded_model
