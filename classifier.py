'''
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import datasetpreprocessing as dpp

# load dataset
df = dpp.preprocess_dataset("data/METABRIC_RNA_Mutation.csv")

# moving class attribute at the end
df1 = df.pop("type_of_breast_surgery")
df["type_of_breast_surgery"] = df1

# split into input and output variables
dfX = df.iloc[:, 0:len(df.columns)-1]
dfY = df.iloc[:, len(df.columns)-1]

X = dfX.values
Y = dfY.values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# baseline model
def create_baseline():
# create model
model = Sequential()
model.add(Dense(603, input_dim=603, activation='relu'))
model.add(Dense(128, input_dim=603, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=10, batch_size=1, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf


# load dataset
df = dpp.preprocess_dataset("data/METABRIC_RNA_Mutation.csv")

# moving class attribute at the end
df1 = df.pop("type_of_breast_surgery")
df["type_of_breast_surgery"] = df1

# split into input and output variables
dfX = df.iloc[:, 0:len(df.columns)-1]
dfY = df.iloc[:, len(df.columns)-1]

X = dfX.values
Y = dfY.values

skf = StratifiedKFold(n_split = 5, shuffle = True) 

VALIDATION_ACCURACY = []
VALIDAITON_LOSS = []

save_dir = '/saved_models/'
fold_var = 1

for train_index, val_index in kf.split(np.zeros(n),Y):
training_data = train_data.iloc[train_index]
validation_data = train_data.iloc[val_index]

train_data_generator = idg.flow_from_dataframe(training_data, directory = image_dir,
x_col = "filename", y_col = "label",
class_mode = "categorical", shuffle = True)
valid_data_generator  = idg.flow_from_dataframe(validation_data, directory = image_dir,
x_col = "filename", y_col = "label",
class_mode = "categorical", shuffle = True)

# CREATE NEW MODEL
model = create_new_model()
# COMPILE NEW MODEL
model.compile(loss='categorical_crossentropy',
optimizer=opt,
metrics=['accuracy'])

# CREATE CALLBACKS
checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var), 
monitor='val_accuracy', verbose=1, 
save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# There can be other callbacks, but just showing one because it involves the model name
# This saves the best model
# FIT THE MODEL
history = model.fit(train_data_generator,
epochs=num_epochs,
callbacks=callbacks_list,
validation_data=valid_data_generator)
#PLOT HISTORY
#		:
#		:

# LOAD BEST MODEL to evaluate the performance of the model
model.load_weights("/saved_models/model_"+str(fold_var)+".h5")

results = model.evaluate(valid_data_generator)
results = dict(zip(model.metrics_names,results))

VALIDATION_ACCURACY.append(results['accuracy'])
VALIDATION_LOSS.append(results['loss'])

tf.keras.backend.clear_session()

fold_var += 1


////////////////////////
'''

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import datetime
import datasetpreprocessing as dpp
import pandas as pd
import math


# Set TF_CONFIG
os.environ['TF_CONFIG'] = json.dumps({
	'cluster': {
		'worker': ["10.0.2.4:1717", "10.0.2.5:1818"]
	},
	'task': {'type': 'worker', 'index': 0}
})


# prepare METABRIC dataset
def metabric_dataset_classifier(batch_size, maxtrainsize, maxvalsize):

	df = pd.read_csv("data/metabric_latent.csv")

	#df = dpp.preprocess_dataset("data/METABRIC_RNA_Mutation.csv")

	target = df.pop("type_of_breast_surgery")
	dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
	dataset = dataset.shuffle(buffer_size=1024)

	dataset_size = len(dataset)
	train_size = int(0.7 * dataset_size)
	val_size = int(0.15 * dataset_size)
	test_size = int(0.15 * dataset_size)

	#repeat trainset to allow multiple epochs
	train_rep = int( math.ceil(maxtrainsize/train_size) )
	train_set = dataset.take(train_size).repeat(train_rep).batch(batch_size)

	#repeat valset to allow multiple epochs
	val_rep = int( math.ceil(maxvalsize/val_size) )
	validation_set =dataset.skip(train_size).take(test_size).repeat(val_rep).batch(batch_size)

	test_set = dataset.skip(train_size + test_size).batch(batch_size)

	return train_set, validation_set, test_set


def main():
	# Prepare directories for logs and checkpoint
	log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	checkpoint_dir = "ckpt/"
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)


	#code to execute when lauching the script
	strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
	print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

	# Open a strategy scope and create/restore the model
	with strategy.scope():# create model
		model = Sequential()
		model.add(Dense(32, input_dim=32, activation='relu'))
		model.add(Dense(16, input_dim=32, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		# Compile model
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	callbacks = [
	# This callback saves a SavedModel every epoch
		tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_dir,
			save_weights_only=True,
			monitor='val_accuracy',
			mode='max',
			save_best_only=True,
			save_freq='epoch'),
		tf.keras.callbacks.TensorBoard(log_dir)
	]


	epochs=10
	steps_per_epoch=70
	validation_steps = 30
	global_batch_size = 4

	train_set, validation_set, test_set = metabric_dataset_classifier(
		batch_size=global_batch_size,
		maxtrainsize=epochs*steps_per_epoch*global_batch_size,
		maxvalsize=epochs*validation_steps*global_batch_size
	)

	model.fit(train_set, epochs=epochs, 
		validation_data = validation_set,
		callbacks=callbacks, verbose=1, 
		steps_per_epoch=steps_per_epoch, validation_steps=validation_steps
	)

	result = model.evaluate(test_set, steps=len(test_set))
	print(result)


if __name__ == '__main__':
	main()