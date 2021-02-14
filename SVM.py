from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np


def metabric_dataset_SVM(dataset_rep = 1):

	df = pd.read_csv("data/metabric_preprocess.csv")
	#df = pd.read_csv("data/metabric_latent.csv")
	df = df.sample(frac=1).reset_index(drop=True)
	#df = dpp.preprocess_dataset("data/METABRIC_RNA_Mutation.csv")

	###getting classes
	target = df.pop("type_of_breast_surgery")

	###split train and test inputs and repeating them
	x_train, x_test = np.split(df.values, [int(0.7 * len(df.values))])
	x_train = np.tile(x_train, (dataset_rep, 1))
	x_test = np.tile(x_test, (dataset_rep, 1))

	###split train and test targets and repeating them
	y_train, y_test = np.split(target.values, [int(0.7 * len(target.values))])
	y_train = np.tile(y_train, dataset_rep)
	y_test = np.tile(y_test, dataset_rep)


	return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = metabric_dataset_SVM(2)
model = svm.SVC()
cv_result = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
	
model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print(cv_result)
print(score)