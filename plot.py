plot_decision_boundary = 1
plot_training_points = 1
plot_new_examples = 0

from InitializeTraining import *
from ProcessTrainingInput import *
from MainLearning import StaticInfo, kernel

import ConfigParser
import sys
import numpy as np

import matplotlib.pyplot as pl


# lets read the config and get essential informations
config = ConfigParser.ConfigParser()
if(len(config.read('config')) == 0):
	print "config file is missing from current directory."
	sys.exit()
print "Reading config file..."

try:
	file_to_predict = config.get('PREDICTION', 'FILE_TO_PREDICT')
	prediction_output = config.get('PREDICTION', 'PREDICTION_OUTPUT')
except Exception, e:
	raise e

trainingInput = fReadInput()
target, training_example, numExample, maxFeature = fReadTrainingFile(trainingInput.training_file)
new_examples = fReadPredictionFile(file_to_predict)

static_info = StaticInfo(target, training_example, numExample, maxFeature, trainingInput.c, trainingInput.kernel_type, trainingInput.kernel_degree, trainingInput.kernel_rbf_para)

threshold_file = "threshold_type" + str(static_info.kernel_type) + trainingInput.training_output
lambdas_file = "lambdas_type" + str(static_info.kernel_type) + trainingInput.training_output
nonzerolambdasindices_file = "nonzerolambdaind_type" + str(static_info.kernel_type) + trainingInput.training_output

threshold = np.genfromtxt(threshold_file).tolist()
lambdas = np.genfromtxt(lambdas_file).tolist()
nonzerolambdaindices = np.genfromtxt(nonzerolambdasindices_file).tolist()



training_X_red = []
training_Y_red = []
training_X_blue = []
training_Y_blue = []




for i in range(len(training_example)):
	if target[i] >= 0:
		training_X_red.append(training_example[i][0].value)
		training_Y_red.append(training_example[i][1].value)
	else:
		training_X_blue.append(training_example[i][0].value)
		training_Y_blue.append(training_example[i][1].value)


# plotting decision boundary
if plot_decision_boundary:
	# Lets define grids
	max_X = 1.2* max(training_X_red and training_X_blue)
	max_Y = 1.2* max(training_Y_red and training_Y_blue)
	min_X = 1.2 * min(training_X_red and training_X_blue)
	min_Y = 1.2 * min(training_Y_red and training_Y_blue)

	int_x = (max_X - min_X)/100. 
	int_y = (max_Y - min_Y)/100.
	x = np.arange(min_X, max_X, int_x)
	y = np.arange(min_Y, max_Y, int_y)
	X, Y = np.meshgrid(x, y)
	Z = np.zeros(X.shape)
	for i in range(len(y)):
		for j in range(len(x)):
			example_point = [cFeature(0, x[j]), cFeature(1, y[i])]
			for k in [int(a) for a in nonzerolambdaindices]:
				Z[i,j] = Z[i,j] + static_info.target[k] * lambdas[k] * kernel(example_point, static_info.training_example[k], static_info)
			Z[i,j] = Z[i,j] - threshold
	
	pl.contour(X, Y, Z, [0], color='black')


# plotting 

if plot_training_points:
	pl.scatter(training_X_red, training_Y_red, color='r')
	pl.scatter(training_X_blue, training_Y_blue, color='b')

if plot_new_examples:
	prediction = np.genfromtxt(prediction_output).tolist()
	predict_X_orange = []
	predict_Y_orange = []
	predict_X_green = []
	predict_Y_green = []

	for i in range(len(prediction)):
		if prediction[i] >= 0:
			predict_X_orange.append(new_examples[i][0].value)
			predict_Y_orange.append(new_examples[i][1].value)
		else:
			predict_X_green.append(new_examples[i][0].value)
			predict_Y_green.append(new_examples[i][1].value)

	pl.scatter(predict_X_orange, predict_Y_orange, color='orange')
	pl.scatter(predict_X_green, predict_Y_green, color='g')



pl.show()
