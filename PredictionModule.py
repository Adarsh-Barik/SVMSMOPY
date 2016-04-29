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

prediction = []

for example in new_examples:
	Z = 0
	for i in [int(x) for x in nonzerolambdaindices]:
		Z = Z + static_info.target[i] * lambdas[i] * kernel(example, static_info.training_example[i], static_info)
	Z = Z - threshold
	#print Z
	if Z >= 0:
		prediction.append(1)
	else:
		prediction.append(-1)

np.savetxt(prediction_output, prediction)

# plotting 

training_X_red = []
training_Y_red = []
training_X_blue = []
training_Y_blue = []


predict_X_orange = []
predict_Y_orange = []
predict_X_green = []
predict_Y_green = []

for i in range(len(training_example)):
	if target[i] >= 0:
		training_X_red.append(training_example[i][0].value)
		training_Y_red.append(training_example[i][1].value)
	else:
		training_X_blue.append(training_example[i][0].value)
		training_Y_blue.append(training_example[i][1].value)

for i in range(len(prediction)):
	if prediction[i] >= 0:
		predict_X_orange.append(new_examples[i][0].value)
		predict_Y_orange.append(new_examples[i][1].value)
	else:
		predict_X_green.append(new_examples[i][0].value)
		predict_Y_green.append(new_examples[i][1].value)

pl.scatter(training_X_red, training_Y_red, color='r')
pl.scatter(training_X_blue, training_Y_blue, color='b')
pl.scatter(predict_X_orange, predict_Y_orange, color='orange')
pl.scatter(predict_X_green, predict_Y_green, color='g')
pl.show()
