from InitializeTraining import *
from ProcessTrainingInput import *
from MainLearning import StaticInfo, kernel

import ConfigParser
import sys
import numpy as np


# lets read the config and get essential informations
config = ConfigParser.ConfigParser()
if(len(config.read('config')) == 0):
	print "config file is missing from current directory."
	sys.exit()
# Get the configuration specific to PREDICTION module
try:
	file_to_predict = config.get('PREDICTION', 'FILE_TO_PREDICT')
	prediction_output = config.get('PREDICTION', 'PREDICTION_OUTPUT')
except Exception, e:
	raise e

# Get the training configuration
trainingInput = fReadInput()
target, training_example, numExample, maxFeature = fReadTrainingFile(trainingInput.training_file)
new_examples = fReadPredictionFile(file_to_predict)

static_info = StaticInfo(target, training_example, numExample, maxFeature, trainingInput.c, trainingInput.kernel_type, trainingInput.kernel_degree, trainingInput.kernel_rbf_para)

# Generate model based on the output files
threshold_file = "threshold_type" + str(static_info.kernel_type) + trainingInput.training_output
lambdas_file = "lambdas_type" + str(static_info.kernel_type) + trainingInput.training_output
nonzerolambdasindices_file = "nonzerolambdaind_type" + str(static_info.kernel_type) + trainingInput.training_output

threshold = np.genfromtxt(threshold_file).tolist()
lambdas = np.genfromtxt(lambdas_file).tolist()
nonzerolambdaindices = np.genfromtxt(nonzerolambdasindices_file).tolist()


# Make predictions
prediction = []


for example in new_examples:
	Z = 0
	for i in [int(a) for a in nonzerolambdaindices]:
		Z = Z + static_info.target[i] * lambdas[i] * kernel(example, static_info.training_example[i], static_info)

	Z = Z - threshold
	# print Z
	if Z >= 0:
		prediction.append(1)
	else:
		prediction.append(-1)

# Save predicted classes to output file
np.savetxt(prediction_output, prediction)
