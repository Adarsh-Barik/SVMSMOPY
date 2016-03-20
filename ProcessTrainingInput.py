# author - adarsh (abarik@purdue.edu)
# This file processes the user input
# User input are stored in a config file
# Following inputs are expected
# KERNEL_TYPE : 0,1,2
# C : Float value - penalty parameter for margin violation
# TRAINING_FILE : path to the training file
# TRAINING_OUTPUT : path to the output file for training

import ConfigParser
import sys
import os.path


class cTrainingInput():
	"""This class will hold the input information"""
	def __init__(self, kernel_type, kernel_degree, kernel_rbf_para, c, training_file, training_output):
		self.kernel_type = kernel_type
		self.c = c
		self.training_file = training_file
		self.training_output = training_output
		self.kernel_degree = kernel_degree
		self.kernel_rbf_para = kernel_rbf_para


def fReadInput():
	"""This function reads the input provided in config file"""
	config = ConfigParser.ConfigParser()
	if(len(config.read('config')) == 0):
		print "config file is missing from current directory."
		sys.exit()
	print "Reading config file..."
	try:
		kernel_type = int(config.get('INPUT', 'KERNEL_TYPE'))
		c = float(config.get('INPUT', 'C'))
		training_file = config.get('INPUT', 'TRAINING_FILE')
		training_output = config.get('INPUT', 'TRAINING_OUTPUT')
		kernel_degree = int(config.get('POLYKERNEL', 'DEGREE'))
		kernel_rbf_para = float(config.get('RBFKERNEL', 'RBFPARA'))
	except ValueError:
		print "ValueError: Check config file to see if something is weird."
		sys.exit()
	except:
		print "Unexpected error:", sys.exc_info()[0]
		sys.exit()

	# Check if we have sensible input
	if kernel_type not in [0, 1, 2]:
		print "Kernel type is not understood."
		sys.exit()
	if not os.path.isfile(training_file):
		print "TRAINING_FILE- ", training_file, " is missing."
		sys.exit()

	training_input = cTrainingInput(kernel_type, kernel_degree, kernel_rbf_para, c, training_file, training_output)
	return training_input

if __name__ == '__main__':
	trialInput = fReadInput()
	print trialInput.kernel_type, trialInput.c, trialInput.training_file, trialInput.training_output
