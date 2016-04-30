# author - adarsh (abarik@purdue.edu)
# This is the main program
# Reference : Platt, John. "Sequential minimal optimization: A fast algorithm for training support vector machines." (1998).

from InitializeTraining import *
from ProcessTrainingInput import *
from collections import namedtuple
from random import shuffle
from math import fabs
import numpy as np

# define tolerance
tol = 1e-3
epsilon = 1e-10


# This function will check if a chosen lambda is eligible for optimization
# If chosen example is eligible then it'll also choose second lambda based
# on second heuristic in the Reference paper
def examineExample(i2, static_info, dynamic_info):
	y2 = static_info.target[i2]
	lambda2 = dynamic_info.lambdas[i2]
	E2 = dynamic_info.error[i2]
	r2 = E2*y2
	if ((r2 < -tol and lambda2 < static_info.C) or (r2 > tol and lambda2 > 0)):
		if(len(dynamic_info.nonboundlambdaindices)):
			# We'll be choosing i1 based on the heuristics presented in reference paper
			if dynamic_info.error[i2] >= 0:
				i1 = dynamic_info.error.index(min(dynamic_info.error))
			else:
				i1 = dynamic_info.error.index(max(dynamic_info.error))
			if takeStep(i1, i2, static_info, dynamic_info):
				return 1

		# if previous i2 doesn't give a positive progress then we'll choose
		# lambda from all non-zero and non-c lambdas at random
		temp_list = dynamic_info.nonboundlambdaindices[:]
		shuffle(temp_list)
		for i1 in temp_list:
			if takeStep(i1, i2, static_info, dynamic_info):
				return 1

		# if above loop fails to give any eligible i1 then we'll loop over
		# all possible lambdas
		temp_list = range(static_info.numExample)
		shuffle(temp_list)
		for i1 in temp_list:
			if takeStep(i1, i2, static_info, dynamic_info):
				return 1
	# KKT satisfies for i2
	return 0


# This function will take two lambdas and try to update them and take a positive step
def takeStep(i1, i2, static_info, dynamic_info):
	# do nothing if both lambdas are same
	if (i1 == i2):
		return 0
	lambda1 = dynamic_info.lambdas[i1]
	lambda2 = dynamic_info.lambdas[i2]

	y1 = static_info.target[i1]
	y2 = static_info.target[i2]
	E1 = dynamic_info.error[i1]
	E2 = dynamic_info.error[i2]
	s = y1*y2

	# Lets compute L and H now
	if y1 is not y2:
		L = max(0, lambda2 - lambda1)
		H = min(static_info.C, static_info.C + lambda2 - lambda1)
	else:
		L = max(0, lambda2 + lambda1 - static_info.C)
		H = min(static_info.C, lambda2 + lambda1)

	if (L == H):
		return 0

	k11 = kernel(static_info.training_example[i1], static_info.training_example[i1], static_info)
	k12 = kernel(static_info.training_example[i1], static_info.training_example[i2], static_info)
	k22 = kernel(static_info.training_example[i2], static_info.training_example[i2], static_info)

	eta = k11 + k22 - 2*k12

	if (eta > 0):
		a2 = lambda2 + y2*(E1 - E2)/eta
		if (a2 < L + epsilon):
			a2 = L
		elif (a2 > H - epsilon):
			a2 = H
	else:
		f1 = y1*(E1 + dynamic_info.threshold) - lambda1*k11 - s*lambda2*k12
		f2 = y2*(E2 + dynamic_info.threshold) - s*lambda1*k12 - lambda2*k22
		L1 = lambda1 + s*(lambda2 - L)
		H1 = lambda1 + s*(lambda2 - H)
		Lobj = L1*f1 + L*f2 + 0.5*L1*L1*k11 + 0.5*L*L*k22 + s*L*L1*k12
		Hobj = H1*f1 + H*f2 + 0.5*H1*H1*k11 + 0.5*H*H*k22 + s*H*H1*k12

		if (Lobj < Hobj - tol):
			a2 = L
		elif (Lobj > Hobj + tol):
			a2 = H
		else:
			a2 = lambda2

	if (fabs(a2 - lambda2) < tol*(a2 + lambda2 + tol)):
		return 0

	a1 = lambda1 + s*(lambda2 - a2)

	# We'll avoid any machine epsilon errors
	if (fabs(a1) < epsilon):
		a1 = 0
	elif (fabs(a1) > static_info.C - epsilon):
		a1 = static_info.C

	# Lets update dynamic informations now
	# Lets update the threshold first
	bold = dynamic_info.threshold
	if (a1 != 0 and a1 != static_info.C):
		dynamic_info.threshold = E1 + y1*(a1 - lambda1)*k11 + y2*(a2 - lambda2)*k12 + dynamic_info.threshold
	elif (a2 != 0 and a2 != static_info.C):
		dynamic_info.threshold = E2 + y1*(a1 - lambda1)*k12 + y2*(a2 - lambda2)*k22 + dynamic_info.threshold
	else:
		b1 = E1 + y1*(a1 - lambda1)*k11 + y2*(a2 - lambda2)*k12 + dynamic_info.threshold
		b2 = E2 + y1*(a1 - lambda1)*k12 + y2*(a2 - lambda2)*k22 + dynamic_info.threshold
		dynamic_info.threshold = 0.5*(b1 + b2)

	# Lets update weights now if kernel_type is linear
	if (static_info.kernel_type == 0):
		dynamic_info = updateWeights(static_info, dynamic_info, a1, a2, i1, i2)

	# update nonbounded and nozero lambdas
	if i1 in dynamic_info.nonboundlambdaindices:
		dynamic_info.nonboundlambdaindices.remove(i1)
	if i2 in dynamic_info.nonboundlambdaindices:
		dynamic_info.nonboundlambdaindices.remove(i2)
	if i1 in dynamic_info.nonzerolambdaindices:
		dynamic_info.nonzerolambdaindices.remove(i1)
	if i2 in dynamic_info.nonzerolambdaindices:
		dynamic_info.nonzerolambdaindices.remove(i2)

	if a1 is not 0:
		dynamic_info.nonzerolambdaindices.append(i1)
		if a1 is not static_info.C:
			dynamic_info.nonboundlambdaindices.append(i1)
	if a2 is not 0:
		dynamic_info.nonzerolambdaindices.append(i2)
		if a2 is not static_info.C:
			dynamic_info.nonboundlambdaindices.append(i2)

	# Lets update the error cache now
	dynamic_info = updateErrorCache(static_info, dynamic_info, i1, i2, a1, a2, bold, dynamic_info.threshold)

	# Lets update the lambdas now
	dynamic_info.lambdas[i1] = a1
	dynamic_info.lambdas[i2] = a2
	return 1


# This function updates the weights
def updateWeights(static_info, dynamic_info, a1, a2, i1, i2):
	x1 = static_info.training_example[i1]
	x2 = static_info.training_example[i2]

	v1 = 0
	v2 = 0
	while (v1 < len(x1)):
		id1 = x1[v1].id
		dynamic_info.weights[id1] = dynamic_info.weights[id1] + static_info.target[i1]*(a1 - dynamic_info.lambdas[i1])*x1[v1].value
		v1 = v1+1

	while(v2 < len(x2)):
		id2 = x2[v2].id
		dynamic_info.weights[id2] = dynamic_info.weights[id2] + static_info.target[i2]*(a2 - dynamic_info.lambdas[i2])*x2[v2].value
		v2 = v2+1
	return dynamic_info


# This function updates the error cache
def updateErrorCache(static_info, dynamic_info, i1, i2, a1, a2, bold, b):
	for i in range(static_info.numExample):
		dynamic_info.error[i] = dynamic_info.error[i] + static_info.target[i1]*(a1 - dynamic_info.lambdas[i1])*kernel(static_info.training_example[i], static_info.training_example[i1], static_info) + static_info.target[i2]*(a2 - dynamic_info.lambdas[i2])*kernel(static_info.training_example[i], static_info.training_example[i2], static_info) - b + bold
	return dynamic_info


# We'll define a function for dot product as vectors may be sparse
def fDotProduct(vector_list1, vector_list2):
	if (len(vector_list1)*len(vector_list2) == 0):
		return 0

	dot_product = 0
	v1 = 0
	v2 = 0

	while(v1 < len(vector_list1) and v2 < len(vector_list2)):
		id1 = vector_list1[v1].id
		id2 = vector_list2[v2].id

		if(id1 == id2):
			dot_product = dot_product + vector_list1[v1].value * vector_list2[v2].value
			v1 = v1+1
			v2 = v2+1
		elif(id1 > id2):
			v2 = v2+1
		else:
			v1 = v1+1
	return dot_product


# Lets define the kernel function now
def kernel(x1, x2, static_info):
	# if kernel is linear
	if static_info.kernel_type == 0:
		return fDotProduct(x1, x2)
	# if kernel is polynomial with degree kernel_degree
	elif static_info.kernel_type == 1:
		return (np.power(1 + fDotProduct(x1, x2), static_info.kernel_degree))
	# if kernel is RBF with sigma = kernel_rbf_para
	elif static_info.kernel_type == 2:
		return np.exp(-fabs(fDotProduct(x1, x1) - 2*fDotProduct(x1, x2) + fDotProduct(x2, x2)) * static_info.kernel_rbf_para)


# We need some static information which won't change
StaticInfo = namedtuple('StaticInfo', 'target training_example numExample maxFeature C kernel_type kernel_degree kernel_rbf_para')


# We also need some dynamic information which we will keep updating
class DynamicInfo():
	"""Keeps dynamic information and will be changed at each iteration"""
	def __init__(self, static_info):
		self.weights = [0]*static_info.maxFeature
		self.lambdas = [0]*static_info.numExample
		self.nonzerolambdaindices = []
		self.nonboundlambdaindices = []
		self.error = []
		for i in range(static_info.numExample):
			self.error.append(0 - static_info.target[i])
		self.threshold = 0

if __name__ == '__main__':

	# Lets get input from config file
	# print "Reading configuration file."
	trainingInput = fReadInput()
	print "Reading training input file."
	target, training_example, numExample, maxFeature = fReadTrainingFile(trainingInput.training_file)

	static_info = StaticInfo(target, training_example, numExample, maxFeature, trainingInput.c, trainingInput.kernel_type, trainingInput.kernel_degree, trainingInput.kernel_rbf_para)


	dynamic_info = DynamicInfo(static_info)

	# when we start no lambdas are changed
	numChanged = 0

	# We choose two strategy to choose lambda1
	# we iterate over the entire training set, determining whether each example
	# violates the KKT conditions. If an example violates the KKT conditions then
	# its eligible for optimization.
	# This is done by keeping the below flag one
	examineAll = 1

	# After one pass through the entire training set, the outer loop iterates
	# over all examples whose lambdas are neither 0 nor C. This will be done by
	# setting examineAll = 0 inside while loop
	iteration = 0
	while (numChanged > 0 or examineAll):
		numChanged = 0
		if examineAll:
			# print "examineAll = 1"
			for i in range(static_info.numExample):
				# print "i: ", i
				numChanged = numChanged + examineExample(i, static_info, dynamic_info)
		else:
			for i in dynamic_info.nonboundlambdaindices:
				numChanged = numChanged + examineExample(i, static_info, dynamic_info)
		if examineAll:
			examineAll = 0
		elif (numChanged == 0):
			examineAll = 1
		iteration = iteration + 1
		print "iteration: ", iteration

	# writing to output files
	if (static_info.kernel_type == 0):
		weights = "weights_linear" + str(static_info.C) + "_" + trainingInput.training_output
		np.savetxt(weights, dynamic_info.weights)
	threshold = "threshold_type" + str(static_info.kernel_type) + trainingInput.training_output
	lambdas = "lambdas_type" + str(static_info.kernel_type) + trainingInput.training_output
	nonzerolambdas = "nonzerolambdaind_type" + str(static_info.kernel_type) + trainingInput.training_output
	np.savetxt(threshold, [dynamic_info.threshold])
	np.savetxt(lambdas, dynamic_info.lambdas)
	np.savetxt(nonzerolambdas, dynamic_info.nonzerolambdaindices)

