# author : adarsh (abarik@purdue.edu)
# This module is used to intialize various parameters to start training


class cFeature():
	"""a feature consists of an id and a nonzero value"""
	def __init__(self, id, value):
		self.id = int(id)
		self.value = float(value)


def fReadTrainingFile(training_file):
	my_file = open(training_file, 'r')
	target_list = []
	data_item_list = []
	number_of_nonzero_list = []
	error_list = []
	maxFeature = 0
	numExample = 0

	for line in my_file:
		line_as_list = line.split()
		target_list.append(int(line_as_list[0]))
		error_list.append(0 - int(line_as_list[0]))
		feature_list = []
		for i in range(1, len(line_as_list)):
			feature_list.append(cFeature(line_as_list[i].split(':')[0], line_as_list[i].split(':')[1]))
		if (int(line_as_list[-1].split(':')[0]) > maxFeature):
			maxFeature = int(line_as_list[-1].split(':')[0])

		data_item_list.append(feature_list)
		number_of_nonzero_list.append(len(line_as_list)-1)
		numExample = numExample + 1

	my_file.close()
	return target_list, data_item_list, numExample, maxFeature+1


# def fInitializeTraining(maxFeature, numExample):
# 	print "Iniatializing.."

# 	weight_list = [0]*maxFeature
# 	lambda_list = [0]*numExample
# 	nonbound_lambda_list = [0]*numExample

# 	return weight_list, lambda_list, nonbound_lambda_list


if __name__ == '__main__':
	target, data_item, numExample, maxFeature = fReadTrainingFile("training_file")
