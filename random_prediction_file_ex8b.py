from random import uniform, choice
generate_training_examples = 0
generate_new_examples = 1

if generate_new_examples:
	myfile = open("predict_500", 'w')
	for i in range(500):
		myfile.write("0:" + str(uniform(-0.7, 0.4)) + " 1:" + str(uniform(-0.6, 0.6)) + "\n")
	myfile.close()

if generate_training_examples:
	myfile1 = open("training_100", 'w')
	for i in range(100):
		myfile1.write(str(choice([-1, 1])) + " 0:" + str(uniform(-0.7, 0.4)) + " 1:" + str(uniform(-0.6, 0.6)) + "\n")
	myfile1.close()
