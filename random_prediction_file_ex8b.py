from random import uniform
from numpy import savetxt

myfile = open("predict_ex8b_500",'w')
for i in range(500):
	myfile.write("0:" + str(uniform(-0.5, 0.4)) + " 1:" + str(uniform(-0.5, 0.4)) + "\n")

myfile.close()