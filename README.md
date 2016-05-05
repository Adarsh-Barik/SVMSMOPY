# SVM SMO in python

The goal of this project is to solve [soft margin problem](https://en.wikipedia.org/wiki/Support_vector_machine#Soft-margin) in [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine). SVM are used extensively in machine learning. I am using [Sequential Minimal Optimization](https://research.microsoft.com/pubs/69644/tr-98-14.pdf) to solve this problem.


## Input format 
The solver can solve problem in any dimension. It has also been optimized to treat the training examples as sparse vector (if possible). A line in input file looks like this:

`1 0:2 1:3 12:4 123:5` and translates as `target_class feature_id(0):value feature_id:value ....` . Notice that feature_id starts with a `0`.

## Configuration file
Configurations are read from a `config` file which looks like this:

```
[INPUT]
# KERNEL_TYPE : 0 - linear, 1 - Polynomial, 2 - RBF
KERNEL_TYPE = 2
C = 2
TRAINING_FILE = ex8b_mod.txt
TRAINING_OUTPUT = training_output

[POLYKERNEL]
DEGREE = 4

[RBFKERNEL]
RBFPARA = 1000
```

Currently, the program supports three types of kernel (linear, polynomial and gaussian). We specify a penalty parameter (C) which is required for each kernel. Training examples are read from `TRAINING_FILE` and output is written to `TRAINING_OUTPUT` files. If polynomial kernel is chosen then we must specify degree of polynomial kernel and if we choose RBF kernel then we must specify a positive RBFPARA which is essentially gamma in [usual formula](https://en.wikipedia.org/wiki/Radial_basis_function_kernel). 

## Codes
* `MainLearning.py` : main program to be run
* `ProcessTrainingInput.py` : Reads input from config file and checks their sanity
* `InitializeTraining.py` : Reads training examples and stores them
*  `PredictionModule.py`  : Classifying new examples (uses config)
*  `plot.py`  : plots results

## Result
I plotted decision boundary for second example from [here](http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex8/ex8.html). 
* choice of kernel : Linear
![alt tag](https://raw.githubusercontent.com/Adarsh-Barik/SVMSMOPY/master/images/SVM_linear_C_1_0_db_tp.png)
![alt tag](https://raw.githubusercontent.com/Adarsh-Barik/SVMSMOPY/master/images/SVM_linear_C_100_0_db_tp.png)
* choice of kernel : Polynomial
![alt tag](https://raw.githubusercontent.com/Adarsh-Barik/SVMSMOPY/master/images/SVM_Poly_degree_2_C_100_0_db_tp.png)
![alt tag](https://raw.githubusercontent.com/Adarsh-Barik/SVMSMOPY/master/images/SVM_Poly_degree_4_C_100_0_db_tp.png)
* choice of kernel : RBF
![alt tag](https://raw.githubusercontent.com/Adarsh-Barik/SVMSMOPY/master/images/SVM_RBF_gamma_1_0_C_5_0_db_tp.png)
![alt tag](https://raw.githubusercontent.com/Adarsh-Barik/SVMSMOPY/master/images/SVM_RBF_gamma_1000_0_C_1_0_db_tp.png)
* Prediction (Orange belongs to Red class and Green belongs to Blue class)
![alt tag](https://raw.githubusercontent.com/Adarsh-Barik/SVMSMOPY/master/images/SVM_RBF_gamma_1000_0_C_3_0_db_tp_pr.png)

