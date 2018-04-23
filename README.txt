CS 6350 - Spring 2018
HW2 - LMS and Perceptron
Taylor Welker
u0778812

To run:  Simply run the file 'run.sh'.  It is preset to pass in data from the dataset-hw2 folder and return the required information for all of problem 5 in 'output.txt'.  Please don't move these files as their paths are set.  

The output each time run.sh is called is placed in output.txt.  Each section is clearly labeled with the required information. The figures representing the cost of the Batch Gradient Descent algorithm and the Stochastic Gradient Descent algorithm are found in the topmost directory labeled 'Batch Gradient Descent.png' and 'Stochastic Gradient Descent.png'.  These are updated every time batchGradientDescent.py and stochasticGradientDescent.py are run (or called by run.sh).


Files:

--Python Files--

Files executed by 'run.sh':
batchGradientDescent.py
stochasticGradientDescent.py
perceptron.py
votedPerceptron.py
averagePerceptron.py

Supporting files:
Attributes.py:  Used to help keep track of the values each attribute can take.  These are the features of a given dataset
Dataset.py: Used to create Datasets that take information from the input csv files and organizes them from the Gradient Descent and Perceptron algorithms.
Example.py: Class object to hold instances of examples within the given dataset.
Utility.py: Contains functions to organize and extract data that are common to most or all of the critical python files.

Extra files:
pr2.py: used to solve problem 2 - Linear Regression.  Can be executed using 'python pr2.py'.  Output goes to console

--Data Files--
*inputs
dataset-hw2: as described in the project description
pr2Training.csv: used to solve problem 2 - Linear Regression.

*outputs
output.csv: where the data from running main.py goes
Batch Gradient Descent.png: holds image of plot relating time steps to cost function
Stochastic Gradient Descent.png: holds image of plot relating training examples to cost function.
