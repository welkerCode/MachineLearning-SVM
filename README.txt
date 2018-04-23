CS 6350 - Spring 2018
HW4 - Stochastic Subgradient SVM
Taylor Welker
u0778812

To run:  Simply run the file 'run.sh'.  It is preset to pass in data from the dataset-hw4 folder and return the required information for all of problem 3 in the console.  Please don't move these files as their paths are set.  

The output each time run.sh is called is placed in the console.  Each section is clearly labeled with the required information.


Files:

--Python Files--

Files executed by 'run.sh':
svm.py

Supporting files:
Attributes.py:  Used to help keep track of the values each attribute can take.  These are the features of a given dataset
Dataset.py: Used to create Datasets that take information from the input csv files and organizes them from the Gradient Descent and Perceptron algorithms.
Example.py: Class object to hold instances of examples within the given dataset.
Utility.py: Contains functions to organize and extract data that are common to most or all of the critical python files.

Extra files:
pr-svm.py: used to solve problem 2c.  Can be executed using 'python pr-svm.py'.  Output goes to console

--Data Files--
*inputs
dataset-hw2: as described in the project description
pr2Training.csv: used to solve problem 2 - Linear Regression.

*outputs
all output goes to the console
