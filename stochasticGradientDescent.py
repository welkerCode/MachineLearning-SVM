'''
Implement the batch gradient descent alg and tune r to find convergence
Convergence can be found by examinging the norm of the weight vector difference ||w_t-w_(t-1)|| at each step t.
    If ||w_t-w_(t-1)|| < 1e-6, then converges.
Weight vector is initialized to 0
Start with high r, and decrease by half each time until you find convergence
Report weight vector and learning rate.  Record cost function value of the training data at each step
Draw figure showing how cost function changes with time steps.  Use final weight vector to calc the cost function value of the test data

'''

from Dataset import *
from Utility import *
import numpy as np


# This runs batch gradient descent on our example, called by the main function
def stochasticGradientDescent(r, max_t, epsilon):


    # Grab the training and testing data
    trainingData, testingData = getGD_Data()

    # Keep trying for smaller and smaller learning rates until you converge
    while r > .0000000000001:

        # According to the description for this part, we have 7 attributes and a label
        # Initialize the weight vector
        prevIterWeights, curIterWeights = initializeWeights(trainingData)

        costs = []

        for t in range(0,max_t):

            # Compute gradient
            for example in trainingData.getExampleList():
                newWeights = []

                yi, xi = getYiXi(example)

                for index in range(0,len(curIterWeights)):
                    xij = xi[index]
                    newWeights.append(curIterWeights[index] + r*(yi - np.dot(np.transpose(curIterWeights), np.array(xi)))*xij)
                prevIterWeights = curIterWeights
                curIterWeights = np.array(newWeights)

                costs.append(calcCost(trainingData,curIterWeights))
                # Check for convergence
                if converges(curIterWeights, prevIterWeights, epsilon):
                    finalCost = calcCost(trainingData, curIterWeights)
                    plotCost(costs, "Stochastic Gradient Descent", "Training Examples")
                    # The analytical solution is w* = (XX^T)^-1 XY
                    return str(curIterWeights), str(r), str(finalCost)

        # If no convergence, then reduce r by half and try again (as long as r > epsilon)
        r = r/2

    return str(curIterWeights), str(r), str(finalCost)



if __name__ == "__main__":
    # Here are some parameters that need to be set prior to running Batch Gradient Descent
    r = .25  # This is the learning rate
    epsilon = .000001  # This is the tolerance to determine convergence
    max_t = 100  # This is the max number of weight training iterations we will give to each learning rate
    stochasticWeights, stochasticLearningRate, finalCost = stochasticGradientDescent(r, max_t, epsilon)
    print "Weight vector: " + stochasticWeights
    print "Learning rate: " + stochasticLearningRate
    print "Final cost: " + finalCost