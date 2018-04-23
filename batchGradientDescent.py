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


# This calculates the gradient for this particular time step
def calcBatchGradient(data, weights):
    '''
    Compute gradient of J(w) at w^t. Call it delta(w^t)
    Evaluate the function for each training example to compute the error and construct the gradient
        dJ/dwj = -sum(for each example)(yi - wTxi)xij
    '''

    # This holds our gradient
    gradient = []

    # For every index in our weight vector
    for index in range(0,len(weights)):

        # Initialize something to hold the sum of the following calculations
        sumTotal = 0

        # For every example in the training data
        for example in data.getExampleList():

            yi, xi = getYiXi(example)

            xij = xi[index]

            #sum((yi-w^T xi)*xij)
            sumTotal += (yi - np.dot(np.transpose(weights), np.array(xi)))*xij

        # Have to take the negative of the sum, and place in our new gradient vector
        delta = -sumTotal
        gradient.append(delta)
    return np.array(gradient)


# This runs batch gradient descent on our example, called by the main function
def batchGradientDescent(r, max_t, epsilon):

    # Grab the training and testing data
    trainingData, testingData = getGD_Data()



    # Keep trying for smaller and smaller learning rates until you converge
    while r > .000000000000000000001:

        prevIterWeights, curIterWeights = initializeWeights(trainingData)

        # According to the description for this part, we have 7 attributes and a label
        # Initialize the weight vector
        costs = []

        for t in range(0,max_t):
            # Compute gradient
            gradient = calcBatchGradient(trainingData, curIterWeights)

            # Update w
            prevIterWeights = curIterWeights
            curIterWeights = curIterWeights - r*gradient

            costs.append(calcCost(trainingData, curIterWeights))

            # Check for convergence
            if converges(curIterWeights, prevIterWeights, epsilon):
                finalCost = calcCost(trainingData,curIterWeights)
                plotCost(costs, "Batch Gradient Descent", "Time Steps")
                analyticalWeight = calcAnalyticalWeight(trainingData)
                return str(curIterWeights), str(r), str(finalCost), str(analyticalWeight)

        # If no convergence, then reduce r by half and try again (as long as r > epsilon)
        #plotCost(costs, "Batch Gradient Descent" + str(r))
        r = r/2

    plotCost(costs, "Batch Gradient Descent", "Time Steps")
    return str(curIterWeights), str(r)



if __name__ == "__main__":
    # Here are some parameters that need to be set prior to running Batch Gradient Descent
    r = 0.001  # This is the initial learning rate
    epsilon = .002  # This is the tolerance to determine convergence
    max_t = 100  # This is the max number of weight training iterations we will give to each learning rate
    batchWeights, batchLearningRate, finalCost, analyticalWeight = batchGradientDescent(r, max_t, epsilon)
    print "Weight vector: " + batchWeights
    print "Learning rate: " + batchLearningRate
    print "Final cost: " + finalCost
    print "Analytical Weight: " + analyticalWeight