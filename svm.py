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
def svm(gamma_0, max_t, N, epsilon, C, gamma_t_choice, d):


    gamma_t = gamma_0

    # Grab the training and testing data
    trainingData, testingData = getGD_Data()


    # According to the description for this part, we have 7 attributes and a label
    # Initialize the weight vector
    prevIterWeights, curIterWeights = initializeWeights(trainingData)

    costs = []



    for t in range(0,max_t):

        random.shuffle(trainingData.exampleList)

        sub_t = 0

        # Compute gradient
        for example in trainingData.getExampleList():

            if gamma_t_choice == 'a':
                gamma_t = gamma_0 / (1 + ((gamma_0 * sub_t) / d))
            else:
                gamma_t = gamma_0 / (1 + sub_t)

            sub_t += 1
            yi, xi = getYiXi(example)


            '''
            gamma_t = learning rate, possible tweaking in between?
            C = hyper parameter that controls tradeoff between a large margin and a small hinge-loss
            N = ????
            '''


            ''' if yi * wTxi <= 1
                    w <- (1-gamma_t)[w0; 0] + gamma_t * C * N * yi * xi (whole vector)
                else
                    w0 <- (w-gamma_t) * w0
            '''

            if yi * np.dot(np.transpose(curIterWeights), xi) <= 1:
                prevIterWeights = curIterWeights
                xi_scalar = gamma_t*C*N*yi
                weight_scalar = (1-gamma_t)
                weightpart = [w * weight_scalar for w in curIterWeights]
                xipart = [x_indv * xi_scalar for x_indv in xi]
                curIterWeights = np.add(np.array(weightpart), np.array(xipart))
            else:
                prevIterWeights = curIterWeights
                curIterWeights = (1 - gamma_t) * curIterWeights

            #costs.append(calcCost(trainingData,curIterWeights))
            # Check for convergence
            if converges(curIterWeights, prevIterWeights, epsilon):
                #finalCost = calcCost(trainingData, curIterWeights)
                #plotCost(costs, "Stochastic Gradient Descent", "Training Examples")
                # The analytical solution is w* = (XX^T)^-1 XY
                return curIterWeights, gamma_t


    return curIterWeights, gamma_t



def testSVM(weight):
    # Grab the training data (code recycled from the Decision Tree Project
    trainingData, testingData = getGD_Data();


    totalTests = 0.0
    totalCorrect = 0.0

    for example in testingData.getExampleList():
        yi, xi = getYiXi(example)
        if yi == 0.0:
            yi = -1.0
        prediction = np.sign(np.dot(np.transpose(weight), np.array(xi)))
        if prediction == np.sign(yi):
            totalCorrect += 1.0
        totalTests += 1.0

    accuracy = totalCorrect / totalTests

    return accuracy

if __name__ == "__main__":



    # Here are some parameters that need to be set prior to running Batch Gradient Descent
    gamma_0 = .1  # This is the learning rate
    epsilon = .000001  # This is the tolerance to determine convergence
    max_t = 50  # This is the max number of weight training iterations we will give to each learning rate
    N = 872 # This is the number of training examples we are using
    C_list = [float(10)/873, float(100)/873, float(300)/873, float(500)/873, float(700)/873]
    d = .15
    gamma_t_choice = 'a'

    for C in C_list:
        svmWeight, svmLearningRate = svm(gamma_0, max_t, N, epsilon, C, gamma_t_choice, d)

        accuracy = testSVM(svmWeight)

        print "-------------------------------------"
        print "With C: " + str(C)
        print "Weight vector: " + str(svmWeight)
        print "Learning rate: " + str(svmLearningRate)
        print "Accuracy: " + str(accuracy)


    print "\n"
    print "\n"
    print "\n"
    print "\n"
    gamma_t_choice = 'b'

    for C in C_list:
        svmWeight, svmLearningRate = svm(gamma_0, max_t, N, epsilon, C, gamma_t_choice, d)

        accuracy = testSVM(svmWeight)

        print "-------------------------------------"
        print "With C: " + str(C)
        print "Weight vector: " + str(svmWeight)
        print "Learning rate: " + str(svmLearningRate)
        print "Accuracy: " + str(accuracy)