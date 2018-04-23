import copy as copy

from Utility import *
from Dataset import *
from batchGradientDescent import calcBatchGradient

def calcGradient():
    trainingData = Dataset('pr2Training.csv')

    wa = [0.0,0.0,0.0,0.0]
    wb = [-1.0,1.0,-1.0,-1.0]
    wc = [.5,-.5,.5,1]

    gradient_A = calcBatchGradient(trainingData, wa)
    gradient_B = calcBatchGradient(trainingData, wb)
    gradient_C = calcBatchGradient(trainingData, wc)

    print "Gradient a: " + str(gradient_A)
    print "Gradient b: " + str(gradient_B)
    print "Gradient c: " + str(gradient_C)
    print "Analytical optimized weight: " + str(calcAnalyticalWeight(trainingData))

def calcStochasticGradient():

    r = 0.1
    w = [0.0, 0.0, 0.0, 0.0]
    weights = []
    weights.append([0.0, 0.0, 0.0, 0.0])
    newWeight = [0.0, 0.0, 0.0, 0.0]
    gradients = []
    trainingData = Dataset('pr2Training.csv')

    # Compute gradient
    for example in trainingData.getExampleList():
        gradient = []
        lastWeight = copy.deepcopy(newWeight)
        yi, xi = getYiXi(example)

        for index in range(0, len(w)):
            xij = xi[index]
            gradient.append(((yi - np.dot(np.transpose(w), np.array(xi))) * xij))
            newWeight[index] = lastWeight[index] + r*gradient[index]
        gradients.append(gradient)
        weights.append(copy.deepcopy(newWeight))



    print "weights: "
    for weight in weights:
        print str(weight)
    print "gradient: "
    for gradient in gradients:
        print str(gradient)


if __name__ == "__main__":
    calcGradient()
    calcStochasticGradient()
