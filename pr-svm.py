import copy as copy

from Utility import *
from Dataset import *
def calcStochasticSVM():

    '''
    learning rate schedule (gamma_t) = gamma_0/(1+(gamma_0 * t / d))
    C_list = {10/873, 100/873, 300/873, 500/873, 700/873}
    schedule yt = y0/(1+t)
    '''
    trainingData = Dataset('pr2Training.csv')
    curIterWeights, prevIterWeights = initializeWeights(trainingData)

    gradients = []
    learning_rate = 0.01
    C = 1
    N = 3

    for example in trainingData.getExampleList():

        yi, xi = getYiXi(example)

        if yi * np.dot(np.transpose(curIterWeights), xi) <= 1:
            xi_scalar = learning_rate * C * N * yi
            xi_scalar
            weight_scalar = (1 - learning_rate)
            weightpart = [w * weight_scalar for w in curIterWeights]
            xipart = [x_indv * xi_scalar for x_indv in xi]
            curIterWeights = np.add(np.array(weightpart), np.array(xipart))
            gradient = np.subtract(curIterWeights, C*N*yi*np.array(xi))
            gradients.append(gradient)
        else:
            curIterWeights = (1 - learning_rate) * curIterWeights
            gradients.append(curIterWeights)


        learning_rate = learning_rate / 2.0


    print "gradient: "
    for gradient in gradients:
        print str(gradient)


if __name__ == "__main__":
    calcStochasticSVM()