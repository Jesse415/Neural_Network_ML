#TODO: CLI accepts 7 arguments:
#      1. nInput: Number of neurons in the input layer
#      2. nHidden: number of neurons in the hidden layer
#      3. nOutput: number of neuronsin output layer
#      4. trainset.csv.gz: the training set
#      5. trainset_label.csv.gz: labels associated with the training set
#      6. testset.csv.gz: the test set
#      7. testset_predict.csv.gz: predicted labels for the test set

#TODO: set default value epochs to 30
#      size of mini-batches to 20
#      Learn rate to 3

#TODO: The nonlinearity used in the neural net id sigmoid function:
#      o(x)=1/1+e^-x

#TODO: Main steps for training a neural net with stochastic gradient
#      descent are:
#      1. Assign random initial weights and biases to the neurons.
#         Each initial weight or bias is a random floating-point number
#         drawn from the standard normal distribution:
#         (mean 0 and variance 1)
#      2. For each training example in a mini-batch, use backpropagation
#         to calculate a gradient estimate, which consists of the
#         following steps:
#         1. Feed forward the input to get the activations of the
#            output layer.
#         2. Calculate derivatives of the cost function for that input
#            with respect to the activations of the output layer.
#         3. Calculate the errors for all the weights and biases of the
#            neurons using backpropagation.
#      3. Update weights (and biases) using stochastic gradient descent:
#                    w->w-(n/m)m/sum/i=1(error^w i)
#         where m is the number of training examples in a mini-batch,
#         error^w i is the error of weight w for input i, and n is the
#         learning rate.
#      4. Repeat this for all mini-batches. Repeat the whole process for
#         specified number of epochs. At the end of each epoch evaluate
#         the network on the test data and display its accuracy.

#TODO: Use the quadratic cost function:
#           C(w,b)=1/2n(n/Sum/i=1)?|f(Xi)-Yi|V?^2?
#      where:
#            w: weights
#            b: biases
#            n: number of test instances
#            Xi: i-th test instance vector
#            Yi: i-th test label vector, i.e. if label for Xi is 8, then
#                Yi will be [0,0,0,0,0,0,0,0,1,0]
#            f(x): Label predicted by the neural network for an input x

#TODO: Hand written digits recognition, encode the output (a number
#      between 0 & 9) by using 10 output neurons. Neuron with the
#      highest activation will be taken as the prediction of the network
#      Output number y has to be a list of 10 numbers, all being 0
#      except for the entry at the correct digit.

#TODO: 1. Create neural net of size [784,30,10]. This network is three
#         layers:
#         784 neurons in the input layer
#         30 neurons in the hidden layer
#         10 neurons in the output layer
#      Then train it on the training data with settings:
#         epoch = 30
#         mini-batch size = 20
#         n = 3.0
#      Test code on the test data (TestDigitX.csv.gz) make plots of test
#      accuracy vs epoch. Report max accuracy. Also, run with second
#      test set (TestDigitX2.csv.gz) output to predictions to
#      PredictDigitY2.gz. Upload both PredictDigitY.csv.gz and
#      PredictDigitY2.gz
#      2. Train new neural nets with same settings as in (1) above but
#         with n = 0.001, 0.1, 1.0, 10, 100. Plot test accuracy vs epoch
#         for each n on the same graph. Report max test accuracy for
#         each n. Create a new neural net each time so it starts from
#         scratch.
#      3. Train new neural nets with same settings but with mini-batch
#         sizes = 1, 5, 10, 20, 100. Plot max test accuracy vs
#         mini-batch size. Which achieves the max test accuracy? Which
#         is slowest?
#      4. Try different hyper-parameter settings (number of epochs, n,
#         and mini-batch size, etc.). Report max test accuracy and the
#         settings.

#      Note: Once code is implemented the algorithm, compare its
#            computed values with manual calculated values from part 1
#            to verify correctness. To varify, make sure that the
#            weights and biases output by the learned network after one
#            epoch are the same as those calculated manually in part 1.
#            Train the small network for 3 epochs and output the weights
#            and biases after each epoch. Report this in report. DO THIS
#            BEFORE RUNNING THE NETWORK ON THE MNIST DATASET.
#TODO: Part 3:
#            1. Now replace the quadratic cost function by a cross
#               entropy cost function.
#         C(w,b)=-1/n(n/Sum/i=1)Yi(ln[f(Xi)])+?(1-Yi)ln[1-f(Xi)]?
#            2. Train nerual net with the specifications as Part 2. What
#               is test max accuracy achieved?
#TODO: What to hand in:
#            1. Well commented python scripts and 2 prediction files:
#               PredictDigitY.csv.gz & PredictDigitY2.csv.gz
#            2. Report explaining algorithms & experimental results from
#               Part 1 - Part 3.
#              (a) Part 1 (20 marks)
#              (b) Part 2 (50 marks)
#              (c) Part 3 (20 marks)
#              (d) Report quality (10 marks)
import sys
import math
import random
import collections
import pickle as cPickle
import gzip
import numpy as np

##############################################################
# The list sizes contains number of neurons
# 2 neurons in first layer, 3 neurons in the second 1 in third
# net = Network([2,3,1])
##############################################################
class Network(object):
    def __init__(self,sizes):
        self.numberLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x)
            for x, y in zip(sizes[:1], sizes[1:])]


    def forward(self, i):
        for b, w in zip(self.biases, self.weights):
            i = sigmoid(np.dot(w, i)+b)
        return i

    ####################################
    # Train the network with miniBatchs
    ####################################
    def StochasticGD(self, trainingData, epochs, miniBatchSize, learnRate, testData = None):
        if testData: nTest = len(testData)
        n = len(trainingData)


        ############################################
        # Randomly shuffle training Data then break
        # into Mini Batchs
        ############################################
        for j in xrange(epochs):
            random.shuffle(trainingData)
            miniBatchs = [
                trainingData[x:x+miniBatchSize]
                for x in xrange(0, n, miniBatchSize)]

        #########################################
        # For each Mini Batch update Weights and
        # Biases for ONE gradient descent
        #########################################
        for miniBatch in miniBatchs:
            self.updateMiniBatch(miniBatch, learnRate)

        ###########################################
        # if Test data is there then check network
        # after each epoch. Track progress
        ###########################################
        if testData:
            print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(testData), nTest))
        else:
            print ("Epoch {0} complete".format(j))

        ###################################################
        # Update Weights and Biases using back-propagation
        ###################################################
        def updateMiniBatch(self, miniBatch, learnRate):
            bias = [np.zeros(b.shape) for b in self.biases]
            weight = [np.zeros(w.shape) for w in self.weights]
            for x, y in miniBatch:
                updateBiases, updateWeights = self.backPropagate(x, y)
                bias = [preB + upB for preB, upB in zip(bias, updateBiases)]
                weight = [preW + upW for preW, upW in zip(weight, updateWeights)]

            self.biases = [b-(learnRate/len(miniBatch))*preB
                   for b, preB in zip(self.biases, bias)]
            self.weights = [w-(learnRate/len(miniBatch))*preW
                   for w, preW in zip(self.weights, weights)]

        def backPropagate(self, x, y):
            bias = [np.zeros(b.shape) for b in self.biases]
            weights = [np.zeros(w.shape) for w in self.weights]
            activation = x
            activation = [x]
            storeZ = []
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                storeZ.append(z)
                activation = sigmoid(z)
                activation.append(activation)
            delta = self.cost(activations[-1], y)*sigmoidPrime(storeZ[-1])
            bias[-1] = delta
            weights[-1] = np.dot(delta, activations[-2].tranpose())
            for i in zrange(2, self.numberLayers):
                z = storeZ[-1]
                sigPrime = sigmoidPrime(z)
            delta = np.dot(delta, activations[-i-1].tranpose())
            return (bias, weights)

        def cost(self, output, y):
            return (output-y)

        def evalutate(self, testData):
            testResults = [(np.argmax(self.forward(x)), y)
                for (x, y) in testData]
            return sum(int(x == y) for (x, y) in testResults)

##################
# Sigmoid Fuction
##################
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

############################
# The derivative of sigmoid
############################
def sigmoidPrime(x):
    return sigmoid(x)*(1-sigmoid(x))

def vectorize(x):
    v = np.zeros((10,1))
    v[x] = 1.0
    return v

def main():

    nInput = int(sys.argv[1])
    nHidden = int(sys.argv[2])
    nOutput = int(sys.argv[3])
    trainX = np.loadtxt(sys.argv[4], delimiter=',')
    trainY = np.loadtxt(sys.argv[5], delimiter=',')
    testX = np.loadtxt(sys.argv[6], delimiter=',')
    testY = np.loadtxt('TestDigitY.csv.gz', delimiter=',')
    print(nInput)
    print(nHidden)
    print(nOutput)
    print(trainX)
    print(trainY)

    #predictY = sys.argv[7] if len(sys.argv) > 7 else None


if __name__ == "__main__":
    main()
        #TODO: First step randomize training set, then break up into
        # batchsizes
        #TODO: Second step take the batch sizes (2D vectors) and make into
        # a 1D vector
        #TODO: Take vector and input into NN, Weights are randomised numbers
        # between 0 & 1 non inclusive
        #TODO: Find error = sum of all errors
        #TODO: back propagate only for each mini batch sample
        #TODO: First round of the training set mini batch calculate error
        # ONCE then fo to the next mini batch untill all training set is done
        # which then equals ONE epoch
        #TODO:Second time "TEST SET" then find the classification accuracy
        # then repeat step for new mini batch

        #NOTE: First sample give 10 numbers, (10 numbers - TARGET)^2 = (10
        # new numbers), Sum them, 1 value is given which equals the error

        #NOTE: if epoch goes up in vcalue of the mean square error, overfitting is happening, STOP the training


############################################################
# For back-propagation algorithm learning multilayer network
# look at Chapter 18 page 755 int eh AI a modern approach
############################################################

