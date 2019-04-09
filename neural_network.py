import matplotlib
import numpy as np

#TODO: Neural network needs 3 layers:
#      1. Input layer
#      2. Hidden layer
#      3. Output layer

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
