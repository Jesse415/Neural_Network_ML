import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class Network():
    def __init__(self, hidden):
        # Initialize biases and weights
        self.weight_one = 0.01*np.random.rand(hidden, 784)
        self.weight_two = 0.01*np.random.rand(10, hidden)
        self.bias_one = 0.01*np.random.rand(hidden, 1)
        self.bias_two = 0.01*np.random.rand(10, 1)

    def forward(self, train_x):
        out_one = sigmoid(
            np.add(np.matmul(self.weight_one, train_x), self.bias_one))
        out_two = sigmoid(
            np.add(np.matmul(self.weight_two, out_one), self.bias_two))
        return out_one, out_two

    def stochasticGD(self, train_x, train_y, epoch, mini_batchSize, learn_rate, test_x, test_y):
        plot_y = []
        # split data into mini batches
        split_x = np.hsplit(train_x, train_x.shape[1] / mini_batchSize)
        split_y = np.hsplit(train_y, train_y.shape[1] / mini_batchSize)
        # For each epoch
        for i in range(epoch):

            # send split_x, split_y to update mini batch
            for x, y in zip(split_x, split_y):
                # forward pass
                out_one, out_two = self.forward(x)

                # back propagate
                # Output layer
                dnet_two = (1-out_two) * out_two * (out_two - y)
                dw_two = np.dot(dnet_two, out_one.T) / mini_batchSize
                db_two = np.sum(dnet_two, axis=1,
                                keepdims=True) / mini_batchSize

                # Hidden layer
                dnet_one = np.multiply(
                    np.dot(self.weight_two.T, dnet_two), out_one,  (1-out_one))
                dw_one = np.dot(dnet_one, x.T) / mini_batchSize
                db_one = np.sum(dnet_one, axis=1,
                                keepdims=True) / mini_batchSize

                # Update weights and biases
                self.weight_one -= learn_rate * dw_one
                self.weight_two -= learn_rate * dw_two
                self.bias_one -= learn_rate * db_one
                self.bias_two -= learn_rate * db_two

            # Do forward pass to get predictions
            predictions = np.argmax(self.forward(test_x)[1], axis=0)
            # Calculate accuracy num of right prediction / total preditions
            acc = sum(int(x == y)
                      for (x, y) in zip(predictions, test_y))/len(predictions)

            print('Epoch {0}: Accuracy {1}'.format(i+1, acc))
            plot_y.append(acc)
        return plot_y

def plot_points(y):
    #z = np.linspace(0,2, 100)
    for i in range(len(y)):
        plt.plot(range(1,31),y[i])

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.title('Mini batch')
    #plt.legend()

def sigmoid(x):
    sum = 1.0/(1.0+np.exp(-x))
    return sum


def main():
    epoch = 30
    mini_batch_size = 20
    learning_rate = 3.0
    n_hidden = 30

    #Hard coded for testing and output to txt
    # Importing data
    print('.........Loading')
    train_x = np.loadtxt('TrainDigitX.csv.gz', delimiter=',')
    train_y = np.loadtxt('TrainDigitY.csv.gz', delimiter=',')
    test_x = np.loadtxt('TestDigitX.csv.gz', delimiter=',')
    test_y = np.loadtxt('TestDigitY.csv.gz', delimiter=',')
    print('Finished importing data')
    print('Running.....')


    orig_stdout = sys.stdout
    file_out = open('Output.txt', 'w')
    sys.stdout = file_out

    matrix_i = np.zeros(shape=(50000, 10))
    for i, value in enumerate(train_y):
        matrix_i[i][int(value)] = 1

    # Shuffle training set and training target once
    indeces = np.random.permutation(50000)
    train_x, train_y = train_x[indeces, :], matrix_i[indeces, :]

    # Transpose train_x and train_y and test
    train_x, train_y, test_x = train_x.T, train_y.T, test_x.T
    total = []
    #uncomment the for loop when adding an array to learning_rate for graphing different variables
    #for i in range(len(learning_rate)):
        # training_data, epochs, mini_batchSize, learn_rate, testData = None
    # Initialize network
    neural_network = Network(n_hidden)
    plot_y = neural_network.stochasticGD(
    train_x, train_y, epoch, mini_batch_size, learning_rate, test_x, test_y)
    total.append(plot_y)
    plot_points(total)#, learning_rate)
    plt.savefig('graph.png')
    plt.close()
    sys.stdout = orig_stdout
    file_out.close()


main()
