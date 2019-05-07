import pickle as cPickle
import gzip

def main():

    print("hello world")

    with gzip.open('../2ICT_2nd_Assignment/TestDigitX.csv.gz', 'rb') as file:
        fileContent = file.read()


    trainingData, validationData, testData = cPickle.load(fileContent)

    trainingInputs = [np.reshape(x, (784, 1)) for x in trainingData[0]]
    trainingResults = [vectorize(y) for y in trainingData[1]]
    trainingData = zip(trainingInputs, trainingResults)
    validationInputs = [np.reshape(x, (784,1)) for x in validationData[0]]
    validationData = zip(validationInputs,validationData[1])
    testInput = [np.reshape(x, (784, 1)) for x in testData[0]]
    testData = zip(testInput, testData[1])

    print("Training Data", trainingData, "validationData", validationData, "testData", testData)

    file.close()

    return 0
if __name__ == "__main__":
    main()
