from NeuralNetwork import NeuralNetwork

import numpy
import matplotlib.pyplot


inputNodes = 784 # dimension of image is 28 x 28, each pixel is an input node
hiddenNodes = 100
outputNodes = 10 # 0..9
learningRate = 0.3

network = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

trainingDataFile = open("resources/mnist_train_100.csv")
trainingData = trainingDataFile.readlines()
trainingDataFile.close()

####
# Train neural network
####
for record in trainingData:
    all_values = record.split(",")

    # scale input data to 0.01 - 0.1
    scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    targets = numpy.zeros(outputNodes) + 0.01
    targets[int(all_values[0])] = 0.99

    network.train(scaled_input, targets)
    pass


####
# Test neural network
####
testDataFile = open("resources/mnist_test_10.csv")
testData = testDataFile.readlines()
testDataFile.close()

testValues = testData[0].split(",")

testImage = (numpy.asfarray(testValues[1:]) / 255.0 * 0.99) + 0.01

# print result
print(network.query(testImage))

# show testdata as image
image_array = numpy.asfarray(testValues[1:]).reshape((28, 28))
matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")
matplotlib.pyplot.show()
