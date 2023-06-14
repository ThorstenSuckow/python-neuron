from NeuralNetwork import NeuralNetwork

inputNodes = 3
hiddenNodes = 3
outputNodes = 3

learningRate = 0.3

network = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)


print(network.train([1.0, 0.5, -1.5], [1.0, 0.5, -1.5]))
print(network.query([1.0, 0.5, -1.5]))

