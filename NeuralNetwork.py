# neural network class definition

class NeuralNetwork:

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # number of nodes for each layer input, hidden and output
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        #learning rate
        self.lr = learningRate
        pass

    # train the neural network
    def train(self):
        pass

    # query the neural network - get the answer of output nodes for
    # an input
    def query(self):
        pass

    pass
