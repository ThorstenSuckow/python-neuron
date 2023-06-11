# neural network class definition
import numpy;
class NeuralNetwork:

    # hnodes = hidden nodes
    # inodes = input nodes
    # onodes = output_nodes

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # number of nodes for each layer input, hidden and output
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        #learning rate
        self.lr = learningRate

        #create weights
        self.weights()
        pass

    # generate weight marices
    def weights(self):

        # O = Output Nodes
        # I = Input Nodes
        # H = Hidden Nodes

        # link weight matrices:
        # -0.5 in each statement takes care of creating  ranges[-0.5, 0.5]

        # weights for input nodes and hidden nodes
        # matrix is hnodes x inodes since weight matrix is multiplicated with input "vectors", resulting in values
        # for the hidden layer:
        #                    a                      au
        #   I: u     w_i_h:  b    H: (w_i_h * I =)  bu
        #                    c                      cu
        #
        #  w_i_h = 3 x 1
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5

        # weights for hidden nodes and output nodes
        # matrix is onodes x hnodes since H is multiplicated with weight matrix, resulting in values for O
        #
        #      au
        #  H:  bu     w_h_o:  x y z      O: (H * w_h_o =)  x * au + y * bu + z * cu
        #      cu
        #
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        pass

    # train the neural network
    def train(self):
        pass

    # query the neural network - get the answer of output nodes for
    # an input
    def query(self):
        pass

    pass
