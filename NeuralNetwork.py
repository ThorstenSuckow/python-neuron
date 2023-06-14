import numpy

# scipy special has sigmoid function expit()
import scipy.special

# neural network class definition
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
        self.initWeights()

        #shortcut for sigmoid as activation function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # generate weight marices
    def initWeights(self):

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
        # see query()
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5

        # weights for hidden nodes and output nodes
        # matrix is onodes x hnodes since H is multiplicated with weight matrix, resulting in values for O
        #
        #      au
        #  H:  bu     w_h_o:  x y z      O: (H * w_h_o =)  x * au + y * bu + z * cu
        #      cu
        #
        # see query()
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        # alternatively use normal distribution with
        # 1st argument 0.0 is the mean of the distribution, second argument is the standard deviation
        # see https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
        # self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.hnodes, self.inodes))

        pass

    # train the neural network
    def train(self, inputs_list, targets_list):

        # convert targets list to 2d array
        targets = numpy.array(targets_list, ndmin=2).T


        # convert inputs to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        #calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        ##
        # Weight Calculations
        ##
        # error is the (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update weights for the links between hidden and output layer
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # update weights for the links between input and hidden layer
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # query the neural network - get the answer of output nodes for
    # an input
    def query(self, inputs_list):

        # convert inputs to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        #calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        pass

    pass
