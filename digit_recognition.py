import random
import math
import fileinput
import bitmap
from mnist import MNIST

# collect all the training and testing images from the mnist database
mndata = MNIST('samples')
images, labels = mndata.load_training()
inputs, outputs = mndata.load_testing()
# convert all the grayscale values to floats 0 to 1, and attach the labels
training = [[[images[i][j] / 255 for j in range(len(images[i]))], labels[i]] for i in range(10000)]
testing = [[[inputs[i][j] / 255 for j in range(len(inputs[i]))], outputs[i]] for i in range(10000)]


class NeuralNetworkNode:
    """Represents a single neuron in a neural network"""

    def __init__(self, inputs, weights, function):
        """NeuralNetworkNode(int inputs, float[inputs + 1] weights, Any function) -> NeuralNetworkNode
        Creates a new neuron
        inputs: the number of inputs it takes
        weights: the initial weights of each input (including the bias which is the last weight)
        function: the activation function"""
        self.inputs = [0] * inputs  # initialize the list of inputs
        if len(weights) == inputs + 1:  # check if the given weights are the correct length
            self.weights = weights
        else:
            raise ValueError("The length of weights must be inputs + 1")
        self.function = function

    def evaluate(self, inputs):
        """NeuralNetworkNode.evaluate(float[] inputs) -> float
        Given an array of inputs with the length given in the constructor, return the activation of that the node"""
        if len(inputs) == len(self.inputs):  # make sure the inputs array is the right length
            self.inputs = inputs.copy()
            # return the sum of the inputs multiplied by the weights put through the activation function
            return self.function(
                sum([self.inputs[i] * self.weights[i] for i in range(len(self.inputs))]) + self.weights[-1])
        else:
            raise ValueError("The length of inputs must be the length given on construction")

    def adjust_weights(self, adjustments):
        """NeuralNetworkNode.adjust_weights(float[] adjustments) -> None
        Given an array of adjustments the same length as this neuron's weights, adjust the weights by their corresponding adjustments"""
        if len(self.weights) == len(adjustments):  # verify that adjustments is the correct length
            self.weights = [self.weights[i] + adjustments[i] for i in range(len(self.weights))]
        else:
            raise ValueError("The length of adjustments must be one more than the length given at construction")

    def get_weights(self):
        """NeuralNetworkNode.get_weights() -> float[]
        Return an array of the weights for this neuron (including the bias as the last weight)"""
        return self.weights

    def set_weight(self, index, weight):
        """NeuralNetworkNode.set_weight(int index, float weight) -> None
        Set the weight at the given index to the given weight"""
        if len(self.weights) > index:  # check whether or not the given index is within the arrray
            self.weights[index] = weight
        else:
            raise IndexError("List index out of range, please specify a value less than or equal to the number of inputs for this node")

    def copy(self):
        """NeuralNetworkNode.copy() -> NeuralNetworkNode
        create a copy of the node"""
        return NeuralNetworkNode(len(self.inputs), self.weights.copy(), self.function)


class NeuralNetwork:
    """Represents a neural network of NeuralNetworkNodes"""

    def __init__(self, layers, lengths, function):
        """NeuralNetwork(int layers, int[layers] lengths, Any function) -> NeuralNetwork
        Create a new neural network with random weights
        layers: the number of layers in the network
        lengths: the amount of nodes in each layer of the netework
        function: the activation function of all the neurons in the network"""
        if len(lengths) == layers:  # check if the length of lengths is correct
            # save all given parameters as attributes so the network can easily be cloned
            self.layers = layers
            self.lengths = lengths
            self.function = function
            # generate a 2D array of all the NeuralNetworkNodes
            self.neurons = []
            last = [1]
            for i in range(layers):
                if i == 0:
                    self.neurons.append([NeuralNetworkNode(1, [1, 0], function) for j in range(lengths[i])])
                else:
                    self.neurons.append(
                        [NeuralNetworkNode(len(last), [random.uniform(-1, 1) for k in range(len(last) + 1)], function) for j in range(lengths[i])])
                last = self.neurons[-1]
        else:
            raise ValueError("The length of lengths must be the number of layers")

    def evaluate(self, inputs):
        """NeuralNetwork.evaluate(float[] inputs) -> float[]
        Calculate the output of the neural network given a list of inputs the same length as the base layer"""
        if len(inputs) == len(self.neurons[0]):  # check if inputs is the same length as the base layer
            last = inputs.copy()
            for i in range(self.layers):  # calculate the value of each layer of neurons
                # using the outputs of the previous layer as the inputs to the next
                if i == 0:
                    last = [self.neurons[i][j].evaluate([last[j]]) for j in range(self.lengths[i])]
                else:
                    last = [self.neurons[i][j].evaluate(last) for j in range(self.lengths[i])]
            return last
        else:
            raise ValueError(
                "The length of inputs must be equal to the length of the first layer (given on construction)")

    def copy(self):
        """NeuralNetwork.copy() -> NeuralNetwork
        Create a new neural network with the same layers but different weights"""
        return NeuralNetwork(self.layers, self.lengths, self.function)

    def clone(self):
        """NeuralNetwork.clone() -> NeuralNetwork
        Create a new neural network with the same layers and weights"""
        clone = self.copy()
        clone.neurons = []
        for layer in self.neurons:
            nodes = []
            for neuron in layer:
                nodes.append(neuron.copy())
            clone.neurons.append(nodes.copy())
        return clone

    def adjust_weights(self, mutability):
        """NeuralNetwork.adjust_weights(float mutability) -> None
        Adjust the weights of all neurons in the network a random float between -mutability and mutability"""
        for i in range(1, len(self.neurons)):
            for j in range(len(self.neurons[i])):
                self.neurons[i][j].adjust_weights([random.uniform(-mutability, mutability) for k in range(len(self.neurons[i - 1]) + 1)])

    def reproduce(self, mutability):
        """NeuralNetwork.reproduce(mutability) -> NeuralNetwork
        Create a clone of the neural network but with random changes made to the weights, determined by mutability"""
        child = self.clone()
        child.adjust_weights(mutability)
        return child

    def evolve(self, mutability, iterations, population, selectivity, data, tests_per_generation, is_parent=False, load_nets=False):
        """NeuralNetwork.evolve(float mutability, int generations, int population, float selectivity, [float[], int][] data, int tests_per_gen, bool is_parent=False, bool load_nets=False) -> None
        Evolve the neural network using a genetic algorithm
        mutability: how much each network differs from its parent
        generations: the number of generations it will run
        population: the number of networks that are tested
        selectivity: the fraction of networks that survive every generation
        data: a list of 2 element lists of a list of floats and an int of training data with matching labels
        tests_per_gen: the amount of training images to show the network each generation
        is_parent: whether or not to use self as a parent to the initial population
        load_nets: whether or not to load the initial population from saved networks"""
        nets = []
        for i in range(population):  # fill the population
            if is_parent:  # with children of the current network
                nets.append(self.reproduce(mutability))
            elif load_nets:  # networks loaded from the files
                copy = self.copy()
                copy.load_from_file("nets\\net%d.txt" %i)
                nets.append(copy)
            else:  # or randomly generated networks
                nets.append(self.copy())
        for i in range(iterations):  # repeat for the specified number of generations
            print("Generation %d" % i)  # print the generation number
            losses = [0] * population  # initialize an empty array of losses
            for j in range(tests_per_generation):  # for the selected amount of training images
                print("\tTraining Image %d" % j)  # print what number image we are on
                test = random.choice(data)  # choose a random image
                for k in range(population):  # test every network in the population
                    response = nets[k].evaluate(test[0])  # generate their response
                    # generate the expected response
                    expected = [0] * 10
                    expected[test[1]] = 1
                    # calculate the loss
                    loss = 0
                    for l in range(len(response)):
                        loss += (response[l] - expected[l]) ** 2
                    losses[k] += loss  # update the sum of losses
            # calculate average losses
            for j in range(population):
                losses[j] /= len(data)
            self.neurons = nets[losses.index(min(losses))].neurons.copy()  # take on the network of the best
            for j in range(int((1 - selectivity) * population)):  # kill the selected amount of the population with the highest loss
                index = losses.index(max(losses))
                nets.pop(index)
                losses.pop(index)
            # have the survivors reproduce
            children = []
            for j in range(len(nets)):
                for k in range(int(1 / selectivity - 1)):
                    children.append(nets[j].reproduce(mutability))
            nets += children
        for j in range(population):  # save the evolved population to files
            nets[j].save_to_file("nets\\net%d.txt" % j)

    def learn(self, iterations, batch_size):
        """TODO: ADD LEARNING USING BACK PROPAGATION"""
        pass

    def save_to_file(self, filename):
        """NeuralNetwork.save_to_file(str filename) -> None
        Save the current neural network to a file named filename"""
        file = open(filename, "w")  # open the file to save to
        # fill the text file with the current weights, each weight separated by commas, each neuron separated by newlines, each layer seperated by new two new lines
        text = ""
        for layer in self.neurons:
            paragraph = ""
            for neuron in layer:
                line = ""
                for weight in neuron.get_weights():
                    line += str(weight) + ","
                paragraph += line.rstrip(", ") + "\n"
            text += paragraph + "\n"
        text = text.rstrip("\n\n")
        file.write(text)
        file.close()  # close the file

    def load_from_file(self, filename):
        """NeuralNetwork.load_from_file(str filename) -> None
        Load saved weights into the current network from a file called filename"""
        file = open(filename, "r")  # open the file to read from
        # split the file into an array of arrays of arrays of floats
        text = file.read()
        paragraphs = text.split("\n\n")
        lines = [paragraph.split("\n") for paragraph in paragraphs]
        weights = [[line.split(",") for line in paragraph] for paragraph in lines]
        # set each weight in the network to the same index in the generated array
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                for k in range(len(weights[i][j])):
                    self.neurons[i][j].set_weight(k, float(weights[i][j][k]))
        file.close()  # close the file


def step(x):
    """step(float x) -> int
    A simple neural network activation function
    Returns 1 if x > 0, returns 0 otherwise"""
    if x > 0:
        return 1
    else:
        return 0


def sigmoid(x):
    """sigmoid(float x) -> float
    A common neural network activation function
    Used to make all neuron activations between 0 and 1
    Defined as 1/(1 + e^(-x))"""
    return 1 / (1 + math.exp(-x))


nn = NeuralNetwork(4, [784, 200, 50, 10], sigmoid)  # create a new neural network with 784 (28 x 28) nodes in the input layer, 2 hidden layers, and 10 nodes in the output layer
nn.evolve(0.1, 50, 100, 0.5, training, 100)  # evolve the network with a genetic algorithm
for c in range(50):  # try out 50 test cases on the best network
    print("Test Case %d" % c)
    # choose a random test case
    index = random.randrange(10000)
    case = testing[index]
    # put the inputs through the network
    out = nn.evaluate(case[0])
    # create a bmp of what the input looks like
    image = bitmap.Bitmap(28, 28)
    for a in range(28):
        for b in range(28):
            image.setPixel(b, 27 - a, tuple([int(case[0][a * 28 + b] * 255)] * 3))
    image.write("inputs\\input%d.bmp" % c)  # save that bmp
    print("\tInput Image: inputs\\input%d.bmp" % c)
    print("Expected Output: %d" % case[1])  # print what the network should output
    print("Computer Output: %d" % out.index(max(out)))  # print the digit the neural network decided
