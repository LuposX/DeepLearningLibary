"""
Author: Lupos
Purpose: Practising coding NN
Date: 17.11.2019
Description: Solving a XOR with a NN.
"""

from typing import List, Dict  # used for typehints

import numpy as np  # used for forward pass, weight init, ect.

# logging
import logging  # used to log errors and info's in a file
from datetime import date  # used to get a name for the log file
import os  # used for creating a folder


class NeuralNetwork:
    def __init__(self, x: List[float], y: List[float], nn_architecture: List[Dict], alpha: float, seed: int) -> None:
        """
        Constructor of the class Neural Network.

        Parameters
        ----------
        x : List[float]
            Input data on which the Neural Network should get trained on
        y : List[float]
            Target/corresponding values for the input data.
        nn_architecture : List[Dict]
            Describes the architecture of the Neural Network.
        alpha : float
            The learning rate.
        seed: int
            Seed for numpy.For creating random values. For creating reproducible results.

        Returns
        -------
        None
        """
        self.level_of_debugging = logging.INFO
        self.logger: object = self.init_logging(self.level_of_debugging)  # initializing of logging

        # Dimension checks
        self.check_input_output_dimension(x, y, nn_architecture)

        np.random.seed(seed)  # set seed for reproducibility

        self.input: List = x
        self.y: List = y
        self.output_model: float = np.zeros(y.shape)
        self.alpha: float = alpha
        self.layer_cache = {"z0": self.input}  # later used for derivatives

        self.nn_architecture: List[Dict] = nn_architecture

        self.weights: List = []  # np.array([])
        self.init_weights()  # initializing of weights
        self.w_d: List = []  # gardient in perspective to the weight

        self.curr_layer: List = []
        self.weight_change: List = []

        self.logger.info("__init__ executed")

    def check_input_output_dimension(self, x, y, nn_architecture):
        """
        Gets executed from the constructor "__init__". Is used
        to check if the dimensions of input and output values correspond to the neuron size
        in the input and output layer.

        Parameters
        ----------
        x
            Input values
        y
            Output Values
        nn_architecture : List[Dict]
             Architecture of the neural network.

        Returns
        -------
        None
        """
        assert len(x[0]) == nn_architecture[0][
            "layer_size"], 'Check the number of input Neurons and "X".'  # check if the first element in "x" has the right shape
        assert len(y[0]) == nn_architecture[-1][
            "layer_size"], 'Check the number of output Neurons and "Y".'  # check if the first element in "y" has the right shape

    # mean square root
    def loss(self, y: List[float], y_hat: List[float]) -> List[float]:
        return 1 / 2 * (y_hat - y) ** 2

    def loss_derivative(self, y: List[float], y_hat: List[float]) -> List[float]:
        return y_hat - y

    def sigmoid(self, x: List[float]) -> List[float]:
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: List[float]) -> List[float]:
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x: List[float]) -> List[float]:
        return np.maximum(0, x)

    def relu_derivative(self, x: List[float]) -> List[float]:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def linear(self, x: List[float]) -> List[float]:
        return x

    def activation_derivative(self, layer: Dict, curr_layer: List[float]) -> List[float]:
        if layer["activation_function"] == "linear":
            return self.linear(curr_layer)

        elif layer["activation_function"] == "relu":
            return self.relu_derivative(curr_layer)

        elif layer["activation_function"] == "sigmoid":
            return self.sigmoid_derivative(curr_layer)

        else:
            raise Exception("Activation function not supported!")

    def communication(self, i: int, how_often: int = 10) -> None:
        """
        Gets executed from the method "train". Communicates information
        about the current status of training progress.

        Parameters
        ----------
        i : int
            A paramter that gets hand over. Current Iteration in a foor loop.
        how_often: int
             Is used to determine the frequently of updates from the training progress.

        Returns
        -------
        None
        """
        if i % how_often == 0:
            print("For Iteration #" + str(i))
            print("Input: " + str(self.input))
            print("Actual Output: " + str(self.y))
            print("Predicted Output: " + str(self.output_model))
            print("Loss: " + str(self.loss(self.y, self.output_model)))
            print("Value of last weight change: " + str(self.weight_change))
            print("\n")

    def init_logging(self, level_of_debugging: str) -> object:
        """
        Gets executed from the constructor "__init__". Initializes the logger.

        Parameters
        ----------
        level_of_debugging: {"logging.DEBUG", "logging.INFO", "logging.CRITICAL", "logging.WARNING", "logging.ERROR"}
            Which error get logged.

        Returns
        -------
        Object
            return a logger object which is used to log errors.
        """
        # creating a directory for "logs" if the directory doesnt exist
        path = os.getcwd()
        name = "logs"
        full_path = path + "\\" + name
        try:
            if not os.path.isdir(full_path):
                os.mkdir(full_path)
        except OSError:
            print("ERROR: Couldn't create a log folder.")

        # create and configure logger
        today = date.today()  # get current date
        today_eu = today.strftime("%d-%m-%Y")  # european date format

        LOG_FORMAT: str = "%(levelname)s  - %(asctime)s - %(message)s"  # logging format

        logging.basicConfig(filename=full_path + "\\" + today_eu + ".log", level=level_of_debugging, format=LOG_FORMAT)
        logger = logging.getLogger()

        # Test logger
        logger.info("------------------------------------------------")
        logger.info("Start of the program")
        logger.info("------------------------------------------------")

        return logger

    # TODO: "init_weights" is work in progress.
    # TODO: "init_weights" init bias.
    def init_weights(self) -> List[float]:
        """
        Gets executed from the constructor "__init__".
        Initializes the weight in the whole Neural Network.

        Returns
        -------
        List
            Weights of the Neural Network.
        """
        self.logger.info("init_weights executed")
        for idx in range(0, len(self.nn_architecture) - 1):  # "len() - 1" because the output layer doesn't has weights
            weights_temp = 2 * np.random.rand(self.nn_architecture[idx]["layer_size"], self.nn_architecture[idx + 1]["layer_size"]) - 1
            self.weights.append(weights_temp)

        return self.weights

    def activate_neuron(self, x: List[float], layer: Dict, save_layer_cache: bool = True) -> List[float]:
        """
        Gets executed from the method "forward" and "full_forward".
        Activates the neurons in the current layer with the specified activation function.

        Parameters
        ----------
        x: List[float]
            This are the values which get activated.
        layer: Dict
             A Dictionary with different attributes about the current layer.
        save_layer_cache:
            Disable it if you don't want to cache the output of a single step of forward propagation.
        Returns
        -------
        List
            Outputs a List with activated values/neurons.
        """

        if layer["activation_function"] == "relu":
            temp_acti = self.relu(x)

            if save_layer_cache:
                # the name of the key of the dict is the index of current layer
                idx_name = self.nn_architecture.index(layer)
                tmp_dict = {"a" + str(idx_name): temp_acti}
                self.layer_cache.update(tmp_dict)

            return temp_acti

        elif layer["activation_function"] == "sigmoid":
            temp_acti = self.sigmoid(x)

            if save_layer_cache:
                # the name of the key of the dict is the index of current layer
                idx_name = self.nn_architecture.index(layer)
                tmp_dict = {"a" + str(idx_name): temp_acti}
                self.layer_cache.update(tmp_dict)

            return temp_acti

        else:
            raise Exception("Activation function not supported!")

    def forward(self, weight: List[float], x: List[float], layer: Dict, save_layer_cache: bool = True) -> List[float]:
        """
        Gets executed from the method "full_forward". This method make´s one
        forward propagation step.

        Parameters
        ----------
        weight : List[float]
            The weights of each associated Neurons in a List.
        x : List[float]
            The Input from the current layer which gets multiplicated with the weights and summed up.
        layer : Dict
            A Dictionary with different attributes about the current layer.
        save_layer_cache: bool
            Disable it if you don't want to cache the output of a single step of forward propagation.

        Returns
        -------
        List
            List with values from the output of the one step forward propagation.
        """

        curr_layer = np.dot(x, weight)

        # the name of the key of the dict is the index of current layer
        idx_name = self.nn_architecture.index(layer)
        tmp_dict = {"z" + str(idx_name): curr_layer}
        self.layer_cache.update(tmp_dict)   # append the "z" value | not activated value

        curr_layer = self.activate_neuron(curr_layer, layer, save_layer_cache)

        return curr_layer

    # TODO: "full_forward" is work in progress
    def full_forward(self) -> List[float]:
        """
        Gets executed from the method "forward_backprop". Makes the full forward propagation
        through the whole Architecture of the Neural Network.

        Returns
        -------
        List
            List with the values of the output Layer.
        """
        self.logger.info("full_forward executed")
        self.layer_cache = {}  # delete cache used from previous iteration
        for idx in range(0, len(self.nn_architecture) - 1):
            self.logger.debug("Current-index (full_forward methode): " + str(idx))

            if self.nn_architecture[idx]["layer_type"] == "input_layer":
                self.layer_cache.update({"z0": self.input})
                self.layer_cache.update({"a0": self.input})
                self.curr_layer = self.forward(self.weights[idx], self.input, self.nn_architecture[idx + 1])  # "idx + 1" to fix issue regarding activation function
            else:
                self.curr_layer = self.forward(self.weights[idx], self.curr_layer, self.nn_architecture[idx + 1])

        return self.curr_layer

    # TODO: "backprop" is work in progress
    def backprop(self) -> None:  # application of the chain rule to find derivative
        """
        Gets executed from the method "forward_backprop". This method handels
        the backpropagation of the Neural Network.

        Returns
        -------
        None
        """
        self.logger.info("Backprop executed")
        for idx, layer in reversed(list(enumerate(nn_architecture))):  # reversed because we go backwards
            if not layer["layer_type"] == "input_layer":  # if we are in the input layer

                # calculating the error term
                if layer["layer_type"] == "output_layer":
                    temp_idx = "z" + str(idx)
                    error_term = self.activation_derivative(layer, self.layer_cache[temp_idx]) * self.loss_derivative(y=self.y, y_hat=self.output_model)
                else:
                    temp_idx = "z" + str(idx)
                    error_term = self.activation_derivative(layer, self.layer_cache[temp_idx]).T * np.dot(self.weights[idx], error_term)

                temp_idx = "a" + str(idx)
                self.weight_change = error_term.T * self.layer_cache[temp_idx]

                self.weights[idx - 1] = self.weights[idx - 1] - (self.alpha * self.weight_change)  # updating weight

    # TODO: "forward_backprop" is work in progress
    def forward_backprop(self) -> None:
        """
        Gets executed from the method "train". This method combines forward propagation
        with backwards propagation.

        Returns
        -------
        None
        """
        self.output_model = self.full_forward()
        self.backprop()

    # TODO: "train" is work in progress
    def train(self, how_often, epochs=20) -> None:
        """
        Execute this method to start training your neural network.

        Parameters
        ----------
        how_often : int
            gets handed over to communication. Is used to determine the frequently of updates from the training progress.
        epochs : int
           determines the epochs of training.

        Returns
        -------
        None
        """
        self.logger.info("Train-method executed")
        for i in range(epochs):
            self.forward_backprop()
            self.communication(i, how_often)


if __name__ == "__main__":
    # data for nn and target
    x = np.array([[0, 1, 1]], dtype=float)
    y = np.array([[1]], dtype=float)

    # nn_architecture is WITH input-layer and output-layer
    nn_architecture = [{"layer_type": "input_layer", "layer_size": 3, "activation_function": "none"},
                       {"layer_type": "hidden_layer", "layer_size": 5, "activation_function": "relu"},
                       {"layer_type": "hidden_layer", "layer_size": 3, "activation_function": "relu"},
                       {"layer_type": "output_layer", "layer_size": 1, "activation_function": "sigmoid"}]

    NeuralNetwork_Inst = NeuralNetwork(x, y, nn_architecture, 1, 5)
    NeuralNetwork_Inst.train(2, 14)