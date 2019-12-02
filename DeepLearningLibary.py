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
    def __init__(self, x: List[float], y: List[float], nn_architecture: List[Dict], alpha: float, seed: int, custom_weights_data: List = [], custom_weights: bool = False) -> None:
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

        self.bias: List = []
        self.weights: List = []  # np.array([])
        self.init_weights(custom_weights, custom_weights_data)  # initializing of weights
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
        assert len(x) == len(y), "Check that X and Y have the corresponding values."

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

    def communication(self, curr_epoch: int, curr_trainingsdata: int, data: List[float], target: List[float], how_often: int = 10) -> None:
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
        if curr_epoch % how_often == 0:
            print("For iteration/trainings-example: #" + str(curr_epoch) + "/#"+ str(curr_trainingsdata))
            print("Input: " + str(data))
            print("Actual Output: " + str(target))
            print("Predicted Output: " + str(self.output_model))
            print("Loss: " + str(self.loss(y=target, y_hat=self.output_model)))
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
    def init_weights(self, custom_weights: bool, custom_weights_data: List) -> List[float]:
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

            if not custom_weights:
                weights_temp = 2 * np.random.rand(self.nn_architecture[idx]["layer_size"], self.nn_architecture[idx + 1]["layer_size"]) - 1
                self.weights.append(weights_temp)

            bias_temp = np.ones(self.nn_architecture[idx + 1]["layer_size"])
            self.bias.append(bias_temp)

        if custom_weights:
            self.weights = custom_weights_data

        return self.weights

    def activate_neuron(self, x: List[float], layer: Dict) -> List[float]:
        """
        Gets executed from the method "forward" and "full_forward".
        Activates the neurons in the current layer with the specified activation function.

        Parameters
        ----------
        x: List[float]
            This are the values which get activated.
        layer: Dict
             A Dictionary with different attributes about the current layer.
        Returns
        -------
        List
            Outputs a List with activated values/neurons.
        """

        if layer["activation_function"] == "relu":
            temp_acti = self.relu(x)

            # the name of the key of the dict is the index of current layer
            idx_name = self.nn_architecture.index(layer)
            self.layer_cache.update({"a" + str(idx_name): temp_acti})

            return temp_acti

        elif layer["activation_function"] == "sigmoid":
            temp_acti = self.sigmoid(x)

            # the name of the key of the dict is the index of current layer
            idx_name = self.nn_architecture.index(layer)
            self.layer_cache.update({"a" + str(idx_name): temp_acti})

            return temp_acti

        else:
            raise Exception("Activation function not supported!")

    def forward(self, weight: List[float], x: List[float], layer: Dict, idx: int) -> List[float]:
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

        Returns
        -------
        List
            List with values from the output of the one step forward propagation.
        """

        curr_layer = np.dot(x, weight) + self.bias[idx]

        # the name of the key of the dict is the index of current layer
        idx_name = self.nn_architecture.index(layer)
        tmp_dict = {"z" + str(idx_name): curr_layer}
        self.layer_cache.update(tmp_dict)   # append the "z" value | not activated value

        curr_layer = self.activate_neuron(curr_layer, layer)

        return curr_layer

    # TODO: "full_forward" is work in progress
    def full_forward(self, data):
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
                self.layer_cache.update({"z0": data})
                self.layer_cache.update({"a0": data})
                self.curr_layer = self.forward(self.weights[idx], data, self.nn_architecture[idx + 1], idx=idx)  # "idx + 1" to fix issue regarding activation function
            else:
                self.curr_layer = self.forward(self.weights[idx], self.curr_layer, self.nn_architecture[idx + 1], idx=idx)

        self.output_model = self.curr_layer

    # TODO: "backprop" is work in progress
    def backprop(self, target: List[float]) -> None:  # application of the chain rule to find derivative
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
                    error_term = self.activation_derivative(layer, self.layer_cache[temp_idx]) * self.loss_derivative(y=target, y_hat=self.output_model)
                else:
                    temp_idx = "z" + str(idx)
                    error_term = self.activation_derivative(layer, self.layer_cache[temp_idx]).T * np.dot(self.weights[idx], error_term)

                temp_idx = "a" + str(idx)
                self.weight_change = error_term.T * self.layer_cache[temp_idx]

                self.weights[idx - 1] = self.weights[idx - 1] - (self.alpha * self.weight_change)  # updating weight


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
        for curr_epoch in range(epochs):
            for idx, trainings_data in enumerate(x):
                self.full_forward(trainings_data)
                self.backprop(self.y[idx])
                self.communication(curr_epoch, idx, target=self.y[idx], data=trainings_data, how_often=how_often)

    def predict(self):
        """
        Used for predicting with the neural network
        """
        print("Predicting")
        print("--------------------")

        running = True
        while(running):
            pred_data = []
            for i in range(0, self.nn_architecture[0]["layer_size"]):
                tmp_input = input("Enter " + str(i) + " value: ")
                pred_data.append(tmp_input)

            self.full_forward(np.asarray([pred_data], dtype=float))
            print("Predicted Output: ", self.output_model)
            print(" ")

            running = input('Enter "exit" if you want to exit. Else press "enter".')
            if running == "exit" or running == "Exit":
                running = False
            else:
                running = True


if __name__ == "__main__":
    # data for nn and target
    x = np.array([[0.7, 0.6]], dtype=float)
    y = np.array([[0.9, 0.1]], dtype=float)

    # nn_architecture is WITH input-layer and output-layer
    nn_architecture = [{"layer_type": "input_layer", "layer_size": 2, "activation_function": "none"},
                       {"layer_type": "hidden_layer", "layer_size": 2, "activation_function": "sigmoid"},
                       {"layer_type": "output_layer", "layer_size": 2, "activation_function": "sigmoid"}]

    weights_data = np.array([[0.3, -0.2, 0.8, -0.6, 0.5, 0.7], [0.2, 0.1, 0.4, -0.4, 0.3, 0.5]], dtype=float)

    NeuralNetwork_Inst = NeuralNetwork(x, y, nn_architecture, 0.3, 5, custom_weights=True, custom_weights_data=weights_data)
    NeuralNetwork_Inst.train(how_often=1, epochs=20)
    # NeuralNetwork_Inst.predict()
