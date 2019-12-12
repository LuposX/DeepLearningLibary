"""
Author: Lupos
Purpose: Practising coding NN
Date: 17.11.2019
Description: test NN
"""

from typing import List, Dict  # used for typehints

import numpy as np  # used for forward pass, weight init, ect.

# logging
import logging  # used to log errors and info's in a file
from datetime import date  # used to get a name for the log file
import os  # used for creating a folder
import pathlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, x: List[float], y: List[float], nn_architecture: List[Dict], alpha: float, seed: int,
                 custom_weights_data: List = [], loss_type: str = "mse", custom_weights: bool = False,
                 level_of_debugging=logging.WARNING) -> None:
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
        self.level_of_debugging = level_of_debugging
        self.logger: object = self.init_logging(self.level_of_debugging)  # initializing of logging
        # Dimension checks
        self.check_input_output_dimension(x, y, nn_architecture)

        np.random.seed(seed)  # set seed for reproducibility

        self.input: List = self.add_bias(x)
        self.y: List = y

        self.output_model: float = np.zeros(y.shape)
        self.alpha: float = alpha
        self.layer_cache = {}  # later used for derivatives
        self.error_term_cache = []

        self.nn_architecture: List[Dict] = nn_architecture

        self.weights: List = []  # np.array([])
        self.init_weights(custom_weights, custom_weights_data)  # initializing of weights
        self.w_d: List = []  # gardient in perspective to the weight

        self.curr_layer: List = []
        self.weight_change_cache: List = []

        self.logger.info("__init__ executed")
        
        # for visuliozing
        self.x_train_loss_history = []
        self.y_train_loss_history = []

        self.bias_weight_tmp = []

        self.loss_type = loss_type
        
    def add_bias(self, x) -> List[float]:
        x = np.array([np.insert(x, 0, 1)])
        return x

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

    def loss_type_choice(self, y: List[float], y_hat: List[float], derivative: bool = False):
        """
        This function chooses the appropiates loss function depending on the loss type
        Parameters
        ----------
        loss_type: type of loss for example "mse" or "cross entropy"

        Returns
        -------

        """
        if not derivative:
            if self.loss_type.lower() == "mse":
                return self.loss_mse(y, y_hat)

            elif "cross" in self.loss_type.lower() and "entropy" in self.loss_type.lower():
                return self.loss_cross_entropy(y=y, y_hat=y_hat)

            else:
                raise NotImplementedError("There is no loss with that name.")


        elif derivative:
            if self.loss_type.lower() == "mse":
                return self.loss_mse_derivative(y, y_hat)

            elif "cross" in self.loss_type.lower() and "entropy" in self.loss_type.lower():
                return self.loss_cross_entropy_derivative(y=y, y_hat=y_hat)

            else:
                raise NotImplementedError("There is no loss with that name.")


        else:
            raise ValueError("Variable derivative not set for loss function.")

    # mean square root
    def loss_mse(self, y: List[float], y_hat: List[float]) -> List[float]:
        loss = np.sum(1 / 2 * (y - y_hat) ** 2)
        return loss

    def loss_mse_derivative(self, y: List[float], y_hat: List[float]) -> List[float]:
        y = np.array([y]).T
        tmp_dev = -(y - y_hat)
        loss_dev = np.array(tmp_dev)
        return loss_dev

    def loss_cross_entropy(self, y: List[float], y_hat: List[float]) -> List[float]:
        loss = []

        for idx, y_hat_item in enumerate(y_hat):
            if y[idx] == 1:
                tmp_loss = -np.log10(y_hat_item)
                loss.append(tmp_loss)
            else:
                tmp_loss = -np.log10(1 - y_hat_item)
                loss.append(tmp_loss)

        loss = np.sum(loss)
        return loss

    def loss_cross_entropy_derivative(self, y: List[float], y_hat: List[float]) -> List[float]:

        y = y.tolist()
        y_hat = y_hat.flatten().tolist()

        loss = []
        for idx, y_hat_item in enumerate(y_hat):
            if y[idx] == 1:
                tmp_loss = -1 * (1 / float(y_hat_item))
                loss.append(tmp_loss)
            else:
                tmp_loss = 1 / (1 - float(y_hat_item))
                loss.append(tmp_loss)

        return np.array(loss)

    def tanh(self, x: List[float]) -> List[float]:
        return 2 / (1 + np.exp(-2 * x)) - 1

    def tanh_derivative(self, x: List[float]) -> List[float]:
        return 1 - (self.tanh(x) ** 2)

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
            return np.array(self.linear(curr_layer))

        elif layer["activation_function"] == "relu":
            return np.array(self.relu_derivative(curr_layer))

        elif layer["activation_function"] == "sigmoid":
            return np.array(self.sigmoid_derivative(curr_layer))

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
            print("Predicted Output: " + str(self.output_model.flatten()))
            print("Loss: " + str(self.loss_type_choice(y=target, y_hat=self.output_model.flatten())))
            print("Value of last weight change: " + str(self.weight_change_cache[-1]))
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
        path = pathlib.Path.cwd()
        name = "logs"
        full_path = path / name
        try:
            if not os.path.isdir(full_path):
                os.mkdir(full_path)
        except OSError:
            print("ERROR: Couldn't create a log folder.")

        # create and configure logger
        today = date.today()  # get current date
        today_eu = today.strftime("%d-%m-%Y")  # european date format

        LOG_FORMAT: str = "%(levelname)s  - %(asctime)s - %(message)s"  # logging format

        logging.basicConfig(filename=full_path / str(today_eu + ".log"), level=level_of_debugging, format=LOG_FORMAT)
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
                # "self.nn_architecture[idx]["layer_size"] + 1" "+ 1" because we also have a bias term
                weights_temp = 2 * np.random.rand(self.nn_architecture[idx + 1]["layer_size"], self.nn_architecture[idx]["layer_size"] + 1) - 1

                self.weights.append(weights_temp)

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

            # add bias to cache when not output layer
            if not layer["layer_type"] == "output_layer":
                tmp_temp_acti_for_chache = self.add_bias(temp_acti)
            else:
                tmp_temp_acti_for_chache = temp_acti.T

            # the name of the key of the dict is the index of current layer
            idx_name = self.nn_architecture.index(layer)
            self.layer_cache.update({"a" + str(idx_name): tmp_temp_acti_for_chache})

            return temp_acti

        elif layer["activation_function"] == "sigmoid":
            temp_acti = self.sigmoid(x)

            # add bias to cache when not output layer
            if not layer["layer_type"] == "output_layer":
                tmp_temp_acti_for_chache = self.add_bias(temp_acti)
            else:
                tmp_temp_acti_for_chache = temp_acti.T

            # the name of the key of the dict is the index of current layer
            idx_name = self.nn_architecture.index(layer)
            self.layer_cache.update({"a" + str(idx_name): tmp_temp_acti_for_chache})

            return temp_acti

        else:
            raise NotImplementedError("Activation function not supported!")

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

        curr_layer = np.dot(weight, x.T)

        # add bias to cache when not output layer
        if not layer["layer_type"] == "output_layer":
            tmp_curr_layer_for_chache = self.add_bias(curr_layer)
        else:
            tmp_curr_layer_for_chache = curr_layer.T

        # the name of the key of the dict is the index of current layer
        idx_name = self.nn_architecture.index(layer)
        tmp_dict = {"z" + str(idx_name):  tmp_curr_layer_for_chache}
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
                self.curr_layer = self.add_bias(self.curr_layer)
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
        self.bias_weight_tmp = []
        self.weight_change_cache = []
        self.error_term_cache = []
        self.logger.info("Backprop executed")
        for idx, layer in reversed(list(enumerate(nn_architecture))):  # reversed because we go backwards
            if not layer["layer_type"] == "input_layer":  # if we are in the input layer

                # calculating the error term
                if layer["layer_type"] == "output_layer":
                    temp_idx = "z" + str(idx)
                    d_a = self.activation_derivative(layer, self.layer_cache[temp_idx])
                    d_J = self.loss_type_choice(y=target, y_hat=self.output_model, derivative=True)
                    error_term = np.array([np.multiply(d_a.flatten(), d_J.flatten())])
                    self.error_term_cache.append(error_term)

                    tmp_matrix_weight = np.asarray(self.weights[idx - 1])
                    tmp_bias_weight_t = np.array(tmp_matrix_weight.T[0])
                    self.bias_weight_tmp.append([tmp_bias_weight_t])
                else:
                    temp_idx = "z" + str(idx)
                    layer_cache_tmp_drop_bias = np.delete(self.layer_cache[temp_idx], 0, 1)

                    d_a = self.activation_derivative(layer, layer_cache_tmp_drop_bias)

                    d_J = 0
                    for item in reversed(self.error_term_cache):
                        tmp_matrix_weight = np.asarray(self.weights[idx - 1])
                        self.bias_weight_tmp.append([tmp_matrix_weight.T[0]])

                        weights_tmp_drop_bias = np.delete(self.weights[idx], 0, 1)
                        d_J = d_J + np.dot(weights_tmp_drop_bias.T, item.T)

                    error_term = d_a.T * d_J
                    error_term = error_term.T
                    self.error_term_cache.append(error_term)

                err_temp = error_term.T
                temp_idx = "a" + str(idx - 1)
                cache_tmp = self.layer_cache[temp_idx]
                cache_tmp = np.delete(cache_tmp, 0, 1)  # delete bias
                weight_change = err_temp * cache_tmp
                self.weight_change_cache.append(weight_change)

        # update weights
        for idx in range(0, len(self.weight_change_cache)):  # reversed because we go backwards
            curr_weight = self.weights[-idx - 1]
            curr_weight = np.delete(curr_weight, 0, 1)  # delete bias
            weight_change_tmp = self.weight_change_cache[idx]

            total_weight_change = self.alpha * weight_change_tmp  # updating weight
            curr_weight = curr_weight - total_weight_change
            self.weights[-idx - 1] = curr_weight

        # update bias
        for i in range(0, len(self.bias_weight_tmp)):
            tmp_weight_bias = np.asarray(self.bias_weight_tmp[i])
            tmp_error_term_bias = np.asarray(self.error_term_cache[i])
            self.bias_weight_tmp[i] = tmp_weight_bias - (self.alpha * tmp_error_term_bias)

        # insert bias in weights
        for i in range(0, len(self.weights)):
            self.weights[i] = np.insert(self.weights[i], obj=0, values=self.bias_weight_tmp[i], axis=1)  # insert the weights for the biases

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

                trainings_data_with_bias = self.add_bias(trainings_data)

                self.full_forward(trainings_data_with_bias)
                self.backprop(self.y[idx])
                self.communication(curr_epoch, idx, target=self.y[idx], data=trainings_data, how_often=how_often)

                self.x_train_loss_history.append(curr_epoch)
                self.y_train_loss_history.append(self.loss_type_choice(y=y[idx], y_hat=self.output_model.flatten()))

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

            pred_data_without_bias = np.asarray([pred_data], dtype=float)

            pred_data.insert(0, 1)  # append bias
            pred_data = np.asarray([pred_data], dtype=float)

            self.full_forward(data=pred_data)

            print("Input: " + str(pred_data_without_bias))
            print("Predicted Output: ", self.output_model.flatten())
            print(" ")

            running = input('Enter "exit" if you want to exit. Else press "enter".')
            if running == "exit" or running == "Exit":
                running = False
            else:
                running = True

    def visulize(self):
        data = {"x": self.x_train_loss_history, "train": self.y_train_loss_history}
        data = pd.DataFrame(data, columns=["x", "train"])

        sns.set_style("darkgrid")
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="x", y="train", data=data, label="train", color="orange")
        plt.xlabel("Time In Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over Time")
        plt.show()

if __name__ == "__main__":
    # data for nn and target
    x = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1]], dtype=float)
    y = np.array([[0, 1, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)

    # nn_architecture is WITH input-layer and output-layer
    nn_architecture = [{"layer_type": "input_layer", "layer_size": 3, "activation_function": "none"},
                       {"layer_type": "hidden_layer", "layer_size": 3, "activation_function": "sigmoid"},
                       {"layer_type": "output_layer", "layer_size": 3, "activation_function": "sigmoid"}]

    weights_data = [np.array([[0.5, 0.1, 0.3, 0.5], [0.5, 0.2, 0.4, 0.6]], dtype=float), np.array([[0.5, 0.7, 0.9], [0.5, 0.8, 0.1]], dtype=float)]
    weights_data = weights_data

    #, custom_weights=True, custom_weights_data=weights_data
    NeuralNetwork_Inst = NeuralNetwork(x, y, nn_architecture, 0.1, 5, loss_type="cross-entropy")
    NeuralNetwork_Inst.train(how_often=20, epochs=300)
    NeuralNetwork_Inst.visulize()
    #NeuralNetwork_Inst.predict()
