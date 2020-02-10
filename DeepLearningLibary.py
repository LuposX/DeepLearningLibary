"""
Author: Lupos
Purpose: Practising coding NN
Date: 17.11.2019
Description: test NN
"""

from utility_libary import *

import random

class NeuralNetwork:
    def __init__(self, x: List[float], y: List[float], nn_architecture: List[Dict], alpha: float, seed: int,
                 custom_weights_data: List = [], logger: object = None, loss_type: str = "mse",
                 custom_weights: bool = False, ) -> None:
        """
        Constructor of the class Neural Network.

        Parameters
        ----------
        logger: object
            If set stuff get logged if not set than not :)
        loss_type: str
            Set the type of loss yo uwant to use for the neural network. e.g "mse", "cross-entropy".
        custom_weights_data: bool
            Set it to true if you want to add custom weights.
        custom_weights: list
            add your custom weights in this list.
        level_of_debugging
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
        self.logger: object = init_logging(logger)  # initializing of logging
        np.random.seed(seed)  # set seed for reproducibility

        self._check_input_output_dimension(x, y, nn_architecture)  # Dimension checks

        self.nn_architecture: List[Dict] = nn_architecture
        self.alpha: float = alpha
        self.loss_type = loss_type
        self.y: List = y
        self.x: List = x
        self.input: List = self._add_bias(x)

        self.output_model: float = np.zeros(y.shape)
        self.layer_value_cache = {}  # later used for derivatives
        self.error_term_cache: List = []
        self.weight_change_cache: List = []
        self.x_train_loss_history: List = []  # for visualizing
        self.y_train_loss_history: List = []  # for visualizing

        self.weights: List = []  # np.array([])
        self._init_weights(custom_weights, custom_weights_data)  # initializing of weights

    def _add_bias(self, x) -> List[float]:
        """
        Is used internal
        Parameters
        ----------
        x

        Returns
        -------

        """
        x = np.array([np.insert(x, 0, 1)])
        return x

    def _check_input_output_dimension(self, x, y, nn_architecture):
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

    def _communication(self, curr_epoch: int, curr_trainingsdata: int, data: List[float], target: List[float],
                      how_often: int = 10) -> None:
        """
        Gets executed from the method "train". Communicates information
        about the current status of training progress.

        Parameters
        ----------
        i : int
            A parameter that gets hand over. Current Iteration in a foor loop.
        how_often: int
             Is used to determine the frequently of updates from the training progress.

        Returns
        -------
        None
        """
        if curr_epoch % how_often == 0:
            print(
                f"For iteration/trainings-example: #" + str(curr_epoch) + "/#" + str(curr_trainingsdata),
                f"Input: {data}",
                f"Actual Output: {target}",
                f"Predicted Output: {self.output_model.flatten()}",
                f"Loss: {loss_type_choice(y=target, y_hat=self.output_model.flatten(), loss_type=self.loss_type)}",
                f"Value of last weight change: ",
                f"{self.weight_change_cache[-1]}"
                "\n",
                sep="\n"
            )

    def _init_weights(self, custom_weights: bool, custom_weights_data: List) -> List[float]:
        """
        Gets executed from the constructor "__init__".
        Initializes the weight in the whole Neural Network.

        Returns
        -------
        List
            Weights of the Neural Network.
        """
        if self.logger:
            self.logger.info("init_weights executed")
        for idx in range(0, len(self.nn_architecture) - 1):  # "len() - 1" because the output layer doesn't has weights

            if not custom_weights:
                # "self.nn_architecture[idx]["layer_size"] + 1" "+ 1" because we also have a bias term
                weights_temp = 2 * np.random.rand(self.nn_architecture[idx + 1]["layer_size"],
                                                  self.nn_architecture[idx]["layer_size"] + 1) - 1

                self.weights.append(weights_temp)

        if custom_weights:
            self.weights = custom_weights_data

        return self.weights

    def _activate_neuron(self, x: List[float], layer: Dict) -> List[float]:
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
            temp_acti = relu(x)

            # add bias to cache when not output layer
            if not layer["layer_type"] == "output_layer":
                tmp_temp_acti_for_chache = self._add_bias(temp_acti)
            else:
                tmp_temp_acti_for_chache = temp_acti.T

            # the name of the key of the dict is the index of current layer
            idx_name = self.nn_architecture.index(layer)
            self.layer_value_cache.update({"a" + str(idx_name): tmp_temp_acti_for_chache})

            return temp_acti


        elif layer["activation_function"] == "sigmoid":
            temp_acti = sigmoid(x)

            # add bias to cache when not output layer
            if not layer["layer_type"] == "output_layer":
                tmp_temp_acti_for_chache = self._add_bias(temp_acti)
            else:
                tmp_temp_acti_for_chache = temp_acti.T

            # the name of the key of the dict is the index of current layer
            idx_name = self.nn_architecture.index(layer)
            self.layer_value_cache.update({"a" + str(idx_name): tmp_temp_acti_for_chache})

            return temp_acti


        elif layer["activation_function"] == "tanh":
            temp_acti = tanh(x)

            # add bias to cache when not output layer
            if not layer["layer_type"] == "output_layer":
                tmp_temp_acti_for_chache = self._add_bias(temp_acti)
            else:
                tmp_temp_acti_for_chache = temp_acti.T

            # the name of the key of the dict is the index of current layer
            idx_name = self.nn_architecture.index(layer)
            self.layer_value_cache.update({"a" + str(idx_name): tmp_temp_acti_for_chache})

            return temp_acti


        else:
            raise NotImplementedError("Activation function not supported!")

    def _forward(self, weight: List[float], x: List[float], layer: Dict, idx: int) -> List[float]:
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
            tmp_curr_layer_for_chache = self._add_bias(curr_layer)
        else:
            tmp_curr_layer_for_chache = curr_layer.T

        # the name of the key of the dict is the index of current layer
        idx_name = self.nn_architecture.index(layer)
        tmp_dict = {"z" + str(idx_name): tmp_curr_layer_for_chache}
        self.layer_value_cache.update(tmp_dict)  # append the "z" value | not activated value

        curr_layer = self._activate_neuron(curr_layer, layer)

        return curr_layer

    def _full_forward(self, data):
        """
        Gets executed from the method "forward_backprop". Makes the full forward propagation
        through the whole Architecture of the Neural Network.

        Returns
        -------
        List
            List with the values of the output Layer.
        """
        if self.logger:
            self.logger.info("full_forward executed")
        self.layer_value_cache = {}  # delete cache used from previous iteration
        for idx in range(0, len(self.nn_architecture) - 1):
            if self.logger:
                self.logger.debug("Current-index (full_forward method): " + str(idx))

            if self.nn_architecture[idx]["layer_type"] == "input_layer":
                self.layer_value_cache.update({"z0": data})
                self.layer_value_cache.update({"a0": data})
                curr_layer = self._forward(self.weights[idx], data, self.nn_architecture[idx + 1],
                                           idx=idx)
            else:
                curr_layer = self._add_bias(curr_layer)
                curr_layer = self._forward(self.weights[idx], curr_layer, self.nn_architecture[idx + 1],
                                           idx=idx)

        self.output_model = curr_layer

    def _backprop(self, target: List[float]) -> None:  # application of the chain rule to find derivative
        """
        Gets executed from the method "forward_backprop". This method is about
        the backpropagation of the Neural Network.

        Returns
        -------
        None
        """
        bias_weight_tmp = []
        self.weight_change_cache = []
        self.error_term_cache = []
        if self.logger:
            self.logger.info("Backprop executed")
        for idx, layer in reversed(list(enumerate(nn_architecture))):  # reversed because we go backwards
            if self.logger:
                self.logger.debug("Current layer: ", str(layer))

            if not layer["layer_type"] == "input_layer":  # if we are in the input layer

                # calculating the error term
                if layer["layer_type"] == "output_layer":
                    temp_idx = "z" + str(idx)
                    d_a = activation_derivative(layer, self.layer_value_cache[temp_idx])
                    d_J = loss_type_choice(y=target, y_hat=self.output_model, loss_type=self.loss_type, derivative=True)
                    error_term = np.array([np.multiply(d_a.flatten(), d_J.flatten())])
                    self.error_term_cache.append(error_term)

                    tmp_matrix_weight = np.asarray(self.weights[idx - 1])
                    tmp_bias_weight_t = np.array(tmp_matrix_weight.T[0])
                    bias_weight_tmp.append([tmp_bias_weight_t])
                else:
                    temp_idx = "z" + str(idx)
                    layer_cache_tmp_drop_bias = np.delete(self.layer_value_cache[temp_idx], 0, 1)

                    d_a = activation_derivative(layer, layer_cache_tmp_drop_bias)

                    d_J = 0
                    for item in reversed(self.error_term_cache):
                        #assert False, "fix that shit in backprop"
                        tmp_matrix_weight = np.asarray(self.weights[idx - 1])
                        bias_weight_tmp.append([tmp_matrix_weight.T[0]])

                        weights_tmp_drop_bias = np.delete(self.weights[idx], 0, 1)
                        d_J = d_J + np.dot(weights_tmp_drop_bias.T, item.T)

                    error_term = d_a.T * d_J
                    error_term = error_term.T
                    self.error_term_cache.append(error_term)

                err_temp = self.error_term_cache[-1].T
                temp_idx = "a" + str(idx - 1)
                cache_tmp = self.layer_value_cache[temp_idx]
                cache_tmp = np.delete(cache_tmp, 0, 1)  # delete bias
                weight_change = err_temp * cache_tmp
                self.weight_change_cache.append(weight_change)

                if self.logger:
                    self.logger.debug("Current error_term: ", str(error_term))

        # update weights
        for idx in range(0, len(self.weight_change_cache)):  # reversed because we go backwards
            curr_weight = self.weights[-idx - 1]
            curr_weight = np.delete(curr_weight, 0, 1)  # delete bias
            weight_change_tmp = self.weight_change_cache[idx]

            total_weight_change = self.alpha * weight_change_tmp  # updating weight
            curr_weight = curr_weight - total_weight_change
            self.weights[-idx - 1] = curr_weight

        # update bias
        for i in range(0, len(bias_weight_tmp)):
            tmp_weight_bias = np.asarray(bias_weight_tmp[i])
            tmp_error_term_bias = np.asarray(self.error_term_cache[i])
            bias_weight_tmp[i] = tmp_weight_bias - (self.alpha * tmp_error_term_bias)

        # insert bias in weights
        for i in range(0, len(self.weights)):
            self.weights[i] = np.insert(self.weights[i], obj=0, values=bias_weight_tmp[i],
                                        axis=1)  # insert the weights for the biases

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
        if self.logger:
            self.logger.info("Train-method executed")

        for curr_epoch in range(epochs):
            if self.logger:
                self.logger.debug("Current number of epochs: ", str(curr_epoch))

            for idx, trainings_data in enumerate(self.x):
                trainings_data_with_bias = self._add_bias(trainings_data)

                self._full_forward(trainings_data_with_bias)
                self._backprop(self.y[idx])
                self._communication(curr_epoch, idx, target=self.y[idx], data=trainings_data, how_often=how_often)

                self.x_train_loss_history.append(curr_epoch)
                self.y_train_loss_history.append(
                    loss_type_choice(y=self.y[idx], y_hat=self.output_model.flatten(), loss_type=self.loss_type))

            # shuffle our data in order to get a good generalization
            tmp_zip_shuffle = list(zip(self.x, self.y))
            random.shuffle(tmp_zip_shuffle)  # shuffle data-set
            self.x, self.y = zip(*tmp_zip_shuffle)


    def predict(self):
        """
        Used for predicting with the neural network
        """
        print("Predicting")
        print("--------------------")

        running = True
        while (running):

            pred_data = []
            for i in range(0, self.nn_architecture[0]["layer_size"]):
                tmp_input = input("Enter " + str(i) + " value: ")
                pred_data.append(tmp_input)

            pred_data_without_bias = np.asarray([pred_data], dtype=float)

            if self.logger:
                self.logger.debug("Current input-predict-data: ", str(pred_data_without_bias))

            pred_data.insert(0, 1)  # append bias
            pred_data = np.asarray([pred_data], dtype=float)

            self._full_forward(data=pred_data)

            print("Input: " + str(pred_data_without_bias))
            print("Predicted Output: ", self.output_model.flatten())
            print(" ")

            if self.logger:
                self.logger.debug("Current output-predict-data: ", str(self.output_model))

            running = input('Enter "exit" if you want to exit. Else press "enter".')
            if running == "exit" or running == "Exit":
                running = False
            else:
                running = True


if __name__ == "__main__":
    # data for nn and target
    x = np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 1]], dtype=float)
    y = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=float)
    #y = np.array([[1], [0], [0], [1]], dtype = float)

    # nn_architecture is WITH input-layer and output-layer
    nn_architecture = [{"layer_type": "input_layer", "layer_size": 3, "activation_function": "none", "idx": 0},
                       {"layer_type": "hidden_layer", "layer_size": 3, "activation_function": "relu", "idx": 1},
                       {"layer_type": "hidden_layer", "layer_size": 3, "activation_function": "relu", "idx": 2},
                       {"layer_type": "hidden_layer", "layer_size": 3, "activation_function": "relu", "idx": 2},
                       {"layer_type": "hidden_layer", "layer_size": 3, "activation_function": "relu", "idx": 2},
                       {"layer_type": "output_layer", "layer_size": 3, "activation_function": "sigmoid", "idx": 3}]

    NeuralNetwork_Inst = NeuralNetwork(x, y, nn_architecture, 0.3, 5, loss_type="cross-entropy")
    NeuralNetwork_Inst.train(how_often=20, epochs=160)
    # NeuralNetwork_Inst.predict()
    visualize(NeuralNetwork_Inst.x_train_loss_history, NeuralNetwork_Inst.y_train_loss_history, "cross-entropy")
