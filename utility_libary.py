# logging
import logging
from datetime import date
import os
import pathlib

from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize(x_train_loss_history: List[int], y_train_loss_history: List[float]):
    """
    Used to visualize the loss of the neural network.
    """
    data = {"x": x_train_loss_history, "train": y_train_loss_history}
    data = pd.DataFrame(data, columns=["x", "train"])

    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="x", y="train", data=data, label="train", color="orange")
    plt.xlabel("Time In Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Time")
    plt.show()


# mean square root
def loss_mse(target: List[float], pred_target: List[float]) -> List[float]:
    """
    Calculates the "Mean-square-loss" of the "target" and "pred_target".

    Parameters
    ----------
    target: List[float]
        the "true" labels for the for the current data set.
    pred_target: List[float]
        the predicted labels of our Neural Network.

    Returns
    -------
    List with loss values in it
    """
    loss = np.sum(1 / 2 * (target - pred_target) ** 2)
    return loss


def loss_mse_derivative(y: List[float], y_hat: List[float]) -> List[float]:
    """
    Calculates the derivative for the "Mean-square-loss" of the "target" and "pred_target".

    Parameters
    ----------
    target: List[float]
        the "true" labels for the for the current data set.
    pred_target: List[float]
        the predicted labels of our Neural Network.

    Returns
    -------
    List with derivatives of the loss values in it
    """
    y = np.array([y]).T
    tmp_dev = -(y - y_hat)
    loss_dev = np.array(tmp_dev)
    return loss_dev


def loss_cross_entropy(target: List[float], pred_target: List[float]) -> List[float]:
    """
    Calculates the "cross-entropy-loss" of the "target" and "pred_target".

    Parameters
    ----------
    target: List[float]
        the "true" labels for the for the current data set.
    pred_target: List[float]
        the predicted labels of our Neural Network.

    Returns
    -------
    List with loss values in it
    """
    loss = []
    for idx, pred_target_item in enumerate(pred_target):
        if target[idx] == 1:
            tmp_loss = -np.log10(pred_target_item)
            loss.append(tmp_loss)
        else:
            tmp_loss = -np.log10(1 - pred_target_item)
            loss.append(tmp_loss)

    loss = np.sum(loss)
    return loss


def loss_cross_entropy_derivative(target: List[float], pred_target: List[float]) -> List[float]:
    """
    Calculates the derivative for the "cross-entropy-loss" of the "target" and "pred_target".

    Parameters
    ----------
    target: List[float]
        the "true" labels for the for the current data set.
    pred_target: List[float]
        the predicted labels of our Neural Network.

    Returns
    -------
    List with derivatives of the loss values in it
    """
    target = target.tolist()
    pred_target = pred_target.flatten().tolist()

    loss = []
    for idx, pred_target_item in enumerate(pred_target):
        if target[idx] == 1:
            tmp_loss = -1 * (1 / float(pred_target_item))
            loss.append(tmp_loss)
        else:
            tmp_loss = 1 / (1 - float(pred_target_item))
            loss.append(tmp_loss)

    return np.array(loss)


def tanh(x: List[float]) -> List[float]:
    """
    Calculates the "tanh" of x. Used as activation function for the Neural Network.

    Parameters
    ----------
    x: List[float]
        X-Input that should be "activated".
    Returns
    -------
    List with the "activated" values.
    """
    return 2 / (1 + np.exp(-2 * x)) - 1


def tanh_derivative(x: List[float]) -> List[float]:
    """
    Calculates derivative of the "tanh" function with respect to x. Used in backpropagation of the Neural Network.

    Parameters
    ----------
    x: List[float]
        X-Input with what we differentiate the tanh function.
    Returns
    -------
    List with the values of the derivative of the tanh function.
    """
    return 1 - (tanh(x) ** 2)


def sigmoid(x: List[float]) -> List[float]:
    """
    Calculates the "sigmoid" of x. Used as activation function for the Neural Network.

    Parameters
    ----------
    x: List[float]
      X-Input that should be "activated".
    Returns
    -------
    List with the "activated" values.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: List[float]) -> List[float]:
    """
    Calculates derivative of the "sigmoid" function with respect to x. Used in backpropagation of the Neural Network.

    Parameters
    ----------
    x: List[float]
        X-Input with what we differentiate the "sigmoid" function.
    Returns
    -------
    List with the values of the derivative of the "sigmoid" function.
    """
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x: List[float]) -> List[float]:
    """
    Calculates the "relu" of x. Used as activation function for the Neural Network.

    Parameters
    ----------
    x: List[float]
      X-Input that should be "activated".
    Returns
    -------
    List with the "activated" values.
    """
    return np.maximum(0, x)


def relu_derivative(x: List[float]) -> List[float]:
    """
    Calculates derivative of the "relu" function with respect to x. Used in backpropagation of the Neural Network.

    Parameters
    ----------
    x: List[float]
       X-Input with what we differentiate the "relu" function.
    Returns
    -------
    List with the values of the derivative of the "relu" function.
    """
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def linear(x: List[float]) -> List[float]:
    """
    Calculates the "linear" of x. Used as activation function for the Neural Network.

    Parameters
    ----------
    x: List[float]
      X-Input that should be "activated".
    Returns
    -------
    List with the "activated" values.
    """
    return x


def init_logging(level_of_debugging: str) -> object:
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
    # create a directory for "logs" if the directory doesn't exist
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


def activation_derivative(layer: Dict, curr_layer: List[float]) -> List[float]:
    """
    Calls the derivative of the right "activation function".

    Parameters
    ----------
    layer: Dict
        Current layer we're in when we iterate over our Neural Network structure(a dictionary).
    curr_layer: List[float]
        The input for the current layer we're in.
    Returns
    -------
    An array with the derivative of the "activation-function".
    """
    if layer["activation_function"] == "linear":
        return np.array(linear(curr_layer))

    elif layer["activation_function"] == "relu":
        return np.array(relu_derivative(curr_layer))

    elif layer["activation_function"] == "sigmoid":
        return np.array(sigmoid_derivative(curr_layer))

    else:
        raise Exception("Activation function not supported!")


def loss_type_choice(y: List[float], y_hat: List[float], loss_type: str, derivative: bool = False):
    """
    This function chooses the appropiates loss function depending on the loss type
    Parameters
    ----------
    loss_type: type of loss for example "mse" or "cross entropy"

    Returns
    -------

    """
    if not derivative:
        if loss_type.lower() == "mse":
            return loss_mse(y, y_hat)

        elif "cross" in loss_type.lower() and "entropy" in loss_type.lower():
            return loss_cross_entropy(target=y, pred_target=y_hat)

        else:
            raise NotImplementedError("There is no loss with that name.")


    elif derivative:
        if loss_type.lower() == "mse":
            return loss_mse_derivative(y, y_hat)

        elif "cross" in loss_type.lower() and "entropy" in loss_type.lower():
            return loss_cross_entropy_derivative(target=y, pred_target=y_hat)

        else:
            raise NotImplementedError("There is no loss with that name.")


    else:
        raise ValueError("Variable derivative not set for loss function.")