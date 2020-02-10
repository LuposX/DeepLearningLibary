import numpy as np


def forward(data, weight):
    return np.dot(weight, data.T)

