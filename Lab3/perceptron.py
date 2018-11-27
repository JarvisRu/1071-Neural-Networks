import numpy as np
import math
import random

class Perceptron():
    def __init__(self, weight, thresold):
        self.weight, self.thresold = weight, thresold
