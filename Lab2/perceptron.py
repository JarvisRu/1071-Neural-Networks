import numpy as np
import math
import random

class Perceptron():
    def __init__(self, dim):
        self.dim = dim + 1
        self.weight = np.zeros((1, self.dim), float)
        self.local_gradient = 0
        self.learning_rate = 0.5
        self.output = 0
        self.__initialize_weight()

        # for momentum
        self.momentum_alpha = 0.9
        self.previous_change = np.zeros((1, self.dim), float)

    def __initialize_weight(self):
        for i in range(self.dim):
            self.weight[0][i] = self.__get_random_weight()

    def __get_random_weight(self):
        return round(random.uniform(-1, 1), 3)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
    
    def set_weight(self, weight):
        self.weight = np.array([ weight ])

    def activate_with_sigmoid(self, input):
        tmp = [-1]
        for i in input:
            tmp.append(i)
        real_input = np.array(tmp)
        v = np.sum(self.weight * real_input)
        self.output = 1 / (1 + np.exp(-1 * v))
        return self.output

    def compute_local_gradient_for_output(self, d):
        self.local_gradient = (d - self.output) * self.output * (1 - self.output)


    def compute_local_gradient_for_hidden(self, output_pers, no):
        sigma_of_pers = 0
        for per in output_pers:
            sigma_of_pers += ( per.local_gradient * np.asscalar(per.weight[0][no]))
        self.local_gradient = self.output * (1 - self.output) * sigma_of_pers

    def adjust_weight(self, input, momentum):
        if not momentum:
            self.__adjust_weight_normal(input)
        else:
            self.__adjust_weight_momentum(input)

    def __adjust_weight_normal(self, input):
        tmp = [-1]
        for i in input:
            tmp.append(i)
        real_input = np.array(tmp)
        self.weight = self.weight + self.learning_rate * (self.local_gradient * real_input)
    
    def __adjust_weight_momentum(self, input):
        tmp = [-1]
        for i in input:
            tmp.append(i)
        real_input = np.array(tmp)
        change = (self.momentum_alpha * self.previous_change) + self.learning_rate * (self.local_gradient * real_input)
        self.previous_change = change
        self.weight = self.weight + change