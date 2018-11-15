import perceptron
import numpy as np
from copy import deepcopy
import random
import time


class MultiClassifier():
    def __init__(self):
        self.__refresh()

    def __refresh(self):
        self.instances, self.labels = [], []
        self.dim, self.classes, self.num_of_data = 0, 0, 0
        self.individual_instances = []

        self.training_times = 0
        self.hasBoundary, self.recognition_boundary = False, 0
        self.learning_rate, self.pro_of_test = 0.0, 0.0
        self.isPocket, self.isMomentum = False, False

        self.training_instances, self.training_labels = [], []
        self.testing_instances, self.testing_labels = [], []
        self.run = 0

        self.training_recog, self.testing_recog = 0.0, 0.0
        self.training_rmse, self.testing_rmse = 0.0, 0.0

    def __regularize_input(self, inputs):
        tmp_set = set(inputs)
        classes = len(tmp_set)
        tmp = list(tmp_set)
        tmp.sort()
        if tmp[0] != 0:
            for i in range(len(inputs)):
                inputs[i] -= tmp[0]
        return classes, inputs

    def __regularize_output(self, outputs):
        result = []
        for output in outputs:
            tmp = 1 if output >= 0.5 else 0
            result.append(tmp)
        return result

    def load_file_info(self, file_name):
        file = open("DataSet/"+file_name, "r")
        self.__refresh()
        # split data
        for line in file:
            col = line.strip().split(" ")
            tmp = []
            for i in range(len(col) - 1):
                tmp.append(float(col[i]))
            self.instances.append(tmp)
            self.labels.append(float(col[-1]))
        file.close()

        # get data info
        self.dim = len(self.instances[0])
        self.num_of_data = len(self.labels)
        self.classes, self.labels = self.__regularize_input(self.labels)

    def split_individual_classes(self):
        for i in range(self.classes):
            self.individual_instances.append([])
        for instance, label in zip(self.instances, self.labels):
            self.individual_instances[label].append(instance)
        return self.individual_instances

    def initialize(self, training_times, recognition_boundary, learning_rate, pro_of_test, isPocket, isMomentum):
        self.training_times = training_times
        if recognition_boundary == 0:
            self.hasBoundary = False
        else:
            self.hasBoundary = True
            self.recognition_boundary = recognition_boundary 
        self.learning_rate = learning_rate
        self.pro_of_test = pro_of_test
        self.isPocket = isPocket
        self.isMomentum = isMomentum
        
    def split_train_test_data(self):
        propotion = (int)(self.pro_of_test * self.num_of_data)
        self.training_instances, self.training_labels = deepcopy(self.instances), deepcopy(self.labels)
        self.testing_instances, self.testing_labels = [], []
        
        num = 0
        start = random.randint(0, self.num_of_data)
        while num < propotion:
            # get index randomly
            add = random.randint(0,5)
            tmp = (start + add) % len(self.training_labels)
            start = tmp

            # do split 
            self.testing_instances.append(self.training_instances[tmp])
            del self.training_instances[tmp]
            
            self.testing_labels.append(self.training_labels[tmp])
            del self.training_labels[tmp]

            num += 1
    
    def do_training(self):
        # using only one perceptron in output layer to deal with binary class
        if self.classes == 2:
            self.__do_training_with_single()
        # using multi perceptron in output layer to deal with multi class
        else:
            self.__do_training_with_multi()

    def __new_perceptorns(self):
        self.hidden_pers = []
        self.output_pers = []
        self.history_weight, self.history_idx = [], 0

        if self.dim == 2:
            self.history_weight.append([])

        # hidden layer
        for i in range(self.dim):
            p = perceptron.Perceptron(self.dim)
            p.set_learning_rate(self.learning_rate)
            if self.dim == 2:
                self.history_weight[self.history_idx].append(p.weight)
            self.hidden_pers.append(p)
        
        # output layer
        if self.classes == 2:
            p = perceptron.Perceptron(self.dim)
            p.set_learning_rate(self.learning_rate)
            if self.dim == 2:
                self.history_weight[self.history_idx].append(p.weight)
            self.output_pers.append(p)
        else:
            for i in range(self.classes):
                p = perceptron.Perceptron(self.dim)
                p.set_learning_rate(self.learning_rate)
                if self.dim == 2:
                    self.history_weight[self.history_idx].append(p.weight)
                self.output_pers.append(p)

    def __record_weight(self):
        self.history_idx += 1
        self.history_weight.append([])
        for hidden_per in self.hidden_pers:
            self.history_weight[self.history_idx].append(hidden_per.weight)
        for output_per in self.output_pers:
            self.history_weight[self.history_idx].append(output_per.weight)

    def __do_training_with_single(self):
        
        self.__new_perceptorns()

        self.run = 0
        self.compute_rmse_times = 0
        self.rmse = []
        is_converge, over_boundary = False, False

        while (self.run < self.training_times) and (not is_converge) and (not over_boundary):
            self.run += 1
            is_converge = True

            for instance, expected_ans in zip(self.training_instances, self.training_labels):
                # feedforward
                hidden_output = []
                output_output = []
                for hidden_per in self.hidden_pers:
                    hidden_output.append(hidden_per.activate_with_sigmoid(instance))
                for output_per in self.output_pers:
                    output_output.append(output_per.activate_with_sigmoid(hidden_output))

                tmp_result = 1 if output_output[0] >= 0.5 else 0
                self.__compute_rmse(tmp_result, expected_ans)
                if tmp_result != expected_ans:
                    is_converge = False

                # back Propagation - compute local gradient
                if not is_converge:
                    self.output_pers[0].compute_local_gradient_for_output(expected_ans)

                    index = 0
                    for hidden_per in self.hidden_pers:
                        index += 1
                        hidden_per.compute_local_gradient_for_hidden(self.output_pers, index)

                    # back Propagation - adjust weight
                    for hidden_per in self.hidden_pers:
                        hidden_per.adjust_weight(instance, self.isMomentum)
                    for output_per in self.output_pers:
                        output_per.adjust_weight(hidden_output, self.isMomentum)
                    
                    # record weight into histroy_weight
                    if (self.dim == 2) and (self.run % 20 == 0):
                        self.__record_weight()

                    # training again with new weight
                    if self.hasBoundary:
                        if self.get_recognition(0) > self.recognition_boundary:
                            print("over_boundary!!")
                            over_boundary = True
                    break

        # record final weight
        if self.dim == 2:
            self.__record_weight()

    def __do_training_with_multi(self):
        
        self.__new_perceptorns()

        self.run = 0
        self.rmse = []
        is_converge, over_boundary = False, False
        
        while (self.run < self.training_times) and (not is_converge) and (not over_boundary):
            self.run += 1
            is_converge = True

            for instance, expected_ans in zip(self.training_instances, self.training_labels):
                # feedforward
                hidden_output = []
                output_output = []
                for hidden_per in self.hidden_pers:
                    hidden_output.append(hidden_per.activate_with_sigmoid(instance))
                for output_per in self.output_pers:
                    output_output.append(output_per.activate_with_sigmoid(hidden_output))

                tmp_result = self.__regularize_output(output_output)
                self.__compute_rmse(tmp_result, expected_ans)
                for i in range(len(tmp_result)):
                    if i != expected_ans and tmp_result[i] == 1:
                        is_converge = False
                        break
                    if i == expected_ans and tmp_result[i] != 1:
                        is_converge = False
                        break

                # back Propagation - compute local gradient
                if not is_converge:
                    index, d = 0, 0
                    for output_per in self.output_pers:
                        d = 1 if index == expected_ans else 0
                        output_per.compute_local_gradient_for_output(d)
                        index += 1
                    index = 0
                    for hidden_per in self.hidden_pers:
                        index += 1
                        hidden_per.compute_local_gradient_for_hidden(self.output_pers, index)

                    # back Propagation - adjust weight
                    for hidden_per in self.hidden_pers:
                        hidden_per.adjust_weight(instance, self.isMomentum)
                    for output_per in self.output_pers:
                        output_per.adjust_weight(hidden_output, self.isMomentum)
                    
                    # record weight into histroy_weight
                    if (self.dim == 2) and (self.run % 20 == 0):
                        self.__record_weight()

                    # training again with new weight
                    if self.hasBoundary:
                        if self.get_recognition(0) > self.recognition_boundary:
                            print("over_boundary!!")
                            over_boundary = True
                    break

        # record final weight
        if self.dim == 2:
            self.__record_weight()

    def get_recognition(self, mode):
        if self.classes == 2:
            return self.__get_recognition_single(mode)
        else:
            return self.__get_recognition_multi(mode)

    def __get_recognition_single(self, mode):
        instances = self.training_instances if mode == 0 else self.testing_instances
        labels = self.training_labels if mode == 0 else self.testing_labels
        successful_num = 0
        for instance, expected_ans in zip(instances, labels):
            # feedforward
            hidden_output = []
            for hidden_per in self.hidden_pers:
                hidden_output.append(hidden_per.activate_with_sigmoid(instance))
            output = self.output_pers[0].activate_with_sigmoid(hidden_output)

            tmp_result = 1 if output >= 0.5 else 0
            if tmp_result == expected_ans:
                successful_num += 1
        recog = successful_num / len(instances)
        if mode == 0:
            self.training_recog = recog
        else:
            self.testing_recog = recog
        return recog
    
    def __get_recognition_multi(self, mode):
        instances = self.training_instances if mode == 0 else self.testing_instances
        labels = self.training_labels if mode == 0 else self.testing_labels
        successful_num = 0
        for instance, expected_ans in zip(instances, labels):
            # feedforward
            hidden_output = []
            output_output = []
            for hidden_per in self.hidden_pers:
                hidden_output.append(hidden_per.activate_with_sigmoid(instance))
            for output_per in self.output_pers:
                output_output.append(output_per.activate_with_sigmoid(hidden_output))
            
            # regularization
            tmp_result = self.__regularize_output(output_output)
            is_successful = True
            for i in range(len(tmp_result)):
                if i != expected_ans and tmp_result[i] == 1:
                    is_successful = False
                    break
                if i == expected_ans and tmp_result[i] != 1:
                    is_successful = False
                    break
            if is_successful:
                successful_num += 1
        recog = successful_num / len(instances)
        if mode == 0:
            self.training_recog = recog
        else:
            self.testing_recog = recog
        return recog

    def get_rmse(self):
        tmp = sum(self.rmse) / len(self.rmse)
        return np.sqrt(tmp)

    def __compute_rmse(self, predict, expected):
        if self.classes == 2:
            tmp = (predict - expected)**2 / 2
            self.rmse.append(tmp)
        else:
            tmp = 0
            for i in range(len(predict)):
                if i != expected:
                    tmp += (predict[i] - 0)**2
                else:
                    tmp += (predict[i] - 1)**2
            tmp /= 2
            self.rmse.append(tmp)

