import perceptron
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import random
import math
import time
import os

def regularization_output(output):
    result = []
    for output in output:
        tmp = 1 if output >= 0.5 else 0
        result.append(tmp)
    return result

def regularization_input(inputs):
    tmp_set = set(inputs)
    classes = len(tmp_set)
    tmp = list(tmp_set)
    tmp.sort()
    if tmp[0] != 0:
        for i in range(len(inputs)):
            inputs[i] -= tmp[0]
    return classes, inputs

def get_recognition_multi(instances, label, hidden_pers, output_pers):
    successful_num = 0
    for instance, expected_ans in zip(instances, label):
        # feedforward
        hidden_output = []
        output_output = []
        for hidden_per in hidden_pers:
            hidden_output.append(hidden_per.activate_with_sigmoid(instance))
        for output_per in output_pers:
            output_output.append(output_per.activate_with_sigmoid(hidden_output))
        
        # regularization
        tmp_result = regularization_output(output_output)
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
    return recog

if __name__ == '__main__':

    file_name = "2Ccircle1.txt"
    # file_name = "2CloseS.txt"
    # file_name = "test.txt"
    # file_name = "perceptron4.txt"
    file = open("DataSet/"+file_name, "r")

    # split data
    instances = []
    label = []
    for line in file:
        col = line.strip().split(" ")
        tmp = []
        for i in range(len(col) - 1):
            tmp.append(float(col[i]))
        instances.append(tmp)
        label.append(int(col[-1]))
    file.close()

    # get data info
    dim = len(instances[0])
    classes, label = regularization_input(label)
    print(set(label))

    # create perceptron lists
    hidden_pers = []
    for i in range(dim):
        p = perceptron.Perceptron(dim)
        p.set_learning_rate(0.8)
        hidden_pers.append(p)
    
    output_pers = []
    for i in range(classes):
        p = perceptron.Perceptron(dim)
        p.set_learning_rate(0.8)
        output_pers.append(p)

    # do training
    training_times = 2000
    boundary = 0.8
    run = 0
    is_converge = False
    over_boundary = False
    while (run < training_times) and (not is_converge) and (not over_boundary):
        run += 1
        is_converge = True
        for instance, expected_ans in zip(instances, label):
            # feedforward
            hidden_output = []
            output_output = []
            for hidden_per in hidden_pers:
                hidden_output.append(hidden_per.activate_with_sigmoid(instance))
            for output_per in output_pers:
                output_output.append(output_per.activate_with_sigmoid(hidden_output))

            tmp_result = regularization_output(output_output)
            for i in range(len(tmp_result)):
                if i != expected_ans and tmp_result[i] == 1:
                    is_converge = False
                    break
                if i == expected_ans and tmp_result[i] != 1:
                    is_converge = False
                    break

            # back Propagation - compute local gradient
            if not is_converge:
                # print("into Propagation")
                index = 0
                d = 0
                for output_per in output_pers:
                    d = 1 if index == expected_ans else 0
                    output_per.compute_local_gradient_for_output(d)
                    index += 1
                index = 0
                for hidden_per in hidden_pers:
                    index += 1
                    hidden_per.compute_local_gradient_for_hidden(output_pers, index)

                # back Propagation - adjust weight
                for hidden_per in hidden_pers:
                    hidden_per.adjust_weight(instance)
                for output_per in output_pers:
                    output_per.adjust_weight(hidden_output)
                
                # retraining again with new weight
                if get_recognition_multi(instances, label, hidden_pers, output_pers) > boundary:
                    print("over boundary!!")
                    over_boundary = True
                break
                
    # final percptron
    print("========Final============")
    print("final run times: ", run)
    
    print(get_recognition_multi(instances, label, hidden_pers, output_pers))