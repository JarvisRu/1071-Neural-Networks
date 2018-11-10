import perceptron
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import random
import math
import time
import os

def find_all_dataset():
    list_file = os.listdir('DataSet') 
    txt_file = []
    for f in list_file:
        if(f.endswith(".txt")):
            txt_file.append(f)
    return txt_file

def load_file_info(file_name):
    file = open("DataSet/"+file_name, "r")
    # using first line to get dimension
    col = file.readline().strip().split(" ")
    dim = len(col) - 1
    feature = []
    label = []
    for n in range(dim):
        feature.append([])
    for n in range(dim):
        feature[n].append(float(col[n]))
    label.append(int(col[-1]))
    for line in file:
        col = line.strip().split(" ")
        for n in range(dim):
            feature[n].append(float(col[n]))
        label.append(int(col[-1]))
    file.close()
    return feature, label

def regularization_output(outputs):
    result = []
    for output in outputs:
        tmp = 1 if outputs > 0.5 else 0
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

def get_recognition(instances, label, hidden_pers, output_pers):
    successful_num = 0
    for instance, expected_ans in zip(instances, label):
        # feedforward
        hidden_output = []
        for hidden_per in hidden_pers:
            hidden_output.append(hidden_per.activate_with_sigmoid(instance))
        output = output_pers[0].activate_with_sigmoid(hidden_output)

        tmp_result = 1 if output >= 0.5 else 0
        if tmp_result == expected_ans:
            successful_num += 1
    recog = successful_num / len(instances)
    return recog



if __name__ == '__main__':

    file_name = "2Ccircle1.txt"
    # file_name = "test.txt"
    # file_name = "perceptron1.txt"
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
    # hidden_pers = []
    # p1 = perceptron.Perceptron(dim)
    # p2 = perceptron.Perceptron(dim)
    # p1.set_weight([-1.2, 1, 1])
    # p2.set_weight([0.3, 1, 1])
    # hidden_pers.append(p1)
    # hidden_pers.append(p2)
    hidden_pers = []
    for i in range(dim):
        p = perceptron.Perceptron(dim)
        p.set_learning_rate(0.7)
        hidden_pers.append(p)
    
    output_pers = []
    p3 = perceptron.Perceptron(dim)
    p3.set_weight([0.5, 0.4, 0.8])
    output_pers.append(p3)

    # do training
    training_times = 5000
    boundary = 0.8
    run = 0
    is_converge = False
    over_boundary = False
    while (run < training_times) and (not is_converge) and (not over_boundary):
        run += 1
        is_converge = True
        for instance, expected_ans in zip(instances, label):
            # print("==========================")
            # feedforward
            hidden_output = []
            output_output = []
            for hidden_per in hidden_pers:
                hidden_output.append(hidden_per.activate_with_sigmoid(instance))
            for output_per in output_pers:
                output_output.append(output_per.activate_with_sigmoid(hidden_output))

            # print(hidden_output)
            # print(output_output)

            tmp_result = 1 if output_output[0] >= 0.5 else 0
            if tmp_result != expected_ans:
                is_converge = False

            # back Propagation - compute local gradient
            if not is_converge:
                # print("into Propagation")
                output_pers[0].compute_local_gradient_for_output(expected_ans)

                index = 0
                for hidden_per in hidden_pers:
                    index += 1
                    hidden_per.compute_local_gradient_for_hidden(output_pers, index)

                # back Propagation - adjust weight
                for hidden_per in hidden_pers:
                    hidden_per.adjust_weight(instance)
                for output_per in output_pers:
                    output_per.adjust_weight(hidden_output)
                
                # training again with new weight
                if get_recognition(instances, label, hidden_pers, output_pers) > boundary:
                    print("over_boundary!!")
                    over_boundary = True
                break
                
    # final percptron
    print("========Final============")
    print("final run times: ", run)
    print(get_recognition(instances, label, hidden_pers, output_pers))