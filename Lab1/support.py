import matplotlib.pyplot as plt
from copy import deepcopy
import random
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

def get_individual_label(label):
    labels = set(label)
    individual_label = list(labels)
    individual_label.sort()
    return individual_label

# treat all noise as class 2
def handle_as_noise(label, individual_label):
    for i in range(len(individual_label) - 2):
        for l in label:
            if(l == individual_label[i+2]):
                l = individual_label[1]
    return label

def find_x2(w0, w1, w2, x1):
    return (w0-(w1*x1)) / w2

def get_random_weight():
    return round(random.uniform(-1, 1), 2)

def get_seperate_points(feature, label, individual_label):
    label1_x1 = []
    label1_x2 = []
    label2_x1 = []
    label2_x2 = []
    for x1, x2, expected_ans in zip(feature[0], feature[1], label):
        if expected_ans == individual_label[0]:
            label1_x1.append(float(x1))
            label1_x2.append(float(x2))
        else:
            label2_x1.append(float(x1))
            label2_x2.append(float(x2))
    return label1_x1, label1_x2, label2_x1, label2_x2

def split_train_test_data(feature, label, pro):
    propotion = (int)(pro * len(label))
    feature_train, label_train = deepcopy(feature), deepcopy(label)
    feature_test = [[], []]
    label_test = []
    num = 0
    start = random.randint(0, len(label))
    while(num < propotion):
        # get index randomly
        add = random.randint(0,5)
        tmp = (start + add) % len(label_train)
        start = tmp
        # do split 
        feature_test[0].append(feature_train[0][tmp])
        del feature_train[0][tmp]
        feature_test[1].append(feature_train[1][tmp])
        del feature_train[1][tmp]
        label_test.append(label_train[tmp])
        del label_train[tmp]
        num += 1
    return feature_train, label_train, feature_test, label_test

def do_training(feature, label, individual_label, weight, learning_rate, run_limit, canvas, ax):
    run = 0
    converge = False
    x0 = -1
    proc_weight = []
    while (not converge) and (run < run_limit):
        converge = True
        for x1, x2, expected_ans in zip(feature[0], feature[1], label):
            tmp = weight[0] * x0 + weight[1] * x1 + weight[2] * x2
            tmp_ans = individual_label[1] if tmp > 0 else individual_label[0]
            if tmp_ans != expected_ans:
                arg = -1 if (tmp_ans > expected_ans) else 1
                weight[0] = weight[0] + arg * learning_rate * x0 
                weight[1] = weight[1] + arg * learning_rate * x1
                weight[2] = weight[2] + arg * learning_rate * x2
                converge = False
                run += 1
                if run % 5 == 0:
                    tmp = deepcopy(weight)
                    proc_weight.append(tmp)
                break
    proc_weight.append(weight)
    return weight, run, proc_weight


def get_recognition(feature, label, weight, individual_label):
    total_num = len(label)
    error_num = 0
    for x1, x2, expected_ans in zip(feature[0], feature[1], label):
        tmp = weight[0] * -1 + weight[1] * x1 + weight[2] * x2
        tmp_ans = individual_label[1] if tmp > 0 else individual_label[0]
        if tmp_ans != expected_ans:
            error_num += 1
    return round(float((total_num - error_num) / total_num) * 100, 3)
