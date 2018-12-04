import os
from copy import deepcopy

def find_training_dataset():
    list_file = os.listdir('Hopfield_dataSet') 
    txt_file = []
    for f in list_file:
        if(f.endswith("Training.txt") or f.endswith("Training.TXT")):
            txt_file.append(f)
    return txt_file

def find_testing_dataset():
    list_file = os.listdir('Hopfield_dataSet') 
    txt_file = []
    for f in list_file:
        if(f.endswith(".txt")):
            txt_file.append(f)
    return txt_file

def check_file(training_file_name, testing_file_name):
    error, over = False, False
    while not error and not over:
        for i, j in zip(training_file_name, testing_file_name):
            if i != '_' and j != '_':
                error = True if i != j else False
            elif i == j and i == '_':
                over = True
                break
            else:
                error = True
                break
    return error

def get_recog(trainings, testings):
    good = 0
    for train, test in zip(trainings, testings):
        if train == test:
            good += 1
    return round(good/len(trainings), 5)
                    