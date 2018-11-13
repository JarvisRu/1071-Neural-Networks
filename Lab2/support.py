import os
from copy import deepcopy

def find_all_dataset():
    list_file = os.listdir('DataSet') 
    txt_file = []
    for f in list_file:
        if(f.endswith(".txt") or f.endswith(".TXT")):
            txt_file.append(f)
    return txt_file

def get_seperate_points(instances, labels):
    tmp = deepcopy(labels)
    tmp.sort()
    individual_labels = set(tmp)
    individual_instances = dict((label,[]) for label in individual_labels)
    for instance, label in zip(instances, labels):
        individual_instances[int(label)].append(instance)

    point_x, point_y = [], []
    index = 0
    for label in individual_instances:
        point_x.append([])
        point_y.append([])
        for instance in individual_instances[label]:
            point_x[index].append(instance[0])
            point_y[index].append(instance[1])
        index += 1
    return point_x, point_y

def get_boudary_of_axis(point_x, point_y):
    min_x = min(min(x) for x in point_x)
    max_x = max(max(x) for x in point_x)
    min_y = min(min(y) for y in point_y)
    max_y = max(max(y) for y in point_y)

    return [min_x, max_x], [min_y, max_y]