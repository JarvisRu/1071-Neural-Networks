import perceptron
import numpy as np
import random
import math

if __name__ == '__main__':

    # file_name = "Basic_Training.txt"
    file_name = "test.txt"
    file = open("Hopfield_dataset/" + file_name, "r")

    # get data
    inputs, img = [], []
    for line in file:
        if len(line.strip()) == 0:
            # convert all char into int
            for i in range(len(img)):
                if img[i] == "0":
                    img[i] = -1
                else:
                    img[i] = 1
            # append img into inputs
            inputs.append(np.array(img))
            img = []
        else:
            line = line.replace(" ","0").strip("\n")
            img += line

    # for last one
    for i in range(len(img)):
        if img[i] == "0":
            img[i] = -1
        else:
            img[i] = 1
    inputs.append(np.array(img))

    file.close()

    num_of_data = len(inputs)
    dim = inputs[0].shape[0]
    print(inputs)
    print(dim)

    # compute w
    original_weight = np.zeros((dim, dim))
    for img in inputs:
        tmp = img.reshape(dim, 1)
        original_weight += tmp * img
    print(original_weight)
    weight_matrix = (1/dim) * (original_weight) - (num_of_data/dim) * np.eye(dim, dtype=int)
    print(weight_matrix)

    # compute thresold
    thresold_list = []
    for row in range(dim):
        thresold_list.append(sum(weight_matrix[row]))
    print(thresold_list)

    # create perceptron lists
    pers = []
    for row in range(dim):
        p = perceptron.Perceptron(weight_matrix[row], thresold_list[row])
        pers.append(p)

    for per in pers:
        print(per.weight, per.thresold)
