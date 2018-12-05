import perceptron
import numpy as np
import random
from copy import deepcopy

class Hopfield():
    def __init__(self):
        self.__refresh()

    def __refresh(self):
        self.inputs, self.testing_inputs = [], []
        self.to_print = []
        self.rows, self.cols = 0, 0
        self.dim, self.num_of_training, self.num_of_testing = 0, 0, 0
        self.pers = []
        self.weight_matrix = np.array([])
        self.thresolds = np.array([])
        self.record = []
        self.overallCorrectNum = 0

    def load_file(self, training_file_name, testing_file_name):
        self.__refresh()
        self.__load_training_data(training_file_name)
        self.__load_testing_data(testing_file_name)
        self.__compute_weight()
        self.__compute_thresold()
        self.__create_perceptron()

    def __load_training_data(self, file_name):
        file = open("Hopfield_dataset/" + file_name, "r")

        self.inputs, img = [], []
        for line in file:
            line = line.replace(" ","0").strip("\n")
            if len(line) == 0:
                # convert all char into int
                for i in range(len(img)):
                    if img[i] == "0":
                        img[i] = -1
                    else:
                        img[i] = 1
                self.inputs.append(np.array(img))
                img = []
            else:
                self.cols = len(line)
                img += line
                
        # for last one
        for i in range(len(img)):
            if img[i] == "0":
                img[i] = -1
            else:
                img[i] = 1
        self.inputs.append(np.array(img))

        # update info
        self.num_of_training = len(self.inputs)
        self.dim = self.inputs[0].shape[0]
        self.rows = int(self.dim / self.cols)
        
        file.close()
        
    def __load_testing_data(self, file_name):
        file = open("Hopfield_dataset/" + file_name, "r")

        self.testing_inputs, img = [], []
        for line in file:
            if len(img) == self.dim:
                # convert all char into int
                for i in range(len(img)):
                    if img[i] == "0":
                        img[i] = -1
                    else:
                        img[i] = 1
                self.testing_inputs.append(np.array(img))
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
        self.testing_inputs.append(np.array(img))

        # update file info
        self.num_of_testing = len(self.testing_inputs)
        file.close()

    def __compute_weight(self):
        original_weight = np.zeros((self.dim, self.dim))
        for img in self.inputs:
            tmp = img.reshape(self.dim, 1)
            original_weight += tmp * img
        self.weight_matrix = (1/self.dim) * (original_weight) - (self.num_of_training/self.dim) * np.eye(self.dim, dtype=int)
 
    def __compute_thresold(self):
        tmp_thresolds = []
        # compute thresold by weight
        # for row in range(self.dim):
        #     tmp_thresolds.append(sum(self.weight_matrix[row]))
        # set thresold as fixed value
        for row in range(self.dim):
            tmp_thresolds.append(0)
        self.thresolds = np.array(tmp_thresolds)

    def __create_perceptron(self):
        self.pers = []
        for row in range(self.dim):
            p = perceptron.Perceptron(self.weight_matrix[row], self.thresolds[row])
            self.pers.append(p)

    def __record_for_print(self, img, i):
        self.record[i].append(np.copy(img))

    def start_association(self):
        # refresh record
        self.record = []
        for i in range(self.num_of_testing):
            self.record.append([])

        # start association synchronously
        i = 0
        for test in self.testing_inputs:
            self.__record_for_print(test, i)
            output = np.copy(test)

            # check if is training set as testing set
            is_training_set = True if np.all(np.equal(test, self.inputs[i])) else False

            # loop until converge or correct
            is_converge = False
            is_correct = False
            while not is_converge and not is_correct:
                # compute tmp vector
                tmp_v = []
                for row in range(self.dim):
                    tmp_v.append(self.pers[row].weight.dot(test))
                tmp_v = np.array(tmp_v)

                # compare with thresold to get output
                for row in range(self.dim):
                    if tmp_v[row] > self.pers[row].thresold:
                        output[row] = 1
                    elif tmp_v[row] < self.pers[row].thresold:
                        output[row] = -1
                
                # try converge condition : output == input or already correct
                if np.all(np.equal(test, output)) :
                    is_converge = True
                else:
                    test = np.copy(output)

                # check for overall recognition
                if not is_training_set:
                    if self.__check_recog(output, i):
                        is_correct = True
                        self.overallCorrectNum += 1
                    
                self.__record_for_print(output, i)
            i += 1

        # compute overall recognition for training_set as testing_set (must wait for converge)
        if is_training_set:
            for i in range(self.num_of_testing):
                if self.__check_recog(self.record[i][-1], i):
                    self.overallCorrectNum += 1


    def __check_recog(self, img, source):
        good = 0
        for output, origin in zip(img, self.inputs[source]):
            if output == origin:
                good += 1
        return True if round(good/len(img), 5) > 0.95 else False