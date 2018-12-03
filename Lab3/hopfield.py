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
        self.rows = self.dim / self.cols
        
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
        for row in range(self.dim):
            tmp_thresolds.append(sum(self.weight_matrix[row]))
        self.thresolds = np.array(tmp_thresolds)

    def __create_perceptron(self):
        self.pers = []
        for row in range(self.dim):
            p = perceptron.Perceptron(self.weight_matrix[row], self.thresolds[row])
            self.pers.append(p)

# ----------------------------------------- Test -----------------------------------------
# to_print = []

# def get_print(output):
#     to_print.append(deepcopy(output))

# if __name__ == '__main__':

#     # get training data
#     # file_name = "Basic_Training.txt"
#     file_name = "test.txt"
#     file = open("Hopfield_dataset/" + file_name, "r")

#     inputs, img = [], []
#     for line in file:
#         if len(line.strip()) == 0:
#             # convert all char into int
#             for i in range(len(img)):
#                 if img[i] == "0":
#                     img[i] = -1
#                 else:
#                     img[i] = 1
#             # append img into inputs
#             inputs.append(np.array(img))
#             img = []
#         else:
#             line = line.replace(" ","0").strip("\n")
#             img += line

#     # for last one
#     for i in range(len(img)):
#         if img[i] == "0":
#             img[i] = -1
#         else:
#             img[i] = 1
#     inputs.append(np.array(img))

#     file.close()

#     num_of_data = len(inputs)
#     dim = inputs[0].shape[0]
#     print("inputs:", inputs)
#     # print("dim:", dim)

#     # compute w
#     original_weight = np.zeros((dim, dim))
#     for img in inputs:
#         tmp = img.reshape(dim, 1)
#         original_weight += tmp * img
#     # print(original_weight)
#     weight_matrix = (1/dim) * (original_weight) - (num_of_data/dim) * np.eye(dim, dtype=int)
#     print("weight:", weight_matrix)

#     # compute thresold
#     thresold_list = []
#     for row in range(dim):
#         thresold_list.append(sum(weight_matrix[row]))
#     thresold_v = np.array(thresold_list)
#     print("thresold:", thresold_list)

#     # create perceptron lists
#     pers = []
#     for row in range(dim):
#         p = perceptron.Perceptron(weight_matrix[row], thresold_list[row])
#         pers.append(p)

#     # for per in pers:
#     #     print(per.weight, per.thresold)

#     # get testing data
#     # file_name = "Basic_Testing.txt"
#     file_name = "test_test.txt"
#     file = open("Hopfield_dataset/" + file_name, "r")

#     testing_inputs, img = [], []
#     for line in file:
#         if len(img) == dim:
#             # convert all char into int
#             for i in range(len(img)):
#                 if img[i] == "0":
#                     img[i] = -1
#                 else:
#                     img[i] = 1
#             # append img into inputs
#             testing_inputs.append(np.array(img))
#             img = []
#         else:
#             line = line.replace(" ","0").strip("\n")
#             img += line

#     # for last one
#     for i in range(len(img)):
#         if img[i] == "0":
#             img[i] = -1
#         else:
#             img[i] = 1
#     testing_inputs.append(np.array(img))

#     file.close()

#     # start association with syn
#     for test in testing_inputs:
#         get_print(test)
#         output = deepcopy(test)

#         #loop until converge
#         is_converge = False
#         while not is_converge:
#             # compute tmp vector
#             tmp_v = []
#             for row in range(dim):
#                 tmp_v.append(pers[row].weight.dot(test))
#             tmp_v = np.array(tmp_v)

#             # compare with thresold to get output
#             for row in range(dim):
#                 # print("compare:", tmp_v[row], thresold_v[row])
#                 if tmp_v[row] > pers[row].thresold:
#                     output[row] = 1
#                 elif tmp_v[row] < pers[row].thresold:
#                     output[row] = -1
            
#             # try converge condition : output == input
#             if (test == output).all() :
#                 is_converge = True
#             else:
#                 print("not converge")
#                 print("test:", test)
#                 print("output:", output)
#                 test = deepcopy(output)
                
#             get_print(output)

#     print(to_print)
    # write in file
    # file = open('write.txt', "w")

    # for img in to_print:
    #     print_img = img.reshape(12, 9).tolist()

    #     for i in range(12):
    #         for j in range(9):
    #             if print_img[i][j] == 1:
    #                 print_img[i][j] = "1"
    #             else:
    #                 print_img[i][j] = " "
    #     print(print_img)

    #     # write to file
    #     for row in range(12):
    #         line = ''.join(print_img[row])
    #         file.write(line + '\n')
    #     file.write('-------------------\n')




