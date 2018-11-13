import support
import multiClassifier
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, pyqtSlot, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QGroupBox, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QSpinBox, QComboBox, QCheckBox)
from matplotlib.backends.backend_qt5agg import FigureCanvas 
from matplotlib.figure import Figure
from copy import deepcopy


class MultiPerceptronView(QWidget):
    def __init__(self):
        super().__init__()
        self.__initUI()
        self.classifier = multiClassifier.MultiClassifier()

    def __initUI(self):
        self.__window_layout = QHBoxLayout()
        self.__status_layout = QVBoxLayout()
        self.__result_layout = QVBoxLayout()

        self.__set_file_box_UI()
        self.__set_dataInfo_box_UI()
        self.__set_setting_box_UI()
        self.__set_control_box_UI()
        self.__set_figure_box_UI()
        self.__set_result_box_UI()

        self.__status_layout.addWidget(self.__file_box, 1)
        self.__status_layout.addWidget(self.__dataInfo_box, 2)
        self.__status_layout.addWidget(self.__setting_box, 3)
        self.__status_layout.addWidget(self.__control_box, 1)

        self.__result_layout.addWidget(self.__figure_box, 3)
        self.__result_layout.addWidget(self.__result_box, 1)

        self.__window_layout.addLayout(self.__status_layout, 1)
        self.__window_layout.addLayout(self.__result_layout, 1)
        self.__window_layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(self.__window_layout)

        self.color = ["blue", "green", "purple", "orange", "black", "pink"]

    def __set_file_box_UI(self):
        self.__file_box = QGroupBox('File')
        hbox = QHBoxLayout()

        file_label = QLabel("Select File :")
        file_label.setAlignment(Qt.AlignCenter)
        self.file_cb = QComboBox()
        self.file_cb.addItems(support.find_all_dataset())
        self.file_cb.setStatusTip("Please select a file as dataset")
        self.file_search_btn = QPushButton('Load File', self)
        self.file_search_btn.setStatusTip("Load file and update file information")
        self.file_search_btn.clicked.connect(self.load_file)

        hbox.addWidget(file_label, 1)
        hbox.addWidget(self.file_cb, 3)
        hbox.addWidget(self.file_search_btn, 1)
        self.__file_box.setLayout(hbox)

    def __set_dataInfo_box_UI(self):
        self.__dataInfo_box = QGroupBox('Data Information')
        vbox = QVBoxLayout()
        name_box = QHBoxLayout()
        number_of_instances_box = QHBoxLayout()
        number_of_feature_box = QHBoxLayout()
        number_of_class_box = QHBoxLayout()

        name_label = QLabel("DataSet Name :")
        name_label.setAlignment(Qt.AlignCenter)
        self.name_text = QLabel(" -- ")
        self.name_text.setAlignment(Qt.AlignCenter)
        name_box.addWidget(name_label, 1)
        name_box.addWidget(self.name_text, 2)

        number_of_instances_label = QLabel("Number of Instances :")
        number_of_instances_label.setAlignment(Qt.AlignCenter)
        self.number_of_instances_text = QLabel(" -- ")
        self.number_of_instances_text.setAlignment(Qt.AlignCenter)
        number_of_instances_box.addWidget(number_of_instances_label, 1)
        number_of_instances_box.addWidget(self.number_of_instances_text, 2)
        
        number_of_feature_label = QLabel("Number of Features :")
        number_of_feature_label.setAlignment(Qt.AlignCenter)
        self.number_of_feature_text = QLabel(" -- ")
        self.number_of_feature_text.setAlignment(Qt.AlignCenter)
        number_of_feature_box.addWidget(number_of_feature_label, 1)
        number_of_feature_box.addWidget(self.number_of_feature_text, 2)
        
        number_of_class_label = QLabel("Number of Classes :")
        number_of_class_label.setAlignment(Qt.AlignCenter)
        self.number_of_class_text = QLabel(" -- ")
        self.number_of_class_text.setAlignment(Qt.AlignCenter)
        number_of_class_box.addWidget(number_of_class_label, 1)
        number_of_class_box.addWidget(self.number_of_class_text, 2)

        vbox.addLayout(name_box)
        vbox.addLayout(number_of_instances_box)
        vbox.addLayout(number_of_feature_box)
        vbox.addLayout(number_of_class_box)
        self.__dataInfo_box.setLayout(vbox)
        
    def __set_setting_box_UI(self):
        self.__setting_box = QGroupBox('Setting')
        # entire setting_box
        vbox = QVBoxLayout()
        # individual section
        # perceptron_box = QHBoxLayout()
        learning_spilt_box = QHBoxLayout()
        converge_condition_box = QVBoxLayout()
        optimized_box = QVBoxLayout()

        # perceptorn section
        # num_Hperceptron_label = QLabel("Number of hidden perceptrons :")
        # self.num_Hperceptron_sbox = QSpinBox()
        # self.num_Hperceptron_sbox.setRange(2,10)
        # self.num_Hperceptron_sbox.setStatusTip('Setting number of hidden perceptrons as ...')

        # perceptron_box.addWidget(num_Hperceptron_label,3)
        # perceptron_box.addWidget(self.num_Hperceptron_sbox,1)

        # learning rate & split section
        learning_rate_label = QLabel("Learning Rate :")
        self.learning_rate_text = QLineEdit()
        self.learning_rate_text.setStatusTip('The learning rate of training')
        learning_spilt_box.addWidget(learning_rate_label, 2)
        learning_spilt_box.addWidget(self.learning_rate_text, 2)
        learning_spilt_box.addStretch(1)

        propotion_of_test_label = QLabel("Testing Data (%) :")
        self.propotion_of_test_text = QLineEdit()
        self.propotion_of_test_text.setStatusTip('testing_data / all_data = ?, default = 0.33')

        learning_spilt_box.addWidget(propotion_of_test_label, 2)
        learning_spilt_box.addWidget(self.propotion_of_test_text, 2)

        # converge condition section 
        converge_condition_label = QLabel("Converge Condition :")
        
        self.training_times_cbox = QCheckBox("Training Times")
        self.training_times_cbox.setChecked(False)
        self.training_times_cbox.stateChanged.connect(self.cbox_change)
        self.training_times_text = QLineEdit()
        self.training_times_text.setStatusTip('Using training times as converge condition')

        self.recognition_cbox = QCheckBox("Recognition")
        self.recognition_cbox.setChecked(False)
        self.recognition_cbox.stateChanged.connect(self.cbox_change)
        self.recognition_text = QLineEdit()
        self.recognition_text.setStatusTip('If training recognition over than ...')

        item_hbox = QHBoxLayout()
        item_hbox.addWidget(self.training_times_cbox,3)
        item_hbox.addWidget(self.training_times_text,3)
        item_hbox.addStretch(1)
        item_hbox.addWidget(self.recognition_cbox,3)
        item_hbox.addWidget(self.recognition_text,3)
        converge_condition_box.addWidget(converge_condition_label)
        converge_condition_box.addLayout(item_hbox)

        # optimized section 
        optimized_label = QLabel("Optimized :")

        self.pocket_cbox = QCheckBox("Pocket")
        self.pocket_cbox.setStatusTip('Using pocket algorithm')

        self.momentum_cbox = QCheckBox("Momentum")
        self.momentum_cbox.setStatusTip('Using momentum to optimize learning')

        option_hbox = QHBoxLayout()
        option_hbox.addWidget(self.pocket_cbox,3)
        option_hbox.addStretch(1)
        option_hbox.addWidget(self.momentum_cbox,3)
        converge_condition_box.addWidget(optimized_label)
        optimized_box.addLayout(option_hbox)
    
        # fit all layout in
        # vbox.addLayout(perceptron_box)
        vbox.addLayout(learning_spilt_box)
        vbox.addLayout(converge_condition_box)
        vbox.addLayout(optimized_box)
        self.__setting_box.setLayout(vbox)
        
    def __set_control_box_UI(self):
        self.__control_box = QGroupBox('Control Panel')
        vbox = QVBoxLayout()
        control_box = QHBoxLayout()
        start_box = QHBoxLayout()

        self.confirm_btn = QPushButton("Confirm")
        self.confirm_btn.setStatusTip('Confirm to use the setting data above ?')
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self.confirm_data)

        self.load_training_data_btn = QPushButton("Load training data")
        self.load_training_data_btn.setStatusTip('Randomly Split dataset into training part and testing part. Load training data and draw it.')
        self.load_training_data_btn.setEnabled(False)
        self.load_training_data_btn.clicked.connect(self.load_training)

        self.load_testing_data_btn = QPushButton("Load testing data")
        self.load_testing_data_btn.setStatusTip('Randomly Split dataset into training part and testing part. Load testing data and draw it.')
        self.load_testing_data_btn.setEnabled(False)
        self.load_testing_data_btn.clicked.connect(self.load_testing)

        control_box.addWidget(self.confirm_btn, 2)
        control_box.addWidget(self.load_training_data_btn, 2)
        control_box.addWidget(self.load_testing_data_btn, 2)

        self.redo_btn = QPushButton("Redo Training")
        self.redo_btn.setStatusTip('Reset all parameter and redo again for better result.')
        self.redo_btn.setEnabled(False)
        self.redo_btn.clicked.connect(self.redo)

        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.setStatusTip('Start training data with parameter above.')
        self.start_training_btn.setEnabled(False)
        self.start_training_btn.clicked.connect(self.start_training)

        self.start_testing_btn = QPushButton("Start Testing")
        self.start_testing_btn.setStatusTip('Testing remaining dataset with weight which get from training.')
        self.start_testing_btn.setEnabled(False)
        self.start_testing_btn.clicked.connect(self.start_testing)

        start_box.addWidget(self.redo_btn, 2)
        start_box.addWidget(self.start_training_btn, 2)
        start_box.addWidget(self.start_testing_btn, 2)

        vbox.addLayout(control_box)
        vbox.addLayout(start_box)
        self.__control_box.setLayout(vbox)

    def __set_figure_box_UI(self):
        self.__figure_box = QGroupBox('Figure')
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        hbox = QHBoxLayout()
        hbox.addWidget(self.canvas)
        self.__figure_box.setLayout(hbox)

    def __set_result_box_UI(self):
        self.__result_box = QGroupBox('Result')
        vbox = QVBoxLayout()
        weight_box = QHBoxLayout()
        tranining_times_result_box = QHBoxLayout()
        training_recognition_box = QHBoxLayout()
        training_rmse_box = QHBoxLayout()
        testing_recognition_box = QHBoxLayout()
        testing_rmse_box = QHBoxLayout()

        weight_result_label = QLabel("Weight : ")
        weight_result_label.setAlignment(Qt.AlignCenter)
        self.weight_result_text = QLabel(" -- ")
        self.weight_result_text.setAlignment(Qt.AlignCenter)
        weight_box.addWidget(weight_result_label, 3)
        weight_box.addWidget(self.weight_result_text, 2)

        training_times_result_label = QLabel("Training times : ")
        training_times_result_label.setAlignment(Qt.AlignCenter)
        self.training_times_result_text = QLabel(" -- ")
        self.training_times_result_text.setAlignment(Qt.AlignCenter)
        tranining_times_result_box.addWidget(training_times_result_label, 3)
        tranining_times_result_box.addWidget(self.training_times_result_text, 2)

        training_recognition_label = QLabel("Recognition rate of training (%) : ")
        training_recognition_label.setAlignment(Qt.AlignCenter)
        self.training_recognition_text = QLabel(" -- ")
        self.training_recognition_text.setAlignment(Qt.AlignCenter)
        training_recognition_box.addWidget(training_recognition_label, 3)
        training_recognition_box.addWidget(self.training_recognition_text, 2)

        training_rmse_label = QLabel("RMSE of training : ")
        training_rmse_label.setAlignment(Qt.AlignCenter)
        self.training_rmse_text = QLabel(" -- ")
        self.training_rmse_text.setAlignment(Qt.AlignCenter)
        training_rmse_box.addWidget(training_rmse_label, 3)
        training_rmse_box.addWidget(self.training_rmse_text, 2)

        testing_recongition_label = QLabel("Recognition rate of testing (%) : ")
        testing_recongition_label.setAlignment(Qt.AlignCenter)
        self.testing_recognition_text = QLabel(" -- ")
        self.testing_recognition_text.setAlignment(Qt.AlignCenter)
        testing_recognition_box.addWidget(testing_recongition_label, 3)
        testing_recognition_box.addWidget(self.testing_recognition_text, 2)

        testing_rmse_label = QLabel("RMSE of testing : ")
        testing_rmse_label.setAlignment(Qt.AlignCenter)
        self.testing_rmse_text = QLabel(" -- ")
        self.testing_rmse_text.setAlignment(Qt.AlignCenter)
        testing_rmse_box.addWidget(testing_rmse_label, 3)
        testing_rmse_box.addWidget(self.testing_rmse_text, 2)

        vbox.addLayout(weight_box)
        vbox.addLayout(tranining_times_result_box)
        vbox.addLayout(training_recognition_box)
        vbox.addLayout(training_rmse_box)
        vbox.addLayout(testing_recognition_box)
        vbox.addLayout(testing_rmse_box)
        self.__result_box.setLayout(vbox)

    # ------------------------------------------------------------------
    @pyqtSlot()
    def load_file(self):
        self.file_name = str(self.file_cb.currentText())
        self.classifier.load_file_info(self.file_name)
        if self.classifier.dim == 2:
            self.draw_points("all")
        else:
            self.ax.clear()
            self.ax.set_title("Over 2 dimension")
            self.canvas.draw()
            QMessageBox.about(self, "Warning", "Exist more than 2 dimension, drawing failed.")
    
        self.update_file_info()
        # self.update_initial_weight()
        # self.draw_points("all")

        # setting panel
        self.learning_rate_text.setText("0.8")
        self.propotion_of_test_text.setText("33")
        self.training_times_cbox.setChecked(True)
        self.training_times_text.setText("1000")
        self.recognition_cbox.setChecked(False)
        self.pocket_cbox.setChecked(False)
        self.momentum_cbox.setChecked(False)
        self.confirm_btn.setText("Confirm")
        # control & result panel
        self.confirm_btn.setEnabled(True)
        self.load_training_data_btn.setEnabled(False)
        self.load_testing_data_btn.setEnabled(False)
        self.start_training_btn.setEnabled(False)
        self.start_testing_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        self.reset_result_text()
    
    @pyqtSlot()
    def confirm_data(self):
        # check status
        if self.learning_rate_text.text() == "":
            self.learning_rate_text.setText("0.8")

        if self.propotion_of_test_text.text() == "":
            self.propotion_of_test_text.setText("33")

        if (not self.training_times_cbox.isChecked()) and (not self.recognition_cbox.isChecked()):
            QMessageBox.about(self, "Warning", "No converge condition was set, set recognition boundary as 0.8")
            self.recognition_cbox.setChecked(True)
            self.recognition_text.setText("0.8")
            self.recognition_boundary = 0.8
            self.training_times = 500000
        else:
            if self.training_times_cbox.isChecked():
                if self.training_times_text.text() == "":
                    self.training_times_text.setText("1000")
                self.training_times = int(self.training_times_text.text())
            else:
                self.training_times = 500000

            if self.recognition_cbox.isChecked():
                if self.recognition_text.text() == "":
                    self.recognition_text.setText("0.9")
                self.recognition_boundary = float(self.recognition_text.text())
            else:
                self.recognition_boundary = 0

        # assign value
        self.learning_rate = float(self.learning_rate_text.text())
        self.pro_of_test = float(int(self.propotion_of_test_text.text())/100)
        self.isPocket = True if (self.pocket_cbox.isChecked()) else False
        self.isMomentum = True if (self.momentum_cbox.isChecked()) else False

        self.update_classifier()
        # set GUI
        self.load_training_data_btn.setEnabled(True)
        self.confirm_btn.setEnabled(False)
        self.reset_result_text()
    
    @pyqtSlot()
    def cbox_change(self):
        self.training_times_text.setEnabled(self.training_times_cbox.isChecked())
        self.recognition_text.setEnabled(self.recognition_cbox.isChecked())

    @pyqtSlot()
    def load_training(self):
        if self.classifier.dim == 2:
            self.draw_points("training")
        # set GUI
        self.load_training_data_btn.setEnabled(False)
        self.start_training_btn.setEnabled(True)
    
    @pyqtSlot()
    def load_testing(self):
        if self.classifier.dim == 2:
            self.draw_points("testing")
        # set GUI
        self.load_testing_data_btn.setEnabled(False)
        self.start_training_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        self.load_testing_data_btn.setEnabled(True)
        self.start_testing_btn.setEnabled(True)

    @pyqtSlot()
    def start_training(self):
        self.classifier.do_training()
        # print(self.classifier.history_weight)
        # print(len(self.classifier.history_weight))
        # print(len(self.classifier.history_weight[0]) - 2)
        if self.classifier.dim == 2:
            self.drawer = Drawer(self.canvas, self.ax, self.training_x, self.training_y, self.classifier.history_weight, self.color)
            self.drawer.start()     
            self.drawer.finish.connect(self.update_training_result)
            # print(self.classifier.output_pers[0].weight)
        else:
            self.update_training_result()
        # self.update_training_result()
        # self.weight_result, self.training_times_result, proc_weight = support.do_training(self.feature_train, self.label_train, self.individual_label, self.weight, self.learning_rate, self.training_times)
        # self.drawer = Drawer(self.canvas, self.ax, self.feature, proc_weight)
        # self.drawer.start()  
        # self.drawer.finish.connect(self.update_training_result)
        
        self.start_training_btn.setEnabled(False)

    @pyqtSlot()
    def update_training_result(self):
        self.classifier.get_recognition(0)
        # self.recog_train = support.get_recognition(self.feature_train, self.label_train, self.weight_result, self.individual_label)
        # weight_res = [ round(w,3) for w in self.weight]
        # set GUI
        # self.weight_result_text.setText(str(weight_res))
        self.training_recognition_text.setText(str(self.classifier.training_recog * 100))
        self.training_times_result_text.setText(str(self.classifier.run))
        self.load_testing_data_btn.setEnabled(True)
        self.redo_btn.setEnabled(True)

    @pyqtSlot()
    def start_testing(self):
        # plot line
        # min_x1, max_x1 = min(self.feature[0])-0.5, max(self.feature[0])+0.5
        # self.ax.plot([min_x1, max_x1], [support.find_x2(self.weight[0], self.weight[1], self.weight[2], min_x1), support.find_x2(self.weight[0], self.weight[1], self.weight[2], max_x1)], color='orange', linewidth=2)
        # self.canvas.draw()
        # # get recognition
        self.classifier.get_recognition(1)
        self.testing_recognition_text.setText(str(self.classifier.testing_recog * 100))
        # self.recog_test = support.get_recognition(self.feature_test, self.label_test, self.weight_result, self.individual_label)
        # # set GUI
        self.confirm_btn.setEnabled(True)
        self.confirm_btn.setText("Redo Again")
        self.load_training_data_btn.setEnabled(False)
        self.load_testing_data_btn.setEnabled(False)
        self.start_training_btn.setEnabled(False)
        self.start_testing_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
    
    @pyqtSlot()
    def redo(self):
        # set GUI
        self.confirm_btn.setEnabled(True)
        self.load_training_data_btn.setEnabled(False)
        self.start_training_btn.setEnabled(False)
        self.load_testing_data_btn.setEnabled(False)
        self.start_testing_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        self.training_recognition_text.setText(" -- ")
        self.training_rmse_text.setText(" -- ")
        self.training_times_result_text.setText(" -- ")
        self.weight_result_text.setText(" -- ")

    # ------------------------------------------------------------------
    def update_file_info(self):
        # set GUI
        self.name_text.setText(self.file_name)
        self.number_of_feature_text.setText(str(self.classifier.dim))
        self.number_of_class_text.setText(str(self.classifier.classes))
        self.number_of_instances_text.setText(str(self.classifier.num_of_data))

    def update_classifier(self):
        self.classifier.initialize(self.training_times, self.recognition_boundary, self.learning_rate, self.pro_of_test, self.isPocket, self.isMomentum)
        self.classifier.split_train_test_data()
        
    def draw_points(self, mode):
        self.ax.clear()
        if(mode == "all"):
            point_x, point_y = support.get_seperate_points(self.classifier.instances, self.classifier.labels)
            boundary_x, boundary_y = support.get_boudary_of_axis(point_x, point_y)
            self.ax.set_title("All data")
            self.ax.set_xlim(boundary_x[0] - 0.5, boundary_x[1] + 0.5)
            self.ax.set_ylim(boundary_y[0] - 0.5, boundary_y[1] + 0.5)
            index = 0
            for x, y in zip(point_x, point_y):
                self.ax.scatter(x, y, c = self.color[index] , s=8)
                index += 1
            self.canvas.draw()
        elif(mode == "training"):
            self.training_x, self.training_y = support.get_seperate_points(self.classifier.training_instances, self.classifier.training_labels)
            boundary_x, boundary_y = support.get_boudary_of_axis(self.training_x, self.training_y)
            self.ax.set_title("Training data")
            self.ax.set_xlim(boundary_x[0] - 0.5, boundary_x[1] + 0.5)
            self.ax.set_ylim(boundary_y[0] - 0.5, boundary_y[1] + 0.5)
            index = 0
            for x, y in zip(self.training_x, self.training_y):
                self.ax.scatter(x, y, c = self.color[index] , s=8)
                index += 1
            self.canvas.draw()
        elif(mode == "testing"):
            point_x, point_y = support.get_seperate_points(self.classifier.testing_instances, self.classifier.testing_labels)
            boundary_x, boundary_y = support.get_boudary_of_axis(point_x, point_y)
            self.ax.set_title("Testing data")
            self.ax.set_xlim(boundary_x[0] - 0.5, boundary_x[1] + 0.5)
            self.ax.set_ylim(boundary_y[0] - 0.5, boundary_y[1] + 0.5)
            index = 0
            for x, y in zip(point_x, point_y):
                self.ax.scatter(x, y, c = self.color[index] , s=8)
                index += 1
            self.canvas.draw()

    def reset_result_text(self):
        self.weight_result_text.setText(" -- ")
        self.training_times_result_text.setText(" -- ")
        self.training_recognition_text.setText(" -- ")
        self.training_rmse_text.setText(" -- ")
        self.testing_recognition_text.setText(" -- ")
        self.testing_rmse_text.setText(" -- ")

        

# Class as Base window, create MultiPerceptronView
class BaseWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(750, 500)
        self.move(350,50)
        self.setWindowTitle('Neural Networks Lab2 - Multilayer Perceptron')
        self.statusBar()
        self.setCentralWidget(MultiPerceptronView())

# Class for drawing 
class Drawer(QThread):
    finish = pyqtSignal()

    def __init__(self, canvas, ax, x, y, history_weight, color):
        super().__init__()
        self.canvas = canvas
        self.ax = ax
        self.x, self.y = x, y 
        self.history_weight = history_weight
        self.color = color
        self.boundary_x, self.boundary_y = 0, 0
        self.num_of_lines = len(self.history_weight[0]) - 2

    def update_coordinate(self, old_x, old_y, weights):
        new_x, new_y = deepcopy(old_x), deepcopy(old_y)
        for xs, ys in zip(new_x, new_y):
            for i in range(len(xs)):
                real_input = np.array([-1, xs[i], ys[i]])
                vx, vy = np.sum(weights[0] * real_input), np.sum(weights[1] * real_input)
                xs[i] = 1 / (1 + math.exp(-1 * vx))
                ys[i] = 1 / (1 + math.exp(-1 * vy))
        return new_x, new_y

    def find_y(self, weights, x):
        weight = weights[0]
        return (np.asscalar(weight[0]) - np.asscalar(weight[1]) * x) / np.asscalar(weight[2])

    def run(self):
        for weights in self.history_weight:
            self.ax.clear()
            # update new x, y based on weight
            updated_x, updated_y = self.update_coordinate(self.x, self.y, weights)
            boundary_x, boundary_y = support.get_boudary_of_axis(updated_x, updated_y)

            x_spacing = (boundary_x[1] - boundary_x[0]) / 10
            y_spacing = (boundary_y[1] - boundary_y[0]) / 10
            self.ax.set_xlim(boundary_x[0] - x_spacing, boundary_x[1] + x_spacing)
            self.ax.set_ylim(boundary_y[0] - y_spacing, boundary_y[1] + y_spacing)
            
            # plot 
            index = 0
            for x, y in zip(updated_x, updated_y):
                self.ax.scatter(x, y, c = self.color[index] , s=8)
                index += 1

            for line in range(self.num_of_lines):
                self.ax.plot([boundary_x[0] - x_spacing, boundary_x[1] + x_spacing], [self.find_y(weights[-1 * line], boundary_x[0] - x_spacing), self.find_y(weights[-1 * line], boundary_x[1] + x_spacing)])
            plt.pause(0.1)
            self.canvas.draw()
        
        # just draw for the last time
        # self.ax.clear()
        # self.updated_x, self.updated_y = self.update_coordinate(self.updated_x, self.updated_y, self.history_weight[-1])
        # boundary_x, boundary_y = support.get_boudary_of_axis(self.updated_x, self.updated_y)

        # x_spacing, y_spacing = (boundary_x[1] - boundary_x[0]) / 10, (boundary_y[1] - boundary_y[0]) / 10
        # self.ax.set_xlim(boundary_x[0] - x_spacing, boundary_x[1] + x_spacing)
        # self.ax.set_ylim(boundary_y[0] - y_spacing, boundary_y[1] + y_spacing)

        # index = 0
        # for x, y in zip(self.updated_x, self.updated_y):
        #     self.ax.scatter(x, y, c = self.color[index] , s=8)
        #     index += 1
        
        # for line in range(self.num_of_lines):
        #     line += 1
        #     self.ax.plot([boundary_x[0] - x_spacing, boundary_x[1] + x_spacing], [self.find_y(self.history_weight[-1][-1 * line], boundary_x[0] - x_spacing), self.find_y(self.history_weight[-1][-1 * line], boundary_x[1] + x_spacing)])

        # self.canvas.draw()
        # trash
        # plt.ion()
        # while len(self.proc) != 0:
        #     weight = self.proc[0]
        #     try:
        #         self.ax.lines.pop(0)
        #     except Exception:
        #         pass
        #     # lines = self.ax.plot([self.min_x1, self.max_x1], [support.find_x2(weight[0], weight[1], weight[2], self.min_x1), support.find_x2(weight[0], weight[1], weight[2], self.max_x1)], color='orange', linewidth=2)
        #     self.canvas.draw()
        #     plt.pause(0.1)
        #     del self.proc[0]
        # plt.ioff()
        # plt.show()
        self.finish.emit()


if __name__ == '__main__':
    sys.argv.append("--style=fusion")
    app = QApplication(sys.argv)
    w = BaseWindow()
    w.show()   
    sys.exit(app.exec_())