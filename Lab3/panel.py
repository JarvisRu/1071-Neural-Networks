import support
import hopfield
import numpy as np
import sys
import math
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QGroupBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QMessageBox, QSpinBox, QComboBox)
from copy import deepcopy

class HopfieldView(QWidget):
    def __init__(self):
        super().__init__()
        self.hopfield = hopfield.Hopfield()
        self.__initUI()

    def __initUI(self):
        self.__window_layout = QVBoxLayout()
        self.__file_layout = QVBoxLayout()
        self.__figure_layout = QHBoxLayout()

        self.__set_file_box_UI()
        self.__set_dataInfo_box_UI()

        self.__set_testing_box_UI()
        self.__set_result_box_UI()

        self.__file_layout.addWidget(self.__file_box, 1)
        self.__file_layout.addWidget(self.__dataInfo_box, 2)

        self.__figure_layout.addWidget(self.__testing_box, 1)
        self.__figure_layout.addWidget(self.__result_box, 1)

        self.__window_layout.addLayout(self.__file_layout, 1)
        self.__window_layout.addLayout(self.__figure_layout, 3)
        self.__window_layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(self.__window_layout)

    def __set_file_box_UI(self):
        self.__file_box = QGroupBox('File')
        hbox = QHBoxLayout()

        training_file_label = QLabel("Select Training File :")
        training_file_label.setAlignment(Qt.AlignCenter)
        self.training_file_cb = QComboBox()
        self.training_file_cb.addItems(support.find_training_dataset())
        self.training_file_cb.setStatusTip("Please select a file as training dataset")
        testing_file_label = QLabel("Select Testing File :")
        testing_file_label.setAlignment(Qt.AlignCenter)
        self.testing_file_cb = QComboBox()
        self.testing_file_cb.addItems(support.find_testing_dataset())
        self.testing_file_cb.setStatusTip("Please select a file as testing dataset")

        self.file_load_btn = QPushButton('Load File', self)
        self.file_load_btn.setStatusTip("Load file and update file information")
        self.file_load_btn.clicked.connect(self.load_both_file)

        hbox.addWidget(training_file_label, 1)
        hbox.addWidget(self.training_file_cb, 3)
        hbox.addWidget(testing_file_label, 1)
        hbox.addWidget(self.testing_file_cb, 3)
        hbox.addWidget(self.file_load_btn, 1)
        self.__file_box.setLayout(hbox)

    def __set_dataInfo_box_UI(self):
        self.__dataInfo_box = QGroupBox('Data Information')
        vbox = QVBoxLayout()
        name_box = QHBoxLayout()
        number_box = QHBoxLayout()
        dim_box = QHBoxLayout()

        training_name_label = QLabel("Training DataSet Name :")
        training_name_label.setAlignment(Qt.AlignCenter)
        self.training_name_text = QLabel(" -- ")
        self.training_name_text.setAlignment(Qt.AlignCenter)
        testing_name_label = QLabel("Testing DataSet Name :")
        testing_name_label.setAlignment(Qt.AlignCenter)
        self.testing_name_text = QLabel(" -- ")
        self.testing_name_text.setAlignment(Qt.AlignCenter)
        name_box.addWidget(training_name_label, 1)
        name_box.addWidget(self.training_name_text, 2)
        name_box.addWidget(testing_name_label, 1)
        name_box.addWidget(self.testing_name_text, 2)

        number_of_training_label = QLabel("Number of Training :")
        number_of_training_label.setAlignment(Qt.AlignCenter)
        self.number_of_training_text = QLabel(" -- ")
        self.number_of_training_text.setAlignment(Qt.AlignCenter)
        number_of_testing_label = QLabel("Number of Testing :")
        number_of_testing_label.setAlignment(Qt.AlignCenter)
        self.number_of_testing_text = QLabel(" -- ")
        self.number_of_testing_text.setAlignment(Qt.AlignCenter)
        number_box.addWidget(number_of_training_label, 1)
        number_box.addWidget(self.number_of_training_text, 2)
        number_box.addWidget(number_of_testing_label, 1)
        number_box.addWidget(self.number_of_testing_text, 2)
        
        real_dim_label = QLabel("Number of dimensions :")
        real_dim_label.setAlignment(Qt.AlignCenter)
        self.real_dim_text = QLabel(" -- ")
        self.real_dim_text.setAlignment(Qt.AlignCenter)
        self.real_dim_text.setStatusTip("rows * cols")
        dim_label = QLabel("Treat dimensions as :")
        dim_label.setAlignment(Qt.AlignCenter)
        self.dim_text = QLabel(" -- ")
        self.dim_text.setAlignment(Qt.AlignCenter)
        self.dim_text.setStatusTip("rows * cols")
        dim_box.addWidget(real_dim_label, 1)
        dim_box.addWidget(self.real_dim_text, 2)
        dim_box.addWidget(dim_label, 1)
        dim_box.addWidget(self.dim_text, 2)

        vbox.addLayout(name_box)
        vbox.addLayout(number_box)
        vbox.addLayout(dim_box)
        self.__dataInfo_box.setLayout(vbox)
        
    def __set_testing_box_UI(self):
        self.__testing_box = QGroupBox('Testing data')
        vbox = QVBoxLayout()
        self.__testing_box.setLayout(vbox)
        
    def __set_result_box_UI(self):
        self.__result_box = QGroupBox("Association Result")
        vbox = QVBoxLayout()
        self.__result_box.setLayout(vbox)

    @pyqtSlot()
    def load_both_file(self):
        self.training_file_name = str(self.training_file_cb.currentText())
        self.testing_file_name = str(self.testing_file_cb.currentText())

        # check file
        if support.check_file(self.training_file_name, self.testing_file_name):
            QMessageBox.about(self, "Warning", "Different source of training and testing data, please check again.")
            self.number_of_training_text.setText(" -- ")
            self.number_of_testing_text.setText(" -- ")
            self.real_dim_text.setText(" -- ")
            self.dim_text.setText(" -- ")
        else:
            self.hopfield.load_file(self.training_file_name, self.testing_file_name)
            # update file_info
            self.training_name_text.setText(self.training_file_name)
            self.testing_name_text.setText(self.testing_file_name)
            self.number_of_training_text.setText(str(self.hopfield.num_of_training))
            self.number_of_testing_text.setText(str(self.hopfield.num_of_testing))
            self.real_dim_text.setText(str(int(self.hopfield.rows)) + " * " + str(int(self.hopfield.cols)))
            self.dim_text.setText(str(self.hopfield.dim) + " * 1")



# Class as Base window, create Hopfield
class BaseWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(750, 500)
        self.move(350,50)
        self.setWindowTitle('Neural Networks Lab3 - Hopfield')
        self.statusBar()
        self.setCentralWidget(HopfieldView())


if __name__ == '__main__':
    sys.argv.append("--style=fusion")
    app = QApplication(sys.argv)
    w = BaseWindow()
    w.show()   
    sys.exit(app.exec_())