import support
import hopfield
import numpy as np
import sys
import math
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QGroupBox, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QPushButton, QMessageBox, QSpinBox, QComboBox, QSlider)
from copy import deepcopy

class HopfieldView(QWidget):
    def __init__(self):
        super().__init__()
        self.hopfield = hopfield.Hopfield()
        self.training_file_name, self.testing_file_name = "", "" 
        self.__initUI()

    def __initUI(self):
        self.__window_layout = QVBoxLayout()
        self.__file_layout = QVBoxLayout()
        self.__figure_layout = QHBoxLayout()

        self.__set_file_box_UI()
        self.__set_dataInfo_box_UI()

        self.__set_training_box_UI()
        self.__set_result_box_UI()

        self.__file_layout.addWidget(self.__file_box, 1)
        self.__file_layout.addWidget(self.__dataInfo_box, 2)

        self.__figure_layout.addWidget(self.__training_box, 1)
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

        self.file_load_btn = QPushButton('Load File + Start Association', self)
        self.file_load_btn.setStatusTip("After loading file and updating file information, start association.")
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
        
    def __set_training_box_UI(self):
        self.__training_box = QGroupBox('Training data')
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        self.grid_training_box = QGridLayout()

        view_training_label = QLabel("View Training Data :")
        view_training_label.setAlignment(Qt.AlignCenter)
        self.view_training_spin = QSpinBox()
        self.view_training_spin.setEnabled(False)
        self.view_training_spin.setValue(1)
        self.view_training_spin.setSingleStep(1)
        self.view_training_spin.valueChanged.connect(self.switch_training_view)
        hbox.addWidget(view_training_label)
        hbox.addWidget(self.view_training_spin)

        vbox.addLayout(hbox, 1)
        vbox.addLayout(self.grid_training_box, 4)
        vbox.addStretch(2)
        self.__training_box.setLayout(vbox)
        
    def __set_result_box_UI(self):
        self.__result_box = QGroupBox("Association Result")
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        slider_box = QHBoxLayout()
        recog_box = QHBoxLayout()
        self.grid_association_box = QGridLayout()

        overall_recog_label = QLabel("Overall Association Recognition :")
        overall_recog_label.setAlignment(Qt.AlignCenter)
        overall_recog_label.setStatusTip("Recognition of all testing dataset, it will be classify as correct if recognition is over 95%")
        self.overall_recog_text = QLabel(" -- ")
        self.overall_recog_text.setAlignment(Qt.AlignCenter)
        self.overall_recog_text.setStatusTip("Recognition of all testing dataset, it will be classify as correct if recognition is over 95%")
        view_association_label = QLabel("View Association Result :")
        view_association_label.setAlignment(Qt.AlignCenter)
        self.view_association_spin = QSpinBox()
        self.view_association_spin.setEnabled(False)
        self.view_association_spin.setValue(0)
        self.view_association_spin.setSingleStep(1)
        self.view_association_spin.valueChanged.connect(self.switch_association_view)
        hbox.addWidget(overall_recog_label)
        hbox.addWidget(self.overall_recog_text)
        hbox.addWidget(view_association_label)
        hbox.addWidget(self.view_association_spin)

        view_step_label = QLabel("View Association step by step :")
        view_step_label.setAlignment(Qt.AlignCenter)
        view_step_label.setStatusTip("View the process of recall step by step")
        self.step_slider = QSlider(Qt.Horizontal)
        self.step_slider.setStatusTip("View the process of recall step by step")
        self.step_slider.setSingleStep(1)
        self.step_slider.setRange(0,1)
        self.step_slider.valueChanged.connect(self.switch_association_step)
        slider_box.addWidget(view_step_label)
        slider_box.addWidget(self.step_slider)

        recog_label = QLabel("Recognition :")
        recog_label.setAlignment(Qt.AlignCenter)
        recog_label.setStatusTip("Recognition of single instance")
        self.recog_text = QLabel(" -- ")
        self.recog_text.setStatusTip("Recognition of single instance")
        self.recog_text.setAlignment(Qt.AlignCenter)
        recog_box.addWidget(recog_label)
        recog_box.addWidget(self.recog_text)

        vbox.addLayout(hbox, 1)
        vbox.addLayout(self.grid_association_box, 4)
        vbox.addLayout(slider_box, 1)
        vbox.addLayout(recog_box, 1)
        self.__result_box.setLayout(vbox)

    @pyqtSlot()
    def load_both_file(self):
        if self.training_file_name == str(self.training_file_cb.currentText()) and self.testing_file_name == str(self.testing_file_cb.currentText()):
            QMessageBox.about(self, "Warning", "Loading the same dataset, do nothing.")
        else:
            self.training_file_name = str(self.training_file_cb.currentText())
            self.testing_file_name = str(self.testing_file_cb.currentText())

            # check file
            if support.check_file(self.training_file_name, self.testing_file_name):
                QMessageBox.about(self, "Warning", "Different source of training and testing data, please check again.")
                self.number_of_training_text.setText(" -- ")
                self.number_of_testing_text.setText(" -- ")
                self.real_dim_text.setText(" -- ")
                self.dim_text.setText(" -- ")
                self.overall_recog_text.setText(" -- ")
                self.recog_text.setText(" -- ")
                self.view_training_spin.setEnabled(False)
                self.view_association_spin.setEnabled(False)
                self.step_slider.setEnabled(False)
                self.step_slider.setValue(0)
            else:
                self.hopfield.load_file(self.training_file_name, self.testing_file_name)
                # start association
                self.hopfield.start_association()
                # update file_info
                self.training_name_text.setText(self.training_file_name)
                self.testing_name_text.setText(self.testing_file_name)
                self.number_of_training_text.setText(str(self.hopfield.num_of_training))
                self.number_of_testing_text.setText(str(self.hopfield.num_of_testing))
                self.real_dim_text.setText(str(self.hopfield.rows) + " * " + str(self.hopfield.cols))
                self.dim_text.setText(str(self.hopfield.dim) + " * 1")
                # initialize figure
                self.__initialize_figure()
                # update figure part
                self.overall_recog_text.setText(str(self.hopfield.overallCorrectNum) + " / " + str(self.hopfield.num_of_training))
                self.view_training_spin.setValue(1)
                self.view_training_spin.setEnabled(True)
                self.view_training_spin.setRange(1, self.hopfield.num_of_training)
                self.view_association_spin.setValue(1)
                self.view_association_spin.setEnabled(True)
                self.view_association_spin.setRange(1, self.hopfield.num_of_testing)
                self.step_slider.setEnabled(True)
                self.step_slider.setValue(0)
                self.switch_training_view()
                self.switch_association_view()

    @pyqtSlot()
    def switch_training_view(self):
        target_view = self.view_training_spin.value()
        self.view_association_spin.setValue(target_view)
        self.__draw_training_view(target_view)

    @pyqtSlot()
    def switch_association_view(self):
        target_view = self.view_association_spin.value()
        if target_view != 0:
            self.view_training_spin.setValue(target_view)
            # draw
            self.__draw_association_view(target_view)
            # reset slider
            self.step_slider.setValue(0)
            self.step_slider.setRange(0, len(self.hopfield.record[target_view - 1])-1)
            # set recognition
            self.recog_text.setText(str(support.get_recog(self.hopfield.inputs[target_view - 1], self.hopfield.record[target_view - 1][-1])))

    def __initialize_figure(self):
        # delete old
        for i in reversed(range(self.grid_training_box.count())): 
            self.grid_training_box.itemAt(i).widget().deleteLater()
        for i in reversed(range(self.grid_association_box.count())): 
            self.grid_association_box.itemAt(i).widget().deleteLater()
            
        # print new
        self.training_rects, self.association_rects = [], []
        for i in range(self.hopfield.rows):
            for j in range(self.hopfield.cols):
                item = QPushButton()
                item.setStyleSheet("background-color: white;")
                item.setEnabled(False)
                self.grid_training_box.addWidget(item, i, j)
                self.training_rects.append(item)

                item2 = QPushButton()
                item2.setStyleSheet("background-color: white;")
                item2.setEnabled(False)
                self.grid_association_box.addWidget(item2, i, j)
                self.association_rects.append(item2)


    def __draw_training_view(self, target):
        target_img = self.hopfield.inputs[target - 1]
        # print all to white first
        for i in range(self.hopfield.dim):
                self.training_rects[i].setStyleSheet("background-color: white;")
        # print by value
        for i in range(self.hopfield.dim):
            if target_img[i] == 1:
                self.training_rects[i].setStyleSheet("background-color: black;")

    def __draw_association_view(self, target):
        target_img = self.hopfield.record[target - 1][0]
        # print all to white first
        for i in range(self.hopfield.dim):
                self.association_rects[i].setStyleSheet("background-color: white;")
        # print by value
        for i in range(self.hopfield.dim):
            if target_img[i] == 1:
                self.association_rects[i].setStyleSheet("background-color: black;")

    def switch_association_step(self):
        step = self.step_slider.value()
        target_img = self.hopfield.record[self.view_association_spin.value() - 1][step]
        # print all to white first
        for i in range(self.hopfield.dim):
                self.association_rects[i].setStyleSheet("background-color: white;")
        # print by value
        for i in range(self.hopfield.dim):
            if target_img[i] == 1:
                self.association_rects[i].setStyleSheet("background-color: black;")


# Class as Base window, create Hopfield
class BaseWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(800, 530)
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