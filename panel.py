import support
import sys
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QGroupBox, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QComboBox)
from matplotlib.backends.backend_qt5agg import FigureCanvas 
from matplotlib.figure import Figure


class Preceptron(QWidget):
    def __init__(self):
        super().__init__()
        self.__initUI()

    def __initUI(self):
        self.__window_layout = QHBoxLayout()
        self.__status_layout = QVBoxLayout()
        self.__result_layout = QVBoxLayout()

        self.__set_file_box_UI()
        self.__set_dataInfo_box_UI()
        self.__set_setting_box_UI()
        self.__set_figure_box_UI()
        self.__set_result_box_UI()

        self.__status_layout.addWidget(self.__file_box, 1)
        self.__status_layout.addWidget(self.__dataInfo_box, 2)
        self.__status_layout.addWidget(self.__setting_box, 2)
        self.__result_layout.addWidget(self.__figure_box, 3)
        self.__result_layout.addWidget(self.__result_box, 1)
        self.__window_layout.addLayout(self.__status_layout, 1)
        self.__window_layout.addLayout(self.__result_layout, 1)

        self.__window_layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(self.__window_layout)

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
        self.name_text = QLabel("NULL")
        self.name_text.setAlignment(Qt.AlignCenter)
        name_box.addWidget(name_label, 1)
        name_box.addWidget(self.name_text, 2)

        number_of_instances_label = QLabel("Number of Instances :")
        number_of_instances_label.setAlignment(Qt.AlignCenter)
        self.number_of_instances_text = QLabel("NULL")
        self.number_of_instances_text.setAlignment(Qt.AlignCenter)
        number_of_instances_box.addWidget(number_of_instances_label, 1)
        number_of_instances_box.addWidget(self.number_of_instances_text, 2)
        
        number_of_feature_label = QLabel("Number of Features :")
        number_of_feature_label.setAlignment(Qt.AlignCenter)
        self.number_of_feature_text = QLabel("NULL")
        self.number_of_feature_text.setAlignment(Qt.AlignCenter)
        number_of_feature_box.addWidget(number_of_feature_label, 1)
        number_of_feature_box.addWidget(self.number_of_feature_text, 2)
        
        number_of_class_label = QLabel("Number of Classes :")
        number_of_class_label.setAlignment(Qt.AlignCenter)
        self.number_of_class_text = QLabel("NULL")
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
        vbox = QVBoxLayout()
        weight_box = QHBoxLayout()
        initalize_box = QHBoxLayout()
        confirm_box = QHBoxLayout()

        weight_label = QLabel("Initalize the weight with Value :")
        weight_label.setAlignment(Qt.AlignCenter)
        self.initial_weight = QLabel("--")
        self.initial_weight.setAlignment(Qt.AlignCenter)
        self.reset_weight_btn = QPushButton("Reset weight")
        self.reset_weight_btn.setEnabled(False)
        self.reset_weight_btn.setStatusTip('Reset initial weight with value from -1 to 1')
        self.reset_weight_btn.clicked.connect(self.update_initial_weight)
        weight_box.addWidget(weight_label,3)
        weight_box.addWidget(self.initial_weight,3)
        weight_box.addWidget(self.reset_weight_btn,1)

        learning_rate_label = QLabel("Learning Rate :")
        learning_rate_label.setAlignment(Qt.AlignCenter)
        self.learning_rate_text = QLineEdit()
        self.learning_rate_text.setStatusTip('The learning rate of training')
        initalize_box.addWidget(learning_rate_label, 2)
        initalize_box.addWidget(self.learning_rate_text, 2)
        initalize_box.addStretch(1)

        training_times_label = QLabel("Training times :")
        training_times_label.setAlignment(Qt.AlignCenter)
        self.training_times_text = QLineEdit()
        self.training_times_text.setStatusTip('Using training times as converge condition')
        initalize_box.addWidget(training_times_label, 2)
        initalize_box.addWidget(self.training_times_text, 2)
        
        self.confirm_btn = QPushButton("Confirm")
        self.confirm_btn.setStatusTip('Confirm to use : Initial weight | Learning rate | Training times')
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self.confirm_data)
        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.setStatusTip('Split dataset into 3 part, training 2 of them.')
        self.start_training_btn.setEnabled(False)
        self.start_testing_btn = QPushButton("Start Testing")
        self.start_testing_btn.setStatusTip('After training, testing remaining dataset.')
        self.start_testing_btn.setEnabled(False)
        confirm_box.addWidget(self.confirm_btn, 2)
        confirm_box.addWidget(self.start_training_btn, 2)
        confirm_box.addWidget(self.start_testing_btn, 2)

        vbox.addLayout(weight_box)
        vbox.addLayout(initalize_box)
        vbox.addLayout(confirm_box)
        self.__setting_box.setLayout(vbox)

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
        hbox = QHBoxLayout()

        file_text = QLabel("Figure")

        hbox.addWidget(file_text)
        self.__figure_box.setLayout(hbox)

    @pyqtSlot()
    def load_file(self):
        self.file_name = str(self.file_cb.currentText())
        self.feature, self.label = support.load_file_info(self.file_name)
        self.individual_label = support.get_individual_label(self.label)
        self.reset_weight_btn.setEnabled(True)
        self.confirm_btn.setEnabled(True)
        self.learning_rate_text.clear()
        self.training_times_text.clear()
        self.update_file_info()
        self.update_initial_weight()
        self.draw_points()

    @pyqtSlot()
    def update_initial_weight(self):
        self.weight = [-1]
        weight_text = " -1 "
        for i in range(self.dimension):
            rand_num = float(support.get_random_weight())
            self.weight.append(rand_num)
            weight_text = weight_text + " " + str(rand_num) + " "
        self.initial_weight.setText(weight_text)
    
    @pyqtSlot()
    def confirm_data(self):
        self.learning_rate = float(self.learning_rate_text.text())
        self.training_times = int(self.training_times_text.text())
        print(self.learning_rate, self.training_times)
        self.start_training_btn.setEnabled(True)

    def update_file_info(self):
        self.dimension = len(self.feature)
        self.name_text.setText(self.file_name)
        self.number_of_feature_text.setText(str(self.dimension))
        self.number_of_class_text.setText(str(len(self.individual_label)))
        self.number_of_instances_text.setText(str(len(self.label)))

    def draw_points(self):
        self.ax.clear()
        label1_x1, label1_x2, label2_x1, label2_x2 = support.get_seperate_points(self.feature, self.label, self.individual_label)
        self.ax.set_xlim(min(self.feature[0])-0.5, max(self.feature[0])+0.5)
        self.ax.set_ylim(min(self.feature[1])-0.5, max(self.feature[1])+0.5)
        self.ax.scatter(label1_x1, label1_x2, c='blue' , s=25)
        self.ax.scatter(label2_x1, label2_x2, c='green' , s=25)
        self.canvas.draw()


class BaseWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(750, 500)
        self.move(350,50)
        self.setWindowTitle('Nerual Network Lab1 - Preceptron')
        self.statusBar()
        self.setCentralWidget(Preceptron())

if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = BaseWindow()
    w.show()   
    sys.exit(app.exec_())