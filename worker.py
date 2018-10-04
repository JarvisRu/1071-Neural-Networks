from PyQt5.QtCore import Qt

class Worker(QThread):
    def __init__(self, parent=None):
        super().__init__()

    def get_data(feature, label, weight, learning_rate, training_times):
        
        

    def run(self):