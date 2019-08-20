from PyQt5.QtWidgets import QDialog
from gui.UIOpenSourceDialog import UIOpenSourceDialog
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import re


class OpenSourceForm(QDialog, UIOpenSourceDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.file_names = []
        self.dissimilarity_name = ""
        self.dissimilarity_index = 0
        self.strategy_name = ""
        self.strategy_index = 0
        self.radiobutton_option = "matfiles"
        self.last_event = "cancel"
        self.normalization_check = False

    def set_previousopensource_option(self, file_names_in, dissimilarity_name_in, dissimilarity_index_in,
                                      radiobutton_option_in, strategy_name_in, strategy_index_in, 
                                      normalization_check_in):
        self.file_names = file_names_in
        self.dissimilarity_name = dissimilarity_name_in
        self.dissimilarity_index = dissimilarity_index_in
        self.strategy_name = strategy_name_in
        self.strategy_index = strategy_index_in
        self.normalization_check = normalization_check_in
        if radiobutton_option_in != "":
            self.radiobutton_option = radiobutton_option_in
        else:
            self.radiobutton_option = "matfiles"
        self.load_previousopensource_option(self.radiobutton_option)

    def load_previousopensource_option(self, radiobutton_option):
        self.normalizationCheckButton.setCheckState(self.normalization_check)
        if radiobutton_option == "matfiles":
            self.matFilesRadioButton.setChecked(True)
            self.pointsFileRadioButton.setChecked(False)
            self.distanceFileRadioButton.setChecked(False)
            self.matFilesLineEdit.setEnabled(True)
            if len(self.file_names) > 0:
                self.matFilesLineEdit.setText(str(self.file_names))
            self.matFilesButton.setEnabled(True)
            self.pointsFileLineEdit.setEnabled(False)
            self.pointsFileLineEdit.setText("")
            self.pointsFileButton.setEnabled(False)
            self.distanceFileLineEdit.setEnabled(False)
            self.distanceFileLineEdit.setText("")
            self.distanceFileButton.setEnabled(False)
            self.dissimilarityBox.setEnabled(True)
            self.dissimilarityBox.setCurrentIndex(self.dissimilarity_index)
            self.strategyBox.setEnabled(True)
            self.strategyBox.setCurrentIndex(self.strategy_index)
            self.radiobutton_option = "matfiles"

        elif radiobutton_option == "pointsfile":
            self.matFilesRadioButton.setChecked(False)
            self.pointsFileRadioButton.setChecked(True)
            self.distanceFileRadioButton.setChecked(False)
            self.pointsFileLineEdit.setEnabled(True)
            if len(self.file_names) > 0:
                self.pointsFileLineEdit.setText(str(self.file_names))
            self.pointsFileButton.setEnabled(True)
            self.matFilesLineEdit.setEnabled(False)
            self.matFilesLineEdit.setText("")
            self.matFilesButton.setEnabled(False)
            self.distanceFileLineEdit.setEnabled(False)
            self.distanceFileLineEdit.setText("")
            self.distanceFileButton.setEnabled(False)
            self.dissimilarityBox.setEnabled(True)
            self.dissimilarityBox.setCurrentIndex(self.dissimilarity_index)
            self.strategyBox.setEnabled(False)
            self.strategyBox.setCurrentIndex(1)
            self.strategy_index = 1
            self.strategy_name = "Point by Point"
            self.radiobutton_option = "pointsfile"

        elif radiobutton_option == "dmatfile":
            self.matFilesRadioButton.setChecked(False)
            self.pointsFileRadioButton.setChecked(False)
            self.distanceFileRadioButton.setChecked(True)
            self.distanceFileLineEdit.setEnabled(True)
            if len(self.file_names) > 0:
                self.distanceFileLineEdit.setText(str(self.file_names))
            self.distanceFileButton.setEnabled(True)
            self.matFilesLineEdit.setEnabled(False)
            self.matFilesLineEdit.setText("")
            self.matFilesButton.setEnabled(False)
            self.pointsFileLineEdit.setEnabled(False)
            self.pointsFileLineEdit.setText("")
            self.pointsFileButton.setEnabled(False)
            self.dissimilarityBox.setCurrentIndex(0)
            self.dissimilarityBox.setEnabled(False)
            self.strategyBox.setEnabled(False)
            self.strategyBox.setCurrentIndex(1)
            self.dissimilarity_index = 0
            self.dissimilarity_name = ""
            self.strategy_index = 1
            self.strategy_name = "Point by Point"
            self.radiobutton_option = "dmatfile"

    def change_opensource_option(self, radio_option):
        if radio_option.text() == "Mat Files":
            if radio_option.isChecked():
                # print(radio_option.text() + " is selected")
                self.matFilesLineEdit.setEnabled(True)
                self.matFilesButton.setEnabled(True)
                self.pointsFileLineEdit.setEnabled(False)
                self.pointsFileLineEdit.setText("")
                self.pointsFileButton.setEnabled(False)
                self.distanceFileLineEdit.setEnabled(False)
                self.distanceFileLineEdit.setText("")
                self.distanceFileButton.setEnabled(False)
                self.dissimilarityBox.setEnabled(True)
                self.strategyBox.setEnabled(True)
                self.radiobutton_option = "matfiles"

        elif radio_option.text() == "Points File":
            if radio_option.isChecked():
                # print(radio_option.text() + " is selected")
                self.pointsFileLineEdit.setEnabled(True)
                self.pointsFileButton.setEnabled(True)
                self.matFilesLineEdit.setEnabled(False)
                self.matFilesLineEdit.setText("")
                self.matFilesButton.setEnabled(False)
                self.distanceFileLineEdit.setEnabled(False)
                self.distanceFileLineEdit.setText("")
                self.distanceFileButton.setEnabled(False)
                self.dissimilarityBox.setEnabled(True)
                self.strategyBox.setEnabled(False)
                self.strategyBox.setCurrentIndex(1)
                self.strategy_index = 1
                self.strategy_name = "Point by Point"
                self.radiobutton_option = "pointsfile"
        elif radio_option.text() == "Distance File":
            if radio_option.isChecked():
                # print(radio_option.text() + " is selected")
                self.distanceFileLineEdit.setEnabled(True)
                self.distanceFileButton.setEnabled(True)
                self.matFilesLineEdit.setEnabled(False)
                self.matFilesLineEdit.setText("")
                self.matFilesButton.setEnabled(False)
                self.pointsFileLineEdit.setEnabled(False)
                self.pointsFileLineEdit.setText("")
                self.pointsFileButton.setEnabled(False)
                self.dissimilarityBox.setCurrentIndex(0)
                self.strategyBox.setEnabled(False)
                self.strategyBox.setCurrentIndex(1)
                self.strategy_index = 1
                self.strategy_name = "Point by Point"
                self.dissimilarity_index = 0
                self.dissimilarity_name = ""
                self.dissimilarityBox.setEnabled(False)
                self.radiobutton_option = "dmatfile"

    def change_dissimilarityBox(self, value):
        if value > 0:
            self.dissimilarity_name = self.dissimilarityBox.currentText()
            self.dissimilarity_index = value
        else:
            self.dissimilarity_name = ""
            self.dissimilarity_index = 0
        
    def change_strategyBox(self, value):
        if value > 0:
            self.strategy_name = self.strategyBox.currentText()
            self.strategy_index = value
        else:
            self.strategy_name = ""
            self.strategy_index = 0    
            
    def check_normalization(self, state):
        self.normalization_check = state
        
    def open_matrices(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "Matlab Matrices (*.mat)", options=options)
        if files:
            files.sort(key=OpenSourceForm.natural_keys)
            self.file_names = []
            self.file_names = files
            self.matFilesLineEdit.setText(str(files))

    def open_data_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Multidimensional Point Files (*.data)",
                                              options=options)
        if file:
            self.file_names = []
            self.file_names.append(file)
            self.pointsFileLineEdit.setText(file)

    def open_dmatrix_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Dissimilarity Matrix File (*.dmat)",
                                              options=options)
        if file:
            self.file_names = []
            self.file_names.append(file)
            self.distanceFileLineEdit.setText(file)

    def accept_button(self):
        if (len(self.file_names) == 0) or (self.radiobutton_option != "dmatfile" and self.dissimilarity_name == "") or (self.radiobutton_option == "matfiles" and (self.strategy_index == "" or self.strategy_index==0)):
            #msg_box = QMessageBox()
            #msg_box.setIcon(QMessageBox.Warning)
            #msg_box.setText("Files or dissimilarity function/strategy not defined.")
            #msg_box.setWindowTitle("Message")
            #msg_box.show()
            #msg_box.exec()
            msg_box = QMessageBox.warning(self, 'Message', "Files or dissimilarity function/strategy not defined.", QMessageBox.Ok, QMessageBox.Ok)
            self.last_event = "cancel"
        else:
            self.last_event = "accept"
            self.close()

    def cancel_button(self):
        self.last_event = "cancel"
        print(self.last_event)
        self.close()

    @staticmethod
    def atoi(text):
        return int(text) if text.isdigit() else text

    @staticmethod
    def natural_keys(text):
        return [OpenSourceForm.atoi(c) for c in re.split('(\d+)', text)]