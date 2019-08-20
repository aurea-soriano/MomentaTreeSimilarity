import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog, QGraphicsScene
from PyQt5 import QtCore, QtGui, QtWidgets
from gui.UIShell4DSimilarity import UIShell4DSimilarity_MainWindow
from form.OpenSourceForm import OpenSourceForm
from process.ProcessGenerator import ProcessGenerator
from normalization.MinMaxNormalization import MinMaxNormalization
import operator


class AppWindow(QMainWindow, UIShell4DSimilarity_MainWindow):
    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.file_names = []
        self.dissimilarity_name = ""
        self.dissimilarity_index = 0
        self.strategy_name = ""
        self.strategy_index = 0
        self.radiobutton_option = ""
        self.njdismatrix = []
        self.normalization_check = False
        self.scene = QGraphicsScene()

    def close_application(self):
        question = QMessageBox.question(self, 'close window', 'Confirm close?', QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.No)
        if question == QMessageBox.Yes:
            print('Yes clicked.')
            sys.exit(app.exec_())
        else:
            print('No clicked.')
            
    def visualize_nj(self, process):
        self.graphicsScene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.graphicsScene)
        node_positions = process.node_positions
        sources = process.sources
        targets = process.targets
       
        minX = float("inf")
        maxX = 0
        minY = float("inf")
        maxY = 0
        for n in node_positions:
            node = node_positions[n]
            valueX = int(node[0])
            valueY = int(node[1])
            if valueX < minX:
                minX = valueX
            if valueX > maxX:
                maxX = valueX
            if valueY < minY:
                minY = valueY
            if valueY > maxY:
                maxY = valueY
       
        for n in node_positions:
            node = node_positions[n]
            valueX = int(node[0])
            valueY = int(node[1])
            valueX= MinMaxNormalization.normalize_value(valueX, minX, maxX, 30, 1000)
            valueY= MinMaxNormalization.normalize_value(valueY, minY, maxY, 30, 450)
            node_positions[n] = (valueX, valueY)
        
        for i in range(0, len(sources)):
            source_node = node_positions[sources[i]]
            source_X = source_node[0]
            source_Y = source_node[1]
            target_node = node_positions[targets[i]]
            target_X = target_node[0]
            target_Y = target_node[1]
            pen = QtGui.QPen(QtCore.Qt.black)
            line = QtCore.QLineF(source_X+10, source_Y+10, target_X+10, target_Y+10)
            self.graphicsScene.addLine(line, pen)
            
        for n in node_positions:
            node = node_positions[n]
            valueX = int(node[0])
            valueY = int(node[1])
            
            if n[0] == 'v':
                pen = QtGui.QPen(QtCore.Qt.black)
                brush = QtGui.QBrush(QtCore.Qt.black)
                
                item = QtWidgets.QGraphicsEllipseItem()
                ellipse = QtCore.QRectF(valueX+7, valueY+7, 4 , 4)
                item.setRect(ellipse)
                item.setBrush(brush)
                item.setPen(pen)
                self.graphicsScene.addItem(item)  
            else:    
                pen = QtGui.QPen(QtCore.Qt.black)
                brush = QtGui.QBrush(QtCore.Qt.darkCyan)
                
                item = QtWidgets.QGraphicsEllipseItem()
                ellipse = QtCore.QRectF(valueX, valueY, 30 , 30)
                item.setRect(ellipse)
                item.setBrush(brush)
                item.setPen(pen)
                item.setToolTip(n)
                self.graphicsScene.addItem(item)  
                
                #self.graphicsScene.addEllipse(ellipse, pen, brush)
                
                text = QtWidgets.QGraphicsTextItem()
                text.setPlainText(n)
                text.setPos(valueX+6, valueY+3)
                self.graphicsScene.addItem(text)  

        
                
                 
    def open_source(self):
        form = OpenSourceForm(self)
        form.setModal(True)
        form.set_previousopensource_option(self.file_names, self.dissimilarity_name,
                                           self.dissimilarity_index, self.radiobutton_option, self.strategy_name,
                                           self.strategy_index, self.normalization_check)
        form.setWindowModality(QtCore.Qt.ApplicationModal)
        form.show()
        form.exec_()
        if form.last_event == "accept":
            print("process")
            self.file_names = form.file_names
            self.radiobutton_option = form.radiobutton_option
            self.dissimilarity_name = form.dissimilarity_name
            self.dissimilarity_index = form.dissimilarity_index
            self.strategy_name = form.strategy_name
            self.strategy_index = form.strategy_index
            self.normalization_check = form.normalization_check
            process = ProcessGenerator(self.file_names, self.dissimilarity_name, self.dissimilarity_index,
                                       self.strategy_name, self.strategy_index,
                                       self.radiobutton_option, False, 39, 38, 56, 58, self.normalization_check)
            self.njdismatrix = process.njdismatrix
            self.visualize_nj(process)      
            self.regressionLineEdit.setText(str(round(process.regression_std_error,2)))
                
    def save_dissimilarity_matrix(self):
        if self.file_names:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file, _ = QFileDialog.getSaveFileName(self, "Save Dissimilarity Matrix", "",
                                                  "Dissimilarity Matrices (*.dmat)")
            if ".dmat" not in file:
                file = file+".dmat"
            self.njdismatrix.save(file)

        else:
            QMessageBox.information(self, "Message", "4D Seismic Matrices not selected.")
    
    def save_scene_image(self):
        if self.file_names:
            print("aurea")
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                  "Image (*.png)")
            if ".png" not in file:
                file = file+".png"
            
            
            isize = self.graphicsScene.sceneRect().size().toSize()
            qimage = QtGui.QImage(isize,QtGui.QImage.Format_ARGB32)
            qimage.fill(QtCore.Qt.white)
            painter = QtGui.QPainter(qimage)
            self.graphicsScene.render(painter)    
            painter.end()
            qimage.save(file)
        else:
            QMessageBox.information(self, "Message", "4D Seismic Matrices not selected.")

 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()
    sys.exit(app.exec_())
