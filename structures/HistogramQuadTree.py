from numpy import *
import numpy as np
from features.Histogram import Histogram
from utils.SubMatrix import SubMatrix
from utils.MatrixReader import MatrixReader


class HistogramPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class HistogramNode:
    def __init__(self, parent_, histogram_, matrix_, depth_, weight_):
        self.parent = parent_
        self.histogram = histogram_
        self.matrix = matrix_
        self.topLeftChild = None
        self.topRightChild = None
        self.botLeftChild = None
        self.botRightChild = None
        self.depth = depth_
        self.weight = weight_


class HistogramQuadTree:
    def __init__(self, matrix_, max_depth):
        self.matrix = np.array(matrix_)
        self.total_depth = 0
        self.max_depth = max_depth
        moments = Histogram(matrix_)
        self.min_size = 2
        self.root = HistogramNode(None, moments.histogram, matrix_, 1, 1)
        if self.root.matrix.shape[0] <= self.min_size or self.root.matrix.shape[1] <= self.min_size:
            return
        else:
            self.insert_node(self.root)
            self.total_depth = 1

    def insert_node(self, parent):
        if parent.matrix.shape[0] <= self.min_size or parent.matrix.shape[1] <= self.min_size:
            return

        if parent.depth > self.max_depth:
            return

        if self.total_depth < parent.depth:
            self.total_depth = parent.depth

        parent_matrix = np.array(parent.matrix)
        # print(parent_matrix.shape)
        height = round(parent_matrix.shape[0] / 2)  # rows
        width = round(parent_matrix.shape[1] / 2)  # cols
        sub_matrix = SubMatrix()

        # calculating the sub matrices and their moments
        # top left
        top_left_matrix = np.array(sub_matrix.get_submatrix(parent_matrix, 0, 0, width - 1, height - 1)).copy()
        top_left_moments = OVMoments(top_left_matrix)
        # print(top_left_matrix.shape)
        parent.topLeftChild = Node(parent, top_left_moments.ovmoments, top_left_matrix, parent.depth + 1, parent.weight/4)
        self.insert_node(parent.topLeftChild)

        # bottom left
        bot_left_matrix = np.array(sub_matrix.get_submatrix(parent_matrix, 0, height, width-1, (height * 2) - 1)).copy()
        bot_left_moments = OVMoments(bot_left_matrix)
        # print(botLeftMatrix.shape)
        parent.botLeftChild = Node(parent, bot_left_moments.ovmoments, bot_left_matrix, parent.depth + 1, parent.weight/4)
        self.insert_node(parent.botLeftChild)

        # top right
        top_right_matrix = np.array(sub_matrix.get_submatrix(parent_matrix, width, 0, (width * 2) - 1, height - 1))
        top_right_moments = OVMoments(top_right_matrix)
        # print(topRightMatrix.shape)
        parent.topRightChild = Node(parent, top_right_moments.ovmoments, top_right_matrix, parent.depth + 1, parent.weight/4)
        self.insert_node(parent.topRightChild)

        # bot right
        bot_right_matrix = np.array(sub_matrix.get_submatrix(parent_matrix, width, height, (width * 2) - 1,
                                                             (height * 2) - 1))
        bot_right_moments = OVMoments(bot_right_matrix)
        # print(botRightMatrix.shape)
        parent.botRightChild = Node(parent, bot_right_moments.ovmoments, bot_right_matrix, parent.depth + 1, parent.weight/4)
        self.insert_node(parent.botRightChild)


def main():
    mat1_path = "/home/aurea/Documents/Shell4DSimilarity/data/size/1.mat"
    path_list = list()
    path_list.append(mat1_path)
    matrix_reader = MatrixReader(path_list)
    QuadTree(matrix_reader.list_matrices[0], 100)


if __name__ == "__main__":
    main()
