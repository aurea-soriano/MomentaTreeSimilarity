from numpy import *
import numpy as np
from features.OVMoments import OVMoments
from utils.SubMatrix import SubMatrix
from utils.MatrixReader import MatrixReader


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Node:
    def __init__(self, parent_, moments_, matrix_, depth_, orientation_):
        self.parent = parent_
        self.moments = moments_
        self.matrix = matrix_
        self.child1 = None
        self.child2 = None
        self.depth = depth_
        self.orientation = orientation_


class AABSTree:
    def __init__(self, matrix_, max_depth):
        self.matrix = np.array(matrix_)
        self.total_depth = 0
        self.max_depth = max_depth
        moments = OVMoments(matrix_)
        self.min_size = 4
        self.root = Node(None, moments.ovmoments, matrix_, 1, 'x')
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

        # calculating new orientation
        if parent.orientation == 'x':
            orientation = 'y'
            # calculating sub matrices and moments
            # child1
            child1_matrix = np.array(sub_matrix.get_submatrix(parent_matrix, 0, 0, (width * 2), height - 1)).copy()
            child1_moments = OVMoments(child1_matrix)
            parent.child1 = Node(parent, child1_moments.ovmoments, child1_matrix, parent.depth + 1, orientation)
            self.insert_node(parent.child1)

            # child2
            child2_matrix = np.array(sub_matrix.get_submatrix(parent_matrix, 0, height-1, (width * 2), (height * 2))).copy()
            child2_moments = OVMoments(child2_matrix)
            parent.child2 = Node(parent, child2_moments.ovmoments, child2_matrix, parent.depth + 1, orientation)
            self.insert_node(parent.child2)
        else:
            orientation = 'x'
            # child1
            child1_matrix = np.array(sub_matrix.get_submatrix(parent_matrix, 0, 0, width - 1, (height * 2))).copy()
            child1_moments = OVMoments(child1_matrix)
            parent.child1 = Node(parent, child1_moments.ovmoments, child1_matrix, parent.depth + 1, orientation)
            self.insert_node(parent.child1)

            # child2
            child2_matrix = np.array(
                sub_matrix.get_submatrix(parent_matrix, width - 1, 0, (width * 2), (height * 2))).copy()
            child2_moments = OVMoments(child2_matrix)
            parent.child2 = Node(parent, child2_moments.ovmoments, child2_matrix, parent.depth + 1, orientation)
            self.insert_node(parent.child2)


def main():
    mat1_path = "/home/aurea/Documents/Shell4DSimilarity/data/size/1.mat"
    path_list = list()
    path_list.append(mat1_path)
    matrix_reader = MatrixReader(path_list)
    AABSTree(matrix_reader.list_matrices[0], 100)


if __name__ == "__main__":
    main()
