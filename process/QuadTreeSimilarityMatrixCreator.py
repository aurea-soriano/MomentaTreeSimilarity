from numpy import *
import numpy as np
from utils.MatrixReader import MatrixReader
from structures.QuadTree import QuadTree
from dissimilarity.ExtendedJaccard import ExtendedJaccard


class QuadTreeSimilarityMatrixCreator:

    def __init__(self, list_matrix_filename, matrix_name):
        matrixreader = MatrixReader(list_matrix_filename)
        print(matrix_name)
        if ".dmat" not in matrix_name:
            matrix_name = matrix_name+".dmat"
        self.matrix_name = matrix_name
        self.list_matrices = matrixreader.list_matrices
        self.list_names = matrixreader.list_names
        self.list_representations = []
        self.similarity_matrix = []
        self.local_similarity = []

        for x in range(0, len(self.list_matrices)):
            wo_nan_matrix = np.array(np.nan_to_num(self.list_matrices[x]))
            quadtree = QuadTree(wo_nan_matrix)
            self.list_representations.append(quadtree)

        self.create_matrix()
        self.calculate_quadtreesimilarity_matrix()

    def calculate_quadtreesimilarity_matrix(self):
        file = open(str(self.matrix_name), "w+")
        print(len(self.list_matrices))
        file.write(str(len(self.list_matrices)) + "\n")
        for i in range(len(self.list_matrices)):
            file.write(self.list_names[i])
            if i != (len(self.list_matrices)-1):
                file.write(";")
        file.write("\n")
        for i in range(len(self.list_matrices)):
            file.write(str(0.0))
            if i != (len(self.list_matrices) - 1):
                file.write(";")
        file.write("\n")
        for i in range(len(self.similarity_matrix)):
            for j in range(len(self.similarity_matrix[i])):
                file.write(str(self.similarity_matrix[i][j]))
                if j != (len(self.similarity_matrix[i]) - 1):
                    file.write(";")
            file.write("\n")

        file.close()

    def create_matrix(self):
        for i in range(len(self.list_matrices)-1):
            similarity_row = []
            for j in range(i+1):
                distance = self.calculate_distance(self.list_representations[i+1], self.list_representations[j])
                similarity_row.append(round(distance, 6))
            self.similarity_matrix.append(similarity_row)

    def calculate_distance(self, quadtree1, quadtree2):
        self.local_similarity = []
        self.traverse(quadtree1.root, quadtree2.root)
        return np.sum(self.local_similarity)

    def traverse(self, node1, node2):
        self.local_similarity.append(ExtendedJaccard(node1.moments, node2.moments))

        if node1.topLeftChild is not None and node2.topLeftChild is not None:
            self.traverse(node1.topLeftChild, node2.topLeftChild)

        if node1.topRightChild is not None and node2.topRightChild is not None:
            self.traverse(node1.topRightChild, node2.topRightChild)

        if node1.botLeftChild is not None and node2.botLeftChild is not None:
            self.traverse(node1.botLeftChild, node2.botLeftChild)

        if node1.botRightChild is not None and node2.botRightChild is not None:
            self.traverse(node1.botRightChild, node2.botRightChild)


def main():
    file_name = "/home/aurea/DIP_seis_obs_234x326_T4018.mat"

    file = open(file_name+".dmat", 'w')
    file.write('whatever')
    file.close()


if __name__ == "__main__":
    main()
