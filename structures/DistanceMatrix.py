import numpy as np
import sys


class DistanceMatrix:

    def __init__(self):
        self.labels = []
        self.ids = []
        self.classes = []
        self.matrix = []
        self.file_name = ""
        # number of points
        self.nr_elements = 0
        # maximum distance in the distance matrix
        self.max_distance = sys.float_info.min
        # minimum distance in the distance matrix
        self.min_distance = sys.float_info.max

    def set_distance(self, index_a, index_b, value):
        assert (index_a >= 0) and (index_a < self.nr_elements) and (index_b >= 0) and (index_b < self.nr_elements),\
            "ERROR: index out of bounds"
        if index_a != index_b:
            if index_a < index_b:
                self.matrix[index_b - 1][index_a] = value
            else:
                self.matrix[index_a - 1][index_b] = value

            if (self.min_distance > value) and (value >= 0.0):
                self.min_distance = value
            elif (self.max_distance < value) and (value >= 0.0):
                self.max_distance = value

    def get_distance(self, index_a, index_b):
        assert (index_a >= 0) and (index_a < self.nr_elements) and (index_b >= 0) and (index_b < self.nr_elements),\
            "ERROR: index out of bounds! " + index_a + "," + index_b + "."
        if index_a == index_b:
            return 0.0
        elif index_a < index_b:
            return self.matrix[index_b - 1][index_a]
        else:
            return self.matrix[index_a - 1][index_b]

    def set_distmatrix(self, distmatrix):
        self.matrix = distmatrix

    def set_element_count(self, nr_elements):
        self.nr_elements = nr_elements

    def save(self, file_name):
        file = open(str(file_name), "w+")
        file.write(str(self.nr_elements) + "\n")
        for i in range(len(self.labels)):
            file.write(self.labels[i])
            if i != (len(self.labels) - 1):
                file.write(";")
        file.write("\n")
        # clusters
        for i in range(len(self.labels)):
            file.write(str(1.0))
            if i != (len(self.labels) - 1):
                file.write(";")
        file.write("\n")

        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                file.write(str(self.matrix[i][j]))
                if j != (len(self.matrix[i]) - 1):
                    file.write(";")
            file.write("\n")
        file.flush()
        file.close()

    def load(self, file_name):
        self.file_name = file_name
        file = open(file_name, "r")

        count = 0
        row_it = 0
        for line in file:
            count += 1
            if count == 1:
                # number of elements
                self.nr_elements = int(line)
                # creating the distance matrix
                self.max_distance = sys.float_info.min  # maximum distance in the distmatrix
                self.min_distance = sys.float_info.max  # minimum distance in the distmatrix
                self.matrix = []
                for i in range(0, self.nr_elements):
                    row = np.zeros(i + 1)
                    self.matrix.append(row)
            elif count == 2:
                # getting the ids
                self.labels = [x.strip() for x in line.split(';')]
                for i in range(1, self.nr_elements+1):
                    self.ids.append(i)

            elif count == 3:
                # getting the classes
                self.classes = [x.strip() for x in line.split(';')]
            elif count > 3:
                if line:
                    values = [x.strip() for x in line.split(';')]
                    for col_it in range(0, len(values)):
                        dist = float(values[col_it].strip())
                        self.set_distance(row_it + 1, col_it, dist)
                    row_it += 1
        file.flush()
        file.close()
